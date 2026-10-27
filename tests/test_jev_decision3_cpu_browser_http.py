import hashlib
import http.client as http_client
import json
import socket
import threading
import time

import jev_decision3_cpu_browser_adapter as adapter
import jev_decision3_cpu_browser_http as http
import pytest
from test_jev_decision3_cpu_browser_adapter import FakeRuntime, request

TOKEN = "a" * 40


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(adapter, "SOURCE_ONLY", False)
    monkeypatch.setattr(http, "SOURCE_ONLY", False)
    monkeypatch.setattr(http.os, "sched_getaffinity", lambda _: set(range(15, 20)))
    runtime = FakeRuntime()
    cpu = adapter.CpuBrowserAdapter(runtime, model_identity="cpu-test", enabled=True)
    events = []
    selected = http.CpuBrowserHTTP(
        cpu,
        adapter_module=adapter,
        token=TOKEN,
        max_requests=2,
        event_sink=events.append,
        enabled=True,
    )
    return selected, runtime, events


def body():
    return json.dumps(request(), separators=(",", ":")).encode()


def test_source_only_and_affinity_gate(monkeypatch):
    monkeypatch.setattr(http, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="opt-in"):
        http.CpuBrowserHTTP(
            None,
            adapter_module=None,
            token=TOKEN,
            max_requests=1,
            event_sink=lambda _: None,
            enabled=True,
        )
    with pytest.raises(ValueError, match="opt-in"):
        http.loopback_server(None, enabled=True)


def test_direct_cpu_core_binds_hashes_and_delivery_without_claiming_client_receipt(service):
    endpoint, runtime, events = service
    started = time.monotonic()
    raw = body()
    response, receipt = endpoint.prepare(raw, started=started)
    assert receipt["status"] == "prepared_not_delivered"
    assert receipt["request_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["response_sha256"] == hashlib.sha256(response).hexdigest()
    assert json.loads(response)["answers"]["operation"]["choice"] == "CLICK"
    assert len(runtime.forwards) == 1
    endpoint.delivery(receipt, delivered=True, started=started)
    assert events[-1]["status"] == "socket_write_returned_client_receipt_unverified"
    assert not endpoint.failed


@pytest.mark.parametrize(
    "raw", [b"", b"{", b'{"x":1,"x":2}', b'{"x":NaN}', b"[]", b"x" * (http.MAX_BODY + 1)]
)
def test_bad_body_latches_failure_before_model(service, raw):
    endpoint, runtime, _events = service
    with pytest.raises((ValueError, UnicodeError)):
        endpoint.prepare(raw, started=time.monotonic())
    assert not runtime.forwards and endpoint.failed
    with pytest.raises(ValueError, match="service failed"):
        endpoint.prepare(body(), started=time.monotonic())


def test_deadline_failure_is_latched_and_does_not_claim_cancellation(service):
    endpoint, runtime, _events = service
    with pytest.raises(ValueError, match="no hard cancellation"):
        endpoint.prepare(body(), started=time.monotonic() - 21)
    assert endpoint.failed and not runtime.forwards
    with pytest.raises(ValueError, match="service failed"):
        endpoint.prepare(body(), started=time.monotonic())


def test_undelivered_or_mismatched_receipt_latches_failure(service):
    endpoint, _runtime, _events = service
    started = time.monotonic()
    _raw, receipt = endpoint.prepare(body(), started=started)
    with pytest.raises(ValueError):
        endpoint.delivery(receipt, delivered=False, started=started)
    assert endpoint.failed
    with pytest.raises(ValueError, match="service failed"):
        endpoint.prepare(body(), started=time.monotonic())


def test_pending_delivery_or_request_limit_rejects_next(service):
    endpoint, _runtime, _events = service
    endpoint.prepare(body(), started=time.monotonic())
    with pytest.raises(ValueError, match="single request"):
        endpoint.prepare(body(), started=time.monotonic())
    assert endpoint.failed


def test_wrong_affinity_latches_without_forward(service, monkeypatch):
    endpoint, runtime, _events = service
    monkeypatch.setattr(http.os, "sched_getaffinity", lambda _: {15, 16})
    with pytest.raises(ValueError, match="five-CPU"):
        endpoint.prepare(body(), started=time.monotonic())
    assert endpoint.failed and not runtime.forwards


def test_event_sink_failure_latches(service):
    endpoint, _runtime, _events = service

    def bad(_value):
        raise RuntimeError("private details")

    endpoint.event_sink = bad
    with pytest.raises(RuntimeError):
        endpoint.prepare(body(), started=time.monotonic())
    assert endpoint.failed


def test_concurrent_request_prevents_inflight_response(service):
    endpoint, runtime, _events = service
    entered, resume = threading.Event(), threading.Event()
    original = runtime.system_one

    def blocked(**kwargs):
        entered.set()
        assert resume.wait(timeout=5)
        return original(**kwargs)

    runtime.system_one = blocked
    outcome = []

    def run():
        try:
            outcome.append(endpoint.prepare(body(), started=time.monotonic()))
        except BaseException as error:
            outcome.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(timeout=5)
        with pytest.raises(ValueError, match="single request"):
            endpoint.prepare(body(), started=time.monotonic())
    finally:
        resume.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(outcome) == 1 and isinstance(outcome[0], ValueError)
    assert endpoint.failed


@pytest.fixture
def listener(service):
    endpoint, runtime, events = service
    server = http.loopback_server(endpoint, enabled=True)
    assert server.server_address[0] == "127.0.0.1" and server.server_address[1] > 0
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    yield server, endpoint, runtime, events
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_loopback_http_success_preserves_response_and_private_receipt(listener):
    server, endpoint, runtime, events = listener
    client = http_client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        client.request(
            "POST",
            "/v1/systemone",
            body=body(),
            headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"},
        )
        response = client.getresponse()
        payload = response.read()
        assert response.status == 200
        assert json.loads(payload)["answers"]["click_target"]["choice"] == "left"
    finally:
        client.close()
    assert len(runtime.forwards) == 1
    deadline = time.monotonic() + 2
    while len(events) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert [event["status"] for event in events] == [
        "prepared_not_delivered",
        "socket_write_returned_client_receipt_unverified",
    ]
    assert not endpoint.failed


def test_unauthorized_loopback_fails_closed_without_model(listener):
    server, endpoint, runtime, _events = listener
    client = http_client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        client.request(
            "POST", "/v1/systemone", body=body(), headers={"Authorization": "Bearer " + "b" * 40}
        )
        response = client.getresponse()
        assert response.status == 400 and response.read() == b""
    finally:
        client.close()
    assert endpoint.failed and not runtime.forwards


def test_ambiguous_framing_is_rejected_before_forward(listener):
    server, endpoint, runtime, _events = listener
    raw = (
        "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer "
        + TOKEN
        + "\r\nContent-Length: 2\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}"
    ).encode()
    with socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=5) as client:
        client.sendall(raw)
        assert b"400" in client.recv(4096)
    assert endpoint.failed and not runtime.forwards


def test_bounded_sequential_requests_then_latch(service):
    endpoint, runtime, _events = service
    for _ in range(2):
        started = time.monotonic()
        _response, receipt = endpoint.prepare(body(), started=started)
        endpoint.delivery(receipt, delivered=True, started=started)
    with pytest.raises(ValueError, match="bounded request count"):
        endpoint.prepare(body(), started=time.monotonic())
    assert endpoint.failed and len(runtime.forwards) == 2


def test_oversize_response_latches_without_delivery(service, monkeypatch):
    endpoint, runtime, events = service
    monkeypatch.setattr(http, "MAX_RESPONSE", 10)
    with pytest.raises(ValueError, match="bounded response"):
        endpoint.prepare(body(), started=time.monotonic())
    assert endpoint.failed and len(runtime.forwards) == 1 and not events


def test_delivery_callback_race_latches_before_next_request(service):
    endpoint, _runtime, _events = service
    started = time.monotonic()
    _response, receipt = endpoint.prepare(body(), started=started)
    entered, resume = threading.Event(), threading.Event()

    def sink(_value):
        entered.set()
        assert resume.wait(timeout=5)

    endpoint.event_sink = sink
    outcome = []

    def deliver():
        try:
            endpoint.delivery(receipt, delivered=True, started=started)
        except BaseException as error:
            outcome.append(error)

    thread = threading.Thread(target=deliver)
    thread.start()
    try:
        assert entered.wait(timeout=5)
        with pytest.raises(ValueError, match="single request"):
            endpoint.prepare(body(), started=time.monotonic())
    finally:
        resume.set()
        thread.join(timeout=5)
    assert not thread.is_alive() and len(outcome) == 1
    assert endpoint.failed


def test_socket_timeout_cannot_cancel_blocked_forward(service, monkeypatch):
    endpoint, runtime, _events = service
    entered, resume = threading.Event(), threading.Event()
    old = runtime.system_one

    def blocked(**kwargs):
        entered.set()
        assert resume.wait(timeout=5)
        return old(**kwargs)

    runtime.system_one = blocked
    monkeypatch.setattr(http, "TIMEOUT_SECONDS", 0.2)
    server = http.loopback_server(endpoint, enabled=True)
    server_thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    server_thread.start()
    outcome = []

    def client_request():
        client = http_client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=2)
        try:
            client.request(
                "POST", "/v1/systemone", body=body(), headers={"Authorization": "Bearer " + TOKEN}
            )
            response = client.getresponse()
            outcome.append(response.status)
            response.read()
        except (OSError, http_client.HTTPException) as error:
            outcome.append(type(error).__name__)
        finally:
            client.close()

    client_thread = threading.Thread(target=client_request)
    client_thread.start()
    try:
        assert entered.wait(timeout=5)
        deadline = time.monotonic() + 2
        while not endpoint.failed and time.monotonic() < deadline:
            time.sleep(0.01)
        assert endpoint.failed
        assert server_thread.is_alive() and not resume.is_set()
    finally:
        resume.set()
        client_thread.join(timeout=5)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
    assert not client_thread.is_alive() and not server_thread.is_alive()
    assert outcome and outcome[0] != 200
    assert len(runtime.forwards) == 1


@pytest.mark.parametrize(
    "headers",
    [
        "Authorization: Bearer "
        + TOKEN
        + "\r\nAuthorization: Bearer "
        + TOKEN
        + "\r\nContent-Length: 2\r\n",
        "Authorization: Bearer "
        + TOKEN
        + "\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n",
    ],
)
def test_ambiguous_authorization_or_transfer_encoding_fails_closed(listener, headers):
    server, endpoint, runtime, _events = listener
    raw = (
        "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\n"
        + headers
        + "Connection: close\r\n\r\n{}"
    ).encode()
    with socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=5) as client:
        client.sendall(raw)
        assert b"400" in client.recv(4096)
    assert endpoint.failed and not runtime.forwards


def test_overlapping_loopback_connections_fail_closed_before_second_forward(service):
    endpoint, runtime, _events = service
    entered, resume = threading.Event(), threading.Event()
    original = runtime.system_one

    def blocked(**kwargs):
        entered.set()
        assert resume.wait(timeout=5)
        return original(**kwargs)

    runtime.system_one = blocked
    server = http.loopback_server(endpoint, enabled=True)
    server_thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    server_thread.start()
    outcomes = []

    def send():
        client = http_client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=4)
        try:
            client.request(
                "POST", "/v1/systemone", body=body(), headers={"Authorization": "Bearer " + TOKEN}
            )
            response = client.getresponse()
            outcomes.append(response.status)
            response.read()
        except (OSError, http_client.HTTPException) as error:
            outcomes.append(type(error).__name__)
        finally:
            client.close()

    first = threading.Thread(target=send)
    second = threading.Thread(target=send)
    first.start()
    try:
        assert entered.wait(timeout=5)
        second.start()
        deadline = time.monotonic() + 2
        while not endpoint.failed and time.monotonic() < deadline:
            time.sleep(0.005)
        assert endpoint.failed
        assert not resume.is_set()
        assert not runtime.forwards
    finally:
        resume.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
    assert not first.is_alive() and not second.is_alive() and not server_thread.is_alive()
    assert len(outcomes) == 2 and all(value != 200 for value in outcomes)
    assert len(runtime.forwards) == 1


def test_overlapping_connection_during_headers_never_calls_model(service):
    endpoint, runtime, _events = service
    server = http.loopback_server(endpoint, enabled=True)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    first = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=4)
    try:
        first.sendall(b"POST /v1/systemone HTTP/1.1\r\n")
        deadline = time.monotonic() + 2
        while not server._connection_active and time.monotonic() < deadline:
            time.sleep(0.005)
        assert server._connection_active
        with socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=4) as second:
            second.sendall(b"POST /v1/systemone HTTP/1.1\r\n")
            deadline = time.monotonic() + 2
            while not endpoint.failed and time.monotonic() < deadline:
                time.sleep(0.005)
            assert endpoint.failed
    finally:
        first.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not thread.is_alive() and not runtime.forwards


def test_nonoverlapping_loopback_connections_preserve_sequential_requests(listener):
    server, endpoint, runtime, _events = listener
    for _ in range(2):
        deadline = time.monotonic() + 2
        while server._connection_active and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not server._connection_active
        client = http_client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        try:
            client.request(
                "POST", "/v1/systemone", body=body(), headers={"Authorization": "Bearer " + TOKEN}
            )
            response = client.getresponse()
            assert response.status == 200
            response.read()
        finally:
            client.close()
    assert len(runtime.forwards) == 2 and not endpoint.failed
