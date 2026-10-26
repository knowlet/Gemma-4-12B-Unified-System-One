"""Real sockets and owned fake children; no model/device/browser execution."""

import ctypes
import hashlib
import http.client
import importlib.util
import json
import os
import signal
import socket
import struct
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("decision3_new_local_transport", DIRECTORY / "local_transport.py")
transport = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = transport
SPEC.loader.exec_module(transport)
TOKEN = "fixture-private-token-" + "a" * 32
BODY = b'{ "model": "d3-edge", "state": "preserve whitespace and \\u4e2d" }'


@pytest.fixture
def selection():
    host = transport.host_observation()
    return transport.Selection("fixture-" + os.urandom(8).hex(), "d3-edge", host["boot_id"],
                               host["MemTotal"], "fixture-uuid-not-a-device-query")


@pytest.fixture
def owned(tmp_path, selection):
    sessions = []

    def start(mode="valid", timeout=2):
        output = tmp_path / str(len(sessions))
        output.mkdir()
        session = transport.OwnedSession(selection, output)
        sessions.append(session)
        command = [sys.executable, str(DIRECTORY / "fake_worker.py"), "--selection",
                   json.dumps(asdict(selection)), "--mode", mode]
        session.start(command, timeout=timeout)
        return session

    yield start
    for session in sessions:
        receipt = session.close(grace=0.2, term=0.3, kill=0.3)
        assert receipt["observed_process_exit_confirmed"], receipt
        assert receipt["direct_child_exit_confirmed"], receipt
        assert not receipt["owned_exit_confirmed"] and not receipt["whole_tree_exit_confirmed"]
        assert not receipt["continuation_authorized"]
        assert receipt["external_owner_boundary_required"]
        assert session.process.poll() is not None


def call_http(endpoint, body=BODY, token=TOKEN):
    connection = http.client.HTTPConnection("127.0.0.1", int(endpoint.rsplit(":", 1)[1]), timeout=2)
    try:
        connection.request("POST", "/predict", body=body,
                           headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_loopback_preserves_original_four_field_envelope_and_closes(owned):
    session = owned()
    gateway = transport.LocalGateway(session)
    server = transport.LoopbackHTTP(gateway, token=TOKEN)
    endpoint = server.start()
    try:
        status, response = call_http(endpoint)
        assert status == 200
        assert json.loads(response) == {"fixture": True, "answer": "WAIT"}
    finally:
        server.close()
    row = json.loads(next(session.output.glob("request-*.json")).read_bytes())
    envelope = row["worker"]["journal"]["original_envelope"]
    assert set(envelope) == {"request_id", "payload_json", "http_sha256", "remaining_seconds"}
    assert envelope["payload_json"].encode() == BODY
    assert envelope["http_sha256"] == hashlib.sha256(BODY).hexdigest()
    assert 0 < envelope["remaining_seconds"] <= 20
    assert row["gateway_action_authorized"] is True
    assert row["worker"]["journal"]["native_calls"] == 0
    receipt = session.close(grace=0.3)
    assert receipt["success"] and not receipt["continuation_authorized"]
    assert receipt["observed_process_exit_confirmed"] and not receipt["forced_cleanup"]
    assert not receipt["model_qualified"]
    assert receipt["final_observation"]["owned"] == []


@pytest.mark.parametrize("mode", ["partial-startup", "truncated-ready", "wrong-ready-nonce",
                                 "wrong-ready-pid", "wrong-ready-start", "wrong-ready-uuid",
                                 "wrong-ready-total"])
def test_startup_failure_never_ready_and_confirms_owned_exit(owned, mode):
    with pytest.raises((transport.TransportError, json.JSONDecodeError)) as raised:
        owned(mode)
    receipt = raised.value.transport_receipt
    assert receipt["observed_process_exit_confirmed"]
    assert not receipt["continuation_authorized"]
    assert not receipt["success"]


def test_startup_watchdog_stops_only_owned_child(owned):
    with pytest.raises(transport.TransportError) as raised:
        owned("startup-timeout", timeout=0.04)
    assert raised.value.transport_receipt["forced_cleanup"]
    assert raised.value.transport_receipt["observed_process_exit_confirmed"]


@pytest.mark.parametrize("mode", ["wrong-response-model", "wrong-response-sha", "wrong-response-request",
                                 "refused"])
def test_bound_response_or_native_refusal_blocks_action(owned, mode):
    session = owned(mode)
    gateway = transport.LocalGateway(session)
    status, body, _ = gateway.request(BODY)
    assert status == 504
    assert "no action" in body.decode()
    assert gateway.blocked and session.blocked
    row = json.loads(next(session.output.glob("request-*.json")).read_bytes())
    assert row["gateway_action_authorized"] is False


def test_worker_deadline_does_not_deliver_late_success_and_retains_it_at_close(owned):
    session = owned("request-timeout")
    gateway = transport.LocalGateway(session, seconds=0.04)
    status, _, _ = gateway.request(BODY)
    assert status == 504
    receipt = session.close(grace=0.4)
    assert receipt["observed_process_exit_confirmed"] and not receipt["continuation_authorized"]
    late = json.loads((session.output / "late-control.json").read_bytes())
    assert late["event"] == "response" and late["success"] is True


def test_queue_deadline_forwards_no_request(owned):
    session = owned()
    gateway = transport.LocalGateway(session, seconds=0.03)
    gateway.lock.acquire()
    try:
        status, _, _ = gateway.request(BODY)
    finally:
        gateway.lock.release()
    assert status == 504 and session.requests == 0
    assert gateway.blocked


def test_final_publication_crossing_deadline_retracts_action(owned, monkeypatch):
    session = owned()
    gateway = transport.LocalGateway(session, seconds=0.06)
    record = session.record

    def delayed_record(name, value):
        result = record(name, value)
        if name.startswith("request-") and value["status"] == 200:
            time.sleep(0.07)
        return result

    monkeypatch.setattr(session, "record", delayed_record)
    status, _, evidence = gateway.request(BODY)
    assert status == 504
    row = json.loads((session.output / ("request-" + evidence["request_id"] + ".json")).read_bytes())
    assert row["late_after_publication"] and row["gateway_action_authorized"] is False


def test_delivery_preparation_crossing_deadline_retracts_action(owned):
    session = owned()
    gateway = transport.LocalGateway(session, seconds=0.05)
    status, body, evidence = gateway.request(BODY)
    assert status == 200
    time.sleep(0.06)
    status, body, evidence = gateway.before_delivery(evidence, body)
    assert status == 504 and b"no action" in body
    assert evidence["status"] == 504 and gateway.blocked


def test_http_header_construction_crossing_deadline_publishes_only_504(owned, monkeypatch):
    session = owned()
    gateway = transport.LocalGateway(session, seconds=0.04)
    server = transport.LoopbackHTTP(gateway, token=TOKEN)
    send_header = transport.BaseHTTPRequestHandler.send_header

    def delayed_header(handler, keyword, value):
        if keyword == "Content-Type" and b" 200 " in handler._headers_buffer[0]:
            assert gateway.lock.locked()
            time.sleep(0.06)
        return send_header(handler, keyword, value)

    monkeypatch.setattr(transport.BaseHTTPRequestHandler, "send_header", delayed_header)
    endpoint = server.start()
    try:
        status, body = call_http(endpoint)
        assert status == 504 and b"no action" in body
        row = json.loads(next(session.output.glob("request-*.json")).read_bytes())
        assert row["late_before_delivery"] and not row["gateway_action_authorized"]
        assert row["status"] == 504
    finally:
        server.close()


def test_http_concurrent_queue_refusal_during_headers_cannot_publish_200(owned, monkeypatch):
    session = owned()
    gateway = transport.LocalGateway(session, seconds=1)
    server = transport.LoopbackHTTP(gateway, token=TOKEN)
    header_started, blocker_done = threading.Event(), threading.Event()
    blocker_result = []
    send_header = transport.BaseHTTPRequestHandler.send_header

    def queued_failure():
        if header_started.wait(timeout=1):
            blocker_result.append(gateway.request(BODY, started=time.monotonic() - 0.99))
        blocker_done.set()

    def concurrent_header(handler, keyword, value):
        if keyword == "Content-Type" and b" 200 " in handler._headers_buffer[0]:
            assert gateway.lock.locked()
            header_started.set()
            assert blocker_done.wait(timeout=1)
        return send_header(handler, keyword, value)

    monkeypatch.setattr(transport.BaseHTTPRequestHandler, "send_header", concurrent_header)
    blocker = threading.Thread(target=queued_failure)
    blocker.start()
    endpoint = server.start()
    started = time.monotonic()
    try:
        status, body = call_http(endpoint)
        assert time.monotonic() - started < gateway.seconds
        assert status == 504 and b"no action" in body
        assert blocker_result[0][0] == 504
        assert gateway.blocked and session.requests == 1
        forwarded = [json.loads(p.read_bytes()) for p in session.output.glob("request-*.json")]
        first = next(row for row in forwarded if "worker" in row)
        assert first["late_before_delivery"] and not first["gateway_action_authorized"]
    finally:
        blocker.join(timeout=1)
        assert not blocker.is_alive()
        server.close()


def test_delivery_failure_binds_exact_response_and_blocks_arm(owned):
    session = owned()
    gateway = transport.LocalGateway(session)
    status, body, evidence = gateway.request(BODY)
    assert status == 200
    gateway.delivery_failed(evidence, body, BrokenPipeError("fixture disconnect"))
    receipt = json.loads(next(session.output.glob("delivery-*.json")).read_bytes())
    assert receipt["request"]["response_sha256"] == hashlib.sha256(body).hexdigest()
    assert receipt["delivered"] is False and gateway.blocked


def test_changed_delivery_receipt_refused(owned):
    session = owned()
    gateway = transport.LocalGateway(session)
    _, body, evidence = gateway.request(BODY)
    path = session.output / ("request-" + evidence["request_id"] + ".json")
    path.write_bytes(b"{}")
    with pytest.raises(transport.TransportError, match="binding differs"):
        gateway.before_delivery(evidence, body)
    assert gateway.blocked and session.blocked


@pytest.mark.parametrize("mode", ["missing-closed", "failed-finalize", "wrong-closed-start", "closed-with-error",
                                 "linger-after-closed"])
def test_incomplete_or_forced_closure_never_authorizes_next_arm(owned, mode):
    session = owned(mode)
    receipt = session.close(grace=0.08, term=0.2, kill=0.2)
    assert receipt["observed_process_exit_confirmed"]
    assert not receipt["success"] and not receipt["continuation_authorized"]
    if mode == "linger-after-closed":
        assert receipt["forced_cleanup"]


def test_unauthenticated_loopback_request_never_reaches_child(owned):
    session = owned()
    server = transport.LoopbackHTTP(transport.LocalGateway(session), token=TOKEN)
    endpoint = server.start()
    try:
        status, _ = call_http(endpoint, token="wrong")
        assert status == 401 and session.requests == 0
        assert call_http(endpoint)[0] == 200
    finally:
        server.close()


def test_http_failure_and_bound_handler_thread_join(owned):
    session = owned("request-timeout")
    server = transport.LoopbackHTTP(transport.LocalGateway(session, seconds=0.03), token=TOKEN)
    endpoint = server.start()
    try:
        assert call_http(endpoint)[0] == 504
    finally:
        server.close()
    assert not server.thread.is_alive()


def test_unrelated_current_process_is_never_owned(owned):
    session = owned()
    observed = session.observe_owned()
    assert os.getpid() not in {r["pid"] for r in observed["owned"]}
    assert {r["pid"] for r in observed["owned"]} == {session.process.pid}
    assert observed["unknown"] == []


def test_detached_descendant_is_found_and_closed(owned):
    session = owned("detached-descendant")
    observed = session.observe_owned()
    assert len(observed["owned"]) == 2
    assert any(row["group"] != session.process.pid for row in observed["owned"])
    receipt = session.close(grace=0.3)
    assert receipt["success"] and not receipt["continuation_authorized"]


def test_unmarked_double_fork_cannot_authorize_whole_tree_continuation(owned):
    library = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    library.prctl.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong,
                              ctypes.c_ulong, ctypes.c_ulong]
    library.prctl.restype = ctypes.c_int
    assert library.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert library.prctl(36, ctypes.c_void_p(1), 0, 0, 0) == 0
    escaped = None
    try:
        session = owned("unmarked-double-fork")
        escaped = session.ready["details"]["escaped_fixture"]
        actual = transport.proc_record(escaped["pid"])
        assert actual["start_ticks"] == escaped["start_ticks"]
        assert actual["parent"] == os.getpid()
        assert actual["group"] == actual["session"] == escaped["pid"]
        environment = transport.read_bound(Path("/proc") / str(escaped["pid"]) / "environ", 65536)
        assert environment == b""
        assert escaped["pid"] not in {row["pid"] for row in session.observe_owned()["owned"]}
        receipt = session.close(grace=0.3)
        assert receipt["success"] and receipt["direct_child_exit_confirmed"]
        assert receipt["observed_process_exit_confirmed"]
        assert transport.proc_record(escaped["pid"])["state"] != "Z"
        assert not receipt["owned_exit_confirmed"] and not receipt["whole_tree_exit_confirmed"]
        assert not receipt["continuation_authorized"]
        assert receipt["continuation_refusal_reason"] == "external_kernel_ownership_boundary_required"
    finally:
        if escaped is not None:
            descriptor = transport.open_pidfd(escaped["pid"])
            try:
                assert transport.proc_record(escaped["pid"])["start_ticks"] == escaped["start_ticks"]
                transport.signal_pidfd(descriptor, signal.SIGTERM)
                until = time.monotonic() + 1
                reaped = False
                while time.monotonic() < until:
                    if os.waitpid(escaped["pid"], os.WNOHANG)[0] == escaped["pid"]:
                        reaped = True
                        break
                    time.sleep(0.01)
                if not reaped:
                    transport.signal_pidfd(descriptor, signal.SIGKILL)
                    assert os.waitpid(escaped["pid"], 0)[0] == escaped["pid"]
                assert not (Path("/proc") / str(escaped["pid"])).exists()
            finally:
                os.close(descriptor)
        assert library.prctl(36, ctypes.c_void_p(previous.value), 0, 0, 0) == 0


def test_real_http_disconnect_records_failed_delivery(owned):
    session = owned("request-timeout")
    server = transport.LoopbackHTTP(transport.LocalGateway(session), token=TOKEN)
    endpoint = server.start()
    client = socket.create_connection(("127.0.0.1", int(endpoint.rsplit(":", 1)[1])), timeout=2)
    try:
        request = ("POST /predict HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer " + TOKEN +
                   "\r\nContent-Length: " + str(len(BODY)) + "\r\nConnection: close\r\n\r\n").encode() + BODY
        client.sendall(request)
        until = time.monotonic() + 1
        while session.requests == 0 and time.monotonic() < until:
            time.sleep(0.005)
        assert session.requests == 1
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        client.close()
        until = time.monotonic() + 2
        while not list(session.output.glob("delivery-*.json")) and time.monotonic() < until:
            time.sleep(0.01)
        assert list(session.output.glob("delivery-*.json"))
        assert server.gateway.blocked and session.blocked
    finally:
        client.close()
        server.close()


def test_boot_bound_total_refused_before_spawning(tmp_path, selection):
    values = {**asdict(selection), "expected_memtotal": selection.expected_memtotal - 4096}
    session = transport.OwnedSession(transport.Selection(**values), tmp_path)
    with pytest.raises(transport.TransportError, match="boot/host total"):
        session.start([sys.executable, str(DIRECTORY / "fake_worker.py")])
    assert session.process is None


def test_unknown_process_evidence_keeps_closure_fail_closed(tmp_path, selection, monkeypatch):
    session = transport.OwnedSession(selection, tmp_path)
    session.start([sys.executable, str(DIRECTORY / "fake_worker.py"), "--selection",
                   json.dumps(asdict(selection))])
    observe = session.observe_owned
    try:
        with monkeypatch.context() as patch:
            patch.setattr(session, "observe_owned", lambda: {
                "owned": [], "unknown": [{"pid": 1, "reason": "Synthetic unreadable ownership"}],
                "vanished": [],
            })
            receipt = session.close(grace=0.2)
        assert not receipt["observed_process_exit_confirmed"] and not receipt["continuation_authorized"]
        assert receipt["final_observation"]["unknown"]
    finally:
        until = time.monotonic() + 1
        while session.process.poll() is None and time.monotonic() < until:
            time.sleep(0.01)
        assert session.process.poll() is not None
        assert observe()["owned"] == []


def test_wrong_http_model_blocks_before_forward(owned):
    session = owned()
    status, _, _ = transport.LocalGateway(session).request(b'{"model":"d3"}')
    assert status == 504 and session.requests == 0


def test_nonfinite_duplicate_or_wrong_envelope_rejected():
    with pytest.raises(transport.TransportError, match="Duplicate"):
        transport.parse(b'{"event":"ready","event":"closed"}')
    with pytest.raises(transport.TransportError, match="Nonfinite"):
        transport.parse(b'{"x":NaN}')
    with pytest.raises(transport.TransportError, match="Exact original"):
        transport.parse_envelope(b'{"request_id":"x","extra":1}')


def test_import_starts_no_model_runtime_or_listener():
    assert not any(name in sys.modules for name in ("torch", "transformers", "modal"))
    assert threading.active_count() >= 1
