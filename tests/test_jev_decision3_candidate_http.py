"""Synthetic candidate HTTP wiring only: no model, GPU or browser is started."""

import copy
import hashlib
import http.client
import json
import threading
import time
from types import SimpleNamespace

import jev_decision3_candidate_http as candidate_http
import pytest
from test_jev_decision3_native import FakeNative, payload


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(candidate_http, "SOURCE_ONLY", False)
    receipts, events, phases = [], [], []
    delivered = threading.Event()
    behavior = {
        "invalid_witness": False,
        "no_witness": False,
        "identity_changed": False,
        "replay_witness": False,
    }

    class Backend(FakeNative):
        def __init__(self):
            super().__init__()
            self.lease = SimpleNamespace(check=lambda: None)

        def predict(self, request, *, deadline=None, evidence_sink=None):
            result = super().predict(request, deadline=deadline, evidence_sink=evidence_sink)
            if not behavior["no_witness"]:
                if behavior["replay_witness"]:
                    duplicate = copy.deepcopy(receipts[-1])
                    # Changing an incidental field cannot make the location new.
                    duplicate["call"] += 1
                    receipts.append(duplicate)
                else:
                    receipts.append(
                        {
                            "call": len(receipts) + 1,
                            "sha256": "a" * 64,
                            "directory": f"/synthetic/call-{len(receipts):04d}",
                        }
                    )
            return result

    backend = Backend()

    def guard(candidate, phase):
        assert candidate is backend
        phases.append(phase)
        return {
            "provider": "changed"
            if behavior["identity_changed"] and phase == "after-request"
            else "fake"
        }

    def verify(receipt, original, legacy):
        assert receipt == receipts[-1]
        assert legacy["decision3_native"]["original_request"] == original
        return not behavior["invalid_witness"]

    def make(*, owner_guard=guard, **kwargs):
        def record(row):
            events.append(row)
            if row["status"] == "socket_write_returned":
                delivered.set()

        return candidate_http.CandidateHTTP(
            backend,
            model_identity="native-test",
            token="t" * 32,
            max_requests=2,
            owner_guard=owner_guard,
            witness_snapshot=lambda: copy.deepcopy(receipts),
            witness_verify=verify,
            event_sink=record,
            **kwargs,
        )

    return SimpleNamespace(
        make=make,
        backend=backend,
        receipts=receipts,
        events=events,
        phases=phases,
        behavior=behavior,
        delivered=delivered,
    )


def test_source_gate_and_caller_identity_required(setup, monkeypatch):
    monkeypatch.setattr(candidate_http, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="independent review"):
        setup.make()
    monkeypatch.setattr(candidate_http, "SOURCE_ONLY", False)
    setup.backend.lease = None
    with pytest.raises(ValueError, match="lease-wrapped"):
        setup.make()


def test_real_loopback_http_preserves_body_hash_witness_and_named_answers(setup):
    service = setup.make()
    server = candidate_http.loopback_server(service)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        host, port = server.server_address
        assert host == "127.0.0.1"
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("POST", "/v1/systemone", body=b"{}", headers={"Authorization": "Bearer wrong"})
        response = conn.getresponse()
        assert response.status == 401
        response.read()
        assert service.calls == 0
        request = payload(3)
        body = json.dumps(request, ensure_ascii=False, indent=2).encode()
        conn.request(
            "POST",
            "/v1/systemone",
            body=body,
            headers={"Authorization": "Bearer " + "t" * 32, "Content-Type": "application/json"},
        )
        response = conn.getresponse()
        raw = response.read()
        assert response.status == 200
        assert setup.delivered.wait(3)
        value = json.loads(raw)
        assert list(value["answers"]) == list(request["questions"])
        assert value["decision3_browser_wire"]["answer_values_changed"] is False
        assert len(setup.backend.calls) == 1
        assert [item["status"] for item in setup.events] == ["prepared", "socket_write_returned"]
        row = setup.events[-1]
        assert row["http_body_sha256"] == hashlib.sha256(body).hexdigest()
        assert row["response_sha256"] == hashlib.sha256(raw).hexdigest()
        assert row["canonical_request_sha256"] != row["http_body_sha256"]
        assert row["witness_receipt"] == setup.receipts[0]
        assert setup.phases == ["initialized", "before-request", "after-request"]
        conn.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert not thread.is_alive()


@pytest.mark.parametrize("mode", ["invalid_witness", "no_witness", "identity_changed"])
def test_failure_latches_and_never_prepares_response(setup, mode):
    service = setup.make()
    setup.behavior[mode] = True
    body = json.dumps(payload(2), separators=(",", ":")).encode()
    with pytest.raises(ValueError):
        service.handle(body, started=time.monotonic())
    assert service.failed is True
    assert not any(event["status"] == "prepared" for event in setup.events)
    with pytest.raises(ValueError, match="prior candidate call failed"):
        service.handle(body, started=time.monotonic())


def test_limits_deadline_and_delivery_failure_are_not_success(setup):
    def clock():
        return 21.0

    service = setup.make(clock=clock)
    body = json.dumps(payload(2)).encode()
    with pytest.raises(ValueError, match="deadline"):
        service.handle(body, started=0.0)
    assert not setup.backend.calls
    assert setup.events[-1]["status"] == "failed"

    service = setup.make()
    response, row = service.handle(body, started=time.monotonic())
    assert response
    service.delivery(row, delivered=False)
    assert service.failed and setup.events[-1]["status"] == "socket_write_failed"


def test_duplicate_json_key_rejected_and_request_budget_enforced(setup):
    service = setup.make()
    with pytest.raises(ValueError, match="duplicate"):
        service.handle(b'{"model":"native-test","model":"native-test"}', started=time.monotonic())
    assert not setup.backend.calls
    service = setup.make()
    body = json.dumps(payload(2)).encode()
    for _ in range(2):
        response, row = service.handle(body, started=time.monotonic())
        assert response
        service.delivery(row, delivered=True)
    with pytest.raises(ValueError, match="request limit"):
        service.handle(body, started=time.monotonic())
    assert len(setup.backend.calls) == 2


def test_late_socket_write_is_recorded_without_claiming_timely_success(setup):
    now = {"value": 1.0}
    service = setup.make(clock=lambda: now["value"])
    body = json.dumps(payload(2)).encode()
    _, row = service.handle(body, started=0.0)
    now["value"] = 21.0
    service.delivery(row, delivered=True)
    assert service.failed
    assert setup.events[-1]["status"] == "socket_write_returned_late"
    assert setup.events[-1]["delivery_elapsed_seconds"] == 21.0


def test_cannot_begin_next_call_until_prior_delivery_is_recorded(setup):
    service = setup.make()
    body = json.dumps(payload(2)).encode()
    _, row = service.handle(body, started=time.monotonic())
    with pytest.raises(ValueError, match="delivery is still pending"):
        service.handle(body, started=time.monotonic())
    assert service.failed and len(setup.backend.calls) == 1
    forged = dict(row, response_sha256="0" * 64)
    with pytest.raises(ValueError, match="matching pending"):
        service.delivery(forged, delivered=True)
    forged = copy.deepcopy(row)
    forged["witness_receipt"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="matching pending"):
        service.delivery(forged, delivered=True)
    forged = dict(row, started_monotonic=row["started_monotonic"] + 1.0)
    with pytest.raises(ValueError, match="matching pending"):
        service.delivery(forged, delivered=True)
    service.delivery(row, delivered=False)
    assert setup.events[-1]["status"] == "socket_write_failed"


def test_delivery_evidence_completion_after_deadline_latches_failure(setup):
    now = {"value": 1.0}
    service = setup.make(clock=lambda: now["value"])
    _, row = service.handle(json.dumps(payload(2)).encode(), started=0.0)
    observed = []

    def slow_sink(event):
        observed.append(event)
        if event["status"] == "socket_write_returned":
            now["value"] = 21.0

    service.event_sink = slow_sink
    service.delivery(row, delivered=True)
    assert service.failed
    assert [event["status"] for event in observed] == [
        "socket_write_returned",
        "delivery_evidence_late",
    ]
    assert observed[-1]["evidence_elapsed_seconds"] == 21.0


@pytest.mark.parametrize("stage", ["header", "body"])
def test_slow_partial_connection_has_absolute_socket_deadline(setup, monkeypatch, stage):
    import socket

    monkeypatch.setattr(candidate_http, "TIMEOUT_SECONDS", 0.8)
    service = setup.make()
    server = candidate_http.loopback_server(service)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    slow = socket.create_connection(server.server_address, timeout=2)
    if stage == "header":
        slow.sendall(b"POST /v1/systemone HTTP/1.1\r\nAuthorization: Bearer ")
    else:
        slow.sendall(
            b"POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer "
            + b"t" * 32
            + b"\r\nContent-Length: 200000\r\n\r\n{"
        )
    stopped = threading.Event()

    def drip():
        while not stopped.wait(0.04):
            try:
                slow.sendall(b"x")
            except OSError:
                break

    drip_thread = threading.Thread(target=drip)
    drip_thread.start()
    try:
        time.sleep(0.08)
        conn = http.client.HTTPConnection(*server.server_address, timeout=5)
        body = json.dumps(payload(2)).encode()
        started = time.monotonic()
        conn.request(
            "POST", "/v1/systemone", body=body, headers={"Authorization": "Bearer " + "t" * 32}
        )
        response = conn.getresponse()
        response.read()
        elapsed = time.monotonic() - started
        assert elapsed < 4.0
        if stage == "header":
            assert response.status == 200
            assert len(setup.backend.calls) == 1
        else:
            assert response.status == 400
            assert service.failed and not setup.backend.calls
        conn.close()
    finally:
        stopped.set()
        drip_thread.join(timeout=2)
        slow.close()
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()
    assert not drip_thread.is_alive() and not worker.is_alive()


@pytest.mark.parametrize("nested", [False, True])
def test_reused_mutable_owner_identity_cannot_hide_change(setup, nested):
    shared = {"model": "original", "metadata": {"weights": "old"}}

    def guard(_backend, phase):
        if phase == "after-request":
            if nested:
                shared["metadata"]["weights"] = "new"
            else:
                shared["model"] = "replacement"
        return shared

    service = setup.make(owner_guard=guard)
    with pytest.raises(ValueError, match="owner identity changed"):
        service.handle(json.dumps(payload(2)).encode(), started=time.monotonic())
    assert service.failed
    assert not any(event["status"] == "prepared" for event in setup.events)


def test_successive_http_connections_reap_socket_watchdog_threads(setup, monkeypatch):
    real_timer = threading.Timer
    timers = []

    def observe_timer(*args, **kwargs):
        timer = real_timer(*args, **kwargs)
        timers.append(timer)
        return timer

    monkeypatch.setattr(candidate_http.threading, "Timer", observe_timer)
    service = setup.make()
    server = candidate_http.loopback_server(service)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        for _ in range(2):
            conn = http.client.HTTPConnection(*server.server_address, timeout=5)
            conn.request(
                "POST",
                "/v1/systemone",
                body=json.dumps(payload(2)).encode(),
                headers={"Authorization": "Bearer " + "t" * 32},
            )
            response = conn.getresponse()
            assert response.status == 200
            response.read()
            conn.close()
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()
    assert len(timers) == 2 and all(not timer.is_alive() for timer in timers)
    assert not worker.is_alive()


@pytest.mark.parametrize("changed", [True, 1.0])
def test_identity_strict_json_types_detect_bool_or_float_in_place_of_integer(setup, changed):
    def guard(_backend, phase):
        return {"model": "fake", "pid": changed if phase == "after-request" else 1}

    service = setup.make(owner_guard=guard)
    with pytest.raises(ValueError, match="owner identity changed"):
        service.handle(json.dumps(payload(2)).encode(), started=time.monotonic())
    assert not any(event["status"] == "prepared" for event in setup.events)


def test_identity_bound_at_service_initialization_before_first_forward(setup):
    identity = {"model": "initial"}
    service = setup.make(owner_guard=lambda _backend, _phase: identity)
    identity["model"] = "replacement"
    with pytest.raises(ValueError, match="owner identity changed"):
        service.handle(json.dumps(payload(2)).encode(), started=time.monotonic())
    assert not setup.backend.calls


@pytest.mark.parametrize("change", ["clear", "replace", "mutate"])
def test_witness_history_must_continue_unmodified_between_calls(setup, change):
    service = setup.make()
    body = json.dumps(payload(2)).encode()
    _, first = service.handle(body, started=time.monotonic())
    service.delivery(first, delivered=True)
    if change == "clear":
        setup.receipts.clear()
    elif change == "replace":
        setup.receipts[:] = [{"call": 99, "sha256": "b" * 64, "directory": "/synthetic/other"}]
    else:
        setup.receipts[0]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="witness snapshot changed between calls"):
        service.handle(body, started=time.monotonic())
    assert service.failed and len(setup.backend.calls) == 1
    assert setup.events[-1]["status"] == "failed"


def test_reused_receipt_directory_rejected_even_if_appended(setup):
    service = setup.make()
    body = json.dumps(payload(2)).encode()
    _, first = service.handle(body, started=time.monotonic())
    service.delivery(first, delivered=True)
    setup.behavior["replay_witness"] = True
    with pytest.raises(ValueError, match="duplicate witness receipt directory"):
        service.handle(body, started=time.monotonic())
    assert service.failed and len(setup.backend.calls) == 2
    assert [event["status"] for event in setup.events] == [
        "prepared",
        "socket_write_returned",
        "failed",
    ]


def test_identical_witness_hashes_can_have_distinct_receipt_directories(setup):
    service = setup.make()
    body = json.dumps(payload(2)).encode()
    for _ in range(2):
        _, row = service.handle(body, started=time.monotonic())
        service.delivery(row, delivered=True)
    assert setup.receipts[0]["sha256"] == setup.receipts[1]["sha256"]
    assert setup.receipts[0]["directory"] != setup.receipts[1]["directory"]
    assert not service.failed


def test_witness_history_modified_during_preparation_cannot_be_returned(setup):
    service = setup.make()
    original_sink = service.event_sink

    def changing_sink(row):
        original_sink(row)
        if row["status"] == "prepared":
            setup.receipts.clear()

    service.event_sink = changing_sink
    with pytest.raises(ValueError, match="witness snapshot changed while preparing"):
        service.handle(json.dumps(payload(2)).encode(), started=time.monotonic())
    assert service.failed
    assert [event["status"] for event in setup.events] == ["prepared", "failed"]


def test_initial_snapshot_must_have_unique_receipt_directories(setup):
    setup.receipts.extend(
        [
            {"directory": "/synthetic/call-0000", "sha256": "a" * 64},
            {"directory": "/synthetic/call-0000", "sha256": "b" * 64},
        ]
    )
    with pytest.raises(ValueError, match="duplicate witness receipt directory"):
        setup.make()


@pytest.mark.parametrize(
    "directory",
    ["relative/path", "/synthetic/./call-0000", "/synthetic/../call-0000", "//synthetic/call-0000"],
)
def test_witness_directory_must_use_canonical_absolute_spelling(setup, directory):
    setup.receipts.append({"directory": directory, "sha256": "a" * 64})
    with pytest.raises(ValueError, match="canonical absolute witness"):
        setup.make()
