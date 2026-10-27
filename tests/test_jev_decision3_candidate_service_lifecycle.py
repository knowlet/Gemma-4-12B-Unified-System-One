"""Persistent loopback lifecycle tests with synthetic backend; no GPU."""

import http.client as http_client
import json
import threading
import time

import jev_decision3_candidate_http as http
import jev_decision3_candidate_service_lifecycle as lifecycle
import pytest
from test_jev_decision3_native import payload

pytest_plugins = ("test_jev_decision3_candidate_http",)


@pytest.fixture
def facade(setup, monkeypatch):
    monkeypatch.setattr(lifecycle, "SOURCE_ONLY", False)
    service = setup.make()
    events, cleaned = [], []
    checks = {"supervisor": True, "closure": True}

    def supervisor():
        if not checks["supervisor"]:
            raise RuntimeError("supervision failed")

    def guard():
        if not checks["closure"]:
            raise RuntimeError("lease closure unconfirmed")

    def cleanup():
        cleaned.append(True)
        return {"owned_backend_closed": True}

    instance = lifecycle.CandidateServiceLifecycle(
        http,
        service,
        supervisor=supervisor,
        closure_guard=guard,
        cleanup=cleanup,
        event_sink=events.append,
        poll_seconds=0.02,
    )
    return instance, setup, events, cleaned, checks


def post(address):
    conn = http_client.HTTPConnection(*address, timeout=3)
    try:
        body = json.dumps(payload(2), ensure_ascii=False).encode()
        conn.request(
            "POST",
            "/v1/systemone",
            body=body,
            headers={"Authorization": "Bearer " + "t" * 32},
        )
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def test_source_gate_and_multiple_real_loopback_calls(facade, monkeypatch):
    instance, setup, events, cleaned, _ = facade
    monkeypatch.setattr(lifecycle, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="independent review"):
        instance.start()
    monkeypatch.setattr(lifecycle, "SOURCE_ONLY", False)
    setup.backend.lease.check()
    address = instance.start()
    for _ in range(2):
        status, body = post(address)
        assert status == 200
        assert json.loads(body)["decision3_browser_wire"]["answer_values_changed"] is False
    assert instance.done.wait(2)
    result = instance.stop()
    assert result["status"] == "stopped" and result["reason"] == "request_limit"
    assert result["request_calls"] == len(setup.receipts) == 2
    assert not result["thread_alive"] and result["cleanup_finished"]
    assert cleaned == [True]
    assert [event["event"] for event in events] == [
        "listening",
        "listener_closed",
        "cleanup_completed",
    ]
    assert instance.stop() == result and cleaned == [True]


def test_request_failure_closes_listener_and_latches_failure(facade):
    instance, setup, _events, cleaned, _checks = facade
    address = instance.start()
    assert post(address)[0] == 200
    setup.behavior["invalid_witness"] = True
    assert post(address)[0] == 400
    assert instance.done.wait(2)
    result = instance.stop()
    assert result["status"] == "failed" and result["reason"] == "request_failed"
    assert result["request_failed"] and result["request_calls"] == 2
    assert cleaned == [True] and len(setup.receipts) == 2


def test_stop_cannot_cleanup_while_forward_is_blocked(facade):
    instance, setup, _events, cleaned, _checks = facade
    entered, release = threading.Event(), threading.Event()
    original = setup.backend.predict

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    setup.backend.predict = blocked
    address = instance.start()
    result = []
    client = threading.Thread(target=lambda: result.append(post(address)))
    client.start()
    try:
        assert entered.wait(2)
        with pytest.raises(TimeoutError, match="cleanup and process closure unconfirmed"):
            instance.stop(timeout_seconds=0.05)
        assert not instance.snapshot()["cleanup_attempted"] and cleaned == []
    finally:
        release.set()
        client.join(timeout=3)
    assert not client.is_alive() and result[0][0] == 200
    state = instance.stop()
    assert state["status"] == "stopped" and not state["thread_alive"] and cleaned == [True]


def test_closure_guard_blocks_cleanup_until_verified(facade):
    instance, _setup, _events, cleaned, checks = facade
    instance.start()
    checks["closure"] = False
    with pytest.raises(RuntimeError, match="lease closure"):
        instance.stop()
    assert cleaned == [] and not instance.snapshot()["cleanup_attempted"]
    checks["closure"] = True
    assert instance.stop()["cleanup_finished"] and cleaned == [True]


def test_supervisor_failure_is_visible_and_cleanup_still_once(facade):
    instance, _setup, _events, cleaned, checks = facade
    instance.start()
    checks["supervisor"] = False
    assert instance.done.wait(2)
    result = instance.stop()
    assert result["status"] == "failed"
    assert result["reason"] == "supervisor_or_listener_failed"
    assert result["failure_type"] == "RuntimeError" and cleaned == [True]


def test_cleanup_evidence_failure_latches_without_retry(facade):
    instance, _setup, _events, cleaned, _checks = facade
    instance.start()

    def failing_sink(event):
        if event["event"] == "cleanup_completed":
            raise RuntimeError("audit sink failed")

    instance.event_sink = failing_sink
    with pytest.raises(RuntimeError, match="audit sink failed"):
        instance.stop()
    assert cleaned == [True] and instance.snapshot()["cleanup_attempted"]
    with pytest.raises(ValueError, match="prior cleanup failed"):
        instance.stop()
    assert cleaned == [True]


def test_concurrent_stop_never_cleans_up_twice(facade):
    instance, _setup, _events, cleaned, _checks = facade
    instance.start()
    barrier = threading.Barrier(3)
    original_cleanup = instance.cleanup

    def slow_cleanup():
        time.sleep(0.05)
        return original_cleanup()

    instance.cleanup = slow_cleanup
    results = []

    def stop():
        barrier.wait()
        try:
            results.append(instance.stop())
        except BaseException as error:
            results.append(error)

    threads = [threading.Thread(target=stop) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=3)
    assert all(not thread.is_alive() for thread in threads)
    assert all(isinstance(result, dict) and result["cleanup_finished"] for result in results)
    assert cleaned == [True]


def test_failed_start_can_still_cleanup_owned_backend(facade):
    instance, _setup, _events, cleaned, _checks = facade

    def fail_listening(event):
        if event["event"] == "listening":
            raise RuntimeError("audit sink failed")

    instance.event_sink = fail_listening
    with pytest.raises(RuntimeError, match="audit sink failed"):
        instance.start()
    result = instance.stop()
    assert result["status"] == "failed" and result["cleanup_finished"]
    assert cleaned == [True] and not result["thread_alive"]


@pytest.mark.parametrize("failure", ["supervisor", "bind"])
def test_pre_listener_start_failure_allows_owned_backend_cleanup(facade, monkeypatch, failure):
    instance, _setup, _events, cleaned, checks = facade
    if failure == "supervisor":
        checks["supervisor"] = False
    else:

        def fail_bind(_service):
            raise OSError("listener bind failed")

        monkeypatch.setattr(http, "loopback_server", fail_bind)
    with pytest.raises((RuntimeError, OSError)):
        instance.start()
    result = instance.stop()
    assert result["status"] == "failed" and result["cleanup_finished"]
    assert cleaned == [True] and not result["thread_alive"]


def test_blocked_listener_event_sink_does_not_hide_unfinished_thread(facade):
    instance, setup, _events, cleaned, _checks = facade
    setup.backend.lease.check()
    instance.service.max_requests = 1
    entered, release = threading.Event(), threading.Event()

    def sink(event):
        if event["event"] == "listener_closed":
            entered.set()
            assert release.wait(3)

    instance.event_sink = sink
    address = instance.start()
    assert post(address)[0] == 200
    assert entered.wait(2)
    try:
        with pytest.raises(TimeoutError, match="cleanup and process closure unconfirmed"):
            instance.stop(timeout_seconds=0.05)
        assert instance.snapshot()["thread_alive"] and cleaned == []
    finally:
        release.set()
    assert instance.stop()["cleanup_finished"] and cleaned == [True]


def test_double_failure_blocks_cleanup_until_listener_really_closes(facade, monkeypatch):
    instance, _setup, _events, cleaned, _checks = facade

    class InvalidListener:
        server_address = ("0.0.0.0", 12345)
        closed = False
        fail_close = True

        def server_close(self):
            if self.fail_close:
                raise OSError("close failed")
            self.closed = True

        def fileno(self):
            return -1 if self.closed else 10

    listener = InvalidListener()
    monkeypatch.setattr(http, "loopback_server", lambda _service: listener)
    with pytest.raises(ValueError, match="loopback listener") as caught:
        instance.start()
    assert any("close failed" in note for note in caught.value.__notes__)
    with pytest.raises(OSError, match="close failed"):
        instance.stop()
    assert not listener.closed and cleaned == [] and not instance.snapshot()["cleanup_attempted"]
    listener.fail_close = False
    state = instance.stop()
    assert listener.closed and state["status"] == "failed" and state["cleanup_finished"]
    assert cleaned == [True]


def test_noop_listener_close_does_not_prove_socket_closure(facade, monkeypatch):
    instance, _setup, _events, cleaned, _checks = facade
    instance.start()
    server = instance.server
    real_close = server.server_close
    monkeypatch.setattr(server, "server_close", lambda: None)
    try:
        with pytest.raises(ValueError, match="listener socket closure unconfirmed"):
            instance.stop()
        assert cleaned == [] and not instance.snapshot()["cleanup_attempted"]
    finally:
        monkeypatch.setattr(server, "server_close", real_close)
        state = instance.stop()
    assert state["cleanup_finished"] and cleaned == [True]
