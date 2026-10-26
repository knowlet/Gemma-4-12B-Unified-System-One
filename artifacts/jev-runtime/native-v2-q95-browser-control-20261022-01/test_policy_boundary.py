"""CPU-only ASGI candidates: no HTTP listener, model, unit or real helper."""

import asyncio
import hashlib
import importlib.util
import json
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location(
    "_gemma_policy_boundary_test", HERE / "policy_boundary.py"
)
m = importlib.util.module_from_spec(spec)
exec(compile((HERE / "policy_boundary.py").read_bytes(), str(spec.origin), "exec"), m.__dict__)


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


class Checker:
    binding = {"selected": "synthetic"}

    def __init__(self, hook=None):
        self.calls, self.hook = [], hook

    def __call__(self, binding, phase):
        assert binding == self.binding
        self.calls.append(phase)
        if self.hook:
            self.hook(phase)
        return {"phase": phase, "helper_admitted": True, "co_resident_admitted": True}


async def candidate(scope, receive, send):
    if scope["type"] == "lifespan":
        await send({"type": "lifespan.startup.complete"})
        return
    await receive()
    await send(
        {"type": "http.response.start", "status": 200, "headers": [(b"x-original", b"kept")]}
    )
    await send({"type": "http.response.body", "body": b'{"actions":[{"type":"done"}]}'})


def run(
    tmp_path,
    checker=None,
    *,
    clock=None,
    app=candidate,
    headers=None,
    method="POST",
    path="/v1/systemone",
    control=None,
    sender=None,
):
    selected = checker or Checker()
    state = control or m.ControlState()
    boundary = m.PolicyBoundary(app, selected, tmp_path, "synthetic-token", state, clock=clock)
    messages = []

    async def receive():
        return {"type": "http.request", "body": b'{"original":"unchanged"}'}

    async def send(message):
        messages.append(message)
        if sender:
            await sender(message)

    scope = {"type": "http", "path": path, "method": method, "headers": headers or []}
    asyncio.run(boundary(scope, receive, send))
    return messages, state, selected


def test_original_candidate_and_body_preserved_with_before_after_receipts(tmp_path):
    messages, control, checker = run(tmp_path)
    body = b'{"actions":[{"type":"done"}]}'
    assert messages[0]["status"] == 200 and messages[1]["body"] == body
    assert (b"x-original", b"kept") in messages[0]["headers"]
    assert (tmp_path / "policy-0000/native-candidate.bin").read_bytes() == body
    assert checker.calls == ["before-native-policy-0", "after-native-policy-0"]
    saved = json.loads((tmp_path / "policy-0000/publication-attempt.json").read_bytes())
    assert saved["selected_sha256"] == hashlib.sha256(body).hexdigest()
    assert saved["client_delivery_proven"] is False
    assert control.failure is None


@pytest.mark.parametrize("phase", ["before", "after"])
def test_helper_refusal_never_publishes_native_success(tmp_path, phase):
    def hook(value):
        if value.startswith(phase):
            raise ValueError("synthetic private details must not leak")

    messages, control, _ = run(tmp_path, Checker(hook))
    assert messages[0]["status"] == 503 and control.failure is not None
    assert b"private details" not in b"".join(p.read_bytes() for p in tmp_path.rglob("*.json"))


def test_aftercheck_crosses_original20_not_an_extra_helper_budget(tmp_path):
    clock = Clock()

    def hook(phase):
        if phase.startswith("after"):
            clock.value = 20

    messages, control, _ = run(tmp_path, Checker(hook), clock=clock)
    assert messages[0]["status"] == 504 and control.failure["type"] == "TimeoutError"
    assert (tmp_path / "policy-0000/native-candidate.bin").is_file()


def test_last_receipt_fsync_delay_is_before_final_gate(tmp_path, monkeypatch):
    clock, original = Clock(), m.write

    def delayed(path, blob):
        original(path, blob)
        if path.name == "selected-admission.json":
            clock.value = 21

    monkeypatch.setattr(m, "write", delayed)
    messages, state, _ = run(tmp_path, clock=clock)
    assert messages[0]["status"] == 504
    assert state.requests[0]["selected_status"] == 504
    assert (
        json.loads((tmp_path / "policy-0000/selected-admission.json").read_bytes())[
            "selected_status"
        ]
        == 200
    )
    assert (
        json.loads((tmp_path / "policy-0000/publication-attempt.json").read_bytes())[
            "selected_status"
        ]
        == 504
    )


def test_delayed_bookkeeping_cannot_follow_the_last_success_gate(tmp_path):
    clock, state = Clock(), m.ControlState()

    class DelayedList(list):
        def append(self, row):
            super().append(row)
            clock.value = 20.5

    state.requests = DelayedList()
    messages, _, _ = run(tmp_path, clock=clock, control=state)
    assert messages[0]["status"] == 504
    assert state.requests[0]["selected_status"] == 504


def test_other_thread_latched_failure_after_bookkeeping_refuses_before_commit(tmp_path):
    state = m.ControlState()

    class ConcurrentList(list):
        def append(self, row):
            super().append(row)
            thread = threading.Thread(target=lambda: state.fail(ValueError("synthetic")))
            thread.start()
            thread.join(timeout=1)
            assert not thread.is_alive()

    state.requests = ConcurrentList()
    messages, _, _ = run(tmp_path, control=state)
    assert messages[0]["status"] == 503


def test_send_failure_latches_and_does_not_attempt_second_response_start(tmp_path):
    starts, state = [], m.ControlState()

    async def fail_send(message):
        starts.append(message["type"])
        raise OSError("synthetic transport closed")

    with pytest.raises(OSError):
        run(tmp_path, control=state, sender=fail_send)
    assert starts == ["http.response.start"] and state.failure is not None


def test_send_yield_does_not_hold_thread_lock_or_deadlock_same_loop_failure(tmp_path):
    state = m.ControlState()

    async def yielding_send(message):
        if message["type"] == "http.response.start":
            await asyncio.sleep(0)
            state.fail(RuntimeError("A failure after committed start cannot retract headers"))

    messages, _, _ = run(tmp_path, control=state, sender=yielding_send)
    assert messages[0]["status"] == 200 and state.failure is not None


def test_local_nf4_text_route_refuses_without_helper_or_native_app(tmp_path):
    messages, state, checker = run(tmp_path, path="/chat/completions")
    assert messages[0]["status"] == 503 and not checker.calls and state.failure


def test_duplicate_auth_interpretation_cannot_bypass_outer_policy_boundary(tmp_path):
    headers = [(b"authorization", b"Bearer synthetic-token"), (b"authorization", b"different")]
    messages, _, checker = run(tmp_path, headers=headers)
    assert messages[0]["status"] == 200 and len(checker.calls) == 2


def test_original_auth_refusal_remains_original_refusal(tmp_path):
    async def unauthorized(scope, receive, send):
        await send({"type": "http.response.start", "status": 401, "headers": []})
        await send({"type": "http.response.body", "body": b'{"detail":"unauthorized"}'})

    messages, state, _ = run(tmp_path, app=unauthorized)
    assert messages[0]["status"] == 401 and state.failure is None


def test_lifespan_delegated_without_helper_calls(tmp_path):
    checker, messages = Checker(), []
    boundary = m.PolicyBoundary(candidate, checker, tmp_path, "token", m.ControlState())

    async def send(message):
        messages.append(message)

    asyncio.run(boundary({"type": "lifespan"}, None, send))
    assert messages == [{"type": "lifespan.startup.complete"}] and not checker.calls


def test_persistence_failure_latches_without_success_publication(tmp_path, monkeypatch):
    state = m.ControlState()

    def refuse(_path, _blob):
        raise OSError("synthetic disk refusal")

    monkeypatch.setattr(m, "write", refuse)
    with pytest.raises(OSError):
        run(tmp_path, control=state)
    assert state.failure is not None
