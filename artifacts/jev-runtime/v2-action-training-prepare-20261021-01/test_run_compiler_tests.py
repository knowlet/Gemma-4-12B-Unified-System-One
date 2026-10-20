"""Fake-only guardian regressions; no systemctl, HTTP, models or actors."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def guardian(monkeypatch):
    path = Path(__file__).with_name("run_compiler_tests.py")
    spec = importlib.util.spec_from_file_location("_test_compiler_guardian", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    writes, commands = {}, []
    monkeypatch.setattr(module, "write", lambda name, value: writes.update({name: value}))
    monkeypatch.setattr(module, "command", lambda *args: commands.append(args) or "")
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return module, writes, commands


def state(invocation="owned", substate="running", pid="123"):
    return {
        "InvocationID": invocation,
        "SubState": substate,
        "MainPID": pid,
        "ControlGroup": "/jev-fake-absent-test-cgroup",
    }


def idle():
    return {"comfy_running": 0, "comfy_pending": 0, "mem_available_bytes": 31_000_000_000}


def statuses(monkeypatch, module, values):
    items = iter(values)
    monkeypatch.setattr(module, "status", lambda: next(items))


def test_monitor_host_error_stops_owned_invocation_and_preserves_primary(guardian, monkeypatch):
    module, writes, commands = guardian
    statuses(monkeypatch, module, [state(), state(), state(substate="dead", pid="0")])

    def broken():
        raise TimeoutError("queue unavailable")

    monkeypatch.setattr(module, "host", broken)
    with pytest.raises(TimeoutError, match="queue unavailable"):
        module.watch("owned", state())
    saved = writes["compiler-tests-root-closure.json"]
    assert commands == [("systemctl", "--user", "stop", module.UNIT)]
    assert saved["reason"] == "guardian_error"
    assert saved["primary_error"]["type"] == "TimeoutError"
    assert saved["host_after"]["observation_error"]["type"] == "TimeoutError"
    assert saved["cgroup_absent_after_stop"]


def test_memory_pressure_stops_only_owned_unit_without_success_claim(guardian, monkeypatch):
    module, writes, commands = guardian
    statuses(monkeypatch, module, [state(), state(), state(substate="dead", pid="0")])
    monkeypatch.setattr(
        module,
        "host",
        lambda: {**idle(), "comfy_running": 1, "mem_available_bytes": module.RESERVE - 1},
    )
    saved = module.watch("owned", state())
    assert saved is writes["compiler-tests-root-closure.json"]
    assert saved["reason"] == "host_reserve_crossed"
    assert commands == [("systemctl", "--user", "stop", module.UNIT)]


def test_foreign_invocation_is_recorded_and_never_stopped(guardian, monkeypatch):
    module, writes, commands = guardian
    statuses(monkeypatch, module, [state("foreign"), state("foreign"), state("foreign")])
    monkeypatch.setattr(module, "host", idle)
    with pytest.raises(AssertionError):
        module.watch("owned", state())
    assert not commands
    saved = writes["compiler-tests-root-closure.json"]
    assert saved["reason"] == "guardian_error" and "not_stopped" in saved["stop_result"]


def test_final_queue_error_does_not_erase_closed_terminal_evidence(guardian, monkeypatch):
    module, writes, commands = guardian
    statuses(monkeypatch, module, [state(substate="exited"), state(), state(pid="0")])
    calls = 0

    def host():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("bad final queue JSON")
        return idle()

    monkeypatch.setattr(module, "host", host)
    saved = module.watch("owned", state())
    assert saved["reason"] == "terminal" and saved["after_stop"]["MainPID"] == "0"
    assert saved["host_after"]["observation_error"]["type"] == "ValueError"
    assert writes and len(commands) == 1


def test_busy_with_ample_ram_can_finish_with_explicit_other_owned_unit(guardian, monkeypatch):
    module, _, commands = guardian
    values = iter([state(), state(substate="exited"), state(), state(pid="0")])
    saved = {}
    monkeypatch.setattr(
        module,
        "host",
        lambda: {**idle(), "comfy_running": 1, "mem_available_bytes": 85_000_000_000},
    )
    result = module.watch(
        "owned",
        state(),
        unit="jev-fake-training.service",
        get_status=lambda: next(values),
        save=lambda name, value: saved.update({name: value}),
    )
    assert result["reason"] == "terminal" and result["after_stop"]["MainPID"] == "0"
    assert commands == [("systemctl", "--user", "stop", "jev-fake-training.service")]
    assert saved
