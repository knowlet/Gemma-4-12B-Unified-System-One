"""Source-only supervisor tests; no systemd, private data or GPU."""

import json
import subprocess
import threading
from pathlib import Path

import jev_decision3_jevbench_coherence_supervisor as supervisor
import pytest


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "SOURCE_ONLY", False)
    output = tmp_path / "worker"
    output.mkdir(mode=0o700)
    unit = "jev-coherence-" + "a" * 32 + ".service"
    invocation = "b" * 32
    plan_sha = "c" * 64
    group = "/user.slice/app.slice/" + unit
    plan = {
        "unit": unit,
        "output_dir": str(output),
        "wall_seconds": 30,
        "python": "/fake/python",
        "source_pins": {
            str(Path(module.__file__).absolute()): "f" * 64
            for module in (
                supervisor,
                supervisor.worker,
                supervisor.common,
                supervisor.child,
                supervisor.common.systemd,
            )
        },
    }
    properties = {
        "InvocationID": invocation,
        "LoadState": "loaded",
        "ControlGroup": group,
        "MainPID": "123",
        "ActiveState": "active",
        "SubState": "running",
        "Type": "exec",
        "MemoryAccounting": "yes",
        "MemoryMax": str(supervisor.common.MEMORY_MAX),
        "MemorySwapMax": "0",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxUSec": "30s",
        "TimeoutStopUSec": "3s",
        "ExecMainCode": "1",
        "ExecMainStatus": "0",
        "Result": "success",
    }
    state = {
        "launched": False,
        "stopped": False,
        "invalid_hello": False,
        "host_failure": False,
        "exit_failure": False,
        "closed": True,
    }
    monkeypatch.setattr(supervisor.worker, "_load_plan", lambda *_: plan)
    monkeypatch.setattr(supervisor.common, "_validate_interpreter", lambda *_: None)
    monkeypatch.setattr(supervisor.common, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(supervisor, "_verified_release", lambda *_: None)
    monkeypatch.setattr(supervisor, "_live_limits", lambda *_: None)
    monkeypatch.setattr(supervisor.common.systemd, "boot_id", lambda: "boot")
    monkeypatch.setattr(supervisor.common.systemd, "process_identity", lambda _: {"startticks": 99})
    monkeypatch.setattr(supervisor.common.systemd, "cgroup_membership", lambda _: group)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)

    def observe(_floor):
        if state["host_failure"] and state["launched"]:
            raise ValueError("queue busy")
        return {"monotonic": 1, "available_bytes": 2, "running": 0, "pending": 0}

    def service_show(*_args, **_kwargs):
        if not state["launched"]:
            return {"LoadState": "not-found", "ActiveState": "inactive", "MainPID": "0"}
        result = dict(properties)
        if state["stopped"] or (output / "go.json").exists():
            result.update(MainPID="0", ActiveState="active", SubState="exited")
            if state["exit_failure"]:
                result.update(ExecMainStatus="1", Result="exit-code")
        return result

    def launch(*_args, **_kwargs):
        state["launched"] = True
        hello = {
            "schema": supervisor.HELLO,
            "pid": 123,
            "startticks": 98 if state["invalid_hello"] else 99,
            "boot_id": "boot",
            "unit": unit,
            "invocation": invocation,
            "plan_sha256": plan_sha,
        }
        (output / "hello.json").write_text(json.dumps(hello))
        terminal = {
            "schema": supervisor.TERMINAL,
            "status": "completed_pending_external_audit",
            "invocation": invocation,
            "plan_sha256": plan_sha,
            "host_supervision": "external",
            "model_score": None,
            "final_cuda_allocated": 0,
            "evaluation": {},
        }
        (output / "terminal.json").write_text(json.dumps(terminal))

    def stop(actual_unit, actual_invocation):
        assert (actual_unit, actual_invocation) == (unit, invocation)
        state["stopped"] = True

    def closed(_group, _unit):
        if not state["closed"]:
            raise ValueError("still populated")
        return "confirmed_unpopulated"

    monkeypatch.setattr(supervisor.common, "_observe", observe)
    monkeypatch.setattr(supervisor.common.systemd, "service_show", service_show)
    monkeypatch.setattr(supervisor.subprocess, "run", launch)
    monkeypatch.setattr(supervisor.common, "_stop_same", stop)
    monkeypatch.setattr(supervisor.common, "_closed_cgroup", closed)
    return plan, state, output, tmp_path / "report", plan_sha


def call(fake):
    _plan, _state, _output, report, digest = fake
    return supervisor.supervise(
        "/fake/plan", digest, "/fake/release", report, supervisor.common.MEMORY_FLOOR
    )


def test_supervised_success_is_only_pending_external_audit(fake):
    plan, state, output, report, digest = fake
    result = call(fake)
    assert result["status"] == "completed_pending_external_audit"
    assert result["model_score"] is None and result["cgroup_closure"] == "confirmed_unpopulated"
    assert json.loads((output / "go.json").read_text()) == {
        "schema": supervisor.GO,
        "invocation": "b" * 32,
        "plan_sha256": digest,
    }
    assert not state["stopped"]
    assert json.loads((report / "closure.json").read_text()) == result
    args = supervisor._unit_args(plan, "/fake/plan", digest, "/fake/release")
    assert "--property=RuntimeMaxSec=30" in args
    assert "--property=MemorySwapMax=0" in args
    assert "--property=KillMode=control-group" in args


@pytest.mark.parametrize("failure", ["invalid_hello", "host_failure", "exit_failure"])
def test_bad_identity_host_or_worker_exit_stop_only_owned_unit(fake, failure):
    _plan, state, output, report, _digest = fake
    state[failure] = True
    with pytest.raises(ValueError):
        call(fake)
    assert state["stopped"] is (failure != "invalid_hello")
    if failure != "exit_failure":
        assert not (output / "go.json").exists()
    assert json.loads((report / "closure.json").read_text())["status"] == "failed_no_score"


def test_racing_unit_created_by_another_process_is_never_stopped(fake, monkeypatch):
    _plan, state, _output, report, _digest = fake

    def conflict(*_args, **_kwargs):
        state["launched"] = True
        raise subprocess.CalledProcessError(1, "systemd-run")

    monkeypatch.setattr(supervisor.subprocess, "run", conflict)
    with pytest.raises(subprocess.CalledProcessError):
        call(fake)
    assert not state["stopped"]
    closure = json.loads((report / "closure.json").read_text())
    assert closure["status"] == "failed_no_score"
    assert closure["invocation"] is None and closure["closure"] == "unknown"


def test_cgroup_not_closed_requires_stop_and_fails(fake):
    _plan, state, _output, report, _digest = fake
    state["closed"] = False
    with pytest.raises(RuntimeError, match="closure required recovery"):
        call(fake)
    assert state["stopped"]
    closure = json.loads((report / "closure.json").read_text())
    assert closure["status"] == "failed_no_score" and closure["closure"] == "unknown"


def test_hard_wall_expiry_stops_same_unit_before_go(fake, monkeypatch):
    _plan, state, output, report, _digest = fake
    times = iter([0, 31])
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: next(times))
    with pytest.raises(ValueError, match="wall deadline"):
        call(fake)
    assert state["stopped"] and not (output / "go.json").exists()
    assert json.loads((report / "closure.json").read_text())["status"] == "failed_no_score"


def test_actual_resource_limit_failure_refuses_go(fake, monkeypatch):
    _plan, state, output, _report, _digest = fake

    def fail(*_args):
        raise ValueError("actual limits differ")

    monkeypatch.setattr(supervisor, "_live_limits", fail)
    with pytest.raises(ValueError, match="actual limits differ"):
        call(fake)
    assert state["stopped"] and not (output / "go.json").exists()


def test_worker_release_only_changes_gate(tmp_path, monkeypatch):
    held = tmp_path / "worker.py"
    release = tmp_path / "released.py"
    held.write_bytes(b"SOURCE_ONLY = True\n")
    release.write_bytes(b"SOURCE_ONLY = False\n")
    pins = {str(path): supervisor.common._sha(path.read_bytes()) for path in (held, release)}
    monkeypatch.setattr(supervisor.worker, "__file__", str(held))
    supervisor._verified_release(release, pins)
    release.write_bytes(b"SOURCE_ONLY = False\n# changed\n")
    pins[str(release)] = supervisor.common._sha(release.read_bytes())
    with pytest.raises(ValueError, match="exactly"):
        supervisor._verified_release(release, pins)


@pytest.mark.parametrize(("text", "seconds"), [("30s", 30), ("2min", 120), ("1h 30min", 5400)])
def test_systemd_runtime_parsing(text, seconds):
    assert supervisor._seconds(text) == seconds


def test_source_gate_precedes_plan():
    with pytest.raises(ValueError, match="independent review"):
        supervisor.supervise("/missing", "a" * 64, "/missing", "/missing", 0)


def test_go_is_not_visible_until_complete_and_cannot_overwrite(tmp_path, monkeypatch):
    original = supervisor.common._write_new
    written, resume = threading.Event(), threading.Event()

    def staged(path, raw):
        original(path, raw)
        written.set()
        assert resume.wait(timeout=5)

    monkeypatch.setattr(supervisor.common, "_write_new", staged)
    errors = []

    def publish():
        try:
            supervisor._publish_go(tmp_path, "b" * 32, "c" * 64, lambda: None)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=publish)
    thread.start()
    try:
        assert written.wait(timeout=5)
        assert not (tmp_path / "go.json").exists()
    finally:
        resume.set()
        thread.join(timeout=5)
    assert not thread.is_alive() and not errors
    old = (tmp_path / "go.json").read_bytes()
    with pytest.raises(FileExistsError):
        supervisor._publish_go(tmp_path, "d" * 32, "e" * 64, lambda: None)
    assert (tmp_path / "go.json").read_bytes() == old


def test_go_deadline_after_staging_never_publishes(tmp_path):
    def expired():
        raise TimeoutError("expired")

    with pytest.raises(TimeoutError, match="expired"):
        supervisor._publish_go(tmp_path, "b" * 32, "c" * 64, expired)
    assert not (tmp_path / "go.json").exists()
    assert not (tmp_path / ".go.tmp").exists()
