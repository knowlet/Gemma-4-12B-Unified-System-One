"""Synthetic owner/watchdog tests. No systemd unit, browser or GPU is started."""

from pathlib import Path
from types import SimpleNamespace

import jev_decision3_link_navigation_features as features
import jev_decision3_link_navigation_gpu_owner as owner
import jev_decision3_link_navigation_pipeline as pipeline
import pytest


def test_comfy_observer_fails_closed_without_mutating_comfy(monkeypatch):
    monkeypatch.setattr(owner, "_read_memavailable", lambda: owner.MEMORY_FLOOR + 1)
    monkeypatch.setattr(owner, "_queue_bytes", lambda: b'{"queue_running":[],"queue_pending":[]}')
    observation = owner._observe(owner.MEMORY_FLOOR)
    assert observation["running"] == observation["pending"] == 0
    for payload in (
        b'{"queue_running":[{}],"queue_pending":[]}',
        b'{"queue_running":[],"queue_pending":[{}]}',
        b'{"queue_running":[],"queue_pending":null}',
        b'{"queue_running":[],"queue_running":[],"queue_pending":[]}',
    ):
        monkeypatch.setattr(owner, "_queue_bytes", lambda payload=payload: payload)
        with pytest.raises((ValueError, TypeError)):
            owner._observe(owner.MEMORY_FLOOR)
    monkeypatch.setattr(owner, "_queue_bytes", lambda: b'{"queue_running":[],"queue_pending":[]}')
    monkeypatch.setattr(owner, "_read_memavailable", lambda: owner.MEMORY_FLOOR - 1)
    with pytest.raises(ValueError, match="memory"):
        owner._observe(owner.MEMORY_FLOOR)

    def unavailable():
        raise TimeoutError("queue unavailable")

    monkeypatch.setattr(owner, "_queue_bytes", unavailable)
    with pytest.raises(TimeoutError):
        owner._observe(owner.MEMORY_FLOOR)


def unit_properties(unit):
    return {
        "InvocationID": "a" * 32,
        "ControlGroup": "/user.slice/app.slice/" + unit,
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "Type": "exec",
        "MemoryAccounting": "yes",
        "MemoryMax": str(owner.MEMORY_MAX),
        "MemorySwapMax": "0",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxUSec": "2min",
        "TimeoutStopUSec": "3s",
        "MainPID": "123",
        "ExecMainCode": "1",
        "ExecMainStatus": "0",
        "Result": "success",
    }


def test_binding_and_stop_reject_replaced_units(monkeypatch):
    unit = "jev-link-feature-" + "b" * 32 + ".service"
    props = unit_properties(unit)
    ready = {"pid": 123, "invocation": "a" * 32, "boot_id": "boot", "startticks": 42}
    monkeypatch.setattr(owner.systemd, "boot_id", lambda: "boot")
    monkeypatch.setattr(owner.systemd, "process_identity", lambda _pid: {"startticks": 42})
    monkeypatch.setattr(owner.systemd, "cgroup_membership", lambda _pid: props["ControlGroup"])
    assert owner._binding(props, ready, unit) == "a" * 32
    for key, value in (
        ("InvocationID", "c" * 32),
        ("MemoryMax", "max"),
        ("MemorySwapMax", "max"),
        ("RuntimeMaxUSec", "infinity"),
        ("MainPID", "124"),
    ):
        bad = dict(props, **{key: value})
        with pytest.raises(ValueError):
            owner._binding(bad, ready, unit)
    calls = []
    monkeypatch.setattr(
        owner.systemd, "service_show", lambda *_args, **_kwargs: dict(props, InvocationID="c" * 32)
    )
    monkeypatch.setattr(owner.systemd, "safe_stop", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="replacement"):
        owner._stop_same(unit, "a" * 32)
    assert calls == []
    monkeypatch.setattr(owner.systemd, "service_show", lambda *_args, **_kwargs: props)
    owner._stop_same(unit, "a" * 32)
    assert calls == [(unit, "a" * 32)]


def plan(tmp_path):
    pins = {
        str(Path(owner.__file__).absolute()): "a" * 64,
        str(Path(features.__file__).absolute()): "b" * 64,
        str(Path(pipeline.__file__).absolute()): "c" * 64,
        str(Path(owner.systemd.__file__).absolute()): "d" * 64,
    }
    return {
        "schema": owner.SCHEMA,
        "collection_dir": str(tmp_path / "collection"),
        "receipt_sha256": "a" * 64,
        "collection_plan_sha256": "b" * 64,
        "cache_root": str(tmp_path / "cache"),
        "modules_cache": str(tmp_path / "modules"),
        "overlay": str(
            Path(owner.__file__).absolute().parent.parent / "artifacts/jev-runtime/overlay"
        ),
        "source_pins": pins,
        "python": "/usr/bin/python3",
        "output_dir": str(tmp_path / "output"),
        "shard_size": 2,
        "shard_index": 0,
        "memory_floor_bytes": owner.MEMORY_FLOOR,
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
    }


def test_unit_limits_and_plan_are_explicit(tmp_path):
    config = plan(tmp_path)
    owner._validate_plan(config)
    args = owner._unit_args(
        "jev-link-feature-x.service", tmp_path / "plan", "d" * 64, tmp_path / "output", config
    )
    assert "--property=MemoryMax=" + str(owner.MEMORY_MAX) in args
    assert "--property=MemorySwapMax=0" in args
    assert "--property=RuntimeMaxSec=120" in args
    assert "--property=AllowedCPUs=15-19" in args
    pos = args.index("/usr/bin/taskset")
    assert args[pos : pos + 4] == ["/usr/bin/taskset", "-c", "15-19", config["python"]]
    assert not any("interrupt" in arg or "free" in arg for arg in args)
    for key, value in (
        ("memory_floor_bytes", owner.MEMORY_FLOOR - 1),
        ("shard_size", 41),
        ("receipt_sha256", "bad"),
        ("cache_root", "relative"),
        ("gpu_uuid", "not-a-uuid"),
        ("gpu_uuid", "12345678-1234-1234-1234-123456789abc"),
    ):
        with pytest.raises(ValueError):
            owner._validate_plan(dict(config, **{key: value}))


def test_parent_stops_only_bound_unit_when_comfy_becomes_busy(tmp_path, monkeypatch):
    config = plan(tmp_path)
    monkeypatch.setattr(owner, "SOURCE_ONLY", False)
    monkeypatch.setattr(owner, "_load_plan", lambda *_: config)
    monkeypatch.setattr(owner, "_validate_interpreter", lambda _plan: None)
    monkeypatch.setattr(owner, "_verify_source_pins", lambda _pins: None)
    monkeypatch.setattr(owner, "_bind_project_source", lambda _root: None)
    monkeypatch.setattr(features, "_check_sources", lambda _pins: None)
    monkeypatch.setattr(pipeline, "admit_training_collection", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(features, "feature_only_shards", lambda *_args, **_kwargs: [{}])
    calls = []

    def observe(_floor):
        calls.append("observe")
        if calls.count("observe") == 3:
            raise ValueError("Comfy queue is not idle")
        return {"running": 0, "pending": 0}

    monkeypatch.setattr(owner, "_observe", observe)
    count = 0

    def show(unit, **_kwargs):
        nonlocal count
        count += 1
        return (
            {"LoadState": "not-found", "ActiveState": "inactive", "MainPID": "0"}
            if count == 1
            else unit_properties(unit)
        )

    monkeypatch.setattr(owner.systemd, "service_show", show)
    monkeypatch.setattr(owner, "_binding", lambda *_args: "a" * 32)
    monkeypatch.setattr(
        owner, "_stop_same", lambda unit, invocation: calls.append(("stop", unit, invocation))
    )
    monkeypatch.setattr(
        owner.systemd, "closed_unit_metadata", lambda *_args, **_kwargs: "unconfirmed-in-test"
    )

    def launch(*_args, **_kwargs):
        (tmp_path / "output" / "ready.json").write_bytes(
            owner._json({"pid": 123, "boot_id": "boot", "startticks": 42})
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(owner.subprocess, "run", launch)
    monkeypatch.setattr(owner.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="Comfy"):
        owner.supervise(tmp_path / "plan", "d" * 64)
    stops = [x for x in calls if isinstance(x, tuple)]
    assert len(stops) == 1 and stops[0][2] == "a" * 32
    events = (tmp_path / "output" / "owner-events.jsonl").read_text()
    assert "monitor_failed" in events and "same_invocation_stop_returned" in events


def test_held_source_cannot_launch(tmp_path):
    with pytest.raises(ValueError, match="review"):
        owner.supervise(tmp_path / "plan", "d" * 64)


def test_active_exited_terminal_requires_same_invocation_and_no_main_pid():
    props = unit_properties("u.service")
    assert owner._terminal_metadata(props, "a" * 32) is None
    props.update(SubState="exited", MainPID="0")
    assert owner._terminal_metadata(props, "a" * 32)["Result"] == "success"
    with pytest.raises(ValueError, match="identity"):
        owner._terminal_metadata(props, "b" * 32)
    props["MainPID"] = "123"
    with pytest.raises(ValueError, match="identity"):
        owner._terminal_metadata(props, "a" * 32)


@pytest.mark.parametrize("code", ["1", "exited"])
def test_successful_exit_accepts_both_systemd_representations(code):
    terminal = {"ExecMainCode": code, "ExecMainStatus": "0", "Result": "success"}
    assert owner._successful_exit(terminal)
    assert not owner._successful_exit(dict(terminal, ExecMainStatus="1"))
    assert not owner._successful_exit(dict(terminal, Result="exit-code"))
    assert not owner._successful_exit(dict(terminal, ExecMainCode="killed"))


def test_project_source_rejects_import_outside_workspace(tmp_path, monkeypatch):
    src = tmp_path / "src"
    (src / "s1").mkdir(parents=True)
    (src / "s1" / "__init__.py").write_text("")
    monkeypatch.setattr(owner.sys, "path", list(owner.sys.path))
    monkeypatch.setattr(
        owner.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__file__=str(src / "s1" / "__init__.py")),
    )
    assert owner._bind_project_source(tmp_path) == src
    monkeypatch.setattr(
        owner.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__file__="/outside/s1/__init__.py"),
    )
    with pytest.raises(ValueError, match="escaped"):
        owner._bind_project_source(tmp_path)


def test_unconfirmed_cgroup_is_not_reported_as_success(tmp_path, monkeypatch):
    config = plan(tmp_path)
    monkeypatch.setattr(owner, "SOURCE_ONLY", False)
    monkeypatch.setattr(owner, "_load_plan", lambda *_: config)
    monkeypatch.setattr(owner, "_validate_interpreter", lambda _plan: None)
    monkeypatch.setattr(owner, "_verify_source_pins", lambda _pins: None)
    monkeypatch.setattr(owner, "_bind_project_source", lambda _root: None)
    monkeypatch.setattr(features, "_check_sources", lambda _pins: None)
    monkeypatch.setattr(pipeline, "admit_training_collection", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(features, "feature_only_shards", lambda *_args, **_kwargs: [{}])
    monkeypatch.setattr(owner, "_observe", lambda _floor: {"running": 0, "pending": 0})
    monkeypatch.setattr(owner, "_binding", lambda *_args: "a" * 32)
    monkeypatch.setattr(owner.time, "sleep", lambda _: None)
    count = 0

    def show(unit, **_kwargs):
        nonlocal count
        count += 1
        if count == 1:
            return {"LoadState": "not-found", "ActiveState": "inactive", "MainPID": "0"}
        props = unit_properties(unit)
        if count >= 3:
            props.update(SubState="exited", MainPID="0")
            (tmp_path / "output" / "worker-result.json").write_bytes(
                owner._json(
                    {
                        "status": "captured_pending_independent_fp64_audit",
                        "invocation": "a" * 32,
                        "plan_sha256": "d" * 64,
                        "final_cuda_allocated": 0,
                    }
                )
            )
        return props

    monkeypatch.setattr(owner.systemd, "service_show", show)
    monkeypatch.setattr(
        owner.systemd, "closed_unit_metadata", lambda *_args, **_kwargs: "retained-unit"
    )

    def populated(*_args):
        raise ValueError("cgroup still populated")

    monkeypatch.setattr(owner, "_closed_cgroup", populated)
    stops = []
    monkeypatch.setattr(
        owner, "_stop_same", lambda unit, invocation: stops.append((unit, invocation))
    )

    def launch(*_args, **_kwargs):
        (tmp_path / "output" / "ready.json").write_bytes(
            owner._json({"pid": 123, "boot_id": "boot", "startticks": 42})
        )

    monkeypatch.setattr(owner.subprocess, "run", launch)
    with pytest.raises(RuntimeError, match="closure is unconfirmed"):
        owner.supervise(tmp_path / "plan", "d" * 64)
    assert len(stops) == 1 and stops[0][1] == "a" * 32
    assert (
        owner._decode((tmp_path / "output" / "owner-closure.json").read_bytes())["closure"]
        == "unknown"
    )
