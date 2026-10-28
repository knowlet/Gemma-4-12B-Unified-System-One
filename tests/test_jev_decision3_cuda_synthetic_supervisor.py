"""Synthetic process, source, and GPU doubles; no model, benchmark, or cases."""

import hashlib
import json

import jev_decision3_candidate_browser_child as child
import jev_decision3_candidate_http_owner as common
import jev_decision3_cpu_experiment_worker as cpu
import jev_decision3_cuda_synthetic_launcher as launcher
import jev_decision3_cuda_synthetic_supervisor as supervisor
import pytest


def encode(value):
    return json.dumps(value, separators=(",", ":")).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture(autouse=True)
def inject_test_dependencies(monkeypatch):
    for module in (launcher, supervisor):
        monkeypatch.setattr(module, "child", child)
        monkeypatch.setattr(module, "common", common)
        monkeypatch.setattr(module, "cpu", cpu)
    monkeypatch.setattr(supervisor, "launcher", launcher)


@pytest.fixture
def plan_fixture(tmp_path, monkeypatch):
    base = tmp_path / "experiments"
    run = base / ("run-" + "a" * 32)
    run.mkdir(parents=True)
    output = run / "worker"
    output.mkdir(mode=0o700)
    path = run / "plan.json"
    source = str(tmp_path / "selected-public.py")
    monkeypatch.setattr(launcher, "BASE", base)
    monkeypatch.setattr(launcher, "_source_set", lambda: {source})
    plan = {
        "schema": launcher.SCHEMA,
        "unit": "jev-cuda-synthetic-" + "a" * 32 + ".service",
        "output_dir": str(output),
        "python": launcher.common.GPU_PYTHON,
        "source_pins": {source: "b" * 64},
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        "wall_seconds": 120,
        "memory_floor_bytes": launcher.common.MEMORY_FLOOR,
    }
    raw = encode(plan)
    path.write_bytes(raw)
    return path, plan, sha(raw)


def test_source_only_gates_before_plan_io(monkeypatch):
    monkeypatch.setattr(launcher, "SOURCE_ONLY", True)
    monkeypatch.setattr(supervisor, "SOURCE_ONLY", True)
    monkeypatch.setattr(launcher, "_plan", lambda *_: pytest.fail("must not read plan"))
    with pytest.raises(ValueError, match="launcher release"):
        launcher.run(None, None)
    with pytest.raises(ValueError, match="supervisor release"):
        supervisor.supervise(None, None, None)


def test_plan_accepts_only_explicit_fixed_source_set(plan_fixture):
    path, plan, digest = plan_fixture
    assert launcher._plan(str(path), digest) == plan
    plan["source_pins"][str(path.parent / "unselected.py")] = "c" * 64
    raw = encode(plan)
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="allowlist"):
        launcher._plan(str(path), sha(raw))


@pytest.mark.parametrize(
    "field,value",
    [
        ("wall_seconds", 121),
        ("wall_seconds", True),
        ("memory_floor_bytes", 1),
        ("gpu_uuid", "cuda:0"),
        ("python", "/bin/python"),
        ("unit", "other.service"),
    ],
)
def test_plan_refuses_weaker_limits_or_identity(plan_fixture, field, value):
    path, plan, _digest = plan_fixture
    plan[field] = value
    raw = encode(plan)
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        launcher._plan(str(path), sha(raw))


def test_unit_has_bounded_cpu_memory_swap_and_hard_wall(plan_fixture):
    path, plan, digest = plan_fixture
    args = supervisor._unit_args(plan, path, digest)
    for required in (
        "--property=MemoryMax=8589934592",
        "--property=MemorySwapMax=0",
        "--property=CPUQuota=500%",
        "--property=AllowedCPUs=15-19",
        "--property=KillMode=control-group",
        "--property=OOMPolicy=kill",
        "--property=RuntimeMaxSec=120",
        "--property=TimeoutStopSec=3s",
    ):
        assert required in args
    assert "-B" in args and "-s" in args
    assert str(launcher._release("bootstrap")) in args
    assert "-I" in args and args[args.index("--mode") + 1] == "launcher"


def test_release_requires_exact_source_gate_change(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    monkeypatch.setattr(launcher, "SCRIPTS", scripts)
    held = scripts / launcher.FILES["worker"]
    release = launcher._release("worker")
    source = b"SOURCE_ONLY = True\nvalue = 1\n"
    released = b"SOURCE_ONLY = False\nvalue = 1\n"
    held.write_bytes(source)
    release.write_bytes(released)
    pins = {str(held): sha(source), str(release): sha(released)}
    launcher._verify_release("worker", pins)
    changed = b"SOURCE_ONLY = False\nvalue = 2\n"
    release.write_bytes(changed)
    pins[str(release)] = sha(changed)
    with pytest.raises(ValueError, match="gate-only"):
        launcher._verify_release("worker", pins)


def make_supervisor(tmp_path, monkeypatch, *, host_failure=False):
    run = tmp_path / "run"
    run.mkdir()
    output = run / "worker"
    output.mkdir(mode=0o700)
    path = run / "plan.json"
    report = run / "supervisor"
    plan = {
        "output_dir": str(output),
        "unit": "jev-cuda-synthetic-" + "a" * 32 + ".service",
        "source_pins": {},
        "memory_floor_bytes": launcher.common.MEMORY_FLOOR,
        "wall_seconds": 120,
        "python": launcher.common.GPU_PYTHON,
    }
    invocation, digest, pid = "d" * 32, "e" * 64, 321
    group = "/user.slice/app.slice/" + plan["unit"]
    running = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "InvocationID": invocation,
        "ControlGroup": group,
        "MainPID": str(pid),
        "Type": "exec",
        "MemoryAccounting": "yes",
        "MemoryMax": str(launcher.common.MEMORY_MAX),
        "MemorySwapMax": "0",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxUSec": "2min",
        "TimeoutStopUSec": "3s",
    }
    terminal = {
        **running,
        "SubState": "exited",
        "MainPID": "0",
        "ExecMainCode": "exited",
        "ExecMainStatus": "0",
        "Result": "success",
        "ControlGroup": "",
    }
    snapshots = iter(
        [
            {"LoadState": "not-found", "ActiveState": "inactive", "MainPID": "0"},
            running,
            terminal,
            terminal,
        ]
    )
    monkeypatch.setattr(supervisor, "SOURCE_ONLY", False)
    monkeypatch.setattr(launcher, "_plan", lambda *_: plan)
    monkeypatch.setattr(launcher, "_source_guard", lambda *_: None)
    monkeypatch.setattr(supervisor.common, "_validate_interpreter", lambda *_: None)
    observation_count = []

    def observe(*_):
        observation_count.append(1)
        if host_failure and len(observation_count) == 3:
            raise ValueError("synthetic ComfyUI activity")
        return {"available_bytes": 100000000000, "running": 0, "pending": 0}

    monkeypatch.setattr(supervisor.common, "_observe", observe)
    monkeypatch.setattr(
        supervisor.common.systemd, "service_show", lambda *_args, **_kwargs: next(snapshots)
    )
    monkeypatch.setattr(supervisor.common.systemd, "boot_id", lambda: "boot")
    monkeypatch.setattr(supervisor.common.systemd, "process_identity", lambda _: {"startticks": 7})
    monkeypatch.setattr(supervisor.common.systemd, "cgroup_membership", lambda _: group)
    monkeypatch.setattr(
        supervisor.common.systemd,
        "closed_unit_metadata",
        lambda *_args, **_kwargs: {"verified": True},
    )
    monkeypatch.setattr(supervisor.os, "getpgid", lambda _: pid)
    monkeypatch.setattr(cpu_limits := supervisor.cpu, "_cpu_limits", lambda *_: None)
    assert cpu_limits is launcher.cpu
    monkeypatch.setattr(supervisor.common, "_closed_cgroup", lambda *_: "absent")
    monkeypatch.setattr(supervisor, "_closed_group", lambda _: "absent")
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    stopped = []
    monkeypatch.setattr(
        supervisor.common, "_stop_same", lambda unit, identity: stopped.append((unit, identity))
    )

    def launch(*_args, **_kwargs):
        hello = {
            "schema": "decision3-cuda-synthetic-hello-v1",
            "pid": pid,
            "startticks": 7,
            "boot_id": "boot",
            "unit": plan["unit"],
            "invocation": invocation,
            "plan_sha256": digest,
            "pgid": pid,
            "cgroup": group,
        }
        supervisor.common._write_new(output / "hello.json", encode(hello))

    monkeypatch.setattr(supervisor.subprocess, "run", launch)
    publish = supervisor.child._publish_control

    def complete(path, raw):
        publish(path, raw)
        result = encode(
            {
                "schema": "decision3-cuda-synthetic-worker-v1",
                "status": "completed_pending_external_audit",
                "device": "cuda:0",
                "calls": 1,
                "score": None,
                "final_allocated_bytes": 0,
            }
        )
        supervisor.common._write_new(output / "result.json", result)
        supervisor.common._write_new(
            output / "terminal.json",
            encode(
                {
                    "schema": "decision3-cuda-synthetic-terminal-v1",
                    "status": "completed_pending_external_audit",
                    "invocation": invocation,
                    "plan_sha256": digest,
                    "result_sha256": sha(result),
                }
            ),
        )

    monkeypatch.setattr(supervisor.child, "_publish_control", complete)
    return path, plan, digest, report, output, stopped, invocation


def test_supervisor_full_synthetic_lifecycle_and_scoped_closure(tmp_path, monkeypatch):
    path, plan, digest, report, output, stopped, invocation = make_supervisor(tmp_path, monkeypatch)
    result = supervisor.supervise(str(path), digest, str(report))
    assert result["status"] == "completed_pending_external_audit"
    assert result["cgroup"] == result["process_group"] == "absent"
    assert result["machine_exclusive"] is False and stopped == []
    assert (output / "go.json").is_file() and (report / "closure.json").is_file()
    events = [json.loads(line) for line in (report / "events.jsonl").read_bytes().splitlines()]
    assert [row["event"] for row in events] == ["prelaunch", "host", "binding_verified", "host"]
    assert events[2]["invocation"] == invocation


def test_supervisor_stops_owned_unit_when_host_becomes_busy(tmp_path, monkeypatch):
    path, plan, digest, report, _output, stopped, invocation = make_supervisor(
        tmp_path, monkeypatch, host_failure=True
    )
    with pytest.raises(ValueError, match="ComfyUI"):
        supervisor.supervise(str(path), digest, str(report))
    assert stopped == [(plan["unit"], invocation)]
    closure = json.loads((report / "closure.json").read_bytes())
    assert (
        closure["status"] == "failed" and closure["cgroup"] == closure["process_group"] == "absent"
    )


def test_supervisor_never_stops_unbound_or_other_unit(tmp_path, monkeypatch):
    path, _plan, digest, report, _output, stopped, _invocation = make_supervisor(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        supervisor, "_bind", lambda *_: (_ for _ in ()).throw(ValueError("identity mismatch"))
    )
    monkeypatch.setattr(
        supervisor, "_recover", lambda *_: (_ for _ in ()).throw(ValueError("no proof"))
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        supervisor.supervise(str(path), digest, str(report))
    assert stopped == []
    closure = json.loads((report / "closure.json").read_bytes())
    assert closure["status"] == "cleanup_unconfirmed"


def test_terminal_rejects_changed_bytes_and_incomplete_response(tmp_path):
    result = encode(
        {
            "schema": "decision3-cuda-synthetic-worker-v1",
            "status": "completed_pending_external_audit",
            "device": "cuda:0",
            "calls": 1,
            "score": None,
            "final_allocated_bytes": 0,
        }
    )
    (tmp_path / "result.json").write_bytes(result)
    terminal = {
        "schema": "decision3-cuda-synthetic-terminal-v1",
        "status": "completed_pending_external_audit",
        "invocation": "d" * 32,
        "plan_sha256": "e" * 64,
        "result_sha256": sha(result),
    }
    (tmp_path / "terminal.json").write_bytes(encode(terminal))
    supervisor._terminal(tmp_path, "d" * 32, "e" * 64)
    (tmp_path / "result.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        supervisor._terminal(tmp_path, "d" * 32, "e" * 64)


@pytest.mark.parametrize(
    "fault", ["pid", "invocation", "startticks", "memory", "swap", "runtime", "group"]
)
def test_binding_rejects_invalid_identity_and_resource_limits(monkeypatch, fault):
    pid, invocation = 321, "d" * 32
    unit = "jev-cuda-synthetic-" + "a" * 32 + ".service"
    group = "/user.slice/app.slice/" + unit
    plan = {"unit": unit}
    digest = "e" * 64
    props = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "InvocationID": invocation,
        "ControlGroup": group,
        "MainPID": str(pid),
        "Type": "exec",
        "MemoryAccounting": "yes",
        "MemoryMax": str(launcher.common.MEMORY_MAX),
        "MemorySwapMax": "0",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxUSec": "2min",
        "TimeoutStopUSec": "3s",
    }
    hello = {
        "schema": "decision3-cuda-synthetic-hello-v1",
        "pid": pid,
        "startticks": 7,
        "boot_id": "boot",
        "unit": unit,
        "invocation": invocation,
        "plan_sha256": digest,
        "pgid": pid,
        "cgroup": group,
    }
    monkeypatch.setattr(supervisor.common.systemd, "boot_id", lambda: "boot")
    monkeypatch.setattr(supervisor.common.systemd, "process_identity", lambda _: {"startticks": 7})
    monkeypatch.setattr(supervisor.common.systemd, "cgroup_membership", lambda _: group)
    monkeypatch.setattr(supervisor.os, "getpgid", lambda _: pid)
    monkeypatch.setattr(supervisor.cpu, "_cpu_limits", lambda *_: None)
    if fault == "pid":
        hello["pid"] = 322
    elif fault == "invocation":
        hello["invocation"] = "f" * 32
    elif fault == "startticks":
        hello["startticks"] = 8
    elif fault == "memory":
        props["MemoryMax"] = "9999999999"
    elif fault == "swap":
        props["MemorySwapMax"] = "1"
    elif fault == "runtime":
        props["RuntimeMaxUSec"] = "3min"
    else:
        hello["cgroup"] = "/some-other-group"
    with pytest.raises(ValueError):
        supervisor._bind(props, hello, plan, digest)


@pytest.mark.parametrize("fail", [False, True])
def test_launcher_runs_only_reviewed_worker_after_bound_handshake(tmp_path, monkeypatch, fail):
    output = tmp_path / "worker"
    output.mkdir(mode=0o700)
    invocation, digest, pid = "d" * 32, "e" * 64, 321
    unit = "jev-cuda-synthetic-" + "a" * 32 + ".service"
    group = "/user.slice/app.slice/" + unit
    plan = {
        "output_dir": str(output),
        "unit": unit,
        "source_pins": {},
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
    }
    monkeypatch.setattr(launcher, "SOURCE_ONLY", False)
    monkeypatch.setattr(launcher, "_plan", lambda *_: plan)
    source_checks = []
    monkeypatch.setattr(launcher, "_source_guard", lambda *_: source_checks.append(True))
    monkeypatch.setattr(launcher, "_group", lambda *_: group)
    monkeypatch.setattr(launcher.common, "_validate_interpreter", lambda *_: None)
    monkeypatch.setattr(launcher.common, "_bind_project_source", lambda *_: None)
    monkeypatch.setattr(launcher.os, "getpgrp", lambda: pid)
    monkeypatch.setattr(launcher.os, "getpid", lambda: pid)
    monkeypatch.setattr(launcher.common.systemd, "process_identity", lambda _: {"startticks": 7})
    monkeypatch.setattr(launcher.common.systemd, "boot_id", lambda: "boot")
    monkeypatch.setenv("INVOCATION_ID", invocation)
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_CACHE",
        "HF_MODULES_CACHE",
        "CUDA_VISIBLE_DEVICES",
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "before-test")

    class FakeTorch:
        def set_num_threads(self, count):
            assert count == 5

        def set_num_interop_threads(self, count):
            assert count == 1

    torch = FakeTorch()
    monkeypatch.setattr(launcher.common, "_bind_overlay", lambda *_: torch)
    calls = []

    class FakeWorker:
        def run(self, **kwargs):
            calls.append(kwargs)
            assert kwargs["torch"] is torch and kwargs["gpu_uuid"] == plan["gpu_uuid"]
            kwargs["source_guard"]()
            kwargs["ownership_guard"]()
            if fail:
                error = ValueError("synthetic worker failure")
                error.cuda_synthetic_evidence = {"cleanup": "unconfirmed"}
                raise error
            return {
                "schema": "decision3-cuda-synthetic-worker-v1",
                "status": "completed_pending_external_audit",
                "device": "cuda:0",
                "calls": 1,
                "score": None,
                "final_allocated_bytes": 0,
            }

    monkeypatch.setattr(launcher.common, "_load_reviewed_release", lambda *_: FakeWorker())
    original_write = launcher.common._write_new

    def write(path, raw):
        original_write(path, raw)
        if path.name == "hello.json":
            original_write(
                output / "go.json",
                encode(
                    {
                        "schema": "decision3-cuda-synthetic-go-v1",
                        "invocation": invocation,
                        "plan_sha256": digest,
                    }
                ),
            )

    monkeypatch.setattr(launcher.common, "_write_new", write)
    if fail:
        with pytest.raises(ValueError, match="synthetic worker failure"):
            launcher.run(str(tmp_path / "plan.json"), digest)
        failure = json.loads((output / "failure.json").read_bytes())
        assert failure["cleanup"] == "unconfirmed" and failure["error_type"] == "ValueError"
        assert not (output / "terminal.json").exists()
    else:
        launcher.run(str(tmp_path / "plan.json"), digest)
        terminal = json.loads((output / "terminal.json").read_bytes())
        assert terminal["result_sha256"] == sha((output / "result.json").read_bytes())
        assert terminal["status"] == "completed_pending_external_audit"
    assert len(calls) == 1 and len(source_checks) >= 3


def test_source_guard_rejects_unpinned_runtime_module_without_opening_it(tmp_path, monkeypatch):
    from types import SimpleNamespace

    allowed = str(tmp_path / "allowed.py")
    unselected = str(tmp_path / "unselected.py")
    pins = {allowed: "a" * 64}
    monkeypatch.setattr(
        launcher.common, "_verify_source_pins", lambda selected: assert_pins(selected, pins)
    )
    monkeypatch.setattr(
        launcher.cpu, "_verify_model_manifest", lambda selected: assert_pins(selected, pins)
    )
    monkeypatch.setattr(launcher, "_verify_release", lambda *_: None)
    monkeypatch.setattr(
        launcher, "sys", SimpleNamespace(modules={"s1.backends": SimpleNamespace(__file__=allowed)})
    )
    launcher._source_guard({"source_pins": pins})
    launcher.sys.modules["s1.unknown"] = SimpleNamespace(__file__=unselected)
    with pytest.raises(ValueError, match="actual project"):
        launcher._source_guard({"source_pins": pins})
    assert not (tmp_path / "unselected.py").exists()


def assert_pins(actual, expected):
    assert actual == expected
