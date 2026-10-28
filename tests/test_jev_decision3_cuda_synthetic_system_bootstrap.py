"""Invented source bytes only: no model, dataset, cases or external execution."""

import hashlib
from types import SimpleNamespace

import jev_decision3_cuda_synthetic_system_bootstrap as boot
import jev_decision3_cuda_synthetic_system_launcher as launcher
import pytest


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def sources(tmp_path, monkeypatch):
    root = tmp_path / "root"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    monkeypatch.setattr(boot, "ROOT", root)
    monkeypatch.setattr(boot, "SCRIPTS", scripts)
    monkeypatch.setattr(boot, "SOURCE_ONLY", False)
    monkeypatch.setattr(boot, "sys", SimpleNamespace(modules={}))
    monkeypatch.setattr(boot, "SystemControl", lambda *_: SimpleNamespace())
    pins = {}
    for name in boot.FILES:
        held, released = scripts / boot.FILES[name], boot._release(name)
        source = ("SOURCE_ONLY = True\nlabel = " + repr(name) + "\n").encode()
        target = source.replace(b"SOURCE_ONLY = True", b"SOURCE_ONLY = False")
        for path, raw in ((held, source), (released, target)):
            path.write_bytes(raw)
            pins[str(path)] = digest(raw)
    for _name, relative in boot.SUPPORT:
        path = scripts / relative
        raw = ("value = " + repr(relative) + "\n").encode()
        path.write_bytes(raw)
        pins[str(path)] = digest(raw)
    for _name, relative, _package in boot.PROJECT:
        path = root / "src/s1" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = ("value = " + repr(relative) + "\n").encode()
        path.write_bytes(raw)
        pins[str(path)] = digest(raw)
    monkeypatch.setattr(boot, "_plan", lambda *_: {"source_pins": pins})
    monkeypatch.setattr(boot, "_source_set", lambda: set(pins))
    return pins


def test_launcher_and_bootstrap_use_the_same_fixed_allowlist(monkeypatch):
    import jev_decision3_candidate_http_owner as common
    import jev_decision3_cpu_experiment_worker as cpu

    monkeypatch.setattr(launcher, "cpu", cpu)
    monkeypatch.setattr(launcher, "common", common)
    assert boot._source_set() == launcher._source_set()


def test_held_bootstrap_rejects_without_reading_plan(monkeypatch):
    monkeypatch.setattr(boot, "SOURCE_ONLY", True)
    monkeypatch.setattr(boot, "_plan", lambda *_: pytest.fail("must not read"))
    with pytest.raises(ValueError, match="bootstrap release"):
        boot.prepare("unused", "unused", mode="launcher")


def test_bootstrap_verifies_all_selected_bytes_before_executing_any(sources, monkeypatch):
    pins = sources
    calls = []
    original = boot._read

    def read(path, expected, **kwargs):
        calls.append(("read", str(path)))
        return original(path, expected, **kwargs)

    def module(name, path, data, **_kwargs):
        assert sum(kind == "read" for kind, _ in calls) == len(pins)
        calls.append(("execute", name))
        if name == "jev_decision3_cuda_synthetic_launcher":
            return SimpleNamespace(_source_set=lambda: set(pins), _plan=lambda *_: None)
        return SimpleNamespace(SOURCE_ONLY=False)

    monkeypatch.setattr(boot, "_read", read)
    monkeypatch.setattr(boot, "_module", module)
    entry = boot.prepare("synthetic", "sha", mode="supervisor")
    assert entry.SOURCE_ONLY is False
    assert calls[0][0] == "read" and calls[-1] == ("execute", "_cuda_synthetic_entry")
    assert set(path for kind, path in calls if kind == "read") == set(pins)


def test_tampered_last_source_prevents_all_workspace_execution(sources, monkeypatch):
    path = boot.ROOT / "src/s1/evaluation/decision3.py"
    path.write_bytes(b"tampered")
    monkeypatch.setattr(boot, "_module", lambda *_args, **_kwargs: pytest.fail("must not execute"))
    with pytest.raises(ValueError, match="source bytes changed"):
        boot.prepare("synthetic", "sha", mode="launcher")


def test_release_gate_difference_cannot_hide_other_changes(sources, monkeypatch):
    path = boot._release("supervisor")
    path.write_bytes(b"SOURCE_ONLY = False\nextra = True\n")
    sources[str(path)] = digest(path.read_bytes())
    monkeypatch.setattr(boot, "_module", lambda *_args, **_kwargs: pytest.fail("must not execute"))
    with pytest.raises(ValueError, match="gate-only"):
        boot.prepare("synthetic", "sha", mode="supervisor")


def test_source_reader_rejects_symlinks_and_hardlinks_before_read(tmp_path):
    source = tmp_path / "plain.py"
    source.write_bytes(b"synthetic")
    alias = tmp_path / "alias.py"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="ordinary"):
        boot._read(alias, digest(b"synthetic"))
    hard = tmp_path / "hard.py"
    hard.hardlink_to(source)
    with pytest.raises(ValueError, match="bounded regular"):
        boot._read(source, digest(b"synthetic"))


def test_module_executes_verified_supplied_bytes_only(tmp_path, monkeypatch):
    name = "_synthetic_bootstrap_test"
    monkeypatch.setattr(boot, "sys", SimpleNamespace(modules={}))
    path = tmp_path / "source.py"
    path.write_bytes(b"value = 'unselected on disk'\n")
    module = boot._module(name, path, b"value = 'verified bytes'\n")
    assert module.value == "verified bytes" and boot.sys.modules[name] is module
    with pytest.raises(ValueError, match="fresh"):
        boot._module(name, path, b"value = 2\n")


def test_bad_plan_inventory_fails_before_any_source_reads(tmp_path, monkeypatch):
    import json

    base = tmp_path / "root"
    run = base / ("run-" + "a" * 32)
    run.mkdir(parents=True)
    path = run / "plan.json"
    plan = {
        "schema": "decision3-cuda-synthetic-system-plan-v1",
        "source_pins": {str(tmp_path / "unexpected.py"): "a" * 64},
    }
    raw = json.dumps(plan).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(boot, "BASE", base)
    monkeypatch.setattr(boot, "_source_set", lambda: {str(tmp_path / "approved.py")})
    with pytest.raises(ValueError, match="source inventory"):
        boot._plan(str(path), digest(raw))
    assert not (tmp_path / "unexpected.py").exists()


def test_full_synthetic_bootstrap_loads_verified_entry_and_injects_dependencies(
    sources, monkeypatch
):
    expected = set(sources)
    held = boot.SCRIPTS / boot.FILES["launcher"]
    released = boot._release("launcher")
    raw = (
        "SOURCE_ONLY = True\ndef _source_set(): return "
        + repr(expected)
        + "\ndef _plan(*args): return None\n"
    ).encode()
    held.write_bytes(raw)
    release_raw = raw.replace(b"SOURCE_ONLY = True", b"SOURCE_ONLY = False")
    released.write_bytes(release_raw)
    sources[str(held)], sources[str(released)] = digest(raw), digest(release_raw)
    entry = boot.prepare("synthetic", "sha", mode="launcher")
    assert entry.SOURCE_ONLY is False
    assert entry.common is boot.sys.modules["jev_decision3_candidate_http_owner"]
    assert entry.child is boot.sys.modules["jev_decision3_candidate_browser_child"]
    assert entry.cpu is boot.sys.modules["jev_decision3_cpu_experiment_worker"]
    assert boot.sys.modules["s1.evaluation.decision3"].value == "evaluation/decision3.py"


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    import json

    base = tmp_path / "runs"
    run = base / ("run-" + "a" * 32)
    run.mkdir(parents=True)
    root = tmp_path / "root-controlled"
    root.mkdir()
    path = root / ("jev-cuda-synthetic-system-broker-" + "a" * 32 + ".py")
    path.write_bytes(b"synthetic broker source")
    sha = digest(path.read_bytes())
    monkeypatch.setattr(boot, "BASE", base)
    monkeypatch.setattr(boot, "BROKER_ROOT", root)
    monkeypatch.setattr(boot, "_root_controlled", lambda _: True)
    monkeypatch.setattr(boot.os, "getuid", lambda: 1001)
    monkeypatch.setattr(boot.os, "geteuid", lambda: 1001)
    plan = str(run / "plan.json")
    plan_sha = "c" * 64
    control = boot.SystemControl(plan, plan_sha, sha)
    unit = "jev-cuda-synthetic-system-" + "a" * 32 + ".service"
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        action = args[-1]
        results = {
            "start": {"status": "started", "unit": unit, "plan_sha256": plan_sha},
            "show": {"InvocationID": "b" * 32},
            "stop": {
                "status": "stop_returned_closure_unconfirmed",
                "unit": unit,
                "invocation": "b" * 32,
            },
        }
        return SimpleNamespace(stdout=json.dumps(results[action]).encode(), stderr=b"")

    monkeypatch.setattr(boot.subprocess, "run", fake_run)
    return SimpleNamespace(
        control=control, path=path, plan=plan, plan_sha=plan_sha, unit=unit, calls=calls
    )


def test_broker_adapter_only_invokes_three_fixed_operations(adapter):
    a = adapter
    a.control.start(a.plan, a.plan_sha)
    assert a.control.show(a.unit)["InvocationID"] == "b" * 32
    hello = {"unit": a.unit, "invocation": "b" * 32, "plan_sha256": a.plan_sha}
    a.control.stop(a.unit, "b" * 32, hello)
    assert [args[-1] for args, _ in a.calls] == ["start", "show", "stop"]
    for args, kwargs in a.calls:
        assert args[:-1] == [
            "/usr/bin/sudo",
            "-n",
            "--",
            "/usr/bin/python3",
            "-I",
            "-B",
            str(a.path),
        ]
        assert kwargs["cwd"] == "/" and kwargs["env"] == {
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
        }
    timeouts = [kwargs["timeout"] for _, kwargs in a.calls]
    assert timeouts[0] == 12 and 0 < timeouts[1] <= 2 and timeouts[2] == 22


def test_adapter_rejects_other_plan_unit_and_unbound_stop_before_execution(adapter):
    a = adapter
    with pytest.raises(ValueError, match="fixed broker plan"):
        a.control.start(a.plan, "f" * 64)
    with pytest.raises(ValueError, match="fixed broker unit"):
        a.control.show("unrelated.service")
    with pytest.raises(ValueError, match="bound broker stop"):
        a.control.stop(a.unit, "b" * 32, {"unit": "unrelated.service"})
    assert a.calls == []


def test_adapter_rejects_broker_tamper_before_execution(adapter):
    adapter.path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source bytes changed"):
        adapter.control.start(adapter.plan, adapter.plan_sha)
    assert adapter.calls == []


def test_adapter_refuses_non_root_controlled_broker(adapter, monkeypatch):
    monkeypatch.setattr(boot, "_root_controlled", lambda _: False)
    with pytest.raises(ValueError, match="root-controlled"):
        adapter.control.show(adapter.unit)
    assert adapter.calls == []


def test_root_controlled_mode_checks():
    assert boot._root_controlled(SimpleNamespace(st_uid=0, st_mode=0o755))
    assert not boot._root_controlled(SimpleNamespace(st_uid=1001, st_mode=0o755))
    assert not boot._root_controlled(SimpleNamespace(st_uid=0, st_mode=0o775))


def test_prepare_requires_broker_sha_before_reading_supervisor_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "SOURCE_ONLY", False)
    monkeypatch.setattr(boot, "BASE", tmp_path)
    monkeypatch.setattr(boot, "_plan", lambda *_: pytest.fail("unexpected plan read"))
    path = str(tmp_path / ("run-" + "a" * 32) / "plan.json")
    with pytest.raises(ValueError, match="broker SHA"):
        boot.prepare(path, "c" * 64, mode="supervisor")


@pytest.mark.parametrize("mode", ["launcher", "supervisor"])
def test_actual_system_entry_bytes_are_verified_and_wired(sources, monkeypatch, mode):
    workspace = __import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"
    for role in boot.FILES:
        held = boot.SCRIPTS / boot.FILES[role]
        released = boot._release(role)
        source = (workspace / boot.FILES[role]).read_bytes()
        marker = b"SOURCE_ONLY = True"
        assert source.count(marker) == 1
        release = source.replace(marker, b"SOURCE_ONLY = False", 1)
        held.write_bytes(source)
        released.write_bytes(release)
        sources[str(held)] = digest(source)
        sources[str(released)] = digest(release)
    original = boot._module

    def load(name, path, raw, **kwargs):
        result = original(name, path, raw, **kwargs)
        if name == "jev_decision3_cuda_synthetic_launcher":
            result._source_set = lambda: set(sources)
            result._plan = lambda *_: None
        return result

    monkeypatch.setattr(boot, "_module", load)
    entry = boot.prepare("synthetic", "sha", mode=mode)
    assert entry.SOURCE_ONLY is False
    if mode == "supervisor":
        assert entry.launcher.FILES["launcher"] == "jev_decision3_cuda_synthetic_system_launcher.py"
        assert entry.system_control is not None
    else:
        assert entry.FILES["launcher"] == "jev_decision3_cuda_synthetic_system_launcher.py"


def test_broker_and_supervisor_construct_the_same_fixed_worker_command(monkeypatch):
    import jev_decision3_candidate_http_owner as common
    import jev_decision3_cuda_synthetic_system_broker as broker
    import jev_decision3_cuda_synthetic_system_supervisor as supervisor

    monkeypatch.setattr(supervisor, "common", common)
    monkeypatch.setattr(supervisor, "launcher", launcher)
    plan_sha = "c" * 64
    monkeypatch.setattr(broker, "PLAN_SHA256", plan_sha)
    nonce = "a" * 32
    unit = "jev-cuda-synthetic-system-" + nonce + ".service"
    path = broker.RUN_ROOT / ("run-" + nonce) / "plan.json"
    plan = {"unit": unit, "python": broker.PYTHON}
    broker_args = broker._command(unit, path)
    supervisor_args = supervisor._unit_args(plan, path, plan_sha)
    assert broker_args[0] == "/usr/bin/systemd-run"
    assert supervisor_args[0] == "systemd-run"
    assert broker_args[1:] == supervisor_args[1:]


def test_show_uses_caller_timeout_after_authentication(adapter, monkeypatch):
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr(boot.time, "monotonic", lambda: next(ticks))
    adapter.control.show(adapter.unit, timeout=1.5)
    assert adapter.calls[-1][1]["timeout"] == 1.25


def test_show_fails_closed_when_authentication_exhausts_timeout(adapter, monkeypatch):
    ticks = iter([10.0, 10.5])
    monkeypatch.setattr(boot.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="deadline"):
        adapter.control.show(adapter.unit, timeout=0.25)
    assert adapter.calls == []


def test_show_propagates_broker_process_timeout(adapter, monkeypatch):
    import subprocess

    def timed_out(args, **kwargs):
        assert 0 < kwargs["timeout"] <= 0.5
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(boot.subprocess, "run", timed_out)
    with pytest.raises(subprocess.TimeoutExpired):
        adapter.control.show(adapter.unit, timeout=0.5)
