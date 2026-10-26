"""Pure fake owner/probe tests only; no units, actors, GPU or real helper."""

import hashlib
import importlib.util
import json
import py_compile
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("_q95_owner_test", HERE / "q95_owner.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
entry_spec = importlib.util.spec_from_file_location("_q95_entry_test", HERE / "launch_q95.py")
ENTRY = importlib.util.module_from_spec(entry_spec)
entry_spec.loader.exec_module(ENTRY)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(M.encode(value))


@pytest.fixture
def selected(tmp_path, monkeypatch):
    root, own, frozen = tmp_path / "repo", tmp_path / "repo/owner", tmp_path / "repo/frozen"
    own.mkdir(parents=True)
    frozen.mkdir()
    monkeypatch.setattr(M, "ROOT", root)
    monkeypatch.setattr(M, "HERE", own)
    monkeypatch.setattr(M, "FROZEN", frozen)
    source, assembly = {}, {}
    for i in range(119):
        p = frozen / f"source-{i}.py"
        p.write_text("# fixture only\n")
        assembly[p.name] = {k: v for k, v in M.pin(p).items() if k != "path"}
        if i < 95:
            source[p.name] = assembly[p.name]
    runtime = {}
    for i in range(348):
        p = root / "runtime" / f"runtime-{i}.py"
        p.parent.mkdir(exist_ok=True)
        p.write_text("# fixture only\n")
        runtime[str(p)] = {k: v for k, v in M.pin(p).items() if k != "path"}
    support = frozen / "support"
    support.mkdir()
    refs = {}
    for key, value in (
        ("execution", {"files": source}),
        ("runtime", {"files": runtime}),
        ("qualification", {}),
        ("design", {}),
        ("inputs", {}),
    ):
        p = support / (key + ".json")
        write(p, value)
        refs[key] = {**M.pin(p), "path": p.relative_to(root).as_posix()}
    tensor = frozen / "model.safetensors"
    tensor.write_bytes(b"synthetic fixture; no model payload")
    s = tensor.stat()
    statrow = {
        "device": s.st_dev,
        "inode": s.st_ino,
        "size_bytes": s.st_size,
        "mtime_ns": s.st_mtime_ns,
    }
    tensor_pin = {k: v for k, v in M.pin(tensor).items() if k != "path"}
    serve = {
        "source_root": str(frozen),
        "cwd": str(frozen),
        "source_count": 95,
        "runtime_file_count": 348,
        "source_files": {**assembly, tensor.name: tensor_pin},
        **refs,
        "argv": ["/fixture/python"],
        "environment_updates": {},
        "environment_unset": [],
    }
    runtime_dir = root / "artifacts/jev-runtime"
    components, fixed = {}, {}
    values = {
        "boundary": {},
        "observer": {},
        "checker": {},
        "serve": serve,
        "assembly": {"files": assembly},
        "tensor": {
            "source": str(tensor),
            "target": str(tensor),
            "source_after": statrow,
            "target_after": statrow,
            "expected_payload_pin": tensor_pin,
        },
        "errors": {},
    }
    for name, value in values.items():
        p = runtime_dir / (name + ".json")
        write(p, value)
        components[name] = M.pin(p)
        fixed[name] = (p.relative_to(runtime_dir).as_posix(), components[name]["sha256"])
    monkeypatch.setattr(M, "FIXED", fixed)
    for key, name in (("owner", "q95_owner.py"), ("entry", "launch_q95.py")):
        p = own / name
        p.write_text("# fixture source only\n")
        components[key] = M.pin(p)
    plan = {
        "profile": M.PROFILE,
        "unit": M.UNIT,
        "nonce": "jev-d3-boundary-" + "a" * 32,
        "boot_id": "fixture-boot",
        "mem_total_bytes": 130595864576,
        "created_monotonic": 0,
        "expires_monotonic": 3600,
        "cgroup": "/fixture-cgroup",
        "output": str(own / "owner-01"),
        "components": components,
        "tensor_stat": M.stamp(tensor),
    }
    path = own / "plan.json"
    write(path, plan)

    def seal():
        write(path, plan)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return SimpleNamespace(
        plan=plan, path=path, seal=seal, serve=serve, root=root, own=own, frozen=frozen
    )


def test_default_dry_does_not_load_or_start_anything(monkeypatch, capsys):
    monkeypatch.setattr(
        ENTRY, "selected", lambda *_: pytest.fail("No source or runtime load in dry mode")
    )
    assert ENTRY.main([]) == 0
    assert json.loads(capsys.readouterr().out)["unit_started"] is False


@pytest.mark.parametrize("argv", [["--run"], ["--request-close"], ["--request-close", "--actor"]])
def test_actual_modes_need_explicit_plan_and_sha(argv):
    with pytest.raises(SystemExit):
        ENTRY.main(argv)


def test_complete_exact_population_and_tensor_stat_without_payload_read(selected, monkeypatch):
    tensor = selected.frozen / "model.safetensors"
    original = Path.open

    def no_tensor(self, *args, **kwargs):
        assert self != tensor
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_tensor)
    plan, serve, refs, _ = M.validate(selected.path, selected.seal())
    assert len(refs) >= 119 + 348
    assert str(tensor) not in {ref["path"] for ref in refs}
    assert M.check_population(refs)
    assert len(M.tensor_identity(plan)) == 2
    assert serve["source_files"] == selected.serve["source_files"]


@pytest.mark.parametrize(
    "fault",
    [
        "sha",
        "profile",
        "unit",
        "nonce",
        "memory",
        "freshness",
        "output",
        "component",
        "source",
        "tensor",
    ],
)
def test_selected_plan_or_population_drift_refuses(selected, fault):
    plan = selected.plan
    if fault == "profile":
        plan["profile"] = "other"
    elif fault == "unit":
        plan["unit"] = "foreign.service"
    elif fault == "nonce":
        plan["nonce"] = "bad"
    elif fault == "memory":
        plan["mem_total_bytes"] = True
    elif fault == "freshness":
        plan["expires_monotonic"] = 3601
    elif fault == "output":
        plan["output"] = str(selected.root / "other/owner-01")
    elif fault == "component":
        plan["components"]["errors"]["sha256"] = "0" * 64
    elif fault == "source":
        (selected.frozen / "source-50.py").write_text("drift")
    elif fault == "tensor":
        plan["tensor_stat"][2] += 1
    digest = selected.seal()
    with pytest.raises(Exception):
        M.validate(selected.path, "0" * 64 if fault == "sha" else digest)


def test_selected_hour_expiry_accepts_exact_addition_without_subtraction_cancellation(selected):
    selected.plan["created_monotonic"] = 13688.74941763
    selected.plan["expires_monotonic"] = selected.plan["created_monotonic"] + 3600
    assert selected.plan["expires_monotonic"] - selected.plan["created_monotonic"] > 3600
    assert M.validate(selected.path, selected.seal())[0] == selected.plan


def test_source_monitor_is_stat_only_and_refuses_replacement(selected, monkeypatch):
    _, _, refs, _ = M.validate(selected.path, selected.seal())
    stamps = M.check_population(refs)
    monkeypatch.setattr(M, "pin", lambda *_: pytest.fail("No full hash per monitor"))
    assert M.check_population(refs, expected_stamps=stamps) == stamps
    (selected.frozen / "source-1.py").write_text("changed")
    with pytest.raises(ValueError, match="metadata changed"):
        M.check_population(refs, expected_stamps=stamps)


@pytest.mark.parametrize("where", ["src/s1", "scripts"])
def test_valid_unselected_project_bytecode_refuses_full_and_monitor_checks(selected, where):
    _, _, refs, _ = M.validate(selected.path, selected.seal())
    stamps = M.check_population(refs)
    source = selected.frozen / where / "unselected.py"
    source.parent.mkdir(parents=True)
    source.write_text("raise RuntimeError('Must never execute')\n")
    py_compile.compile(str(source), doraise=True)
    assert list(source.parent.rglob("*.pyc"))
    with pytest.raises(ValueError, match="Cached project bytecode"):
        M.check_population(refs)
    with pytest.raises(ValueError, match="Cached project bytecode"):
        M.check_population(refs, expected_stamps=stamps)


def test_pin_streams_multiple_chunks_without_read_bytes(tmp_path, monkeypatch):
    p = tmp_path / "large"
    blob = b"chunk" * 600000
    p.write_bytes(blob)
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("Bounded streaming required"))
    assert M.pin(p)["sha256"] == hashlib.sha256(blob).hexdigest()


@pytest.mark.parametrize("initial", [False, True])
def test_host_floor_exact_boundary_and_no_helper_memcg_substitution(selected, initial):
    plan = selected.plan
    reserve = max(24 * M.GIB, (plan["mem_total_bytes"] + 4) // 5)
    available = reserve + (2 * M.BASE if initial else 2 * M.GIB)
    row = {
        "boot_id": plan["boot_id"],
        "mem_total_bytes": plan["mem_total_bytes"],
        "mem_available_bytes": available,
        "comfy_state": "absent_confirmed",
        "comfy_running": 0,
        "comfy_pending": 0,
    }
    probe = SimpleNamespace(observe=lambda **_: row)
    assert M.host(plan, probe, initial=initial) == row
    row["mem_available_bytes"] -= 1
    with pytest.raises(ValueError, match="host reserve"):
        M.host(plan, probe, initial=initial)


@pytest.mark.parametrize("fault", ["boot", "total", "busy", "unknown", "typed", "probe"])
def test_host_unknown_busy_identity_and_probe_errors_refuse(selected, fault):
    row = {
        "boot_id": selected.plan["boot_id"],
        "mem_total_bytes": selected.plan["mem_total_bytes"],
        "mem_available_bytes": selected.plan["mem_total_bytes"],
        "comfy_state": "http_idle",
        "comfy_running": 0,
        "comfy_pending": 0,
    }
    if fault == "boot":
        row["boot_id"] = "other"
    elif fault == "total":
        row["mem_total_bytes"] += 1
    elif fault == "busy":
        row.update(comfy_state="http_busy", comfy_running=1)
    elif fault == "unknown":
        row["comfy_state"] = "unknown"
    elif fault == "typed":
        row["comfy_pending"] = False

    def observe(**_):
        if fault == "probe":
            raise RuntimeError("probe failed")
        return row

    with pytest.raises(Exception):
        M.host(selected.plan, SimpleNamespace(observe=observe))


def test_observer_error_evidence_is_bounded_without_private_exception_message():
    error = RuntimeError("private credential must not appear")
    error.evidence = {"known": "x" * 70000}
    value = M.error_record(error)
    assert value["observation_evidence"]["truncated"] is True
    assert "private credential" not in json.dumps(value)


def test_close_request_requires_exact_selected_plan_and_nonce(selected):
    out = Path(selected.plan["output"])
    out.mkdir()
    assert M.close_requested(selected.plan, "digest") is False
    write(out / "close-request.json", {"plan_sha256": "digest", "nonce": "other"})
    with pytest.raises(ValueError):
        M.close_requested(selected.plan, "digest")
    write(out / "close-request.json", {"plan_sha256": "digest", "nonce": selected.plan["nonce"]})
    assert M.close_requested(selected.plan, "digest") is True


@pytest.mark.parametrize("fault", ["nonce", "invocation", "boot", "pid", "cap", "kernel", "state"])
def test_actor_admission_refuses_wrong_owner_or_readback(selected, monkeypatch, fault):
    plan = selected.plan
    actor = {
        "pid": 123,
        "start_ticks": 4,
        "pgid": 123,
        "sid": 123,
        "uid": 1000,
        "cgroup": plan["cgroup"],
        "state": "S",
    }
    out = Path(plan["output"])
    out.mkdir()
    invocation = "b" * 32
    write(
        out / "admission.json",
        {
            "plan_sha256": "digest",
            "nonce": plan["nonce"],
            "invocation_id": invocation,
            "actor": actor,
        },
    )
    props = {"MemoryMax": str(40 * M.GIB)}
    row = {
        **props,
        "InvocationID": invocation,
        "Description": plan["nonce"],
        "ControlGroup": plan["cgroup"],
        "MainPID": "123",
        "ActiveState": "active",
        "SubState": "running",
    }
    cg = selected.root / "cg" / plan["cgroup"].lstrip("/")
    cg.mkdir(parents=True)
    (cg / "memory.max").write_text(str(40 * M.GIB))
    (cg / "memory.swap.max").write_text("0")
    if fault == "nonce":
        row["Description"] = "other"
    elif fault == "invocation":
        row["InvocationID"] = "c" * 32
    elif fault == "pid":
        row["MainPID"] = "124"
    elif fault == "cap":
        row["MemoryMax"] = "1"
    elif fault == "kernel":
        (cg / "memory.swap.max").write_text("1")
    fresh_actor = {**actor, "state": "R"} if fault == "state" else actor
    owner = SimpleNamespace(
        status=lambda _: row,
        boot=lambda: "other" if fault == "boot" else plan["boot_id"],
        process=lambda _: fresh_actor,
        CGROUP_ROOT=selected.root / "cg",
    )
    monkeypatch.setattr(
        M, "selection", lambda *_: SimpleNamespace(cgroup=plan["cgroup"], properties=lambda: props)
    )
    monkeypatch.setattr(M.os, "getpid", lambda: 123)
    monkeypatch.setenv("JEV_D3_BOUNDARY_NONCE", plan["nonce"])
    if fault == "state":
        assert M.actor_admission(plan, "digest", owner)["actor"] == actor
    else:
        with pytest.raises(ValueError):
            M.actor_admission(plan, "digest", owner)


def test_helper_descriptor_seed_is_exact_public_contract_without_co_resident():
    fields = "binding worker listener service_directory private_config interpreter source_root qualification execution runtime registry expected_memtotal".split()
    identity = {key: {} for key in fields}
    identity.update(schema_id="evidence-only", authentication_admitted=False, scope="evidence only")
    seed = M.helper_selection_seed(
        identity, SimpleNamespace(SCHEMA="decision3-q95-live-helper-selection-v1")
    )
    assert set(seed) == set(fields) | {"schema_id"}
    assert seed["schema_id"] == "decision3-q95-live-helper-selection-v1"
    assert "co_resident" not in seed and "authentication_admitted" not in seed


@pytest.mark.parametrize("fault", ["probe", "cleanup", "final_host"])
def test_monitor_failure_closes_only_owned_unit_and_attempts_all_final_checks(
    selected, monkeypatch, fault
):
    plan, out = selected.plan, Path(selected.plan["output"])
    actor = {
        "pid": 123,
        "start_ticks": 4,
        "pgid": 123,
        "sid": 123,
        "uid": 1000,
        "cgroup": plan["cgroup"],
        "state": "S",
    }
    events, host_calls = [], []

    def save(directory, name, value):
        write(directory / name, value)

    def command(*_args, **_kwargs):
        save(out, "actor-intent.json", {"plan_sha256": "digest", "actor": actor})
        events.append("launched-only-selected-unit")

    class Boundary:
        def __init__(self, selected_unit, *_):
            assert selected_unit.unit == M.UNIT

        def verify(self):
            return {"actor": actor, "unit": {"InvocationID": "b" * 32}}

        def owned_status(self):
            return {"ActiveState": "active", "SubState": "running", "MainPID": "123"}

        def close(self, *, primary_error):
            assert primary_error["type"] == "RuntimeError"
            events.append("authenticated-owned-close")
            if fault == "cleanup":
                raise ValueError("Secondary close failed")
            return {
                "whole_unit_exit_confirmed": True,
                "observed_cgroup_cohort_exit_confirmed": True,
            }

    props = {"RuntimeMaxUSec": "3h", "TimeoutStopUSec": "20s"}
    unit = SimpleNamespace(
        unit=M.UNIT, nonce=plan["nonce"], cgroup=plan["cgroup"], properties=lambda: props
    )
    owner = SimpleNamespace(
        boot=lambda: plan["boot_id"],
        status=lambda _: {"LoadState": "not-found"},
        CGROUP_ROOT=selected.root / "absent-cgroup",
        UnitBoundary=Boundary,
        save=save,
        command=command,
    )
    monkeypatch.setattr(M, "selection", lambda *_: unit)
    monkeypatch.setattr(M, "load", lambda *_: SimpleNamespace())
    monkeypatch.setattr(M.time, "monotonic", lambda: 1)
    monkeypatch.setattr(M.time, "sleep", lambda _: None)

    def population(*_args, **_kwargs):
        events.append("population-final-or-initial")
        return {}

    monkeypatch.setattr(M, "check_population", population)
    monkeypatch.setattr(M, "tensor_identity", lambda _: events.append("tensor-stat-check"))

    def host(*_args, **_kwargs):
        host_calls.append(1)
        if len(host_calls) == 2 or (fault == "final_host" and len(host_calls) == 3):
            error = RuntimeError("Primary host probe failure")
            error.evidence = {"state": "unknown"}
            raise error
        return {"known": True}

    monkeypatch.setattr(M, "host", host)
    assert M.controller_main(plan, {"argv": ["/fixture/python"]}, [], b"{}", "digest", owner) == 1
    result = json.loads((out / "owner-final.json").read_bytes())
    assert result["primary_error"]["type"] == "RuntimeError"
    assert result["primary_error"]["observation_evidence"] == {"state": "unknown"}
    assert events.count("authenticated-owned-close") == 1
    assert events.count("population-final-or-initial") == 2 and len(host_calls) == 3
    assert result["completed"] is False
