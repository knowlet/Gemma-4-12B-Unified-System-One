"""Pure local fixtures: no real systemd, Comfy, GPU, model or network calls."""

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, SOURCE / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


owner = load("_test_d3_owner", "owner_boundary.py")
controller = load("_test_d3_controller", "run_probe.py")
service = load("_test_d3_probe", "probe_service.py")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    here = root / "artifacts/jev-runtime/decision3-local-owner-boundary-20261022-01"
    here.mkdir(parents=True)
    for module in (owner, controller):
        monkeypatch.setattr(module, "ROOT", root)
        monkeypatch.setattr(module, "HERE", here)
    cgroot, proc = tmp_path / "cgroups", tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(owner, "CGROUP_ROOT", cgroot)
    monkeypatch.setattr(owner, "PROC", proc)
    monkeypatch.setattr(owner, "boot", lambda: "selected-boot")
    selection = owner.Selection(
        controller.UNIT, "jev-d3-boundary-" + "a" * 32, "selected-boot", 64 * 1024**2, 30, 2
    )
    group = cgroot / selection.cgroup.lstrip("/")
    group.mkdir(parents=True)
    for p, text in {
        "memory.max": str(64 * 1024**2),
        "memory.swap.max": "0",
        "cgroup.type": "domain",
        "cgroup.procs": "101\n303\n",
    }.items():
        (group / p).write_text(text)
    for directory in (group.parent, group.parent.parent):
        (directory / "cgroup.procs").write_text("")
    output = here / "probe-01"
    output.mkdir()
    for name in ("owner_boundary.py", "run_probe.py", "probe_service.py"):
        (here / name).write_text("fake code; never imported or executed")
    refs = [
        owner.pin(here / name) for name in ("owner_boundary.py", "run_probe.py", "probe_service.py")
    ]
    rows = {
        101: {
            "pid": 101,
            "start_ticks": 11,
            "pgid": 101,
            "sid": 101,
            "uid": os.getuid(),
            "state": "S",
            "cgroup": selection.cgroup,
        },
        303: {
            "pid": 303,
            "start_ticks": 33,
            "pgid": 303,
            "sid": 303,
            "uid": os.getuid(),
            "state": "S",
            "cgroup": selection.cgroup,
        },
    }
    for pid in rows:
        (proc / str(pid)).mkdir()

    def process(pid):
        if pid not in rows:
            raise FileNotFoundError(pid)
        return rows[pid].copy()

    monkeypatch.setattr(owner, "process", process)
    state = {
        **selection.properties(),
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "InvocationID": "b" * 32,
        "ControlGroup": selection.cgroup,
        "MainPID": "101",
        "ExecMainPID": "101",
        "Result": "success",
    }
    monkeypatch.setattr(owner, "status", lambda _selection: state.copy())
    calls = []

    def command(*argv, **kwargs):
        calls.append((argv, kwargs))
        assert argv == ("systemctl", "--user", "stop", selection.unit)
        rows.clear()
        shutil.rmtree(group)
        state.update(MainPID="0", ActiveState="inactive", SubState="dead", ControlGroup="")
        return ""

    monkeypatch.setattr(owner, "command", command)
    monkeypatch.setattr(owner.time, "sleep", lambda _seconds: None)
    boundary = owner.UnitBoundary(selection, output, refs)
    return {
        "root": root,
        "here": here,
        "group": group,
        "proc": proc,
        "output": output,
        "selection": selection,
        "refs": refs,
        "state": state,
        "rows": rows,
        "calls": calls,
        "boundary": boundary,
    }


def test_kernel_membership_covers_detached_markerless_child(setup):
    value = setup["boundary"].verify()
    assert [r["pid"] for r in value["cohort"]["members"]] == [101, 303]
    assert value["cohort"]["members"][1]["sid"] != value["actor"]["sid"]
    assert value["migration"]["no_escape_enforced"] is False


def test_complete_owned_close_has_no_generic_continuation(setup):
    boundary = setup["boundary"]
    boundary.verify()
    result = boundary.close()
    assert result["whole_unit_exit_confirmed"] is True
    assert result["observed_cgroup_cohort_exit_confirmed"] is True
    assert len(result["tracked_cohort"]) == 2
    assert len(result["two_process_scans"]) == 2
    for key in (
        "no_escape_enforced",
        "whole_tree_exit_confirmed",
        "continuation_authorized",
        "model_qualified",
    ):
        assert result[key] is False
    assert boundary.close() is result
    assert len(setup["calls"]) == 1
    assert setup["calls"][0][1] == {"timeout": 7}


@pytest.mark.parametrize(
    "field,value", [("Description", "foreign"), ("InvocationID", ""), ("ControlGroup", "/other")]
)
def test_wrong_unit_auth_never_stops(setup, field, value):
    setup["state"][field] = value
    result = setup["boundary"].close()
    assert setup["calls"] == []
    assert result["whole_unit_exit_confirmed"] is False
    assert result["closure_errors"]


def test_changed_invocation_never_stops(setup):
    setup["boundary"].verify()
    setup["state"]["InvocationID"] = "c" * 32
    result = setup["boundary"].close()
    assert setup["calls"] == []
    assert result["whole_unit_exit_confirmed"] is False


def test_changed_boot_never_stops(setup, monkeypatch):
    setup["boundary"].verify()
    monkeypatch.setattr(owner, "boot", lambda: "other-boot")
    result = setup["boundary"].close()
    assert setup["calls"] == []
    assert result["whole_unit_exit_confirmed"] is False
    assert "observation_error" in result["two_process_scans"][0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("KillMode", "process"),
        ("MemoryMax", "infinity"),
        ("MemorySwapMax", "1"),
        ("RuntimeMaxUSec", "infinity"),
        ("Description", "wrong"),
    ],
)
def test_property_readback_refuses_mismatch(setup, field, value):
    setup["state"][field] = value
    with pytest.raises(ValueError):
        setup["boundary"].verify()
    assert setup["calls"] == []


@pytest.mark.parametrize(
    "field,value", [("memory.max", "max"), ("memory.swap.max", "max"), ("cgroup.type", "threaded")]
)
def test_kernel_readback_refuses_mismatch(setup, field, value):
    (setup["group"] / field).write_text(value)
    with pytest.raises(ValueError, match="kernel"):
        setup["boundary"].verify()


def test_actor_must_have_actual_own_session(setup):
    setup["rows"][101]["sid"] = 1
    with pytest.raises(ValueError, match="session"):
        setup["boundary"].verify()


def test_unverified_unit_cannot_claim_closure(setup):
    result = setup["boundary"].close()
    assert len(setup["calls"]) == 1
    assert result["whole_unit_exit_confirmed"] is False


def test_source_drift_does_not_skip_owned_cleanup(setup):
    setup["boundary"].verify()
    (setup["here"] / "owner_boundary.py").write_text("changed")
    result = setup["boundary"].close()
    assert len(setup["calls"]) == 1
    assert result["sources_unchanged"] is False
    assert result["whole_unit_exit_confirmed"] is False


def test_final_inventory_error_preserves_primary_and_stops_own_unit(setup, monkeypatch):
    setup["boundary"].verify()

    def broken():
        raise PermissionError("fixture")

    monkeypatch.setattr(setup["boundary"], "members", broken)
    result = setup["boundary"].close(primary_error={"type": "TimeoutError", "message": "primary"})
    assert len(setup["calls"]) == 1
    assert result["primary_error"]["message"] == "primary"
    assert result["closure_errors"][0]["stage"] == "final_cohort"
    assert result["whole_unit_exit_confirmed"] is False


def test_reauthentication_after_inventory_never_stops_replacement(setup, monkeypatch):
    setup["boundary"].verify()

    def replaced():
        setup["state"]["InvocationID"] = "c" * 32
        return {"members": [], "vanished": []}

    monkeypatch.setattr(setup["boundary"], "members", replaced)
    result = setup["boundary"].close()
    assert setup["calls"] == []
    assert result["whole_unit_exit_confirmed"] is False


@pytest.mark.parametrize("kind", ["unknown", "matches", "failure"])
def test_missing_process_proof_never_claims_complete(setup, monkeypatch, kind):
    setup["boundary"].verify()

    def scan():
        if kind == "failure":
            raise PermissionError("fixture")
        return {
            "matches": [303] if kind == "matches" else [],
            "unknown": [303] if kind == "unknown" else [],
        }

    monkeypatch.setattr(setup["boundary"], "scan_absence", scan)
    result = setup["boundary"].close()
    assert result["whole_unit_exit_confirmed"] is False


def test_surviving_cgroup_refuses_closure(setup, monkeypatch):
    setup["boundary"].verify()
    monkeypatch.setattr(owner, "command", lambda *a, **kw: "")
    setup["state"].update(MainPID="0", ActiveState="inactive")
    result = setup["boundary"].close()
    assert result["cgroup_absent"] is False
    assert result["whole_unit_exit_confirmed"] is False


def test_unloaded_owned_unit_after_stop_is_supported(setup, monkeypatch):
    setup["boundary"].verify()
    stop = owner.command

    def unload(*argv, **kwargs):
        result = stop(*argv, **kwargs)
        setup["state"].update(
            LoadState="not-found",
            Description=setup["selection"].unit,
            InvocationID="",
            ControlGroup="",
        )
        return result

    monkeypatch.setattr(owner, "command", unload)
    assert setup["boundary"].close()["whole_unit_exit_confirmed"] is True


@pytest.mark.parametrize(
    "unit",
    [
        "jevbench-gx10-20261022-02.service",
        "ComfyUI.service",
        "../jev-d3-foo.service",
        "jev-d3-foo.service;bad",
    ],
)
def test_only_d3_unit_namespace(unit):
    with pytest.raises(ValueError, match="namespace"):
        owner.Selection(unit, "jev-d3-boundary-" + "a" * 32, "boot", 64 * 1024**2, 30, 2)


@pytest.mark.parametrize("change", ["empty", "absolute", "traversal", "duplicate", "extra"])
def test_source_identity_population_rejects_bad_refs(setup, change):
    refs = copy.deepcopy(setup["refs"])
    if change == "empty":
        refs = []
    elif change == "absolute":
        refs[0]["path"] = str(setup["here"] / "owner_boundary.py")
    elif change == "traversal":
        refs[0]["path"] = "../outside"
    elif change == "duplicate":
        refs.append(refs[0])
    else:
        refs[0]["trusted"] = True
    with pytest.raises(ValueError):
        owner.sources_match(refs)


def test_symlink_source_refused(setup):
    source = setup["here"] / "owner_boundary.py"
    target = setup["here"] / "other.py"
    target.write_text(source.read_text())
    source.unlink()
    source.symlink_to(target)
    with pytest.raises(ValueError, match="Ordinary"):
        owner.sources_match(setup["refs"])


def test_evidence_is_exclusive_and_rejects_nan(setup):
    owner.save(setup["output"], "example.json", {"a": 1})
    with pytest.raises(FileExistsError):
        owner.save(setup["output"], "example.json", {"a": 2})
    with pytest.raises(ValueError):
        owner.save(setup["output"], "nan.json", {"a": float("nan")})


def test_existing_boundary_evidence_refuses_reuse(setup):
    owner.save(setup["output"], "unit-readback.json", {})
    with pytest.raises(ValueError, match="reuse"):
        owner.UnitBoundary(setup["selection"], setup["output"], setup["refs"])


def test_failed_readback_write_cannot_qualify_closure(setup, monkeypatch):
    save = owner.save

    def fail_readback(directory, name, value):
        if name == "unit-readback.json":
            raise OSError("fixture failed persistence")
        return save(directory, name, value)

    monkeypatch.setattr(owner, "save", fail_readback)
    with pytest.raises(OSError, match="persistence"):
        setup["boundary"].verify()
    assert setup["boundary"].readback is None
    assert setup["boundary"].close()["whole_unit_exit_confirmed"] is False


@pytest.mark.parametrize("publisher", [owner.save, service.publish])
def test_receipt_visible_only_after_complete_content(setup, monkeypatch, publisher):
    link = os.link

    def publication(source, target):
        assert not target.exists()
        assert json.loads(source.read_bytes()) == {"ready": True}
        assert source.stat().st_mode & 0o777 == 0o600
        link(source, target)

    monkeypatch.setattr(os, "link", publication)
    publisher(setup["output"], "atomic.json", {"ready": True})
    assert json.loads((setup["output"] / "atomic.json").read_bytes()) == {"ready": True}


def test_default_controller_and_entry_are_dry_without_import_or_commands(monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry performed side effect")

    monkeypatch.setattr(controller, "load_plan", forbidden)
    monkeypatch.setattr(controller, "load_owner", forbidden)
    monkeypatch.setattr(service, "load_plan", forbidden)
    monkeypatch.setattr(service.os, "fork", forbidden)
    assert controller.main([]) == 0
    assert service.main([]) == 0
    assert "DRY" in capsys.readouterr().out


def test_explicit_execution_requires_plan_before_owner_import(monkeypatch):
    monkeypatch.setattr(controller, "load_owner", lambda: pytest.fail("owner imported"))
    with pytest.raises(ValueError, match="plan"):
        controller.main(["--run"])


def test_missing_or_wrong_sealed_digest_reads_no_owner(setup):
    path = setup["here"] / "plan.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        controller.load_plan(path, None)
    with pytest.raises(ValueError, match="bounded"):
        controller.load_plan(path, "0" * 64)


def test_scan_detects_recorded_process_after_cgroup_migration(setup):
    setup["boundary"].verify()
    setup["rows"][303]["cgroup"] = "/migrated"
    rows = setup["boundary"].scan_absence()["matches"]
    assert {r["pid"] for r in rows} == {101, 303}


def test_pid_reuse_does_not_match_old_identity_or_other_session(setup):
    setup["boundary"].verify()
    setup["rows"][303].update(start_ticks=444, cgroup="/other")
    rows = setup["boundary"].scan_absence()["matches"]
    assert {r["pid"] for r in rows} == {101}


def test_streaming_source_hash_never_calls_read_bytes(setup, monkeypatch):
    path = setup["root"] / "large-fixture"
    data = b"chunk" * (1024**2)
    path.write_bytes(data)
    monkeypatch.setattr(Path, "read_bytes", lambda _p: pytest.fail("whole-file read"))
    ref = owner.pin(path)
    assert ref["sha256"] == hashlib.sha256(data).hexdigest()
    assert ref["size_bytes"] == len(data)


def test_controller_refuses_preexisting_output_before_launch(setup):
    plan = {
        "profile": owner.PROFILE,
        "unit": controller.UNIT,
        "memory_max_bytes": 64 * 1024**2,
        "runtime_seconds": 30,
        "stop_seconds": 2,
        "output": str(setup["output"]),
        "interpreter": str(Path(sys.executable).resolve()),
        "sources": setup["refs"],
    }
    path = setup["here"] / "plan.json"
    blob = json.dumps(plan).encode()
    path.write_bytes(blob)
    with pytest.raises(ValueError, match="overwrite"):
        controller.load_plan(path, hashlib.sha256(blob).hexdigest())
