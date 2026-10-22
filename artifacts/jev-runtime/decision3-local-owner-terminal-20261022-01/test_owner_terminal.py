"""Pure metadata fixtures only; no real units, processes, models or network."""

import ast
import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "decision3-local-owner-boundary-20261022-01"


def load(name, path, expected=None):
    blob = path.read_bytes()
    if expected is not None:
        assert hashlib.sha256(blob).hexdigest() == expected
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(blob, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


original_tests = load(
    "_selected_original_boundary_pure_tests",
    PARENT / "test_owner_boundary.py",
    "c4f7b1296ca3689078b45faf3dfec3d5b4f3be7c949aafc6239712b1a316f2e6",
)
owner = load("_selected_terminal_owner", HERE / "owner_boundary.py")
original_tests.owner = owner
for name, value in vars(original_tests).items():
    if name == "setup" or (
        name.startswith("test_")
        and name != "test_controller_refuses_preexisting_output_before_launch"
    ):
        globals()[name] = value


def natural_terminal(setup, *, active="failed", sub="failed", control_group=""):
    boundary = setup["boundary"]
    boundary.verify()
    setup["rows"].clear()
    shutil.rmtree(setup["group"])
    setup["state"].update(
        MainPID="0", ActiveState=active, SubState=sub, ControlGroup=control_group, ExecMainPID="101"
    )
    return boundary


@pytest.mark.parametrize("active,sub", [("failed", "failed"), ("inactive", "dead")])
@pytest.mark.parametrize("retained_cgroup", [False, True])
def test_already_terminal_owned_identity_needs_no_signal_or_stop(
    setup, active, sub, retained_cgroup
):
    boundary = natural_terminal(
        setup,
        active=active,
        sub=sub,
        control_group=setup["selection"].cgroup if retained_cgroup else "",
    )
    primary = {"type": "ValueError", "message": "actual earlier refusal"}
    result = boundary.close(primary_error=primary)
    assert setup["calls"] == []
    assert result["closure_mode"] == "already_terminal_verified"
    assert result["stop_required"] is False
    assert result["stop_result"] is None
    assert len(result["already_terminal_identity_samples"]) == 6
    assert result["already_terminal_identity_sandwich_verified"] is True
    assert result["whole_unit_exit_confirmed"] is True
    assert result["observed_cgroup_cohort_exit_confirmed"] is True
    assert result["primary_error"] == primary
    for key in [
        "continuation_authorized",
        "whole_tree_exit_confirmed",
        "no_escape_enforced",
        "model_qualified",
    ]:
        assert result[key] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("Description", "foreign"),
        ("InvocationID", "c" * 32),
        ("InvocationID", ""),
        ("MainPID", "101"),
        ("ExecMainPID", "999"),
        ("ControlGroup", "/foreign"),
        ("ActiveState", "active"),
        ("SubState", "running"),
        ("LoadState", "not-found"),
        ("MemoryMax", "infinity"),
        ("MemorySwapMax", "1"),
    ],
)
def test_terminal_identity_mismatch_never_authorizes_stop_or_closure(setup, field, value):
    boundary = natural_terminal(setup)
    setup["state"][field] = value
    result = boundary.close()
    assert setup["calls"] == []
    assert result["whole_unit_exit_confirmed"] is False
    assert result["already_terminal_identity_sandwich_verified"] is False


@pytest.mark.parametrize(
    "missing", ["readback", "actor", "invocation", "tracked", "saved_readback"]
)
def test_terminal_requires_complete_persisted_prior_live_authority(setup, missing):
    boundary = natural_terminal(setup)
    if missing == "saved_readback":
        (setup["output"] / "unit-readback.json").write_text("{}")
    elif missing == "tracked":
        boundary.tracked.clear()
    else:
        setattr(boundary, missing, None)
    result = boundary.close()
    assert setup["calls"] == []
    assert result["whole_unit_exit_confirmed"] is False


@pytest.mark.parametrize("kind", ["invocation", "boot", "cgroup", "source", "unknown", "matches"])
def test_terminal_scan_sandwich_detects_drift_and_missing_absence(setup, monkeypatch, kind):
    boundary = natural_terminal(setup)
    scan = boundary.scan_absence
    calls = 0

    def drift():
        nonlocal calls
        calls += 1
        row = scan()
        if calls == 1:
            if kind == "invocation":
                setup["state"]["InvocationID"] = "c" * 32
            elif kind == "boot":
                monkeypatch.setattr(owner, "boot", lambda: "new-boot")
            elif kind == "cgroup":
                setup["group"].mkdir()
            elif kind == "source":
                (setup["here"] / "owner_boundary.py").write_text("drift")
            elif kind == "unknown":
                row["unknown"] = [{"pid": "303", "type": "PermissionError"}]
            else:
                row["matches"] = [{"pid": 303, "start_ticks": 33}]
        return row

    monkeypatch.setattr(boundary, "scan_absence", drift)
    result = boundary.close()
    assert setup["calls"] == []
    assert result["whole_unit_exit_confirmed"] is False


def test_live_status_and_resource_proofs_are_identical_to_frozen_parent():
    parent = ast.parse((PARENT / "owner_boundary.py").read_bytes())
    successor = ast.parse((HERE / "owner_boundary.py").read_bytes())
    old = next(
        node
        for node in parent.body
        if isinstance(node, ast.ClassDef) and node.name == "UnitBoundary"
    )
    new = next(
        node
        for node in successor.body
        if isinstance(node, ast.ClassDef) and node.name == "UnitBoundary"
    )
    for name in ["__init__", "owned_status", "members", "verify", "scan_absence"]:
        left = next(
            node for node in old.body if isinstance(node, ast.FunctionDef) and node.name == name
        )
        right = next(
            node for node in new.body if isinstance(node, ast.FunctionDef) and node.name == name
        )
        assert ast.dump(left) == ast.dump(right)
    old_selection = next(
        node for node in parent.body if isinstance(node, ast.ClassDef) and node.name == "Selection"
    )
    new_selection = next(
        node
        for node in successor.body
        if isinstance(node, ast.ClassDef) and node.name == "Selection"
    )
    assert ast.dump(old_selection) == ast.dump(new_selection)
