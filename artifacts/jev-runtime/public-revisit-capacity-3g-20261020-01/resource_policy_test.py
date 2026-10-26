"""Pure3GiB policy/readback fixtures; no actor, real cgroup or framework changes."""

import ast
import builtins
import copy
import hashlib
import importlib.util
import os
import runpy
from pathlib import Path

import pytest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("cpu3g_tested", HERE / "cpu_prepare_form_3g.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
BOOT = "11111111-2222-4333-8444-555555555555"
BINDING = {"boot_id": BOOT, "cgroup_path": c.EXPECTED_CGROUP}


@pytest.fixture(autouse=True)
def pure_only(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "transformers",
            "tokenizers",
            "numpy",
            "peft",
            "httpx",
            "playwright",
        }:
            pytest.fail("Pure resource test imported runtime: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(c, "_UNIT_ANCHOR", None)


def fake_unit(tmp_path):
    proc, mount = tmp_path / "EXPLICIT_FAKE_PROC", tmp_path / "EXPLICIT_FAKE_CGROUP"
    pid = os.getpid()
    process = proc / str(pid)
    process.mkdir(parents=True)
    boot = proc / "sys/kernel/random/boot_id"
    boot.parent.mkdir(parents=True)
    boot.write_text(BOOT + "\n")
    (process / "cgroup").write_text("0::" + c.EXPECTED_CGROUP + "\n")
    (process / "mountinfo").write_text(f"1 0 0:1 / {mount} rw - cgroup2 cgroup rw\n")
    tail = ["S", "1", "111", "222"] + ["0"] * 16
    tail[19] = "33333"
    (process / "stat").write_text(f"{pid} (EXPLICIT FAKE) " + " ".join(tail) + "\n")
    directory = mount / c.EXPECTED_CGROUP.lstrip("/")
    directory.mkdir(parents=True)
    files = {
        "memory.max": str(3 * c.GIB),
        "memory.high": "max",
        "memory.current": "1024",
        "memory.peak": "2048",
        "memory.swap.max": "0",
        "memory.swap.current": "0",
        "pids.current": "1",
        "cgroup.procs": str(pid),
        "memory.events": "\n".join(k + " 0" for k in sorted(c.UNIT_EVENTS)),
        "cgroup.events": "populated 1\nfrozen 0",
        "cgroup.stat": "nr_descendants 0\nnr_dying_descendants 0",
    }
    for name, value in files.items():
        (directory / name).write_text(value + "\n")
    return proc, mount, directory


def observation(tmp_path):
    proc, mount, directory = fake_unit(tmp_path)
    evidence = c.assert_current_unit(BINDING, proc=proc, cgroup_root=mount)
    return evidence, proc, mount, directory


def test_same_reserve_new3g_formula_and_unchanged_as_limits():
    obs = {"MemTotal": 130595860480, "MemAvailable": 31000000000, "boot_id": BOOT}
    formula = c.resource_formula(obs, BOOT, prospective=True)
    assert formula == {
        "reserve_bytes": 26119172096,
        "required_bytes": 29340397568,
        "original_CPU_floor_bytes": 30414139392,
        "swap_or_resident_credit_bytes": 0,
    }
    assert c.resource_formula(obs, BOOT, prospective=False)["required_bytes"] == 26119172096
    assert c.limit_target("parent") == (2 * c.GIB, 8 * c.GIB)
    assert c.limit_target("worker") == (8 * c.GIB, 8 * c.GIB)
    assert c.LIMITS["tracked_parent_plus_owned_RSS_bytes"] == 3 * c.GIB
    assert c.PROFILE != "CPU_REFERENCE_ONLY_V3" and c.GATE_SCHEMA.endswith("3g-v1")


def test_all_original_nonresource_helpers_are_identical_ast():
    old = (
        c.ROOT
        / "artifacts/jev-runtime/owned-form-progress-collection-admission-20261019-01/cpu_prepare_form_v3.py"
    )
    raw = old.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "b9b8c7a04464c9e5c41c7a704f6b1a46ae0de7fe1ee80d28adf9f208959bdf55"
    )
    before = {
        n.name: ast.dump(n, include_attributes=False)
        for n in ast.parse(raw).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    after = {
        n.name: ast.dump(n, include_attributes=False)
        for n in ast.parse(Path(c.__file__).read_bytes()).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    changed = {
        "CPUReferenceGate",
        "validate_gate_evidence",
        "OwnedMemoryMonitor",
        "validate_monitor_evidence",
    }
    assert all(after[name] == body for name, body in before.items() if name not in changed)
    assert all(
        name in before
        for name in (
            "load_processor",
            "capture_reference",
            "processor_evidence",
            "compile_references_v2",
        )
    )


def test_readback_and_pure_offline_validator_need_no_current_env(tmp_path, monkeypatch):
    evidence, *_ = observation(tmp_path)
    for key in ("JEV_CPU3G_CGROUP", "JEV_CPU3G_BOOT"):
        monkeypatch.delenv(key, raising=False)
    before = copy.deepcopy(evidence)
    assert c.validate_unit_evidence(evidence, BINDING) is evidence
    assert evidence == before and evidence["memory_max_bytes"] == 3 * c.GIB


@pytest.mark.parametrize(
    "filename,value",
    [
        ("memory.max", "max"),
        ("memory.max", str(10 * 1024**3)),
        ("memory.high", "1"),
        ("memory.swap.max", "1"),
        ("memory.swap.current", "1"),
        ("memory.current", str(3 * 1024**3 + 1)),
        ("memory.events", "low 0\nhigh 0\nmax 1\noom 0\noom_kill 0\noom_group_kill 0"),
        ("memory.events", "low 0\nhigh 0\nmax 0\noom 1\noom_kill 1\noom_group_kill 0"),
        ("cgroup.events", "populated 1\nfrozen 1"),
        ("cgroup.stat", "nr_descendants 1\nnr_dying_descendants 0"),
        ("cgroup.procs", "999999999"),
        ("pids.current", "0"),
    ],
)
def test_wrong_or_missing_hardcap_readback_refuses(tmp_path, filename, value):
    proc, mount, directory = fake_unit(tmp_path)
    (directory / filename).write_text(value + "\n")
    with pytest.raises((ValueError, KeyError)):
        c.assert_current_unit(BINDING, proc=proc, cgroup_root=mount)


def test_wrong_membership_mount_symlink_and_unit_inode_refuse(tmp_path, monkeypatch):
    evidence, proc, mount, directory = observation(tmp_path)
    monkeypatch.setattr(
        c, "_UNIT_ANCHOR", (*c.unit_identity(evidence)[:-1], evidence["cgroup_inode"] + 1)
    )
    with pytest.raises(ValueError, match="inode changed"):
        c.assert_current_unit(BINDING, proc=proc, cgroup_root=mount)
    monkeypatch.setattr(c, "_UNIT_ANCHOR", None)
    (proc / str(os.getpid()) / "cgroup").write_text("0::/unrelated.service\n")
    with pytest.raises(ValueError, match="exact sameboot"):
        c.assert_current_unit(BINDING, proc=proc, cgroup_root=mount)
    (proc / str(os.getpid()) / "cgroup").write_text("0::" + c.EXPECTED_CGROUP + "\n")
    target = directory / "memory.max"
    target.unlink()
    target.symlink_to(directory / "memory.current")
    with pytest.raises(ValueError, match="Ordinary"):
        c.assert_current_unit(BINDING, proc=proc, cgroup_root=mount)


@pytest.mark.parametrize(
    "key,value",
    [
        ("memory_max_bytes", True),
        ("memory_peak_bytes", 3 * 1024**3 + 1),
        ("pids_current", False),
        ("observed_monotonic", float("nan")),
        ("cgroup_inode", 0),
    ],
)
def test_offline_unit_evidence_rejects_typed_edges(tmp_path, key, value):
    evidence, *_ = observation(tmp_path)
    evidence[key] = value
    with pytest.raises(ValueError):
        c.validate_unit_evidence(evidence, BINDING)


def monitor_blob(evidence):
    parent = copy.deepcopy(evidence["caller"])
    obs = {"MemTotal": 130595860480, "MemAvailable": 31000000000, "boot_id": BOOT}
    row = {
        "schema_id": c.MONITOR_SCHEMA,
        "profile": c.PROFILE,
        "sequence": 0,
        "monotonic": 1.0,
        "expected_boot_id": BOOT,
        "admitted": True,
        "error": None,
        "tracked": [{**parent, "rss_bytes": 1234}],
        "tracked_RSS_bytes": 1234,
        "observation": obs,
        **c.resource_formula(obs, BOOT, prospective=False),
        "parent_limits": {"role": "parent", "soft_bytes": 2 * c.GIB, "hard_bytes": 8 * c.GIB},
        "unit": evidence,
    }
    cleanup = {
        "confirmed": True,
        "remaining": [],
        "tracked_PID_start_ticks": {},
        "tracked_groups": [],
    }
    return row, parent, cleanup


def test_closed_monitor_accepts_unit_keyword_or_pure_fixed_boot_default(tmp_path, monkeypatch):
    evidence, *_ = observation(tmp_path)
    row, parent, cleanup = monitor_blob(evidence)
    monkeypatch.setattr(
        c, "assert_current_unit", lambda *_: pytest.fail("Offline validator queried live cgroup")
    )
    blob = c.encode(row) + b"\n"
    assert c.validate_monitor_evidence(blob, BOOT, parent=parent, cleanup=cleanup) == 1
    assert (
        c.validate_monitor_evidence(blob, BOOT, parent=parent, cleanup=cleanup, unit=BINDING) == 1
    )
    row["tracked_RSS_bytes"] = 3 * c.GIB + 1
    row["tracked"][0]["rss_bytes"] = row["tracked_RSS_bytes"]
    with pytest.raises(ValueError):
        c.validate_monitor_evidence(c.encode(row) + b"\n", BOOT, parent=parent, cleanup=cleanup)


def test_monitor_rejects_cross_unit_caller_and_hardcap_event(tmp_path):
    evidence, *_ = observation(tmp_path)
    row, parent, cleanup = monitor_blob(evidence)
    row["unit"]["caller"]["start_ticks"] += 1
    with pytest.raises(ValueError, match="exact unit and parent"):
        c.validate_monitor_evidence(c.encode(row) + b"\n", BOOT, parent=parent, cleanup=cleanup)
    row["unit"]["caller"]["start_ticks"] -= 1
    row["unit"]["memory_events"]["oom_kill"] = 1
    with pytest.raises(ValueError, match="OOM"):
        c.validate_monitor_evidence(c.encode(row) + b"\n", BOOT, parent=parent, cleanup=cleanup)


def test_helper_standalone_cli_cannot_launch_uncapped_form_worker(monkeypatch):
    monkeypatch.setattr(
        c.subprocess, "Popen", lambda *_a, **_k: pytest.fail("Helper CLI started actor")
    )
    with pytest.raises(SystemExit, match="Component only"):
        runpy.run_path(c.__file__, run_name="__main__")
