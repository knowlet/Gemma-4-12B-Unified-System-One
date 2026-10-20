"""V2 policy/readback and bounded owned tiny-child exit regression; no framework or unit starts."""

import ast
import builtins
import copy
import hashlib
import importlib.util
import os
import runpy
import shutil
import subprocess
import sys
import time
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
    assert c.PROFILE != "CPU_REFERENCE_ONLY_V3" and c.GATE_SCHEMA.endswith("3g-v2")


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
        "process_rss",
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
        "tracked": [
            {
                **parent,
                "rss_bytes": 1234,
                "rss_state": "live",
                "proc_states": ["S", "S", "S"],
                "absence_checks": 0,
            }
        ],
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


def fake_process(tmp_path, *, state="S", rss=1024):
    proc = tmp_path / "EXPLICIT_FAKE_RSS_PROC"
    identity = {"pid": 998877, "start_ticks": 34567, "pgid": 998877, "sid": 998877}
    directory = proc / str(identity["pid"])
    directory.mkdir(parents=True)
    tail = [state, "1", str(identity["pgid"]), str(identity["sid"])] + ["0"] * 16
    tail[19] = str(identity["start_ticks"])
    (directory / "stat").write_text(str(identity["pid"]) + " (FAKE RSS) " + " ".join(tail))
    status = "Name:\tFAKE RSS\nState:\t" + state + " (explicit fixture)\n"
    if rss is not None:
        status += "VmRSS:\t" + str(rss) + " kB\n"
    (directory / "status").write_text(status)
    return proc, directory, identity


def old_helper():
    path = HERE.parent / "public-revisit-capacity-3g-20261020-01/cpu_prepare_form_3g.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "82c7add649276785b64c4fbe24e945e918f555cb2b6bcf96dd61997734bd5334"
    )
    spec = importlib.util.spec_from_file_location("frozen_cpu3g_v1_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_former_monitor_fails_same_identity_zombie_fixture_successor_records_zero(tmp_path):
    proc, _, identity = fake_process(tmp_path, state="Z", rss=None)
    with pytest.raises(ValueError, match="Unknown/reused tracked RSS identity"):
        old_helper().process_rss(identity, proc)
    row = c.process_rss(identity, proc)
    assert row == {
        **identity,
        "rss_bytes": 0,
        "rss_state": "zombie",
        "proc_states": ["Z", "Z", "Z"],
        "absence_checks": 0,
    }


def test_live_rss_is_retained_and_same_identity_is_required(tmp_path):
    proc, _, identity = fake_process(tmp_path)
    row = c.process_rss(identity, proc)
    assert row["rss_bytes"] == c.MIB and row["rss_state"] == "live"
    for field in ("start_ticks", "pgid", "sid"):
        wrong = {**identity, field: identity[field] + 1}
        with pytest.raises(ValueError, match="identity changed before"):
            c.process_rss(wrong, proc)


def test_live_missing_rss_and_fake_zombie_with_rss_refuse(tmp_path):
    proc, directory, identity = fake_process(tmp_path, rss=None)
    with pytest.raises(ValueError, match="Unknown tracked RSS state"):
        c.process_rss(identity, proc)
    (directory / "status").write_text("State:\tZ (zombie)\nVmRSS:\t0 kB\n")
    raw = (directory / "stat").read_text().replace(") S ", ") Z ")
    (directory / "stat").write_text(raw)
    with pytest.raises(ValueError, match="terminal zero"):
        c.process_rss(identity, proc)


def test_reaped_between_status_reads_requires_two_full_pid_absences(tmp_path, monkeypatch):
    proc, directory, identity = fake_process(tmp_path)
    original_read, original_stat = Path.read_text, Path.lstat
    checks = []

    def disappear(path, *args, **kwargs):
        if path == directory / "status":
            shutil.rmtree(directory)
            raise FileNotFoundError(path)
        return original_read(path, *args, **kwargs)

    def counted(path, *args, **kwargs):
        if path == directory:
            checks.append(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", disappear)
    monkeypatch.setattr(Path, "lstat", counted)
    row = c.process_rss(identity, proc)
    assert row == {
        **identity,
        "rss_bytes": 0,
        "rss_state": "absent",
        "proc_states": [],
        "absence_checks": 2,
    }
    assert len(checks) == 2


def test_missing_status_existing_pid_and_reappearing_pid_refuse(tmp_path, monkeypatch):
    proc, directory, identity = fake_process(tmp_path)
    (directory / "status").unlink()
    with pytest.raises(ValueError, match="PID directory still exists"):
        c.process_rss(identity, proc)
    shutil.rmtree(directory)
    original = Path.lstat
    checks = 0

    def reappearing(path, *args, **kwargs):
        nonlocal checks
        if path == directory:
            checks += 1
            if checks == 2:
                directory.mkdir()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", reappearing)
    with pytest.raises(ValueError, match="PID directory still exists"):
        c.process_rss(identity, proc)
    assert checks == 2


def test_permission_error_never_becomes_zero_rss(tmp_path, monkeypatch):
    proc, directory, identity = fake_process(tmp_path)
    original = Path.read_text

    def denied(path, *args, **kwargs):
        if path == directory / "status":
            raise PermissionError("EXPLICIT fake permission failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(PermissionError, match="EXPLICIT"):
        c.process_rss(identity, proc)


@pytest.mark.parametrize("state", ["zombie", "absent"])
def test_offline_terminal_zero_may_leave_unit_but_requires_closed_exact_identity(tmp_path, state):
    evidence, *_ = observation(tmp_path)
    row, parent, cleanup = monitor_blob(evidence)
    child = {
        "pid": 998877,
        "start_ticks": 34567,
        "pgid": 998877,
        "sid": 998877,
        "rss_bytes": 0,
        "rss_state": state,
        "proc_states": ["Z", "Z", "Z"] if state == "zombie" else [],
        "absence_checks": 0 if state == "zombie" else 2,
    }
    row["tracked"].append(child)
    cleanup["tracked_PID_start_ticks"][str(child["pid"])] = child["start_ticks"]
    cleanup["tracked_groups"].append(child["pgid"])
    assert child["pid"] not in evidence["processes"]
    assert (
        c.validate_monitor_evidence(c.encode(row) + b"\n", BOOT, parent=parent, cleanup=cleanup)
        == 1
    )
    cleanup["tracked_PID_start_ticks"][str(child["pid"])] += 1
    with pytest.raises(ValueError, match="owned closed"):
        c.validate_monitor_evidence(c.encode(row) + b"\n", BOOT, parent=parent, cleanup=cleanup)


@pytest.mark.parametrize(
    "patch",
    [
        {"rss_bytes": 1},
        {"proc_states": ["S", "Z", "Z"]},
        {"absence_checks": 2},
        {"rss_state": "unknown"},
        {"absence_checks": False},
    ],
)
def test_offline_forged_terminal_zero_refuses(patch):
    row = {
        "pid": 11,
        "start_ticks": 22,
        "pgid": 11,
        "sid": 11,
        "rss_bytes": 0,
        "rss_state": "zombie",
        "proc_states": ["Z", "Z", "Z"],
        "absence_checks": 0,
    }
    row.update(patch)
    with pytest.raises(ValueError):
        c.validate_rss_observation(row)


def test_monitor_parent_cannot_be_zombie_or_absent(tmp_path):
    evidence, *_ = observation(tmp_path)
    row, parent, cleanup = monitor_blob(evidence)
    row["tracked"][0].update(rss_bytes=0, rss_state="zombie", proc_states=["Z"] * 3)
    row["tracked_RSS_bytes"] = 0
    with pytest.raises(ValueError, match="parent remains live"):
        c.validate_monitor_evidence(c.encode(row) + b"\n", BOOT, parent=parent, cleanup=cleanup)


@pytest.mark.skipif(not hasattr(os, "WNOWAIT"), reason="Linux waitid WNOWAIT required")
def test_real_owned_tiny_child_zombie_and_reap_without_unit_or_framework():
    child = subprocess.Popen(
        [sys.executable, "-B", "-s", "-c", "pass"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5
        info = None
        while info is None and time.monotonic() < deadline:
            info = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
            if info is None:
                time.sleep(0.005)
        assert info is not None, "Owned tiny child must exit within5s"
        raw = (Path("/proc") / str(child.pid) / "stat").read_text()
        tail = raw[raw.rfind(")") + 2 :].split()
        identity = {
            "pid": child.pid,
            "start_ticks": int(tail[19]),
            "pgid": int(tail[2]),
            "sid": int(tail[3]),
        }
        assert tail[0] == "Z"
        with pytest.raises(ValueError, match="Unknown/reused tracked RSS identity"):
            old_helper().process_rss(identity)
        assert c.process_rss(identity)["rss_state"] == "zombie"
        assert child.wait(timeout=5) == 0
        assert c.process_rss(identity)["rss_state"] == "absent"
    finally:
        # The only spawned child is a bounded Python `pass`; wait reaps only it.
        child.wait(timeout=5)
