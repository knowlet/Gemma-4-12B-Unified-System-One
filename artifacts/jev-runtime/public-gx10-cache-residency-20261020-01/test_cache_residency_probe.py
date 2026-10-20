"""Pure synthetic backend tests. No checkpoint mmap/mincore/payload is executed."""

import importlib.util
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

BASE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "cache_residency_probe_under_test", BASE / "cache_residency_probe.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.fixture
def plan():
    return json.loads((BASE / "plan-01.json").read_text())


class FakeOps:
    def __init__(self, *, fail=None, unmap_fail=False, short=False, flags=1):
        self.calls = []
        self.fail, self.unmap_fail, self.short, self.flags = fail, unmap_fail, short, flags

    def map(self, fd, offset, length):
        self.calls.append(("map", fd, offset, length))
        if self.fail == "map":
            raise OSError("fake-map-primary")
        return 4096

    def residency(self, address, length, pages):
        self.calls.append(("mincore", address, length, pages))
        if self.fail == "mincore":
            raise OSError("fake-mincore-primary")
        return bytes([self.flags]) * (pages - int(self.short))

    def unmap(self, address, length):
        self.calls.append(("unmap", address, length))
        if self.unmap_fail:
            raise OSError("fake-unmap-secondary")


def scan(ops, *, gate=None, record=None, clock=lambda: 0):
    gate = gate or (lambda stage, chunk: {"sequence": chunk})
    return probe.scan_fd(
        91, ops=ops, gate_step=gate, record=record or (lambda _: None), deadline=1, clock=clock
    )


def test_default_dry_does_not_bind_or_open_checkpoint(monkeypatch, capsys):
    def prohibited(*args, **kwargs):
        raise AssertionError("dry path attempted an effect")

    monkeypatch.setattr(probe, "LibcMincore", prohibited)
    monkeypatch.setattr(probe, "inspect_checkpoint", prohibited)
    monkeypatch.setattr(probe, "verify_sources", prohibited)
    assert probe.main([]) == 0
    assert json.loads(capsys.readouterr().out)["actual_measurement"] is False


def test_explicit_run_requires_owned_output_and_both_reviewed_hashes():
    with pytest.raises(ValueError, match="two reviewed SHA256"):
        probe.main(["--run"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("snapshot", "/foreign-checkpoint"),
        ("canonical_blob", probe.SNAPSHOT),
        ("checkpoint_size_bytes", probe.CHECKPOINT_BYTES - 1),
        ("chunk_bytes", probe.CHUNK + 4096),
        ("page_size", 65536),
        ("vector_cap_bytes", 131072),
        ("wall_seconds", 121),
        ("output_cap_bytes", 1024**3),
        ("swap_or_resident_credit_bytes", 1),
        ("swap_or_resident_credit_bytes", False),
    ],
)
def test_fixed_plan_refuses_scope_or_budget_changes(plan, key, value):
    plan[key] = value
    with pytest.raises(ValueError):
        probe.fixed_plan(plan)


def test_fixed_plan_refuses_boolean_inode_metadata(plan):
    plan["checkpoint_metadata"]["inode"] = True
    with pytest.raises(ValueError, match="Strict integer"):
        probe.fixed_plan(plan)


def test_window_partition_covers_exact_file_without_aliases_or_overlap():
    windows = list(probe.windows())
    assert len(windows) == 90
    assert windows[0][0] == 0
    assert sum(length for _, length, _ in windows) == probe.CHECKPOINT_BYTES
    previous = 0
    for offset, length, pages in windows:
        assert offset == previous and offset % 4096 == 0
        assert 0 < length <= 256 * 1024**2
        assert pages == (length + 4095) // 4096 and pages <= 65536
        previous += length
    assert windows[-1][1] % 4096 != 0


def test_all_resident_counts_partial_tail_as_exact_file_bytes_and_unmaps():
    ops, records, gates = FakeOps(), [], []

    def gate(stage, chunk):
        gates.append((stage, chunk))
        return {"sequence": chunk + 2}

    result = scan(ops, gate=gate, record=records.append)
    assert result["resident_file_bytes"] == probe.CHECKPOINT_BYTES
    assert result["resident_page_bytes"] == ((probe.CHECKPOINT_BYTES + 4095) // 4096) * 4096
    assert len(records) == len(gates) == result["chunks"] == 90
    assert all(
        row["mapping_unmapped"] and row["gate_sequence"] == row["chunk"] + 2 for row in records
    )
    assert [call[0] for call in ops.calls] == ["map", "mincore", "unmap"] * 90


def test_only_low_bit_defines_residency():
    result = scan(FakeOps(flags=0xFE))
    assert result["resident_pages"] == result["resident_file_bytes"] == 0


def test_refused_gate_prevents_mapping():
    ops = FakeOps()

    def gate(*args):
        raise ValueError("fresh CPU refused")

    with pytest.raises(ValueError, match="fresh CPU refused"):
        scan(ops, gate=gate)
    assert ops.calls == []


def test_deadline_stops_before_gate_and_mapping():
    ops = FakeOps()
    with pytest.raises(ValueError, match="wall time"):
        scan(ops, clock=lambda: 2)
    assert ops.calls == []


def test_failed_map_has_no_unmap_or_retry():
    ops = FakeOps(fail="map")
    with pytest.raises(OSError, match="fake-map-primary"):
        scan(ops)
    assert [c[0] for c in ops.calls] == ["map"]


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_mincore_primary_is_preserved_and_mapping_cleanup_attempted_once(cleanup_failure):
    ops = FakeOps(fail="mincore", unmap_fail=cleanup_failure)
    with pytest.raises(OSError, match="fake-mincore-primary") as captured:
        scan(ops)
    assert [c[0] for c in ops.calls] == ["map", "mincore", "unmap"]
    assert bool(getattr(captured.value, "__notes__", [])) == cleanup_failure


def test_incomplete_vector_is_unknown_and_mapping_is_unmapped():
    ops = FakeOps(short=True)
    with pytest.raises(ValueError, match="Complete residency"):
        scan(ops)
    assert [c[0] for c in ops.calls] == ["map", "mincore", "unmap"]


def test_unmap_failure_refuses_snapshot_without_recording_success():
    ops, rows = FakeOps(unmap_fail=True), []
    with pytest.raises(OSError, match="fake-unmap-secondary"):
        scan(ops, record=rows.append)
    assert rows == [] and [c[0] for c in ops.calls] == ["map", "mincore", "unmap"]


@pytest.fixture
def synthetic_checkpoint(tmp_path, monkeypatch, plan):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    blob = tmp_path / probe.BLOB
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"synthetic metadata fixture only")
    snapshot = tmp_path / probe.SNAPSHOT
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(blob)
    metadata = plan["checkpoint_metadata"]

    class FileAPI:
        euid = metadata["uid"]
        changed_fd = False
        open_mismatch = False

        def __init__(self):
            self.calls = []
            self.fstats = 0

        def _stat(self, changed=False):
            x = metadata
            return SimpleNamespace(
                st_dev=x["dev"],
                st_ino=x["inode"] + int(changed),
                st_size=x["size_bytes"],
                st_uid=x["uid"],
                st_gid=x["gid"],
                st_mode=stat.S_IFREG | x["mode"],
                st_nlink=x["nlink"],
                st_mtime_ns=x["mtime_ns"],
                st_ctime_ns=x["ctime_ns"],
            )

        def stat(self, path, *, follow_symlinks):
            assert path == blob and follow_symlinks is False
            return self._stat()

        def geteuid(self):
            return self.euid

        def open(self, path, flags):
            self.calls.append(("open", path, flags))
            assert flags == os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
            return 91

        def fstat(self, fd):
            assert fd == 91
            self.fstats += 1
            return self._stat(self.open_mismatch or (self.changed_fd and self.fstats > 1))

        def close(self, fd):
            self.calls.append(("close", fd))

    return plan, FileAPI()


def inspect(plan, api, ops=None):
    return probe.inspect_checkpoint(
        plan,
        gate_step=lambda *_: {"sequence": 0},
        ops=ops or FakeOps(),
        record=lambda _: None,
        file_api=api,
        clock=lambda: 0,
    )


def test_inode_owner_required_even_when_synthetic_backend_would_report_every_page(
    synthetic_checkpoint,
):
    plan, api = synthetic_checkpoint
    api.euid += 1
    ops = FakeOps()
    with pytest.raises(ValueError, match="Inode owner"):
        inspect(plan, api, ops)
    assert api.calls == [] and ops.calls == []


def test_open_descriptor_mismatch_closes_without_mapping(synthetic_checkpoint):
    plan, api = synthetic_checkpoint
    api.open_mismatch = True
    ops = FakeOps()
    with pytest.raises(ValueError, match="Opened descriptor"):
        inspect(plan, api, ops)
    assert [c[0] for c in api.calls] == ["open", "close"] and ops.calls == []


def test_final_descriptor_drift_refuses_completed_snapshot_but_closes(synthetic_checkpoint):
    plan, api = synthetic_checkpoint
    api.changed_fd = True
    with pytest.raises(ValueError, match="metadata stable"):
        inspect(plan, api)
    assert [c[0] for c in api.calls] == ["open", "close"]


def test_hardlink_aliases_are_not_added_and_descriptor_is_closed(synthetic_checkpoint):
    plan, api = synthetic_checkpoint
    result = inspect(plan, api)
    assert result["inode_count"] == 1 and result["aliases_not_added"] == 2
    assert result["queried_file_bytes"] == probe.CHECKPOINT_BYTES and result["descriptor_closed"]
    assert [c[0] for c in api.calls] == ["open", "close"]


def test_measurement_failure_also_closes_owned_descriptor(synthetic_checkpoint):
    plan, api = synthetic_checkpoint
    with pytest.raises(OSError, match="fake-mincore-primary"):
        inspect(plan, api, FakeOps(fail="mincore"))
    assert [c[0] for c in api.calls] == ["open", "close"]


def test_native_wrapper_uses_only_prot_none_private_and_kernel_vector(monkeypatch):
    calls = []

    class Function:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args):
            calls.append((self.name, *args[:2]))
            if self.name == "mmap":
                assert args == (None, 8192, 0, 2, 91, 4096)
                return 4096
            if self.name == "mincore":
                args[2][0], args[2][1] = 0x81, 0xFE
            return 0

    libc = SimpleNamespace(**{name: Function(name) for name in ["mmap", "mincore", "munmap"]})
    monkeypatch.setattr(probe.ctypes, "CDLL", lambda *a, **kw: libc)
    ops = probe.LibcMincore()
    address = ops.map(91, 4096, 8192)
    assert ops.residency(address, 8192, 2) == b"\x81\xfe"
    ops.unmap(address, 8192)
    assert [c[0] for c in calls] == ["mmap", "mincore", "munmap"]


def test_original_cpu_gate_exact_threshold_and_refusal_are_durable(tmp_path, plan):
    Gate = probe.original_gate_class(plan)
    total = 130595860480
    threshold = max(24 * 1024**3, (total + 4) // 5) + 4 * 1024**3
    observation = {"MemTotal": total, "MemAvailable": threshold, "boot_id": "synthetic-boot"}
    path = tmp_path / "original-gate.jsonl"
    gate = Gate(path, "synthetic-boot", observe=lambda: dict(observation))
    row = gate.check(probe.CPU_PHASE, "cpu")
    assert row["required_bytes"] == threshold and row["admitted"]
    observation["MemAvailable"] -= 1
    with pytest.raises(ValueError, match="memory admission refused"):
        gate.check(probe.CPU_PHASE, "cpu")
    persisted = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["admitted"] for r in persisted] == [True, False]
    assert [r["sequence"] for r in persisted] == [0, 1]
    assert all(
        type(r["swap_or_resident_credit_bytes"]) is int and r["swap_or_resident_credit_bytes"] == 0
        for r in persisted
    )


def test_original_cpu_gate_refuses_other_boot_and_persists_refusal(tmp_path, plan):
    Gate = probe.original_gate_class(plan)
    path = tmp_path / "wrong-boot.jsonl"
    gate = Gate(
        path,
        "expected-boot",
        observe=lambda: {
            "MemTotal": 130595860480,
            "MemAvailable": 121000000000,
            "boot_id": "different-boot",
        },
    )
    with pytest.raises(ValueError, match="memory admission refused"):
        gate.check(probe.CPU_PHASE, "cpu")
    record = json.loads(path.read_text())
    assert record["admitted"] is False and record["swap_or_resident_credit_bytes"] == 0


def test_dependency_drift_is_refused_before_original_import(plan, monkeypatch):
    plan["source_dependencies"][probe.DEPENDENCIES[0]]["sha256"] = "0" * 64
    monkeypatch.setattr(
        probe.importlib.util, "spec_from_file_location", lambda *a: pytest.fail("drift imported")
    )
    with pytest.raises(ValueError, match="source drift before import"):
        probe.original_gate_class(plan)


def test_evidence_reserve_refuses_growth_before_terminal_budget_exhausted(tmp_path):
    (tmp_path / "owned.bin").write_bytes(b"x" * (probe.OUTPUT_CAP - probe.TERMINAL_RESERVE))
    with pytest.raises(ValueError, match="terminal reserve"):
        probe.write_json(tmp_path, "measurement.json", {"data": "small"})
    assert not (tmp_path / "measurement.json.next").exists()


def test_orchestrator_persists_unknown_on_probe_failure_without_retry(tmp_path, plan, monkeypatch):
    monkeypatch.setattr(probe, "BASE", tmp_path)
    monkeypatch.setattr(probe, "verify_sources", lambda *a: {"synthetic": True})
    monkeypatch.setattr(probe, "host_metrics", lambda: {"boot_id": plan["expected_boot_id"]})
    monkeypatch.setattr(probe, "process_identity", lambda: {"pid": 1, "start_ticks": 2, "pgid": 3})

    class Gate:
        sequence = 0

        def __init__(self, *a):
            pass

        def check(self, *a):
            self.sequence += 1
            return {
                "admitted": True,
                "swap_or_resident_credit_bytes": 0,
                "sequence": self.sequence - 1,
            }

    monkeypatch.setattr(probe, "original_gate_class", lambda _: Gate)
    monkeypatch.setattr(probe, "LibcMincore", FakeOps)
    attempts = []

    def fail(*a, **kw):
        attempts.append(1)
        raise OSError("synthetic protected failure")

    monkeypatch.setattr(probe, "inspect_checkpoint", fail)
    output = tmp_path / "actual-synthetic"
    assert probe.run(plan, output, "synthetic-source", "synthetic-plan") == 2
    terminal = json.loads((output / "terminal.json").read_text())
    assert terminal["status"] == "UNKNOWN" and terminal["complete"] is False
    assert terminal["error"]["message"] == "synthetic protected failure" and attempts == [1]
    assert terminal["host_before"]["boot_id"] == terminal["host_after"]["boot_id"]
