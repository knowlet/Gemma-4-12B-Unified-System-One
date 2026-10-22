"""Explicit fake native/OS evidence, never CUDA, model construction or a unit."""

import builtins
import copy
import json
from types import SimpleNamespace

import pytest

from s1 import browser_native_resources as r

BOOT = "11111111-2222-4333-8444-555555555555"
ACTOR = {"pid": 12345, "start_ticks": 123, "pgid": 12345, "sid": 12345}


@pytest.fixture(autouse=True)
def no_framework_or_actor(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "tokenizers", "peft", "playwright"}:
            raise AssertionError("Fake resource test imported framework: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(r.subprocess, "Popen", lambda *_a, **_k: pytest.fail("Actor started"))


def host(available=118 * r.GIB, *, total=r.TOTAL, boot=BOOT):
    return {
        "boot_id": boot,
        "MemTotal": total,
        "MemAvailable": available,
        "MemFree": 40 * r.GIB,
        "Cached": 2 * r.GIB,
        "SReclaimable": 0,
        "SwapTotal": 0,
        "SwapFree": 0,
        "Hugepagesize": 2 * r.MIB,
        "HugePages_Total": 0,
        "HugePages_Free": 0,
        "Hugetlb": 0,
    }


def unit(_expected=r.EXPECTED_UNIT):
    invocation = "a" * 32
    return {
        "path": r.EXPECTED_UNIT,
        "device": 33,
        "inode": 123,
        "actor": ACTOR,
        "InvocationID": invocation,
        "memory_max": r.CAP,
        "memory_high": "max",
        "memory_current": 2 * r.GIB,
        "memory_peak": 3 * r.GIB,
        "swap_max": 0,
        "swap_current": 0,
        "events": {"low": 0, "high": 0, "max": 0, "oom": 0, "oom_kill": 0, "oom_group_kill": 0},
        "cgroup_events": {"populated": 1, "frozen": 0},
        "cgroup_stat": {"nr_descendants": 0, "nr_dying_descendants": 0},
        "members": [ACTOR["pid"]],
        "properties": r._expected_properties(r.EXPECTED_UNIT, ACTOR["pid"], invocation),
    }


def native(*, total=r.TOTAL, boot=BOOT):
    h = r.mechanism()
    metadata = json.loads(r._component("driver")[1])["cuda_link_origins"][0]
    fraction = float((r.CAP - 4 * r.MIB) / total)
    return {
        "identity": {
            "context_device": 0,
            "integrated": 1,
            "name": "NVIDIA GB10",
            "total_bytes": total,
            "uuid": r.EXPECTED_UUID.replace("-", ""),
            "boot_id": boot,
            "library_metadata": {
                k: metadata[k] for k in ("resolved", "size_bytes", "mtime_ns", "device", "inode")
            },
            "symbol_addresses": {k: 123456 + i for i, k in enumerate(h.UMA_SYMBOLS)},
            "load_mode": "exact-canonical-RTLD_NOLOAD",
            "pin_scope": "EXPLICIT_FAKE_METADATA_ONLY",
        },
        "allocator": {
            "uuid": r.EXPECTED_UUID,
            "name": "NVIDIA GB10",
            "total_bytes": total,
            "fraction": fraction,
            "backend": "native",
            "allowed_bytes_upper": int(fraction * total),
        },
        "cuda": {
            "free": 0,
            "total": total,
            "allocated": 24 * r.GIB,
            "reserved": 26 * r.GIB,
            "peak_allocated": 25 * r.GIB,
            "peak_reserved": 27 * r.GIB,
        },
    }


class FakeAPI:
    def __init__(self, observation):
        assert observation["MemAvailable"] >= r.floor(observation, loaded=False)["required_bytes"]
        self.releases = 0

    def snapshot(self, _host):
        return native(total=_host["MemTotal"], boot=_host["boot_id"])

    def release(self):
        self.releases += 1


def exercise(tmp_path, *, parent=True, steps=142, total=r.TOTAL, boot=BOOT):
    observed = host(total=total, boot=boot)
    observed["MemAvailable"] = r.floor(observed, loaded=False)["required_bytes"]
    g = r.Gate(
        tmp_path / "resource-gates.jsonl",
        boot,
        observer=lambda: copy.deepcopy(observed),
        unit_reader=unit,
        api_factory=FakeAPI,
    )
    g.check("before-any-project-framework-or-parent-payload", "cpu")
    if parent:
        g.check("immediately-before-legacy-parent-capture-and-Torch", "cpu")
    g.check("before-checkpoint-payload-hash-and-BF16-loader", "load")
    g.check("before-any-BF16-model-construction", "train")
    checks = []

    def checked_model():
        checks.append("EXPLICIT_FAKE_COMPLETE_MODEL_CHECK")
        observed["MemAvailable"] = r.floor(observed, loaded=True)["required_bytes"]

    g.mark_model_loaded(SimpleNamespace(quantization="none", max_context=8192), checked_model)
    if parent:
        g.check("before-parent-injection", "train")
    for split, count in (("train", 142), ("development", 37)):
        for i in range(count):
            g.check(f"before-BF16-processor-compile-{split}-{i}", "train")
    g.check("immediately-before-unchanged-train_actions-optimizer", "train")
    for phase, count in (
        ("before-native-train-forward", steps),
        ("before-native-development-forward", 37),
    ):
        for _ in range(count):
            g.check(phase, "train")
    g.check("after-native-training", "train")
    return g, checks


def test_floors_retain_full_reserve_and_no_residency_credit():
    h = host()
    assert r.floor(h, loaded=False) == {
        "reserve_bytes": 26119172096,
        "required_bytes": 116313485312,
        "old_train_required_bytes": 106438961088,
    }
    assert r.floor(h, loaded=True)["required_bytes"] == 77658779648
    assert r.CAP == 42949672960 and r.STOP == 42412802048


@pytest.mark.parametrize(
    "total,boot",
    [(r.TOTAL, BOOT), (r.TOTAL + 4096, "22222222-3333-4444-8555-666666666666")],
)
def test_physical_total_is_selected_for_each_boot_without_rewriting_raw_evidence(
    tmp_path, total, boot
):
    g, _ = exercise(tmp_path, steps=1, total=total, boot=boot)
    result = r.validate_evidence(
        g.path.read_bytes(),
        expected_unit=r.EXPECTED_UNIT,
        expected_boot_id=boot,
        steps=1,
        parent_present=True,
    )
    rows = [json.loads(x) for x in g.path.read_bytes().splitlines()]
    assert g.expected_memtotal == total
    assert all(row["observation"]["MemTotal"] == total for row in rows)
    assert all(
        row["native"]["identity"]["total_bytes"]
        == row["native"]["allocator"]["total_bytes"]
        == row["native"]["cuda"]["total"]
        == total
        for row in rows
        if row["native"] is not None
    )
    assert (
        result["startup_required_bytes"]
        == r.floor(host(total=total), loaded=False)["required_bytes"]
    )
    assert r.CAP == 40 * r.GIB and r.STOP == r.CAP - r.GIB // 2


def test_live_host_total_cannot_drift_within_the_selected_boot(tmp_path):
    observed = host()
    g = r.Gate(
        tmp_path / "gate.jsonl",
        BOOT,
        observer=lambda: observed,
        unit_reader=unit,
        api_factory=FakeAPI,
    )
    g.check("before-any-project-framework-or-parent-payload", "cpu")
    observed["MemTotal"] += 4096
    with pytest.raises(ValueError, match="selected total"):
        g.check("immediately-before-legacy-parent-capture-and-Torch", "cpu")
    assert g.expected_memtotal == r.TOTAL
    assert json.loads(g.path.read_text().splitlines()[-1])["admitted"] is False


@pytest.mark.parametrize("part", ["identity", "allocator", "cuda"])
def test_native_totals_must_all_equal_the_selected_raw_host_total(part):
    total = r.TOTAL + 4096
    n = native(total=total)
    n[part]["total" if part == "cuda" else "total_bytes"] = r.TOTAL
    with pytest.raises(ValueError):
        r.validate_native(n, host(total=total), BOOT)


def test_offline_evidence_refuses_internally_consistent_total_change(tmp_path):
    g, _ = exercise(tmp_path, steps=1)
    rows = [json.loads(x) for x in g.path.read_bytes().splitlines()]
    for row in rows[1:]:
        observed = row["observation"]
        observed["MemTotal"] += 4096
        policy = r.floor(observed, loaded=row["loaded"])
        observed["MemAvailable"] = max(observed["MemAvailable"], policy["required_bytes"])
        row.update(policy)
        row["original_train_would_refuse"] = (
            observed["MemAvailable"] < policy["old_train_required_bytes"]
        )
        if row["native"] is not None:
            row["native"] = native(total=observed["MemTotal"])
    with pytest.raises(ValueError, match="selected total"):
        r.validate_evidence(
            b"\n".join(r.encode(row) for row in rows),
            expected_unit=r.EXPECTED_UNIT,
            expected_boot_id=BOOT,
            steps=1,
            parent_present=True,
        )


@pytest.mark.parametrize("loaded", [False, True])
def test_host_floor_one_byte_boundary_and_refusal_record(tmp_path, loaded):
    observed = host()
    observed["MemAvailable"] = r.floor(observed, loaded=loaded)["required_bytes"] - 1
    g = r.Gate(
        tmp_path / "gate.jsonl",
        BOOT,
        observer=lambda: observed,
        unit_reader=unit,
        api_factory=FakeAPI,
    )
    if loaded:
        # Explicit fake resident state only tests numeric gate; offline transition validation is separate.
        g.loaded = True
    with pytest.raises(ValueError, match="host floor"):
        g.check("before-any-project-framework-or-parent-payload", "cpu")
    row = json.loads(g.path.read_bytes())
    assert row["admitted"] is False and row["error"]["type"] == "ValueError"
    assert row["resident_credit_bytes"] == row["swap_credit_bytes"] == 0


@pytest.mark.parametrize("parent", [False, True])
def test_complete_all179_full142_37_profile_evidence(tmp_path, parent):
    g, checks = exercise(tmp_path, parent=parent)
    result = r.validate_evidence(
        g.path.read_bytes(),
        expected_unit=r.EXPECTED_UNIT,
        expected_boot_id=BOOT,
        steps=142,
        parent_present=parent,
    )
    assert result["rows"] == (366 if parent else 364)
    assert result["train_forwards"] == 142 and result["development_forwards"] == 37
    assert checks == ["EXPLICIT_FAKE_COMPLETE_MODEL_CHECK"]
    assert g.api.releases == 179 + 142 + 37 + 1
    evidence = g.evidence(steps=142, parent_present=parent)
    assert evidence["size_bytes"] == len(g.path.read_bytes())
    rows = [json.loads(x) for x in g.path.read_bytes().splitlines()]
    assert all(
        row["native"]["cuda"]["free"] == 0 and row["old_cuda_free_would_refuse"] is True
        for row in rows
        if row["native"] is not None
    )
    assert all(row["original_train_would_refuse"] is True for row in rows if row["loaded"])


@pytest.mark.parametrize(
    "mutation",
    [
        "backend",
        "fraction",
        "uuid",
        "integrated",
        "bool_free",
        "negative_free",
        "over_total",
        "soft_stop",
        "driver_metadata",
    ],
)
def test_native_identity_allocator_and_raw_counter_refusals(mutation):
    n = native()
    if mutation == "backend":
        n["allocator"]["backend"] = "cudaMallocAsync"
    elif mutation == "fraction":
        n["allocator"]["fraction"] = 1.0
    elif mutation == "uuid":
        n["identity"]["uuid"] = "00000000-0000-0000-0000-000000000000"
    elif mutation == "integrated":
        n["identity"]["integrated"] = True
    elif mutation == "bool_free":
        n["cuda"]["free"] = False
    elif mutation == "negative_free":
        n["cuda"]["free"] = -1
    elif mutation == "over_total":
        n["cuda"]["free"] = r.TOTAL + 1
    elif mutation == "soft_stop":
        n["cuda"]["reserved"] = r.STOP
    else:
        n["identity"]["library_metadata"]["inode"] += 1
    with pytest.raises(ValueError):
        r.validate_native(n, host(), BOOT)


@pytest.mark.parametrize("field", ["memory_max", "swap_current", "events", "properties", "members"])
def test_unit_pressure_bounds_and_identity_are_not_waived(field):
    u = unit()
    if field == "memory_max":
        u[field] += 1
    elif field == "swap_current":
        u[field] = 1
    elif field == "events":
        u[field]["oom"] = 1
    elif field == "properties":
        u[field]["MainPID"] = "1"
    else:
        u[field] = []
    with pytest.raises(ValueError):
        r.validate_unit(u, r.EXPECTED_UNIT)


def test_failed_model_check_cannot_transition_loaded(tmp_path):
    g = r.Gate(tmp_path / "gate.jsonl", BOOT, observer=host, unit_reader=unit, api_factory=FakeAPI)
    g.check("before-checkpoint-payload-hash-and-BF16-loader", "load")

    def failed():
        raise ValueError("EXPLICIT_FAKE_WRONG_FULL_MODEL")

    with pytest.raises(ValueError, match="WRONG_FULL_MODEL"):
        g.mark_model_loaded(SimpleNamespace(quantization="none", max_context=8192), failed)
    assert g.loaded is False and g.sequence == 1


@pytest.mark.parametrize(
    "mutation", ["loaded", "phase", "cache_release", "unit_inode", "symbol", "boot"]
)
def test_sealed_evidence_rejects_forged_transitions_and_stage_drift(tmp_path, mutation):
    g, _ = exercise(tmp_path, steps=1)
    rows = [json.loads(x) for x in g.path.read_bytes().splitlines()]
    if mutation == "loaded":
        rows[0]["loaded"] = True
    elif mutation == "phase":
        rows.pop(4)
    elif mutation == "cache_release":
        rows[6]["own_unused_cache_released"] = False
    elif mutation == "unit_inode":
        rows[-1]["unit"]["inode"] += 1
    elif mutation == "symbol":
        rows[-1]["native"]["identity"]["symbol_addresses"]["cuCtxGetDevice"] += 1
    else:
        rows[-1]["observation"]["boot_id"] = "different"
    with pytest.raises(ValueError):
        r.validate_evidence(
            b"\n".join(r.encode(x) for x in rows),
            expected_unit=r.EXPECTED_UNIT,
            expected_boot_id=BOOT,
            steps=1,
            parent_present=True,
        )


def test_primary_resource_refusal_survives_journal_failure(tmp_path, monkeypatch):
    observed = host(1)
    g = r.Gate(
        tmp_path / "gate.jsonl",
        BOOT,
        observer=lambda: observed,
        unit_reader=unit,
        api_factory=FakeAPI,
    )

    def broken(_row):
        raise OSError("EXPLICIT_FAKE_STORAGE_FAILURE")

    monkeypatch.setattr(g, "_append", broken)
    with pytest.raises(ValueError, match="Host reserve") as exc:
        g.check("before-any-project-framework-or-parent-payload", "cpu")
    assert isinstance(exc.value.__cause__, OSError)
    assert any("evidence also failed" in note for note in exc.value.__notes__)


def test_fixed_component_capture_is_separate_and_corruption_refuses(tmp_path):
    destination = tmp_path / "components"
    descriptor = r.capture_components(destination)
    assert descriptor["artifact_path"] == "inputs/resource-components"
    assert len(descriptor["files"]) == 3
    assert r.verify_components(destination, descriptor) is True
    path = destination / next(iter(descriptor["files"]))
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="bytes differ"):
        r.verify_components(destination, descriptor)


def test_optin_real_source_and_all179_capture_reaches_pre_model_sentinel(tmp_path, monkeypatch):
    """Real source/data/component capture; only unit/checkpoint/constructor are fakes."""
    import importlib.metadata

    import jev_browser_fit_native_actions as fitter

    phases = []

    class FakeGate:
        def __init__(self, path, _boot):
            self.path = path
            path.write_bytes(b"")

        def check(self, phase, scope):
            phases.append((phase, scope))

    class PreModelSentinel(RuntimeError):
        pass

    def stop_before_model(_seed):
        raise PreModelSentinel("EXPLICIT_FAKE_PRE_MODEL_SENTINEL")

    monkeypatch.setattr(fitter, "_environment", lambda: None)
    monkeypatch.setattr(r, "Gate", FakeGate)
    monkeypatch.setattr(fitter, "_checkpoint", lambda: {"EXPLICIT_FAKE_CHECKPOINT": True})
    monkeypatch.setattr(fitter, "_construct_model", stop_before_model)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "EXPLICIT_FAKE_VERSION")
    output = tmp_path / "fit"
    with pytest.raises(PreModelSentinel, match="PRE_MODEL_SENTINEL"):
        fitter.fit_native_actions(
            historical_root=r.ROOT, output_dir=output, steps=142, resource_profile=r.PROFILE
        )
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert receipt["stage"] == "failed" and receipt["model_load_started"] is True
    assert receipt["model_loads"] == receipt["training_updates"] == 0
    assert receipt["error"]["type"] == "PreModelSentinel"
    assert receipt["native_data"]["counts"] == {"train": 142, "development": 37, "total": 179}
    assert len(receipt["native_data"]["files"]) == 45
    assert r.verify_components(
        output / "inputs/resource-components", receipt["resource_components"]
    )
    sources = receipt["source_manifest"]["files"]
    assert "s1/browser_native_resources.py" in sources
    assert all(name.endswith(".py") for name in sources)
    assert all(ref["path"] not in sources for ref in r.COMPONENTS.values())
    assert receipt["source_verification"]["success"] is True
    assert phases == [
        ("before-any-project-framework-or-parent-payload", "cpu"),
        ("before-checkpoint-payload-hash-and-BF16-loader", "load"),
        ("before-any-BF16-model-construction", "train"),
    ]
