"""Pure resource-policy boundaries and identity drift; no live observations."""

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

HERE = Path(__file__).absolute().parent
SPEC = importlib.util.spec_from_file_location(
    "_serial_resource_policy_test", HERE / "resource_policy.py"
)
M = importlib.util.module_from_spec(SPEC)
exec(
    compile(
        (HERE / "resource_policy.py").read_bytes(),
        str(HERE / "resource_policy.py"),
        "exec",
        dont_inherit=True,
    ),
    M.__dict__,
)


@pytest.fixture
def inputs():
    expected = {
        "boot_id": "90eeafaa-7c2f-45f0-8b45-f74ef2ee1005",
        "mem_total_bytes": 130595864576,
        "gpu_uuid": "e3429caa-b057-cbe0-917d-d3b85b313be7",
        "cgroup_path": "/sys/fs/cgroup/test.slice/full.service",
        "resident_bytes": M.RESIDENT_BYTES,
        "cpu_cap_bytes": M.CPU_CAP_BYTES,
        "cuda_cap_bytes": M.CUDA_CAP_BYTES,
        "buffer_bytes": M.BUFFER_BYTES,
    }
    counters = {"low": 0, "high": 0, "max": 0, "oom": 0, "oom_kill": 0, "oom_group_kill": 0}
    rows = []
    for index, path in enumerate((expected["cgroup_path"], "/sys/fs/cgroup/test.slice")):
        rows.append(
            {
                "path": path,
                "version": "cgroup2",
                "device": 1,
                "inode": 100 + index,
                "limit": M.CPU_CAP_BYTES if index == 0 else None,
                "current": 128 * 1024**2 if index == 0 else 20 * M.GIB,
                "peak": 128 * 1024**2 if index == 0 else 20 * M.GIB,
                "swap_limit": 0 if index == 0 else None,
                "swap_current": 0,
                "events": dict(counters),
                "swap_events": {"high": 0, "max": 0, "fail": 0},
            }
        )
    rows[1]["events"]["oom"] = 5  # Ancestor history is recorded, not relabeled own OOM.
    snapshot = {
        "boot_id": expected["boot_id"],
        "proc_meminfo": {
            "MemTotal": expected["mem_total_bytes"],
            "MemAvailable": 96986133300,
            "SwapTotal": 32 * M.GIB,
            "SwapFree": 16 * M.GIB,
        },
        "cgroup_membership": "0::/test.slice/full.service\n",
        "cgroup_mountinfo": "30 20 0:28 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        "cgroup_memory": rows,
        "cuda": None,
    }
    return snapshot, expected


def initial(inputs):
    snapshot, expected = inputs
    result = M.decision(snapshot, expected, "before-framework")
    assert result["admitted"], result
    return result


def constructor(inputs):
    snapshot, expected = copy.deepcopy(inputs)
    baseline = initial(inputs)["baseline"]
    snapshot["proc_meminfo"]["MemAvailable"] = (
        max(24 * M.GIB, (expected["mem_total_bytes"] + 4) // 5) + M.BUFFER_BYTES
    )
    snapshot["cgroup_memory"][0].update(current=2 * M.GIB, peak=2 * M.GIB)
    snapshot["raw_driver"] = {
        "boot_id": expected["boot_id"],
        "device_count": 1,
        "context_device": 0,
        "integrated": 1,
        "name": "NVIDIA GB10",
        "uuid": expected["gpu_uuid"],
        "total_bytes": expected["mem_total_bytes"],
    }
    fraction = M.allocator_fraction(expected["mem_total_bytes"])
    snapshot["cuda"] = {
        "device": 0,
        "name": "NVIDIA GB10",
        "uuid": "GPU-" + expected["gpu_uuid"],
        "free_bytes": M.RESIDENT_BYTES + M.BUFFER_BYTES,
        "total_bytes": expected["mem_total_bytes"],
        "allocated_bytes": 0,
        "reserved_bytes": 0,
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "allocator_backend": "native",
        "allocator_fraction": fraction,
        "allocator_allowed_bytes_upper": math.floor(fraction * expected["mem_total_bytes"]),
    }
    return snapshot, expected, baseline


def steady(inputs):
    snapshot, expected, baseline = constructor(inputs)
    gate = M.decision(snapshot, expected, "before-constructor", baseline)
    assert gate["admitted"], gate
    snapshot["cuda"].update(
        free_bytes=M.BUFFER_BYTES,
        allocated_bytes=M.RESIDENT_BYTES,
        reserved_bytes=55_000_000_000,
        peak_allocated_bytes=M.RESIDENT_BYTES,
        peak_reserved_bytes=55_000_000_000,
    )
    return snapshot, expected, gate["baseline"]


def assert_refused(result, reason=None):
    assert result["admitted"] is False
    assert result["baseline"] is None
    assert result["refusal_reasons"]
    if reason:
        assert reason in result["refusal_reasons"], result
    json.dumps(result, allow_nan=False)


def test_exact_conservative_arithmetic_no_credit_and_original_floor_distinct(inputs):
    result = initial(inputs)
    assert M.CPU_CAP_BYTES == 6442450944
    assert M.CUDA_CAP_BYTES == 62277025792
    assert M.BUFFER_BYTES == 2147483648
    assert result["host"]["reserve_bytes"] == 26119172916
    assert result["host"]["required_available_bytes"] == 96986133300
    assert result["host"]["reserve_bytes"] + 2 * M.RESIDENT_BYTES + M.BUFFER_BYTES == 132618706804
    assert result["budgets"]["swap_or_resident_credit_bytes"] == 0
    assert result["budgets"]["free_pools_summed"] is False


def test_initial_boundary_and_host_floor_is_independent_from_6gib_memcg(inputs):
    snapshot, expected = inputs
    gate = initial(inputs)
    assert M.CPU_CAP_BYTES < gate["host"]["reserve_bytes"]
    snapshot["proc_meminfo"]["MemAvailable"] -= 1
    assert_refused(
        M.decision(snapshot, expected, "before-framework"), "host_available_below_phase_floor"
    )


@pytest.mark.parametrize("phase", ["before-constructor", "steady"])
def test_later_host_and_cuda_boundaries(inputs, phase):
    snapshot, expected, baseline = (
        constructor(inputs) if phase == "before-constructor" else steady(inputs)
    )
    assert M.decision(snapshot, expected, phase, baseline)["admitted"]
    snapshot["proc_meminfo"]["MemAvailable"] -= 1
    assert_refused(
        M.decision(snapshot, expected, phase, baseline), "host_available_below_phase_floor"
    )
    snapshot["proc_meminfo"]["MemAvailable"] += 1
    snapshot["cuda"]["free_bytes"] -= 1
    assert_refused(M.decision(snapshot, expected, phase, baseline), "cuda_free_below_phase_floor")


@pytest.mark.parametrize("phase", ["before-constructor", "steady"])
def test_later_phase_requires_successful_initial_and_constructor_baselines(inputs, phase):
    snapshot, expected, _ = constructor(inputs)
    assert_refused(M.decision(snapshot, expected, phase), "successful_initial_baseline_required")
    if phase == "steady":
        assert_refused(
            M.decision(snapshot, expected, phase, initial(inputs)["baseline"]),
            "successful_constructor_baseline_required",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("mem_total_bytes", True),
        ("cpu_cap_bytes", 80 * 1024**3),
        ("cuda_cap_bytes", 59 * 1024**3),
        ("buffer_bytes", 0),
        ("resident_bytes", 1),
        ("boot_id", "not-a-boot-id"),
        ("gpu_uuid", "missing"),
        ("cgroup_path", "/sys/fs/cgroup/a/../full.service"),
    ],
)
def test_typed_selected_policy_cannot_be_relabelled(inputs, field, value):
    snapshot, expected = inputs
    expected[field] = value
    assert_refused(M.decision(snapshot, expected, "before-framework"))


@pytest.mark.parametrize(
    "fault",
    [
        "boot",
        "total",
        "boolavailable",
        "missingancestor",
        "reordered",
        "duplicate",
        "missingevent",
        "missinginode",
        "submount",
        "membership",
        "swapshape",
    ],
)
def test_malformed_or_incomplete_observations_refuse(inputs, fault):
    snapshot, expected = inputs
    if fault == "boot":
        snapshot["boot_id"] = "00000000-0000-0000-0000-000000000000"
    elif fault == "total":
        snapshot["proc_meminfo"]["MemTotal"] -= 4096
    elif fault == "boolavailable":
        snapshot["proc_meminfo"]["MemAvailable"] = True
    elif fault == "missingancestor":
        snapshot["cgroup_memory"].pop()
    elif fault == "reordered":
        snapshot["cgroup_memory"].reverse()
    elif fault == "duplicate":
        snapshot["cgroup_memory"][1] = copy.deepcopy(snapshot["cgroup_memory"][0])
    elif fault == "missingevent":
        del snapshot["cgroup_memory"][0]["events"]["oom"]
    elif fault == "missinginode":
        del snapshot["cgroup_memory"][0]["inode"]
    elif fault == "submount":
        snapshot["cgroup_mountinfo"] = "30 20 0:28 /hidden /sys/fs/cgroup rw - cgroup2 cgroup rw"
    elif fault == "membership":
        snapshot["cgroup_membership"] = "0::/other.service"
    elif fault == "swapshape":
        snapshot["proc_meminfo"]["SwapFree"] = snapshot["proc_meminfo"]["SwapTotal"] + 1
    assert_refused(M.decision(snapshot, expected, "before-framework"))


@pytest.mark.parametrize(
    "field,value", [("device", 2), ("inode", 900), ("limit", 5 * 1024**3), ("swap_limit", 1)]
)
def test_cgroup_identity_or_limits_drift_within_run_refuse(inputs, field, value):
    snapshot, expected, baseline = constructor(inputs)
    snapshot["cgroup_memory"][0][field] = value
    assert_refused(
        M.decision(snapshot, expected, "before-constructor", baseline),
        "cgroup_identity_or_caps_drift",
    )


@pytest.mark.parametrize(
    "fault", ["oom", "oom_kill", "max", "swapcurrent", "swapfail", "cpucurrent", "cpupeak"]
)
def test_selected_pressure_oom_swap_and_cpu_cap_are_hard_refusals(inputs, fault):
    snapshot, expected, baseline = constructor(inputs)
    leaf = snapshot["cgroup_memory"][0]
    if fault in {"oom", "oom_kill", "max"}:
        leaf["events"][fault] += 1
    elif fault == "swapcurrent":
        leaf["swap_current"] = 1
    elif fault == "swapfail":
        leaf["swap_events"]["fail"] = 1
    elif fault == "cpucurrent":
        leaf.update(current=M.CPU_CAP_BYTES + 1, peak=M.CPU_CAP_BYTES + 1)
    else:
        leaf["peak"] = M.CPU_CAP_BYTES + 1
    assert_refused(M.decision(snapshot, expected, "before-constructor", baseline))


def test_ancestor_counters_are_recorded_separately_and_remaining_cpu_headroom_checked(inputs):
    snapshot, expected, baseline = constructor(inputs)
    ancestor = snapshot["cgroup_memory"][1]
    ancestor["events"]["oom"] += 1
    gate = M.decision(snapshot, expected, "before-constructor", baseline)
    assert gate["admitted"], gate
    assert gate["cgroups"][1]["event_deltas"]["oom"] == 1
    assert gate["cgroups"][0]["event_deltas"]["oom"] == 0
    ancestor["events"]["oom"] = 4
    assert_refused(
        M.decision(snapshot, expected, "before-constructor", baseline),
        "cgroup_event_counter_reset_or_drift",
    )
    # A finite ancestor cap is selected at initial admission, then kept exact.
    inputs[0]["cgroup_memory"][1]["limit"] = 26 * M.GIB
    snapshot, expected, baseline = constructor(inputs)
    snapshot["cgroup_memory"][1].update(current=23 * M.GIB, peak=23 * M.GIB)
    assert_refused(
        M.decision(snapshot, expected, "before-constructor", baseline),
        "ancestor_cpu_headroom_insufficient",
    )


@pytest.mark.parametrize(
    "source,field,value",
    [
        ("raw_driver", "boot_id", "00000000-0000-0000-0000-000000000000"),
        ("raw_driver", "total_bytes", 130595860480),
        ("raw_driver", "integrated", 0),
        ("raw_driver", "device_count", 2),
        ("raw_driver", "context_device", 1),
        ("raw_driver", "name", "other GPU"),
        ("raw_driver", "uuid", "00000000-0000-0000-0000-000000000000"),
        ("cuda", "total_bytes", 130595860480),
        ("cuda", "name", "other GPU"),
        ("cuda", "uuid", "00000000-0000-0000-0000-000000000000"),
    ],
)
def test_raw_driver_torch_and_host_must_match(inputs, source, field, value):
    snapshot, expected, baseline = constructor(inputs)
    snapshot[source][field] = value
    assert_refused(M.decision(snapshot, expected, "before-constructor", baseline))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 1, -0.1, 1.1])
def test_allocator_fraction_must_be_finite_native_float(inputs, value):
    snapshot, expected, baseline = constructor(inputs)
    snapshot["cuda"]["allocator_fraction"] = value
    assert_refused(M.decision(snapshot, expected, "before-constructor", baseline))


def test_fraction_uses_actual_raw_total_and_new_boot_total_is_not_normalized(inputs):
    snapshot, expected, baseline = constructor(inputs)
    cuda = snapshot["cuda"]
    cuda["allocator_fraction"] = M.CUDA_CAP_BYTES / 130595860480
    cuda["allocator_allowed_bytes_upper"] = math.floor(
        cuda["allocator_fraction"] * cuda["total_bytes"]
    )
    assert_refused(
        M.decision(snapshot, expected, "before-constructor", baseline),
        "allocator_readback_or_actual_denominator_mismatch",
    )
    snapshot, expected = inputs
    expected["mem_total_bytes"] += 4096
    snapshot["proc_meminfo"]["MemTotal"] += 4096
    snapshot["proc_meminfo"]["MemAvailable"] += 4096
    snapshot["boot_id"] = expected["boot_id"] = "00000000-0000-0000-0000-000000000001"
    fresh, expected, baseline = constructor((snapshot, expected))
    gate = M.decision(fresh, expected, "before-constructor", baseline)
    assert gate["admitted"], gate
    assert gate["cuda"]["fraction_denominator_bytes"] == 130595868672


def test_allocator_peaks_and_within_run_readback_are_preserved(inputs):
    snapshot, expected, baseline = steady(inputs)
    gate = M.decision(snapshot, expected, "steady", baseline)
    assert gate["admitted"], gate
    assert gate["cuda"]["peak_reserved_bytes"] == 55_000_000_000
    snapshot["cuda"]["peak_reserved_bytes"] = M.CUDA_CAP_BYTES + 1
    assert_refused(
        M.decision(snapshot, expected, "steady", baseline), "cuda_allocator_cap_exceeded"
    )
    snapshot, expected, baseline = steady(inputs)
    snapshot["cuda"]["allocator_backend"] = "cudaMallocAsync"
    assert_refused(M.decision(snapshot, expected, "steady", baseline))


@pytest.mark.parametrize("key", ["limit", "swap_limit"])
def test_missing_ancestor_limit_is_not_treated_as_unlimited(inputs, key):
    snapshot, expected = inputs
    del snapshot["cgroup_memory"][1][key]
    assert_refused(
        M.decision(snapshot, expected, "before-framework"), "complete_cgroup_fields_required"
    )


def test_within_run_fraction_drift_is_refused_even_with_same_conservative_cap(inputs):
    snapshot, expected, baseline = steady(inputs)
    cuda = snapshot["cuda"]
    cuda["allocator_fraction"] = math.nextafter(cuda["allocator_fraction"], 1.0)
    cuda["allocator_allowed_bytes_upper"] = math.floor(
        cuda["allocator_fraction"] * cuda["total_bytes"]
    )
    assert M.CUDA_CAP_BYTES - 1 <= cuda["allocator_allowed_bytes_upper"] <= M.CUDA_CAP_BYTES
    assert_refused(
        M.decision(snapshot, expected, "steady", baseline), "allocator_identity_or_cap_drift"
    )


def test_no_input_mutation_or_arbitrary_extra_evidence_and_json_refusals(inputs):
    snapshot, expected = inputs
    snapshot["secret_payload"] = "not-a-resource-fact"
    original = copy.deepcopy((snapshot, expected))
    result = M.decision(snapshot, expected, "before-framework")
    assert result["admitted"]
    assert (snapshot, expected) == original
    assert "not-a-resource-fact" not in json.dumps(result)
    for phase in (None, [], "wrong"):
        assert_refused(M.decision(snapshot, expected, phase))
    assert_refused(M.decision(None, expected, "before-framework"))


def test_host_sampler_starts_at_steady_floor_and_never_performs_cuda_admission(inputs):
    snapshot, expected = inputs
    snapshot["proc_meminfo"]["MemAvailable"] = 28266656564
    result = M.host_decision(snapshot, expected)
    assert result["admitted"], result
    assert result["scope"] == result["baseline"]["scope"] == "host-sampler"
    assert result["phase"] == "host-steady"
    assert result["host"]["required_available_bytes"] == 28266656564
    assert result["cuda_checks_performed"] is False
    assert result["model_admission_performed"] is False
    assert_refused(
        M.host_decision(snapshot, expected, initial=True), "host_available_below_phase_floor"
    )
    snapshot["proc_meminfo"]["MemAvailable"] -= 1
    assert_refused(
        M.host_decision(snapshot, expected, result["baseline"]), "host_available_below_phase_floor"
    )


def test_host_only_baseline_cannot_authorize_worker_and_worker_baseline_not_sampler(inputs):
    snapshot, expected, worker_initial = constructor(inputs)
    host_baseline = M.host_decision(inputs[0], expected)["baseline"]
    assert_refused(
        M.decision(snapshot, expected, "before-constructor", host_baseline),
        "successful_initial_baseline_required",
    )
    assert_refused(
        M.host_decision(inputs[0], expected, worker_initial), "successful_initial_baseline_required"
    )


@pytest.mark.parametrize(
    "fault", ["inode", "cap", "swap", "oom", "counter_reset", "missingancestor"]
)
def test_host_sampler_reuses_strict_cgroup_and_counter_validation(inputs, fault):
    snapshot, expected = inputs
    baseline = M.host_decision(snapshot, expected)["baseline"]
    leaf = snapshot["cgroup_memory"][0]
    if fault == "inode":
        leaf["inode"] += 1
    elif fault == "cap":
        leaf["limit"] += 1
    elif fault == "swap":
        leaf["swap_current"] = 1
    elif fault == "oom":
        leaf["events"]["oom"] += 1
    elif fault == "counter_reset":
        snapshot["cgroup_memory"][1]["events"]["oom"] -= 1
    else:
        snapshot["cgroup_memory"].pop()
    assert_refused(M.host_decision(snapshot, expected, baseline))


def test_host_initial_floor_and_typed_initial_flag(inputs):
    snapshot, expected = inputs
    result = M.host_decision(snapshot, expected, initial=True)
    assert result["admitted"], result
    assert result["host"]["required_available_bytes"] == 96986133300
    assert_refused(M.host_decision(snapshot, expected, initial=1), "invalid_phase")


@pytest.mark.parametrize("scope", ["worker", "host"])
def test_repeated_initial_checks_keep_original_baseline_and_full_floor(inputs, scope):
    snapshot, expected = inputs
    evaluate = (
        (lambda baseline: M.decision(snapshot, expected, "before-framework", baseline))
        if scope == "worker"
        else (lambda baseline: M.host_decision(snapshot, expected, baseline, initial=True))
    )
    first = evaluate(None)
    assert first["admitted"]
    original_baseline = copy.deepcopy(first["baseline"])
    snapshot["cgroup_memory"][0].update(current=256 * 1024**2, peak=256 * 1024**2)
    snapshot["cgroup_memory"][1]["events"]["oom"] += 1
    second = evaluate(first["baseline"])
    assert second["admitted"], second
    assert second["host"]["required_available_bytes"] == 96986133300
    assert second["baseline"] == first["baseline"] == original_baseline
    assert second["cgroups"][1]["event_deltas"]["oom"] == 1
    snapshot["proc_meminfo"]["MemAvailable"] -= 1
    assert_refused(evaluate(second["baseline"]), "host_available_below_phase_floor")


@pytest.mark.parametrize("scope", ["worker", "host"])
@pytest.mark.parametrize(
    "fault",
    [
        "inode",
        "limit",
        "event_reset",
        "selected_oom",
        "selected_max",
        "selected_swap_current",
        "selected_swap_fail",
    ],
)
def test_repeated_initial_identity_and_event_drift_refuse(inputs, scope, fault):
    snapshot, expected = inputs
    evaluate = (
        (lambda baseline: M.decision(snapshot, expected, "before-framework", baseline))
        if scope == "worker"
        else (lambda baseline: M.host_decision(snapshot, expected, baseline, initial=True))
    )
    baseline = evaluate(None)["baseline"]
    if fault == "inode":
        snapshot["cgroup_memory"][0]["inode"] += 1
    elif fault == "limit":
        snapshot["cgroup_memory"][1]["limit"] = 100 * M.GIB
    elif fault == "event_reset":
        snapshot["cgroup_memory"][1]["events"]["oom"] -= 1
    elif fault == "selected_oom":
        snapshot["cgroup_memory"][0]["events"]["oom"] += 1
    elif fault == "selected_max":
        snapshot["cgroup_memory"][0]["events"]["max"] += 1
    elif fault == "selected_swap_current":
        snapshot["cgroup_memory"][0]["swap_current"] += 1
    else:
        snapshot["cgroup_memory"][0]["swap_events"]["fail"] += 1
    assert_refused(evaluate(baseline))


@pytest.mark.parametrize("bound", ["driver", "allocator", "both"])
def test_constructor_bound_baseline_cannot_reenter_initial_phase(inputs, bound):
    snapshot, expected, baseline = steady(inputs)
    if bound == "driver":
        baseline["allocator"] = None
    elif bound == "allocator":
        baseline["driver"] = None
    # Remove current CUDA facts: absence cannot erase prior constructor binding.
    snapshot.pop("raw_driver")
    snapshot["cuda"] = None
    snapshot["proc_meminfo"]["MemAvailable"] = expected["mem_total_bytes"]
    assert_refused(
        M.decision(snapshot, expected, "before-framework", baseline),
        "initial_phase_after_cuda_forbidden",
    )
