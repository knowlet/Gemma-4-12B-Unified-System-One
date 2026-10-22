"""Pure proposed full-D3 serial-load resource policy; no probing or side effects.

Research basis: profile-proposal.json SHA3a72ba74900b28e8e505c9acba2ed9c23eb6d7107afa6a8c1f4e394e553c9235.
The historical CORE132618706804-byte floor remains unchanged and refusing.
This policy adds independent CPU/CUDA budgets conservatively; it grants no UMA,
resident, swap or combined-free-pool credit and makes no measured-peak claim.
Only a separately reviewed collector/owner can authenticate supplied observations.
"""

from __future__ import annotations

import copy
import math
import re
import uuid
from pathlib import PurePosixPath

GIB = 1024**3
CPU_CAP_BYTES = 6 * GIB
CUDA_CAP_BYTES = 58 * GIB
BUFFER_BYTES = 2 * GIB
RESIDENT_BYTES = 52176025120
REFERENCE_MEMTOTAL_BYTES = 130595864576
PROFILE = "decision3-gx10-serial-mmap-measured-phase1-v1"
PHASES = {"before-framework", "before-constructor", "steady"}
CGROUP_MOUNT = "/sys/fs/cgroup"
EXPECTED_FIELDS = {
    "boot_id",
    "mem_total_bytes",
    "gpu_uuid",
    "cgroup_path",
    "resident_bytes",
    "cpu_cap_bytes",
    "cuda_cap_bytes",
    "buffer_bytes",
}
OOM_EVENTS = {"max", "oom", "oom_kill", "oom_group_kill"}
BASELINE_SCHEMA = "decision3-serial-resource-baseline-v1"


class InvalidObservation(ValueError):
    """An input-schema refusal, identified by a constant reason code."""


def need(value, reason):
    if not value:
        raise InvalidObservation(reason)


def integer(value, *, minimum=0):
    return type(value) is int and minimum <= value <= 2**64 - 1


def uuid_key(value, *, device=False):
    need(type(value) is str and len(value) <= 40, "invalid_uuid")
    raw = value[4:] if device and value.startswith("GPU-") else value
    need(
        bool(re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", raw)),
        "invalid_uuid",
    )
    return uuid.UUID(raw).hex


def selected_expected(value):
    need(type(value) is dict and set(value) == EXPECTED_FIELDS, "invalid_expected_fields")
    uuid_key(value["boot_id"])
    uuid_key(value["gpu_uuid"], device=True)
    need(integer(value["mem_total_bytes"], minimum=1), "invalid_selected_total")
    need(type(value["cgroup_path"]) is str, "invalid_selected_cgroup")
    path = PurePosixPath(value["cgroup_path"])
    need(
        str(path) == value["cgroup_path"]
        and ".." not in path.parts
        and path.is_relative_to(CGROUP_MOUNT)
        and str(path) != CGROUP_MOUNT,
        "invalid_selected_cgroup",
    )
    for key, exact in (
        ("cpu_cap_bytes", CPU_CAP_BYTES),
        ("cuda_cap_bytes", CUDA_CAP_BYTES),
        ("buffer_bytes", BUFFER_BYTES),
        ("resident_bytes", RESIDENT_BYTES),
    ):
        need(type(value[key]) is int and value[key] == exact, "selected_budget_changed")
    return copy.deepcopy(value)


def allocator_fraction(total_bytes):
    """A conservative fraction computed from the authenticated actual denominator."""
    need(integer(total_bytes, minimum=CUDA_CAP_BYTES), "invalid_allocator_total")
    fraction = math.nextafter(CUDA_CAP_BYTES / total_bytes, 0.0)
    need(
        CUDA_CAP_BYTES - 1 <= math.floor(fraction * total_bytes) <= CUDA_CAP_BYTES,
        "allocator_fraction_not_representable",
    )
    return fraction


def event_counters(value, required):
    need(
        type(value) is dict and required <= set(value) and len(value) <= 32,
        "incomplete_cgroup_event_counters",
    )
    need(
        all(
            type(k) is str and re.fullmatch(r"[a-z_]{1,32}", k) and integer(v)
            for k, v in value.items()
        ),
        "invalid_cgroup_event_counters",
    )
    return dict(value)


def cgroup_rows(snapshot, expected):
    leaf = PurePosixPath(expected["cgroup_path"])
    membership = "/" + str(leaf.relative_to(CGROUP_MOUNT))
    need(
        snapshot.get("cgroup_membership", "").strip() == "0::" + membership,
        "selected_cgroup_membership_differs",
    )
    mounts = snapshot.get("cgroup_mountinfo")
    need(type(mounts) is str and len(mounts) <= 65536, "invalid_cgroup_mountinfo")
    memory_mounts = []
    for line in mounts.splitlines():
        before, separator, after = line.partition(" - ")
        suffix = after.split()
        if separator and suffix and suffix[0] in {"cgroup", "cgroup2"}:
            memory_mounts.append((before.split(), suffix))
    need(len(memory_mounts) == 1, "complete_unified_cgroup_mount_required")
    fields, suffix = memory_mounts[0]
    need(
        len(fields) >= 6
        and len(suffix) >= 3
        and suffix[0] == "cgroup2"
        and fields[3] == "/"
        and fields[4] == CGROUP_MOUNT,
        "complete_unified_cgroup_mount_required",
    )
    paths, directory = [], leaf
    while str(directory) != CGROUP_MOUNT:
        paths.append(str(directory))
        directory = directory.parent
    rows = snapshot.get("cgroup_memory")
    need(
        type(rows) is list and 0 < len(rows) == len(paths) <= 32,
        "complete_cgroup_ancestry_required",
    )
    result = []
    for raw, path in zip(rows, paths):
        need(
            type(raw) is dict and raw.get("path") == path and raw.get("version") == "cgroup2",
            "complete_cgroup_ancestry_required",
        )
        need(
            {
                "path",
                "version",
                "device",
                "inode",
                "limit",
                "current",
                "peak",
                "swap_limit",
                "swap_current",
                "events",
                "swap_events",
            }
            <= set(raw),
            "complete_cgroup_fields_required",
        )
        row = {
            key: raw.get(key)
            for key in (
                "path",
                "version",
                "device",
                "inode",
                "limit",
                "current",
                "peak",
                "swap_limit",
                "swap_current",
            )
        }
        need(integer(row["device"]) and integer(row["inode"], minimum=1), "invalid_cgroup_inode")
        need(row["limit"] is None or integer(row["limit"], minimum=1), "invalid_cgroup_limit")
        need(
            integer(row["current"]) and integer(row["peak"]) and row["current"] <= row["peak"],
            "invalid_cgroup_usage",
        )
        need(row["swap_limit"] is None or integer(row["swap_limit"]), "invalid_swap_limit")
        need(integer(row["swap_current"]), "invalid_swap_usage")
        row["events"] = event_counters(raw.get("events"), OOM_EVENTS)
        row["swap_events"] = event_counters(raw.get("swap_events"), {"max", "fail"})
        result.append(row)
    return result


def group_binding(row):
    return {
        k: copy.deepcopy(row[k])
        for k in (
            "path",
            "version",
            "device",
            "inode",
            "limit",
            "swap_limit",
            "events",
            "swap_events",
        )
    }


def baseline_state(baseline, expected, rows, phase, scope):
    if baseline is None:
        need(
            phase == "before-framework" or scope == "host-sampler",
            "successful_initial_baseline_required",
        )
        return {
            "schema_id": BASELINE_SCHEMA,
            "scope": scope,
            "expected": copy.deepcopy(expected),
            "cgroups": [group_binding(row) for row in rows],
            "driver": None,
            "allocator": None,
        }
    need(
        type(baseline) is dict
        and set(baseline) == {"schema_id", "scope", "expected", "cgroups", "driver", "allocator"}
        and baseline["schema_id"] == BASELINE_SCHEMA
        and baseline["scope"] == scope
        and selected_expected(baseline["expected"]) == expected,
        "successful_initial_baseline_required",
    )
    if phase == "before-framework":
        need(
            baseline["driver"] is None and baseline["allocator"] is None,
            "initial_phase_after_cuda_forbidden",
        )
    need(
        type(baseline["cgroups"]) is list and len(baseline["cgroups"]) == len(rows),
        "cgroup_baseline_incomplete",
    )
    for old, current in zip(baseline["cgroups"], rows):
        need(
            type(old) is dict and set(old) == set(group_binding(current)),
            "cgroup_baseline_incomplete",
        )
        for key in ("path", "version", "device", "inode", "limit", "swap_limit"):
            need(
                type(old[key]) is type(current[key]) and old[key] == current[key],
                "cgroup_identity_or_caps_drift",
            )
        for key, required in (("events", OOM_EVENTS), ("swap_events", {"max", "fail"})):
            counters = event_counters(old[key], required)
            need(
                set(counters) == set(current[key])
                and all(current[key][k] >= v for k, v in counters.items()),
                "cgroup_event_counter_reset_or_drift",
            )
    return copy.deepcopy(baseline)


def driver_identity(raw, expected):
    need(type(raw) is dict, "raw_driver_required")
    fields = (
        "boot_id",
        "device_count",
        "context_device",
        "integrated",
        "name",
        "uuid",
        "total_bytes",
    )
    row = {key: raw.get(key) for key in fields}
    need(
        row["boot_id"] == expected["boot_id"]
        and row["name"] == "NVIDIA GB10"
        and type(row["device_count"]) is int
        and row["device_count"] == 1
        and type(row["context_device"]) is int
        and row["context_device"] == 0
        and type(row["integrated"]) is int
        and row["integrated"] == 1
        and type(row["total_bytes"]) is int
        and row["total_bytes"] == expected["mem_total_bytes"]
        and uuid_key(row["uuid"], device=True) == uuid_key(expected["gpu_uuid"], device=True),
        "raw_driver_host_device_mismatch",
    )
    return row


def cuda_evidence(raw, driver, expected, phase):
    need(type(raw) is dict, "cuda_observation_required")
    fields = (
        "device",
        "name",
        "uuid",
        "free_bytes",
        "total_bytes",
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "allocator_backend",
        "allocator_fraction",
        "allocator_allowed_bytes_upper",
    )
    row = {key: raw.get(key) for key in fields}
    need(
        type(row["device"]) is int
        and row["device"] == 0
        and row["name"] == driver["name"] == "NVIDIA GB10"
        and uuid_key(row["uuid"], device=True) == uuid_key(driver["uuid"], device=True)
        and integer(row["total_bytes"], minimum=1)
        and row["total_bytes"] == driver["total_bytes"] == expected["mem_total_bytes"],
        "raw_cuda_driver_host_mismatch",
    )
    for key in (
        "free_bytes",
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "allocator_allowed_bytes_upper",
    ):
        need(integer(row[key]), "invalid_cuda_counter")
    need(
        row["free_bytes"] <= row["total_bytes"]
        and row["allocated_bytes"] <= row["reserved_bytes"]
        and row["allocated_bytes"] <= row["peak_allocated_bytes"]
        and row["reserved_bytes"] <= row["peak_reserved_bytes"]
        and row["peak_allocated_bytes"] <= row["peak_reserved_bytes"],
        "inconsistent_cuda_counters",
    )
    fraction = row["allocator_fraction"]
    need(
        row["allocator_backend"] == "native"
        and type(fraction) is float
        and math.isfinite(fraction)
        and 0 < fraction <= 1
        and row["allocator_allowed_bytes_upper"] == math.floor(fraction * row["total_bytes"])
        and CUDA_CAP_BYTES - 1 <= row["allocator_allowed_bytes_upper"] <= CUDA_CAP_BYTES,
        "allocator_readback_or_actual_denominator_mismatch",
    )
    row["fraction_denominator_bytes"] = row["total_bytes"]
    row["required_free_bytes"] = BUFFER_BYTES + (
        RESIDENT_BYTES if phase == "before-constructor" else 0
    )
    return row


def _decision(snapshot, expected, phase, baseline, *, host_only):
    """Shared host/cgroup predicate; public worker entry never skips CUDA checks."""
    scope = "host-sampler" if host_only else "worker"
    result = {
        "schema_id": "decision3-serial-resource-decision-v1",
        "profile": PROFILE,
        "scope": scope,
        "cuda_checks_performed": False,
        "phase": phase if type(phase) is str and phase in PHASES else None,
        "admitted": False,
        "refusal_reasons": [],
        "baseline": None,
    }
    try:
        need(type(phase) is str and phase in PHASES, "invalid_phase")
        selected = selected_expected(expected)
        need(type(snapshot) is dict, "invalid_snapshot")
        need(snapshot.get("boot_id") == selected["boot_id"], "host_boot_drift")
        proc = snapshot.get("proc_meminfo")
        need(
            type(proc) is dict
            and type(proc.get("MemTotal")) is int
            and proc["MemTotal"] == selected["mem_total_bytes"]
            and integer(proc.get("MemAvailable"), minimum=1)
            and proc["MemAvailable"] <= proc["MemTotal"],
            "host_memory_identity_or_type",
        )
        need(
            integer(proc.get("SwapTotal"))
            and integer(proc.get("SwapFree"))
            and proc["SwapFree"] <= proc["SwapTotal"],
            "invalid_host_swap_observation",
        )
        reserve = max(24 * GIB, (selected["mem_total_bytes"] + 4) // 5)
        floor = (
            reserve
            + BUFFER_BYTES
            + (CPU_CAP_BYTES + CUDA_CAP_BYTES if phase == "before-framework" else 0)
        )
        rows = cgroup_rows(snapshot, selected)
        bound = baseline_state(baseline, selected, rows, phase, scope)
        leaf = rows[0]
        need(
            type(leaf["limit"]) is int
            and leaf["limit"] == CPU_CAP_BYTES
            and type(leaf["swap_limit"]) is int
            and leaf["swap_limit"] == 0,
            "selected_cpu_or_swap_cap_differs",
        )
        reasons = result["refusal_reasons"]
        if proc["MemAvailable"] < floor:
            reasons.append("host_available_below_phase_floor")
        if leaf["current"] > CPU_CAP_BYTES or leaf["peak"] > CPU_CAP_BYTES:
            reasons.append("selected_cpu_cap_exceeded")
        if leaf["swap_current"] != 0 or any(leaf["swap_events"].values()):
            reasons.append("selected_swap_activity")
        if any(leaf["events"][key] for key in OOM_EVENTS):
            reasons.append("selected_pressure_or_oom_activity")
        # CPU headroom is independent from the full host reserve. Never take a
        # min(6GiB memcg headroom, host MemAvailable) against the full host floor.
        remaining_cpu = max(0, CPU_CAP_BYTES - leaf["current"])
        groups = []
        for row, old in zip(rows, bound["cgroups"]):
            evidence = copy.deepcopy(row)
            evidence["headroom_bytes"] = (
                None if row["limit"] is None else max(0, row["limit"] - row["current"])
            )
            evidence["event_deltas"] = {k: v - old["events"][k] for k, v in row["events"].items()}
            evidence["swap_event_deltas"] = {
                k: v - old["swap_events"][k] for k, v in row["swap_events"].items()
            }
            if (
                row is not leaf
                and row["limit"] is not None
                and row["limit"] - row["current"] < remaining_cpu
            ):
                reasons.append("ancestor_cpu_headroom_insufficient")
            groups.append(evidence)
        result.update(
            host={
                "boot_id": snapshot["boot_id"],
                "mem_total_bytes": proc["MemTotal"],
                "available_bytes": proc["MemAvailable"],
                "reserve_bytes": reserve,
                "required_available_bytes": floor,
                "swap_total_bytes": proc["SwapTotal"],
                "swap_free_bytes": proc["SwapFree"],
            },
            budgets={
                "cpu_cap_bytes": CPU_CAP_BYTES,
                "cuda_cap_bytes": CUDA_CAP_BYTES,
                "buffer_bytes": BUFFER_BYTES,
                "resident_bytes": RESIDENT_BYTES,
                "swap_or_resident_credit_bytes": 0,
                "free_pools_summed": False,
            },
            cgroups=groups,
            remaining_cpu_budget_bytes=remaining_cpu,
        )
        driver = snapshot.get("raw_driver") if not host_only else None
        if not host_only and driver is not None:
            driver = driver_identity(driver, selected)
            need(bound["driver"] is None or bound["driver"] == driver, "raw_driver_identity_drift")
            bound["driver"] = driver
            result["raw_driver"] = driver
        cuda = snapshot.get("cuda") if not host_only else None
        if host_only:
            # A parent sampler has no CUDA context. Its scoped host predicate
            # cannot substitute for worker initial/constructor admission.
            need(
                bound["driver"] is None and bound["allocator"] is None,
                "host_sampler_baseline_scope_differs",
            )
        elif phase == "before-framework":
            need(cuda is None, "before_framework_cuda_already_present")
        else:
            need(driver is not None, "raw_driver_required")
            cuda = cuda_evidence(cuda, driver, selected, phase)
            config = {
                key: cuda[key]
                for key in (
                    "device",
                    "name",
                    "uuid",
                    "total_bytes",
                    "allocator_backend",
                    "allocator_fraction",
                    "allocator_allowed_bytes_upper",
                )
            }
            need(
                bound["allocator"] is None or bound["allocator"] == config,
                "allocator_identity_or_cap_drift",
            )
            need(
                phase != "steady" or bound["allocator"] is not None,
                "successful_constructor_baseline_required",
            )
            bound["allocator"] = config
            if cuda["free_bytes"] < cuda["required_free_bytes"]:
                reasons.append("cuda_free_below_phase_floor")
            if (
                max(
                    cuda["reserved_bytes"],
                    cuda["peak_reserved_bytes"],
                    cuda["allocated_bytes"],
                    cuda["peak_allocated_bytes"],
                )
                > CUDA_CAP_BYTES
            ):
                reasons.append("cuda_allocator_cap_exceeded")
            result["cuda"] = cuda
            result["cuda_checks_performed"] = True
        result["refusal_reasons"] = list(dict.fromkeys(reasons))
        result["admitted"] = not result["refusal_reasons"]
        result["baseline"] = bound if result["admitted"] else None
    except InvalidObservation as error:
        result["refusal_reasons"].append(str(error))
    except (KeyError, TypeError, ValueError, AttributeError):
        # No arbitrary input/exception strings are copied into evidence.
        result["refusal_reasons"].append("malformed_resource_observation")
    return result


def decision(snapshot, expected, phase, baseline=None):
    """Worker JSON facts; refused inputs never return a usable baseline.

    Later phases require successful worker before-framework admission; pass the
    returned constructor baseline onward to bind the actual allocator. A host-
    sampler baseline is deliberately insufficient. This is a pure predicate,
    not live admission, model qualification or a process ownership proof.
    """
    return _decision(snapshot, expected, phase, baseline, host_only=False)


def host_decision(snapshot, expected, baseline=None, *, initial=False):
    """Independent parent sampling: host/cgroups only, never CUDA admission.

    Default starts at R+2GiB with its own event/identity baseline. initial=True
    selects the full R+6GiB+58GiB+2GiB host floor. Neither mode authorizes the
    worker or supplies its before-framework/constructor baseline.
    """
    phase = ("before-framework" if initial else "steady") if type(initial) is bool else None
    result = _decision(snapshot, expected, phase, baseline, host_only=True)
    result["phase"] = (
        ("host-initial" if initial else "host-steady") if type(initial) is bool else None
    )
    result["model_admission_performed"] = False
    return result
