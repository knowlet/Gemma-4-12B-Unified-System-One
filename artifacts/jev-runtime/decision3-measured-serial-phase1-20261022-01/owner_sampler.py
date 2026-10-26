"""External50ms host/cgroup sampler for an already authenticated6GiB unit.

No unit launch and no CUDA/framework calls. The caller keeps this sampler
outside the selected model unit and owns its lifetime/whole-unit final closure.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
BOUNDARY = (
    ROOT / "artifacts/jev-runtime/decision3-local-owner-terminal-20261022-01/owner_boundary.py"
)
BOUNDARY_SHA = "7a50c47d6d7f696973c666cb7a40ce47dd0a7624667a18422c8c60c0db72e4e5"
PERIOD_SECONDS = 0.05


def require(value, message):
    if not value:
        raise ValueError(message)


def save_new(path, value):
    blob = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    )
    require(len(blob) <= 1024**2, "Bounded sampler receipt")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())


def run_owned_sampler(boundary, policy, observations, expected, directory, stop_event, deadline):
    """Block outside the owned unit; any refusal closes only its selected unit.

    policy/observations are checked component modules selected by the external
    descriptor, not arbitrary trust callbacks. Return is not continuation proof.
    """
    require(
        hashlib.sha256(BOUNDARY.read_bytes()).hexdigest() == BOUNDARY_SHA
        and Path(boundary.verify.__func__.__code__.co_filename) == BOUNDARY,
        "Exact existing selected unit boundary",
    )
    selected = boundary.selection
    require(
        selected.memory_max_bytes == 6 * 1024**3
        and selected.boot_id == expected["boot_id"]
        and "/sys/fs/cgroup" + selected.cgroup == expected["cgroup_path"]
        and boundary.readback is not None
        and boundary.actor is not None,
        "Previously authenticated measured6GiB unit required",
    )
    boundary.owned_status()
    actor = {key: boundary.actor[key] for key in ("pid", "start_ticks", "pgid", "sid")}
    directory = Path(directory).absolute()
    require(
        directory.is_dir()
        and not any(p.is_symlink() for p in (directory, *directory.parents))
        and not (directory / "sampler.json").exists(),
        "Fresh sampler output",
    )
    require(
        type(deadline) is float and time.monotonic() < deadline <= time.monotonic() + 10800,
        "Selected bounded external monitor deadline",
    )
    return _sample_loop(
        policy,
        observations,
        expected,
        actor,
        directory,
        stop_event,
        deadline,
        lambda error: boundary.close(primary_error=error),
    )


def _sample_loop(
    policy,
    observations,
    expected,
    actor,
    directory,
    stop_event,
    deadline,
    close_owned,
    *,
    clock=time.monotonic,
):
    started, previous, baseline = clock(), None, None
    result = {
        "schema_id": "decision3-measured-phase1-host-sampler-v1",
        "period_seconds": PERIOD_SECONDS,
        "scope": "Observed50ms host/cgroup samples; no between-sample or CUDA peak guarantee",
        "cuda_observed": False,
        "score": None,
        "samples": 0,
        "max_gap_seconds": 0.0,
        "min_host_available_bytes": None,
        "cgroups": {},
        "last_samples": [],
        "error": None,
        "source_and_whole_unit_closure_external": True,
        "continuation_authorized": False,
    }
    primary = None
    close_attempted = False
    try:
        while not stop_event.is_set():
            now = clock()
            require(now < deadline, "External sampler deadline")
            snapshot = observations.host_snapshot(expected, process=actor)
            decision = policy.host_decision(snapshot, expected, baseline)
            row = {"elapsed_seconds": clock() - started, "snapshot": snapshot, "decision": decision}
            result["last_samples"] = (result["last_samples"] + [row])[-32:]
            if previous is not None:
                result["max_gap_seconds"] = max(result["max_gap_seconds"], now - previous)
            previous = now
            result["samples"] += 1
            require(decision["admitted"] is True, "Host sampler resource refusal")
            baseline = decision["baseline"]
            available = snapshot["proc_meminfo"]["MemAvailable"]
            result["min_host_available_bytes"] = min(
                result["min_host_available_bytes"] or available, available
            )
            for group in decision["cgroups"]:
                record = result["cgroups"].setdefault(
                    group["path"],
                    {
                        "max_current_bytes": 0,
                        "max_kernel_peak_bytes": 0,
                        "minimum_headroom_bytes": group["headroom_bytes"],
                        "latest": None,
                    },
                )
                record["max_current_bytes"] = max(record["max_current_bytes"], group["current"])
                record["max_kernel_peak_bytes"] = max(
                    record["max_kernel_peak_bytes"], group["peak"]
                )
                if group["headroom_bytes"] is not None:
                    record["minimum_headroom_bytes"] = min(
                        record["minimum_headroom_bytes"], group["headroom_bytes"]
                    )
                record["latest"] = group
            stop_event.wait(max(0.0, min(PERIOD_SECONDS - (clock() - now), deadline - clock())))
        require(result["samples"] > 0, "At least one successful external sample required")
    except BaseException as error:
        primary = error
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            close_attempted = True
            result["owned_close"] = close_owned(result["error"])
        except BaseException as secondary:
            result["owned_close_error"] = {
                "type": type(secondary).__name__,
                "message": str(secondary),
            }
            error.add_note("Authenticated selected-unit cleanup also failed")
    finally:
        result["elapsed_seconds"] = clock() - started
        if previous is not None:
            result["max_gap_seconds"] = max(result["max_gap_seconds"], clock() - previous)
        result["successful_stop_requested"] = primary is None
        try:
            save_new(directory / "sampler.json", result)
        except BaseException as secondary:
            if primary is None:
                primary = secondary
            else:
                primary.add_note("Sampler receipt persistence also failed")
            if not close_attempted:
                close_attempted = True
                try:
                    close_owned(
                        {
                            "type": type(primary).__name__,
                            "message": "Sampler evidence persistence failed",
                        }
                    )
                except BaseException:
                    primary.add_note(
                        "Authenticated selected-unit cleanup also failed after persistence refusal"
                    )
    if primary is not None:
        raise primary
    return result
