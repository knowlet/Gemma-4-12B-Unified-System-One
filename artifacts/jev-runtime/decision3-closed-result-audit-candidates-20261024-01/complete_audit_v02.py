"""HELD standalone complete1479 closed-result audit v02 for full06 actual04.

No candidate code has been run or parsed during authoring. Default CLI does not
read actual evidence. --execute is for root only AFTER its source review and the
original NN/controller are terminal. No partial-stage fallback is accepted here.
It never imports a target module, runs a scorer/model, repairs an original flag,
or accepts a later absence observation in place of the original closure.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
RUNTIME = ROOT / "artifacts/jev-runtime"
OWNER = RUNTIME / "llama-cpp-owner-20261024-07"
CAMPAIGN = RUNTIME / "decision3-publisher-batch4-cpu-both-20261024-06"
NATIVE = RUNTIME / "decision3-publisher-batch4-cpu-20261024-03"
OUTPUT = OWNER / "actual-04"
ACTUAL = OUTPUT / "edge-both"
PUBLIC = "legacy-public231"
COHERENCE = "jevbench-mini"
SCHEMA = "authenticated-native-batch4-stage-completion-v1"
PROFILE = "decision3-edge-gx10-publisher-cpu2g-coexist-batch4-both-stage-completions-ece-schema-20261024-v1"
OWNER_PROFILE = "llama-cpp-cpu4-2g-coexist-command-owned-v1"
UNIT = "jev-d3-llama-cpp-cpu4-2g-coexist-20261024-07.service"
POLICY = "comfy-observed-cpu-coexist-v1"
SPEC_SHA = "fbaa2fbe1fb2bcd95236a4ccf6f71e69b8b6a4feae78f98429ff04d6c22ecfee"
REVIEW_SHA = "5d090a58eaea8145aec9f76fd63203a0258d217110b7e0c063d5a5edd1ab3cb3"
PLAN_SHA = "351500bcdfdd0b1c3c6096b21bd767321e761a0af79a64ed080dad518d9d575c"
NATIVE_PLAN_SHA = "5efcef370b0ae09a78d334d6e00c4a4e39ebbaae9ad18fd822581e8979e1b6e4"
BENCHMARK_SHA = "b5caa794d864917a637034bd1cebe0487c9eec0e5c3c423835d24fca4f397021"
OWNER_SHA = "685a9f809f429f4e9658083dc8530b563740625f9ba7816908fdc58002de256d"
CAP = 2147483648
TOTAL = 130595864576
RESERVE = 26119172916
AUTH_KEYS = {
    "unchanged_batch_authenticate",
    "actual_runtime_identity_and_cpu_BF16_FP32_dtypes_validated",
    "all78_scorer_sources_plus2_dataset_metadata_authenticated",
    "campaign_plan_prerequisites_and_every_audit_ref_authenticated",
}
UNIVERSAL_FALSE = (
    "no_escape_enforced",
    "whole_tree_exit_confirmed",
    "continuation_authorized",
    "model_qualified",
)
SMOKE_QUALIFICATIONS = (
    "smoke_complete",
    "external_authenticated_closure",
    "one_actual_constructor",
    "two_actual_four_row_events",
    "durable_call_20s",
    "whole_command_120s",
    "final_full_model_and_sources_authenticated",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary unambiguous evidence path required",
    )
    return path


def signature(info):
    return tuple(
        getattr(info, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    )


def digest(path):
    path = ordinary(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), "Regular selected file required")
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
        after = os.fstat(stream.fileno())
    require(
        signature(before) == signature(after) == signature(path.stat()),
        "Selected bytes changed while hashing",
    )
    return h.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def ref(path):
    path = ordinary(path)
    label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    return {"path": label, "sha256": digest(path), "size_bytes": path.stat().st_size}


def path_from_ref(row):
    require(type(row) is dict and type(row.get("path")) is str, "Typed selected reference required")
    path = Path(row["path"])
    return ordinary(path if path.is_absolute() else ROOT / path)


def join_ref(row, *, require_sha=True):
    path = path_from_ref(row)
    info = path.stat()
    require(stat.S_ISREG(info.st_mode), "Regular pinned artifact required")
    size = row.get("size_bytes", row.get("size"))
    if size is not None:
        require(type(size) is int and 0 <= size == info.st_size, "Pinned artifact size differs")
    if "stat" in row:
        require(
            type(row["stat"]) is dict
            and row["stat"]
            and all(type(v) is int and getattr(info, k) == v for k, v in row["stat"].items()),
            "Original selected stat identity differs",
        )
    if row.get("sha256") is not None:
        require(
            type(row["sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            and digest(path) == row["sha256"],
            "Pinned artifact SHA differs",
        )
    else:
        require(
            not require_sha and "stat" in row, "Selected SHA or original stat-only scope required"
        )
    return path


def unique_pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def read_json(path, observations, *, maximum=16 * 1024**2, expected=None):
    path = ordinary(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum,
            "Bounded regular evidence JSON required",
        )
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    require(
        len(raw) == before.st_size
        and signature(before) == signature(after) == signature(path.stat()),
        "Evidence changed while reading",
    )
    sha = hashlib.sha256(raw).hexdigest()
    require(expected is None or sha == expected, "Selected JSON SHA differs")
    observations[path] = {"path": str(path), "sha256": sha, "size_bytes": len(raw)}
    return json.loads(
        raw,
        object_pairs_hook=unique_pairs,
        parse_constant=lambda _: require(False, "Nonfinite JSON refused"),
    )


def finite(value, *, minimum=0, maximum=math.inf, positive=False):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > minimum if positive else value >= minimum)
        and value <= maximum
    )


def validate_limits(limits):
    require(
        type(limits) is dict
        and set(limits) == {"memory.max", "memory.swap.max", "cpu.max"}
        and limits.get("memory.max") == str(CAP)
        and limits.get("memory.swap.max") == "0"
        and type(limits.get("cpu.max")) is str,
        "Original exact memory/swap and CPU fields required",
    )
    values = limits["cpu.max"].split()
    require(
        len(values) == 2 and all(re.fullmatch(r"[0-9]+", value) for value in values),
        "Original finite numeric quota/period required",
    )
    quota, period = map(int, values)
    require(period > 0 and quota == 4 * period, "Original four-CPU quota/period ratio differs")


def validate_host(gate, boot_id, *, initial):
    require(
        type(gate) is dict
        and gate.get("host_observation_policy") == POLICY
        and gate.get("initial") is initial,
        "Original host gate scope differs",
    )
    row = gate["observation"]
    require(
        row.get("boot_id") == boot_id
        and type(row.get("mem_total_bytes")) is int
        and row["mem_total_bytes"] == TOTAL,
        "Known same-boot host total required",
    )
    reserve = max(24 * 1024**3, (row["mem_total_bytes"] + 4) // 5)
    required = reserve + (CAP if initial else 0)
    require(
        reserve == RESERVE == gate.get("reserve_bytes")
        and gate.get("required_available_bytes") == required
        and type(row.get("mem_available_bytes")) is int
        and required <= row["mem_available_bytes"] <= TOTAL,
        "Original full host floor failed or unknown",
    )
    running, pending = row.get("comfy_running"), row.get("comfy_pending")
    require(
        type(running) is int and running >= 0 and type(pending) is int and pending >= 0,
        "Typed original Comfy counts required",
    )
    evidence = row.get("comfy_evidence")
    require(type(evidence) is dict, "Original Comfy evidence required")
    if row.get("comfy_state") in {"http_busy", "http_idle"}:
        require(
            evidence.get("transport") == "HTTP"
            and type(evidence.get("http_status")) is int
            and evidence["http_status"] == 200
            and type(evidence.get("response_sha256")) is str
            and re.fullmatch(r"[0-9a-f]{64}", evidence["response_sha256"])
            and type(evidence.get("comfy_running")) is int
            and type(evidence.get("comfy_pending")) is int
            and evidence["comfy_running"] == running
            and evidence["comfy_pending"] == pending
            and (row["comfy_state"] == "http_busy") == (running > 0 or pending > 0),
            "Strict known HTTP evidence differs",
        )
    else:
        require(
            row.get("comfy_state") == "absent_confirmed"
            and running == pending == 0
            and evidence.get("transport") == "connection_refused",
            "Original confirmed absence evidence required",
        )


def selected_sources(config, spec):
    require(
        type(spec.get("sources")) is list
        and len(spec["sources"]) == 54
        and type(config.get("base_sources")) is list
        and len(config["base_sources"]) == 5
        and config.get("sources") == config["base_sources"] + spec["sources"]
        and len(config["sources"]) == 59,
        "Exact54+5 source union required",
    )
    labels = set()
    for row in config["sources"]:
        require(
            set(row) == {"path", "sha256", "size_bytes"}
            and not Path(row["path"]).is_absolute()
            and row["path"] not in labels
            and type(row["size_bytes"]) is int
            and 0 <= row["size_bytes"] <= 1048576,
            "Original distinct bounded source record required",
        )
        labels.add(row["path"])
        join_ref(row)
    require(
        any(
            path_from_ref(row) == OWNER / "controller.py" and row["sha256"] == OWNER_SHA
            for row in config["base_sources"]
        ),
        "Original owner07 source pin missing",
    )
    require(
        config.get("binary_stats") == spec.get("binary_stats"), "Selected interpreter stats differ"
    )
    for row in spec["binary_stats"]:
        require(set(row) == {"path", "stat"}, "Original stat-only executable record required")
        join_ref(row, require_sha=False)


def check_original_closure(config, receipt, closure, readback):
    require(closure == receipt.get("closure"), "Original closure differs from owner receipt")
    require(
        closure.get("closure_errors") == []
        and closure.get("primary_error") == receipt["primary_error"],
        "Original primary/closure error record differs",
    )
    for key in (
        "post_stop_terminal_verified",
        "cgroup_absent",
        "sources_unchanged",
        "whole_unit_exit_confirmed",
        "observed_cgroup_cohort_exit_confirmed",
    ):
        require(closure.get(key) is True, "Original current closure is not fully positive: " + key)
    require(
        all(closure.get(key) is False for key in UNIVERSAL_FALSE),
        "Universal/continuation/model nonqualification flags must remain false",
    )
    scans = closure.get("two_process_scans")
    require(
        type(scans) is list
        and len(scans) == 2
        and all(
            type(row) is dict and row.get("matches") == row.get("unknown") == [] for row in scans
        ),
        "Original exact two empty known scans required; no later replacement",
    )
    selection = closure["selection"]
    require(
        selection
        == {
            "unit": UNIT,
            "nonce": config["nonce"],
            "boot_id": config["boot_id"],
            "memory_max_bytes": CAP,
            "runtime_seconds": 10800,
            "stop_seconds": 3,
        },
        "Original closure selected unit/config differs",
    )
    actor, before, after = closure["actor"], closure["before_stop"], closure["after_stop"]
    group = f"/user.slice/user-{actor['uid']}.slice/user@{actor['uid']}.service/app.slice/{UNIT}"
    require(
        actor["pid"] == actor["pgid"] == actor["sid"]
        and actor["cgroup"] == group
        and type(actor["start_ticks"]) is int
        and actor["start_ticks"] > 0,
        "Original admitted wrapper identity differs",
    )
    require(
        type(closure.get("invocation_id")) is str
        and re.fullmatch(r"[0-9a-f]{32}", closure["invocation_id"]),
        "Original invocation ID required",
    )
    properties = {
        "Type": "exec",
        "Description": config["nonce"],
        "MemoryAccounting": "yes",
        "MemoryMax": str(CAP),
        "MemoryHigh": "infinity",
        "MemorySwapMax": "0",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxUSec": "3h",
        "TimeoutStopUSec": "3s",
        "KillSignal": "15",
        "FinalKillSignal": "9",
        "SendSIGKILL": "yes",
        "RemainAfterExit": "yes",
        "Restart": "no",
        "Delegate": "no",
    }
    require(
        readback.get("actor") == actor
        and readback.get("kernel")
        == {"memory.max": str(CAP), "memory.swap.max": "0", "cgroup.type": "domain"}
        and all(readback["unit"].get(k) == v for k, v in properties.items())
        and readback["unit"].get("InvocationID") == closure["invocation_id"]
        and readback["unit"].get("MainPID") == str(actor["pid"])
        and readback["unit"].get("ControlGroup") == group,
        "Original live admission readback differs",
    )
    require(
        after.get("MainPID") == "0"
        and (
            (
                after.get("Description") == config["nonce"]
                and after.get("InvocationID") in (closure["invocation_id"], "")
                and after.get("ActiveState") in {"inactive", "failed"}
            )
            or (
                after.get("LoadState") == "not-found"
                and after.get("InvocationID") == ""
                and after.get("ControlGroup") == ""
            )
        ),
        "Original post-stop terminal state differs",
    )
    if closure.get("closure_mode") == "authenticated_stop":
        require(
            closure.get("stop_required") is True
            and type(closure.get("stop_result")) is str
            and before.get("Description") == config["nonce"]
            and before.get("InvocationID") == closure["invocation_id"]
            and before.get("MainPID") == str(actor["pid"])
            and before.get("ControlGroup") == group,
            "Original exact authenticated stop required",
        )
    else:
        samples = closure.get("already_terminal_identity_samples")
        require(
            closure.get("closure_mode") == "already_terminal_verified"
            and closure.get("stop_required") is False
            and closure.get("already_terminal_identity_sandwich_verified") is True
            and type(samples) is list
            and len(samples) == 6
            and all(row == before for row in samples)
            and before.get("LoadState") == "loaded"
            and before.get("Description") == config["nonce"]
            and before.get("InvocationID") == closure["invocation_id"]
            and before.get("MainPID") == "0"
            and before.get("ExecMainPID") == str(actor["pid"])
            and (before.get("ActiveState"), before.get("SubState"))
            in {("failed", "failed"), ("inactive", "dead")}
            and before.get("ControlGroup") in ("", group)
            and all(before.get(k) == v for k, v in properties.items()),
            "Only original fully verified terminal reconciliation is admissible",
        )
    return actor


def public_metrics(aggregate):
    integers = (
        "n_planned",
        "n_attempted",
        "n_scorable",
        "n_valid",
        "n_correct",
        "n_renormalized",
        "calibration_n",
    )
    numbers = (
        "accuracy",
        "macro_accuracy",
        "coverage",
        "schema_validity",
        "schema_validity_strict",
        "operational_success",
        "brier_mean",
        "ordinal_mae",
        "price_per_1000_decisions_usd",
        "ledger_charged_usd",
    )
    require(
        type(aggregate) is dict
        and aggregate.get("complete") is True
        and aggregate.get("n_planned") == aggregate.get("n_attempted") == 231,
        "Original complete public231 aggregate required",
    )
    projected = {key: aggregate[key] for key in integers + numbers}
    require(
        all(type(projected[k]) is int and 0 <= projected[k] <= 231 for k in integers),
        "Typed public global counts required",
    )
    require(
        projected["n_correct"] <= projected["n_scorable"],
        "Correct public count cannot exceed the official scorable population",
    )
    require(
        all(projected[k] is None or finite(projected[k], minimum=-math.inf) for k in numbers),
        "Finite typed official scalar metrics required",
    )
    ece = aggregate["ece"]
    if ece is None:
        require(projected["calibration_n"] == 0, "Absent ECE requires zero pairs")
    else:
        require(
            type(ece) is dict
            and set(ece) == {"ece", "n", "bins"}
            and finite(ece["ece"], maximum=1)
            and type(ece["n"]) is int
            and ece["n"] == projected["calibration_n"]
            and type(ece["bins"]) is list
            and len(ece["bins"]) == 10,
            "Exact original structured ECE schema required",
        )
        for index, item in enumerate(ece["bins"]):
            require(
                type(item) is dict
                and set(item) == {"lo", "hi", "n", "mean_confidence", "accuracy"}
                and finite(item["lo"], maximum=1)
                and finite(item["hi"], maximum=1)
                and item["lo"] == index / 10
                and item["hi"] == (index + 1) / 10
                and type(item["n"]) is int
                and 0 <= item["n"] <= ece["n"],
                "Original typed ECE global bin differs",
            )
            require(
                (item["mean_confidence"] is None and item["accuracy"] is None)
                if item["n"] == 0
                else (
                    finite(item["mean_confidence"], maximum=1)
                    and finite(item["accuracy"], maximum=1)
                ),
                "Original ECE bin means differ",
            )
        require(sum(item["n"] for item in ece["bins"]) == ece["n"], "ECE global pair count differs")
    projected.update(ece=ece, complete=True)
    # Official summarize.metric uses n_scorable, not an invented denominator.
    require(
        aggregate["accuracy"]
        == (aggregate["n_correct"] / aggregate["n_scorable"] if aggregate["n_scorable"] else None),
        "Official aggregate accuracy/count consistency differs",
    )
    return projected


def coherence_metrics(aggregate):
    """Original stage_metrics projection: six globals, not item scores/errors."""
    require(type(aggregate) is dict, "Original coherence aggregate required")
    expected_counts = {"requests": 1248, "cases": 240, "relations": 50, "planned_tests": 1200}
    require(
        all(
            type(aggregate.get(k)) is int and aggregate[k] == value
            for k, value in expected_counts.items()
        ),
        "Original full coherence populations required",
    )
    result = {key: aggregate[key] for key in ("overall", "interval", *expected_counts)}
    require(
        result["overall"] is None or finite(result["overall"], minimum=-math.inf),
        "Original coherence overall must be finite or absent",
    )
    interval = result["interval"]
    require(
        interval is None
        or (
            type(interval) in (list, tuple)
            and len(interval) == 2
            and all(finite(v, minimum=-math.inf) for v in interval)
        ),
        "Original coherence interval must be two finite values or absent",
    )
    if interval is not None:
        result["interval"] = list(interval)
    return result


def validate_latency(row, count):
    require(
        type(row) is dict
        and set(row)
        == {
            "count",
            "median_seconds",
            "p95_seconds",
            "max_seconds",
            "quantile_method",
            "startup_included",
            "browser_20_second_qualification",
            "unit",
        }
        and type(row.get("count")) is int
        and row["count"] == count,
        "Original complete typed latency population required",
    )
    require(
        all(
            finite(row[key], maximum=180)
            for key in ("median_seconds", "p95_seconds", "max_seconds")
        )
        and row["median_seconds"] <= row["p95_seconds"] <= row["max_seconds"]
        and row["quantile_method"] == "nearest_rank_p95; ordinary_median"
        and row["startup_included"] is False
        and row["browser_20_second_qualification"] is False
        and row["unit"]
        == "one unchanged complete state/questions callback; native question chunks of up to4",
        "Original complete callback180s latency scope differs",
    )


def stable_wrapper(left, right):
    return all(
        left.get(key) == right.get(key)
        for key in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")
    )


def aware_epoch(value):
    require(type(value) is str, "Original UTC timestamp string required")
    parsed = datetime.datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, "Explicit timezone on original terminal timestamp required")
    result = parsed.timestamp()
    require(finite(result, positive=True), "Finite original terminal epoch required")
    return result


def counter_diagnostic(snapshot, group):
    """Typed observations only; never impose a new zero-pressure/peak gate."""
    require(
        type(snapshot) is dict
        and snapshot.get("cgroup") == group
        and finite(snapshot.get("observed_monotonic"), positive=True)
        and type(snapshot.get("counters")) is dict,
        "Original owned counter snapshot required",
    )
    counters = snapshot["counters"]
    require(
        all(
            type(counters.get(k)) is int and counters[k] >= 0
            for k in ("memory.current", "memory.peak")
        )
        and type(counters.get("memory.events")) is dict
        and {"oom", "oom_kill"} <= set(counters["memory.events"])
        and all(type(v) is int and v >= 0 for v in counters["memory.events"].values())
        and type(counters.get("cpu.stat")) is dict
        and all(type(v) is int and v >= 0 for v in counters["cpu.stat"].values()),
        "Original numeric memory/CPU counters required",
    )
    return {
        "observed_monotonic": snapshot["observed_monotonic"],
        "counters": counters,
        "configured_memcg_limit_bytes": CAP,
        "observed_peak_within_configured_limit": counters["memory.peak"] <= CAP,
        "pressure_or_oom_event_observed": any(
            value > 0
            for key, value in counters["memory.events"].items()
            if key in {"max", "oom", "oom_kill", "oom_group_kill"}
        ),
        "role": "diagnostic; not an additional score acceptance threshold or combined physical cap proof",
    }


def selected_stage_sources(plan, spec, benchmark_path):
    argv = spec["commands"][0]["argv"]

    def argument(name):
        require(argv.count(name) == 1, "Exact original single CLI field required")
        return argv[argv.index(name) + 1]

    smoke_path, audit_path = Path(argument("--smoke-receipt")), Path(argument("--smoke-audit"))
    smoke_sha, audit_sha = argument("--smoke-receipt-sha256"), argument("--smoke-audit-sha256")
    sources = [
        (CAMPAIGN / "edge-plan.json", PLAN_SHA),
        (CAMPAIGN / "campaign.py", plan["campaign_sha256"]),
        (NATIVE / "actor.py", plan["batch_actor_sha256"]),
        (NATIVE / "freeze.json", plan["batch_freeze_sha256"]),
        (NATIVE / "edge-plan.json", NATIVE_PLAN_SHA),
        (benchmark_path, BENCHMARK_SHA),
        (
            NATIVE / "resource_helper.py",
            "7d90f15adee339ed292510d3f5c10c0b33b505279de32643fe1c3da425554552",
        ),
        (
            NATIVE / "source_audit.py",
            "ae299d87cbf7338bb9ff29055efc335e96a44b73a29370e08428aad2fb2b68d3",
        ),
        (smoke_path, smoke_sha),
        (audit_path, audit_sha),
    ]
    return sources, smoke_path, smoke_sha, audit_path, audit_sha


def verify_prerequisite(audit, native, smoke_path, smoke_sha):
    require(
        audit.get("schema") == "independent-native-batch4-synthetic-result-audit-v1"
        and audit.get("review_kind") == "closed_real_smoke_result"
        and audit.get("result") in {"PASS", "PASS_WITH_LIMITS"}
        and audit.get("model_key") == "edge"
        and audit.get("owner_unit") == UNIT
        and audit.get("memory_max_bytes") == CAP
        and audit.get("batch_size") == 4
        and audit.get("host_observation_policy") == POLICY
        and audit.get("model_revision") == native["model"]["revision"]
        and audit.get("model_identity_sha256") == native["model"]["model_sha256"]
        and audit.get("manifest_sha256") == native["model"]["manifest_sha256"]
        and all(audit.get("qualification", {}).get(key) is True for key in SMOKE_QUALIFICATIONS),
        "Original independent closed smoke prerequisite differs",
    )
    refs = audit["artifact_refs"]
    require(
        type(refs) is dict
        and {"state", "plan", "owner_receipt", "root_execution", "spec", "closure_observation"}
        <= set(refs)
        and path_from_ref(refs["state"]) == smoke_path
        and refs["state"]["sha256"] == smoke_sha
        and path_from_ref(refs["plan"]) == NATIVE / "edge-plan.json"
        and refs["plan"]["sha256"] == NATIVE_PLAN_SHA,
        "Exact original smoke/audit references required",
    )
    for row in refs.values():
        path = path_from_ref(row)
        require(
            path.suffix.lower() not in {".safetensors", ".bin", ".pt", ".npz", ".gguf", ".so"}
            and path.name not in {"tokenizer.json", "merges.txt", "vocab.json"}
            and path.stat().st_size <= 1048576,
            "Original selected small prerequisite ref required",
        )
        join_ref(row)


def execute():
    observations = {}
    terminal_path = OWNER / "root-execution-full06-edge-both-04.json"
    receipt_path = OUTPUT / "receipt.json"
    # No actual state/stage/score/model read before BOTH original terminal anchors.
    require(
        ordinary(terminal_path).is_file() and ordinary(receipt_path).is_file(),
        "Original root terminal and original owner receipt must both exist first",
    )
    terminal = read_json(terminal_path, observations)
    receipt = read_json(receipt_path, observations)
    require(
        type(terminal.get("exit_code")) is int
        and terminal["exit_code"] == 0
        and finite(terminal.get("wall_seconds"), maximum=10800, positive=True),
        "Complete audit requires original successful root terminal within owner budget",
    )
    require(type(terminal.get("stderr")) is str, "Original stderr capture must be a string")
    stderr_bytes = terminal["stderr"].encode("utf-8")
    stderr_diagnostic = {
        "nonempty": bool(stderr_bytes),
        "utf8_bytes": len(stderr_bytes),
        "sha256": hashlib.sha256(stderr_bytes).hexdigest(),
        "empty_stderr_acceptance_gate_added": False,
        "raw_text_included": False,
    }
    root_started = aware_epoch(terminal["started_at_utc"])
    root_finished = aware_epoch(terminal["finished_at_utc"])
    require(root_started < root_finished, "Ordered original root clocks required")
    require(
        receipt.get("completed") is True
        and receipt.get("primary_error") is None
        and receipt.get("secondary_errors") == []
        and receipt.get("sources_unchanged") is True
        and receipt.get("comfy_service_modified") is False
        and receipt.get("whole_tree_exit_confirmed") is False
        and receipt.get("profile") == OWNER_PROFILE
        and receipt.get("unit_attempted") is True,
        "Original owner success/source/nonqualification predicates required",
    )
    config = receipt["selected_config"]
    require(
        config.get("selection_sha256") == terminal.get("selection_sha256") == SPEC_SHA
        and terminal.get("independent_source_selection_review_sha256") == REVIEW_SHA
        and config.get("profile") == OWNER_PROFILE
        and config.get("unit") == UNIT
        and config.get("output") == str(OUTPUT)
        and config.get("runtime_seconds") == 10800
        and config.get("command_count") == 1
        and config.get("mem_total_bytes") == TOTAL
        and type(config.get("nonce")) is str
        and re.fullmatch(r"jev-d3-boundary-[0-9a-f]{32}", config["nonce"]),
        "Exact current selected owner/spec/invocation differs",
    )
    config_path = OUTPUT / "sealed-config.json"
    require(read_json(config_path, observations) == config, "Original sealed config differs")
    config_sha = digest(config_path)
    spec_path = CAMPAIGN / "root-edge-both-command-spec-01.json"
    require(config.get("selection") == str(spec_path), "Exact original spec path required")
    controller_argv = [
        str(ROOT / ".venv/bin/python"),
        "-B",
        "-s",
        str(OWNER / "controller.py"),
        "--run",
        "--expected-entry-sha256",
        OWNER_SHA,
        "--selection",
        str(spec_path),
        "--selection-sha256",
        SPEC_SHA,
        "--output",
        str(OUTPUT),
    ]
    require(
        terminal.get("profile") == "root-selected-full06-retry04-Edge-coexist-terminal-v1"
        and terminal.get("argv") == controller_argv,
        "Original current root controller profile/argv differs",
    )
    spec = read_json(spec_path, observations, maximum=1048576, expected=SPEC_SHA)
    require(
        spec.get("profile") == "llama-cpp-command-spec-v1"
        and spec.get("runtime_seconds") == 10800
        and len(spec.get("commands", [])) == 1,
        "Original command profile/count/deadline differs",
    )
    selected_sources(config, spec)
    closure_path = OUTPUT / "external-boundary-closure.json"
    readback_path = OUTPUT / "unit-readback.json"
    closure = read_json(closure_path, observations)
    readback = read_json(readback_path, observations)
    actor = check_original_closure(config, receipt, closure, readback)
    require(
        closure.get("closure_mode") == "authenticated_stop",
        "Original successful owner requires its authenticated live-main stop",
    )
    owner_state = receipt["state"]
    require(
        owner_state.get("actor_success") is True
        and owner_state.get("preclose_exit_memory_authenticated") is True
        and owner_state.get("preclose_main_live") is True,
        "Original successful live-main/command result predicates required",
    )
    validate_host(owner_state["initial_host"], config["boot_id"], initial=True)
    validate_host(receipt["final_host"], config["boot_id"], initial=False)
    admission_path = OUTPUT / "admission.json"
    admission = read_json(admission_path, observations)
    require(
        admission.get("config_sha256") == config_sha
        and admission.get("nonce") == config["nonce"]
        and admission.get("invocation_id") == closure["invocation_id"]
        and stable_wrapper(admission["actor"], actor),
        "Original admission/config/closure identity differs",
    )
    bootstrap_path = OUTPUT / "bootstrap-identity.json"
    bootstrap = read_json(bootstrap_path, observations)
    require(
        bootstrap == owner_state["bootstrap_admission"]["identity"]
        and bootstrap.get("config_sha256") == config_sha
        and bootstrap.get("nonce") == config["nonce"]
        and bootstrap.get("torch_not_imported") is True
        and stable_wrapper(bootstrap["identity"]["process"], actor)
        and bootstrap.get("sys_executable") == str(ROOT / ".venv/bin/python")
        and bootstrap.get("observed_main_thread_affinity") == [16, 17, 18, 19]
        and bootstrap.get("python_flags") == {"no_user_site": 1, "dont_write_bytecode": True},
        "Original stdlib bootstrap/config/affinity identity differs",
    )
    bootstrap_argv = [
        str(ROOT / ".venv/bin/python"),
        "-B",
        "-s",
        str(OWNER / "controller.py"),
        "--internal-stage",
        "actor",
        "--config",
        str(config_path),
        "--config-sha256",
        config_sha,
    ]
    require(
        bootstrap["identity"].get("argv") == bootstrap_argv
        and bootstrap["identity"].get("cwd") == str(ROOT)
        and bootstrap["identity"].get("exe")
        == str((ROOT / ".venv/bin/python").resolve(strict=True)),
        "Original exact bootstrap executable/argv/cwd differs",
    )
    actor_exit = owner_state["actor_exit"]
    actor_exit_path = OUTPUT / "actor-exit.json"
    require(
        read_json(actor_exit_path, observations) == actor_exit
        and actor_exit.get("config_sha256") == config_sha
        and stable_wrapper(actor_exit["actor"], actor)
        and type(actor_exit.get("entry_exit_code")) is int
        and actor_exit["entry_exit_code"] == 0
        and actor_exit.get("error") is None
        and actor_exit.get("planned_command_count") == 1
        and len(actor_exit.get("command_results", [])) == 1,
        "Original selected wrapper successful result/config differs",
    )
    require(
        finite(owner_state.get("closure_finished_monotonic"), positive=True)
        and finite(actor_exit.get("hold_started_monotonic"), positive=True)
        and actor_exit.get("postreceipt_hold_seconds") == 5
        and actor_exit.get("hold_deadline_monotonic") == actor_exit["hold_started_monotonic"] + 5
        and owner_state["closure_finished_monotonic"] < actor_exit["hold_deadline_monotonic"],
        "Original authenticated closure must finish inside the successful wrapper hold",
    )
    preclose = owner_state["preclose_exit_memory_raw"]
    require(
        preclose.get("Description") == config["nonce"]
        and preclose.get("InvocationID") == closure["invocation_id"]
        and preclose.get("MainPID") == str(actor["pid"]),
        "Original preclose observation identity differs",
    )

    # Only now read the actual campaign state, plans and stage/result aggregates.
    state_path = ACTUAL / "state.json"
    state = read_json(state_path, observations)
    command = actor_exit["command_results"][0]
    require(
        type(command.get("returncode")) is int
        and command["returncode"] == 0
        and finite(command.get("elapsed_seconds"), maximum=10800, positive=True),
        "Original selected command must succeed within owner budget",
    )
    require(
        state.get("status") == "both_original_suites_complete_native_batch4_cpu_profile_only"
        and state.get("profile") == PROFILE
        and state.get("model_key") == "edge"
        and state.get("plan_sha256") == PLAN_SHA
        and all(
            type(state.get(key)) is int and state[key] == 1479
            for key in ("started_calls", "completed_calls", "raw_native_returns")
        )
        and type(state.get("model_loads")) is int
        and state["model_loads"] == 1
        and state.get("cuda_initialized_final") is False
        and state.get(
            "final_full_model_manifest_runtime_and78_sources_plus2_metadata_authenticated"
        )
        is True
        and state.get("whole_campaign_deadline_seconds") == 10000
        and state.get("per_call_deadline_seconds") == 180
        and state.get("torch_threads_final") == {"intraop": 4, "interop": 1}
        and set(state.get("completed_stage_receipts", {})) == {PUBLIC, COHERENCE}
        and set(state.get("latencies", {})) == {PUBLIC, COHERENCE}
        and finite(state.get("elapsed_seconds"), maximum=10000, positive=True)
        and state["elapsed_seconds"] <= command["elapsed_seconds"] <= terminal["wall_seconds"],
        "Full original1479/state/final-auth/campaign and owner clock predicates required",
    )
    limits = state["owned_limits"]
    validate_limits(limits)
    plan_path, native_path = CAMPAIGN / "edge-plan.json", NATIVE / "edge-plan.json"
    benchmark_path = RUNTIME / "decision3-publisher-lite-cpu-20261023-01/benchmark-source-plan.json"
    plan = read_json(plan_path, observations, maximum=1048576, expected=PLAN_SHA)
    native = read_json(native_path, observations, maximum=1048576, expected=NATIVE_PLAN_SHA)
    benchmark = read_json(benchmark_path, observations, maximum=1048576, expected=BENCHMARK_SHA)
    require(
        plan["model"] == native["model"]
        and plan["profile"] == PROFILE
        and native["owner_unit"] == UNIT
        and native["owner_profile"] == OWNER_PROFILE
        and native["host_observation_policy"] == POLICY
        and len(native["model_files"]) == 20
        and len(native["runtime_files"]) == 28
        and len(benchmark["source_files"]) == 78
        and len(benchmark["existing_dataset_metadata"]) == 2,
        "Exact model/runtime/scorer inventory differs",
    )
    expected_model = {
        **native["model"],
        "profile": PROFILE,
        "device": "cpu",
        "backbone_dtype": "bfloat16",
        "readout_dtype": "float32",
        "temperature": 1.0,
        "batch_size": 4,
        "context_limit": 16384,
        "context_policy": "reject_no_truncation",
        "permutation_average": False,
        "constructor_count": 1,
        "benchmark_heaven_revision": benchmark["public"]["revision"],
        "coherence_revision": benchmark["coherence"]["revision"],
        "source_plan_sha256": BENCHMARK_SHA,
    }
    require(state.get("model_identity") == expected_model, "Actual complete model identity differs")
    proc = state["process"]
    require(
        proc.get("affinity") == [16, 17, 18, 19]
        and proc.get("cgroup") == "0::" + actor["cgroup"]
        and proc.get("pgid") == actor["pgid"]
        and proc.get("sid") == actor["sid"],
        "Actual model process and admitted wrapper group differ",
    )
    start_environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "ACCELERATE_USE_CPU": "true",
    }
    require(
        all(proc.get("environment", {}).get(k) == v for k, v in start_environment.items()),
        "Original pre-loader START_ENV differs",
    )
    started_path, intent_path = (
        OUTPUT / "command-01-started.json",
        OUTPUT / "command-01-intent.json",
    )
    started, intent = read_json(started_path, observations), read_json(intent_path, observations)
    observed = command["observed_identity"]
    require(
        started.get("observed_identity") == observed
        and started.get("pid")
        == command.get("pid")
        == proc.get("pid")
        == observed["process"].get("pid")
        and observed["process"].get("start_ticks") == proc.get("starttime_ticks")
        and all(observed["process"].get(k) == actor[k] for k in ("pgid", "sid", "uid", "cgroup")),
        "Original independent child PID/start/UID/cohort join differs",
    )
    expected_argv = [part.replace("{OUTPUT}", str(OUTPUT)) for part in spec["commands"][0]["argv"]]
    expected_cwd = spec["commands"][0]["cwd"].replace("{OUTPUT}", str(OUTPUT))
    require(
        command.get("argv") == intent.get("argv") == observed.get("argv") == expected_argv
        and command.get("cwd")
        == intent.get("cwd")
        == observed.get("cwd")
        == expected_cwd
        == str(ROOT)
        and command.get("resolved_executable")
        == intent.get("resolved_executable")
        == observed.get("exe")
        == proc.get("executable")
        and command.get("executable_stat_at_start") == intent.get("executable_stat_at_start")
        and intent.get("shell") is False,
        "Original exact command executable/argv/cwd/stat join differs",
    )
    command_output_path = OUTPUT / "command-01-output.json"
    command_output = read_json(command_output_path, observations)
    require(
        command_output.get("result") == command
        and command_output.get("primary_error") is None
        and command_output.get("cleanup_errors") == []
        and command_output.get("child_poll_at_finalization") == 0,
        "Original selected command output/cleanup result differs",
    )
    executable_pin = next(row for row in spec["binary_stats"] if row["path"] == proc["executable"])
    require(
        command["executable_stat_at_start"] == executable_pin["stat"],
        "Original selected executable stat differs",
    )
    sources, smoke_path, smoke_sha, audit_path, audit_sha = selected_stage_sources(
        plan, spec, benchmark_path
    )
    require(
        sources[6][1] == native["helper_sha256"] and sources[7][1] == native["source_audit_sha256"],
        "Original helper/audit source pins differ",
    )
    for path, sha in sources:
        require(digest(path) == sha, "Postclosed selected stage source/prerequisite changed")
    audit = read_json(audit_path, observations, maximum=1048576, expected=audit_sha)
    verify_prerequisite(audit, native, smoke_path, smoke_sha)
    require(
        state.get("prerequisite")
        == {
            "path": str(audit_path.resolve()),
            "sha256": audit_sha,
            "result": audit["result"],
            "qualification": audit["qualification"],
            "artifact_refs": audit["artifact_refs"],
            "limits": audit.get("limits", []),
        },
        "Actual complete campaign prerequisite descriptor differs",
    )

    scores = state["benchmark_scores"]
    require(
        type(scores) is dict
        and set(scores)
        == {"profile", "legacy_public231", "coherence_mini", "official_rank", "browser_score"}
        and scores["profile"] == PROFILE
        and scores["official_rank"] is None
        and scores["browser_score"] is None,
        "Original complete scores container differs",
    )
    projected = {
        PUBLIC: public_metrics(scores["legacy_public231"]),
        COHERENCE: coherence_metrics(scores["coherence_mini"]),
    }
    require(
        type(scores["coherence_mini"].get("errors")) is list,
        "Original Report.errors list must be preserved",
    )
    errors = scores["coherence_mini"]["errors"]
    # Report.errors lists backend/base/build/judge/minimize exceptions; original
    # campaign does not require an empty list. No raw error text is copied.
    error_diagnostic = {
        "count": len(errors),
        "canonical_sha256": canonical_digest(errors),
        "zero_error_acceptance_gate_added": False,
        "source": "pinned coherence Report.errors -> result['errors']; original campaign preserves this list",
        "raw_error_text_included": False,
    }
    stage_refs, previous_cleanup = {}, None
    began, expected_invocation = None, None
    for stage, count in ((PUBLIC, 231), (COHERENCE, 1248)):
        selected = state["completed_stage_receipts"][stage]
        require(selected.get("stage") == stage, "Selected stage reference label differs")
        completion_path = join_ref(selected)
        require(
            completion_path == ACTUAL / "stage-completions" / stage / "completion.json",
            "Exact current stage completion path required",
        )
        completion = read_json(completion_path, observations)
        require(
            completion.get("schema") == SCHEMA
            and completion.get("stage") == stage
            and completion.get("profile") == PROFILE
            and completion.get("status")
            == "authenticated_stage_complete_pending_independent_current_root_audit"
            and completion.get("owner_nonce") == config["nonce"]
            and completion.get("whole_campaign_seconds") == 10000
            and completion.get("overall_campaign_complete") is False
            and completion.get("external_current_command_and_owner_closure_required") is True
            and completion.get("cooperative_publication_not_hard_realtime_or_no_escape") is True,
            "Original authenticated current stage completion differs",
        )
        evidence_path = join_ref(completion["evidence"])
        require(
            evidence_path == completion_path.parent / "evidence.json",
            "Exact current stage evidence path required",
        )
        evidence = read_json(evidence_path, observations)
        cumulative = 231 if stage == PUBLIC else 1479
        require(
            evidence.get("schema") == SCHEMA
            and evidence.get("stage") == stage
            and evidence.get("profile") == PROFILE
            and evidence.get("model_key") == "edge"
            and evidence.get("status") == "evidence_only_not_a_stage_completion"
            and evidence.get("counts")
            == {
                "stage_callbacks": count,
                "cumulative_callbacks": cumulative,
                "started": cumulative,
                "completed": cumulative,
                "raw_native_returns": cumulative,
            }
            and set(evidence.get("full_authentication", {})) == AUTH_KEYS
            and all(v is True for v in evidence["full_authentication"].values())
            and evidence.get("metrics") == projected[stage]
            and set(evidence["metrics"]) == set(projected[stage])
            and evidence.get("raw_tasks_gold_responses_or_per_item_scores_included") is False,
            "Exact typed complete stage population/authentication/metric schema differs",
        )
        stage_owner = evidence["owner"]
        require(
            set(stage_owner) == {"unit", "profile", "nonce", "process", "limits"}
            and stage_owner.get("unit") == UNIT
            and stage_owner.get("profile") == OWNER_PROFILE
            and stage_owner.get("nonce") == config["nonce"]
            and stage_owner.get("limits") == limits
            and all(
                stage_owner["process"].get(k) == proc.get(k)
                for k in (
                    "pid",
                    "starttime_ticks",
                    "pgid",
                    "sid",
                    "cgroup",
                    "affinity",
                    "executable",
                )
            )
            and stage_owner["process"].get("environment") == native["environment"],
            "Stage immutable process identity or post-loader CPU_ENV differs",
        )
        validate_limits(stage_owner["limits"])
        invocation = evidence["invocation"]
        require(
            invocation.get("output") == str(ACTUAL)
            and invocation.get("plan_sha256") == PLAN_SHA
            and invocation.get("campaign_started_unix") == state.get("campaign_started_unix")
            and finite(
                invocation.get("campaign_started_unix"), minimum=root_started, maximum=root_finished
            ),
            "Stage current output/plan/epoch invocation differs",
        )
        if expected_invocation is None:
            expected_invocation, began = invocation, invocation["campaign_started_monotonic"]
            require(
                finite(began, positive=True)
                and finite(started.get("started_monotonic"), positive=True)
                and started["started_monotonic"] <= began,
                "Original execute-entry clock differs",
            )
        else:
            require(
                invocation == expected_invocation,
                "Both stages must share one exact execute-entry clock",
            )
        clock, cleanup = evidence["clock"], completion["after_stage_deadline_cleanup_monotonic"]
        publication_elapsed = selected["elapsed_campaign_seconds_after_publication"]
        require(
            finite(clock.get("stage_started_monotonic"), minimum=began)
            and finite(
                clock.get("authenticated_monotonic"), minimum=clock["stage_started_monotonic"]
            )
            and finite(cleanup, minimum=clock["authenticated_monotonic"])
            and finite(clock.get("elapsed_campaign_seconds"), maximum=10000)
            and clock["elapsed_campaign_seconds"] <= clock["authenticated_monotonic"] - began
            and finite(publication_elapsed, maximum=10000)
            and cleanup - began <= publication_elapsed <= state["elapsed_seconds"]
            and clock.get("whole_campaign_seconds") == 10000
            and clock.get("per_call_seconds") == 180
            and clock.get("owner_seconds") == 10800,
            "Original shared whole10000 stage/auth/publication clocks differ",
        )
        if previous_cleanup is not None:
            require(
                previous_cleanup <= clock["stage_started_monotonic"],
                "Original public then coherence order differs",
            )
        previous_cleanup = cleanup
        validate_latency(evidence["latency"], count)
        require(
            evidence["latency"] == state["latencies"][stage], "Final exact stage latency changed"
        )
        pins = evidence["pins"]
        require(
            pins.get("model") == native["model"]
            and pins.get("manifest") == native["manifest"]
            and pins.get("model_files") == native["model_files"]
            and pins.get("runtime_files") == native["runtime_files"]
            and pins.get("owner_sources") == native["owner_sources"]
            and pins.get("benchmark_source_plan_sha256") == BENCHMARK_SHA
            and pins.get("scorer_sources") == benchmark["source_files"]
            and pins.get("dataset_metadata") == benchmark["existing_dataset_metadata"]
            and pins.get("campaign_sources_and_prerequisites")
            == [{"path": str(path), "sha256": sha} for path, sha in sources],
            "Exact stage model/runtime/scorer/prerequisite pin union differs",
        )
        stage_refs[stage] = {"completion": ref(completion_path), "evidence": ref(evidence_path)}

    # Rehash selected original model20+manifest/runtime28/owner and scorer78+data2
    # only now, with terminal/current closure established. No task/gold rescore.
    records = (
        [native["manifest"]]
        + native["model_files"]
        + native["runtime_files"]
        + native["owner_sources"]
        + benchmark["source_files"]
        + benchmark["existing_dataset_metadata"]
    )
    checked = {}
    for record in records:
        path = path_from_ref(record)
        join_ref(record, require_sha=bool(record.get("sha256")))
        checked[str(path)] = {
            "sha256": record.get("sha256"),
            "stat_only": record.get("sha256") is None,
        }
    summary_path, manifest_path = (
        ACTUAL / "public231/summary.json",
        ACTUAL / "public231/manifest.json",
    )
    summary, manifest = (
        read_json(summary_path, observations),
        read_json(manifest_path, observations),
    )
    require(
        public_metrics(summary) == projected[PUBLIC]
        and summary.get("identity") == expected_model
        and summary.get("source_revision") == benchmark["public"]["revision"]
        and summary.get("official_rank") is None
        and summary.get("official_composite") is None
        and summary.get("sealed_evaluation") == "not_run"
        and manifest.get("n_attempted") == 231
        and manifest.get("n_valid") == projected[PUBLIC]["n_valid"]
        and manifest.get("status") in {"completed", "completed_with_errors"}
        and manifest.get("identity") == expected_model
        and manifest.get("artifact_sha256", {}).get("summary.json") == digest(summary_path),
        "Original official public summary/manifest does not match stage/global aggregate",
    )
    results_path = ACTUAL / "results.json"
    results = read_json(results_path, observations)
    require(
        results.get("status") == "pending_durable_campaign_validation"
        and results.get("benchmark_scores") == scores
        and results.get("latencies") == state["latencies"]
        and results.get("model_identity") == expected_model
        and results.get("browser_score") is None
        and results.get("latency_startup_included") is False,
        "Original durable complete results/state handoff differs",
    )
    before_memory = counter_diagnostic(state["before_load_owned_counters"], proc["cgroup"])
    after_memory = counter_diagnostic(state["after_final_auth_owned_counters"], proc["cgroup"])
    require(
        before_memory["observed_monotonic"] <= began + state["elapsed_seconds"]
        and before_memory["observed_monotonic"]
        <= after_memory["observed_monotonic"]
        <= began + state["elapsed_seconds"],
        "Original counter clock scopes differ",
    )
    last_properties = owner_state["last_live_properties"]
    require(
        type(last_properties.get("kernel_memory_peak_bytes")) is int
        and last_properties["kernel_memory_peak_bytes"] >= 0
        and type(last_properties.get("kernel_memory_events")) is dict,
        "Original final live kernel diagnostics required",
    )
    final_events = last_properties["kernel_memory_events"]["counters"]
    require(
        type(final_events) is dict
        and {"oom", "oom_kill"} <= set(final_events)
        and all(type(v) is int and v >= 0 for v in final_events.values()),
        "Original final live memory event counters must be numeric",
    )
    raw_peak = preclose.get("MemoryPeak")
    manager_peak = int(raw_peak) if type(raw_peak) is str and raw_peak.isdigit() else None
    memory_diagnostic = {
        "selected_memcg_limit_bytes": CAP,
        "swap_limit_bytes": 0,
        "before_load": before_memory,
        "after_final_authentication": after_memory,
        "final_live_kernel_peak_bytes": last_properties["kernel_memory_peak_bytes"],
        "final_live_kernel_events": final_events,
        "systemd_preclose_peak_bytes": manager_peak,
        "final_live_peak_within_selected_limit": last_properties["kernel_memory_peak_bytes"] <= CAP,
        "final_live_pressure_or_oom_observed": any(
            v > 0
            for k, v in final_events.items()
            if k in {"max", "oom", "oom_kill", "oom_group_kill"}
        ),
        "zero_pressure_or_peak_within_cap_score_gate_added": False,
        "scope": "Typed selected-unit sampling/peaks only; diagnostic separate from original full-score qualification; no combined physical UMA cap or universal all-process fit proof",
    }
    # Bookend every parsed artifact so a concurrent metadata edit cannot silently
    # change which original evidence was joined during the postclosed hashes.
    for row in observations.values():
        join_ref(row)
    require(closure == receipt["closure"], "Original closure remains immutable")
    result = {
        "schema": "root-closed-full06-actual04-complete-aggregate-review-v02",
        "status": "PASS_WITH_LIMITS_BOTH_ORIGINAL_SUITES_AND_SELECTED_UNIT_COHORT_CLOSURE",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reviewer": "root closed source/identity/typed aggregate review; no model execution or item-level rescore",
        "model": expected_model,
        "public231": projected[PUBLIC],
        "coherence_mini": projected[COHERENCE],
        "coherence_error_diagnostic": error_diagnostic,
        "latencies": state["latencies"],
        "callbacks": 1479,
        "overall_campaign_complete": True,
        "owner_completed": True,
        "root_exit_code": terminal["exit_code"],
        "root_wall_seconds": terminal["wall_seconds"],
        "root_stderr_diagnostic": stderr_diagnostic,
        "command_seconds": command["elapsed_seconds"],
        "campaign_recorded_seconds": state["elapsed_seconds"],
        "clock_scope": "Original whole10000 starts at campaign execute entry; child command includes startup/import/return/log overhead and is bounded by owner10800; successful original return proves post-cleanup literal gates returned",
        "selected_source_records_authenticated": 59,
        "postclosed_native_runtime_owner_and_benchmark_records_authenticated": len(checked),
        "runtime_scope": "Original SHA when supplied plus selected stat-only identities; not full native/transitive DSO closure",
        "memory_diagnostic": memory_diagnostic,
        "original_failure_records": {
            "owner_primary_error": None,
            "owner_secondary_errors": [],
            "command_primary_error": None,
            "command_cleanup_errors": [],
        },
        "identity": {
            "boot_id": config["boot_id"],
            "unit": UNIT,
            "nonce": config["nonce"],
            "invocation_id": closure["invocation_id"],
            "model_pid_start": [proc["pid"], proc["starttime_ticks"]],
        },
        "artifact_refs": {
            "root_terminal": ref(terminal_path),
            "owner_receipt": ref(receipt_path),
            "sealed_config": ref(config_path),
            "admission": ref(admission_path),
            "bootstrap": ref(bootstrap_path),
            "state": ref(state_path),
            "results": ref(results_path),
            "actor_exit": ref(actor_exit_path),
            "original_closure": ref(closure_path),
            "original_readback": ref(readback_path),
            "command_started": ref(started_path),
            "command_intent": ref(intent_path),
            "command_output": ref(command_output_path),
            "stages": stage_refs,
            "official_public_summary": ref(summary_path),
            "official_public_manifest": ref(manifest_path),
            "audit_source": ref(Path(__file__).resolve()),
        },
        "closure": {
            key: closure[key]
            for key in (
                "closure_mode",
                "post_stop_terminal_verified",
                "cgroup_absent",
                "sources_unchanged",
                "whole_unit_exit_confirmed",
                "observed_cgroup_cohort_exit_confirmed",
                *UNIVERSAL_FALSE,
            )
        },
        "limits": [
            "Both original suites/counts and their complete authenticated stage receipts were joined to this successful current invocation and original scoped closure.",
            "Original Report.errors is preserved by count/hash; an empty-list requirement was not added or used to alter official scores.",
            "Memory pressure/peak values are typed diagnostics; no new zero-pressure or peak-size score threshold was added.",
            "CPU quota accepts exactly the original four-times-positive-period rule; native/stage limits are joined exactly.",
            "Official pinned scorers ran in the original authenticated campaign; this audit does not recompute item-level scores or claim full DSO closure.",
            "CPU4/2GiB D3Edge only. No composite, leaderboard rank, non-MINI sealed/full benchmark, training, browser, speedup, harmless coexistence or competitor win.",
            "Original universal no-escape, whole-tree, continuation and model-qualified flags remain false.",
        ],
        "raw_tasks_gold_answers_errors_responses_features_or_model_bytes_included": False,
    }
    destination = CAMPAIGN / "root-closed-actual04-complete-result-review-v02.json"
    with ordinary(destination).open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require(read_json(destination, {}) == result, "Written root complete audit reread differs")
    print(
        json.dumps(
            {
                "status": result["status"],
                "public231": projected[PUBLIC],
                "coherence_mini": projected[COHERENCE],
                "coherence_error_count": len(errors),
                "review": ref(destination),
            },
            ensure_ascii=False,
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "HELD_SOURCE_ONLY_V02_NOT_EXECUTED_OR_CHECKED",
                    "actual_evidence_read": False,
                    "purpose": "Complete1479 current-closure typed aggregate audit only after root source review and original terminal",
                }
            )
        )
        return 0
    execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
