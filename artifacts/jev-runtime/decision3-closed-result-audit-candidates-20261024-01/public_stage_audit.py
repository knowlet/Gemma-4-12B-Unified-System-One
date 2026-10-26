"""Held source candidate for a CLOSED public231-only audit of full06 actual04.

No candidate code has been run or parsed during authoring. Default CLI does not
read actual evidence. --execute is for root only AFTER its source review and the
original NN/controller are terminal. This is not the full1479 success audit.
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


def validate_latency(row):
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
        and row["count"] == 231,
        "Full231 original typed latency population required",
    )
    require(
        all(
            finite(row[key], maximum=180, positive=True)
            for key in ("median_seconds", "p95_seconds", "max_seconds")
        )
        and row["median_seconds"] <= row["p95_seconds"] <= row["max_seconds"]
        and row["quantile_method"] == "nearest_rank_p95; ordinary_median"
        and row["startup_included"] is False
        and row["browser_20_second_qualification"] is False
        and row["unit"]
        == "one unchanged complete state/questions callback; native question chunks of up to4",
        "Original complete callback latency policy differs",
    )


def aware_epoch(value):
    require(type(value) is str, "Original UTC timestamp string required")
    parsed = datetime.datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, "Explicit timezone on original terminal timestamp required")
    result = parsed.timestamp()
    require(finite(result, positive=True), "Finite original terminal epoch required")
    return result


def execute():
    observations = {}
    terminal_path = OWNER / "root-execution-full06-edge-both-04.json"
    receipt_path = OUTPUT / "receipt.json"
    # No state, score, stage, model or payload read before BOTH terminal anchors.
    require(
        ordinary(terminal_path).is_file() and ordinary(receipt_path).is_file(),
        "Original root terminal and original owner receipt must both exist first",
    )
    terminal = read_json(terminal_path, observations)
    receipt = read_json(receipt_path, observations)
    require(
        type(terminal.get("exit_code")) is int
        and terminal["exit_code"] != 0
        and finite(terminal.get("wall_seconds"), maximum=10800, positive=True),
        "This fallback requires a closed nonzero full-campaign terminal",
    )
    root_started = aware_epoch(terminal["started_at_utc"])
    root_finished = aware_epoch(terminal["finished_at_utc"])
    require(root_started < root_finished, "Original root terminal clocks required")
    require(
        receipt.get("completed") is False
        and receipt.get("primary_error") is not None
        and type(receipt.get("secondary_errors")) is list
        and receipt.get("sources_unchanged") is True
        and receipt.get("comfy_service_modified") is False
        and receipt.get("whole_tree_exit_confirmed") is False
        and receipt.get("profile") == OWNER_PROFILE,
        "Original failure and source/nonqualification flags must be preserved",
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
        "Exact current selected invocation/config differs",
    )
    require(
        read_json(OUTPUT / "sealed-config.json", observations) == config,
        "Original sealed config changed",
    )
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
        spec.get("runtime_seconds") == 10800 and len(spec.get("commands", [])) == 1,
        "Original command/budget selection differs",
    )
    selected_sources(config, spec)
    closure_path = OUTPUT / "external-boundary-closure.json"
    closure = read_json(closure_path, observations)
    readback = read_json(OUTPUT / "unit-readback.json", observations)
    actor = check_original_closure(config, receipt, closure, readback)
    require(receipt.get("unit_attempted") is True, "Original unit attempt/admission required")
    validate_host(receipt["state"]["initial_host"], config["boot_id"], initial=True)
    validate_host(receipt["final_host"], config["boot_id"], initial=False)
    admission = read_json(OUTPUT / "admission.json", observations)
    require(
        admission.get("nonce") == config["nonce"]
        and admission.get("invocation_id") == closure["invocation_id"]
        and admission.get("actor") == actor
        and admission.get("config_sha256") == digest(OUTPUT / "sealed-config.json"),
        "Original admission identity/config join differs",
    )

    # Actual stage/state reads start only after original terminal and scoped closure passed.
    state_path = ACTUAL / "state.json"
    state = read_json(state_path, observations)
    require(
        state.get("profile") == PROFILE
        and state.get("model_key") == "edge"
        and state.get("plan_sha256") == PLAN_SHA
        and state.get("model_loads") == 1
        and state.get("benchmark_scores") is None
        and state.get("status")
        in {
            "failed_or_incomplete_no_complete_score_claim",
            "running_original_public231_then_coherence_mini",
        }
        and state.get("whole_campaign_deadline_seconds") == 10000
        and state.get("per_call_deadline_seconds") == 180
        and set(state.get("completed_stage_receipts", {})) == {PUBLIC},
        "Only an incomplete campaign with exactly one completed public stage is admissible",
    )
    validate_limits(state.get("owned_limits"))
    plan = read_json(CAMPAIGN / "edge-plan.json", observations, maximum=1048576, expected=PLAN_SHA)
    native = read_json(
        NATIVE / "edge-plan.json", observations, maximum=1048576, expected=NATIVE_PLAN_SHA
    )
    benchmark_path = RUNTIME / "decision3-publisher-lite-cpu-20261023-01/benchmark-source-plan.json"
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
        "Exact original model/runtime/scorer inventory differs",
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
    require(state.get("model_identity") == expected_model, "Actual stage model identity differs")
    proc = state["process"]
    require(
        proc.get("affinity") == [16, 17, 18, 19]
        and proc.get("cgroup") == "0::" + actor["cgroup"]
        and proc.get("pgid") == actor["pgid"]
        and proc.get("sid") == actor["sid"],
        "Actual model process/owner group identity differs",
    )
    start_environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "ACCELERATE_USE_CPU": "true",
    }
    require(
        all(proc.get("environment", {}).get(k) == v for k, v in start_environment.items()),
        "Original pre-loader offline CPU environment differs",
    )
    started_path = OUTPUT / "command-01-started.json"
    intent_path = OUTPUT / "command-01-intent.json"
    started = read_json(started_path, observations)
    intent = read_json(intent_path, observations)
    command_process = started["observed_identity"]["process"]
    require(
        started.get("pid") == proc.get("pid") == command_process.get("pid")
        and command_process.get("start_ticks") == proc.get("starttime_ticks")
        and all(command_process.get(k) == actor[k] for k in ("pgid", "sid", "uid", "cgroup")),
        "Observed original command PID/start/unit join differs",
    )
    expected_argv = [part.replace("{OUTPUT}", str(OUTPUT)) for part in spec["commands"][0]["argv"]]
    require(
        intent.get("argv") == started["observed_identity"].get("argv") == expected_argv
        and intent.get("cwd") == started["observed_identity"].get("cwd") == str(ROOT)
        and intent.get("resolved_executable")
        == started["observed_identity"].get("exe")
        == proc.get("executable")
        and intent.get("shell") is False,
        "Observed exact command/exec/cwd differs",
    )

    stage_ref = state["completed_stage_receipts"][PUBLIC]
    require(stage_ref.get("stage") == PUBLIC, "Selected public stage reference label differs")
    completion_path = join_ref(stage_ref)
    require(
        completion_path == ACTUAL / "stage-completions" / PUBLIC / "completion.json",
        "Exact current public completion path required",
    )
    completion = read_json(completion_path, observations)
    require(
        completion.get("schema") == SCHEMA
        and completion.get("stage") == PUBLIC
        and completion.get("profile") == PROFILE
        and completion.get("status")
        == "authenticated_stage_complete_pending_independent_current_root_audit"
        and completion.get("owner_nonce") == config["nonce"]
        and completion.get("overall_campaign_complete") is False
        and completion.get("external_current_command_and_owner_closure_required") is True
        and completion.get("cooperative_publication_not_hard_realtime_or_no_escape") is True
        and completion.get("whole_campaign_seconds") == 10000,
        "Current public completion is missing/demoted/unknown",
    )
    evidence_path = join_ref(completion["evidence"])
    require(
        evidence_path == completion_path.parent / "evidence.json",
        "Exact current public evidence path required",
    )
    evidence = read_json(evidence_path, observations)
    require(
        evidence.get("schema") == SCHEMA
        and evidence.get("stage") == PUBLIC
        and evidence.get("profile") == PROFILE
        and evidence.get("model_key") == "edge"
        and evidence.get("status") == "evidence_only_not_a_stage_completion"
        and evidence.get("counts")
        == {
            "stage_callbacks": 231,
            "cumulative_callbacks": 231,
            "started": 231,
            "completed": 231,
            "raw_native_returns": 231,
        }
        and set(evidence.get("full_authentication", {})) == AUTH_KEYS
        and all(value is True for value in evidence["full_authentication"].values())
        and evidence.get("raw_tasks_gold_responses_or_per_item_scores_included") is False,
        "Full public stage authentication/counts absent",
    )
    stage_owner = evidence["owner"]
    require(
        set(stage_owner) == {"unit", "profile", "nonce", "process", "limits"}
        and stage_owner.get("unit") == UNIT
        and stage_owner.get("profile") == OWNER_PROFILE
        and stage_owner.get("nonce") == config["nonce"]
        and stage_owner.get("limits") == state["owned_limits"]
        and all(
            stage_owner["process"].get(k) == proc.get(k)
            for k in ("pid", "starttime_ticks", "pgid", "sid", "cgroup", "affinity", "executable")
        )
        and stage_owner["process"].get("environment") == native["environment"],
        "Public stage stable identity or selected post-loader CPU environment differs",
    )
    invocation = evidence["invocation"]
    require(
        invocation.get("output") == str(ACTUAL)
        and invocation.get("plan_sha256") == PLAN_SHA
        and invocation.get("campaign_started_unix") == state.get("campaign_started_unix")
        and finite(
            invocation.get("campaign_started_unix"), minimum=root_started, maximum=root_finished
        ),
        "Public stage exact invocation differs",
    )
    began = invocation["campaign_started_monotonic"]
    clock = evidence["clock"]
    cleanup = completion["after_stage_deadline_cleanup_monotonic"]
    publication_elapsed = stage_ref["elapsed_campaign_seconds_after_publication"]
    require(
        finite(began, positive=True)
        and finite(clock["stage_started_monotonic"], minimum=began)
        and finite(started.get("started_monotonic"), positive=True)
        and started["started_monotonic"] <= began
        and clock["stage_started_monotonic"] <= clock["authenticated_monotonic"] <= cleanup
        and finite(clock["elapsed_campaign_seconds"], maximum=10000, positive=True)
        and clock["elapsed_campaign_seconds"] <= clock["authenticated_monotonic"] - began
        and finite(publication_elapsed, maximum=10000, positive=True)
        and publication_elapsed <= terminal["wall_seconds"]
        and cleanup - began <= publication_elapsed
        and clock.get("whole_campaign_seconds") == 10000
        and clock.get("per_call_seconds") == 180
        and clock.get("owner_seconds") == 10800,
        "Original whole-clock/authentication/publication scopes differ",
    )
    validate_latency(evidence["latency"])
    if PUBLIC in state.get("latencies", {}):
        require(
            state["latencies"][PUBLIC] == evidence["latency"],
            "Final retained public latency changed",
        )

    # A validated later callback is a deliberately conservative sufficient
    # witness of complete() returning past its final handoff. This actual04
    # audit refuses the wider entry-only/first-coherence-timeout case.
    continuation_path = ACTUAL / "native-evidence/00231-validated.json"
    continuation = read_json(continuation_path, observations, maximum=1048576)
    require(
        continuation.get("index") == 231
        and continuation.get("stage") == COHERENCE
        and continuation.get("status") == "validated_pending_durable_evidence",
        "Positive same-invocation later-suite continuation required",
    )
    before = continuation["before_owned_counters"]
    require(
        before.get("cgroup") == proc["cgroup"]
        and finite(
            before.get("observed_monotonic"), minimum=began + publication_elapsed, positive=True
        )
        and before["observed_monotonic"] - began <= terminal["wall_seconds"]
        and finite(continuation.get("validation_seconds"), maximum=180),
        "Later callback identity/time does not prove public final handoff returned",
    )
    counters = before["counters"]
    require(
        type(counters.get("memory.peak")) is int
        and counters["memory.peak"] >= 0
        and type(counters.get("memory.current")) is int
        and counters["memory.current"] >= 0
        and type(counters.get("memory.events")) is dict
        and {"max", "oom", "oom_kill"} <= set(counters["memory.events"])
        and all(type(value) is int and value >= 0 for value in counters["memory.events"].values()),
        "Typed original public-through-continuation diagnostics required",
    )
    continuation_memory = {
        "observed_monotonic": before["observed_monotonic"],
        "memory.current": counters["memory.current"],
        "memory.peak": counters["memory.peak"],
        "memory.events": counters["memory.events"],
        "configured_memory_max_bytes": CAP,
        "observed_peak_within_configured_limit": counters["memory.peak"] <= CAP,
        "pressure_or_oom_event_observed": any(
            value > 0
            for key, value in counters["memory.events"].items()
            if key in {"max", "oom", "oom_kill", "oom_group_kill"}
        ),
        "role": "diagnostic only; no new zero-pressure public-score acceptance gate",
    }
    coherence_completion_path = ACTUAL / "stage-completions" / COHERENCE / "completion.json"
    if coherence_completion_path.exists():
        later_receipt = read_json(coherence_completion_path, observations, maximum=1048576)
        require(
            later_receipt.get("status") == "failed_no_stage_qualification",
            "This audit does not handle two completed stages or unknown later receipts",
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
        and pins.get("dataset_metadata") == benchmark["existing_dataset_metadata"],
        "Public stage exact model/runtime/scorer pins differ",
    )
    argv = spec["commands"][0]["argv"]

    def argument(name):
        return argv[argv.index(name) + 1]

    smoke_path, audit_path = Path(argument("--smoke-receipt")), Path(argument("--smoke-audit"))
    smoke_sha, audit_sha = argument("--smoke-receipt-sha256"), argument("--smoke-audit-sha256")
    expected_sources = [
        (CAMPAIGN / "edge-plan.json", PLAN_SHA),
        (CAMPAIGN / "campaign.py", plan["campaign_sha256"]),
        (NATIVE / "actor.py", plan["batch_actor_sha256"]),
        (NATIVE / "freeze.json", plan["batch_freeze_sha256"]),
        (NATIVE / "edge-plan.json", NATIVE_PLAN_SHA),
        (benchmark_path, BENCHMARK_SHA),
        (NATIVE / "resource_helper.py", native["helper_sha256"]),
        (NATIVE / "source_audit.py", native["source_audit_sha256"]),
        (smoke_path, smoke_sha),
        (audit_path, audit_sha),
    ]
    require(
        pins.get("campaign_sources_and_prerequisites")
        == [{"path": str(path), "sha256": sha} for path, sha in expected_sources],
        "Exact selected stage prerequisite source union differs",
    )
    for path, sha in expected_sources:
        require(digest(path) == sha, "Postclosed selected stage source/prerequisite changed")
    audit = read_json(audit_path, observations, maximum=1048576, expected=audit_sha)
    require(
        audit.get("schema") == "independent-native-batch4-synthetic-result-audit-v1"
        and audit.get("review_kind") == "closed_real_smoke_result"
        and audit.get("result") in {"PASS", "PASS_WITH_LIMITS"}
        and audit.get("model_key") == "edge"
        and audit.get("owner_unit") == UNIT
        and audit.get("memory_max_bytes") == CAP
        and audit.get("host_observation_policy") == POLICY
        and audit.get("model_revision") == native["model"]["revision"]
        and audit.get("model_identity_sha256") == native["model"]["model_sha256"]
        and audit.get("manifest_sha256") == native["model"]["manifest_sha256"]
        and all(audit.get("qualification", {}).get(key) is True for key in SMOKE_QUALIFICATIONS),
        "Original accepted closed smoke prerequisite differs",
    )
    audit_refs = audit["artifact_refs"]
    require(
        {"state", "plan", "owner_receipt", "root_execution", "spec", "closure_observation"}
        <= set(audit_refs)
        and path_from_ref(audit_refs["state"]) == smoke_path
        and audit_refs["state"]["sha256"] == smoke_sha
        and path_from_ref(audit_refs["plan"]) == NATIVE / "edge-plan.json"
        and audit_refs["plan"]["sha256"] == NATIVE_PLAN_SHA,
        "Exact original smoke audit refs differ",
    )
    for row in audit_refs.values():
        require(path_from_ref(row).stat().st_size <= 1048576, "Original small audit ref required")
        join_ref(row)

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
        # Stat-only selected runtime identities keep their original limited scope.
        join_ref(record, require_sha=bool(record.get("sha256")))
        checked[str(path)] = {
            "sha256": record.get("sha256"),
            "stat_only": record.get("sha256") is None,
        }

    summary_path = ACTUAL / "public231/summary.json"
    manifest_path = ACTUAL / "public231/manifest.json"
    summary = read_json(summary_path, observations)
    manifest = read_json(manifest_path, observations)
    projected = public_metrics(summary)
    require(
        evidence.get("metrics") == projected,
        "Official typed public aggregate differs from authenticated stage",
    )
    require(
        summary.get("identity") == expected_model
        and summary.get("source_revision") == benchmark["public"]["revision"]
        and summary.get("official_rank") is None
        and summary.get("official_composite") is None
        and summary.get("sealed_evaluation") == "not_run"
        and manifest.get("n_attempted") == 231
        and manifest.get("n_valid") == projected["n_valid"]
        and manifest.get("status") in {"completed", "completed_with_errors"}
        and manifest.get("identity") == expected_model
        and manifest.get("artifact_sha256", {}).get("summary.json") == digest(summary_path),
        "Original official public summary/manifest identity differs",
    )
    peak_raw = receipt["state"].get("preclose_exit_memory_raw", {}).get("MemoryPeak")
    peak = int(peak_raw) if type(peak_raw) is str and peak_raw.isdigit() else None
    # A later failure's full-run peak/events are not a public fit qualification.
    # Preserve their original evidence through immutable owner refs, without
    # rewriting or demanding overall success/final campaign authentication.
    for path, row in observations.items():
        join_ref(row)
    errors = {
        "primary_error": receipt["primary_error"],
        "secondary_errors": receipt["secondary_errors"],
    }
    result = {
        "schema": "root-closed-full06-actual04-public-stage-only-review-v1",
        "status": "PASS_WITH_LIMITS_PUBLIC231_ONLY_OVERALL_CAMPAIGN_FAILED_OR_INCOMPLETE",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reviewer": "root closed aggregate/source/identity review; no item-level rescore or model execution",
        "scope": "Exactly one authenticated complete public231 stage; later coherence continuation proved, coherence completion not qualified",
        "model": expected_model,
        "public231": projected,
        "public_latency": evidence["latency"],
        "coherence_mini": None,
        "coherence_complete_qualified": False,
        "overall_campaign_complete": False,
        "owner_completed": receipt["completed"],
        "root_exit_code": terminal["exit_code"],
        "root_wall_seconds": terminal["wall_seconds"],
        "campaign_state_status": state["status"],
        "full_campaign_final_authentication_claimed": False,
        "failure_preservation": {
            "original_owner_primary_and_secondary_errors_unchanged": True,
            "primary_error_present": True,
            "secondary_error_count": len(receipt["secondary_errors"]),
            "exact_error_object_canonical_sha256": canonical_digest(errors),
            "original_errors_location": "artifact_refs.owner_receipt; errors remain there exactly, never replaced or cleared",
            "raw_error_text_or_command_stdout_stderr_copied": False,
        },
        "selected_source_records_authenticated": 59,
        "postclosed_native_runtime_owner_and_benchmark_records_authenticated": len(checked),
        "runtime_scope": "Original selected SHA when present plus original stat-only identities; no full native DSO closure",
        "public_complete_callbacks": 231,
        "whole_run_peak_bytes_observed": peak,
        "public_through_continuation_memory_snapshot": continuation_memory,
        "whole_run_memory_fit_qualified": False,
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
            "sealed_config": ref(OUTPUT / "sealed-config.json"),
            "state": ref(state_path),
            "original_closure": ref(closure_path),
            "original_readback": ref(OUTPUT / "unit-readback.json"),
            "command_started": ref(started_path),
            "command_intent": ref(intent_path),
            "public_completion": ref(completion_path),
            "public_evidence": ref(evidence_path),
            "later_suite_continuation": ref(continuation_path),
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
            "Original failed root/owner/campaign records are preserved; this does not convert them into a full success.",
            "A positive same-invocation later callback proves public complete() returned after handoff; a marker alone is refused.",
            "Original current closure must itself be positive. Later absence observations never repair false original flags.",
            "All231 public tasks remain in the original official aggregate; no tasks, options, budgets or scores are repaired.",
            "No qualified coherence/full1479 score, rank, sealed result, browser, training, speedup, fit or competitor win.",
            "CPU4/2GiB D3Edge profile only; universal no-escape/whole-tree/continuation/model flags remain false.",
        ],
        "raw_tasks_gold_answers_responses_features_or_model_bytes_included": False,
    }
    destination = CAMPAIGN / "root-public-stage-only-actual04-result-review-01.json"
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
    require(read_json(destination, {}) == result, "Written root audit reread differs")
    print(
        json.dumps(
            {
                "status": result["status"],
                "public231": projected,
                "coherence_mini": None,
                "root_exit_code_preserved": terminal["exit_code"],
                "owner_completed_preserved": receipt["completed"],
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
                    "status": "HELD_SOURCE_ONLY_NOT_EXECUTED_OR_CHECKED",
                    "actual_evidence_read": False,
                    "purpose": "Closed public231-only fallback after exact original terminal/receipt/closure and root source review",
                }
            )
        )
        return 0
    execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
