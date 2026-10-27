"""Held, I/O-free v2 CPU profile audit of caller-supplied synthetic bytes.

Caller must independently authenticate sources, collection, process identity and
closure. This module cannot certify them, dynamic dependencies or exclusivity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics

SOURCE_ONLY = True
MODES = ("native", "whole", "ordered_pairs", "length_pairs")
KEYS = tuple(f"synthetic_{index}" for index in range(4))
OPTIONS = ("alpha", "beta", "gamma")
TARGETS = (3117, 4196, 4579, 3126)
MANIFEST = "3bae8989a6feb36a66bf4d8c44112dc63199d0dc69f564dc895fe368b1e08e59"


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _decode(raw, limit):
    _need(type(raw) is bytes and 0 < len(raw) <= limit, "bounded bytes required")

    def pairs(items):
        output = {}
        for key, value in items:
            _need(key not in output, "duplicate JSON key")
            output[key] = value
        return output

    def bad_constant(_value):
        raise ValueError("non-finite JSON constant")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)
    _need(type(value) is dict, "JSON object required")
    return value


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _probabilities(answer):
    _need(
        type(answer) is dict
        and set(answer) == {"type", "probabilities", "confidence", "choice"}
        and answer["type"] == "choice"
        and type(answer["probabilities"]) is dict
        and tuple(answer["probabilities"]) == OPTIONS,
        "complete ordered answer required",
    )
    values = list(answer["probabilities"].values())
    _need(
        all(_number(value) and 0 <= value <= 1 for value in values)
        and math.isclose(sum(values), 1, abs_tol=1e-6, rel_tol=1e-6)
        and answer["choice"] == OPTIONS[values.index(max(values))],
        "invalid choice probabilities",
    )
    confidence = max(
        0.0,
        min(1.0, 1 + sum(p * math.log(p) for p in values if p > 0) / math.log(len(values))),
    )
    _need(
        _number(answer["confidence"])
        and math.isclose(answer["confidence"], confidence, abs_tol=1e-12, rel_tol=1e-12),
        "invalid confidence",
    )
    return values


def _metrics(mode, lengths):
    indexes = list(range(4))
    if mode == "length_pairs":
        indexes.sort(key=lambda index: (-lengths[index], index))
    width = 4 if mode == "whole" else 2
    groups = [indexes[start : start + width] for start in range(0, 4, width)]
    return {
        "native_run_calls": 1 if mode == "whole" else 2,
        "forward_groups": len(groups),
        "unpadded_tokens": sum(lengths),
        "padded_slots": sum(
            len(group) * max(lengths[index] for index in group) for group in groups
        ),
    }


def _audit_legacy(
    *, closure_bytes, terminal_bytes, result_bytes, hello_bytes, plan_sha, invocation, model
):
    """Audit supplied bytes only; return no performance data for failed closure."""
    _need(SOURCE_ONLY is False, "reviewed audit release required")
    _need(type(plan_sha) is str and re.fullmatch(r"[0-9a-f]{64}", plan_sha), "plan SHA required")
    _need(
        invocation is None
        or (type(invocation) is str and re.fullmatch(r"[0-9a-f]{32}", invocation)),
        "valid invocation or None required",
    )
    closure = _decode(closure_bytes, 65536)
    _need(
        closure.get("schema") == "decision3-cpu-watcher-closure-v1"
        and closure.get("plan_sha256") == plan_sha
        and "invocation" in closure
        and closure["invocation"] == invocation
        and closure.get("status")
        in {"failed", "cleanup_unconfirmed", "completed_pending_external_audit"}
        and closure.get("machine_exclusive") is False,
        "matching closure required",
    )
    scope = (
        "caller-supplied bytes only; source authentication, plan bytes, synthetic input content, "
        "hello unit and cgroup, dynamic dependencies and machine exclusivity unverified; "
        "caller must independently verify them; synthetic timing, not benchmark score"
    )
    if closure["status"] != "completed_pending_external_audit":
        return {"status": "incomplete", "closure_status": closure["status"], "scope": scope}
    _need(type(invocation) is str, "successful invocation required")
    _need(type(model) is str and model.strip() and len(model) <= 4096, "model name required")
    _need(
        closure.get("cgroup") in {"absent", "confirmed_unpopulated"}
        and closure.get("process_group") == "absent"
        and closure.get("unit") == "retained-unit"
        and closure.get("diagnostic")
        == {"last_entered_stage": "result_write", "failure_category": None},
        "successful verified closure required",
    )
    hello = _decode(hello_bytes, 4096)
    _need(
        hello.get("schema") == "decision3-cpu-hello-v1"
        and hello.get("plan_sha256") == plan_sha
        and hello.get("invocation") == invocation
        and type(hello.get("pid")) is int
        and hello["pid"] > 0
        and hello.get("pgid") == hello["pid"]
        and type(hello.get("startticks")) is int
        and hello["startticks"] > 0
        and type(hello.get("boot_id")) is str
        and bool(hello["boot_id"])
        and type(hello.get("unit")) is str
        and type(hello.get("cgroup")) is str,
        "matching worker hello required",
    )
    terminal = _decode(terminal_bytes, 4096)
    _need(
        set(terminal) == {"schema", "status", "invocation", "plan_sha256", "result_sha256"}
        and terminal["schema"] == "decision3-cpu-terminal-v1"
        and terminal["status"] == "completed_pending_external_audit"
        and terminal["invocation"] == invocation
        and terminal["plan_sha256"] == plan_sha
        and type(result_bytes) is bytes
        and hashlib.sha256(result_bytes).hexdigest() == terminal["result_sha256"],
        "matching terminal and result digest required",
    )
    result = _decode(result_bytes, 4 * 1024 * 1024)
    _need(
        result.get("schema") == "decision3-cpu-padding-experiment-result-v1"
        and result.get("publisher_manifest_sha256") == MANIFEST,
        "result schema and claimed manifest required",
    )
    questions = result.get("questions")
    length_map = result.get("head_token_lengths")
    _need(
        type(questions) is dict
        and tuple(questions) == KEYS
        and type(length_map) is dict
        and tuple(length_map) == KEYS
        and all(
            type(length_map[k]) is int and abs(length_map[k] - target) <= 32
            for k, target in zip(KEYS, TARGETS, strict=True)
        ),
        "original ordered head lengths required",
    )
    lengths = [length_map[key] for key in KEYS]
    total = sum(lengths)
    _need(
        type(result.get("input_tokens")) is int and result["input_tokens"] == total,
        "token denominator differs",
    )
    samples = result.get("samples")
    _need(
        type(samples) is list and len(samples) == 20, "one warmup and four measured rounds required"
    )
    measured = {mode: [] for mode in MODES}
    cycles = []
    for index, sample in enumerate(samples):
        cycle = index // 4 - 1
        mode = MODES[index % 4] if cycle == -1 else MODES[(index % 4 + cycle) % 4]
        _need(
            type(sample) is dict
            and sample.get("cycle") == cycle
            and type(sample.get("cycle")) is int
            and sample.get("phase") == ("warmup" if cycle == -1 else "measured")
            and sample.get("mode") == mode
            and type(sample.get("elapsed_ns")) is int
            and sample["elapsed_ns"] > 0
            and type(sample.get("peak_rss_kib")) is int
            and sample["peak_rss_kib"] > 0,
            "complete ordered timings required",
        )
        response = sample.get("response")
        _need(
            type(response) is dict
            and response.get("model") == model
            and type(response.get("model")) is str
            and type(response.get("answers")) is dict
            and tuple(response["answers"]) == KEYS
            and response.get("usage") == {"input_tokens": total, "output_tokens": 0}
            and type(response["usage"]["input_tokens"]) is int
            and type(response["usage"]["output_tokens"]) is int,
            "complete response required",
        )
        for key in KEYS:
            values = _probabilities(response["answers"][key])
            if mode != "native":
                raw = sample.get("native_probabilities")
                _need(
                    type(raw) is dict
                    and tuple(raw) == KEYS
                    and type(raw[key]) is list
                    and len(raw[key]) == 3
                    and all(_number(x) and 0 <= x <= 1 for x in raw[key])
                    and math.isclose(sum(raw[key]), 1, abs_tol=1e-5, rel_tol=0)
                    and all(
                        math.isclose(a, b / sum(raw[key]), abs_tol=1e-12, rel_tol=1e-12)
                        for a, b in zip(values, raw[key], strict=True)
                    ),
                    "raw probabilities differ",
                )
        if mode == "native":
            _need(
                sample.get("native_probabilities") is None and sample.get("metrics") is None,
                "native raw data must be unavailable",
            )
        else:
            metrics = sample.get("metrics")
            _need(
                type(metrics) is dict
                and all(type(v) is int for v in metrics.values())
                and metrics == _metrics(mode, lengths),
                "padding metrics differ",
            )
        if cycle >= 0:
            measured[mode].append(sample)
    for cycle in range(4):
        native = measured["native"][cycle]
        for mode in MODES[1:]:
            sample = measured[mode][cycle]
            actual, reference = sample["response"], native["response"]
            cycles.append(
                {
                    "cycle": cycle,
                    "mode": mode,
                    "max_probability_abs_diff": max(
                        abs(
                            actual["answers"][key]["probabilities"][option]
                            - reference["answers"][key]["probabilities"][option]
                        )
                        for key in KEYS
                        for option in OPTIONS
                    ),
                    "max_native_probability_abs_diff": None,
                    "max_confidence_abs_diff": max(
                        abs(
                            actual["answers"][key]["confidence"]
                            - reference["answers"][key]["confidence"]
                        )
                        for key in KEYS
                    ),
                    "same_choices": all(
                        actual["answers"][key]["choice"] == reference["answers"][key]["choice"]
                        for key in KEYS
                    ),
                    "same_usage": actual["usage"] == reference["usage"],
                    "latency_ratio_to_native": sample["elapsed_ns"] / native["elapsed_ns"],
                }
            )
    recorded = result.get("comparisons")
    _need(type(recorded) is list and len(recorded) == 12, "complete comparisons required")
    for expected, actual in zip(cycles, recorded, strict=True):
        _need(
            type(actual) is dict
            and actual == {k: v for k, v in expected.items() if k != "latency_ratio_to_native"},
            "recorded comparison differs",
        )
    timing = {}
    for mode in MODES:
        elapsed = [row["elapsed_ns"] for row in measured[mode]]
        timing[mode] = {
            "samples_ns": elapsed,
            "p50_ns": statistics.median(elapsed),
            "p95_nearest_rank_ns": max(elapsed),
        }
    return {
        "status": "complete",
        "scope": scope,
        "head_token_lengths": length_map,
        "input_tokens": total,
        "padding": {mode: _metrics(mode, lengths) for mode in MODES[1:]},
        "cycles": cycles,
        "timing": timing,
        "limitations": "Four measured rounds only; p95 is the maximum. Native raw probability rows are unavailable. Candidate timings include snapshot checks.",
    }


def _plan_evidence(raw, plan_sha, expected_pins, schema, profile=None):
    _need(type(plan_sha) is str and re.fullmatch(r"[0-9a-f]{64}", plan_sha), "plan SHA required")
    _need(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == plan_sha, "plan bytes differ")
    plan = _decode(raw, 1024 * 1024)
    fields = {
        "schema",
        "unit",
        "output_dir",
        "python",
        "source_pins",
        "wall_seconds",
        "memory_floor_bytes",
        "rounds",
    }
    if profile is not None:
        fields.add("cpu_profile")
    _need(
        set(plan) == fields
        and plan["schema"] == schema
        and plan["rounds"] == 4
        and type(plan["rounds"]) is int
        and type(plan["unit"]) is str
        and re.fullmatch(r"jev-cpu-[0-9a-f]{32}\.service", plan["unit"])
        and type(plan["wall_seconds"]) is int
        and 30 <= plan["wall_seconds"] <= 3600
        and type(plan["memory_floor_bytes"]) is int
        and plan["memory_floor_bytes"] >= 34709107508
        and type(plan["output_dir"]) is str
        and plan["output_dir"].endswith(
            "/run-" + plan["unit"].removeprefix("jev-cpu-").removesuffix(".service") + "/worker"
        )
        and ".." not in plan["output_dir"].split("/")
        and plan["python"] == "/home/albert_hsieh/ComfyUI/.venv/bin/python"
        and type(plan["source_pins"]) is dict
        and type(expected_pins) is dict
        and plan["source_pins"] == expected_pins
        and len(expected_pins) == 19
        and all(
            type(path) is str
            and path.startswith("/")
            and type(digest) is str
            and re.fullmatch(r"[0-9a-f]{64}", digest)
            for path, digest in expected_pins.items()
        ),
        "matching reviewed plan and source pins required",
    )
    if profile is not None:
        _need(plan["cpu_profile"] == profile, "cpu10 profile required")
    suffix = "_v2_release.py" if profile is not None else "_release.py"
    for name in (
        "jev_decision3_cpu_experiment_worker",
        "jev_decision3_cpu_experiment_watcher",
        "jev_decision3_cpu_padding_candidate",
    ):
        _need(
            sum(path.endswith("/" + name + suffix) for path in expected_pins) == 1,
            "profile-specific release pins required",
        )
        other_suffix = "_release.py" if profile is not None else "_v2_release.py"
        _need(
            not any(path.endswith("/" + name + other_suffix) for path in expected_pins),
            "other profile release pins forbidden",
        )
    return plan


def _events(raw, plan, hello, invocation):
    _need(type(raw) is bytes and 0 < len(raw) <= 4 * 1024 * 1024, "bounded watcher events required")
    lines = raw.splitlines()
    _need(
        0 < len(lines) <= 5000 and all(0 < len(line) <= 65536 for line in lines),
        "bounded watcher event count required",
    )
    events = [_decode(line, 65536) for line in lines]
    pre = [event for event in events if event.get("event") == "prelaunch"]
    hosts = [event for event in events if event.get("event") == "host"]
    bindings = [event for event in events if event.get("event") == "binding_verified"]
    observations = [event for event in events if event.get("event") == "cpu_observation"]
    _need(
        len(pre) == 1 and hosts and len(bindings) == 1 and observations,
        "complete watcher observations required",
    )
    _need(
        events[0] is pre[0]
        and events.index(pre[0])
        < events.index(hosts[0])
        < events.index(bindings[0])
        < events.index(observations[0]),
        "watcher event order invalid",
    )
    times = [event.get("monotonic") for event in [pre[0], *hosts]]
    _need(
        all(_number(value) for value in times) and all(a <= b for a, b in zip(times, times[1:])),
        "watcher host times invalid",
    )
    for event in [*pre, *hosts]:
        _need(
            type(event.get("running")) is int
            and event["running"] == 0
            and type(event.get("pending")) is int
            and event["pending"] == 0
            and type(event.get("available_bytes")) is int
            and event["available_bytes"] >= plan["memory_floor_bytes"]
            and _number(event.get("monotonic")),
            "watcher host guard failed",
        )
    binding = bindings[0]
    _need(
        binding.get("invocation") == invocation
        and binding.get("pid") == hello["pid"]
        and binding.get("pgid") == hello["pgid"]
        and binding.get("cgroup") == hello["cgroup"]
        and binding.get("cpu_profile") == "cpu10"
        and binding.get("cpu_list") == "5-9,15-19",
        "cpu10 watcher binding differs",
    )
    group = hello.get("cgroup")
    _need(
        type(group) is str
        and group.startswith("/")
        and all(part not in ("", ".", "..") for part in group[1:].split("/")),
        "canonical cgroup required",
    )
    expected_levels = len(group[1:].split("/")) + 1
    _need(expected_levels <= 64, "bounded ancestor depth required")
    report = []
    for item in observations:
        status = item.get("status")
        _need(status in {"observed", "partial", "unverified"}, "telemetry status invalid")
        if status == "unverified":
            _need(set(item) == {"event", "status"}, "invalid unverified telemetry")
            report.append({"status": "unverified", "levels": []})
            continue
        levels = item.get("levels")
        _need(
            type(levels) is list and len(levels) == expected_levels,
            "complete ancestor levels required",
        )
        parsed = []
        for number, level in enumerate(levels):
            _need(
                type(level) is dict
                and set(level)
                == {"ancestor_level", "cpu_max", "cpu_stat", "quota_status", "throttle_status"}
                and type(level["ancestor_level"]) is int
                and level["ancestor_level"] == number,
                "ordered ancestor levels required",
            )
            quota_status, throttle_status = level["quota_status"], level["throttle_status"]
            _need(
                quota_status in {"observed", "root_missing", "unverified"}
                and throttle_status in {"observed", "unverified"}
                and (quota_status != "root_missing" or number == len(levels) - 1),
                "ancestor observation status invalid",
            )
            quota = level["cpu_max"]
            cap = None
            if quota_status == "observed":
                _need(
                    type(quota) is str and re.fullmatch(r"(?:max|[0-9]+) [0-9]+", quota),
                    "quota telemetry invalid",
                )
                amount, period = quota.split()
                _need(int(period) > 0, "quota period invalid")
                if amount != "max":
                    cap = int(amount) / int(period)
            else:
                _need(quota is None, "unverified quota must be absent")
            stats = level["cpu_stat"]
            if throttle_status == "observed":
                _need(
                    type(stats) is dict
                    and set(stats) == {"nr_periods", "nr_throttled", "throttled_usec"}
                    and all(type(value) is int and value >= 0 for value in stats.values()),
                    "throttle telemetry invalid",
                )
            else:
                _need(stats is None, "unverified throttle values must be absent")
            parsed.append(
                {
                    "ancestor_level": number,
                    "quota_status": quota_status,
                    "cpu_quota_cores": cap,
                    "throttle_status": throttle_status,
                    "cpu_stat": stats,
                }
            )
        _need(
            parsed[0]["quota_status"] != "observed" or parsed[0]["cpu_quota_cores"] == 10.0,
            "leaf quota differs from cpu10 profile",
        )
        _need(
            status
            == (
                "observed"
                if all(
                    level["quota_status"] == level["throttle_status"] == "observed"
                    for level in levels
                )
                else "partial"
            ),
            "telemetry completeness mismatch",
        )
        report.append({"status": status, "levels": parsed})
    # Periodic cgroup snapshots need not span the exact timing interval.
    finite_caps = [
        level["cpu_quota_cores"]
        for observation in report
        for level in observation["levels"]
        if level["cpu_quota_cores"] is not None
    ]
    return {
        "observations": report,
        "smallest_observed_quota_cores": min(finite_caps) if finite_caps else None,
        "ancestor_completeness": "complete"
        if all(item["status"] == "observed" for item in report)
        else "incomplete_or_unverified",
        "interpretation": "snapshots are observational; quota availability, exclusive CPU and throttle deltas during individual requests are not established",
    }


def audit_v2(
    *,
    plan_bytes,
    approved_source_pins,
    events_bytes,
    closure_bytes,
    terminal_bytes,
    result_bytes,
    hello_bytes,
    plan_sha,
    invocation,
    model,
    baseline=None,
):
    """Audit independently supplied v2 evidence; optional baseline is also caller supplied."""
    _need(SOURCE_ONLY is False, "reviewed audit release required")
    plan = _plan_evidence(
        plan_bytes,
        plan_sha,
        approved_source_pins,
        "decision3-cpu-padding-experiment-plan-v2",
        "cpu10",
    )
    summary = _audit_legacy(
        closure_bytes=closure_bytes,
        terminal_bytes=terminal_bytes,
        result_bytes=result_bytes,
        hello_bytes=hello_bytes,
        plan_sha=plan_sha,
        invocation=invocation,
        model=model,
    )
    if summary["status"] != "complete":
        return summary
    hello = _decode(hello_bytes, 4096)
    _need(
        hello.get("unit") == plan["unit"]
        and hello.get("cgroup", "").endswith("/app.slice/" + plan["unit"]),
        "hello unit and cgroup differ from plan",
    )
    result = _decode(result_bytes, 4 * 1024 * 1024)
    _need(result.get("cpu_profile") == "cpu10", "result cpu10 profile required")
    telemetry = _events(events_bytes, plan, hello, invocation)
    summary["cpu_profile"] = "cpu10"
    summary["cpu_telemetry"] = telemetry
    summary["under_20_seconds"] = {
        mode: sum(ns < 20_000_000_000 for ns in info["samples_ns"])
        for mode, info in summary["timing"].items()
    }
    summary["scope"] += (
        "; v2 plan and submitted events checked, but caller must authenticate all source and evidence bytes"
    )
    if baseline is not None:
        _need(
            type(baseline) is dict
            and set(baseline)
            == {
                "plan_bytes",
                "approved_source_pins",
                "closure_bytes",
                "terminal_bytes",
                "result_bytes",
                "hello_bytes",
                "plan_sha",
                "invocation",
                "model",
            },
            "explicit baseline evidence required",
        )
        base_plan = _plan_evidence(
            baseline["plan_bytes"],
            baseline["plan_sha"],
            baseline["approved_source_pins"],
            "decision3-cpu-padding-experiment-plan-v1",
        )
        base = _audit_legacy(
            **{
                key: baseline[key]
                for key in (
                    "closure_bytes",
                    "terminal_bytes",
                    "result_bytes",
                    "hello_bytes",
                    "plan_sha",
                    "invocation",
                    "model",
                )
            }
        )
        _need(base["status"] == "complete", "completed baseline required")
        base_hello = _decode(baseline["hello_bytes"], 4096)
        _need(
            base_hello.get("unit") == base_plan["unit"]
            and base_hello.get("cgroup", "").endswith("/app.slice/" + base_plan["unit"]),
            "baseline hello differs from plan",
        )
        base_result = _decode(baseline["result_bytes"], 4 * 1024 * 1024)
        same = (
            json.dumps(result.get("state"), ensure_ascii=False, separators=(",", ":"))
            == json.dumps(base_result.get("state"), ensure_ascii=False, separators=(",", ":"))
            and json.dumps(result.get("questions"), ensure_ascii=False, separators=(",", ":"))
            == json.dumps(base_result.get("questions"), ensure_ascii=False, separators=(",", ":"))
            and result.get("head_token_lengths") == base_result.get("head_token_lengths")
            and model == baseline["model"]
        )
        comparison = {
            "serialized_inputs_and_model_label_match": same,
            "scope": "independent sequential runs; model weights, tokenizer, host activity and source authenticity not established; timings are not paired",
        }
        if same:
            comparison["p50_ratio_v2_to_baseline"] = {
                mode: summary["timing"][mode]["p50_ns"] / base["timing"][mode]["p50_ns"]
                for mode in MODES
            }
        summary["baseline_comparison"] = comparison
    return summary
