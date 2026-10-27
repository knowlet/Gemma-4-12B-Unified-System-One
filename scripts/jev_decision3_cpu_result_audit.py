"""Held, I/O-free audit of caller-supplied synthetic CPU experiment bytes.

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


def audit(*, closure_bytes, terminal_bytes, result_bytes, hello_bytes, plan_sha, invocation, model):
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
