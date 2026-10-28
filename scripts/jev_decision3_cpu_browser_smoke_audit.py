"""Held, I/O-free audit of four caller-supplied synthetic browser smoke slots.

The caller must independently authenticate the supplied plan, closure, client
receipt, action and independent DOM captures; supplied digest attestations do
not establish their provenance. No model, browser, service or file is opened.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

SOURCE_ONLY = True
PROFILES = ("v7", "v7-links", "v7-links", "v7")
TARGETS = {"CLICK": "click_target", "TYPE_TEXT": "type_text_target", "SELECT": "select_target"}
OPERATIONS = {*TARGETS, "SCROLL_DOWN", "SCROLL_UP", "WAIT", "DONE", "BLOCKED"}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _decode(raw, limit):
    _need(type(raw) is bytes and 0 < len(raw) <= limit, "bounded JSON bytes required")

    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError("nonfinite JSON constant")

    def finite(value):
        result = float(value)
        _need(math.isfinite(result), "nonfinite JSON number")
        return result

    result = json.loads(
        raw, object_pairs_hook=pairs, parse_constant=reject_constant, parse_float=finite
    )
    _need(type(result) is dict, "JSON object required")
    return result


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _attested(value, attestations):
    return _sha(_encode(value)) in attestations


def _labels(question):
    _need(
        type(question) is dict and question.get("type", "choice") == "choice",
        "choice question required",
    )
    present = [key for key in ("criteria", "options", "levels") if key in question]
    _need(len(present) == 1, "one criteria field required")
    options = question[present[0]]
    if type(options) is dict:
        labels = list(options)
    else:
        _need(
            type(options) is list and all(type(option) is str for option in options),
            "ordered criteria required",
        )
        labels = options
    _need(
        1 <= len(labels) <= 255
        and all(type(label) is str and label.strip() for label in labels)
        and len(set(labels)) == len(labels),
        "bounded distinct option labels required",
    )
    return labels


def _answer(answer, labels, *, native):
    _need(
        type(answer) is dict
        and set(answer)
        == (
            {"type", "choice", "probabilities", "confidence"}
            if native
            else {"id", "type", "choice", "probabilities", "confidence", "margin"}
        )
        and answer.get("type") == "choice"
        and type(answer.get("probabilities")) is dict
        and list(answer["probabilities"]) == labels,
        "complete ordered answer probabilities required",
    )
    values = list(answer["probabilities"].values())
    _need(
        all(
            type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
            for value in values
        )
        and math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        and answer.get("choice") == labels[values.index(max(values))]
        and type(answer.get("confidence")) in (int, float)
        and math.isfinite(answer["confidence"])
        and 0 <= answer["confidence"] <= 1,
        "complete finite model choice required",
    )
    _need(len(values) != 1 or values == [1.0], "singleton probability must be exact")
    if native:
        expected_confidence = (
            1.0
            if len(values) == 1
            else max(
                0.0,
                min(
                    1.0,
                    1
                    + sum(value * math.log(value) for value in values if value > 0)
                    / math.log(len(values)),
                ),
            )
        )
        _need(
            math.isclose(answer["confidence"], expected_confidence, rel_tol=1e-12, abs_tol=1e-12),
            "native entropy confidence differs",
        )
    else:
        ordered = sorted(values, reverse=True)
        expected_margin = 1.0 if len(ordered) == 1 else ordered[0] - ordered[1]
        _need(
            math.isclose(answer["confidence"], ordered[0], rel_tol=1e-12, abs_tol=1e-12)
            and type(answer["margin"]) in (int, float)
            and math.isfinite(answer["margin"])
            and math.isclose(answer["margin"], expected_margin, rel_tol=1e-12, abs_tol=1e-12),
            "named confidence or margin differs",
        )
    return values


def _call(call, *, alias, plan_sha, invocation, attestations):
    _need(
        type(call) is dict
        and set(call) == {"call_id", "request", "response", "delivery", "action"}
        and type(call["call_id"]) is int
        and 1 <= call["call_id"] <= 16
        and type(call["request"]) is str
        and type(call["response"]) is str,
        "complete call evidence required",
    )
    request_raw, response_raw = call["request"].encode(), call["response"].encode()
    request = _decode(request_raw, 1_000_000)
    response = _decode(response_raw, 16 * 1024**2)
    questions = request.get("questions")
    _need(
        request.get("model") == alias
        and type(questions) is dict
        and "operation" in questions
        and 1 <= len(questions) <= 4
        and not set(questions) - {"operation", *TARGETS.values()},
        "complete browser request required",
    )
    labels = {key: _labels(value) for key, value in questions.items()}
    _need(
        not set(labels["operation"]) - OPERATIONS
        and all(
            (operation in labels["operation"]) == (target in labels)
            for operation, target in TARGETS.items()
        ),
        "operation and target inventory differs",
    )
    answers, native = response.get("answers"), response.get("decision3_native")
    _need(
        response.get("model") == alias
        and type(answers) is dict
        and list(answers) == list(questions)
        and type(native) is dict
        and native.get("version") == "jev_decision3_native_v1"
        and native.get("status") == "completed"
        and _encode(native.get("original_request")) == _encode(request)
        and native.get("original_request_sha256") == _sha(_encode(request)),
        "response/request identity differs",
    )
    wire = response.get("decision3_browser_wire")
    cpu = response.get("decision3_cpu")
    _need(
        type(wire) is dict
        and wire.get("profile") == "jev_decision3_browser_named_answers_v1"
        and wire.get("source_profile") == "jev_decision3_native_v1"
        and wire.get("conversion") == "ordered_list_to_named_object"
        and wire.get("answer_values_changed") is False
        and wire.get("question_ids") == list(questions)
        and wire.get("answer_count") == len(questions)
        and wire.get("original_request_sha256") == _sha(_encode(request))
        and type(cpu) is dict
        and cpu.get("profile") == "decision3-cpu-browser-response-v1"
        and cpu.get("device") == "cpu",
        "CPU named response metadata differs",
    )
    raw = native.get("raw_response")
    _need(
        type(raw) is dict
        and raw.get("model") == "d3-edge"
        and type(raw.get("answers")) is dict
        and list(raw["answers"]) == list(questions),
        "complete publisher response required",
    )
    usage = raw.get("usage")
    _need(
        type(usage) is dict
        and type(usage.get("input_tokens")) is int
        and usage["input_tokens"] > 0
        and type(usage.get("output_tokens")) is int
        and usage["output_tokens"] == 0
        and response.get("usage") == usage,
        "publisher usage differs",
    )
    lengths = cpu.get("native_head_lengths")
    _need(
        type(lengths) is dict
        and list(lengths) == list(questions)
        and all(type(value) is int and 1 <= value <= 16384 for value in lengths.values())
        and sum(lengths.values()) == usage["input_tokens"],
        "head token usage differs",
    )
    for key in questions:
        values = _answer(answers[key], labels[key], native=False)
        raw_values = _answer(raw["answers"][key], labels[key], native=True)
        _need(
            answers[key].get("id") == key
            and answers[key]["choice"] == raw["answers"][key]["choice"]
            and all(
                math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                for a, b in zip(values, raw_values, strict=True)
            ),
            "named answer differs from native answer",
        )
    operation = answers["operation"]["choice"]
    target = answers[TARGETS[operation]]["choice"] if operation in TARGETS else None
    delivery, action = call["delivery"], call["action"]
    _need(
        type(delivery) is dict
        and delivery.get("status") == "client_received_and_handler_closed"
        and delivery.get("plan_sha256") == plan_sha
        and delivery.get("invocation") == invocation
        and type(delivery.get("call_id")) is int
        and delivery["call_id"] == call["call_id"]
        and delivery.get("request_sha256") == _sha(request_raw)
        and delivery.get("response_sha256") == _sha(response_raw)
        and delivery.get("client_response_sha256") == _sha(response_raw)
        and delivery.get("socket_write_returned") is True
        and delivery.get("handler_closed") is True
        and type(delivery.get("socket_write_elapsed_seconds")) in (int, float)
        and math.isfinite(delivery["socket_write_elapsed_seconds"])
        and type(delivery.get("request_to_socket_write_elapsed_seconds")) in (int, float)
        and math.isfinite(delivery["request_to_socket_write_elapsed_seconds"])
        and 0
        <= delivery["socket_write_elapsed_seconds"]
        <= delivery["request_to_socket_write_elapsed_seconds"]
        < 20,
        "matching externally verified timely delivery required",
    )
    _need(
        type(action) is dict
        and action.get("status") == "executed"
        and action.get("plan_sha256") == plan_sha
        and action.get("invocation") == invocation
        and type(action.get("call_id")) is int
        and action["call_id"] == call["call_id"]
        and action.get("operation") == operation
        and action.get("target") == target,
        "actual action must match the complete model choice",
    )
    return _attested(delivery, attestations) and _attested(action, attestations)


def audit(
    *,
    plan_bytes,
    evidence_bytes,
    closure_bytes,
    expected_plan_sha256,
    expected_invocation,
    externally_verified_sha256s,
):
    """Check four evidence slots; missing external attestations never yield a score."""
    _need(SOURCE_ONLY is False, "reviewed smoke audit release required")
    _need(
        _digest(expected_plan_sha256)
        and type(plan_bytes) is bytes
        and _sha(plan_bytes) == expected_plan_sha256,
        "exact plan bytes required",
    )
    _need(
        expected_invocation is None
        or (
            type(expected_invocation) is str and re.fullmatch(r"[0-9a-f]{32}", expected_invocation)
        ),
        "invocation required",
    )
    _need(
        type(externally_verified_sha256s) in (set, frozenset)
        and len(externally_verified_sha256s) <= 128
        and all(_digest(value) for value in externally_verified_sha256s),
        "bounded external attestations required",
    )
    plan, evidence, closure = (
        _decode(plan_bytes, 1_000_000),
        _decode(evidence_bytes, 32 * 1024**2),
        _decode(closure_bytes, 65536),
    )
    slots = plan.get("slots")
    _need(
        plan.get("schema") == "decision3-cpu-browser-smoke-plan-v1"
        and type(plan.get("model")) is str
        and plan["model"].strip()
        and type(slots) is list
        and len(slots) == 4,
        "four-slot smoke plan required",
    )
    for index, slot in enumerate(slots):
        _need(
            type(slot) is dict
            and set(slot) == {"slot", "profile", "fixture_sha256", "expected_dom_sha256"}
            and type(slot["slot"]) is int
            and slot["slot"] == index
            and slot["profile"] == PROFILES[index]
            and _digest(slot["fixture_sha256"])
            and _digest(slot["expected_dom_sha256"]),
            "balanced plan slot required",
        )
    _need(len({slot["fixture_sha256"] for slot in slots}) == 1, "same synthetic fixture required")
    _need(
        evidence.get("schema") == "decision3-cpu-browser-smoke-evidence-v1"
        and evidence.get("plan_sha256") == expected_plan_sha256
        and evidence.get("invocation") == expected_invocation
        and type(evidence.get("slots")) is list
        and len(evidence["slots"]) == 4,
        "complete smoke evidence inventory required",
    )
    _need(
        closure.get("schema") == "decision3-cpu-browser-smoke-closure-v1"
        and closure.get("plan_sha256") == expected_plan_sha256
        and closure.get("invocation") == expected_invocation,
        "matching smoke closure required",
    )
    scope = (
        "caller-supplied synthetic evidence and digest attestations only; caller must authenticate sources, "
        "capture origins, request-to-socket-write timing, delivery, independent DOM oracle, common success "
        "criteria across plan slots and process cleanup; socket-write timing does not certify client "
        "end-to-end latency; reported failed slot events are not independently validated here and do not "
        "prove execution started; timeout must be recorded as an externally attested failed slot; "
        "no real-world benchmark claim"
    )
    closed = (
        type(expected_invocation) is str
        and closure.get("status") == "closed"
        and closure.get("cgroup") in {"absent", "confirmed_unpopulated"}
        and closure.get("process_group") == "absent"
        and closure.get("browser_children") == "absent"
        and closure.get("listener") == "closed"
        and _attested(closure, externally_verified_sha256s)
    )
    if not closed:
        return {
            "status": "unknown",
            "score": None,
            "denominator": {"v7": 2, "v7-links": 2},
            "slots": [
                {"slot": i, "profile": profile, "outcome": "unknown"}
                for i, profile in enumerate(PROFILES)
            ],
            "scope": scope,
        }
    outcomes, previous_failed, last_call = [], False, 0
    for index, row in enumerate(evidence["slots"]):
        _need(
            type(row) is dict
            and set(row) == {"slot", "profile", "status", "calls", "final_dom"}
            and type(row["slot"]) is int
            and row["slot"] == index
            and row["profile"] == PROFILES[index]
            and row["status"] in {"completed", "failed", "not_executed"}
            and type(row["calls"]) is list
            and len(row["calls"]) <= 4,
            "ordered slot evidence required",
        )
        _need(
            not previous_failed or row["status"] == "not_executed", "failed service cannot resume"
        )
        if row["status"] != "completed":
            _need(
                row["final_dom"] is None and (row["status"] != "not_executed" or not row["calls"]),
                "failed slot cannot claim success",
            )
            previous_failed = True
            outcome = row["status"] if _attested(row, externally_verified_sha256s) else "unknown"
        else:
            _need(row["calls"], "completed slot requires real model calls")
            verified = _attested(row, externally_verified_sha256s)
            for call in row["calls"]:
                _need(
                    type(call) is dict
                    and type(call.get("call_id")) is int
                    and call["call_id"] == last_call + 1,
                    "contiguous ordered call IDs required",
                )
                last_call = call["call_id"]
                verified = (
                    _call(
                        call,
                        alias=plan["model"],
                        plan_sha=expected_plan_sha256,
                        invocation=expected_invocation,
                        attestations=externally_verified_sha256s,
                    )
                    and verified
                )
            dom = row["final_dom"]
            _need(
                type(dom) is dict
                and set(dom)
                == {
                    "capture_type",
                    "plan_sha256",
                    "invocation",
                    "call_id",
                    "fixture_sha256",
                    "html",
                    "sha256",
                }
                and dom["capture_type"] == "independent_post_action_dom"
                and dom["plan_sha256"] == expected_plan_sha256
                and dom["invocation"] == expected_invocation
                and type(dom["call_id"]) is int
                and dom["call_id"] == last_call
                and dom["fixture_sha256"] == slots[index]["fixture_sha256"]
                and type(dom["html"]) is str
                and len(dom["html"].encode()) <= 4 * 1024**2
                and _sha(dom["html"].encode()) == dom["sha256"],
                "independent final DOM evidence required",
            )
            verified = _attested(dom, externally_verified_sha256s) and verified
            outcome = (
                ("success" if dom["sha256"] == slots[index]["expected_dom_sha256"] else "failed")
                if verified
                else "unknown"
            )
        outcomes.append({"slot": index, "profile": PROFILES[index], "outcome": outcome})
    if any(row["outcome"] == "unknown" for row in outcomes):
        for row in outcomes:
            if row["outcome"] == "success":
                row["outcome"] = "unscored"
        return {
            "status": "unknown",
            "score": None,
            "denominator": {"v7": 2, "v7-links": 2},
            "slots": outcomes,
            "scope": scope,
        }
    counts = {
        profile: sum(row["outcome"] == "success" for row in outcomes if row["profile"] == profile)
        for profile in ("v7", "v7-links")
    }
    return {
        "status": "audited_caller_attested",
        "denominator": {"v7": 2, "v7-links": 2},
        "score": {
            profile: {
                "successes": count,
                "scheduled": 2,
                "completed": sum(
                    row["status"] == "completed"
                    for row in evidence["slots"]
                    if row["profile"] == profile
                ),
                "reported_failed": sum(
                    row["status"] == "failed"
                    for row in evidence["slots"]
                    if row["profile"] == profile
                ),
                "not_executed": sum(
                    row["status"] == "not_executed"
                    for row in evidence["slots"]
                    if row["profile"] == profile
                ),
            }
            for profile, count in counts.items()
        },
        "slots": outcomes,
        "scope": scope,
    }
