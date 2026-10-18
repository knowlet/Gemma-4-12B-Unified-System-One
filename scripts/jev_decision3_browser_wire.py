"""Opt-in, request-bound answer-container conversion for the pinned Jev caller.

The legacy Decision3 browser adapter returns an ordered answer list. Pinned
upstream choose() requires a named object. This helper changes only that
container and adds separate wire provenance; it performs no policy inference.
Existing API routes and frozen native/replay sources do not import this module.
"""

from __future__ import annotations

import copy
import hashlib
import json

PROFILE = "jev_decision3_browser_named_answers_v1"
LEGACY_PROFILE = "jev_decision3_native_v1"
METADATA_KEY = "decision3_browser_wire"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise ValueError("Wire conversion requires finite UTF-8 JSON") from exc
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            _require(all(type(key) is str for key in item), "JSON object keys must be strings")
            pending.extend(item.values())
        elif type(item) is list:
            pending.extend(item)
        else:
            _require(
                item is None or type(item) in (str, bool, int, float),
                "Only JSON values are supported",
            )


def named_browser_response(request, response):
    """Deep-copy a validated legacy result into the caller's named-answer shape.

    Every request head must appear once in the original request order. Invalid
    or already converted formats are refused; answers are never repaired,
    reordered, synthesized, filtered or renormalized.
    """
    _json(request)
    _json(response)
    _require(
        type(request) is dict and type(response) is dict, "JSON request/response objects required"
    )
    questions = request.get("questions")
    _require(
        type(questions) is dict and bool(questions), "Ordered named request questions required"
    )
    ids = list(questions)
    _require(all(type(key) is str and key.strip() for key in ids), "Nonempty question IDs required")
    answers = response.get("answers")
    _require(type(answers) is list, "Legacy ordered answer list required")
    _require(METADATA_KEY not in response, "Wire metadata already present; conversion refused")
    audit = response.get("decision3_native")
    _require(
        type(audit) is dict and audit.get("version") == LEGACY_PROFILE,
        "Explicit legacy Decision3 native profile required",
    )
    request_json = json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    request_sha = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
    _require(
        audit.get("original_request_sha256") == request_sha
        and json.dumps(
            audit.get("original_request"),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        == request_json,
        "Legacy response belongs to a different original request",
    )
    actual = []
    for answer in answers:
        _require(
            type(answer) is dict and type(answer.get("id")) is str, "Named answer objects required"
        )
        actual.append(answer["id"])
    _require(len(actual) == len(set(actual)), "Duplicate answer IDs refused")
    _require(set(actual) == set(ids), "Missing or unknown answer IDs refused")
    _require(actual == ids, "Answer order differs from the original request")
    result = copy.deepcopy(response)
    result["answers"] = {key: answer for key, answer in zip(ids, result["answers"], strict=True)}
    result[METADATA_KEY] = {
        "profile": PROFILE,
        "source_profile": LEGACY_PROFILE,
        "question_ids": ids,
        "answer_count": len(ids),
        "original_request_sha256": request_sha,
        "conversion": "ordered_list_to_named_object",
        "answer_values_changed": False,
        "scope": "Wire container only; no policy, probability or action changes",
    }
    return result
