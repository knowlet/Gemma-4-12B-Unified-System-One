"""Lossless schema bridge for Jev Ultrafast's observed operation/target heads.

The browser policy and executor stay upstream. This callable replaces only its
TypeSafe decision transport; it does not plan actions or generate field values.
"""

from __future__ import annotations

import copy
import json
import time

from s1.backends import UnsupportedRequest, normalize_response
from s1.contracts import MAX_OPTIONS, DecisionRequest
from s1.errors import RequestValidationError

TYPESAFE_DECISION_URL = "https://api.typesafe.ai/v1/systemone"


def _capture(value):
    """Keep invalid/non-JSON output auditable without corrupting trace JSON."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False)), "json"
    except (TypeError, ValueError, OverflowError):
        return repr(value), "python_repr_not_json"


def _description(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    raise RequestValidationError("Ultrafast descriptions must be strings or JSON objects/lists")


def map_decision_request(body):
    """Preserve state, question order, labels, and structured descriptions.

    A one-candidate target is already determined by the observed action space.
    It is reported explicitly rather than padded with a fabricated candidate.
    Larger heads are rejected whole; no observed target is silently discarded.
    """
    if not isinstance(body, dict) or set(body) != {"model", "state", "questions"}:
        raise RequestValidationError("expected Ultrafast model, state, and questions")
    try:
        json.dumps(body, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RequestValidationError("Ultrafast request must contain finite JSON values") from exc
    if not isinstance(body["model"], str) or not body["model"]:
        raise RequestValidationError("Ultrafast model must be a nonempty string")
    if not isinstance(body["state"], (str, dict)):
        raise RequestValidationError("Ultrafast state must be a string or object")
    questions = body["questions"]
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise RequestValidationError("Ultrafast needs 1..64 named questions")
    mapped, singletons = {}, {}
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not 1 <= len(question_id) <= 128:
            raise RequestValidationError("invalid Ultrafast question id")
        if not isinstance(question, dict) or set(question) != {
            "type",
            "instructions",
            "criteria",
        }:
            raise RequestValidationError("expected Ultrafast type, instructions, and criteria")
        if question["type"] != "choice":
            raise UnsupportedRequest("Ultrafast bridge supports only observed choice heads")
        instructions = _description(question["instructions"])
        if not 1 <= len(instructions) <= 16000:
            raise RequestValidationError("Ultrafast instructions need 1..16000 characters")
        criteria = question["criteria"]
        if not isinstance(criteria, dict) or not criteria:
            raise RequestValidationError("Ultrafast criteria must be a nonempty named object")
        if len(criteria) > MAX_OPTIONS:
            raise UnsupportedRequest(
                f"Ultrafast head {question_id!r} has {len(criteria)} options; "
                f"the backend contract supports at most {MAX_OPTIONS}; nothing was truncated"
            )
        if any(not isinstance(label, str) or not label.strip() for label in criteria):
            raise RequestValidationError("Ultrafast choice labels must be nonempty strings")
        descriptions = {label: _description(value) for label, value in criteria.items()}
        if len(descriptions) == 1:
            label = next(iter(descriptions))
            singletons[question_id] = {
                "id": question_id,
                "type": "choice",
                "choice": label,
                "probabilities": {label: 1.0},
                "confidence": 1.0,
                "margin": 1.0,
            }
        else:
            mapped[question_id] = {
                "type": "choice",
                "instructions": instructions,
                "criteria": descriptions,
            }
    request = (
        DecisionRequest.model_validate({"state": copy.deepcopy(body["state"]), "questions": mapped})
        if mapped
        else None
    )
    return request, singletons


class UltrafastDecisionBridge:
    """Callable decision transport with one append-only attempt record per call.

    ``backend_calls`` counts actual ``backend.predict`` invocations, including
    failed invocations. Singleton heads incur no model call. Credentials are
    deliberately absent from records. Field-text requests must be routed to a
    separate, explicitly configured text transport by the caller.
    """

    def __init__(self, backend):
        self.backend = backend
        self.records = []
        self.backend_calls = 0

    def __call__(self, url, key, body):
        del key  # The local backend has no need for the upstream transport key.
        started = time.perf_counter()
        captured, body_format = _capture(body)
        record = {
            "attempt": len(self.records) + 1,
            "body": captured,
            "body_format": body_format,
            "mapped_request": None,
            "singleton_heads": [],
            "raw_response": None,
            "raw_response_format": None,
            "normalized_response": None,
            "backend_calls": 0,
            "reported_cost_usd": None,
            "error": None,
        }
        try:
            if url != TYPESAFE_DECISION_URL:
                raise UnsupportedRequest("Ultrafast bridge accepts only the TypeSafe decision URL")
            request, singletons = map_decision_request(body)
            record["singleton_heads"] = list(singletons)
            record["mapped_request"] = request.model_dump() if request else None
            raw, normalized = {}, {"answers": {}}
            if request is not None:
                record["backend_calls"] = 1
                self.backend_calls += 1
                raw = self.backend.predict(request)
                record["raw_response"], record["raw_response_format"] = _capture(raw)
                normalized = normalize_response(request, raw)
            answers = {**normalized["answers"], **singletons}
            result = {
                "answers": {name: answers[name] for name in body["questions"]},
                "model": getattr(self.backend, "name", type(self.backend).__name__),
                "usage": normalized.get("usage"),
            }
            # Force trace serializability before returning anything executable.
            json.dumps(result, ensure_ascii=False, allow_nan=False)
            record["normalized_response"] = copy.deepcopy(result)
            return result
        except Exception as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            record["latency_ms"] = (time.perf_counter() - started) * 1000
            self.records.append(record)
