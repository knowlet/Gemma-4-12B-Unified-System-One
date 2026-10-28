"""Lossless schema bridge for Jev Ultrafast's observed operation/target heads.

The browser policy and executor stay upstream. This callable replaces only its
TypeSafe decision transport; it does not plan actions or generate field values.
"""

from __future__ import annotations

import copy
import json
import math
import time

from s1.backends import UnsupportedRequest, normalize_response
from s1.contracts import MAX_OPTIONS, DecisionRequest
from s1.errors import BackendResponseError, RequestValidationError

TYPESAFE_DECISION_URL = "https://api.typesafe.ai/v1/systemone"
HIERARCHICAL_HEAD_POLICY = {
    "name": "balanced-contiguous-all-groups-v1",
    "grouping": "ceil(N/52) contiguous groups in original order, sizes differing by at most one",
    "scoring": "One joint core request with unchanged ordinary heads, each parent head, and every conditional group head; no pruning.",
    "probability_semantics": "uncalibrated_hierarchical_approximation",
    "aggregation": "P(candidate) = P(group) * P(candidate | group)",
    "max_groups": MAX_OPTIONS,
    "core_max_options": MAX_OPTIONS,
    "state_or_candidate_truncation": False,
    "calibration_claim": None,
    "separate_call_parity_claim": None,
}


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


def _has_oversized_head(body):
    return (
        isinstance(body, dict)
        and isinstance(body.get("questions"), dict)
        and any(
            isinstance(question, dict)
            and isinstance(question.get("criteria"), dict)
            and len(question["criteria"]) > MAX_OPTIONS
            for question in body["questions"].values()
        )
    )


def _hierarchical_plan(body):
    """Expand all structural heads in place, then validate one joint core request."""
    transformed, plans, joint = copy.deepcopy(body), [], {}
    used_ids, next_id = set(body["questions"]), 1
    for question_id, question in body["questions"].items():
        if (
            not isinstance(question, dict)
            or not isinstance(question.get("criteria"), dict)
            or len(question["criteria"]) <= MAX_OPTIONS
        ):
            joint[question_id] = copy.deepcopy(question)
            continue
        criteria = question["criteria"]
        instructions = _description(question.get("instructions"))
        if not 1 <= len(instructions) <= 16000:
            raise RequestValidationError("Ultrafast instructions need 1..16000 characters")
        if any(not isinstance(label, str) or not label.strip() for label in criteria):
            raise RequestValidationError("Ultrafast choice labels must be nonempty strings")
        for description in criteria.values():
            _description(description)
        count = (len(criteria) + MAX_OPTIONS - 1) // MAX_OPTIONS
        if count > MAX_OPTIONS:
            raise UnsupportedRequest(
                f"Ultrafast head {question_id!r} needs {count} groups; "
                f"hierarchical policy supports at most {MAX_OPTIONS}; nothing was truncated"
            )
        base, remainder = divmod(len(criteria), count)
        items, groups, offset = list(criteria.items()), [], 0
        parent = copy.deepcopy(question)
        parent["criteria"] = {}
        parent["instructions"] = {
            "original_instructions": copy.deepcopy(question.get("instructions")),
            "hierarchy": {
                "stage": "parent",
                "rule": "Choose the group containing the best original candidate under the original instructions; group members retain their original IDs and descriptions.",
            },
        }
        joint[question_id] = parent
        for index in range(count):
            size = base + int(index < remainder)
            members = dict(items[offset : offset + size])
            offset += size
            group_id = f"group_{index + 1}"
            parent["criteria"][group_id] = {"members": copy.deepcopy(members)}
            while f"__s1_hierarchy_{next_id}" in used_ids:
                next_id += 1
            head_id = f"__s1_hierarchy_{next_id}"
            used_ids.add(head_id)
            next_id += 1
            joint[head_id] = {
                "type": question.get("type"),
                "instructions": {
                    "original_instructions": copy.deepcopy(question.get("instructions")),
                    "hierarchy": {
                        "stage": "conditional",
                        "group": group_id,
                        "rule": "Condition on the best original candidate belonging to this group; choose the best original candidate within it under the original instructions.",
                    },
                },
                "criteria": copy.deepcopy(members),
            }
            groups.append({"id": group_id, "candidate_ids": list(members), "head_id": head_id})
        plans.append({"question_id": question_id, "groups": groups})
    # Reusing the original mapper also validates unchanged outer/question keys,
    # all ordinary heads, and all parent heads before any backend dispatch.
    transformed["questions"] = joint
    mapped, singletons = map_decision_request(transformed)
    return mapped, singletons, plans


class UltrafastDecisionBridge:
    """Callable decision transport with one append-only attempt record per call.

    ``backend_calls`` counts actual ``backend.predict`` invocations, including
    failed invocations. Singleton heads incur no model call. Credentials are
    deliberately absent from records. Field-text requests must be routed to a
    separate, explicitly configured text transport by the caller.
    """

    def __init__(self, backend, *, oversized_head_policy="reject", backbone_forward_count=None):
        if oversized_head_policy not in ("reject", "hierarchical"):
            raise ValueError("oversized head policy must be reject or hierarchical")
        self.backend = backend
        self.oversized_head_policy = oversized_head_policy
        self.backbone_forward_count = backbone_forward_count
        self.records = []
        self.backend_calls = 0

    def _subcall(self, request, record):
        started = time.perf_counter()
        call = {
            "kind": "joint",
            "internal_head_ids": [question.id for question in request.questions],
            "mapped_request": request.model_dump(),
            "raw_response": None,
            "raw_response_format": None,
            "normalized_response": None,
            "backend_calls": 1,
            "backbone_forwards": None,
            "observed_completed_forward_delta": None,
            "forward_count_basis": "successful_backbone_returns",
            "forward_count_interpretation": None,
            "forward_observation_error": None,
            "reported_cost_usd": None,
            "error": None,
        }
        record["hierarchy"]["subcalls"].append(call)
        forward_start = None
        if self.backbone_forward_count is not None:
            try:
                forward_start = self.backbone_forward_count()
                if type(forward_start) is not int or forward_start < 0:
                    raise ValueError("backbone forward count must be a nonnegative integer")
            except Exception as exc:
                call["forward_observation_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                forward_start = None
        record["backend_calls"] += 1
        self.backend_calls += 1
        try:
            raw = self.backend.predict(request)
            call["raw_response"], call["raw_response_format"] = _capture(raw)
            record["raw_response"] = call["raw_response"]
            record["raw_response_format"] = call["raw_response_format"]
            normalized = normalize_response(request, raw)
            call["normalized_response"] = copy.deepcopy(normalized)
            return normalized
        except BaseException as exc:
            call["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            if forward_start is not None:
                try:
                    forward_end = self.backbone_forward_count()
                    if type(forward_end) is not int or forward_end < forward_start:
                        raise ValueError("backbone forward count decreased or changed type")
                    delta = forward_end - forward_start
                    call["observed_completed_forward_delta"] = delta
                    if call["error"] is None:
                        call["backbone_forwards"] = delta
                        call["forward_count_interpretation"] = "exact_successful_forwards"
                    else:
                        # A dispatch that raises before its forward post-hook
                        # may be missing from this successful-return counter.
                        call["forward_count_interpretation"] = "lower_bound"
                except Exception as exc:
                    call["forward_observation_error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
            record["hierarchy"]["backbone_forwards"] = call["backbone_forwards"]
            for key in (
                "observed_completed_forward_delta",
                "forward_count_basis",
                "forward_count_interpretation",
                "forward_observation_error",
            ):
                record["hierarchy"][key] = call[key]
            call["latency_ms"] = (time.perf_counter() - started) * 1000

    def _hierarchical(self, body, record):
        request, singletons, plans = _hierarchical_plan(body)
        record["singleton_heads"] = list(singletons)
        record["mapped_request"] = request.model_dump()
        record["hierarchy"] = {
            "policy": copy.deepcopy(HIERARCHICAL_HEAD_POLICY),
            "internal_head_count": len(request.questions),
            "heads": [
                {
                    "question_id": plan["question_id"],
                    "groups": [
                        {
                            "id": group["id"],
                            "candidate_ids": group["candidate_ids"],
                            "head_id": group["head_id"],
                        }
                        for group in plan["groups"]
                    ],
                }
                for plan in plans
            ],
            "subcalls": [],
        }
        normalized = self._subcall(request, record)
        answers = {**normalized["answers"], **singletons}
        for plan in plans:
            question_id = plan["question_id"]
            group_probs = normalized["answers"][question_id]["probabilities"]
            probabilities = {}
            for group in plan["groups"]:
                within = normalized["answers"][group["head_id"]]["probabilities"]
                for label in group["candidate_ids"]:
                    probabilities[label] = group_probs[group["id"]] * within[label]
            if (
                list(probabilities) != list(body["questions"][question_id]["criteria"])
                or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())
                or not math.isclose(math.fsum(probabilities.values()), 1, rel_tol=0, abs_tol=1e-10)
            ):
                raise BackendResponseError("invalid complete hierarchical probability distribution")
            ranked = sorted(probabilities.values(), reverse=True)
            answers[question_id] = {
                "id": question_id,
                "type": "choice",
                "choice": max(probabilities, key=probabilities.get),
                "probabilities": probabilities,
                "confidence": ranked[0],
                "margin": ranked[0] - ranked[1],
                "probability_semantics": HIERARCHICAL_HEAD_POLICY["probability_semantics"],
            }
        result = {
            "answers": {name: answers[name] for name in body["questions"]},
            "model": getattr(self.backend, "name", type(self.backend).__name__),
            "usage": None,
        }
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        record["normalized_response"] = copy.deepcopy(result)
        return result

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
            if self.oversized_head_policy == "hierarchical" and _has_oversized_head(body):
                return self._hierarchical(body, record)
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
