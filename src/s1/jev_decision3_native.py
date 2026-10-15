"""Separate full-head Decision3 browser profile with explicit native255 DTOs.

Raw ordered JSON instructions/criteria reach the publisher's own serializer.
The generic 52-option contracts and native20 hierarchy are unchanged. Every
requested head, including a singleton, is evaluated by the native checkpoint.
"""

from __future__ import annotations

import copy
import hashlib
import time
from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from .contracts import StrictModel
from .errors import BackendResponseError, BackendTimeoutError, RequestValidationError
from .evaluation.decision3 import (
    ARCHITECTURAL_CONTEXT,
    CONTEXT_LIMIT,
    MAX_OPTIONS,
    configuration_blockers,
    normalize_native_response,
)
from .jev_generated import _request
from .jev_hierarchy import _check_json, _evidence

PROFILE = "jev_decision3_native_v1"


def profile_manifest():
    return {
        "profile": PROFILE,
        "scope": "publisher-native full browser request; every requested head evaluated",
        "candidate_cap": MAX_OPTIONS,
        "singleton_policy": "native_forward; no forced answer or skipped head",
        "question_policy": "all requested heads in original order, no operation-first or hierarchy",
        "native_hard_context_cap": None,
        "architectural_context_metadata": ARCHITECTURAL_CONTEXT,
        "project_context_cap": CONTEXT_LIMIT,
        "budget_seconds": 20,
        "timeout_policy": "one cooperative timer across parse/preflight/forward; before and after blocking calls; no hard cancellation",
        "external_http_timeout_seconds": 25,
        "external_http_responsibility": "unchanged upstream caller; this synchronous adapter cannot enforce delivery or hard-cancel HTTP25",
        "limitations": [
            "256..400 candidate heads are unsupported and rejected intact; no pruning",
            "native numerical replay, task success, speed and peak memory are unverified",
            "context16384 is a prospective project resource cap, not a publisher measured limit",
        ],
    }


class Decision3NativeQuestion(StrictModel):
    """Normal validated browser-only choice DTO, independent of Question's52 cap."""

    id: str = Field(min_length=1, max_length=128)
    type: Literal["choice"] = "choice"
    instructions: str | dict[str, Any] | list[Any]
    criteria: dict[str, Any] = Field(min_length=1, max_length=MAX_OPTIONS)

    @model_validator(mode="after")
    def finite_native_values(self):
        _check_json(self.instructions)
        _check_json(self.criteria)
        if any(not label.strip() for label in self.criteria):
            raise ValueError("native candidate labels must be nonempty")
        return self

    def labels(self):
        return list(self.criteria)

    def descriptions(self):
        return list(self.criteria.values())


class Decision3NativeRequest(StrictModel):
    state: str | dict[str, Any]
    questions: list[Decision3NativeQuestion] = Field(min_length=1, max_length=64)
    media: list = Field(default_factory=list, max_length=0)
    model: str | None = None

    @model_validator(mode="after")
    def unique_finite(self):
        _check_json(self.state)
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate native question IDs")
        return self


def parse_native_payload(payload, *, model_identity):
    """Validate full original native inventory, then every real native255 head.

    _request's representative COPY checks only the shared strict outer schema;
    no representative, truncated candidate set or model_copy reaches inference.
    This explicit DTO normally validates every original candidate again.
    """
    original = copy.deepcopy(payload)
    _, source, labels = _request(original, model_identity)
    if any(len(offered) > MAX_OPTIONS for offered in labels.values()):
        raise RequestValidationError("Decision3 native head exceeds255 candidates; none pruned")
    data = copy.deepcopy(original)
    converted = []
    for key, raw in data["questions"].items():
        entry = copy.deepcopy(raw)
        entry["id"] = key
        for alias in ("options", "levels"):
            if alias in entry:
                entry["criteria"] = entry.pop(alias)
        converted.append(entry)
    data["questions"] = converted
    try:
        parsed = Decision3NativeRequest.model_validate(data)
    except (ValidationError, ValueError) as exc:
        raise RequestValidationError(f"invalid Decision3 native request: {exc}") from exc
    return parsed, original, source, labels


class JevDecision3NativeAdapter:
    def __init__(self, backend, *, model_identity=None, evidence_sink=None):
        spec = getattr(backend, "spec", None)
        if spec is None or getattr(spec, "adapter", None) != "decision3":
            raise ValueError(
                "Decision3 native browser profile requires its explicit pinned provider"
            )
        reasons = configuration_blockers(spec)
        if reasons:
            raise ValueError("invalid Decision3 native browser provider: " + ", ".join(reasons))
        self.backend = backend
        self.identity = backend.name if model_identity is None else model_identity
        if not isinstance(self.identity, str) or not self.identity.strip():
            raise ValueError("configured Decision3 native model alias must be nonempty")
        self.evidence_sink = evidence_sink

    def predict(self, payload):
        started = time.monotonic()
        audit = {
            "profile": profile_manifest(),
            "version": PROFILE,
            "status": "parsing",
            "original_request": _evidence(payload),
            "backend_predict_calls": 0,
            "native_question_rows": 0,
            "native_events": [],
            "raw_response": None,
        }

        def emit(event, data):
            if self.evidence_sink is None:
                return
            try:
                self.evidence_sink({"event": event, "profile": PROFILE, "data": _evidence(data)})
            except BaseException as exc:
                audit.setdefault("evidence_sink_errors", []).append(
                    {"type": type(exc).__name__, "message": str(exc)}
                )

        def deadline():
            if time.monotonic() - started >= 20:
                raise BackendTimeoutError(
                    "Decision3 native cooperative20s budget exceeded; no hard cancel"
                )

        def native_event(event, evidence):
            audit["native_events"].append({"event": event, "evidence": _evidence(evidence)})
            emit(event, evidence)

        try:
            deadline()
            parsed, original, source, labels = parse_native_payload(
                payload, model_identity=self.identity
            )
            audit.update(
                original_request=original,
                original_request_sha256=hashlib.sha256(source).hexdigest(),
                candidate_labels=labels,
                native_question_rows=len(parsed.questions),
                status="preflight",
            )
            emit("planned", audit)
            deadline()
            audit["backend_predict_calls"] += 1
            # Ordinary validation and a full deep copy, never an unchecked wide model_copy.
            request = Decision3NativeRequest.model_validate(copy.deepcopy(parsed.model_dump()))
            raw = self.backend.predict(request, deadline=deadline, evidence_sink=native_event)
            audit["raw_response"] = _evidence(raw)
            emit("backend_raw_returned", raw)
            deadline()
            expected_native_name = self.backend.spec.model_id.rsplit("/", 1)[-1]
            if not isinstance(raw, dict) or raw.get("model") != expected_native_name:
                raise BackendResponseError(
                    "Decision3 native browser response changed checkpoint identity"
                )
            result = normalize_native_response(request, raw)
            answers = [copy.deepcopy(result["answers"][q.id]) for q in request.questions]
            audit.update(status="completed", usage=_evidence(result.get("usage")))
            if "cost" in result:
                audit["cost"] = _evidence(result["cost"])
            deadline()
            emit("completed", audit)
            deadline()
            return {**result, "answers": answers, "model": self.identity, "decision3_native": audit}
        except BaseException as exc:
            audit.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
            if hasattr(exc, "decision3_provider_evidence"):
                audit["provider_failure_evidence"] = _evidence(exc.decision3_provider_evidence)
            exc.decision3_native = audit
            emit("failed", audit)
            raise
        finally:
            audit["elapsed_ms"] = (time.monotonic() - started) * 1000
