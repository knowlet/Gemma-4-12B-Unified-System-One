"""Opt-in wire compatibility for the pinned Jev Ultrafast choice policy.

The ordinary DecisionRequest contract stays strict. Structured browser
instructions and candidate descriptions are serialized without discarding any
fields; singleton target heads are resolved without a model call. Unsupported
candidate counts fail explicitly rather than dropping visible browser targets.
"""

from __future__ import annotations

import json

from pydantic import Field, ValidationError, model_validator

from .backends import normalize_response
from .contracts import MAX_OPTIONS, DecisionRequest, Question
from .errors import BackendResponseError, RequestValidationError

SINGLETON_TARGETS = frozenset({"click_target", "type_text_target", "select_target"})


class _JevQuestion(Question):
    @model_validator(mode="after")
    def valid_criteria(self):
        if (
            self.type == "choice"
            and self.id in SINGLETON_TARGETS
            and isinstance(self.criteria, dict)
            and len(self.criteria) == 1
        ):
            if not next(iter(self.criteria)).strip():
                raise ValueError("criteria must have unique, nonempty labels")
            return self
        return super().valid_criteria()


class _JevRequest(DecisionRequest):
    model: str | None = Field(default=None, min_length=1, max_length=512)
    questions: list[_JevQuestion] = Field(min_length=1, max_length=64)


def _json_text(value, *, field):
    if isinstance(value, str):
        return value
    if not isinstance(value, (dict, list)):
        raise RequestValidationError(f"{field} must be text or a JSON object/array")
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise RequestValidationError(f"{field} must contain finite JSON data") from exc


def _parse_payload(payload, *, model_identity):
    if not isinstance(payload, dict):
        raise RequestValidationError("Jev request must be an object")
    data = dict(payload)
    questions = data.get("questions")
    if isinstance(questions, dict):
        entries = list(questions.items())
    elif isinstance(questions, list):
        entries = [(q.get("id") if isinstance(q, dict) else None, q) for q in questions]
    else:
        raise RequestValidationError("Jev questions must be a named object or a list")
    normalized = []
    for question_id, raw in entries:
        if not isinstance(raw, dict):
            raise RequestValidationError("each Jev question must be an object")
        question = dict(raw)
        if "id" in question and question["id"] != question_id:
            raise RequestValidationError("question id disagrees with its dictionary key")
        question["id"] = question_id
        if "instructions" in question:
            question["instructions"] = _json_text(
                question["instructions"], field=f"question {question_id!r} instructions"
            )
        # Keep aliases subject to Question's existing ambiguity checks.
        for key in ("criteria", "options", "levels"):
            criteria = question.get(key)
            if isinstance(criteria, dict):
                if len(criteria) > MAX_OPTIONS:
                    raise RequestValidationError(
                        f"Jev question {question_id!r} has {len(criteria)} candidates; "
                        f"configured S1 supports at most {MAX_OPTIONS}. No targets were truncated."
                    )
                question[key] = {
                    label: _json_text(
                        description, field=f"question {question_id!r} candidate {label!r}"
                    )
                    for label, description in criteria.items()
                }
        normalized.append(question)
    data["questions"] = normalized
    try:
        parsed = _JevRequest.model_validate(data)
    except ValidationError as exc:
        raise RequestValidationError(f"invalid Jev request: {exc}") from exc
    if parsed.model is not None and parsed.model != model_identity:
        raise RequestValidationError(
            f"requested model {parsed.model!r} differs from configured model {model_identity!r}; "
            "this endpoint does not switch models"
        )
    return parsed


def _singleton_answer(question):
    label = question.labels()[0]
    return {
        "id": question.id,
        "type": "choice",
        "choice": label,
        "probabilities": {label: 1.0},
        "confidence": 1.0,
        "margin": 1.0,
    }


class JevCompatibilityAdapter:
    """Wrap one configured backend; model_identity is an explicit wire alias.

    Use ``predict(raw_json)`` at an opt-in HTTP boundary under the backend's
    existing inference lock. The result is already normalized, including any
    singleton heads that are deliberately absent from the model input.
    """

    def __init__(self, backend, *, model_identity=None):
        self.backend = backend
        self.model_identity = backend.name if model_identity is None else model_identity
        if not isinstance(self.model_identity, str) or not self.model_identity.strip():
            raise ValueError("configured Jev model identity must be nonempty text")

    def predict(self, payload):
        parsed = _parse_payload(payload, model_identity=self.model_identity)
        singletons = {q.id: _singleton_answer(q) for q in parsed.questions if len(q.labels()) == 1}
        model_questions = [q for q in parsed.questions if q.id not in singletons]
        if model_questions:
            request = DecisionRequest(
                state=parsed.state,
                questions=[Question.model_validate(q.model_dump()) for q in model_questions],
                media=parsed.media,
            )
            result = normalize_response(request, self.backend.predict(request))
        else:
            result = {"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}
        model = result.get("model", self.backend.name)
        if not isinstance(model, str) or not model.strip():
            raise BackendResponseError("backend model identity must be nonempty text")
        answers = {**result["answers"], **singletons}
        return {
            **result,
            "model": model,
            "answers": {q.id: answers[q.id] for q in parsed.questions},
            "jev_compatibility": {
                "model_identity": self.model_identity,
                "singleton_heads": list(singletons),
            },
        }
