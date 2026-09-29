"""Validated public contracts shared by inference, HTTP and evaluation."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_OPTIONS = 52


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ImageInput(StrictModel):
    type: Literal["image"] = "image"
    data: str = Field(min_length=1, max_length=16_000_000, description="Base64 image bytes")
    timestamp_seconds: float | None = Field(default=None, ge=0)


class AudioInput(StrictModel):
    type: Literal["audio"] = "audio"
    samples: list[float] = Field(min_length=1, max_length=480_000)
    sampling_rate: Literal[16000] = 16000

    @model_validator(mode="after")
    def valid_waveform(self):
        if any(abs(x) > 1 for x in self.samples):
            raise ValueError("audio samples must be normalized to [-1, 1]")
        return self


Media = Annotated[ImageInput | AudioInput, Field(discriminator="type")]


class Question(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    type: Literal["choice", "noul", "score"] = "choice"
    instructions: str = Field(min_length=1, max_length=16000)
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def aliases(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            for alias in ("options", "levels"):
                if alias in value:
                    if "criteria" in value:
                        raise ValueError("use criteria or options/levels, not both")
                    value["criteria"] = value.pop(alias)
        return value

    @model_validator(mode="after")
    def valid_criteria(self):
        if self.type == "noul":
            if self.criteria is not None and (
                not isinstance(self.criteria, dict) or set(self.criteria) != {"false", "true"}
            ):
                raise ValueError("noul criteria must contain exactly false and true")
        else:
            if not self.criteria or not 2 <= len(self.criteria) <= MAX_OPTIONS:
                raise ValueError(f"{self.type} needs 2..{MAX_OPTIONS} criteria")
            keys = list(self.criteria)
            if any(not k.strip() for k in keys) or len(set(keys)) != len(keys):
                raise ValueError("criteria must have unique, nonempty labels")
            if self.type == "score":
                # Also rejects nonfinite numeric labels before computing expectations.
                self.score_values()
        return self

    def labels(self) -> list[str]:
        if self.type == "noul":
            return ["false", "true"]
        if self.type == "score" and isinstance(self.criteria, list):
            return [str(i) for i in range(len(self.criteria))]
        return list(self.criteria or [])

    def descriptions(self) -> list[str]:
        if self.type == "noul":
            return [self.criteria[k] if self.criteria else k for k in ("false", "true")]
        return (
            list(self.criteria.values()) if isinstance(self.criteria, dict) else list(self.criteria)
        )

    def score_values(self) -> list[float]:
        labels = self.labels()
        try:
            values = [float(k) for k in labels]
        except ValueError:
            return [float(i) for i in range(len(labels))]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("score levels must be finite")
        if any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError("numeric score levels must be strictly increasing")
        return values


class DecisionRequest(StrictModel):
    state: str | dict[str, Any]
    questions: list[Question] = Field(min_length=1, max_length=64)
    media: list[Media] = Field(default_factory=list, max_length=8)

    @model_validator(mode="before")
    @classmethod
    def named_questions(cls, value):
        if isinstance(value, dict) and isinstance(value.get("questions"), dict):
            value = dict(value)
            questions = []
            for name, q in value["questions"].items():
                if not isinstance(q, dict):
                    raise ValueError("each named question must be an object")
                if "id" in q and q["id"] != name:
                    raise ValueError("question id disagrees with its dictionary key")
                questions.append({**q, "id": name})
            value["questions"] = questions
        return value

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [q.id for q in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError("question ids must be unique")
        return self

    def named_questions_payload(self) -> dict:
        return {q.id: q.model_dump(exclude={"id"}, exclude_none=True) for q in self.questions}


def validate_distribution(values, size: int) -> list[float]:
    result = [float(x) for x in values]
    if len(result) != size or not all(math.isfinite(x) and 0 <= x <= 1 for x in result):
        raise ValueError("invalid probability distribution")
    if not math.isclose(sum(result), 1.0, abs_tol=1e-5):
        raise ValueError("probabilities must sum to one")
    return result


def answer_from_probabilities(question: Question, values) -> dict:
    labels = question.labels()
    p = validate_distribution(values, len(labels))
    best = max(range(len(p)), key=p.__getitem__)
    ordered = sorted(p, reverse=True)
    answer = {
        "id": question.id,
        "type": question.type,
        "probabilities": dict(zip(labels, p)),
        "confidence": p[best],
        "margin": ordered[0] - ordered[1],
    }
    if question.type == "choice":
        answer["choice"] = labels[best]
    elif question.type == "noul":
        answer["noul"] = p[1]
    else:
        answer.update(
            level=labels[best], score=sum(v * x for v, x in zip(question.score_values(), p))
        )
    return answer
