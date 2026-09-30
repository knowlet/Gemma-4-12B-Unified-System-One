"""Evaluation-only metadata stays outside the inference payload."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from s1.benchmark import Case

METADATA_FIELDS = (
    "group_id",
    "task_id",
    "language",
    "schema_id",
    "soft_gold",
    "critical_questions",
)


class EvaluationCase(Case):
    split: Literal["train", "development", "calibration", "test"] = "test"
    group_id: str | None = Field(default=None, min_length=1)
    task_id: str | None = Field(default=None, min_length=1)
    language: str | None = Field(default=None, min_length=1)
    schema_id: str | None = Field(default=None, min_length=1)
    soft_gold: dict[str, dict[str, float]] | None = None
    critical_questions: list[str] | None = None

    @model_validator(mode="after")
    def validate_annotations(self):
        from s1.contracts import validate_distribution

        questions = {q.id: q for q in self.request.questions}
        if self.critical_questions is not None:
            if (
                len(set(self.critical_questions)) != len(self.critical_questions)
                or set(self.critical_questions) - questions.keys()
            ):
                raise ValueError("critical_questions must contain unique request question ids")
        if self.soft_gold is not None:
            if set(self.soft_gold) - questions.keys():
                raise ValueError("soft_gold has unknown question ids")
            for key, target in self.soft_gold.items():
                labels = questions[key].labels()
                if set(target) != set(labels):
                    raise ValueError("soft_gold must cover all candidate labels")
                validate_distribution([target[label] for label in labels], len(labels))
        return self

    def metadata(self) -> dict:
        return {
            "case_id": self.id,
            "group_id": self.group_id or self.id,
            "task_id": self.task_id,
            "language": self.language,
            "schema_id": self.schema_id,
            "split": self.split,
        }


def load_cases(path) -> list[EvaluationCase]:
    cases = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    cases.append(EvaluationCase.model_validate_json(line))
                except ValueError:
                    # Validation messages can contain prompt text or media contents.
                    raise ValueError(f"invalid evaluation case at line {number}") from None
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("dataset must be nonempty and have unique case ids")
    return cases


def fingerprint(cases) -> str:
    # Preserve v1 fingerprints when none of the optional metadata was supplied.
    rows = [
        case.model_dump_json(
            exclude={name for name in METADATA_FIELDS if getattr(case, name, None) is None}
        )
        for case in cases
    ]
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def audit_splits(paths) -> dict:
    """Reject source-group, ID, or identical-request leakage across splits."""
    seen = {"id": {}, "group": {}, "request": {}}
    counts, overlaps = {}, []
    for path in paths:
        for case in load_cases(path):
            counts[case.split] = counts.get(case.split, 0) + 1
            request_key = hashlib.sha256(
                json.dumps(
                    case.request.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            for kind, key in (
                ("id", case.id),
                ("group", case.group_id or case.id),
                ("request", request_key),
            ):
                previous = seen[kind].setdefault(key, set())
                if previous - {case.split}:
                    overlaps.append(
                        {
                            "kind": kind,
                            "splits": sorted(previous | {case.split}),
                            "key_sha256": hashlib.sha256(key.encode()).hexdigest(),
                        }
                    )
                previous.add(case.split)
    return {"valid": not overlaps, "split_counts": counts, "overlaps": overlaps}


def request_fingerprint(request):
    return hashlib.sha256(
        json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
