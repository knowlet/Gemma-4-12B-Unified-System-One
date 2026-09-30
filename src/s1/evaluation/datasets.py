"""Evaluation-only metadata stays outside the inference payload."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field

from s1.benchmark import Case

METADATA_FIELDS = ("group_id", "task_id", "language", "schema_id")


class EvaluationCase(Case):
    split: Literal["train", "development", "calibration", "test"] = "test"
    group_id: str | None = Field(default=None, min_length=1)
    task_id: str | None = Field(default=None, min_length=1)
    language: str | None = Field(default=None, min_length=1)
    schema_id: str | None = Field(default=None, min_length=1)

    def metadata(self) -> dict:
        return {
            "case_id": self.id,
            "group_id": self.group_id or self.id,
            "task_id": self.task_id,
            "language": self.language,
            "schema_id": self.schema_id,
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
