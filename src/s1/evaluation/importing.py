"""Local public-data conversion with immutable source provenance and no gold-based sampling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .artifacts import write_json
from .datasets import EvaluationCase, fingerprint
from .experiments import write_cases


def import_dataset(source, output, *, format, source_url, revision, license, original_split, split):
    if not all((source_url, revision, license, original_split)):
        raise ValueError("source URL, immutable revision, license and original split are required")
    raw = Path(source).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    cases = []
    for index, row in enumerate(rows):
        if format == "boolq":
            if type(row.get("answer")) is not bool:
                raise ValueError("BoolQ conversion requires published boolean labels")
            state = row["passage"]
            question = {"id": "answer", "type": "noul", "instructions": row["question"]}
            gold = "true" if row["answer"] else "false"
            language = "en"
        elif format == "ocnli":
            if row.get("label") not in ("entailment", "neutral", "contradiction"):
                raise ValueError(
                    "OCNLI requires released labels; unlabeled official test cannot be scored"
                )
            state = {"premise": row["sentence1"], "hypothesis": row["sentence2"]}
            question = {
                "id": "answer",
                "type": "choice",
                "instructions": "判斷前提與假設的關係。",
                "criteria": {"entailment": "蘊含", "neutral": "中立", "contradiction": "矛盾"},
            }
            gold, language = row["label"], "zh"
        elif format == "choice":
            # Full supplied ontology only. Never choose negatives using the gold.
            state = row["state"]
            question = {
                "id": "answer",
                "type": "choice",
                "instructions": row["question"],
                "criteria": row["options"],
            }
            gold, language = row["answer"], row.get("language")
        else:
            raise ValueError("unknown dataset format")
        group = (
            row.get("group_id")
            or hashlib.sha256(
                json.dumps(state, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
        )
        cases.append(
            EvaluationCase.model_validate(
                {
                    "id": f"{format}-{original_split}-{row.get('id', index)}",
                    "split": split,
                    "group_id": str(group),
                    "task_id": f"{format}-{original_split}",
                    "language": language,
                    "schema_id": format,
                    "request": {"state": state, "questions": [question]},
                    "gold": {"answer": gold},
                }
            )
        )
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError("empty input or duplicate source IDs")
    output = Path(output)
    provenance = output.with_suffix(".provenance.json")
    if provenance.exists() or output.exists():
        raise FileExistsError("dataset or provenance output exists")
    manifest = {
        "schema_version": 2,
        "source_url": source_url,
        "source_revision": revision,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "license": license,
        "original_split": original_split,
        "evaluation_split": split,
        "converter": format,
        "dataset_sha256": fingerprint(cases),
        "cases": len(cases),
        "candidate_policy": "full_supplied_ontology; no_gold_selected_distractors",
        "label_availability_verified_by": "input_schema; provenance is operator-supplied",
    }
    write_cases(output, cases)
    write_json(provenance, manifest)
    return manifest
