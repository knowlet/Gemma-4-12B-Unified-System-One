"""One JSONL dataset, one contract, per-decision records and honest coverage."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from .backends import UnsupportedRequest, normalize_response
from .contracts import DecisionRequest, StrictModel
from .metrics import summarize


class Case(StrictModel):
    id: str = Field(min_length=1)
    split: Literal["train", "calibration", "test"] = "test"
    request: DecisionRequest
    gold: dict[str, str | bool]

    @model_validator(mode="after")
    def validate_gold(self):
        if set(self.gold) != {q.id for q in self.request.questions}:
            raise ValueError("gold must cover exactly the request question ids")
        for q in self.request.questions:
            label = self.gold[q.id]
            if q.type == "noul" and isinstance(label, bool):
                label = "true" if label else "false"
                self.gold[q.id] = label
            if label not in q.labels():
                raise ValueError(f"gold for {q.id} must be a valid label")
        return self


def load_cases(path) -> list[Case]:
    cases = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    cases.append(Case.model_validate_json(line))
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError("dataset must be nonempty and have unique case ids")
    return cases


def fingerprint(cases):
    serialized = "\n".join(c.model_dump_json() for c in cases)
    return hashlib.sha256(serialized.encode()).hexdigest()


def metrics_for(records):
    valid = [r for r in records if r["status"] == "ok"]
    if not valid:
        return {"n": 0}
    width = max(len(r["probabilities"]) for r in valid)
    probabilities = np.zeros((len(valid), width))
    golds, counts = [], []
    for i, r in enumerate(valid):
        keys = list(r["probabilities"])
        probabilities[i, : len(keys)] = list(r["probabilities"].values())
        golds.append(keys.index(r["gold"]))
        counts.append(len(keys))
    metrics = summarize(probabilities, golds, counts)
    score_errors = [abs(r["score"] - r["gold_score"]) for r in valid if r["type"] == "score"]
    if score_errors:
        metrics["score_mae"] = float(np.mean(score_errors))
    return metrics


def evaluate(backend, cases, *, warmup=0):
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if not cases or any(c.split != "test" for c in cases):
        raise ValueError("evaluation requires a nonempty, test-only split")
    warmup_errors = []
    for _ in range(warmup):
        try:
            backend.predict(cases[0].request)
        except Exception as exc:
            warmup_errors.append(type(exc).__name__)
    records, latencies = [], []
    for case in cases:
        started = time.perf_counter()
        try:
            response = normalize_response(case.request, backend.predict(case.request))
        except Exception as exc:
            status = "unsupported" if isinstance(exc, UnsupportedRequest) else "error"
            for q in case.request.questions:
                records.append(
                    {
                        "case_id": case.id,
                        "question_id": q.id,
                        "type": q.type,
                        "status": status,
                        "error_type": type(exc).__name__,
                    }
                )
            continue
        latency = (time.perf_counter() - started) * 1000
        latencies.append(latency)
        for q in case.request.questions:
            answer = response["answers"][q.id]
            record = {
                "case_id": case.id,
                "question_id": q.id,
                "type": q.type,
                "status": "ok",
                "gold": case.gold[q.id],
                "latency_ms": latency,
                "probabilities": answer["probabilities"],
            }
            if q.type == "score":
                record.update(
                    score=answer["score"],
                    gold_score=q.score_values()[q.labels().index(case.gold[q.id])],
                )
            records.append(record)
    counts = dict(Counter(r["status"] for r in records))
    report = {
        "schema_version": 1,
        "backend": backend.name,
        "metadata": getattr(backend, "metadata", {}),
        "dataset_sha256": fingerprint(cases),
        "case_ids": [c.id for c in cases],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "warmup": warmup,
        "warmup_errors": warmup_errors,
        "counts": counts,
        "coverage": counts.get("ok", 0) / len(records),
        "metrics": metrics_for(records),
        "by_type": {
            kind: metrics_for([r for r in records if r["type"] == kind])
            for kind in ("choice", "noul", "score")
        },
        "latency_ms": {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "requests": len(latencies),
        }
        if latencies
        else None,
        "records": records,
    }
    return report


def compare_reports(reports):
    """Recompute scores on the exact successful decision intersection."""
    if len(reports) < 2 or len({r["dataset_sha256"] for r in reports}) != 1:
        raise ValueError("comparison requires at least two reports for the same dataset")
    if len({r["backend"] for r in reports}) != len(reports):
        raise ValueError("comparison backend names must be unique")
    keys = [
        set((r["case_id"], r["question_id"]) for r in report["records"] if r["status"] == "ok")
        for report in reports
    ]
    common = set.intersection(*keys)
    return {
        "dataset_sha256": reports[0]["dataset_sha256"],
        "common_decisions": len(common),
        "metrics": {
            report["backend"]: metrics_for(
                [r for r in report["records"] if (r["case_id"], r["question_id"]) in common]
            )
            for report in reports
        },
    }


def write_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
