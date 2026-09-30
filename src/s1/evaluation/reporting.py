"""Recomputable E1 counts and descriptive quality; inferential statistics follow in E2."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .artifacts import read_records, write_json


def summarize_run(directory) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    requests = read_records(directory, "requests")
    predictions = read_records(directory, "predictions")
    errors = read_records(directory, "errors")
    models = {}
    for cell in manifest["plan"]["models"]:
        model_id = cell["model_id"]
        rows = [row for row in predictions if row["model_id"] == model_id]
        calls = [row for row in requests if row["model_id"] == model_id]
        measured = [row for row in calls if row["phase"] == "measurement"]
        valid = [row for row in rows if row["status"] == "ok"]
        repetitions = cell["identity"]["profile"]["repetitions"]
        expected = sum(cell["coverage"]["decisions"].values()) * repetitions
        eligible = cell["coverage"]["decisions"]["eligible"] * repetitions
        attempted = sum(row["dispatched_at_s"] is not None for row in measured)
        actual_correct = sum(row["actual_label"] == row["gold"] for row in valid)
        standardized_correct = sum(row["standardized_argmax"] == row["gold"] for row in valid)
        latencies = [row["latency_ms"] for row in measured if row["status"] == "ok"]
        actual_choices = [row for row in valid if row["type"] == "choice"]
        models[model_id] = {
            **manifest["execution"][model_id],
            "request_counts": dict(Counter(row["status"] for row in measured)),
            "decision_counts": dict(Counter(row["status"] for row in rows)),
            "denominators": {
                "total_decisions": expected,
                "recorded_decisions": len(rows),
                "eligible_decisions": eligible,
                "valid_decisions": len(valid),
                "attempted_measured_requests": attempted,
                "source_groups": len({row["group_id"] for row in rows}),
            },
            "records_complete": len(rows) == expected,
            "warmup_counts": dict(
                Counter(row["status"] for row in calls if row["phase"] == "warmup")
            ),
            "execution_coverage": len(valid) / eligible if attempted and eligible else None,
            "quality": {
                "actual_label_accuracy": actual_correct / len(valid) if valid else None,
                "actual_choice_accuracy": sum(
                    row["actual_label"] == row["gold"] for row in actual_choices
                )
                / len(actual_choices)
                if actual_choices
                else None,
                "standardized_argmax_accuracy": standardized_correct / len(valid)
                if valid
                else None,
                "operational_correctness": actual_correct / eligible
                if attempted and eligible
                else None,
            },
            "successful_request_latency_ms": {
                "requests": len(latencies),
                "p50": float(np.percentile(latencies, 50)),
                "p95": float(np.percentile(latencies, 95)),
            }
            if latencies
            else None,
            "error_count": sum(row["model_id"] == model_id for row in errors),
        }
    return {
        "schema_version": 2,
        "plan_id": manifest["plan"]["plan_id"],
        "run_status": manifest["status"],
        "models": models,
        "limitations": [
            "Descriptive metrics only; repeated decisions are not independent samples.",
            "Warmup is excluded from quality and latency percentiles, but consumes request budget.",
            "Noul/Score labels use ordered distribution argmax; Score expectation is saved separately.",
            "Failed/timeout requests have no successful latency; see counts and termination times.",
        ],
    }


def write_summary(directory):
    result = summarize_run(directory)
    write_json(Path(directory) / "summary.json", result)
    return result
