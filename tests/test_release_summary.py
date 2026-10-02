"""Fail-closed alignment and provenance checks for release summaries."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "summarize_release", ROOT / "scripts/summarize_release.py"
)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def cases(split="test", count=3):
    from s1.evaluation.datasets import EvaluationCase

    return [
        EvaluationCase.model_validate(
            {
                "id": f"{split}-{i}",
                "group_id": f"group-{split}-{i}",
                "task_id": "boolq",
                "split": split,
                "request": {
                    "state": f"independent state {split} {i}",
                    "questions": {
                        "answer": {"type": "noul", "instructions": "Is the assertion correct?"}
                    },
                },
                "gold": {"answer": bool(i % 2)},
            }
        )
        for i in range(count)
    ]


def cuda_report(population):
    from s1.benchmark import fingerprint
    from s1.evaluation.datasets import fingerprint as release_fingerprint

    manifest = {"identity": "run", "recipe": {"model": "base", "revision": "a" * 40}}
    identity = {"dataset_sha256": release_fingerprint(population)}
    records = []
    for value in summary.decision_index(population).values():
        records.append(
            {
                "case_id": value["case_id"],
                "question_id": value["question_id"],
                "type": value["type"],
                "gold": value["gold"],
                "status": "ok",
                "probabilities": {
                    label: (0.8 if label == value["gold"] else 0.2) for label in value["labels"]
                },
                "latency_ms": 10.0,
            }
        )
    return (
        {
            "backend": "base",
            "coverage": 1.0,
            "counts": {"ok": len(records)},
            "case_ids": [case.id for case in population],
            "dataset_sha256": fingerprint(population),
            "warmup_errors": [],
            "metadata": {
                "identity": "run",
                "base_model": "base",
                "base_revision": "a" * 40,
                "temperature": 1.0,
                "release_dataset_sha256": identity["dataset_sha256"],
            },
            "records": records,
        },
        identity,
        manifest,
    )


def test_cuda_complete_population_recomputes_metrics():
    population = cases()
    report, identity, manifest = cuda_report(population)
    rows = summary.validate_cuda(report, population, identity, manifest, "base", 1.0)
    result = summary.summarize_rows(rows, "cuda")
    assert result["coverage"] == 1 and result["decisions"] == 3
    assert result["metrics"]["accuracy"] == 1
    assert result["metrics"]["nll"] == pytest.approx(-np.log(0.8))
    assert result["metrics"]["brier"] == pytest.approx(0.08)
    assert result["metrics"]["ece_15"] == pytest.approx(0.2)
    assert result["latency_ms"]["scope"] == "CUDA end-to-end backend request"


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "gold", "labels", "status"])
def test_cuda_rejects_incomplete_or_misaligned_records_despite_full_summary(corruption):
    population = cases()
    report, identity, manifest = cuda_report(population)
    if corruption == "missing":
        report["records"].pop()
    elif corruption == "duplicate":
        report["records"].append(copy.deepcopy(report["records"][0]))
    elif corruption == "gold":
        report["records"][0]["gold"] = "true"
    elif corruption == "labels":
        report["records"][0]["probabilities"] = dict(
            reversed(list(report["records"][0]["probabilities"].items()))
        )
    else:
        report["records"][0]["status"] = "error"
    with pytest.raises(ValueError):
        summary.validate_cuda(report, population, identity, manifest, "base", 1.0)


def test_paired_delta_direction_determinism_and_no_shared_subset():
    population = cases(count=4)
    report, identity, manifest = cuda_report(population)
    right = summary.validate_cuda(report, population, identity, manifest, "base", 1.0)
    left = copy.deepcopy(right)
    left[0]["probabilities"].reverse()
    result = summary.paired_accuracy(left, right, seed=7, resamples=200)
    assert result["delta"] == 0.25
    assert result["decisions"] == result["source_groups"] == 4
    assert result == summary.paired_accuracy(left, right, seed=7, resamples=200)
    with pytest.raises(ValueError, match="complete population"):
        summary.paired_accuracy(left, right[:-1])
    right[0]["gold"] = "true"
    with pytest.raises(ValueError, match="golds"):
        summary.paired_accuracy(left, right)


def mlx_report(population, role="test"):
    from s1.evaluation.datasets import request_fingerprint

    records = []
    for case in population:
        answers = {}
        for question in case.request.questions:
            labels = question.labels()
            probabilities = dict(zip(labels, [0.25, 0.75]))
            answers[question.id] = {
                "type": question.type,
                "labels": labels,
                "gold": case.gold[question.id],
                "raw_logits": [float(np.log(0.25)), float(np.log(0.75))],
                "probabilities": probabilities,
                "prediction": labels[1],
            }
        records.append(
            {
                "id": case.id,
                "split": case.split,
                "status": "ok",
                "group_id": case.group_id,
                "request_sha256": request_fingerprint(case.request),
                "answers": answers,
                "prepared_forward_samples_ms": [20.0, 22.0],
            }
        )
    return {
        "calibration": {"temperature": 1.0},
        "measurement": {"repeat_per_case": 2},
        "cases": {role: records},
        "summary": {
            role: {
                "expected_cases": len(population),
                "successful_cases": len(population),
                "case_coverage": 1,
                "expected_decisions": len(population),
                "scored_decisions": len(population),
                "decision_coverage": 1,
            }
        },
    }


def test_mlx_complete_records_preserve_prepared_latency_scope():
    population = cases()
    report = mlx_report(population)
    rows = summary.validate_mlx_split(report, "test", population)
    result = summary.summarize_rows(rows, "mlx")
    assert result["latency_ms"]["samples"] == 6
    assert result["latency_ms"]["requests"] == 3
    assert "prepared tensors" in result["latency_ms"]["scope"]
    assert result["latency_ms"]["p50"] == 21.0


@pytest.mark.parametrize("corruption", ["missing", "group", "question", "temperature", "samples"])
def test_mlx_rejects_partial_or_misaligned_population(corruption):
    population = cases()
    report = mlx_report(population)
    record = report["cases"]["test"][0]
    if corruption == "missing":
        report["cases"]["test"].pop()
    elif corruption == "group":
        record["group_id"] = "different-source"
    elif corruption == "question":
        record["answers"]["wrong"] = record["answers"].pop("answer")
    elif corruption == "temperature":
        report["calibration"]["temperature"] = 2.0
    else:
        record["prepared_forward_samples_ms"].pop()
    with pytest.raises(ValueError):
        summary.validate_mlx_split(report, "test", population)


def test_mlx_source_receipt_requires_all_config_index_and_weight_shards():
    files = [
        {"path": name, "size_bytes": 10, "sha256": character * 64}
        for name, character in (
            ("config.json", "a"),
            ("model-00001-of-00001.safetensors", "b"),
            ("model.safetensors.index.json", "c"),
        )
    ]
    manifest = {"recipe": {"model": "base", "revision": "d" * 40}}
    artifacts = {"merged": {row["path"]: row["sha256"] for row in files}}
    artifacts["merged"]["s1_config.json"] = "e" * 64
    report = {
        "status": "ok",
        "release_validation_complete": True,
        "source_model": "base",
        "source_revision": "d" * 40,
        "prompt_version": 1,
        "dataset_manifest_sha256": "f" * 64,
        "source_s1_config_sha256": "e" * 64,
        "source_checkpoint_files": files,
        "source_checkpoint_sha256": summary.digest(files),
        "measurement": {"not_end_to_end_request_latency": True},
    }
    summary.validate_mlx_provenance(report, manifest, artifacts, "f" * 64)
    bad = copy.deepcopy(report)
    bad["source_checkpoint_files"].pop()
    bad["source_checkpoint_sha256"] = summary.digest(bad["source_checkpoint_files"])
    with pytest.raises(ValueError, match="inventory"):
        summary.validate_mlx_provenance(bad, manifest, artifacts, "f" * 64)
    bad = copy.deepcopy(report)
    bad["source_checkpoint_files"][1]["sha256"] = "0" * 64
    bad["source_checkpoint_sha256"] = summary.digest(bad["source_checkpoint_files"])
    with pytest.raises(ValueError, match="artifact receipt"):
        summary.validate_mlx_provenance(bad, manifest, artifacts, "f" * 64)


def test_calibration_requires_exact_frozen_calibration_labels():
    from s1.training import fit_temperature

    population = cases("calibration")
    records = []
    for truth in summary.decision_index(population).values():
        records.append(
            {
                "case_id": truth["case_id"],
                "question_id": truth["question_id"],
                "labels": truth["labels"],
                "gold_index": truth["gold_index"],
                "logits": [0.0, 1.0],
            }
        )
    fit = fit_temperature([r["logits"] for r in records], [r["gold_index"] for r in records])
    report = {**fit, "records": records}
    assert summary.validate_calibration(report, population) == fit
    report["records"][0]["gold_index"] = 1
    with pytest.raises(ValueError, match="calibration split"):
        summary.validate_calibration(report, population)
