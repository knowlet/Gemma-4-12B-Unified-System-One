"""Recovery summaries must demonstrate actual held-out adapter training."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recovery_collector", ROOT / "scripts/collect_comparison.py"
)
COLLECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECT)


def proof_fixture(tmp_path):
    model = "gemma-ce128-s0-nf4-recovered"
    row = {
        "model_id": model,
        "checkpoint": "gemma",
        "revision": "f" * 40,
        "adapter_sha256": "b" * 64,
        "recovery": {"method": "fixed_nf4_qlora_continuation", "seed": 0},
        "artifacts": {"training": "training-receipt.json"},
    }
    proof = {
        "status": "completed",
        "actual_optimizer_updates": 100,
        "adapter_sha256": row["adapter_sha256"],
        "recovery": copy.deepcopy(row["recovery"]),
        "base_model": row["checkpoint"],
        "base_revision": row["revision"],
        "observed_training_ids": [f"train-{i}" for i in range(100)],
        "observed_training_request_sha256": [f"train-hash-{i}" for i in range(100)],
        "calibration_reference": [{"request_sha256": f"calibration-{i}"} for i in range(16)],
        "frozen_parameter_sample_sha256_before": "frozen-digest",
        "frozen_parameter_sample_sha256_after": "frozen-digest",
        "lora_sha256_before": "original-digest",
        "lora_sha256_after": "updated-digest",
    }
    artifact = tmp_path / model / "training-receipt.json"
    artifact.parent.mkdir()
    records = {i: {"request_sha256": f"test-hash-{i}"} for i in range(128)}
    return {"source": tmp_path / "campaign.json", "row": row}, proof, artifact, records


def write_proof(attempt, proof, artifact):
    artifact.write_text(json.dumps(proof))
    attempt["row"]["artifact_sha256"] = {
        "training-receipt.json": hashlib.sha256(artifact.read_bytes()).hexdigest()
    }


def test_observed_recovery_proof_preserves_actual_updates_and_hash(tmp_path):
    attempt, proof, artifact, records = proof_fixture(tmp_path)
    write_proof(attempt, proof, artifact)
    result = COLLECT.verify_recovery_training_evidence(attempt, records, tmp_path / "summary.json")
    assert result["actual_optimizer_updates"] == 100
    assert result["observed_distinct_training_cases"] == 100
    assert result["held_out_request_overlap"] is False
    assert (
        result["training_receipt_sha256"]
        == attempt["row"]["artifact_sha256"]["training-receipt.json"]
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "updates",
        "adapter",
        "duplicates",
        "leak",
        "calibration_leak",
        "frozen",
        "no_training",
        "hash",
    ],
)
def test_planned_recipe_does_not_replace_actual_training_evidence(tmp_path, tamper):
    attempt, proof, artifact, records = proof_fixture(tmp_path)
    if tamper == "updates":
        proof["actual_optimizer_updates"] = 0
    elif tamper == "adapter":
        proof["adapter_sha256"] = "a-different-adapter"
    elif tamper == "duplicates":
        proof["observed_training_ids"][1] = proof["observed_training_ids"][0]
    elif tamper == "leak":
        proof["observed_training_request_sha256"][0] = "test-hash-3"
    elif tamper == "calibration_leak":
        proof["calibration_reference"][0]["request_sha256"] = "test-hash-3"
    elif tamper == "frozen":
        proof["frozen_parameter_sample_sha256_after"] = "changed"
    elif tamper == "no_training":
        proof["lora_sha256_after"] = proof["lora_sha256_before"]
    write_proof(attempt, proof, artifact)
    if tamper == "hash":
        artifact.write_text(artifact.read_text() + "\n")
    with pytest.raises(ValueError):
        COLLECT.verify_recovery_training_evidence(attempt, records, tmp_path / "summary.json")
