"""Retry provenance and exact-input pairing for live comparison collection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "collect_comparison", ROOT / "scripts/collect_comparison.py"
)
COLLECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECT)


def campaign(tmp_path, name, day, models, *, media_hash=None):
    directory = tmp_path / name
    directory.mkdir()
    runs = []
    for model, status, correct in models:
        row = {
            "model_id": model,
            "checkpoint": "pinned-checkpoint",
            "revision": "f" * 40,
            "precision": "bfloat16",
            "quantization": "nf4" if "-nf4" in model else "int8" if "-int8" in model else "none",
            "adapter_sha256": "a" * 64 if model.startswith("gemma-ce128") else None,
            "status": status,
            "gpu": "NVIDIA A100 80GB PCIe",
            "started_at_unix": 1_790_000_000 + day,
            "boolq": None
            if correct is None
            else {
                "expected": 128,
                "recorded": 128,
                "valid": 128,
                "correct": correct,
                "accuracy": correct / 128,
                "dataset_sha256": COLLECT.DATASET_PINS["text-test.jsonl"][1],
            },
            "media": None,
            "load": [],
            "artifacts": {"predictions": "boolq/predictions.jsonl"},
        }
        runs.append(row)
        if correct is not None:
            predictions = directory / model / "boolq"
            predictions.mkdir(parents=True)
            cases, records = [], []
            for i in range(128):
                cases.append(
                    {
                        "case_id": f"case-{i}",
                        "group_id": f"group-{i // 2}",
                        "decisions": 1,
                        "request_sha256": hashlib.sha256(f"prompt-{i}".encode()).hexdigest(),
                    }
                )
                records.append(
                    {
                        "model_id": model,
                        "case_id": f"case-{i}",
                        "question_id": "answer",
                        "repetition": 0,
                        "group_id": f"group-{i // 2}",
                        "task_id": "boolq",
                        "schema_id": "boolq",
                        "type": "noul",
                        "gold": "true",
                        "actual_label": "true" if i < correct else "false",
                        "status": "ok",
                        "eligibility": "eligible",
                    }
                )
            (predictions / "predictions.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records)
            )
            manifest = {
                "status": "completed",
                "plan": {
                    "dataset_sha256": COLLECT.DATASET_PINS["text-test.jsonl"][1],
                    "models": [
                        {
                            "model_id": model,
                            "cases": cases,
                            "identity": {
                                "model": {
                                    "id": model,
                                    "model_id": row["checkpoint"],
                                    "revision": row["revision"],
                                    "precision": row["precision"],
                                    "quantization": row["quantization"],
                                    "decision_adapter_sha256": row["adapter_sha256"],
                                }
                            },
                        }
                    ],
                },
                "execution": {
                    model: {
                        "status": "completed",
                        "telemetry": {
                            key: row[key] for key in ("revision", "precision", "quantization")
                        },
                    }
                },
            }
            (predictions / "run_manifest.json").write_text(json.dumps(manifest))
            row["artifact_sha256"] = {
                f"boolq/{name}": COLLECT.digest(predictions / name)
                for name in ("predictions.jsonl", "run_manifest.json")
            }
            (directory / model / "receipt.json").write_text(json.dumps(row))
    path = directory / "campaign.json"
    path.write_text(
        json.dumps(
            {
                "campaign_id": name,
                "generated_at": "2026-10-01T08:00:00+08:00",
                "status": "completed",
                "policy": {"gpu": "A100-80GB"},
                "runs": runs,
                "dataset": {
                    "sha256": COLLECT.DATASET_PINS["text-test.jsonl"][1],
                    "cases": 128,
                    "media_sha256": media_hash or COLLECT.DATASET_PINS["media-test.jsonl"][1],
                },
            }
        )
    )
    return path


def update_consistent_identity(path, index, key, value):
    """Change a model's saved identity consistently, leaving pair eligibility to validation."""
    data = json.loads(path.read_text())
    row = data["runs"][index]
    row[key] = value
    manifest_path = path.parent / row["model_id"] / "boolq/run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    plan_key = {"checkpoint": "model_id", "adapter_sha256": "decision_adapter_sha256"}.get(key, key)
    manifest["plan"]["models"][0]["identity"]["model"][plan_key] = value
    if key in ("revision", "precision", "quantization"):
        manifest["execution"][row["model_id"]]["telemetry"][key] = value
    manifest_path.write_text(json.dumps(manifest))
    row["artifact_sha256"]["boolq/run_manifest.json"] = COLLECT.digest(manifest_path)
    path.write_text(json.dumps(data))


def test_latest_success_selected_without_cherry_picking_and_all_attempts_retained(tmp_path):
    first = campaign(tmp_path, "first", 1, [("gemma-base-bf16", "completed", 120)])
    second = campaign(tmp_path, "second", 2, [("gemma-base-bf16", "completed", 110)])
    third = campaign(tmp_path, "third", 3, [("gemma-base-bf16", "failed", None)])
    result = COLLECT.collect([third, first, second], tmp_path / "summary.json", resamples=100)
    assert result["runs"][0]["boolq"]["correct"] == 110
    assert result["runs"][0]["selected_attempt"] == "second:gemma-base-bf16"
    assert len(result["attempts"]) == 3
    assert sum(a["selected"] for a in result["attempts"]) == 1
    assert all(a["source_campaign"] for a in result["attempts"])
    assert result["status"] == "incomplete"
    assert "gemma-ce128-s2-nf4" in result["missing_models"]


def test_latest_incomplete_attempt_selected_only_when_none_completed(tmp_path):
    first = campaign(tmp_path, "first", 1, [("gemma-base-bf16", "failed", None)])
    second = campaign(tmp_path, "second", 2, [("gemma-base-bf16", "running", None)])
    result = COLLECT.collect([first, second], tmp_path / "summary.json", resamples=100)
    assert result["runs"][0]["status"] == "running"
    assert result["unresolved_models"] == ["gemma-base-bf16"]


def test_clef_is_optional_for_old_seventeen_model_campaigns(tmp_path):
    original = [*COLLECT.EXPECTED_MODELS, *COLLECT.RECOVERY_MODELS]
    path = campaign(tmp_path, "historical", 1, [(model, "failed", None) for model in original])
    result = COLLECT.collect([path], tmp_path / "summary.json", resamples=100)
    assert len(result["expected_models"]) == 17
    assert "clef-local" not in result["expected_models"]
    assert "clef-local" not in result["missing_models"]
    assert all(pair["right"] != "clef-local" for pair in result["comparisons"])


def test_clef_campaign_adds_expected_model_and_exact_input_base_comparison(tmp_path):
    path = campaign(
        tmp_path,
        "clef",
        2,
        [("gemma-base-bf16", "completed", 111), ("clef-local", "completed", 114)],
    )
    result = COLLECT.collect([path], tmp_path / "summary.json", resamples=100)
    assert "clef-local" in result["expected_models"]
    assert "clef-local" not in result["missing_models"]
    pair = next(c for c in result["comparisons"] if c["right"] == "clef-local")
    assert pair["left"] == "gemma-base-bf16"
    assert pair["kind"] == "same_input_generalist"
    assert pair["status"] == "computed"
    assert pair["common_decisions"] == 128
    assert pair["operational_accuracy_delta"]["estimate"] == 3 / 128


def test_failed_clef_attempt_remains_required_without_publishing_a_score(tmp_path):
    path = campaign(
        tmp_path,
        "clef-failed",
        2,
        [("gemma-base-bf16", "completed", 111), ("clef-local", "failed", None)],
    )
    result = COLLECT.collect([path], tmp_path / "summary.json", resamples=100)
    assert "clef-local" in result["expected_models"]
    assert "clef-local" in result["unresolved_models"]
    assert result["status"] == "incomplete"
    pair = next(c for c in result["comparisons"] if c["right"] == "clef-local")
    assert pair["status"] == "not_available"
    assert pair["reason"] == "saved BoolQ predictions or run manifest unavailable"
    assert "operational_accuracy_delta" not in pair


def test_pre_execution_bad_dataset_retained_but_evaluated_mismatch_rejected(tmp_path):
    failed = campaign(
        tmp_path, "failed", 1, [("gemma-base-bf16", "failed", None)], media_hash="other"
    )
    good = campaign(tmp_path, "good", 2, [("gemma-base-bf16", "completed", 111)])
    result = COLLECT.collect([failed, good], tmp_path / "summary.json", resamples=100)
    bad_attempt = result["attempts"][0]
    assert bad_attempt["dataset_compatible"] is bad_attempt["selected"] is False
    assert bad_attempt["selection_reason"] == "incompatible pre-execution attempt"
    measured = campaign(
        tmp_path, "measured", 3, [("gemma-base-bf16", "completed", 111)], media_hash="other"
    )
    with pytest.raises(ValueError, match="different dataset identities"):
        COLLECT.collect([good, measured], tmp_path / "summary.json", resamples=100)


@pytest.mark.parametrize("mode", ["int8", "nf4"])
def test_complete_same_seed_pair_uses_source_groups_and_pins_prediction_artifacts(tmp_path, mode):
    path = campaign(
        tmp_path,
        "measured",
        1,
        [
            ("gemma-ce128-s0-bf16", "completed", 116),
            (f"gemma-ce128-s0-{mode}", "completed", 117),
        ],
    )
    result = COLLECT.collect([path], tmp_path / "summary.json", resamples=100)
    pair = next(c for c in result["comparisons"] if c["right"] == f"gemma-ce128-s0-{mode}")
    assert pair["status"] == "computed"
    delta = pair["operational_accuracy_delta"]
    assert delta["estimate"] == 1 / 128
    assert delta["groups"] == 64
    assert delta["observations"] == 128
    assert delta["interval"] is not None
    assert pair["common_decisions"] == 128
    assert pair["left_evidence"]["predictions_sha256"]
    assert pair["right_evidence"]["manifest_sha256"]
    assert pair["multiple_comparisons_adjusted"] is False


@pytest.mark.parametrize("mode", ["int8", "nf4"])
@pytest.mark.parametrize(
    ("index", "key", "value"),
    [
        (0, "precision", "float32"),
        (1, "precision", "float32"),
        (0, "quantization", "candidate_mode"),
        (1, "quantization", "none"),
        (1, "quantization", "other_mode"),
    ],
)
def test_same_seed_pair_rejects_consistent_but_wrong_precision_or_mode(
    tmp_path, mode, index, key, value
):
    candidate = f"gemma-ce128-s0-{mode}"
    path = campaign(
        tmp_path,
        "measured",
        1,
        [("gemma-ce128-s0-bf16", "completed", 116), (candidate, "completed", 117)],
    )
    if value == "candidate_mode":
        value = mode
    elif value == "other_mode":
        value = "int8" if mode == "nf4" else "nf4"
    update_consistent_identity(path, index, key, value)
    result = COLLECT.collect([path], tmp_path / "summary.json", resamples=100)
    assert len(result["runs"]) == 2  # Each individual measurement remains internally valid.
    pair = next(c for c in result["comparisons"] if c["right"] == candidate)
    assert pair["status"] == "not_available"
    assert "unquantized BF16 reference" in pair["reason"]
    assert "operational_accuracy_delta" not in pair


@pytest.mark.parametrize("key", ["checkpoint", "revision", "adapter_sha256"])
def test_same_seed_pair_requires_same_identity_between_consistent_individual_runs(tmp_path, key):
    path = campaign(
        tmp_path,
        "measured",
        1,
        [("gemma-ce128-s0-bf16", "completed", 116), ("gemma-ce128-s0-nf4", "completed", 117)],
    )
    update_consistent_identity(path, 1, key, "different")
    result = COLLECT.collect([path], tmp_path / "summary.json", resamples=100)
    pair = next(c for c in result["comparisons"] if c["right"] == "gemma-ce128-s0-nf4")
    assert pair["status"] == "not_available"
    assert "identical checkpoint and adapter identities" in pair["reason"]


def test_same_seed_pair_does_not_treat_two_missing_adapter_digests_as_identical(tmp_path):
    path = campaign(
        tmp_path,
        "measured",
        1,
        [("gemma-ce128-s0-bf16", "completed", 116), ("gemma-ce128-s0-nf4", "completed", 117)],
    )
    for index in (0, 1):
        update_consistent_identity(path, index, "adapter_sha256", None)
    result = COLLECT.collect([path], tmp_path / "summary.json", resamples=100)
    pair = next(c for c in result["comparisons"] if c["right"] == "gemma-ce128-s0-nf4")
    assert pair["status"] == "not_available"
    assert "operational_accuracy_delta" not in pair


@pytest.mark.parametrize("tamper", ["request", "gold", "missing", "counts", "adapter"])
def test_paired_comparison_rejects_incompatible_or_incomplete_evidence(tmp_path, tamper):
    path = campaign(
        tmp_path,
        "measured",
        1,
        [
            ("gemma-ce128-s0-bf16", "completed", 116),
            ("gemma-ce128-s0-nf4", "completed", 117),
        ],
    )
    root = path.parent / "gemma-ce128-s0-nf4" / "boolq"
    if tamper == "request":
        manifest = json.loads((root / "run_manifest.json").read_text())
        manifest["plan"]["models"][0]["cases"][0]["request_sha256"] = "a" * 64
        (root / "run_manifest.json").write_text(json.dumps(manifest))
    elif tamper in ("gold", "missing"):
        records = [
            json.loads(line) for line in (root / "predictions.jsonl").read_text().splitlines()
        ]
        if tamper == "gold":
            records[-1]["gold"] = "false"
            records[-1]["actual_label"] = "true"  # Preserve correctness count while changing gold.
        else:
            records.pop()
        (root / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    else:
        value = json.loads(path.read_text())
        if tamper == "adapter":
            value["runs"][-1]["adapter_sha256"] = "b" * 64
        else:
            value["runs"][-1]["boolq"]["correct"] = 115
        path.write_text(json.dumps(value))
    if tamper in ("missing", "counts", "adapter"):
        with pytest.raises(ValueError):
            COLLECT.collect([path], tmp_path / "summary.json", resamples=100)
        return
    # A self-consistent individual receipt can still be incompatible with its pair.
    value = json.loads(path.read_text())
    value["runs"][-1]["artifact_sha256"] = {
        f"boolq/{name}": COLLECT.digest(root / name)
        for name in ("predictions.jsonl", "run_manifest.json")
    }
    path.write_text(json.dumps(value))
    result = COLLECT.collect([path], tmp_path / "summary.json", resamples=100)
    pair = next(c for c in result["comparisons"] if c["right"] == "gemma-ce128-s0-nf4")
    assert pair["status"] == "not_available"
    assert "operational_accuracy_delta" not in pair
    assert pair["reason"]


def test_duplicate_campaign_input_and_ambiguous_model_attempt_are_rejected(tmp_path):
    path = campaign(tmp_path, "single", 1, [("gemma-base-bf16", "completed", 111)])
    with pytest.raises(ValueError, match="unique"):
        COLLECT.collect([path, path], tmp_path / "summary.json", resamples=100)
    value = json.loads(path.read_text())
    value["runs"].append(value["runs"][0])
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="duplicate model"):
        COLLECT.collect([path], tmp_path / "summary.json", resamples=100)


@pytest.mark.parametrize("valid_parent", [True, False])
def test_recovery_pair_permits_new_adapter_only_with_verified_training_lineage(
    tmp_path, valid_parent
):
    path = campaign(
        tmp_path,
        "recovery",
        1,
        [
            ("gemma-ce128-s0-nf4", "completed", 110),
            ("gemma-ce128-s0-nf4-recovered", "completed", 116),
        ],
    )
    data = json.loads(path.read_text())
    recovered = data["runs"][-1]
    recovered["adapter_sha256"] = "b" * 64
    recovered["recovery"] = {
        "method": "fixed_nf4_qlora_continuation",
        "parent_adapter_sha256": "a" * 64 if valid_parent else "c" * 64,
        "additional_optimizer_updates": 100,
        "train_dataset_sha256": COLLECT.RECOVERY_TRAIN_SHA256,
        "calibration_dataset_sha256": COLLECT.RECOVERY_CALIBRATION_SHA256,
        "no_test_fitting": True,
        "exploratory": True,
        "recipe_fixed_before_execution": True,
    }
    model_root = path.parent / recovered["model_id"]
    manifest_path = model_root / "boolq/run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["plan"]["models"][0]["identity"]["model"]["decision_adapter_sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest))
    recovered["artifact_sha256"]["boolq/run_manifest.json"] = COLLECT.digest(manifest_path)
    training_path = model_root / "training-receipt.json"
    training_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "actual_optimizer_updates": 100,
                "base_model": recovered["checkpoint"],
                "base_revision": recovered["revision"],
                "adapter_sha256": recovered["adapter_sha256"],
                "recovery": recovered["recovery"],
                "observed_training_ids": [f"train-{i}" for i in range(100)],
                "observed_training_request_sha256": [
                    hashlib.sha256(f"train-{i}".encode()).hexdigest() for i in range(100)
                ],
                "calibration_reference": [
                    {"request_sha256": hashlib.sha256(f"calibration-{i}".encode()).hexdigest()}
                    for i in range(16)
                ],
                "frozen_parameter_sample_sha256_before": {"embedding": "c" * 64},
                "frozen_parameter_sample_sha256_after": {"embedding": "c" * 64},
                "lora_sha256_before": "d" * 64,
                "lora_sha256_after": "e" * 64,
            }
        )
    )
    recovered["artifacts"]["training"] = "training-receipt.json"
    recovered["artifact_sha256"]["training-receipt.json"] = COLLECT.digest(training_path)
    path.write_text(json.dumps(data))
    result = COLLECT.collect([path], tmp_path / "summary.json", resamples=100)
    pair = next(c for c in result["comparisons"] if c["right"] == "gemma-ce128-s0-nf4-recovered")
    assert pair["kind"] == "exploratory_extra_training_recovery"
    assert "additional training" in pair["interpretation"]
    assert pair["status"] == ("computed" if valid_parent else "not_available")
    assert "gemma-ce128-s2-nf4-recovered" in result["expected_models"]
    assert "gemma-ce128-s2-nf4-recovered" in result["missing_models"]
    if valid_parent:
        assert pair["operational_accuracy_delta"]["estimate"] == 6 / 128


@pytest.mark.parametrize(
    "artifact", ["boolq/predictions.jsonl", "boolq/run_manifest.json", "load.json"]
)
@pytest.mark.parametrize("tamper", ["changed", "missing"])
def test_selected_score_requires_all_recorded_artifact_hashes(tmp_path, artifact, tamper):
    path = campaign(tmp_path, "measured", 1, [("gemma-base-bf16", "completed", 111)])
    data = json.loads(path.read_text())
    artifact_path = path.parent / "gemma-base-bf16" / artifact
    if artifact == "load.json":
        artifact_path.write_text("[]")
        data["runs"][0]["artifact_sha256"][artifact] = COLLECT.digest(artifact_path)
        path.write_text(json.dumps(data))
    if tamper == "missing":
        artifact_path.unlink()
    else:
        artifact_path.write_text(artifact_path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch|unavailable"):
        COLLECT.collect([path], tmp_path / "summary.json", resamples=100)


@pytest.mark.parametrize(
    "key", ["checkpoint", "revision", "precision", "quantization", "adapter_sha256"]
)
def test_selected_score_rejects_receipt_identity_mislabeling(tmp_path, key):
    path = campaign(tmp_path, "measured", 1, [("gemma-base-bf16", "completed", 111)])
    data = json.loads(path.read_text())
    data["runs"][0][key] = "different"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=f"receipt {key} differs"):
        COLLECT.collect([path], tmp_path / "summary.json", resamples=100)


@pytest.mark.parametrize("key", ["revision", "precision", "quantization"])
def test_selected_score_rejects_runtime_identity_disagreeing_with_its_plan(tmp_path, key):
    path = campaign(tmp_path, "measured", 1, [("gemma-base-bf16", "completed", 111)])
    data = json.loads(path.read_text())
    manifest_path = path.parent / "gemma-base-bf16/boolq/run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["execution"]["gemma-base-bf16"]["telemetry"][key] = "different"
    manifest_path.write_text(json.dumps(manifest))
    data["runs"][0]["artifact_sha256"]["boolq/run_manifest.json"] = COLLECT.digest(manifest_path)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=f"runtime {key} differs"):
        COLLECT.collect([path], tmp_path / "summary.json", resamples=100)


def test_selected_score_requires_saved_hashes_even_without_a_comparison_partner(tmp_path):
    path = campaign(tmp_path, "measured", 1, [("gemma-base-bf16", "completed", 111)])
    data = json.loads(path.read_text())
    del data["runs"][0]["artifact_sha256"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing BoolQ"):
        COLLECT.collect([path], tmp_path / "summary.json", resamples=100)


def test_jev_omni_adds_a_verified_same_input_pair_only_when_attempted(tmp_path):
    path = campaign(
        tmp_path,
        "jev-omni",
        1,
        [("gemma-base-bf16", "completed", 111), ("jev-omni-local", "completed", 116)],
    )
    result = COLLECT.collect([path], tmp_path / "summary.json", resamples=100)
    assert "jev-omni-local" in result["expected_models"]
    assert "jev-omni-local" not in result["missing_models"]
    comparison = next(row for row in result["comparisons"] if row["right"] == "jev-omni-local")
    assert comparison["status"] == "computed"
    assert comparison["kind"] == "same_input_generalist"
    assert comparison["operational_accuracy_delta"]["estimate"] == 5 / 128
