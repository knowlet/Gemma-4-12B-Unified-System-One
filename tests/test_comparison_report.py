"""Evidence integrity guards for the comparison document."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_comparison", ROOT / "scripts/summarize_comparison.py"
)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def historical():
    return {
        "data": {
            "datasets": {
                "text-test.jsonl": {"cases": 128, "dataset_sha256": "boolq-fixed-hash"},
                "media-test.jsonl": {"cases": 52, "dataset_sha256": "media-fixed-hash"},
            }
        }
    }


def campaign():
    return {
        "campaign_id": "test",
        "dataset": {"sha256": "boolq-fixed-hash", "cases": 128},
        "runs": [
            {
                "model_id": "candidate",
                "status": "completed_with_errors",
                "boolq": {"expected": 128, "recorded": 128, "valid": 127, "correct": 115},
                "media": {"native_status": "unsupported"},
            }
        ],
    }


def test_errors_stay_in_denominator_and_unsupported_media_is_not_zero():
    result = REPORT.build_summary(historical(), campaign())
    run = result["runs"][0]
    assert run["boolq"]["accuracy"] == 115 / 128
    assert run["media"]["native_status"] == "unsupported"
    assert run["media"].get("accuracy") is None
    assert run["memory"]["peak_allocated_bytes"] is None


def test_reject_changed_dataset_despite_same_case_count():
    data = campaign()
    data["dataset"]["sha256"] = "a-different-128-cases"
    with pytest.raises(ValueError, match="dataset hash differs"):
        REPORT.build_summary(historical(), data)


def test_measured_accuracy_requires_dataset_identity():
    data = campaign()
    data["dataset"] = {}
    with pytest.raises(ValueError, match="require the dataset hash"):
        REPORT.build_summary(historical(), data)


@pytest.mark.parametrize(
    "replacement",
    [
        {"accuracy": 115 / 127},
        {"correct": 128, "valid": 127},
        {"recorded": 129},
        {"expected": 127},
    ],
)
def test_reject_conditional_accuracy_or_inconsistent_counts(replacement):
    data = campaign()
    data["runs"][0]["boolq"].update(replacement)
    with pytest.raises(ValueError):
        REPORT.build_summary(historical(), data)


def test_blocked_preflight_does_not_inherit_historical_metrics(tmp_path):
    data = {
        "campaign_id": "preflight",
        "status": "blocked",
        "runs": [{"model_id": "gemma-int4", "status": "blocked", "precision": "int4"}],
    }
    history = historical()
    historical_path = tmp_path / "history.json"
    historical_path.write_text(json.dumps(history))
    result = REPORT.build_summary(history, data)
    rendered = REPORT.render(result, history, historical_path, None, tmp_path / "report.md")
    assert "gemma-int4 / int4 | blocked | not measured | not measured" in rendered
    assert "No new completed load measurements" in rendered
    assert result["dataset"]["campaign_match"] is None
    assert result["runs"][0]["boolq"].get("accuracy") is None


def test_saved_historical_evidence_preserves_seed_and_media_scope():
    history = json.loads(REPORT.DEFAULT_HISTORICAL.read_text())
    rows = REPORT.historical_rows(history)
    ce = next(row for row in rows if row["model_id"] == "Gemma 12B + ce, 128/class")
    assert ce["boolq"]["accuracy"] == pytest.approx((116 + 117 + 117) / (128 * 3))
    assert ce["boolq"]["seeds"] == [0, 1, 2]
    assert ce["media"]["correct"] == 34
    assert ce["media_status"] == "seed 0 only"
    assert ce["nominal_weight_gb"] == 24
    assert "peak_allocated_bytes" not in ce


@pytest.mark.parametrize("media_hash", [None, "different-media-cases"])
def test_measured_media_requires_matching_input_identity(media_hash):
    data = campaign()
    data["dataset"]["media_sha256"] = media_hash
    data["runs"][0]["media"] = {"recorded": 52, "valid": 52, "correct": 34}
    with pytest.raises(ValueError, match="media dataset hash"):
        REPORT.build_summary(historical(), data)


def test_complete_measured_row_keeps_latency_and_memory_boundaries():
    data = campaign()
    data["dataset"]["media_sha256"] = "media-fixed-hash"
    run = data["runs"][0]
    run["boolq"].update(mean_ms=105.0, p99_ms=207.0)
    run["media"] = {"recorded": 52, "valid": 52, "correct": 34}
    run["memory"] = {
        "model_footprint_bytes": 7_000_000_000,
        "peak_allocated_bytes": 9_000_000_000,
        "peak_reserved_bytes": 11_000_000_000,
    }
    run["load"] = [{"requests": 128, "successful": 128, "p99_ms": 999.0}]
    result = REPORT.build_summary(historical(), data)["runs"][0]
    assert result["boolq"]["p99_ms"] == 207.0
    assert result["load"][0]["p99_ms"] == 999.0
    assert result["memory"]["peak_allocated_bytes"] == 9_000_000_000
    assert result["memory"]["model_footprint_bytes"] == 7_000_000_000


def seeded_campaign(quantized_mode="nf4"):
    runs = []
    for mode in ("bf16", quantized_mode):
        for seed in range(3):
            runs.append(
                {
                    "model_id": f"gemma-ce128-s{seed}-{mode}",
                    "gpu": "test GPU",
                    "status": "completed",
                    "checkpoint": "pinned-checkpoint",
                    "revision": "f" * 40,
                    "adapter_sha256": f"{seed:064x}",
                    "precision": "bfloat16",
                    "quantization": "none" if mode == "bf16" else mode,
                    "boolq": {
                        "expected": 128,
                        "recorded": 128,
                        "valid": 128,
                        "correct": 116 + int(mode != "bf16"),
                    },
                    "memory": {"peak_allocated_bytes": 7_000_000_000 + seed * 100_000_000},
                    "acceptance": {"status": "passed"},
                }
            )
    return {"dataset": {"sha256": "boolq-fixed-hash"}, "runs": runs}


@pytest.mark.parametrize("mode", ["int8", "nf4"])
def test_seed_summary_uses_all_seeds_worst_memory_and_paired_delta(mode):
    summary = REPORT.build_summary(historical(), seeded_campaign(mode))
    group = next(g for g in summary["seed_aggregates"] if g["format"] == mode)
    assert group["status"] == "complete"
    assert group["completed_seeds"] == [0, 1, 2]
    assert group["accuracy_mean"] == 117 / 128
    assert group["worst_peak_allocated_bytes"] == 7_200_000_000
    assert group["paired_mean_delta_vs_bf16_pp"] == 100 / 128
    assert group["accuracy_memory_target_status"] == "passed"


@pytest.mark.parametrize("mode", ["int8", "nf4"])
@pytest.mark.parametrize("key", ["checkpoint", "revision", "adapter_sha256"])
@pytest.mark.parametrize("change", ["mismatch", "both_missing"])
def test_seed_delta_requires_matching_present_base_and_adapter_identity(mode, key, change):
    data = seeded_campaign(mode)
    if change == "mismatch":
        data["runs"][4][key] = "different"
    else:
        data["runs"][1].pop(key)
        data["runs"][4].pop(key)
    summary = REPORT.build_summary(historical(), data)
    group = next(g for g in summary["seed_aggregates"] if g["format"] == mode)
    assert group["accuracy_mean"] == 117 / 128
    assert group["paired_mean_delta_vs_bf16_pp"] is None


@pytest.mark.parametrize("mode", ["int8", "nf4"])
@pytest.mark.parametrize(
    ("index", "key", "value"),
    [
        (1, "precision", "float32"),
        (4, "precision", "float32"),
        (1, "quantization", "candidate_mode"),
        (4, "quantization", "none"),
        (4, "quantization", "other_mode"),
    ],
)
def test_seed_delta_requires_unquantized_bf16_to_declared_quantization(mode, index, key, value):
    data = seeded_campaign(mode)
    if value == "candidate_mode":
        value = mode
    elif value == "other_mode":
        value = "int8" if mode == "nf4" else "nf4"
    data["runs"][index][key] = value
    summary = REPORT.build_summary(historical(), data)
    group = next(g for g in summary["seed_aggregates"] if g["format"] == mode)
    assert group["paired_mean_delta_vs_bf16_pp"] is None


def test_missing_seed_cannot_silently_raise_average_or_pass():
    data = seeded_campaign()
    data["runs"] = data["runs"][:-1]
    summary = REPORT.build_summary(historical(), data)
    group = next(g for g in summary["seed_aggregates"] if g["format"] == "nf4")
    assert group["missing_seeds"] == [2]
    assert group["status"] == "incomplete"
    assert group["accuracy_mean"] is None
    assert group["worst_peak_allocated_bytes"] is None
    assert group["accuracy_memory_target_status"] == "not_measured"


def test_failed_load_and_mixed_hardware_do_not_become_complete_pass():
    data = seeded_campaign()
    data["runs"][-1]["status"] = "failed"
    summary = REPORT.build_summary(historical(), data)
    group = next(g for g in summary["seed_aggregates"] if g["format"] == "nf4")
    assert group["all_run_stages_completed"] is False
    data["runs"][-1]["gpu"] = "different GPU"
    summary = REPORT.build_summary(historical(), data)
    group = next(g for g in summary["seed_aggregates"] if g["format"] == "nf4")
    assert group["status"] == "hardware_mismatch"
    assert group["accuracy_mean"] is None
    assert group["accuracy_memory_target_status"] == "not_measured"


def test_accuracy_seed_mean_preserves_mixed_skus_with_shared_configured_tier():
    data = seeded_campaign()
    for index, run in enumerate(data["runs"]):
        run["configured_gpu_tier"] = "A100-80GB"
        run["gpu"] = "A100 PCIe" if index % 2 else "A100 SXM"
    summary = REPORT.build_summary(historical(), data)
    group = next(g for g in summary["seed_aggregates"] if g["format"] == "nf4")
    assert group["status"] == "complete"
    assert group["observed_gpus"] == ["A100 PCIe", "A100 SXM"]
    assert group["configured_gpu_tiers"] == ["A100-80GB"]
    assert group["accuracy_mean"] == 117 / 128
    assert "latency" not in group


def test_recovery_accuracy_is_separate_from_unchanged_original_nf4_group():
    data = seeded_campaign()
    recovered = []
    for seed, original in enumerate(data["runs"][3:]):
        original["adapter_sha256"] = f"{seed:064x}"
        child = json.loads(json.dumps(original))
        child["model_id"] += "-recovered"
        child["adapter_sha256"] = f"{seed + 3:064x}"
        child["recovery"] = {"parent_adapter_sha256": original["adapter_sha256"]}
        child["boolq"]["correct"] = 118
        recovered.append(child)
    data["runs"].extend(recovered)
    groups = {g["format"]: g for g in REPORT.build_summary(historical(), data)["seed_aggregates"]}
    assert groups["nf4"]["accuracy_mean"] == 117 / 128
    assert groups["nf4-recovered"]["accuracy_mean"] == 118 / 128
    assert groups["nf4-recovered"]["additional_training"] is True
    assert groups["nf4-recovered"]["paired_mean_delta_vs_bf16_pp"] is None
    assert groups["nf4-recovered"]["paired_mean_delta_vs_original_nf4_pp"] == 100 / 128
    data["runs"].pop()
    groups = {g["format"]: g for g in REPORT.build_summary(historical(), data)["seed_aggregates"]}
    assert groups["nf4"]["accuracy_mean"] == 117 / 128
    assert groups["nf4-recovered"]["accuracy_mean"] is None
    assert groups["nf4-recovered"]["missing_seeds"] == [2]


def test_published_recovery_summary_does_not_claim_raw_receipts_are_included(tmp_path):
    history = json.loads(REPORT.DEFAULT_HISTORICAL.read_text())
    campaign_path = ROOT / "docs/validation/2026-10-01/comparison-summary.json"
    saved = json.loads(campaign_path.read_text())
    summary = REPORT.build_summary(history, saved)
    rendered = REPORT.render(
        summary, history, REPORT.DEFAULT_HISTORICAL, campaign_path, tmp_path / "report.md"
    )
    assert "quality results and artifact hashes for all three recovery seeds" in rendered
    assert "Individual recovery receipts and prediction artifacts are not included" in rendered
    assert "quality receipts are available" not in rendered
    groups = {group["format"]: group for group in summary["seed_aggregates"]}
    assert groups["int8"]["paired_mean_delta_vs_bf16_pp"] == pytest.approx(100 / 384)
    assert groups["nf4"]["paired_mean_delta_vs_bf16_pp"] == pytest.approx(-1500 / 384)
