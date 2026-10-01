import copy
import json
import math

import pytest

from s1.evaluation.datasets import audit_splits
from s1.evaluation.metrics import binary_metrics, quality_metrics, risk_coverage
from s1.evaluation.statistics import cluster_interval, paired_comparison


def row(identifier, p, gold, *, kind="choice", **extra):
    labels = list(p)
    best = max(labels, key=p.__getitem__)
    return {
        "request_id": identifier,
        "case_id": identifier,
        "question_id": "q",
        "repetition": 0,
        "group_id": identifier,
        "task_id": "task",
        "schema_id": "schema",
        "language": "en",
        "type": kind,
        "status": "ok",
        "eligibility": "eligible",
        "probabilities": p,
        "labels": labels,
        "actual_label": best,
        "standardized_argmax": best,
        "gold": gold,
        **extra,
    }


def test_hard_soft_and_calibration_metrics_have_hand_computable_values():
    records = [
        row("a", {"a": 0.75, "b": 0.25}, "a", soft_gold={"a": 0.5, "b": 0.5}),
        row("b", {"a": 0.75, "b": 0.25}, "b"),
    ]
    metrics = quality_metrics(records)
    assert metrics["nll"] == pytest.approx(-(math.log(0.75) + math.log(0.25)) / 2)
    assert metrics["brier_sum"] == pytest.approx(0.625)
    assert metrics["ece_15"] == pytest.approx(0.25)
    assert metrics["soft_targets"]["cross_entropy"] == pytest.approx(metrics["nll"])
    assert metrics["soft_targets"]["brier_sum"] == pytest.approx(0.125)
    assert sum(b["n"] for b in metrics["reliability"]) == 2


def test_score_metrics_use_ordinal_and_numeric_distances_separately():
    metrics = quality_metrics(
        [row("s", {"10": 0.25, "20": 0.5, "50": 0.25}, "50", kind="score", score=25, gold_score=50)]
    )
    assert metrics["score"]["mae"] == 25
    assert metrics["score"]["ordinal_error"] == 1
    assert metrics["score"]["rps_normalized"] == pytest.approx((0.25**2 + 0.75**2) / 2)


def test_binary_metrics_handle_ties_and_single_class():
    metrics = binary_metrics([0.9, 0.5, 0.5, 0.1], [True, True, False, False])
    assert metrics["auroc"] == pytest.approx(0.875)
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["fpr"] == 0
    assert metrics["fnr"] == 0.5
    assert binary_metrics([0.2], [False])["auroc"] is None
    assert binary_metrics([], [])["average_precision"] is None


def test_risk_curve_is_invariant_to_order_within_ties():
    assert risk_coverage([0.8, 0.8], [True, False]) == risk_coverage([0.8, 0.8], [False, True])
    assert risk_coverage([0.8, 0.8], [True, False])["aurc"] == 0.5


def test_absent_probabilities_do_not_get_fabricated_distribution_metrics():
    record = row("a", {"a": 1, "b": 0}, "a")
    record["probabilities"] = None
    metrics = quality_metrics([record])
    assert metrics["n"] == 1
    assert metrics["probability_n"] == 0
    assert metrics["nll"] is None
    assert metrics["brier_sum"] is None


def test_bootstrap_resamples_groups_and_repeated_rows_cannot_add_groups():
    result = cluster_interval([0, 0, 1, 1], ["a", "a", "b", "b"], seed=12)
    assert result["groups"] == 2
    assert result["estimate"] == 0.5
    assert result == cluster_interval([0, 0, 1, 1], ["a", "a", "b", "b"], seed=12)
    assert cluster_interval([1] * 100, ["a"] * 100)["interval"] is None


def test_paired_failures_are_zero_operational_but_not_conditional():
    left = [row("a", {"a": 1, "b": 0}, "a"), row("b", {"a": 1, "b": 0}, "a")]
    right = copy.deepcopy(left)
    right[1].update(status="timeout", actual_label=None, probabilities=None)
    result = paired_comparison(left, right)
    assert result["conditional_accuracy_delta"]["estimate"] == 0
    assert result["operational_accuracy_delta"]["estimate"] == -0.5
    assert result["operational_accuracy_delta"]["groups"] == 2
    with pytest.raises(ValueError, match="duplicate"):
        paired_comparison(left + left, right)


def test_split_audit_rejects_group_and_semantic_request_overlap(benchmark_case, tmp_path):
    a, b = benchmark_case.model_dump(), benchmark_case.model_dump()
    a.update(split="train", group_id="document")
    b.update(id="other", split="test", group_id="document")
    path = tmp_path / "splits.jsonl"
    path.write_text(json.dumps(a) + "\n" + json.dumps(b))
    audit = audit_splits([path])
    assert not audit["valid"]
    assert {overlap["kind"] for overlap in audit["overlaps"]} == {"group", "request"}
