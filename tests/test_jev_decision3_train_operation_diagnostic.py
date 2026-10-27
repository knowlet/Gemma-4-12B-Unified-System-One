"""CPU-only synthetic tests; no production data, model or GPU."""

import copy

import numpy as np
import pytest
from jev_decision3_train_operation_diagnostic import analyze_train


def example():
    return {
        "features": np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
        "baseline": np.array([[0, 0], [0, 0], [100, 100]], dtype=np.float32),
        "delta": np.array([[0, 0], [1, 1], [0, 0]], dtype=np.float32),
        "targets": np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        "counts": np.array([2, 2, 2], dtype=np.int64),
        "row_map": [
            {"head_id": "operation", "criteria_order": ["CLICK", "TYPE_TEXT"]},
            {
                "source_row": {
                    "head_id": "operation",
                    "ordered_criteria_ids": ["TYPE_TEXT", "CLICK"],
                }
            },
            {"head_id": "operation", "criteria_order": ["CLICK", "DONE"]},
        ],
        "old_rows": 2,
        "train_only": True,
    }


def test_order_collisions_and_class_confusion_with_masked_tail():
    data = example()
    before = copy.deepcopy(data)
    result = analyze_train(**data)
    assert result["rows"] == 3
    assert result["ambiguous_ordinal_count"] == 2
    assert result["rows_with_ambiguous_ordinal"] == 3
    assert result["ordinal_operation_counts"]["0"] == {"CLICK": 2, "TYPE_TEXT": 1}
    assert result["groups"]["old"]["original_hits"] == 1
    assert result["groups"]["old"]["candidate_hits"] == 1
    assert result["groups"]["new"]["candidate_hits"] == 1
    assert result["single_label_metrics"]["TYPE_TEXT"]["rows"] == 2
    assert result["single_label_metrics"]["TYPE_TEXT"]["candidate_click"] == 1
    assert result["predicted_operation_counts"]["candidate"] == {
        "CLICK": 1,
        "DONE": 1,
        "TYPE_TEXT": 1,
    }
    for key in ("features", "baseline", "delta", "targets", "counts"):
        np.testing.assert_array_equal(data[key], before[key])
    assert data["row_map"] == before["row_map"]


def test_zero_delta_matches_original_and_empty_group_has_null_metrics():
    data = example()
    data["delta"][:] = 0
    data["old_rows"] = 3
    result = analyze_train(**data)
    assert result["groups"]["old"]["original_hits"] == result["groups"]["old"]["candidate_hits"]
    assert result["groups"]["old"]["original_ce"] == pytest.approx(np.log(2))
    assert result["groups"]["old"]["candidate_ce"] == pytest.approx(np.log(2))
    assert result["groups"]["new"]["candidate_hits"] is None


def test_multiple_valid_soft_labels_are_counted_without_inventing_one_teacher():
    data = example()
    data["targets"][0, :2] = 0.5
    result = analyze_train(**data)
    assert result["multi_label_rows"] == 1
    assert result["single_label_teacher_counts"]["TYPE_TEXT"] == 1
    assert result["groups"]["old"]["original_hits"] == 2
    assert result["groups"]["old"]["candidate_hits"] == 1


@pytest.mark.parametrize(
    "corrupt",
    ["not_train", "menu_order", "illegal_tail", "nan", "bad_count", "duplicate", "wrong_head"],
)
def test_invalid_or_unadmitted_inputs_are_rejected(corrupt):
    data = example()
    if corrupt == "not_train":
        data["train_only"] = False
    elif corrupt == "menu_order":
        data["row_map"][0]["criteria_order"].append("extra")
    elif corrupt == "illegal_tail":
        data["targets"][0] = [0, 0, 1]
    elif corrupt == "nan":
        data["delta"][0, 0] = np.nan
    elif corrupt == "bad_count":
        data["counts"][0] = 0
    elif corrupt == "duplicate":
        data["row_map"][0]["criteria_order"] = ["CLICK", "CLICK"]
    else:
        data["row_map"][0]["head_id"] = "click_target"
    with pytest.raises(ValueError):
        analyze_train(**data)
