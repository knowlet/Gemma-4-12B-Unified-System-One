"""Synthetic checks only; no benchmark tasks or model execution."""

import math

import jevbench_heaven_scorer_audit as audit
import pytest


def test_event_encoding_distinguishes_candidate_order_and_rejects_nonfinite():
    first = {"questions": {"synthetic": {"criteria": {"a": "A", "b": "B"}}}}
    reversed_options = {"questions": {"synthetic": {"criteria": {"b": "B", "a": "A"}}}}
    assert audit.sha(audit.encode(first)) != audit.sha(audit.encode(reversed_options))
    with pytest.raises(ValueError):
        audit.encode({"value": math.nan})


def test_evidence_reader_rejects_parent_and_symlink_escape(tmp_path):
    selected = tmp_path / "selected"
    selected.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("synthetic")
    (selected / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="evidence_path"):
        audit.read(selected, "../outside")
    with pytest.raises(ValueError, match="evidence_path"):
        audit.read(selected, "link")


def test_evidence_reader_rejects_nonfinite_json(tmp_path):
    (tmp_path / "synthetic.json").write_text('{"value": NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        audit.read_json(tmp_path, "synthetic.json")


def test_comparison_preserves_json_types_and_scorer_sorted_keys():
    assert audit.same_json({"b": 2, "a": 1}, {"a": 1, "b": 2})
    assert not audit.same_ordered({"b": 2, "a": 1}, {"a": 1, "b": 2})
    for same_looking in (True, 1.0):
        assert not audit.same_json({"value": 1}, {"value": same_looking})
        assert not audit.same_ordered({"value": 1}, {"value": same_looking})
