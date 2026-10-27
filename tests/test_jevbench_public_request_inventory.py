import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import jevbench_public_request_inventory as inventory
import pytest


def test_summary_counts_and_preserves_synthetic_requests():
    items = [
        (
            "private-synthetic-state",
            {
                "first": {"type": "noul", "instructions": "synthetic-noul", "criteria": None},
                "second": {
                    "type": "score",
                    "instructions": "synthetic-score",
                    "criteria": ["low", "high"],
                },
            },
        ),
        (
            {"synthetic": "state"},
            {
                "decision": {
                    "type": "choice",
                    "instructions": "synthetic-choice",
                    "criteria": {"a": "first", "b": "second"},
                },
            },
        ),
        (
            "synthetic-invalid",
            {"invalid": {"type": "choice", "instructions": "x", "criteria": {"only": "one"}}},
        ),
    ]
    before = copy.deepcopy(items)
    result = inventory.summarize(items)
    assert items == before
    assert result["totals"]["requests"] == 3
    assert result["totals"]["contract_accepted"] == 2
    assert result["totals"]["contract_rejected"] == 1
    assert result["question_types"] == {"choice": 2, "noul": 1, "score": 1}
    assert result["choice_score_option_counts"] == {"1": 1, "2": 2}
    assert result["native_changes"]["null_criteria_omitted_heads"] == 1
    text = json.dumps(result)
    assert "private-synthetic-state" not in text
    assert "synthetic-noul" not in text
    assert "synthetic-invalid" not in text


def test_summary_detects_head_and_criteria_order_changes(monkeypatch):
    original = inventory.native_payload

    def reordered(request):
        payload = original(request)
        questions = payload["questions"]
        questions["first"]["criteria"] = dict(
            reversed(list(questions["first"]["criteria"].items()))
        )
        payload["questions"] = dict(reversed(list(questions.items())))
        return payload

    monkeypatch.setattr(inventory, "native_payload", reordered)
    question = {"type": "choice", "instructions": "synthetic", "criteria": {"a": "A", "b": "B"}}
    result = inventory.summarize([("state", {"first": question, "second": question})])
    changes = result["native_changes"]
    assert changes["head_order_or_membership_changed_requests"] == 1
    assert changes["criteria_order_or_membership_changed_heads"] == 1
    assert result["totals"]["native_structural_equal"] == 1
    assert result["totals"]["native_ordered_json_changed"] == 1


def test_summary_rejection_and_output_are_aggregate_only(monkeypatch):
    monkeypatch.setattr(inventory, "request_blockers", lambda request: ["untrusted-secret"])
    q = {"type": "noul", "instructions": "private-instruction"}
    result = inventory.summarize([("private-state", {"head": q})])
    assert result["totals"]["decision3_static_blocked"] == 1
    assert result["decision3_blockers"] == {"other": 1}
    assert "private" not in json.dumps(result)
    assert "untrusted-secret" not in json.dumps(result)


def test_summary_conversion_failure_is_counted(monkeypatch):
    def fail(_):
        raise ValueError("private-data")

    monkeypatch.setattr(inventory, "native_payload", fail)
    result = inventory.summarize([("state", {"head": {"type": "noul", "instructions": "x"}})])
    assert result["totals"]["native_conversion_rejected"] == 1
    assert "private-data" not in json.dumps(result)


def test_full_frozen_plan_expansion_deduplicates_like_official():
    q = {"type": "noul", "instructions": "synthetic"}
    cases = [
        SimpleNamespace(id="a", state="state-a", questions={"head": q}),
        SimpleNamespace(id="b", state="state-b", questions={"head": q}),
    ]

    class Plan:
        by = {
            "rel": [
                {"case": "a", "requests": "same"},
                {"case": "a", "requests": "new"},
                {"case": "b", "requests": "same"},
            ]
        }

        def __len__(self):
            return 3

    def decode(case, requests):
        return {
            "named": case.questions
            if requests == "same"
            else {"new-head": {"type": "noul", "instructions": "other"}}
        }

    suite = SimpleNamespace(cases=cases, plan=Plan())
    before = copy.deepcopy([case.questions for case in cases])
    result = inventory.coherence_inventory(
        suite, {"plan": {"requests": 3, "trials": 3}}, decode, lambda name, question: None
    )
    assert result["case_question_unique_requests"] == 3
    assert result["trial_request_occurrences"] == 3
    assert result["summary"]["totals"]["official_schema_accepted_questions"] == 3
    assert [case.questions for case in cases] == before
    with pytest.raises(ValueError, match="population"):
        inventory.coherence_inventory(
            suite, {"plan": {"requests": 2, "trials": 3}}, decode, lambda name, question: None
        )


def test_source_pin_rejects_tampering_and_preloaded_module(tmp_path, monkeypatch):
    relative = "src/jevbench/__init__.py"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"value = 1\n")
    config = {"files": {relative: hashlib.sha256(path.read_bytes()).hexdigest()}}
    finder = inventory._verified_coherence(tmp_path, config)
    assert finder.sources[relative] == b"value = 1\n"
    monkeypatch.setitem(sys.modules, "jevbench", SimpleNamespace())
    with pytest.raises(RuntimeError, match="already loaded"):
        inventory._verified_coherence(tmp_path, config)
    monkeypatch.delitem(sys.modules, "jevbench")
    path.write_bytes(b"value = 2\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        inventory._verified_coherence(tmp_path, config)


def test_rejects_unpinned_import():
    finder = inventory._VerifiedFinder(None, {})
    with pytest.raises(ImportError, match="unpinned"):
        finder.find_spec("jevbench.surprise")
