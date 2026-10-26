"""Read-only replay of saved random-fixture aggregate and request joins."""

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def audit(directory=HERE / "actual-01"):
    receipt = json.loads((directory / "receipt.json").read_bytes())
    ledger = json.loads((directory / "join-ledger.json").read_bytes())
    assert receipt["completed"] and receipt["closed_observed_processes"]
    assert receipt["quality_score"] is None and receipt["model_qualified"] is False
    assert receipt["server_bootstrap_qualified"] is False
    assert not receipt["cleanup_errors"] and not receipt["blocked_nonlocal_requests"]
    assert receipt["real_model_calls"] == receipt["comfy_calls"] == 0
    policies, helpers = ledger["policy"], ledger["helper"]
    assert len(policies) == receipt["policy_calls"] == receipt["native_calls"] == 23
    assert len(helpers) == receipt["helper_calls"] == 22
    for row in policies:
        ids, wire = row["question_ids"], row["wire"]
        assert ids == list(row["answers"]) == wire["question_ids"]
        assert wire["answer_count"] == len(ids)
        assert wire["original_request_sha256"] == row["request_sha256"]
        assert wire["answer_values_changed"] is False
        assert set(ids) <= {"operation", "click_target", "type_text_target", "select_target"}
        assert row["answers"]["operation"]["choice"] == "FIND_TEXT"
        for key in ids:
            answer = row["answers"][key]
            probabilities = answer["probabilities"]
            assert set(probabilities) == set(row["candidate_labels"][key])
            assert 1 <= len(probabilities) <= 255
            assert all(
                type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1
                for p in probabilities.values()
            )
            assert abs(sum(probabilities.values()) - 1) < 1e-6
            assert probabilities[answer["choice"]] == max(probabilities.values())
    assert all(
        row["schema_valid"] and isinstance(row["query"], str) and 0 < len(row["query"]) <= 2000
        for row in helpers
    )
    cases = receipt["cases"]
    assert len(cases) == 7
    for row in cases[:6]:
        assert row["actions"] == row["decisions"] == 1
        assert helpers[row["helper_index"]]["query"] == row["feedback"]["query"]
        assert policies[row["policy_index"]]["answers"]["operation"]["choice"] == "FIND_TEXT"
        assert row["feedback"]["task_success"] == "not_evaluated"
    positive, hidden, missing, stale, uncertain, late, budget = cases
    assert positive["feedback"]["status"] == "found"
    assert not positive["before"]["in_view"] and positive["after"]["in_view"]
    assert hidden["feedback"]["status"] == missing["feedback"]["status"] == "not_found"
    assert (
        not hidden["feedback"]["scroll_requested"] and not missing["feedback"]["scroll_requested"]
    )
    assert stale["feedback"]["status"] == "stale" and stale["feedback"]["scroll_requested"] is False
    assert (
        uncertain["feedback"]["status"] == "stale"
        and uncertain["feedback"]["scroll_requested"] is None
    )
    assert uncertain["after"]["in_view"] is True
    assert late["feedback"]["status"] == "deadline" and late["feedback"]["late"] is True
    assert late["feedback"]["scroll_requested"] is False and not late["after"]["in_view"]
    assert budget["actions"] == 16 and budget["decisions"] == 17
    assert budget["seventeenth_action_refused"] and budget["helper_calls_after_budget"] == 0
    assert all(row["same_process_absent"] for row in receipt["process_closure"])
    return {
        "audit_passed": True,
        "cases": 7,
        "policy_request_joins": 23,
        "schema_valid_helper_joins": 22,
        "submitted_heads_total": sum(len(r["question_ids"]) for r in policies),
        "observed_chromium_processes_closed": len(receipt["process_closure"]),
        "quality_score": None,
        "receipt_sha256": hashlib.sha256((directory / "receipt.json").read_bytes()).hexdigest(),
        "ledger_sha256": hashlib.sha256((directory / "join-ledger.json").read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True))
