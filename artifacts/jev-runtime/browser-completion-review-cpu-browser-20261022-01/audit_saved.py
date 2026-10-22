"""Read only the exact own fixture artifacts and write a fresh aggregate audit."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PINS = {
    "run_fixture.py": "173e4c93fc61cbd36e11fb190f7346ff6905908d9f9e9c50e7ea653885ca6502",
    "actual-01/receipt.json": "e4ae43bcc88f0f5c6bc5f100d264f7696736bcae69b29f9a1930affd53c78e46",
    "actual-01/join-ledger.json": "4fbab4c92e080beb2549af30bc46ea3ee826c318d890f4c24aa4a1ff6bd7e3bd",
}


def ref(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def main():
    for label, digest in PINS.items():
        assert ref(HERE / label)["sha256"] == digest
    output = HERE / "actual-01"
    receipt = json.loads((output / "receipt.json").read_bytes())
    ledger = json.loads((output / "join-ledger.json").read_bytes())
    assert {p.name for p in output.iterdir()} == {"receipt.json", "join-ledger.json"}
    assert receipt["completed"] and receipt["closed_observed_processes"]
    assert receipt["browser_close_returned"] and receipt["server_and_playwright_contexts_closed"]
    assert receipt["before"]["in_view"] is False and receipt["after"]["in_view"] is True
    assert receipt["decision_operations"] == ["DONE", "FIND_TEXT", "DONE"]
    assert receipt["history_kinds"] == ["completion_review", "find_text", "completion_review"]
    assert receipt["review_outcomes"] == ["vetoed", "approved"]
    assert receipt["total_native_logical_slots"] == 5 and receipt["history_slots"] == 3
    assert len(ledger["policy"]) == 3 and len(ledger["review"]) == 2 and len(ledger["helper"]) == 1
    native = ledger["simulated_native_calls"]
    assert [x["scope"] for x in native] == ["policy", "review", "policy", "policy", "review"]
    assert [x["chosen"] for x in native] == ["DONE", "INCOMPLETE", "FIND_TEXT", "DONE", "COMPLETE"]
    for row, item in zip(ledger["policy"], [n for n in native if n["scope"] == "policy"]):
        assert row["head_ids"] == item["head_ids"] == row["wire"]["question_ids"]
        # Persistence sorts dictionary keys. Explicit ordered arrays, not saved
        # named-answer dictionary iteration, bind the original head inventory.
        assert set(row["head_ids"]) == set(row["answers"])
        assert row["request_sha256"] == row["wire"]["original_request_sha256"]
        assert row["wire"]["answer_values_changed"] is False
    for row, item in zip(ledger["review"], [n for n in native if n["scope"] == "review"]):
        assert row["head_ids"] == item["head_ids"] == ["visible_completion"]
        assert row["native_payload_sha256"] == item["native_payload_sha256"]
        assert (
            row["candidate_labels"]
            == item["candidate_labels"]
            == {"visible_completion": ["COMPLETE", "INCOMPLETE"]}
        )
        assert row["answer"]["choice"] == item["chosen"]
        assert row["remaining_seconds_at_return"] > 0
    assert [x["visible_goal_text_in_request"] for x in ledger["review"]] == [False, True]
    assert ledger["helper"][0]["query_sha256"] == receipt["generated_query_sha256"]
    assert ledger["helper"][0]["schema_valid"]
    assert not receipt["cleanup_errors"] and receipt["blocked_nonlocal_requests"] == 0
    assert receipt["native_api_server_returned_events"] == 3
    assert not any(k.startswith("Input.") for k in receipt["cdp_method_counts"])
    fresh = []
    for row in receipt["observed_processes"]:
        try:
            raw = (Path("/proc") / str(row["pid"]) / "stat").read_text()
            start = int(raw.rsplit(")", 1)[1].split()[19])
        except (FileNotFoundError, ProcessLookupError):
            start = None
        assert start != row["start_ticks"]
        fresh.append(
            {
                "pid": row["pid"],
                "selected_start_ticks": row["start_ticks"],
                "current_start_ticks": start,
                "same_process_absent": True,
            }
        )
    value = {
        "schema_id": "completion-browser-saved-data-audit-v1",
        "status": "PASS",
        "scope": "Read-only own fixture joins and fresh observed PID/start closure; no rerun.",
        "fixture_only": True,
        "auditor": ref(Path(__file__)),
        "audited_receipt": ref(output / "receipt.json"),
        "audited_ledger": ref(output / "join-ledger.json"),
        "executed_source": ref(HERE / "run_fixture.py"),
        "fixed_operations": ["DONE", "FIND_TEXT", "DONE"],
        "history_slots": 3,
        "native_logical_slots": 5,
        "policy_calls": 3,
        "review_calls": 2,
        "strict_schema_helper_calls": 1,
        "policy_original_request_wire_joins": True,
        "native_head_candidate_order_joins": True,
        "storage_order_scope": "Named dictionaries sorted on persistence; explicit head/candidate lists preserve original order.",
        "review_visible_goal_text": [False, True],
        "review_deadline_remaining_seconds": [
            x["remaining_seconds_at_return"] for x in ledger["review"]
        ],
        "actual_scroll_y_before_after": [receipt["before"]["y"], receipt["after"]["y"]],
        "observed_cohort_fresh_closure": fresh,
        "exact_output_index": [ref(output / p) for p in ("receipt.json", "join-ledger.json")],
        "generic_real_server_route_qualified": False,
        "quality_score": None,
        "whole_tree_exit_claimed": False,
    }
    with (HERE / "saved-data-audit.json").open("xb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps({"status": "PASS", "observed_cohort": len(fresh)}))


if __name__ == "__main__":
    main()
