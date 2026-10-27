"""Synthetic reconciliation only; no model, browser or private cases."""

import copy
import hashlib
import json

import jev_done_review_reconciliation as audit
import pytest


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fixture():
    data = dict(
        events=[],
        runner_records=[],
        captures=[],
        bindings=[],
        service_events=[],
        client_closure={"confirmed": True, "remaining": [], "tracked_PID_start_ticks": {"7": 8}},
    )
    for i, status in enumerate(("review_call", "replan_call"), 1):
        req, resp = {"model": "fake", "step": i}, {"result": i}
        rb, sb = wire(req), wire(resp)
        data["events"].append(
            {
                "status": status,
                "request_sha256": sha(audit._encode(req, sorted_keys=True)),
                "response_sha256": sha(audit._encode(resp, sorted_keys=True)),
            }
        )
        data["runner_records"].append({"kind": "policy", "request": req, "response": resp})

        def desc(b):
            return {"sha256": sha(b), "size_bytes": len(b)}

        data["captures"].append(
            {
                "record": {
                    "sequence": i,
                    "pid": 7,
                    "status": "response",
                    "request_source": "response-attached-request",
                    "http_status": 200,
                    "request_bytes": desc(rb),
                    "response_bytes": desc(sb),
                },
                "request_bytes": rb,
                "response_bytes": sb,
            }
        )
        binding = {
            "schema": "decision3-semantic-browser-binding-v1",
            "status": "prepared_not_delivered",
            "call": i,
            "original_request": req,
            "named_response": resp,
            "http_body_sha256": sha(rb),
            "response_sha256": sha(sb),
            "witness_receipt": {"fake": i},
        }
        data["bindings"].append(binding)
        for event_status in ("prepared", "socket_write_returned"):
            data["service_events"].append(
                {
                    "call": i,
                    "status": event_status,
                    "http_body_sha256": sha(rb),
                    "response_sha256": sha(sb),
                    "witness_receipt": {"fake": i},
                }
            )
    return data


def test_success_aggregate_only():
    data = fixture()
    data["runner_records"].insert(0, {"kind": "text"})
    result = audit.reconcile(**data)
    assert result["matched_events"] == {"review_call": 1, "replan_call": 1}
    assert result["status"] == "supplied_evidence_consistent"
    assert result["source_origin_verified"] is False
    assert result["semantic_delta_execution_verified"] is False
    assert result["browser_success"] is None
    assert "fake" not in json.dumps(result)


def test_no_events_is_not_a_verified_review():
    data = fixture()
    data["events"] = []
    assert audit.reconcile(**data)["status"] == "no_success_events_observed"


@pytest.mark.parametrize("location", ["event", "runner", "capture", "binding"])
def test_duplicates_are_unknown(location):
    data = fixture()
    if location == "event":
        data["events"].append(copy.deepcopy(data["events"][0]))
    elif location == "runner":
        data["runner_records"].append(copy.deepcopy(data["runner_records"][0]))
    elif location == "capture":
        cap = copy.deepcopy(data["captures"][0])
        cap["record"]["sequence"] = 3
        data["captures"].append(cap)
    else:
        binding = copy.deepcopy(data["bindings"][0])
        binding["call"] = 3
        data["bindings"].append(binding)  # even an undelivered duplicate is ambiguous
    result = audit.reconcile(**data)
    assert result["matched_events"]["review_call"] == 0
    assert result["unresolved_events"]["review_call"] >= 1


def test_redaction_and_rewrite_do_not_match():
    data = fixture()
    data["runner_records"][0]["request"] = {"model": "[REDACTED]", "step": 1}
    result = audit.reconcile(**data)
    assert result["unresolved_events"]["review_call"] == 1
    assert result["matched_events"]["replan_call"] == 1


def test_missing_delivery_and_explicit_failures_remain_unknown():
    data = fixture()
    data["service_events"].pop(1)
    data["events"].extend([{"status": "review_failed"}, {"status": "replan_failed"}])
    data["runner_records"].append({"kind": "policy", "error": "transport"})
    data["captures"].append(
        {
            "record": {
                "sequence": 3,
                "pid": 7,
                "status": "transport_exception",
                "exception_type": "Timeout",
            }
        }
    )
    result = audit.reconcile(**data)
    assert result["unresolved_events"]["review_call"] == 1
    assert result["observed_events"]["review_failed"] == 1
    assert result["observed_events"]["replan_failed"] == 1
    assert result["observed_runner_policy_errors"] == 1
    assert result["observed_capture_transport_exceptions"] == 1
    assert result["unmatched_capture_attribution"] == "unknown"


def test_runner_error_with_same_pair_cannot_match():
    data = fixture()
    data["runner_records"][0]["error"] = "failed"
    assert audit.reconcile(**data)["unresolved_events"]["review_call"] == 1


def test_wire_bytes_binding_and_order_are_required():
    data = fixture()
    data["bindings"][0]["original_request"] = {"step": 1, "model": "fake"}
    assert audit.reconcile(**data)["unresolved_events"]["review_call"] == 1
    data = fixture()
    data["events"].reverse()
    with pytest.raises(ValueError, match="order"):
        audit.reconcile(**data)


def test_modified_capture_and_unconfirmed_closure_fail_closed():
    data = fixture()
    data["captures"][0]["request_bytes"] += b" "
    with pytest.raises(ValueError, match="capture bytes"):
        audit.reconcile(**data)
    data = fixture()
    data["client_closure"]["confirmed"] = False
    with pytest.raises(ValueError, match="closure"):
        audit.reconcile(**data)


def test_duplicate_keys_are_not_silently_parsed():
    data = fixture()
    item = data["captures"][0]
    item["request_bytes"] = b'{"step":1,"step":1}'
    item["record"]["request_bytes"] = {
        "sha256": sha(item["request_bytes"]),
        "size_bytes": len(item["request_bytes"]),
    }
    result = audit.reconcile(**data)
    assert result["unresolved_events"]["review_call"] == 1
    assert result["observed_capture_unparseable_responses"] == 1


@pytest.mark.parametrize("status", ["review_failed", "replan_failed"])
def test_failed_event_is_unresolved_even_after_success(status):
    data = fixture()
    data["events"].append({"status": status})
    assert audit.reconcile(**data)["status"] == "unresolved"
    data["events"] = [{"status": status}]
    assert audit.reconcile(**data)["status"] == "unresolved"


def test_other_events_are_counted_and_limit_is_unresolved():
    data = fixture()
    data["events"].extend(
        [{"status": "done_supported"}, {"status": "replanned"}, {"status": "review_limit_blocked"}]
    )
    result = audit.reconcile(**data)
    assert result["status"] == "unresolved"
    assert result["observed_events"]["done_supported"] == 1
    assert result["observed_events"]["replanned"] == 1
    assert result["observed_events"]["review_limit_blocked"] == 1
