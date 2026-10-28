import hashlib
import json

import jev_decision3_cpu_browser_adapter as adapter
import jev_decision3_cpu_browser_smoke_audit as audit
import pytest
from test_jev_decision3_cpu_browser_adapter import FakeRuntime, request


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def evidence(monkeypatch):
    monkeypatch.setattr(audit, "SOURCE_ONLY", False)
    monkeypatch.setattr(adapter, "SOURCE_ONLY", False)
    fixture_sha = sha(b"synthetic page fixture")
    dom = "<html><body>synthetic complete</body></html>"
    dom_sha = sha(dom.encode())
    plan = {
        "schema": "decision3-cpu-browser-smoke-plan-v1",
        "model": "cpu-test",
        "slots": [
            {
                "slot": index,
                "profile": profile,
                "fixture_sha256": fixture_sha,
                "expected_dom_sha256": dom_sha,
            }
            for index, profile in enumerate(audit.PROFILES)
        ],
    }
    plan_sha = sha(encode(plan))
    invocation = "a" * 32
    runtime = FakeRuntime()
    cpu = adapter.CpuBrowserAdapter(runtime, model_identity="cpu-test", enabled=True)
    rows = []
    for index, profile in enumerate(audit.PROFILES):
        payload = request()
        raw_request = encode(payload)
        raw_response = encode(cpu.predict(payload))
        call_id = index + 1
        delivery = {
            "status": "client_received_and_handler_closed",
            "plan_sha256": plan_sha,
            "invocation": invocation,
            "call_id": call_id,
            "request_sha256": sha(raw_request),
            "response_sha256": sha(raw_response),
            "client_response_sha256": sha(raw_response),
            "socket_write_returned": True,
            "handler_closed": True,
            "socket_write_elapsed_seconds": 1.5,
            "request_to_socket_write_elapsed_seconds": 2.0,
        }
        action = {
            "status": "executed",
            "plan_sha256": plan_sha,
            "invocation": invocation,
            "call_id": call_id,
            "operation": "CLICK",
            "target": "left",
        }
        capture = {
            "capture_type": "independent_post_action_dom",
            "plan_sha256": plan_sha,
            "invocation": invocation,
            "call_id": call_id,
            "fixture_sha256": fixture_sha,
            "html": dom,
            "sha256": dom_sha,
        }
        rows.append(
            {
                "slot": index,
                "profile": profile,
                "status": "completed",
                "calls": [
                    {
                        "call_id": call_id,
                        "request": raw_request.decode(),
                        "response": raw_response.decode(),
                        "delivery": delivery,
                        "action": action,
                    }
                ],
                "final_dom": capture,
            }
        )
    result = {
        "schema": "decision3-cpu-browser-smoke-evidence-v1",
        "plan_sha256": plan_sha,
        "invocation": invocation,
        "slots": rows,
    }
    closure = {
        "schema": "decision3-cpu-browser-smoke-closure-v1",
        "plan_sha256": plan_sha,
        "invocation": invocation,
        "status": "closed",
        "cgroup": "absent",
        "process_group": "absent",
        "browser_children": "absent",
        "listener": "closed",
    }
    return {
        "plan": plan,
        "result": result,
        "closure": closure,
        "plan_sha": plan_sha,
        "invocation": invocation,
    }


def attest(value):
    entries = [value["closure"]]
    for row in value["result"]["slots"]:
        entries.append(row)
        for call in row["calls"]:
            entries += [call["delivery"], call["action"]]
        if row["final_dom"] is not None:
            entries.append(row["final_dom"])
    return {sha(encode(entry)) for entry in entries}


def execute(value, *, attestations=None):
    return audit.audit(
        plan_bytes=encode(value["plan"]),
        evidence_bytes=encode(value["result"]),
        closure_bytes=encode(value["closure"]),
        expected_plan_sha256=value["plan_sha"],
        expected_invocation=value["invocation"],
        externally_verified_sha256s=attest(value) if attestations is None else attestations,
    )


def test_four_balanced_slots_require_complete_attested_model_action_and_dom(evidence):
    result = execute(evidence)
    assert result["status"] == "audited_caller_attested"
    assert result["denominator"] == {"v7": 2, "v7-links": 2}
    assert result["score"] == {
        "v7": {
            "successes": 2,
            "scheduled": 2,
            "completed": 2,
            "reported_failed": 0,
            "not_executed": 0,
        },
        "v7-links": {
            "successes": 2,
            "scheduled": 2,
            "completed": 2,
            "reported_failed": 0,
            "not_executed": 0,
        },
    }
    assert [row["profile"] for row in result["slots"]] == list(audit.PROFILES)
    assert "caller must authenticate" in result["scope"]
    assert "common success criteria across plan slots" in result["scope"]
    assert "does not certify client end-to-end latency" in result["scope"]
    assert "do not prove execution started" in result["scope"]
    assert "timeout must be recorded as an externally attested failed slot" in result["scope"]


def test_failure_and_not_executed_remain_in_full_denominator(evidence):
    for index in range(1, 4):
        row = evidence["result"]["slots"][index]
        row.update(status="failed" if index == 1 else "not_executed", calls=[], final_dom=None)
    result = execute(evidence)
    assert [row["outcome"] for row in result["slots"]] == [
        "success",
        "failed",
        "not_executed",
        "not_executed",
    ]
    assert result["score"] == {
        "v7": {
            "successes": 1,
            "scheduled": 2,
            "completed": 1,
            "reported_failed": 0,
            "not_executed": 1,
        },
        "v7-links": {
            "successes": 0,
            "scheduled": 2,
            "completed": 0,
            "reported_failed": 1,
            "not_executed": 1,
        },
    }


def test_unverified_cleanup_never_emits_success_or_score(evidence):
    evidence["closure"]["status"] = "cleanup_unconfirmed"
    result = execute(evidence)
    assert result["status"] == "unknown" and result["score"] is None
    assert all(row["outcome"] == "unknown" for row in result["slots"])


def test_reported_failed_empty_call_requires_attestation_and_does_not_assert_start(evidence):
    rows = evidence["result"]["slots"]
    rows[0].update(status="failed", calls=[], final_dom=None)
    for row in rows[1:]:
        row.update(status="not_executed", calls=[], final_dom=None)
    verified = attest(evidence)
    verified.remove(sha(encode(rows[0])))
    result = execute(evidence, attestations=verified)
    assert result["status"] == "unknown" and result["score"] is None
    assert result["slots"][0]["outcome"] == "unknown"
    result = execute(evidence)
    assert result["slots"][0]["outcome"] == "failed"
    assert result["score"]["v7"] == {
        "successes": 0,
        "scheduled": 2,
        "completed": 0,
        "reported_failed": 1,
        "not_executed": 1,
    }


def test_missing_external_attestation_prevents_success_and_score(evidence):
    verified = attest(evidence)
    verified.remove(sha(encode(evidence["result"]["slots"][0]["final_dom"])))
    result = execute(evidence, attestations=verified)
    assert result["status"] == "unknown" and result["score"] is None
    assert all(row["outcome"] != "success" for row in result["slots"])


def test_missing_closure_attestation_never_emits_success(evidence):
    verified = attest(evidence)
    verified.remove(sha(encode(evidence["closure"])))
    result = execute(evidence, attestations=verified)
    assert result["status"] == "unknown" and result["score"] is None
    assert all(row["outcome"] == "unknown" for row in result["slots"])


@pytest.mark.parametrize(
    "fault",
    [
        "profile",
        "missing_slot",
        "invalid_dom_hash",
        "wrong_dom_fixture",
        "wrong_action",
        "bad_delivery",
        "missing_response",
        "bad_response_digest",
        "no_calls",
        "resume_after_failure",
        "repeated_call",
        "late_delivery",
        "slow_inference_fast_write",
        "missing_total_time",
        "write_longer_than_total",
    ],
)
def test_inconsistent_slot_or_evidence_is_never_scored(evidence, fault):
    rows = evidence["result"]["slots"]
    if fault == "profile":
        rows[1]["profile"] = "v7"
    elif fault == "missing_slot":
        rows.pop()
    elif fault == "invalid_dom_hash":
        rows[0]["final_dom"]["sha256"] = "b" * 64
    elif fault == "wrong_dom_fixture":
        rows[0]["final_dom"]["fixture_sha256"] = "b" * 64
    elif fault == "wrong_action":
        rows[0]["calls"][0]["action"]["target"] = "right"
    elif fault == "bad_delivery":
        rows[0]["calls"][0]["delivery"]["handler_closed"] = False
    elif fault == "missing_response":
        rows[0]["calls"][0]["response"] = ""
    elif fault == "bad_response_digest":
        rows[0]["calls"][0]["delivery"]["client_response_sha256"] = "b" * 64
    elif fault == "no_calls":
        rows[0]["calls"] = []
    elif fault == "resume_after_failure":
        rows[1].update(status="failed", calls=[], final_dom=None)
    elif fault == "repeated_call":
        rows[1]["calls"][0]["call_id"] = 1
    elif fault == "late_delivery":
        rows[0]["calls"][0]["delivery"]["socket_write_elapsed_seconds"] = 20.0
        rows[0]["calls"][0]["delivery"]["request_to_socket_write_elapsed_seconds"] = 20.0
    elif fault == "slow_inference_fast_write":
        rows[0]["calls"][0]["delivery"].update(
            socket_write_elapsed_seconds=1.0,
            request_to_socket_write_elapsed_seconds=36.0,
        )
    elif fault == "missing_total_time":
        rows[0]["calls"][0]["delivery"].pop("request_to_socket_write_elapsed_seconds")
    elif fault == "write_longer_than_total":
        rows[0]["calls"][0]["delivery"]["request_to_socket_write_elapsed_seconds"] = 1.0
    with pytest.raises(ValueError):
        execute(evidence)


def test_dom_oracle_mismatch_is_a_failed_attempt(evidence):
    dom = evidence["result"]["slots"][0]["final_dom"]
    dom["html"] = "<html>not complete</html>"
    dom["sha256"] = sha(dom["html"].encode())
    result = execute(evidence)
    assert result["slots"][0]["outcome"] == "failed"
    assert result["score"]["v7"] == {
        "successes": 1,
        "scheduled": 2,
        "completed": 2,
        "reported_failed": 0,
        "not_executed": 0,
    }


def test_source_gate_does_not_consume_inputs(monkeypatch):
    monkeypatch.setattr(audit, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="release required"):
        audit.audit(
            plan_bytes=None,
            evidence_bytes=None,
            closure_bytes=None,
            expected_plan_sha256=None,
            expected_invocation=None,
            externally_verified_sha256s=None,
        )


@pytest.mark.parametrize(
    "fault",
    [
        "missing_head",
        "missing_option",
        "native_choice",
        "wrong_usage",
        "wire_metadata",
        "head_lengths",
        "named_confidence",
        "named_margin",
        "native_entropy",
        "named_error",
        "native_error",
    ],
)
def test_model_response_inconsistency_is_rejected_even_with_matching_delivery_digest(
    evidence, fault
):
    call = evidence["result"]["slots"][0]["calls"][0]
    response = json.loads(call["response"])
    if fault == "missing_head":
        response["answers"].pop("click_target")
    elif fault == "missing_option":
        response["answers"]["click_target"]["probabilities"].pop("right")
    elif fault == "native_choice":
        response["decision3_native"]["raw_response"]["answers"]["click_target"]["choice"] = "right"
    elif fault == "wrong_usage":
        response["usage"]["input_tokens"] += 1
    elif fault == "wire_metadata":
        response["decision3_browser_wire"]["answer_values_changed"] = True
    elif fault == "named_confidence":
        response["answers"]["operation"]["confidence"] = 0.01
    elif fault == "named_margin":
        response["answers"]["operation"]["margin"] = 0.01
    elif fault == "native_entropy":
        response["decision3_native"]["raw_response"]["answers"]["operation"]["confidence"] = 0.01
    elif fault == "named_error":
        response["answers"]["operation"]["error"] = "invalid_model_output"
    elif fault == "native_error":
        response["decision3_native"]["raw_response"]["answers"]["operation"]["error"] = (
            "invalid_model_output"
        )
    else:
        response["decision3_cpu"]["native_head_lengths"]["operation"] += 1
    response_raw = encode(response)
    call["response"] = response_raw.decode()
    call["delivery"]["response_sha256"] = sha(response_raw)
    call["delivery"]["client_response_sha256"] = sha(response_raw)
    with pytest.raises(ValueError):
        execute(evidence)


def test_prelaunch_unknown_closure_does_not_issue_score_or_success(evidence):
    evidence["invocation"] = None
    evidence["result"]["invocation"] = None
    evidence["closure"].update(invocation=None, status="not_started")
    result = execute(evidence)
    assert result["status"] == "unknown" and result["score"] is None
    assert all(row["outcome"] == "unknown" for row in result["slots"])


def test_native_and_named_singleton_confidence_and_margin_are_recomputed(evidence):
    endpoint = adapter.CpuBrowserAdapter(FakeRuntime(), model_identity="cpu-test", enabled=True)
    payload = request(singleton=True)
    response = endpoint.predict(payload)
    raw_request, raw_response = encode(payload), encode(response)
    call = evidence["result"]["slots"][0]["calls"][0]
    call["request"], call["response"] = raw_request.decode(), raw_response.decode()
    call["delivery"]["request_sha256"] = sha(raw_request)
    call["delivery"]["response_sha256"] = call["delivery"]["client_response_sha256"] = sha(
        raw_response
    )
    call["action"]["operation"], call["action"]["target"] = "DONE", None
    assert execute(evidence)["slots"][0]["outcome"] == "success"
    response["answers"]["operation"]["margin"] = 0.0
    raw_response = encode(response)
    call["response"] = raw_response.decode()
    call["delivery"]["response_sha256"] = call["delivery"]["client_response_sha256"] = sha(
        raw_response
    )
    with pytest.raises(ValueError, match="confidence or margin"):
        execute(evidence)
