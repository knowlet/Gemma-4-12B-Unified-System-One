import copy
import hashlib
import json

import jev_decision3_cpu_experiment_worker as worker
import jev_decision3_cpu_padding_candidate as candidate
import jev_decision3_cpu_result_audit as audit
import pytest
from test_jev_decision3_cpu_experiment import FakeRuntime


def encode(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


@pytest.fixture
def evidence(monkeypatch):
    monkeypatch.setattr(audit, "SOURCE_ONLY", False)
    monkeypatch.setattr(candidate, "SOURCE_ONLY", False)
    result = worker._experiment(FakeRuntime(), candidate, 4)
    for index, sample in enumerate(result["samples"]):
        sample["elapsed_ns"] = (index + 1) * 100
    result["publisher_manifest_sha256"] = audit.MANIFEST
    result["model_verification"] = "synthetic"
    invocation, plan = "b" * 32, "c" * 64
    closure = {
        "schema": "decision3-cpu-watcher-closure-v1",
        "status": "completed_pending_external_audit",
        "invocation": invocation,
        "plan_sha256": plan,
        "cgroup": "absent",
        "process_group": "absent",
        "machine_exclusive": False,
        "unit": "retained-unit",
        "diagnostic": {"last_entered_stage": "result_write", "failure_category": None},
    }
    hello = {
        "schema": "decision3-cpu-hello-v1",
        "pid": 123,
        "pgid": 123,
        "startticks": 42,
        "boot_id": "synthetic",
        "unit": "test.service",
        "cgroup": "/test",
        "invocation": invocation,
        "plan_sha256": plan,
    }
    terminal = {
        "schema": "decision3-cpu-terminal-v1",
        "status": "completed_pending_external_audit",
        "invocation": invocation,
        "plan_sha256": plan,
        "result_sha256": hashlib.sha256(encode(result)).hexdigest(),
    }
    return {
        "result": result,
        "closure": closure,
        "hello": hello,
        "terminal": terminal,
        "plan_sha": plan,
        "invocation": invocation,
        "model": "synthetic",
    }


def execute(evidence, *, hash_result=True):
    result_bytes = encode(evidence["result"])
    terminal = copy.deepcopy(evidence["terminal"])
    if hash_result:
        terminal["result_sha256"] = hashlib.sha256(result_bytes).hexdigest()
    return audit.audit(
        closure_bytes=encode(evidence["closure"]),
        terminal_bytes=encode(terminal),
        result_bytes=result_bytes,
        hello_bytes=encode(evidence["hello"]),
        plan_sha=evidence["plan_sha"],
        invocation=evidence["invocation"],
        model=evidence["model"],
    )


def test_source_gate_rejects_before_consuming_bytes(monkeypatch):
    monkeypatch.setattr(audit, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="release required"):
        audit.audit(
            closure_bytes=None,
            terminal_bytes=None,
            result_bytes=None,
            hello_bytes=None,
            plan_sha=None,
            invocation=None,
            model=None,
        )


def test_recomputes_complete_four_rounds_and_percentiles(evidence):
    output = execute(evidence)
    assert output["status"] == "complete"
    assert output["input_tokens"] == 15018
    assert list(output["head_token_lengths"].values()) == [3117, 4196, 4579, 3126]
    assert [output["padding"][mode]["padded_slots"] for mode in audit.MODES[1:]] == [
        18316,
        17550,
        15410,
    ]
    assert len(output["cycles"]) == 12
    native_times = [
        sample["elapsed_ns"]
        for sample in evidence["result"]["samples"]
        if sample["phase"] == "measured" and sample["mode"] == "native"
    ]
    ordered = sorted(native_times)
    assert output["timing"]["native"]["samples_ns"] == native_times
    assert output["timing"]["native"]["p50_ns"] == (ordered[1] + ordered[2]) / 2
    assert output["timing"]["native"]["p95_nearest_rank_ns"] == ordered[-1]
    assert "unverified" in output["scope"]


@pytest.mark.parametrize("status", ["failed", "cleanup_unconfirmed"])
def test_failed_or_unconfirmed_closure_never_reports_timings(evidence, status):
    evidence["closure"]["status"] = status
    evidence["result"] = {"not": "a completed result"}
    output = execute(evidence)
    assert output["status"] == "incomplete"
    assert "timing" not in output and "cycles" not in output


@pytest.mark.parametrize(
    "fault",
    [
        "cgroup",
        "group",
        "unit",
        "stage",
        "failure",
        "identity",
        "digest",
        "hello",
        "length",
        "order",
        "samples",
        "cycle",
        "option",
        "probability",
        "raw",
        "usage",
        "model",
        "metric",
        "comparison",
        "zero_time",
    ],
)
def test_rejects_inconsistent_completed_evidence(evidence, fault):
    result, closure = evidence["result"], evidence["closure"]
    if fault == "cgroup":
        closure["cgroup"] = "unconfirmed"
    elif fault == "group":
        closure["process_group"] = "unconfirmed"
    elif fault == "unit":
        closure["unit"] = "unknown"
    elif fault == "stage":
        closure["diagnostic"]["last_entered_stage"] = "inference"
    elif fault == "failure":
        closure["diagnostic"]["failure_category"] = "other"
    elif fault == "identity":
        closure["invocation"] = "d" * 32
    elif fault == "digest":
        evidence["terminal"]["result_sha256"] = "f" * 64
    elif fault == "hello":
        evidence["hello"]["pgid"] = 124
    elif fault == "length":
        result["head_token_lengths"]["synthetic_0"] = 1
    elif fault == "order":
        result["samples"][4], result["samples"][5] = result["samples"][5], result["samples"][4]
    elif fault == "samples":
        result["samples"].pop()
    elif fault == "cycle":
        result["samples"][4]["cycle"] = 1
    elif fault == "option":
        result["samples"][4]["response"]["answers"]["synthetic_0"]["probabilities"].pop("alpha")
    elif fault == "probability":
        result["samples"][4]["response"]["answers"]["synthetic_0"]["probabilities"]["alpha"] = 0.9
    elif fault == "raw":
        result["samples"][5]["native_probabilities"]["synthetic_0"] = [0.1, 0.4, 0.5]
    elif fault == "usage":
        result["samples"][4]["response"]["usage"]["input_tokens"] = True
    elif fault == "model":
        result["samples"][4]["response"]["model"] = "other"
    elif fault == "metric":
        result["samples"][5]["metrics"]["padded_slots"] += 1
    elif fault == "comparison":
        result["comparisons"][0]["same_choices"] = False
    elif fault == "zero_time":
        result["samples"][4]["elapsed_ns"] = 0
    with pytest.raises(ValueError):
        execute(evidence, hash_result=(fault != "digest"))


def test_rejects_duplicate_keys_and_nonfinite_json(evidence):
    for bad in (b'{"x":1,"x":2}', b'{"x": NaN}'):
        with pytest.raises(ValueError):
            audit._decode(bad, 1024)


@pytest.mark.parametrize("status", ["failed", "cleanup_unconfirmed"])
def test_prelaunch_failure_without_invocation_is_incomplete(evidence, status):
    evidence["closure"]["status"] = status
    evidence["closure"]["invocation"] = None
    value = audit.audit(
        closure_bytes=encode(evidence["closure"]),
        terminal_bytes=None,
        result_bytes=None,
        hello_bytes=None,
        plan_sha=evidence["plan_sha"],
        invocation=None,
        model=None,
    )
    assert value["status"] == "incomplete" and "timing" not in value
    assert "synthetic input content" in value["scope"]
    assert "plan bytes" in value["scope"] and "hello unit and cgroup" in value["scope"]


def test_success_without_invocation_is_rejected(evidence):
    evidence["closure"]["invocation"] = None
    with pytest.raises(ValueError, match="successful invocation"):
        audit.audit(
            closure_bytes=encode(evidence["closure"]),
            terminal_bytes=None,
            result_bytes=None,
            hello_bytes=None,
            plan_sha=evidence["plan_sha"],
            invocation=None,
            model=None,
        )


def test_prelaunch_failure_still_requires_plan_and_invocation_match(evidence):
    evidence["closure"]["status"] = "failed"
    evidence["closure"]["invocation"] = None
    evidence["closure"]["plan_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="matching closure"):
        audit.audit(
            closure_bytes=encode(evidence["closure"]),
            terminal_bytes=None,
            result_bytes=None,
            hello_bytes=None,
            plan_sha=evidence["plan_sha"],
            invocation=None,
            model=None,
        )
