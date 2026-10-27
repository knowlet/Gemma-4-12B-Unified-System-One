import copy
import hashlib

import jev_decision3_cpu_v2_result_audit as audit
import pytest
from test_jev_decision3_cpu_result_audit import encode, evidence  # noqa: F401


@pytest.fixture
def v2(evidence, monkeypatch):  # noqa: F811
    monkeypatch.setattr(audit, "SOURCE_ONLY", False)
    names = (
        "jev_decision3_cpu_experiment_worker",
        "jev_decision3_cpu_experiment_watcher",
        "jev_decision3_cpu_padding_candidate",
    )
    pins = {"/approved/" + name + "_v2_release.py": "a" * 64 for name in names}
    pins.update({f"/approved/source-{index}": "b" * 64 for index in range(16)})
    unit = "jev-cpu-" + "a" * 32 + ".service"
    group = "/user.slice/app.slice/" + unit
    plan = {
        "schema": "decision3-cpu-padding-experiment-plan-v2",
        "cpu_profile": "cpu10",
        "unit": unit,
        "output_dir": "/approved/run-" + "a" * 32 + "/worker",
        "python": "/home/albert_hsieh/ComfyUI/.venv/bin/python",
        "source_pins": pins,
        "wall_seconds": 600,
        "memory_floor_bytes": 34709107508,
        "rounds": 4,
    }
    plan_bytes = encode(plan)
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    evidence["result"]["cpu_profile"] = "cpu10"
    evidence["closure"]["plan_sha256"] = plan_sha
    evidence["hello"].update(plan_sha256=plan_sha, unit=unit, cgroup=group)
    evidence["terminal"]["plan_sha256"] = plan_sha
    stat = {"nr_periods": 10, "nr_throttled": 1, "throttled_usec": 100}
    observations = [
        {
            "ancestor_level": 0,
            "cpu_max": "1000000 100000",
            "cpu_stat": stat,
            "quota_status": "observed",
            "throttle_status": "observed",
        },
        {
            "ancestor_level": 1,
            "cpu_max": "500000 100000",
            "cpu_stat": stat,
            "quota_status": "observed",
            "throttle_status": "observed",
        },
        {
            "ancestor_level": 2,
            "cpu_max": "max 100000",
            "cpu_stat": stat,
            "quota_status": "observed",
            "throttle_status": "observed",
        },
        {
            "ancestor_level": 3,
            "cpu_max": None,
            "cpu_stat": None,
            "quota_status": "root_missing",
            "throttle_status": "unverified",
        },
    ]
    events = [
        {
            "event": "prelaunch",
            "running": 0,
            "pending": 0,
            "available_bytes": 40000000000,
            "monotonic": 1.0,
        },
        {
            "event": "host",
            "running": 0,
            "pending": 0,
            "available_bytes": 40000000000,
            "monotonic": 2.0,
        },
        {
            "event": "binding_verified",
            "invocation": evidence["invocation"],
            "pid": 123,
            "pgid": 123,
            "cgroup": group,
            "cpu_profile": "cpu10",
            "cpu_list": "5-9,15-19",
        },
        {"event": "cpu_observation", "status": "partial", "levels": observations},
    ]
    return {"evidence": evidence, "plan": plan, "events": events}


def execute(value, *, baseline=None):
    ev = value["evidence"]
    plan_bytes = encode(value["plan"])
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    result_bytes = encode(ev["result"])
    terminal = copy.deepcopy(ev["terminal"])
    terminal["result_sha256"] = hashlib.sha256(result_bytes).hexdigest()
    return audit.audit_v2(
        plan_bytes=plan_bytes,
        approved_source_pins=value["plan"]["source_pins"],
        events_bytes=b"\n".join(encode(event) for event in value["events"]),
        closure_bytes=encode(ev["closure"]),
        terminal_bytes=encode(terminal),
        result_bytes=result_bytes,
        hello_bytes=encode(ev["hello"]),
        plan_sha=plan_sha,
        invocation=ev["invocation"],
        model=ev["model"],
        baseline=baseline,
    )


def make_baseline(value):
    data = value["evidence"]
    plan = copy.deepcopy(value["plan"])
    plan["schema"] = "decision3-cpu-padding-experiment-plan-v1"
    plan.pop("cpu_profile")
    plan["source_pins"] = {
        path.replace("_v2_release.py", "_release.py"): digest
        for path, digest in plan["source_pins"].items()
    }
    raw_plan = encode(plan)
    sha = hashlib.sha256(raw_plan).hexdigest()
    result = copy.deepcopy(data["result"])
    result.pop("cpu_profile")
    result_raw = encode(result)
    closure, hello, terminal = (
        copy.deepcopy(data[name]) for name in ("closure", "hello", "terminal")
    )
    for item in (closure, hello, terminal):
        item["plan_sha256"] = sha
    terminal["result_sha256"] = hashlib.sha256(result_raw).hexdigest()
    return {
        "plan_bytes": raw_plan,
        "approved_source_pins": plan["source_pins"],
        "closure_bytes": encode(closure),
        "hello_bytes": encode(hello),
        "terminal_bytes": encode(terminal),
        "result_bytes": result_raw,
        "plan_sha": sha,
        "invocation": data["invocation"],
        "model": data["model"],
    }


def test_v2_records_partial_ancestors_and_prevents_false_exclusive_claim(v2):
    value = execute(v2)
    assert value["status"] == "complete" and value["cpu_profile"] == "cpu10"
    assert value["cpu_telemetry"]["smallest_observed_quota_cores"] == 5
    assert value["cpu_telemetry"]["ancestor_completeness"] == "incomplete_or_unverified"
    assert value["under_20_seconds"]["native"] == 4
    assert (
        "source" in value["scope"] and "not established" in value["cpu_telemetry"]["interpretation"]
    )


def test_v1_baseline_comparison_requires_matching_synthetic_inputs(v2):
    baseline = make_baseline(v2)
    value = execute(v2, baseline=baseline)
    assert value["baseline_comparison"]["serialized_inputs_and_model_label_match"] is True
    assert value["baseline_comparison"]["p50_ratio_v2_to_baseline"]["native"] == 1
    v2["evidence"]["result"]["state"]["note"] = "different"
    comparison = execute(v2, baseline=baseline)["baseline_comparison"]
    assert comparison["serialized_inputs_and_model_label_match"] is False
    assert "p50_ratio_v2_to_baseline" not in comparison


@pytest.mark.parametrize(
    "fault",
    [
        "plan_profile",
        "result_profile",
        "old_release",
        "wrong_hello",
        "wrong_binding",
        "wrong_cpu_list",
        "wrong_leaf_quota",
        "unsafe_host",
        "event_order",
        "bad_telemetry",
        "missing_telemetry",
    ],
)
def test_rejects_inconsistent_v2_profile_or_observations(v2, fault):
    if fault == "plan_profile":
        v2["plan"]["cpu_profile"] = "cpu5"
    elif fault == "result_profile":
        v2["evidence"]["result"]["cpu_profile"] = "cpu5"
    elif fault == "old_release":
        pins = v2["plan"]["source_pins"]
        old = next(path for path in pins if path.endswith("worker_v2_release.py"))
        pins[old.replace("_v2_release.py", "_release.py")] = pins.pop(old)
    elif fault == "wrong_hello":
        v2["evidence"]["hello"]["cgroup"] = "/wrong"
    elif fault == "wrong_binding":
        v2["events"][2]["invocation"] = "d" * 32
    elif fault == "wrong_cpu_list":
        v2["events"][2]["cpu_list"] = "15-19"
    elif fault == "wrong_leaf_quota":
        v2["events"][3]["levels"][0]["cpu_max"] = "500000 100000"
    elif fault == "unsafe_host":
        v2["events"][1]["running"] = 1
    elif fault == "event_order":
        v2["events"][1], v2["events"][2] = v2["events"][2], v2["events"][1]
    elif fault == "bad_telemetry":
        v2["events"][3]["status"] = "observed"
    elif fault == "missing_telemetry":
        v2["events"].pop()
    with pytest.raises(ValueError):
        execute(v2)


def test_unverified_telemetry_never_claims_ancestor_capacity(v2):
    v2["events"][3] = {"event": "cpu_observation", "status": "unverified"}
    info = execute(v2)["cpu_telemetry"]
    assert info["ancestor_completeness"] == "incomplete_or_unverified"
    assert info["smallest_observed_quota_cores"] is None


def test_failed_v2_closure_has_no_timing_or_required_events(v2):
    v2["evidence"]["closure"]["status"] = "failed"
    v2["events"] = []
    result = execute(v2)
    assert result["status"] == "incomplete" and "timing" not in result


def test_reordered_options_do_not_get_a_cross_run_ratio(v2):
    baseline = make_baseline(v2)
    question = v2["evidence"]["result"]["questions"]["synthetic_0"]
    question["criteria"] = dict(reversed(list(question["criteria"].items())))
    comparison = execute(v2, baseline=baseline)["baseline_comparison"]
    assert comparison["serialized_inputs_and_model_label_match"] is False
    assert "p50_ratio_v2_to_baseline" not in comparison


def test_plan_sha_and_approved_pin_inventory_are_required(v2):
    plan_bytes = encode(v2["plan"])
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    wrong_pins = dict(v2["plan"]["source_pins"])
    wrong_pins[next(iter(wrong_pins))] = "f" * 64
    with pytest.raises(ValueError, match="reviewed plan"):
        audit._plan_evidence(
            plan_bytes, plan_sha, wrong_pins, "decision3-cpu-padding-experiment-plan-v2", "cpu10"
        )
    with pytest.raises(ValueError, match="plan bytes"):
        audit._plan_evidence(
            plan_bytes,
            "c" * 64,
            v2["plan"]["source_pins"],
            "decision3-cpu-padding-experiment-plan-v2",
            "cpu10",
        )


def test_incomplete_baseline_cannot_produce_cross_run_ratio(v2):
    baseline = make_baseline(v2)
    closure = copy.deepcopy(v2["evidence"]["closure"])
    closure["status"] = "cleanup_unconfirmed"
    closure["plan_sha256"] = baseline["plan_sha"]
    baseline["closure_bytes"] = encode(closure)
    with pytest.raises(ValueError, match="completed baseline"):
        execute(v2, baseline=baseline)


def test_v2_source_only_gate_rejects_without_processing_input(monkeypatch):
    monkeypatch.setattr(audit, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="release required"):
        audit.audit_v2(
            plan_bytes=None,
            approved_source_pins=None,
            events_bytes=None,
            closure_bytes=None,
            terminal_bytes=None,
            result_bytes=None,
            hello_bytes=None,
            plan_sha=None,
            invocation=None,
            model=None,
        )


def test_truncated_ancestor_list_cannot_claim_completeness(v2):
    observation = v2["events"][3]
    observation["levels"] = observation["levels"][:1]
    observation["status"] = "observed"
    with pytest.raises(ValueError, match="complete ancestor levels"):
        execute(v2)


def test_noncanonical_cgroup_cannot_supply_ambiguous_depth(v2):
    group = v2["evidence"]["hello"]["cgroup"].replace("/user.slice/", "/user.slice/../")
    v2["evidence"]["hello"]["cgroup"] = group
    v2["events"][2]["cgroup"] = group
    with pytest.raises(ValueError, match="canonical cgroup"):
        execute(v2)
