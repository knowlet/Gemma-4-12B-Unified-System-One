import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from s1.evaluation.artifacts import read_records
from s1.evaluation.contracts import ProfileSpec
from s1.evaluation.gates import assess_gate
from s1.evaluation.planning import create_plan
from s1.evaluation.registry import Registry
from s1.evaluation.runner import execute
from s1.evaluation.statistics import compare_runs, paired_comparison

ROOT = Path(__file__).resolve().parents[2]


def _assess(tmp_path, pair, **overrides):
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps({"comparisons": [pair]}))
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "left_experiment_id": pair["left_experiment_id"],
                "right_experiment_id": pair["right_experiment_id"],
                "accuracy_margin": 0.01,
                **overrides,
            }
        )
    )
    return assess_gate(comparison, gate)


def _pair(left, right):
    return {
        "left_experiment_id": "a" * 64,
        "right_experiment_id": "b" * 64,
        **paired_comparison(left, right, resamples=100),
    }


def _rows(count=30):
    return [
        {
            "case_id": f"case-{i}",
            "question_id": "q",
            "repetition": 0,
            "group_id": f"group-{i}",
            "task_id": None,
            "schema_id": None,
            "type": "choice",
            "gold": "yes",
            "actual_label": "yes",
            "status": "ok",
            "eligibility": "eligible",
        }
        for i in range(count)
    ]


def test_default_gate_rejects_90_percent_support_loss_through_real_execution(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    cases = []
    for i in range(300):
        question = (
            {"type": "choice", "instructions": "Choose yes", "criteria": {"yes": "yes", "no": "no"}}
            if i < 30
            else {"type": "noul", "instructions": "Choose false"}
        )
        cases.append(
            {
                "id": f"case-{i}",
                "group_id": f"group-{i}",
                "request": {"state": f"case {i}", "questions": {"q": question}},
                "gold": {"q": "yes" if i < 30 else False},
            }
        )
    dataset.write_text("\n".join(map(json.dumps, cases)))
    config = ROOT / "configs/benchmarks"
    registry = Registry.load(
        config / "models.toml", config / "suites.toml", config / "profiles.toml"
    )
    uniform = registry.models["uniform"]
    models = {
        "reference": uniform.model_copy(update={"id": "reference"}),
        "candidate": uniform.model_copy(
            update={
                "id": "candidate",
                "capabilities": uniform.capabilities.model_copy(update={"primitives": ("choice",)}),
            }
        ),
    }
    suite = registry.suites["smoke"].model_copy(update={"dataset": str(dataset)})
    profile = ProfileSpec(
        id="scope",
        suite="smoke",
        models=("reference", "candidate"),
        budget={"max_requests": 600, "max_estimated_cost_usd": 0.0},
    )
    registry = replace(
        registry, models=models, suites={"smoke": suite}, profiles={"scope": profile}
    )
    plan = create_plan(registry, "scope", lockfile=ROOT / "uv.lock", enable_models=list(models))
    assert plan["models"][0]["coverage"]["decisions"]["eligible"] == 300
    assert plan["models"][1]["coverage"]["decisions"]["unsupported"] == 270
    execute(
        registry,
        "scope",
        output=tmp_path / "run",
        lockfile=ROOT / "uv.lock",
        enable_models=list(models),
    )
    pair = compare_runs([tmp_path / "run"], resamples=100)["comparisons"][0]
    interval = pair["operational_accuracy_delta"]
    assert interval["groups"] == interval["observations"] == 30
    assert interval["estimate"] == 0
    assert interval["interval"] == [0, 0]
    rows = read_records(tmp_path / "run", "predictions")
    assert sum(row["status"] == "unsupported" for row in rows) == 270
    result = _assess(tmp_path, pair)
    assert result["status"] == "inconclusive"
    assert {"eligibility_mismatch", "population_not_fully_eligible", "insufficient_support"} <= set(
        result["reasons"]
    )
    assert result["population"]["decisions"] == 300
    assert result["population"]["joint_support_fraction"] == 0.1

    result = _assess(tmp_path, pair, population="common_eligible", min_support_fraction=0.1)
    assert result["status"] == "passed"
    assert result["reasons"] == []
    assert result["population"]["decisions"] == 30
    assert result["population"]["total_decisions"] == 300
    assert "subset only" in result["scope"]
    assert (
        _assess(tmp_path, pair, population="common_eligible", min_support_fraction=0.11)["status"]
        == "inconclusive"
    )


def test_identical_jointly_unsupported_rows_do_not_establish_full_population_acceptance(tmp_path):
    rows = _rows(60)
    for row in rows[30:]:
        row.update(status="unsupported", eligibility="unsupported", actual_label=None)
    pair = _pair(rows, copy.deepcopy(rows))
    assert pair["population"]["eligibility_mismatches"] == 0
    result = _assess(tmp_path, pair)
    assert result["status"] == "inconclusive"
    assert "population_not_fully_eligible" in result["reasons"]
    assert (
        _assess(tmp_path, pair, population="common_eligible", min_support_fraction=0.5)["status"]
        == "passed"
    )


@pytest.mark.parametrize(
    "change,reason", [("missing", "unmatched_decisions"), ("not_run", "unexecuted_decisions")]
)
@pytest.mark.parametrize("policy", ["all_decisions", "common_eligible"])
def test_missing_or_not_run_slots_cannot_disappear_from_gate(tmp_path, change, reason, policy):
    left = _rows(31)
    right = copy.deepcopy(left)
    if change == "missing":
        right.pop()
    else:
        right[-1].update(status="not_run", actual_label=None)
    pair = _pair(left, right)
    assert pair["operational_accuracy_delta"]["groups"] == 30
    result = _assess(tmp_path, pair, population=policy, min_support_fraction=0.5)
    assert result["status"] == "inconclusive"
    assert reason in result["reasons"]


def test_runtime_failures_stay_in_population_and_fail_noninferiority(tmp_path):
    left = _rows()
    right = copy.deepcopy(left)
    for row in right:
        row.update(status="timeout", actual_label=None)
    result = _assess(tmp_path, _pair(left, right))
    assert result["status"] == "failed"
    assert result["reasons"] == []
    assert result["population"]["joint_support_fraction"] == 1
    assert result["evidence"]["estimate"] == -1


def test_population_hash_pins_repetitions_and_the_selected_subset(tmp_path):
    left = _rows(60)
    right = copy.deepcopy(left)
    for row in right[30:]:
        row.update(status="unsupported", eligibility="unsupported", actual_label=None)
    pair = _pair(left, right)
    identity = pair["population"]["jointly_eligible_identity_sha256"]
    result = _assess(
        tmp_path,
        pair,
        population="common_eligible",
        min_support_fraction=0.5,
        population_sha256=identity,
    )
    assert result["status"] == "passed"
    assert result["population"]["identity_sha256"] == identity

    shifted = copy.deepcopy(right)
    shifted[0].update(status="unsupported", eligibility="unsupported", actual_label=None)
    shifted[30].update(status="ok", eligibility="eligible", actual_label="yes")
    changed = _pair(left, shifted)
    assert changed["population"]["jointly_eligible_decisions"] == 30
    result = _assess(
        tmp_path,
        changed,
        population="common_eligible",
        min_support_fraction=0.5,
        population_sha256=identity,
    )
    assert result["status"] == "inconclusive"
    assert "population_identity_mismatch" in result["reasons"]
    repeated_left, repeated_right = copy.deepcopy(left), copy.deepcopy(right)
    for row in repeated_left + repeated_right:
        row["repetition"] = 1
    repeated = _pair(repeated_left, repeated_right)
    assert repeated["population"]["identity_sha256"] != pair["population"]["identity_sha256"]
    assert repeated["population"]["jointly_eligible_identity_sha256"] != identity


def test_legacy_comparison_without_population_evidence_is_inconclusive(tmp_path):
    pair = _pair(_rows(), _rows())
    del pair["population"]
    result = _assess(tmp_path, pair)
    assert result["status"] == "inconclusive"
    assert result["reasons"] == ["missing_or_invalid_population_evidence"]
