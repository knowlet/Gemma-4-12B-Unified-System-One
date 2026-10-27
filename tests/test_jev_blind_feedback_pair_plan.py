"""Synthetic contract tests only; no benchmark cases or external resources."""

from copy import deepcopy

import pytest

from scripts import jev_blind_feedback_pair_plan as subject


def synthetic_plan():
    pins = {name: "a" * 64 for name in subject._PIN_NAMES}
    arm = {
        "review_enabled": True,
        "feedback_enabled": False,
        "pins": pins,
        "configuration_sha256": "b" * 64,
        "budgets": {
            "actions": 16,
            "policy_calls": 16,
            "text_calls": 4,
            "review_calls": 2,
            "time_ms": 60000,
        },
    }
    on = deepcopy(arm)
    on["feedback_enabled"] = True
    return {
        "schema": "synthetic-feedback-pair-plan-v1",
        "exposure_status": "unknown",
        "population_sha256": "c" * 64,
        "selection_commitment_sha256": "d" * 64,
        "ordering_commitment_sha256": "e" * 64,
        "selected_cases": 2,
        "planned_runs": 4,
        "arms": {"off": arm, "on": on},
        "schedule": [
            {"token": "1" * 32, "seed": 1, "order": ["off", "on"]},
            {"token": "2" * 32, "seed": 2, "order": ["on", "off"]},
        ],
    }


def test_valid_synthetic_plan_is_only_declared_consistency():
    plan = synthetic_plan()
    before = deepcopy(plan)
    assert subject.SOURCE_ONLY is True
    assert subject.validate_plan(plan) == {
        "status": "supplied_plan_consistent",
        "exposure_status": "unknown",
        "selected_cases": 2,
        "planned_runs": 4,
        "authenticity_and_preregistration": "unverified",
        "runtime_conformance": "unverified",
    }
    assert plan == before


def change(path, value):
    def mutate(plan):
        current = plan
        for key in path[:-1]:
            current = current[key]
        current[path[-1]] = value

    return mutate


@pytest.mark.parametrize(
    "mutate",
    [
        change(["exposure_status"], "unseen"),
        change(["population_sha256"], "not a hash"),
        change(["selection_commitment_sha256"], "f" * 63),
        change(["ordering_commitment_sha256"], "G" * 64),
        change(["selected_cases"], True),
        change(["selected_cases"], 3),
        change(["planned_runs"], 2),
        change(["arms", "off", "review_enabled"], False),
        change(["arms", "on", "feedback_enabled"], False),
        change(["arms", "on", "pins", "model"], "f" * 64),
        change(["arms", "on", "pins", "case_and_judge"], "f" * 64),
        change(["arms", "off", "pins", "runner"], None),
        change(["arms", "on", "configuration_sha256"], "f" * 64),
        change(["arms", "on", "budgets", "policy_calls"], 17),
        change(["arms", "on", "budgets", "review_calls"], 0),
        change(["arms", "on", "budgets", "time_ms"], True),
        change(["schedule", 0, "token"], "https://example.com/"),
        change(["schedule", 0, "token"], "2" * 32),
        change(["schedule", 0, "seed"], True),
        change(["schedule", 0, "order"], ["on", "on"]),
        change(["schedule", 0, "order"], ["off"]),
        change(["schedule"], []),
    ],
)
def test_inconsistent_or_unproven_plan_rejected(mutate):
    plan = synthetic_plan()
    mutate(plan)
    with pytest.raises(ValueError):
        subject.validate_plan(plan)


def test_extra_fields_and_missing_source_pins_rejected():
    plan = synthetic_plan()
    plan["arms"]["on"]["other_setting"] = True
    with pytest.raises(ValueError):
        subject.validate_plan(plan)
    plan = synthetic_plan()
    del plan["arms"]["on"]["pins"]["case_and_judge"]
    with pytest.raises(ValueError):
        subject.validate_plan(plan)
    plan = synthetic_plan()
    plan["schedule"][0]["goal"] = "must not be supplied"
    with pytest.raises(ValueError):
        subject.validate_plan(plan)
