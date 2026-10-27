"""Validate a synthetic feedback-pair plan supplied entirely in memory.

This source-only validator does not access data, files, network, a model, or a
browser. It checks internal consistency only: the caller must independently
establish pin authenticity, preregistration, token secrecy, population
eligibility, original-judge identity, and that actual runs follow the plan.
Equal supplied pins do not establish any of those facts. Exposure is always
unknown; this interface cannot certify an unseen case or a valid blind trial.
"""

import re

SOURCE_ONLY = True

_PIN_NAMES = frozenset(
    {
        "runner",
        "case_and_judge",
        "model",
        "model_config",
        "semantic_adapter",
        "review_owner",
        "review_route",
        "text_helper",
        "browser_harness",
        "supervisor",
        "selector",
    }
)
_BUDGET_NAMES = frozenset({"actions", "policy_calls", "text_calls", "review_calls", "time_ms"})


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _exact_dict(value, keys, message):
    _need(type(value) is dict and set(value) == set(keys), message)


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


def _arm(value, feedback):
    _exact_dict(
        value,
        {"review_enabled", "feedback_enabled", "pins", "configuration_sha256", "budgets"},
        "exact arm configuration required",
    )
    _need(value["review_enabled"] is True, "both arms must enable review")
    _need(value["feedback_enabled"] is feedback, "explicit feedback setting required")
    _exact_dict(value["pins"], _PIN_NAMES, "complete source and judge pins required")
    _need(all(_digest(v) for v in value["pins"].values()), "invalid source pin")
    _need(_digest(value["configuration_sha256"]), "invalid common configuration pin")
    _exact_dict(value["budgets"], _BUDGET_NAMES, "complete budgets required")
    _need(
        all(_integer(v, 0, 3_600_000) for v in value["budgets"].values())
        and value["budgets"]["policy_calls"] > 0
        and value["budgets"]["review_calls"] > 0
        and value["budgets"]["time_ms"] > 0,
        "invalid budgets",
    )


def validate_plan(plan):
    """Validate declared pair settings; return only non-sensitive aggregate status.

    ``schedule`` contains each opaque token once, a shared case seed, and the
    order of its two planned arms. A token's format cannot establish entropy or
    secrecy, and commitments cannot establish when they were made. No result
    returned by this function is evidence of runtime or source authenticity.
    """
    _exact_dict(
        plan,
        {
            "schema",
            "exposure_status",
            "population_sha256",
            "selection_commitment_sha256",
            "ordering_commitment_sha256",
            "selected_cases",
            "planned_runs",
            "arms",
            "schedule",
        },
        "exact plan fields required",
    )
    _need(
        type(plan["schema"]) is str and plan["schema"] == "synthetic-feedback-pair-plan-v1",
        "invalid plan schema",
    )
    _need(
        type(plan["exposure_status"]) is str and plan["exposure_status"] == "unknown",
        "case exposure must remain unknown",
    )
    for key in ("population_sha256", "selection_commitment_sha256", "ordering_commitment_sha256"):
        _need(_digest(plan[key]), "invalid population or commitment pin")
    count = plan["selected_cases"]
    _need(_integer(count, 1, 1024), "invalid preregistered denominator")
    _need(
        _integer(plan["planned_runs"], 2, 2048) and plan["planned_runs"] == 2 * count,
        "both planned arms must be in the denominator",
    )
    arms = plan["arms"]
    _exact_dict(arms, {"off", "on"}, "exactly two feedback arms required")
    _arm(arms["off"], False)
    _arm(arms["on"], True)
    for key in ("pins", "configuration_sha256", "budgets", "review_enabled"):
        _need(arms["off"][key] == arms["on"][key], "arm configurations differ")
    schedule = plan["schedule"]
    _need(type(schedule) is list and len(schedule) == count, "incomplete schedule")
    seen = set()
    for pair in schedule:
        _exact_dict(pair, {"token", "seed", "order"}, "exact paired schedule required")
        token = pair["token"]
        _need(
            type(token) is str and re.fullmatch(r"[0-9a-f]{32}", token) is not None,
            "opaque token format required",
        )
        _need(token not in seen, "duplicate case token")
        seen.add(token)
        _need(_integer(pair["seed"], 0, 2**64 - 1), "invalid shared case seed")
        _need(
            type(pair["order"]) is list
            and all(type(value) is str for value in pair["order"])
            and pair["order"] in (["off", "on"], ["on", "off"]),
            "invalid arm order",
        )
    return {
        "status": "supplied_plan_consistent",
        "exposure_status": "unknown",
        "selected_cases": count,
        "planned_runs": plan["planned_runs"],
        "authenticity_and_preregistration": "unverified",
        "runtime_conformance": "unverified",
    }
