"""Apply explicitly supplied noninferiority gates; uncertainty is never a pass."""

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from s1.contracts import StrictModel


class Gate(StrictModel):
    left_experiment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    right_experiment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    accuracy_margin: float = Field(ge=0, le=1)
    min_groups: int = Field(default=30, ge=2)
    population: Literal["all_decisions", "common_eligible"] = "all_decisions"
    min_support_fraction: float = Field(default=1, gt=0, le=1)
    population_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class PopulationEvidence(StrictModel):
    decisions: int = Field(ge=0, strict=True)
    groups: int = Field(ge=0, strict=True)
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    jointly_eligible_decisions: int = Field(ge=0, strict=True)
    jointly_eligible_groups: int = Field(ge=0, strict=True)
    jointly_eligible_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    eligibility_mismatches: int = Field(ge=0, strict=True)
    left_only: int = Field(ge=0, strict=True)
    right_only: int = Field(ge=0, strict=True)
    left_not_run: int = Field(ge=0, strict=True)
    right_not_run: int = Field(ge=0, strict=True)


def _assess_population(comparison, gate, interval):
    try:
        population = PopulationEvidence.model_validate(comparison.get("population"))
    except ValidationError:
        return ["missing_or_invalid_population_evidence"], None
    reasons = []
    if population.left_only or population.right_only:
        reasons.append("unmatched_decisions")
    if population.left_not_run or population.right_not_run:
        reasons.append("unexecuted_decisions")
    if gate.population == "all_decisions":
        decisions, groups, identity = (
            population.decisions,
            population.groups,
            population.identity_sha256,
        )
        if population.eligibility_mismatches:
            reasons.append("eligibility_mismatch")
        if population.jointly_eligible_decisions != population.decisions:
            reasons.append("population_not_fully_eligible")
    else:
        decisions, groups, identity = (
            population.jointly_eligible_decisions,
            population.jointly_eligible_groups,
            population.jointly_eligible_identity_sha256,
        )
    support = (
        population.jointly_eligible_decisions / population.decisions if population.decisions else 0
    )
    if not 0 < support <= 1 or support < gate.min_support_fraction:
        reasons.append("insufficient_support")
    if gate.population_sha256 is not None and identity != gate.population_sha256:
        reasons.append("population_identity_mismatch")
    # A confidence interval must represent every decision in the declared population.
    if interval.get("observations") != decisions or interval["groups"] != groups:
        reasons.append("population_evidence_mismatch")
    return reasons, {
        "policy": gate.population,
        "decisions": decisions,
        "groups": groups,
        "identity_sha256": identity,
        "total_decisions": population.decisions,
        "joint_support_fraction": support,
        "coverage": population.model_dump(),
    }


def assess_gate(comparison_path, gate_path):
    comparison = json.loads(Path(comparison_path).read_text())
    gate = Gate.model_validate_json(Path(gate_path).read_text())
    matches = [
        r
        for r in comparison["comparisons"]
        if r["left_experiment_id"] == gate.left_experiment_id
        and r["right_experiment_id"] == gate.right_experiment_id
    ]
    if len(matches) != 1:
        raise ValueError("gate must select exactly one pinned experiment pair")
    interval = matches[0]["operational_accuracy_delta"]
    reasons, population = _assess_population(matches[0], gate, interval)
    result = "inconclusive"
    if not reasons and interval["groups"] >= gate.min_groups and interval["interval"] is not None:
        lower, upper = interval["interval"]
        if lower >= -gate.accuracy_margin:
            result = "passed"
        elif upper < -gate.accuracy_margin:
            result = "failed"
    return {
        "status": result,
        "gate": gate.model_dump(),
        "evidence": interval,
        "population": population,
        "reasons": reasons,
        "scope": (
            "operational accuracy on all declared decision slots"
            if gate.population == "all_decisions"
            else "operational accuracy on the declared common-eligible subset only"
        )
        + "; does not imply latency, critical-error or retention acceptance",
    }
