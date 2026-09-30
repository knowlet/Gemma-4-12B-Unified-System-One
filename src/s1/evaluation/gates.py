"""Apply explicitly supplied noninferiority gates; uncertainty is never a pass."""

import json
from pathlib import Path

from pydantic import Field

from s1.contracts import StrictModel


class Gate(StrictModel):
    left_experiment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    right_experiment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    accuracy_margin: float = Field(ge=0, le=1)
    min_groups: int = Field(default=30, ge=2)


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
    result = "inconclusive"
    if interval["groups"] >= gate.min_groups and interval["interval"] is not None:
        lower, upper = interval["interval"]
        if lower >= -gate.accuracy_margin:
            result = "passed"
        elif upper < -gate.accuracy_margin:
            result = "failed"
    return {
        "status": result,
        "gate": gate.model_dump(),
        "evidence": interval,
        "scope": "operational accuracy only; does not imply latency, critical-error or retention acceptance",
    }
