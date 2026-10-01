import json
from dataclasses import replace
from pathlib import Path

import pytest

from s1.evaluation.calibration import fit_from_run, validate_calibration
from s1.evaluation.contracts import ModelSpec, ProfileSpec
from s1.evaluation.datasets import load_cases
from s1.evaluation.registry import Registry
from s1.evaluation.runner import execute

ROOT = Path(__file__).resolve().parents[2]


def registry():
    directory = ROOT / "configs/benchmarks"
    return Registry.load(
        directory / "models.toml", directory / "suites.toml", directory / "profiles.toml"
    )


def test_temperature_fit_rejects_test_run(tmp_path):
    execute(
        registry(),
        "smoke",
        output=tmp_path / "test",
        lockfile=ROOT / "uv.lock",
        enable_models=["uniform"],
    )
    with pytest.raises(ValueError, match="calibration-only"):
        fit_from_run(tmp_path / "test", "uniform", tmp_path / "temperature.json")


def test_fit_reload_and_overlap_guards(tmp_path):
    reg = registry()
    cases = load_cases(reg.dataset_path(reg.suites["smoke"]))
    calibration_cases = []
    for case in cases:
        data = case.model_dump()
        data.update(id="cal-" + case.id, group_id="cal-" + case.id, split="calibration")
        data["request"]["state"] = "Calibration-only fixture: " + str(data["request"]["state"])
        calibration_cases.append(data)
    dataset = tmp_path / "calibration.jsonl"
    dataset.write_text("\n".join(json.dumps(case) for case in calibration_cases))
    profile = ProfileSpec(id="cal", suite="smoke", split="calibration", models=("uniform",))
    cal_reg = replace(
        reg,
        suites={"smoke": reg.suites["smoke"].model_copy(update={"dataset": str(dataset)})},
        profiles={"cal": profile},
    )
    execute(
        cal_reg,
        "cal",
        output=tmp_path / "cal-run",
        lockfile=ROOT / "uv.lock",
        enable_models=["uniform"],
    )
    fitted = fit_from_run(tmp_path / "cal-run", "uniform", tmp_path / "temperature.json")
    data = reg.models["uniform"].model_dump()
    data.update(
        calibration="domain",
        base_calibration="none",
        calibration_file=str(tmp_path / "temperature.json"),
        calibration_sha256=fitted["artifact_sha256"],
    )
    spec = ModelSpec.model_validate(data)
    assert validate_calibration(spec, cases)["temperature"] > 0
    with pytest.raises(ValueError, match="overlaps"):
        validate_calibration(spec, load_cases(dataset))
    domain_reg = replace(reg, models={"uniform": spec})
    result = execute(
        domain_reg,
        "smoke",
        output=tmp_path / "calibrated-test",
        lockfile=ROOT / "uv.lock",
        enable_models=["uniform"],
    )
    assert result["run_status"] == "completed"
    assert (
        result["models"]["uniform"]["telemetry"]["calibration_temperature"] == fitted["temperature"]
    )
    with pytest.raises(ValueError, match="mismatch"):
        validate_calibration(spec.model_copy(update={"revision": "different"}))
