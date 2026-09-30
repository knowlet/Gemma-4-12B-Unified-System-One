import hashlib
import json
from pathlib import Path

import pytest

from s1.evaluation.adapters import load_adapter
from s1.evaluation.calibration import base_identity
from s1.evaluation.checkpoints import directory_digest
from s1.evaluation.contracts import ModelSpec, ProfileSpec, SuiteSpec
from s1.evaluation.datasets import EvaluationCase, request_fingerprint
from s1.evaluation.experiments import write_cases
from s1.evaluation.planning import create_plan
from s1.evaluation.registry import Registry
from s1.evaluation.supervision import fit_baseline
from s1.evaluation.workflow import Episode, execute_workflows

ROOT = Path(__file__).resolve().parents[2]
CAPABILITIES = {
    "modalities": ["text"],
    "primitives": ["choice"],
    "max_options": 2,
    "max_questions": 1,
    "probabilities": "complete",
}


def request(state):
    return {
        "state": state,
        "questions": [
            {
                "id": "action",
                "type": "choice",
                "instructions": "Choose next action",
                "criteria": {"complete": "Declare completion", "escalate": "Ask worker"},
            }
        ],
    }


def scenario():
    return Episode.model_validate(
        {
            "id": "episode-test",
            "group_id": "source-test",
            "family": "ci",
            "seed": 0,
            "initial_state": "initial",
            "max_steps": 1,
            "states": {
                "initial": {"observation": request("initial test"), "goal_satisfied": True},
                # Every state must be audited, including one outside the observed replay.
                "later": {"observation": request("later test"), "goal_satisfied": True},
            },
        }
    )


def uniform(**updates):
    return ModelSpec(
        **{
            "id": "uniform",
            "adapter": "uniform",
            "enabled": True,
            "precision": "float64",
            "execution_mode": "sequential",
            "calibration": "none",
            "capabilities": CAPABILITIES,
            **updates,
        }
    )


def registry(spec, case, tmp_path):
    write_cases(tmp_path / "test.jsonl", [case])
    return Registry(
        models={spec.id: spec},
        suites={
            "test": SuiteSpec(
                id="test", dataset="test.jsonl", track="generalist", information_view="raw_text"
            )
        },
        profiles={"test": ProfileSpec(id="test", suite="test", models=(spec.id,))},
        suite_directory=tmp_path,
        baseline_revision="f8390676ae942699a71a85ae9a99cfcd7b9b3806",
    )


def evaluation_case(episode):
    return EvaluationCase(
        id=episode.id,
        group_id=episode.group_id,
        request=episode.states["later"].observation,
        gold={"action": "complete"},
    )


def run(reg, episode, tmp_path, **kwargs):
    path = tmp_path / "episodes.jsonl"
    path.write_text(episode.model_dump_json() + "\n")
    return execute_workflows(
        reg,
        path,
        next(iter(reg.models)),
        next(iter(reg.models)),
        output=tmp_path / "workflow",
        lockfile=ROOT / "uv.lock",
        max_cost_usd=1.0,
        **kwargs,
    )


@pytest.mark.parametrize("overlap", ["id", "group", "request"])
def test_workflow_matches_planner_training_overlap_before_loading(tmp_path, overlap):
    episode = scenario()
    heldout = evaluation_case(episode)
    training = [
        EvaluationCase(
            id=episode.id if overlap == "id" else f"train-{i}",
            group_id=episode.group_id if overlap == "group" else f"source-train-{i}",
            split="train",
            request=heldout.request if overlap == "request" and i == 0 else request(f"train {i}"),
            gold={"action": label},
        )
        for i, label in enumerate(("complete", "escalate"))
    ]
    # Unique IDs are required for the real training path.
    if overlap == "id":
        training[1].id = "train-1"
    write_cases(tmp_path / "train.jsonl", training)
    fit = fit_baseline(tmp_path / "train.jsonl", tmp_path / "baseline", method="prior")
    spec = uniform(
        id="prior",
        adapter="prior",
        cost_per_request_usd=0.0,
        artifact_file=fit["artifact_file"],
        artifact_sha256=fit["artifact_sha256"],
    )
    reg = registry(spec, heldout, tmp_path)
    plan = create_plan(reg, "test", lockfile=ROOT / "uv.lock")
    assert "evaluation_overlaps_training" in plan["models"][0]["reasons"]
    loaded = []

    def factory(spec, environment):
        loaded.append(spec.id)
        return load_adapter(spec, environment)

    result = run(reg, episode, tmp_path, adapter_factory=factory)
    assert result["status"] == "not_run"
    assert "evaluation_overlaps_training" in result["reasons"]
    assert loaded == []


class RecordingAdapter:
    def __init__(self, telemetry):
        self.resolved = telemetry
        self.calls = 0
        self.closed = False

    def telemetry(self):
        return self.resolved

    def predict(self, request):
        self.calls += 1
        return {
            "answers": {
                "action": {
                    "choice": "complete",
                    "probabilities": {"complete": 1.0, "escalate": 0.0},
                }
            },
            "cost_usd": 0,
        }

    def close(self):
        self.closed = True


def test_workflow_precision_mismatch_prevents_inference_and_persists_failure(tmp_path):
    episode = scenario()
    reg = registry(uniform(), evaluation_case(episode), tmp_path)
    adapter = RecordingAdapter({"precision": "float32", "context_policy": "not_applicable"})
    result = run(reg, episode, tmp_path, adapter_factory=lambda *_: adapter)
    assert result["status"] == "setup_error"
    assert adapter.calls == 0
    assert adapter.closed
    manifest = json.loads((tmp_path / "workflow/workflow_manifest.json").read_text())
    assert manifest["execution"]["uniform"]["status"] == "setup_error"
    assert "adapter_setup_failed" in manifest["execution"]["uniform"]["reasons"]
    assert manifest["execution"]["uniform"]["error_type"] == "ValueError"


@pytest.mark.parametrize("kind", ["gemma", "laya"])
def test_workflow_revision_mismatch_prevents_inference(tmp_path, kind):
    episode = scenario()
    spec = uniform(
        id=kind,
        adapter=kind,
        model_id="pinned-fixture",
        revision="a" * 40,
        precision="float32" if kind == "gemma" else "provider",
        execution_mode="sequential" if kind == "gemma" else "provider",
        calibration="none" if kind == "gemma" else "checkpoint",
        device="cpu",
        cost_per_request_usd=0.0,
    )
    reg = registry(spec, evaluation_case(episode), tmp_path)
    adapter = RecordingAdapter(
        {"revision": "b" * 40, "precision": spec.precision, "context_policy": "reject"}
    )
    result = run(reg, episode, tmp_path, adapter_factory=lambda *_: adapter)
    assert result["status"] == "setup_error"
    assert adapter.calls == 0
    assert adapter.closed


def test_workflow_validated_telemetry_is_persisted_before_first_prediction(tmp_path):
    episode = scenario()
    reg = registry(uniform(), evaluation_case(episode), tmp_path)
    observed = {
        "revision": "local-fixture",
        "device": "cpu",
        "precision": "float64",
        "temperature": 1.0,
        "context_limit": 512,
        "context_policy": "reject",
    }

    class Persisted(RecordingAdapter):
        def predict(self, request):
            manifest = json.loads((tmp_path / "workflow/workflow_manifest.json").read_text())
            assert manifest["execution"]["uniform"]["telemetry"] == {
                **observed,
                "calibration_temperature": None,
            }
            return super().predict(request)

    adapter = Persisted(observed)
    result = run(reg, episode, tmp_path, adapter_factory=lambda *_: adapter)
    assert result["status"] == "completed"
    assert adapter.calls == 4
    assert adapter.closed
    manifest = json.loads((tmp_path / "workflow/workflow_manifest.json").read_text())
    assert manifest["execution"]["uniform"]["status"] == "completed"


@pytest.mark.parametrize("split", ["train", "calibration"])
@pytest.mark.parametrize("overlap", ["id", "group", "request"])
def test_all_workflow_states_reject_adapter_fitting_overlap(tmp_path, split, overlap):
    episode = scenario()
    case = evaluation_case(episode)
    provenance = {
        f"{part}_{kind}": []
        for part in ("train", "calibration")
        for kind in ("ids", "groups", "requests")
    }
    values = {"id": case.id, "group": case.group_id, "request": request_fingerprint(case.request)}
    keys = {"id": "ids", "group": "groups", "request": "requests"}
    provenance[f"{split}_{keys[overlap]}"] = [values[overlap]]
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "training-provenance.json").write_text(json.dumps(provenance))
    spec = uniform(
        id="gemma",
        adapter="gemma",
        model_id="fixture",
        revision="a" * 40,
        precision="float32",
        device="cpu",
        cost_per_request_usd=0.0,
        decision_adapter_path=str(checkpoint),
        decision_adapter_sha256=directory_digest(checkpoint),
    )
    reg = registry(spec, case, tmp_path)
    assert (
        "evaluation_overlaps_adapter_training"
        in create_plan(reg, "test", lockfile=ROOT / "uv.lock")["models"][0]["reasons"]
    )
    result = run(reg, episode, tmp_path, adapter_factory=lambda *_: pytest.fail("loaded overlap"))
    assert result["status"] == "not_run"
    assert "evaluation_overlaps_adapter_training" in result["reasons"]


@pytest.mark.parametrize("overlap", ["id", "group", "request"])
def test_workflow_rejects_domain_calibration_overlap_before_loading(tmp_path, overlap):
    episode = scenario()
    case = evaluation_case(episode)
    calibration_file = tmp_path / "calibration.json"
    spec = uniform(
        calibration="domain",
        base_calibration="none",
        calibration_file=str(calibration_file),
        calibration_sha256="0" * 64,
    )
    calibration = {
        "base_identity": base_identity(spec),
        "split": "calibration",
        "transform": "epsilon_regularized_probability_power",
        "floor": 1e-12,
        "temperature": 1.0,
        "case_ids": [case.id] if overlap == "id" else [],
        "group_ids": [case.group_id] if overlap == "group" else [],
        "request_sha256": [request_fingerprint(case.request)] if overlap == "request" else [],
    }
    calibration_file.write_text(json.dumps(calibration))
    spec = spec.model_copy(
        update={"calibration_sha256": hashlib.sha256(calibration_file.read_bytes()).hexdigest()}
    )
    reg = registry(spec, case, tmp_path)
    assert (
        "invalid_or_overlapping_calibration"
        in create_plan(reg, "test", lockfile=ROOT / "uv.lock")["models"][0]["reasons"]
    )
    result = run(reg, episode, tmp_path, adapter_factory=lambda *_: pytest.fail("loaded overlap"))
    assert result["status"] == "not_run"
    assert "invalid_or_overlapping_calibration" in result["reasons"]


def test_workflow_validates_both_models_before_inference_and_closes_both(tmp_path):
    episode = scenario()
    small, strong = uniform(id="small"), uniform(id="strong")
    reg = registry(small, evaluation_case(episode), tmp_path)
    reg.models[strong.id] = strong
    adapters = {
        "small": RecordingAdapter({"precision": "float64", "context_policy": "not_applicable"}),
        "strong": RecordingAdapter({"precision": "float32", "context_policy": "not_applicable"}),
    }
    path = tmp_path / "episodes.jsonl"
    path.write_text(episode.model_dump_json() + "\n")
    result = execute_workflows(
        reg,
        path,
        small.id,
        strong.id,
        output=tmp_path / "workflow",
        lockfile=ROOT / "uv.lock",
        adapter_factory=lambda spec, _: adapters[spec.id],
    )
    assert result["status"] == "setup_error"
    assert all(adapter.calls == 0 and adapter.closed for adapter in adapters.values())
    manifest = json.loads((tmp_path / "workflow/workflow_manifest.json").read_text())
    assert manifest["execution"]["small"]["telemetry"]["precision"] == "float64"
    assert manifest["execution"]["strong"]["status"] == "setup_error"
    assert not (tmp_path / "workflow/episodes.jsonl").exists()


def test_workflow_manifest_marks_adapter_with_routed_inference_errors(tmp_path):
    episode = scenario()
    reg = registry(uniform(), evaluation_case(episode), tmp_path)

    class Failing(RecordingAdapter):
        def predict(self, request):
            self.calls += 1
            raise ValueError("private request contents")

    adapter = Failing({"precision": "float64", "context_policy": "not_applicable"})
    result = run(reg, episode, tmp_path, adapter_factory=lambda *_: adapter)
    assert result["status"] == "completed"
    manifest_path = tmp_path / "workflow/workflow_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["execution"]["uniform"]["status"] == "completed_with_errors"
    assert "private request contents" not in manifest_path.read_text()
    assert adapter.closed
