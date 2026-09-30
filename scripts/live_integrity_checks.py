"""Actual runtime telemetry and real fitted-artifact workflow boundary checks.

The caller owns a loaded CUDA FP32 checkpoint and the completed LoRA training
artifacts. This module neither starts cloud work nor closes the borrowed model.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _observation(state):
    from s1.contracts import DecisionRequest

    return DecisionRequest.model_validate(
        {
            "state": state,
            "questions": [
                {
                    "id": "action",
                    "type": "choice",
                    "instructions": "Choose the next action based on the visible status.",
                    "criteria": {
                        "complete": "All required checks passed; declare completion.",
                        "escalate": "The visible checks are incomplete; ask the worker.",
                    },
                }
            ],
        }
    )


def _episode(identifier, group, *, later_request=None):
    from s1.evaluation.workflow import Episode

    initial = _observation("All required checks have passed. The task is complete.")
    later = later_request or _observation("A separate unreachable verification state.")
    return Episode.model_validate(
        {
            "id": identifier,
            "group_id": group,
            "family": "ci",
            "seed": 0,
            "initial_state": "initial",
            "max_steps": 1,
            "states": {
                "initial": {"observation": initial, "goal_satisfied": True},
                "unreachable": {"observation": later, "goal_satisfied": True},
            },
        }
    )


def run(model, root: Path, output: Path, dataset_dir: Path, training_dir: Path):
    """Verify actual FP32 telemetry and completed training-artifact provenance."""
    import torch

    from s1.evaluation.adapters import ReferenceAdapter
    from s1.evaluation.checkpoints import directory_digest
    from s1.evaluation.contracts import ModelSpec
    from s1.evaluation.datasets import EvaluationCase, fingerprint, load_cases, request_fingerprint
    from s1.evaluation.experiments import write_cases
    from s1.evaluation.integrity import fitting_provenance_blockers
    from s1.evaluation.registry import Registry
    from s1.evaluation.supervision import fit_baseline
    from s1.evaluation.workflow import execute_workflows

    root, output = Path(root), Path(output)
    dataset_dir, training_dir = Path(dataset_dir), Path(training_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {
        "status": "running",
        "checkpoint": model.name,
        "revision": model.revision,
        "precision": str(model.head.weight.dtype).removeprefix("torch."),
        "checks": [],
        "trained_artifacts": [],
        "limitations": [
            "Workflow fixtures validate boundaries and telemetry, not representative task quality.",
            "Blocked adapter fixtures validate real fitted provenance without loading adapter weights.",
            "BoolQ fitting requests are NOUL; request-only workflow overlap is exercised with a real fitted Choice prior.",
            "This module borrows the caller's checkpoint and does not close or replace it.",
        ],
    }
    report_path = output / "integrity-checks.json"

    def persist(stage):
        report["stage"] = stage
        report["wall_seconds"] = time.perf_counter() - started
        _write(report_path, report)
        print(
            json.dumps(
                {
                    "check": "live_integrity",
                    "stage": stage,
                    "wall_seconds": round(report["wall_seconds"], 3),
                }
            ),
            flush=True,
        )

    def record(name, **evidence):
        report["checks"].append({"name": name, "status": "passed", **evidence})
        persist(name)

    def workflow(registry, spec, episode, name, *, load_actual):
        directory = output / name
        source = output / f"{name}.episodes.jsonl"
        source.write_text(episode.model_dump_json() + "\n", encoding="utf-8")
        actual_forwards, factory_calls, manifest_checks = [0], [0], [0]
        loaded = []

        def observed_backbone(*args):
            actual_forwards[0] += 1

        class Borrowed(ReferenceAdapter):
            def __init__(self, requested):
                super().__init__(requested, SimpleNamespace(model=model))
                self.closed = False

            def predict(self, request):
                manifest = json.loads((directory / "workflow_manifest.json").read_text())
                telemetry = manifest["execution"][self.spec.id]["telemetry"]
                if telemetry["revision"] != model.revision or telemetry["precision"] != "float32":
                    raise AssertionError("actual telemetry was not persisted before prediction")
                if manifest["execution"][self.spec.id]["status"] != "ready":
                    raise AssertionError("workflow dispatched a model before its setup was ready")
                manifest_checks[0] += 1
                return super().predict(request)

            def close(self):
                # Only this wrapper closes; the caller's CUDA model stays alive.
                self.backend = None
                self.closed = True

        def factory(requested, environment):
            factory_calls[0] += 1
            if not load_actual:
                raise AssertionError("provenance overlap must block before adapter loading")
            adapter = Borrowed(requested)
            loaded.append(adapter)
            return adapter

        hook = model.backbone.register_forward_pre_hook(observed_backbone)
        try:
            result = execute_workflows(
                replace(registry, models={spec.id: spec}),
                source,
                spec.id,
                spec.id,
                output=directory,
                lockfile=root / "uv.lock",
                max_calls=5,
                max_cost_usd=1.0,
                adapter_factory=factory,
            )
        finally:
            hook.remove()
        manifest = json.loads((directory / "workflow_manifest.json").read_text())
        return {
            "status": result["status"],
            "reasons": result["reasons"],
            "actual_backbone_forwards": actual_forwards[0],
            "adapter_factory_calls": factory_calls[0],
            "manifest_checks_before_prediction": manifest_checks[0],
            "wrappers_closed": all(adapter.closed for adapter in loaded),
            "execution": manifest["execution"],
            "manifest": str((directory / "workflow_manifest.json").relative_to(output)),
            "episode_has_unreachable_state": "unreachable" in episode.states,
        }

    try:
        if not str(model.device).startswith("cuda") or model.head.weight.dtype != torch.float32:
            raise ValueError("live integrity checks require the caller's actual CUDA FP32 model")
        if len(model.revision or "") not in (40, 64):
            raise ValueError("live integrity checks require an immutable checkpoint revision")
        configs = root / "configs/benchmarks"
        registry = Registry.load(
            configs / "models.toml", configs / "suites.toml", configs / "profiles.toml"
        )
        spec = registry.models["gemma-g4"].model_copy(
            update={
                "id": "actual-gemma-fp32",
                "enabled": True,
                "model_id": model.name,
                "revision": model.revision,
                "processor_revision": model.revision,
                "device": str(model.device),
                "precision": "float32",
                "calibration": "none",
                "context_limit": model.max_context,
                "cost_per_request_usd": 0.02,
            }
        )
        episode = _episode("integrity-disjoint-episode", "integrity-disjoint-source")
        positive = workflow(registry, spec, episode, "actual-telemetry", load_actual=True)
        execution = positive["execution"][spec.id]
        if (
            positive["status"] != "completed"
            or positive["actual_backbone_forwards"] < 1
            or positive["manifest_checks_before_prediction"] != positive["actual_backbone_forwards"]
            or execution["status"] != "completed"
            or execution["telemetry"]["revision"] != model.revision
            or execution["telemetry"]["precision"] != "float32"
            or not positive["wrappers_closed"]
        ):
            raise AssertionError("actual workflow telemetry was not valid before inference")
        record("actual_workflow_telemetry", **positive)
        wrong_revision = "0" * len(model.revision)
        if wrong_revision == model.revision:
            wrong_revision = "1" * len(model.revision)
        mismatches = {
            "revision": {"revision": wrong_revision, "processor_revision": wrong_revision},
            "precision": {"precision": "bfloat16"},
        }
        for kind, changes in mismatches.items():
            requested = spec.model_copy(update=changes)
            result = workflow(registry, requested, episode, f"mismatch-{kind}", load_actual=True)
            execution = result["execution"][spec.id]
            if (
                result["status"] != "setup_error"
                or result["actual_backbone_forwards"] != 0
                or result["adapter_factory_calls"] != 1
                or execution["status"] != "setup_error"
                or execution.get("error_type") != "ValueError"
                or not result["wrappers_closed"]
            ):
                raise AssertionError(f"actual {kind} mismatch did not stop before inference")
            record(f"actual_{kind}_mismatch_before_inference", **result)

        training_path = training_dir / "training-checks.json"
        training = json.loads(training_path.read_text())
        if training["status"] != "passed" or not training["runs"]:
            raise ValueError("completed actual training checks and adapter artifacts are required")
        report["training_source"] = {
            "report_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
            "base_model": training["base_model"],
            "base_revision": training["base_revision"],
            "steps_per_cell": training["steps_per_cell"],
            "shots_per_class": training["shots_per_class"],
            "adapters": len(training["runs"]),
        }
        if training["base_model"] != model.name or training["base_revision"] != model.revision:
            raise ValueError("fitted artifacts differ from the actual borrowed checkpoint identity")
        calibration = load_cases(dataset_dir / "text-calibration.jsonl")
        heldout = load_cases(dataset_dir / "text-test.jsonl")
        if fingerprint(calibration) != training["datasets"]["text-calibration"]["sha256"]:
            raise ValueError("calibration dataset differs from the actual training report")
        if fingerprint(heldout) != training["datasets"]["text-test"]["sha256"]:
            raise ValueError("heldout dataset differs from the actual training report")
        selected_methods = set()
        for trained in training["runs"]:
            if trained["status"] != "passed":
                raise ValueError("an adapter did not finish actual training validation")
            adapter_path = training_dir / trained["adapter_path"]
            digest = directory_digest(adapter_path)
            if digest != trained["adapter_sha256"]:
                raise AssertionError("actual trained adapter directory hash changed")
            support_path = training_dir / "support" / f"{trained['cell_id']}.jsonl"
            fitted = load_cases(support_path)
            if fingerprint(fitted) != trained["support_sha256"]:
                raise AssertionError("actual selected fitting data changed")
            provenance_path = adapter_path / "training-provenance.json"
            provenance = json.loads(provenance_path.read_text())
            if (
                hashlib.sha256(provenance_path.read_bytes()).hexdigest()
                != trained["provenance_sha256"]
            ):
                raise AssertionError("actual trained provenance hash changed")
            artifact_spec = spec.model_copy(
                update={
                    "decision_adapter_path": str(adapter_path),
                    "decision_adapter_sha256": digest,
                    "calibration": "checkpoint",
                }
            )
            rejected = []
            for split, source in (("train", fitted[0]), ("calibration", calibration[0])):
                keys = {
                    "id": (f"{split}_ids", source.id),
                    "group": (f"{split}_groups", source.group_id or source.id),
                    "request": (f"{split}_requests", request_fingerprint(source.request)),
                }
                for dimension, (key, value) in keys.items():
                    if value not in provenance[key]:
                        raise AssertionError("saved provenance omitted actual fitting identities")
                    probe = SimpleNamespace(
                        id=f"integrity-clean-{split}-{dimension}",
                        group_id=f"integrity-clean-group-{split}-{dimension}",
                        request=heldout[0].request,
                    )
                    if dimension == "id":
                        probe.id = source.id
                    elif dimension == "group":
                        probe.group_id = source.group_id or source.id
                    else:
                        probe.request = source.request
                    reasons = fitting_provenance_blockers(artifact_spec, [probe])
                    if reasons != ["evaluation_overlaps_adapter_training"]:
                        raise AssertionError(
                            f"actual {split}/{dimension} fitting overlap was accepted"
                        )
                    rejected.append({"split": split, "dimension": dimension, "reasons": reasons})
            if fitting_provenance_blockers(artifact_spec, heldout):
                raise AssertionError("disjoint actual heldout data was incorrectly rejected")
            report["trained_artifacts"].append(
                {
                    "id": trained["id"],
                    "adapter_sha256": digest,
                    "provenance_sha256": trained["provenance_sha256"],
                    "fitting_cases": len(fitted),
                    "calibration_cases": len(calibration),
                    "rejected_dimensions": rejected,
                    "disjoint_heldout_cases_accepted": len(heldout),
                    "adapter_weights_loaded_in_this_check": False,
                }
            )
            # Both matched objectives receive the actual workflow boundary
            # checks; every adapter above receives all six provenance checks.
            if trained["method"] not in selected_methods:
                selected_methods.add(trained["method"])
                for split, source in (("train", fitted[0]), ("calibration", calibration[0])):
                    for dimension in ("id", "group"):
                        identifier = (
                            source.id
                            if dimension == "id"
                            else f"integrity-episode-{split}-{dimension}"
                        )
                        group = (
                            source.group_id or source.id
                            if dimension == "group"
                            else f"integrity-group-{split}-{dimension}"
                        )
                        scenario = _episode(identifier, group)
                        name = f"adapter-{trained['method']}-{split}-{dimension}"
                        result = workflow(
                            registry, artifact_spec, scenario, name, load_actual=False
                        )
                        if (
                            result["status"] != "not_run"
                            or result["reasons"] != ["evaluation_overlaps_adapter_training"]
                            or result["actual_backbone_forwards"] != 0
                            or result["adapter_factory_calls"] != 0
                        ):
                            raise AssertionError(
                                "actual fitted adapter workflow was not blocked before loading"
                            )
                        record(
                            f"actual_adapter_workflow_{trained['method']}_{split}_{dimension}",
                            **result,
                        )
            persist(f"adapter_provenance:{trained['id']}")
        record(
            "all_actual_adapters_fitting_provenance",
            adapters=len(report["trained_artifacts"]),
            rejected_dimension_checks=6 * len(report["trained_artifacts"]),
            disjoint_heldout_checks=len(report["trained_artifacts"]),
        )

        # Fit a genuine Choice prior so request-only overlap can occur in a
        # valid workflow state, with disjoint episode ID and source group.
        prior_request = _observation("An actual CPU prior fitting source in an unreachable state.")
        train_cases = [
            EvaluationCase(
                id=f"integrity-prior-train-{i}",
                group_id=f"integrity-prior-source-{i}",
                split="train",
                request=prior_request
                if i == 0
                else _observation("A second CPU prior training source."),
                gold={"action": label},
            )
            for i, label in enumerate(("complete", "escalate"))
        ]
        train_path = output / "prior-train.jsonl"
        write_cases(train_path, train_cases)
        fitted = fit_baseline(train_path, output / "prior", method="prior")
        prior_spec = ModelSpec(
            id="actual-fitted-prior",
            adapter="prior",
            enabled=True,
            precision="float64",
            execution_mode="sequential",
            calibration="none",
            cost_per_request_usd=0.0,
            artifact_file=fitted["artifact_file"],
            artifact_sha256=fitted["artifact_sha256"],
            capabilities=spec.capabilities.model_copy(
                update={"modalities": ("text",), "primitives": ("choice",), "batch": "none"}
            ),
        )
        prior_episode = _episode(
            "integrity-prior-heldout-episode",
            "integrity-prior-heldout-source",
            later_request=prior_request,
        )
        identities = {
            "id": {case.id for case in train_cases},
            "group": {case.group_id for case in train_cases},
            "request": {request_fingerprint(case.request) for case in train_cases},
        }
        if prior_episode.id in identities["id"] or prior_episode.group_id in identities["group"]:
            raise AssertionError("request-only workflow fixture accidentally overlaps IDs/groups")
        if (
            request_fingerprint(prior_episode.states["initial"].observation)
            in identities["request"]
        ):
            raise AssertionError(
                "request-only workflow overlap is not isolated to the unreachable state"
            )
        result = workflow(
            registry, prior_spec, prior_episode, "prior-unreachable-request", load_actual=False
        )
        if (
            result["status"] != "not_run"
            or result["reasons"] != ["evaluation_overlaps_training"]
            or result["adapter_factory_calls"] != 0
            or result["actual_backbone_forwards"] != 0
        ):
            raise AssertionError("actual fitted prior overlap in an unreachable state was accepted")
        record(
            "actual_fitted_prior_unreachable_request_overlap",
            **result,
            prior_artifact_sha256=fitted["artifact_sha256"],
            actual_cpu_fitting_cases=len(train_cases),
            overlapping_state="unreachable",
            episode_ids_disjoint=True,
            episode_source_groups_disjoint=True,
            reachable_initial_request_disjoint=True,
        )
        if selected_methods != {"ce", "ce_brier"}:
            raise AssertionError(
                "actual fitting boundary checks require both matched training objectives"
            )
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        (output / "integrity-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        report["checks_passed"] = len(report["checks"])
        report["borrowed_model_precision_on_return"] = str(model.head.weight.dtype).removeprefix(
            "torch."
        )
        report["borrowed_model_revision_on_return"] = model.revision
        persist(report["status"])
    return report
