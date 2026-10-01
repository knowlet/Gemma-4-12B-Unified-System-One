"""Aggregate actual live receipts and descriptive paired source-group intervals.

No runtime result or model manifest is manufactured. Prediction comparisons use
copied rows and require identical recorded request identities before bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from s1.evaluation.statistics import paired_comparison

JOBS = {
    "20260930-v2-live-02/checkpoint": "ap-d7sHBbjvAJ7qey70u2sdo4",
    "20261001-v2-live-03/training": "ap-W6NzNTxhnfQK3VovSJk7uY",
    "20261001-v2-live-03/baselines": "ap-iwnAEC27b0CvZdi9AO0OCu",
    "20261001-v2-live-03/services": "ap-G8bfbdhLl0uAv8DHPYcTme",
    "20261001-v2-live-03/diagnostics": "ap-2ZKm7x2JOtmjblqYdoT3Hv",
    "20261001-v2-live-04/stress": "ap-DYbNYZxTGrvW77LPq4cCNa",
    "20261001-v2-live-04/training-full": "ap-WtiHAD5uDycSRHHaaO5dmz",
    "20261001-v2-live-04/baselines-full": "ap-KYotBV6Nq9DRBsz2TtY2dV",
    "20261001-v2-live-04/checkpoint-fp32": "ap-i54t80cLOMjRNUM9WcpUmt",
    "20261001-v2-live-05/integrity": "ap-Ky8N1kngPEnQXxOCShUMLr",
    "20261001-v2-live-05/stress": "ap-qfkoeHABpdp1hLC64HBdVe",
}
REQUIRED = (
    "checkpoint-fp32",
    "training",
    "training-full",
    "baselines",
    "baselines-full",
    "services",
    "stress",
    "diagnostics",
    "integrity",
)


def _read(path: Path):
    return json.loads(path.read_text())


def _hash(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _copied_comparison_rows(path: Path):
    rows = [dict(row) for row in _rows(path)]
    manifest_path = path.parent / "run_manifest.json"
    identities = {}
    if manifest_path.exists():
        manifest = _read(manifest_path)
        for cell in manifest["plan"]["models"]:
            for case in cell["cases"]:
                identities[(cell["model_id"], case["case_id"])] = case["request_sha256"]
    for row in rows:
        row.setdefault("repetition", 0)
        if "request_sha256" not in row:
            row["request_sha256"] = identities.get((row["model_id"], row["case_id"]))
        if not row["request_sha256"]:
            raise ValueError("prediction file has no recorded request identity")
    return rows


def _observed(rows):
    valid = [row for row in rows if row["status"] == "ok"]
    correct = sum(row["actual_label"] == row["gold"] for row in valid)
    return {
        "recorded_decisions": len(rows),
        "valid_decisions": len(valid),
        "correct": correct,
        "groups": len({row["group_id"] for row in rows}),
        "operational_accuracy": correct / len(rows) if rows else None,
        "conditional_accuracy": correct / len(valid) if valid else None,
    }


def _compact_metrics(value):
    """Keep scalar metrics; large bins and slices remain in hashed raw artifacts."""
    if not isinstance(value, dict):
        return value
    return {
        key: _compact_metrics(item)
        for key, item in value.items()
        if not isinstance(item, list)
        and key not in ("slices", "selective", "schema_macro_f1", "critical_fields", "noul")
    }


def _integrity_evidence(directory, inner, training):
    """Audit expected blocked outcomes explicitly, instead of treating them as failures."""
    artifacts = inner.get("trained_artifacts", [])
    fitted = {run["id"]: run for run in training.get("runs", [])}
    dimensions = {
        (split, dimension)
        for split in ("train", "calibration")
        for dimension in ("id", "group", "request")
    }
    audited_artifacts = []
    for artifact in artifacts:
        run = fitted.get(artifact["id"], {})
        rejected = artifact.get("rejected_dimensions", [])
        passed = (
            {(entry["split"], entry["dimension"]) for entry in rejected} == dimensions
            and len(rejected) == 6
            and all(
                entry["reasons"] == ["evaluation_overlaps_adapter_training"] for entry in rejected
            )
            and artifact.get("adapter_sha256") == run.get("adapter_sha256")
            and artifact.get("provenance_sha256") == run.get("provenance_sha256")
            and artifact.get("disjoint_heldout_cases_accepted")
            == training.get("datasets", {}).get("text-test", {}).get("cases")
            and artifact.get("adapter_weights_loaded_in_this_check") is False
        )
        audited_artifacts.append({**artifact, "boundary_passed": passed})
    checks = []
    for check in inner.get("checks", []):
        name, status = check["name"], check["status"]
        forwards, factories = (
            check.get("actual_backbone_forwards"),
            check.get("adapter_factory_calls"),
        )
        if name == "actual_workflow_telemetry":
            telemetry = next(iter(check["execution"].values())).get("telemetry", {})
            expected, passed = (
                "completed with validated FP32 telemetry before actual forwards",
                (
                    status == "completed"
                    and forwards > 0
                    and check["manifest_checks_before_prediction"] == forwards
                    and factories == 1
                    and telemetry.get("precision") == "float32"
                    and telemetry.get("revision") == inner.get("revision")
                    and check.get("wrappers_closed") is True
                ),
            )
        elif name in (
            "actual_revision_mismatch_before_inference",
            "actual_precision_mismatch_before_inference",
        ):
            execution = next(iter(check["execution"].values()))
            expected, passed = (
                "setup_error and zero actual forwards",
                (
                    status == "setup_error"
                    and forwards == 0
                    and factories == 1
                    and execution.get("error_type") == "ValueError"
                    and check.get("wrappers_closed") is True
                ),
            )
        elif name.startswith("actual_adapter_workflow_"):
            expected, passed = (
                "provenance block before adapter loading",
                (
                    status == "not_run"
                    and forwards == 0
                    and factories == 0
                    and check.get("reasons") == ["evaluation_overlaps_adapter_training"]
                ),
            )
        elif name == "actual_fitted_prior_unreachable_request_overlap":
            expected, passed = (
                "unreachable-state request-only provenance block before loading",
                (
                    status == "not_run"
                    and forwards == 0
                    and factories == 0
                    and check.get("reasons") == ["evaluation_overlaps_training"]
                    and check.get("overlapping_state") == "unreachable"
                    and all(
                        check.get(key) is True
                        for key in (
                            "episode_ids_disjoint",
                            "episode_source_groups_disjoint",
                            "reachable_initial_request_disjoint",
                        )
                    )
                ),
            )
        elif name == "all_actual_adapters_fitting_provenance":
            expected, passed = (
                "every trained artifact rejects six dimensions and accepts disjoint heldout",
                (
                    status == "passed"
                    and check.get("adapters") == len(artifacts)
                    and check.get("rejected_dimension_checks")
                    == sum(len(artifact["rejected_dimensions"]) for artifact in artifacts)
                    and check.get("disjoint_heldout_checks") == len(artifacts)
                    and len(artifacts) == len(fitted)
                    and all(artifact["boundary_passed"] for artifact in audited_artifacts)
                ),
            )
        else:
            expected, passed = "recognized integrity boundary check", False
        evidence = {**check, "expected_outcome": expected, "boundary_passed": bool(passed)}
        if check.get("manifest"):
            path = directory / "checks" / check["manifest"]
            if path.exists():
                manifest = _read(path)
                evidence["manifest_sha256"] = _hash(path)
                evidence["boundary_passed"] &= (
                    manifest["status"] == status and manifest["execution"] == check["execution"]
                )
            else:
                evidence["boundary_passed"] = False
        checks.append(evidence)
    return {
        "status": inner.get("status"),
        "checkpoint": inner.get("checkpoint"),
        "revision": inner.get("revision"),
        "precision": inner.get("precision"),
        "training_source": inner.get("training_source"),
        "checks": checks,
        "trained_artifacts": audited_artifacts,
        "verified_boundary_checks": sum(check["boundary_passed"] for check in checks),
        "verified_rejected_dimensions": sum(
            len(artifact["rejected_dimensions"])
            for artifact in audited_artifacts
            if artifact["boundary_passed"]
        ),
        "verified_disjoint_heldout_checks": sum(
            artifact["boundary_passed"] for artifact in audited_artifacts
        ),
        "all_boundaries_verified": bool(checks)
        and inner.get("status") == "passed"
        and all(check["boundary_passed"] for check in checks),
        "limitations": inner.get("limitations"),
    }


def aggregate(root: Path, *, jobs=None, resamples=2000):
    root = Path(root).resolve()
    base = root / "artifacts/live-validation"
    jobs = {
        **{key: f"https://modal.com/apps/knowlet/main/{value}" for key, value in JOBS.items()},
        **(jobs or {}),
    }
    sources, latest, receipts = [], {}, {}
    for path in sorted(base.glob("20*/*/receipt.json")):
        receipt = _read(path)
        key = str(path.parent.relative_to(base))
        stage = receipt.get("stage", path.parent.name)
        inner = receipt.get("result", {})
        source = {
            "key": key,
            "stage": stage,
            "artifact": str(path.relative_to(root)),
            "artifact_sha256": _hash(path),
            "job_url": jobs.get(key),
            "outer_status": receipt.get("status"),
            "result_status": inner.get("status"),
            "started_at_unix": receipt.get("started_at_unix"),
            "elapsed_seconds": receipt.get("elapsed_seconds"),
            "gpu": receipt.get("gpu"),
            "gpu_memory_bytes": receipt.get("gpu_memory_bytes"),
            "campaign_checkpoint": receipt.get("campaign_checkpoint")
            or {"model_id": receipt.get("model_id"), "revision": receipt.get("revision")},
            "campaign_checkpoint_is_evaluated_identity": False,
            "identity_policy": "actual inner model/checkpoint identities and resolved telemetry are authoritative",
            "packages": receipt.get("packages"),
            "failures": inner.get("failures", []),
            "error": receipt.get("error"),
        }
        sources.append(source)
        receipts[key] = (path.parent, receipt)
        old = latest.get(stage)
        if old is None or (source["started_at_unix"] or 0) > (old["started_at_unix"] or 0):
            latest[stage] = source
    provenance_path = base / "datasets/provenance.json"
    provenance = _read(provenance_path)
    local_path = base / "local-checks.json"
    local = _read(local_path) if local_path.exists() else {"status": "not_recorded"}
    result = {
        "schema_version": 1,
        "generated_at_taipei": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
        "evidence_policy": "actual receipts and prediction records; no pending stage is passed",
        "receipts": sources,
        "latest_stages": latest,
        "missing_stages": [stage for stage in REQUIRED if stage not in latest],
        "local_checks": local,
        "local_checks_artifact_sha256": _hash(local_path) if local_path.exists() else None,
        "data": {
            key: provenance[key]
            for key in (
                "dataset_format_version",
                "seed",
                "purpose",
                "scope_limit",
                "selection_policy",
                "sources",
                "source_files",
                "datasets",
                "media_bundle_sha256",
                "M0_annotation",
                "M2_annotation",
                "split_audit",
                "full_train_split_audit",
                "full_train_superset",
            )
        },
        "dataset_provenance_sha256": _hash(provenance_path),
        "lock_sha256": _hash(root / "uv.lock"),
        "aggregation_script_sha256": _hash(Path(__file__)),
        "source_identity_by_stage": {},
        "quality": [],
        "training": {},
        "baseline_curve": {},
        "baseline_checkpoints": {},
        "batch_checks": {},
        "pipeline": {},
        "workflows": {},
        "load": {},
        "diagnostics": {},
        "integrity": {},
        "comparisons": [],
        "comparison_rejections": [],
        "statistical_policy": {
            "unit": "source group; paired decision outcomes",
            "confidence": 0.95,
            "resamples": resamples,
            "seed": 20260930,
            "multiple_comparisons_adjusted": False,
            "interpretation": "descriptive, unadjusted; no post-hoc noninferiority acceptance",
            "normalization": "repetition=0 added only to copied training prediction rows",
            "matching": "ID/question/repetition plus recorded request hash and gold/group/task/schema/type",
        },
        "external_providers": [
            {
                "model": "jev-http",
                "status": "not_run",
                "endpoint_env": "JEV_BENCHMARK_ENDPOINT",
                "token_env": "JEV_BENCHMARK_TOKEN",
            },
            {
                "model": "kev",
                "status": "not_run",
                "endpoint_env": "KEV_BENCHMARK_ENDPOINT",
                "token_env": "configure ModelSpec.token_env if this deployment requires a bearer token",
            },
            {
                "model": "decider",
                "status": "not_run",
                "endpoint_env": "DECIDER_BENCHMARK_ENDPOINT",
                "token_env": "configure ModelSpec.token_env if this deployment requires a bearer token",
            },
            {
                "model": "agentjev",
                "status": "not_run",
                "endpoint_env": "AGENTJEV_BENCHMARK_ENDPOINT",
                "token_env": "configure ModelSpec.token_env if this deployment requires a bearer token",
            },
        ],
    }
    for provider in result["external_providers"]:
        provider["status"] = "blocked_missing_prerequisites"
        provider["additional_prerequisites"] = [
            "accessible deployed service and immutable resolved model revision",
            "provider-reported cost metadata or explicit unknown-cost policy",
        ]
    comparison_sources = {}

    def active(stage):
        if stage not in latest:
            return None, None
        return receipts[latest[stage]["key"]]

    def quality(stage, name, path, *, model_id=None, reported=None, dataset_hash=None, **extra):
        if not path.exists():
            return
        rows = _rows(path)
        if model_id is not None:
            rows = [row for row in rows if row["model_id"] == model_id]
        record = {
            "stage": stage,
            "model": name,
            "prediction_artifact": str(path.relative_to(root)),
            "prediction_sha256": _hash(path),
            "dataset_sha256": dataset_hash,
            "observed": _observed(rows),
            "task_observations": {
                task: _observed([row for row in rows if row["task_id"] == task])
                for task in sorted({row["task_id"] for row in rows})
            },
            "reported": reported or {},
            **extra,
        }
        result["quality"].append(record)
        comparison_sources[f"{stage}:{name}"] = (path, model_id, dataset_hash, extra)

    for stage in ("checkpoint", "checkpoint-fp32"):
        directory, receipt = active(stage)
        if directory is None:
            continue
        checks, inner = directory / "checks", receipt["result"]
        result["batch_checks"][stage] = {
            "status": inner.get("status"),
            "precision": inner.get("precision"),
            "runner_run_count": len(inner.get("runner_runs", [])),
            "failures": inner.get("failures", []),
            "runtime_sweeps": [],
            "checks": {},
        }
        for name in (
            "g1-g2-runner-parity",
            "g1-g2-numerical-parity",
            "g2-g4-native-parity",
            "question-independence",
        ):
            path = checks / f"{name}.json"
            if path.exists():
                value = _read(path)
                result["batch_checks"][stage]["checks"][name] = {
                    "artifact_sha256": _hash(path),
                    "passed": value.get("passed"),
                    "comparisons": value.get("comparisons"),
                    "execution": value.get("execution"),
                }
        for run in inner.get("runner_runs", []):
            run_dir = checks / run["id"]
            manifest_path = run_dir / "run_manifest.json"
            if manifest_path.exists():
                runtime = _read(manifest_path)["plan"]["runtime"]
                result["source_identity_by_stage"][stage] = {
                    "source_sha256": runtime["source_sha256"],
                    "lock_sha256": runtime["lock_sha256"],
                }
            result["batch_checks"][stage]["runtime_sweeps"].append(
                {
                    "id": run["id"],
                    "status": run["status"],
                    "cases": run["cases"],
                    "batch_size": run["batch_size"],
                    "observed_execution": run["observed_execution"],
                    "models": {
                        key: {
                            field: stats.get(field)
                            for field in (
                                "denominators",
                                "records_complete",
                                "systems",
                                "telemetry",
                            )
                        }
                        for key, stats in run["models"].items()
                    },
                }
            )
            if run["id"].startswith("public-"):
                for name, stats in run["models"].items():
                    quality(
                        stage,
                        f"{run['id']}:{name}",
                        run_dir / "predictions.jsonl",
                        model_id=name,
                        reported=stats.get("quality"),
                        dataset_hash=run["dataset_sha256"],
                        precision=inner.get("precision"),
                    )

    for stage in ("training", "training-full"):
        directory, receipt = active(stage)
        if directory is None:
            continue
        inner, checks = receipt["result"], directory / "checks"
        result["training"][stage] = {
            key: inner.get(key)
            for key in (
                "status",
                "shots_per_class",
                "seeds",
                "steps_per_cell",
                "planned_cells",
                "planned_optimizer_updates",
                "optimizer_updates",
                "precision",
                "gradient_checkpointing",
                "reload_scope",
                "base_weights_merged",
                "datasets",
                "split_audit",
                "wall_seconds",
            )
        }
        compact = []
        for run in inner.get("runs", []):
            compact.append(
                {
                    key: run.get(key)
                    for key in (
                        "id",
                        "seed",
                        "shots_per_class",
                        "method",
                        "status",
                        "optimizer_updates",
                        "finite_losses",
                        "trainable_parameters_changed",
                        "trainable_parameters",
                        "adapter_reload",
                        "adapter_sha256",
                        "provenance_sha256",
                        "support_sha256",
                        "fitting_provenance",
                        "heldout_text",
                        "heldout_media",
                        "calibration",
                    )
                }
            )
            for modality in ("text", "media"):
                quality(
                    stage,
                    f"{run['id']}:{modality}",
                    checks / "predictions" / f"{run['id']}-{modality}.jsonl",
                    reported=run.get(f"heldout_{modality}"),
                    dataset_hash=inner.get("datasets", {})
                    .get(f"{modality}-test", {})
                    .get("sha256"),
                    support_sha256=run.get("support_sha256"),
                    cell_id=run.get("cell_id"),
                    method=run.get("method"),
                    shots=run.get("shots_per_class"),
                    seed=run.get("seed"),
                )
        result["training"][stage]["runs"] = compact
        for modality in ("text", "media"):
            quality(
                stage,
                f"base:{modality}",
                checks / "predictions" / f"base-{modality}.jsonl",
                reported=inner.get("base_retention", {}).get(modality),
                dataset_hash=inner.get("datasets", {}).get(f"{modality}-test", {}).get("sha256"),
            )

    directory, receipt = active("baselines")
    if directory:
        inner = receipt["result"]
        result["baseline_checkpoints"] = inner.get("checkpoints", {})
        for name, stats in inner.get("models", {}).items():
            if stats.get("directory"):
                quality(
                    "baselines",
                    name,
                    directory / "checks" / stats["directory"] / "predictions.jsonl",
                    reported={
                        "quality": stats.get("quality"),
                        "metrics": stats.get("metrics"),
                        "expected_context_rejections": stats.get(
                            "expected_context_rejection_count"
                        ),
                        "unexpected_errors": stats.get("unexpected_error_count"),
                    },
                    dataset_hash=inner.get("datasets", {}).get("test_sha256"),
                )
    directory, receipt = active("baselines-full")
    if directory:
        inner = receipt["result"]
        result["baseline_curve"] = {
            key: inner.get(key)
            for key in (
                "status",
                "methods",
                "shots_per_class",
                "seeds",
                "expected_fits",
                "setfit_optimizer_steps_per_cell",
                "actual_setfit_optimizer_updates",
                "datasets",
                "checkpoint",
            )
        }
        result["baseline_curve"]["fits"] = [
            {
                key: fit.get(key)
                for key in (
                    "id",
                    "cell_id",
                    "method",
                    "seed",
                    "shots_per_class",
                    "support_sha256",
                    "status",
                    "training",
                    "artifact_sha256",
                    "quality",
                    "denominators",
                )
            }
            for fit in inner.get("fits", [])
        ]
        for fit in inner.get("fits", []):
            quality(
                "baselines-full",
                fit["id"],
                directory / "checks" / fit["evaluation_directory"] / "predictions.jsonl",
                reported={"quality": fit.get("quality"), "metrics": fit.get("metrics")},
                dataset_hash=inner.get("datasets", {}).get("test_sha256"),
                support_sha256=fit.get("support_sha256"),
                cell_id=fit.get("cell_id"),
                method=fit.get("method"),
                shots=fit.get("shots_per_class"),
                seed=fit.get("seed"),
            )

    directory, receipt = active("services")
    if directory:
        services = receipt["result"].get("results", {})
        pipeline = services.get("pipeline", {})
        result["pipeline"] = {
            key: pipeline.get(key)
            for key in (
                "status",
                "preprocessor",
                "reported_total_cost_usd",
                "allocated_device_wall_seconds",
                "synchronized_decision_wall_seconds",
                "decision_calls",
                "failures",
            )
        }
        for view, entry in pipeline.get("runs", {}).items():
            for name, stats in entry.get("summary", {}).get("models", {}).items():
                quality(
                    "pipeline",
                    view,
                    directory / "checks/pipeline" / f"media-{view.lower()}" / "predictions.jsonl",
                    model_id=name,
                    reported=stats.get("quality"),
                )
            result["pipeline"][view] = {
                key: entry.get(key)
                for key in (
                    "run_status",
                    "actual_predictions",
                    "expected_predictions",
                    "records_complete",
                    "counterfactual",
                )
            }
        for name, manifest_path in (
            ("gemma", directory / "checks/pipeline/workflow/workflow_manifest.json"),
            ("laya", directory / "checks/workflow/workflow_manifest.json"),
        ):
            if manifest_path.exists():
                manifest = _read(manifest_path)
                result["workflows"][name] = {
                    "status": manifest["status"],
                    "manifest_sha256": _hash(manifest_path),
                    "source_sha256": manifest["source_sha256"],
                    "execution": manifest["execution"],
                    "policies": _read(manifest_path.parent / "workflow_summary.json"),
                    "episodes": len(_rows(manifest_path.parent / "episodes.jsonl")),
                }
        result["load"]["services"] = services.get("http_load", {})
        laya = services.get("laya", {})
        for name, stats in laya.get("text", {}).get("models", {}).items():
            quality(
                "laya",
                name,
                directory / "checks/text/predictions.jsonl",
                model_id=name,
                reported=stats.get("quality"),
                dataset_hash=provenance["datasets"]["text-test.jsonl"]["dataset_sha256"],
            )
    directory, receipt = active("stress")
    if directory:
        result["load"]["stress"] = receipt.get("result", {})
        result["load"]["stress"]["evidence_key"] = latest["stress"]["key"]
        result["load"]["stress"]["gpu"] = receipt.get("gpu")
        identity_path = directory / "checks/server-identity.json"
        if identity_path.exists():
            result["load"]["stress"]["server_identity"] = _read(identity_path)
            result["load"]["stress"]["server_identity_sha256"] = _hash(identity_path)
    directory, receipt = active("integrity")
    if directory:
        result["integrity"] = _integrity_evidence(
            directory, receipt.get("result", {}), result["training"].get("training-full", {})
        )
    directory, receipt = active("diagnostics")
    if directory:
        inner = receipt.get("result", {})
        result["diagnostics"] = {
            "status": inner.get("status"),
            "errors": inner.get("errors"),
            "fp32_exercised": inner.get("fp32_exercised"),
            "hypotheses": inner.get("hypotheses"),
            "experiments": [
                {
                    key: experiment.get(key)
                    for key in (
                        "case",
                        "condition",
                        "precision",
                        "attention_implementations",
                        "batch_size",
                        "candidate_logits",
                        "probabilities",
                        "readout_batch_logit_delta",
                        "readout_batch_probability_delta",
                        "difference_from_original_single",
                    )
                }
                for experiment in inner.get("experiments", [])
            ],
        }

    def comparison(left_id, right_id, *, kind):
        left_path, left_model, left_hash, left_extra = comparison_sources[left_id]
        right_path, right_model, right_hash, right_extra = comparison_sources[right_id]
        try:
            if left_hash and right_hash and left_hash != right_hash:
                raise ValueError("dataset fingerprints differ")
            if kind == "matched_support" and left_extra.get("support_sha256") != right_extra.get(
                "support_sha256"
            ):
                raise ValueError("support fingerprints differ")
            left, right = _copied_comparison_rows(left_path), _copied_comparison_rows(right_path)
            if left_model:
                left = [row for row in left if row["model_id"] == left_model]
            if right_model:
                right = [row for row in right if row["model_id"] == right_model]

            def key(row):
                return row["case_id"], row["question_id"], row["repetition"]

            a, b = {key(row): row for row in left}, {key(row): row for row in right}
            if a.keys() != b.keys() or any(
                a[k]["request_sha256"] != b[k]["request_sha256"] for k in a
            ):
                raise ValueError("recorded request identities differ")
            value = paired_comparison(left, right, seed=20260930, resamples=resamples)
            result["comparisons"].append(
                {"left": left_id, "right": right_id, "kind": kind, **value}
            )
        except ValueError as exc:
            result["comparison_rejections"].append(
                {"left": left_id, "right": right_id, "reason": str(exc)}
            )

    for stage in ("training", "training-full"):
        prefix = f"{stage}:"
        for key in list(comparison_sources):
            if key.startswith(prefix) and not key.startswith(prefix + "base:"):
                modality = key.rsplit(":", 1)[1]
                baseline = f"{stage}:base:{modality}"
                if baseline in comparison_sources:
                    comparison(baseline, key, kind="base_retention")
        methods = defaultdict(dict)
        for key, source in comparison_sources.items():
            if key.startswith(prefix) and key.endswith(":text") and source[3].get("cell_id"):
                methods[source[3]["cell_id"]][source[3]["method"]] = key
        for cell in methods.values():
            if "ce" in cell and "ce_brier" in cell:
                comparison(cell["ce"], cell["ce_brier"], kind="matched_support")
    for key, source in comparison_sources.items():
        if key.startswith("training-full:") and key.endswith(":text") and source[3].get("cell_id"):
            for baseline_key, baseline_source in comparison_sources.items():
                if (
                    baseline_key.startswith("baselines-full:")
                    and baseline_source[3].get("cell_id") == source[3]["cell_id"]
                ):
                    comparison(baseline_key, key, kind="matched_support")
    base_key = "training-full:base:text"
    if base_key in comparison_sources:
        for key in comparison_sources:
            if key.startswith(("baselines:", "baselines-full:", "laya:")):
                comparison(base_key, key, kind="public_text_same_input")

    accounting = {}
    for filename in ("rates.json", "billing-report.json", "apps.json", "apps-final.json"):
        path = base / "accounting" / filename
        if path.exists():
            accounting[filename] = {"artifact_sha256": _hash(path), "recorded": _read(path)}
    billing = accounting.get("billing-report.json", {}).get("recorded", [])
    accounting["recorded_billed_usd"] = sum(float(row["cost"]) for row in billing)
    accounting["scope"] = (
        "Completed billing buckets only; pending/unreported costs are not zero or final."
    )
    rates = accounting.get("rates.json", {}).get("recorded", {})
    billed_apps = {row["object_id"] for row in billing}
    estimates = []
    for source in sources:
        if source["elapsed_seconds"] is None or not rates:
            continue
        hours = source["elapsed_seconds"] / 3600
        gpu_cost = hours * float(rates["gpu_hour_cost_a100_80gb"]) if source["gpu"] else 0.0
        memory_gib = 16 if source["stage"].startswith("baselines") else 48
        allocated_window = gpu_cost + hours * (
            4 * float(rates["cpu_hour_cost"]) + memory_gib * float(rates["mem_gib_hour_cost"])
        )
        app_id = source["job_url"].rsplit("/", 1)[-1] if source.get("job_url") else None
        estimates.append(
            {
                "key": source["key"],
                "recorded_elapsed_seconds": source["elapsed_seconds"],
                "gpu_only_window_usd": gpu_cost,
                "requested_allocation_window_usd": allocated_window,
                "has_completed_billing_bucket": app_id in billed_apps if app_id else None,
            }
        )
    unreported = [entry for entry in estimates if entry["has_completed_billing_bucket"] is False]
    accounting["execution_window_estimate"] = {
        "official_rates": {
            key: rates.get(key)
            for key in ("gpu_hour_cost_a100_80gb", "cpu_hour_cost", "mem_gib_hour_cost")
        },
        "requested_resources": "4 CPUs; 48 GiB host memory for GPU stages, 16 GiB for CPU baseline stages; A100-80GB tier",
        "receipt_windows": estimates,
        "all_recorded_windows_gpu_only_usd": sum(
            entry["gpu_only_window_usd"] for entry in estimates
        ),
        "all_recorded_windows_allocation_usd": sum(
            entry["requested_allocation_window_usd"] for entry in estimates
        ),
        "unreported_windows_gpu_only_usd": sum(
            entry["gpu_only_window_usd"] for entry in unreported
        ),
        "unreported_windows_allocation_usd": sum(
            entry["requested_allocation_window_usd"] for entry in unreported
        ),
        "recorded_bill_plus_unreported_window_estimate_usd": accounting["recorded_billed_usd"]
        + sum(entry["requested_allocation_window_usd"] for entry in unreported),
        "scope": "Bounds for recorded execution windows only, not the final bill; initialization outside receipts, unmatched apps, storage/egress and pending bucket extensions are excluded. Billing and these windows can overlap; all-window estimate is not added to billing.",
    }
    result["accounting"] = accounting
    unfinished = [
        stage
        for stage in REQUIRED
        if stage not in latest or latest[stage]["outer_status"] != "completed"
    ]
    pending_local = [
        key
        for key, value in local.items()
        if isinstance(value, dict) and value.get("status") == "running"
    ]
    result["incomplete"] = unfinished + [f"local:{key}" for key in pending_local]
    if result["integrity"] and not result["integrity"]["all_boundaries_verified"]:
        result["incomplete"].append("integrity:boundaries_unverified")
    result["runtime_status"] = (
        "partial" if result["incomplete"] else "completed_for_available_access"
    )
    result["status"] = "partial" if result["incomplete"] else "completed_with_findings"
    result["external_access_status"] = "blocked_missing_prerequisites"
    result["findings"] = [
        "BF16 G2/G4 probabilities changed with batch shape; controlled mask/head experiments did not resolve it.",
        "FP32 parity is reported separately; BF16 numerical failures are retained as observed limitations.",
        "Stress runtime success does not imply the 1000ms SLO passed at every concurrency/load.",
        "External Jev/Typesafe/AgentJev deployments have no live evaluation receipts.",
        "M2 is label-derived oracle text; workflows are deterministic replay fixtures, not real repairs/UI tasks.",
        "Historical CPU baseline outer Gemma fields were campaign context; inner MiniLM checkpoint identities are authoritative.",
        "Historical stress 04 server identity recorded its pre-multiplier offered rate; latest stress 05 records actual rates per cell.",
    ]
    return result


def render_report(summary):
    lines = [
        "# Live validation, 2026-10-01 (Asia/Taipei)",
        "",
        f"Runtime coverage: **{summary['runtime_status']}**; evidence outcome: **{summary['status']}**. This report is generated from actual receipts and saved predictions.",
        "Real runtime completion is separate from model quality, latency SLOs, and production acceptance.",
        "External deployment evaluations remain blocked by missing endpoint, identity, authentication, and cost prerequisites.",
        "",
        "## Execution evidence",
        "",
        "| Stage | Runtime result | Hardware | Wall seconds | Job |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for stage, source in summary["latest_stages"].items():
        job = f"[Modal]({source['job_url']})" if source.get("job_url") else "No linked job recorded"
        seconds = source.get("elapsed_seconds")
        lines.append(
            f"| {stage} | {source['outer_status']} / {source.get('result_status') or 'see nested results'} | "
            f"{source.get('gpu') or 'CPU'} | {seconds:.2f} | {job} |"
            if seconds is not None
            else f"| {stage} | {source['outer_status']} | {source.get('gpu') or 'CPU'} | unknown | {job} |"
        )
    if summary["incomplete"]:
        lines += ["", "Still incomplete: " + ", ".join(summary["incomplete"]) + "."]
    local = summary["local_checks"]
    if local.get("pytest"):
        lines += [
            "",
            f"Local pytest: {local['pytest'].get('passed')} passed, {local['pytest'].get('skipped')} skipped, "
            f"{local['pytest'].get('warnings')} warnings. Ruff/lock/build outcomes remain explicit in live-summary.json.",
        ]
    lines += [
        "",
        "## Public data and screening quality",
        "",
        "BoolQ keeps all passage/question text in a single fixed schema. The 256-case full train is a strict superset of the 128-case pilot; "
        "16 calibration and 128 official labeled validation cases are disjoint by ID, passage group, and request hash.",
        "Media uses 32 official MNIST test images and 20 official FSDD test recordings; speaker grouping leaves six independent audio groups.",
        "",
        "| Observation | Correct / decisions | Operational accuracy |",
        "| --- | ---: | ---: |",
    ]
    for row in summary["quality"]:
        if (
            row["stage"] in ("baselines", "pipeline", "laya")
            or row["model"].startswith("base:")
            or row["stage"] == "checkpoint-fp32"
        ):
            observed = row["observed"]
            lines.append(
                f"| {row['stage']} / {row['model']} | {observed['correct']} / {observed['recorded_decisions']} | "
                f"{observed['operational_accuracy']:.2%} |"
            )
    lines += [
        "",
        "M0 removes digit evidence and is separately labeled unknown. M2 uses source-label oracle descriptions; it is a diagnostic upper-information view, "
        "not independently transcribed input or a free product capability. M3 uses actual Tesseract and pinned CPU Whisper through real loopback HTTP; recognized text and errors are saved.",
    ]
    for row in summary["quality"]:
        if (row["stage"], row["model"]) in (
            ("training-full", "base:media"),
            ("pipeline", "M3"),
        ):
            tasks = ", ".join(
                f"{name}: {observed['correct']}/{observed['recorded_decisions']}"
                for name, observed in row["task_observations"].items()
            )
            lines.append(f"{row['stage']} / {row['model']} by task: {tasks}.")
    for row in summary["quality"]:
        rejected = row["reported"].get("expected_context_rejections")
        if rejected:
            lines.append(
                f"{row['model']} rejected {rejected} oversized context(s), retained as operational errors in the "
                f"{row['observed']['recorded_decisions']}-case denominator."
            )
    lines += [
        "These digit-only subsets support task-specific screening, not broad visual/audio quality claims.",
        "",
        "## Matched learning curves",
        "",
        "| Curve | Fits / cells | Support shots per class | Seeds | Actual optimizer updates | Runtime status |",
        "| --- | ---: | --- | --- | ---: | --- |",
    ]
    for name, training in summary["training"].items():
        lines.append(
            f"| {name}, CE and CE+Brier | {len(training['runs'])} | {training.get('shots_per_class')} | "
            f"{training.get('seeds')} | {training.get('optimizer_updates')} | {training.get('status')} |"
        )
    curve = summary["baseline_curve"]
    if curve:
        lines.append(
            f"| Prior / TFIDF / SetFit | {len(curve.get('fits', []))} | {curve.get('shots_per_class')} | "
            f"{curve.get('seeds')} | {curve.get('actual_setfit_optimizer_updates')} SetFit; prior/TFIDF closed form | {curve.get('status')} |"
        )
    grouped = defaultdict(list)
    for row in summary["quality"]:
        if (
            row["stage"] in ("training-full", "baselines-full")
            and row.get("method")
            and not row["model"].endswith(":media")
        ):
            grouped[(row["stage"], row["method"], row["shots"])].append(row)
    if grouped:
        lines += [
            "",
            "| Full curve / method | Shots/class | Seed count | Mean accuracy [min, max] | Mean NLL | Mean Brier sum | Mean ECE-15 |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
        for (stage, method, shots), rows in sorted(grouped.items()):
            values = [row["observed"]["operational_accuracy"] for row in rows]
            metrics = [row["reported"].get("metrics", row["reported"]) for row in rows]
            metric_means = [
                f"{statistics.mean(value[key] for value in metrics):.4f}"
                if all(value.get(key) is not None for value in metrics)
                else "unavailable"
                for key in ("nll", "brier_sum", "ece_15")
            ]
            lines.append(
                f"| {stage} / {method} | {shots} | {len(values)} | {statistics.mean(values):.2%} "
                f"[{min(values):.2%}, {max(values):.2%}] | {' | '.join(metric_means)} |"
            )
    lines += [
        "",
        f"Computed {len(summary['comparisons'])} paired source-group bootstrap comparisons using 2,000 resamples and seed 20260930. "
        "Intervals are unadjusted descriptive 95% CIs; no post-hoc noninferiority acceptance is performed. "
        "Every comparison checks saved request hashes and gold/group/task/schema/type metadata. Repetition 0 is added only in temporary comparison copies.",
        "Optimizer updates and reload equality are recorded per cell. Reload deserializes each adapter independently on the same frozen base; full merged model reload is not claimed.",
    ]
    lines += [
        "",
        "Selected descriptive comparisons below use all 128 fixed public text cases. The direction is candidate minus the frozen BF16 base; seed 0 rows illustrate the full curve without selecting the best seed.",
        "",
        "| Candidate | Accuracy delta, percentage points | 95% source-group CI |",
        "| --- | ---: | --- |",
    ]
    targets = {
        "training-full:shots-128-seed-0-ce:text",
        "training-full:shots-128-seed-0-ce_brier:text",
        "baselines-full:shots-128-seed-0-prior",
        "baselines-full:shots-128-seed-0-tfidf",
        "baselines-full:shots-128-seed-0-setfit",
        "laya:laya-general",
    }
    for comparison in summary["comparisons"]:
        if comparison["left"] == "training-full:base:text" and comparison["right"] in targets:
            delta = comparison["operational_accuracy_delta"]
            low, high = delta["interval"]
            lines.append(
                f"| {comparison['right']} | {delta['estimate'] * 100:+.3f} | "
                f"[{low * 100:+.3f}, {high * 100:+.3f}] |"
            )
    lines += ["", "## Batching, pipelines, workflows, and load", ""]
    for stage, batch in summary["batch_checks"].items():
        checks = ", ".join(
            f"{key}: {value.get('passed')}" for key, value in batch["checks"].items()
        )
        lines.append(
            f"- {stage} ({batch.get('precision')}): {batch['runner_run_count']} runner sweeps; {checks}."
        )
    lines += [
        "",
        "BF16 shape-dependent backbone/logit differences persisted in controlled mask and readout experiments. "
        "The explicit FP32 path and its rerun are a separate numerical control; default BF16 failures remain in the evidence. "
        "Batches, candidate counts, question counts, context lengths, and repeated-state sweeps verify execution, not representative task quality.",
    ]
    for name, workflow in summary["workflows"].items():
        statuses = {key: value["status"] for key, value in workflow["execution"].items()}
        lines += [
            "",
            f"Workflow {name}: {workflow['episodes']} policy-episode records; manifest {workflow['status']}; adapter execution {statuses}. "
            "Finite-state replay checks environment goal outcomes. It does not perform actual CI repairs or UI/document operations.",
            "",
            "| Policy | Successful / episodes | False completions | Fallback episode rate |",
            "| --- | ---: | ---: | ---: |",
        ]
        for policy, stats in workflow["policies"].items():
            lines.append(
                f"| {policy} | {stats['successful_episodes']} / {stats['episodes']} | "
                f"{stats['false_completions']} | {stats['fallback_episode_rate']:.2%} |"
            )
    for name, load in summary["load"].items():
        cells = load.get("cells", [])
        if not cells:
            continue
        lines += [
            "",
            f"Load {name}: {sum(cell['requests'] for cell in cells):,} requests, "
            f"{sum(cell['successful'] for cell in cells):,} successful, "
            f"{sum(cell.get('observed_backbone_forwards', cell['actual_model_forwards']) for cell in cells):,} recorded real model forwards.",
            "",
            "| Workload | Successful / requests | Offered requests/s | Peak inflight | p99 ms | SLO misses |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for cell in cells:
            systems = cell["systems"]
            rate = cell.get("offered_requests_per_second")
            rate_text = (
                f"{rate:.4f}"
                if rate is not None
                else "closed loop"
                if cell["id"].startswith("closed_loop")
                else "not recorded"
            )
            lines.append(
                f"| {cell['id']} | {cell['successful']} / {cell['requests']} | {rate_text} | {cell.get('peak_client_inflight', 'not recorded')} | "
                f"{systems.get('successful_p99_ms', 0):.3f} | {systems.get('deadline_misses')} |"
            )
            if cell.get("p99_block_bootstrap_95ci_ms"):
                lines.append(
                    f"\n10,000-request p99 block-bootstrap 95% CI: {cell['p99_block_bootstrap_95ci_ms']} ms. "
                    "Twenty contiguous time blocks from one machine/workload; no deployment-population claim.\n"
                )
    lines += [
        "",
        "The serving checks verify a protected /decide route and the intentionally public health route. "
        "High-concurrency stress can complete all forwards while failing the 1,000 ms SLO. Loopback tests do not measure WAN/deployment latency.",
        "Historical stress 04 recorded the offered rate before workload multipliers; latest stress 05 supersedes it with per-cell actual rates. Historical receipts retain their measured latencies and hardware identities.",
        "",
        "## Actual fitted-artifact and telemetry boundaries",
        "",
    ]
    integrity = summary["integrity"]
    if integrity:
        lines += [
            f"Integrity status: {integrity['status']}; independently audited {integrity['verified_boundary_checks']} boundary checks, "
            f"{integrity['verified_rejected_dimensions']} train/calibration ID/group/request rejection dimensions across "
            f"{len(integrity['trained_artifacts'])} actual saved LoRA artifacts, and {integrity['verified_disjoint_heldout_checks']} disjoint heldout acceptance checks.",
            "Actual FP32 workflow telemetry was present before four real backbone forwards. Revision and precision mismatches each stopped with zero forwards; "
            "provenance overlap fixtures stopped before adapter loading. A genuinely fitted CPU prior rejected request-only overlap in an unreachable state while episode IDs/groups and the initial request stayed disjoint.",
            "Expected setup_error/not_run child outcomes are passing boundary observations, not successful model executions. "
            "Blocked adapter checks inspect actual artifact/provenance hashes without reloading their weights; original adapter reload checks remain separate.",
        ]
    lines += [
        "",
        "## Provenance, costs, and remaining access",
        "",
    ]
    for name, source in summary["data"]["sources"].items():
        lines.append(
            f"- {name}: [{source['license']}]({source['license_source']}), "
            f"[source]({source['source_url']}), immutable revision `{source['source_revision']}`."
        )
    lines += [
        "",
        f"Lock SHA-256: `{summary['lock_sha256']}`. Dataset/source/adapter hashes, model revisions, actual counts and job links are in "
        "[live-summary.json](live-summary.json). Raw predictions, receipts and comparison JSON remain under artifacts/live-validation.",
        "Reported per-request preprocessing/provider costs remain unknown. Operator ceilings are reservations, not provider invoices. "
        f"The completed billing buckets sum to ${summary['accounting']['recorded_billed_usd']:.6f}; pending/unreported buckets prevent a final bill claim.",
        "",
        "External provider runs are not completed:",
        "",
    ]
    for provider in summary["external_providers"]:
        lines.append(
            f"- {provider['model']}: `{provider['endpoint_env']}`; token configuration: `{provider['token_env']}`; "
            "an accessible deployment with immutable resolved model revision and provider cost metadata or an explicit unknown-cost policy."
        )
    cost = summary["accounting"]["execution_window_estimate"]
    final_apps = summary["accounting"].get("apps-final.json", {}).get("recorded", [])
    lines += [
        "",
        f"Using recorded official A100-80GB/CPU/memory rates, the unreported execution windows estimate "
        f"${cost['unreported_windows_gpu_only_usd']:.4f} GPU-only to ${cost['unreported_windows_allocation_usd']:.4f} with requested CPU/host memory. "
        f"Recorded billing plus those window allocations is approximately ${cost['recorded_bill_plus_unreported_window_estimate_usd']:.4f}. "
        "This is not a final invoice: initialization outside receipts, storage/egress and pending bucket extensions are excluded. "
        "The $20 cap applied to this chosen campaign; function timeout ceilings are not automatically summed as a budget.",
        f"The final app snapshot records {len(final_apps)} jobs, all stopped with zero tasks: "
        f"{bool(final_apps) and all(app.get('state') == 'stopped' and str(app.get('tasks')) == '0' for app in final_apps)}.",
        "Historical CPU baseline outer Gemma fields described campaign context. Actual inner MiniLM identities and resolved runtime telemetry are authoritative.",
        "",
        "## Reproduction",
        "",
        "Use an authenticated Modal profile with the `gemma-unified-system-one` volume. These commands start ephemeral jobs; they do not deploy a service. "
        "Use a fresh run name and preserve receipts/datasets before the next stage.",
        "",
        "```sh",
        "uv sync --all-extras --group dev",
        "uv run python scripts/prepare_live_validation_data.py --output artifacts/live-validation/datasets",
        "uv run modal run apps/modal/validate_v2.py --stage checkpoint --run your-run",
        "uv run modal run apps/modal/validate_v2.py --stage checkpoint-fp32 --run your-run",
        "uv run modal run apps/modal/validate_v2.py --stage training-full --run your-run",
        "uv run modal run apps/modal/validate_v2.py --stage baselines-full --run your-run",
        "uv run modal run apps/modal/validate_v2.py --stage services --run your-run",
        "uv run modal run apps/modal/validate_v2.py --stage stress --run your-run",
        "uv run modal run apps/modal/validate_v2.py --stage integrity --run your-run --training-run your-run",
        "uv run python scripts/summarize_live_validation.py",
        "```",
        "",
        "Pilot training, six CPU baseline methods and diagnostics use `--stage training`, `--stage baselines`, and `--stage diagnostics`. "
        "Integrity must use the run containing completed training-full artifacts; choose per-stage run names when rerunning an existing stage.",
    ]
    lines += [
        "",
        "Public small-sample screening does not establish broad visual/audio understanding, pretraining cleanliness, production reliability, "
        "a statistically accepted replacement decision, or published model/deployment readiness.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--job", action="append", default=[], help="run/stage=Modal URL")
    args = parser.parse_args()
    jobs = dict(item.split("=", 1) for item in args.job)
    summary = aggregate(args.root, jobs=jobs)
    base = args.root / "artifacts/live-validation"
    _write(
        base / "paired-comparisons.json",
        {
            "policy": summary["statistical_policy"],
            "comparisons": summary["comparisons"],
            "rejections": summary["comparison_rejections"],
        },
    )
    _write(base / "final-summary.json", summary)
    compact = {key: value for key, value in summary.items() if key != "accounting"}
    compact["quality"] = [
        {**row, "reported": _compact_metrics(row["reported"])} for row in summary["quality"]
    ]
    compact["comparisons"] = [
        {
            key: value[key]
            for key in (
                "left",
                "right",
                "kind",
                "direction",
                "common_decisions",
                "population",
                "operational_accuracy_delta",
                "conditional_accuracy_delta",
            )
        }
        for value in summary["comparisons"]
    ]
    compact["baseline_curve"] = {
        **summary["baseline_curve"],
        "fits": [
            {**fit, "training": _compact_metrics(fit.get("training"))}
            for fit in summary["baseline_curve"].get("fits", [])
        ],
    }
    compact["integrity"] = {
        **summary["integrity"],
        "artifact_dimension_rejection_reason": "evaluation_overlaps_adapter_training",
        "trained_artifacts": [
            {
                **artifact,
                "rejected_dimensions": [
                    f"{entry['split']}/{entry['dimension']}"
                    for entry in artifact["rejected_dimensions"]
                ],
            }
            for artifact in summary["integrity"].get("trained_artifacts", [])
        ],
    }
    compact["accounting"] = {
        "recorded_billed_usd": summary["accounting"]["recorded_billed_usd"],
        "scope": summary["accounting"]["scope"],
        "execution_window_estimate": summary["accounting"]["execution_window_estimate"],
        "final_apps": summary["accounting"].get("apps-final.json", {}).get("recorded"),
        "artifact_sha256": {
            key: value["artifact_sha256"]
            for key, value in summary["accounting"].items()
            if isinstance(value, dict) and "artifact_sha256" in value
        },
    }
    compact["raw_summary_artifact"] = {
        "path": "artifacts/live-validation/final-summary.json",
        "sha256": _hash(base / "final-summary.json"),
    }
    compact["raw_comparisons_artifact"] = {
        "path": "artifacts/live-validation/paired-comparisons.json",
        "sha256": _hash(base / "paired-comparisons.json"),
    }
    docs = args.root / "docs/validation/2026-10-01"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "live-summary.json").write_text(
        json.dumps(compact, separators=(",", ":"), allow_nan=False) + "\n"
    )
    if (docs / "live-summary.json").stat().st_size >= 500_000:
        raise ValueError("committable evidence summary exceeds 500KB")
    (docs / "live-validation.md").write_text(render_report(summary))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "receipts": len(summary["receipts"]),
                "comparisons": len(summary["comparisons"]),
                "comparison_rejections": len(summary["comparison_rejections"]),
                "incomplete": summary["incomplete"],
                "committable_json_bytes": (docs / "live-summary.json").stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
