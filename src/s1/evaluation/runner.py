"""Sequential E1 runner with frozen eligibility and gold-free adapter inputs."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Literal

from pydantic import Field

from s1.contracts import StrictModel
from s1.errors import BackendTimeoutError

from .adapters import execution_blockers, load_adapter
from .artifacts import Artifacts
from .contracts import ModelSpec
from .datasets import fingerprint, load_cases
from .planning import create_plan
from .reporting import write_summary
from .responses import execution_info, normalize, pipeline_info


class RuntimeTelemetry(StrictModel):
    revision: str | None = None
    device: str | None = None
    precision: Literal["float64", "float32", "float16", "bfloat16", "quantized", "provider"]
    temperature: float | None = Field(default=None, gt=0)
    calibration_temperature: float | None = Field(default=None, gt=0)
    context_limit: int | None = Field(default=None, ge=1)
    context_policy: Literal["reject", "truncate", "provider", "not_applicable"]


def _telemetry(adapter, spec):
    result = RuntimeTelemetry.model_validate(adapter.telemetry())
    if spec.adapter in ("gemma", "laya") and result.revision != spec.revision:
        raise ValueError("resolved model revision differs from the plan")
    if result.precision != spec.precision:
        raise ValueError("resolved precision differs from the plan")
    return result.model_dump(mode="json")


def _predictions(case, request_record, response):
    for question in case.request.questions:
        row = {
            key: request_record[key]
            for key in (
                "request_id",
                "model_id",
                "experiment_id",
                "repetition",
                "eligibility",
                "status",
            )
        }
        row.update(
            **case.metadata(),
            question_id=question.id,
            type=question.type,
            gold=case.gold[question.id],
            actual_label=None,
            standardized_argmax=None,
            probabilities=None,
        )
        row.update(
            labels=question.labels(),
            score_values=question.score_values() if question.type == "score" else None,
            soft_gold=(case.soft_gold or {}).get(question.id),
            critical=question.id in (case.critical_questions or []),
            modalities=sorted({"text", *(m.type for m in case.request.media)}),
            candidate_count=len(question.labels()),
            questions_per_request=len(case.request.questions),
        )
        if response is not None:
            answer = response["answers"][question.id]
            dist = answer["probabilities"]
            if dist is None:
                row["actual_label"] = answer["label"]
                if question.type == "score":
                    row["score"] = question.score_values()[question.labels().index(answer["label"])]
                    row["gold_score"] = question.score_values()[
                        question.labels().index(row["gold"])
                    ]
                yield row
                continue
            best = max(question.labels(), key=dist.__getitem__)
            row.update(
                probabilities=dist,
                standardized_argmax=best,
                actual_label=answer["choice"] if question.type == "choice" else best,
            )
            if question.type == "noul":
                row["noul"] = answer["noul"]
            if question.type == "score":
                row["score"] = answer["score"]
                row["gold_score"] = question.score_values()[
                    question.labels().index(case.gold[question.id])
                ]
        yield row


def _record_call(
    artifacts,
    adapter,
    cell,
    case,
    repetition,
    index,
    phase,
    eligibility,
    now,
    reason=None,
    completed=None,
    slo_ms=None,
):
    row = {
        "request_id": f"{cell['model_id']}:{repetition}:{phase}:{index}",
        "model_id": cell["model_id"],
        "experiment_id": cell["experiment_id"],
        **case.metadata(),
        "repetition": repetition,
        "phase": phase,
        "decisions": len(case.request.questions),
        "eligibility": eligibility,
        "status": "not_run" if reason else eligibility,
        "reason": reason,
        "scheduled_at_s": None,
        "dispatched_at_s": None,
        "all_required_answers_ready_at_s": None,
        "terminated_at_s": None,
        "latency_ms": None,
    }
    response = None
    if completed is not None:
        completed = dict(completed)
        response = completed.pop("response")
        error = completed.pop("error_type")
        row.update(completed)
        if error:
            artifacts.append(
                "errors",
                {
                    "model_id": cell["model_id"],
                    "request_id": row["request_id"],
                    "phase": phase,
                    "error_type": error,
                    "status": row["status"],
                },
            )
    elif adapter is not None and eligibility == "eligible" and reason is None:
        row["scheduled_at_s"] = row["dispatched_at_s"] = now()
        try:
            # A mutating adapter cannot change golds, later repetitions or normalization.
            result = adapter.predict(case.request.model_copy(deep=True))
            row["adapter_execution"] = execution_info(result)
            row["pipeline"] = pipeline_info(result.get("pipeline"))
            row["probability_postprocessing"] = (
                "bounded_four_decimal_renormalization"
                if result.get("probability_postprocessing")
                == "bounded_four_decimal_renormalization"
                else None
            )
            response = normalize(
                case.request, result, cell["identity"]["model"]["capabilities"]["probabilities"]
            )
        except BaseException as exc:
            row["pipeline"] = pipeline_info(getattr(exc, "evaluation_pipeline", None))
            interrupted = not isinstance(exc, Exception)
            row["status"] = (
                "interrupted"
                if interrupted
                else "timeout"
                if isinstance(exc, BackendTimeoutError)
                else "error"
            )
            row["terminated_at_s"] = now()
            artifacts.append(
                "errors",
                {
                    "model_id": cell["model_id"],
                    "request_id": row["request_id"],
                    "phase": phase,
                    "error_type": type(exc).__name__,
                    "status": row["status"],
                },
            )
            if interrupted:
                artifacts.append("requests", row)
                if phase == "measurement":
                    for prediction in _predictions(case, row, None):
                        artifacts.append("predictions", prediction)
                raise
        else:
            row["status"] = "ok"
            row["all_required_answers_ready_at_s"] = row["terminated_at_s"] = now()
            row["latency_ms"] = (row["terminated_at_s"] - row["dispatched_at_s"]) * 1000
        row["deadline_miss"] = (
            (row["status"] != "ok" or row["latency_ms"] > slo_ms) if slo_ms is not None else None
        )
    artifacts.append("requests", row)
    if phase == "measurement":
        for prediction in _predictions(case, row, response):
            artifacts.append("predictions", prediction)
    return row["status"]


def execute(
    registry,
    profile_id,
    *,
    output,
    lockfile,
    environment=None,
    enable_models=(),
    adapter_factory=load_adapter,
):
    # Copy environment once so endpoint identity and construction use the same values.
    environment = dict(environment or {})
    plan = create_plan(
        registry,
        profile_id,
        lockfile=lockfile,
        environment=environment,
        enable_models=enable_models,
    )
    profile = registry.profiles[profile_id]
    cases = load_cases(registry.dataset_path(registry.suites[profile.suite]))
    if fingerprint(cases) != plan["dataset_sha256"]:
        raise ValueError("dataset changed after preflight")
    started = time.perf_counter()

    def now():
        return time.perf_counter() - started

    manifest = {
        "schema_version": 2,
        "status": "running",
        "plan": plan,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "timing": "monotonic seconds since run start; latency measured from scheduled arrival",
        "execution_mode": profile.load_mode,
        "retries": 0,
        "runner_result_cache": False,
        "seed_policy": "recorded_only; no stochastic sampling or support selection in E1",
        "execution": {
            cell["model_id"]: {"status": "not_run", "reasons": ["not_started"], "telemetry": None}
            for cell in plan["models"]
        },
    }
    with Artifacts(output) as artifacts:
        artifacts.manifest(manifest)
        try:
            for cell in plan["models"]:
                spec = ModelSpec.model_validate(cell["identity"]["model"])
                reasons = list(cell["reasons"])
                if not plan["can_execute"]:
                    reasons.append("plan_not_executable")
                reasons.extend(execution_blockers(spec, profile))
                execution = {
                    "status": "not_run",
                    "reasons": list(dict.fromkeys(reasons)),
                    "telemetry": None,
                }
                manifest["execution"][spec.id] = execution
                adapter = None
                failures = False
                running = False
                try:
                    if not reasons:
                        try:
                            setup_started = now()
                            adapter = adapter_factory(spec, environment)
                            execution["telemetry"] = _telemetry(adapter, spec)
                            execution["setup_ms"] = (now() - setup_started) * 1000
                            execution["status"] = "running"
                        except Exception as exc:
                            execution["status"] = "setup_error"
                            execution["reasons"].append("adapter_setup_failed")
                            artifacts.append(
                                "errors",
                                {
                                    "model_id": spec.id,
                                    "phase": "setup",
                                    "error_type": type(exc).__name__,
                                },
                            )
                    # Persist the setup result before making the first prediction.
                    artifacts.manifest(manifest)
                    eligible = [
                        case
                        for case, entry in zip(cases, cell["cases"])
                        if entry["eligibility"] == "eligible"
                    ]
                    running = execution["status"] == "running"
                    use_loadgen = (
                        profile.batch_size > 1
                        or profile.concurrency > 1
                        or profile.load_mode != "closed_loop"
                    )
                    for repetition in range(profile.repetitions):
                        if running:
                            for index in range(profile.warmup):
                                status = _record_call(
                                    artifacts,
                                    adapter,
                                    cell,
                                    eligible[0],
                                    repetition,
                                    index,
                                    "warmup",
                                    "eligible",
                                    now,
                                )
                                failures |= status in ("error", "timeout")
                        for index, (case, entry) in enumerate(zip(cases, cell["cases"])):
                            if running and use_loadgen and entry["eligibility"] == "eligible":
                                continue
                            status = _record_call(
                                artifacts,
                                adapter if running else None,
                                cell,
                                case,
                                repetition,
                                index,
                                "measurement",
                                entry["eligibility"],
                                now,
                                reason=None if running else execution["reasons"],
                                slo_ms=profile.slo_ms,
                            )
                            failures |= status in ("error", "timeout")
                        if running and use_loadgen:
                            from .loadgen import run_load

                            indexed = [
                                (i, case)
                                for i, (case, entry) in enumerate(zip(cases, cell["cases"]))
                                if entry["eligibility"] == "eligible"
                            ]
                            for index, case, completed in run_load(
                                adapter,
                                indexed,
                                profile,
                                spec.capabilities.probabilities,
                                now,
                                repetition,
                            ):
                                status = _record_call(
                                    artifacts,
                                    None,
                                    cell,
                                    case,
                                    repetition,
                                    index,
                                    "measurement",
                                    "eligible",
                                    now,
                                    completed=completed,
                                )
                                failures |= status in ("error", "timeout")
                    if running:
                        execution["status"] = "completed_with_errors" if failures else "completed"
                        if hasattr(adapter, "resources"):
                            execution["resources"] = adapter.resources()
                finally:
                    if adapter is not None:
                        try:
                            adapter.close()
                        except Exception as exc:
                            artifacts.append(
                                "errors",
                                {
                                    "model_id": spec.id,
                                    "phase": "close",
                                    "error_type": type(exc).__name__,
                                },
                            )
                            execution["status"] = (
                                "completed_with_errors" if running else "setup_error"
                            )
                    # Release model references before loading the next backend.
                    adapter = None
                    artifacts.manifest(manifest)
            statuses = [state["status"] for state in manifest["execution"].values()]
            if any(status in ("setup_error", "completed_with_errors") for status in statuses):
                manifest["status"] = "completed_with_errors"
            elif any(status == "completed" for status in statuses):
                missing_enabled = any(
                    cell["identity"]["model"]["enabled"]
                    and manifest["execution"][cell["model_id"]]["status"] == "not_run"
                    for cell in plan["models"]
                )
                manifest["status"] = "partial" if missing_enabled else "completed"
            else:
                manifest["status"] = "not_run"
        except BaseException:
            manifest["status"] = "interrupted"
            for execution in manifest["execution"].values():
                if execution["status"] == "running":
                    execution["status"] = "interrupted"
            raise
        finally:
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            artifacts.manifest(manifest)
    return write_summary(output)
