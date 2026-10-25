"""Explicit operation-first browser scheduling for native specialist decisions.

The generic compatibility and full-head hierarchy remain separate profiles.
This profile preserves all input candidates but evaluates only operation and its
selected target. Native numerical equivalence and performance require later
same-input model replay; independent-question source contracts alone do not prove
them, particularly when splitting a native Decider shared/batch execution path.
"""

from __future__ import annotations

import copy
import hashlib
import time

from .backends import normalize_response
from .contracts import DecisionRequest, Question, validate_distribution
from .errors import BackendResponseError, RequestValidationError
from .jev import _parse_payload, _singleton_answer
from .jev_generated import TARGETS, _request
from .jev_hierarchy import JevHierarchicalAdapter, _evidence

PROFILE = "jev_browser_operation_first_v1"


def profile_manifest():
    return {
        "profile": PROFILE,
        "scope": "browser-only native specialist operation and selected active target",
        "original_request": "all original heads/candidates/order preserved; no target pruning",
        "unused_target_heads": "not_evaluated; no distributions fabricated",
        "active_probabilities": "native calibrated per-head or complete hierarchical group-times-conditional distribution",
        "budget_coverage": "original any target head>20: one cooperative20s; original narrow: HTTP25 only",
        "http_timeout_enforcement": "unchanged upstream browser client; this synchronous adapter does not enforce or hard-cancel HTTP25",
        "limitations": [
            "different scheduling and failure behavior from full-head native hierarchy",
            "unused-head runtime/context failures intentionally no longer block this profile",
            "narrow multi-question requests may use more backend calls after splitting",
            "blocking calls cannot be hard-cancelled; HTTP delivery differs from computation",
            "native numerical/argmax equivalence, speed and task success are unverified",
        ],
    }


class JevBrowserOperationFirstAdapter:
    """One configured native specialist; no changes to its checkpoint or prompts."""

    def __init__(self, backend, *, model_identity=None, evidence_sink=None):
        spec = getattr(backend, "spec", None)
        capabilities = getattr(spec, "capabilities", None)
        if (
            getattr(spec, "adapter", None) not in {"decider", "jev_omni"}
            or getattr(spec, "calibration", None) != "checkpoint"
            or getattr(spec, "precision", None) != "bfloat16"
            or getattr(capabilities, "independent_questions", None) is not True
        ):
            raise ValueError(
                "operation-first browser profile requires a native BF16 checkpoint-calibrated independent-question specialist"
            )
        self.backend = backend
        self.builder = JevHierarchicalAdapter(
            backend, model_identity=model_identity, batch_stages=False
        )
        self.identity = self.builder.compatible.model_identity
        self.evidence_sink = evidence_sink

    def _emit(self, event, audit, *, stage_index=None):
        if self.evidence_sink is None:
            return
        data = {"profile": PROFILE, "status": audit["status"]}
        if event == "planned":
            data.update(
                original_request=audit["original_request"],
                original_request_sha256=audit["original_request_sha256"],
                original_head_labels=audit["original_head_labels"],
                budget_coverage=audit["budget_coverage"],
            )
        elif stage_index is not None:
            data["stage_index"] = stage_index
            stage = audit["stages"][stage_index]
            keys = (
                ("request", "scope", "head", "group")
                if event == "stage_started"
                else ("raw_response",)
                if event == "stage_raw_returned"
                else tuple(stage)
            )
            data["stage"] = {key: stage[key] for key in keys if key in stage}
        else:
            data.update(
                selected_operation=audit.get("selected_operation"),
                active_head=audit.get("active_head"),
                unused_target_heads=audit.get("unused_target_heads"),
                unused_target_heads_status="not_evaluated",
                backend_predict_calls=audit["backend_predict_calls"],
                error=audit.get("error"),
            )
        try:
            self.evidence_sink({"event": event, "audit": _evidence(data)})
        except BaseException as exc:
            audit.setdefault("evidence_sink_errors", []).append(
                {"type": type(exc).__name__, "message": str(exc)}
            )

    def _deadline(self, audit, started):
        if audit.get("original_request_wide"):
            self.builder._deadline(started)

    def _run(self, stage, audit, started, identities, reported_usage):
        index = len(audit["stages"])
        audit["stages"].append(stage.evidence)
        stage_started, error_stage = time.monotonic(), "preflight"
        try:
            self._deadline(audit, started)
            if stage.request is None:
                result = {
                    "answers": copy.deepcopy(stage.forced_answer),
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
                stage.evidence["status"] = "deterministic"
            else:
                preflight_started = time.monotonic()
                try:
                    stage.evidence["preflight"] = self.builder._preflight(stage.request)
                finally:
                    stage.evidence["preflight_elapsed_ms"] = (
                        time.monotonic() - preflight_started
                    ) * 1000
                self._deadline(audit, started)
                error_stage = "prediction"
                audit["backend_predict_calls"] += 1
                audit["native_question_rows"] += len(stage.request.questions)
                stage.evidence["status"] = "running"
                self._emit("stage_started", audit, stage_index=index)
                raw = self.backend.predict(stage.request.model_copy(deep=True))
                stage.evidence["raw_response"] = _evidence(raw)
                self._emit("stage_raw_returned", audit, stage_index=index)
                result = normalize_response(stage.request, raw)
                model = result.get("model", self.backend.name)
                if not isinstance(model, str) or not model.strip():
                    raise BackendResponseError("backend model identity must be nonempty text")
                identities.add(model)
                if len(identities) > 1:
                    raise BackendResponseError(
                        "operation-first backend reported different models across stages"
                    )
                stage.evidence.update(status="completed", model=model)
            stage.evidence["response"] = _evidence(result)
            stage.evidence["usage"] = _evidence(result.get("usage"))
            reported_usage.append(result.get("usage"))
            self._deadline(audit, started)
            return result
        except BaseException as exc:
            stage.evidence.update(
                status="failed",
                error_stage=error_stage,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            if hasattr(exc, "generation"):
                stage.evidence["error_generation"] = _evidence(exc.generation)
            raise
        finally:
            stage.evidence["elapsed_ms"] = (time.monotonic() - stage_started) * 1000
            stage.evidence["latency_scope"] = (
                "preflight_and_native_stage_without_device_synchronize"
            )
            self._emit("stage_finished", audit, stage_index=index)

    def _project(self, payload, head):
        projected = copy.deepcopy(payload)
        projected["questions"] = {head: copy.deepcopy(payload["questions"][head])}
        return projected

    def _ordinary_stage(self, payload, head, *, scope):
        parsed = _parse_payload(self._project(payload, head), model_identity=self.identity)
        question = parsed.questions[0]
        if len(question.labels()) == 1:
            stage = self.builder._stage(
                None, scope=scope, head=head, forced_answer={head: _singleton_answer(question)}
            )
            stage.evidence["request"] = {
                "state": copy.deepcopy(parsed.state),
                "questions": [question.model_dump()],
                "mathematically_forced_singleton": True,
            }
            return stage
        return self.builder._stage(
            DecisionRequest(
                state=copy.deepcopy(parsed.state),
                questions=[Question.model_validate(question.model_dump())],
            ),
            scope=scope,
            head=head,
        )

    def predict(self, payload):
        started = time.monotonic()
        audit = {
            "version": PROFILE,
            "profile": profile_manifest(),
            "status": "planning",
            "heads": {},
            "stages": [],
            "backend_predict_calls": 0,
            "backend_predict_batch_calls": 0,
            "native_question_rows": 0,
            "unused_target_heads_status": "not_evaluated",
            "timeout_policy": "original_width_based; cooperative_before_and_after_blocking_calls_no_hard_cancel",
            "limits": {
                "branch_width": 20,
                "targets_per_head": 400,
                "native_stage_requests": 64,
                "wide_budget_seconds": 20,
            },
            "original_request": _evidence(payload),
        }
        try:
            _, source, labels = _request(payload, self.identity)
            audit.update(
                original_request=copy.deepcopy(payload),
                original_request_sha256=hashlib.sha256(source).hexdigest(),
                original_head_labels=copy.deepcopy(labels),
                payload_bytes=len(source),
                original_request_wide=any(
                    len(values) > 20 for head, values in labels.items() if head != "operation"
                ),
            )
            audit["budget_coverage"] = (
                "wide20_cooperative_and_HTTP25"
                if audit["original_request_wide"]
                else "narrow_HTTP25_only_no_native20"
            )
            self._deadline(audit, started)
            self._emit("planned", audit)
            identities, usages = set(), []
            operation_stage = self._ordinary_stage(payload, "operation", scope="operation")
            operation_result = self._run(operation_stage, audit, started, identities, usages)
            operation_answer = operation_result["answers"]["operation"]
            operation = operation_answer["choice"]
            active = TARGETS.get(operation)
            audit.update(
                selected_operation=operation,
                active_head=active,
                unused_target_heads=[
                    head for head in labels if head != "operation" and head != active
                ],
            )
            self._emit("operation_selected", audit)
            answers = {"operation": operation_answer}
            singleton_heads = []
            if active:
                self._deadline(audit, started)
                audit["active_labels"] = copy.deepcopy(labels[active])
                if len(labels[active]) <= 20:
                    stage = self._ordinary_stage(payload, active, scope="active_target")
                    result = self._run(stage, audit, started, identities, usages)
                    answers[active] = result["answers"][active]
                    if stage.request is None:
                        singleton_heads.append(active)
                else:
                    target_audit = {"heads": {}, "stages": []}
                    plan = self.builder._plan(self._project(payload, active), target_audit)
                    if plan is None:
                        raise RequestValidationError(
                            "active wide target failed to produce its complete hierarchy"
                        )
                    _, stages = plan
                    if (
                        sum(stage.request is not None for stage in stages)
                        + audit["backend_predict_calls"]
                        > 64
                    ):
                        raise RequestValidationError(
                            "operation-first native stage work limit exceeded"
                        )
                    audit["heads"] = target_audit["heads"]
                    metadata = audit["heads"][active]
                    conditionals = []
                    for stage in stages:
                        result = self._run(stage, audit, started, identities, usages)
                        if stage.evidence["scope"] == "group":
                            metadata["group_probabilities"] = result["answers"][active][
                                "probabilities"
                            ]
                        else:
                            conditionals.append(
                                (stage.evidence["group"], result["answers"][active])
                            )
                    joint = {}
                    for group, answer in conditionals:
                        for label, probability in answer["probabilities"].items():
                            joint[label] = metadata["group_probabilities"][group] * probability
                    joint = {label: joint[label] for label in metadata["original_labels"]}
                    try:
                        validate_distribution(joint.values(), len(joint))
                    except ValueError as exc:
                        raise BackendResponseError(
                            "invalid complete active hierarchy distribution"
                        ) from exc
                    metadata["joint_probabilities"] = joint
                    best = max(joint, key=joint.__getitem__)
                    probabilities = sorted(joint.values(), reverse=True)
                    answers[active] = {
                        "id": active,
                        "type": "choice",
                        "choice": best,
                        "probabilities": joint,
                        "confidence": joint[best],
                        "margin": probabilities[0] - probabilities[1],
                    }
            self._deadline(audit, started)
            audit.update(status="completed", elapsed_ms=(time.monotonic() - started) * 1000)
            self._emit("operation_first_returned", audit)
            usage = {}
            for key in ("input_tokens", "output_tokens"):
                values = [item.get(key) if isinstance(item, dict) else None for item in usages]
                usage[key] = (
                    sum(values)
                    if all(type(value) is int and value >= 0 for value in values)
                    else None
                )
            return {
                "model": next(iter(identities), self.backend.name),
                "answers": answers,
                "usage": usage,
                "hierarchy": audit,
                "jev_browser_profile": PROFILE,
                "unused_target_heads": audit["unused_target_heads"],
                "unused_target_heads_status": "not_evaluated",
                "jev_compatibility": {
                    "model_identity": self.identity,
                    "singleton_heads": singleton_heads,
                },
            }
        except BaseException as exc:
            previous = getattr(exc, "hierarchy", None)
            if previous is not None and previous is not audit:
                audit["backend_error_hierarchy"] = _evidence(previous)
            audit.update(
                status="failed",
                elapsed_ms=(time.monotonic() - started) * 1000,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            exc.hierarchy = audit
            if not hasattr(exc, "generation"):
                exc.generation = audit
            self._emit("operation_first_failed", audit)
            raise
