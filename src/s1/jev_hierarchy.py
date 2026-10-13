"""Explicit hierarchical Jev choices, with every conditional distribution measured.

The core 52-option contract is unchanged. This adapter is a separate opt-in
policy: recognized browser target heads above the branching limit are factored
into group and conditional choices. Its probabilities describe that hierarchy,
not the calibration of a flat native classifier. Timeouts are cooperative;
blocking backend calls cannot be interrupted by this synchronous adapter.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass

from pydantic import ValidationError

from .backends import normalize_response
from .contracts import DecisionRequest, Question, validate_distribution
from .errors import BackendResponseError, BackendTimeoutError, RequestValidationError
from .jev import (
    SINGLETON_TARGETS,
    JevCompatibilityAdapter,
    _json_text,
    _parse_payload,
    _singleton_answer,
)


def _evidence(value, depth=0):
    """Keep invalid provider output inspectable in a JSON error response."""
    if depth > 64:
        return {"evidence_error": "nesting exceeds 64 levels"}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            return {
                "non_json_mapping": [
                    {"key": _evidence(key, depth + 1), "value": _evidence(item, depth + 1)}
                    for key, item in value.items()
                ]
            }
        return {key: _evidence(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_evidence(item, depth + 1) for item in value]
    if type(value) is float and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if value is None or type(value) in (str, int, float, bool):
        return value
    return {"non_json_type": type(value).__qualname__}


def _json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _check_json(value, depth=0):
    if depth > 64:
        raise RequestValidationError("Jev hierarchy JSON nesting exceeds 64 levels")
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise RequestValidationError("Jev hierarchy JSON object keys must be strings")
        for item in value.values():
            _check_json(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _check_json(item, depth + 1)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise RequestValidationError("Jev hierarchy requires JSON data")


def _criteria(raw, question_id):
    present = [name for name in ("criteria", "options", "levels") if name in raw]
    if len(present) > 1:
        raise RequestValidationError("use criteria or options/levels, not both")
    name = present[0] if present else "criteria"
    value = raw.get(name)
    if isinstance(value, dict):
        labels, descriptions = list(value), list(value.values())
    elif isinstance(value, list):
        labels, descriptions = list(value), list(value)
    else:
        return name, value, None
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise RequestValidationError(f"question {question_id!r} needs nonempty string labels")
    if len(set(labels)) != len(labels):
        raise RequestValidationError(f"question {question_id!r} has duplicate labels")
    if isinstance(value, list) and any(not isinstance(item, str) for item in descriptions):
        raise RequestValidationError("list criteria must contain strings")
    # Validate EVERY original description, including those outside the small
    # representative request used to validate the unchanged outer contract.
    normalized = {
        label: _json_text(description, field=f"question {question_id!r} candidate {label!r}")
        for label, description in zip(labels, descriptions)
    }
    return name, value, normalized


@dataclass
class _Stage:
    request: DecisionRequest | None
    evidence: dict
    forced_answer: dict | None = None


class JevHierarchicalAdapter:
    """One configured backend and a bounded, complete two-level decision tree.

    Small requests use JevCompatibilityAdapter exactly. A hierarchy stages every
    group before returning any answer; failures expose ``hierarchy`` and
    ``generation`` evidence on the original exception, never partial answers.
    With explicit ``batch_stages=True``, all independent native stage requests
    use one supported backend batch call, preserving the ordered-list contract.
    """

    def __init__(
        self,
        backend,
        *,
        model_identity=None,
        branch_width=20,
        max_targets=400,
        max_stage_calls=64,
        max_payload_bytes=1_000_000,
        budget_seconds=20,
        batch_stages=False,
    ):
        if type(batch_stages) is not bool:
            raise ValueError("batch_stages must be a boolean")
        for value, name, maximum in (
            (branch_width, "branch_width", 20),
            (max_targets, "max_targets", 400),
            (max_stage_calls, "max_stage_calls", 64),
            (max_payload_bytes, "max_payload_bytes", 1_000_000),
        ):
            if type(value) is not int or value < 1 or value > maximum:
                raise ValueError(f"{name} must be an integer in 1..{maximum}")
        if branch_width < 2:
            raise ValueError("branch_width must be at least two")
        if (
            type(budget_seconds) not in (int, float)
            or not math.isfinite(budget_seconds)
            or budget_seconds <= 0
        ):
            raise ValueError("budget_seconds must be positive and finite")
        self.backend = backend
        self.compatible = JevCompatibilityAdapter(backend, model_identity=model_identity)
        self.branch_width = branch_width
        self.max_targets = max_targets
        self.max_stage_calls = max_stage_calls
        self.max_payload_bytes = max_payload_bytes
        self.budget_seconds = budget_seconds
        self.batch_stages = batch_stages

    def _deadline(self, started):
        if time.monotonic() - started >= self.budget_seconds:
            raise BackendTimeoutError(
                "Jev hierarchy cooperative time budget exceeded; blocking calls cannot be hard-cancelled"
            )

    def _plan(self, payload, audit):
        if not isinstance(payload, dict):
            raise RequestValidationError("Jev request must be an object")
        _check_json(payload)
        try:
            # Named question/candidate insertion order affects both native
            # prompts and partitioning, so the source identity must retain it.
            encoded = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (ValueError, TypeError, OverflowError) as exc:
            raise RequestValidationError("Jev hierarchy requires finite JSON data") from exc
        if len(encoded) > self.max_payload_bytes:
            raise RequestValidationError("Jev hierarchy request exceeds its byte limit")
        audit["original_request_sha256"] = hashlib.sha256(encoded).hexdigest()
        audit["payload_bytes"] = len(encoded)
        data = copy.deepcopy(payload)
        questions = data.get("questions")
        if isinstance(questions, dict):
            entries = list(questions.items())
        elif isinstance(questions, list):
            entries = [(q.get("id") if isinstance(q, dict) else None, q) for q in questions]
        else:
            raise RequestValidationError("Jev questions must be a named object or a list")
        if not 1 <= len(entries) <= 64:
            raise RequestValidationError("Jev hierarchy needs 1..64 questions")
        representative, wide = [], {}
        for question_id, raw in entries:
            if not isinstance(raw, dict):
                raise RequestValidationError("each Jev question must be an object")
            item = dict(raw)
            if "id" in item and item["id"] != question_id:
                raise RequestValidationError("question id disagrees with its dictionary key")
            item["id"] = question_id
            key, original, normalized = _criteria(raw, question_id)
            if (
                normalized is not None
                and len(normalized) > self.branch_width
                and question_id in SINGLETON_TARGETS
                and raw.get("type", "choice") == "choice"
            ):
                if question_id in wide:
                    raise RequestValidationError("question ids must be unique")
                count = len(normalized)
                if any(len(label) > 128 for label in normalized):
                    raise RequestValidationError("hierarchical target labels exceed 128 characters")
                if any(len(description) > 16_000 for description in normalized.values()):
                    raise RequestValidationError(
                        "hierarchical target description exceeds 16000 characters"
                    )
                if count > min(self.max_targets, self.branch_width**2):
                    raise RequestValidationError(
                        f"target head {question_id!r} has {count} candidates; hierarchy limit is "
                        f"{min(self.max_targets, self.branch_width**2)}; no targets were truncated"
                    )
                wide[question_id] = {"raw": raw, "original": original, "criteria": normalized}
                item[key] = dict(list(normalized.items())[: self.branch_width])
            representative.append(item)
        data["questions"] = representative
        parsed = _parse_payload(data, model_identity=self.compatible.model_identity)
        if not wide:
            return None
        if parsed.media:
            raise RequestValidationError(
                "Jev target hierarchy currently supports text-only observations"
            )
        # Only three named target heads can be expanded. Duplicate IDs have also
        # passed the ordinary parser's full-question uniqueness validation.
        if len(wide) > 3:
            raise RequestValidationError("at most three browser target heads can use hierarchy")
        audit["deterministic_heads"] = [q.id for q in parsed.questions if len(q.labels()) == 1]
        stages = []
        ordinary = [q for q in parsed.questions if q.id not in wide]
        if ordinary:
            # Keep singleton resolution in the existing compatibility adapter;
            # this stage's actual native request excludes forced singleton heads.
            model_questions = [
                Question.model_validate(q.model_dump()) for q in ordinary if len(q.labels()) > 1
            ]
            request = (
                DecisionRequest(state=copy.deepcopy(parsed.state), questions=model_questions)
                if model_questions
                else None
            )
            forced = {q.id: _singleton_answer(q) for q in ordinary if len(q.labels()) == 1}
            stages.append(self._stage(request, scope="ordinary", forced_answer=forced))
            stages[-1].evidence["deterministic_heads"] = list(forced)
        for question in parsed.questions:
            if question.id not in wide:
                continue
            entry = wide[question.id]
            labels = list(entry["criteria"])
            count = math.ceil(len(labels) / self.branch_width)
            size, remainder = divmod(len(labels), count)
            groups, offset = [], 0
            for index in range(count):
                length = size + (index < remainder)
                members = labels[offset : offset + length]
                groups.append({"id": f"group{index}", "members": members})
                offset += length
            audit["heads"][question.id] = {"original_labels": labels, "groups": groups}
            original_values = (
                entry["original"]
                if isinstance(entry["original"], dict)
                else dict(zip(labels, entry["original"]))
            )
            group_criteria = {
                group["id"]: _json(
                    {
                        "members": [
                            {"target_id": label, "criteria": original_values[label]}
                            for label in group["members"]
                        ]
                    }
                )
                for group in groups
            }
            root = self._question(question.id, entry["raw"], "group", group_criteria)
            stages.append(
                self._stage(
                    DecisionRequest(state=copy.deepcopy(parsed.state), questions=[root]),
                    scope="group",
                    head=question.id,
                )
            )
            for group in groups:
                criteria = {label: entry["criteria"][label] for label in group["members"]}
                if len(criteria) == 1:
                    label = next(iter(criteria))
                    answer = {
                        "id": question.id,
                        "type": "choice",
                        "choice": label,
                        "probabilities": {label: 1.0},
                        "confidence": 1.0,
                        "margin": 1.0,
                    }
                    stage = self._stage(
                        None,
                        scope="conditional",
                        head=question.id,
                        group=group["id"],
                        forced_answer={question.id: answer},
                    )
                    stage.evidence["request"] = {
                        "state": copy.deepcopy(parsed.state),
                        "questions": [{"id": question.id, "type": "choice", "criteria": criteria}],
                        "mathematically_forced_singleton": True,
                    }
                else:
                    leaf = self._question(
                        question.id, entry["raw"], "conditional", criteria, group=group
                    )
                    stage = self._stage(
                        DecisionRequest(state=copy.deepcopy(parsed.state), questions=[leaf]),
                        scope="conditional",
                        head=question.id,
                        group=group["id"],
                    )
                stages.append(stage)
        audit["stages"] = [stage.evidence for stage in stages]
        calls = sum(stage.request is not None for stage in stages)
        audit["planned_backend_calls"] = calls
        audit["planned_stage_predictions"] = calls
        audit["planned_backend_calls_scope"] = "native_stage_requests_not_batch_method_calls"
        if calls > self.max_stage_calls:
            raise RequestValidationError("Jev hierarchy exceeds its backend-call work limit")
        return parsed, stages

    def _question(self, question_id, raw, scope, criteria, *, group=None):
        hierarchy = {"stage": scope, "head": question_id}
        if group is not None:
            hierarchy.update(group=group["id"], members=group["members"])
        rules = (
            "Choose the group containing the best original observed target for this operation. "
            "Groups are not browser actions. Consider every listed member."
            if scope == "group"
            else "Choose the best original observed target among the members of this group."
        )
        original_instructions = raw.get("instructions")
        supplied = original_instructions
        if isinstance(supplied, str):
            try:
                supplied = json.loads(supplied)
            except ValueError:
                supplied = None
        instruction_body = {
            "original_instructions": original_instructions,
            "hierarchy": hierarchy,
            "rules": rules,
        }
        if isinstance(supplied, dict) and isinstance(supplied.get("goal"), str):
            instruction_body["goal"] = supplied["goal"]
        instructions = _json(instruction_body)
        try:
            return Question(id=question_id, instructions=instructions, criteria=criteria)
        except ValidationError as exc:
            raise RequestValidationError(f"invalid expanded Jev question: {exc}") from exc

    @staticmethod
    def _stage(request, *, scope, head=None, group=None, forced_answer=None):
        evidence = {
            "scope": scope,
            "head": head,
            "group": group,
            "status": "planned",
            "request": request.model_dump(exclude_none=True) if request is not None else None,
        }
        return _Stage(request, evidence, forced_answer)

    def _preflight(self, request):
        backend = self.backend
        spec = getattr(backend, "spec", None)
        maximum = getattr(getattr(spec, "capabilities", None), "max_options", None)
        if maximum is not None and any(len(q.labels()) > maximum for q in request.questions):
            raise RequestValidationError(
                "a hierarchy stage exceeds this backend's declared option capacity"
            )
        if callable(getattr(backend, "preflight", None)):
            metadata = backend.preflight(request.model_copy(deep=True))
            return {"scope": "backend_hook", "metadata": _evidence(metadata)}
        mode = getattr(backend, "prompt_mode", None)
        if mode in {"user", "reasoned", "grounded", "goal-first"} and callable(
            getattr(backend, "_prepare_question", None)
        ):
            counts = [
                backend._prepare_question(request.model_copy(deep=True), q)[1]
                for q in request.questions
            ]
            return {
                "scope": "base_context_only" if mode == "reasoned" else "full_native_context",
                "input_tokens": counts,
            }
        model = getattr(backend, "model", None)
        if callable(getattr(model, "prepare", None)):
            inputs, _, _ = model.prepare(request.model_copy(deep=True))
            return {
                "scope": "full_native_context",
                "input_tokens": int(inputs["input_ids"].shape[1]),
            }
        if getattr(spec, "adapter", None) == "decider":
            from .evaluation.providers import typesafe_payload

            _, _, rows = backend.engine._system_one_items(
                **typesafe_payload(request), independent=True, max_state_tokens=2**31 - 1
            )
            counts = [len(row["ids"]) for row in rows]
            if any(count > spec.context_limit for count in counts):
                raise RequestValidationError(
                    "Decider context limit exceeded; no hierarchy stage executed"
                )
            return {"scope": "full_native_context", "input_tokens": counts}
        return {"scope": "unavailable; runtime failures are preserved"}

    @staticmethod
    def _batch_metadata(stages):
        """Fold shared provider telemetry once; request-owned usage stays per stage."""
        batches = {}
        allocated = []
        whole_adapter_batch = None
        for index, stage in stages:
            raw = stage.evidence.get("raw_response")
            execution = raw.get("execution") if isinstance(raw, dict) else None
            if not isinstance(execution, dict):
                allocated.append(None)
                continue
            count = execution.get("forward_calls")
            allocated.append(
                count
                if execution.get("forward_calls_scope") == "allocated_once_per_shared_browser_batch"
                and type(count) is int
                and count >= 0
                else None
            )
            # The native Gemma adapter repeats the same call-wide execution
            # object on every response. Its explicit scope permits one count,
            # never a sum across rows. Unscoped counts remain unknown.
            if execution.get("scope") == "whole_adapter_batch":
                if whole_adapter_batch is None:
                    whole_adapter_batch = {
                        "stage_indices": [],
                        "execution": execution,
                        "forward_calls": count if type(count) is int and count >= 0 else None,
                    }
                elif whole_adapter_batch["execution"] != execution:
                    variants = whole_adapter_batch.setdefault(
                        "inconsistent_shared_metadata", [whole_adapter_batch["execution"]]
                    )
                    if execution not in variants:
                        variants.append(execution)
                    whole_adapter_batch["forward_calls"] = None
                whole_adapter_batch["stage_indices"].append(index)
            batch_id = execution.get("batch_id")
            if not isinstance(batch_id, str) or not batch_id:
                continue
            batch = batches.setdefault(
                batch_id,
                {
                    "batch_id": batch_id,
                    "stage_indices": [],
                    "forward_refs": [],
                    "batch_chunk_ids": [],
                },
            )
            batch["stage_indices"].append(index)
            for key in (
                "batch_forward_calls",
                "batch_elapsed_ms",
                "latency_scope",
                "forward_calls_scope",
                "max_forward_batch_size",
            ):
                if key in execution:
                    value = execution[key]
                    if key not in batch:
                        batch[key] = value
                    elif batch[key] != value:
                        conflicts = batch.setdefault("inconsistent_shared_metadata", {})
                        variants = conflicts.setdefault(key, [batch[key]])
                        if value not in variants:
                            variants.append(value)
            refs = execution.get("forward_refs")
            if isinstance(refs, list):
                for ref in refs:
                    if ref not in batch["forward_refs"]:
                        batch["forward_refs"].append(ref)
            chunks = execution.get("batch_chunk_ids")
            if isinstance(chunks, list):
                for chunk in chunks:
                    if chunk not in batch["batch_chunk_ids"]:
                        batch["batch_chunk_ids"].append(chunk)
        return {
            "batches": list(batches.values()),
            "allocated_forward_calls": (
                sum(allocated) if all(value is not None for value in allocated) else None
            ),
            "forward_calls_scope": "sum_of_request_allocations_not_shared_batch_totals",
            "whole_adapter_batch": whole_adapter_batch,
        }

    @staticmethod
    def _retain_batch_error(stages, audit, exc):
        """Keep the backend's genuine completed responses after a failed batch."""
        if not hasattr(exc, "batch_evidence"):
            return
        evidence = _evidence(exc.batch_evidence)
        audit["batch_error_evidence"] = evidence
        if not isinstance(evidence, dict):
            return
        indices = evidence.get("completed_response_indices")
        responses = evidence.get("completed_responses")
        if not isinstance(indices, list) or not isinstance(responses, list):
            return
        for response_index, raw in zip(indices, responses):
            if type(response_index) is not int or not 0 <= response_index < len(stages):
                continue
            _, stage = stages[response_index]
            stage.evidence.update(
                raw_response=raw,
                response=raw,
                status="returned_unvalidated",
            )

    def _predict_batch(self, stages, audit, started):
        native = [(index, stage) for index, stage in enumerate(stages) if stage.request is not None]
        execution = {
            "status": "running",
            "backend_call_status": "not_started",
            "stage_indices": [index for index, _ in native],
            "expected_response_count": len(native),
            "latency_scope": "whole_hierarchy_backend_batch",
        }
        audit["batch_execution"] = execution
        for response_index, (_, stage) in enumerate(native):
            stage.evidence.update(
                elapsed_ms=None,
                latency_scope="shared_backend_batch",
                batch_stage_index=response_index,
            )
        self._deadline(started)
        batch_started = time.monotonic()
        try:
            audit["backend_predict_batch_calls"] += 1
            audit["stage_batching"]["used"] = True
            execution["backend_call_status"] = "running"
            raw_results = self.backend.predict_batch(
                [stage.request.model_copy(deep=True) for _, stage in native]
            )
            execution["backend_call_status"] = "returned"
            # Record the entire return before inspecting count, identity or
            # probabilities; later invalid rows must not erase genuine outputs.
            execution["raw_responses"] = _evidence(raw_results)
            if not isinstance(raw_results, list):
                raise BackendResponseError("hierarchy predict_batch must return an ordered list")
            execution["response_count"] = len(raw_results)
            for (_, stage), raw in zip(native, raw_results):
                stage.evidence.update(
                    raw_response=_evidence(raw),
                    response=_evidence(raw),
                    status="returned_unvalidated",
                )
            execution["backend_execution"] = self._batch_metadata(native)
            if len(raw_results) != len(native):
                raise BackendResponseError(
                    "hierarchy predict_batch response count differs from requests"
                )
            # A blocking batch is cooperative, like the sequential calls. Keep
            # all returned evidence even if the deadline expired during it.
            self._deadline(started)
            execution["status"] = "returned_unvalidated"
            return {index: raw for (index, _), raw in zip(native, raw_results, strict=True)}
        except Exception as exc:
            if execution["backend_call_status"] == "running":
                execution["backend_call_status"] = "raised"
            self._retain_batch_error(native, audit, exc)
            execution["backend_execution"] = self._batch_metadata(native)
            execution.update(
                status="failed",
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise
        finally:
            execution["elapsed_ms"] = (time.monotonic() - batch_started) * 1000

    def predict(self, payload):
        started = time.monotonic()
        audit = {
            "version": "jev_hierarchy_v1",
            "probabilities_semantics": "hierarchical_policy_distribution_not_flat_calibration",
            "partition": "balanced_observation_order",
            "branch_width": self.branch_width,
            "status": "planning",
            "heads": {},
            "stages": [],
            "backend_predict_calls": 0,
            "backend_predict_batch_calls": 0,
            "timeout_policy": "cooperative_before_and_after_blocking_calls_no_hard_cancel",
            "limits": {
                "targets_per_head": min(self.max_targets, self.branch_width**2),
                "backend_calls": self.max_stage_calls,
                "backend_calls_scope": "native_stage_requests_not_batch_method_calls",
                "payload_bytes_after_json_decode": self.max_payload_bytes,
                "expanded_label_characters": 128,
                "expanded_description_characters": 16_000,
                "budget_seconds": self.budget_seconds,
            },
        }
        try:
            plan = self._plan(payload, audit)
            if plan is None:
                return self.compatible.predict(payload)
            parsed, stages = plan
            method_available = callable(getattr(self.backend, "predict_batch", None))
            supported = getattr(self.backend, "supports_stage_batching", True) is not False
            batch_available = method_available and supported
            use_batch = self.batch_stages and batch_available
            audit["stage_batching"] = {
                "requested": self.batch_stages,
                "method_available": method_available,
                "supported": supported,
                "available": batch_available,
                "used": False,
                "fallback_reason": (
                    "backend_method_unavailable"
                    if self.batch_stages and not method_available
                    else "backend_declares_stage_batching_unsupported"
                    if self.batch_stages and not supported
                    else None
                ),
            }
            audit["status"] = "preflight"
            for stage in stages:
                self._deadline(started)
                if stage.request is not None:
                    preflight_started = time.monotonic()
                    try:
                        stage.evidence["preflight"] = self._preflight(stage.request)
                    except Exception as exc:
                        stage.evidence.update(
                            status="failed",
                            error=f"{type(exc).__name__}: {exc}",
                            error_stage="preflight",
                        )
                        raise
                    finally:
                        stage.evidence["preflight_elapsed_ms"] = (
                            time.monotonic() - preflight_started
                        ) * 1000
            batch_results = self._predict_batch(stages, audit, started) if use_batch else None
            answers, identities, reported_usage = {}, set(), []
            for stage_index, stage in enumerate(stages):
                self._deadline(started)
                stage_started = time.monotonic()
                try:
                    if stage.request is None:
                        result = {
                            "answers": stage.forced_answer,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        }
                        stage.evidence["status"] = "deterministic"
                    else:
                        if batch_results is None:
                            audit["backend_predict_calls"] += 1
                            raw_result = self.backend.predict(stage.request.model_copy(deep=True))
                        else:
                            raw_result = batch_results[stage_index]
                        stage.evidence["raw_response"] = _evidence(raw_result)
                        stage.evidence["response"] = _evidence(raw_result)
                        result = normalize_response(stage.request, raw_result)
                        if stage.forced_answer:
                            result["answers"].update(stage.forced_answer)
                        model = result.get("model", self.backend.name)
                        if not isinstance(model, str) or not model.strip():
                            raise BackendResponseError(
                                "backend model identity must be nonempty text"
                            )
                        identities.add(model)
                        stage.evidence["model"] = model
                        stage.evidence["model_identity_source"] = (
                            "response" if "model" in result else "configured_backend_name"
                        )
                        if len(identities) > 1:
                            raise BackendResponseError(
                                "hierarchy backend reported different models across stages"
                            )
                        stage.evidence["status"] = "completed"
                    stage.evidence["response"] = _evidence(result)
                    stage.evidence["usage"] = _evidence(result.get("usage"))
                    reported_usage.append(result.get("usage"))
                    self._deadline(started)
                except Exception as exc:
                    if hasattr(exc, "generation"):
                        stage.evidence["error_generation"] = _evidence(exc.generation)
                    stage.evidence.update(
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                        error_stage="prediction",
                    )
                    raise
                finally:
                    if batch_results is None or stage.request is None:
                        stage.evidence["elapsed_ms"] = (time.monotonic() - stage_started) * 1000
                if stage.evidence["scope"] == "ordinary":
                    answers.update(result["answers"])
                elif stage.evidence["scope"] == "group":
                    audit["heads"][stage.evidence["head"]]["group_probabilities"] = result[
                        "answers"
                    ][stage.evidence["head"]]["probabilities"]
            for head, metadata in audit["heads"].items():
                joint = {}
                for stage in stages:
                    if stage.evidence["scope"] == "conditional" and stage.evidence["head"] == head:
                        mass = metadata["group_probabilities"][stage.evidence["group"]]
                        for label, probability in stage.evidence["response"]["answers"][head][
                            "probabilities"
                        ].items():
                            joint[label] = mass * probability
                joint = {label: joint[label] for label in metadata["original_labels"]}
                try:
                    validate_distribution(joint.values(), len(joint))
                except ValueError as exc:
                    raise BackendResponseError("invalid joint hierarchy distribution") from exc
                metadata["joint_probabilities"] = joint
                best = max(joint, key=joint.__getitem__)
                values = sorted(joint.values(), reverse=True)
                answers[head] = {
                    "id": head,
                    "type": "choice",
                    "choice": best,
                    "probabilities": joint,
                    "confidence": joint[best],
                    "margin": values[0] - values[1],
                }
            if use_batch:
                self._deadline(started)
                audit["batch_execution"]["status"] = "completed"
            audit.update(status="completed", elapsed_ms=(time.monotonic() - started) * 1000)
            usage = {}
            for key in ("input_tokens", "output_tokens"):
                values = [
                    item.get(key) if isinstance(item, dict) else None for item in reported_usage
                ]
                usage[key] = (
                    sum(values)
                    if all(type(value) is int and value >= 0 for value in values)
                    else None
                )
            return {
                "model": next(iter(identities), self.backend.name),
                "answers": {q.id: answers[q.id] for q in parsed.questions},
                "usage": usage,
                "hierarchy": audit,
                "jev_compatibility": {
                    "model_identity": self.compatible.model_identity,
                    "singleton_heads": audit["deterministic_heads"],
                },
            }
        except Exception as exc:
            previous_hierarchy = getattr(exc, "hierarchy", None)
            if previous_hierarchy is not None and previous_hierarchy is not audit:
                audit["backend_error_hierarchy"] = _evidence(previous_hierarchy)
            if "batch_execution" in audit:
                audit["batch_execution"]["status"] = "failed"
                if not hasattr(exc, "batch_evidence"):
                    execution = audit["batch_execution"]
                    indices = [
                        stage["batch_stage_index"]
                        for stage in audit["stages"]
                        if "batch_stage_index" in stage and "raw_response" in stage
                    ]
                    responses = [
                        stage["raw_response"]
                        for stage in audit["stages"]
                        if "batch_stage_index" in stage and "raw_response" in stage
                    ]
                    exc.batch_evidence = {
                        "completed_response_indices": indices,
                        "completed_responses": responses,
                        "raw_responses": execution.get("raw_responses"),
                        "validation_scope": "returned_native_rows_not_all_validated",
                        "backend_execution": execution.get("backend_execution"),
                    }
                audit["batch_error_evidence"] = _evidence(exc.batch_evidence)
            audit.update(
                status="failed",
                elapsed_ms=(time.monotonic() - started) * 1000,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            exc.hierarchy = audit
            if not hasattr(exc, "generation"):
                exc.generation = audit
            raise
