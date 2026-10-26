"""Held candidate DTO/provider integration; no publisher identity substitution.

Injected original parsing, DTO validation, payload conversion, normalization and
named-wire functions remain authoritative. Only the explicitly named accepted
operation-only delta backend produces probabilities. No model/framework is
imported here. Dev informed model selection; this wiring supplies no new unseen
evaluation or official score claim.
"""

from __future__ import annotations

import copy
import hashlib
import time

SOURCE_ONLY = False
PROFILE = "jev-dsl01-operation-candidate-cpu5-browser-v1"
ALIAS = "d3-edge-dsl01-operation-delta-cpu5-candidate"
LEGACY_ABI = "jev_decision3_native_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def released():
    require(SOURCE_ONLY is False, "SOURCE_ONLY: delta candidate adapter/provider is held")


class CandidateProvider:
    """Raw DTO->original ordered payload->adapted raw; no normalization here."""

    name = ALIAS

    def __init__(
        self,
        *,
        spec,
        native,
        observer,
        delta_backend,
        runtime,
        torch,
        model,
        delta,
        capability,
        capability_sha256,
        native_payload,
        verify_owned,
    ):
        released()
        require(callable(verify_owned), "current original owned selection required")
        verify_owned()
        require(
            type(capability_sha256) is str
            and len(capability_sha256) == 64
            and all(character in "0123456789abcdef" for character in capability_sha256),
            "root-authenticated operation capability SHA remains pending",
        )
        require(
            delta_backend.SOURCE_ONLY is False
            and delta_backend.ALIAS == ALIAS
            and delta_backend.PROFILE == capability["profile"]
            and delta_backend.CAPABILITY_SCHEMA == "root-accepted-dsl01-operation-train-delta-v1"
            and delta_backend.CAPABILITY_SHA256 == capability_sha256,
            "separately released/qualified candidate backend required",
        )
        self.spec, self.native, self.observer = spec, native, observer
        self.delta_backend, self.runtime, self.torch = delta_backend, runtime, torch
        self.model, self.delta, self.capability = model, delta, copy.deepcopy(capability)
        self.capability_sha256 = capability_sha256
        self.native_payload, self.verify_owned = native_payload, verify_owned

    def predict(self, request, *, deadline, evidence_sink):
        released()
        trace = {
            "profile": PROFILE,
            "alias": ALIAS,
            "only_adapted_head_id": "operation",
            "publisher_native_numerics": False,
            "events": [],
            "raw_response": None,
            "helper_calls": 0,
        }
        backend = None

        def emit(event, data):
            trace["events"].append({"event": event, "evidence": copy.deepcopy(data)})
            evidence_sink(event, data)
            deadline()

        try:
            deadline()
            self.verify_owned()
            deadline()
            payload = self.native_payload(request)
            trace["ordered_native_payload"] = copy.deepcopy(payload)
            trace["question_order"] = list(payload["questions"])
            backend = self.delta_backend.DeltaHeadBackend(
                self.native,
                self.observer,
                self.runtime,
                self.torch,
                self.model,
                self.delta,
                root_capability=self.capability,
                root_capability_sha256=self.capability_sha256,
                deadline=deadline,
                evidence_sink=emit,
            )
            raw = backend(**payload)
            trace["raw_response"] = copy.deepcopy(raw)
            trace["delta_head_evidence"] = copy.deepcopy(backend.last_evidence)
            emit("candidate_raw_returned_before_alias_validation", raw)
            require(
                isinstance(raw, dict) and raw.get("model") == ALIAS,
                "candidate raw alias changed; publisher identity refused",
            )
            deadline()
            return raw
        except BaseException as error:
            if backend is not None:
                try:
                    trace["delta_head_evidence"] = copy.deepcopy(backend.last_evidence)
                except BaseException as secondary:
                    error.add_note(
                        "Delta failure evidence retention also failed: " + repr(secondary)
                    )
            error.decision3_provider_evidence = trace
            raise


def adapter_type(dependencies, *, verify_owned, clock=time.monotonic):
    """Explicit candidate predict; never bypass original publisher's name guard.

    The base spec remains the real checkpoint descriptor. CPU resource and delta
    candidate identities are separate and are not asserted as CUDA qualification.
    """
    released()
    require(callable(verify_owned), "current owned verifier required")

    class CandidateAdapter:
        def __init__(self, backend, *, model_identity=None, evidence_sink=None):
            released()
            verify_owned()
            spec = backend.spec
            require(
                spec.adapter == "decision3"
                and spec.device == "cpu"
                and spec.precision == "bfloat16"
                and spec.context_limit == 16384,
                "honest original CPU BF16/context16384 base spec required",
            )
            blockers = [
                reason
                for reason in dependencies["configuration_blockers"](spec)
                if reason != "decision3_requires_cuda_bfloat16"
            ]
            require(not blockers, "non-device base blockers retained: " + ", ".join(blockers))
            require(
                backend.name == ALIAS and model_identity in (None, ALIAS),
                "exact adapted candidate identity required",
            )
            self.backend, self.identity, self.evidence_sink = backend, ALIAS, evidence_sink

        def predict(self, payload):
            started = clock()
            audit = {
                "version": LEGACY_ABI,
                "candidate_profile": PROFILE,
                "scope": "legacy v1 DTO/named-wire ABI only; adapted CPU candidate, not publisher-native numerics",
                "publisher_native_numerics": False,
                "candidate_alias": ALIAS,
                "original_request": copy.deepcopy(payload),
                "status": "parsing",
                "backend_predict_calls": 0,
                "native_question_rows": 0,
                "native_events": [],
                "raw_response": None,
            }

            def emit(event, data):
                if self.evidence_sink is not None:
                    try:
                        self.evidence_sink(
                            {"event": event, "profile": PROFILE, "data": copy.deepcopy(data)}
                        )
                    except BaseException as error:
                        audit.setdefault("evidence_sink_errors", []).append(
                            {"type": type(error).__name__, "message": str(error)}
                        )
                        raise  # Required retention failure cannot authorize a result.

            def deadline():
                if clock() - started >= 20:
                    raise dependencies["timeout_type"](
                        "candidate cooperative20s exceeded; no hard cancellation"
                    )

            def event(name, data):
                audit["native_events"].append({"event": name, "evidence": copy.deepcopy(data)})
                emit(name, data)

            try:
                deadline()
                parsed, original, source, labels = dependencies["parse_native_payload"](
                    payload, model_identity=ALIAS
                )
                audit.update(
                    original_request=original,
                    original_request_sha256=hashlib.sha256(source).hexdigest(),
                    candidate_labels=labels,
                    native_question_rows=len(parsed.questions),
                    status="preflight",
                )
                emit("planned", audit)
                deadline()
                # Preserve normal DTO validation; no unchecked model_copy or
                # representative/pruned head reaches the adapted backend.
                request = dependencies["request_type"].model_validate(
                    copy.deepcopy(parsed.model_dump())
                )
                audit["backend_predict_calls"] += 1
                raw = self.backend.predict(request, deadline=deadline, evidence_sink=event)
                audit["raw_response"] = copy.deepcopy(raw)
                emit("backend_raw_returned", raw)
                deadline()
                if not isinstance(raw, dict) or raw.get("model") != ALIAS:
                    raise dependencies["response_error_type"](
                        "raw candidate model must equal the distinct delta alias"
                    )
                result = dependencies["normalize_native_response"](request, raw)
                answers = [copy.deepcopy(result["answers"][q.id]) for q in request.questions]
                audit.update(status="completed", usage=copy.deepcopy(result.get("usage")))
                deadline()
                emit("completed", audit)
                deadline()
                return {**result, "answers": answers, "model": ALIAS, "decision3_native": audit}
            except BaseException as error:
                audit.update(
                    status="failed", error={"type": type(error).__name__, "message": str(error)}
                )
                if hasattr(error, "decision3_provider_evidence"):
                    try:
                        audit["provider_failure_evidence"] = copy.deepcopy(
                            error.decision3_provider_evidence
                        )
                    except BaseException as secondary:
                        error.add_note(
                            "Provider failure evidence copy also failed: " + repr(secondary)
                        )
                error.decision3_native = audit
                try:
                    emit("failed", audit)
                except BaseException as secondary:
                    error.add_note("Failed-event retention also failed: " + repr(secondary))
                raise
            finally:
                audit["elapsed_ms"] = (clock() - started) * 1000

    return CandidateAdapter


def create_application(
    *,
    bridge,
    owned_release,
    provider,
    dependencies,
    output,
    cpu_identity,
    backend_wrapper,
    clock=time.monotonic,
):
    """Inject into unchanged serialized Gateway, including its final ASGI20 gate."""
    released()
    owned_release.verify()
    require(
        provider.name == ALIAS and cpu_identity["candidate_alias"] == ALIAS,
        "candidate HTTP identity differs",
    )
    wrapped = backend_wrapper(provider)
    kind = adapter_type(dependencies, verify_owned=owned_release.verify, clock=clock)
    return bridge._Gateway(
        output=output,
        backend=wrapped,
        adapter_type=kind,
        named_browser_response=dependencies["named_browser_response"],
        cpu_identity=cpu_identity,
        # Nested isinstance tuple preserves genuine TimeoutError and the
        # existing project timeout object without replacing the primary error.
        timeout_type=(dependencies["timeout_type"], TimeoutError),
        validation_type=dependencies["validation_type"],
        owned_release=owned_release,
        clock=clock,
    )
