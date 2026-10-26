"""Local and Modal use the same application and validation boundary."""

from __future__ import annotations

import secrets
import threading
import time

from .contracts import DecisionRequest
from .errors import (
    BackendResponseError,
    BackendTimeoutError,
    BackendTransportError,
    RequestValidationError,
)


def create_app(
    backend,
    *,
    api_key=None,
    max_batch=16,
    jev_compatible=False,
    jev_hierarchical=False,
    jev_hierarchical_batch=False,
    jev_browser_operation_first=False,
    jev_decision3_native=False,
    jev_generated_actions=False,
    jev_generated_action_grammar=False,
    jev_generated_prompt_format="raw-json-v1",
    model_identity=None,
    request_evidence_sink=None,
    jev_action_profile="standard",
):
    from fastapi import Depends, FastAPI, Header, HTTPException

    from .jev_action_profiles import ACTION_PROFILES, DEFAULT_ACTION_PROFILE, resolve_protocol
    from .jev_prompt import PROMPT_FORMATS

    if type(jev_action_profile) is not str or jev_action_profile not in ACTION_PROFILES:
        raise ValueError("Unknown explicit browser action profile")
    experimental_actions = jev_action_profile != DEFAULT_ACTION_PROFILE
    if experimental_actions and (
        not jev_compatible
        or type(jev_decision3_native) is not bool
        or type(jev_generated_actions) is not bool
        or jev_decision3_native == jev_generated_actions
        or jev_hierarchical
        or jev_hierarchical_batch
        or jev_browser_operation_first
    ):
        raise ValueError(
            "Experimental action profile requires one explicit compatible native/generated adapter"
        )
    action_protocol = resolve_protocol(jev_action_profile) if experimental_actions else None

    if (
        type(jev_generated_prompt_format) is not str
        or jev_generated_prompt_format not in PROMPT_FORMATS
    ):
        raise ValueError("unsupported generated-action prompt format")
    if jev_generated_prompt_format != "raw-json-v1" and not jev_generated_actions:
        raise ValueError("shared-element prompt requires generated actions")

    if type(jev_decision3_native) is not bool:
        raise ValueError("Decision3 native profile must be an explicit boolean")
    decision3_class = None
    if jev_decision3_native:
        if (
            not jev_compatible
            or jev_hierarchical
            or jev_hierarchical_batch
            or jev_browser_operation_first
            or jev_generated_actions
            or jev_generated_action_grammar
            or not callable(request_evidence_sink)
        ):
            raise ValueError(
                "Decision3 native requires compatibility and server evidence without other decision profiles"
            )
        if action_protocol is None:
            from .jev_decision3_native import JevDecision3NativeAdapter

            decision3_class = JevDecision3NativeAdapter
        else:
            decision3_class = action_protocol.native_adapter_class
        decision3_class(backend, model_identity=model_identity)
    if jev_hierarchical and not jev_compatible:
        raise ValueError("Jev hierarchical choices require the explicit compatible endpoint")
    if jev_hierarchical_batch and not jev_hierarchical:
        raise ValueError("Jev hierarchy batching requires hierarchical choices")
    if type(jev_browser_operation_first) is not bool:
        raise ValueError("Jev browser operation-first must be an explicit boolean")
    operation_first_class = None
    if jev_browser_operation_first:
        if (
            not jev_compatible
            or not jev_hierarchical
            or jev_hierarchical_batch
            or jev_generated_actions
            or jev_generated_action_grammar
        ):
            raise ValueError(
                "browser operation-first requires compatibility and hierarchy without batching/generated actions"
            )
        from .jev_browser_operation_first import JevBrowserOperationFirstAdapter

        operation_first_class = JevBrowserOperationFirstAdapter
        operation_first_class(backend, model_identity=model_identity)
    if type(jev_generated_actions) is not bool:
        raise ValueError("Jev generated actions must be an explicit boolean")
    if type(jev_generated_action_grammar) is not bool:
        raise ValueError("Jev generated action grammar must be an explicit boolean")
    if jev_generated_action_grammar and not jev_generated_actions:
        raise ValueError("Jev generated action grammar requires generated actions")
    generated_adapter = None
    if jev_generated_actions:
        if not jev_compatible or jev_hierarchical or jev_hierarchical_batch:
            raise ValueError(
                "Jev generated actions require compatibility without hierarchy/batching"
            )
        if action_protocol is None:
            from .jev_generated import JevGeneratedActionAdapter

            generated_class = JevGeneratedActionAdapter
        else:
            generated_class = action_protocol.generated_adapter_class

        generated_adapter = generated_class(
            backend,
            model_identity=model_identity,
            constrain_actions=jev_generated_action_grammar,
            prompt_format=jev_generated_prompt_format,
        )
    app = FastAPI(title="Gemma Unified System One", version="0.1.0")
    lock = threading.Lock()
    app.state.request_evidence_errors = []
    app.state.request_evidence_timings = []

    def authorize(authorization: str | None = Header(default=None)):
        if api_key and not secrets.compare_digest(authorization or "", f"Bearer {api_key}"):
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    @app.get("/healthz")
    def health():
        result = {"status": "ok", "model": backend.name}
        if action_protocol is not None:
            import copy

            result.update(
                action_profile=jev_action_profile,
                action_profile_provenance=copy.deepcopy(action_protocol.provenance),
            )
        return result

    def invoke(request, *, compatible=False):
        from .backends import normalize_response

        emit = None
        if compatible and request_evidence_sink is not None:
            import hashlib
            import json

            from .browser_server_evidence import json_evidence

            request_id, sequence, started = secrets.token_hex(16), 0, time.monotonic()

            def emit(event, data):
                nonlocal sequence
                event_sequence = sequence
                sequence += 1
                observer_started = time.perf_counter()
                try:
                    entry = {
                        "request_id": request_id,
                        "sequence": event_sequence,
                        "event": event,
                        "recorded_unix": time.time(),
                        "elapsed_ms": (time.monotonic() - started) * 1000,
                        "data": json_evidence(data),
                    }
                    request_evidence_sink(entry)
                except BaseException as exc:
                    app.state.request_evidence_errors.append(
                        {
                            "request_id": request_id,
                            "event": event,
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                finally:
                    app.state.request_evidence_timings.append(
                        {
                            "request_id": request_id,
                            "event": event,
                            "elapsed_ms": (time.perf_counter() - observer_started) * 1000,
                        }
                    )

            try:
                encoded = json.dumps(
                    request, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                ).encode()
                request_sha = hashlib.sha256(encoded).hexdigest()
            except (ValueError, TypeError, OverflowError, UnicodeError):
                request_sha = None
            emit(
                "request_started",
                {
                    "payload": request,
                    "original_request_sha256": request_sha,
                    "model_identity": model_identity,
                    **(
                        {
                            "action_profile": jev_action_profile,
                            "action_profile_provenance": action_protocol.provenance,
                        }
                        if action_protocol is not None
                        else {}
                    ),
                },
            )

        def error_detail(exc):
            generation = getattr(exc, "generation", None)
            hierarchy = getattr(exc, "hierarchy", None)
            decision3_native = getattr(exc, "decision3_native", None)
            if compatible and (
                generation is not None or hierarchy is not None or decision3_native is not None
            ):
                detail = {"message": str(exc)}
                if generation is not None and generation is not hierarchy:
                    detail["generation"] = generation
                if hierarchy is not None:
                    detail["hierarchy"] = hierarchy
                if decision3_native is not None:
                    detail["decision3_native"] = decision3_native
                return detail
            return str(exc)

        def compute():
            with lock:
                if emit is not None:
                    emit("inference_lock_acquired", {})
                if compatible:
                    if generated_adapter is not None:
                        return generated_adapter.predict(
                            request,
                            evidence_sink=(lambda event: emit("generated_action_checkpoint", event))
                            if emit is not None
                            else None,
                        )
                    from .jev import JevCompatibilityAdapter

                    adapter = JevCompatibilityAdapter
                    options = (
                        {"evidence_sink": lambda event: emit("hierarchy_checkpoint", event)}
                        if emit is not None
                        else {}
                    )
                    if decision3_class is not None:
                        adapter = decision3_class
                        if emit is not None:
                            options["evidence_sink"] = lambda event: emit(
                                "decision3_native_checkpoint", event
                            )
                    elif operation_first_class is not None:
                        adapter = operation_first_class
                        if emit is not None:
                            options["evidence_sink"] = lambda event: emit(
                                "browser_operation_first_checkpoint", event
                            )
                    elif jev_hierarchical:
                        from .jev_hierarchy import JevHierarchicalAdapter

                        adapter = JevHierarchicalAdapter
                        options["batch_stages"] = jev_hierarchical_batch
                    return adapter(backend, model_identity=model_identity, **options).predict(
                        request
                    )
                return normalize_response(request, backend.predict(request))

        try:
            try:
                result = compute()
            except BaseException as exc:
                if emit is not None:
                    details = {"error": {"type": type(exc).__name__, "message": str(exc)}}
                    for key in (
                        "hierarchy",
                        "generation",
                        "batch_evidence",
                        "prefix_evidence",
                        "decision3_native",
                    ):
                        if hasattr(exc, key):
                            details[key] = getattr(exc, key)
                    emit("server_failed", details)
                raise
            if emit is not None:
                emit("server_returned", {"response": result})
            return result
        except RequestValidationError as exc:
            raise HTTPException(status_code=422, detail=error_detail(exc)) from exc
        except BackendTimeoutError as exc:
            raise HTTPException(status_code=504, detail=error_detail(exc)) from exc
        except (BackendResponseError, BackendTransportError) as exc:
            raise HTTPException(status_code=502, detail=error_detail(exc)) from exc
        except Exception as exc:
            if compatible and (
                getattr(exc, "generation", None) is not None
                or getattr(exc, "hierarchy", None) is not None
                or getattr(exc, "decision3_native", None) is not None
            ):
                raise HTTPException(status_code=502, detail=error_detail(exc)) from exc
            raise

    @app.post("/decide/batch", dependencies=[Depends(authorize)])
    @app.post("/v1/systemone/batch", dependencies=[Depends(authorize)])
    def decide_batch(requests: list[DecisionRequest]):
        from .backends import normalize_response

        if not requests or len(requests) > max_batch:
            raise HTTPException(
                status_code=422, detail=f"batch must contain 1..{max_batch} requests"
            )
        try:
            with lock:
                if hasattr(backend, "predict_batch"):
                    raw = backend.predict_batch(requests)
                else:
                    raw = [backend.predict(r) for r in requests]
                if not isinstance(raw, list) or len(raw) != len(requests):
                    raise BackendResponseError(
                        "batch backend must return one response per request as a list"
                    )
                return [normalize_response(req, resp) for req, resp in zip(requests, raw)]
        except RequestValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except BackendTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except (BackendResponseError, BackendTransportError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/decide", dependencies=[Depends(authorize)])
    def decide(request: DecisionRequest):
        return invoke(request)

    if jev_compatible:

        @app.post("/v1/systemone", dependencies=[Depends(authorize)])
        def jev_decide(request: dict):
            return invoke(request, compatible=True)

    else:
        app.post("/v1/systemone", dependencies=[Depends(authorize)])(decide)

    return app
