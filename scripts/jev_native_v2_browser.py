#!/usr/bin/env python3
"""Explicit current-NF4 raw/V2 evaluation on the unchanged full V7 browser suite.

Dry by default. This is not the historical Q95/BF16 qualification or a Decision3
warm campaign. Root must separately confirm the owned process/cgroup has exited.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import contextvars
import hashlib
import importlib.util
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "jev-current-nf4-v7-browser-v1"
IDENTITY = "generated-actions-nf4"
FORMATS = ("raw-json-v1", "shared-elements-v2")
V7 = "composed-visible-text-settled-v7"
V7_SHA = "d9f4236d1d46c9472c3d6c1551342639ecb870e6c0dd419246927645410ea868"
BASE_BYTES = 23_919_549_408
GIB = 1024**3
MAX_BODY = 1_000_000
CACHE = ROOT / "artifacts/jev-runtime/field-json-grammar-20261018-01/full-compile-01/cache"
CACHE_SHA = "6e5ac130fa1f5b6bb315f71d17cb8d3d1b8f2552eef319a1e695c06ebf5b0dc6"
REVISION = "a66f836b56605039fe040f330180e336d19b3362"
BASE_ID = "knowlet/Gemma-4-12B-Unified-System-One"
MODEL_CACHE = (
    ROOT
    / "artifacts/jev-local-hf/hub/models--knowlet--Gemma-4-12B-Unified-System-One/snapshots"
    / REVISION
)
CHROMIUM = "/home/albert_hsieh/.cache/ms-playwright/chromium_headless_shell-1208/chrome-linux/headless_shell"
KNOWN4 = (
    "public-arxiv-title-filter-results-v1",
    "public-arxiv-original-paper-open-v1",
    "public-python-counter-most-common-v1",
    "public-nasa-mars-facts-moons-v1",
)
PUBLIC6 = tuple(f"heldout-public-{i:03d}" for i in range(6))
TRACE = contextvars.ContextVar("native_v2_browser_http_trace", default=None)
MATRIX = (
    ROOT / "artifacts/jev-runtime/public-affordance-matrix-harness-20261019-01/matrix_common.py"
)


def require(value, message):
    if not value:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def pin(path):
    blob = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}


def write(path, value, *, raw=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = value if raw else encode(value) + b"\n"
    require(type(blob) is bytes and len(blob) <= 16 * 1024**2, "Bounded evidence file")
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())


def ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary path without symlink or traversal required",
    )
    return path


class ResourceJournal:
    """Existing full load formula, separately named browser journal; no residency credit."""

    def __init__(self, path, boot_id, *, observe=None):
        self.path, self.boot_id, self.observe = Path(path), boot_id, observe or self.snapshot
        self.lock, self.sequence, self.fatal = threading.RLock(), 0, None
        self.memtotal = None
        write(self.path, b"", raw=True)

    @staticmethod
    def snapshot():
        values = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                number, unit = value.split()
                require(unit == "kB", "Memory observation units")
                values[key] = int(number) * 1024
        return {**values, "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}

    def check(self, phase, scope="load"):
        with self.lock:
            require(self.fatal is None, "Earlier resource/evidence refusal remains latched")
            require(scope in {"cpu", "load"} and self.sequence < 32768, "Browser resource bound")
            row = {"sequence": self.sequence, "phase": phase, "scope": scope, "admitted": False}
            primary = None
            try:
                obs = self.observe()
                require(
                    obs["boot_id"] == self.boot_id
                    and type(obs["MemTotal"]) is int
                    and type(obs["MemAvailable"]) is int
                    and 0 < obs["MemAvailable"] <= obs["MemTotal"],
                    "Fresh same-boot positive memory observation required",
                )
                if self.memtotal is None:
                    self.memtotal = obs["MemTotal"]
                require(obs["MemTotal"] == self.memtotal, "Host physical memory changed")
                reserve = max(24 * GIB, (obs["MemTotal"] + 4) // 5)
                required = reserve + (4 * GIB if scope == "cpu" else 2 * BASE_BYTES)
                row.update(
                    observation=obs,
                    reserve_bytes=reserve,
                    required_bytes=required,
                    swap_or_resident_credit_bytes=0,
                    monotonic=time.monotonic(),
                )
                require(obs["MemAvailable"] >= required, "Full conservative browser floor refused")
                row["admitted"] = True
            except BaseException as exc:
                primary = exc
                row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            try:
                blob = encode(row) + b"\n"
                with self.path.open("ab") as stream:
                    stream.write(blob)
                    stream.flush()
                    os.fsync(stream.fileno())
                self.sequence += 1
            except BaseException as exc:
                self.fatal = exc
                if primary is not None:
                    primary.add_note("Resource evidence failed: " + repr(exc))
                    raise primary from exc
                raise
            if primary is not None:
                self.fatal = primary
                raise primary
            return row


def action_settings(
    action_profile="standard",
    text_endpoint=None,
    text_model=None,
    text_api_key_env="S1_TEXT_API_KEY",
):
    """Validate an explicit local shared helper before any model construction."""
    import re
    from urllib.parse import urlsplit

    from s1.jev_action_profiles import ACTION_PROFILES, DEFAULT_ACTION_PROFILE

    require(
        type(action_profile) is str and action_profile in ACTION_PROFILES,
        "Unknown explicit browser action profile",
    )
    require(
        (text_endpoint is None) == (text_model is None),
        "External helper requires both endpoint and model",
    )
    experimental = action_profile != DEFAULT_ACTION_PROFILE
    require(
        not experimental or text_endpoint is not None,
        "FIND_TEXT requires an explicit shared external HTTP text helper",
    )
    if text_endpoint is None:
        require(text_api_key_env == "S1_TEXT_API_KEY", "No external helper for this key setting")
        return {}
    require(
        type(text_endpoint) is str and text_endpoint == text_endpoint.strip(),
        "Explicit helper endpoint",
    )
    parsed = urlsplit(text_endpoint)
    require(
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        and parsed.port is not None
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment,
        "Shared GX10 helper requires an explicit loopback HTTP endpoint",
    )
    require(
        type(text_model) is str and bool(text_model.strip()) and text_model == text_model.strip(),
        "Explicit shared helper model",
    )
    require(
        type(text_api_key_env) is str
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text_api_key_env)
        and text_api_key_env
        not in {
            "S1_POLICY_API_KEY",
            "TEXT_MODEL_REASONING",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONNOUSERSITE",
            "PYTHONPATH",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "TOKENIZERS_PARALLELISM",
            "HF_HOME",
            "HF_HUB_CACHE",
        },
        "Distinct external helper key environment name",
    )
    result = {
        "action_profile": action_profile,
        "helper_tokens": None,
        "text_helper": {
            "endpoint": text_endpoint.rstrip("/"),
            "model": text_model,
            "api_key_env": text_api_key_env,
            "transport": "original_httpx",
            "original_request_max_tokens": 1024,
            "native_limit_authenticated": False,
            "external_runtime_receipt_required": True,
        },
    }
    if experimental:
        result["profile"] = "jev-current-nf4-v7-find-text-experimental-v1"
        result["experimental_scope"] = (
            "Nine operations; original ten cases, judges and sixteen decisions"
        )
    return result


def metadata_plan(
    adapter,
    output,
    upstream,
    prompt_format,
    *,
    action_profile="standard",
    text_endpoint=None,
    text_model=None,
    text_api_key_env="S1_TEXT_API_KEY",
):
    settings = action_settings(action_profile, text_endpoint, text_model, text_api_key_env)
    require(type(prompt_format) is str and prompt_format in FORMATS, "Explicit raw or V2 format")
    adapter, output, upstream = map(ordinary, (adapter, output, upstream))
    require(
        adapter.is_dir() and upstream.is_dir() and not output.exists(), "Fresh output and inputs"
    )
    require(
        not any(
            output.is_relative_to(p) or p.is_relative_to(output)
            for p in (adapter.parent, upstream, CACHE, ROOT / "src", ROOT / "scripts")
        ),
        "Output must not overlap the adapter, upstream, grammar or source trees",
    )
    manifest = ordinary(adapter / "manifest.json")
    require(manifest.is_file() and manifest.stat().st_size <= 4_000_000, "Small adapter manifest")
    from s1.browser_action_prompt import expected_format

    metadata = json.loads(manifest.read_bytes())
    expected_format(metadata, prompt_format)
    return {
        "profile": PROFILE,
        "prompt_format": prompt_format,
        "adapter": str(adapter),
        "adapter_manifest": pin(manifest),
        "output": str(output),
        "upstream": str(upstream),
        "model_identity": IDENTITY,
        "quantization": "nf4",
        "observation_profile": V7,
        "case_ids": list(KNOWN4 + PUBLIC6),
        "groups": ["known4", "public6"],
        "seed": 20261014,
        "repeats": 1,
        "max_steps": 16,
        "model_seconds": 20,
        "helper_seconds": 20,
        "helper_tokens": 128,
        "context": 8192,
        "HTTP_seconds": 25,
        "historical_reference": "Q95/BF16 V7 5/10; not a precision/helper matched control",
        "metadata_authenticated": False,
        "model_execution": False,
        **settings,
    }


def verify_adapter(path, prompt_format):
    from s1.browser_action_adapter import verify_action_adapter
    from s1.browser_action_prompt import expected_format

    verified = verify_action_adapter(
        path,
        model_id=BASE_ID,
        revision=REVISION,
        quantization="nf4",
        expected_prompt_format=prompt_format,
    )
    expected_format(verified.telemetry(), prompt_format)
    return verified


def construct_model():
    import torch

    from s1.unified import UnifiedDecisionModel

    torch.set_num_threads(2)
    torch.manual_seed(20261014)
    torch.backends.cuda.enable_cudnn_sdp(False)
    return UnifiedDecisionModel(
        BASE_ID,
        revision=REVISION,
        device="cuda",
        precision="bfloat16",
        quantization="nf4",
        max_context=8192,
        attn_implementation="sdpa",
    )


class CurrentNF4HelperGuard:
    """Helper protocol bridge, not the historical Q95 full-weight guard.

    full=True checks the same complete runtime identities/frozen state as check;
    it makes no claim of an old BF16 numerical or whole-head qualification.
    Original verified metadata and decision/base context managers are delegated.
    """

    def __init__(self, model):
        original = model.action_adapter
        require(not isinstance(original, CurrentNF4HelperGuard), "Wrap verified adapter once")
        self.original, self.model, self.lm = original, model, model.lm
        self.metadata = original.metadata
        self.head, self.backbone, self.temperature = model.head, model.backbone, model.temperature
        self.head_version = model.head.weight._version
        original._check()
        model.action_adapter = self
        self.check(full=True)

    def _check(self, *, disabled=False):
        require(type(disabled) is bool, "Explicit adapter state")
        require(
            self.model.action_adapter is self
            and self.original.model is self.model
            and self.original.lm is self.lm
            and self.model.lm is self.lm
            and self.model.head is self.head
            and self.model.backbone is self.backbone
            and self.head.weight._version == self.head_version
            and self.model.temperature == self.temperature
            and self.metadata is self.original.metadata,
            "Original verified adapter/model/head/temperature identity changed",
        )
        if disabled:
            status = self.lm.get_model_status()
            require(
                status.enabled is False
                and status.merged_adapters == []
                and status.active_adapters == ["default"]
                and status.available_adapters == ["default"],
                "Actual PEFT adapters must be disabled, unmerged and exclusively default",
            )
            require(not any(p.requires_grad for p in self.lm.parameters()), "Frozen helper base")
        else:
            self.original._check()

    def check(self, *, full=False):
        require(type(full) is bool, "Explicit current-runtime check flag")
        self._check()

    @contextlib.contextmanager
    def decision_context(self):
        self.check()
        with self.original.decision_context():
            yield

    @contextlib.contextmanager
    def base_context(self):
        self.check()
        try:
            with self.original.base_context():
                self._check(disabled=True)
                yield
        finally:
            self.check()


def prepare_helper(model):
    import jev_field_json_grammar as grammar
    import jev_field_syntax_helper as helper

    manifest = CACHE / "manifest.json"
    require(pin(manifest)["sha256"] == CACHE_SHA, "Selected full native grammar manifest")
    cache = json.loads(manifest.read_bytes())
    return helper.prepare_helper(
        model,
        cache_dir=CACHE,
        tokenizer_path=MODEL_CACHE / "tokenizer.json",
        generation_path=MODEL_CACHE / "generation_config.json",
        expected_cache_manifest_sha256=CACHE_SHA,
        expected_tokenizer_sha256=cache["tokenizer_sha256"],
        expected_generation_sha256=cache["generation_config_sha256"],
        expected_grammar_source_sha256=cache["source_sha256"],
        expected_helper_source_sha256=pin(helper.__file__)["sha256"],
        expected_guard_source_sha256=pin(__file__)["sha256"],
        expected_decoder_identity=grammar.DECODER_IDENTITY,
    )


def register_native_gate(model, gate):
    # PEFT.generate delegates to the base model and can bypass PEFT.forward.
    # The original generated-action observer uses this same actual backbone seam.
    return model.backbone.register_forward_pre_hook(
        lambda *_args, **_kwargs: gate.check("before-native-forward") and None,
        with_kwargs=True,
    )


def helper_context(payload, system_prompt):
    require(
        type(payload) is dict
        and set(payload) == {"model", "max_tokens", "response_format", "reasoning", "messages"},
        "Original helper request keys required",
    )
    require(
        payload["model"] == IDENTITY
        and type(payload["max_tokens"]) is int
        and payload["max_tokens"] == 1024
        and encode(payload["response_format"]) == encode({"type": "json_object"})
        and encode(payload["reasoning"]) == encode({"enabled": False}),
        "Original helper model/limits/no-reasoning contract required",
    )
    messages = payload["messages"]
    require(
        type(messages) is list
        and len(messages) == 2
        and messages[0] == {"role": "system", "content": system_prompt}
        and type(messages[1]) is dict
        and set(messages[1]) == {"role", "content"}
        and messages[1]["role"] == "user"
        and type(messages[1]["content"]) is str,
        "Original helper messages required",
    )
    context = json.loads(messages[1]["content"])
    require(
        type(context) is dict and set(context) == {"goal", "field", "page", "recent_actions"},
        "Original complete field context required",
    )
    return context


def http_deadline(started, *, clock=time.monotonic):
    if clock() - started >= 20:
        raise TimeoutError("Whole HTTP model/helper20s exceeded")


def _replace_http_evidence(directory, name, value, *, raw=False):
    """Atomically rebind only this request's final response/exchange, mode0600."""
    require(name in {"response.bin", "exchange.json"}, "Owned HTTP terminal evidence only")
    directory = ordinary(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = ordinary(directory / name)
    require(not target.exists() or target.is_file(), "Ordinary owned terminal file required")
    pending = directory / ("." + name + "." + secrets.token_hex(8) + ".pending")
    primary = None
    try:
        write(pending, value, raw=raw)
        os.replace(pending, target)
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            pending.unlink(missing_ok=True)
        except BaseException as exc:
            if primary is None:
                raise
            primary.add_note("HTTP pending evidence cleanup also failed: " + repr(exc))


def make_application(
    model,
    prepared,
    journal,
    gate,
    output,
    token,
    prompt_format,
    system_prompt,
    *,
    clock=None,
    action_profile="standard",
    allow_local_helper=True,
):
    """One current NF4 model, serial policy/helper requests, no historical qualification."""
    require(type(allow_local_helper) is bool, "Explicit local helper selection required")
    require(
        action_profile == "standard" or allow_local_helper is False,
        "Experimental action profile requires a shared external helper",
    )
    import jev_field_syntax_helper as helper
    from fastapi import Header, HTTPException
    from fastapi.responses import JSONResponse, Response

    from s1.api import create_app
    from s1.browser_policy import BrowserDecisionBackend

    def sink(event):
        trace = TRACE.get()
        require(trace is not None, "Owned HTTP trace required")
        trace["api_request_id"] = event["request_id"]
        journal(event)

    backend = BrowserDecisionBackend(
        model,
        prompt_mode="native",
        analysis_style="detailed",
        decision_readout=None,
        prefix_cache=False,
        prefix_suffix_batch_size=1,
    )
    app = create_app(
        backend,
        api_key=token,
        jev_compatible=True,
        jev_generated_actions=True,
        jev_generated_action_grammar=True,
        jev_generated_prompt_format=prompt_format,
        model_identity=IDENTITY,
        request_evidence_sink=sink,
        **({"jev_action_profile": action_profile} if action_profile != "standard" else {}),
    )
    serial = asyncio.Lock()
    sequence = 0
    clock = time.monotonic if clock is None else clock

    @app.middleware("http")
    async def record_http(request, call_next):
        nonlocal sequence
        started = clock()
        async with serial:
            number, sequence = sequence, sequence + 1
            directory = Path(output) / "http" / f"{number:04d}"
            trace = {
                "request_id": secrets.token_hex(16),
                "api_request_id": None,
                "started": started,
            }
            previous = TRACE.get()
            handle = TRACE.set(trace)
            primary = None
            status, body = 500, encode({"detail": "Request failed; no action authorized"})
            record = {
                "sequence": number,
                "path": request.url.path,
                "method": request.method,
                "client_delivery": "unknown; join the actual browser client evidence",
                "request_id": trace["request_id"],
            }
            try:
                http_deadline(started, clock=clock)
                require(
                    (request.method, request.url.path)
                    in {
                        ("GET", "/healthz"),
                        ("POST", "/v1/systemone"),
                        ("POST", "/chat/completions"),
                    },
                    "Only the declared current browser API routes are available",
                )
                original = await request.body()
                require(len(original) <= MAX_BODY, "Bounded original HTTP body")
                write(directory / "request.bin", original, raw=True)
                gate.check("before-http-request")
                http_deadline(started, clock=clock)
                response = await call_next(request)
                candidate = b"".join([part async for part in response.body_iterator])
                write(directory / "candidate-response.bin", candidate, raw=True)
                status, body = response.status_code, candidate
                gate.check("after-http-request")
                require(not app.state.request_evidence_errors, "Policy evidence failure is latched")
                http_deadline(started, clock=clock)
            except BaseException as exc:
                primary = exc
                status = 504 if isinstance(exc, TimeoutError) else 502
                record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                body = encode({"detail": record["error"]})
                if not isinstance(exc, Exception):
                    raise
            finally:
                try:
                    TRACE.reset(handle)
                except BaseException as exc:
                    try:
                        TRACE.set(previous)
                    except BaseException as extra:
                        exc.add_note("HTTP TRACE fallback also failed: " + repr(extra))
                    if primary is not None:
                        primary.add_note("HTTP TRACE reset also failed: " + repr(exc))
                        raise primary from exc
                    raise

            def publish():
                _replace_http_evidence(directory, "response.bin", body, raw=True)
                record.update(
                    status=status,
                    api_request_id=trace["api_request_id"],
                    request=pin(directory / "request.bin")
                    if (directory / "request.bin").exists()
                    else None,
                    response=pin(directory / "response.bin"),
                    elapsed_seconds=clock() - started,
                    elapsed_seconds_scope="after_response_binding_before_this_exchange_fsync",
                    deadline_gate="after_all_publication_at_ASGI_http.response.start",
                )
                _replace_http_evidence(directory, "exchange.json", record)

            def finalize(delivery, start_message):
                nonlocal status, body, primary

                def bind():
                    delivery.status_code, delivery.body = status, body
                    delivery.init_headers(
                        {"X-Request-ID": trace["api_request_id"] or trace["request_id"]}
                    )
                    return {
                        **start_message,
                        "status": delivery.status_code,
                        "headers": delivery.raw_headers,
                    }

                try:
                    publish()
                    message = bind()
                    if status == 200:
                        try:
                            http_deadline(started, clock=clock)
                        except TimeoutError as exc:
                            primary = exc
                            status = 504
                            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                            body = encode({"detail": record["error"]})
                            # Refusal publication can be late; it never authorizes
                            # the retained candidate and needs no recursive gate.
                            publish()
                            message = bind()
                except BaseException as exc:
                    if primary is not None and exc is not primary:
                        primary.add_note("HTTP final evidence also failed: " + repr(exc))
                        raise primary from exc
                    raise
                return message

            class PublishedResponse(Response):
                async def __call__(self, scope, receive, send):
                    async def final_send(message):
                        if message["type"] == "http.response.start":
                            # Reacquire safely at this ASGI boundary instead of
                            # transferring a held lock across dispatch's return.
                            # Queue time remains inside the original20s budget.
                            async with serial:
                                message = finalize(self, message)
                        # This is the last cooperative authorization boundary.
                        # Awaiting send does not prove network/client delivery.
                        await send(message)

                    await super().__call__(scope, receive, final_send)

            return PublishedResponse(
                body,
                status_code=status,
                media_type="application/json",
                headers={"X-Request-ID": trace["api_request_id"] or trace["request_id"]},
            )

    @app.post("/chat/completions")
    def chat(payload: dict, authorization: str | None = Header(default=None)):
        if not allow_local_helper:
            raise HTTPException(status_code=404, detail="Shared external helper selected")
        if not secrets.compare_digest(authorization or "", "Bearer " + token):
            raise HTTPException(status_code=401, detail="Invalid bearer token")
        trace = TRACE.get()
        require(trace is not None, "Owned helper request")
        rid, counter = trace["request_id"], 0

        def emit(name, data):
            nonlocal counter
            journal(
                {
                    "request_id": rid,
                    "sequence": counter,
                    "event": name,
                    "recorded_unix": time.time(),
                    "data": data,
                }
            )
            counter += 1

        emit("request_started", {"payload": payload, "helper_profile": PROFILE})
        try:
            context = helper_context(payload, system_prompt)
            text, generation = helper.generate_field_text(
                prepared,
                context,
                started=trace["started"],
                evidence_sink=lambda event: emit("helper_checkpoint", event),
            )
            require(
                generation["success"] is True and generation["response_authorized"] is True,
                "Constrained helper did not authorize a value",
            )
            result = {
                "model": IDENTITY,
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"text": text}, ensure_ascii=False),
                        }
                    }
                ],
                "usage": generation["usage"],
            }
            emit("server_returned", {"response": result, "generation": generation})
            return result
        except Exception as exc:
            emit(
                "server_failed",
                {
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "generation": getattr(exc, "generation", None),
                },
            )
            return JSONResponse(
                {"detail": "Helper refused; no field text authorized"}, status_code=502
            )

    return app


def matrix_common():
    spec = importlib.util.spec_from_file_location("_jev_native_browser_owned_cleanup", MATRIX)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def composition_paths(action_profile="standard"):
    labels = (
        "scripts/jev_native_v2_browser.py",
        "scripts/jev_browser_profile_v7_benchmark.py",
        "scripts/jev_observation_profile_v7.py",
        "scripts/jev_heldout_public_cases.py",
        "scripts/jev_field_syntax_helper.py",
        "scripts/jev_field_json_grammar.py",
        str(MATRIX.relative_to(ROOT)),
    )
    result = {label: ROOT / label for label in labels}
    if action_profile != "standard":
        from s1.jev_action_profiles import resolve_protocol

        protocol = resolve_protocol(action_profile)
        for row in protocol.provenance["source_dependencies"]:
            result[row["path"]] = ROOT / row["path"]
        for label in ("src/s1/jev_action_profiles.py", "scripts/jev_find_text_browser.py"):
            result[label] = ROOT / label
    return result


def capture_composition(output, action_profile="standard"):
    result = {}
    for label, path in composition_paths(action_profile).items():
        blob = path.read_bytes()
        write(Path(output) / "composition-sources" / label, blob, raw=True)
        result[label] = pin(path)
    write(Path(output) / "composition-sources.json", result)
    return result


def verify_composition(output, expected, action_profile="standard"):
    require(
        set(expected) == set(composition_paths(action_profile)),
        "Composition source population changed",
    )
    require(
        all(
            pin(path) == expected[label] == pin(Path(output) / "composition-sources" / label)
            for label, path in composition_paths(action_profile).items()
        ),
        "Composition source bytes changed",
    )


def output_inventory(output):
    files, total = {}, 0
    for path in sorted(Path(output).rglob("*")):
        require(not path.is_symlink(), "Owned output contains a symlink")
        if path.is_dir():
            continue
        require(path.is_file() and len(files) < 20_000, "Bounded ordinary output population")
        row = pin(path)
        total += row["size_bytes"]
        require(total <= 4 * GIB, "Owned browser evidence exceeded4GiB")
        files[path.relative_to(output).as_posix()] = row
    return files


def run_client(argv, env, output):
    """Reuse the existing PID/start ownership and bounded group cleanup primitive."""
    common = matrix_common()
    known, groups, child, primary = {}, set(), None, None
    secondary, streams = [], []
    try:
        for name in ("stdout.txt", "stderr.txt"):
            streams.append((Path(output) / name).open("xb"))
        child = subprocess.Popen(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=streams[0],
            stderr=streams[1],
            start_new_session=True,
        )
        identity = common.process_table().get(child.pid)
        require(
            identity is not None
            and identity["ppid"] == os.getpid()
            and identity["pid"] == identity["pgid"] == identity["sid"] == child.pid
            and type(identity["start_ticks"]) is int
            and identity["start_ticks"] > 0,
            "Actual owned browser child PID/start/PGID/SID required",
        )
        # extend_owned expands authenticated seeds; an empty mapping never
        # discovers even the main child. Seed before poll can reap a fast exit.
        known[child.pid] = identity["start_ticks"]
        groups.add(identity["pgid"])
        deadline = time.monotonic() + 7200
        while True:
            common.extend_owned(child.pid, known, groups)
            if child.poll() is not None:
                break
            require(time.monotonic() < deadline, "Owned browser group wall bound exceeded")
            time.sleep(0.1)
        require(child.returncode == 0, "V7 browser process failed; partial evidence retained")
    except BaseException as exc:
        primary = exc
    finally:
        if child is not None:
            closure = {"confirmed": False, "error": "Owned cleanup did not complete"}
            try:
                closure = common.stop_owned(child, known, groups)
            except BaseException as exc:
                closure["error"] = {"type": type(exc).__name__, "message": str(exc)}
                secondary.append(("Owned browser cleanup", exc))
            try:
                write(Path(output) / "owned-client-closure.json", closure)
            except BaseException as exc:
                secondary.append(("Owned browser closure publication", exc))
            if not closure["confirmed"]:
                secondary.append(
                    (
                        "Owned browser closure",
                        RuntimeError("Owned browser process group closure unconfirmed"),
                    )
                )
        for stream in reversed(streams):
            try:
                stream.close()
            except BaseException as exc:
                secondary.append(("Browser log stream close", exc))
    if primary is None and secondary:
        primary = secondary[0][1]
    if primary is not None:
        for stage, exc in secondary:
            if exc is not primary:
                primary.add_note(stage + ": " + repr(exc))
        raise primary


def run_groups(
    output,
    upstream,
    endpoint,
    token,
    chromium,
    *,
    completed=None,
    action_profile="standard",
    text_helper=None,
):
    if action_profile == "standard":
        import jev_browser_profile_v7_benchmark as v7
    else:
        import jev_find_text_browser as v7
    helper_settings = action_settings(
        action_profile,
        None if text_helper is None else text_helper["endpoint"],
        None if text_helper is None else text_helper["model"],
        "S1_TEXT_API_KEY" if text_helper is None else text_helper["api_key_env"],
    )
    selected_helper = helper_settings.get("text_helper")

    groups = []
    for name, mode, ids in (("known4", "original", KNOWN4), ("public6", "heldout", PUBLIC6)):
        bundle, manifest = v7.prepare_bundle(
            Path(output) / name / "runtime-bundle", case_module=mode
        )
        argv = [
            sys.executable,
            "-B",
            "-s",
            str(bundle / "runner.py"),
            "--upstream-path",
            str(upstream),
            "--endpoint",
            endpoint + "/v1/systemone",
            "--model",
            IDENTITY,
            "--output-dir",
            str(Path(output) / name / "results"),
            "--text-endpoint",
            endpoint if selected_helper is None else selected_helper["endpoint"],
            "--text-model",
            IDENTITY if selected_helper is None else selected_helper["model"],
            "--api-key-env",
            "S1_POLICY_API_KEY",
            "--text-api-key-env",
            "S1_TEXT_API_KEY" if selected_helper is None else selected_helper["api_key_env"],
            "--repeats",
            "1",
            "--max-steps",
            "16",
            "--seed",
            "20261014",
            "--suite-split",
            "development",
            "--chromium-executable",
            chromium,
            "--observation-profile",
            V7,
            "--observation-profile-helper-sha256",
            V7_SHA,
            "--public-cases",
            *ids,
        ]
        env = {
            **os.environ,
            "S1_POLICY_API_KEY": token,
            **({"S1_TEXT_API_KEY": token} if selected_helper is None else {}),
            "TEXT_MODEL_REASONING": "none",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join(
                str(ROOT / p) for p in ("artifacts/jev-runtime/overlay", "src", "scripts")
            ),
        }
        primary = None
        try:
            run_client(argv, env, Path(output) / name)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            try:
                v7.verify_bundle(bundle, manifest)
            except BaseException as exc:
                if primary is None:
                    raise
                primary.add_note("V7 bundle final verification: " + repr(exc))
        result = json.loads((Path(output) / name / "results/summary.json").read_bytes())
        require(result["attempts"] == len(ids), "Every fixed group case must be retained")
        group = {"group": name, "denominator": len(ids), "summary": result}
        groups.append(group)
        if completed is not None:
            completed(group)
        require(not result.get("observation_profile_failed"), "V7 observation evidence failed")
    return groups


def write_final_receipt(output, receipt):
    # Reuse the existing atomic writer so a late index failure can replace the
    # earlier pending-closure receipt with a failed receipt without truncation.
    from jev_local_model import _write_receipt

    _write_receipt(Path(output) / "receipt.json", receipt, qualified=True)


def run(plan, *, expected_boot_id, chromium):
    """Explicit actor boundary. Caller owns an external process/cgroup closure check."""
    from s1.browser_action_adapter import attach_action_adapter
    from s1.browser_evidence import capture_runtime_sources, verify_runtime_sources
    from s1.browser_server_evidence import ServerEvidenceJournal

    require(expected_boot_id, "Explicit expected boot ID required")
    action_profile = plan.get("action_profile", "standard")
    text_helper = plan.get("text_helper")
    action_settings(
        action_profile,
        None if text_helper is None else text_helper["endpoint"],
        None if text_helper is None else text_helper["model"],
        "S1_TEXT_API_KEY" if text_helper is None else text_helper["api_key_env"],
    )
    output = ordinary(plan["output"])
    require(not output.exists(), "Fresh execution output")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HOME=str(ROOT / "artifacts/jev-local-hf"),
        HF_HUB_CACHE=str(ROOT / "artifacts/jev-local-hf/hub"),
    )
    output.mkdir(parents=True)
    receipt = {
        **plan,
        "stage": "setup",
        "model_execution": False,
        "groups": [],
        "process_exit_confirmed": False,
        "outer_closure_required": True,
    }
    gate = ResourceJournal(output / "resources.jsonl", expected_boot_id)
    model = hook = server = thread = journal = sources = None
    composition = None
    primary = None
    own_source = pin(__file__)
    try:
        gate.check("before-artifact-and-framework", "cpu")
        write(output / "plan.json", plan)
        verified = verify_adapter(plan["adapter"], plan["prompt_format"])
        require(
            verified.manifest_sha256 == plan["adapter_manifest"]["sha256"],
            "Selected adapter manifest changed after metadata planning",
        )
        receipt["metadata_authenticated"] = True
        sources = capture_runtime_sources(output)
        composition = (
            capture_composition(output)
            if action_profile == "standard"
            else capture_composition(output, action_profile)
        )
        gate.check("immediately-before-model-constructor")
        receipt["model_load_started"] = True
        write(
            output / "model-constructor-intent.json",
            {
                "model": BASE_ID,
                "revision": REVISION,
                "quantization": "nf4",
                "monotonic": time.monotonic(),
                "boot_id": expected_boot_id,
                "constructor_completion_unknown": True,
            },
        )
        model = construct_model()
        receipt["model_load_returned"] = True
        receipt["model_execution"] = True
        write(
            output / "model-loaded.json",
            {
                "model": model.name,
                "revision": model.revision,
                "quantization": model.quantization,
                "temperature": model.temperature,
                "max_context": model.max_context,
            },
        )
        receipt["attached_adapter"] = attach_action_adapter(model, verified)
        guard = CurrentNF4HelperGuard(model)
        receipt["helper_guard"] = {
            "profile": "current-nf4-helper-runtime-identity-v1",
            "wrapped_type": type(guard.original).__name__,
            "source": pin(__file__),
            "original_guard_source": pin(
                __import__(type(guard.original).__module__, fromlist=["__file__"]).__file__
            ),
            "coverage": "Original enabled/disabled PEFT contexts, frozen state, model/lm/head/version/temperature identities; no Q95 fullweight qualification",
        }
        hook = register_native_gate(model, gate)
        if text_helper is None:
            gate.check("before-full-helper-preparation")
            prepared = prepare_helper(model)
            write(output / "helper-prepared.json", prepared.receipt())
        else:
            prepared = None
            write(output / "external-helper-selection.json", text_helper)
        questions = ast.parse((Path(plan["upstream"]) / "jev_ultrafast/questions.py").read_bytes())
        system_prompt = next(
            ast.literal_eval(node.value)
            for node in questions.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "TEXT_VALUE" for t in node.targets)
        )
        journal = ServerEvidenceJournal(output / "server")
        token = secrets.token_urlsafe(32)
        app = make_application(
            model,
            prepared,
            journal,
            gate,
            output,
            token,
            plan["prompt_format"],
            system_prompt,
            **(
                {"action_profile": action_profile, "allow_local_helper": False}
                if text_helper is not None
                else {}
            ),
        )
        import uvicorn

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        endpoint = "http://127.0.0.1:" + str(listener.getsockname()[1])
        receipt["endpoint"] = endpoint
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        end = time.monotonic() + 15
        while not server.started and thread.is_alive() and time.monotonic() < end:
            time.sleep(0.02)
        require(server.started, "Owned loopback server did not start")
        write(
            output / "loopback-ready.json",
            {
                "profile": plan.get("profile", PROFILE),
                **({"action_profile": action_profile} if action_profile != "standard" else {}),
                "endpoint": endpoint,
                "model": IDENTITY,
                "prompt_format": plan["prompt_format"],
                "boot_id": expected_boot_id,
                "pid": os.getpid(),
                "pgid": os.getpgid(0),
                "sid": os.getsid(0),
                "qualified": False,
                "source": own_source,
            },
        )
        receipt["stage"] = "browser"
        run_groups(
            output,
            plan["upstream"],
            endpoint,
            token,
            chromium,
            completed=receipt["groups"].append,
            **(
                {"action_profile": action_profile, "text_helper": text_helper}
                if text_helper is not None
                else {}
            ),
        )
        receipt["attempts"] = sum(group["summary"]["attempts"] for group in receipt["groups"])
        receipt["successes"] = sum(group["summary"]["successes"] for group in receipt["groups"])
        require(receipt["attempts"] == 10 and gate.fatal is None, "Complete fixed ten required")
        receipt["stage"] = "complete_pending_external_closure"
    except BaseException as exc:
        primary = exc
        receipt.update(stage="failed", error={"type": type(exc).__name__, "message": str(exc)})
    finally:
        errors, secondary = [], []
        receipt["cleanup_errors"] = errors

        def failed(stage, exc):
            errors.append(stage + ": " + repr(exc))
            secondary.append((stage, exc))
            receipt["stage"] = "failed"
            first = primary if primary is not None else secondary[0][1]
            receipt.setdefault("error", {"type": type(first).__name__, "message": str(first)})

        thread_drained = thread is None
        receipt["server_thread_drained"] = thread_drained
        try:
            if server is not None:
                server.should_exit = True
        except BaseException as exc:
            failed("Server shutdown request", exc)
        try:
            if thread is not None:
                thread.join(timeout=30)
                thread_drained = not thread.is_alive()
                receipt["server_thread_drained"] = thread_drained
                if not thread_drained:
                    failed(
                        "Server drain",
                        RuntimeError("Server thread still live; external owned cleanup required"),
                    )
        except BaseException as exc:
            failed("Server thread join; closure unconfirmed", exc)
        try:
            if hook is not None and thread_drained:
                hook.remove()
        except BaseException as exc:
            failed("Forward hook cleanup", exc)
        try:
            if journal is not None:
                snapshot = journal.snapshot()
                snapshot["local_service_thread_drained"] = thread_drained
                write(output / "server-index.json", snapshot)
                if (
                    snapshot["pending_request_ids"]
                    or snapshot["errors"]
                    or not snapshot["all_requests_sealed"]
                ):
                    failed(
                        "Server evidence closure",
                        RuntimeError("Server evidence pending or unsealed"),
                    )
        except BaseException as exc:
            failed("Server evidence closure", exc)
        try:
            if sources is not None:
                receipt["source_verification"] = verify_runtime_sources(sources)
                if not receipt["source_verification"]["success"]:
                    failed("Source closure", RuntimeError("Current execution sources changed"))
            if composition is not None:
                if action_profile == "standard":
                    verify_composition(output, composition)
                else:
                    verify_composition(output, composition, action_profile)
            if own_source != pin(__file__):
                failed("Source closure", RuntimeError("Browser entrypoint changed"))
        except BaseException as exc:
            failed("Source closure", exc)
        try:
            if gate.fatal is None and thread_drained:
                gate.check("after-model-service-drain", "cpu")
                if receipt.get("metadata_authenticated"):
                    final_adapter = verify_adapter(plan["adapter"], plan["prompt_format"])
                    require(
                        final_adapter.manifest_sha256 == verified.manifest_sha256
                        and final_adapter.receipt_sha256 == verified.receipt_sha256
                        and final_adapter.files == verified.files,
                        "Selected full adapter evidence changed during browser run",
                    )
        except BaseException as exc:
            failed("Final resource/artifact closure", exc)
        receipt["planned_denominator"] = 10
        receipt["completed_group_attempts"] = sum(
            group["summary"]["attempts"] for group in receipt["groups"]
        )
        receipt["completed_group_successes"] = sum(
            group["summary"]["successes"] for group in receipt["groups"]
        )
        receipt["partial_group_count_scope"] = (
            "Completed group summaries only; interrupted group attempts require the retained client evidence audit"
        )
        files = None
        try:
            if thread_drained:
                files = output_inventory(output)
        except BaseException as exc:
            failed("Owned output inventory", exc)
        publication_failed = False
        try:
            write_final_receipt(output, receipt)
        except BaseException as exc:
            publication_failed = True
            failed("Final receipt publication", exc)
        if files is not None and not publication_failed:
            try:
                files["receipt.json"] = pin(output / "receipt.json")
                write(
                    output / "output-index.json",
                    {
                        "scope": "Complete owned output excluding this index, after local server drain; external process/cgroup closure still required",
                        "files": files,
                        "file_count": len(files),
                        "total_bytes": sum(row["size_bytes"] for row in files.values()),
                    },
                )
            except BaseException as exc:
                publication_failed = True
                failed("Final output index publication", exc)
        if publication_failed:
            try:
                write_final_receipt(output, receipt)
            except BaseException as exc:
                failed("Failed finalization receipt publication", exc)
    if primary is None and secondary:
        primary = secondary[0][1]
    if primary is not None:
        for stage, exc in secondary:
            if exc is not primary:
                primary.add_note(stage + ": " + repr(exc))
        raise primary
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt-format", choices=FORMATS, default="shared-elements-v2")
    parser.add_argument(
        "--action-profile",
        choices=("standard", "jev-find-text-experimental-v1"),
        default="standard",
    )
    parser.add_argument("--text-endpoint")
    parser.add_argument("--text-model")
    parser.add_argument("--text-api-key-env", default="S1_TEXT_API_KEY")
    parser.add_argument(
        "--upstream-path", type=Path, default=ROOT / "artifacts/jev-runtime/upstream"
    )
    parser.add_argument("--chromium-executable", default=CHROMIUM)
    parser.add_argument("--expected-boot-id")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.run and not args.expected_boot_id:
        parser.error("--run requires --expected-boot-id")
    plan = metadata_plan(
        args.action_adapter,
        args.output_dir,
        args.upstream_path,
        args.prompt_format,
        action_profile=args.action_profile,
        text_endpoint=args.text_endpoint,
        text_model=args.text_model,
        text_api_key_env=args.text_api_key_env,
    )
    result = (
        run(plan, expected_boot_id=args.expected_boot_id, chromium=args.chromium_executable)
        if args.run
        else plan
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
