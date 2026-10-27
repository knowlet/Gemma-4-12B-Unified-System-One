"""Held synthetic V7/V7-links target-selection smoke, not an agent benchmark.

A caller must independently supervise the supplied local model service and the
whole process, including a hard wall and resource/cleanup checks. This module
cannot authenticate the endpoint, source execution, GPU, or process closure.
It never clicks links and returns every planned attempt, including failures.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
import sys
import threading
import time
import types
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import jev_observation_profile_v7 as v7
import jev_observation_profile_v7_links as links

SOURCE_ONLY = True
MAX_RESPONSE = 16 * 1024 * 1024
MAX_REQUEST = 1_000_000
MAX_WALL_SECONDS = 120
_LOCK = threading.Lock()


def _need(value, message):
    if not value:
        raise ValueError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _source(root, label):
    path = root / label
    _need(path.is_absolute() and path.resolve() == path, "ordinary upstream source required")
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        _need(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= v7.MAX_SOURCE_BYTES,
            "bounded ordinary upstream source required",
        )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(v7.MAX_SOURCE_BYTES + 1)
        after = os.fstat(fd)
        current = os.stat(path, follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        _need(
            tuple(getattr(before, f) for f in fields)
            == tuple(getattr(after, f) for f in fields)
            == tuple(getattr(current, f) for f in fields)
            and len(raw) == before.st_size
            and _sha(raw) == v7.CANONICAL_SHA256[label],
            "fixed upstream source differs",
        )
        return raw
    finally:
        os.close(fd)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, "duplicate response JSON key")
        result[key] = value
    return result


def _endpoint(value):
    _need(type(value) is str, "explicit loopback endpoint required")
    try:
        parsed = urlsplit(value)
        _need(
            parsed.scheme == "http"
            and parsed.hostname == "127.0.0.1"
            and parsed.port
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and "?" not in value
            and "#" not in value
            and "\\" not in value
            and not any(ord(c) < 33 or ord(c) == 127 for c in value),
            "explicit credential-free loopback endpoint required",
        )
    except ValueError:
        raise ValueError("explicit credential-free loopback endpoint required") from None
    return value


def _sources(root, links_sha256):
    root = Path(root)
    _need(root.is_absolute() and root.is_dir() and root.resolve() == root, "upstream root required")
    links._source_pin(links_sha256)
    snapshot = _source(root, links.SNAPSHOT)
    model = _source(root, links.MODEL)
    questions = _source(root, "jev_ultrafast/questions.py")
    parent_snapshot = v7._derive_snapshot(v7._PARENT._derive_v3(snapshot))
    return {
        "v7": (parent_snapshot, model),
        "v7-links": (links._derive_snapshot(parent_snapshot), links._derive_model(model)),
    }, questions


def _modules(sources, questions, model_alias):
    package_name = "_jev_loopback_smoke_package"
    names = [package_name, package_name + ".questions"]
    _need(not any(name in sys.modules for name in names), "fresh smoke module names required")
    package = types.ModuleType(package_name)
    package.__path__ = []
    questions_module = types.ModuleType(names[1])
    questions_module.__package__ = package_name
    sys.modules[package_name] = package
    sys.modules[names[1]] = questions_module
    modules = {}
    try:
        exec(
            compile(questions, "pinned_questions.py", "exec", dont_inherit=True),
            questions_module.__dict__,
        )
        for name, (_, source) in sources.items():
            module = types.ModuleType(package_name + ".model_" + name.replace("-", "_"))
            module.__package__ = package_name
            exec(compile(source, "pinned_model.py", "exec", dont_inherit=True), module.__dict__)
            module.os = types.SimpleNamespace(
                environ={"TYPESAFE_API_KEY": "synthetic", "TYPESAFE_MODEL": model_alias}
            )
            modules[name] = module
        return modules
    except BaseException:
        for module in modules.values():
            module.CLIENT.close()
        raise
    finally:
        if sys.modules.get(names[1]) is questions_module:
            del sys.modules[names[1]]
        if sys.modules.get(package_name) is package:
            del sys.modules[package_name]


class CaptureSinkError(RuntimeError):
    """The private caller failed to accept a capture event."""


class SupervisorGuardError(RuntimeError):
    """The external guard refused or failed; no further call is allowed."""


def _guard(callback):
    try:
        result = callback()
        if result is not None and result is not True:
            close = getattr(result, "close", None)
            if callable(close):
                close()
            raise ValueError("synchronous guard must approve")
    except Exception:
        raise SupervisorGuardError("external guard refused") from None


def _emit(sink, phase, value):
    try:
        result = sink({"phase": phase, "data": copy.deepcopy(value)})
        if result is not None:
            close = getattr(result, "close", None)
            if callable(close):
                close()
            raise ValueError("synchronous event sink required")
    except Exception:
        raise CaptureSinkError("capture sink failed") from None


def _post(client, endpoint, token, request, deadline, observation, event_sink):
    remaining = deadline - time.monotonic()
    _need(remaining > 0, "cooperative deadline exceeded before call")
    raw = json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    _need(len(raw) <= MAX_REQUEST, "synthetic request exceeds bound")
    observation["request"] = copy.deepcopy(request)
    observation["request_sha256"] = _sha(raw)
    observation["request_bytes_utf8"] = raw.decode("utf-8")
    _emit(event_sink, "request_prepared", observation)
    observation["http_started"] = True
    try:
        with client.stream(
            "POST",
            endpoint,
            content=raw,
            headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
            timeout=min(remaining, 25),
        ) as response:
            observation["http_status"] = response.status_code
            data = bytearray()
            for chunk in response.iter_bytes():
                if len(data) + len(chunk) > MAX_RESPONSE:
                    observation["response_complete"] = False
                    observation["response_prefix_bytes_retained_before_overflow"] = len(data)
                    observation["response_bytes_omitted_reason"] = "response_limit"
                    raise ValueError("response exceeds bound")
                data.extend(chunk)
            raw_response = bytes(data)
            unescaped = re.sub(
                rb"\\u00([0-7][0-9a-fA-F])",
                lambda match: bytes([int(match.group(1), 16)]),
                raw_response,
            )
            if token.encode() in raw_response or token.encode() in unescaped:
                observation["response_complete"] = False
                observation["response_rejected_secret"] = True
                raise ValueError("private token found in response")
            # The bearer is restricted to ASCII header-safe characters. The
            # raw and ASCII JSON-unescaped scans above cover direct and \u00XX
            # encodings before any response bytes are handed to the sink.
            observation["response_complete"] = True
            observation["response_sha256"] = _sha(raw_response)
            observation["response_bytes_hex"] = raw_response.hex()
            _emit(event_sink, "response_captured", observation)
            _need(response.status_code == 200, "model endpoint returned non-200")
            _need(time.monotonic() < deadline, "cooperative deadline exceeded after call")
            try:
                parsed = json.loads(
                    raw_response,
                    object_pairs_hook=_unique_pairs,
                    parse_constant=lambda _: (_ for _ in ()).throw(
                        ValueError("nonfinite response")
                    ),
                )
                # Refuse lone surrogate strings that cannot be represented as
                # ordinary UTF-8, without discarding the captured raw bytes.
                json.dumps(parsed, ensure_ascii=False, allow_nan=False).encode()
            except (ValueError, UnicodeDecodeError, UnicodeEncodeError, RecursionError):
                parsed = None
            _need(parsed is not None, "model response must be unambiguous UTF-8 JSON")
            return parsed
    except Exception as exc:
        observation["transport_or_response_error"] = type(exc).__name__
        if not isinstance(exc, CaptureSinkError):
            _emit(event_sink, "transport_or_response_failed", observation)
        raise


def run(
    *,
    upstream_root,
    links_sha256,
    endpoint,
    token,
    model_alias,
    chromium_executable,
    wall_seconds,
    supervisor_check,
    event_sink,
):
    """Evaluate four local synthetic selections without clicking any target.

    ``supervisor_check`` is a caller-supplied synchronous guard, not proof of
    supervision; it must return None or True and its refusal is fatal.
    Internal deadlines are cooperative; an independent owner must hard-stop the
    complete process and verify resource/endpoint identity and closure.
    """
    _need(SOURCE_ONLY is False, "held synthetic smoke requires independent review")
    endpoint = _endpoint(endpoint)
    _need(
        type(token) is str
        and 32 <= len(token) <= 256
        and token.isascii()
        and all(c.isalnum() or c in "-_" for c in token),
        "private bearer token required",
    )
    _need(
        type(model_alias) is str and 0 < len(model_alias) <= 128 and token not in model_alias,
        "model alias required",
    )
    _need(
        type(chromium_executable) is str
        and Path(chromium_executable).is_absolute()
        and Path(chromium_executable).is_file(),
        "explicit Chromium executable required",
    )
    _need(
        type(wall_seconds) in (int, float)
        and math.isfinite(wall_seconds)
        and 0 < wall_seconds <= MAX_WALL_SECONDS
        and callable(supervisor_check)
        and callable(event_sink),
        "bounded wall, external guard and private event sink required",
    )
    _need(_LOCK.acquire(blocking=False), "serial smoke ownership required")
    modules = {}
    schedule = (
        ("same-first", "v7"),
        ("same-first", "v7-links"),
        ("cross-first", "v7-links"),
        ("cross-first", "v7"),
    )
    trials = [
        {
            "position": position,
            "profile": arm,
            "status": "not_started",
            "expected_same_origin_index": "1" if position == "same-first" else "2",
            "http_started": False,
            "request": None,
            "response_complete": None,
        }
        for position, arm in schedule
    ]
    deadline = time.monotonic() + wall_seconds
    fatal_error = None
    source_observations = None
    cleanup_errors = []

    def cleanup(component, close):
        try:
            close()
        except Exception as exc:
            cleanup_errors.append({"component": component, "error_type": type(exc).__name__})

    try:
        _emit(event_sink, "planned", trials)
        sources, questions = _sources(upstream_root, links_sha256)
        source_observations = {
            "upstream_snapshot_sha256": v7.CANONICAL_SHA256[links.SNAPSHOT],
            "upstream_model_sha256": v7.CANONICAL_SHA256[links.MODEL],
            "upstream_questions_sha256": v7.CANONICAL_SHA256["jev_ultrafast/questions.py"],
            "v7_helper_sha256": links.V7_SHA256,
            "v7_links_helper_sha256": links_sha256,
            "derived": {
                name: {"snapshot_sha256": _sha(pair[0]), "model_sha256": _sha(pair[1])}
                for name, pair in sources.items()
            },
        }
        modules = _modules(sources, questions, model_alias)
        from playwright.sync_api import sync_playwright

        _guard(supervisor_check)
        _need(time.monotonic() < deadline, "cooperative deadline exceeded before browser")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=chromium_executable, headless=True, args=["--no-sandbox"]
            )
            try:
                context = browser.new_context(viewport={"width": 1120, "height": 780})
                try:
                    page = context.new_page()
                    unexpected = []
                    order = {"value": None}

                    def route(route):
                        if route.request.url == "http://127.0.0.1:53001/synthetic":
                            first = order["value"] == "same-first"
                            a, b = (
                                ("/same", "http://127.0.0.1:53002/other")
                                if first
                                else ("http://127.0.0.1:53002/other", "/same")
                            )
                            body = f'<!doctype html><title>Synthetic</title><main><a href="{a}">Open</a><br><a href="{b}">Open</a></main>'
                            route.fulfill(status=200, content_type="text/html", body=body)
                        else:
                            unexpected.append("blocked")
                            route.abort()

                    context.route("**/*", route)
                    client = httpx.Client(trust_env=False, follow_redirects=False)
                    try:
                        for position, sequence in (
                            ("same-first", ("v7", "v7-links")),
                            ("cross-first", ("v7-links", "v7")),
                        ):
                            order["value"] = position
                            _guard(supervisor_check)
                            _need(
                                time.monotonic() < deadline,
                                "cooperative deadline exceeded before page",
                            )
                            page.goto(
                                "http://127.0.0.1:53001/synthetic",
                                timeout=max(1, int((deadline - time.monotonic()) * 1000)),
                            )
                            states = {
                                name: page.evaluate(sources[name][0].decode())
                                for name in ("v7", "v7-links")
                            }
                            baseline = {
                                k: value
                                for k, value in states["v7"].items()
                                if k != "observation_profile"
                            }
                            enriched = {
                                k: value
                                for k, value in states["v7-links"].items()
                                if k not in {"observation_profile", "link_destinations"}
                            }
                            _need(baseline == enriched, "paired synthetic observations differ")
                            for arm in sequence:
                                observation = next(
                                    row
                                    for row in trials
                                    if row["position"] == position and row["profile"] == arm
                                )
                                try:
                                    _guard(supervisor_check)
                                    _need(
                                        time.monotonic() < deadline, "cooperative deadline exceeded"
                                    )
                                    module = modules[arm]
                                    module.post_json = lambda _url, _key, body, obs=observation: (
                                        _post(
                                            client, endpoint, token, body, deadline, obs, event_sink
                                        )
                                    )
                                    answer = module.choose(
                                        states[arm],
                                        "Select the link whose destination has the same origin as this page.",
                                        [],
                                    )
                                    observation.update(
                                        status="response_validated",
                                        operation=answer["operation"],
                                        selected_target=answer["target"],
                                        selected_action=answer["choice"],
                                        complete_answers=answer["raw_answers"],
                                    )
                                except Exception as exc:
                                    observation.update(
                                        status="failed", error_type=type(exc).__name__
                                    )
                                    if isinstance(exc, SupervisorGuardError):
                                        _emit(event_sink, "trial_finished", observation)
                                        raise
                                    if isinstance(exc, CaptureSinkError):
                                        raise
                                _emit(event_sink, "trial_finished", observation)
                            _need(not unexpected, "unexpected browser request blocked")
                    finally:
                        cleanup("httpx_client", client.close)
                finally:
                    cleanup("browser_context", context.close)
            finally:
                cleanup("browser", browser.close)
    except Exception as exc:
        fatal_error = type(exc).__name__
    finally:
        for module in modules.values():
            cleanup("upstream_http_client", module.CLIENT.close)
        _LOCK.release()
    return {
        "status": "synthetic_smoke_incomplete"
        if fatal_error
        or cleanup_errors
        or any(row["status"] != "response_validated" for row in trials)
        else "synthetic_smoke_pending_external_audit",
        "fatal_error_type": fatal_error,
        "cleanup_errors": cleanup_errors,
        "trials": trials,
        "source_observations": source_observations,
        "expected_trials": len(schedule),
        "clicks_executed": 0,
        "source_authenticity": "unverified",
        "model_identity": "unverified",
        "capture_completeness": "unverified",
        "response_bytes_scope": "httpx_decoded_content_not_raw_wire",
        "http_started_is_delivery_proof": False,
        "hard_deadline": "external_supervisor_required",
        "task_success": None,
        "blind_selection": "unverified",
    }
