"""Held single-arm browser actor for a declared same-model feedback pair.

This actor uses the pinned DONE-review bridge and private capture bootstrap to
invoke the original bundled runner. Selection, the other arm, process/GPU
supervision, original-judge provenance, numerical audit and OS closure remain
external responsibilities. It does not publish an accepted score or a
supervisor actor-complete record. The actor attempts to retain observed captures,
including failures and unmatched exchanges, in the private output directory.
Validation or write failures can leave capture completeness unconfirmed.

The supplied plan and its commitments are not proof of blind selection, case
exposure, source authenticity or actual runtime conformance. Released copies
must be independently reviewed and supervised before any real invocation.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import importlib.util
import io
import json
import os
import re
import stat
import sys
import threading
from pathlib import Path
from urllib.parse import urlsplit

SOURCE_ONLY = True
_LOCK = threading.Lock()
_MAX_SOURCE = 8 * 1024 * 1024
_MAX_CAPTURE = 16 * 1024 * 1024
_ENV = "JEV_FEEDBACK_CHILD_TOKEN"


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            _need(key not in out, "duplicate JSON key")
            out[key] = value
        return out

    def invalid(_value):
        raise ValueError("nonfinite JSON value")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _encode(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _no_secret(raw, secret):
    """Reject plain or JSON-escaped token material before private persistence."""
    _need(type(raw) is bytes, "capture bytes required")
    token = secret.encode("utf-8")
    escaped = json.dumps(secret, ensure_ascii=True)[1:-1].encode("ascii")
    _need(token not in raw and escaped not in raw, "private token in captured data")
    try:
        parsed = json.loads(raw, object_pairs_hook=lambda pairs: pairs)
    except (ValueError, UnicodeDecodeError):
        return

    def walk(value):
        if isinstance(value, str):
            _need(secret not in value, "private token in captured data")
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(key)
                walk(item)

    walk(parsed)


def _absolute(value):
    _need(type(value) is str, "absolute path required")
    path = Path(value)
    _need(
        path.is_absolute() and ".." not in path.parts and path.resolve() == path,
        "ordinary path required",
    )
    return path


def _read(path, expected, limit=_MAX_SOURCE):
    _need(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected), "SHA256 required")
    path = _absolute(str(path))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, "bounded file required")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        after = os.fstat(fd)
        current = os.stat(path, follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        _need(
            tuple(getattr(before, f) for f in fields)
            == tuple(getattr(after, f) for f in fields)
            == tuple(getattr(current, f) for f in fields)
            and len(raw) == before.st_size
            and _sha(raw) == expected,
            "source bytes or identity changed",
        )
        return raw
    finally:
        os.close(fd)


def _release(pair):
    _need(
        type(pair) is dict and set(pair) == {"held", "held_sha256", "released", "released_sha256"},
        "source pair required",
    )
    held = _read(pair["held"], pair["held_sha256"])
    released = _read(pair["released"], pair["released_sha256"])
    marker = b"SOURCE_ONLY" + b" = True"
    _need(
        held.count(marker) == 1
        and released == held.replace(marker, b"SOURCE_ONLY" + b" = False", 1),
        "gate-only release required",
    )
    return released


def _module(path, raw, name):
    _need(name not in sys.modules, "fresh module namespace required")
    spec = importlib.util.spec_from_file_location(name, path)
    _need(spec is not None and spec.loader is not None, "source loader required")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _private_dir(path):
    path = _absolute(path)
    info = path.stat()
    _need(
        path.is_dir()
        and info.st_uid == os.getuid()
        and info.st_mode & 0o077 == 0
        and not any(path.iterdir()),
        "fresh empty owned private directory required",
    )
    return path


def _write_new(path, raw):
    _need(type(raw) is bytes and len(raw) <= _MAX_CAPTURE, "bounded private bytes required")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _append(path, raw):
    _need(type(raw) is bytes and len(raw) <= 65536, "bounded private event required")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        _need(stat.S_ISREG(os.fstat(fd).st_mode), "ordinary journal required")
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(raw + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _pinned_sources(config, pins):
    sources = config["source_files"]
    _need(
        type(sources) is dict and set(sources) == set(pins),
        "complete plan source inventory required",
    )
    for name, record in sources.items():
        _need(
            type(record) is dict
            and set(record) == {"path", "sha256"}
            and record["sha256"] == pins[name],
            "source inventory differs from plan",
        )
        _read(record["path"], record["sha256"])


class _BoundedOutput(io.StringIO):
    """Refuse unbounded captured stdout before it consumes actor memory."""

    def write(self, text):
        _need(type(text) is str, "text output required")
        _need(self.tell() + len(text) <= _MAX_CAPTURE // 4, "runner output too large")
        return super().write(text)


def _local_endpoint(value, *, allow_path=False):
    _need(type(value) is str, "private endpoint required")
    url = urlsplit(value)
    _need(
        url.scheme == "http"
        and url.hostname == "127.0.0.1"
        and url.port
        and url.username is None
        and url.password is None
        and (
            url.path in ("", "/")
            or (
                allow_path
                and url.path.startswith("/")
                and all(part not in (".", "..") for part in url.path.split("/"))
            )
        )
        and not url.query
        and not url.fragment,
        "local HTTP endpoint required",
    )
    return value


def _case_token(secret, case_id):
    _need(type(secret) is bytes and 32 <= len(secret) <= 128, "private selector key required")
    _need(
        type(case_id) is str
        and 1 <= len(case_id) <= 128
        and not case_id.startswith("-")
        and all(c.isalnum() or c in "_-" for c in case_id),
        "bounded private case identifier required",
    )
    return hmac.new(
        secret, b"jev-feedback-case-v1\x00" + case_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()[:32]


def run_arm(
    plan_bytes,
    expected_plan_sha256,
    config,
    *,
    pair_index,
    arm_name,
    case_id,
    selector_key,
    endpoint,
    api_key,
    private_dir,
):
    """Run one original-bundle attempt; caller must independently supervise it.

    The caller privately supplies the selected case and candidate endpoint. This
    function verifies a keyed token match and declared settings, but cannot
    establish the selector's population, preregistration, or readiness identity.
    The other arm is a separate invocation with an independent private dir.
    """
    _need(SOURCE_ONLY is False, "held feedback actor requires independent review")
    _need(sys.dont_write_bytecode is True, "no-bytecode actor required")
    _need(
        type(plan_bytes) is bytes
        and _sha(plan_bytes) == expected_plan_sha256
        and len(plan_bytes) <= 1024 * 1024,
        "pinned bounded pair plan required",
    )
    plan = _json(plan_bytes)
    _need(
        type(pair_index) is int
        and 0 <= pair_index < plan["selected_cases"]
        and arm_name in ("off", "on"),
        "planned arm required",
    )
    pair = plan["schedule"][pair_index]
    _need(
        hmac.compare_digest(_case_token(selector_key, case_id), pair["token"]),
        "private selector token differs",
    )
    arm = plan["arms"][arm_name]
    _need(
        type(config) is dict and _sha(_encode(config)) == arm["configuration_sha256"],
        "pinned common execution configuration required",
    )
    required = {
        "schema",
        "source_files",
        "actor_source",
        "pair_verifier",
        "review_bridge",
        "capture_bootstrap",
        "owner",
        "capture",
        "bundle_dir",
        "bundle_manifest_sha256",
        "runner_sha256",
        "profile_helper_sha256",
        "upstream_path",
        "model",
        "chromium_executable",
        "observation_profile",
        "text_endpoint",
        "text_model",
        "max_reviews",
        "destination_attribution",
    }
    _need(
        set(config) == required and config["schema"] == "jev-feedback-original-actor-v1",
        "exact actor configuration required",
    )
    verifier = config["pair_verifier"]
    _need(
        type(verifier) is dict
        and set(verifier) == {"path", "sha256"}
        and _absolute(verifier["path"]).suffix == ".py",
        "pinned pair-plan verifier required",
    )
    verifier_raw = _read(verifier["path"], verifier["sha256"])
    _need(
        _absolute(config["actor_source"]["released"]) == Path(__file__).absolute(),
        "pinned released actor required",
    )
    _release(config["actor_source"])
    verifier_name = "_jev_feedback_pair_verifier"
    verifier_module = _module(verifier["path"], verifier_raw, verifier_name)
    try:
        _need(
            Path(verifier_module.__file__).absolute() == _absolute(verifier["path"])
            and sys.modules.get(verifier_name) is verifier_module
            and callable(getattr(verifier_module, "validate_plan", None)),
            "pinned pair-plan verifier identity required",
        )
        verifier_module.validate_plan(plan)
        _need(
            Path(verifier_module.__file__).absolute() == _absolute(verifier["path"]),
            "pair-plan verifier path changed",
        )
    finally:
        _need(
            sys.modules.get(verifier_name) is verifier_module,
            "pair-plan verifier identity changed",
        )
        del sys.modules[verifier_name]
    _need(
        type(config["destination_attribution"]) is bool
        and type(config["model"]) is str
        and config["model"]
        and type(config["max_reviews"]) is int
        and 1 <= config["max_reviews"] <= 4,
        "bounded model and review limit required",
    )
    _need(
        type(arm["budgets"]["actions"]) is int and 1 <= arm["budgets"]["actions"] <= 16,
        "bounded runner steps required",
    )
    _need(
        1 <= arm["budgets"]["policy_calls"] <= 16
        and 0 <= arm["budgets"]["text_calls"] <= 16
        and 1 <= arm["budgets"]["review_calls"] <= 4,
        "bounded call budget required",
    )
    _need(
        config["max_reviews"] <= arm["budgets"]["review_calls"],
        "review limit exceeds declared budget",
    )
    _need(type(api_key) is str and 32 <= len(api_key) <= 1024, "private child token required")
    endpoint = _local_endpoint(endpoint)
    _absolute(config["upstream_path"])
    _absolute(config["chromium_executable"])
    out = _private_dir(private_dir)
    _pinned_sources(config, arm["pins"])
    _need(
        config["source_files"]["review_owner"]["path"] == config["owner"]["held"]
        and config["source_files"]["review_owner"]["sha256"] == config["owner"]["held_sha256"],
        "review owner source pin required",
    )
    bridge_raw, bootstrap_raw = (
        _release(config["review_bridge"]),
        _release(config["capture_bootstrap"]),
    )
    _release(config["owner"])
    _release(config["capture"])
    # Refuse heldout bundles before the bootstrap has occasion to inspect them.
    manifest_path = _absolute(config["bundle_dir"]) / "bundle-manifest.json"
    manifest = _json(_read(manifest_path, config["bundle_manifest_sha256"], limit=1024 * 1024))
    _need(
        type(manifest) is dict
        and manifest.get("case_module") == "original"
        and type(manifest.get("files")) is dict
        and "jev_heldout_public_cases.py" not in manifest["files"],
        "original public-case bundle required",
    )
    _need(
        manifest["files"]["original_jev_browser_benchmark.py"]["sha256"] == arm["pins"]["runner"]
        and manifest["files"]["jev_public_web_cases.py"]["sha256"] == arm["pins"]["case_and_judge"],
        "pinned original runner and judge bundle required",
    )
    _need(_LOCK.acquire(blocking=False), "single actor ownership required")
    bridge = bootstrap = None
    old_key = os.environ.get(_ENV)
    primary = None
    captures = 0
    started = False
    stdout = None
    try:
        bridge = _module(
            config["review_bridge"]["released"], bridge_raw, "_jev_feedback_review_bridge"
        )
        bootstrap = _module(
            config["capture_bootstrap"]["released"],
            bootstrap_raw,
            "_jev_feedback_capture_bootstrap",
        )
        _need(
            bridge.SOURCE_ONLY is False
            and bootstrap.SOURCE_ONLY is False
            and callable(bridge.run)
            and callable(bootstrap.run),
            "released pinned bootstraps required",
        )
        # Existing bootstrap performs its own full pinned bundle and release checks.
        bootstrap._bundle(
            config["bundle_dir"],
            config["bundle_manifest_sha256"],
            config["runner_sha256"],
            config["profile_helper_sha256"],
        )
        _write_new(out / "capture-events.jsonl", b"")
        _write_new(out / "review-events.jsonl", b"")

        def capture(event):
            nonlocal captures
            _need(
                type(event) is dict
                and type(event.get("sequence")) is int
                and event["sequence"] == captures + 1,
                "ordered capture required",
            )
            expected = {
                "sequence",
                "status",
                "request_source",
                "request_bytes",
                "response_bytes",
                "http_status",
                "exception_type",
                "scope",
            }
            if config["destination_attribution"]:
                expected.add("destination_match")
                _need(
                    event.get("destination_match")
                    in {"policy_endpoint", "text_endpoint", "shared_endpoint", "unknown"},
                    "valid destination label required",
                )
            _need(set(event) == expected, "exact capture event required")
            n = event["sequence"]
            meta = {k: v for k, v in event.items() if k not in ("request_bytes", "response_bytes")}
            for part in ("request", "response"):
                raw = event.get(part + "_bytes")
                if raw is None:
                    meta[part + "_file"] = None
                    meta[part + "_sha256"] = None
                else:
                    _no_secret(raw, api_key)
                    filename = f"capture-{n:04d}-{part}.bin"
                    _write_new(out / filename, raw)
                    meta[part + "_file"] = filename
                    meta[part + "_sha256"] = _sha(raw)
            encoded = _encode(meta)
            _no_secret(encoded, api_key)
            _append(out / "capture-events.jsonl", encoded)
            captures = n
            _need(
                captures <= arm["budgets"]["policy_calls"] + arm["budgets"]["text_calls"],
                "captured call budget exceeded",
            )

        def review(event):
            encoded = _encode(event)
            _no_secret(encoded, api_key)
            _append(out / "review-events.jsonl", encoded)

        args = [
            "--upstream-path",
            config["upstream_path"],
            "--endpoint",
            endpoint,
            "--model",
            config["model"],
            "--output-dir",
            str(out / "runner"),
            "--repeats",
            "1",
            "--max-steps",
            str(arm["budgets"]["actions"]),
            "--seed",
            str(pair["seed"]),
            "--suite-split",
            "development",
            "--api-key-env",
            _ENV,
            "--chromium-executable",
            config["chromium_executable"],
            "--observation-profile",
            config["observation_profile"],
            "--public-cases",
            case_id,
        ]
        if config["profile_helper_sha256"] is not None:
            _need(
                type(config["profile_helper_sha256"]) is str
                and re.fullmatch(r"[0-9a-f]{64}", config["profile_helper_sha256"]),
                "profile helper pin required",
            )
            args += ["--observation-profile-helper-sha256", config["profile_helper_sha256"]]
        _need(
            type(config["text_model"]) is str and config["text_model"],
            "common text helper required",
        )
        args += [
            "--text-endpoint",
            _local_endpoint(config["text_endpoint"], allow_path=True),
            "--text-model",
            config["text_model"],
        ]
        os.environ[_ENV] = api_key
        started = True
        stdout = _BoundedOutput()
        with contextlib.redirect_stdout(stdout):
            bridge.run(
                bootstrap,
                enabled=True,
                feedback_enabled=arm["feedback_enabled"],
                max_reviews=config["max_reviews"],
                event_sink=review,
                held_owner_path=config["owner"]["held"],
                held_owner_sha256=config["owner"]["held_sha256"],
                released_owner_path=config["owner"]["released"],
                released_owner_sha256=config["owner"]["released_sha256"],
                bundle_dir=config["bundle_dir"],
                manifest_sha256=config["bundle_manifest_sha256"],
                runner_sha256=config["runner_sha256"],
                profile_helper_sha256=config["profile_helper_sha256"],
                held_capture_path=config["capture"]["held"],
                held_capture_sha256=config["capture"]["held_sha256"],
                released_capture_path=config["capture"]["released"],
                released_capture_sha256=config["capture"]["released_sha256"],
                sink=capture,
                arguments=args,
                **({"destination_attribution": True} if config["destination_attribution"] else {}),
            )
        return {
            "status": "runner_returned_pending_external_audit",
            "capture_count": captures,
            "browser_success": None,
            "source_authenticity": "unverified",
            "selector_and_blinding": "unverified",
            "os_closure": "unverified",
            "budget_enforcement": "external",
        }
    except BaseException as error:
        primary = error
        raise
    finally:
        problems = []
        if stdout is not None:
            try:
                output = stdout.getvalue().encode("utf-8")
                _no_secret(output, api_key)
                _write_new(out / "runner-stdout.txt", output)
            except BaseException:
                problems.append("private runner output was not preserved")
        if started and os.environ.get(_ENV) != api_key:
            problems.append("private token environment changed")
        if old_key is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = old_key
        for name, module in (
            ("_jev_feedback_review_bridge", bridge),
            ("_jev_feedback_capture_bootstrap", bootstrap),
        ):
            if module is not None:
                if sys.modules.get(name) is module:
                    del sys.modules[name]
                else:
                    problems.append("loaded module identity changed")
        try:
            _pinned_sources(config, arm["pins"])
            _release(config["review_bridge"])
            _release(config["capture_bootstrap"])
            _release(config["owner"])
            _release(config["capture"])
            _release(config["actor_source"])
            _read(config["pair_verifier"]["path"], config["pair_verifier"]["sha256"])
        except BaseException:
            problems.append("source pin changed during actor execution")
        _LOCK.release()
        if problems:
            if primary is None:
                raise RuntimeError("; ".join(problems))
            primary.add_note("actor integrity or cleanup unconfirmed")
