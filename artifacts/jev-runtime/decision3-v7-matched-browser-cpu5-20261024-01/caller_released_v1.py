"""Held thin caller for the original V7 known4+public6 matched comparison.

Only a selected owned worker may call one original case. No model/service,
resource guardian, alternative judge, retry loop or action repair is supplied.
Raw HTTP observation is private and passes each original CLIENT.post once.
Source authoring only: no checks or actual/data reads have been performed.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

SOURCE_ONLY = False
CAPABILITY_SHA256 = '4dbbb9470d011931854f6473c0840ef7f0746c4194c719cde6ab454c3e79d463'
V7_LAUNCHER_SHA256 = '31450f9002ad216c32b77751997eb27748c491d7bc757c4410294d3704fdfb9e'
ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
V7_LAUNCHER = ROOT / "scripts/jev_browser_profile_v7_benchmark.py"
PROFILE = "composed-visible-text-settled-v7"
PROFILE_HELPER_SHA256 = "d9f4236d1d46c9472c3d6c1551342639ecb870e6c0dd419246927645410ea868"
TEXT_MODEL = "generated-actions-nf4"
KNOWN4 = (
    "public-arxiv-title-filter-results-v1",
    "public-arxiv-original-paper-open-v1",
    "public-python-counter-most-common-v1",
    "public-nasa-mars-facts-moons-v1",
)
PUBLIC6 = tuple(f"heldout-public-{index:03d}" for index in range(6))
CASE_IDS = KNOWN4 + PUBLIC6


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def released():
    require(SOURCE_ONLY is False, "SOURCE_ONLY: matched caller is held before arguments/IO")
    require(
        pin(CAPABILITY_SHA256) and pin(V7_LAUNCHER_SHA256),
        "selected capability/launcher pins remain pending",
    )


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def reference(path):
    path = Path(path).absolute()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": sha(raw), "size_bytes": len(raw)}


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def retain(path, raw):
    path = Path(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)
    return {"path": str(path.absolute()), "sha256": sha(raw), "size_bytes": len(raw)}


def retain_json(path, value):
    raw = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode(
            "utf-8"
        )
        + b"\n"
    )
    return retain(path, raw)


def header_bytes(headers):
    # HTTPX raw pairs preserve duplicate headers and their byte spelling.
    return [
        [base64.b64encode(key).decode("ascii"), base64.b64encode(value).decode("ascii")]
        for key, value in headers.raw
    ]


class HTTPRecorder:
    """Pass-through observation; original post_json alone decides real retries.

    Ordinary observation failures are latched until the case boundary, so they
    do not manufacture transport exceptions and new upstream retries. An original
    transport BaseException is always rethrown unchanged. A successful response
    is returned unchanged even if ordinary retention failed; no completion may
    then be accepted. A fatal observation BaseException escapes immediately when
    no original transport error is already primary.
    """

    def __init__(self, output, *, clock=time.perf_counter):
        self.output, self.clock = Path(output), clock
        self.rows, self.failures = [], []
        self.output.mkdir(mode=0o700, exist_ok=False)
        sync_directory(self.output.parent)

    def _capture(self, row, response, primary):
        try:
            request = (
                response.request if response is not None else getattr(primary, "request", None)
            )
            require(request is not None, "actual serialized HTTP request bytes unavailable")
            request_raw = request.content
            require(type(request_raw) is bytes, "actual HTTP request.content bytes required")
            target = self.output / ("attempt-" + str(row["index"]).zfill(4))
            target.mkdir(mode=0o700, exist_ok=False)
            sync_directory(self.output)
            row["request"] = retain(target / "request.bin", request_raw)
            row["request_headers_base64"] = header_bytes(request.headers)
            row["request_sha256"] = sha(request_raw)
            if response is not None:
                response_raw = response.content
                require(type(response_raw) is bytes, "actual HTTP response.content bytes required")
                row["response"] = retain(target / "response.bin", response_raw)
                row["response_headers_base64"] = header_bytes(response.headers)
                row["response_sha256"] = sha(response_raw)
                row["http_status"] = response.status_code
                row["server_request_id"] = response.headers.get("X-Request-ID")
            else:
                row.update(
                    response=None,
                    response_headers_base64=None,
                    response_sha256=None,
                    http_status=None,
                    server_request_id=None,
                )
            row["capture_complete"] = True
            row["metadata_ref"] = retain_json(target / "metadata.json", row)
        except BaseException as error:
            row["capture_complete"] = False
            row["observation_error"] = {"type": type(error).__name__, "message": str(error)}
            self.failures.append(error)
            if primary is not None:
                primary.add_note("HTTP observation also failed: " + repr(error))
            elif not isinstance(error, Exception):
                raise

    def wrap(self, original_post):
        def observed(url, **kwargs):
            row = {
                "index": len(self.rows) + 1,
                "endpoint": str(url),
                "started_unix": time.time(),
                "capture_complete": False,
            }
            self.rows.append(row)
            started = self.clock()
            try:
                response = original_post(url, **kwargs)  # Exactly once, unchanged kwargs.
            except BaseException as error:
                row.update(
                    latency_ms=(self.clock() - started) * 1000,
                    transport_error={"type": type(error).__name__, "message": str(error)},
                )
                self._capture(row, None, error)
                raise
            else:
                row["latency_ms"] = (self.clock() - started) * 1000
                self._capture(row, response, None)
                return response

        return observed

    def finish(self):
        record = {
            "schema": "jev-v7-matched-client-http-attempts-v1",
            "private": True,
            "http_attempts": len(self.rows),
            "attempts": self.rows,
            "capture_complete": not self.failures
            and all(row["capture_complete"] for row in self.rows),
            "body_scope": "Exact HTTPX request.content/response.content, as in the original client observer; no packet/compression-wire-byte claim",
            "transport_behavior_changed": False,
        }
        try:
            result = retain_json(self.output / "http-attempts.json", record)
        except BaseException as error:
            if self.failures:
                primary = self.failures[0]
                if error is not primary:
                    primary.add_note("Final HTTP trace retention also failed: " + repr(error))
                raise primary from error
            raise
        if self.failures:
            raise self.failures[0]
        return result


@contextmanager
def observe_http(runner, recorder):
    original_loader = runner.load_upstream

    @contextmanager
    def observed_loader(*args, **kwargs):
        with original_loader(*args, **kwargs) as modules:
            _agent, model = modules
            original_post = model.CLIENT.post
            model.CLIENT.post = recorder.wrap(original_post)
            try:
                yield modules
            finally:
                model.CLIENT.post = original_post

    runner.load_upstream = observed_loader
    try:
        yield
    finally:
        runner.load_upstream = original_loader


def import_source(name, path, expected):
    path = Path(path)
    require(
        pin(expected) and not any(p.is_symlink() for p in (path, *path.parents)),
        "selected ordinary source required",
    )
    raw = path.read_bytes()
    require(
        len(raw) <= 1024**2 and sha(raw) == expected and name not in sys.modules,
        "fresh exact source bytes required",
    )
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "source loader unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@contextmanager
def bundle_imports(bundle, manifest):
    # The original V7 run_groups uses these same project roots; bundle helpers
    # must be fresh when original and heldout groups share an owned process.
    names = {Path(name).stem for name in manifest["files"] if name.endswith(".py")}
    saved = {name: sys.modules[name] for name in names if name in sys.modules}
    for name in names:
        sys.modules.pop(name, None)
    previous = list(sys.path)
    sys.path[:0] = [
        str(bundle),
        str(ROOT / "artifacts/jev-runtime/overlay"),
        str(ROOT / "src"),
        str(ROOT / "scripts"),
    ]
    try:
        yield
    finally:
        sys.path[:] = previous
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


@contextmanager
def helper_reasoning_disabled():
    # Exact environment selection used by original native_v2.run_groups.
    previous = os.environ.get("TEXT_MODEL_REASONING")
    os.environ["TEXT_MODEL_REASONING"] = "none"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TEXT_MODEL_REASONING", None)
        else:
            os.environ["TEXT_MODEL_REASONING"] = previous


def call_case(
    case_id,
    output,
    *,
    policy_endpoint,
    policy_model,
    text_endpoint,
    text_model=TEXT_MODEL,
    upstream,
    chromium,
    capability_sha256,
    verify_owned,
    policy_api_key=None,
    text_api_key=None,
):
    released()
    require(
        capability_sha256 == CAPABILITY_SHA256 and callable(verify_owned),
        "selected owned caller capability required",
    )
    require(case_id in CASE_IDS, "only the original known4+public6 ten cases are allowed")
    require(
        text_model == TEXT_MODEL and type(text_endpoint) is str and bool(text_endpoint.strip()),
        "the explicit shared generated-actions-nf4 helper is required",
    )
    require(
        all(
            type(value) is str and bool(value.strip()) for value in (policy_endpoint, policy_model)
        ),
        "explicit independent policy endpoint/model required",
    )
    verify_owned()  # Trusted worker authenticates current owner, cap bytes and separate NF4 helper.
    output = Path(output).absolute()
    require(
        not any(p.is_symlink() for p in (output, *output.parents)), "ordinary fresh output required"
    )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    retain(output / ".gitignore", b"*\n")
    mode, group = ("original", "known4") if case_id in KNOWN4 else ("heldout", "public6")
    record = {
        "schema": "jev-v7-matched-single-case-v1",
        "case_id": case_id,
        "group": group,
        "case_denominator": 1,
        "suite_denominator": 10,
        "infrastructure_complete": False,
        "task_success": None,
        "profile": PROFILE,
        "profile_helper_sha256": PROFILE_HELPER_SHA256,
        "policy_endpoint": policy_endpoint,
        "policy_model": policy_model,
        "text_endpoint": text_endpoint,
        "text_model": text_model,
        "text_model_reasoning": "none",
        "capability_sha256": capability_sha256,
        "helper_runtime_qualification_claimed": False,
        "seed": 20261014,
        "repeats": 1,
        "max_steps": 16,
        "action_profile": "standard",
    }
    v7 = runner = recorder = manifest = bundle = None
    primary = None
    module_suffix = case_id.replace("-", "_")
    v7_name, runner_name = (
        "_matched_v7_launcher_" + module_suffix,
        "_matched_v7_runner_" + module_suffix,
    )
    try:
        v7 = import_source(v7_name, V7_LAUNCHER, V7_LAUNCHER_SHA256)
        require(v7.PROFILE == PROFILE, "original V7 launcher profile differs")
        bundle, manifest = v7.prepare_bundle(output / "runtime-bundle", case_module=mode)
        require(
            manifest["profile"] == PROFILE
            and manifest["case_module"] == mode
            and manifest["profile_helper_sha256"] == PROFILE_HELPER_SHA256,
            "original selected observation helper/source differs",
        )
        v7.verify_bundle(bundle, manifest)
        recorder = HTTPRecorder(output / "http")
        with bundle_imports(bundle, manifest):
            runner = import_source(
                runner_name, bundle / "runner.py", manifest["files"]["runner.py"]["sha256"]
            )
            with observe_http(runner, recorder), helper_reasoning_disabled():
                result = runner.run_benchmark(
                    upstream_path=upstream,
                    endpoint=policy_endpoint,
                    model=policy_model,
                    output_dir=output / "results",
                    repeats=1,
                    max_steps=16,
                    seed=20261014,
                    suite_split="development",
                    api_key=policy_api_key,
                    text_endpoint=text_endpoint,
                    text_model=text_model,
                    text_api_key=text_api_key,
                    chromium_executable=chromium,
                    public_case_ids=[case_id],
                    observation_profile=PROFILE,
                    observation_profile_helper_sha256=PROFILE_HELPER_SHA256,
                )
        require(
            result["attempts"] == 1 and len(result["outcomes"]) == 1,
            "the original selected case must retain its complete attempt",
        )
        outcome = result["outcomes"][0]
        require(
            outcome["case_id"] == case_id + "-000"
            and outcome["task_id"] == case_id
            and type(outcome["success"]) is bool,
            "original case identity/judge outcome differs",
        )
        record["task_success"] = outcome["success"]  # False is a completed benchmark result.
        record["summary_ref"] = reference(output / "results/summary.json")
        require(
            not result.get("observation_profile_failed"), "original V7 observation integrity failed"
        )
    except BaseException as error:
        primary = error
    finally:
        for finish in (
            (lambda: record.update(http_trace_ref=recorder.finish()))
            if recorder is not None
            else None,
            (lambda: v7.verify_bundle(bundle, manifest)) if manifest is not None else None,
            verify_owned,
        ):
            if finish is None:
                continue
            try:
                finish()
            except BaseException as error:
                if primary is None:
                    primary = error
                elif primary is not error:
                    primary.add_note("Matched-case finalization also failed: " + repr(error))
        if manifest is not None:
            try:
                record["bundle_manifest_ref"] = reference(bundle / "bundle-manifest.json")
            except BaseException as error:
                if primary is None:
                    primary = error
                elif primary is not error:
                    primary.add_note("Bundle manifest reference also failed: " + repr(error))
        record["infrastructure_complete"] = primary is None
        record["error"] = (
            None if primary is None else {"type": type(primary).__name__, "message": str(primary)}
        )
        try:
            record["receipt_ref"] = retain_json(output / "caller-receipt.json", record)
        except BaseException as error:
            record["infrastructure_complete"] = False
            if primary is None:
                primary = error
            elif primary is not error:
                primary.add_note("Caller receipt retention also failed: " + repr(error))
        sys.modules.pop(runner_name, None)
        sys.modules.pop(v7_name, None)
    if primary is not None:
        primary.jev_v7_case_record = record
        raise primary
    return record


def aggregate_cases(records):
    require(
        type(records) is list
        and len(records) == 10
        and [row["case_id"] for row in records] == list(CASE_IDS),
        "exact original ten case records required; partial comparisons cannot report a score",
    )
    require(
        all(
            row.get("group") == ("known4" if row["case_id"] in KNOWN4 else "public6")
            and row.get("profile") == PROFILE
            and row.get("profile_helper_sha256") == PROFILE_HELPER_SHA256
            and row.get("seed") == 20261014
            and row.get("repeats") == 1
            and row.get("max_steps") == 16
            and row.get("text_model") == TEXT_MODEL
            and row.get("text_model_reasoning") == "none"
            and row.get("action_profile") == "standard"
            for row in records
        )
        and len({row["policy_model"] for row in records}) == 1
        and len({row["text_endpoint"] for row in records}) == 1,
        "matched profile/helper/policy and original group/settings required",
    )
    complete = all(row["infrastructure_complete"] is True for row in records)
    require(
        all(
            type(row["task_success"]) is bool
            for row in records
            if row["infrastructure_complete"] is True
        ),
        "completed cases require original judge results",
    )
    successes = sum(
        row["infrastructure_complete"] is True and row["task_success"] is True for row in records
    )
    return {
        "schema": "jev-v7-matched-exact10-aggregate-v1",
        "denominator": 10,
        "infrastructure_complete": complete,
        "successes": successes,
        "success_rate": successes / 10 if complete else None,
        "completed_cases": sum(row["infrastructure_complete"] is True for row in records),
        "failed_tasks": sum(
            row["infrastructure_complete"] is True and row["task_success"] is False
            for row in records
        ),
        "profile": PROFILE,
        "profile_helper_sha256": PROFILE_HELPER_SHA256,
        "seed": 20261014,
        "repeats": 1,
        "max_steps": 16,
    }


def main(_argv=None):
    released()
    raise RuntimeError(
        "An authenticated owned worker must call call_case; standalone launch is not authorized"
    )


if __name__ == "__main__":
    raise SystemExit(main())
