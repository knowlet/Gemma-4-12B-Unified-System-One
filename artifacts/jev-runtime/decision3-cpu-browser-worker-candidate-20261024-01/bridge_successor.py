"""Unselected explicit owned-binding successor to held b4cf00a bridge source.

No loader/server/guardian entry is provided. A process-local release closure
must revalidate the original owner admission/command/source/resource binding.
It is never derived from an HTTP request or environment bypass. The only worker
entry remains SOURCE_ONLY=True. This copied successor is unexecuted/unfrozen.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import secrets
import stat
import threading
import time
import re
from pathlib import Path


OWNED_WORKER_BINDING_VERSION = "owned-decision3-cpu-browser-binding-v1"
PROFILE = "decision3-cpu-native-http-two-controls-source-candidate-v1"
LEGACY_ABI = "jev_decision3_native_v1"
CONTEXT = 16384
BUDGET = 20
CALLER_HTTP_TIMEOUT = 25
MAX_BODY_BYTES = 1024 * 1024
OPERATIONS = {
    "CLICK", "TYPE_TEXT", "SELECT", "SCROLL_DOWN", "SCROLL_UP", "WAIT", "DONE", "BLOCKED"
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class _OwnedRelease:
    """A verified invariant proof plus live original-authority revalidation."""

    def __init__(self, check):
        _require(callable(check), "Explicit original owned-selection verifier required")
        self._check = check
        self.proof = copy.deepcopy(check())
        self._validate(self.proof)

    @staticmethod
    def _validate(proof):
        _require(type(proof) is dict and proof.get("schema") == OWNED_WORKER_BINDING_VERSION,
                 "Explicit owned-binding proof schema required")
        _require(proof.get("owner_unit") == "jev-d3-llama-cpp-cpu4-2g-coexist-20261024-07.service"
                 and proof.get("owner_profile") == "llama-cpp-cpu4-2g-coexist-command-owned-v1"
                 and proof.get("device") == "cpu" and proof.get("precision") == "bfloat16"
                 and proof.get("context_limit") == CONTEXT and proof.get("memory_max_bytes") == 2147483648,
                 "Exact original owner07/native03 CPU identity required")
        _require(proof.get("worker_pid") == os.getpid()
                 and type(proof.get("worker_start_ticks")) is int and proof["worker_start_ticks"] > 0
                 and type(proof.get("nonce")) is str and re.fullmatch(r"jev-d3-boundary-[0-9a-f]{32}", proof["nonce"])
                 and type(proof.get("invocation_id")) is str and re.fullmatch(r"[0-9a-f]{32}", proof["invocation_id"]),
                 "Actual process/admission/nonce proof required")
        for key in ("plan_sha256", "config_sha256", "selection_sha256"):
            _require(type(proof.get(key)) is str and re.fullmatch(r"[0-9a-f]{64}", proof[key]), "Exact original selection hashes required")
        _require(proof.get("native_model", {}).get("model_id") == "vllm-sr/d3-edge"
                 and proof["native_model"].get("revision") == "cab71e6ad3461b8086d51ddb572c656babd63b0b"
                 and proof["native_model"].get("model_sha256") == "6e00d04abbdbdf6d35c9e12065769d9edaacc3bfe9404583891797718d239dc8",
                 "Pinned original Edge model proof required")
        _json_bytes(proof)

    def verify(self):
        current = self._check()
        self._validate(current)
        _require(current == self.proof, "Original owned binding changed")
        return copy.deepcopy(current)


def bind_owned_selection(check):
    """No new guardian: check reuses the selected original owner/controller."""
    return _OwnedRelease(check)


def _release_gate(owned_release):
    _require(type(owned_release) is _OwnedRelease, "A process-local owned-selection release is required")
    return owned_release.verify()


def create_application(plan_path, *, owned_release, actor, runtime, torch, model,
                       model_identity, output, cpu_identity, backend_wrapper):
    proof = _release_gate(owned_release)
    _require(str(Path(plan_path).absolute()) == proof["plan_path"]
             and model == proof["native_model"] and model_identity == "d3-edge-cpu-toy",
             "Application plan/model/alias differs from original owned selection")
    _require(cpu_identity.get("nonce") == proof["nonce"]
             and cpu_identity.get("invocation_id") == proof["invocation_id"]
             and cpu_identity.get("pid") == proof["worker_pid"], "CPU HTTP provenance differs from owned worker")
    actor.validate_runtime(runtime, torch, model)
    dependencies = _draft_dependencies(owned_release=owned_release)
    spec = dependencies["spec_factory"]("d3-edge", device="cpu")
    backend = _BackendShim(spec=spec, name=model_identity, actor=actor, runtime=runtime,
                           torch=torch, model=model, native_payload=dependencies["native_payload"],
                           owned_release=owned_release)
    _require(callable(backend_wrapper), "Original-error/fatal backend guard required")
    backend = backend_wrapper(backend)
    from s1.errors import BackendTimeoutError, RequestValidationError

    return _Gateway(output=output, backend=backend, adapter_type=dependencies["adapter_type"],
                    named_browser_response=dependencies["named_browser_response"],
                    cpu_identity=cpu_identity, timeout_type=BackendTimeoutError,
                    validation_type=RequestValidationError, owned_release=owned_release)


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _duplicates(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON object key refused")
        result[key] = value
    return result


def _decode(body):
    def invalid_constant(value):
        raise ValueError("Nonfinite JSON refused: " + value)

    result = json.loads(body.decode("utf-8"), object_pairs_hook=_duplicates, parse_constant=invalid_constant)
    _json_bytes(result)  # Also rejects invalid UTF-8 scalar strings; no repair.
    _require(type(result) is dict, "JSON object required")
    return result


def _toy_scope(payload):
    """Reject an out-of-scope WHOLE request; never select/filter/reorder values.

    The pinned choose() builds operation from observed targets + controls, then
    DONE/BLOCKED. The two-button page has no fill/select targets. All original
    offered operation values remain intact, including TYPE_TEXT/SELECT should
    an unusual caller offer them without a corresponding target head. A legal
    wrong choice, DONE, or BLOCKED is never corrected by this bridge.
    """
    questions = payload.get("questions")
    _require(type(questions) is dict and "operation" in questions, "Named operation head required")
    _require(set(questions) <= {"operation", "click_target"}, "Toy only: freeform/text/select/other heads unsupported")
    for question in questions.values():
        _require(type(question) is dict and question.get("type") == "choice", "Toy only: choice heads required")
    offered = questions["operation"].get("criteria")
    _require(type(offered) is dict and offered and set(offered) <= OPERATIONS, "Original standard operation choices required")
    _require({"DONE", "BLOCKED"} <= set(offered), "Pinned caller DONE/BLOCKED choices must remain offered")
    if "click_target" in questions:
        candidates = questions["click_target"].get("criteria")
        _require(type(candidates) is dict and 1 <= len(candidates) <= 2, "Toy requires one or two intact click candidates")
    _require(not payload.get("media"), "Toy only: text observation, no media")


def _cpu_adapter_type(base_adapter, configuration_blockers, *, owned_release):
    class CPUAdapter(base_adapter):
        # Inherit predict verbatim: all native parse/DTO/normalize, all original
        # ordered heads, cooperative20s, raw/error audit and singleton execution.
        def __init__(self, backend, *, model_identity=None, evidence_sink=None):
            _release_gate(owned_release)
            spec = getattr(backend, "spec", None)
            _require(spec is not None and spec.adapter == "decision3", "Explicit pinned Decision3 spec required")
            _require(spec.device == "cpu", "Honest CPU device required; CUDA spoofing refused")
            _require(spec.precision == "bfloat16", "CPU BF16 is required independently of the combined CUDA blocker")
            _require(spec.context_limit == CONTEXT, "CPU rejection-only context16384 required")
            blockers = [reason for reason in configuration_blockers(spec) if reason != "decision3_requires_cuda_bfloat16"]
            _require(not blockers, "Non-device native blockers retained: " + ", ".join(blockers))
            self.backend = backend
            self.identity = backend.name if model_identity is None else model_identity
            _require(isinstance(self.identity, str) and bool(self.identity.strip()), "Explicit model alias required")
            self.evidence_sink = evidence_sink

    return CPUAdapter


class _RowWitness:
    """Actual first-layer entry observer, including genuine 1..4 row tails.

    A hook entry proves entry/shape, not successful completion of that forward.
    The unmodified backend raw-return event separately proves native completion.
    No tensors are read for values, changed, padded, duplicated, or returned.
    """

    def __init__(self, torch, ids, lengths, emit, clock=time.monotonic):
        self.torch, self.emit, self.clock = torch, emit, clock
        self.expected = [
            {"question_ids": ids[start:start + 4],
             "shape": [len(ids[start:start + 4]), max(lengths[key] for key in ids[start:start + 4]), 1024]}
            for start in range(0, len(ids), 4)
        ]
        self.events = []

    def __call__(self, _module, args):
        _require(len(self.events) < len(self.expected) and bool(args), "Unexpected additional decoder entry")
        hidden = args[0]
        expected = self.expected[len(self.events)]
        event = {
            "question_ids": list(expected["question_ids"]),
            "shape": list(hidden.shape), "dtype": str(hidden.dtype),
            "device": hidden.device.type, "observed_monotonic": self.clock(),
            "inputs_or_outputs_modified": False, "scope": "first_layer_entered_not_forward_completed",
        }
        self.events.append(event)
        self.emit("actual_decoder_entered", event)
        _require(event["shape"] == expected["shape"] and 1 <= event["shape"][0] <= 4, "Actual native batch/tail shape differs")
        _require(hidden.dtype == self.torch.bfloat16 and hidden.device.type == "cpu" and not self.torch.cuda.is_initialized(), "CPU BF16 with no initialized CUDA required")
        return None

    def completed(self):
        _require(len(self.events) == len(self.expected), "Missing genuine decoder entries/rows")
        return {
            "observed_decoder_entries": len(self.events),
            "observed_question_rows": sum(event["shape"][0] for event in self.events),
            "actual_shapes": [event["shape"] for event in self.events],
            "question_order": [key for event in self.events for key in event["question_ids"]],
            "native_raw_returned_and_validated": True,
            "dummy_questions_added": 0, "helper_calls": 0,
        }


class _BackendShim:
    """Protocol adapter over an ALREADY loaded, selected frozen CPU backend.

    native_payload makes the ordinary ordered DTO conversion. The original
    NativeBatch4Backend performs runtime/preparation/raw validation. This shim
    returns its raw unchanged; inherited predict normalizes exactly once.
    """

    def __init__(self, *, spec, name, actor, runtime, torch, model, native_payload, owned_release):
        _release_gate(owned_release)
        self.spec, self.name = spec, name
        self.actor, self.runtime, self.torch, self.model = actor, runtime, torch, model
        self.native_payload = native_payload

    def predict(self, request, *, deadline, evidence_sink):
        payload = self.native_payload(request)
        ids = list(payload["questions"])
        hook = witness = None
        trace = {"question_order": ids, "events": [], "raw_response": None, "helper_calls": 0}
        primary = None

        def emit(event, data):
            trace["events"].append({"event": event, "evidence": copy.deepcopy(data)})
            evidence_sink(event, data)
            deadline()

        def native_event(event, evidence):
            nonlocal witness, hook
            if event == "prepared":
                _require(witness is None and evidence["original_question_order"] == ids, "Original preparation order differs")
                lengths = evidence["lengths"]
                _require(set(lengths) == set(ids), "Every original question length required")
                witness = _RowWitness(self.torch, ids, lengths, emit)
                hook = self.runtime.backbone.language_model.layers[0].register_forward_pre_hook(witness)
            if event == "raw_returned_before_validation":
                trace["raw_response"] = copy.deepcopy(evidence)
            emit(event, evidence)

        try:
            deadline()
            backend = self.actor.NativeBatch4Backend(
                self.runtime, self.torch, self.model, context_limit=CONTEXT, evidence_sink=native_event
            )
            raw = backend(**payload)
            _require(witness is not None, "No original prepared event/forward witness")
            emit("native_rows_completed", witness.completed())
            deadline()
            return raw
        except BaseException as exc:
            primary = exc
            exc.decision3_provider_evidence = trace
            raise
        finally:
            if hook is not None:
                try:
                    hook.remove()
                except BaseException as cleanup:
                    if primary is not None:
                        primary.add_note("Decoder witness cleanup also failed: " + repr(cleanup))
                    else:
                        cleanup.decision3_provider_evidence = trace
                        raise


def _ordinary_directory(path):
    _require(path.is_absolute(), "Absolute owned output required")
    for parent in (path, *path.parents):
        mode = parent.lstat().st_mode
        _require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), "Owned output and every ancestor must be ordinary directories")


def _fsync_directory(path):
    directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class _Record:
    """Generated paths only; immutable events and HTTP-only atomic final files."""

    def __init__(self, output, request_id):
        _ordinary_directory(output)
        self.path = output / request_id
        self.path.mkdir(mode=0o700, exist_ok=False)
        _fsync_directory(output)
        self.event_count = 0
        self.errors = []
        self.fatal_error = None

    def write(self, name, data, *, replace=False):
        allowed = name in {"request.bin", "request.json", "candidate.json", "response.bin", "exchange.json", "delivery.json"}
        _require(allowed or (name.startswith("event-") and name.endswith(".json") and name[6:-5].isdigit()), "Fixed owned evidence filename required")
        _require(not replace or name in {"response.bin", "exchange.json"}, "Only final HTTP records may be atomically replaced")
        _ordinary_directory(self.path)
        target = self.path / name
        temporary = self.path / (".pending-" + secrets.token_hex(12))
        primary = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            fd = os.open(temporary if replace else target, flags, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if replace:
                os.replace(temporary, target)
            _fsync_directory(self.path)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            try:
                if temporary.exists():
                    temporary.unlink()
            except BaseException as cleanup:
                if primary is None:
                    raise
                primary.add_note("Evidence temporary cleanup also failed: " + repr(cleanup))

    def event(self, evidence):
        self.event_count += 1
        try:
            self.write(f"event-{self.event_count:06d}.json", _json_bytes(evidence))
        except BaseException as exc:
            # The original adapter also isolates observer failures. This ledger
            # makes missing evidence fatal to 200 authorization, not to primary.
            self.errors.append({"type": type(exc).__name__, "message": str(exc)})
            if not isinstance(exc, Exception):
                if self.fatal_error is None:
                    self.fatal_error = exc
                raise

    def raise_fatal(self):
        # Original adapter observers isolate BaseException. Retain the actual
        # abort object and rethrow at the next gateway boundary; this does not
        # guarantee immediate interruption inside the unmodified native call.
        if self.fatal_error is not None:
            raise self.fatal_error


class _DeadlineExceeded(TimeoutError):
    pass


def _new_output(path, *, owned_release):
    _release_gate(owned_release)
    path = Path(path).absolute()
    _ordinary_directory(path.parent)
    path.mkdir(mode=0o700, exist_ok=False)
    fd = os.open(path / ".gitignore", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(b"*\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path)
    _fsync_directory(path.parent)
    return path


class _Gateway:
    """Private UNSELECTED draft ASGI callable; no server/service entry provided.

    The future guarded factory must supply a fresh, ordinary, ignored output,
    authenticated CPU identity, original adapter/wire dependencies and one owned
    runtime. No API key, Authorization header, environment, or source crawl is
    recorded. Busy requests do no inference. All model calls remain serialized.
    """

    def __init__(self, *, output, backend, adapter_type, named_browser_response,
                 cpu_identity, timeout_type, validation_type, owned_release, clock=time.monotonic):
        _release_gate(owned_release)
        _ordinary_directory(output)
        _require((output / ".gitignore").read_bytes() == b"*\n", "Owned output must be ignored before any request")
        _require(cpu_identity.get("device") == "cpu" and cpu_identity.get("precision") == "bfloat16" and cpu_identity.get("context_limit") == CONTEXT, "Separate honest CPU identity required")
        self.output, self.backend, self.adapter_type = output, backend, adapter_type
        self.named_browser_response = named_browser_response
        self.cpu_identity = copy.deepcopy(cpu_identity)
        self.timeout_type, self.validation_type, self.clock = timeout_type, validation_type, clock
        self.serial = threading.Lock()

    async def __call__(self, scope, receive, send):
        began = self.clock()
        request_id = secrets.token_hex(16)
        status, primary, candidate, record = 500, None, None, None
        lock_owned = False
        body = b""
        body_complete = False
        request_recorded = False
        candidate_retained = False

        def deadline():
            if record is not None:
                record.raise_fatal()
            if self.clock() - began >= BUDGET:
                raise _DeadlineExceeded("CPU HTTP cooperative20s expired; no hard cancellation/delivery guarantee")

        def error_body(exc):
            return {"error": {"type": type(exc).__name__, "message": str(exc)},
                    "request_id": request_id, "cpu_profile": PROFILE}

        try:
            record = _Record(self.output, request_id)
            _require(scope.get("type") == "http", "HTTP only; no lifespan/service entry supplied")
            if scope.get("method") != "POST" or scope.get("path") != "/v1/systemone":
                raise ValueError("Only POST /v1/systemone is supported; no helper/freeform endpoint")
            lock_owned = self.serial.acquire(blocking=False)
            if not lock_owned:
                status = 503
                raise RuntimeError("Owned CPU native runtime busy; no inference performed")
            while True:
                deadline()
                message = await receive()
                deadline()
                _require(message.get("type") == "http.request", "Request disconnected/unsupported ASGI event")
                body += message.get("body", b"")
                _require(len(body) <= MAX_BODY_BYTES, "Whole request exceeds toy body limit; none truncated")
                if not message.get("more_body", False):
                    body_complete = True
                    break
            record.write("request.bin", body)
            request_recorded = True
            payload = _decode(body)
            canonical = _json_bytes(payload)
            record.write("request.json", _json_bytes({
                "request_id": request_id, "body_sha256": _sha(body),
                "request_body_complete": body_complete,
                "canonical_request_sha256": _sha(canonical), "request": payload,
                "question_order": list(payload.get("questions", {})),
                "cpu_identity": self.cpu_identity, "legacy_v1_scope": "wire ABI only; no CUDA qualification",
                "helper_calls": 0, "external_caller_timeout_seconds": CALLER_HTTP_TIMEOUT,
            }))
            _toy_scope(payload)
            deadline()
            adapter = self.adapter_type(self.backend, evidence_sink=record.event)
            deadline()  # Owned-selection revalidation is part of the outer20s.
            result = adapter.predict(payload)
            deadline()
            candidate = self.named_browser_response(payload, result)
            candidate["decision3_cpu_http"] = {
                "profile": PROFILE, "cpu_identity": self.cpu_identity,
                "legacy_version_scope": "v1 DTO/response ABI only, not CUDA eligibility",
                "request_id": request_id, "body_sha256": _sha(body),
                "canonical_request_sha256": _sha(canonical), "helper_calls": 0,
                "candidates_or_probabilities_changed": False,
            }
            record.write("candidate.json", _json_bytes(candidate))
            candidate_retained = True
            if record.errors:
                raise RuntimeError("Native observer evidence incomplete; successful HTTP refused")
            deadline()
            status = 200
        except BaseException as exc:
            primary = record.fatal_error if record is not None and record.fatal_error is not None else exc
            if primary is not exc:
                primary.add_note("Failure after observer abort: " + repr(exc))
            if record is not None and not request_recorded:
                try:
                    record.write("request.bin", body)
                except BaseException as secondary:
                    primary.add_note("Partial raw request preservation also failed: " + repr(secondary))
            if status != 503:
                status = 504 if isinstance(primary, (_DeadlineExceeded, self.timeout_type)) else (422 if isinstance(primary, (ValueError, self.validation_type)) else 500)

        try:
            response_body = _json_bytes(candidate if status == 200 else error_body(primary))
            def prepare_terminal():
                # This sample precedes terminal record writes. The final gate
                # BELOW includes every write/fsync and header construction.
                exchange = {
                    "request_id": request_id, "status": status,
                    "response_sha256": _sha(response_body),
                    "body_sha256": _sha(body) if body_complete else None,
                    "captured_body_sha256": _sha(body), "request_body_complete": body_complete,
                    "elapsed_ms_sampled_before_terminal_write": (self.clock() - began) * 1000,
                    "timing_scope": "ASGI entry through publication and final200authorization; excludes delivery after response.start",
                    "candidate_constructed": candidate is not None,
                    "candidate_retained": candidate_retained, "observer_errors": record.errors,
                    "primary_error": None if primary is None else {"type": type(primary).__name__, "message": str(primary)},
                    "helper_calls": 0, "network_delivery_guaranteed": False,
                }
                record.write("response.bin", response_body, replace=True)
                record.write("exchange.json", _json_bytes(exchange), replace=True)
                return {"type": "http.response.start", "status": status, "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(response_body)).encode("ascii")),
                    (b"x-request-id", request_id.encode("ascii")),
                ]}

            _require(record is not None, "Could not create owned evidence record")
            start_message = prepare_terminal()
            if status == 200:
                try:
                    deadline()  # Last authorization immediately before ASGI start.
                except _DeadlineExceeded as exc:
                    primary, status = exc, 504
                    response_body = _json_bytes(error_body(exc))
                    start_message = prepare_terminal()  # Candidate stays immutable.
            await send(start_message)
            await send({"type": "http.response.body", "body": response_body})
        except BaseException as secondary:
            # Never substitute an evidence/send failure for an existing primary.
            if primary is not None:
                primary.add_note("HTTP publication/delivery also failed: " + repr(secondary))
            else:
                primary = secondary
            raise primary
        finally:
            if lock_owned:
                self.serial.release()
        if primary is not None and not isinstance(primary, Exception):
            raise primary  # Resource abort/interrupt must escape the owned actor.


def _draft_dependencies(*, owned_release):
    """Called only after the selected worker's ONE original native load."""
    _release_gate(owned_release)
    from s1.evaluation.decision3 import configuration_blockers, decision3_spec, native_payload
    from s1.jev_decision3_native import JevDecision3NativeAdapter
    from scripts.jev_decision3_browser_wire import named_browser_response

    return {
        "adapter_type": _cpu_adapter_type(JevDecision3NativeAdapter, configuration_blockers, owned_release=owned_release),
        "spec_factory": decision3_spec, "native_payload": native_payload,
        "named_browser_response": named_browser_response,
    }
