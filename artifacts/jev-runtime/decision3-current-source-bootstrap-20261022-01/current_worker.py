#!/usr/bin/env python3
"""Explicit current-source Decision3 native/warmup/request helpers.

Forked source text, no historical module mutation and no actor/CLI entrypoint.
Native request, wire conversion and publisher text48 bodies remain unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import stat
import tempfile
import time
from pathlib import Path

ROOT = Path('/home/albert_hsieh/Gemma-4-12B-Unified-System-One')
SCRIPT = Path(__file__).resolve()
CORE_PATH = ROOT / "artifacts/jev-runtime/decision3-current-source-bootstrap-20261022-01/current_core.py"
CORE_SHA = "13b77299787d77587de2253a5f953b101113fc8d694edd27ee651d709be0e435"
WRAPPER_PATH = ROOT / "apps/modal/decision3_public_replay.py"
WRAPPER_SHA = "1cd90e5e98ad1c7c63985cdc5c9b7795021bf252b6ca1b5437bdc653d065f533"
WIRE_PATH = ROOT / "scripts/jev_decision3_browser_wire.py"
WIRE_SHA = "c048d7779e60fce5d1a56b38e26b9ddf5ed9e1ff9c6adc478786d0267cca3027"
SERVICE_PATH = Path('/home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/decision3-matched-v7-warm-readiness-20261018-01/source-candidate-01/decision3_browser_service.py')
INTERPRETER = "/opt/decision3/bin/python"
PROFILE = "decision3-gx10-current-source-rpc-v1"
WARM_PROFILE = "decision3-gx10-current-publisher-text48-v1"
WARM_LENGTHS = (37, 64, 320, 333, 1000, 1024)
MAX_BODY, MAX_FRAME, MAX_RESPONSE, MAX_REQUESTS = 1_000_000, 4 * 1024**2, 16 * 1024**2, 192
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
CORE = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def parse(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def finite(value):
        number = float(value)
        require(math.isfinite(number), "Nonfinite JSON number")
        return number

    return json.loads(
        blob,
        object_pairs_hook=pairs,
        parse_float=finite,
        parse_constant=lambda value: require(False, "Nonfinite JSON " + value),
    )


def ordinary(path, *, directory=False):
    path = Path(path).absolute()
    require(not any(entry.is_symlink() for entry in (path, *path.parents)), "Linked trusted path")
    mode = path.stat().st_mode
    require(stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode), "Ordinary path required")
    return path


def pin(path):
    path = ordinary(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before, hashed = os.fstat(stream.fileno()), hashlib.sha256()
        for block in iter(lambda: stream.read(1024**2), b""):
            hashed.update(block)
        after = os.fstat(stream.fileno())
    current = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(
        tuple(getattr(before, key) for key in fields)
        == tuple(getattr(after, key) for key in fields)
        == tuple(getattr(current, key) for key in fields),
        "Source changed while hashing",
    )
    return {"sha256": hashed.hexdigest(), "size_bytes": before.st_size}


def persist(path, value, *, terminal=False):
    """The pure request helper's persistence seam; production uses core bounds."""
    if CORE is not None:
        return CORE.write(path, value, terminal=terminal)
    # A source/import failure still needs a durable failed terminal receipt.
    path = Path(path)
    ordinary(path.parent, directory=True)
    blob = encode(value)
    require(len(blob) <= MAX_RESPONSE, "Single evidence JSON byte bound")
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def error_info(exc):
    result = {"type": type(exc).__name__, "message": str(exc)}
    if getattr(exc, "__notes__", None):
        result["notes"] = list(exc.__notes__)
    if exc.__cause__ is not None:
        result["cause"] = {"type": type(exc.__cause__).__name__, "message": str(exc.__cause__)}
    return result


def evidence_value(value):
    """Retain finite JSON exactly and label malformed, unencodable evidence."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"invalid_nonfinite_float": repr(value)}
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: evidence_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [evidence_value(item) for item in value]
    return {"invalid_native_type": type(value).__name__}


def same_uuid(left, right):
    def normalized(value):
        return str(value).strip().lower().removeprefix("gpu-")

    return bool(normalized(left)) and normalized(left) == normalized(right)


def startup_configuration(profile="cold", workload=None):
    if profile == "cold":
        require(workload is None, "Cold startup has no warm workload setting")
        return None
    require(profile == "publisher-text48-v1", "Unknown explicit startup profile")
    require(workload in {"public-six", "known-four"}, "Explicit warm workload required")
    return {
        "schema_id": "decision3-publisher-startup-v1",
        "profile": profile,
        "workload": workload,
        "max_browser_envelopes": 96 if workload == "public-six" else 64,
        "lengths": list(WARM_LENGTHS),
        "images": False,
        "native_batch_size": 8,
        "permutation_average": False,
    }


def validate_startup(value):
    if value is None:
        return None
    require(type(value) is dict, "External startup object required")
    expected = startup_configuration(value.get("profile"), value.get("workload"))
    require(
        expected is not None
        and json.dumps(value, sort_keys=True, allow_nan=False)
        == json.dumps(expected, sort_keys=True, allow_nan=False),
        "Exact typed external startup configuration required",
    )
    return expected


def publisher_warmup(
    runtime, torch, gates, output, state, gpu_uuid, source_pin, *, clock=time.monotonic
):
    """Audit the genuine pinned publisher method; synthetic work never enters RPC."""
    require(
        runtime.fast is None
        and not getattr(torch.version, "hip", None)
        and type(runtime.batch_size) is int
        and runtime.batch_size == 8
        and runtime.permutation_average is False,
        "Text48 requires native NVIDIA batch8 without graph capture/permutation",
    )
    for function in (runtime.warmup, runtime.probabilities):
        require(
            pin(function.__func__.__code__.co_filename) == source_pin,
            "Actual publisher warmup/probabilities source differs",
        )
    tokens = runtime.token_ids
    require(
        type(tokens) is list
        and len(tokens) == 255
        and all(type(token) is int and token >= 0 for token in tokens),
        "Actual255 publisher token IDs required",
    )
    path, started = Path(output) / "warmup.json", clock()
    report = {
        "profile": "publisher-text48-v1",
        "stage": "starting",
        "images": False,
        "lengths": list(WARM_LENGTHS),
        "publisher_source": source_pin,
        "publisher_warmup_invocations": 0,
        "attempted_probability_calls": 0,
        "entered_probability_calls": 0,
        "completed_probability_calls": 0,
        "validated_probability_calls": 0,
        "synthetic_sequence_rows": 0,
        "synthetic_input_tokens": 0,
        "records": [],
        "error": None,
        "secondary_errors": [],
        "peak_scope": "process cumulative including load/warmup",
        "counter_scope": "publisher API entry calls, not GPU kernel launches",
    }

    def checkpoint():
        report["elapsed_seconds"] = clock() - started
        state["warmup_calls"] = report["entered_probability_calls"]
        state["publisher_warmup_invocations"] = report["publisher_warmup_invocations"]
        state["warmup"] = {
            key: copy.deepcopy(value) for key, value in report.items() if key != "records"
        }
        state["warmup"]["journal_path"] = str(path)
        try:
            persist(path, report)
            state["warmup"]["journal_pin"] = pin(path)
        except BaseException:
            state["warmup_failure_evidence"] = evidence_value(report)
            raise

    original = runtime.probabilities
    owned = "probabilities" in runtime.__dict__
    previous = runtime.__dict__.get("probabilities")

    def audited(sequences, counts):
        index = report["attempted_probability_calls"]
        require(index < 48, "Publisher warmup exceeded48 calls")
        size, length = index // 6 + 1, WARM_LENGTHS[index % 6]
        expected = [
            [tokens[(i + j) % len(tokens)] for j in range(max(8, length - 7 * i))]
            for i in range(size)
        ]
        require(
            encode(sequences) == encode(expected) and encode(counts) == encode([2] * size),
            "Publisher synthetic schedule differs",
        )
        row = {
            "index": index,
            "sequences": copy.deepcopy(sequences),
            "counts": copy.deepcopy(counts),
            "stage": "before_gate",
            "raw_returned": False,
            "raw_probabilities": None,
            "secondary_errors": [],
        }
        report["records"].append(row)
        report["attempted_probability_calls"] += 1
        call_started, primary = clock(), None
        try:
            checkpoint()
            torch.cuda.synchronize(0)
            gate = gates.check(f"warmup-before-probabilities-{index:02d}")
            require(
                same_uuid(gate["snapshot"]["cuda"]["uuid"], gpu_uuid), "Warmup CUDA UUID differs"
            )
            row["resources_before"] = gate["snapshot"]
            row["stage"] = "publisher_entry"
            report["entered_probability_calls"] += 1
            report["synthetic_sequence_rows"] += size
            report["synthetic_input_tokens"] += sum(map(len, sequences))
            checkpoint()
            result = original(sequences, counts)
            row.update(raw_returned=True, raw_probabilities=evidence_value(result))
            report["completed_probability_calls"] += 1
            checkpoint()  # Retain raw values before validation or the final CUDA fence.
            require(
                type(result) is list
                and len(result) == size
                and all(
                    type(values) is list
                    and len(values) == 2
                    and all(
                        type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values
                    )
                    and abs(sum(values) - 1) < 0.02
                    for values in result
                ),
                "Malformed actual warmup probabilities",
            )
            report["validated_probability_calls"] += 1
            row["stage"] = "complete"
        except BaseException as exc:
            primary = exc
            row.update(stage="failed", error=error_info(exc))
        finally:
            try:
                torch.cuda.synchronize(0)
                row["cuda_synchronized_elapsed_seconds"] = clock() - call_started
                if CORE is not None:
                    row["resources_after"] = CORE.resource_snapshot(torch)
                checkpoint()
            except BaseException as exc:
                row["secondary_errors"].append(error_info(exc))
                if primary is None:
                    primary = exc
                else:
                    primary.add_note("Warmup fence/evidence also failed: " + repr(exc))
        if primary is not None:
            raise primary
        return result

    primary = None
    try:
        runtime.probabilities = audited
        report.update(stage="running", publisher_warmup_invocations=1)
        checkpoint()
        publisher_seconds = runtime.warmup(lengths=WARM_LENGTHS, images=False)
        report["publisher_reported_seconds"] = evidence_value(publisher_seconds)
        require(
            type(publisher_seconds) in (int, float)
            and math.isfinite(publisher_seconds)
            and publisher_seconds >= 0,
            "Invalid publisher warmup elapsed time",
        )
        require(
            all(
                report[key] == 48
                for key in (
                    "attempted_probability_calls",
                    "entered_probability_calls",
                    "completed_probability_calls",
                    "validated_probability_calls",
                )
            )
            and report["synthetic_sequence_rows"] == 216
            and report["synthetic_input_tokens"] == 96544,
            "Incomplete publisher text48 warmup",
        )
        report["stage"] = "complete"
    except BaseException as exc:
        primary = exc
        report.update(stage="failed", error=error_info(exc))
    finally:
        try:
            if owned:
                runtime.probabilities = previous
            else:
                del runtime.probabilities
        except BaseException as exc:
            report["secondary_errors"].append(error_info(exc))
            if primary is None:
                primary = exc
            else:
                primary.add_note("Publisher method restoration also failed: " + repr(exc))
        try:
            checkpoint()
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note("Final warmup evidence also failed: " + repr(exc))
    if primary is not None:
        raise primary
    return state["warmup"]


class DeadlineBackend:
    """Pass the outer timer through the legacy adapter's ordinary backend call."""

    def __init__(self, backend, check, counts=None):
        self.backend, self.check, self.counts = backend, check, counts

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def predict(self, request, *, deadline=None, evidence_sink=None):
        def check():
            self.check()
            if deadline is not None:
                deadline()

        def event(kind, evidence):
            # The raw sink runs before either post-forward deadline check.
            if evidence_sink is not None:
                evidence_sink(kind, evidence)
            check()

        check()
        if self.counts is not None:
            self.counts["backend_predict_entries"] += 1
        result = self.backend.predict(request, deadline=check, evidence_sink=event)
        check()
        return result


def process_request(
    envelope_bytes,
    *,
    adapter,
    wire,
    torch,
    gates,
    output,
    model,
    nonce,
    gpu_uuid,
    startup=None,
    started=None,
    clock=time.monotonic,
):
    """Run one unchanged native request, retaining raw output even after timeout.

    No framework import occurs here. Tests can supply fake dependencies and
    replace persist(). Every failure closes the campaign; late converted JSON
    remains evidence and is never offered as an HTTP response.
    """
    started = clock() if started is None else started
    remaining, primary = None, None
    output = Path(output)
    row = {
        "profile": PROFILE,
        "model": model,
        "nonce": nonce,
        "gpu_uuid": gpu_uuid,
        "request_id": None,
        "http_sha256": None,
        "stage": "parsing",
        "success": False,
        "campaign_stop": True,
        "raw_returned": False,
        "raw_response": None,
        "resources_before": None,
        "resources_after": None,
        "events": [],
        "secondary_errors": [],
        "helper_calls": 0,
        "optimizer_updates": 0,
        "task_score": None,
        "wire_body_utf8": None,
    }
    if startup is not None:
        row.update(profile=WARM_PROFILE, startup=validate_startup(startup))
    directory = output / "requests"
    path = directory / "invalid-request.json"
    sink_errors = []
    previous_sink = getattr(adapter, "evidence_sink", None)
    previous_backend = getattr(adapter, "backend", None)
    sink_installed = backend_installed = False
    native_restorations = []

    def checkpoint():
        row["journal_path"] = str(path)
        row["elapsed_seconds"] = clock() - started
        persist(path, row)

    def check():
        if sink_errors:
            raise RuntimeError("Request evidence persistence failed") from sink_errors[0]
        if remaining is not None and clock() - started >= remaining:
            raise TimeoutError("Full remaining browser request budget exceeded")

    def secondary(exc, stage):
        nonlocal primary
        row["secondary_errors"].append({"stage": stage, **error_info(exc)})
        if primary is None:
            primary = exc
        else:
            primary.add_note(stage + " also failed: " + repr(exc))

    def raw_from(evidence, *, overwrite=False):
        if type(evidence) is not dict:
            return
        for key in ("decision3_provider_evidence", "provider_failure_evidence"):
            if type(evidence.get(key)) is dict:
                raw_from(evidence[key], overwrite=True)
        if evidence.get("raw_response") is not None and (overwrite or not row["raw_returned"]):
            row.update(raw_returned=True, raw_response=evidence_value(evidence["raw_response"]))

    def event(evidence):
        try:
            require(type(evidence) is dict, "Native evidence object required")
            data = evidence.get("data")
            raw_from(data, overwrite=evidence.get("event") == "native_raw_returned")
            if evidence.get("event") == "backend_raw_returned" and not row["raw_returned"]:
                row.update(raw_returned=True, raw_response=evidence_value(data))
            row["events"].append(evidence_value(evidence))
            # Save exact returned values before any outer post-forward deadline.
            checkpoint()
        except BaseException as exc:
            sink_errors.append(exc)
            row["secondary_errors"].append({"stage": "native_evidence", **error_info(exc)})
        check()

    try:
        require(
            type(envelope_bytes) is bytes and len(envelope_bytes) <= MAX_FRAME, "RPC frame bound"
        )
        envelope = parse(envelope_bytes.decode("utf-8"))
        require(type(envelope) is dict, "RPC envelope object required")
        require(
            set(envelope) == {"request_id", "payload_json", "http_sha256", "remaining_seconds"},
            "Exact request envelope fields required",
        )
        request_id = envelope["request_id"]
        require(
            type(request_id) is str and SAFE_ID.fullmatch(request_id), "Safe request ID required"
        )
        row["request_id"] = request_id
        directory.mkdir(exist_ok=True)
        ordinary(directory, directory=True)
        candidate_path = directory / (request_id + ".json")
        require(not candidate_path.exists(), "Request ID already journaled")
        path = candidate_path
        require(len(list(directory.iterdir())) < MAX_REQUESTS, "At most192 requests per worker")
        payload_json = envelope["payload_json"]
        require(type(payload_json) is str, "Original HTTP payload string required")
        payload_bytes = payload_json.encode("utf-8")
        require(len(payload_bytes) <= MAX_BODY, "Original HTTP body byte bound")
        http_sha = hashlib.sha256(payload_bytes).hexdigest()
        require(envelope["http_sha256"] == http_sha, "Exact HTTP body SHA differs")
        remaining = envelope["remaining_seconds"]
        require(
            type(remaining) in (int, float) and math.isfinite(remaining) and 0 < remaining <= 20,
            "Remaining budget must be finite, positive and at most20 seconds",
        )
        row.update(
            http_sha256=http_sha,
            payload_json=payload_json,
            remaining_seconds=remaining,
            envelope_utf8=envelope_bytes.decode("utf-8"),
        )
        checkpoint()
        check()
        payload = parse(payload_bytes)
        require(
            type(payload) is dict and payload.get("model") == model, "Original model alias differs"
        )
        row["original_request"] = copy.deepcopy(payload)
        torch.cuda.synchronize(0)
        if CORE is not None:
            row["resources_before"] = CORE.resource_snapshot(torch)
        check()
        gates.check("browser-before-request-" + request_id)
        check()
        adapter.evidence_sink = event
        sink_installed = True
        if previous_backend is not None:
            counts = None
            if startup is not None:
                counts = {
                    "backend_predict_entries": 0,
                    "system_one_entries": 0,
                    "probabilities_entries": 0,
                    "sequence_rows": 0,
                    "input_tokens": 0,
                    "scope": "actual publisher API entries; not GPU kernel launch counts",
                }
                row["task_native_calls"] = counts

                def count_method(target, name, field):
                    original = getattr(target, name)
                    owned, previous = name in target.__dict__, target.__dict__.get(name)

                    def counted(*args, **kwargs):
                        counts[field] += 1
                        if name == "probabilities":
                            sequences = args[0] if args else kwargs["sequences"]
                            counts["sequence_rows"] += len(sequences)
                            counts["input_tokens"] += sum(map(len, sequences))
                        return original(*args, **kwargs)

                    setattr(target, name, counted)
                    native_restorations.append((target, name, owned, previous))

                count_method(previous_backend.model, "system_one", "system_one_entries")
                count_method(previous_backend.runtime, "probabilities", "probabilities_entries")
            adapter.backend = DeadlineBackend(previous_backend, check, counts)
            backend_installed = True
        row["stage"] = "native_predict"
        checkpoint()
        check()
        legacy = adapter.predict(payload)
        row["legacy_response"] = evidence_value(legacy)
        if type(legacy) is dict:
            raw_from(legacy.get("decision3_native"))
        checkpoint()
        check()
        converted = wire.named_browser_response(payload, legacy)
        row["wire_body_utf8"] = encode(converted).decode("utf-8")
        row["stage"] = "finalizing"
        check()
    except BaseException as exc:
        primary = exc
        for name in ("decision3_native", "decision3_provider_evidence"):
            if hasattr(exc, name):
                value = evidence_value(getattr(exc, name))
                row[name] = value
                raw_from(value, overwrite=name == "decision3_provider_evidence")
    finally:
        for target, name, owned, previous in reversed(native_restorations):
            try:
                if owned:
                    setattr(target, name, previous)
                else:
                    delattr(target, name)
            except BaseException as exc:
                secondary(exc, "restore_native_counter_method")
        if sink_installed:
            try:
                adapter.evidence_sink = previous_sink
            except BaseException as exc:
                secondary(exc, "restore_native_sink")
        if backend_installed:
            try:
                adapter.backend = previous_backend
            except BaseException as exc:
                secondary(exc, "restore_native_backend")
        try:
            torch.cuda.synchronize(0)
            row["cuda_synchronized_elapsed_seconds"] = clock() - started
            if CORE is not None:
                row["resources_after"] = CORE.resource_snapshot(torch)
        except BaseException as exc:
            secondary(exc, "final_cuda_fence_or_resource_observation")
        try:
            if remaining is not None:
                check()
        except BaseException as exc:
            secondary(exc, "final_request_deadline")
    if primary is None:
        try:
            row.update(stage="complete", success=True, campaign_stop=False)
            checkpoint()
            check()
            response = {
                "event": "response",
                "request_id": row["request_id"],
                "http_sha256": row["http_sha256"],
                "nonce": nonce,
                "model": model,
                "gpu_uuid": gpu_uuid,
                "success": True,
                "campaign_stop": False,
                "wire_body_utf8": row["wire_body_utf8"],
                "journal": row,
            }
            require(len(encode(response)) <= MAX_RESPONSE, "RPC response byte bound")
            check()
            return response
        except BaseException as exc:
            primary = exc
    row.update(stage="failed", success=False, campaign_stop=True, error=error_info(primary))
    try:
        # Evidence may retain a late converted body; the transport field is None.
        directory.mkdir(exist_ok=True)
        ordinary(directory, directory=True)
        checkpoint()
    except BaseException as exc:
        secondary(exc, "failed_request_persistence")
        row["error"] = error_info(primary)
    return {
        "event": "response",
        "request_id": row["request_id"],
        "http_sha256": row["http_sha256"],
        "nonce": nonce,
        "model": model,
        "gpu_uuid": gpu_uuid,
        "success": False,
        "campaign_stop": True,
        "wire_body_utf8": None,
        "journal": row,
    }


def load_file(name, path):
    expected = {CORE_PATH: CORE_SHA, WIRE_PATH: WIRE_SHA}
    require(path in expected, "Only the selected CORE/wire source may execute")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Ordinary selected source")
    require(path.is_file() and path.stat().st_size <= 1024**2, "Bounded selected source")
    blob = path.read_bytes()
    require(hashlib.sha256(blob).hexdigest() == expected[path], "Executed source SHA differs")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "File module spec required")
    module = importlib.util.module_from_spec(spec)
    exec(compile(blob, str(path), "exec", dont_inherit=True), module.__dict__)
    require(Path(module.__file__).resolve() == path, "Imported helper origin differs")
    return module


def authenticate_sources(manifest):
    global CORE
    require(pin(CORE_PATH)["sha256"] == CORE_SHA, "Reviewed replay core differs")
    if CORE is None:
        CORE = load_file("decision3_gx10_current_replay_core", CORE_PATH)
    identities = CORE.authenticate_source(CORE_SHA)
    expected = {
        **{ROOT / label: sha for label, sha in CORE.SOURCE_PINS.items()},
        CORE_PATH: CORE_SHA,
        WRAPPER_PATH: WRAPPER_SHA,
        WIRE_PATH: WIRE_SHA,
        SCRIPT: None,
        SERVICE_PATH: None,
    }
    require(
        type(manifest) is dict
        and manifest.get("schema_id") == "decision3-gx10-current-source-inventory-v1"
        and manifest.get("files") == 18
        and type(manifest.get("rows")) is list
        and len(manifest["rows"]) == len(expected) == 18,
        "Exact18 current source inventory required",
    )
    actual = {}
    for row in manifest["rows"]:
        require(
            type(row) is dict and set(row) == {"path", "pin", "classification"},
            "Exact source row fields required",
        )
        require(type(row["path"]) is str, "Absolute source path required")
        path = Path(row["path"])
        require(
            path.is_absolute() and path in expected and path not in actual, "Source path differs"
        )
        require(
            row["classification"]
            == (
                "selected_current_source13"
                if path in {ROOT / label for label in CORE.SOURCE_PINS}
                else "selected_current_composition_source"
            ),
            "Source classification differs",
        )
        actual[path] = pin(path)
        require(actual[path] == row["pin"], "Selected source pin differs: " + str(path))
        require(
            expected[path] is None or actual[path]["sha256"] == expected[path],
            "Fixed reviewed source SHA differs: " + str(path),
        )
    require(set(actual) == set(expected), "Source inventory path set differs")
    return identities


def matched_context_spec(spec):
    """Validated8192 admission derivative; original16384 source/math stay frozen."""
    require(type(spec.context_limit) is int and spec.context_limit == 16384,
            "Original Decision3 context authority changed")
    data = spec.model_dump()
    data["context_limit"] = 8192
    result = type(spec).model_validate(data)
    require(result.model_dump() == data and type(result.context_limit) is int
            and result.context_limit == 8192, "Matched validated8192 context required")
    return result

