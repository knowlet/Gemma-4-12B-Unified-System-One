#!/usr/bin/env python3
"""Fresh-process, offline Decision3 browser worker with ordered local JSONL RPC.

Only the worker entrypoint imports the framework. The unchanged native adapter
receives each original browser request, and the reviewed wire helper changes
only its answer container. This worker owns no HTTP listener or browser.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import stat
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
CORE_PATH = ROOT / "scripts/jev_decision3_public_dev22_gpu_replay.py"
CORE_SHA = "9919e3b1dd5d5a68377b8e74c1b313c9a368a1716400d32af72e8a548ef02b4c"
WRAPPER_PATH = ROOT / "apps/modal/decision3_public_replay.py"
WRAPPER_SHA = "1cd90e5e98ad1c7c63985cdc5c9b7795021bf252b6ca1b5437bdc653d065f533"
WIRE_PATH = ROOT / "scripts/jev_decision3_browser_wire.py"
WIRE_SHA = "c048d7779e60fce5d1a56b38e26b9ddf5ed9e1ff9c6adc478786d0267cca3027"
SERVICE_PATH = ROOT / "apps/modal/decision3_browser_service.py"
ACTION_PROFILE_PATH = ROOT / "src/s1/jev_action_profiles.py"
ACTION_PROFILES = ("standard", "jev-find-text-experimental-v1")
DEFAULT_ACTION_PROFILE = "standard"
INTERPRETER = "/opt/decision3/bin/python"
PROFILE = "decision3-browser-local-rpc-v1"
WARM_PROFILE = "decision3-browser-publisher-text48-v1"
WARM_LENGTHS = (37, 64, 320, 333, 1000, 1024)
MAX_BODY, MAX_FRAME, MAX_RESPONSE, MAX_REQUESTS = 1_000_000, 4 * 1024**2, 16 * 1024**2, 192
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
CORE = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected_action_profile(config, override=None):
    """Missing legacy configuration keeps the original eight-operation path."""
    require(type(config) is dict, "Worker config object required")
    profile = config.get("action_profile", DEFAULT_ACTION_PROFILE)
    require(
        type(profile) is str and profile in ACTION_PROFILES,
        "Unknown explicit worker action profile",
    )
    if override is not None:
        require(
            type(override) is str and override in ACTION_PROFILES,
            "Unknown explicit worker action profile override",
        )
        require(
            "action_profile" not in config or profile == override,
            "Worker CLI/config action profiles differ",
        )
        profile = override
    require(
        profile != DEFAULT_ACTION_PROFILE or "action_profile_source" not in config,
        "Standard worker does not select an experimental action-profile source",
    )
    return profile


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


def authenticate_action_profile(config, provenance=None):
    """Authenticate the opt-in selector and its seven declared small sources.

    The original exact17 inventory remains unchanged. The explicit selector pin
    is additional configuration; its byte-pinned factory supplies the dependency
    projection, checked before Torch and again during final authentication.
    """
    if selected_action_profile(config) == DEFAULT_ACTION_PROFILE:
        return None
    row = config.get("action_profile_source")
    require(
        type(row) is dict
        and set(row) == {"path", "pin"}
        and row["path"] == str(ACTION_PROFILE_PATH),
        "Exact experimental action-profile selector source required",
    )
    expected = row["pin"]
    require(
        type(expected) is dict
        and set(expected) == {"sha256", "size_bytes"}
        and type(expected["sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", expected["sha256"])
        and type(expected["size_bytes"]) is int
        and 0 < expected["size_bytes"] <= 200_000,
        "Bounded experimental action-profile selector pin required",
    )
    require(
        ordinary(ACTION_PROFILE_PATH).stat().st_size == expected["size_bytes"],
        "Experimental action-profile selector size differs",
    )
    require(pin(ACTION_PROFILE_PATH) == expected, "Experimental action-profile selector changed")
    if provenance is not None:
        dependencies = provenance.get("source_dependencies")
        require(
            type(dependencies) is list and len(dependencies) == 7,
            "Exact seven experimental action-profile dependencies required",
        )
        seen = set()
        for dependency in dependencies:
            require(
                type(dependency) is dict
                and set(dependency) == {"path", "sha256"}
                and type(dependency["path"]) is str,
                "Exact experimental action-profile dependency row required",
            )
            label = dependency["path"]
            relative = Path(label)
            require(
                relative.parts
                and not relative.is_absolute()
                and ".." not in relative.parts
                and relative.as_posix() == label
                and label.endswith(".py")
                and label not in seen
                and type(dependency["sha256"]) is str
                and re.fullmatch(r"[0-9a-f]{64}", dependency["sha256"]),
                "Unique repository-relative action-profile source required",
            )
            seen.add(label)
            path = ordinary(ROOT / relative)
            require(0 < path.stat().st_size <= 200_000, "Bounded action-profile source required")
            require(
                pin(path)["sha256"] == dependency["sha256"],
                "Experimental action-profile dependency changed: " + label,
            )
    return copy.deepcopy(row)


class ActionProfileWire:
    """Keep the selected native adapter and checked wire on the same profile."""

    def __init__(self, profile, module):
        self.profile, self.module = profile, module

    def named_browser_response(self, request, response):
        return self.module.named_browser_response(self.profile, request, response)


def resolve_action_profile(config, state):
    profile = selected_action_profile(config)
    if profile == DEFAULT_ACTION_PROFILE:
        return None, None
    source = authenticate_action_profile(config)
    module = importlib.import_module("s1.jev_action_profiles")
    require(
        Path(module.__file__).resolve() == ACTION_PROFILE_PATH
        and module.ACTION_PROFILES == ACTION_PROFILES
        and module.DEFAULT_ACTION_PROFILE == DEFAULT_ACTION_PROFILE,
        "Imported action-profile selector origin/API differs",
    )
    protocol = module.resolve_protocol(profile)
    provenance = protocol.provenance
    require(
        isinstance(protocol.native_adapter_class, type)
        and type(provenance) is dict
        and provenance.get("action_profile") == profile
        and provenance.get("operation_count") == 9
        and provenance.get("implementation") == "checked_private_derivatives"
        and provenance.get("native_profile") == "jev_decision3_native_find_text_experimental_v1"
        and provenance.get("wire_profile") == "jev_decision3_browser_find_text_experimental_v1",
        "Selected experimental native/wire protocol differs",
    )
    authenticate_action_profile(config, provenance)
    state.update(
        action_profile=profile,
        action_profile_source=source,
        action_profile_provenance=copy.deepcopy(provenance),
    )
    return protocol.native_adapter_class, ActionProfileWire(profile, module)


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
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "File module spec required")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == path, "Imported helper origin differs")
    return module


def authenticate_sources(manifest):
    global CORE
    require(pin(CORE_PATH)["sha256"] == CORE_SHA, "Reviewed replay core differs")
    if CORE is None:
        CORE = load_file("jev_decision3_browser_replay_core", CORE_PATH)
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
        and manifest.get("schema_id") == "decision3-browser-source-inventory-v1"
        and manifest.get("files") == 17
        and type(manifest.get("rows")) is list
        and len(manifest["rows"]) == len(expected) == 17,
        "Exact17 source inventory required",
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
                "reviewed_public_source12"
                if path in {ROOT / label for label in CORE.SOURCE_PINS}
                else "reviewed_service_composition_source"
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


def package_versions():
    tree = ast.parse(ordinary(WRAPPER_PATH).read_bytes())
    literals = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PACKAGES" for target in node.targets)
    ]
    require(len(literals) == 1 and type(literals[0]) is dict, "Reviewed package map required")
    result = {}
    for name, wanted in literals[0].items():
        actual = importlib.metadata.version(name)
        require(
            actual == wanted or name == "causal-conv1d" and actual.split("+")[0] == wanted,
            "Pinned Decision3 package differs: " + name,
        )
        result[name] = actual
    return result


def validate_output(path):
    path = ordinary(path, directory=True)
    root = ROOT / "artifacts/jev-d3-native"
    require(path.is_relative_to(root), "Browser output must remain under replay output root")
    parts = path.relative_to(root).parts
    require(
        len(parts) == 2
        and parts[0].startswith("browser-")
        and SAFE_ID.fullmatch(parts[0].removeprefix("browser-"))
        and SAFE_ID.fullmatch(parts[1]),
        "Exact browser campaign/arm output shape required",
    )
    require(
        set(entry.name for entry in path.iterdir()).issubset({"config.json", "stderr.log"}),
        "Fresh browser arm output required",
    )
    return path


def initialize(config, state, output):
    selected_action_profile(config)
    require(
        sys.executable == INTERPRETER
        and sys.version_info[:2] == (3, 13)
        and sys.flags.no_user_site
        and sys.dont_write_bytecode,
        "Dedicated Decision3 Python3.13 -B -s worker required",
    )
    startup = validate_startup(config.get("startup"))
    initialization_started = time.monotonic() if startup is not None else None
    if startup is not None:
        state["startup_timing"] = {}
    identities = authenticate_sources(config["source_manifest"])
    authenticate_action_profile(config)
    if startup is not None:
        state["startup_timing"]["initial_source_authentication_seconds"] = (
            time.monotonic() - initialization_started
        )
    name = config["model"]
    require(name in CORE.MODEL_ORDER and config["alias"] == name, "Fixed Decision3 alias required")
    identity = identities[name]
    gates = CORE.Gates(output, identity, config["expected_boot_id"], config["expected_memtotal"])
    state["_identity"], state["_gates"] = identity, gates
    state["initial_resources"] = CORE.resource_snapshot()
    gates.check("before-input-and-framework")
    cache_started = time.monotonic() if startup is not None else None
    state["cache_before"] = CORE.authenticate_cache(name, identity, output, gates, "initial")
    if startup is not None:
        state["startup_timing"]["initial_cache_authentication_seconds"] = (
            time.monotonic() - cache_started
        )
    # Generated publisher module copies are runtime cache, never raw evidence.
    # Keep them fresh per owned arm outside the core's2048-file evidence tree.
    modules = ROOT / "artifacts/jev-d3-browser-module-cache" / config["nonce"] / name
    require(not modules.exists(), "Fresh owned publisher module cache required")
    modules.mkdir(parents=True)
    ordinary(modules, directory=True)
    state["publisher_module_cache"] = str(modules)
    os.environ.update(
        HF_HUB_CACHE=str(CORE.HUB),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_TOKEN_PATH="/nonexistent-d3-public-token",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
        HF_ENABLE_PARALLEL_LOADING="false",
        HF_PARALLEL_LOADING_WORKERS="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_MODULES_CACHE=str(modules),
    )
    blocked = []
    state["_blocked"] = blocked

    def deny_network(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            blocked.append(event)
            raise RuntimeError("Network forbidden in offline browser model worker")

    sys.addaudithook(deny_network)
    sys.path.insert(0, str(ROOT / "src"))
    action_adapter, action_wire = resolve_action_profile(config, state)
    import torch

    from s1.evaluation.decision3 import CHECKPOINTS, Decision3Backend, decision3_spec
    from s1.jev_decision3_native import JevDecision3NativeAdapter

    state["_torch"] = torch
    for label in CORE.SOURCE_PINS:
        module_name = label.removeprefix("src/").removesuffix(".py").replace("/", ".")
        if module_name.endswith(".__init__"):
            module_name = module_name.removesuffix(".__init__")
        module = sys.modules.get(module_name)
        require(
            module is not None and Path(module.__file__).resolve() == ROOT / label,
            "Imported production module origin differs: " + module_name,
        )
    require(
        CHECKPOINTS == identities and torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "Fixed identities and exactly one real CUDA GPU required",
    )
    gates.torch = torch
    gate = gates.check("before-model-constructor")
    require(
        same_uuid(gate["snapshot"]["cuda"]["uuid"], config["expected_gpu_uuid"]),
        "Actual CUDA UUID differs from selected parent GPU",
    )
    state["packages"] = package_versions()
    state["cuda_runtime"] = {
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    state["stage"] = "model_loading"
    persist(output / "receipt.json", public_state(state))
    load_started = time.monotonic() if startup is not None else None
    if startup is not None:
        state["startup_timing"]["before_model_constructor_seconds"] = (
            load_started - initialization_started
        )
    backend = Decision3Backend(
        decision3_spec(name, device="cuda:0"),
        expected_boot_id=config["expected_boot_id"],
        expected_memtotal=config["expected_memtotal"],
    )
    # Keep an owned backend available to finally even if geometry checks fail.
    state["_backend"] = backend
    state["_torch"] = torch
    if startup is not None:
        state["startup_timing"]["model_constructor_seconds"] = time.monotonic() - load_started
    runtime, width = backend.runtime, identity["readout_tensor_bytes"] // (255 * 4)
    require(
        all(
            parameter.device == torch.device("cuda:0") and parameter.dtype == torch.bfloat16
            for parameter in runtime.backbone.parameters()
        )
        and runtime.readout.device == torch.device("cuda:0")
        and runtime.readout.dtype == torch.float32
        and list(runtime.readout.shape) == [255, width],
        "Actual CUDA placement/BF16 backbone/FP32 readout differs",
    )
    require(
        sum(parameter.numel() for parameter in runtime.backbone.parameters())
        + runtime.readout.numel()
        == identity["loaded_parameters"],
        "Actual loaded parameter count differs",
    )
    provenance = runtime.provenance()
    require(
        provenance["model_name"] == name
        and provenance["repo_id"] == identity["model_id"]
        and provenance["pooling"] == "last",
        "Actual publisher provenance differs",
    )
    torch.cuda.synchronize(0)
    state["runtime"] = {
        "provenance": provenance,
        "runtime_info": runtime.runtime_info(),
        "kernels": runtime.kernels,
        "fast_enabled": runtime.fast is not None,
        "fast_skipped": runtime.fast_skipped,
        "provider_telemetry": backend.telemetry(),
        "readout_shape": list(runtime.readout.shape),
        "loaded_parameters": identity["loaded_parameters"],
        "packages": state["packages"],
    }
    state["resources_after_load"] = CORE.resource_snapshot(torch)
    startup = validate_startup(config.get("startup"))
    if startup is not None:
        warm_started = time.monotonic()
        state["stage"] = "publisher_warmup"
        persist(output / "receipt.json", public_state(state))
        publisher_warmup(
            runtime,
            torch,
            gates,
            output,
            state,
            config["expected_gpu_uuid"],
            state["cache_before"]["files"]["d3_runtime.py"],
        )
        state["startup_timing"]["warmup_instrumented_seconds"] = time.monotonic() - warm_started
        post_started = time.monotonic()
        authenticate_sources(config["source_manifest"])
        state["cache_post_warmup"] = CORE.authenticate_cache(
            name, identity, output, gates, "post-warmup"
        )
        require(
            state["cache_post_warmup"] == state["cache_before"],
            "Post-warmup complete cache changed",
        )
        require(
            all(
                p.device == torch.device("cuda:0") and p.dtype == torch.bfloat16
                for p in runtime.backbone.parameters()
            )
            and runtime.readout.device == torch.device("cuda:0")
            and runtime.readout.dtype == torch.float32
            and list(runtime.readout.shape) == [255, width]
            and sum(p.numel() for p in runtime.backbone.parameters()) + runtime.readout.numel()
            == identity["loaded_parameters"]
            and runtime.provenance() == provenance
            and package_versions() == state["packages"],
            "Post-warmup geometry/provenance/package version differs",
        )
        torch.cuda.synchronize(0)
        state["resources_after_warmup"] = CORE.resource_snapshot(torch)
        require(
            same_uuid(state["resources_after_warmup"]["cuda"]["uuid"], config["expected_gpu_uuid"]),
            "Post-warmup GPU differs",
        )
        state["post_warmup_checks"] = {
            "source_manifest_sha256": hashlib.sha256(encode(config["source_manifest"])).hexdigest(),
            "readout_shape": list(runtime.readout.shape),
            "loaded_parameters": identity["loaded_parameters"],
            "backbone_dtype": "bfloat16",
            "readout_dtype": "float32",
            "device": "cuda:0",
            "packages": state["packages"],
            "provenance": provenance,
        }
        state["startup_timing"]["postwarm_authentication_seconds"] = time.monotonic() - post_started
    if action_adapter is None:
        wire = load_file("jev_decision3_browser_named_wire", WIRE_PATH)
        adapter = JevDecision3NativeAdapter(backend, model_identity=config["alias"])
    else:
        wire = action_wire
        adapter = action_adapter(backend, model_identity=config["alias"])
    return adapter, wire, torch, gates, identity, blocked


def public_state(state):
    return {key: value for key, value in state.items() if not key.startswith("_")}


def send(control, message):
    blob = encode(message)
    require(len(blob) <= MAX_RESPONSE, "Control response byte bound")
    control.write(blob + b"\n")
    control.flush()


def run(config, control):
    selected_action_profile(config)
    output = validate_output(config["output_dir"])
    require(
        type(config.get("nonce")) is str and SAFE_ID.fullmatch(config["nonce"]),
        "Safe nonce required",
    )
    require(
        type(config.get("expected_boot_id")) is str and config["expected_boot_id"],
        "Boot ID required",
    )
    require(
        type(config.get("expected_memtotal")) is int and config["expected_memtotal"] > 0,
        "Host total required",
    )
    require(
        type(config.get("expected_gpu_uuid")) is str and config["expected_gpu_uuid"],
        "GPU UUID required",
    )
    state = {
        "profile": PROFILE,
        "stage": "initializing",
        "model": config.get("model"),
        "alias": config.get("alias"),
        "nonce": config["nonce"],
        "gpu_uuid": config["expected_gpu_uuid"],
        "pid": os.getpid(),
        "requests": [],
        "error": None,
        "secondary_errors": [],
        "success": False,
        "campaign_stop": False,
        "helper_calls": 0,
        "optimizer_updates": 0,
        "warmup_calls": 0,
        "task_score": None,
        "final_resources": None,
    }
    startup = validate_startup(config.get("startup"))
    if startup is not None:
        state.update(
            profile=WARM_PROFILE,
            startup=startup,
            browser_envelopes_seen=0,
            publisher_warmup_invocations=0,
        )
    started, primary, adapter, torch, identity, blocked = (
        time.monotonic(),
        None,
        None,
        None,
        None,
        [],
    )

    def failed(exc, stage):
        nonlocal primary
        if state.get("_request_error") is not None:
            # The structured request error is already primary even though it is
            # not an exception object in this outer control loop.
            state["secondary_errors"].append({"stage": stage, **error_info(exc)})
        if primary is None:
            primary = exc
        else:
            primary.add_note(stage + " also failed: " + repr(exc))
            if state.get("_request_error") is None:
                state["secondary_errors"].append({"stage": stage, **error_info(exc)})
        state.update(
            stage="failed",
            success=False,
            campaign_stop=True,
            error=state.get("_request_error", error_info(primary)),
        )

    try:
        persist(output / "receipt.json", state)
        adapter, wire, torch, gates, identity, blocked = initialize(config, state, output)
        if startup is not None:
            state["startup_timing"]["total_worker_to_ready_seconds"] = time.monotonic() - started
        state.update(stage="ready", success=True)
        persist(output / "receipt.json", public_state(state))
        ready = {
            "event": "ready",
            "model": config["model"],
            "alias": config["alias"],
            "nonce": config["nonce"],
            "gpu_uuid": config["expected_gpu_uuid"],
            "pid": os.getpid(),
            "resources": state["resources_after_load"],
            "runtime": state["runtime"],
        }
        if "action_profile" in state:
            ready.update(
                {
                    key: copy.deepcopy(state[key])
                    for key in (
                        "action_profile",
                        "action_profile_source",
                        "action_profile_provenance",
                    )
                }
            )
        if startup is not None:
            ready.update(
                {
                    key: copy.deepcopy(state[key])
                    for key in (
                        "startup",
                        "warmup",
                        "warmup_calls",
                        "publisher_warmup_invocations",
                        "cache_before",
                        "cache_post_warmup",
                        "post_warmup_checks",
                        "startup_timing",
                    )
                }
            )
        if startup is not None:
            ready["resources_after_load"] = state["resources_after_load"]
            ready["resources"] = state["resources_after_warmup"]
        send(control, ready)
        while True:
            line = sys.stdin.buffer.readline(MAX_FRAME + 1)
            require(
                line and len(line) <= MAX_FRAME and line.endswith(b"\n"),
                "Bounded complete RPC line required",
            )
            received_started = time.monotonic() if startup is not None else None
            # Avoid parsing an HTTP request before process_request starts its timer.
            if len(line) <= 64 and re.fullmatch(rb'\s*\{\s*"command"\s*:\s*"close"\s*\}\s*', line):
                state["stage"] = "closing"
                break
            if startup is not None:
                state["browser_envelopes_seen"] += 1
                if state["browser_envelopes_seen"] > startup["max_browser_envelopes"]:
                    state["envelope_limit_refusal"] = {
                        "limit": startup["max_browser_envelopes"],
                        "raw_envelope_hex": line.hex(),
                        "sha256": hashlib.sha256(line).hexdigest(),
                    }
                    persist(output / "receipt.json", public_state(state), terminal=True)
                    raise ValueError(
                        "External warm workload browser envelope limit exceeded; no inference"
                    )
                persist(output / "receipt.json", public_state(state))
            require(len(state["requests"]) < MAX_REQUESTS, "At most192 requests per worker")
            response = process_request(
                line,
                adapter=adapter,
                wire=wire,
                torch=torch,
                gates=gates,
                output=output,
                model=config["model"],
                nonce=config["nonce"],
                gpu_uuid=config["expected_gpu_uuid"],
                **(
                    {"startup": startup, "started": received_started} if startup is not None else {}
                ),
            )
            journal = response["journal"]
            if not response["success"]:
                # Preserve the original failure/raw response before journal pin,
                # receipt writes or cleanup can fail independently.
                state.update(stage="failed", success=False, campaign_stop=True)
                state["error"] = journal.get("error")
                state["_request_error"] = copy.deepcopy(state["error"])
                state["failed_request_response"] = copy.deepcopy(response)
            journal_path = Path(journal["journal_path"])
            state["requests"].append(
                {
                    "request_id": journal["request_id"],
                    "http_sha256": journal["http_sha256"],
                    "success": response["success"],
                    "campaign_stop": response["campaign_stop"],
                    "elapsed_seconds": journal["elapsed_seconds"],
                    "journal": {"path": str(journal_path), "pin": CORE.pin(journal_path)},
                }
            )
            persist(output / "receipt.json", public_state(state))
            send(control, response)
            if not response["success"]:
                break
    except BaseException as exc:
        failed(exc, "worker_or_control")
    finally:
        torch = torch if torch is not None else state.pop("_torch", None)
        backend = adapter.backend if adapter is not None else state.pop("_backend", None)
        identity = identity if identity is not None else state.pop("_identity", None)
        gates = locals().get("gates", state.pop("_gates", None))
        blocked = state.pop("_blocked", blocked)
        if torch is not None:
            try:
                torch.cuda.synchronize(0)
            except BaseException as exc:
                failed(exc, "final_cuda_fence")
        if backend is not None:
            try:
                backend.close()
            except BaseException as exc:
                failed(exc, "backend_close")
        if torch is not None:
            try:
                state["final_resources"] = CORE.resource_snapshot(torch)
                require(
                    same_uuid(
                        state["final_resources"]["cuda"]["uuid"], config["expected_gpu_uuid"]
                    ),
                    "Final CUDA UUID differs",
                )
            except BaseException as exc:
                failed(exc, "final_resources")
        if identity is not None:
            try:
                state["cache_after"] = CORE.authenticate_cache(
                    config["model"], identity, output, gates, "final"
                )
            except BaseException as exc:
                failed(exc, "final_cache_authentication")
        try:
            authenticate_sources(config["source_manifest"])
            authenticate_action_profile(config, state.get("action_profile_provenance"))
        except BaseException as exc:
            failed(exc, "final_source_authentication")
        state["network_attempts_blocked"] = blocked
        if blocked:
            failed(RuntimeError("Offline worker attempted network access"), "network_attempts")
        if state["stage"] == "closing" and primary is None and not state["campaign_stop"]:
            state.update(stage="complete", success=True)
        elif state["stage"] != "complete":
            state.update(stage="failed", success=False, campaign_stop=True)
        state["elapsed_seconds"] = time.monotonic() - started
        if primary is not None:
            state["error"] = state.get("_request_error", error_info(primary))
        try:
            persist(output / "receipt.json", public_state(state), terminal=True)
        except BaseException as exc:
            failed(exc, "terminal_receipt_persistence")
        try:
            send(
                control,
                {
                    "event": "closed",
                    "success": state["stage"] == "complete" and state["success"],
                    "campaign_stop": state["campaign_stop"],
                    "receipt": public_state(state),
                },
            )
        except BaseException as exc:
            failed(exc, "closed_control_delivery")
            try:
                persist(output / "receipt.json", public_state(state), terminal=True)
            except BaseException as persistence_exc:
                failed(persistence_exc, "failed_control_receipt_persistence")
    return 0 if state["stage"] == "complete" and state["success"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--action-profile", choices=ACTION_PROFILES)
    args = parser.parse_args(argv)
    # FD-level redirection also catches C extension and publisher print output.
    control = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    config_path = ordinary(args.worker)
    require(config_path.stat().st_size <= MAX_FRAME, "Worker config byte bound")
    config = parse(config_path.read_bytes())
    require(type(config) is dict, "Worker config object required")
    profile = selected_action_profile(config, args.action_profile)
    if args.action_profile is not None:
        config["action_profile"] = profile
    try:
        return run(config, control)
    finally:
        control.close()


if __name__ == "__main__":
    raise SystemExit(main())
