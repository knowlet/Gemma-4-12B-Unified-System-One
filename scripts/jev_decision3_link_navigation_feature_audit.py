"""Rehydrate and independently replay one closed link-navigation GPU shard.

This module does not launch a model, browser, service or optimizer. The caller
must separately authenticate its selected plan and exact owner file inventory,
run the audit in a distinct supervised process, and retain that process's own
terminal and resource evidence. A replay is not fit or model qualification.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import stat
from pathlib import Path

import jev_decision3_link_navigation_features as features
import jev_decision3_link_navigation_gpu_owner as owner
import jev_decision3_link_navigation_pipeline as pipeline
import numpy as np

from s1.evaluation.decision3 import native_payload
from s1.jev_decision3_native import parse_native_payload

MAX_FILE = 16 * 1024 * 1024
MAX_GPU_PYTHON_FILE = 64 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
MAX_FILES = 128


def _need(value, message):
    if not value:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _pin(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("nonfinite JSON")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def _read_closed(root, pins):
    root = Path(root)
    _need(
        root.is_absolute()
        and root.is_dir()
        and not any(p.is_symlink() for p in (root, *root.parents)),
        "ordinary absolute owner directory required",
    )
    _need(
        type(pins) is dict and 1 <= len(pins) <= MAX_FILES,
        "bounded independently selected file inventory required",
    )
    for name, ref in pins.items():
        _need(
            type(name) is str
            and re.fullmatch(r"[A-Za-z0-9_.-]+", name)
            and name not in {".", ".."}
            and type(ref) is dict
            and set(ref) == {"sha256", "size_bytes"}
            and _pin(ref["sha256"])
            and type(ref["size_bytes"]) is int
            and 0 <= ref["size_bytes"] <= MAX_FILE,
            "invalid owner file pin",
        )
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        before = os.fstat(directory)
        names = os.listdir(directory)
        _need(set(names) == set(pins), "owner directory is not a closed exact inventory")
        blobs, observations, total = {}, {}, 0
        for name in names:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            try:
                initial = os.fstat(fd)
                _need(
                    stat.S_ISREG(initial.st_mode)
                    and initial.st_nlink == 1
                    and initial.st_size == pins[name]["size_bytes"],
                    "owner entry must be an exact ordinary file",
                )
                total += initial.st_size
                _need(total <= MAX_TOTAL, "owner inventory exceeds aggregate bound")
                with os.fdopen(fd, "rb", closefd=False) as stream:
                    data = stream.read(MAX_FILE + 1)
                final = os.fstat(fd)

                def identity(x):
                    return (x.st_dev, x.st_ino, x.st_mode, x.st_size, x.st_mtime_ns, x.st_ctime_ns)

                _need(
                    len(data) == initial.st_size
                    and identity(initial) == identity(final)
                    and _sha(data) == pins[name]["sha256"],
                    "owner entry bytes or identity changed",
                )
                observations[name] = identity(final)
                blobs[name] = data
            finally:
                os.close(fd)
        _need(set(os.listdir(directory)) == set(names), "owner population changed during read")
        for name, expected in observations.items():
            current = os.stat(name, dir_fd=directory, follow_symlinks=False)
            _need(
                (
                    current.st_dev,
                    current.st_ino,
                    current.st_mode,
                    current.st_size,
                    current.st_mtime_ns,
                    current.st_ctime_ns,
                )
                == expected,
                "owner entry changed after read",
            )
        after = os.fstat(directory)
        _need(
            (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns),
            "owner directory changed during read",
        )
        return blobs
    finally:
        os.close(directory)


def _ref(ref, blobs, pins, name):
    _need(
        type(ref) is dict
        and set(ref) == {"path", "sha256", "size_bytes"}
        and ref["path"] == name
        and name in blobs
        and ref["sha256"] == pins[name]["sha256"]
        and ref["size_bytes"] == pins[name]["size_bytes"],
        "owner artifact reference differs",
    )
    return blobs[name]


def _array(data, shape):
    stream = io.BytesIO(data)
    value = np.load(stream, allow_pickle=False)
    _need(
        type(value) is np.ndarray
        and value.dtype == np.dtype("<f4")
        and value.shape == shape
        and value.flags.c_contiguous
        and stream.tell() == len(data)
        and bool(np.isfinite(value).all()),
        "exact bounded finite FP32 NPY required",
    )
    value.flags.writeable = False
    return value


def _check_sources(pins):
    _need(type(pins) is dict and bool(pins), "source pin inventory required")
    for label, digest in pins.items():
        path = Path(label)
        _need(
            type(label) is str
            and path.is_absolute()
            and _pin(digest)
            and not any(p.is_symlink() for p in (path, *path.parents)),
            "ordinary pinned source required",
        )
        limit = MAX_GPU_PYTHON_FILE if str(path) == owner.GPU_PYTHON_REAL else MAX_FILE
        if str(path) == owner.GPU_PYTHON_REAL:
            _need(digest == owner.GPU_PYTHON_SHA256, "fixed GPU Python digest required")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            _need(
                stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit,
                "bounded ordinary source required",
            )
            computed = hashlib.sha256()
            length = 0
            with os.fdopen(fd, "rb", closefd=False) as stream:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    length += len(chunk)
                    _need(length <= limit, "source file exceeds bound")
                    computed.update(chunk)
            after = os.fstat(fd)
            named = path.stat(follow_symlinks=False)

            def identity(value):
                return (
                    value.st_dev,
                    value.st_ino,
                    value.st_mode,
                    value.st_size,
                    value.st_mtime_ns,
                    value.st_ctime_ns,
                )

            _need(
                identity(before) == identity(after) == identity(named)
                and length == before.st_size
                and computed.hexdigest() == digest,
                "source pin differs",
            )
        finally:
            os.close(fd)


def _closure(blobs, plan, plan_sha):
    ready, go = _decode(blobs["ready.json"]), _decode(blobs["go.json"])
    worker, closure = _decode(blobs["worker-result.json"]), _decode(blobs["owner-closure.json"])
    events = [_decode(line) for line in blobs["owner-events.jsonl"].splitlines()]
    _need(
        type(ready) is dict
        and set(ready) == {"pid", "startticks", "boot_id", "invocation"}
        and type(ready["pid"]) is int
        and ready["pid"] > 0
        and type(ready["startticks"]) is int
        and ready["startticks"] > 0
        and type(ready["boot_id"]) is str
        and _pin(plan_sha),
        "worker ready identity required",
    )
    invocation = ready["invocation"]
    _need(
        type(invocation) is str
        and re.fullmatch(r"[0-9a-f]{32}", invocation)
        and go == {"invocation": invocation, "plan_sha256": plan_sha},
        "owner handshake differs",
    )
    _need(
        type(worker) is dict
        and worker.get("schema") == "decision3-link-navigation-gpu-owner-worker-v1"
        and worker.get("status") == "captured_pending_independent_fp64_audit"
        and worker.get("invocation") == invocation
        and worker.get("plan_sha256") == plan_sha
        and worker.get("initial_cuda_allocated") == worker.get("final_cuda_allocated") == 0
        and worker.get("fit_eligible") is False
        and worker.get("model_score") is None
        and not any(
            key in worker for key in ("cleanup_error_type", "close_error_type", "error_type")
        ),
        "complete worker and cleared CUDA allocator required",
    )
    gpu = worker.get("gpu")
    _need(
        type(gpu) is dict
        and type(gpu.get("uuid")) is str
        and gpu["uuid"].lower().removeprefix("gpu-")
        == plan["gpu_uuid"].lower().removeprefix("gpu-")
        and gpu.get("capability") == [12, 1]
        and type(gpu.get("name")) is str
        and type(gpu.get("torch_cuda")) is str,
        "selected GPU identity differs",
    )
    _need(
        type(closure) is dict
        and closure.get("schema") == "decision3-link-navigation-gpu-owner-closure-v1"
        and closure.get("launch_attempted") is True
        and closure.get("invocation") == invocation
        and closure.get("fit_eligible") is False
        and closure.get("model_score") is None
        and closure.get("cgroup_closure") in {"absent", "confirmed_unpopulated"}
        and closure.get("closure") in {"retained-unit", "confirmed-unit-GC"}
        and not closure.get("closure_error_type"),
        "verified cgroup closure required",
    )
    unit = closure.get("unit")
    _need(
        type(unit) is str and re.fullmatch(r"jev-link-feature-[0-9a-f]{32}\.service", unit),
        "owned unit identity required",
    )
    terminal, final = closure.get("terminal"), closure.get("final_unit")
    _need(
        type(terminal) is dict
        and terminal.get("InvocationID") == invocation
        and terminal.get("MainPID") == "0"
        and terminal.get("ExecMainCode") in {"exited", "1"}
        and terminal.get("ExecMainStatus") == "0"
        and terminal.get("Result") == "success"
        and (
            terminal.get("ActiveState") in {"inactive", "failed"}
            or (terminal.get("ActiveState") == "active" and terminal.get("SubState") == "exited")
        ),
        "same successful terminal invocation required",
    )
    _need(
        type(final) is dict
        and final.get("MainPID") == "0"
        and final.get("Result") in {"success", ""},
        "closed final unit required",
    )
    if closure["closure"] == "retained-unit":
        _need(
            final.get("LoadState") == "loaded" and final.get("InvocationID") == invocation,
            "final retained invocation differs",
        )
    else:
        _need(
            closure.get("stop_returned") is True
            and final.get("LoadState") == "not-found"
            and final.get("InvocationID") == ""
            and final.get("ActiveState") == "inactive"
            and final.get("ControlGroup") == "",
            "unit GC needs confirmed same invocation stop",
        )
    _need(
        type(events) is list
        and events
        and all(type(event) is dict for event in events)
        and not any(
            event.get("event") in {"monitor_failed", "closure_unknown", "stop_unconfirmed"}
            for event in events
        ),
        "unbroken owner monitoring required",
    )
    requested = [e for e in events if e.get("event") == "unit_requested"]
    bindings = [e for e in events if e.get("event") == "binding_verified"]
    observations = [e for e in events if e.get("event") in {"prelaunch_host", "host"}]
    _need(
        len(requested) == len(bindings) == 1
        and requested[0].get("unit") == unit
        and requested[0].get("plan_sha256") == plan_sha,
        "unique owner launch required",
    )
    binding = bindings[0]
    group = binding.get("cgroup")
    _need(
        binding.get("invocation") == invocation
        and binding.get("pid") == ready["pid"]
        and binding.get("startticks") == ready["startticks"]
        and binding.get("boot_id") == ready["boot_id"]
        and type(group) is str
        and group.startswith("/")
        and ".." not in Path(group).parts
        and group.endswith("/app.slice/" + unit),
        "owner cgroup binding differs",
    )
    _need(
        not terminal.get("ControlGroup") or terminal["ControlGroup"] == group,
        "terminal cgroup differs",
    )
    _need(
        events.index(requested[0]) < events.index(binding)
        and len(observations) >= 3
        and any(event.get("event") == "host" for event in events[events.index(binding) + 1 :])
        and observations[0].get("event") == "prelaunch_host"
        and all(
            type(e.get("available_bytes")) is int
            and e["available_bytes"] >= plan["memory_floor_bytes"]
            and type(e.get("running")) is int
            and type(e.get("pending")) is int
            and e["running"] == e["pending"] == 0
            and type(e.get("monotonic")) in (float, int)
            and math.isfinite(e["monotonic"])
            for e in observations
        )
        and all(a["monotonic"] <= b["monotonic"] for a, b in zip(observations, observations[1:])),
        "host supervision evidence differs",
    )
    return worker, closure


def _replay(case, row, baseline):
    wire = row["payload"]
    _need(
        _sha(_canonical(wire)) == row["wire_sha256"] == case["wire_sha256"]
        and wire.get("model") == features.ALIAS,
        "full ordered original wire differs",
    )
    questions = wire["questions"]
    heads = list(questions)
    _need(
        1 <= len(heads) <= 4
        and case["head_order"] == heads
        and case["criteria_order"] == [list(questions[head]["criteria"]) for head in heads],
        "original head and criteria order differs",
    )
    parsed, _, _, _ = parse_native_payload(wire, model_identity=features.ALIAS)
    _need(
        _sha(_canonical(native_payload(parsed))) == case["native_payload_sha256"],
        "native payload digest differs",
    )
    preflight = case["preflight"]
    _need(
        type(preflight) is dict
        and set(preflight) == {"lengths", "errors", "project_context_limit"}
        and type(preflight["errors"]) in (dict, list)
        and not preflight["errors"]
        and preflight["project_context_limit"] == 16384
        and type(preflight["lengths"]) is dict
        and list(preflight["lengths"]) == heads
        and all(type(x) is int and 0 < x <= 16384 for x in preflight["lengths"].values()),
        "preflight differs",
    )
    shape = [len(heads), max(preflight["lengths"].values()), 1024]
    _need(
        case["first_layer_shape"] == case["final_norm_shape"] == shape,
        "same-call layer geometry differs",
    )
    raw = case["raw_response"]
    _need(
        type(raw) is dict
        and raw.get("model") == "d3-edge"
        and type(raw.get("answers")) is dict
        and list(raw["answers"]) == heads
        and _sha(_canonical(raw)) == case["raw_response_sha256"]
        and type(raw.get("usage")) is dict
        and set(raw["usage"]) == {"input_tokens", "output_tokens"}
        and all(type(value) is int for value in raw["usage"].values())
        and raw["usage"]
        == {"input_tokens": sum(preflight["lengths"].values()), "output_tokens": 0},
        "native response or usage differs",
    )
    matrix = case["features"]
    _need(
        _sha(matrix.tobytes(order="C")) == case["features_fp32_sha256"],
        "same-call feature bytes differ",
    )
    logits = matrix.astype(np.float64) @ baseline.astype(np.float64).T
    _need(bool(np.isfinite(logits).all()), "nonfinite FP64 logits")
    error = 0.0
    for i, head in enumerate(heads):
        labels = list(questions[head]["criteria"])
        answer = raw["answers"][head]
        _need(
            1 <= len(labels) <= 255
            and type(answer) is dict
            and answer.get("type") == "choice"
            and not answer.get("error")
            and type(answer.get("probabilities")) is dict
            and list(answer["probabilities"]) == labels,
            "native legal probabilities differ",
        )
        values = list(answer["probabilities"].values())
        _need(
            all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in values)
            and math.isclose(sum(values), 1.0, abs_tol=1e-6),
            "invalid native probabilities",
        )
        legal = logits[i, : len(labels)]
        weights = np.exp(legal - legal.max())
        expected = weights / weights.sum()
        error = max(
            error, *(abs(float(a) - float(b)) for a, b in zip(values, expected, strict=True))
        )
        _need(
            error <= 1e-5
            and answer.get("choice")
            == labels[int(np.argmax(expected))]
            == max(answer["probabilities"], key=answer["probabilities"].__getitem__),
            "FP64 native replay differs",
        )
        entropy = -sum(value * math.log(value) for value in values if value > 0)
        confidence = (
            1.0 if len(labels) == 1 else max(0.0, min(1.0, 1.0 - entropy / math.log(len(labels))))
        )
        _need(
            type(answer.get("confidence")) in (int, float)
            and math.isfinite(answer["confidence"])
            and abs(answer["confidence"] - confidence) <= 1e-5,
            "native entropy confidence differs",
        )
    return error


def audit_shard(
    root,
    *,
    owner_plan_bytes,
    expected_owner_plan_sha256,
    expected_files,
    expected_source_pins,
    admitted,
):
    """Read an exact externally pinned owner output and return a merge-ready shard."""
    _need(
        type(owner_plan_bytes) is bytes
        and 0 < len(owner_plan_bytes) <= 1024 * 1024
        and _pin(expected_owner_plan_sha256)
        and _sha(owner_plan_bytes) == expected_owner_plan_sha256,
        "exact externally pinned owner plan required",
    )
    plan = _decode(owner_plan_bytes)
    owner._validate_plan(plan)
    _need(
        plan["python"] == owner.GPU_PYTHON
        and str(Path(plan["python"]).resolve(strict=True)) == owner.GPU_PYTHON_REAL
        and plan["source_pins"].get(owner.GPU_PYTHON_REAL) == owner.GPU_PYTHON_SHA256,
        "fixed pinned GPU Python required",
    )
    _need(
        plan["output_dir"] == str(root)
        and plan["source_pins"] == expected_source_pins
        and plan["receipt_sha256"] == admitted.receipt_sha256
        and plan["collection_plan_sha256"] == admitted.plan_sha256,
        "plan, selected source or admitted collection differs",
    )
    required = (owner.__file__, features.__file__, pipeline.__file__, owner.systemd.__file__)
    _need(
        all(str(Path(path).absolute()) in expected_source_pins for path in required),
        "owner, feature, admission and systemd source pins required",
    )
    _check_sources(expected_source_pins)
    blobs = _read_closed(root, expected_files)
    worker, closure = _closure(blobs, plan, expected_owner_plan_sha256)
    capture = copy.deepcopy(worker["capture"])
    _need(
        type(capture) is dict
        and capture.get("schema") == features.SCHEMA
        and capture.get("status") == "captured_pending_independent_fp64_audit"
        and capture.get("fit_eligible") is False
        and capture.get("model_score") is None
        and capture.get("model_id") == features.MODEL_ID
        and capture.get("revision") == features.REVISION
        and capture.get("manifest_sha256") == features.MANIFEST_SHA256
        and capture.get("model_sha256") == features.MODEL_SHA256
        and capture.get("baseline_sha256") == features.W0_SHA256
        and capture.get("source_pins") == expected_source_pins,
        "pinned captured model and source required",
    )
    expected_shards = features.feature_only_shards(admitted, shard_size=plan["shard_size"])
    _need(plan["shard_index"] < len(expected_shards), "selected admitted shard required")
    selected = expected_shards[plan["shard_index"]]
    for key in (
        "index",
        "start",
        "stop",
        "total",
        "ordered_identity_sha256",
        "receipt_sha256",
        "plan_sha256",
    ):
        _need(capture.get(key) == selected[key], "admitted shard identity or coverage differs")
    rows = pipeline.compile_wire_projection(admitted)
    cases = capture["cases"]
    _need(
        type(cases) is list and len(cases) == selected["stop"] - selected["start"],
        "complete shard cases required",
    )
    expected_names = {
        "ready.json",
        "go.json",
        "worker-result.json",
        "owner-closure.json",
        "owner-events.jsonl",
        "worker.log",
        "baseline.npy",
    }
    for index in range(len(cases)):
        expected_names.update({f"features-{index:04d}.npy", f"raw-{index:04d}.json"})
    _need(set(blobs) == expected_names, "unexpected or incomplete owner artifact inventory")
    baseline = _array(_ref(capture["baseline"], blobs, expected_files, "baseline.npy"), (255, 1024))
    _need(_sha(baseline.tobytes(order="C")) == features.W0_SHA256, "original FP32 W0 differs")
    capture["baseline"] = baseline
    max_error, checked_heads, legal = 0.0, 0, 0
    for index, case in enumerate(cases):
        ordinal = selected["start"] + index
        row = rows[ordinal]
        _need(
            case["global_ordinal"] == ordinal and case["case_id"] == row["case_id"],
            "original case order differs",
        )
        case["features"] = _array(
            _ref(case["features"], blobs, expected_files, f"features-{index:04d}.npy"),
            (len(case["head_order"]), 1024),
        )
        raw_bytes = _ref(case["raw_response"], blobs, expected_files, f"raw-{index:04d}.json")
        _need(
            _sha(raw_bytes) == case["raw_response_sha256"],
            "raw file and same-call response digest differ",
        )
        case["raw_response"] = _decode(raw_bytes)
        max_error = max(max_error, _replay(case, row, baseline))
        checked_heads += len(case["head_order"])
        legal += sum(len(choices) for choices in case["criteria_order"])
    _check_sources(expected_source_pins)
    _need(
        _read_closed(root, expected_files) == blobs,
        "owner inventory changed during independent replay",
    )
    audit = {
        "status": "fp64_replayed_pending_external_admission",
        "fit_eligible": False,
        "fit_authorized": False,
        "model_score": None,
        "invocation": closure["invocation"],
        "plan_sha256": expected_owner_plan_sha256,
        "owner_files_sha256": _sha(_canonical(expected_files)),
        "source_pins_sha256": _sha(_canonical(expected_source_pins)),
        "ordered_identity_sha256": capture["ordered_identity_sha256"],
        "shard_index": capture["index"],
        "cases": len(cases),
        "heads": checked_heads,
        "legal_probabilities": legal,
        "max_fp64_probability_error": max_error,
    }
    return {"shard": capture, "audit": audit}
