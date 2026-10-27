"""Supervise one candidate HTTP witness in a fresh user transient unit.

Source-only pending independent review. Parent independently observes ComfyUI
and physical memory, including while the child is blocked in native forward.
It never sends a ComfyUI mutation. One admitted TRAIN wire crosses a real
loopback socket. The selected cooperative forward lease is shared by inode;
it does not protect against other GPU users that do not cooperate. All
witnesses remain quarantined pending independent owner and FP64 audits.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import http.client as http_client
import importlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

import jev_stage_memory_profile_v2 as systemd

SOURCE_ONLY = True
SCHEMA = "decision3-candidate-http-owner-plan-v1"
MEMORY_MAX = 8 * 1024**3
MEMORY_FLOOR = 34709107508
WALL_SECONDS = 120
MAX_QUEUE_BYTES = 1024 * 1024
GPU_PYTHON = "/home/albert_hsieh/ComfyUI/.venv/bin/python"
GPU_PYTHON_REAL = (
    "/home/albert_hsieh/.local/share/uv/python/cpython-3.13.14-linux-aarch64-gnu/bin/python3.13"
)
GPU_PYTHON_SHA256 = "92ed50fd9dde3654d421d165214a95361e1889210b3d2063001d6c2e75eef2ab"


def _need(value, message):
    if not value:
        raise ValueError(message)


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        data, object_pairs_hook=pairs, parse_constant=lambda _: _need(False, "nonfinite JSON")
    )


def _load_plan(path, expected_sha):
    _need(re.fullmatch(r"[0-9a-f]{64}", expected_sha) is not None, "explicit plan SHA required")
    path = Path(path)
    _need(
        path.is_absolute() and path.is_file() and not path.is_symlink(),
        "ordinary absolute plan required",
    )
    raw = path.read_bytes()
    _need(
        0 < len(raw) <= 1024 * 1024 and _sha(raw) == expected_sha,
        "exact bounded owner plan required",
    )
    plan = _decode(raw)
    _validate_plan(plan)
    return plan


def _validate_plan(plan):
    keys = {
        "schema",
        "collection_dir",
        "receipt_sha256",
        "collection_plan_sha256",
        "cache_root",
        "modules_cache",
        "overlay",
        "source_pins",
        "python",
        "output_dir",
        "shard_size",
        "shard_index",
        "memory_floor_bytes",
        "gpu_uuid",
        "deltas",
        "transport",
    }
    _need(
        type(plan) is dict and set(plan) == keys and plan["schema"] == SCHEMA,
        "exact owner plan required",
    )
    for key in ("collection_dir", "cache_root", "modules_cache", "overlay", "python", "output_dir"):
        _need(
            type(plan[key]) is str
            and Path(plan[key]).is_absolute()
            and ".." not in Path(plan[key]).parts,
            "absolute selected paths required",
        )
    for key in ("receipt_sha256", "collection_plan_sha256"):
        _need(
            type(plan[key]) is str and re.fullmatch(r"[0-9a-f]{64}", plan[key]),
            "admission SHA pins required",
        )
    _need(
        type(plan["source_pins"]) is dict
        and bool(plan["source_pins"])
        and all(
            type(path) is str
            and Path(path).is_absolute()
            and type(digest) is str
            and re.fullmatch(r"[0-9a-f]{64}", digest)
            for path, digest in plan["source_pins"].items()
        ),
        "explicit source SHA inventory required",
    )
    _need(
        type(plan["shard_size"]) is int
        and plan["shard_size"] == 1
        and type(plan["shard_index"]) is int
        and plan["shard_index"] >= 0,
        "bounded shard selection required",
    )
    transport = plan["transport"]
    _need(
        type(transport) is dict
        and set(transport) == {"http_release", "lease_release", "lock", "journal", "nonce", "stat"}
        and all(
            type(transport[key]) is str
            and Path(transport[key]).is_absolute()
            and ".." not in Path(transport[key]).parts
            for key in ("http_release", "lease_release", "lock", "journal")
        )
        and type(transport["nonce"]) is str
        and re.fullmatch(r"[0-9a-f]{32}", transport["nonce"])
        and type(transport["stat"]) is dict
        and set(transport["stat"]) == {"lock", "journal"}
        and all(
            type(row) is dict
            and set(row) == {"device", "inode"}
            and all(type(number) is int and number >= 0 for number in row.values())
            for row in transport["stat"].values()
        ),
        "exact absolute HTTP/lease release and shared inode identities required",
    )
    _need(
        type(plan["deltas"]) is dict and set(plan["deltas"]) == {"operation", "click_target"},
        "both pinned candidate deltas required",
    )
    for value in plan["deltas"].values():
        _need(
            type(value) is dict
            and set(value) == {"path", "file_sha256", "fp32_sha256"}
            and type(value["path"]) is str
            and Path(value["path"]).is_absolute()
            and ".." not in Path(value["path"]).parts
            and all(
                type(value[key]) is str and re.fullmatch(r"[0-9a-f]{64}", value[key])
                for key in ("file_sha256", "fp32_sha256")
            ),
            "absolute path and two independent delta digests required",
        )
    _need(
        type(plan["memory_floor_bytes"]) is int and plan["memory_floor_bytes"] >= MEMORY_FLOOR,
        "minimum host physical memory floor required",
    )
    _need(
        type(plan["gpu_uuid"]) is str
        and re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", plan["gpu_uuid"]
        ),
        "explicit selected GPU UUID required",
    )
    workspace = Path(__file__).absolute().parent.parent
    _need(
        Path(plan["overlay"]) == workspace / "artifacts/jev-runtime/overlay",
        "fixed workspace overlay required",
    )


def _bind_project_source(workspace):
    src = workspace / "src"
    _need(src.is_dir() and not src.is_symlink(), "workspace source directory required")
    sys.path.insert(0, str(src))
    s1 = importlib.import_module("s1")
    _need(
        Path(s1.__file__).resolve() == (src / "s1/__init__.py").resolve(),
        "actual s1 import escaped selected workspace",
    )
    return src


def _bind_overlay(workspace, pins):
    overlay = workspace / "artifacts/jev-runtime/overlay"
    _need(overlay.is_dir() and not overlay.is_symlink(), "fixed workspace overlay required")
    sys.path.insert(0, str(overlay))
    _need(
        all(importlib.util.find_spec(name) is None for name in ("fla", "causal_conv1d", "kernels")),
        "unreviewed optional GPU kernels are forbidden",
    )
    versions = {
        "torch": "2.13.0+cu130",
        "transformers": "5.17.0",
        "safetensors": "0.8.0",
        "huggingface_hub": "1.33.0",
        "accelerate": "1.15.0",
    }
    _need(
        {name: importlib.metadata.version(name) for name in versions} == versions,
        "fixed GPU package versions required",
    )
    modules = {
        name: importlib.import_module(name)
        for name in ("torch", "transformers", "transformers.models.qwen3_5.modeling_qwen3_5")
    }
    _need(
        all(
            Path(modules[name].__file__).resolve().is_relative_to(overlay.resolve())
            for name in ("transformers", "transformers.models.qwen3_5.modeling_qwen3_5")
        ),
        "actual transformer modules must come from pinned overlay",
    )
    _need(
        all(str(Path(module.__file__).absolute()) in pins for module in modules.values()),
        "actual GPU module source pins required",
    )
    return modules["torch"]


def _verify_source_pins(pins):
    for path, digest in pins.items():
        source = Path(path)
        _need(
            source.is_absolute() and source.is_file() and not source.is_symlink(),
            "ordinary pinned source required",
        )
        with source.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        _need(actual == digest, "selected source bytes changed")


def _validate_interpreter(plan):
    selected = Path(plan["python"])
    _need(
        str(selected) == GPU_PYTHON and str(selected.resolve()) == GPU_PYTHON_REAL,
        "fixed GPU Python path required",
    )
    _need(
        plan["source_pins"].get(GPU_PYTHON_REAL) == GPU_PYTHON_SHA256,
        "independent GPU Python digest required",
    )
    with selected.open("rb") as stream:
        _need(
            hashlib.file_digest(stream, "sha256").hexdigest() == GPU_PYTHON_SHA256,
            "actual GPU Python binary changed",
        )
    probe = subprocess.run(
        [
            str(selected),
            "-I",
            "-c",
            "import json,sys;print(json.dumps([*sys.version_info[:3],sys.executable]))",
        ],
        check=True,
        capture_output=True,
        timeout=10,
    )
    _need(
        _decode(probe.stdout) == [3, 13, 14, GPU_PYTHON],
        "actual GPU Python version or executable differs",
    )


def _write_new(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())


def _read_memavailable():
    lines = Path("/proc/meminfo").read_text().splitlines()
    values = [line.split() for line in lines if line.startswith("MemAvailable:")]
    _need(
        len(values) == 1 and len(values[0]) == 3 and values[0][2] == "kB",
        "valid physical memory observation required",
    )
    return int(values[0][1]) * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise ValueError("Comfy queue redirect forbidden")


def _queue_bytes():
    request = urllib.request.Request("http://127.0.0.1:8188/queue", method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(request, timeout=2) as response:
        _need(response.status == 200, "Comfy queue unavailable")
        raw = response.read(MAX_QUEUE_BYTES + 1)
    _need(len(raw) <= MAX_QUEUE_BYTES, "bounded Comfy queue response required")
    return raw


def _observe(floor):
    raw = _queue_bytes()
    data = _decode(raw)
    _need(
        type(data) is dict
        and type(data.get("queue_running")) is list
        and type(data.get("queue_pending")) is list,
        "typed Comfy queue required",
    )
    available = _read_memavailable()
    observation = {
        "monotonic": time.monotonic(),
        "available_bytes": available,
        "running": len(data["queue_running"]),
        "pending": len(data["queue_pending"]),
    }
    _need(available >= floor, "physical memory below selected floor")
    _need(not observation["running"] and not observation["pending"], "Comfy queue is not idle")
    return observation


def _binding(properties, ready, unit):
    invocation, group = properties["InvocationID"], properties["ControlGroup"]
    _need(re.fullmatch(r"[0-9a-f]{32}", invocation) is not None, "unique unit invocation required")
    _need(
        properties["LoadState"] == "loaded"
        and properties["ActiveState"] == "active"
        and group.endswith("/app.slice/" + unit),
        "active owned app unit required",
    )
    _need(
        properties["Type"] == "exec"
        and properties["MemoryAccounting"] == "yes"
        and properties["MemoryMax"] == str(MEMORY_MAX)
        and properties["MemorySwapMax"] == "0"
        and properties["OOMPolicy"] == "kill"
        and properties["KillMode"] == "control-group"
        and properties["RuntimeMaxUSec"] == "2min"
        and properties["TimeoutStopUSec"] == "3s",
        "actual cgroup unit limits differ",
    )
    pid = ready["pid"]
    _need(
        type(pid) is int
        and int(properties["MainPID"]) == pid
        and ready["invocation"] == invocation
        and ready["boot_id"] == systemd.boot_id()
        and ready["startticks"] == systemd.process_identity(pid)["startticks"]
        and systemd.cgroup_membership(pid) == group,
        "unit, invocation, process and cgroup identity differ",
    )
    return invocation


def _stop_same(unit, invocation):
    current = systemd.service_show(unit, timeout=2)
    _need(
        current["InvocationID"] == invocation and current["LoadState"] == "loaded",
        "refuse stopping unknown or replacement invocation",
    )
    systemd.safe_stop(unit, invocation)


def _closed_cgroup(group, unit):
    _need(
        type(group) is str
        and group.startswith("/")
        and ".." not in Path(group).parts
        and group.endswith("/app.slice/" + unit),
        "exact previously bound cgroup required",
    )
    path = Path("/sys/fs/cgroup") / group.lstrip("/")
    if not path.exists():
        return "absent"
    values = dict(line.split() for line in (path / "cgroup.events").read_text().splitlines())
    _need(values.get("populated") == "0", "owned cgroup still populated")
    return "confirmed_unpopulated"


def _unit_args(unit, plan_path, plan_sha, output, plan):
    properties = {
        "Type": "exec",
        "MemoryAccounting": "yes",
        "MemoryMax": str(MEMORY_MAX),
        "MemorySwapMax": "0",
        "CPUQuota": "500%",
        "AllowedCPUs": "15-19",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxSec": str(WALL_SECONDS),
        "TimeoutStopSec": "3s",
        "RemainAfterExit": "yes",
        "StandardOutput": "append:" + str(output / "worker.log"),
        "StandardError": "append:" + str(output / "worker.log"),
    }
    return [
        "systemd-run",
        "--user",
        "--slice=app.slice",
        "--unit=" + unit,
        *["--property=" + k + "=" + v for k, v in properties.items()],
        "/usr/bin/taskset",
        "-c",
        "15-19",
        plan["python"],
        "-B",
        "-s",
        str(Path(__file__).absolute()),
        "--child",
        "--plan",
        str(plan_path),
        "--plan-sha256",
        plan_sha,
        "--unit",
        unit,
    ]


def _successful_exit(terminal):
    """Accept systemd's numeric and textual representations of CLD_EXITED."""
    return (
        terminal["ExecMainCode"] in {"1", "exited"}
        and terminal["ExecMainStatus"] == "0"
        and terminal["Result"] == "success"
    )


def _terminal_metadata(properties, invocation):
    terminal = properties["ActiveState"] in {"inactive", "failed"} or (
        properties["ActiveState"] == "active" and properties["SubState"] == "exited"
    )
    if not terminal:
        return None
    _need(
        invocation is not None
        and properties["InvocationID"] == invocation
        and properties["MainPID"] == "0",
        "terminal unit identity must be verified",
    )
    return {
        key: properties[key]
        for key in (
            "InvocationID",
            "ActiveState",
            "SubState",
            "MainPID",
            "ExecMainCode",
            "ExecMainStatus",
            "Result",
            "ControlGroup",
        )
    }


def _own_cgroup(unit):
    _need(os.sched_getaffinity(0) == {15, 16, 17, 18, 19}, "CPU affinity differs")
    group = systemd.cgroup_membership(os.getpid())
    _need(group.endswith("/app.slice/" + unit), "selected unit cgroup required")
    cg = Path("/sys/fs/cgroup") / group.lstrip("/")
    _need(
        (cg / "memory.max").read_text().strip() == str(MEMORY_MAX)
        and (cg / "memory.swap.max").read_text().strip() == "0",
        "8 GiB and no swap required",
    )
    quota, period = map(int, (cg / "cpu.max").read_text().split())
    _need(quota == period * 5, "five CPU quota required")
    return group


def _load_deltas(plan, torch):
    """Load exactly the externally pinned NPY and FP32 matrix bytes."""
    import numpy as np

    loaded = {}
    for head, ref in plan["deltas"].items():
        path = Path(ref["path"])
        _need(path.is_file() and path.resolve() == path, "ordinary delta file required")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            raw = stream.read(2 * 1024**2 + 1)
        _need(len(raw) <= 2 * 1024**2 and _sha(raw) == ref["file_sha256"], "delta file changed")
        array = np.load(io.BytesIO(raw), allow_pickle=False)
        _need(
            isinstance(array, np.ndarray)
            and array.shape == (255, 1024)
            and array.dtype == np.dtype("<f4")
            and array.flags.c_contiguous
            and np.isfinite(array).all()
            and _sha(array.tobytes()) == ref["fp32_sha256"],
            "complete pinned finite FP32 delta required",
        )
        loaded[head] = torch.from_numpy(array.copy())
    return loaded


def _same_native(left, right):
    """Compare state, question order and criteria order, ignoring question field order."""
    import jev_decision3_link_navigation_features as features

    if (
        type(left) is not dict
        or type(right) is not dict
        or set(left) != {"state", "questions"}
        or set(right) != {"state", "questions"}
        or type(left["questions"]) is not dict
        or type(right["questions"]) is not dict
        or list(left["questions"]) != list(right["questions"])
        or features._canonical(left["state"]) != features._canonical(right["state"])
    ):
        return False
    for name in left["questions"]:
        a, b = left["questions"][name], right["questions"][name]
        if (
            type(a) is not dict
            or type(b) is not dict
            or set(a) != {"type", "instructions", "criteria"}
            or set(b) != {"type", "instructions", "criteria"}
            or any(features._canonical(a[key]) != features._canonical(b[key]) for key in a)
        ):
            return False
    return True


def _drop_exception_frames(error):
    """Keep the exception and stack locations without retaining its frame objects."""
    stack = []
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        tb = current.__traceback__
        if tb is not None:
            stack.extend(
                {"function": item.name, "line": item.lineno} for item in traceback.extract_tb(tb)
            )
            traceback.clear_frames(tb)
            current.__traceback__ = None
        for child in (current.__cause__, current.__context__):
            if child is not None:
                pending.append(child)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
    return stack


def _load_reviewed_release(path, held_name, pins, module_name):
    """Load an independently pinned source-identical release, never flip a gate."""
    held = Path(__file__).absolute().parent / held_name
    released = Path(path)
    _need(
        str(held) in pins and str(released) in pins and released != held,
        "held and independently reviewed release source pins required",
    )
    _verify_source_pins({str(held): pins[str(held)], str(released): pins[str(released)]})
    source, actual = held.read_bytes(), released.read_bytes()
    marker = b"SOURCE_ONLY = True"
    _need(
        source.count(marker) == 1 and actual == source.replace(marker, b"SOURCE_ONLY = False", 1),
        "reviewed release must differ only by the explicit source gate",
    )
    spec = importlib.util.spec_from_file_location(module_name, released)
    _need(spec is not None and spec.loader is not None, "release module loader required")
    module = importlib.util.module_from_spec(spec)
    _need(module_name not in sys.modules, "fresh release module name required")
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise
    _need(module.SOURCE_ONLY is False, "reviewed released module required")
    return module


def _shared_identity(transport):
    for name in ("lock", "journal"):
        path = Path(transport[name])
        _need(
            path.is_file() and path.resolve() == path and not path.is_symlink(),
            "ordinary existing shared lock and journal required",
        )
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            _need(
                {"device": info.st_dev, "inode": info.st_ino} == transport["stat"][name],
                "shared lock and journal identity changed",
            )
        finally:
            os.close(fd)


def _lease_events(transport, pid):
    _shared_identity(transport)
    fd = os.open(transport["journal"], os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        _need(info.st_size <= 16 * 1024**2, "bounded shared forward journal required")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            lines = stream.read().splitlines()
    finally:
        os.close(fd)
    _shared_identity(transport)
    rows = [_decode(line) for line in lines]
    own = [row for row in rows if row.get("nonce") == transport["nonce"] and row.get("pid") == pid]
    _need(
        len(own) == 2
        and [row.get("event") for row in own] == ["acquired", "completed"]
        and all(row.get("kind") == "decision3-gpu" for row in own)
        and type(own[0].get("thread")) is int
        and own[0].get("thread") == own[1].get("thread")
        and type(own[0].get("monotonic_ns")) is int
        and type(own[1].get("monotonic_ns")) is int
        and own[0]["monotonic_ns"] <= own[1]["monotonic_ns"],
        "matching acquired and completed same-thread forward lease records required",
    )
    return own


def _run_one_http(http, service, body, check):
    """Return actual client-read bytes for one admitted wire."""
    _need(type(body) is bytes and bool(body), "one bounded original train wire required")
    server = http.loopback_server(service)
    host, port = server.server_address
    _need(host == "127.0.0.1" and type(port) is int and 0 < port <= 65535, "loopback only")
    server.timeout = 3
    result = {}
    conn = http_client.HTTPConnection(host, port, timeout=http.TIMEOUT_SECONDS + 2)

    def client():
        try:
            conn.request(
                "POST",
                "/v1/systemone",
                body=body,
                headers={
                    "Authorization": "Bearer " + service.token,
                    "Content-Type": "application/json",
                },
            )
            response = conn.getresponse()
            result["status"] = response.status
            result["content_length"] = response.getheader("Content-Length")
            result["body"] = response.read(http.MAX_RESPONSE + 1)
        except BaseException as error:
            result["error"] = error
        finally:
            conn.close()

    worker = threading.Thread(target=client, name="candidate-http-client", daemon=True)
    try:
        check()
        worker.start()
        server.handle_request()
        worker.join(timeout=http.TIMEOUT_SECONDS + 2)
        _need(not worker.is_alive(), "HTTP client did not finish")
        _need("error" not in result, "HTTP client failed to receive response")
        raw = result.get("body")
        _need(
            result.get("status") == 200
            and type(raw) is bytes
            and 0 < len(raw) <= http.MAX_RESPONSE
            and result.get("content_length") == str(len(raw))
            and service.calls == 1
            and not service.failed
            and service.pending is None,
            "one complete timely HTTP client receipt required",
        )
        check()
        return raw
    finally:
        conn.close()
        server.server_close()
        if worker.is_alive():
            worker.join(timeout=2)
        _need(not worker.is_alive(), "HTTP client thread closure unconfirmed")


def _capture_candidate(plan, admitted, backend, torch, output, check):
    """Run one admitted train wire over a lease-protected loopback socket."""
    _need(SOURCE_ONLY is False, "held source requires independent review before execution")
    import jev_decision3_link_navigation_adapter as candidate
    import jev_decision3_link_navigation_adapter_audit as audit
    import jev_decision3_link_navigation_features as features

    from s1.backends import normalize_response
    from s1.contracts import validate_distribution
    from s1.evaluation.decision3 import native_payload
    from s1.jev_decision3_native import parse_native_payload
    from s1.jev_generated import _request
    from s1.jev_hierarchy import _check_json

    pins = plan["source_pins"]
    transport = plan["transport"]
    _shared_identity(transport)
    http = _load_reviewed_release(
        transport["http_release"], "jev_decision3_candidate_http.py", pins, "_owned_candidate_http"
    )
    lease_module = _load_reviewed_release(
        transport["lease_release"], "jev_gpu_forward_lease.py", pins, "_owned_forward_lease"
    )
    actual_sources = (
        candidate.__file__,
        candidate.reference.__file__,
        audit.__file__,
        inspect.getfile(type(backend)),
        inspect.getfile(type(backend.model)),
        inspect.getfile(type(backend.runtime)),
        inspect.getfile(parse_native_payload),
        inspect.getfile(native_payload),
        inspect.getfile(_request),
        inspect.getfile(_check_json),
        inspect.getfile(normalize_response),
        inspect.getfile(validate_distribution),
        inspect.getfile(http.DeadlineBackend),
        inspect.getfile(http.JevDecision3NativeAdapter),
        http.wire.__file__,
    )
    _need(
        all(str(Path(p).absolute()) in pins for p in actual_sources), "actual source pins required"
    )
    features._check_sources(pins)
    features._validate_manifest(backend.runtime, plan["cache_root"])
    shards = features.feature_only_shards(admitted, shard_size=plan["shard_size"])
    _need(plan["shard_index"] < len(shards), "admitted training shard required")
    shard = shards[plan["shard_index"]]
    sink, receipts = audit.make_witness_writer(
        output / "witnesses", maximum_calls=len(shard["wires"])
    )
    deltas = _load_deltas(plan, torch)
    expected = {name: ref["fp32_sha256"] for name, ref in plan["deltas"].items()}

    def guard(provider, _owned_deltas):
        check()
        _need(provider is backend, "only the owned provider may be used")
        features._check_sources(pins)
        features._validate_manifest(provider.runtime, plan["cache_root"])
        return {"provider": "single-owned-child", "source_inventory_checked": True}

    wrapper = candidate.DualHeadDecision3Backend(
        backend, deltas, expected, torch, identity_guard=guard, witness_sink=sink
    )
    lease = lease_module.ForwardLease(
        transport["lock"], transport["journal"], transport["stat"], nonce=transport["nonce"]
    )
    serial = lease_module.SerialBackend(wrapper, lease, torch=torch)
    completed = []
    primary = None
    try:
        for row in shard["wires"]:
            check()
            wire = row["payload"]
            _need(
                features._sha(features._canonical(wire)) == row["wire_sha256"], "train wire changed"
            )
            parsed, _, _, _ = parse_native_payload(wire, model_identity=features.ALIAS)
            native = native_payload(parsed)
            _need(
                _same_native(native, {"state": wire["state"], "questions": wire["questions"]}),
                "original request changed during native serialization",
            )
            before = len(receipts)
            events = []

            def owner_guard(actual, _phase):
                check()
                _shared_identity(transport)
                features._check_sources(pins)
                features._validate_manifest(backend.runtime, plan["cache_root"])
                _need(
                    actual is serial and actual.backend is wrapper and actual.lease is lease,
                    "owned wrapper and lease required",
                )
                return {"provider": "single-owned-child", "source_inventory_checked": True}

            def witness_verify(receipt, payload, legacy):
                document = Path(receipt["directory"]) / "witness.json"
                raw = document.read_bytes()
                _need(_sha(raw) == receipt["manifest_sha256"], "witness manifest changed")
                witness = _decode(raw)
                provider = legacy["decision3_native"]["raw_response"]["decision3_provider_evidence"]
                return bool(
                    features._canonical(payload) == features._canonical(wire)
                    and features._canonical(legacy["decision3_native"]["original_request"])
                    == features._canonical(wire)
                    and _same_native(provider["native_payload"], native)
                    and _same_native(witness["metadata"]["native_payload"], native)
                    and features._canonical(witness["metadata"]["composed_raw"])
                    == features._canonical(provider["raw_response"])
                    and witness["witness_sha256"] == receipt["witness_sha256"]
                    and receipt["delta_sha256"] == expected
                )

            service = http.CandidateHTTP(
                serial,
                model_identity=features.ALIAS,
                token=secrets.token_urlsafe(32),
                max_requests=1,
                owner_guard=owner_guard,
                witness_snapshot=lambda: list(receipts),
                witness_verify=witness_verify,
                event_sink=lambda event: events.append(event),
            )
            response_raw = _run_one_http(http, service, features._canonical(wire), check)
            _need(len(receipts) == before + 1, "one witness per completed HTTP predict required")
            receipt = receipts[-1]
            _need(
                len(events) == 2
                and [event["status"] for event in events] == ["prepared", "socket_write_returned"]
                and events[-1]["http_body_sha256"] == row["wire_sha256"]
                and events[-1]["response_sha256"] == _sha(response_raw)
                and events[-1]["witness_receipt"] == receipt
                and lease.fd is None
                and lease.held is None
                and lease.fatal is None,
                "matching timely socket delivery, client receipt and completed lease required",
            )
            response = _decode(response_raw)
            provider = response["decision3_native"]["raw_response"]["decision3_provider_evidence"]
            lease_events = _lease_events(transport, os.getpid())
            evidence_raw = _json({"http_events": events, "lease_events": lease_events})
            _write_new(output / "http-evidence.json", evidence_raw)
            check()
            _need(len(response_raw) <= 8 * 1024**2, "bounded private response required")
            response_name = f"response-{before:04d}.json"
            _write_new(output / response_name, response_raw)
            check()
            completed.append(
                {
                    "wire_sha256": row["wire_sha256"],
                    "response_file": response_name,
                    "response_sha256": features._sha(response_raw),
                    "raw_response_sha256": features._sha(
                        features._canonical(provider["raw_response"])
                    ),
                    "receipt": receipt,
                }
            )
        _need(len(completed) == len(shard["wires"]) == len(receipts), "complete shard required")
        check()
        _load_deltas(plan, torch)
        features._check_sources(pins)
        features._validate_manifest(backend.runtime, plan["cache_root"])
        return {
            "ordered_identity_sha256": shard["ordered_identity_sha256"],
            "receipt_sha256": shard["receipt_sha256"],
            "plan_sha256": shard["plan_sha256"],
            "shard_index": shard["index"],
            "completed": completed,
            "http_evidence_sha256": _sha(evidence_raw),
            "fit_eligible": False,
            "model_score": None,
        }
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            wrapper.close()
        except BaseException as cleanup_error:
            if primary is None:
                raise
            primary.add_note("Candidate wrapper close also failed: " + repr(cleanup_error))


def _child(plan_path, plan_sha, unit):
    _need(SOURCE_ONLY is False, "held source requires independent review before execution")
    plan = _load_plan(plan_path, plan_sha)
    output = Path(plan["output_dir"])
    _own_cgroup(unit)
    invocation = os.environ.get("INVOCATION_ID", "")
    _need(re.fullmatch(r"[0-9a-f]{32}", invocation), "systemd invocation environment required")
    pid = os.getpid()
    ready = {
        "pid": pid,
        "startticks": systemd.process_identity(pid)["startticks"],
        "boot_id": systemd.boot_id(),
        "invocation": invocation,
    }
    _write_new(output / "ready.json", _json(ready))
    deadline = time.monotonic() + 20
    while not (output / "go.json").is_file():
        _need(time.monotonic() < deadline, "parent binding handshake timed out")
        time.sleep(0.1)
    _need(
        _decode((output / "go.json").read_bytes())
        == {"invocation": invocation, "plan_sha256": plan_sha},
        "verified parent handshake required",
    )
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_CACHE": plan["cache_root"],
            "HF_MODULES_CACHE": plan["modules_cache"],
            "CUDA_VISIBLE_DEVICES": plan["gpu_uuid"],
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "5",
            "MKL_NUM_THREADS": "5",
            "OPENBLAS_NUM_THREADS": "5",
        }
    )
    workspace = Path(__file__).absolute().parent.parent
    _bind_project_source(workspace)
    _verify_source_pins(plan["source_pins"])
    torch = _bind_overlay(workspace, plan["source_pins"])
    import jev_decision3_link_navigation_features as features
    import jev_decision3_link_navigation_pipeline as pipeline

    from s1.evaluation.decision3 import Decision3Backend, decision3_spec

    _need(
        all(
            getattr(module, "__file__", None)
            and Path(module.__file__).resolve().is_relative_to((workspace / "src").resolve())
            for name, module in tuple(sys.modules.items())
            if name == "s1" or name.startswith("s1.")
        ),
        "actual project modules must come from workspace source",
    )

    started = time.monotonic()

    def check():
        _need(time.monotonic() - started < WALL_SECONDS - 5, "worker cooperative deadline exceeded")
        _own_cgroup(unit)

    backend = None
    primary = None
    status = {
        "schema": "decision3-link-navigation-candidate-owner-worker-v1",
        "status": "running",
        "fit_eligible": False,
        "model_score": None,
        "invocation": invocation,
        "plan_sha256": plan_sha,
    }
    try:
        admitted = pipeline.admit_training_collection(
            plan["collection_dir"],
            expected_receipt_sha256=plan["receipt_sha256"],
            expected_plan_sha256=plan["collection_plan_sha256"],
        )
        features._check_sources(plan["source_pins"])
        check()
        torch.set_num_threads(5)
        torch.set_num_interop_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        _need(
            torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "exactly one selected GPU required",
        )
        gpu = torch.cuda.get_device_properties(0)
        _need(
            str(gpu.uuid).lower().removeprefix("gpu-")
            == plan["gpu_uuid"].lower().removeprefix("gpu-")
            and (gpu.major, gpu.minor) == (12, 1)
            and torch.cuda.is_bf16_supported(including_emulation=False),
            "selected GB10 with native BF16 required",
        )
        status["gpu"] = {
            "uuid": str(gpu.uuid),
            "name": gpu.name,
            "capability": [gpu.major, gpu.minor],
            "torch_cuda": torch.version.cuda,
        }
        torch.cuda.synchronize(0)
        status["initial_cuda_allocated"] = torch.cuda.memory_allocated(0)
        _need(status["initial_cuda_allocated"] == 0, "fresh process CUDA allocator required")
        backend = Decision3Backend(decision3_spec("d3-edge", device="cuda:0"))
        status["capture"] = _capture_candidate(plan, admitted, backend, torch, output, check)
        status["status"] = "candidate_captured_pending_independent_fp64_audit"

    except BaseException as error:
        primary = error
        status.update(status="failed_no_accepted_features", error_type=type(error).__name__)
        status["error_stack"] = _drop_exception_frames(error)
    finally:
        try:
            if backend is not None:
                backend.close()
        except BaseException as error:
            status["close_error_type"] = type(error).__name__
            status["status"] = "failed_cleanup_unconfirmed"
            _drop_exception_frames(error)
            if primary is None:
                primary = error
            else:
                primary.add_note("backend close also failed: " + repr(error))
        backend = None
        gc.collect()
        try:
            torch.cuda.synchronize(0)
            clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
            if callable(clear):
                clear()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(0)
            status["final_cuda_allocated"] = torch.cuda.memory_allocated(0)
            _need(status["final_cuda_allocated"] == 0, "worker CUDA allocation not cleared")
        except BaseException as error:
            status["cleanup_error_type"] = type(error).__name__
            status["status"] = "failed_cleanup_unconfirmed"
            _drop_exception_frames(error)
            if primary is None:
                primary = error
            else:
                primary.add_note("CUDA cleanup also failed: " + repr(error))
        _write_new(output / "worker-result.json", _json(status))
    if primary is not None:
        raise primary
    return 0


def _append_event(path, value):
    raw = _json(value) + b"\n"
    _need(len(raw) <= 8192, "bounded owner observation required")
    with path.open("ab") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _verify_candidate_result(worker, shard, plan, output):
    """Independently bind successful child receipts to the admitted shard."""
    import jev_decision3_link_navigation_features as features

    capture = worker.get("capture")
    _need(
        type(capture) is dict
        and capture.get("receipt_sha256") == shard["receipt_sha256"]
        and capture.get("plan_sha256") == shard["plan_sha256"]
        and capture.get("ordered_identity_sha256") == shard["ordered_identity_sha256"]
        and capture.get("shard_index") == shard["index"]
        and capture.get("fit_eligible") is False
        and capture.get("model_score") is None,
        "worker capture must bind the admitted train shard",
    )
    completed = capture.get("completed")
    _need(type(completed) is list and len(completed) == len(shard["wires"]), "all wires required")
    evidence_path = output / "http-evidence.json"
    _need(
        evidence_path.is_file()
        and evidence_path.resolve() == evidence_path
        and type(capture.get("http_evidence_sha256")) is str,
        "pinned HTTP evidence file required",
    )
    evidence_raw = evidence_path.read_bytes()
    _need(
        len(evidence_raw) <= 8 * 1024**2 and _sha(evidence_raw) == capture["http_evidence_sha256"],
        "HTTP evidence bytes changed",
    )
    evidence = _decode(evidence_raw)
    ready = _decode((output / "ready.json").read_bytes())
    _need(
        type(evidence) is dict
        and set(evidence) == {"http_events", "lease_events"}
        and evidence["lease_events"] == _lease_events(plan["transport"], ready["pid"]),
        "parent must independently verify completed shared forward lease",
    )
    expected_deltas = {key: ref["fp32_sha256"] for key, ref in plan["deltas"].items()}
    for index, (item, wire) in enumerate(zip(completed, shard["wires"], strict=True)):
        _need(
            type(item) is dict
            and set(item)
            == {"wire_sha256", "response_file", "response_sha256", "raw_response_sha256", "receipt"}
            and item["wire_sha256"] == wire["wire_sha256"]
            and item["response_file"] == f"response-{index:04d}.json"
            and all(
                type(item[key]) is str and re.fullmatch(r"[0-9a-f]{64}", item[key])
                for key in ("response_sha256", "raw_response_sha256")
            ),
            "ordered wire and response hashes required",
        )
        receipt = item["receipt"]
        directory = output / "witnesses" / f"call-{index:04d}"
        _need(
            type(receipt) is dict
            and receipt.get("directory") == str(directory)
            and directory.is_dir()
            and directory.resolve() == directory
            and receipt.get("delta_sha256") == expected_deltas,
            "owned witness directory and delta pins required",
        )
        manifest = directory / "witness.json"
        response_file = output / item["response_file"]
        _need(
            manifest.is_file()
            and manifest.resolve() == manifest
            and response_file.is_file()
            and response_file.resolve() == response_file,
            "ordinary witness and response files required",
        )
        raw = manifest.read_bytes()
        response_raw = response_file.read_bytes()
        _need(
            len(raw) <= 8 * 1024**2
            and _sha(raw) == receipt.get("manifest_sha256")
            and len(response_raw) <= 8 * 1024**2
            and _sha(response_raw) == item["response_sha256"],
            "witness or response file changed",
        )
        witness, response = _decode(raw), _decode(response_raw)
        expected_native = {key: wire["payload"][key] for key in ("state", "questions")}
        provider = response["decision3_native"]["raw_response"]["decision3_provider_evidence"]
        http_events = evidence["http_events"]
        _need(
            witness.get("witness_sha256") == receipt.get("witness_sha256")
            and _same_native(witness["metadata"]["native_payload"], expected_native)
            and _same_native(provider["native_payload"], expected_native)
            and features._sha(features._canonical(witness["metadata"]["composed_raw"]))
            == item["raw_response_sha256"]
            and features._canonical(provider["raw_response"])
            == features._canonical(witness["metadata"]["composed_raw"]),
            "witness request and response must match completed call",
        )
        _need(
            type(http_events) is list
            and len(http_events) == 2
            and [event.get("status") for event in http_events]
            == ["prepared", "socket_write_returned"]
            and http_events[-1].get("http_body_sha256") == wire["wire_sha256"]
            and http_events[-1].get("response_sha256") == item["response_sha256"]
            and http_events[-1].get("witness_receipt") == receipt
            and features._canonical(response["decision3_native"]["original_request"])
            == features._canonical(wire["payload"])
            and response["decision3_browser_wire"]["answer_values_changed"] is False,
            "one wire, same-call witness, delivered response and client bytes required",
        )


def supervise(plan_path, plan_sha):
    """Launch and independently watch one exclusively owned transient unit."""
    _need(SOURCE_ONLY is False, "held source requires independent review before execution")
    plan = _load_plan(plan_path, plan_sha)
    output = Path(plan["output_dir"])
    _need(
        output.parent.is_dir()
        and not any(path.is_symlink() for path in (output.parent, *output.parent.parents)),
        "ordinary existing output parent required",
    )
    output.mkdir(mode=0o700, exist_ok=False)
    journal = output / "owner-events.jsonl"
    _write_new(journal, b"")
    unit = "jev-link-candidate-" + secrets.token_hex(16) + ".service"
    invocation, bound_group, launch_attempted, terminal, stop_confirmed = (
        None,
        None,
        False,
        None,
        False,
    )
    started = time.monotonic()
    try:
        # Check source bytes and all admission pins without loading a model.
        _validate_interpreter(plan)
        _verify_source_pins(plan["source_pins"])
        _bind_project_source(Path(__file__).absolute().parent.parent)
        import jev_decision3_link_navigation_adapter as candidate
        import jev_decision3_link_navigation_adapter_audit as audit
        import jev_decision3_link_navigation_features as features
        import jev_decision3_link_navigation_pipeline as pipeline

        features._check_sources(plan["source_pins"])
        _need(
            str(Path(__file__).absolute()) in plan["source_pins"]
            and str(Path(features.__file__).absolute()) in plan["source_pins"]
            and str(Path(pipeline.__file__).absolute()) in plan["source_pins"]
            and str(Path(systemd.__file__).absolute()) in plan["source_pins"],
            "owner, watchdog, extractor and admission source pins required",
        )
        _need(
            str(Path(candidate.__file__).absolute()) in plan["source_pins"]
            and str(Path(audit.__file__).absolute()) in plan["source_pins"],
            "candidate and witness source pins required",
        )
        admitted = pipeline.admit_training_collection(
            plan["collection_dir"],
            expected_receipt_sha256=plan["receipt_sha256"],
            expected_plan_sha256=plan["collection_plan_sha256"],
        )
        shards = features.feature_only_shards(admitted, shard_size=plan["shard_size"])
        _need(plan["shard_index"] < len(shards), "selected admitted shard required")
        _append_event(journal, {"event": "prelaunch_host", **_observe(plan["memory_floor_bytes"])})
        initial = systemd.service_show(unit, timeout=2)
        _need(
            initial["LoadState"] == "not-found"
            and initial["ActiveState"] == "inactive"
            and initial["MainPID"] == "0",
            "fresh absent unit required",
        )
        command = _unit_args(unit, Path(plan_path), plan_sha, output, plan)
        launch_attempted = True
        subprocess.run(command, check=True, capture_output=True, timeout=10)
        _append_event(journal, {"event": "unit_requested", "unit": unit, "plan_sha256": plan_sha})
        while True:
            # This runs in the parent, independently of a blocked child forward.
            observation = _observe(plan["memory_floor_bytes"])
            _append_event(journal, {"event": "host", **observation})
            properties = systemd.service_show(unit, timeout=2)
            if invocation is None and (output / "ready.json").is_file():
                ready = _decode((output / "ready.json").read_bytes())
                invocation = _binding(properties, ready, unit)
                bound_group = properties["ControlGroup"]
                _write_new(
                    output / "go.json", _json({"invocation": invocation, "plan_sha256": plan_sha})
                )
                _append_event(
                    journal,
                    {
                        "event": "binding_verified",
                        "invocation": invocation,
                        "pid": ready["pid"],
                        "boot_id": ready["boot_id"],
                        "startticks": ready["startticks"],
                        "cgroup": properties["ControlGroup"],
                    },
                )
            elif invocation is not None:
                _need(
                    properties["InvocationID"] == invocation
                    and properties["LoadState"] == "loaded",
                    "owned invocation changed during monitoring",
                )
            terminal = _terminal_metadata(properties, invocation)
            if terminal is not None:
                break
            _need(
                time.monotonic() - started < WALL_SECONDS + 5, "parent monitoring deadline exceeded"
            )
            time.sleep(0.5)
        _verify_source_pins(plan["source_pins"])
        if _successful_exit(terminal):
            worker = _decode((output / "worker-result.json").read_bytes())
            _need(
                worker.get("status") == "candidate_captured_pending_independent_fp64_audit"
                and worker.get("invocation") == invocation
                and worker.get("plan_sha256") == plan_sha
                and worker.get("final_cuda_allocated") == 0,
                "successful worker and CUDA cleanup evidence required",
            )
            _verify_candidate_result(worker, shards[plan["shard_index"]], plan, output)
    except BaseException as error:
        try:
            _append_event(
                journal,
                {
                    "event": "monitor_failed",
                    "error_type": type(error).__name__,
                    "reason": str(error)[:500],
                    "invocation": invocation,
                },
            )
        except BaseException as log_error:
            error.add_note("Owner journal write also failed: " + repr(log_error))
        if invocation is None and launch_attempted and (output / "ready.json").is_file():
            try:
                ready = _decode((output / "ready.json").read_bytes())
                properties = systemd.service_show(unit, timeout=2)
                invocation = _binding(properties, ready, unit)
                bound_group = properties["ControlGroup"]
            except BaseException as bind_error:
                error.add_note("Owned unit binding could not be verified: " + repr(bind_error))
        if invocation is not None:
            try:
                _stop_same(unit, invocation)
                stop_confirmed = True
                try:
                    _append_event(
                        journal,
                        {"event": "same_invocation_stop_returned", "invocation": invocation},
                    )
                except BaseException as log_error:
                    error.add_note("Owner journal write also failed: " + repr(log_error))
            except BaseException as stop_error:
                error.add_note("Owned unit stop not confirmed: " + type(stop_error).__name__)
                try:
                    _append_event(
                        journal,
                        {"event": "stop_unconfirmed", "error_type": type(stop_error).__name__},
                    )
                except BaseException as log_error:
                    error.add_note("Owner journal write also failed: " + repr(log_error))
        else:
            try:
                _append_event(
                    journal,
                    {
                        "event": "closure_unknown",
                        "reason": "no bound invocation; RuntimeMaxSec remains the last limit",
                    },
                )
            except BaseException as log_error:
                error.add_note("Owner journal write also failed: " + repr(log_error))
        raise
    finally:
        active_error = sys.exception()
        closure = {
            "schema": "decision3-link-navigation-candidate-owner-closure-v1",
            "unit": unit,
            "invocation": invocation,
            "launch_attempted": launch_attempted,
            "terminal": terminal,
            "stop_returned": stop_confirmed,
            "closure": "unknown",
            "fit_eligible": False,
            "model_score": None,
        }
        if invocation is not None:
            try:
                properties = systemd.service_show(unit, timeout=2)
                closure["closure"] = systemd.closed_unit_metadata(
                    properties, invocation, stopped=stop_confirmed
                )
                closure["cgroup_closure"] = _closed_cgroup(bound_group, unit)
                closure["final_unit"] = {
                    key: properties[key]
                    for key in (
                        "InvocationID",
                        "LoadState",
                        "ActiveState",
                        "MainPID",
                        "ControlGroup",
                        "Result",
                    )
                }
            except BaseException as error:
                closure["closure_error_type"] = type(error).__name__
                closure["closure"] = "unknown"
                try:
                    _stop_same(unit, invocation)
                    stop_confirmed = True
                    closure["stop_returned"] = True
                    properties = systemd.service_show(unit, timeout=2)
                    closure["closure"] = systemd.closed_unit_metadata(
                        properties, invocation, stopped=True
                    )
                    closure["cgroup_closure"] = _closed_cgroup(bound_group, unit)
                    closure["final_unit"] = {
                        key: properties[key]
                        for key in (
                            "InvocationID",
                            "LoadState",
                            "ActiveState",
                            "MainPID",
                            "ControlGroup",
                            "Result",
                        )
                    }
                except BaseException as retry_error:
                    closure["closure"] = "unknown"
                    closure["closure_retry_error_type"] = type(retry_error).__name__
        try:
            _write_new(output / "owner-closure.json", _json(closure))
        except BaseException as write_error:
            if active_error is not None:
                active_error.add_note("Owner closure evidence write failed: " + repr(write_error))
            else:
                raise
        if invocation is not None and closure["closure"] == "unknown":
            if active_error is not None:
                active_error.add_note("Owned unit or cgroup closure is unconfirmed")
            else:
                raise RuntimeError("Owned unit or cgroup closure is unconfirmed")
    return terminal


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--unit")
    args = parser.parse_args(argv)
    _need(SOURCE_ONLY is False, "held source requires independent review before execution")
    if args.child:
        _need(
            type(args.unit) is str
            and re.fullmatch(r"jev-link-candidate-[0-9a-f]{32}\.service", args.unit),
            "fresh owned unit required",
        )
        return _child(args.plan, args.plan_sha256, args.unit)
    _need(args.unit is None, "parent cannot choose an existing unit")
    terminal = supervise(args.plan, args.plan_sha256)
    return 0 if _successful_exit(terminal) else 1


if __name__ == "__main__":
    raise SystemExit(main())
