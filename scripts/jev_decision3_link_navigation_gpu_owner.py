"""Supervise one link-navigation GPU shard in a fresh user transient unit.

Source-only pending independent review. Parent independently observes ComfyUI
and physical memory, including while the child is blocked in native forward.
It never sends a ComfyUI mutation. All captures remain pending FP64 audit.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import jev_stage_memory_profile_v2 as systemd

SOURCE_ONLY = True
SCHEMA = "decision3-link-navigation-gpu-owner-plan-v1"
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
        and 1 <= plan["shard_size"] <= 40
        and type(plan["shard_index"]) is int
        and plan["shard_index"] >= 0,
        "bounded shard selection required",
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


def _child(plan_path, plan_sha, unit):
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
        "schema": "decision3-link-navigation-gpu-owner-worker-v1",
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
        result = features.capture_shard(
            admitted,
            shard_size=plan["shard_size"],
            shard_index=plan["shard_index"],
            backend=backend,
            torch=torch,
            source_pins=plan["source_pins"],
            cache_root=plan["cache_root"],
            guard=check,
            deadline=check,
        )
        import numpy as np

        def save_tensor(name, value):
            path = output / name
            with path.open("xb") as stream:
                np.save(stream, value.detach().cpu().numpy(), allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            data = path.read_bytes()
            return {"path": name, "sha256": _sha(data), "size_bytes": len(data)}

        baseline = save_tensor("baseline.npy", result.pop("baseline"))
        cases = []
        for index, case in enumerate(result.pop("cases")):
            feature = save_tensor(f"features-{index:04d}.npy", case.pop("features"))
            raw = case.pop("raw_response")
            raw_name = f"raw-{index:04d}.json"
            raw_bytes = _json(raw)
            _write_new(output / raw_name, raw_bytes)
            cases.append(
                {
                    **case,
                    "features": feature,
                    "raw_response": {
                        "path": raw_name,
                        "sha256": _sha(raw_bytes),
                        "size_bytes": len(raw_bytes),
                    },
                }
            )
        status["capture"] = {**result, "baseline": baseline, "cases": cases}
        status["status"] = "captured_pending_independent_fp64_audit"
    except BaseException as error:
        primary = error
        status.update(status="failed_no_accepted_features", error_type=type(error).__name__)
    finally:
        try:
            if backend is not None:
                backend.close()
        except BaseException as error:
            status["close_error_type"] = type(error).__name__
            status["status"] = "failed_cleanup_unconfirmed"
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
    unit = "jev-link-feature-" + secrets.token_hex(16) + ".service"
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
                worker.get("status") == "captured_pending_independent_fp64_audit"
                and worker.get("invocation") == invocation
                and worker.get("plan_sha256") == plan_sha
                and worker.get("final_cuda_allocated") == 0,
                "successful worker and CUDA cleanup evidence required",
            )
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
            "schema": "decision3-link-navigation-gpu-owner-closure-v1",
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
            and re.fullmatch(r"jev-link-feature-[0-9a-f]{32}\.service", args.unit),
            "fresh owned unit required",
        )
        return _child(args.plan, args.plan_sha256, args.unit)
    _need(args.unit is None, "parent cannot choose an existing unit")
    terminal = supervise(args.plan, args.plan_sha256)
    return 0 if _successful_exit(terminal) else 1


if __name__ == "__main__":
    raise SystemExit(main())
