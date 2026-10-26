"""Context/tiny-operator GPU mechanism probe; dry by default, no models."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import resource
import struct
import time
from pathlib import Path

PROFILE = "gpu-context-probe-fixture-v2"
EXPECTED_GPU = "GPU-e3429caa-b057-cbe0-917d-d3b85b313be7"
ALLOCATOR_CAP_BYTES = 64 * 1024**2
MAX_FILE = 1024**2
PLANNED_STAGES = [
    "stdlib-before-torch",
    "torch-imported-preinit",
    "cuda-init-returned",
    "cuda-init-synchronized",
    "own-allocator-cap-set",
    "tiny-allocation-fill-enqueued",
    "tiny-fill-synchronized",
    "tiny-matmul-synchronized",
    "tiny-readback-oracle",
    "own-tensors-released-empty-cache",
]


class DeadlinePartial(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise ValueError(message)


def ordinary(path):
    path = Path(path).absolute()
    need(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary unlinked path required",
    )
    return path


def read(path):
    path = ordinary(path)
    need(path.is_file(), "Regular small selected file required")
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE + 1)
    need(len(data) <= MAX_FILE, "Small source/fixture bound")
    return data


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    need(len(data) <= MAX_FILE, "Small evidence bound")
    with ordinary(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def binary_stat(path):
    s = ordinary(path).stat()
    return {key: getattr(s, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns")}


def selection(path, digest, entry_digest):
    blob = read(path)
    need(sha(blob) == digest, "Independent selected manifest SHA")
    value = json.loads(blob)
    need(
        set(value) == {"profile", "sources", "versions", "binary_stats"}
        and value["profile"] == PROFILE,
        "Exact CPU fixture selection",
    )
    for row in value["sources"].values():
        data = read(row["path"])
        need(
            len(data) == row["size_bytes"] and sha(data) == row["sha256"], "Selected source changed"
        )
    own = value["sources"]["entry"]
    need(
        ordinary(own["path"]) == ordinary(__file__) and own["sha256"] == entry_digest,
        "Independent entry SHA/path",
    )
    for row in value["binary_stats"]:
        need(binary_stat(row["path"]) == row["stat"], "Selected binary stat identity changed")
    return value


def identity():
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": os.getpid(),
        "start_ticks": int(fields[19]),
        "pgid": os.getpgrp(),
        "sid": os.getsid(0),
    }


def exception(error):
    return {
        "type": type(error).__name__,
        "message": str(error)[:2000],
        "notes": getattr(error, "__notes__", []),
    }


def check_deadline(deadline, minimum_seconds=0.25):
    if deadline - time.monotonic() < minimum_seconds:
        raise DeadlinePartial("Insufficient parent-sealed deadline for the next stage")


def memory_snapshot():
    started = time.monotonic()
    status = {}
    for line in read(f"/proc/{os.getpid()}/status").decode().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap"):
            amount, unit = value.split()
            need(unit == "kB", "Linux process memory units")
            status[key + "_bytes"] = int(amount) * 1024
    rows = read(f"/proc/{os.getpid()}/cgroup").decode().splitlines()
    groups = [row[3:] for row in rows if row.startswith("0::")]
    need(
        len(groups) == 1 and groups[0].startswith("/") and ".." not in Path(groups[0]).parts,
        "Own unified cgroup path",
    )
    cgroup = ordinary(Path("/sys/fs/cgroup") / groups[0].lstrip("/"))
    memory = {
        "path": groups[0],
        "memory_current_bytes": int(read(cgroup / "memory.current")),
        "memory_peak_bytes": int(read(cgroup / "memory.peak")),
    }
    memory["memory_stat"] = {
        key: int(value)
        for key, value in (
            line.split() for line in read(cgroup / "memory.stat").decode().splitlines()
        )
        if key in ("anon", "file", "shmem", "kernel", "pgfault", "pgmajfault")
    }
    need(
        all(key in status for key in ("VmRSS_bytes", "RssAnon_bytes", "RssFile_bytes")),
        "Required process RSS breakdown",
    )
    return {
        "process": status,
        "own_cgroup": memory,
        "monotonic_before": started,
        "monotonic_after": time.monotonic(),
        "scope": "Sequential non-atomic own-process/own-cgroup snapshot; page cache release and disk IO bytes are not inferred.",
    }


def normalize_uuid(value):
    need(isinstance(value, str), "CUDA UUID must be original string")
    value = value.removeprefix("GPU-")
    need(
        re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value) is not None,
        "Canonical CUDA UUID",
    )
    return value


def cuda_snapshot(torch, metrics_allowed):
    started = time.monotonic()
    initialized = None if torch is None else torch.cuda.is_initialized()
    value = {
        "initialized": initialized,
        "metrics_seconds": 0.0,
        "free_bytes": None,
        "total_bytes": None,
        "allocated_bytes": None,
        "reserved_bytes": None,
        "max_allocated_bytes": None,
        "max_reserved_bytes": None,
    }
    if not metrics_allowed:
        value["scope"] = (
            "CUDA metrics unmeasured before first explicit synchronize; no lazy-initializing query invoked."
        )
        return value
    need(initialized is True, "GPU metrics require prior explicit init and synchronize")
    free, total = torch.cuda.mem_get_info(0)
    need(0 <= free <= total and total > 0, "Actual CUDA global free/total query")
    value.update(
        free_bytes=int(free),
        total_bytes=int(total),
        allocated_bytes=torch.cuda.memory_allocated(0),
        reserved_bytes=torch.cuda.memory_reserved(0),
        max_allocated_bytes=torch.cuda.max_memory_allocated(0),
        max_reserved_bytes=torch.cuda.max_memory_reserved(0),
        metrics_seconds=time.monotonic() - started,
        scope="Global CUDA free is shared, not an owned-process delta; allocator counters omit context/driver allocations. Metrics may perturb timing or synchronize. Never add free bytes to host MemAvailable.",
    )
    return value


def stage(output, result, name, torch=None, metrics_allowed=False, details=None, completed=True):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    memory = memory_snapshot()
    host = {}
    for line in read("/proc/meminfo").decode().splitlines():
        key, _, value = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            amount, unit = value.split()
            need(unit == "kB", "Host memory units")
            host[key + "_bytes"] = int(amount) * 1024
    gpu_error = None
    try:
        gpu = cuda_snapshot(torch, metrics_allowed)
    except BaseException as error:
        gpu_error = error
        gpu = {
            "initialized": None if torch is None else torch.cuda.is_initialized(),
            "metrics_error": exception(error),
        }
    record = {
        "stage": name,
        "elapsed_seconds": time.monotonic() - result["started_monotonic"],
        "monotonic": time.monotonic(),
        "peak_RSS_bytes": usage.ru_maxrss * 1024,
        "ru_minflt": usage.ru_minflt,
        "ru_majflt": usage.ru_majflt,
        "memory": memory,
        "host": host,
        "CUDA": gpu,
        "details": details,
        "snapshot_scope": "Sequential non-atomic own process/cgroup, host and global CUDA samples; owner admission remains external.",
    }
    path = output / ("stage-%02d.json" % len(result["stages"]))
    write(path, record)
    result["stages"].append(
        {"path": str(path), "sha256": sha(read(path)), "size_bytes": path.stat().st_size}
    )
    if completed and gpu_error is None:
        result["completed_stages"].append(name)
    if gpu_error is not None:
        raise gpu_error


def release_owned(torch, owned, timings, primary):
    errors = []

    def retain(error):
        nonlocal primary
        errors.append(exception(error))
        if primary is None:
            primary = error
        else:
            primary.add_note("Own GPU cleanup also failed: " + str(error))

    if torch.cuda.is_initialized():
        started = time.monotonic()
        try:
            torch.cuda.synchronize(0)
        except BaseException as error:
            retain(error)
        finally:
            timings["cleanup_sync_seconds"] = time.monotonic() - started
    owned.clear()
    gc.collect()
    if torch.cuda.is_initialized():
        started = time.monotonic()
        try:
            torch.cuda.empty_cache()
        except BaseException as error:
            retain(error)
        finally:
            timings["own_empty_cache_seconds"] = time.monotonic() - started
    return primary, errors


def run_torch(output, selected, result, deadline):
    check_deadline(deadline, 2)
    started = time.monotonic()
    import torch

    result["timings"]["torch_import_seconds"] = time.monotonic() - started
    result["versions"] = {"torch": str(torch.__version__)}
    need(
        str(torch.__version__) == selected["versions"]["torch"]
        and set(selected["versions"]) == {"torch"},
        "Selected actual Torch version",
    )
    need(
        ordinary(torch.__file__) == ordinary(selected["sources"]["torch"]["path"]),
        "Selected Torch package origin",
    )
    need(torch.cuda.is_initialized() is False, "No initialization before explicit probe stage")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    stage(output, result, "torch-imported-preinit", torch)
    owned, primary, metrics_allowed = [], None, False

    def timed(name, operation):
        before = time.monotonic()
        try:
            return operation()
        finally:
            result["timings"][name] = time.monotonic() - before

    try:
        check_deadline(deadline, 2)
        timed("init_call_seconds", torch.cuda.init)
        need(torch.cuda.is_initialized() is True, "PyTorch CUDA lazy initialization returned")
        result["CUDA"]["initialized_after_init_call"] = True
        stage(
            output,
            result,
            "cuda-init-returned",
            torch,
            details={
                "scope": "PyTorch lazy-init returned; no GPU metrics yet and no complete physical context cost claim."
            },
        )
        check_deadline(deadline, 2)
        timed("init_sync_seconds", lambda: torch.cuda.synchronize(0))
        metrics_allowed = True
        result["GPU_context_measured"] = True
        started = time.monotonic()
        need(torch.cuda.device_count() == 1, "Exactly one visible GPU")
        props = torch.cuda.get_device_properties(0)
        raw_uuid = str(props.uuid)
        result["CUDA"]["native_uuid_type"] = type(props.uuid).__name__
        result["CUDA"]["device_uuid_raw"] = raw_uuid
        normalized = normalize_uuid(raw_uuid)
        need(
            normalized == normalize_uuid(EXPECTED_GPU), "Logical0 joins selected physical GPU UUID"
        )
        result["CUDA"].update(
            logical_device_index=0,
            device_uuid_raw=raw_uuid,
            device_uuid_normalized=normalized,
            device_name=props.name,
            is_integrated=int(props.is_integrated),
            device_total_memory_bytes=int(props.total_memory),
            allocator_backend=torch.cuda.memory.get_allocator_backend(),
            identity_query_seconds=time.monotonic() - started,
        )
        stage(output, result, "cuda-init-synchronized", torch, True, {"identity": result["CUDA"]})
        check_deadline(deadline, 2)
        need(
            props.total_memory >= ALLOCATOR_CAP_BYTES,
            "Known positive total for fixed own allocator cap",
        )
        fraction = ALLOCATOR_CAP_BYTES / props.total_memory
        timed(
            "allocator_cap_set_seconds",
            lambda: torch.cuda.set_per_process_memory_fraction(fraction, 0),
        )
        actual_fraction = torch.cuda.get_per_process_memory_fraction(0)
        need(
            math.isclose(actual_fraction, fraction, rel_tol=1e-6, abs_tol=1e-12),
            "Allocator fraction API roundtrip",
        )
        result["own_allocator_cap"] = {
            "requested_bytes": ALLOCATOR_CAP_BYTES,
            "requested_fraction": fraction,
            "actual_fraction": actual_fraction,
            "scope": "Own PyTorch caching allocator only. Not a context/driver/cuBLAS/physical UMA or cgroup hard cap. Never enlarged on failure.",
        }
        stage(output, result, "own-allocator-cap-set", torch, True, result["own_allocator_cap"])
        check_deadline(deadline, 2)
        started = time.monotonic()
        with torch.inference_mode():
            owned.append(torch.ones((16, 16), dtype=torch.bfloat16, device="cuda:0"))
            owned.append(torch.ones((16, 16), dtype=torch.bfloat16, device="cuda:0"))
        result["timings"]["allocation_fill_enqueue_seconds"] = time.monotonic() - started
        need(
            all(
                x.dtype == torch.bfloat16
                and x.device.type == "cuda"
                and x.device.index == 0
                and list(x.shape) == [16, 16]
                for x in owned
            ),
            "Only two tiny BF16 inputs on selected GPU",
        )
        stage(
            output,
            result,
            "tiny-allocation-fill-enqueued",
            torch,
            True,
            {
                "input_bytes": 1024,
                "first_kernel_scope": "ones fill may execute before matmul; this is not a claim that matmul is the first CUDA kernel.",
            },
        )
        check_deadline(deadline, 2)
        timed("allocation_fill_sync_seconds", lambda: torch.cuda.synchronize(0))
        stage(output, result, "tiny-fill-synchronized", torch, True)
        check_deadline(deadline, 2)
        started = time.monotonic()
        with torch.inference_mode():
            owned.append(torch.matmul(owned[0], owned[1]))
        result["timings"]["matmul_enqueue_seconds"] = time.monotonic() - started
        timed("matmul_sync_seconds", lambda: torch.cuda.synchronize(0))
        result["timings"]["matmul_enqueue_plus_sync_seconds"] = time.monotonic() - started
        stage(output, result, "tiny-matmul-synchronized", torch, True)
        check_deadline(deadline, 2)
        started = time.monotonic()
        cpu_output = owned[2].to(device="cpu")
        result["timings"]["tiny_CPU_readback_seconds"] = time.monotonic() - started
        need(
            cpu_output.dtype == torch.bfloat16
            and cpu_output.device.type == "cpu"
            and list(cpu_output.shape) == [16, 16],
            "Exact tiny output dtype/shape",
        )
        expected = torch.full((16, 16), 16, dtype=torch.bfloat16, device="cpu")
        need(
            torch.equal(cpu_output, expected),
            "Closed-form ones16x16 matmul oracle must be exactly16",
        )
        actual_bytes = bytes(cpu_output.view(torch.uint8).reshape(-1).tolist())
        expected_bytes = struct.pack("=H", 0x4180) * 256
        need(actual_bytes == expected_bytes, "Independent BF16 bit-pattern oracle")
        result["tiny_operator"] = {
            "dtype": "bfloat16",
            "shape": [16, 16],
            "completed": True,
            "all_exact_16": True,
            "readback_bytes": len(actual_bytes),
            "output_sha256": sha(actual_bytes),
            "closed_form_expected_sha256": sha(expected_bytes),
            "scope": "Tiny fill/matmul and512B CPU readback only; no model transfer, layer, accuracy or throughput qualification.",
        }
        stage(output, result, "tiny-readback-oracle", torch, True, result["tiny_operator"])
        del cpu_output, expected, actual_bytes, expected_bytes
        check_deadline(deadline)
    except BaseException as error:
        primary = error
    finally:
        primary, cleanup_errors = release_owned(torch, owned, result["timings"], primary)
        if torch.cuda.is_initialized():
            try:
                stage(
                    output,
                    result,
                    "own-tensors-released-empty-cache"
                    if primary is None
                    else "failure-own-tensors-release-empty-cache",
                    torch,
                    metrics_allowed and deadline - time.monotonic() >= 0.25,
                    {
                        "context_destroyed": False,
                        "scope": "Only unused own allocator cache released; authenticated owner process closure is still required.",
                    },
                    completed=primary is None,
                )
            except BaseException as error:
                cleanup_errors.append(exception(error))
                if primary is None:
                    primary = error
                else:
                    primary.add_note("Own release/cache cleanup also failed: " + str(error))
        result["cleanup_errors"] = cleanup_errors
        result["CUDA"]["initialized_final"] = torch.cuda.is_initialized()
    if primary is not None:
        raise primary
    check_deadline(deadline)
    need(
        result["completed_stages"] == PLANNED_STAGES
        and result["CUDA"]["initialized_final"] is True,
        "Every planned stage and initialized CUDA state required",
    )
    result["context_and_tiny_operator_completed"] = True
    result["mechanism_checks_completed"] = True


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--profile")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--entry-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.run:
        print(
            json.dumps(
                {
                    "profile": PROFILE,
                    "mode": "DRY_NO_TORCH_GPU_MODEL_OR_WRITES",
                    "planned_stages": PLANNED_STAGES,
                }
            )
        )
        return 0
    need(
        args.profile == PROFILE
        and args.selection
        and args.selection_sha256
        and args.entry_sha256
        and args.output,
        "Explicit profile/source/output selection",
    )
    need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == EXPECTED_GPU
        and "ACCELERATE_USE_CPU" not in os.environ,
        "Explicit single selected GPU environment",
    )
    need(
        all(
            os.environ.get(k) == "1"
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        )
        and os.environ.get("USE_TF") == "0"
        and os.environ.get("USE_FLAX") == "0",
        "Single CPU thread and unrelated frameworks disabled",
    )
    deadline = float(os.environ["JEV_GPU_PROBE_DEADLINE_MONOTONIC"])
    need(math.isfinite(deadline) and deadline > 0, "Finite parent-sealed deadline")
    selected = selection(args.selection, args.selection_sha256, args.entry_sha256)
    hardware = json.loads(read(selected["sources"]["hardware_identity"]["path"]))
    need(
        hardware["exit_code"] == 0
        and hardware["stdout"].split(",", 1)[0] == EXPECTED_GPU
        and hardware["boot_id"] == read("/proc/sys/kernel/random/boot_id").decode().strip(),
        "Selected root hardware identity and boot",
    )
    output = ordinary(args.output)
    need(not output.exists(), "Fresh nonexistent output required")
    output.mkdir(mode=0o700)
    started = time.monotonic()
    result = {
        "profile": PROFILE,
        "CPU_ONLY": False,
        "GPU_context_measured": False,
        "GPU_model_transfer_measured": False,
        "no_model_loaded": True,
        "model_loaded": False,
        "model_header_read": False,
        "model_payload_read": False,
        "model_mmap": False,
        "large_model_runtime_or_quality_qualified": False,
        "passed": False,
        "completed": False,
        "mechanism_checks_completed": False,
        "context_and_tiny_operator_completed": False,
        "planned_stages": PLANNED_STAGES,
        "completed_stages": [],
        "stages": [],
        "timings": {},
        "tiny_operator": {"completed": False},
        "CUDA": {
            "logical_device_index": 0,
            "selected_uuid": EXPECTED_GPU,
            "initialized_final": None,
        },
        "identity": identity(),
        "started_monotonic": started,
        "parent_deadline_monotonic": deadline,
        "entry_sha256": args.entry_sha256,
        "source_selection_sha256": args.selection_sha256,
        "status": "started",
        "limits": "Context and tiny BF16 fill/matmul only; no model/header/payload/mmap,UMA physical cap,12B,8192,20s,N3/Q95 or benchmark qualification.",
    }
    write(output / "intent.json", result)
    primary = None
    try:
        stage(output, result, "stdlib-before-torch")
        run_torch(output, selected, result, deadline)
        need(
            result["mechanism_checks_completed"]
            and result["context_and_tiny_operator_completed"]
            and result["completed_stages"] == PLANNED_STAGES,
            "All planned context/operator stages required",
        )
    except BaseException as error:
        primary = error
    finally:
        try:
            need(
                selection(args.selection, args.selection_sha256, args.entry_sha256) == selected,
                "Final selected sources and native binary stat identities",
            )
            result["source_integrity_final"] = True
        except BaseException as error:
            result["source_integrity_final"] = False
            if primary is None or isinstance(primary, DeadlinePartial):
                if primary is not None:
                    error.add_note("Earlier deadline partial: " + str(primary))
                primary = error
            else:
                primary.add_note("Final source refusal: " + str(error))
        partial = isinstance(primary, DeadlinePartial) and not result.get("cleanup_errors")
        result.update(
            status="partial_deadline"
            if partial
            else "failed"
            if primary is not None
            else "completed_mechanism_observations",
            passed=primary is None,
            completed=primary is None,
            elapsed_seconds=time.monotonic() - started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            identity_after=identity(),
            error=None if primary is None else exception(primary),
        )
        if primary is not None:
            result["mechanism_checks_completed"] = result["context_and_tiny_operator_completed"] = (
                False
            )
        write(output / "receipt.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_stages": len(result["completed_stages"]),
                "model_qualified": False,
            }
        )
    )
    return 2 if partial else 1 if primary is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
