"""Selected stdlib resource helper extracted from frozen Lite smoke utilities.

No Lite model loader/plan/entry is included; actual publisher NN math remains in
the separately pinned native batch4 actor. Distinct owner07 CPU-coexist policy.
The controller owns strict Comfy HTTP observation; this leaf host_gate retains
only its original full R+2GiB memory check and makes no Comfy HTTP claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

HOST_OBSERVATION_POLICY = "comfy-observed-cpu-coexist-v1"


RESERVE = 26_119_172_916


MEMORY_MAX = 2_147_483_648


RUNTIME_PYTHON = Path("/home/albert_hsieh/ComfyUI/.venv/bin/python")


OVERLAY = ROOT / "artifacts/jev-runtime/overlay"


START_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "ACCELERATE_USE_CPU": "true",
}


CPU_ENV = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_ENABLE_PARALLEL_LOADING": "false",
    "HF_PARALLEL_LOADING_WORKERS": "1",
    "USE_HUB_KERNELS": "0",
    "D3_FAST": "0",
    "D3_FUSED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
}


KERNEL_NAMES = (
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "causal_conv1d_fn",
    "causal_conv1d_update",
)


VERSIONS = {
    "torch": "2.13.0+cu130",
    "transformers": "5.17.0",
    "safetensors": "0.8.0",
    "huggingface_hub": "1.33.0",
    "accelerate": "1.15.0",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def unique(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate JSON key")
        value[key] = item
    return value


def read_json(path):
    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    # Strict serialization happens before any temporary file is created.
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def diagnostic(value):
    raw = repr(value).encode("utf-8", errors="backslashreplace")
    return {
        "encoding": "python_repr_diagnostic_only",
        "repr_prefix": raw[:1_048_576].decode("utf-8", errors="backslashreplace"),
        "repr_bytes": len(raw),
        "repr_sha256": hashlib.sha256(raw).hexdigest(),
        "diagnostic_truncated": len(raw) > 1_048_576,
    }


def ensure_elapsed(began, seconds, stage):
    elapsed = time.monotonic() - began
    require(
        math.isfinite(elapsed) and 0 <= elapsed <= seconds,
        stage + " completed after its fixed deadline",
    )
    return elapsed


def finalize_state(output, state, primary):
    try:
        write_json(output / "state.json", state)
    except BaseException as secondary:
        if primary is None:
            raise
        primary.add_note("Final Lite state write failed: " + repr(secondary))
        try:
            write_json(
                output / "final-evidence-error.json",
                {
                    "status": "failed_no_complete_claim",
                    "primary_error": {"type": type(primary).__name__, "message": str(primary)},
                    "secondary_error": {
                        "type": type(secondary).__name__,
                        "message": str(secondary),
                    },
                    "state_diagnostic": diagnostic(state),
                },
            )
        except BaseException as tertiary:
            primary.add_note("Fallback Lite evidence write failed: " + repr(tertiary))


def process_identity():
    stat = Path("/proc/self/stat").read_text()
    tail = stat[stat.rindex(")") + 2 :].split()
    return {
        "pid": os.getpid(),
        "starttime_ticks": int(tail[19]),
        "pgid": os.getpgrp(),
        "sid": os.getsid(0),
        "cgroup": Path("/proc/self/cgroup").read_text().strip(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "executable": str(Path(sys.executable).resolve()),
        "environment": {key: os.environ.get(key) for key in {**START_ENV, **CPU_ENV}},
    }


def owned_limits(identity):
    relative = identity["cgroup"].split("::", 1)[1].lstrip("/")
    group = Path("/sys/fs/cgroup") / relative
    return {
        name: (group / name).read_text().strip()
        for name in ("memory.max", "memory.swap.max", "cpu.max")
    }


def host_gate():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        parts = line.split()
        if parts[0] in ("MemTotal:", "MemAvailable:"):
            values[parts[0][:-1]] = int(parts[1]) * 1024
    reserve = max(24 * 1024**3, (values["MemTotal"] + 4) // 5)
    require(reserve == RESERVE, "host total/reserve differs")
    required = RESERVE + MEMORY_MAX
    require(values["MemAvailable"] >= required, "unchanged reserve plus 2GiB startup floor refused")
    return {
        **values,
        "reserve_bytes": reserve,
        "required_available_bytes": required,
        "swap_credit_bytes": 0,
        "cache_credit_bytes": 0,
        "already_loaded_credit_bytes": 0,
    }


def check_record(record, *, full):
    path = Path(record["path"])
    info = path.stat()
    require(info.st_size == record["size"], "file size differs: " + str(path))
    if "stat" in record:
        require(
            {key: getattr(info, key) for key in record["stat"]} == record["stat"],
            "file identity changed: " + str(path),
        )
    if full:
        require(sha(path) == record["sha256"], "file SHA256 differs: " + str(path))


def binding_identity(fn):
    seen, stack = set(), [fn]
    while stack:
        value = stack.pop()
        if id(value) in seen or not callable(value):
            continue
        seen.add(id(value))
        module = getattr(value, "__module__", "") or ""
        if module.startswith(("fla", "causal_conv1d", "kernels")):
            raise RuntimeError("GPU-only bound kernel: " + module)
        for cell in getattr(value, "__closure__", None) or ():
            try:
                stack.append(cell.cell_contents)
            except ValueError:
                pass
    return "torch-reference"


class Deadline:
    def __init__(self, seconds, output, stage):
        self.seconds, self.output, self.stage = seconds, output, stage
        self.started = time.monotonic()
        self.timer = threading.Timer(seconds, self.expire)
        self.timer.daemon = True

    def expire(self):
        try:
            write_json(
                self.output / (self.stage + "-timeout.json"),
                {
                    "status": "deadline_exceeded",
                    "stage": self.stage,
                    "deadline_seconds": self.seconds,
                    "elapsed_seconds": time.monotonic() - self.started,
                    "pid": os.getpid(),
                    "exit_code": 124,
                },
            )
        finally:
            os._exit(124)

    def __enter__(self):
        self.timer.start()
        return self

    def __exit__(self, *args):
        self.timer.cancel()
