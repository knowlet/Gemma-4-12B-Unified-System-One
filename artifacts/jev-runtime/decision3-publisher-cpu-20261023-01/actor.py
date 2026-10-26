#!/usr/bin/env python3
"""Dry-default publisher Edge CPU actor; the existing generic owner contains execution."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
UNIT = "jev-d3-llama-cpp-cpu4-20261023-03.service"
PROFILE = "decision3-publisher-edge-cpu-batch1-smoke-v1"
RESERVE = 26_119_172_916
MEMORY_MAX = 2_147_483_648
WALL_SECONDS = 120
FORWARD_SECONDS = 20
THREADS = 4
RUNTIME_PYTHON = Path("/home/albert_hsieh/ComfyUI/.venv/bin/python")
SNAPSHOT = ROOT / "artifacts/jev-local-hf/hub/models--vllm-sr--d3-edge/snapshots/cab71e6ad3461b8086d51ddb572c656babd63b0b"
OVERLAY = ROOT / "artifacts/jev-runtime/overlay"
START_ENV = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "ACCELERATE_USE_CPU": "true"}
CPU_ENV = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false", "HF_ENABLE_PARALLEL_LOADING": "false", "HF_PARALLEL_LOADING_WORKERS": "1", "USE_HUB_KERNELS": "0", "D3_FAST": "0", "D3_FUSED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
MODEL_MODULES = {"torch", "transformers", "safetensors", "numpy", "s1"}
KERNEL_NAMES = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update")
VERSIONS = {"torch": "2.13.0+cu130", "transformers": "5.17.0", "safetensors": "0.8.0", "huggingface_hub": "1.33.0", "accelerate": "1.15.0"}


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
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp-" + str(threading.get_ident()))
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def process_identity():
    stat = Path("/proc/self/stat").read_text()
    tail = stat[stat.rindex(")") + 2:].split()
    return {"pid": os.getpid(), "starttime_ticks": int(tail[19]), "pgid": os.getpgrp(), "sid": os.getsid(0), "cgroup": Path("/proc/self/cgroup").read_text().strip(), "affinity": sorted(os.sched_getaffinity(0)), "executable": str(Path(sys.executable).resolve()), "environment": {key: os.environ.get(key) for key in {**START_ENV, **CPU_ENV}}}


def validate_owner(identity, limits):
    lines = identity["cgroup"].splitlines()
    require(len(lines) == 1 and lines[0].startswith("0::/"), "expected unified cgroup identity")
    require(lines[0].split("::", 1)[1].endswith("/" + UNIT), "execution requires selected existing owner unit")
    require(identity["affinity"] == [16, 17, 18, 19], "owned CPU affinity differs")
    require(limits["memory.max"] == str(MEMORY_MAX) and limits["memory.swap.max"] == "0", "owned memory/swap limits differ")
    quota, period = map(int, limits["cpu.max"].split())
    require(quota == 4 * period and period > 0, "owned CPU quota differs")
    require(all(identity["environment"].get(key) == value for key, value in START_ENV.items()), "CPU/offline owner environment differs")


def owned_limits(identity):
    relative = identity["cgroup"].split("::", 1)[1].lstrip("/")
    group = Path("/sys/fs/cgroup") / relative
    return {name: (group / name).read_text().strip() for name in ("memory.max", "memory.swap.max", "cpu.max")}


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
    return {**values, "reserve_bytes": reserve, "required_available_bytes": required, "swap_credit_bytes": 0, "cache_credit_bytes": 0, "already_loaded_credit_bytes": 0}


def check_record(record, *, full):
    path = Path(record["path"])
    info = path.stat()
    require(info.st_size == record["size"], "file size differs: " + str(path))
    if "stat" in record:
        require({key: getattr(info, key) for key in record["stat"]} == record["stat"], "file identity changed: " + str(path))
    if full:
        require(sha(path) == record["sha256"], "file SHA256 differs: " + str(path))


def load_plan(path, expected_sha):
    require(expected_sha and sha(path) == expected_sha, "selected plan SHA256 differs")
    plan = read_json(path)
    require(plan["profile"] == PROFILE and plan["actor_sha256"] == sha(__file__), "selected actor/profile changed")
    require(plan["snapshot"] == str(SNAPSHOT) and plan["revision"] == "cab71e6ad3461b8086d51ddb572c656babd63b0b", "selected snapshot differs")
    require(plan["wall_seconds"] == WALL_SECONDS and plan["forward_seconds"] == FORWARD_SECONDS, "deadline declaration differs")
    require(plan["reserve_bytes"] == RESERVE and plan["memory_max_bytes"] == MEMORY_MAX, "budget declaration differs")
    require(plan["batch_size"] == 1 and plan["threads"] == THREADS, "CPU/batch profile differs")
    require(plan["runtime_python"] == str(RUNTIME_PYTHON) and Path(sys.executable).resolve() == RUNTIME_PYTHON.resolve(), "selected interpreter differs")
    require(plan["environment"] == {**START_ENV, **CPU_ENV}, "selected environment declaration differs")
    records = plan["model_files"]
    require(len(records) == 20 and len({row["name"] for row in records}) == 20, "runtime file inventory differs")
    for row in records:
        require(row["path"] == str(SNAPSHOT / row["name"]) and Path(row["name"]).name == row["name"], "unsafe model file path")
    return plan


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


def validate_runtime(runtime, torch):
    require(str(runtime.device) == "cpu", "publisher runtime is not CPU")
    require(runtime.batch_size == 1 and runtime.temperature == 1.0 and runtime.max_length is None and runtime.readout_dtype == "float32" and runtime.attention_mode == "noncausal_full_attention" and not runtime.permutation_average, "publisher numerical policy differs")
    require(all(p.device.type == "cpu" and p.dtype == torch.bfloat16 for p in runtime.backbone.parameters()), "backbone is not entirely CPU BF16")
    require(runtime.readout.device.type == "cpu" and runtime.readout.dtype == torch.float32 and tuple(runtime.readout.shape) == (255, 1024), "readout is not original CPU FP32 255x1024")
    require(runtime.fast is None and not torch.cuda.is_initialized(), "publisher activated GPU path")
    require(set(runtime.kernels) >= {"torch_chunk_gated_delta_rule", "causal_conv1d_fn"} and all(value == "torch-reference" for key, value in runtime.kernels.items() if key != "fla"), "publisher kernel report is not Torch reference")
    config = runtime.backbone.config.text_config
    require(config.hidden_size == 1024 and config.num_hidden_layers == 12 and config.layer_types == ["linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "full_attention", "full_attention", "full_attention", "full_attention"], "Edge architecture differs")


def validate_answer(raw, request, tokens):
    require(isinstance(raw, dict) and raw.get("model") == "d3-edge", "unexpected actual publisher model identity")
    require(set(raw.get("answers", {})) == {"q"}, "response question inventory differs")
    answer = raw["answers"]["q"]
    require(answer.get("type") == "choice", "response is not a choice")
    keys = list(request["questions"]["q"]["criteria"])
    probabilities = answer.get("probabilities")
    require(isinstance(probabilities, dict) and list(probabilities) == keys, "candidate ordering/inventory differs")
    values = list(probabilities.values())
    require(all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in values), "invalid probabilities")
    require(abs(sum(values) - 1) <= 1e-6, "distribution does not sum to one")
    require(answer.get("choice") == keys[max(range(len(values)), key=values.__getitem__)], "choice is not legal argmax")
    entropy = -sum(value * math.log(value) for value in values if value > 0)
    expected_confidence = 1 - entropy / math.log(len(keys))
    confidence = answer.get("confidence")
    require(type(confidence) in (int, float) and math.isfinite(confidence) and abs(confidence - expected_confidence) <= 1e-6, "confidence entropy differs")
    require(raw.get("usage") == {"input_tokens": tokens, "output_tokens": 0}, "native token usage differs")
    return {"contract_valid": True, "quality_or_expected_answer_claim": False}


class Deadline:
    def __init__(self, seconds, output, stage):
        self.seconds, self.output, self.stage = seconds, output, stage
        self.started = time.monotonic()
        self.timer = threading.Timer(seconds, self.expire)
        self.timer.daemon = True

    def expire(self):
        try:
            write_json(self.output / (self.stage + "-timeout.json"), {"status": "deadline_exceeded", "stage": self.stage, "deadline_seconds": self.seconds, "elapsed_seconds": time.monotonic() - self.started, "pid": os.getpid(), "exit_code": 124})
        finally:
            os._exit(124)

    def __enter__(self):
        self.timer.start()
        return self

    def __exit__(self, *args):
        self.timer.cancel()


def load_publisher_cpu(plan, output, state):
    """Single real constructor shared by smoke and a later separately selected score worker."""
    require(state.get("model_loads", 0) == 0, "second model constructor is forbidden")
    identity = process_identity()
    validate_owner(identity, owned_limits(identity))
    require(not MODEL_MODULES.intersection(sys.modules), "loader began with model libraries imported")
    state["host_before_authentication"] = host_gate()
    for record in plan["runtime_files"]:
        check_record(record, full="sha256" in record)
    check_record(plan["manifest"], full=True)
    for record in plan["model_files"]:
        check_record(record, full=True)
    state["authenticated_model_files"] = [{"name": row["name"], "size": row["size"], "sha256": row["sha256"]} for row in plan["model_files"]]
    state["status"] = "model_files_fully_authenticated"
    write_json(output / "state.json", state)
    os.environ.update(CPU_ENV)
    # This exact existing overlay is local to this actor process.
    sys.path.insert(0, str(OVERLAY))
    os.environ["HF_HOME"] = str(output / "hf-cache")
    os.environ["HF_MODULES_CACHE"] = str(output / "hf-cache/modules")
    for package in ("fla", "causal_conv1d", "kernels"):
        require(importlib.util.find_spec(package) is None, "GPU-only optional package visible: " + package)
    from importlib import metadata
    state["package_versions"] = {name: metadata.version(name) for name in VERSIONS}
    require(state["package_versions"] == VERSIONS, "selected runtime package versions differ")
    state["actual_environment"] = {key: os.environ.get(key) for key in {**START_ENV, **CPU_ENV}}
    state["status"] = "importing_torch"
    write_json(output / "state.json", state)
    import torch
    require(Path(torch.__file__).resolve() == RUNTIME_PYTHON.parent.parent / "lib/python3.13/site-packages/torch/__init__.py", "Torch imported from unselected environment")
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    require(torch.get_num_threads() == THREADS and torch.get_num_interop_threads() == 1 and not torch.cuda.is_initialized(), "CPU threading or CUDA initialization differs")
    # Authenticated local publisher module; no AutoModel remote-code cache or wrapper math.
    sys.path.insert(0, str(SNAPSHOT))
    from d3_runtime import D3, kernel_report
    from transformers.models.qwen3_5 import modeling_qwen3_5
    require(Path(modeling_qwen3_5.__file__).resolve() == (OVERLAY / "transformers/models/qwen3_5/modeling_qwen3_5.py").resolve(), "Transformers imported from unselected environment")
    state["actual_bindings"] = {name: binding_identity(getattr(modeling_qwen3_5, name)) for name in KERNEL_NAMES}
    state["publisher_kernel_report_before_load"] = kernel_report()
    require(all(value == "torch-reference" for key, value in state["publisher_kernel_report_before_load"].items() if key != "fla"), "publisher pre-load kernel binding differs")
    require(not torch.cuda.is_initialized(), "CUDA initialized before model load")
    state["status"] = "loading_publisher_cpu"
    write_json(output / "state.json", state)
    model_started = time.monotonic()
    runtime = D3.from_pretrained(str(SNAPSHOT), device="cpu", batch_size=1, verify="full", local_files_only=True, permutation_average=False)
    state["model_loads"] = 1
    state["model_load_seconds"] = time.monotonic() - model_started
    validate_runtime(runtime, torch)
    state["native_provenance"] = runtime.provenance()
    state["native_runtime_info"] = runtime.runtime_info()
    return runtime, torch


def execute(plan_path, expected_sha, output):
    require(not MODEL_MODULES.intersection(sys.modules), "actor began with model libraries imported")
    require(sys.flags.no_user_site and sys.dont_write_bytecode, "runtime requires -B -s")
    plan = load_plan(plan_path, expected_sha)
    # Refuse before any payload/source import unless already inside the selected owner.
    identity = process_identity()
    limits = owned_limits(identity)
    validate_owner(identity, limits)
    gate = host_gate()
    output = Path(output).absolute()
    require(output.is_relative_to(HERE.parent) and not output.exists(), "new output must be in runtime artifact tree")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = {"profile": PROFILE, "status": "authenticating", "plan_sha256": expected_sha, "process": identity, "owned_limits": limits, "host_startup_admission": gate, "model_loads": 0, "forward_calls": 0, "warmup_calls": 0, "benchmark_calls": 0, "browser_calls": 0, "helper_models": 0, "no_quality_claim": True}
    write_json(output / "state.json", state)
    with Deadline(WALL_SECONDS, output, "actor"):
        try:
            runtime, torch = load_publisher_cpu(plan, output, state)
            request = copy.deepcopy(plan["request"])
            prepared = runtime.prepare(**request)
            require(prepared.keys == ["q"] and not prepared.errors and set(prepared.sequences) == {"q"}, "native preparation rejected synthetic request")
            tokens = len(prepared.sequences["q"])
            require(0 < tokens <= 512, "synthetic input exceeds declared cap; no truncation")
            state["native_input_tokens"] = tokens
            state["full_synthetic_request"] = request
            state["status"] = "running_single_native_request"
            state["forward_calls"] = 1
            write_json(output / "state.json", state)
            forward_started = time.monotonic()
            with Deadline(FORWARD_SECONDS, output, "forward"):
                raw = runtime.system_one(**copy.deepcopy(request))
            elapsed = time.monotonic() - forward_started
            state["raw_response"] = raw
            state["forward_seconds"] = elapsed
            state["status"] = "raw_response_retained"
            write_json(output / "state.json", state)
            require(elapsed <= FORWARD_SECONDS, "native request returned after fixed deadline")
            state["validation"] = validate_answer(raw, request, tokens)
            require(not torch.cuda.is_initialized(), "CUDA initialized during publisher CPU request")
            state["cuda_initialized_final"] = torch.cuda.is_initialized()
            state["torch_threads"] = {"intraop": torch.get_num_threads(), "interop": torch.get_num_interop_threads()}
            state["status"] = "publisher_cpu_single_request_complete_no_benchmark_claim"
            return state
        except BaseException as exc:
            state["status"] = "failed"
            state["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            state["elapsed_seconds"] = time.monotonic() - started
            write_json(output / "state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", type=Path, default=HERE / "plan.json")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"profile": PROFILE, "status": "dry_default_no_model_import_or_payload_read", "plan_path": str(args.plan), "requires_existing_owner": UNIT, "request_count": 1, "batch_size": 1, "threads": THREADS, "memory_max_bytes": MEMORY_MAX, "reserve_bytes": RESERVE, "wall_seconds": WALL_SECONDS, "forward_seconds": FORWARD_SECONDS, "model_execution": False}))
        return
    require(args.output is not None, "execution requires new output directory")
    result = execute(args.plan, args.plan_sha256, args.output)
    print(json.dumps({"profile": PROFILE, "status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
