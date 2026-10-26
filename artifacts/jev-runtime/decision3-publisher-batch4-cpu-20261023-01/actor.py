#!/usr/bin/env python3
"""Dry-default native batch-four CPU probe and reusable publisher backend."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
HELPER = ROOT / "artifacts/jev-runtime/decision3-publisher-lite-cpu-20261023-01/actor.py"
HELPER_SHA = "33859d8b57bc906e477dce6ebf9d76ba4175ce775abb55cd9c42510f6de0bc9e"
SOURCE_AUDIT_SHA = "ae299d87cbf7338bb9ff29055efc335e96a44b73a29370e08428aad2fb2b68d3"
UNIT = "jev-d3-llama-cpp-cpu4-3g-long-20261023-05.service"
OWNER_PROFILE = "llama-cpp-cpu4-3g-long-command-owned-v1"
OWNER_CONTROLLER_SHA = "81a492bb00a430c8728e0fcafe94a4220961312e521e8c0d7863c7f601c8ef2e"
RESERVE = 26_119_172_916
MEMORY_MAX = 3_221_225_472
WALL_SECONDS = 120
CALL_SECONDS = 20
BATCH_SIZE = 4
CALLS = 2
HEADS = ("color", "shape", "number", "direction")
MODEL_MODULES = {"torch", "transformers", "safetensors", "numpy", "s1"}
MODELS = {
    "edge": {
        "model_id": "vllm-sr/d3-edge",
        "name": "d3-edge",
        "revision": "cab71e6ad3461b8086d51ddb572c656babd63b0b",
        "manifest_sha256": "3bae8989a6feb36a66bf4d8c44112dc63199d0dc69f564dc895fe368b1e08e59",
        "model_sha256": "6e00d04abbdbdf6d35c9e12065769d9edaacc3bfe9404583891797718d239dc8",
        "layers": [
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "full_attention",
            "full_attention",
            "full_attention",
        ],
    },
    "lite": {
        "model_id": "vllm-sr/d3-lite",
        "name": "d3-lite",
        "revision": "9c10703193afc7c7d28c88201c0364bfbbf58320",
        "manifest_sha256": "31f0ff94ff26619c0a063ccac803edcfe34c56c92e5c9c50b703b62893f125dd",
        "model_sha256": "fda5cfc7bc9063fbb7cd40b50fa9a0963d83a4790a3d6a11a356833fd1171992",
        "layers": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        * 6,
    },
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_verified(name, path, expected):
    require(
        Path(path).suffix == ".py" and Path(path).stat().st_size < 200_000,
        "small selected Python source required",
    )
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "selected dependency source changed")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_helper():
    return import_verified("_batch4_frozen_generic_helper", HELPER, HELPER_SHA)


def validate_owner(identity, limits, start_env):
    lines = identity["cgroup"].splitlines()
    require(
        len(lines) == 1 and lines[0].startswith("0::/") and lines[0].endswith("/" + UNIT),
        "selected owner05 cgroup required",
    )
    require(identity["affinity"] == [16, 17, 18, 19], "CPU affinity differs")
    require(
        limits["memory.max"] == str(MEMORY_MAX) and limits["memory.swap.max"] == "0",
        "3GiB/swap0 owner limits differ",
    )
    quota, period = map(int, limits["cpu.max"].split())
    require(period > 0 and quota == 4 * period, "CPU quota differs")
    require(
        all(identity["environment"].get(key) == value for key, value in start_env.items()),
        "CPU/offline startup environment differs",
    )


def load_plan(helper, path, expected):
    require(expected and sha(path) == expected, "selected plan SHA differs")
    plan = helper.read_json(path)
    key = plan.get("model_key")
    require(
        key in MODELS and plan.get("model") == MODELS[key],
        "actual selected Edge/Lite identity differs",
    )
    require(
        plan.get("profile") == f"decision3-{key}-publisher-cpu-batch4-synthetic-20261023-v1"
        and plan.get("actor_sha256") == sha(__file__),
        "batch4 profile/source differs",
    )
    require(
        plan.get("helper_sha256") == HELPER_SHA
        and plan.get("source_audit_sha256") == SOURCE_AUDIT_SHA,
        "selected generic helper/AST audit differs",
    )
    require(
        plan.get("owner_unit") == UNIT
        and plan.get("owner_profile") == OWNER_PROFILE
        and plan.get("reserve_bytes") == RESERVE
        and plan.get("memory_max_bytes") == MEMORY_MAX,
        "owner05 profile/budget differs",
    )
    require(
        any(
            row["path"]
            == str(ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261023-05/controller.py")
            and row["sha256"] == OWNER_CONTROLLER_SHA
            for row in plan["owner_sources"]
        ),
        "selected owner05 controller identity differs",
    )
    require(
        plan.get("batch_size") == BATCH_SIZE
        and plan.get("calls") == CALLS
        and plan.get("wall_seconds") == WALL_SECONDS
        and plan.get("call_seconds") == CALL_SECONDS,
        "batch/count/deadline declaration differs",
    )
    require(
        plan.get("runtime_python") == str(helper.RUNTIME_PYTHON)
        and Path(sys.executable).resolve() == helper.RUNTIME_PYTHON.resolve(),
        "selected runtime interpreter differs",
    )
    require(
        plan.get("threads") == 4
        and plan.get("environment") == {**helper.START_ENV, **helper.CPU_ENV},
        "declared CPU environment/threads differs",
    )
    expected_snapshot = (
        ROOT
        / f"artifacts/jev-local-hf/hub/models--vllm-sr--d3-{key}/snapshots/{MODELS[key]['revision']}"
    )
    require(
        plan.get("snapshot") == str(expected_snapshot)
        and plan["manifest"]["sha256"] == MODELS[key]["manifest_sha256"],
        "snapshot/manifest differs",
    )
    records = plan["model_files"]
    require(
        len(records) == 20
        and len({row["name"] for row in records}) == 20
        and all(
            Path(row["name"]).name == row["name"]
            and row["path"] == str(expected_snapshot / row["name"])
            for row in records
        ),
        "complete safe runtime file set required",
    )
    require(
        list(plan["request"]["questions"]) == list(HEADS)
        and all(q["type"] == "choice" for q in plan["request"]["questions"].values()),
        "four ordered independent developer choices required",
    )
    return plan


def authenticate(helper, plan):
    for record in plan["runtime_files"]:
        helper.check_record(record, full="sha256" in record)
    helper.check_record(plan["manifest"], full=True)
    for record in plan["model_files"]:
        helper.check_record(record, full=True)
    for record in plan["owner_sources"]:
        helper.check_record(record, full=True)


def validate_runtime(runtime, torch, model):
    require(
        runtime.model_name == model["name"]
        and runtime.manifest["identity"]["model_sha256"] == model["model_sha256"],
        "actual model identity differs",
    )
    require(
        str(runtime.device) == "cpu"
        and runtime.batch_size == 4
        and runtime.temperature == 1.0
        and runtime.max_length is None
        and runtime.readout_dtype == "float32"
        and runtime.attention_mode == "noncausal_full_attention"
        and not runtime.permutation_average,
        "native batch4 CPU numerical policy differs",
    )
    parameters = list(runtime.backbone.parameters())
    require(
        parameters
        and all(
            p.device.type == "cpu" and p.dtype == torch.bfloat16 and not p.requires_grad
            for p in parameters
        )
        and runtime.backbone.training is False,
        "backbone must be genuine CPU BF16 evaluation weights",
    )
    require(
        runtime.readout.device.type == "cpu"
        and runtime.readout.dtype == torch.float32
        and tuple(runtime.readout.shape) == (255, 1024),
        "original CPU FP32 readout required",
    )
    config = runtime.backbone.config.text_config
    require(
        config.hidden_size == 1024
        and config.num_hidden_layers == len(model["layers"])
        and config.layer_types == model["layers"],
        "actual model architecture differs",
    )
    require(runtime.fast is None and not torch.cuda.is_initialized(), "GPU/fast path activated")
    require(
        set(runtime.kernels) >= {"torch_chunk_gated_delta_rule", "causal_conv1d_fn"}
        and all(
            value == "torch-reference" for key, value in runtime.kernels.items() if key != "fla"
        ),
        "actual reference CPU kernels required",
    )


def load_runtime(helper, plan, output, state):
    """Single actual publisher constructor; reusable by a later selected campaign."""
    require(
        state.get("model_loads", 0) == 0 and not MODEL_MODULES.intersection(sys.modules),
        "one fresh constructor with no preimported model libraries required",
    )
    identity = helper.process_identity()
    validate_owner(identity, helper.owned_limits(identity), helper.START_ENV)
    state["host_before_full_authentication"] = helper.host_gate()
    authenticate(helper, plan)
    state["initial_full_model_manifest_runtime_sources_authenticated"] = True
    source = import_verified("_batch4_source_audit", HERE / "source_audit.py", SOURCE_AUDIT_SHA)
    state["publisher_source_AST_contract"] = source.verify_file(
        Path(plan["snapshot"]) / "d3_runtime.py"
    )
    os.environ.update(helper.CPU_ENV)
    sys.path.insert(0, str(helper.OVERLAY))
    os.environ["HF_HOME"] = str(output / "hf-cache")
    os.environ["HF_MODULES_CACHE"] = str(output / "hf-cache/modules")
    for package in ("fla", "causal_conv1d", "kernels"):
        require(
            importlib.util.find_spec(package) is None, "GPU-only optional kernel package visible"
        )
    from importlib import metadata

    state["package_versions"] = {name: metadata.version(name) for name in helper.VERSIONS}
    require(state["package_versions"] == helper.VERSIONS, "selected package versions differ")
    state["actual_environment"] = {
        key: os.environ.get(key) for key in {**helper.START_ENV, **helper.CPU_ENV}
    }
    state["status"] = "importing_CPU_framework"
    helper.write_json(output / "state.json", state)
    import torch

    require(
        Path(torch.__file__).resolve()
        == helper.RUNTIME_PYTHON.parent.parent / "lib/python3.13/site-packages/torch/__init__.py",
        "Torch origin differs",
    )
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    require(
        torch.get_num_threads() == 4
        and torch.get_num_interop_threads() == 1
        and not torch.cuda.is_initialized(),
        "CPU threading/CUDA state differs",
    )
    sys.path.insert(0, plan["snapshot"])
    from d3_runtime import D3
    from transformers.models.qwen3_5 import modeling_qwen3_5

    require(
        Path(modeling_qwen3_5.__file__).resolve()
        == (helper.OVERLAY / "transformers/models/qwen3_5/modeling_qwen3_5.py").resolve(),
        "Transformers origin differs",
    )
    state["actual_kernel_bindings"] = {
        name: helper.binding_identity(getattr(modeling_qwen3_5, name))
        for name in helper.KERNEL_NAMES
    }
    state["status"] = "constructing_original_publisher_CPU_batch4"
    helper.write_json(output / "state.json", state)
    began = time.monotonic()
    runtime = D3.from_pretrained(
        plan["snapshot"],
        device="cpu",
        batch_size=4,
        verify="full",
        local_files_only=True,
        permutation_average=False,
    )
    state["model_loads"] = 1
    state["model_load_seconds"] = time.monotonic() - began
    validate_runtime(runtime, torch, plan["model"])
    state["native_provenance"] = runtime.provenance()
    state["native_runtime_info"] = runtime.runtime_info()
    return runtime, torch


def own_counters(identity):
    path = Path("/sys/fs/cgroup") / identity["cgroup"].split("::", 1)[1].lstrip("/")
    counters = {
        "memory.current": int((path / "memory.current").read_text()),
        "memory.peak": int((path / "memory.peak").read_text()),
    }
    for name in ("memory.events", "cpu.stat"):
        counters[name] = {
            parts[0]: int(parts[1])
            for line in (path / name).read_text().splitlines()
            if len(parts := line.split()) == 2
        }
    return {
        "cgroup": identity["cgroup"],
        "observed_monotonic": time.monotonic(),
        "counters": counters,
        "scope": "Only this selected owned unit; includes actor/framework/cache/wrapper activity, not native-only peak or external closure",
    }


class CPUProfileAbort(BaseException):
    """A resource/identity refusal must escape upstream per-case handlers."""


class NativeBatch4Backend:
    """One already loaded runtime; unchanged full requests for both future suites."""

    def __init__(self, runtime, torch, model, context_limit=16384, evidence_sink=None):
        require(context_limit == 16384, "declared rejection-only benchmark context differs")
        self.runtime, self.torch, self.model = runtime, torch, model
        self.context_limit = context_limit
        self.evidence_sink = evidence_sink
        self.check_runtime()

    def check_runtime(self):
        try:
            validate_runtime(self.runtime, self.torch, self.model)
        except Exception as exc:
            raise CPUProfileAbort("Native batch4 CPU identity/numerics changed") from exc

    def prepare(self, state, questions):
        prepared = self.runtime.prepare(
            state=copy.deepcopy(state), questions=copy.deepcopy(questions)
        )
        lengths = {key: len(tokens) for key, tokens in prepared.sequences.items()}
        lengths.update(prepared.lengths)
        require(
            not prepared.errors
            and prepared.keys == list(questions)
            and set(lengths) == set(questions),
            "publisher rejected complete original request",
        )
        require(
            all(type(n) is int and 0 < n <= self.context_limit for n in lengths.values()),
            "context cap exceeded; nothing truncated",
        )
        return prepared, lengths

    def __call__(self, state, questions):
        self.check_runtime()
        _, lengths = self.prepare(state, questions)
        if self.evidence_sink is not None:
            self.evidence_sink(
                "prepared", {"lengths": lengths, "original_question_order": list(questions)}
            )
        raw = self.runtime.system_one(
            state=copy.deepcopy(state), questions=copy.deepcopy(questions)
        )
        if self.evidence_sink is not None:
            self.evidence_sink("raw_returned_before_validation", raw)
        validate_native(raw, questions, lengths, self.model)
        self.check_runtime()
        return raw

    def answers(self, state, questions):
        return self(state, questions)["answers"]


def validate_native(raw, questions, lengths, model):
    require(
        isinstance(raw, dict)
        and raw.get("model") == model["name"]
        and list(raw.get("answers", {})) == list(questions),
        "actual model/question order differs",
    )
    require(
        raw.get("usage") == {"input_tokens": sum(lengths.values()), "output_tokens": 0},
        "complete native usage differs",
    )
    json.dumps(raw, allow_nan=False)


def validate_smoke_answers(raw, request, lengths, model):
    validate_native(raw, request["questions"], lengths, model)
    for key, question in request["questions"].items():
        answer = raw["answers"][key]
        require(
            answer.get("type") == "choice"
            and list(answer.get("probabilities", {})) == list(question["criteria"]),
            "all original choice candidates/order required",
        )
        probabilities = answer["probabilities"]
        values = list(probabilities.values())
        require(
            all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in values)
            and abs(sum(values) - 1) <= 1e-6,
            "finite complete choice distribution required",
        )
        require(
            answer.get("choice") == max(probabilities, key=probabilities.__getitem__),
            "legal native argmax required",
        )
        entropy = -sum(value * math.log(value) for value in values if value > 0)
        expected = 1 - entropy / math.log(len(values))
        require(
            type(answer.get("confidence")) in (float, int)
            and math.isfinite(answer["confidence"])
            and abs(answer["confidence"] - expected) <= 1e-6,
            "original entropy confidence required",
        )
    return {"contract_valid": True, "semantic_expected_answer_or_quality_claim": False}


class ShapeWitness:
    """Observation hook: returns None and never reads/modifies tensor values."""

    def __init__(self, torch, expected_width):
        self.torch, self.expected_width, self.events = torch, expected_width, []

    def __call__(self, _module, args):
        require(
            args and tuple(args[0].shape) == (4, self.expected_width, 1024),
            "actual four-row decoder input required",
        )
        hidden = args[0]
        require(
            hidden.dtype == self.torch.bfloat16
            and hidden.device.type == "cpu"
            and not self.torch.cuda.is_initialized(),
            "actual CPU BF16 first-layer input required",
        )
        self.events.append(
            {
                "shape": list(hidden.shape),
                "dtype": str(hidden.dtype),
                "device": hidden.device.type,
                "observed_monotonic": time.monotonic(),
                "inputs_or_outputs_modified": False,
            }
        )
        return None


def execute(plan_path, expected, output):
    require(
        sys.flags.no_user_site
        and sys.dont_write_bytecode
        and not MODEL_MODULES.intersection(sys.modules),
        "fresh -B -s actor with no model imports required",
    )
    helper = load_helper()
    plan = load_plan(helper, plan_path, expected)
    identity = helper.process_identity()
    limits = helper.owned_limits(identity)
    validate_owner(identity, limits, helper.START_ENV)
    gate = helper.host_gate()
    output = Path(output).resolve(strict=False)
    require(
        output.is_relative_to(HERE.parent.resolve()) and not output.exists(),
        "fresh resolved output in runtime tree required",
    )
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = {
        "profile": plan["profile"],
        "plan_sha256": expected,
        "model_key": plan["model_key"],
        "status": "authenticating",
        "process": identity,
        "owned_limits": limits,
        "host_startup_admission": gate,
        "model_loads": 0,
        "native_request_calls": 0,
        "benchmark_calls": 0,
        "browser_calls": 0,
        "quality_or_batch1_parity_claim": False,
        "calls": [],
    }
    helper.write_json(output / "state.json", state)
    primary, handle = None, None
    with helper.Deadline(WALL_SECONDS, output, "actor"):
        try:
            state["before_load_owned_counters"] = own_counters(identity)
            runtime, torch = load_runtime(helper, plan, output, state)
            request = copy.deepcopy(plan["request"])
            backend = NativeBatch4Backend(runtime, torch, plan["model"])
            prepared, lengths = backend.prepare(**request)
            require(
                list(lengths) == list(HEADS)
                and set(prepared.sequences) == set(HEADS)
                and len(set(lengths.values())) > 1
                and max(lengths.values()) <= 512,
                "four differing full short synthetic prompts required; nothing truncated",
            )
            witness = ShapeWitness(torch, max(lengths.values()))
            handle = runtime.backbone.language_model.layers[0].register_forward_pre_hook(witness)
            state["input_tokens_per_question"] = lengths
            state["padded_shape_expected"] = [4, max(lengths.values()), 1024]
            state["full_developer_request"] = request
            state["status"] = "running_two_ordered_native_batch4_requests"
            for index in range(CALLS):
                row = {
                    "index": index,
                    "status": "running",
                    "before_owned_counters": own_counters(identity),
                }
                state["calls"].append(row)
                state["native_request_calls"] += 1
                helper.write_json(output / "state.json", state)
                began = time.monotonic()
                with helper.Deadline(CALL_SECONDS, output, "call-" + str(index)):
                    raw = runtime.system_one(**copy.deepcopy(request))
                    raw_path = output / ("call-" + str(index) + "-raw.json")
                    try:
                        helper.write_json(
                            raw_path, {"raw_response": raw, "encoding": "strict_json"}
                        )
                        row["raw_response"] = raw
                    except (TypeError, ValueError, OverflowError):
                        row["raw_response_diagnostic"] = helper.diagnostic(raw)
                        helper.write_json(
                            raw_path,
                            {
                                "invalid_raw_diagnostic": row["raw_response_diagnostic"],
                                "valid_response_claim": False,
                            },
                        )
                    row["validation"] = validate_smoke_answers(raw, request, lengths, plan["model"])
                    validate_runtime(runtime, torch, plan["model"])
                    require(
                        len(witness.events) == index + 1,
                        "exactly one actual four-row forward per request required",
                    )
                    row["actual_first_layer_event"] = witness.events[-1]
                    row["after_owned_counters"] = own_counters(identity)
                    row["validation_seconds"] = helper.ensure_elapsed(
                        began, CALL_SECONDS, "batch4 response validation"
                    )
                    row["status"] = "validated"
                    helper.write_json(output / "state.json", state)
                    row["durable_validation_seconds"] = helper.ensure_elapsed(
                        began, CALL_SECONDS, "batch4 durable validation"
                    )
            require(
                state["model_loads"] == 1
                and state["native_request_calls"] == 2
                and len(witness.events) == 2,
                "one constructor/two actual four-row forwards required",
            )
            state["actual_first_layer_events"] = witness.events
            handle.remove()
            handle = None
            authenticate(helper, plan)
            validate_runtime(runtime, torch, plan["model"])
            require(
                sha(__file__) == plan["actor_sha256"]
                and sha(HELPER) == HELPER_SHA
                and sha(HERE / "source_audit.py") == SOURCE_AUDIT_SHA,
                "selected sources changed during probe",
            )
            state["final_full_model_manifest_runtime_source_identities_unchanged"] = True
            state["after_final_auth_owned_counters"] = own_counters(identity)
            state["cuda_initialized_final"] = torch.cuda.is_initialized()
            state["elapsed_seconds"] = helper.ensure_elapsed(
                started, WALL_SECONDS, "whole batch4 probe"
            )
            state["status"] = "native_batch4_two_call_synthetic_probe_complete_no_benchmark_claim"
            helper.write_json(output / "state.json", state)
            helper.ensure_elapsed(started, WALL_SECONDS, "whole batch4 durable evidence")
            return state
        except BaseException as exc:
            primary = exc
            state["status"] = "failed_no_batch4_qualification"
            state["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            cleanup_primary = None
            if handle is not None:
                try:
                    handle.remove()
                except BaseException as secondary:
                    if primary is None:
                        primary = secondary
                        cleanup_primary = secondary
                        state["status"] = "failed_no_batch4_qualification"
                        state["error"] = {
                            "type": type(secondary).__name__,
                            "message": str(secondary),
                        }
                    else:
                        primary.add_note("Observation hook cleanup failed: " + repr(secondary))
            state["elapsed_seconds"] = time.monotonic() - started
            helper.finalize_state(output, state, primary)
            if cleanup_primary is not None:
                raise cleanup_primary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_default_no_model_or_job_execution",
                    "models": list(MODELS),
                    "batch_size": 4,
                    "ordered_independent_questions": 4,
                    "native_calls": 2,
                    "constructor_count": 1,
                    "owner_unit": UNIT,
                    "owner_profile": OWNER_PROFILE,
                    "memory_max_bytes": MEMORY_MAX,
                    "startup_available_bytes": RESERVE + MEMORY_MAX,
                    "whole_seconds": WALL_SECONDS,
                    "durable_call_seconds": CALL_SECONDS,
                    "benchmark_or_batch1_score_inherited": False,
                }
            )
        )
        return
    require(
        args.plan is not None and args.output is not None,
        "explicit selected plan and fresh output required",
    )
    result = execute(args.plan, args.plan_sha256, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
