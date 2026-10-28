"""Observe frozen batched/singleton decisions around unchanged native text generation.

This MPS-only control never repairs logits, scores tasks, or selects a fallback.
Telemetry copies bounded chunks to CPU and synchronizes execution: its timings
describe an instrumented diagnostic, not an uninstrumented speed benchmark.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PHASES = (
    "batched_before",
    "singletons_before",
    "text_generation",
    "batched_after",
    "singletons_after",
)
TELEMETRY_POLICY = {
    "version": "cpu-copy-then-cast-v2",
    "device_readback": "Copy original dtype to CPU, then cast values to float32 or row masks to bool on CPU.",
    "combined_device_dtype_transfer": False,
    "prior_observer_invalidation": {
        "run": "artifacts/mps-numerics/20261028-native-control-01",
        "status": "invalid_for_unchanged_model_comparisons",
        "reason": "The earlier combined MPS device/dtype observer can misread BF16 offset views and may mutate int64 input masks. Its BF16 telemetry and unchanged-model claims are invalidated; saved responses remain historical outputs of that instrumented run.",
        "unaffected": "The original external browser baseline had no diagnostic observer and is not invalidated by this observer defect.",
    },
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def capture(value):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False)), "json"
    except (TypeError, ValueError, OverflowError):
        return repr(value), "python_repr_not_json"


def load_module(path, name):
    module = SimpleNamespace(__file__=str(path), __name__=name)
    raw = Path(path).read_bytes()
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def selected_record(path, index):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not 1 <= index <= len(rows):
        raise ValueError("trace index must select an existing one-based record")
    return copy.deepcopy(rows[index - 1])


def read_inputs(decision_trace, decision_record, helper_trace, helper_record):
    decision = selected_record(decision_trace, decision_record)
    helper = selected_record(helper_trace, helper_record)
    request = decision.get("mapped_request")
    if not isinstance(request, dict) or not request.get("questions"):
        raise ValueError("selected decision trace has no complete mapped request")
    if not isinstance(helper.get("context"), dict) or not isinstance(
        helper.get("instruction"), str
    ):
        raise ValueError("selected helper trace lacks original instruction/context")
    if (
        helper.get("do_sample") is not False
        or type(helper.get("max_new_tokens")) is not int
        or not 1 <= helper["max_new_tokens"] <= 1024
    ):
        raise ValueError("helper must declare its original bounded greedy generation settings")
    return request, {
        key: copy.deepcopy(helper[key])
        for key in ("context", "instruction", "max_new_tokens", "do_sample")
    }


def inventory(root):
    return {
        str(path.relative_to(root)): digest(path)
        for path in sorted(Path(root).rglob("*"))
        if path.is_file()
    }


def freeze_inputs(args, guard_bytes):
    frozen = args.output / "sources"
    for source in sorted((args.source_root / "src/s1").rglob("*.py")):
        if source.is_symlink():
            raise ValueError("frozen S1 source must contain regular files")
        target = frozen / source.relative_to(args.source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if (
        not (frozen / "src/s1/unified.py").is_file()
        or not (frozen / "src/s1/backends.py").is_file()
    ):
        raise ValueError("source root must contain the full frozen src/s1 package")
    script = frozen / "scripts/diagnose_mps_numerics.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), script)
    (script.parent / "resume_mps_benchmarks.py").write_bytes(guard_bytes)
    for name, source in (
        ("decision-trace.json", args.decision_trace),
        ("helper-trace.json", args.helper_trace),
        ("checkpoint-manifest.json", args.checkpoint / "artifact-manifest.json"),
    ):
        target = frozen / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return inventory(frozen)


def verify_checkpoint(path, expected_manifest):
    path = Path(path).resolve()
    if digest(path / "artifact-manifest.json") != expected_manifest:
        raise ValueError("checkpoint manifest changed after input freeze")
    manifest = json.loads((path / "artifact-manifest.json").read_text())
    rows = manifest.get("files", [])
    names = [row["path"] for row in rows]
    if len(set(names)) != len(names) or not {
        "model.safetensors",
        "config.json",
        "s1_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    } <= set(names):
        raise ValueError("checkpoint manifest lacks unique required model files")
    for row in rows:
        relative = Path(row["path"])
        target = (path / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not target.is_relative_to(path)
            or not target.is_file()
        ):
            raise ValueError("unsafe/missing checkpoint member")
        if target.stat().st_size != row["size_bytes"] or digest(target) != row["sha256"]:
            raise ValueError("checkpoint member differs from manifest")
    return {"path": str(path), "manifest_sha256": expected_manifest, "verified_files": rows}


def load_runtime(output):
    source = (output / "sources/src").resolve()
    for name, module in tuple(sys.modules.items()):
        if name == "s1" or name.startswith("s1."):
            location = getattr(module, "__file__", None)
            if not location or not Path(location).resolve().is_relative_to(source):
                raise RuntimeError("S1 imported before diagnostic frozen source was installed")
    sys.path.insert(0, str(source))
    return SimpleNamespace(
        torch=importlib.import_module("torch"),
        backends=importlib.import_module("s1.backends"),
        contracts=importlib.import_module("s1.contracts"),
        unified=importlib.import_module("s1.unified"),
        gemma=importlib.import_module("s1.evaluation.gemma"),
    )


def tensor_statistics(tensor, torch, row_mask=None, max_values=262144):
    """Count every value with bounded CPU copies; preserve valid vs padded rows."""
    detached = tensor.detach()
    width = detached.shape[-1] if detached.ndim else 1
    values = detached.reshape(-1, width)
    row_mask = row_mask.reshape(-1).to(device="cpu").bool() if row_mask is not None else None
    if row_mask is not None and row_mask.numel() != values.shape[0]:
        row_mask = None

    def empty():
        return {
            "values": 0,
            "finite": 0,
            "nan": 0,
            "positive_inf": 0,
            "negative_inf": 0,
            "finite_min": None,
            "finite_max": None,
        }

    result = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "all": empty(),
    }
    if row_mask is not None:
        result.update(valid_query_rows=empty(), padded_query_rows=empty())

    def add(target, cpu):
        finite = torch.isfinite(cpu)
        target["values"] += cpu.numel()
        target["finite"] += int(finite.sum().item())
        target["nan"] += int(torch.isnan(cpu).sum().item())
        target["positive_inf"] += int(torch.isposinf(cpu).sum().item())
        target["negative_inf"] += int(torch.isneginf(cpu).sum().item())
        if finite.any().item():
            low, high = float(cpu[finite].min().item()), float(cpu[finite].max().item())
            target["finite_min"] = (
                low if target["finite_min"] is None else min(low, target["finite_min"])
            )
            target["finite_max"] = (
                high if target["finite_max"] is None else max(high, target["finite_max"])
            )

    rows_per_chunk = max(1, max_values // width)
    for start in range(0, values.shape[0], rows_per_chunk):
        # MPS combined device+dtype conversion can mishandle nonzero storage
        # offsets and shrinking casts. Copy dtype-preserving bytes first; every
        # dtype conversion below runs on CPU and cannot write an MPS input mask.
        cpu = values[start : start + rows_per_chunk].to(device="cpu").float()
        add(result["all"], cpu)
        if row_mask is not None:
            mask = row_mask[start : start + rows_per_chunk]
            add(result["valid_query_rows"], cpu[mask])
            add(result["padded_query_rows"], cpu[~mask])
    return result


@contextmanager
def observe_model(model, runtime, events, phase):
    """Temporary observers return no replacement tensors or changed logits."""
    handles, originals, state = [], [], {"row_mask": None}

    def record(name, tensor, mask=None):
        events.append(
            {
                "phase": phase,
                "event": name,
                "statistics": tensor_statistics(tensor, runtime.torch, mask),
            }
        )

    def before(_module, _args, kwargs):
        ids = kwargs.get("input_ids")
        embedded = kwargs.get("inputs_embeds")
        length = (
            ids.shape[1] if ids is not None else embedded.shape[1] if embedded is not None else None
        )
        mask = kwargs.get("attention_mask")
        state["row_mask"] = (
            mask[:, -length:] if hasattr(mask, "ndim") and mask.ndim == 2 and length else None
        )

    def hidden(_module, _args, output):
        value = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
        record("unified_hidden", value, state["row_mask"])

    def layer_hook(index):
        def observe(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            record(f"decoder_layer_{index}", value, state["row_mask"])

        return observe

    try:
        handles.append(model.backbone.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(model.backbone.register_forward_hook(hidden))
        for index, layer in enumerate(model.backbone.language_model.layers):
            handles.append(layer.register_forward_hook(layer_hook(index)))
        handles.append(
            model.head.register_forward_pre_hook(
                lambda _module, args: record("generation_readout_hidden", args[0])
            )
        )
        handles.append(
            model.head.register_forward_hook(
                lambda _module, _args, output: record("generation_raw_head_logits", output)
            )
        )
        handles.append(
            model.lm.register_forward_hook(
                lambda _module, _args, output: record("generation_softcapped_logits", output.logits)
            )
        )
        for module in (runtime.unified, runtime.gemma):
            original = module.project_candidates

            def wrapper(*args, _original=original, **kwargs):
                hidden = args[0] if args else kwargs["hidden"]
                counts = args[3] if len(args) > 3 else kwargs["nopts"]
                record("candidate_readout_hidden", hidden)
                logits = _original(*args, **kwargs)
                active = runtime.torch.cat(
                    [
                        logits[index, :count]
                        for index, count in enumerate(counts.detach().cpu().tolist())
                    ]
                )
                record("active_candidate_logits", active)
                events.append(
                    {
                        "phase": phase,
                        "event": "candidate_count_mask",
                        "option_counts": counts.detach().cpu().tolist(),
                        "masked_slots": int(logits.numel() - active.numel()),
                        "expected_mask_value": "negative_inf",
                    }
                )
                return logits

            originals.append((module, original))
            module.project_candidates = wrapper
        yield
    finally:
        for module, original in reversed(originals):
            module.project_candidates = original
        for handle in reversed(handles):
            handle.remove()


def generate_native(model, torch, helper):
    result = {
        "instruction": helper["instruction"],
        "context": copy.deepcopy(helper["context"]),
        "max_new_tokens": helper["max_new_tokens"],
        "do_sample": False,
        "raw_text": None,
        "generated_token_ids": None,
        "error": None,
    }
    try:
        text = model.processor.apply_chat_template(
            [
                {"role": "system", "content": helper["instruction"]},
                {"role": "user", "content": json.dumps(helper["context"], ensure_ascii=False)},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        result["rendered_prompt"] = text
        inputs = model.processor(text=text, return_tensors="pt", add_special_tokens=False)
        if inputs["input_ids"].shape[1] > model.max_context:
            raise ValueError("helper context exceeds original model limit; nothing truncated")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        result["input_tokens"] = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output = model.lm.generate(
                **inputs, max_new_tokens=helper["max_new_tokens"], do_sample=False
            )
        tokens = output[0, inputs["input_ids"].shape[1] :]
        result["generated_token_ids"] = tokens.cpu().tolist()
        result["raw_text"] = model.tok.decode(tokens, skip_special_tokens=True)
        parsed = json.loads(result["raw_text"])
        if (
            not isinstance(parsed, dict)
            or set(parsed) != {"text"}
            or not isinstance(parsed["text"], str)
            or not parsed["text"].strip()
            or len(parsed["text"]) > 2000
        ):
            raise ValueError("native helper returned no valid upstream field value")
        result["parsed"] = parsed
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return result


def measured_call(operation, runtime, guard):
    started = time.perf_counter()
    result = {"response": None, "response_encoding": None, "error": None}
    try:
        runtime.torch.mps.synchronize()
        response = operation()
        runtime.torch.mps.synchronize()
        result["response"], result["response_encoding"] = capture(response)
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        result["allocator_maintenance"] = guard.maintain_mps_allocator(runtime.torch)
        result["wall_ms_including_maintenance"] = (time.perf_counter() - started) * 1000
    return result


def execute_experiment(backend, runtime, guard, request, helper, output):
    results, all_events = [], []
    baseline = runtime.contracts.DecisionRequest.model_validate(copy.deepcopy(request))
    for phase in PHASES:
        events, phase_result = (
            [],
            {
                "phase": phase,
                "backend_calls": 0,
                "calls": [],
                "question_order": [q.id for q in baseline.questions],
                "request_sha256": json_hash(request),
                "errors": [],
                "reported_cost_usd": None,
            },
        )
        with observe_model(backend.model, runtime, events, phase):
            if phase == "text_generation":
                call = measured_call(
                    lambda: generate_native(backend.model, runtime.torch, helper), runtime, guard
                )
                phase_result["generation_calls"] = 1
                phase_result["calls"].append(call)
            elif phase.startswith("batched"):
                phase_result["backend_calls"] = 1
                phase_result["calls"].append(
                    measured_call(
                        lambda: backend.predict(baseline.model_copy(deep=True)), runtime, guard
                    )
                )
            else:
                single = copy.copy(backend)
                single.question_mode = "independent"
                answers, complete = {}, True
                for question in baseline.questions:
                    part = baseline.model_copy(
                        deep=True, update={"questions": [question.model_copy(deep=True)]}
                    )
                    call = measured_call(lambda part=part: single.predict(part), runtime, guard)
                    call.update(
                        question_id=question.id, question_order_index=len(phase_result["calls"])
                    )
                    phase_result["calls"].append(call)
                    phase_result["backend_calls"] += 1
                    if call["error"] is None and call["response_encoding"] == "json":
                        raw = call["response"]
                        if (
                            isinstance(raw, dict)
                            and isinstance(raw.get("answers"), dict)
                            and set(raw["answers"]) == {question.id}
                        ):
                            answers[question.id] = raw["answers"][question.id]
                        else:
                            complete = False
                    else:
                        complete = False
                phase_result["merged_answers"] = answers if complete else None
        phase_result["telemetry"] = events
        for call in phase_result["calls"]:
            if call["error"]:
                phase_result["errors"].append(call["error"])
            if call["allocator_maintenance"]["error"]:
                phase_result["errors"].append(
                    {"phase": "allocator_maintenance", **call["allocator_maintenance"]["error"]}
                )
            if (
                phase == "text_generation"
                and isinstance(call["response"], dict)
                and call["response"].get("error")
            ):
                phase_result["errors"].append(call["response"]["error"])
        results.append(phase_result)
        all_events.extend(events)
        write_json(output / f"{phase}.json", phase_result)
        write_json(output / "results.json", results)
    return results


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    guard_path = ROOT / "scripts/resume_mps_benchmarks.py"
    guard_bytes = guard_path.read_bytes()
    guard = load_module(guard_path, "_diagnostic_process_guard")
    receipt = {
        "status": "initializing",
        "pid": os.getpid(),
        "phases": list(PHASES),
        "predecessors": [],
        "question_mode": args.question_mode,
        "official_rank": None,
        "scoring": "not performed",
        "telemetry": "Full tensor counts in bounded CPU chunks, including valid/padded rows at every decoder layer; observers alter timing/synchronization, never tensor values.",
        "telemetry_policy": copy.deepcopy(TELEMETRY_POLICY),
        "allocator_policy": copy.deepcopy(guard.ALLOCATOR_POLICY),
        "packages": {
            name: guard.package_version(name) for name in ("torch", "transformers", "numpy")
        },
    }

    def save():
        write_json(args.output / "receipt.json", receipt)

    save()
    try:
        receipt["predecessors"] = [guard.capture_handle(pid) for pid in args.wait_for_pid]
        receipt["source_sha256"] = freeze_inputs(args, guard_bytes)
        frozen = args.output / "sources"
        request, helper = read_inputs(
            frozen / "inputs/decision-trace.json",
            args.decision_record,
            frozen / "inputs/helper-trace.json",
            args.helper_record,
        )
        receipt.update(
            status="waiting",
            request_sha256=json_hash(request),
            helper_sha256=json_hash(helper),
            decision_record=args.decision_record,
            helper_record=args.helper_record,
        )
        save()
        for handle in receipt["predecessors"]:
            guard.wait_handle(handle, save, args.poll_seconds)
        if inventory(frozen) != receipt["source_sha256"]:
            raise ValueError("diagnostic frozen inputs/source changed while queued")
        receipt["checkpoint"] = verify_checkpoint(
            args.checkpoint, digest(frozen / "inputs/checkpoint-manifest.json")
        )
        runtime = load_runtime(args.output)
        backend = runtime.backends.GemmaBackend(
            str(args.checkpoint),
            device="mps",
            dtype="bfloat16",
            attn_implementation="sdpa",
            question_mode=args.question_mode,
        )
        if backend.model.device != "mps":
            raise ValueError("diagnostic requires the complete native MPS model")
        receipt.update(status="running", runtime=backend.metadata)
        receipt["allocator_before_experiment"] = guard.maintain_mps_allocator(runtime.torch)
        save()
        if receipt["allocator_before_experiment"]["error"]:
            raise RuntimeError("MPS allocator maintenance failed before diagnostic")
        results = execute_experiment(backend, runtime, guard, request, helper, args.output)
        receipt.update(
            status="completed_diagnostic",
            phase_errors={row["phase"]: row["errors"] for row in results},
            backend_calls=sum(row["backend_calls"] for row in results),
        )
        save()
        return 0 if not any(row["errors"] for row in results) else 1
    except Exception as exc:
        receipt.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
        save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("decision-trace", "helper-trace", "checkpoint", "source-root", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--decision-record", type=int, required=True)
    parser.add_argument("--helper-record", type=int, required=True)
    parser.add_argument(
        "--question-mode", choices=("independent", "causal_multislot"), default="independent"
    )
    parser.add_argument("--wait-for-pid", type=int, action="append", default=[])
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args(argv)
    if (
        min(args.decision_record, args.helper_record) < 1
        or any(pid <= 0 for pid in args.wait_for_pid)
        or len(set(args.wait_for_pid)) != len(args.wait_for_pid)
        or not 0 < args.poll_seconds <= 60
    ):
        parser.error(
            "positive one-based record indices, unique positive PIDs and poll interval in (0,60] required"
        )
    for name in ("decision_trace", "helper_trace", "checkpoint", "source_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
