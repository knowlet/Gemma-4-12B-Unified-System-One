"""Replay 20 frozen successful callbacks to measure a new MPS runtime epoch.

No warmup, labels, model selection, retry, watermark override, or parallel work.
The first measured response is retained. Torch and archived S1 load only after
all captured predecessors finish and frozen input/source checks pass.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REQUEST_LIMIT = 20
DRIVER_FRACTION = 0.8
MAX_DRIVER_GROWTH_BYTES = 1024**3
MAX_PROBABILITY_DELTA = 1e-2
RUNTIME_FIELDS = (
    "model",
    "revision",
    "question_mode",
    "temperature",
    "device",
    "dtype",
    "attn_implementation",
    "compile_mode",
    "max_forward_batch_size",
    "max_prepared_prompts",
    "transformers_version",
    "max_context",
    "precision",
    "quantization",
    "quantization_details",
    "adapter_path",
)
MPS_ENVIRONMENT = (
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "PYTORCH_MPS_FAST_MATH",
    "PYTORCH_MPS_PREFER_METAL",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
)


class DriverLimit(RuntimeError):
    pass


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def capture(value):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False)), "json"
    except (TypeError, ValueError, OverflowError):
        return repr(value), "python_repr_not_json"


def error_record(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


def load_module(path, name):
    module = SimpleNamespace(__file__=str(path), __name__=name)
    exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    return module


def inventory(root):
    return {
        path.relative_to(root).as_posix(): digest(path)
        for path in sorted(Path(root).rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def select_records(path):
    """First 20 successful callback rows, in original order; never choose by delta."""
    selected = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        row = json.loads(line)
        if row.get("error") is not None or not isinstance(row.get("response"), dict):
            continue
        if row.get("allocator_maintenance", {}).get("error") is not None:
            continue
        if not isinstance(row.get("request"), dict) or not row["request"].get("questions"):
            raise ValueError("successful callback lacks a complete request")
        selected.append({"source_line": line_number, "original": copy.deepcopy(row)})
        if len(selected) == REQUEST_LIMIT:
            return selected
    raise ValueError("trace must contain at least 20 successful callbacks")


def compare_responses(request, baseline, candidate):
    """Descriptive fixed numerical comparison; no labels or quality scoring."""
    questions = request["questions"]
    if isinstance(questions, list):
        questions = {q["id"]: q for q in questions}
    if not isinstance(questions, dict) or not questions:
        raise ValueError("nonempty request questions required")
    distributions = []
    for response in (baseline, candidate):
        answers = response.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise ValueError("response question ids differ from request")
        values = []
        for qid, question in questions.items():
            criteria = question.get("criteria")
            labels = (
                ["false", "true"]
                if question["type"] == "noul"
                else list(criteria)
                if isinstance(criteria, dict) or question["type"] == "choice"
                else [str(i) for i in range(len(criteria))]
            )
            answer = answers[qid]
            probabilities = answer.get("probabilities")
            if (
                answer.get("id", qid) != qid
                or answer.get("type", question["type"]) != question["type"]
                or not isinstance(probabilities, dict)
                or set(probabilities) != set(labels)
            ):
                raise ValueError("response probability schema differs from request")
            vector = [probabilities[label] for label in labels]
            if (
                any(
                    isinstance(p, bool)
                    or not isinstance(p, (int, float))
                    or not math.isfinite(p)
                    or not 0 <= p <= 1
                    for p in vector
                )
                or abs(sum(vector) - 1) > 1e-5
            ):
                raise ValueError("invalid response probability distribution")
            values.append(vector)
        distributions.append(values)
    differences, agreement = [], []
    for old, new in zip(*distributions, strict=True):
        differences.extend(abs(a - b) for a, b in zip(old, new, strict=True))
        agreement.append(
            max(range(len(old)), key=old.__getitem__) == max(range(len(new)), key=new.__getitem__)
        )
    maximum = max(differences)
    return {
        "max_probability_delta": maximum,
        "all_argmax_agree": all(agreement),
        "questions": len(agreement),
        "max_allowed_probability_delta": MAX_PROBABILITY_DELTA,
        "passed": maximum <= MAX_PROBABILITY_DELTA and all(agreement),
    }


def memory_snapshot(torch):
    torch.mps.synchronize()
    result = {
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
    }
    if min(result.values()) < 0 or result["recommended_max_bytes"] <= 0:
        raise RuntimeError("valid MPS memory observations required")
    return result


def check_driver(memory):
    if memory["driver_allocated_bytes"] >= DRIVER_FRACTION * memory["recommended_max_bytes"]:
        raise DriverLimit("driver memory reached 0.8 of recommended maximum before next forward")


def trim_memory(torch):
    row = {"before": None, "after": None, "error": None}
    started = time.perf_counter()
    try:
        row["before"] = memory_snapshot(torch)
        torch.mps.empty_cache()
        row["after"] = memory_snapshot(torch)
    except Exception as exc:
        row["error"] = error_record(exc)
    row["wall_ms"] = (time.perf_counter() - started) * 1000
    return row


@contextmanager
def observe_forwards(model, torch, row, event_path):
    pending = []

    def before(_module, _args, kwargs):
        memory = memory_snapshot(torch)
        check_driver(memory)
        row["forwards"] += 1
        inputs = kwargs.get("input_ids")
        event = {
            "request_index": row["index"],
            "forward": row["forwards"],
            "input_shape": list(inputs.shape) if inputs is not None else None,
            "before": memory,
            "after": None,
        }
        pending.append(event)
        append_json(event_path, {**event, "event": "before_forward"})

    def after(_module, _args, _output):
        if pending:
            event = pending.pop()
            event["after"] = memory_snapshot(torch)
            row["forward_events"].append(event)
            append_json(event_path, {**event, "event": "after_forward"})

    handles = []
    try:
        handles.append(model.backbone.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(model.backbone.register_forward_hook(after, always_call=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


def summarize(rows, baseline_after_trim, terminal_error=None, stop_reason=None):
    complete = len(rows) == REQUEST_LIMIT
    valid_rows = all(
        row.get("error") is None
        and row.get("forwards", 0) > 0
        and row.get("allocator_maintenance", {}).get("error") is None
        and row.get("allocator_maintenance", {}).get("after") is not None
        for row in rows
    )
    baseline = baseline_after_trim["driver_allocated_bytes"]
    growth = [
        row["allocator_maintenance"]["after"]["driver_allocated_bytes"] - baseline
        for row in rows
        if row.get("allocator_maintenance", {}).get("after") is not None
    ]
    memory_pass = (
        complete
        and valid_rows
        and terminal_error is None
        and stop_reason in (None, "request_limit")
        and len(growth) == REQUEST_LIMIT
        and max(growth) <= MAX_DRIVER_GROWTH_BYTES
    )
    comparisons = [row.get("comparison") for row in rows]
    numerical_pass = (
        complete
        and valid_rows
        and terminal_error is None
        and all(item is not None and item["passed"] for item in comparisons)
    )
    return {
        "requested_callbacks": REQUEST_LIMIT,
        "completed_callbacks": len(rows),
        "actual_forwards": sum(row.get("forwards", 0) for row in rows),
        "warmups": 0,
        "first_sample_retained": True,
        "post_load_trim_driver_bytes": baseline,
        "post_trim_driver_growth_bytes": growth,
        "max_post_trim_driver_growth_bytes": max(growth) if growth else None,
        "max_allowed_driver_growth_bytes": MAX_DRIVER_GROWTH_BYTES,
        "memory_gate_passed": memory_pass,
        "numerical_gate_passed": numerical_pass,
        "max_allowed_probability_delta": MAX_PROBABILITY_DELTA,
        "max_probability_delta": max(
            (r["max_probability_delta"] for r in comparisons if r), default=None
        ),
        "all_argmax_agree": all(r["all_argmax_agree"] for r in comparisons if r)
        if all(comparisons)
        else False,
        "stop_reason": stop_reason,
        "error": terminal_error,
        "interpretation": "Instrumented memory replay and descriptive numerical agreement, not model quality or selection.",
    }


def execute_replay(backend, runtime, records, output, baseline_after_trim):
    rows, terminal_error, stop_reason = [], None, "request_limit"
    for index, selected in enumerate(records, 1):
        if index > REQUEST_LIMIT:
            break
        try:
            before = memory_snapshot(runtime.torch)
            check_driver(before)
        except Exception as exc:
            terminal_error = error_record(exc)
            stop_reason = (
                "driver_limit" if isinstance(exc, DriverLimit) else "memory_observation_error"
            )
            break
        original = selected["original"]
        row = {
            "index": index,
            "source_line": selected["source_line"],
            "request": copy.deepcopy(original["request"]),
            "old_response": copy.deepcopy(original["response"]),
            "old_latency_ms": original.get("latency_ms"),
            "before_request": before,
            "response": None,
            "error": None,
            "comparison": None,
            "forwards": 0,
            "forward_events": [],
        }
        started = time.perf_counter()
        try:
            request = runtime.contracts.DecisionRequest.model_validate(row["request"])
            with observe_forwards(
                backend.model, runtime.torch, row, output / "memory-events.jsonl"
            ):
                response = backend.predict(request)
                runtime.torch.mps.synchronize()
            row["response"], row["response_encoding"] = capture(response)
            runtime.backends.normalize_response(request, response)
            runtime.backends.normalize_response(request, row["old_response"])
            row["comparison"] = compare_responses(row["request"], row["old_response"], response)
            if row["forwards"] == 0:
                raise RuntimeError("callback did not execute an observed real forward")
        except BaseException as exc:
            row["error"] = error_record(exc)
        finally:
            row["inference_wall_ms"] = (time.perf_counter() - started) * 1000
            row["allocator_maintenance"] = trim_memory(runtime.torch)
            row["wall_ms"] = (time.perf_counter() - started) * 1000
            append_json(output / "callbacks.jsonl", row)
            rows.append(row)
            write_json(
                output / "summary.json", summarize(rows, baseline_after_trim, row["error"], None)
            )
        if row["error"] or row["allocator_maintenance"]["error"]:
            terminal_error = row["error"] or row["allocator_maintenance"]["error"]
            stop_reason = (
                "driver_limit" if terminal_error["type"] == "DriverLimit" else "callback_error"
            )
            break
    summary = summarize(rows, baseline_after_trim, terminal_error, stop_reason)
    write_json(output / "summary.json", summary)
    return summary


def freeze_inputs(args, guard_bytes):
    parent_bytes = (args.parent_run / "receipt.json").read_bytes()
    parent = json.loads(parent_bytes)
    frozen = args.output / "sources"
    hashes = {k: v for k, v in parent["source_inventory"].items() if k.startswith("src/s1/")}
    if not {"src/s1/unified.py", "src/s1/backends.py"} <= set(hashes):
        raise ValueError("parent receipt lacks frozen S1 sources")
    for name, expected in hashes.items():
        source = args.parent_run / "sources" / name
        if source.is_symlink() or digest(source) != expected["sha256"]:
            raise ValueError("archived parent S1 differs from parent execution receipt")
        target = frozen / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name, raw in (
        ("benchmark_mps_memory.py", Path(__file__).read_bytes()),
        ("resume_mps_benchmarks.py", guard_bytes),
    ):
        target = frozen / "scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    records = select_records(args.trace)
    write_json(frozen / "inputs/selected-callbacks.json", records)
    (frozen / "inputs/parent-receipt.json").write_bytes(parent_bytes)
    shutil.copyfile(args.trace, frozen / "inputs/original-callbacks.jsonl")
    checkpoint = parent["checkpoint"]
    shutil.copyfile(
        Path(checkpoint["path"]) / "artifact-manifest.json",
        frozen / "inputs/checkpoint-manifest.json",
    )
    if digest(frozen / "inputs/checkpoint-manifest.json") != checkpoint["manifest_sha256"]:
        raise ValueError("checkpoint manifest differs from parent before queue")
    return parent, inventory(frozen)


def load_runtime(output):
    source = (output / "sources/src").resolve()
    for name, module in tuple(sys.modules.items()):
        if name == "s1" or name.startswith("s1."):
            location = getattr(module, "__file__", None)
            if not location or not Path(location).resolve().is_relative_to(source):
                raise RuntimeError("S1 imported before frozen source was installed")
    sys.path.insert(0, str(source))
    return SimpleNamespace(
        torch=importlib.import_module("torch"),
        backends=importlib.import_module("s1.backends"),
        contracts=importlib.import_module("s1.contracts"),
    )


def run(args):
    if len(set(args.wait_for_pid)) != len(args.wait_for_pid) or any(
        pid <= 0 or pid == os.getpid() for pid in args.wait_for_pid
    ):
        raise ValueError("unique positive predecessor PIDs required")
    if not 0 < args.poll_seconds <= 60:
        raise ValueError("poll interval must be in (0,60]")
    args.output.mkdir(parents=True, exist_ok=False)
    guard_path = ROOT / "scripts/resume_mps_benchmarks.py"
    guard_bytes = guard_path.read_bytes()
    guard = load_module(guard_path, "_memory_replay_guard")
    receipt = {
        "status": "initializing",
        "pid": os.getpid(),
        "predecessors": [],
        "request_limit": REQUEST_LIMIT,
        "max_driver_fraction": DRIVER_FRACTION,
        "max_driver_growth_bytes": MAX_DRIVER_GROWTH_BYTES,
        "max_probability_delta": MAX_PROBABILITY_DELTA,
        "packages": {
            name: guard.package_version(name)
            for name in ("torch", "torchvision", "transformers", "numpy")
        },
        "mps_environment": {name: os.environ.get(name) for name in MPS_ENVIRONMENT},
        "labels_used": False,
        "official_rank": None,
        "reported_cost_usd": None,
    }

    def save():
        write_json(args.output / "receipt.json", receipt)

    save()
    try:
        receipt["predecessors"] = [guard.capture_handle(pid) for pid in args.wait_for_pid]
        parent, receipt["source_sha256"] = freeze_inputs(args, guard_bytes)
        receipt["parent"] = str(args.parent_run)
        receipt["checkpoint"] = parent["checkpoint"]
        receipt["runtime_expected"] = parent["runtime_loaded"]
        receipt["runtime_change"] = {
            "torch": [parent["runtime_loaded"]["torch_version"], receipt["packages"]["torch"]]
        }
        receipt["status"] = "waiting"
        save()
        frozen = args.output / "sources"
        guard = load_module(
            frozen / "scripts/resume_mps_benchmarks.py", "_frozen_memory_replay_guard"
        )
        for handle in receipt["predecessors"]:
            guard.wait_handle(handle, save, args.poll_seconds)
        if inventory(frozen) != receipt["source_sha256"]:
            raise ValueError("frozen source or requests changed while queued")
        guard.verify_checkpoint(receipt["checkpoint"])
        if inventory(frozen) != receipt["source_sha256"]:
            raise ValueError("frozen source or requests changed during checkpoint verification")
        expected = receipt["runtime_expected"]
        if any(value is None for value in receipt["packages"].values()):
            raise ValueError("required runtime packages are unavailable after predecessor waits")
        if not set(RUNTIME_FIELDS) | {"torch_version"} <= set(expected):
            raise ValueError("parent lacks complete native runtime identity")
        if any(
            expected.get(k) != v
            for k, v in {
                "device": "mps",
                "dtype": "bfloat16",
                "attn_implementation": "sdpa",
                "question_mode": "causal_multislot",
            }.items()
        ):
            raise ValueError("parent does not have the fixed native MPS replay policy")
        if receipt["packages"]["transformers"] != expected["transformers_version"]:
            raise ValueError("only the Torch/TorchVision runtime may differ from the parent")
        if receipt["packages"]["numpy"] != parent["packages"]["numpy"]:
            raise ValueError("NumPy differs from the parent runtime")
        runtime = load_runtime(args.output)
        receipt["memory_before_load"] = memory_snapshot(runtime.torch)
        backend = runtime.backends.GemmaBackend(
            receipt["checkpoint"]["path"],
            device="mps",
            dtype="bfloat16",
            attn_implementation="sdpa",
            question_mode="causal_multislot",
        )
        receipt["runtime_loaded"] = copy.deepcopy(backend.metadata)
        if any(backend.metadata.get(k) != expected.get(k) for k in RUNTIME_FIELDS):
            raise ValueError("loaded model differs from frozen parent runtime policy")
        if backend.metadata.get("torch_version") != receipt["packages"]["torch"]:
            raise ValueError("Torch package changed while queued")
        receipt["torch_git_version"] = getattr(runtime.torch.version, "git_version", None)
        receipt["post_load_memory"] = memory_snapshot(runtime.torch)
        receipt["post_load_trim"] = trim_memory(runtime.torch)
        save()
        if receipt["post_load_trim"]["error"]:
            raise RuntimeError("post-load MPS cache trim failed")
        receipt["status"] = "running"
        save()
        records = json.loads((frozen / "inputs/selected-callbacks.json").read_text())
        summary = execute_replay(
            backend, runtime, records, args.output, receipt["post_load_trim"]["after"]
        )
        receipt.update(status="completed_replay", summary=summary)
        save()
        if summary["error"] and summary["error"]["type"] == "KeyboardInterrupt":
            return 130
        return 0 if summary["memory_gate_passed"] and summary["numerical_gate_passed"] else 1
    except BaseException as exc:
        receipt.update(status="error", error=error_record(exc))
        save()
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-run", type=Path, default=ROOT / "artifacts/ultrafast/20261028-benchmark-resume-02"
    )
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-for-pid", type=int, action="append", default=[])
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args(argv)
    args.parent_run = args.parent_run.resolve()
    args.trace = (
        args.trace or args.parent_run / "causal_multislot/coherence-mini-model-calls.jsonl"
    ).resolve()
    args.output = args.output.resolve()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
