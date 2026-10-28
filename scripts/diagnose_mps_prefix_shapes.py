"""Bounded development-only MPS diagnostic for independent execution shapes.

A uses padded full prompts, B runs complete singleton prompts, and D reuses the
native prefix. This is an instrumented numerical diagnostic, not a speed claim.
It does not change production defaults, fit calibration, or consume task golds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_PROBABILITY_DELTA = 0.01
SCHEDULE = [("warmup", name) for name in ("A", "B", "D")] + [
    ("measured", name) for name in ("A", "B", "D", "D", "B", "A")
]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_sha256(value):
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def load_development_snapshot(trace_root, task_id, decision_index):
    """Read one pre-existing mapped request; task outcomes and golds are unused."""
    from s1.backends import GEMMA_MAX_FORWARD_BATCH_SIZE
    from s1.contracts import DecisionRequest

    if not task_id.startswith("dev-") or Path(task_id).name != task_id:
        raise ValueError("only a development task ID is allowed")
    if decision_index < 1:
        raise ValueError("decision index is one-based and must be positive")
    folder = Path(trace_root) / "independent" / f"{task_id}-1"
    task_path, trace_path = folder / "task.json", folder / "decision-transport.json"
    task = json.loads(task_path.read_text())
    if task.get("id") != task_id or task.get("split") != "development":
        raise ValueError("reserved/transfer snapshots cannot drive this diagnostic")
    rows = json.loads(trace_path.read_text())
    if not isinstance(rows, list) or not 1 <= decision_index <= len(rows):
        raise ValueError("decision index is outside the recorded development trace")
    row = rows[decision_index - 1]
    if row.get("error") is not None or row.get("mapped_request") is None:
        raise ValueError("selected snapshot lacks a valid mapped request")
    request = DecisionRequest.model_validate(row["mapped_request"])
    if request.media or not 2 <= len(request.questions) <= GEMMA_MAX_FORWARD_BATCH_SIZE:
        raise ValueError("bounded browser diagnostic needs 2..8 text-only independent heads")
    return request, {
        "task_id": task_id,
        "split": "development",
        "decision_index": decision_index,
        "task_path": str(task_path),
        "task_sha256": sha256(task_path),
        "trace_path": str(trace_path),
        "trace_sha256": sha256(trace_path),
        "request_sha256": content_sha256(request.model_dump()),
        "labels_or_outcomes_used": False,
    }


def source_paths(source_root):
    source_root = Path(source_root).resolve()
    return [
        *source_root.glob("src/s1/**/*.py"),
        source_root / "scripts/diagnose_mps_prefix_shapes.py",
        source_root / "scripts/run_ultrafast.py",
        source_root / "scripts/resume_mps_benchmarks.py",
    ]


def archive_sources(output, source_root=ROOT):
    """Freeze relevant source bytes before waiting or loading a checkpoint."""
    source_root = Path(source_root).resolve()
    destination = Path(output) / "sources"
    hashes = {}
    for source in source_paths(source_root):
        relative = source.relative_to(source_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative.as_posix()] = sha256(target)
    return hashes


def validate_frozen_sources(output, hashes, source_root=ROOT):
    source_root = Path(source_root).resolve()
    names = {path.relative_to(source_root).as_posix() for path in source_paths(source_root)}
    if names != set(hashes):
        raise ValueError("source inventory changed while queued; refusing model load")
    for name, expected in hashes.items():
        if (
            sha256(source_root / name) != expected
            or sha256(Path(output) / "sources" / name) != expected
        ):
            raise ValueError("source bytes changed while queued; refusing model load")


def load_wait_tools(output):
    """Use the archived, Torch-free process observer rather than a mutable helper."""
    path = Path(output) / "sources/scripts/resume_mps_benchmarks.py"
    spec = importlib.util.spec_from_file_location("_s1_prefix_shape_wait", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wait_for_predecessors(pids, report, output):
    if not pids:
        report["predecessors"] = []
        return
    observer = load_wait_tools(output)
    # Capture all identities before waiting on any one of them.
    report["predecessors"] = [observer.capture_handle(pid) for pid in pids]

    def save():
        write_json(Path(output) / "report.json", report)

    save()
    for handle in report["predecessors"]:
        observer.wait_handle(handle, save, 15)


def slot_layout(model, request):
    prefix = model.prepare_prefix(request)
    prefix_length = int(prefix["input_ids"].shape[1])
    result = []
    for question in request.questions:
        tokens, slots, _ = model.prepare_suffix([question], prefix_length)
        if tokens.shape[0] != 1 or slots.numel() != 1:
            raise ValueError("one exact native answer slot per independent head is required")
        absolute = int(slots.item())
        result.append(
            {
                "question_id": question.id,
                "absolute_slot": absolute,
                "suffix_slot": absolute - prefix_length,
                "full_tokens": prefix_length + int(tokens.shape[1]),
            }
        )
    return {"prefix_tokens": prefix_length, "heads": result}


def selected_slots(variant, forward_index, layout, batch_size):
    heads = layout["heads"]
    if variant == "A":
        if forward_index != 0 or batch_size != len(heads):
            raise ValueError("unexpected A full-batch forward shape")
        return [(i, head["absolute_slot"], head["question_id"]) for i, head in enumerate(heads)]
    if variant == "B":
        if batch_size != 1 or forward_index >= len(heads):
            raise ValueError("unexpected B singleton forward shape")
        head = heads[forward_index]
        return [(0, head["absolute_slot"], head["question_id"])]
    if variant == "D":
        if forward_index == 0 and batch_size == 1:
            return []
        if forward_index != 1 or batch_size != len(heads):
            raise ValueError("unexpected D prefix/suffix forward shape")
        return [(i, head["suffix_slot"], head["question_id"]) for i, head in enumerate(heads)]
    raise ValueError("unknown diagnostic variant")


class ForwardRecorder:
    """Clone only selected final vectors inside the hook; copy to CPU after timing."""

    def __init__(self, variant, layout):
        self.variant, self.layout = variant, layout
        self.rows = []

    def __call__(self, _module, _args, kwargs, output):
        ids = kwargs["input_ids"]
        index = len(self.rows)
        row = {
            "forward_index": index,
            "input_shape": list(ids.shape),
            "use_cache": kwargs.get("use_cache"),
            "has_past_key_values": kwargs.get("past_key_values") is not None,
            "attention_mask_shape": (
                list(kwargs["attention_mask"].shape) if "attention_mask" in kwargs else None
            ),
            "position_ids_shape": (
                list(kwargs["position_ids"].shape) if "position_ids" in kwargs else None
            ),
            "selected_slots": [],
        }
        self.rows.append(row)
        for batch_index, slot, question_id in selected_slots(
            self.variant, index, self.layout, ids.shape[0]
        ):
            hidden = output.last_hidden_state
            if not 0 <= slot < hidden.shape[1]:
                raise ValueError("selected answer slot is outside the native forward output")
            row["selected_slots"].append(
                {
                    "question_id": question_id,
                    "batch_index": batch_index,
                    "output_slot": slot,
                    "hidden_dtype": str(hidden.dtype),
                    "_tensor": hidden[batch_index, slot].detach().clone(),
                }
            )

    def finish(self):
        rows = self.rows
        for row in rows:
            for slot in row["selected_slots"]:
                tensor = slot.pop("_tensor")
                values = None
                try:
                    values = tensor.to(device="cpu").float().tolist()
                    json.dumps(values, allow_nan=False)
                    slot["hidden_vector"] = values
                except Exception as exc:
                    slot["hidden_vector"] = None
                    slot["capture_error"] = {"type": type(exc).__name__, "message": str(exc)}
                    if values is not None:
                        slot["hidden_vector_repr"] = repr(values)
        return rows


def compare(request, left, right):
    import numpy as np

    from s1.backends import normalize_response

    left, right = normalize_response(request, left), normalize_response(request, right)
    differences, argmax = [], []
    for question in request.questions:
        a = left["answers"][question.id]["probabilities"]
        b = right["answers"][question.id]["probabilities"]
        labels = question.labels()
        differences.extend(abs(a[key] - b[key]) for key in labels)
        argmax.append(
            np.argmax([a[key] for key in labels]) == np.argmax([b[key] for key in labels])
        )
    maximum = max(differences)
    agreement = bool(all(argmax))
    return {
        "max_probability_delta": maximum,
        "all_argmax_agree": agreement,
        "gate_passed": agreement and maximum <= MAX_PROBABILITY_DELTA,
    }


def summarize(request, samples):
    if [(row["phase"], row["variant"]) for row in samples] != SCHEDULE:
        raise ValueError("all planned warmup and measured attempts must be retained")
    if any(row.get("error") is not None for row in samples):
        raise ValueError("failed attempts cannot be excluded from the diagnostic")
    measured = {
        variant: [
            row for row in samples if row["phase"] == "measured" and row["variant"] == variant
        ]
        for variant in ("A", "B", "D")
    }
    result = {"comparisons": {}, "repeat_max_probability_delta": {}}
    for left, right in (("A", "B"), ("A", "D"), ("B", "D")):
        comparisons = [
            compare(request, a["raw_response"], b["raw_response"])
            for a in measured[left]
            for b in measured[right]
        ]
        result["comparisons"][f"{left}_vs_{right}"] = {
            "max_probability_delta": max(row["max_probability_delta"] for row in comparisons),
            "all_argmax_agree": all(row["all_argmax_agree"] for row in comparisons),
            "gate_passed": all(row["gate_passed"] for row in comparisons),
        }
    for variant, rows in measured.items():
        result["repeat_max_probability_delta"][variant] = max(
            compare(request, rows[0]["raw_response"], row["raw_response"])["max_probability_delta"]
            for row in rows
        )
    result["all_pairwise_gates_passed"] = all(
        row["gate_passed"] for row in result["comparisons"].values()
    )
    return result


def run_schedule(model, request, layout, operations, report, output, *, synchronize, memory, clock):
    """Injectable measurement contract, allowing CPU fake forwards in tests."""
    from s1.backends import normalize_response

    report["samples"] = []
    for phase, variant in SCHEDULE:
        recorder = ForwardRecorder(variant, layout)
        sample = {
            "phase": phase,
            "variant": variant,
            "raw_response": None,
            "error": None,
            "memory_before": memory(),
        }
        report["samples"].append(sample)
        hook = model.backbone.register_forward_hook(recorder, with_kwargs=True)
        started, failure = None, None
        try:
            synchronize()
            started = clock()
            raw = operations[variant](request)
            sample["raw_response"] = raw
            synchronize()
            sample["latency_ms"] = (clock() - started) * 1000
            normalize_response(request, raw)
            expected_forwards = {"A": 1, "B": len(layout["heads"]), "D": 2}[variant]
            if len(recorder.rows) != expected_forwards:
                raise ValueError("native forward count differs from the declared variant")
        except Exception as exc:
            failure = exc
            sample["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if started is not None and "latency_ms" not in sample:
                synchronize()
                sample["latency_ms"] = (clock() - started) * 1000
        finally:
            hook.remove()
            sample["forwards"] = recorder.finish()
            if failure is None and any(
                "capture_error" in slot
                for row in sample["forwards"]
                for slot in row["selected_slots"]
            ):
                failure = RuntimeError("selected-slot vector capture failed")
                sample["error"] = {"type": type(failure).__name__, "message": str(failure)}
            sample["memory_after"] = memory()
            # Invalid raw values remain auditable without breaking the receipt.
            try:
                json.dumps(sample["raw_response"], allow_nan=False)
            except (TypeError, ValueError, OverflowError):
                sample["raw_response_repr"] = repr(sample.pop("raw_response"))
                sample["raw_response"] = None
            write_json(Path(output) / "report.json", report)
        if failure is not None:
            raise failure
    return summarize(request, report["samples"])


def run_real_diagnostic(args, request, report):
    """All Torch/model imports and accelerator access occur after queue validation."""
    import torch
    from run_ultrafast import verify_checkpoint

    from s1.evaluation.gemma import predict_batch, predict_sequential
    from s1.evaluation.prefix import predict_independent_prefix
    from s1.profiling import memory_snapshot, synchronize_device
    from s1.unified import UnifiedDecisionModel

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; no alternate device is allowed")
    report["checkpoint"] = verify_checkpoint(args.model)
    started = time.perf_counter()
    model = UnifiedDecisionModel(
        str(args.model.resolve()), device="mps", dtype="bfloat16", attn_implementation="sdpa"
    )
    synchronize_device("mps")
    report["load_seconds"] = time.perf_counter() - started
    if model.device != "mps" or model.dtype != "bfloat16":
        raise RuntimeError("diagnostic requires the native MPS BF16 runtime")
    report["model"] = {
        "name": model.name,
        "revision": model.revision,
        "device": model.device,
        "dtype": model.dtype,
        "temperature": model.temperature,
        "attn_implementation": model.attn_implementation,
        "max_context": model.max_context,
        "compile_mode": getattr(model, "compile_mode", None),
        "quantization": model.quantization,
        "softcap": model.softcap,
    }
    report["slot_layout"] = slot_layout(model, request)
    report["status"] = "running"
    operations = {
        "A": lambda r: predict_batch(model, [r], independent=True)[0],
        "B": lambda r: predict_sequential(model, r),
        "D": lambda r: predict_independent_prefix(model, [r])[0],
    }
    return run_schedule(
        model,
        request,
        report["slot_layout"],
        operations,
        report,
        args.output,
        synchronize=lambda: synchronize_device("mps"),
        memory=lambda: memory_snapshot("mps"),
        clock=time.perf_counter,
    )


def main(argv=None):
    from s1.profiling import environment_metadata

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/release/checkpoint")
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--development-task", default="dev-design-free")
    parser.add_argument("--decision-index", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-for-pid", type=int, action="append", default=[])
    args = parser.parse_args(argv)
    if any(pid <= 0 for pid in args.wait_for_pid) or len(set(args.wait_for_pid)) != len(
        args.wait_for_pid
    ):
        parser.error("wait-for-pid values must be positive and unique")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "initializing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Single development observation; MPS numerical diagnostic; no production adoption or speed claim",
        "environment": environment_metadata(),
        "max_probability_delta": MAX_PROBABILITY_DELTA,
        "gate": "all argmax agree and every measured pair has max probability delta <= 0.01",
        "schedule": [{"phase": phase, "variant": variant} for phase, variant in SCHEDULE],
        "variants": {
            "A": "padded complete independent prompts in one native batch",
            "B": "complete singleton prompts in separate native forwards",
            "D": "request-local native prefix then cached independent suffixes",
        },
        "timing_scope": "synchronized instrumented execution; selected-vector clone hooks included, CPU vector export excluded",
        "defaults_changed": False,
        "reported_cost_usd": None,
        "samples": [],
    }
    write_json(args.output / "report.json", report)
    try:
        request, report["input"] = load_development_snapshot(
            args.trace_root, args.development_task, args.decision_index
        )
        report["request"] = request.model_dump()
        write_json(args.output / "request.json", report["request"])
        report["request_file_sha256"] = sha256(args.output / "request.json")
        report["source_sha256"] = archive_sources(args.output)
        report["status"] = "waiting" if args.wait_for_pid else "initializing"
        write_json(args.output / "report.json", report)
        wait_for_predecessors(args.wait_for_pid, report, args.output)
        validate_frozen_sources(args.output, report["source_sha256"])
        if (
            content_sha256(request.model_dump()) != report["input"]["request_sha256"]
            or sha256(args.output / "request.json") != report["request_file_sha256"]
        ):
            raise ValueError("frozen input changed while queued")
        report["source_validation"] = "unchanged after predecessor termination"
        report["summary"] = run_real_diagnostic(args, request, report)
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write_json(args.output / "report.json", report)
    print(json.dumps(report["summary"]), flush=True)
    return 0 if report["summary"]["all_pairwise_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
