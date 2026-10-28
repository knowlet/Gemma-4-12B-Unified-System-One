"""Bounded MPS comparison of independent questions with request-local prefix reuse.

Only development browser snapshots and the existing native modality regressions
are consumed. This experiment does not fit calibration or alter model defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from s1.backends import normalize_response
from s1.contracts import DecisionRequest
from s1.profiling import environment_metadata, memory_snapshot, synchronize_device

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_frozen_sources(paths, expected):
    for path in paths:
        if sha256(path) != expected[str(Path(path).resolve().relative_to(ROOT))]:
            raise RuntimeError("source changed while queued; no model was loaded")


def process_identity(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart=,stat=,command="], capture_output=True, text=True
    )
    if result.stderr.strip():
        raise RuntimeError("predecessor ps observation failed; refusing concurrent MPS load")
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    if result.returncode not in (0, 1) or result.stdout.strip():
        raise RuntimeError("predecessor ps observation failed; refusing concurrent MPS load")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError as exc:
        raise RuntimeError("cannot verify predecessor process identity") from exc
    raise RuntimeError("predecessor is live but ps returned no identity")


def wait_for_process(pid, output):
    """Wait for the identified process to exit; a reused PID is a different process."""
    identity = process_identity(pid)
    if identity is None:
        raise ValueError("requested predecessor process is not live")

    # The state field can change, so use the fixed lstart prefix + command.
    def stable(value):
        return value[:24], value[24:].split(maxsplit=1)[-1]

    initial = stable(identity)
    write_json(output / "predecessor.json", {"pid": pid, "identity": identity})
    while (current := process_identity(pid)) is not None and stable(current) == initial:
        state = current[24:].split(maxsplit=1)[0]
        if state.startswith("Z"):
            break
        time.sleep(15)
    write_json(output / "predecessor-completed.json", {"pid": pid, "last_observation": current})


def load_cases(trace_root, task_id, native_dataset):
    if not task_id.startswith("dev-"):
        raise ValueError("prefix experiments must use development tasks")
    folder = Path(trace_root) / "independent" / f"{task_id}-1"
    task = json.loads((folder / "task.json").read_text())
    if task["id"] != task_id or task["split"] != "development":
        raise ValueError("reserved/transfer snapshots cannot drive this experiment")
    trace = folder / "decision-transport.json"
    cases = []
    trace_rows = json.loads(trace.read_text())
    if not trace_rows:
        raise ValueError("development trace cannot be empty")
    for index, row in enumerate(trace_rows):
        if row["error"] is not None or row["mapped_request"] is None:
            raise ValueError("development snapshot lacks a valid mapped request")
        request = DecisionRequest.model_validate(row["mapped_request"])
        cases.append((f"browser-{task_id}-{index + 1}", request))
    for line in Path(native_dataset).read_text().splitlines():
        row = json.loads(line)
        if row["split"] != "test":
            raise ValueError("native regression requires test-only inputs")
        cases.append((f"native-{row['id']}", DecisionRequest.model_validate(row["request"])))
    ids = [case_id for case_id, _ in cases]
    if not cases or len(set(ids)) != len(ids):
        raise ValueError("nonempty unique cases required")
    return cases, {
        "development_trace": str(trace),
        "development_trace_sha256": sha256(trace),
        "native_dataset": str(native_dataset),
        "native_dataset_sha256": sha256(native_dataset),
        "case_ids": ids,
        "labels_used": False,
    }


def compare_answers(request, baseline, candidate):
    baseline = normalize_response(request, baseline)
    candidate = normalize_response(request, candidate)
    differences, agreement = [], []
    for question in request.questions:
        original = baseline["answers"][question.id]
        changed = candidate["answers"][question.id]
        labels = question.labels()
        a = [original["probabilities"][key] for key in labels]
        b = [changed["probabilities"][key] for key in labels]
        differences.extend(abs(x - y) for x, y in zip(a, b, strict=True))
        agreement.append(int(np.argmax(a)) == int(np.argmax(b)))
    return {
        "max_probability_delta": max(differences),
        "all_argmax_agree": all(agreement),
        "decisions": len(agreement),
    }


def summarize_case(request, samples, max_delta):
    if any(
        sample["profile"] not in {"baseline", "prefix"}
        or sample["phase"] not in {"warmup", "measured"}
        for sample in samples
    ):
        raise ValueError("unknown measured profile or phase cannot be excluded from the population")
    measured = [sample for sample in samples if sample["phase"] == "measured"]
    baselines = [sample for sample in measured if sample["profile"] == "baseline"]
    candidates = [sample for sample in measured if sample["profile"] == "prefix"]
    if not baselines or len(baselines) != len(candidates):
        raise ValueError("balanced nonempty baseline/candidate measurements required")
    # Compare every measured output to every measured baseline: repeated BF16
    # baseline instability is retained instead of choosing its most favorable sample.
    comparisons = [
        compare_answers(request, a["response"], b["response"])
        for a in baselines
        for b in candidates
    ]
    baseline_drift = [
        compare_answers(request, baselines[0]["response"], a["response"]) for a in baselines
    ]
    max_observed = max(row["max_probability_delta"] for row in comparisons)
    agreement = all(row["all_argmax_agree"] for row in comparisons)
    timing = {
        profile: np.percentile([sample["latency_ms"] for sample in group], [50, 95]).tolist()
        for profile, group in (("baseline", baselines), ("prefix", candidates))
    }
    return {
        "samples_per_profile": len(baselines),
        "latency_ms_p50_p95": timing,
        "median_latency_ratio": timing["prefix"][0] / timing["baseline"][0],
        "max_probability_delta": max_observed,
        "all_argmax_agree": agreement,
        "baseline_repeat_max_delta": max(row["max_probability_delta"] for row in baseline_drift),
        "parity_gate_passed": agreement and max_observed <= max_delta,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/release/checkpoint")
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--development-task", default="dev-design-free")
    parser.add_argument(
        "--native-dataset", type=Path, default=ROOT / "examples/benchmarks/mps.jsonl"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repeat", type=int, default=2, help="ABBA blocks per input; no samples discarded"
    )
    parser.add_argument("--max-probability-delta", type=float, default=0.01)
    parser.add_argument("--wait-for-pid", type=int)
    args = parser.parse_args()
    if args.repeat < 1 or not 0 <= args.max_probability_delta <= 1:
        parser.error("positive repeats and probability delta in [0,1] required")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "waiting" if args.wait_for_pid else "initializing",
        "environment": environment_metadata(),
        "repeat_abba_blocks": args.repeat,
        "max_probability_delta": args.max_probability_delta,
        "gate": "all argmax agree and max absolute probability difference <= predeclared bound",
        "defaults_changed": False,
        "runtime": "MPS only",
        "reported_cost_usd": None,
        "cases": [],
    }
    write_json(args.output / "report.json", report)
    try:
        from run_ultrafast import archive_sources, verify_checkpoint

        source_paths = [
            *ROOT.glob("src/s1/**/*.py"),
            Path(__file__),
            ROOT / "scripts/run_ultrafast.py",
        ]
        archive_sources(source_paths, args.output / "sources")
        report["source_sha256"] = {
            str(path.relative_to(args.output / "sources")): sha256(path)
            for path in (args.output / "sources").rglob("*")
            if path.is_file()
        }
        write_json(args.output / "report.json", report)
        if args.wait_for_pid:
            wait_for_process(args.wait_for_pid, args.output)
        validate_frozen_sources(source_paths, report["source_sha256"])
        import torch

        from s1.evaluation.gemma import predict_batch
        from s1.evaluation.prefix import predict_independent_prefix
        from s1.unified import UnifiedDecisionModel

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable; no CPU/CUDA fallback")
        report["checkpoint"] = verify_checkpoint(args.model)
        cases, report["dataset"] = load_cases(
            args.trace_root, args.development_task, args.native_dataset
        )
        started = time.perf_counter()
        model = UnifiedDecisionModel(
            str(args.model.resolve()), device="mps", dtype="bfloat16", attn_implementation="sdpa"
        )
        synchronize_device("mps")
        report["load_seconds"] = time.perf_counter() - started
        report["status"] = "running"
        report["model"] = {
            "name": model.name,
            "revision": model.revision,
            "device": model.device,
            "dtype": model.dtype,
            "temperature": model.temperature,
            "attn_implementation": model.attn_implementation,
        }
        report["memory_after_load"] = memory_snapshot("mps")
        operations = {
            "baseline": lambda request: predict_batch(model, [request], independent=True)[0],
            "prefix": lambda request: predict_independent_prefix(model, [request])[0],
        }
        for case_id, request in cases:
            record = {"id": case_id, "request": request.model_dump(), "samples": []}
            report["cases"].append(record)
            schedules = [("warmup", ["baseline", "prefix"])] + [
                (
                    "measured",
                    ["baseline", "prefix", "prefix", "baseline"]
                    if block % 2 == 0
                    else ["prefix", "baseline", "baseline", "prefix"],
                )
                for block in range(args.repeat)
            ]
            for phase, order in schedules:
                for profile in order:
                    record["active_attempt"] = {
                        "phase": phase,
                        "profile": profile,
                        "ordinal": len(record["samples"]) + 1,
                    }
                    synchronize_device("mps")
                    started = time.perf_counter()
                    response = operations[profile](request)
                    synchronize_device("mps")
                    elapsed = (time.perf_counter() - started) * 1000
                    normalize_response(request, response)
                    record["samples"].append(
                        {
                            "phase": phase,
                            "profile": profile,
                            "latency_ms": elapsed,
                            "response": response,
                            "memory": memory_snapshot("mps"),
                        }
                    )
                    record.pop("active_attempt")
                    write_json(args.output / "report.json", report)
            record["summary"] = summarize_case(
                request, record["samples"], args.max_probability_delta
            )
            print(json.dumps({"case": case_id, **record["summary"]}), flush=True)
        report["status"] = "completed"
        report["parity_gate_passed"] = all(
            row["summary"]["parity_gate_passed"] for row in report["cases"]
        )
        report["scope"] = (
            "Bounded MPS probability/runtime experiment; browser success and both JevBench regressions still required"
        )
    except Exception as exc:
        report.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write_json(args.output / "report.json", report)
    return 0 if report["parity_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
