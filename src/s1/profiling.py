"""Synchronized request timing and separate phase diagnostics for local inference.

Phase timings use additional forwards. They are intentionally never added to,
or substituted for, end-to-end timings of the public ``predict`` method.
"""

from __future__ import annotations

import math
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from .backends import normalize_response
from .benchmark import fingerprint
from .contracts import answer_from_probabilities


def environment_metadata():
    packages = {}
    for package in ("torch", "transformers"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        **packages,
        "mps_environment": {
            name: os.environ.get(name)
            for name in (
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "PYTORCH_MPS_FAST_MATH",
                "PYTORCH_MPS_PREFER_METAL",
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                "PYTORCH_MPS_LOW_WATERMARK_RATIO",
            )
        },
    }


def synchronize_device(device):
    """Complete outstanding device work before reading a host wall clock."""
    import torch

    kind = torch.device(device).type
    if kind == "mps":
        torch.mps.synchronize()
    elif kind == "cuda":
        torch.cuda.synchronize(device)


def memory_snapshot(device):
    """Allocator observations at one instant, explicitly not peak memory."""
    snapshot = {"kind": "snapshot_not_peak", "device": str(device), "unit": "bytes"}
    if str(device).split(":")[0] != "mps":
        snapshot["available"] = False
        return snapshot
    import torch

    for name in ("current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"):
        try:
            snapshot[name] = int(getattr(torch.mps, name)())
        except (AttributeError, RuntimeError) as exc:
            snapshot[name] = None
            snapshot[f"{name}_error"] = str(exc)
    snapshot["available"] = snapshot["current_allocated_memory"] is not None
    return snapshot


def _timed(operation, synchronize, clock):
    synchronize()
    started = clock()
    result = operation()
    synchronize()
    elapsed_ms = (clock() - started) * 1000
    if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
        raise ValueError("timer produced an invalid duration")
    return result, elapsed_ms


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("inference produced a nonfinite output")
    if isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite(child)


def _validate_response(request, response):
    _finite(response)
    normalize_response(request, response)


def _latency_summary(latencies, decisions):
    total_ms = sum(latencies)
    return {
        "request_count": len(latencies),
        "decision_count": decisions,
        "total_measured_ms": total_ms,
        "latency_ms": {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        }
        if latencies
        else None,
        "requests_per_second": len(latencies) * 1000 / total_ms if total_ms > 0 else None,
        "decisions_per_second": decisions * 1000 / total_ms if total_ms > 0 else None,
    }


def profile_phases(model, request, *, synchronize, clock=time.perf_counter, set_stage=None):
    """Run a separate forward and release all prepared tensors before returning."""
    import torch

    from .unified import project_candidates

    set_stage = set_stage or (lambda stage: None)
    model.lm.eval()
    with torch.inference_mode():
        set_stage("phase_prepare")
        (inputs, slots, nopts), prepare_ms = _timed(
            lambda: model.prepare(request), synchronize, clock
        )
        input_tokens = int(inputs["input_ids"].shape[-1])
        set_stage("phase_backbone_forward")
        output, forward_ms = _timed(
            lambda: model.backbone(**inputs, use_cache=False, return_dict=True),
            synchronize,
            clock,
        )
        set_stage("phase_projection_softmax_cpu")

        def project():
            hidden = output.last_hidden_state[0, slots]
            logits = project_candidates(hidden, model.head, model.letters, nopts, model.softcap)
            return (logits / model.temperature).softmax(-1).cpu().tolist()

        probabilities, projection_ms = _timed(project, synchronize, clock)
    _finite(probabilities)
    answers = {
        q.id: answer_from_probabilities(q, p[: len(q.labels())])
        for q, p in zip(request.questions, probabilities)
    }
    _validate_response(request, {"answers": answers})
    return {
        "prepare_ms": prepare_ms,
        "backbone_forward_ms": forward_ms,
        "projection_softmax_cpu_ms": projection_ms,
        "input_tokens": input_tokens,
        "answers": answers,
    }


def profile_model(
    model,
    cases,
    *,
    warmup=1,
    repeat=3,
    synchronize=None,
    clock=time.perf_counter,
    phase_runner=profile_phases,
    progress=None,
):
    """Profile an already-loaded model, retaining partial results on failure.

    ``synchronize``, ``clock`` and ``phase_runner`` are injectable to test the
    measurement contract without downloading weights or using an accelerator.
    The command-line runner only constructs a real pretrained model.
    """
    if warmup < 0 or repeat < 1:
        raise ValueError("warmup must be nonnegative and repeat must be positive")
    if not cases or any(case.split != "test" for case in cases):
        raise ValueError("profiling requires a nonempty, test-only dataset")
    synchronize = synchronize or (lambda: synchronize_device(model.device))
    progress = progress or (lambda message: None)
    report = {
        "schema_version": 1,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_metadata(),
        "model": {
            "name": model.name,
            "resolved_revision": model.revision,
            "device": str(model.device),
            "dtype": getattr(model, "dtype", None),
            "attn_implementation": getattr(model, "attn_implementation", None),
            "temperature": model.temperature,
        },
        "dataset_sha256": fingerprint(cases),
        "case_ids": [case.id for case in cases],
        "planned_request_count": len(cases) * repeat,
        "planned_decision_count": sum(len(case.request.questions) for case in cases) * repeat,
        "warmup_per_case": warmup,
        "repeat_per_case": repeat,
        "measurement": {
            "end_to_end": "synchronized full predict; excludes warmup and phase profiling",
            "phases": "separate forwards; synchronized boundaries add overhead; not an E2E sum",
            "throughput": "serial successful E2E requests only; decisions count questions",
            "tokens": "processor-expanded input length, including all question answer slots",
            "failure_policy": "stop at first error and preserve completed samples",
        },
        "memory_snapshots": {"before_profile": memory_snapshot(model.device)},
        "cases": [],
    }
    stage, case_id, iteration = "start", None, None

    def set_stage(value):
        nonlocal stage
        stage = value

    try:
        for case in cases:
            case_id = case.id
            record = {
                "id": case.id,
                "question_count": len(case.request.questions),
                "input_tokens": None,
                "warmup_samples": [],
                "end_to_end_samples": [],
                "phase_samples": [],
            }
            report["cases"].append(record)
            stage = "warmup"
            progress(f"{case.id}: {warmup} warmup request(s)")
            for iteration in range(warmup):
                response, elapsed_ms = _timed(
                    lambda: model.predict(case.request), synchronize, clock
                )
                _validate_response(case.request, response)
                record["warmup_samples"].append(
                    {"latency_ms": elapsed_ms, "answers": response["answers"]}
                )
            stage = "end_to_end"
            progress(f"{case.id}: {repeat} measured full request(s)")
            for iteration in range(repeat):
                response, elapsed_ms = _timed(
                    lambda: model.predict(case.request), synchronize, clock
                )
                _validate_response(case.request, response)
                record["end_to_end_samples"].append(
                    {"latency_ms": elapsed_ms, "answers": response["answers"]}
                )
            progress(f"{case.id}: {repeat} separate phase profile(s)")
            for iteration in range(repeat):
                stage = "phase_profile"
                sample = phase_runner(
                    model, case.request, synchronize=synchronize, clock=clock, set_stage=set_stage
                )
                _finite(sample)
                _validate_response(case.request, {"answers": sample["answers"]})
                if record["input_tokens"] not in (None, sample["input_tokens"]):
                    raise ValueError("input token count changed between repeated phase samples")
                record["input_tokens"] = sample["input_tokens"]
                record["phase_samples"].append(sample)
        report["status"] = "ok"
    except Exception as exc:
        report["status"] = "error"
        report["failure"] = {
            "stage": stage,
            "case_id": case_id,
            "iteration": iteration,
            "type": type(exc).__name__,
            "message": str(exc),
        }
    latencies, decisions, tokens = [], 0, 0
    known_tokens = True
    for record in report["cases"]:
        case_latencies = [sample["latency_ms"] for sample in record["end_to_end_samples"]]
        count = len(case_latencies)
        case_decisions = count * record["question_count"]
        record["summary"] = _latency_summary(case_latencies, case_decisions)
        record["phase_latency_ms"] = (
            {
                key: {
                    "p50": float(np.percentile([s[key] for s in record["phase_samples"]], 50)),
                    "p95": float(np.percentile([s[key] for s in record["phase_samples"]], 95)),
                }
                for key in ("prepare_ms", "backbone_forward_ms", "projection_softmax_cpu_ms")
            }
            if record["phase_samples"]
            else None
        )
        latencies.extend(case_latencies)
        decisions += case_decisions
        if count and record["input_tokens"] is None:
            known_tokens = False
        elif record["input_tokens"] is not None:
            tokens += count * record["input_tokens"]
    report["summary"] = _latency_summary(latencies, decisions)
    report["summary"]["total_measured_input_tokens"] = tokens if known_tokens else None
    report["summary"]["request_coverage"] = len(latencies) / report["planned_request_count"]
    report["summary"]["decision_coverage"] = decisions / report["planned_decision_count"]
    report["summary"]["phase_request_count"] = sum(
        len(record["phase_samples"]) for record in report["cases"]
    )
    report["summary"]["phase_coverage"] = (
        report["summary"]["phase_request_count"] / report["planned_request_count"]
    )
    attempted = {record["id"] for record in report["cases"]}
    report["unattempted_case_ids"] = [case.id for case in cases if case.id not in attempted]
    report["memory_snapshots"]["after_profile"] = memory_snapshot(model.device)
    return report
