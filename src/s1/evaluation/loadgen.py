"""Bounded concurrency and fixed/Poisson arrivals with uncensored completion times.

Arrival times are computed before workers run, so saturation cannot reduce offered
load. Backend timeouts remain the backend's responsibility; an SLO is not a fake
completion time or a claim that an in-process forward can be safely cancelled.
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from s1.errors import BackendTimeoutError

from .responses import execution_info, normalize, pipeline_info


def arrival_offsets(count, mode, rate, seed):
    if mode == "closed_loop":
        return [None] * count
    if mode not in ("fixed", "poisson") or rate is None or rate <= 0:
        raise ValueError("invalid arrival configuration")
    rng = random.Random(seed)
    offsets, current = [], 0.0
    for _ in range(count):
        offsets.append(current)
        current += 1 / rate if mode == "fixed" else rng.expovariate(rate)
    return offsets


def run_load(adapter, indexed_cases, profile, probability_mode, now, repetition):
    batches = [
        indexed_cases[i : i + profile.batch_size]
        for i in range(0, len(indexed_cases), profile.batch_size)
    ]
    offsets = arrival_offsets(
        len(batches), profile.load_mode, profile.arrival_rate, profile.seed + repetition
    )
    epoch = now()

    def worker(batch_index, batch, offset):
        scheduled = epoch + offset if offset is not None else None
        if scheduled is not None:
            time.sleep(max(0, scheduled - now()))
        dispatched = now()
        scheduled = dispatched if scheduled is None else scheduled
        try:
            requests = [case.request.model_copy(deep=True) for _, case in batch]
            if profile.batch_size > 1:
                results = adapter.predict_batch(requests)
                if not isinstance(results, list) or len(results) != len(batch):
                    raise ValueError("batch response count mismatch")
            else:
                results = [adapter.predict(requests[0])]
        except Exception as exc:
            results = [exc] * len(batch)
        completed = []
        for (index, case), result in zip(batch, results):
            response, error = None, None
            try:
                if isinstance(result, Exception):
                    raise result
                response = normalize(case.request, result, probability_mode)
                status = "ok"
            except Exception as exc:
                error = type(exc).__name__
                status = "timeout" if isinstance(exc, BackendTimeoutError) else "error"
            terminated = now()
            completed.append(
                (
                    index,
                    case,
                    {
                        "status": status,
                        "response": response,
                        "error_type": error,
                        "scheduled_at_s": scheduled,
                        "dispatched_at_s": dispatched,
                        "terminated_at_s": terminated,
                        "all_required_answers_ready_at_s": terminated if status == "ok" else None,
                        "latency_ms": (terminated - scheduled) * 1000 if status == "ok" else None,
                        "service_latency_ms": (terminated - dispatched) * 1000
                        if status == "ok"
                        else None,
                        "dispatch_lag_ms": (dispatched - scheduled) * 1000,
                        "deadline_miss": (
                            status != "ok" or (terminated - scheduled) * 1000 > profile.slo_ms
                        )
                        if profile.slo_ms is not None
                        else None,
                        "batch_id": f"{repetition}:{batch_index}",
                        "batch_size": len(batch),
                        "adapter_execution": execution_info(result),
                        "probability_postprocessing": (
                            "bounded_four_decimal_renormalization"
                            if isinstance(result, dict)
                            and result.get("probability_postprocessing")
                            == "bounded_four_decimal_renormalization"
                            else None
                        ),
                        "pipeline": pipeline_info(
                            result.get("pipeline")
                            if isinstance(result, dict)
                            else getattr(result, "evaluation_pipeline", None)
                        ),
                    },
                )
            )
        return completed

    # Flush records on the owning thread as each batch completes. Futures never
    # write artifacts concurrently. Draining workers precedes adapter.close().
    pool = ThreadPoolExecutor(max_workers=profile.concurrency)
    try:
        futures = [
            pool.submit(worker, i, batch, offset)
            for i, (batch, offset) in enumerate(zip(batches, offsets))
        ]
        for future in as_completed(futures):
            yield from future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def systems_metrics(calls, predictions):
    """Per-repetition measurement windows exclude warmup and model setup."""
    import numpy as np

    measured = [r for r in calls if r["phase"] == "measurement" and r["scheduled_at_s"] is not None]
    if not measured:
        return None
    repetitions = {r["repetition"] for r in measured}
    seconds = sum(
        max(r["terminated_at_s"] for r in measured if r["repetition"] == repetition)
        - min(r["scheduled_at_s"] for r in measured if r["repetition"] == repetition)
        for repetition in repetitions
    )
    success = [r for r in measured if r["status"] == "ok"]
    ontime = {r["request_id"] for r in success if r.get("deadline_miss") is False}
    correct = [r for r in predictions if r["status"] == "ok" and r["actual_label"] == r["gold"]]
    critical = {}
    for r in predictions:
        if r.get("critical"):
            critical.setdefault(r["request_id"], []).append(
                r["status"] == "ok" and r["actual_label"] == r["gold"]
            )
    latencies = [r["latency_ms"] for r in success]
    has_slo = any(r.get("deadline_miss") is not None for r in measured)
    return {
        "measurement_seconds": seconds,
        "requests_per_second": len(success) / seconds if seconds else None,
        "decisions_per_second": sum(r["decisions"] for r in success) / seconds if seconds else None,
        "correct_within_slo_per_second": sum(r["request_id"] in ontime for r in correct) / seconds
        if seconds and has_slo
        else None,
        "critical_correct_within_slo_requests_per_second": sum(
            key in ontime and all(values) for key, values in critical.items()
        )
        / seconds
        if seconds and critical and has_slo
        else None,
        "deadline_misses": sum(r.get("deadline_miss") is True for r in measured)
        if has_slo
        else None,
        "dispatch_lag_p95_ms": float(
            np.percentile([r.get("dispatch_lag_ms", 0) for r in measured], 95)
        ),
        "successful_p99_ms": float(np.percentile(latencies, 99)) if latencies else None,
        "p99_precision": "screening_only" if len(success) < 10000 else "check_time_blocks_and_ci",
        "timeout_observations": "censored; excluded from successful latency percentiles",
    }
