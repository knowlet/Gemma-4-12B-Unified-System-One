"""Matched public-data quality, memory and HTTP load campaign.

Every cell owns its model lifetime. The ordinary evaluation runner retains its
eligibility, provenance, prediction, and runtime-identity checks. Load uses the
same BoolQ cases and the existing uncensored fixed/Poisson/closed-loop generator.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .adapters import load_adapter
from .artifacts import read_records, write_json
from .contracts import Budget, ProfileSpec, SuiteSpec
from .datasets import audit_splits, fingerprint, load_cases, request_fingerprint
from .metrics import quality_metrics
from .runner import execute

DATASET_PINS = {
    "text-test.jsonl": (128, "7bdb25dc3d8a0371bb73909fe55597de86f66e808bc42d1a5a1cb17d08f44d41"),
    "media-test.jsonl": (52, "7f3dff9c8fcfb164df5f8ca05b0fdaadbfb3a84812bfbe4c9b1906b6954d9f35"),
}
GEMMA_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
CE_DIGESTS = (
    "36c2a0b51a2e13e8f2a3bfad85855451b85461087f490822e69dbd7846e478e2",
    "7c2b5fa18e37c354fc50aaef5a3adc330844469f00f2cc1f57a8987794a48d94",
    "06644c2692e844bce802985eecbc82744d73b428026f0926ed0f94dba9f45866",
)


def verify_datasets(directory):
    directory = Path(directory)
    result = {}
    for name, (count, digest) in DATASET_PINS.items():
        cases = load_cases(directory / name)
        if len(cases) != count or fingerprint(cases) != digest:
            raise ValueError(f"matched historical dataset required: {name}")
        result[name] = {"cases": count, "sha256": digest}
    paths = [
        directory / name
        for name in ("text-train-full.jsonl", "text-calibration.jsonl", *DATASET_PINS)
    ]
    audit = audit_splits(paths)
    if not audit["valid"]:
        raise ValueError("train/calibration/test overlap")
    return {
        "sha256": result["text-test.jsonl"]["sha256"],
        "cases": 128,
        "media_sha256": result["media-test.jsonl"]["sha256"],
        "media_comparability": "exact52 archived requests, case IDs, labels and historical dataset fingerprint",
        "files": result,
        "split_audit": audit,
    }


def campaign_specs(registry, adapter_root, *, device="cuda"):
    """Predeclare all seeds and precisions; never choose an adapter by test score."""
    common = {"enabled": True, "device": device, "cost_per_request_usd": 0.02}
    base = registry.models["gemma-g4"].model_copy(
        update={
            **common,
            "id": "gemma-base-bf16",
            "revision": GEMMA_REVISION,
            "precision": "bfloat16",
            "quantization": "none",
        }
    )
    specs = [base]
    for seed, digest in enumerate(CE_DIGESTS):
        for mode in ("none", "int8", "nf4"):
            precision = "bf16" if mode == "none" else mode
            specs.append(
                base.model_copy(
                    update={
                        "id": f"gemma-ce128-s{seed}-{precision}",
                        "precision": "bfloat16",
                        "quantization": mode,
                        "calibration": "checkpoint",
                        "decision_adapter_path": str(
                            Path(adapter_root) / f"shots-128-seed-{seed}-ce"
                        ),
                        "decision_adapter_sha256": digest,
                    }
                )
            )
    specs.append(
        registry.models["laya-general"].model_copy(
            update={
                **common,
                "revision": LAYA_REVISION,
                "precision": "provider",
                "capabilities": registry.models["laya-general"].capabilities.model_copy(
                    update={
                        "max_options": 52,
                        "max_questions": 64,
                        "concurrent_requests": False,
                        "independent_questions": False,
                    }
                ),
            }
        )
    )
    for name in ("decider-local", "kev-local", "agentjev-local", "clef-local", "jev-omni-local"):
        specs.append(registry.models[name].model_copy(update=common))
    return specs


def planned_row(spec, status="not_run", error=None):
    return {
        "model_id": spec.id,
        "checkpoint": spec.model_id,
        "revision": spec.revision,
        "precision": spec.precision,
        "quantization": spec.quantization,
        "adapter_sha256": spec.decision_adapter_sha256,
        "status": status,
        "error": error,
        "gpu": None,
        "boolq": None,
        "media": None,
        "memory": None,
        "load": [],
        "artifacts": {},
    }


def percentiles(values):
    return {
        **{f"p{p}_ms": float(np.percentile(values, p)) if values else None for p in (50, 95, 99)},
        "mean_ms": float(np.mean(values)) if values else None,
    }


def summarize_quality(output, cases):
    predictions = read_records(output, "predictions")
    requests = read_records(output, "requests")
    expected = sum(len(c.request.questions) for c in cases)
    valid = [r for r in predictions if r["status"] == "ok"]
    correct = sum(r["actual_label"] == r["gold"] for r in valid)
    attempted = any(r["status"] in ("ok", "error", "timeout") for r in predictions)
    unsupported = sum(r["eligibility"] == "unsupported" for r in predictions)
    eligible = expected - unsupported
    metrics = quality_metrics(predictions)
    by_task = {}
    for task in sorted({c.task_id for c in cases}):
        task_rows = [r for r in predictions if r["task_id"] == task]
        task_expected = sum(len(c.request.questions) for c in cases if c.task_id == task)
        task_unsupported = sum(r["eligibility"] == "unsupported" for r in task_rows)
        task_valid = [r for r in task_rows if r["status"] == "ok"]
        task_attempted = any(r["status"] in ("ok", "error", "timeout") for r in task_rows)
        task_correct = sum(r["actual_label"] == r["gold"] for r in task_valid)
        task_eligible = task_expected - task_unsupported
        by_task[task] = {
            "expected": task_expected,
            "eligible": task_eligible,
            "unsupported": task_unsupported,
            "valid": len(task_valid),
            "correct": task_correct if task_attempted else None,
            "accuracy": task_correct / task_expected
            if task_attempted and not task_unsupported
            else None,
            "supported_accuracy": task_correct / task_eligible
            if task_attempted and task_eligible
            else None,
            "native_status": "unsupported"
            if task_unsupported == task_expected
            else "completed"
            if len(task_valid) == task_expected
            else "partial_support"
            if len(task_valid) + task_unsupported == task_expected
            else "partial",
        }
    return {
        "expected": expected,
        "recorded": len(predictions),
        "valid": len(valid),
        "correct": correct if attempted else None,
        "accuracy": correct / expected if attempted and not unsupported else None,
        "eligible": eligible,
        "supported_accuracy": correct / eligible if attempted and eligible else None,
        "status": "completed" if len(valid) == expected else "partial" if attempted else "not_run",
        "dataset_sha256": fingerprint(cases),
        "errors": sum(r["status"] in ("error", "timeout") for r in predictions),
        "unsupported": unsupported,
        "nll": metrics.get("nll"),
        "brier_sum": metrics.get("brier_sum"),
        "ece_15": metrics.get("ece_15"),
        **percentiles(
            [
                r["latency_ms"]
                for r in requests
                if r["phase"] == "measurement" and r["status"] == "ok"
            ]
        ),
        "latency_scope": "CUDA-synchronized adapter wall time; excludes setup and warmup",
        "by_task": by_task,
    }


def acceptance(row):
    """An engineering target on a fixed screening set, not a population guarantee."""
    mode = row.get("quantization")
    if mode not in ("int8", "nf4"):
        return {"status": "not_applicable"}
    quality, memory = row.get("boolq") or {}, row.get("memory") or {}
    accuracy, peak = quality.get("accuracy"), memory.get("peak_allocated_bytes")
    cap = 12_000_000_000 if mode == "int8" else 8_000_000_000
    measured = quality.get("valid") == 128 and accuracy is not None and peak is not None
    return {
        "status": "passed"
        if measured and accuracy >= 0.90 and peak <= cap
        else "failed"
        if measured
        else "not_measured",
        "accuracy_at_least_90pct": accuracy >= 0.90 if measured else None,
        "accuracy_above_89pct": accuracy > 0.89 if measured else None,
        "peak_allocated_cap_bytes": cap,
        "memory_target_pass": peak <= cap if measured else None,
        "scope": "fixed 128-case point estimate and measured process CUDA allocation; not minimum GPU capacity",
    }


class SharedAdapter:
    def __init__(self, adapter, sync=lambda: None):
        self.adapter, self.sync = adapter, sync
        self.name = adapter.spec.id
        self.calls = 0

    def predict(self, request):
        self.sync()
        result = self.adapter.predict(request)
        self.sync()
        self.calls += 1
        return result

    def telemetry(self):
        return self.adapter.telemetry()

    def close(self):
        pass


def run_quality(registry, spec, shared, data, output, lockfile):
    result = {}
    for name, filename, media in (
        ("boolq", "text-test.jsonl", False),
        ("media", "media-test.jsonl", True),
    ):
        suite = SuiteSpec(
            id=name,
            dataset=str(Path(data, filename).resolve()),
            dataset_sha256=DATASET_PINS[filename][1],
            track="native_multimodal"
            if media
            else "specialist"
            if spec.decision_adapter_path
            else "generalist",
            information_view="native_media" if media else "raw_text",
        )
        native = not media or bool({"image", "audio"} & set(spec.capabilities.modalities or ()))
        profile = ProfileSpec(
            id=name,
            suite=name,
            models=(spec.id,),
            warmup=3 if native else 0,
            budget=Budget(max_requests=200, max_estimated_cost_usd=4.0),
        )
        current = replace(
            registry, models={spec.id: spec}, suites={name: suite}, profiles={name: profile}
        )
        execute(
            current,
            name,
            output=output / name,
            lockfile=lockfile,
            adapter_factory=lambda *_: shared,
        )
        result[name] = summarize_quality(output / name, load_cases(data / filename))
        if media:
            quality = result[name]
            quality["native_status"] = (
                "unsupported"
                if quality["unsupported"] == quality["expected"]
                else "partial_support"
                if quality["unsupported"] > 0
                and quality["valid"] + quality["unsupported"] == quality["expected"]
                else quality["status"]
            )
    return result


def run_http_load(shared, cases, output, serve, *, count=128, arrival_rate=5.0):
    from s1.backends import HTTPBackend

    from .loadgen import run_load, systems_metrics

    url, server, thread = serve(shared)
    client = HTTPBackend(url + "/decide", token="ephemeral-validation", timeout=120)
    result = []
    schedule = [("closed_loop", c) for c in (1, 4, 16, 64)] + [("fixed", 16), ("poisson", 16)]
    try:
        for mode, concurrency in schedule:
            name = f"{mode}-c{concurrency}-n{count}"
            profile = ProfileSpec(
                id=name,
                suite="load",
                models=(shared.name,),
                concurrency=concurrency,
                load_mode=mode,
                seed=20260930,
                arrival_rate=arrival_rate if mode != "closed_loop" else None,
                slo_ms=1000.0,
                budget=Budget(max_requests=count),
            )
            calls, predictions = [], []
            before = shared.calls
            indexed = [(i, cases[i % len(cases)]) for i in range(count)]
            with (output / f"{name}.jsonl").open("x") as stream:
                for index, case, row in run_load(
                    client, indexed, profile, "complete", time.perf_counter, 0
                ):
                    response = row.pop("response")
                    row.update(
                        request_id=f"{name}:{index}",
                        phase="measurement",
                        repetition=0,
                        decisions=len(case.request.questions),
                        case_id=case.id,
                        request_sha256=request_fingerprint(case.request),
                    )
                    calls.append(row)
                    if response:
                        for question in case.request.questions:
                            answer = response["answers"][question.id]
                            label = answer.get("choice") or max(
                                question.labels(), key=answer["probabilities"].get
                            )
                            predictions.append(
                                {
                                    "request_id": row["request_id"],
                                    "status": "ok",
                                    "actual_label": label,
                                    "gold": case.gold[question.id],
                                }
                            )
                    stream.write(json.dumps({**row, "response": response}, allow_nan=False) + "\n")
                    stream.flush()
            good = [r for r in calls if r["status"] == "ok"]
            result.append(
                {
                    "id": name,
                    "arrival": mode,
                    "concurrency": concurrency,
                    "arrival_rate": profile.arrival_rate,
                    "requests": count,
                    "successful": len(good),
                    "slo_misses": sum(r["deadline_miss"] for r in calls),
                    "completed_backend_calls": shared.calls - before,
                    **percentiles([r["latency_ms"] for r in good]),
                    "systems": systems_metrics(calls, predictions),
                    "scope": "same 128 BoolQ cases; real loopback HTTP; serialized model; no result cache; screening p99",
                }
            )
            write_json(output / "load-summary.json", {"cells": result})
    finally:
        client.close()
        server.should_exit = True
        thread.join(timeout=15)
        if thread.is_alive():
            raise RuntimeError("HTTP load server failed to stop")
    return result


def run_cell(registry, spec, data, output, lockfile, serve, *, count=128, arrival_rate=5.0):
    import importlib.metadata

    import torch

    from s1.resources import memory_snapshot

    data, output, lockfile = Path(data), Path(output), Path(lockfile)
    dataset = verify_datasets(data)
    output.mkdir(parents=True, exist_ok=False)
    row = planned_row(spec, "running")
    row.update(
        dataset=dataset,
        started_at_unix=time.time(),
        lock_sha256=hashlib.sha256(lockfile.read_bytes()).hexdigest(),
        packages={
            n: importlib.metadata.version(n)
            for n in ("torch", "transformers", "peft", "bitsandbytes", "laya")
        },
    )
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        row["gpu"] = torch.cuda.get_device_name()
    adapter = None
    started = time.perf_counter()
    write_json(output / "receipt.json", row)
    try:
        adapter = load_adapter(spec, {})
        row["load_seconds"] = time.perf_counter() - started
        row["telemetry"] = adapter.telemetry()
        model = getattr(adapter.backend, "model", None)
        if model is None and hasattr(adapter.backend, "agent"):
            model = getattr(adapter.backend.agent, "model", None)
        module = model if isinstance(model, torch.nn.Module) else getattr(model, "lm", model)
        shared = SharedAdapter(
            adapter, torch.cuda.synchronize if spec.device.startswith("cuda") else lambda: None
        )
        row["loading_memory"] = memory_snapshot(module, device=spec.device)
        if spec.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        row.update(run_quality(registry, spec, shared, data, output, lockfile))
        print(f"{spec.id}: quality complete, BoolQ {row['boolq']['correct']}/128", flush=True)
        row["memory"] = memory_snapshot(module, device=spec.device)
        row["memory"]["cuda_peak_scope"] = "post_load_quality_warmup_media_and_http_load"
        write_json(output / "receipt.json", row)
        # Do not serve a model whose runtime identity/provenance failed quality setup.
        if row["boolq"]["valid"] == 0:
            raise RuntimeError(
                "quality harness produced zero valid decisions; inspect manifests/errors"
            )
        row["load"] = run_http_load(
            shared,
            load_cases(data / "text-test.jsonl"),
            output,
            serve,
            count=count,
            arrival_rate=arrival_rate,
        )
        row["memory"] = memory_snapshot(module, device=spec.device)
        row["status"] = (
            "completed"
            if row["boolq"]["valid"] == 128
            and all(c["successful"] == c["requests"] for c in row["load"])
            and row["media"].get("native_status") in ("completed", "partial_support", "unsupported")
            else "completed_with_errors"
        )
    except Exception as exc:
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if row.get("memory"):
            row["memory"]["cuda_peak_scope"] = "post_load_quality_warmup_media_and_http_load"
            row["memory"]["host_rss_bytes"] = row["memory"].get("process_peak_rss_bytes")
        row["elapsed_seconds"] = time.perf_counter() - started
        row["acceptance"] = acceptance(row)
        row["artifacts"] = {
            "receipt": "receipt.json",
            "predictions": "boolq/predictions.jsonl",
            "media_predictions": "media/predictions.jsonl",
            "load": "load-summary.json",
        }
        row["artifact_sha256"] = {
            str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "receipt.json"
        }
        write_json(output / "receipt.json", row)
        if adapter is not None:
            adapter.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row
