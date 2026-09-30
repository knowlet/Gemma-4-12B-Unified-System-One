"""Real HTTP serving/load and pinned Laya checks on ephemeral GPU workers."""

from __future__ import annotations

import gc
import json
import socket
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

from s1.contracts import DecisionRequest
from s1.evaluation.adapters import ReferenceAdapter
from s1.evaluation.artifacts import write_json
from s1.evaluation.contracts import Budget, ProfileSpec, SuiteSpec
from s1.evaluation.datasets import EvaluationCase
from s1.evaluation.loadgen import run_load, systems_metrics
from s1.evaluation.registry import Registry
from s1.evaluation.runner import execute
from s1.evaluation.workflow import execute_workflows

LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"


def _registry(root):
    config = root / "configs/benchmarks"
    return Registry.load(config / "models.toml", config / "suites.toml", config / "profiles.toml")


def _serve(backend):
    import httpx
    import uvicorn

    from s1.api import create_app

    listener = socket.socket()
    server = thread = None
    try:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(create_app(backend, api_key="ephemeral-validation"), log_level="error")
        )
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{port}"
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        if not server.started:
            raise RuntimeError("HTTP server did not start")
        with httpx.Client() as client:
            unauthenticated = client.post(url + "/decide", json={})
            authenticated = client.get(
                url + "/healthz", headers={"Authorization": "Bearer ephemeral-validation"}
            )
        assert unauthenticated.status_code == 401
        assert authenticated.status_code == 200
        return url, server, thread
    except BaseException:
        try:
            if server is not None:
                server.should_exit = True
            if thread is not None and thread.ident is not None:
                thread.join(timeout=10)
        finally:
            listener.close()
        raise


def _load(model, output, *, stress=False):
    import numpy as np
    import torch

    from s1.backends import HTTPBackend

    class Backend:
        name = f"gemma:{model.name}"
        forwards = 0

        def predict(self, request):
            result = model.predict(request)
            torch.cuda.synchronize()
            self.forwards += 1
            return result

    backend = Backend()
    url, server, thread = _serve(backend)

    class ObservedHTTP(HTTPBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.guard = threading.Lock()
            self.inflight = self.peak_inflight = 0

        def predict(self, request):
            with self.guard:
                self.inflight += 1
                self.peak_inflight = max(self.peak_inflight, self.inflight)
            try:
                return super().predict(request)
            finally:
                with self.guard:
                    self.inflight -= 1

    adapter = ObservedHTTP(url + "/decide", token="ephemeral-validation", timeout=60)
    actual_forwards = [0]

    def observed_backbone(module, args, kwargs):
        actual_forwards[0] += 1

    hook = model.backbone.register_forward_pre_hook(observed_backbone, with_kwargs=True)
    request = DecisionRequest(
        state="The object is red.",
        questions={
            "color": {
                "type": "choice",
                "instructions": "Choose the object's color.",
                "criteria": {"red": "red", "blue": "blue"},
            }
        },
    )
    case = EvaluationCase(
        id="serving-color",
        group_id="synthetic-serving-color",
        split="test",
        request=request,
        gold={"color": "red"},
    )
    cells = []
    try:
        warmup = []
        for _ in range(8):
            start = time.perf_counter()
            adapter.predict(request)
            warmup.append(time.perf_counter() - start)
        rate = 0.8 / float(np.median(warmup))
        if stress:
            rate *= 1.25 / 0.8
        write_json(
            output / "server-identity.json",
            {
                "model": model.name,
                "revision": model.revision,
                "device": model.device,
                "precision": str(model.head.weight.dtype),
                "context_limit": model.max_context,
                "route": "/decide",
                "auth_positive_and_negative_pass": True,
                "server_policy": "actual CUDA forwards serialized by public API lock; no result cache",
                "client": "real loopback TCP HTTP; network/WAN latency not measured",
                "workload": "repeated synthetic short request; runtime/load evidence only",
                "warmup_seconds": warmup,
                "offered_requests_per_second": rate,
            },
        )
        schedule = [
            (mode, concurrency, 64)
            for mode in ("fixed", "poisson")
            for concurrency in (1, 4, 16, 64)
        ] + [("fixed", 16, 10000)]
        if stress:
            schedule = [
                (mode, concurrency, 128)
                for mode in ("closed_loop", "fixed", "poisson")
                for concurrency in (1, 4, 16, 64)
            ]
        for index, (mode, concurrency, count) in enumerate(schedule):
            name = f"{mode}-c{concurrency}-n{count}"
            print(f"load {name} starting", flush=True)
            profile = ProfileSpec(
                id=name,
                suite="load",
                models=("gemma-http",),
                concurrency=concurrency,
                load_mode=mode,
                arrival_rate=rate if mode != "closed_loop" else None,
                slo_ms=1000.0,
                seed=20260930,
                budget=Budget(max_requests=count, max_estimated_cost_usd=0.0),
            )
            before = backend.forwards
            before_actual = actual_forwards[0]
            adapter.peak_inflight = 0
            started = time.perf_counter()
            calls, predictions = [], []
            with (output / f"{name}.jsonl").open("w") as stream:
                for _, current, result in run_load(
                    adapter,
                    [(i, case) for i in range(count)],
                    profile,
                    "complete",
                    time.perf_counter,
                    index,
                ):
                    response = result.pop("response")
                    row = {
                        **result,
                        "request_id": f"{name}:{len(calls)}",
                        "phase": "measurement",
                        "repetition": index,
                        "decisions": 1,
                    }
                    calls.append(row)
                    if response:
                        predictions.append(
                            {
                                "request_id": row["request_id"],
                                "status": "ok",
                                "gold": "red",
                                "actual_label": response["answers"]["color"]["choice"],
                            }
                        )
                    stream.write(json.dumps({**row, "response": response}, allow_nan=False) + "\n")
                    if len(calls) % 500 == 0:
                        stream.flush()
                        print(f"load {name}: {len(calls)}/{count}", flush=True)
            success = [r for r in calls if r["status"] == "ok"]
            summary = {
                "id": name,
                "requests": count,
                "successful": len(success),
                "actual_model_forwards": backend.forwards - before,
                "observed_backbone_forwards": actual_forwards[0] - before_actual,
                "configured_concurrency": concurrency,
                "load_mode": mode,
                "offered_requests_per_second": profile.arrival_rate,
                "peak_client_inflight": adapter.peak_inflight,
                "elapsed_seconds": time.perf_counter() - started,
                "systems": systems_metrics(calls, predictions),
                "correct": sum(r["actual_label"] == r["gold"] for r in predictions),
            }
            if count == 10000 and len(success) == 10000:
                ordered = sorted(success, key=lambda r: r["scheduled_at_s"])
                blocks = np.array_split([r["latency_ms"] for r in ordered], 20)
                rng = np.random.default_rng(20260930)
                boot = [
                    float(
                        np.percentile(
                            np.concatenate([blocks[i] for i in rng.integers(0, 20, 20)]), 99
                        )
                    )
                    for _ in range(1000)
                ]
                summary["p99_time_blocks_ms"] = [float(np.percentile(b, 99)) for b in blocks]
                summary["p99_block_bootstrap_95ci_ms"] = np.percentile(boot, [2.5, 97.5]).tolist()
                summary["p99_scope"] = (
                    "20 contiguous time blocks; one machine/workload, no deployment population inference"
                )
            cells.append(summary)
            write_json(
                output / ("stress-checks.json" if stress else "load-checks.json"), {"cells": cells}
            )
            print(json.dumps(summary), flush=True)
        assert all(
            c["successful"] == c["requests"]
            and c["actual_model_forwards"] == c["requests"]
            and c["observed_backbone_forwards"] == c["requests"]
            for c in cells
        )
        return {
            "cells": cells,
            "measured_actual_forwards": sum(c["actual_model_forwards"] for c in cells),
        }
    finally:
        hook.remove()
        adapter.close()
        server.should_exit = True
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("HTTP server failed to stop after load workers drained")


def _laya(root, output, dataset_dir):
    from s1.backends import LayaBackend

    registry = _registry(root)
    spec = registry.models["laya-general"].model_copy(
        update={
            "enabled": True,
            "revision": LAYA_REVISION,
            "device": "cuda",
            "precision": "provider",
            "cost_per_request_usd": 0.02,
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
    backend = LayaBackend(spec.model_id, revision=spec.revision, device="cuda", max_len=512)

    class Shared(ReferenceAdapter):
        def close(self):
            pass

    def factory(current, environment):
        return Shared(current, backend)

    suite = SuiteSpec(
        id="laya-public",
        dataset=str(dataset_dir / "text-test.jsonl"),
        track="generalist",
        information_view="raw_text",
        notes="Laya declared max_len512 truncates; full public inputs preserved",
    )
    profile = ProfileSpec(
        id="laya-public",
        suite=suite.id,
        models=(spec.id,),
        warmup=1,
        budget=Budget(max_requests=200, max_estimated_cost_usd=4.0),
    )
    current = replace(
        registry, models={spec.id: spec}, suites={suite.id: suite}, profiles={profile.id: profile}
    )
    try:
        result = execute(
            current,
            profile.id,
            output=output / "text",
            lockfile=root / "uv.lock",
            adapter_factory=factory,
        )
        workflow = execute_workflows(
            current,
            root / "examples/benchmarks/workflows.jsonl",
            spec.id,
            spec.id,
            output=output / "workflow",
            lockfile=root / "uv.lock",
            max_calls=2000,
            max_cost_usd=40.0,
            adapter_factory=factory,
        )
        write_json(
            output / "laya-checks.json",
            {"text": result, "workflow": workflow, "resolved_backend_metadata": backend.metadata},
        )
        workflow_manifest = json.loads((output / "workflow/workflow_manifest.json").read_text())
        if (
            result["run_status"] != "completed"
            or workflow["status"] != "completed"
            or any(
                current["status"] != "completed"
                for current in workflow_manifest["execution"].values()
            )
        ):
            raise AssertionError("Laya text or workflow runtime recorded failures; see artifacts")
        return {"text": result, "workflow": workflow, "metadata": backend.metadata}
    finally:
        backend = None
        gc.collect()


def run(root: Path, output: Path, dataset_dir: Path, model_id: str, revision: str):
    import torch

    from s1.unified import UnifiedDecisionModel

    output.mkdir(parents=True, exist_ok=False)
    results, failures = {}, []
    model = UnifiedDecisionModel(model_id, revision=revision, device="cuda")
    try:
        from live_pipeline_checks import run as pipeline_check

        for name, operation in (
            ("pipeline", lambda: pipeline_check(model, root, output / "pipeline", dataset_dir)),
            ("http_load", lambda: _load(model, output)),
        ):
            try:
                results[name] = operation()
                if isinstance(results[name], dict) and results[name].get("failures"):
                    failures.append({"stage": name, "nested_failures": results[name]["failures"]})
                if name == "pipeline":
                    manifest = json.loads(
                        (output / "pipeline/workflow/workflow_manifest.json").read_text()
                    )
                    invalid = {
                        model_id: entry
                        for model_id, entry in manifest["execution"].items()
                        if entry["status"] != "completed"
                    }
                    if invalid:
                        failures.append({"stage": name, "workflow_execution_failures": invalid})
            except Exception as exc:
                failures.append(
                    {"stage": name, "error_type": type(exc).__name__, "error": str(exc)}
                )
                (output / f"{name}-failure.txt").write_text(traceback.format_exc())
                print(traceback.format_exc(), flush=True)
    finally:
        model = None
        gc.collect()
        torch.cuda.empty_cache()
    try:
        results["laya"] = _laya(root, output, dataset_dir)
    except Exception as exc:
        failures.append({"stage": "laya", "error_type": type(exc).__name__, "error": str(exc)})
        (output / "laya-failure.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    receipt = {"schema_version": 1, "results": results, "failures": failures}
    write_json(output / "services-checks.json", receipt)
    return receipt
