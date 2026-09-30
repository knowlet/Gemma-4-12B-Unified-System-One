"""Explicit live Gemma checks using one already-loaded, pinned checkpoint.

The caller owns model loading and cloud budgeting. This module neither downloads
weights nor starts cloud jobs. Its public-data metrics are screening observations;
the contract and synthetic workloads are never representative quality evidence.
"""

from __future__ import annotations

import base64
import gc
import io
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from s1.contracts import DecisionRequest, Question
from s1.evaluation.adapters import ReferenceAdapter
from s1.evaluation.artifacts import read_records, write_json
from s1.evaluation.contracts import Budget, ProfileSpec, SuiteSpec
from s1.evaluation.datasets import fingerprint, load_cases
from s1.evaluation.experiments import behavior_comparison, prepare_sweep, write_cases
from s1.evaluation.registry import Registry
from s1.evaluation.runner import execute
from s1.unified import project_candidates

PROBABILITY_ATOL = 0.01
LOGIT_ATOL = 0.03
# Operator accounting ceiling, not measured per-request billing. The parent job
# separately records GPU wall time and stops the cloud workload at its budget.
REQUEST_COST_CEILING_USD = 0.02


def _sync(model):
    import torch

    if str(model.device).startswith("cuda"):
        torch.cuda.synchronize(model.device)


def _release():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _ObservedAdapter(ReferenceAdapter):
    """Observe real forwards; closing this wrapper leaves its shared model alive."""

    def __init__(self, spec, model, trace_path):
        super().__init__(spec, SimpleNamespace(model=model))
        self.model, self.trace_path, self.forwards = model, Path(trace_path), []
        self.lm_calls = 0
        self.hooks = [
            model.backbone.register_forward_pre_hook(self._backbone, with_kwargs=True),
            model.lm.register_forward_pre_hook(self._lm, with_kwargs=True),
        ]

    def _lm(self, module, args, kwargs):
        self.lm_calls += 1

    def _backbone(self, module, args, kwargs):
        inputs = kwargs.get("input_ids")
        if inputs is None:
            inputs = next((arg for arg in args if hasattr(arg, "shape") and arg.ndim == 2), None)
        attention = kwargs.get("attention_mask")
        lengths = (
            attention.sum(-1).detach().cpu().tolist()
            if attention is not None and attention.ndim == 2
            else [int(inputs.shape[1])] * int(inputs.shape[0])
            if inputs is not None
            else []
        )
        self.forwards.append(
            {
                "batch_size": int(inputs.shape[0]) if inputs is not None else None,
                "padded_sequence_tokens": int(inputs.shape[1]) if inputs is not None else None,
                "sequence_tokens": [int(value) for value in lengths],
                "processor_tensors": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in kwargs.items()
                    if hasattr(value, "shape")
                },
                "use_cache": kwargs.get("use_cache"),
            }
        )

    def _call(self, requests, *, batch):
        self.forwards, self.lm_calls = [], 0
        _sync(self.model)
        started = time.perf_counter()
        error = None
        try:
            results = super().predict_batch(requests) if batch else [super().predict(requests[0])]
            _sync(self.model)
            execution = {
                "forward_calls": len(self.forwards),
                "batch_sizes": [row["batch_size"] for row in self.forwards],
                "sequence_tokens": [n for row in self.forwards for n in row["sequence_tokens"]],
                "scope": "whole_adapter_batch",
            }
            for result in results:
                result["execution"] = execution
            return results if batch else results[0]
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            raise
        finally:
            with self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "model_id": self.spec.id,
                            "requests": len(requests),
                            "questions": sum(len(request.questions) for request in requests),
                            "elapsed_ms": (time.perf_counter() - started) * 1000,
                            "backbone_forward_calls": len(self.forwards),
                            "lm_forward_calls": self.lm_calls,
                            "forwards": self.forwards,
                            "error": error,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
            if error:
                _release()

    def predict(self, request):
        return self._call([request], batch=False)

    def predict_batch(self, requests):
        return self._call(requests, batch=True)

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.backend = None


def _distribution_delta(request, left, right):
    rows = []
    for question in request.questions:
        a, b = left["answers"][question.id], right["answers"][question.id]
        pa, pb = a["probabilities"], b["probabilities"]
        best_a, best_b = max(pa, key=pa.get), max(pb, key=pb.get)
        delta = max(abs(pa[label] - pb[label]) for label in question.labels())
        rows.append(
            {
                "question_id": question.id,
                "probability_max_abs_delta": delta,
                "argmax_match": best_a == best_b,
                "passed": delta <= PROBABILITY_ATOL and best_a == best_b,
            }
        )
    return rows


def _fallback_media():
    import numpy as np
    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (224, 224), (255, 0, 0)).save(stream, format="PNG")
    return (
        {"type": "image", "data": base64.b64encode(stream.getvalue()).decode()},
        {"type": "audio", "samples": (0.1 * np.sin(np.arange(16000) * 0.04)).tolist()},
    )


def _parity_requests(contract_cases, media_cases):
    image, audio = _fallback_media()
    for case in media_cases:
        for medium in case.request.media:
            if medium.type == "image":
                image = medium.model_dump(mode="json")
                break
        if any(m.type == "image" for m in case.request.media):
            break
    for case in media_cases:
        for medium in case.request.media:
            if medium.type == "audio":
                audio = medium.model_dump(mode="json")
                break
        if any(m.type == "audio" for m in case.request.media):
            break
    base = contract_cases[0].request
    requests = {
        "text-short": base.model_copy(deep=True),
        "text-unequal-length": base.model_copy(
            deep=True,
            update={"state": "Neutral context. " * 128 + str(base.state)},
        ),
    }
    for name, media in (("image", [image]), ("audio", [audio]), ("mixed", [image, audio])):
        requests[name] = DecisionRequest.model_validate(
            {
                "state": "Observe the supplied media. This is a batching parity check.",
                "media": media,
                "questions": [
                    {
                        "id": "observed",
                        "instructions": "Does the supplied media contain evidence?",
                        "type": "noul",
                    },
                    {
                        "id": "kind",
                        "instructions": "Which information was supplied?",
                        "criteria": {"visual": "Visual", "audio": "Audio", "both": "Both"},
                    },
                ],
            }
        )
        requests[f"{name}-unequal-length"] = requests[name].model_copy(
            deep=True, update={"state": "Neutral context. " * 24 + requests[name].state}
        )
    return requests


def _readout_parity(model, cases):
    import torch

    rows = []
    for case in cases:
        for question in case.request.questions:
            try:
                with torch.inference_mode():
                    inputs, slots, counts = model.prepare(
                        case.request.model_copy(update={"questions": [question]})
                    )
                    output = model.backbone(**inputs, use_cache=False, return_dict=True)
                    hidden = output.last_hidden_state[0, slots]
                    candidate = project_candidates(
                        hidden, model.head, model.letters, counts, model.softcap
                    )[:, : len(question.labels())]
                    full = model.head(hidden.to(model.head.weight.dtype))
                    if model.softcap is not None:
                        full = torch.tanh(full / model.softcap) * model.softcap
                    full = full[:, model.letters[: len(question.labels())]].float()
                    pc, pf = (
                        (candidate / model.temperature).softmax(-1),
                        (full / model.temperature).softmax(-1),
                    )
                    logit_delta = (candidate - full).abs().max().item()
                    probability_delta = (pc - pf).abs().max().item()
                    argmax_match = bool(torch.equal(pc.argmax(-1), pf.argmax(-1)))
                    rows.append(
                        {
                            "case_id": case.id,
                            "question_id": question.id,
                            "sequence_tokens": inputs["input_ids"].shape[1],
                            "logit_max_abs_delta": logit_delta,
                            "probability_max_abs_delta": probability_delta,
                            "argmax_match": argmax_match,
                            "passed": logit_delta <= LOGIT_ATOL
                            and probability_delta <= PROBABILITY_ATOL
                            and argmax_match,
                        }
                    )
                    del inputs, output, hidden, candidate, full, pc, pf
            except Exception as exc:
                rows.append(
                    {
                        "case_id": case.id,
                        "question_id": question.id,
                        "passed": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    }
                )
                _release()
    return {
        "kind": "same_hidden_full_vs_candidate_projection",
        "logit_atol": LOGIT_ATOL,
        "probability_atol": PROBABILITY_ATOL,
        "comparisons": rows,
        "passed": all(row["passed"] for row in rows),
    }


def run(model, root: Path, output: Path, dataset_dir: Path) -> dict:
    """Run live tests and v2 benchmarks without reloading or freeing ``model``."""
    import torch

    root, output, dataset_dir = Path(root), Path(output), Path(dataset_dir)
    output.mkdir(parents=True, exist_ok=False)
    if not str(model.device).startswith("cuda") or model.head.weight.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("live checkpoint checks require the caller's CUDA BF16 or FP32 model")
    precision = "float32" if model.head.weight.dtype == torch.float32 else "bfloat16"
    if not model.revision or len(model.revision) not in (40, 64):
        raise ValueError("live checkpoint checks require an immutable checkpoint revision")
    original_temperature = model.temperature
    config = root / "configs/benchmarks"
    registry = Registry.load(
        config / "models.toml", config / "suites.toml", config / "profiles.toml"
    )
    selected_models = {}
    for name in ("gemma-g0", "gemma-g1", "gemma-g2", "gemma-g3", "gemma-g4"):
        selected_models[name] = registry.models[name].model_copy(
            update={
                "enabled": True,
                "model_id": model.name,
                "revision": model.revision,
                "processor_revision": model.revision,
                "device": str(model.device),
                "precision": precision,
                "context_limit": model.max_context,
                "calibration": "none",
                "cost_per_request_usd": REQUEST_COST_CEILING_USD,
                "hardware_label": torch.cuda.get_device_name(model.device),
            }
        )
    registry = replace(registry, models=selected_models)
    contract_cases = load_cases(root / "examples/benchmarks/text-contract.jsonl")
    media_cases = load_cases(dataset_dir / "media-test.jsonl")
    started = time.perf_counter()
    receipt = {
        "kind": "pinned_pretrained_checkpoint_live_validation",
        "status": "running",
        "checkpoint": model.name,
        "revision": model.revision,
        "precision": str(model.head.weight.dtype),
        "device": str(model.device),
        "hardware": torch.cuda.get_device_name(model.device),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(model.device).total_memory,
        "temperature": 1.0,
        "runner_result_cache": False,
        "runner_runs": [],
        "failures": [],
        "limitations": [
            "Contract/synthetic fixtures measure implementation and systems behavior, not quality.",
            "Public datasets are bounded screening samples; no production/noninferiority claim.",
            "CUDA is synchronized around observed adapter calls; hook accounting adds overhead.",
            "Per-request costs are operator ceilings, not actual Modal charges.",
            "Batch sizes count requests; each question expands to an independent sequence in G4.",
        ],
    }

    def persist():
        receipt["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "receipt.json", receipt)

    def benchmark(
        name,
        dataset,
        model_names,
        *,
        repetitions=1,
        warmup=1,
        batch_size=1,
        track="systems",
        information_view="raw_text",
    ):
        cases = load_cases(dataset)
        suite = SuiteSpec(
            id=name,
            dataset=str(Path(dataset).resolve()),
            track=track,
            information_view=information_view,
            dataset_sha256=fingerprint(cases),
        )
        requests = len(model_names) * repetitions * (len(cases) + warmup)
        profile = ProfileSpec(
            id=name,
            suite=name,
            models=tuple(model_names),
            repetitions=repetitions,
            warmup=warmup,
            batch_size=batch_size,
            require_native_batch=batch_size > 1,
            budget=Budget(
                max_requests=requests,
                max_estimated_cost_usd=requests * REQUEST_COST_CEILING_USD + 0.01,
            ),
        )
        configured = replace(registry, suites={name: suite}, profiles={name: profile})
        directory = output / name
        trace = output / f"{name}-forwards.jsonl"
        print(f"LIVE_CHECK {name}: {len(cases)} cases; {list(model_names)}", flush=True)
        torch.cuda.reset_peak_memory_stats(model.device)
        summary = execute(
            configured,
            name,
            output=directory,
            lockfile=root / "uv.lock",
            enable_models=model_names,
            adapter_factory=lambda spec, environment: _ObservedAdapter(spec, model, trace),
        )
        errors = read_records(directory, "errors")
        traces = (
            [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
        )
        forward_rows = [row for call in traces for row in call["forwards"]]
        token_lengths = [n for row in forward_rows for n in row["sequence_tokens"]]
        entry = {
            "id": name,
            "directory": str(directory),
            "status": summary["run_status"],
            "cases": len(cases),
            "repetitions": repetitions,
            "warmup": warmup,
            "batch_size": batch_size,
            "dataset_sha256": fingerprint(cases),
            "models": {
                key: {
                    field: stats[field]
                    for field in (
                        "status",
                        "reasons",
                        "telemetry",
                        "resources",
                        "denominators",
                        "records_complete",
                        "quality",
                        "systems",
                        "request_counts",
                        "execution_coverage",
                        "error_count",
                        "successful_request_latency_ms",
                    )
                    if field in stats
                }
                for key, stats in summary["models"].items()
            },
            "forward_trace": trace.name,
            "observed_execution": {
                "backbone_forward_calls": len(forward_rows),
                "lm_forward_calls": sum(call["lm_forward_calls"] for call in traces),
                "processed_sequences": len(token_lengths),
                "sequence_token_min": min(token_lengths) if token_lengths else None,
                "sequence_token_max": max(token_lengths) if token_lengths else None,
                "max_sequence_batch_size": max(
                    (row["batch_size"] or 0 for row in forward_rows), default=0
                ),
                "failed_adapter_calls": sum(call["error"] is not None for call in traces),
            },
            "errors": errors,
        }
        receipt["runner_runs"].append(entry)
        if summary["run_status"] != "completed" or errors:
            receipt["failures"].append(
                {"check": name, "status": summary["run_status"], "errors": errors}
            )
        persist()
        _release()
        return entry

    def direct(name, callback):
        print(f"LIVE_CHECK {name}", flush=True)
        try:
            result = callback()
        except Exception as exc:
            result = {"passed": False, "error_type": type(exc).__name__, "error": str(exc)[:1000]}
            _release()
        write_json(output / f"{name}.json", result)
        if not result["passed"]:
            receipt["failures"].append({"check": name, "artifact": f"{name}.json"})
        persist()
        return result

    model.set_temperature(1.0)
    try:
        benchmark(
            "g0-g4-contract",
            root / "examples/benchmarks/text-contract.jsonl",
            tuple(selected_models),
            repetitions=3,
            track="contract",
        )
        # Compare actual saved runner decisions, independently of raw-head parity.
        paired = behavior_comparison(
            output / "g0-g4-contract", output / "g0-g4-contract", "gemma-g1", "gemma-g2"
        )
        paired["passed"] = (
            paired["choice_flip_rate"] == 0
            and paired["not_both_valid"] == 0
            and paired["max_probability_delta"] is not None
            and paired["max_probability_delta"] <= PROBABILITY_ATOL
        )
        direct("g1-g2-runner-parity", lambda: paired)
        direct("g1-g2-numerical-parity", lambda: _readout_parity(model, contract_cases))

        def batching_parity():
            requests = _parity_requests(contract_cases, media_cases)
            failures, left = [], {}
            sequential = _ObservedAdapter(
                selected_models["gemma-g2"], model, output / "native-parity-forwards.jsonl"
            )
            try:
                for name, request in requests.items():
                    try:
                        left[name] = sequential.predict(request)
                    except Exception as exc:
                        failures.append(
                            {
                                "input_kind": name,
                                "stage": "sequential",
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:1000],
                            }
                        )
            finally:
                sequential.close()
            batched = _ObservedAdapter(
                selected_models["gemma-g4"], model, output / "native-parity-forwards.jsonl"
            )
            try:
                successful = {name: requests[name] for name in left}
                right = batched.predict_batch(list(successful.values())) if successful else []
            finally:
                batched.close()
            rows = [
                {"input_kind": name, "decisions": _distribution_delta(request, left[name], result)}
                for (name, request), result in zip(successful.items(), right)
            ]
            return {
                "probability_atol": PROBABILITY_ATOL,
                "comparisons": rows,
                "execution": right[0]["execution"] if right else None,
                "failures": failures,
                "passed": bool(rows)
                and not failures
                and all(d["passed"] for row in rows for d in row["decisions"]),
            }

        direct("g2-g4-native-parity", batching_parity)

        def independence():
            rows = []
            unrelated = Question(
                id="unrelated-control",
                instructions="Choose the first option.",
                criteria={"first": "The first option", "second": "The second"},
            )
            for mode in ("gemma-g2", "gemma-g4"):
                adapter = _ObservedAdapter(
                    selected_models[mode], model, output / "independence-forwards.jsonl"
                )
                try:
                    for case in (contract_cases[0], contract_cases[4], contract_cases[8]):
                        request = case.request
                        baseline = adapter.predict(request)
                        variants = {
                            "reverse_question_order": request.model_copy(
                                update={"questions": list(reversed(request.questions))}
                            ),
                            "add_unrelated_question_first": request.model_copy(
                                update={"questions": [unrelated, *request.questions]}
                            ),
                        }
                        for variant, changed in variants.items():
                            result = adapter.predict(changed)
                            comparisons = _distribution_delta(request, baseline, result)
                            rows.append(
                                {
                                    "model": mode,
                                    "case_id": case.id,
                                    "variant": variant,
                                    "decisions": comparisons,
                                }
                            )
                        for question in request.questions:
                            alone = request.model_copy(update={"questions": [question]})
                            result = adapter.predict(alone)
                            comparisons = _distribution_delta(alone, baseline, result)
                            rows.append(
                                {
                                    "model": mode,
                                    "case_id": case.id,
                                    "variant": "question_alone",
                                    "decisions": comparisons,
                                }
                            )
                finally:
                    adapter.close()
            return {
                "probability_atol": PROBABILITY_ATOL,
                "comparisons": rows,
                "passed": all(d["passed"] for row in rows for d in row["decisions"]),
            }

        direct("question-independence", independence)
        # Screening samples can be small. A full B=32 still needs 32 requests;
        # preserve source groups when repeating inputs for this systems-only run.
        batch_cases = load_cases(dataset_dir / "text-test.jsonl")
        batch_dataset = dataset_dir / "text-test.jsonl"
        if len(batch_cases) < 32:
            batch_dataset = output / "native-batch-systems.jsonl"
            batch_cases = [
                batch_cases[index % len(batch_cases)].model_copy(
                    deep=True, update={"id": f"batch-system-{index}"}
                )
                for index in range(32)
            ]
            write_cases(batch_dataset, batch_cases)
            receipt["native_batch_dataset"] = {
                "kind": "repeated_public_inputs_systems_only",
                "cases": 32,
                "dataset_sha256": fingerprint(batch_cases),
                "quality_claim": False,
            }
        for batch_size in (1, 2, 4, 8, 16, 32):
            benchmark(
                f"native-batch-{batch_size}",
                batch_dataset,
                ("gemma-g4",),
                batch_size=batch_size,
            )
        sweep = prepare_sweep(output / "synthetic-datasets", cases_per_cell=2)
        for cell in sweep["cells"]:
            benchmark(
                f"sweep-{cell['id']}",
                output / "synthetic-datasets" / cell["dataset"],
                ("gemma-g4",),
            )
        benchmark(
            "public-text-quality",
            dataset_dir / "text-test.jsonl",
            ("gemma-g4",),
            batch_size=4,
            track="generalist",
        )
        benchmark(
            "public-media-quality",
            dataset_dir / "media-test.jsonl",
            ("gemma-g4",),
            batch_size=1,
            track="native_multimodal",
            information_view="native_media",
        )
        receipt["status"] = "passed" if not receipt["failures"] else "completed_with_failures"
    except BaseException:
        receipt["status"] = "interrupted"
        raise
    finally:
        model.set_temperature(original_temperature)
        _release()
        persist()
    return receipt
