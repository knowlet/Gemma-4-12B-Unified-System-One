"""Matched campaign invariants without downloading production model weights."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1.contracts import DecisionRequest, answer_from_probabilities
from s1.errors import BackendTimeoutError
from s1.evaluation.adapters import execution_blockers
from s1.evaluation.comparison import (
    SharedAdapter,
    acceptance,
    campaign_specs,
    run_http_load,
    summarize_quality,
)
from s1.evaluation.contracts import ModelSpec, ProfileSpec
from s1.evaluation.datasets import EvaluationCase, fingerprint, request_fingerprint
from s1.evaluation.registry import Registry

ROOT = Path(__file__).resolve().parents[1]


def cases(count=4):
    return [
        EvaluationCase(
            id=f"boolq-{i}",
            group_id=f"passage-{i}",
            task_id="boolq",
            request=DecisionRequest(
                state=f"passage and question {i}",
                questions={"answer": {"type": "noul", "instructions": "Is the claim true?"}},
            ),
            gold={"answer": "true" if i % 2 else "false"},
        )
        for i in range(count)
    ]


def save_quality(tmp_path, dataset, statuses):
    predictions = []
    for i, (case, status) in enumerate(zip(dataset, statuses)):
        predictions.append(
            {
                "request_id": f"req-{i}",
                "case_id": case.id,
                "task_id": case.task_id,
                "question_id": "answer",
                "type": "noul",
                "labels": ["false", "true"],
                "status": status,
                "eligibility": "unsupported" if status == "not_run" else "eligible",
                "actual_label": "false" if status == "ok" else None,
                "gold": case.gold["answer"],
                "probabilities": {"false": 0.8, "true": 0.2} if status == "ok" else None,
            }
        )
    requests = [{"phase": "warmup", "status": "ok", "latency_ms": 9999.0}] + [
        {
            "phase": "measurement",
            "status": status,
            "latency_ms": float(10 + 20 * i) if status == "ok" else None,
        }
        for i, status in enumerate(statuses)
    ]
    for name, rows in (("predictions", predictions), ("requests", requests)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_quality_summary_keeps_failures_in_eligible_case_denominator(tmp_path):
    dataset = cases()
    save_quality(tmp_path, dataset, ["ok", "ok", "error", "not_run"])
    summary = summarize_quality(tmp_path, dataset)
    assert summary["expected"] == summary["recorded"] == 4
    assert summary["valid"] == 2
    assert summary["correct"] == 1
    assert summary["accuracy"] is None
    assert summary["eligible"] == 3
    assert summary["supported_accuracy"] == pytest.approx(1 / 3)
    assert summary["errors"] == summary["unsupported"] == 1
    assert summary["status"] == "partial"
    assert summary["dataset_sha256"] == fingerprint(dataset)
    assert summary["mean_ms"] == summary["p50_ms"] == 20.0
    assert summary["p99_ms"] == pytest.approx(29.8)
    assert summary["nll"] is not None


def test_image_only_support_keeps_audio_quality_unknown(tmp_path):
    dataset = [
        case.model_copy(update={"task_id": "mnist" if i < 2 else "fsdd"})
        for i, case in enumerate(cases())
    ]
    save_quality(tmp_path, dataset, ["ok", "ok", "not_run", "not_run"])
    summary = summarize_quality(tmp_path, dataset)
    assert summary["expected"] == summary["recorded"] == 4
    assert summary["eligible"] == summary["valid"] == summary["unsupported"] == 2
    assert summary["correct"] == 1
    assert summary["accuracy"] is None
    assert summary["supported_accuracy"] == 0.5
    image, audio = summary["by_task"]["mnist"], summary["by_task"]["fsdd"]
    assert image["accuracy"] == image["supported_accuracy"] == 0.5
    assert image["native_status"] == "completed"
    assert audio["correct"] is audio["accuracy"] is audio["supported_accuracy"] is None
    assert audio["unsupported"] == audio["expected"] == 2
    assert audio["eligible"] == audio["valid"] == 0
    assert audio["native_status"] == "unsupported"


@pytest.mark.parametrize("first_status,expected", [("ok", "partial_support"), ("error", "partial")])
def test_partial_media_completion_requires_every_supported_case(
    tmp_path, monkeypatch, first_status, expected
):
    from s1.evaluation import comparison

    config = ROOT / "configs/benchmarks"
    registry = Registry.load(
        config / "models.toml", config / "suites.toml", config / "profiles.toml"
    )
    base = registry.models["gemma-g4"]
    spec = base.model_copy(
        update={
            "capabilities": base.capabilities.model_copy(update={"modalities": ("text", "image")})
        }
    )
    dataset = cases()
    monkeypatch.setattr(comparison, "load_cases", lambda _: dataset)

    def execute(current, profile, *, output, **kwargs):
        assert current.profiles[profile].warmup == 3
        output.mkdir()
        statuses = [first_status, "ok", "not_run", "not_run"] if profile == "media" else ["ok"] * 4
        save_quality(output, dataset, statuses)

    monkeypatch.setattr(comparison, "execute", execute)
    result = comparison.run_quality(registry, spec, None, tmp_path, tmp_path, ROOT / "uv.lock")
    assert result["media"]["native_status"] == expected
    assert result["media"]["unsupported"] == 2


def test_unsupported_media_has_no_manufactured_zero_accuracy(tmp_path):
    dataset = cases()
    save_quality(tmp_path, dataset, ["not_run"] * 4)
    summary = summarize_quality(tmp_path, dataset)
    assert summary["accuracy"] is summary["correct"] is None
    assert summary["status"] == "not_run"
    assert summary["unsupported"] == 4
    assert summary["nll"] is None
    assert summary["mean_ms"] is summary["p99_ms"] is None


def test_attempted_errors_have_zero_operational_accuracy_and_no_latency(tmp_path):
    dataset = cases()
    save_quality(tmp_path, dataset, ["error", "timeout", "error", "timeout"])
    summary = summarize_quality(tmp_path, dataset)
    assert summary["valid"] == summary["correct"] == 0
    assert summary["accuracy"] == 0.0
    assert summary["errors"] == 4
    assert summary["status"] == "partial"
    assert summary["mean_ms"] is summary["p99_ms"] is None


def test_every_predeclared_model_has_executable_precision_policy():
    config = ROOT / "configs/benchmarks"
    registry = Registry.load(
        config / "models.toml", config / "suites.toml", config / "profiles.toml"
    )
    specs = campaign_specs(registry, "/saved-adapters")
    assert len(specs) == 15
    assert len({spec.id for spec in specs}) == 15
    for spec in specs:
        ModelSpec.model_validate(spec.model_dump())
        profile = ProfileSpec(id="quality", suite="boolq", models=(spec.id,))
        assert execution_blockers(spec, profile) == [], spec.id
    trained = [spec for spec in specs if spec.decision_adapter_path]
    assert len(trained) == 9
    for seed in range(3):
        rows = [spec for spec in trained if f"-s{seed}-" in spec.id]
        assert {spec.quantization for spec in rows} == {"none", "int8", "nf4"}
        assert len({spec.decision_adapter_sha256 for spec in rows}) == 1
        assert len({spec.decision_adapter_path for spec in rows}) == 1


@pytest.mark.parametrize(
    "mode,accuracy,peak,valid,expected",
    [
        ("int8", 116 / 128, 12_000_000_000, 128, "passed"),
        ("int8", 116 / 128, 12_000_000_001, 128, "failed"),
        ("nf4", 116 / 128, 8_000_000_000, 128, "passed"),
        ("nf4", 116 / 128, 8_000_000_001, 128, "failed"),
        ("nf4", 114 / 128, 7_000_000_000, 128, "failed"),
        ("nf4", 116 / 128, None, 128, "not_measured"),
        ("int8", 116 / 128, 10_000_000_000, 127, "not_measured"),
    ],
)
def test_acceptance_requires_measured_memory_and_complete_accuracy(
    mode, accuracy, peak, valid, expected
):
    result = acceptance(
        {
            "quantization": mode,
            "boolq": {"valid": valid, "accuracy": accuracy},
            "memory": {"peak_allocated_bytes": peak, "model_footprint_bytes": 1},
        }
    )
    assert result["status"] == expected
    if accuracy == 114 / 128:
        assert result["accuracy_above_89pct"] is True
        assert result["accuracy_at_least_90pct"] is False


def test_unquantized_or_unmeasured_is_never_a_quantization_pass():
    assert acceptance({"quantization": "none"})["status"] == "not_applicable"
    assert acceptance({"quantization": "nf4"})["status"] == "not_measured"


@pytest.mark.parametrize("fail_one", [False, True])
def test_real_http_load_replays_same_cases_and_retains_failures(tmp_path, fail_one):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    module_spec = importlib.util.spec_from_file_location(
        "comparison_live_services", ROOT / "scripts/live_services_checks.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    dataset = cases(3)

    class CaseBackend:
        spec = SimpleNamespace(id="fake-case-model")

        def __init__(self):
            self.states = []

        def predict(self, request):
            self.states.append(request.state)
            index = int(request.state.rsplit(" ", 1)[1])
            if fail_one and index == 1:
                raise BackendTimeoutError("test timeout")
            return {
                "answers": {
                    question.id: answer_from_probabilities(
                        question, [0.1, 0.9] if index % 2 else [0.9, 0.1]
                    )
                    for question in request.questions
                }
            }

    backend = CaseBackend()
    shared = SharedAdapter(backend)
    live_servers = []

    def serve(adapter):
        result = module._serve(adapter)
        live_servers.append(result)
        return result

    summary = run_http_load(shared, dataset, tmp_path, serve, count=3, arrival_rate=10000.0)
    assert len(summary) == 6
    assert Counter(backend.states) == {case.request.state: 6 for case in dataset}
    for cell in summary:
        assert cell["requests"] == 3
        assert cell["successful"] == cell["completed_backend_calls"] == (2 if fail_one else 3)
        assert cell["slo_misses"] >= int(fail_one)
        rows = [
            json.loads(line) for line in (tmp_path / f"{cell['id']}.jsonl").read_text().splitlines()
        ]
        assert {r["case_id"] for r in rows} == {case.id for case in dataset}
        assert {r["request_sha256"] for r in rows} == {
            request_fingerprint(case.request) for case in dataset
        }
        assert sum(row["status"] != "ok" for row in rows) == int(fail_one)
        assert cell["p99_ms"] is not None
    assert not live_servers[0][2].is_alive()
