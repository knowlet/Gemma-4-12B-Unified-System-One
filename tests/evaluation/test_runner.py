import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from s1.backends import UniformBackend, UnsupportedRequest
from s1.benchmark import fingerprint as v1_fingerprint
from s1.contracts import DecisionRequest
from s1.errors import BackendTimeoutError
from s1.evaluation.adapters import ReferenceAdapter, execution_blockers, load_adapter
from s1.evaluation.artifacts import read_records
from s1.evaluation.contracts import ModelSpec, ProfileSpec
from s1.evaluation.datasets import EvaluationCase, fingerprint, load_cases
from s1.evaluation.registry import Registry
from s1.evaluation.reporting import summarize_run
from s1.evaluation.runner import execute

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "benchmarks"


@pytest.fixture
def registry():
    return Registry.load(CONFIG / "models.toml", CONFIG / "suites.toml", CONFIG / "profiles.toml")


def run(registry, tmp_path, **kwargs):
    return execute(
        registry,
        "smoke",
        output=tmp_path / "run",
        lockfile=ROOT / "uv.lock",
        enable_models=["uniform"],
        **kwargs,
    )


def no_warmup(registry, **updates):
    data = registry.profiles["smoke"].model_dump()
    data.update(warmup=0, repetitions=1, **updates)
    return replace(registry, profiles={"smoke": ProfileSpec.model_validate(data)})


def factory(backend):
    return lambda spec, environment: ReferenceAdapter(spec, backend)


def test_uniform_records_and_summary_can_be_recomputed(registry, tmp_path):
    result = run(registry, tmp_path)
    assert result["run_status"] == "completed"
    stats = result["models"]["uniform"]
    assert stats["request_counts"] == {"ok": 9}
    assert stats["decision_counts"] == {"ok": 15}
    assert stats["warmup_counts"] == {"ok": 3}
    assert stats["execution_coverage"] == 1
    assert stats["quality"]["actual_label_accuracy"] == pytest.approx(0.4)
    assert stats["quality"]["actual_choice_accuracy"] == pytest.approx(0.5)
    assert stats["successful_request_latency_ms"]["requests"] == 9
    assert summarize_run(tmp_path / "run") == result
    assert not read_records(tmp_path / "run", "errors")
    rows = read_records(tmp_path / "run", "requests")
    assert len(rows) == 12
    assert len({row["request_id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["scheduled_at_s"] == row["dispatched_at_s"]
        assert (
            row["dispatched_at_s"]
            <= row["all_required_answers_ready_at_s"]
            == row["terminated_at_s"]
        )
        assert row["latency_ms"] >= 0


@pytest.mark.parametrize(
    "failure,status",
    [
        (ValueError("private request contents"), "error"),
        (UnsupportedRequest("provider changed its mind"), "error"),
        (BackendTimeoutError("secret endpoint token"), "timeout"),
    ],
)
def test_runtime_failures_remain_in_eligible_denominator(registry, tmp_path, failure, status):
    class Broken:
        def predict(self, request):
            raise failure

    result = run(no_warmup(registry), tmp_path, adapter_factory=factory(Broken()))
    stats = result["models"]["uniform"]
    assert result["run_status"] == "completed_with_errors"
    assert stats["decision_counts"] == {status: 5}
    assert stats["denominators"]["eligible_decisions"] == 5
    assert stats["quality"]["actual_label_accuracy"] is None
    assert stats["quality"]["operational_correctness"] == 0
    assert stats["execution_coverage"] == 0
    assert stats["successful_request_latency_ms"] is None
    for row in read_records(tmp_path / "run", "requests"):
        assert row["all_required_answers_ready_at_s"] is None
        assert row["terminated_at_s"] >= row["dispatched_at_s"]
        assert row["latency_ms"] is None
    text = "".join(path.read_text() for path in (tmp_path / "run").iterdir())
    assert str(failure) not in text


def test_actual_choice_ties_are_separate_from_standard_argmax(registry, benchmark_case, tmp_path):
    path = tmp_path / "case.jsonl"
    path.write_text(benchmark_case.model_dump_json())
    suite = registry.suites["smoke"].model_copy(update={"dataset": str(path)})
    registry = replace(no_warmup(registry), suites={"smoke": suite})

    class Tied(UniformBackend):
        def predict(self, request):
            result = super().predict(request)
            result["answers"]["route"]["choice"] = "technical"
            return result

    result = run(registry, tmp_path, adapter_factory=factory(Tied()))
    quality = result["models"]["uniform"]["quality"]
    assert quality["actual_choice_accuracy"] == 0
    assert quality["actual_label_accuracy"] == pytest.approx(1 / 3)
    assert quality["standardized_argmax_accuracy"] == pytest.approx(2 / 3)


def test_partial_response_marks_whole_request_failed(registry, tmp_path):
    class Partial(UniformBackend):
        def predict(self, request):
            result = super().predict(request)
            if len(request.questions) == 2:
                result["answers"].pop(request.questions[0].id)
            return result

    result = run(no_warmup(registry), tmp_path, adapter_factory=factory(Partial()))
    stats = result["models"]["uniform"]
    assert stats["decision_counts"] == {"error": 4, "ok": 1}
    assert stats["execution_coverage"] == pytest.approx(1 / 5)
    assert stats["successful_request_latency_ms"]["requests"] == 1


def test_gold_and_metadata_are_not_adapter_inputs_and_mutations_are_isolated(registry, tmp_path):
    seen = []

    class Mutating(UniformBackend):
        def predict(self, request):
            assert isinstance(request, DecisionRequest)
            assert set(request.model_dump()) == {"state", "questions", "media"}
            assert request.state != "mutated"
            seen.append(request.state)
            request.state = "mutated"
            return super().predict(request)

    run(registry, tmp_path, adapter_factory=factory(Mutating()))
    assert len(seen) == 12
    for path in (tmp_path / "run").iterdir():
        contents = path.read_text()
        assert "charged twice" not in contents
        assert "mutated" not in contents


def test_setup_failure_is_not_a_zero_quality_score_and_closes_adapter(registry, tmp_path):
    closed = []

    class BadTelemetry:
        def telemetry(self):
            raise ValueError("secret credentials")

        def close(self):
            closed.append(True)

    result = run(registry, tmp_path, adapter_factory=lambda *_: BadTelemetry())
    stats = result["models"]["uniform"]
    assert result["run_status"] == "completed_with_errors"
    assert stats["status"] == "setup_error"
    assert stats["decision_counts"] == {"not_run": 15}
    assert all(value is None for value in stats["quality"].values())
    assert stats["execution_coverage"] is None
    assert closed == [True]
    assert read_records(tmp_path / "run", "errors")[0]["phase"] == "setup"
    assert "secret credentials" not in json.dumps(result)


def test_factory_failure_still_writes_all_records(registry, tmp_path):
    def missing(*_):
        raise ModuleNotFoundError("optional runtime missing")

    result = run(registry, tmp_path, adapter_factory=missing)
    assert result["models"]["uniform"]["decision_counts"] == {"not_run": 15}


def test_warmup_errors_are_reported_and_excluded_from_quality(registry, tmp_path):
    class FirstFails(UniformBackend):
        calls = 0

        def predict(self, request):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("warmup error")
            return super().predict(request)

    result = run(registry, tmp_path, adapter_factory=factory(FirstFails()))
    stats = result["models"]["uniform"]
    assert result["run_status"] == "completed_with_errors"
    assert stats["warmup_counts"] == {"error": 1, "ok": 2}
    assert stats["request_counts"] == {"ok": 9}
    assert stats["decision_counts"] == {"ok": 15}


def test_close_error_preserves_results(registry, tmp_path):
    class BadClose(UniformBackend):
        def close(self):
            raise RuntimeError("private cleanup details")

    result = run(registry, tmp_path, adapter_factory=factory(BadClose()))
    assert result["run_status"] == "completed_with_errors"
    assert result["models"]["uniform"]["decision_counts"] == {"ok": 15}
    assert read_records(tmp_path / "run", "errors")[-1]["phase"] == "close"


def test_budget_prevents_loading(registry, tmp_path):
    data = registry.profiles["smoke"].model_dump()
    data["budget"]["max_requests"] = 0
    registry = replace(registry, profiles={"smoke": ProfileSpec.model_validate(data)})
    called = []
    result = run(registry, tmp_path, adapter_factory=lambda *_: called.append(True))
    assert called == []
    assert result["run_status"] == "not_run"
    assert "request_budget_exceeded" in result["models"]["uniform"]["reasons"]


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"batch_size": 2}, "batch_unavailable"),
    ],
)
def test_executor_does_not_emulate_unsupported_parallel_modes(registry, tmp_path, changes, reason):
    data = registry.profiles["smoke"].model_dump()
    data.update(changes)
    registry = replace(registry, profiles={"smoke": ProfileSpec.model_validate(data)})
    result = run(registry, tmp_path, adapter_factory=lambda *_: pytest.fail("must not load"))
    assert result["run_status"] == "not_run"
    assert reason in result["models"]["uniform"]["reasons"]


def test_existing_run_directory_is_never_overwritten(registry, tmp_path):
    (tmp_path / "run").mkdir()
    sentinel = tmp_path / "run" / "keep.txt"
    sentinel.write_text("existing results")
    with pytest.raises(FileExistsError):
        run(registry, tmp_path, adapter_factory=lambda *_: pytest.fail("must not load"))
    assert sentinel.read_text() == "existing results"


def test_interruption_keeps_manifest_and_completed_records(registry, tmp_path):
    closed = []

    class Interrupt(UniformBackend):
        calls = 0

        def predict(self, request):
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt()
            return super().predict(request)

        def close(self):
            closed.append(True)

    with pytest.raises(KeyboardInterrupt):
        run(no_warmup(registry), tmp_path, adapter_factory=factory(Interrupt()))
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert closed == [True]
    requests = read_records(tmp_path / "run", "requests")
    assert [row["status"] for row in requests] == ["ok", "interrupted"]
    assert requests[-1]["terminated_at_s"] is not None
    partial = summarize_run(tmp_path / "run")["models"]["uniform"]
    assert partial["records_complete"] is False
    assert partial["denominators"]["total_decisions"] == 5
    assert partial["denominators"]["eligible_decisions"] == 5
    assert partial["denominators"]["recorded_decisions"] == 4


def test_missing_http_credentials_leave_not_run_records(registry, tmp_path):
    spec = registry.models["uniform"].model_copy(
        update={
            "adapter": "http",
            "revision": "service-v1",
            "precision": "provider",
            "execution_mode": "provider",
            "calibration": "provider",
            "endpoint_env": "TEST_ENDPOINT",
            "token_env": "TEST_TOKEN",
        }
    )
    registry = replace(registry, models={"uniform": spec})
    result = run(
        registry,
        tmp_path,
        environment={"TEST_ENDPOINT": "https://example.invalid/decide"},
        adapter_factory=lambda *_: pytest.fail("must not load"),
    )
    assert result["run_status"] == "not_run"
    assert "missing_credentials" in result["models"]["uniform"]["reasons"]
    assert result["models"]["uniform"]["decision_counts"] == {"not_run": 15}


def test_runtime_precision_mismatch_stops_before_prediction(registry, tmp_path):
    class WrongPrecision:
        def telemetry(self):
            return {"precision": "float32", "context_policy": "not_applicable"}

        def predict(self, request):
            pytest.fail("must not predict after telemetry mismatch")

        def close(self):
            pass

    result = run(registry, tmp_path, adapter_factory=lambda *_: WrongPrecision())
    assert result["models"]["uniform"]["status"] == "setup_error"


def test_text_suite_tracks_source_groups_without_counting_translations_as_independent(
    registry, tmp_path
):
    result = execute(
        registry,
        "text-smoke",
        output=tmp_path / "run",
        lockfile=ROOT / "uv.lock",
        enable_models=["uniform"],
    )
    stats = result["models"]["uniform"]
    assert stats["request_counts"] == {"ok": 10}
    assert stats["decision_counts"] == {"ok": 18}
    assert stats["warmup_counts"] == {"ok": 1}
    assert stats["denominators"]["source_groups"] == 5
    assert stats["records_complete"] is True
    rows = read_records(tmp_path / "run", "predictions")
    assert {row["language"] for row in rows} == {"en", "zh-TW"}
    assert {row["task_id"] for row in rows} == {"ticket-routing", "ci-status", "ordinal-sentiment"}


def test_executor_incompatible_enabled_model_marks_run_partial(registry, tmp_path):
    models = dict(registry.models)
    models["another"] = models["uniform"].model_copy(
        update={"id": "another", "enabled": True, "precision": "float32"}
    )
    data = registry.profiles["smoke"].model_dump()
    data.update(models=("uniform", "another"), budget={"max_requests": 24})
    registry = replace(
        registry, models=models, profiles={"smoke": ProfileSpec.model_validate(data)}
    )
    result = run(registry, tmp_path)
    assert result["run_status"] == "partial"
    assert result["models"]["uniform"]["status"] == "completed"
    assert result["models"]["another"]["status"] == "not_run"


def test_v1_fingerprints_preserved_and_v2_metadata_recorded(benchmark_case, tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(benchmark_case.model_dump_json())
    cases = load_cases(path)
    assert fingerprint(cases) == v1_fingerprint([benchmark_case])
    data = cases[0].model_dump()
    data.update(group_id="document-1", task_id="routing", language="zh-TW", schema_id="tickets")
    annotated = EvaluationCase.model_validate(data)
    assert fingerprint([annotated]) != fingerprint(cases)
    assert annotated.metadata()["group_id"] == "document-1"
    assert "gold" not in annotated.request.model_dump()


def test_invalid_dataset_errors_do_not_include_prompt_text(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"private-case","request":{"state":"secret message"}}')
    with pytest.raises(ValueError, match="line 1") as exc:
        load_cases(path)
    assert "secret" not in str(exc.value)


@pytest.mark.parametrize("precision", ["bfloat16", "float32"])
def test_adapter_factory_forwards_settings_without_loading_real_models(
    registry, monkeypatch, precision
):
    import s1.backends

    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return UniformBackend()

    monkeypatch.setattr(s1.backends, "GemmaBackend", fake)
    monkeypatch.setattr(s1.backends, "LayaBackend", fake)
    gemma = ModelSpec(
        id="g",
        adapter="gemma",
        model_id="local/model",
        revision="a" * 40,
        device="cuda:1",
        precision=precision,
        calibration="none",
        context_limit=2048,
    )
    load_adapter(gemma, {}).close()
    assert calls[-1] == (
        ("local/model",),
        {
            "revision": "a" * 40,
            "device": "cuda:1",
            "precision": precision,
            "quantization": "none",
            "max_context": 2048,
            "temperature": 1.0,
        },
    )
    laya = ModelSpec(
        id="l",
        adapter="laya",
        model_id="local/laya",
        revision="b" * 40,
        subfolder="multilingual",
        context_limit=512,
    )
    load_adapter(laya, {}).close()
    assert calls[-1][1]["subfolder"] == "multilingual"
    assert calls[-1][1]["max_len"] == 512


def test_gemma_precision_and_processor_policy_cannot_be_ignored(registry):
    spec = registry.models["gemma-g3"].model_copy(
        update={"device": "cpu", "processor_revision": "different"}
    )
    reasons = execution_blockers(spec, registry.profiles["smoke"])
    assert "gemma_device_precision_mismatch" in reasons
    assert "separate_processor_revision_unavailable" in reasons


@pytest.mark.parametrize(
    "device,precision,supported",
    [
        ("cuda:1", "float32", True),
        ("cuda:1", "bfloat16", True),
        ("cuda:1", "float16", False),
        ("cpu", "float32", True),
        ("cpu", "bfloat16", False),
    ],
)
def test_gemma_explicit_precision_is_checked_against_runtime_support(
    registry, device, precision, supported
):
    spec = registry.models["gemma-g3"].model_copy(update={"device": device, "precision": precision})
    reasons = execution_blockers(spec, registry.profiles["smoke"])
    assert ("gemma_device_precision_mismatch" not in reasons) == supported


def test_cli_run_and_recompute(tmp_path):
    output = tmp_path / "cli-run"
    command = [
        sys.executable,
        "-m",
        "s1.cli",
        "eval",
        "run",
        "--profile",
        "smoke",
        "--enable-model",
        "uniform",
        "--output",
        str(output),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    expected = json.loads((output / "summary.json").read_text())
    recomputed = subprocess.run(
        [sys.executable, "-m", "s1.cli", "eval", "summarize", str(output)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert recomputed.returncode == 0, recomputed.stderr
    assert json.loads(recomputed.stdout) == expected
    repeated = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert repeated.returncode == 2
    assert json.loads((output / "summary.json").read_text()) == expected
