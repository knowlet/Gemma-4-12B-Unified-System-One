"""Measurement contract tests; fake inference and clocks never use MPS."""

import gc
import json
import runpy
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1 import profiling
from s1.contracts import answer_from_probabilities


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.events = []

    def __call__(self):
        self.events.append("clock")
        return self.now

    def synchronize(self):
        self.events.append("synchronize")


class FakeModel:
    name = "fixture-only"
    revision = "a" * 40
    device = "cpu"
    dtype = "float32"
    attn_implementation = "eager"
    temperature = 1.0

    def __init__(self, clock):
        self.clock = clock
        self.calls = 0

    def predict(self, request):
        self.calls += 1
        self.clock.events.append("predict")
        # Make the warmup observably slower than measured requests.
        self.clock.now += 1.0 if self.calls == 1 else 0.02
        return {
            "answers": {
                q.id: answer_from_probabilities(q, [1 / len(q.labels())] * len(q.labels()))
                for q in request.questions
            }
        }


def fake_phases(model, request, *, synchronize, clock, set_stage):
    set_stage("phase_projection_softmax_cpu")
    return {
        "prepare_ms": 3.0,
        "backbone_forward_ms": 9.0,
        "projection_softmax_cpu_ms": 1.0,
        "input_tokens": 42,
        "answers": {
            q.id: answer_from_probabilities(q, [1 / len(q.labels())] * len(q.labels()))
            for q in request.questions
        },
    }


def test_full_request_boundaries_and_phases_are_separate(benchmark_case):
    clock = FakeClock()
    model = FakeModel(clock)
    report = profiling.profile_model(
        model,
        [benchmark_case],
        warmup=1,
        repeat=3,
        synchronize=clock.synchronize,
        clock=clock,
        phase_runner=fake_phases,
    )
    assert report["status"] == "ok"
    assert clock.events == ["synchronize", "clock", "predict", "synchronize", "clock"] * 4
    case = report["cases"][0]
    assert case["warmup_samples"][0]["latency_ms"] == pytest.approx(1000)
    assert [s["latency_ms"] for s in case["end_to_end_samples"]] == pytest.approx([20] * 3)
    assert len(case["phase_samples"]) == 3
    assert case["phase_latency_ms"]["backbone_forward_ms"]["p50"] == 9
    summary = report["summary"]
    assert summary["request_count"] == 3
    assert summary["decision_count"] == 9
    assert summary["latency_ms"] == pytest.approx({"p50": 20, "p95": 20})
    assert summary["decisions_per_second"] == pytest.approx(150)
    assert summary["total_measured_input_tokens"] == 126
    assert summary["request_coverage"] == summary["phase_coverage"] == 1.0
    assert report["memory_snapshots"]["after_profile"]["kind"] == "snapshot_not_peak"
    json.dumps(report, allow_nan=False)


def test_nonfinite_warmup_fails_without_successful_measurements(benchmark_case):
    clock = FakeClock()
    model = FakeModel(clock)
    original = model.predict

    def nonfinite(request):
        response = original(request)
        response["answers"]["route"]["probabilities"]["billing"] = float("nan")
        return response

    model.predict = nonfinite
    report = profiling.profile_model(
        model,
        [benchmark_case],
        synchronize=clock.synchronize,
        clock=clock,
        phase_runner=fake_phases,
    )
    assert report["status"] == "error"
    assert report["failure"]["stage"] == "warmup"
    assert report["failure"]["case_id"] == benchmark_case.id
    assert report["summary"]["request_count"] == 0
    assert report["summary"]["latency_ms"] is None
    json.dumps(report, allow_nan=False)


def test_phase_failure_keeps_successful_full_requests(benchmark_case):
    clock = FakeClock()

    def failed_phase(model, request, *, synchronize, clock, set_stage):
        set_stage("phase_backbone_forward")
        raise RuntimeError("fixture forward failure")

    report = profiling.profile_model(
        FakeModel(clock),
        [benchmark_case, benchmark_case.model_copy(update={"id": "unattempted"})],
        warmup=0,
        repeat=2,
        synchronize=clock.synchronize,
        clock=clock,
        phase_runner=failed_phase,
    )
    assert report["status"] == "error"
    assert report["failure"]["stage"] == "phase_backbone_forward"
    assert report["summary"]["request_count"] == 2
    assert report["summary"]["total_measured_input_tokens"] is None
    assert report["case_ids"] == [benchmark_case.id, "unattempted"]
    assert report["unattempted_case_ids"] == ["unattempted"]
    assert report["planned_request_count"] == 4
    assert report["planned_decision_count"] == 12
    assert report["summary"]["request_coverage"] == 0.5
    assert report["summary"]["decision_coverage"] == 0.5
    assert report["summary"]["phase_coverage"] == 0


@pytest.mark.inference
def test_phase_boundaries_and_prepared_tensor_lifetimes(decision_request):
    torch = pytest.importorskip("torch")
    clock = FakeClock()
    tensor_refs = []
    head = torch.nn.Linear(2, 26, bias=False)
    head.register_forward_hook(lambda *args: pytest.fail("full vocabulary head called"))

    def prepare(request):
        clock.events.append("prepare")
        clock.now += 0.003
        inputs = torch.tensor([[1, 2, 3]])
        tensor_refs.append(weakref.ref(inputs))
        return {"input_ids": inputs}, torch.tensor([0, 1, 2]), torch.tensor([2, 2, 2])

    def backbone(**kwargs):
        assert kwargs["use_cache"] is False
        assert kwargs["return_dict"] is True
        clock.events.append("forward")
        clock.now += 0.009
        output = torch.ones((1, 3, 2))
        tensor_refs.append(weakref.ref(output))
        return SimpleNamespace(last_hidden_state=output)

    model = SimpleNamespace(
        lm=SimpleNamespace(eval=lambda: None),
        prepare=prepare,
        backbone=backbone,
        head=head,
        letters=torch.arange(26),
        softcap=None,
        temperature=1.0,
    )
    sample = profiling.profile_phases(
        model,
        decision_request,
        synchronize=clock.synchronize,
        clock=clock,
    )
    assert sample["input_tokens"] == 3
    assert sample["prepare_ms"] == pytest.approx(3)
    assert sample["backbone_forward_ms"] == pytest.approx(9)
    assert clock.events == [
        "synchronize",
        "clock",
        "prepare",
        "synchronize",
        "clock",
        "synchronize",
        "clock",
        "forward",
        "synchronize",
        "clock",
        "synchronize",
        "clock",
        "synchronize",
        "clock",
    ]
    gc.collect()
    assert all(ref() is None for ref in tensor_refs)
    json.dumps(sample, allow_nan=False)


def script_main():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_mps.py"
    return runpy.run_path(str(path))["main"]


def test_script_writes_load_error_report(tmp_path, monkeypatch, benchmark_case):
    from s1 import unified

    def failed_load(*args, **kwargs):
        raise RuntimeError("fixture missing weights")

    monkeypatch.setattr(unified, "UnifiedDecisionModel", failed_load)
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(benchmark_case.model_dump_json() + "\n")
    output = tmp_path / "report.json"
    result = script_main()(
        [
            "--model",
            str(tmp_path),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--device",
            "cpu",
        ]
    )
    report = json.loads(output.read_text())
    assert result == 1
    assert report["failure"]["stage"] == "load_model"
    assert report["load_time_ms"] >= 0
    assert report["cases"] == []


@pytest.mark.parametrize(
    "arguments", [["--temperature", "nan"], ["--temperature", "inf"], ["--max-context", "0"]]
)
def test_script_configuration_error_is_serializable_before_load(tmp_path, monkeypatch, arguments):
    from s1 import unified

    def unexpected_load(*args, **kwargs):
        pytest.fail("invalid configuration must be rejected before loading weights")

    monkeypatch.setattr(unified, "UnifiedDecisionModel", unexpected_load)
    output = tmp_path / "report.json"
    result = script_main()(["--output", str(output), *arguments])
    report = json.loads(output.read_text())
    assert result == 1
    assert report["failure"]["stage"] == "configuration"
    assert report["load_time_ms"] is None


def test_script_pins_revision_before_loading(tmp_path, monkeypatch, benchmark_case):
    from s1 import unified

    calls = []
    sha = "b" * 40

    def info(name, *, revision):
        calls.append(("resolve", name, revision))
        return SimpleNamespace(sha=sha)

    def load(name, **kwargs):
        calls.append(("load", name, kwargs["revision"]))
        return SimpleNamespace(device="cpu")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=info)),
    )
    monkeypatch.setattr(unified, "UnifiedDecisionModel", load)
    monkeypatch.setattr(profiling, "synchronize_device", lambda device: None)
    monkeypatch.setattr(
        profiling,
        "profile_model",
        lambda *args, **kwargs: {"status": "ok", "memory_snapshots": {}, "cases": []},
    )
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(benchmark_case.model_dump_json() + "\n")
    output = tmp_path / "report.json"
    assert (
        script_main()(
            [
                "--model",
                "fixture/model",
                "--revision",
                "main",
                "--dataset",
                str(dataset),
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert calls == [("resolve", "fixture/model", "main"), ("load", "fixture/model", sha)]
    assert json.loads(output.read_text())["pinned_revision"] == sha
