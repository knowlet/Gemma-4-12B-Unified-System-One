"""Pure score boundary tests; never import Torch or construct a model."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).absolute().parent
SPEC = importlib.util.spec_from_file_location(
    "_jevbench_score_worker_test", HERE / "score_worker.py"
)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def test_dry_run_has_no_execution_side_effects(monkeypatch, capsys):
    monkeypatch.setattr(worker, "execute", lambda _boot: pytest.fail("Execution in dry mode"))
    worker.main([])
    assert capsys.readouterr().out.strip() == "DRY_NO_MODEL_OR_OUTPUT"


def test_admission_converts_resource_exception_to_unscorable_abort():
    class Gate:
        def check(self, _phase, _scope):
            raise ValueError("host memory refused")

    try:
        worker.Admission(Gate()).check("forward")
    except Exception:
        pytest.fail("Resource abort was catchable as an ordinary model failure")
    except worker.ScoreAbort as exc:
        assert isinstance(exc.__cause__, ValueError)


def test_wrong_published_weight_digest_refused_before_any_model_import():
    import sys

    from jev_public_action_artifact import BASE_CHECKPOINT

    assert "torch" not in sys.modules
    observed = {**BASE_CHECKPOINT, "sha256": "0" * 64}
    with pytest.raises(worker.ScoreAbort, match="published base checkpoint"):
        worker.authenticate_checkpoint(
            "cache",
            lambda cache: Path(cache) / "fake-weight",
            lambda *_a, **_kw: observed,
            BASE_CHECKPOINT,
        )
    assert "torch" not in sys.modules


def test_actual_backbone_hook_and_cpu_drain_use_distinct_scopes(tmp_path):
    seen = []
    gate = worker.Admission(SimpleNamespace(check=lambda *args: seen.append(args)))
    predictor = worker.Predictor(None, None, gate, tmp_path, None)
    predictor.stage = "jevbench-mini"
    predictor.hook(None, None)
    gate.check("post-model-work-drain", "cpu")
    assert seen == [
        ("actual-backbone-forward:jevbench-mini", "load"),
        ("post-model-work-drain", "cpu"),
    ]


def test_predictor_preserves_complete_request_and_native_answer(tmp_path):
    seen, order = [], []

    def predict(request):
        seen.append(request)
        return {"answers": {"d": {"type": "noul", "noul": 0.7}}}

    predictor = worker.Predictor(
        SimpleNamespace(predict=predict),
        SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: order.append("sync"))),
        SimpleNamespace(check=lambda *args: order.append(args)),
        tmp_path,
        lambda **kwargs: kwargs,
        lambda: order.append("guard"),
    )
    predictor.stage = "jevbench-mini"
    state, questions = {"nested": [1, 2]}, {"d": {"type": "noul", "instructions": "Full"}}
    assert predictor.answers(state, questions) == {"d": {"type": "noul", "noul": 0.7}}
    assert seen == [{"state": state, "questions": questions}]
    events = [
        json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()
    ]
    assert events[0]["request"] == seen[0]
    assert events[1]["response"]["answers"]["d"]["noul"] == 0.7
    assert order == [("before-predict:jevbench-mini",), "guard", "sync", "sync"]
    assert json.loads((tmp_path / "progress.json").read_text())["completed_predictions"] == 1


def test_prediction_primary_survives_secondary_evidence_failure(tmp_path, monkeypatch):
    primary = worker.ScoreAbort("forward resource refusal")

    def fail(_request):
        raise primary

    predictor = worker.Predictor(
        SimpleNamespace(predict=fail),
        SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)),
        SimpleNamespace(check=lambda *_args: None),
        tmp_path,
        lambda **kwargs: kwargs,
    )
    real_append = worker.append

    def append(path, value):
        if value["event"] == "failed":
            raise OSError("secondary disk failure")
        real_append(path, value)

    monkeypatch.setattr(worker, "append", append)
    with pytest.raises(worker.ScoreAbort) as got:
        predictor({}, {})
    assert got.value is primary
    assert "secondary disk failure" in primary.__notes__[0]


def test_cuda_oom_records_failure_then_escapes_as_environment_abort(tmp_path):
    class FakeCUDAOutOfMemory(RuntimeError):
        pass

    oom = FakeCUDAOutOfMemory("fake capacity limit")

    def predict(_request):
        raise oom

    predictor = worker.Predictor(
        SimpleNamespace(predict=predict),
        SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda: None, OutOfMemoryError=FakeCUDAOutOfMemory)
        ),
        SimpleNamespace(check=lambda *_args: None),
        tmp_path,
        lambda **kwargs: kwargs,
    )
    try:
        predictor({}, {})
    except Exception:
        pytest.fail("CUDA capacity failure can be mistaken for a model answer failure")
    except worker.ScoreAbort as exc:
        assert exc.__cause__ is oom
    events = [
        json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "failed"
    assert events[-1]["error"]["type"] == "FakeCUDAOutOfMemory"


def test_official_public_runner_does_not_convert_environment_abort(tmp_path):
    from run_jevbench_public import run_public

    calls = []

    def refused(_state, _questions):
        calls.append(1)
        raise worker.ScoreAbort("before actual model forward")

    with pytest.raises(worker.ScoreAbort):
        run_public(
            refused,
            worker.HEAVEN,
            tmp_path / "public231",
            {"model_id": "not-a-model-pure-test", "revision": "pure"},
        )
    assert calls == [1]
    assert json.loads((tmp_path / "public231/manifest.json").read_text())["status"] == "running"
    assert not (tmp_path / "public231/summary.json").exists()


def test_score_source_population_can_be_captured_and_detects_drift(tmp_path, monkeypatch):
    refs = worker.capture_sources(tmp_path)
    assert worker.verify_sources(refs)
    assert any(label.endswith("score_worker.py") for label in refs)
    assert any(label.endswith("jevbench-mini/tests.jsonl.gz") for label in refs)
    assert any(label.endswith("run_jevbench_public.py") for label in refs)
    one = tmp_path / "substitute.py"
    one.write_text("changed")
    original = worker.source_paths()
    monkeypatch.setattr(worker, "source_paths", lambda: {**original, "extra.py": one})
    assert not worker.verify_sources(refs)
