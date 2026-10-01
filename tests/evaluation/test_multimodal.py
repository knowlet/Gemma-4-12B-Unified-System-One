import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from s1.backends import HTTPBackend, UniformBackend
from s1.contracts import AudioInput
from s1.errors import BackendResponseError
from s1.evaluation.datasets import load_cases
from s1.evaluation.multimodal import PipelineAdapter, counterfactual_summary, prepare_media_views
from s1.evaluation.registry import Registry
from s1.evaluation.runner import execute


def test_views_require_separate_missing_media_gold_and_fixed_text(benchmark_case, tmp_path):
    case = benchmark_case.model_dump(mode="json")
    case.update(group_id="recording-pair")
    case["request"]["media"] = [{"type": "audio", "samples": [0.1]}]
    row = {
        "case": case,
        "missing_media_gold": case["gold"],
        "verified_text": "Verified transcription",
    }
    other = json.loads(json.dumps(row))
    other["case"]["id"] = "other"
    other["case"]["request"]["media"][0]["samples"] = [-0.2]
    other["case"]["gold"]["route"] = "technical"
    source = tmp_path / "bundle.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in [row, other]))
    report = prepare_media_views(source, tmp_path / "views")
    assert len(report["counterfactual_pairs"]) == 1
    assert not load_cases(tmp_path / "views/M0.jsonl")[0].request.media
    assert load_cases(tmp_path / "views/M1.jsonl")[0].request.media
    assert "verified_media_description" in load_cases(tmp_path / "views/M2.jsonl")[0].request.state
    root = Path(__file__).resolve().parents[2]
    config = root / "configs/benchmarks"
    registry = Registry.load(
        config / "models.toml", config / "suites.toml", config / "profiles.toml"
    )
    suite = registry.suites["smoke"].model_copy(
        update={
            "dataset": str(tmp_path / "views/M1.jsonl"),
            "track": "native_multimodal",
            "information_view": "native_media",
        }
    )
    profile = registry.profiles["smoke"].model_copy(update={"warmup": 0, "repetitions": 1})
    registry = replace(registry, suites={"smoke": suite}, profiles={"smoke": profile})
    execute(
        registry,
        "smoke",
        output=tmp_path / "run",
        lockfile=root / "uv.lock",
        enable_models=["uniform"],
    )
    score = counterfactual_summary(tmp_path / "run", "uniform", tmp_path / "views/media-views.json")
    assert score["required_change_pairs"] == 1
    assert score["operational_pair_accuracy"] == 0
    run_manifest = tmp_path / "run/run_manifest.json"
    data = json.loads(run_manifest.read_text())
    data["status"] = "interrupted"
    run_manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="finished"):
        counterfactual_summary(tmp_path / "run", "uniform", tmp_path / "views/media-views.json")
    other["case"]["request"]["state"] = "different textual hint"
    source.write_text("\n".join(json.dumps(r) for r in [row, other]))
    with pytest.raises(ValueError, match="fixed"):
        prepare_media_views(source, tmp_path / "invalid")


def test_pipeline_measures_real_stage_and_preserves_cost_when_decision_fails(decision_request):
    decision_request.media = [AudioInput(samples=[0.1])]

    def handler(request):
        payload = json.loads(request.content)
        assert set(payload) == {"media"}
        return httpx.Response(200, json={"text": "A spoken refund request", "cost_usd": 0.03})

    class Decision(UniformBackend):
        def predict(self, request):
            assert not request.media
            assert request.state["media_observation"] == "A spoken refund request"
            return super().predict(request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        preprocessor = HTTPBackend("https://example.invalid/asr", client=client)
        adapter = PipelineAdapter(Decision(), preprocessor)
        result = adapter.predict(decision_request)
        assert result["pipeline"]["preprocess_reported_cost_usd"] == 0.03
        assert result["pipeline"]["preprocess_ms"] > 0

        class Broken:
            def predict(self, request):
                raise BackendResponseError("private provider content")

        adapter.adapter = Broken()
        with pytest.raises(BackendResponseError) as exc:
            adapter.predict(decision_request)
        assert exc.value.evaluation_pipeline["preprocess_reported_cost_usd"] == 0.03
