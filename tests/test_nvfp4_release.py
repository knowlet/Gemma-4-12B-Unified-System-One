"""Offline release gates: source identity, packed reload parity and evaluation."""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def exporter(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "test_export_nvfp4_release", ROOT / "scripts/export_nvfp4_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def __getitem__(self, item):
        return Tensor(self.value[item])

    def __sub__(self, other):
        return Tensor(self.value - other.value)

    def abs(self):
        return Tensor(np.abs(self.value))

    def max(self):
        return self.value.max()


@pytest.fixture
def release_env(exporter, tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    sidecar = {"temperature": 2.0, "training_recipe_sha256": "a" * 64}
    (source / "s1_config.json").write_text(json.dumps(sidecar))
    (source / "model.safetensors").write_bytes(b"verified trained source")
    expected = {path.name: exporter.file_sha256(path) for path in source.iterdir()}
    source_receipt = tmp_path / "source-receipt.json"
    source_receipt.write_text(json.dumps({"merged": expected}))
    questions = [SimpleNamespace(labels=lambda: ["false", "true"])]
    data = {
        split: [
            SimpleNamespace(
                id=f"{split}-{i}",
                request=SimpleNamespace(questions=questions, state=f"{split}-{i}"),
            )
            for i in range(2)
        ]
        for split in ("calibration", "test", "media", "regression_text")
    }
    identities = {name: {"dataset_sha256": name + "-identity"} for name in data}
    state = SimpleNamespace(
        source=source,
        output=tmp_path / "output",
        source_receipt=source_receipt,
        expected=expected,
        data=data,
        events=[],
        models=[],
        commits=0,
        drift=False,
        fail_evaluation=False,
        fail_calibration=False,
    )

    class Model:
        def __init__(self, name, **kwargs):
            self.path = Path(name)
            self.temperature = kwargs.get("temperature", sidecar["temperature"])
            self.kind = (
                "packed" if self.path.name == "package" else kwargs.get("quantization", "bf16")
            )
            self.lm = self
            self.quantization_details = {"mode": kwargs.get("quantization", "none")}
            self.processor = SimpleNamespace(save_pretrained=self.save_processor)
            state.models.append(self)
            state.events.append(("load", self.kind))

        def save_processor(self, path):
            (path / "tokenizer.json").write_text("{}")

        def logits(self, request):
            offset = 1.0 if state.drift and self.kind == "packed" else 0.0
            return Tensor([[3.0 + offset, 1.0, float("-inf")]])

    def save(model, path):
        state.events.append(("save", model.kind))
        path.mkdir()
        (path / "model.safetensors").write_bytes(b"packed weights")
        (path / "config.json").write_text("{}")

    def calibrate(model, cases):
        assert model.kind == "packed"
        assert cases is data["calibration"]
        state.events.append(("calibrate", [case.id for case in cases]))
        if state.fail_calibration:
            raise ValueError("calibration failed")
        model.temperature = 1.5
        return {"temperature": 1.5, "n": len(cases)}

    def evaluate(model, cases, root, prefix, commit):
        state.events.append(("evaluate", model.kind))
        if state.fail_evaluation and model.kind == "packed":
            raise RuntimeError("incomplete held-out evaluation")
        return {
            split: {"coverage": 1.0, "metrics": {"n": len(cases[split]), "acc": 0.75}}
            for split in ("test", "media", "regression_text")
        }

    def commit():
        state.commits += 1

    state.commit = commit
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            inference_mode=nullcontext,
            equal=lambda left, right: np.array_equal(left.value, right.value),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "s1.nvfp4",
        SimpleNamespace(save_nvfp4=save, validate_nvfp4_artifact=lambda path: {"status": "ok"}),
    )
    monkeypatch.setitem(sys.modules, "s1.unified", SimpleNamespace(UnifiedDecisionModel=Model))
    monkeypatch.setattr(exporter, "load_release_data", lambda path: (data, identities))
    monkeypatch.setattr(exporter, "_runtime", lambda: {"gpu": "test Blackwell"})
    monkeypatch.setattr(exporter, "_free", lambda: None)
    monkeypatch.setattr(exporter, "calibrate", calibrate)
    monkeypatch.setattr(exporter, "_evaluate", evaluate)
    return state


def run(exporter, env):
    return exporter.run_release(
        env.source, "unused-injected-datasets", env.output, env.source_receipt, commit=env.commit
    )


def test_completed_release_evaluates_reloaded_weights_and_binds_calibration(exporter, release_env):
    result = run(exporter, release_env)
    assert result["status"] == "ok"
    assert release_env.events == [
        ("load", "bf16"),
        ("evaluate", "bf16"),
        ("load", "nvfp4"),
        ("save", "nvfp4"),
        ("load", "packed"),
        ("calibrate", ["calibration-0", "calibration-1"]),
        ("evaluate", "packed"),
    ]
    package = release_env.output / "package"
    conversion = json.loads((package / "conversion-manifest.json").read_text())
    evaluation = json.loads((package / "evaluation.json").read_text())
    assert evaluation["conversion_manifest_sha256"] == exporter.file_sha256(
        package / "conversion-manifest.json"
    )
    assert evaluation["temperature"] == 1.5 and evaluation["bf16_temperature"] == 2.0
    assert all(row["exact_logits_equal"] for row in evaluation["reload_parity"])
    assert json.loads((package / "s1_config.json").read_text())["temperature"] == 1.5
    assert json.loads((release_env.source / "s1_config.json").read_text())["temperature"] == 2.0
    assert conversion["files"]["s1_config.json"]["sha256"] == exporter.file_sha256(
        package / "s1_config.json"
    )
    assert not {"conversion-manifest.json", "evaluation.json"} & conversion["files"].keys()
    assert release_env.commits >= 3


def test_source_mismatch_fails_before_model_load_and_preserves_failure_receipt(
    exporter, release_env
):
    (release_env.source / "model.safetensors").write_bytes(b"unverified source")
    with pytest.raises(ValueError, match="source checksum mismatch"):
        run(exporter, release_env)
    assert release_env.models == []
    receipt = json.loads((release_env.output / "receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert not (release_env.output / "package").exists()


def test_reload_drift_blocks_calibration_and_success_artifacts(exporter, release_env):
    release_env.drift = True
    with pytest.raises(RuntimeError, match="changed candidate logits"):
        run(exporter, release_env)
    assert not any(event[0] == "calibrate" for event in release_env.events)
    assert not (release_env.output / "package/conversion-manifest.json").exists()
    assert not (release_env.output / "package/evaluation.json").exists()
    assert json.loads((release_env.output / "receipt.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("phase", ["calibration", "evaluation"])
def test_failed_validation_never_emits_success_evaluation(exporter, release_env, phase):
    setattr(release_env, f"fail_{phase}", True)
    with pytest.raises((ValueError, RuntimeError), match="failed|incomplete"):
        run(exporter, release_env)
    assert not (release_env.output / "package/evaluation.json").exists()
    assert json.loads((release_env.output / "receipt.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("failure", ["coverage", "warmup"])
def test_population_failure_is_saved_then_stops_remaining_splits(
    exporter, tmp_path, monkeypatch, failure
):
    report = {"coverage": 0.5 if failure == "coverage" else 1.0, "warmup_errors": []}
    if failure == "warmup":
        report["warmup_errors"] = ["RuntimeError"]
    calls = []

    def evaluate(model, cases, warmup):
        calls.append(cases)
        return report

    monkeypatch.setitem(sys.modules, "s1.benchmark", SimpleNamespace(evaluate=evaluate))
    model = SimpleNamespace(reset_memory_peak=lambda: None, memory_snapshot=lambda: {})
    data = {name: [name] for name in ("test", "media", "regression_text")}
    with pytest.raises(RuntimeError, match="did not complete every request"):
        exporter._evaluate(model, data, tmp_path, "nvfp4", lambda: None)
    assert calls == [["test"]]
    saved = json.loads((tmp_path / "nvfp4-test.json").read_text())
    assert saved["coverage"] == report["coverage"]
    assert not (tmp_path / "nvfp4-media.json").exists()
