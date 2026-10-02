"""Exercise cloud entrypoint routing without importing Modal or starting jobs."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def cloud_module(monkeypatch):
    class Image:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()

        def uv_sync(self, **kwargs):
            self.extras = kwargs["extras"]
            return self

        def env(self, environment):
            self.environment = environment
            return self

        def add_local_python_source(self, *args):
            return self

        def add_local_dir(self, *args):
            return self

    def decorate(**kwargs):
        return lambda function: function

    fake_modal = SimpleNamespace(
        is_local=lambda: True,
        App=lambda *a: SimpleNamespace(function=decorate, cls=decorate, local_entrypoint=decorate),
        Volume=SimpleNamespace(from_name=lambda *a, **kw: SimpleNamespace(commit=lambda: None)),
        Image=Image,
        concurrent=decorate,
        enter=decorate,
        asgi_app=decorate,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.delenv("S1_HF_SECRET", raising=False)
    monkeypatch.setenv("S1_QUANTIZATION", "nf4")
    monkeypatch.setenv("S1_ADAPTER_PATH", "/vol/ce-adapter")
    monkeypatch.setenv("S1_PRECISION", "bfloat16")
    path = Path(__file__).resolve().parents[1] / "apps/modal/unified.py"
    spec = importlib.util.spec_from_file_location("s1_modal_quantization_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cloud_benchmark_and_server_forward_same_frozen_options(cloud_module, monkeypatch):
    from s1 import backends, benchmark

    calls = []
    monkeypatch.setattr(backends, "GemmaBackend", lambda *args, **kw: calls.append(kw))
    monkeypatch.setattr(benchmark, "evaluate", lambda *a, **kw: {"counts": {"ok": 1}})
    assert cloud_module.benchmark([], "gemma") == {"counts": {"ok": 1}}
    cloud_module.Server().load()
    expected = {
        "revision": cloud_module.revision,
        "quantization": "nf4",
        "adapter_path": "/vol/ce-adapter",
        "precision": "bfloat16",
    }
    assert calls == [expected, expected]
    assert "quantization" in cloud_module.runtime.extras
    assert cloud_module.runtime.environment["S1_QUANTIZATION"] == "nf4"
    assert cloud_module.runtime.environment["S1_ADAPTER_PATH"] == "/vol/ce-adapter"


def test_cloud_model_smoke_forwards_options_and_reports_memory(cloud_module, monkeypatch):
    from s1 import unified

    calls = []
    fake_model = SimpleNamespace(
        name="test-model",
        revision="a" * 40,
        quantization="nf4",
        quantization_details={"method": "bitsandbytes"},
        predict=lambda request: {"media_count": len(request.media)},
        memory_snapshot=lambda: {"peak_allocated_bytes": 123},
    )

    def load(*args, **kwargs):
        calls.append(kwargs)
        return fake_model

    monkeypatch.setattr(unified, "UnifiedDecisionModel", load)
    monkeypatch.setitem(
        sys.modules,
        "PIL",
        SimpleNamespace(
            Image=SimpleNamespace(
                new=lambda *a: SimpleNamespace(save=lambda buffer, **kw: buffer.write(b"fixture")),
            )
        ),
    )
    result = cloud_module.model_smoke()
    assert calls == [
        {
            "revision": cloud_module.revision,
            "quantization": "nf4",
            "adapter_path": "/vol/ce-adapter",
            "precision": "bfloat16",
        }
    ]
    assert result["memory"] == {"peak_allocated_bytes": 123}
    assert set(result["results"]) == {"text", "image", "audio"}


def test_cloud_laya_and_training_reject_quantized_environment(cloud_module):
    with pytest.raises(ValueError, match="require Gemma"):
        cloud_module.benchmark([], "laya")
    with pytest.raises(ValueError, match="unset them before training"):
        cloud_module.train([], [])


def test_public_backend_metadata_records_quantized_adapter(monkeypatch):
    from s1 import unified
    from s1.backends import GemmaBackend

    model = SimpleNamespace(
        name="test",
        revision="a" * 40,
        temperature=1.2,
        device="cuda",
        dtype="bfloat16",
        attn_implementation="sdpa",
        max_context=16384,
        head=SimpleNamespace(weight=SimpleNamespace(dtype="torch.bfloat16")),
        quantization="nf4",
        quantization_details={"method": "bitsandbytes"},
    )
    monkeypatch.setattr(unified, "UnifiedDecisionModel", lambda *a, **kw: model)
    backend = GemmaBackend(adapter_path="/vol/ce-adapter", quantization="nf4")
    assert backend.metadata["quantization"] == "nf4"
    assert backend.metadata["adapter_path"] == "/vol/ce-adapter"
    assert backend.metadata["precision"] == "bfloat16"
