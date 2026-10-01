"""Policy and accounting checks do not require inference dependencies or GPUs."""

import importlib.util
import logging
import sys
from types import SimpleNamespace

import pytest

from s1 import resources
from s1.quantization import (
    DENSE_MODULES,
    inspect_quantization,
    limit_expected_int8_cast_warning,
    loading_kwargs,
)
from s1.resources import memory_snapshot, reset_memory_peak


@pytest.mark.parametrize("mode", ["int8", "nf4"])
@pytest.mark.parametrize("device,precision", [("cpu", "bfloat16"), ("cuda", "float32")])
def test_invalid_quantization_runtime_fails_before_loading(mode, device, precision):
    with pytest.raises(ValueError, match="CUDA and bfloat16"):
        loading_kwargs(mode, device=device, precision=precision)


def test_no_quantization_needs_no_optional_dependencies():
    assert loading_kwargs("none", device="cpu", precision="float32") == {}
    with pytest.raises(ValueError, match="quantization"):
        loading_kwargs("fp8", device="cuda", precision="bfloat16")
    with pytest.raises(ValueError, match="explicit"):
        inspect_quantization(SimpleNamespace(is_quantized=True), "none")


@pytest.mark.parametrize("mode", ["int8", "nf4"])
def test_quantization_keeps_candidate_head_and_media_dense(monkeypatch, mode):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(BitsAndBytesConfig=dict))
    kwargs = loading_kwargs(mode, device="cuda:1", precision="bfloat16")
    assert kwargs["device_map"] == {"": "cuda:1"}
    config = kwargs["quantization_config"]
    assert config["llm_int8_skip_modules"] == list(DENSE_MODULES)
    if mode == "nf4":
        assert config["load_in_4bit"] is True
        assert config["bnb_4bit_compute_dtype"] == "bf16"
        assert config["bnb_4bit_use_double_quant"] is True
    else:
        assert config["load_in_8bit"] is True
        assert config["llm_int8_threshold"] == 6.0


def test_missing_conversion_and_wrong_scope_are_detected(monkeypatch):
    class Int8:
        pass

    class NF4:
        pass

    monkeypatch.setitem(
        sys.modules,
        "bitsandbytes",
        SimpleNamespace(nn=SimpleNamespace(Linear8bitLt=Int8, Linear4bit=NF4)),
    )
    with pytest.raises(ValueError, match="did not convert"):
        inspect_quantization(SimpleNamespace(named_modules=lambda: []), "int8")
    with pytest.raises(ValueError, match="differs"):
        inspect_quantization(
            SimpleNamespace(named_modules=lambda: [("model.language_model.q_proj", NF4())]),
            "int8",
        )
    with pytest.raises(ValueError, match="media projection"):
        inspect_quantization(
            SimpleNamespace(named_modules=lambda: [("model.embed_audio.projection", NF4())]),
            "nf4",
        )


def test_memory_bytes_and_explicit_peak_window(monkeypatch):
    calls = []
    cuda = SimpleNamespace(
        synchronize=lambda device: calls.append(("synchronize", device)),
        reset_peak_memory_stats=lambda device: calls.append(("reset", device)),
        mem_get_info=lambda device: (100, 1000),
        get_device_name=lambda device: "test GPU",
        memory_allocated=lambda device: 600,
        memory_reserved=lambda device: 700,
        max_memory_allocated=lambda device: 650,
        max_memory_reserved=lambda device: 800,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    reset_memory_peak("cuda:1")
    receipt = memory_snapshot(SimpleNamespace(get_memory_footprint=lambda: 512), "cuda:1")
    assert calls == [("synchronize", "cuda:1"), ("reset", "cuda:1"), ("synchronize", "cuda:1")]
    assert receipt["model_footprint_bytes"] == 512
    assert receipt["allocated_bytes"] == 600
    assert receipt["peak_allocated_bytes"] == 650
    assert receipt["reserved_bytes"] == 700
    assert receipt["peak_reserved_bytes"] == 800
    assert receipt["gpu_total_bytes"] == 1000
    assert receipt["device_used_bytes"] == 900
    assert receipt["process_peak_scope"] == (
        "process_lifetime" if resources.resource is not None else "unavailable"
    )


@pytest.mark.parametrize("platform,rss_bytes", [("linux", 7168), ("darwin", 7)])
def test_cpu_memory_does_not_query_cuda(monkeypatch, platform, rss_bytes):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(resources, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(
        resources,
        "resource",
        SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=7)),
    )
    reset_memory_peak("cpu")
    receipt = memory_snapshot(device="cpu")
    assert receipt["process_peak_rss_bytes"] == rss_bytes
    assert receipt["process_peak_rss_kib"] == rss_bytes / 1024
    assert receipt["process_peak_scope"] == "process_lifetime"
    assert "peak_allocated_bytes" not in receipt


def test_memory_module_import_and_cpu_snapshot_without_unix_resource(monkeypatch):
    monkeypatch.setitem(sys.modules, "resource", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    spec = importlib.util.spec_from_file_location("resources_without_unix", resources.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.reset_memory_peak("cpu")
    receipt = module.memory_snapshot(SimpleNamespace(get_memory_footprint=lambda: 512), "cpu")
    assert receipt["model_footprint_bytes"] == 512
    assert receipt["process_peak_scope"] == "unavailable"
    assert "process_peak_rss_bytes" not in receipt
    assert "process_peak_rss_kib" not in receipt
    assert "peak_allocated_bytes" not in receipt


def test_int8_logging_keeps_first_cast_and_all_other_diagnostics(caplog):
    logger = logging.getLogger("bitsandbytes.autograd._functions")
    original_filters = logger.filters[:]
    logger.filters.clear()
    message = "MatMul8bitLt: inputs will be cast from %s to float16 during quantization"
    try:
        first = limit_expected_int8_cast_warning()
        assert limit_expected_int8_cast_warning() is first
        with caplog.at_level(logging.WARNING):
            logger.warning(message, "torch.bfloat16")
            logger.warning(message, "torch.bfloat16")
            # Reinstalling for another model in the same process must not reset it.
            limit_expected_int8_cast_warning()
            logger.warning(message, "torch.bfloat16")
            for _ in range(2):
                logger.warning(message, "torch.float32")
                logger.warning("a different quantization warning")
            logger.error(message, "torch.bfloat16")
            logging.getLogger("another.module").warning(message, "torch.bfloat16")
        expected = message % "torch.bfloat16"
        matching_warnings = [
            r
            for r in caplog.records
            if r.name == logger.name and r.levelno == logging.WARNING and r.getMessage() == expected
        ]
        assert len(matching_warnings) == 1
        assert len(caplog.records) == 7
        assert any(r.levelno == logging.ERROR for r in caplog.records)
        assert any(r.name == "another.module" for r in caplog.records)
    finally:
        logger.filters[:] = original_filters


def test_int8_log_once_is_thread_safe():
    from concurrent.futures import ThreadPoolExecutor

    from s1.quantization import _ExpectedInt8CastOnce

    once = _ExpectedInt8CastOnce()
    record = logging.LogRecord(
        "bitsandbytes.autograd._functions",
        logging.WARNING,
        "",
        0,
        "MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization",
        (),
        None,
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(once.filter, [record] * 100)) == 1
