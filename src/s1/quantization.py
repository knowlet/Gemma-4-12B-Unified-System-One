"""Explicit single-GPU bitsandbytes inference; no silent precision fallback."""

from __future__ import annotations

import logging
import threading

QUANTIZATION_MODES = ("none", "int8", "nf4")
# The readout indexes dense LM-head rows directly. These lightweight media
# projections are also kept dense to preserve the existing processor contract.
DENSE_MODULES = ("lm_head", "model.embed_vision", "model.embed_audio")
_INT8_LOGGER = "bitsandbytes.autograd._functions"
_INT8_CAST_MESSAGE = (
    "MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)
_LOG_FILTER_LOCK = threading.Lock()


class _ExpectedInt8CastOnce(logging.Filter):
    """Retain the first expected cast message without hiding other diagnostics."""

    def __init__(self):
        super().__init__()
        self._seen = False
        self._lock = threading.Lock()

    def filter(self, record):
        if (
            record.name != _INT8_LOGGER
            or record.levelno != logging.WARNING
            or record.getMessage() != _INT8_CAST_MESSAGE
        ):
            return True
        with self._lock:
            if self._seen:
                return False
            self._seen = True
            return True


def limit_expected_int8_cast_warning():
    """Log the known bitsandbytes BF16 activation cast once per process.

    bitsandbytes 0.50.2 logs this expected conversion on every Linear8bitLt
    forward. Attach only to that module's logger; do not change logger levels,
    Python warning filters, other cast dtypes, errors, or unrelated diagnostics.
    """
    logger = logging.getLogger(_INT8_LOGGER)
    with _LOG_FILTER_LOCK:
        existing = next((f for f in logger.filters if isinstance(f, _ExpectedInt8CastOnce)), None)
        if existing is not None:
            return existing
        once = _ExpectedInt8CastOnce()
        logger.addFilter(once)
        return once


def loading_kwargs(mode, *, device, precision):
    """Return from_pretrained kwargs, validating before checkpoint downloads."""
    if mode not in QUANTIZATION_MODES:
        raise ValueError("quantization must be none, int8, or nf4")
    if mode == "none":
        return {}
    if not str(device).startswith("cuda") or precision != "bfloat16":
        raise ValueError("int8/nf4 quantization requires CUDA and bfloat16 precision")
    import torch
    from transformers import BitsAndBytesConfig

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise ImportError("install gemma-system-one[inference,quantization] for int8/nf4") from exc

    options = {"llm_int8_skip_modules": list(DENSE_MODULES)}
    if mode == "int8":
        limit_expected_int8_cast_warning()
        options.update(load_in_8bit=True, llm_int8_threshold=6.0)
    else:
        options.update(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    # Automatic offloading would change both the memory and latency experiment.
    return {
        "quantization_config": BitsAndBytesConfig(**options),
        "device_map": {"": str(device)},
    }


def inspect_quantization(lm, mode):
    """Verify actual conversion and keep an auditable, compact runtime receipt."""
    if mode == "none":
        if getattr(lm, "is_quantized", False):
            raise ValueError("pre-quantized checkpoint requires an explicit quantization mode")
        return {"method": "none", "quantized_linear_modules": 0}
    import bitsandbytes as bnb

    expected = bnb.nn.Linear8bitLt if mode == "int8" else bnb.nn.Linear4bit
    quantized_types = (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)
    modules = [
        (name, module) for name, module in lm.named_modules() if isinstance(module, quantized_types)
    ]
    if not modules:
        raise ValueError("requested quantization did not convert any linear modules")
    if any(not isinstance(module, expected) for _, module in modules):
        raise ValueError("loaded quantization differs from the requested mode")
    if any("language_model" not in name.split(".") for name, _ in modules):
        raise ValueError("quantization unexpectedly converted a media projection or LM head")
    expected_storage = "torch.int8" if mode == "int8" else "torch.uint8"
    if any(str(module.weight.dtype) != expected_storage for _, module in modules):
        raise ValueError("quantized modules do not contain the expected packed weight storage")
    base = lm.get_base_model() if hasattr(lm, "peft_config") else lm
    head = base.get_output_embeddings()
    if isinstance(head, quantized_types) or not head.weight.is_floating_point():
        raise ValueError("candidate readout requires an unquantized floating-point LM head")
    return {
        "method": "bitsandbytes",
        "mode": mode,
        "quantized_linear_modules": len(modules),
        "dense_modules": list(DENSE_MODULES),
        "head_dtype": str(head.weight.dtype).removeprefix("torch."),
        "nf4_compute_dtype": "bfloat16" if mode == "nf4" else None,
        "double_quantization": mode == "nf4",
        "int8_outlier_threshold": 6.0 if mode == "int8" else None,
        "int8_activation_quantization_dtype": "float16" if mode == "int8" else None,
        "int8_expected_cast_warning": "once_per_process" if mode == "int8" else None,
        "cpu_offload": False,
    }
