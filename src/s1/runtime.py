"""Resolve optional Torch runtime settings without importing Torch at package import."""

from __future__ import annotations


def resolve_device(device=None):
    import torch

    if device is not None and str(device) != "auto":
        return str(torch.device(device))
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype, device):
    import torch

    supported = {name: getattr(torch, name) for name in ("float32", "float16", "bfloat16")}
    if dtype is None or dtype == "auto":
        device_type = torch.device(device).type
        if device_type == "cuda":
            return torch.bfloat16
        if device_type == "mps":
            # MPS gained bfloat16 support in macOS 14. Keep float32 on older
            # systems: the real 12B checkpoint produced invalid image-request
            # probabilities in float16, even though tiny-model tests passed.
            return torch.bfloat16 if torch.backends.mps.is_macos_or_newer(14, 0) else torch.float32
        return torch.float32
    if isinstance(dtype, str) and dtype in supported:
        return supported[dtype]
    if isinstance(dtype, torch.dtype) and dtype in supported.values():
        return dtype
    raise ValueError("dtype must be auto, float32, float16, or bfloat16")
