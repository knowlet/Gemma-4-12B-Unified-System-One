"""Finite CPU BF16 bytes for the local Gemma4 Unified converter only.

The surrounding converter is llama.cpp, MIT, pinned in source-manifest.json.
This helper keeps matrices in their original BF16 representation. Norms and
scalars continue through the upstream float32 path.
"""
import numpy as np
import torch
import gguf
import sys


def _eager_raw(tensor):
    if sys.byteorder != 'little':
        raise ValueError('Little-endian host required for BF16 byte export')
    if (tensor.dtype != torch.bfloat16 or tensor.device.type != 'cpu'
            or not tensor.is_contiguous() or tensor.ndim < 2):
        raise ValueError('Contiguous multidimensional CPU BF16 tensor required')
    raw = tensor.view(torch.uint8).numpy()
    words = raw.view(np.uint16).reshape(-1)
    # Bound temporary masks; infinity and NaN are refused before tensor writing.
    for start in range(0, words.size, 1 << 19):
        if np.any((words[start:start + (1 << 19)] & 0x7f80) == 0x7f80):
            raise ValueError('Nonfinite BF16 tensor refused by byte export')
    return raw


def raw_bf16_numpy(tensor):
    if tensor.dtype != torch.bfloat16 or len(tensor.shape) < 2:
        raise ValueError('Multidimensional BF16 metadata required')
    shape = tuple(tensor.shape[:-1]) + (tensor.shape[-1] * 2,)
    return gguf.LazyNumpyTensor(
        meta=gguf.LazyNumpyTensor.meta_with_dtype_and_shape(np.uint8, shape),
        args=(tensor,), func=_eager_raw)
