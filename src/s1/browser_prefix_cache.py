"""CPU BF16 SDPA key-axis alignment for independent causal prefix forks.

Only masked-out columns are restored, as zeros, in transient attention tensors.
The native sliding layers still retain exactly their usual window tail. This
keeps SDPA's key reduction coordinates consistent with full-context execution;
it does not reconstruct evicted KV or claim arbitrary-model bitwise equivalence.
"""

from __future__ import annotations

import copy

import torch
from torch.nn.functional import pad
from transformers.cache_utils import Cache, DynamicCache, DynamicSlidingWindowLayer


def _needs_alignment(layer) -> bool:
    return (
        isinstance(layer, DynamicSlidingWindowLayer)
        and layer.is_initialized
        and layer.keys.device.type == "cpu"
        and layer.keys.dtype == torch.bfloat16
    )


class AbsoluteKeyDynamicCache(DynamicCache):
    """A fork-local view over native layers, for fixed-position causal suffixes.

    Constructed only from independently copied, non-offloaded native layers.
    Mask sizes and update returns must change together: zero-filled evicted
    positions are outside the sliding window of every query in the suffix.
    Gemma's shared-KV consumers receive these same transient, aligned tensors.
    """

    key_layout = "absolute_zero_padded_cpu_bf16_sdpa"

    def __init__(self, layers):
        Cache.__init__(self, layers=layers)
        self.alignment_records = []

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        if layer_idx < len(self.layers) and _needs_alignment(self.layers[layer_idx]):
            return self.layers[layer_idx].get_seq_length() + query_length, 0
        return super().get_mask_sizes(query_length, layer_idx)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        keys, values = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        layer = self.layers[layer_idx]
        if _needs_alignment(layer):
            missing = layer.get_seq_length() - keys.shape[-2]
            if missing > 0:
                keys = pad(keys, (0, 0, missing, 0))
                values = pad(values, (0, 0, missing, 0))
                self.alignment_records.append(
                    {
                        "layer_idx": layer_idx,
                        "masked_left_key_columns": missing,
                        "attention_key_columns": keys.shape[-2],
                        "stored_key_columns": layer.keys.shape[-2],
                        "transient_returned_kv_bytes": keys.numel() * keys.element_size()
                        + values.numel() * values.element_size(),
                    }
                )
        return keys, values


def fork_prefix_cache(cache: DynamicCache, *, attention_implementation: str) -> DynamicCache:
    """Deep-copy without global patches, backend changes, or prefix mutations."""
    if (
        type(cache) is DynamicCache
        and not cache.offloading
        and attention_implementation == "sdpa"
        and any(_needs_alignment(layer) for layer in cache.layers)
    ):
        return AbsoluteKeyDynamicCache(copy.deepcopy(cache.layers))
    return copy.deepcopy(cache)
