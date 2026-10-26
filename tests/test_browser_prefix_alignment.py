"""Regressions for CPU BF16 SDPA alignment without changing native readout."""

import copy

import pytest
import test_browser_prefix as original

from s1.browser_prefix import predict_prefix

model = original.model
torch = original.torch
pytestmark = pytest.mark.inference


@pytest.mark.parametrize("prefix", [32, 63, 64, 65, 1371])
@pytest.mark.parametrize("suffix", [1, 65])
def test_absolute_mask_and_transient_padding_preserve_native_cache(model, prefix, suffix):
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_sliding_window_causal_mask

    from s1.browser_prefix_cache import AbsoluteKeyDynamicCache, fork_prefix_cache

    config = model.backbone.config.get_text_config()
    native = DynamicCache(config=config)
    keys = torch.randn(1, 1, prefix + suffix, 8, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    native.update(keys[:, :, :prefix], values[:, :, :prefix], 0)
    original_tail = native.layers[0].keys.clone()
    fork = fork_prefix_cache(native, attention_implementation="sdpa")
    assert isinstance(fork, AbsoluteKeyDynamicCache)
    assert fork.get_mask_sizes(suffix, 0) == (prefix + suffix, 0)
    missing = max(prefix - config.sliding_window + 1, 0)
    mask = create_sliding_window_causal_mask(
        config=config,
        inputs_embeds=torch.zeros(1, suffix, config.hidden_size, dtype=torch.bfloat16),
        attention_mask=torch.ones(1, prefix + suffix, dtype=torch.long),
        past_key_values=fork,
        position_ids=torch.arange(prefix, prefix + suffix).unsqueeze(0),
    )
    if mask is None:
        # SDPA may omit the mask for a one-token query that can see all keys.
        assert suffix == 1
        allowed = torch.ones(1, 1, suffix, prefix + suffix, dtype=torch.bool)
    else:
        allowed = mask if mask.dtype == torch.bool else mask == 0
    assert not allowed[..., :missing].any()
    qpos = torch.arange(prefix, prefix + suffix)[:, None]
    kpos = torch.arange(prefix + suffix)[None, :]
    expected_mask = (qpos >= kpos) & (qpos - kpos < config.sliding_window)
    assert torch.equal(allowed[0, 0], expected_mask)
    returned_keys, returned_values = fork.update(keys[:, :, prefix:], values[:, :, prefix:], 0)
    assert torch.equal(returned_keys[:, :, missing:], keys[:, :, missing:])
    assert torch.equal(returned_values[:, :, missing:], values[:, :, missing:])
    assert torch.count_nonzero(returned_keys[:, :, :missing]) == 0
    assert torch.count_nonzero(returned_values[:, :, :missing]) == 0
    assert native.get_seq_length() == prefix
    assert fork.get_seq_length() == prefix + suffix
    assert fork.layers[0].keys.shape[-2] == min(prefix + suffix, config.sliding_window - 1)
    assert torch.equal(native.layers[0].keys, original_tail)
    assert fork.layers[0].keys.data_ptr() != native.layers[0].keys.data_ptr()
    assert len(fork.layers) == config.num_hidden_layers - config.num_kv_shared_layers
    if missing:
        record = fork.alignment_records[0]
        assert record["masked_left_key_columns"] == missing
        expected_bytes = (
            returned_keys.numel() + returned_values.numel()
        ) * returned_keys.element_size()
        assert record["transient_returned_kv_bytes"] == expected_bytes
    else:
        assert fork.alignment_records == []


@pytest.mark.parametrize(
    ("dtype", "implementation"), [(torch.float32, "sdpa"), (torch.bfloat16, "eager")]
)
def test_unaffected_execution_keeps_native_key_layout(model, dtype, implementation):
    from transformers.cache_utils import DynamicCache

    from s1.browser_prefix_cache import fork_prefix_cache

    native = DynamicCache(config=model.backbone.config.get_text_config())
    keys = torch.randn(1, 1, 70, 8, dtype=dtype)
    native.update(keys, keys.clone(), 0)
    fork = fork_prefix_cache(native, attention_implementation=implementation)
    assert type(fork) is DynamicCache
    assert fork.get_mask_sizes(3, 0) == native.get_mask_sizes(3, 0)
    assert fork.layers[0].keys.data_ptr() != native.layers[0].keys.data_ptr()


def test_sdpa_output_matches_full_key_axis_with_real_bf16_tensors(model):
    from torch.nn.functional import scaled_dot_product_attention
    from transformers.cache_utils import DynamicCache

    from s1.browser_prefix_cache import fork_prefix_cache

    # The actual failing row's dimensions: the eviction offset 1308 is not
    # aligned with SDPA reduction blocks. There is no model/winner mock.
    prefix, total, window = 1371, 3046, 64
    query = torch.randn(1, 2, total, 8, dtype=torch.bfloat16)
    keys = torch.randn(1, 1, total, 8, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    pos = torch.arange(total)
    mask = (pos[:, None] >= pos) & (pos[:, None] - pos < window)
    expected = scaled_dot_product_attention(
        query,
        keys.repeat_interleave(2, 1),
        values.repeat_interleave(2, 1),
        attn_mask=mask,
        scale=1.0,
    )[:, :, prefix:]
    native = DynamicCache(config=model.backbone.config.get_text_config())
    native.update(keys[:, :, :prefix], values[:, :, :prefix], 0)
    fork = fork_prefix_cache(native, attention_implementation="sdpa")
    aligned_keys, aligned_values = fork.update(keys[:, :, prefix:], values[:, :, prefix:], 0)
    actual = scaled_dot_product_attention(
        query[:, :, prefix:],
        aligned_keys.repeat_interleave(2, 1),
        aligned_values.repeat_interleave(2, 1),
        attn_mask=mask[prefix:],
        scale=1.0,
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_original_bf16_winner_with_52_candidates_in_first_suffix_chunk(model, batch_size):
    model.lm.to(dtype=torch.bfloat16)
    rows = original.prepared_rows(model, "goal-first")
    rows = [rows[2], rows[0], rows[1]]
    before = copy.deepcopy(rows)
    expected = original.full_answers(model, rows)
    assert expected[0]["choice"] == "item-37"
    actual = predict_prefix(model, rows, prompt_mode="goal-first", max_suffix_batch_size=batch_size)
    for row, answer, reference in zip(rows, actual, expected, strict=True):
        question = row["request"].questions[0].id
        result = answer["answers"][question]
        assert result["choice"] == reference["choice"]
        assert result["probabilities"] == pytest.approx(reference["probabilities"], abs=2e-3)
        assert answer["execution"]["attention_key_layout"] == "absolute_zero_padded_cpu_bf16_sdpa"
    trace = actual[0]["prefix"]
    assert trace["forward_calls"] == 1 + (len(rows) + batch_size - 1) // batch_size
    assert trace["cache_layers"][0]["key_shape"][-2] == 63
    assert all(branch["attention_key_alignment"] for branch in trace["branches"])
    for row, saved in zip(rows, before, strict=True):
        for key in row["prepared"][0]:
            assert torch.equal(row["prepared"][0][key], saved["prepared"][0][key])
