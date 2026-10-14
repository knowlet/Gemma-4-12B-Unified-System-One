"""Genuine tiny Gemma CPU equivalence for independent prefix-cache branches."""

import copy
import json
from types import SimpleNamespace

import pytest

from s1.browser_policy import BrowserDecisionBackend
from s1.browser_prefix import BrowserPrefixUnsupported, predict_prefix
from s1.contracts import DecisionRequest, ImageInput, answer_from_probabilities
from s1.errors import BackendResponseError, RequestValidationError
from s1.unified import UnifiedDecisionModel, project_candidates

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4UnifiedConfig"):
    pytest.skip("Gemma 4 Unified requires the inference extra", allow_module_level=True)

pytestmark = pytest.mark.inference


class Tokenizer:
    pad_token_id = 0
    unk_token_id = -1

    def encode(self, text, **_kwargs):
        return [ord(character) % 240 for character in text]


class Processor:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return "<user>" + messages[0]["content"][0]["text"] + "</user><assistant>"

    def __call__(self, *, text, return_tensors, add_special_tokens):
        assert return_tensors == "pt" and add_special_tokens is False
        ids = torch.tensor([self.tokenizer.encode(text)])
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": torch.zeros_like(ids),
            "token_type_ids": torch.full_like(ids, 7),
        }


@pytest.fixture
def model():
    torch.manual_seed(94)
    config = transformers.Gemma4UnifiedConfig(
        text_config={
            "vocab_size": 256,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "layer_types": ["sliding_attention", "full_attention"] * 2,
            "num_kv_shared_layers": 2,
            "sliding_window": 64,
            "max_position_embeddings": 8192,
            "final_logit_softcapping": 3.0,
        },
        vision_config={
            "patch_size": 2,
            "pooling_kernel_size": 1,
            "mm_embed_dim": 8,
            "output_proj_dims": 8,
            "mm_posemb_size": 4,
        },
        audio_config={"audio_embed_dim": 8},
        image_token_id=250,
        audio_token_id=251,
        video_token_id=252,
    )
    lm = transformers.Gemma4UnifiedForConditionalGeneration(config).eval()
    return UnifiedDecisionModel.from_components(lm, Processor(), temperature=2.5)


def prepared_rows(model, mode="user"):
    backend = BrowserDecisionBackend(model, prompt_mode=mode)
    state = {
        "page": {"title": "Observed browser page", "text": "A common visible record. " * 5},
        "elements": [{"index": "1", "label": "Observed control"}],
        "recent_actions": [],
    }
    rows = []
    for index, options in enumerate([2, 5, 52]):
        request = DecisionRequest(
            state=state,
            questions=[
                {
                    "id": f"head-{index}",
                    "instructions": json.dumps(
                        {
                            "goal": "Open the requested record after applying the search",
                            "stage": index,
                        }
                    ),
                    "criteria": {
                        f"item-{option}": "Observed target " + str(option)
                        for option in range(options)
                    },
                }
            ],
        )
        inputs, length = backend._prepare_question(request, request.questions[0])
        inputs["position_ids"] = torch.arange(length).unsqueeze(0)
        rows.append(
            {
                "request": request,
                "prepared": (inputs, torch.tensor([length - 1]), torch.tensor([options])),
            }
        )
    return rows


def full_answers(model, rows):
    output = []
    with torch.inference_mode():
        for row in rows:
            inputs, slots, counts = row["prepared"]
            result = model.backbone(**inputs, use_cache=False, return_dict=True)
            hidden = result.last_hidden_state[0, slots]
            logits = project_candidates(hidden, model.head, model.letters, counts, model.softcap)
            probabilities = (logits / model.temperature).softmax(-1).cpu().tolist()[0]
            question = row["request"].questions[0]
            output.append(
                answer_from_probabilities(question, probabilities[: len(question.labels())])
            )
    return output


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["user", "grounded", "goal-first"])
def test_real_gemma_prefix_matches_full_context_across_sliding_and_shared_kv_layers(
    model, dtype, mode
):
    model.lm.to(dtype=dtype)
    rows = prepared_rows(model, mode)
    original_requests = [row["request"].model_dump() for row in rows]
    original_tensors = [
        {key: value.clone() for key, value in row["prepared"][0].items()} for row in rows
    ]
    expected = full_answers(model, rows)
    calls, prefix = [], {}

    def inspect_forward(_module, _args, kwargs, output):
        assert kwargs["use_cache"] is True
        calls.append(
            {
                key: kwargs[key].clone()
                for key in [
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "mm_token_type_ids",
                    "token_type_ids",
                ]
            }
        )
        cache = output.past_key_values
        if kwargs["past_key_values"] is None:
            prefix["cache"] = cache
            prefix["tensors"] = [
                (layer.keys.clone(), layer.values.clone()) for layer in cache.layers
            ]
        else:
            assert cache is not prefix["cache"]
            for layer, original_layer in zip(cache.layers, prefix["cache"].layers, strict=True):
                assert layer.keys.data_ptr() != original_layer.keys.data_ptr()

    hook = model.backbone.register_forward_hook(inspect_forward, with_kwargs=True)
    try:
        actual = predict_prefix(model, rows, prompt_mode=mode)
    finally:
        hook.remove()
    # Splitting a genuine SDPA forward changes floating-point reduction shapes;
    # compare within dtype precision while requiring the same winning label.
    tolerance = 2e-3 if dtype == torch.bfloat16 else 1e-5
    for row, response, reference in zip(rows, actual, expected, strict=True):
        question = row["request"].questions[0]
        answer = response["answers"][question.id]
        assert answer["probabilities"] == pytest.approx(reference["probabilities"], abs=tolerance)
        assert answer["choice"] == reference["choice"]
        assert list(answer["probabilities"]) == question.labels()
    trace = actual[0]["prefix"]
    assert trace["status"] == "completed"
    p = trace["prefix_tokens"]
    assert p > 64
    assert all(suffix > 64 for suffix in trace["suffix_tokens"])
    assert model.backbone.config.text_config.num_kv_shared_layers == 2
    assert len(trace["cache_layers"]) == 2
    assert trace["cache_layers"][0]["is_sliding"] is True
    assert trace["cache_layers"][0]["key_shape"][-2] == 63
    assert (
        trace["cache_layers"][0]["key_storage_bytes"]
        >= trace["cache_layers"][0]["key_tensor_bytes"]
    )
    assert trace["cache_layers"][1]["is_sliding"] is False
    assert prefix["cache"].get_seq_length() == p
    for layer, (keys, values) in zip(prefix["cache"].layers, prefix["tensors"], strict=True):
        torch.testing.assert_close(layer.keys, keys, atol=0, rtol=0)
        torch.testing.assert_close(layer.values, values, atol=0, rtol=0)
    assert len(calls) == trace["forward_calls"] == 4
    forwards_by_ref = {
        (item["chunk_id"], item["forward_index"]): item for item in trace["forwards"]
    }
    for response in actual:
        for ref in response["execution"]["forward_refs"]:
            assert (ref["chunk_id"], ref["forward_index"]) in forwards_by_ref
    assert sum(response["execution"]["forward_calls"] for response in actual) == 4
    assert (
        sum(response["usage"]["input_tokens"] for response in actual)
        == trace["computed_input_tokens"]
    )
    assert trace["computed_input_tokens"] == p + sum(trace["suffix_tokens"])
    assert trace["full_context_input_tokens"] == sum(trace["context_tokens"])
    assert trace["computed_input_tokens"] < trace["full_context_input_tokens"]
    assert len({response["execution"]["batch_id"] for response in actual}) == 1
    assert len({response["latency_ms"] for response in actual}) == 1
    assert all("prefix" not in response for response in actual[1:])
    assert [row["request"].model_dump() for row in rows] == original_requests
    for index, row in enumerate(rows):
        inputs = row["prepared"][0]
        length = inputs["input_ids"].shape[1]
        assert calls[index + 1]["input_ids"].shape[1] == length - p
        torch.testing.assert_close(
            calls[index + 1]["attention_mask"], inputs["attention_mask"], atol=0, rtol=0
        )
        assert calls[index + 1]["position_ids"].tolist() == [list(range(p, length))]
        for key in ["input_ids", "mm_token_type_ids", "token_type_ids"]:
            torch.testing.assert_close(calls[index + 1][key], inputs[key][:, p:], atol=0, rtol=0)
        for key, value in inputs.items():
            torch.testing.assert_close(value, original_tensors[index][key], atol=0, rtol=0)
    json.dumps(trace, allow_nan=False)


def test_branch_order_cannot_contaminate_independent_cache_answers(model):
    rows = prepared_rows(model)
    original = predict_prefix(model, rows, prompt_mode="user")
    reversed_results = predict_prefix(model, list(reversed(rows)), prompt_mode="user")
    for first, reverse in zip(original, reversed(reversed_results), strict=True):
        assert list(first["answers"]) == list(reverse["answers"])
        question_id = next(iter(first["answers"]))
        assert first["answers"][question_id]["probabilities"] == pytest.approx(
            reverse["answers"][question_id]["probabilities"], abs=1e-6
        )


def test_all_sequence_tensor_keys_limit_prefix_and_identical_rows_keep_answer_in_suffix(model):
    first = prepared_rows(model)[0]
    second = copy.deepcopy(first)
    second["prepared"][0]["token_type_ids"][0, 73] = 9
    result = predict_prefix(model, [first, second], prompt_mode="user")
    assert result[0]["prefix"]["prefix_tokens"] == 73
    exact = predict_prefix(model, [first, copy.deepcopy(first)], prompt_mode="user")
    length = first["prepared"][0]["input_ids"].shape[1]
    assert exact[0]["prefix"]["prefix_tokens"] == length - 1
    assert exact[0]["prefix"]["suffix_tokens"] == [1, 1]


def test_right_padding_preserves_full_masks_and_original_last_true_readout(model):
    from torch.nn.functional import pad

    rows = prepared_rows(model)
    for index, row in enumerate(rows):
        inputs = row["prepared"][0]
        padding = index + 3
        length = inputs["input_ids"].shape[1] + padding
        for key in inputs:
            inputs[key] = pad(inputs[key], (0, padding), value=0)
        inputs["position_ids"] = torch.arange(length).unsqueeze(0)
    expected = full_answers(model, rows)
    actual = predict_prefix(model, rows, prompt_mode="user")
    for row, response, reference in zip(rows, actual, expected, strict=True):
        question_id = row["request"].questions[0].id
        assert response["answers"][question_id]["probabilities"] == pytest.approx(
            reference["probabilities"], abs=1e-5
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "media",
        "mask_gap",
        "mask_value",
        "keys",
        "count",
        "slot",
        "positions",
        "budget",
        "modality",
        "vocabulary",
    ],
)
def test_every_original_row_is_validated_on_cpu_before_any_prefill(model, corruption):
    rows = prepared_rows(model)
    row = rows[-1]
    inputs, slots, counts = row["prepared"]
    if corruption == "media":
        row["request"].media = [ImageInput(data="not-decoded")]
    elif corruption == "mask_gap":
        inputs["attention_mask"][0, 4] = 0
    elif corruption == "mask_value":
        inputs["attention_mask"][0, 4] = 2
    elif corruption == "keys":
        del inputs["token_type_ids"]
    elif corruption == "count":
        counts[0] -= 1
    elif corruption == "slot":
        slots[0] -= 1
    elif corruption == "positions":
        inputs["position_ids"][0, 3] = 7
    elif corruption == "budget":
        model.max_context = inputs["input_ids"].shape[1] - 1
    elif corruption == "modality":
        inputs["mm_token_type_ids"][0, 3] = 1
    else:
        inputs["input_ids"][0, -2] = model.backbone.config.text_config.vocab_size
    calls = []
    hook = model.backbone.register_forward_hook(lambda *_args: calls.append(None))
    try:
        with pytest.raises(RequestValidationError) as error:
            predict_prefix(model, rows, prompt_mode="user")
    finally:
        hook.remove()
    assert calls == []
    trace = error.value.prefix_evidence
    assert trace is error.value.batch_evidence
    assert trace["forwards"] == []
    assert trace["completed_response_indices"] == []
    assert trace["validation_row_index"] == 2


@pytest.mark.parametrize("mode", ["native", "reasoned", "unknown"])
def test_unsupported_modes_reject_explicitly_without_changing_input_or_model(model, mode):
    rows = prepared_rows(model)
    with pytest.raises(BrowserPrefixUnsupported, match="user browser modes") as error:
        predict_prefix(model, rows, prompt_mode=mode)
    assert error.value.prefix_evidence["forwards"] == []


def test_readout_callback_receives_actual_hidden_counts_and_native_softcap(model):
    rows = prepared_rows(model)
    default = predict_prefix(model, rows, prompt_mode="user")
    calls = []

    def explicit_readout(hidden, counts, softcap):
        calls.append((tuple(hidden.shape), counts.tolist(), softcap))
        return project_candidates(hidden, model.head, model.letters, counts, softcap)

    actual = predict_prefix(model, rows, prompt_mode="user", readout=explicit_readout)
    assert [counts for _shape, counts, _softcap in calls] == [[2], [5], [52]]
    assert all(shape == (1, 16) and softcap == model.softcap for shape, _counts, softcap in calls)
    for original, result in zip(default, actual, strict=True):
        question_id = next(iter(original["answers"]))
        assert original["answers"][question_id]["probabilities"] == pytest.approx(
            result["answers"][question_id]["probabilities"], abs=1e-6
        )


def test_readout_cannot_quietly_replace_unavailable_candidate_mask(model):
    rows = prepared_rows(model)
    with pytest.raises(BackendResponseError, match="masked candidate") as error:
        predict_prefix(model, rows, prompt_mode="user", readout=lambda *_args: torch.zeros((1, 52)))
    assert error.value.prefix_evidence["completed_response_indices"] == []
    assert len(error.value.prefix_evidence["forwards"]) == 2


def test_suffix_failure_preserves_genuine_completed_answers_and_failed_stage_inputs(model):
    rows = prepared_rows(model)
    expected = full_answers(model, rows)
    calls = []

    def fail_second_suffix(_module, _args, _kwargs):
        calls.append(None)
        if len(calls) == 3:
            raise RuntimeError("failed suffix branch")

    hook = model.backbone.register_forward_pre_hook(fail_second_suffix, with_kwargs=True)
    try:
        with pytest.raises(RuntimeError, match="failed suffix branch") as error:
            predict_prefix(model, rows, prompt_mode="user")
    finally:
        hook.remove()
    trace = error.value.prefix_evidence
    assert trace["completed_response_indices"] == [0]
    question_id = rows[0]["request"].questions[0].id
    assert trace["completed_responses"][0]["answers"][question_id][
        "probabilities"
    ] == pytest.approx(expected[0]["probabilities"], abs=1e-6)
    assert [forward["status"] for forward in trace["forwards"]] == [
        "returned",
        "returned",
        "failed",
    ]
    assert trace["rows"][1]["request"] == rows[1]["request"].model_dump(exclude_none=True)
    assert trace["forwards"][-1]["row_index"] == 1
    assert trace["forward_calls"] == 3
    assert trace["elapsed_ms"] > 0
    assert trace["branches"][-1]["status"] == "failed"
    assert trace["branches"][-1]["failure_stage"] == "forwarding"
    json.dumps(trace, allow_nan=False)


def test_noncausal_or_unsupported_backbone_requires_explicit_fallback_decision(model):
    rows = prepared_rows(model)
    model.backbone.config.text_config.use_bidirectional_attention = "all"
    with pytest.raises(BrowserPrefixUnsupported, match="bidirectional") as error:
        predict_prefix(model, rows, prompt_mode="user")
    assert error.value.prefix_evidence["forwards"] == []
    model.backbone = SimpleNamespace()
    with pytest.raises(BrowserPrefixUnsupported, match="native Gemma4Unified") as error:
        predict_prefix(model, rows, prompt_mode="user")
    assert error.value.prefix_evidence["forwards"] == []


def test_length_dependent_rotary_modes_fail_before_any_prefix_forward(model):
    rows = prepared_rows(model)
    model.backbone.config.text_config.rope_parameters["full_attention"]["rope_type"] = "dynamic"
    with pytest.raises(BrowserPrefixUnsupported, match="fixed default/proportional") as error:
        predict_prefix(model, rows, prompt_mode="user")
    assert error.value.prefix_evidence["forwards"] == []


def compact_rows(model, *, prefix=70, suffixes=(6, 80, 9), padding=(2, 5, 0)):
    """Small genuine text tensors with mixed lengths and known causal overlap."""
    rows = []
    for index, (suffix, extra, count) in enumerate(zip(suffixes, padding, (2, 52, 5), strict=True)):
        request = DecisionRequest(
            state={"page": {"text": "Observed controls"}},
            questions=[
                {
                    "id": f"row-{index}",
                    "instructions": "Choose an observed control",
                    "criteria": {f"node-{n}": f"Control {n}" for n in range(count)},
                }
            ],
        )
        ids = torch.cat(
            (
                torch.arange(prefix) % 29 + 3,
                torch.full((suffix,), 40 + index),
                torch.zeros(extra, dtype=torch.long),
            )
        ).unsqueeze(0)
        mask = torch.ones_like(ids)
        if extra:
            mask[:, -extra:] = 0
        inputs = {
            "input_ids": ids,
            "attention_mask": mask,
            "mm_token_type_ids": torch.zeros_like(ids),
            "token_type_ids": torch.full_like(ids, 7),
            "position_ids": torch.arange(ids.shape[1]).unsqueeze(0),
        }
        rows.append(
            {
                "request": request,
                "prepared": (inputs, torch.tensor([prefix + suffix - 1]), torch.tensor([count])),
            }
        )
    return rows


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("prefix,suffixes", [(15, (4, 70, 9)), (70, (6, 80, 9))])
def test_two_suffix_rows_match_full_and_single_cache_with_right_padding_and_shared_kv(
    model, dtype, prefix, suffixes
):
    from transformers.cache_utils import DynamicCache

    model.lm.to(dtype=dtype)
    rows = compact_rows(model, prefix=prefix, suffixes=suffixes)
    original = [{key: value.clone() for key, value in row["prepared"][0].items()} for row in rows]
    full = full_answers(model, rows)
    sequential = predict_prefix(model, rows, prompt_mode="grounded")
    calls, shared, repeats = [], {}, []
    native_repeat = DynamicCache.batch_repeat_interleave

    def repeat(cache, count):
        repeats.append((cache, count, cache.get_seq_length()))
        return native_repeat(cache, count)

    def before(_module, _args, kwargs):
        calls.append({key: kwargs[key].clone() for key in rows[0]["prepared"][0]})
        fork = kwargs["past_key_values"]
        if fork is not None:
            assert fork is not shared["cache"] and fork.get_seq_length() == prefix
            for layer, saved in zip(fork.layers, shared["layers"], strict=True):
                assert layer.keys.shape[0] == kwargs["input_ids"].shape[0]
                for batch_index in range(layer.keys.shape[0]):
                    torch.testing.assert_close(layer.keys[batch_index], saved[0][0], atol=0, rtol=0)

    def after(_module, _args, kwargs, output):
        if kwargs["past_key_values"] is None:
            shared["cache"] = output.past_key_values
            shared["layers"] = [
                (layer.keys.clone(), layer.values.clone(), layer.get_seq_length())
                for layer in output.past_key_values.layers
            ]

    from unittest.mock import patch

    pre_hook = model.backbone.register_forward_pre_hook(before, with_kwargs=True)
    post_hook = model.backbone.register_forward_hook(after, with_kwargs=True)
    try:
        with patch.object(DynamicCache, "batch_repeat_interleave", repeat):
            actual = predict_prefix(model, rows, prompt_mode="grounded", max_suffix_batch_size=2)
    finally:
        pre_hook.remove()
        post_hook.remove()
    tolerance = 2e-3 if dtype == torch.bfloat16 else 1e-5
    for row, response, one, reference in zip(rows, actual, sequential, full, strict=True):
        question_id = row["request"].questions[0].id
        answer = response["answers"][question_id]
        assert answer["probabilities"] == pytest.approx(reference["probabilities"], abs=tolerance)
        assert answer["probabilities"] == pytest.approx(
            one["answers"][question_id]["probabilities"], abs=tolerance
        )
        assert answer["choice"] == reference["choice"] == one["answers"][question_id]["choice"]
        assert list(answer["probabilities"]) == row["request"].questions[0].labels()
    assert len(repeats) == 1 and repeats[0][1:] == (2, prefix)
    assert repeats[0][0] is not shared["cache"]
    assert shared["cache"].get_seq_length() == prefix
    for layer, (keys, values, logical) in zip(
        shared["cache"].layers, shared["layers"], strict=True
    ):
        assert layer.get_seq_length() == logical
        torch.testing.assert_close(layer.keys, keys, atol=0, rtol=0)
        torch.testing.assert_close(layer.values, values, atol=0, rtol=0)
    trace = actual[0]["prefix"]
    assert len(calls) == trace["forward_calls"] == 3
    assert [call["input_ids"].shape[0] for call in calls] == [1, 2, 1]
    assert trace["max_suffix_batch_size"] == trace["max_forward_batch_size"] == 2
    assert [response["passes"] for response in actual] == [2, 0, 1]
    assert sum(response["execution"]["forward_calls"] for response in actual) == 3
    assert [forward["row_indices"] for forward in trace["forwards"]] == [[], [0, 1], [2]]
    physical = sum(call["input_ids"].numel() for call in calls)
    logical = prefix + sum(row["prepared"][0]["input_ids"].shape[1] - prefix for row in rows)
    assert (
        trace["computed_input_tokens"]
        == physical
        == sum(response["usage"]["input_tokens"] for response in actual)
    )
    assert trace["logical_computed_input_tokens"] == logical
    assert trace["padding_input_tokens"] == physical - logical > 0
    assert trace["full_context_input_tokens"] == sum(
        row["prepared"][0]["input_ids"].shape[1] for row in rows
    )
    for offset, indices in enumerate(((0, 1), (2,)), 1):
        call = calls[offset]
        width = call["input_ids"].shape[1]
        assert call["position_ids"].tolist() == [list(range(prefix, prefix + width))] * len(indices)
        for batch_index, index in enumerate(indices):
            inputs = original[index]
            suffix = inputs["input_ids"].shape[1] - prefix
            for key in ("input_ids", "mm_token_type_ids", "token_type_ids"):
                torch.testing.assert_close(
                    call[key][batch_index, :suffix], inputs[key][0, prefix:], atol=0, rtol=0
                )
            torch.testing.assert_close(
                call["attention_mask"][batch_index, : inputs["attention_mask"].shape[1]],
                inputs["attention_mask"][0],
                atol=0,
                rtol=0,
            )
            assert not call["attention_mask"][
                batch_index, inputs["attention_mask"].shape[1] :
            ].any()
        branch = trace["branches"][offset - 1]
        assert branch["cache_logical_tokens_after_forward"] == prefix + width
        assert all(
            layer["logical_tokens"] == prefix + width
            for layer in branch["cache_layers_after_forward"]
        )
    for row, saved in zip(rows, original, strict=True):
        for key, value in row["prepared"][0].items():
            torch.testing.assert_close(value, saved[key], atol=0, rtol=0)
    json.dumps(actual, allow_nan=False)


@pytest.mark.parametrize("size", [True, False, None, 0, 3, 1.0, "2", float("nan")])
def test_suffix_batch_size_is_strict_before_any_model_forward(model, size):
    calls = []
    hook = model.backbone.register_forward_pre_hook(lambda *_args: calls.append(None))
    try:
        with pytest.raises(BrowserPrefixUnsupported, match="integer 1 or 2") as error:
            predict_prefix(
                model, compact_rows(model), prompt_mode="user", max_suffix_batch_size=size
            )
    finally:
        hook.remove()
    assert calls == [] and error.value.prefix_evidence["forwards"] == []
    json.dumps(error.value.prefix_evidence, allow_nan=False)


def test_two_row_readout_callback_scores_every_candidate_with_actual_counts(model):
    rows = compact_rows(model)
    calls = []

    def readout(hidden, counts, softcap):
        calls.append((tuple(hidden.shape), counts.tolist(), softcap))
        return project_candidates(hidden, model.head, model.letters, counts, softcap)

    reference = predict_prefix(model, rows, prompt_mode="user", max_suffix_batch_size=2)
    actual = predict_prefix(
        model, rows, prompt_mode="user", readout=readout, max_suffix_batch_size=2
    )
    assert calls == [((2, 16), [2, 52], model.softcap), ((1, 16), [5], model.softcap)]
    assert [response["answers"] for response in actual] == [
        response["answers"] for response in reference
    ]


@pytest.mark.parametrize(
    "corruption", ["media", "mask", "budget", "positions", "count", "padding_token"]
)
def test_two_row_mode_validates_every_original_context_before_prefill(model, corruption):
    rows = compact_rows(model)
    inputs, _slots, counts = rows[-1]["prepared"]
    if corruption == "media":
        rows[-1]["request"].media = [ImageInput(data="unread")]
    elif corruption == "mask":
        inputs["attention_mask"][0, 4] = 0
    elif corruption == "budget":
        model.max_context = rows[0]["prepared"][0]["input_ids"].shape[1]
    elif corruption == "positions":
        inputs["position_ids"][0, -1] += 2
    elif corruption == "count":
        counts[0] = 51
    else:
        model.backbone.config.text_config.pad_token_id = -1
    calls = []
    hook = model.backbone.register_forward_pre_hook(lambda *_args: calls.append(None))
    try:
        with pytest.raises(RequestValidationError) as error:
            predict_prefix(model, rows, prompt_mode="user", max_suffix_batch_size=2)
    finally:
        hook.remove()
    assert calls == [] and error.value.prefix_evidence["forwards"] == []


def test_two_row_cache_order_cannot_contaminate_answers(model):
    rows = compact_rows(model)
    forward = predict_prefix(model, rows, prompt_mode="user", max_suffix_batch_size=2)
    reverse = predict_prefix(
        model, list(reversed(rows)), prompt_mode="user", max_suffix_batch_size=2
    )
    for first, last in zip(forward, reversed(reverse), strict=True):
        question_id = next(iter(first["answers"]))
        assert first["answers"][question_id]["probabilities"] == pytest.approx(
            last["answers"][question_id]["probabilities"], abs=1e-5
        )


def test_failed_second_suffix_chunk_retains_both_completed_real_rows(model):
    rows = compact_rows(model)
    calls = []

    def fail(_module, _args, _kwargs):
        calls.append(None)
        if len(calls) == 3:
            raise RuntimeError("second suffix chunk failed")

    hook = model.backbone.register_forward_pre_hook(fail, with_kwargs=True)
    try:
        with pytest.raises(RuntimeError, match="second suffix chunk") as error:
            predict_prefix(model, rows, prompt_mode="user", max_suffix_batch_size=2)
    finally:
        hook.remove()
    trace = error.value.prefix_evidence
    assert trace["completed_response_indices"] == [0, 1]
    assert len(trace["completed_responses"]) == 2
    assert trace["forwards"][-1]["row_indices"] == [2]
    assert trace["branches"][-1]["failure_stage"] == "forwarding"
    assert len(trace["branches"][0]["raw_probabilities"][1]["probabilities"]) == 52
    assert trace["forwarded_input_tokens"] == sum(
        item["input_tokens"] for item in trace["forwards"]
    )
    assert trace["completed_computed_input_tokens"] == sum(
        item["input_tokens"] for item in trace["forwards"] if item["status"] == "returned"
    )
    json.dumps(trace, allow_nan=False)


def test_partial_answer_failure_keeps_completed_row_and_all_raw_chunk_distributions(
    model, monkeypatch
):
    import s1.browser_prefix as prefix_module

    rows = compact_rows(model)
    native_answer = prefix_module.answer_from_probabilities
    called = []

    def interrupted(question, probabilities):
        called.append(question.id)
        if len(called) == 2:
            raise RuntimeError("second row formatting failed")
        return native_answer(question, probabilities)

    monkeypatch.setattr(prefix_module, "answer_from_probabilities", interrupted)
    with pytest.raises(RuntimeError, match="second row formatting") as caught:
        predict_prefix(model, rows, prompt_mode="user", max_suffix_batch_size=2)
    evidence = caught.value.prefix_evidence
    assert evidence["completed_response_indices"] == [0]
    assert len(evidence["completed_responses"]) == 1
    assert evidence["branches"][-1]["failure_stage"] == "readout"
    assert [item["row_index"] for item in evidence["branches"][0]["raw_probabilities"]] == [0, 1]
    assert len(evidence["branches"][0]["raw_probabilities"][1]["probabilities"]) == 52
    assert evidence["completed_computed_input_tokens"] == sum(
        item["input_tokens"] for item in evidence["forwards"]
    )
    json.dumps(evidence, allow_nan=False)


def test_batched_evidence_preserves_rows_with_the_same_question_id(model):
    rows = compact_rows(model)[:2]
    for row in rows:
        question = row["request"].questions[0].model_copy(update={"id": "shared-head"})
        row["request"] = row["request"].model_copy(update={"questions": [question]})
    actual = predict_prefix(model, rows, prompt_mode="user", max_suffix_batch_size=2)
    saved = actual[0]["prefix"]["branches"][0]["row_answers"]
    assert [item["row_index"] for item in saved] == [0, 1]
    assert [item["answers"] for item in saved] == [response["answers"] for response in actual]
    assert len(saved[0]["answers"]["shared-head"]["probabilities"]) == 2
    assert len(saved[1]["answers"]["shared-head"]["probabilities"]) == 52


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batched_cache_counters_and_layers_match_native_padded_single_suffixes(model, dtype):
    model.lm.to(dtype=dtype)
    rows = compact_rows(model)
    shared, branches = {}, []

    def after(_module, _args, kwargs, output):
        if kwargs["past_key_values"] is None:
            shared["cache"] = output.past_key_values
        else:
            branches.append(
                (
                    {key: kwargs[key].clone() for key in rows[0]["prepared"][0]},
                    [
                        (layer.keys.clone(), layer.values.clone(), layer.get_seq_length())
                        for layer in output.past_key_values.layers
                    ],
                )
            )

    hook = model.backbone.register_forward_hook(after, with_kwargs=True)
    try:
        predict_prefix(model, rows, prompt_mode="user", max_suffix_batch_size=2)
    finally:
        hook.remove()
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-5
    with torch.inference_mode():
        for inputs, layers in branches:
            for batch_index in range(inputs["input_ids"].shape[0]):
                native = model.backbone(
                    **{key: value[batch_index : batch_index + 1] for key, value in inputs.items()},
                    past_key_values=copy.deepcopy(shared["cache"]),
                    use_cache=True,
                    return_dict=True,
                ).past_key_values
                for expected, (keys, values, logical) in zip(native.layers, layers, strict=True):
                    # Native caches advance by physical padded query width;
                    # original per-row context lengths remain separate evidence.
                    assert expected.get_seq_length() == logical
                    assert expected.keys.shape[1:] == keys.shape[1:]
                    torch.testing.assert_close(
                        expected.keys[0], keys[batch_index], atol=tolerance, rtol=tolerance
                    )
                    torch.testing.assert_close(
                        expected.values[0], values[batch_index], atol=tolerance, rtol=tolerance
                    )
