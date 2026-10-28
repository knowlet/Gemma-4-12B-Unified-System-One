"""Real tiny native Unified parity for the explicit prefix-cache experiment."""

# ruff: noqa: E402

from __future__ import annotations

import base64
import copy
import io

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
import test_unified as native_fixtures
from PIL import Image

from s1.contracts import AudioInput, ImageInput
from s1.errors import RequestValidationError
from s1.evaluation.gemma import predict_batch, predict_sequential
from s1.evaluation.prefix import predict_independent_prefix
from s1.schema import LETTERS

pytestmark = pytest.mark.inference


@pytest.fixture
def model():
    return native_fixtures.model.__wrapped__()


def assert_answers_close(actual, reference, *, tolerance=1e-5):
    assert list(actual["answers"]) == list(reference["answers"])
    for name, answer in actual["answers"].items():
        expected = reference["answers"][name]
        assert list(answer["probabilities"]) == list(expected["probabilities"])
        torch.testing.assert_close(
            torch.tensor(list(answer["probabilities"].values())),
            torch.tensor(list(expected["probabilities"].values())),
            atol=tolerance,
            rtol=tolerance,
        )
        label = (
            "choice"
            if answer["type"] == "choice"
            else "level"
            if answer["type"] == "score"
            else None
        )
        if label:
            assert answer[label] == expected[label]


def add_media(request, modality):
    if modality in ("image", "mixed"):
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), "blue").save(buffer, format="PNG")
        request.media.append(ImageInput(data=base64.b64encode(buffer.getvalue()).decode()))
    if modality in ("audio", "mixed"):
        request.media.append(AudioInput(samples=[0.2] * 16))


@pytest.mark.parametrize("modality", ["text", "image", "audio", "mixed"])
def test_split_reassembly_is_exactly_the_original_prepare_contract(
    model, decision_request, modality
):
    add_media(decision_request, modality)
    prefix = model.prepare_prefix(decision_request)
    length = prefix["input_ids"].shape[1]
    suffix, slots, counts = model.prepare_suffix(decision_request.questions, length)
    inputs, actual_slots, actual_counts = model.prepare(decision_request)
    # Independently reproduce the pre-refactor algorithm, including text bytes,
    # original processor tensors and the exact answer-slot positions.
    legacy = dict(
        model.processor(
            **{
                "text": "state",
                "return_tensors": "pt",
                "add_special_tokens": False,
                **({"images": [None]} if modality in ("image", "mixed") else {}),
                **(
                    {"audio": [None], "sampling_rate": 16000}
                    if modality in ("audio", "mixed")
                    else {}
                ),
            }
        )
    )
    old_suffix, old_slots, old_counts = [], [], []
    for question in decision_request.questions:
        lines = [f"\nQuestion ({question.type}): {question.instructions}\n"]
        for letter, label, description in zip(LETTERS, question.labels(), question.descriptions()):
            lines.append(f"({letter}) {label}: {description}\n")
        lines.append("Answer: (")
        old_suffix.extend(model.tok.encode("".join(lines), add_special_tokens=False))
        old_slots.append(length + len(old_suffix) - 1)
        old_counts.append(len(question.labels()))
    legacy["input_ids"] = torch.cat([legacy["input_ids"], torch.tensor([old_suffix])], dim=1)
    for key, fill in (("attention_mask", 1), ("mm_token_type_ids", 0), ("token_type_ids", 0)):
        if key in legacy:
            legacy[key] = torch.cat(
                [legacy[key], legacy[key].new_full((1, len(old_suffix)), fill)], dim=1
            )
    assert set(legacy) == set(inputs) == set(prefix)
    for name in legacy:
        torch.testing.assert_close(inputs[name], legacy[name], rtol=0, atol=0)
    torch.testing.assert_close(suffix, torch.tensor([old_suffix]), rtol=0, atol=0)
    assert actual_slots.tolist() == slots.tolist() == old_slots
    assert actual_counts.tolist() == counts.tolist() == old_counts


@pytest.mark.parametrize("modality", ["text", "image", "audio", "mixed"])
def test_real_unified_prefix_matches_independent_references(model, decision_request, modality):
    add_media(decision_request, modality)
    sequential = predict_sequential(model, decision_request)
    batched = predict_batch(model, [decision_request])[0]
    calls = []
    hook = model.backbone.register_forward_hook(
        lambda _module, _args, output: calls.append(output.past_key_values.get_seq_length())
    )
    try:
        actual = predict_independent_prefix(model, [decision_request])[0]
    finally:
        hook.remove()
    assert len(calls) == 2
    assert_answers_close(actual, sequential)
    assert_answers_close(actual, batched)
    execution = actual["execution"]
    assert execution["forward_calls"] == 2
    assert execution["prefix_prefill_calls"] == execution["suffix_forward_calls"] == 1
    assert execution["batch_sizes"] == [1, 3]
    assert execution["cached_prefix_tokens"] == [calls[0]]


def test_longer_than_sliding_window_prefix_metadata_survives_branch_copy(
    model, decision_request, monkeypatch
):
    processor = model.processor

    class LongPrefixProcessor(native_fixtures.Processor):
        tokenizer = processor.tokenizer

        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            for key, fill in (("input_ids", 10), ("attention_mask", 1), ("mm_token_type_ids", 0)):
                result[key] = torch.cat([result[key], result[key].new_full((1, 90), fill)], dim=1)
            return result

    monkeypatch.setattr(model, "processor", LongPrefixProcessor())
    # Nine differently sized heads require two branches of one prefix cache.
    questions = []
    for index in range(9):
        question = decision_request.questions[index % 3].model_copy(deep=True)
        question.id = f"q{index}"
        question.instructions += " extra" * index
        questions.append(question)
    decision_request.questions = questions
    reference = predict_batch(model, [decision_request])[0]
    actual = predict_independent_prefix(model, [decision_request])[0]
    assert actual["execution"]["cached_prefix_tokens"] == [94]
    assert actual["execution"]["batch_sizes"] == [1, 8, 1]
    assert actual["execution"]["suffix_forward_calls"] == 2
    assert_answers_close(actual, reference)
    reordered = decision_request.model_copy(deep=True)
    reordered.questions.reverse()
    repeated = predict_independent_prefix(model, [reordered, decision_request])
    assert list(repeated[0]["answers"]) == [question.id for question in reordered.questions]
    for name, answer in actual["answers"].items():
        assert_answers_close(
            {"answers": {name: repeated[0]["answers"][name]}}, {"answers": {name: answer}}
        )
    assert_answers_close(repeated[1], actual)
    assert repeated[1]["execution"]["cached_prefix_tokens"] == [94, 94]


def test_native_media_tensors_only_enter_complete_prefix_backbone(model, decision_request):
    add_media(decision_request, "mixed")
    calls = []

    def before(_module, args, kwargs):
        calls.append({key: value for key, value in kwargs.items() if key != "past_key_values"})

    hook = model.backbone.register_forward_pre_hook(before, with_kwargs=True)
    try:
        predict_independent_prefix(
            model, [decision_request, decision_request.model_copy(deep=True)]
        )
    finally:
        hook.remove()
    assert len(calls) == 4
    for prefix, suffix in (calls[:2], calls[2:]):
        assert {
            "pixel_values",
            "image_position_ids",
            "input_features",
            "input_features_mask",
        } <= set(prefix)
        assert not {
            "pixel_values",
            "input_features",
            "image_position_ids",
            "input_features_mask",
        } & set(suffix)
        assert suffix["input_ids"].shape[0] == 3
        assert suffix["position_ids"][0, 0].item() == prefix["input_ids"].shape[1]
        assert (
            suffix["attention_mask"].shape[1]
            == prefix["input_ids"].shape[1] + suffix["input_ids"].shape[1]
        )
        assert not suffix["mm_token_type_ids"].any()


def test_one_question_explicitly_uses_reference_path(model, decision_request):
    decision_request.questions = decision_request.questions[:1]
    expected = predict_batch(model, [decision_request])[0]
    actual = predict_independent_prefix(model, [decision_request])[0]
    assert actual["answers"] == expected["answers"]
    assert actual["execution"]["prefix_prefill_calls"] == 0
    assert actual["execution"]["single_question_reference_requests"] == 1
    assert actual["execution"]["single_question_reference_calls"] == 1


def test_does_not_mutate_requests_or_reuse_cache_after_weights_change(model, decision_request):
    original = copy.deepcopy(decision_request.model_dump())
    predict_independent_prefix(model, [decision_request])
    with torch.no_grad():
        model.backbone.language_model.layers[0].self_attn.q_proj.weight.add_(0.1)
    actual = predict_independent_prefix(model, [decision_request])[0]
    expected = predict_batch(model, [decision_request])[0]
    assert_answers_close(actual, expected)
    assert decision_request.model_dump() == original


def test_refuses_empty_batches_and_bidirectional_text(model, decision_request):
    with pytest.raises(ValueError, match="empty batch"):
        predict_independent_prefix(model, [])
    model.backbone.config.text_config.use_bidirectional_attention = "all"
    with pytest.raises(ValueError, match="bidirectional text"):
        predict_independent_prefix(model, [decision_request])


def test_later_oversized_suffix_is_rejected_before_any_backbone_call(model, decision_request):
    model.max_context = 256
    decision_request.questions[-1].instructions = "x" * 300
    calls = []
    hook = model.backbone.register_forward_pre_hook(lambda *_args: calls.append(1))
    try:
        with pytest.raises(RequestValidationError, match="context tokens; nothing was truncated"):
            predict_independent_prefix(model, [decision_request])
    finally:
        hook.remove()
    assert calls == []


def test_rejects_nonunified_backbone_before_forward(model, decision_request):
    model.backbone.config.model_type = "gemma4"
    calls = []
    hook = model.backbone.register_forward_pre_hook(lambda *_args: calls.append(1))
    try:
        with pytest.raises(ValueError, match="complete Gemma 4 Unified backbone"):
            predict_independent_prefix(model, [decision_request])
    finally:
        hook.remove()
    assert calls == []
