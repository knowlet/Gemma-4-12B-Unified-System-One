"""Research prompt role boundaries, native media, slots and G4 batching contracts."""

import base64
import copy
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1.contracts import AudioInput, DecisionRequest, ImageInput
from s1.errors import RequestValidationError
from s1.evaluation.gemma import predict_batch
from s1.unified import UnifiedDecisionModel

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")


spec = importlib.util.spec_from_file_location(
    "breakthrough_prompt", Path(__file__).parents[1] / "scripts/breakthrough_prompt.py"
)
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)
pytestmark = pytest.mark.inference


class Tokenizer:
    unk_token_id = 255
    pad_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(character) for character in text]


class Processor:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.messages, self.calls, self.prefixes = [], [], []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        self.messages.append(copy.deepcopy(messages))
        content = "".join(
            entry["text"] if entry["type"] == "text" else f"<{entry['type']}>"
            for entry in messages[0]["content"]
        )
        return f"<user>{content}</user><assistant>"

    def __call__(self, **kwargs):
        assert kwargs["return_tensors"] == "pt" and kwargs["add_special_tokens"] is False
        self.calls.append(kwargs)
        text = kwargs["text"].replace("<image>", chr(250) * 2).replace("<audio>", chr(251) * 3)
        ids = torch.tensor([self.tokenizer.encode(text)])
        types = (ids == 250).long() + (ids == 251).long() * 3
        result = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": types,
            "token_type_ids": types.clone(),
        }
        if "images" in kwargs:
            count = 2 * len(kwargs["images"])
            result["pixel_values"] = torch.arange(count * 4).float().reshape(1, count, 4)
            result["image_position_ids"] = torch.zeros((1, count, 2), dtype=torch.long)
        if "audio" in kwargs:
            count = 3 * len(kwargs["audio"])
            result["input_features"] = torch.ones((1, count, 4))
            result["input_features_mask"] = torch.ones((1, count), dtype=torch.bool)
        self.prefixes.append({key: value.clone() for key, value in result.items()})
        return result


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.last_inputs = None

    def forward(self, **inputs):
        assert inputs.pop("use_cache") is False and inputs.pop("return_dict") is True
        self.last_inputs = inputs
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        batch = ids.shape[0]
        for key in ("pixel_values", "image_position_ids", "input_features", "input_features_mask"):
            if key in inputs:
                assert inputs[key].shape[0] == batch
        # A causal readout depends on real prefix tokens, never right padding.
        total = (ids.float() * mask).cumsum(1) / 1000
        length = mask.float().cumsum(1) / 100
        media = inputs["mm_token_type_ids"].float().cumsum(1) / 10
        hidden = torch.stack([total, length, media, total.sin()], dim=-1) * self.scale
        return SimpleNamespace(last_hidden_state=hidden)


class LM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Backbone()
        self.head = torch.nn.Linear(4, 256, bias=True)
        self.config = SimpleNamespace(
            _name_or_path="research-fixture",
            text_config=SimpleNamespace(final_logit_softcapping=3.0, max_position_embeddings=4096),
        )

    def get_output_embeddings(self):
        return self.head


@pytest.fixture
def model():
    torch.manual_seed(31)
    return UnifiedDecisionModel.from_components(LM(), Processor(), temperature=2.5)


@pytest.fixture
def case_request():
    return DecisionRequest(
        state={"ticket": "A customer asks about an invoice."},
        questions=[
            {
                "id": "route",
                "type": "choice",
                "instructions": "Route the supplied ticket.",
                "criteria": {"billing": "Payment requests", "technical": "Software faults"},
            },
            {
                "id": "rating",
                "type": "score",
                "instructions": "Rate frustration using this rubric, without other questions.",
                "criteria": ["calm", "impatient", "angry"],
            },
            {"id": "refund", "type": "noul", "instructions": "Was a refund requested?"},
        ],
    )


def with_media(case_request, modality):
    case_request = case_request.model_copy(deep=True)
    if modality in ("image", "mixed"):
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), "blue").save(buffer, format="PNG")
        case_request.media.append(
            ImageInput(data=base64.b64encode(buffer.getvalue()).decode(), timestamp_seconds=1.25)
        )
    if modality in ("audio", "mixed"):
        case_request.media.append(AudioInput(samples=[0.2, -0.1, 0.0]))
    return case_request


@pytest.mark.parametrize("modality", ["text", "image", "audio", "mixed"])
def test_questions_are_inside_user_and_readout_is_after_assistant(model, case_request, modality):
    case_request = with_media(case_request, modality).model_copy(
        update={"questions": case_request.questions[:1]}
    )
    proxy = research.wrap_model(model, "user_question")
    inputs, slots, counts = proxy.prepare(case_request)
    messages = model.processor.messages[-1]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    user_text = "".join(entry.get("text", "") for entry in messages[0]["content"])
    assert case_request.questions[0].instructions in user_text
    assert "Question (choice)" in user_text
    assert "billing: Payment requests" in user_text and "technical: Software faults" in user_text
    assert "Answer: (" not in user_text
    ids = inputs["input_ids"][0].tolist()
    rendered = "".join(chr(value) for value in ids)
    assert rendered.split("</user><assistant>")[1] == "\nAnswer: ("
    assert ids[slots[0]] == ord("(") and slots.tolist() == [len(ids) - 1]
    assert counts.tolist() == [2]


def test_user_state_question_separator_survives_native_per_block_trimming(model, case_request):
    class NativeTrimmingProcessor(Processor):
        def apply_chat_template(self, messages, **kwargs):
            # The real Gemma template strips each individual text block before
            # concatenation; a newline in a separate block cannot separate it.
            messages = copy.deepcopy(messages)
            for message in messages:
                for entry in message["content"]:
                    if entry["type"] == "text":
                        entry["text"] = entry["text"].strip()
            return super().apply_chat_template(messages, **kwargs)

    model.processor = NativeTrimmingProcessor()
    model.tok = model.processor.tokenizer
    request = case_request.model_copy(
        update={"state": "Please refund the duplicate.", "questions": case_request.questions[:1]}
    )
    inputs, slots, _ = research.wrap_model(model, "user_question").prepare(request)
    rendered = model.processor.calls[-1]["text"]
    user, assistant = rendered.split("</user><assistant>")
    assert "Please refund the duplicate.\nQuestion (choice):" in user
    assert request.questions[0].instructions in user
    assert ".Question" not in user and assistant == ""
    assert model.processor.messages[-1][0]["role"] == "user"
    assert inputs["input_ids"][0, slots[0]].item() == ord("(")


@pytest.mark.parametrize("modality", ["text", "image", "audio", "mixed"])
def test_native_tensors_and_expanded_media_slots_are_retained(model, case_request, modality):
    case_request = with_media(case_request, modality).model_copy(
        update={"questions": case_request.questions[:1]}
    )
    inputs, slots, _ = research.wrap_model(model, "user_question").prepare(case_request)
    prefix = model.processor.prefixes[-1]
    count = inputs["input_ids"].shape[1] - prefix["input_ids"].shape[1]
    assert count == len(model.tok.encode("\nAnswer: ("))
    for key in ("input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"):
        torch.testing.assert_close(inputs[key][:, :-count], prefix[key], atol=0, rtol=0)
    assert inputs["attention_mask"][0, slots].tolist() == [1]
    assert inputs["mm_token_type_ids"][0, slots].tolist() == [0]
    assert inputs["token_type_ids"][0, slots].tolist() == [0]
    for key in set(prefix) - {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}:
        torch.testing.assert_close(inputs[key], prefix[key], atol=0, rtol=0)
    if modality in ("image", "mixed"):
        assert (inputs["input_ids"] == 250).sum().item() == 2
        assert model.processor.calls[-1]["images"][0].getpixel((0, 0)) == (0, 0, 255)
        assert "Frame at 1.25s:" in model.processor.calls[-1]["text"]
    if modality in ("audio", "mixed"):
        assert (inputs["input_ids"] == 251).sum().item() == 3
        assert model.processor.calls[-1]["sampling_rate"] == 16000
        np.testing.assert_array_equal(
            model.processor.calls[-1]["audio"][0], np.asarray([0.2, -0.1, 0.0], dtype=np.float32)
        )


@pytest.mark.parametrize("mode", research.MODES)
def test_instance_view_does_not_rebind_supplied_model_or_change_readout(model, case_request, mode):
    prepare_function, keys = model.prepare.__func__, set(model.__dict__)
    proxy = research.wrap_model(model, mode)
    assert proxy is not model
    assert proxy.lm is model.lm and proxy.backbone is model.backbone
    assert (
        proxy.head is model.head and proxy.processor is model.processor and proxy.tok is model.tok
    )
    assert proxy.letters is model.letters
    assert proxy.softcap == model.softcap and proxy.temperature == model.temperature
    result = research.predict(model, case_request, mode)
    assert result["execution"]["independent_questions"] is True
    assert result["execution"]["research_prompt_mode"] == mode
    assert model.prepare.__func__ is prepare_function and set(model.__dict__) == keys


@pytest.mark.parametrize("modality", ["text", "image", "audio", "mixed"])
def test_independent_batch_matches_sequential_native_head_with_real_masks(
    model, case_request, modality
):
    case_request = with_media(case_request, modality)
    handle = model.head.register_forward_hook(lambda *args: pytest.fail("full LM head called"))
    try:
        actual = research.predict(model, case_request, "user_question")
    finally:
        handle.remove()
    assert actual["execution"]["batch_sizes"] == [3]
    assert actual["execution"]["forward_calls"] == 1
    assert len(set(actual["execution"]["sequence_tokens"])) > 1
    padded = model.backbone.last_inputs
    assert (padded["attention_mask"] == 0).any()
    assert (padded["input_ids"][padded["attention_mask"] == 0] == model.tok.pad_token_id).all()
    assert (padded["mm_token_type_ids"][padded["attention_mask"] == 0] == 0).all()
    proxy = research.wrap_model(model, "user_question")
    for question in case_request.questions:
        part = case_request.model_copy(update={"questions": [question]})
        inputs, slots, _ = proxy.prepare(part)
        hidden = proxy.backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state[
            0, slots
        ]
        # The unchanged full native head provides an independent numerical oracle.
        native = model.head(hidden)[:, model.letters[: len(question.labels())]]
        expected = (torch.tanh(native / model.softcap) * model.softcap / model.temperature).softmax(
            -1
        )
        torch.testing.assert_close(
            torch.tensor(list(actual["answers"][question.id]["probabilities"].values())),
            expected[0],
            atol=1e-7,
            rtol=1e-6,
        )


def test_current_mode_is_existing_independent_helper(model, case_request):
    expected = predict_batch(model, [case_request], independent=True)[0]
    actual = research.predict(model, case_request, "current")
    actual["execution"].pop("research_prompt_mode")
    assert actual == expected


def test_52_candidates_keep_original_semantic_labels(model, case_request):
    case_request = case_request.model_copy(deep=True)
    case_request.questions = case_request.questions[:1]
    case_request.questions[0].criteria = {
        f"label-{index}": "Same description" for index in range(52)
    }
    response = research.predict(model, case_request, "user_question")
    answer = response["answers"]["route"]
    assert list(answer["probabilities"]) == list(case_request.questions[0].criteria)
    assert len(answer["probabilities"]) == 52 and sum(
        answer["probabilities"].values()
    ) == pytest.approx(1)
    assert answer["choice"] in case_request.questions[0].criteria


def test_preparation_rejects_multislot_but_predict_splits_questions(model, case_request):
    with pytest.raises(RequestValidationError, match="one question"):
        research.wrap_model(model, "user_question").prepare(case_request)
    assert len(research.predict(model, case_request, "user_question")["answers"]) == 3


def test_context_limit_accounts_for_complete_user_question_without_truncation(model, case_request):
    case_request = case_request.model_copy(update={"questions": case_request.questions[:1]})
    proxy = research.wrap_model(model, "user_question")
    expected = proxy.prepare(case_request)[0]["input_ids"]
    proxy.max_context = expected.shape[1]
    torch.testing.assert_close(
        proxy.prepare(case_request)[0]["input_ids"], expected, atol=0, rtol=0
    )
    proxy.max_context -= 1
    with pytest.raises(RequestValidationError, match="nothing was truncated"):
        proxy.prepare(case_request)
    assert model.max_context == 4096


def test_training_view_preserves_gradient_through_backbone_and_head(model, case_request):
    proxy = research.wrap_model(model, "user_question")
    proxy.lm.train()
    case_request = case_request.model_copy(update={"questions": case_request.questions[:1]})
    logits = proxy.logits(case_request)
    torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
    assert model.backbone.scale.grad is not None and model.backbone.scale.grad.abs() > 0
    assert model.head.weight.grad is not None and model.head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("data", ["bad-base64", base64.b64encode(b"not an image").decode()])
def test_malformed_native_images_fail_before_forward(model, case_request, data):
    case_request = case_request.model_copy(update={"questions": case_request.questions[:1]})
    case_request.media = [ImageInput(data=data)]
    with pytest.raises(RequestValidationError, match="image"):
        research.predict(model, case_request, "user_question")
    assert model.backbone.last_inputs is None


def test_unknown_mode_is_rejected(model):
    with pytest.raises(ValueError, match="mode"):
        research.wrap_model(model, "unknown")
