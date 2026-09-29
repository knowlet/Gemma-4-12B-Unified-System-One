"""Real tiny Gemma 4 Unified forwards with deterministic processor fixtures."""

import base64
import io

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from PIL import Image
from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration

from s1.contracts import AudioInput, ImageInput
from s1.unified import UnifiedDecisionModel, candidate_ids

pytestmark = pytest.mark.inference


class Tokenizer:
    unk_token_id = 255

    def encode(self, text, **kwargs):
        return [ord(c) % 240 for c in text]


class Processor:
    tokenizer = Tokenizer()

    def save_pretrained(self, path):
        (path / "processor-test.json").write_text("{}")

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "state"

    def __call__(self, **kwargs):
        ids = [2, 10, 20, 30]
        types = [0] * 4
        result = {}
        if "images" in kwargs:
            ids.extend([250, 250])
            types.extend([1, 1])
            result["pixel_values"] = torch.arange(24, dtype=torch.float32).reshape(1, 2, 12)
            result["image_position_ids"] = torch.tensor([[[0, 0], [0, 1]]])
        if "audio" in kwargs:
            ids.extend([251, 251])
            types.extend([3, 3])
            result["input_features"] = torch.ones((1, 2, 8))
            result["input_features_mask"] = torch.ones((1, 2), dtype=torch.bool)
        result.update(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
            mm_token_type_ids=torch.tensor([types]),
        )
        return result


@pytest.fixture
def model():
    torch.manual_seed(42)
    config = Gemma4UnifiedConfig(
        text_config={
            "vocab_size": 256,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 64,
            "max_position_embeddings": 1024,
            "final_logit_softcapping": 3.0,
        },
        vision_config={
            "patch_size": 2,
            "pooling_kernel_size": 1,
            "mm_embed_dim": 16,
            "output_proj_dims": 16,
            "mm_posemb_size": 8,
        },
        audio_config={"audio_embed_dim": 8},
        image_token_id=250,
        audio_token_id=251,
        video_token_id=252,
    )
    lm = Gemma4UnifiedForConditionalGeneration(config).eval()
    return UnifiedDecisionModel.from_components(lm, Processor())


@pytest.mark.parametrize("modality", ["text", "image", "audio", "mixed"])
def test_projection_matches_full_lm_head(model, decision_request, modality):
    if modality in ("image", "mixed"):
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
        decision_request.media.append(ImageInput(data=base64.b64encode(buffer.getvalue()).decode()))
    if modality in ("audio", "mixed"):
        decision_request.media.append(AudioInput(samples=[0.1] * 16))
    inputs, slots, counts = model.prepare(decision_request)
    assert slots[-1] == inputs["input_ids"].shape[1] - 1
    assert inputs["mm_token_type_ids"][0, slots].tolist() == [0, 0, 0]
    with torch.inference_mode():
        full = model.lm(**inputs, use_cache=False).logits[0, slots][:, model.letters]
        candidate = model.logits(decision_request)
    for i, count in enumerate(counts):
        torch.testing.assert_close(candidate[i, :count], full[i, :count], atol=1e-5, rtol=1e-5)
        assert torch.isneginf(candidate[i, count:]).all()
    # A selected-row implementation must never call the full output head.
    handle = model.head.register_forward_hook(lambda *args: pytest.fail("full LM head was called"))
    calls = []
    hook = model.backbone.register_forward_hook(lambda *args: calls.append(1))
    try:
        result = model.predict(decision_request)
    finally:
        handle.remove()
        hook.remove()
    assert len(calls) == result["passes"] == 1
    assert len(result["answers"]) == 3


def test_context_rejection_and_temperature(model, decision_request):
    model.max_context = 5
    with pytest.raises(ValueError, match="nothing was truncated"):
        model.prepare(decision_request)
    for temperature in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            model.set_temperature(temperature)


def test_corrupt_image_is_a_validation_error(model, decision_request):
    decision_request.media.append(ImageInput(data=base64.b64encode(b"not an image").decode()))
    with pytest.raises(ValueError, match="invalid"):
        model.prepare(decision_request)


def test_context_sensitive_candidates_rejected():
    class Broken(Tokenizer):
        def encode(self, text, **kwargs):
            ids = super().encode(text)
            return ids[:-2] + [200] if text.endswith("(A") else ids

    with pytest.raises(ValueError, match="boundary"):
        candidate_ids(Broken())


def test_training_lora_calibration_and_checkpoint(model, benchmark_case, tmp_path, monkeypatch):
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

    from s1.training import train_model

    lm = get_peft_model(
        model.lm, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"])
    )
    model = UnifiedDecisionModel.from_components(lm, Processor())
    train = benchmark_case.model_copy(deep=True)
    train.split = "train"
    calibration = benchmark_case.model_copy(deep=True)
    calibration.id = "calibration-ticket"
    calibration.split = "calibration"
    calibration.request.state = "Please refund a duplicated invoice payment."
    before = {
        name: value.detach().clone() for name, value in lm.named_parameters() if value.requires_grad
    }
    report = train_model(model, [train], [calibration], steps=2, lr=0.01)
    assert report["steps"] == 2
    assert report["calibration"]["n"] == 3
    assert any(
        not torch.equal(before[name], value)
        for name, value in lm.named_parameters()
        if name in before
    )
    expected = model.predict(train.request)["answers"]
    model.save(tmp_path)
    assert (tmp_path / "processor-test.json").is_file()
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: Processor())
    restored = UnifiedDecisionModel(str(tmp_path))
    assert restored.temperature == model.temperature
    for key, answer in restored.predict(train.request)["answers"].items():
        assert answer["probabilities"] == pytest.approx(expected[key]["probabilities"], abs=1e-5)


def test_training_rejects_test_split_and_leakage(model, benchmark_case):
    from s1.training import train_model

    with pytest.raises(ValueError, match="split"):
        train_model(model, [benchmark_case], [benchmark_case], steps=1)
    train = benchmark_case.model_copy(deep=True)
    train.split = "train"
    calibration = benchmark_case.model_copy(deep=True)
    calibration.split = "calibration"
    calibration.id = "different-id"
    with pytest.raises(ValueError, match="requests overlap"):
        train_model(model, [train], [calibration], steps=1)


def test_legacy_large_choice_evaluation_is_gold_independent():
    import numpy as np

    from s1.engine import score_examples
    from s1.schema import Example, Q

    class LegacyTokenizer(Tokenizer):
        bos_token_id = 2
        pad_token_id = 0

    class FakeModel:
        tok = LegacyTokenizer()
        temperature = 1.0

        def forward_batch(self, batch):
            return torch.zeros((len(batch["nopts"]), 52)).masked_fill(
                torch.arange(52)[None, :] >= batch["nopts"][:, None], float("-inf")
            )

    question = Q("choose", [str(i) for i in range(77)], gold=0)
    first = score_examples(FakeModel(), [Example("state", [question])])[0][0]
    question.gold = 76
    second = score_examples(FakeModel(), [Example("state", [question])])[0][0]
    assert len(first) == 77
    assert first.sum() == pytest.approx(1)
    np.testing.assert_array_equal(first, second)
