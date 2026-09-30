"""Real tiny Gemma 4 Unified forwards with deterministic processor fixtures."""

import base64
import io

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from PIL import Image
from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration

from s1.contracts import AudioInput, ImageInput
from s1.errors import RequestValidationError
from s1.unified import UnifiedDecisionModel, candidate_ids, project_candidates

pytestmark = pytest.mark.inference


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("modality", ["text", "image", "audio", "mixed"])
def test_native_batch_independence(model, decision_request, dtype, modality):
    from s1.evaluation.gemma import predict_batch, predict_sequential

    model.lm.to(dtype=dtype)
    if modality in ("image", "mixed"):
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), "blue").save(buffer, format="PNG")
        decision_request.media.append(ImageInput(data=base64.b64encode(buffer.getvalue()).decode()))
    if modality in ("audio", "mixed"):
        decision_request.media.append(AudioInput(samples=[0.2] * 16))
    reference = predict_sequential(model, decision_request)
    changed = decision_request.model_copy(deep=True)
    changed.questions[0].instructions = "A completely unrelated and longer question"
    changed.questions.reverse()
    # An unrelated text-only request is processed in a separate shape bucket.
    text_only = decision_request.model_copy(deep=True)
    text_only.media = []
    calls = []
    hook = model.backbone.register_forward_hook(lambda _, args, output: calls.append(1))
    try:
        batch = predict_batch(model, [decision_request, changed, text_only])
    finally:
        hook.remove()
    assert len(calls) == (1 if modality == "text" else 2)
    assert sum(batch[0]["execution"]["batch_sizes"]) == 9
    # Different BF16 GEMM shapes can round differently. This is a numerical
    # tolerance check, not a claim of bitwise batch parity.
    tolerance = 2e-3 if dtype == torch.bfloat16 else 1e-5
    for q in decision_request.questions:
        expected = list(reference["answers"][q.id]["probabilities"].values())
        actual = list(batch[0]["answers"][q.id]["probabilities"].values())
        torch.testing.assert_close(
            torch.tensor(actual), torch.tensor(expected), atol=tolerance, rtol=tolerance
        )
        assert actual.index(max(actual)) == expected.index(max(expected))
        if q.id != decision_request.questions[0].id:
            other = list(batch[1]["answers"][q.id]["probabilities"].values())
            torch.testing.assert_close(
                torch.tensor(other), torch.tensor(expected), atol=tolerance, rtol=tolerance
            )
    full = predict_batch(model, [decision_request], readout="full")[0]
    same_batch = predict_batch(model, [decision_request])[0]
    for q in decision_request.questions:
        torch.testing.assert_close(
            torch.tensor(list(full["answers"][q.id]["probabilities"].values())),
            torch.tensor(list(same_batch["answers"][q.id]["probabilities"].values())),
            atol=1e-5,
            rtol=1e-5,
        )


def test_generation_is_one_hard_label_per_question(model, decision_request):
    from s1.evaluation.gemma import predict_generated
    from s1.evaluation.responses import normalize

    result = normalize(decision_request, predict_generated(model, decision_request), "none")
    for question in decision_request.questions:
        assert result["answers"][question.id]["label"] in question.labels()
        assert result["answers"][question.id]["probabilities"] is None


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
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_projection_matches_full_lm_head(model, decision_request, modality, dtype):
    model.lm.to(dtype=dtype)
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
        torch.testing.assert_close(
            candidate[i, :count], full[i, :count].float(), atol=1e-5, rtol=1e-5
        )
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


@pytest.mark.parametrize("softcap", [None, 30.0])
def test_bfloat16_softcap_preserves_native_tie_and_mask(softcap):
    head = torch.nn.Linear(1, 4, bias=True, dtype=torch.bfloat16)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[0.0], [14.625], [14.75], [0.0]]))
        head.bias.zero_()
    hidden = torch.ones((2, 1), dtype=torch.bfloat16)
    ids = torch.tensor([1, 2])
    counts = torch.tensor([2, 1])
    native = head(hidden)[:, ids]
    if softcap is not None:
        native = torch.tanh(native / softcap) * softcap
        assert native[0, 0] == native[0, 1]
    expected = native.float()
    expected[1, 1] = float("-inf")
    actual = project_candidates(hidden, head, ids, counts, softcap)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.argmax(-1).tolist() == expected.argmax(-1).tolist()


def test_context_rejection_and_temperature(model, decision_request):
    model.max_context = 5
    with pytest.raises(RequestValidationError, match="nothing was truncated"):
        model.prepare(decision_request)
    for temperature in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            model.set_temperature(temperature)


def test_corrupt_image_is_a_validation_error(model, decision_request):
    decision_request.media.append(ImageInput(data=base64.b64encode(b"not an image").decode()))
    with pytest.raises(RequestValidationError, match="invalid"):
        model.prepare(decision_request)


@pytest.mark.parametrize("data", ["!invalid-base64!", "圖片", "bm90IGFuIGltYWdl"])
def test_invalid_image_is_still_a_client_error(model, request_data, data):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from s1.api import create_app

    request_data["media"] = [{"type": "image", "data": data}]
    client = TestClient(create_app(model), raise_server_exceptions=False)
    for path in ("/decide", "/v1/systemone"):
        response = client.post(path, json=request_data)
        assert response.status_code == 422
        assert "invalid" in response.json()["detail"]


@pytest.mark.parametrize("explicit_temperature", [None, 2.0])
@pytest.mark.parametrize("commit", [None, "a" * 40])
def test_constructor_uses_resolved_calibration_revision(
    model, monkeypatch, explicit_temperature, commit
):
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from s1 import unified

    model.lm.config._commit_hash = commit
    resolved_revision = commit or "main"
    model_calls = []
    config_calls = []

    def load_model(name, **kwargs):
        model_calls.append((name, kwargs["revision"]))
        return model.lm

    def load_config(name, *, revision):
        config_calls.append((name, revision))
        return 3.0

    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: Processor())
    monkeypatch.setattr(AutoModelForMultimodalLM, "from_pretrained", load_model)
    monkeypatch.setattr(unified, "load_temperature", load_config)
    restored = UnifiedDecisionModel(
        "owner/checkpoint", revision="main", temperature=explicit_temperature, device="cpu"
    )
    assert model_calls == [("owner/checkpoint", "main")]
    assert restored.revision == resolved_revision
    assert restored.temperature == (3.0 if explicit_temperature is None else explicit_temperature)
    assert config_calls == (
        [("owner/checkpoint", resolved_revision)] if explicit_temperature is None else []
    )


def test_local_and_hub_checkpoint_calibration_match(model, decision_request, tmp_path, monkeypatch):
    import huggingface_hub
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    config_path = tmp_path / "s1_config.json"
    config_path.write_text('{"temperature": 3.0}')
    commit = "a" * 40
    model.lm.config._commit_hash = commit
    downloads = []

    def download(name, filename, *, revision):
        downloads.append((name, filename, revision))
        return str(config_path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: Processor())
    monkeypatch.setattr(
        AutoModelForMultimodalLM, "from_pretrained", lambda *args, **kwargs: model.lm
    )
    local = UnifiedDecisionModel(str(tmp_path), device="cpu")
    assert downloads == []
    remote = UnifiedDecisionModel("owner/checkpoint", revision="main", device="cpu")
    assert downloads == [("owner/checkpoint", "s1_config.json", commit)]
    assert local.temperature == remote.temperature == 3.0
    assert local.predict(decision_request)["answers"] == remote.predict(decision_request)["answers"]


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


@pytest.mark.parametrize("reordered", ["state", "nested_state", "state_list_item", "criteria"])
def test_training_rejects_reordered_mapping_keys(benchmark_case, reordered):
    from s1.training import train_model

    train = benchmark_case.model_copy(deep=True)
    train.split = "train"
    train.request.state = {
        "ticket": {"topic": "billing", "refund": True},
        "events": [{"kind": "payment", "amount": 10}],
    }
    calibration = train.model_copy(deep=True)
    calibration.id = "calibration-ticket"
    calibration.split = "calibration"
    if reordered == "state":
        calibration.request.state = dict(reversed(calibration.request.state.items()))
    elif reordered == "nested_state":
        state = calibration.request.state
        state["ticket"] = dict(reversed(state["ticket"].items()))
    elif reordered == "state_list_item":
        events = calibration.request.state["events"]
        events[0] = dict(reversed(events[0].items()))
    else:
        question = calibration.request.questions[0]
        question.criteria = dict(reversed(question.criteria.items()))

    assert train.request.model_dump() == calibration.request.model_dump()
    before = (train.request.model_dump_json(), calibration.request.model_dump_json())
    assert before[0] != before[1]
    # Reject leakage before any model/optimizer access, without modifying inputs.
    with pytest.raises(ValueError, match="requests overlap"):
        train_model(None, [train], [calibration], steps=1)
    assert before == (train.request.model_dump_json(), calibration.request.model_dump_json())


@pytest.mark.parametrize(
    "different", ["state_value", "state_list_order", "question_order", "media_order"]
)
def test_training_overlap_check_preserves_request_differences(model, benchmark_case, different):
    from s1.training import train_model

    train = benchmark_case.model_copy(deep=True)
    train.split = "train"
    train.request.state = {"events": ["paid", "refunded"]}
    train.request.media = [AudioInput(samples=[0.1] * 16), AudioInput(samples=[0.2] * 16)]
    calibration = train.model_copy(deep=True)
    calibration.id = "calibration-ticket"
    calibration.split = "calibration"
    if different == "state_value":
        calibration.request.state["events"][1] = "disputed"
    elif different == "state_list_order":
        calibration.request.state["events"].reverse()
    elif different == "question_order":
        calibration.request.questions.reverse()
    else:
        calibration.request.media.reverse()

    before = (train.request.model_dump_json(), calibration.request.model_dump_json())
    report = train_model(model, [train], [calibration], steps=1)
    assert report["steps"] == 1
    assert before == (train.request.model_dump_json(), calibration.request.model_dump_json())


def test_legacy_large_choice_evaluation_is_gold_independent(monkeypatch):
    import numpy as np

    from s1 import engine
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
    noul = Q("refund?", ["no", "yes"], kind="noul", descs=["no refund", "refund requested"])
    original_decide = engine.decide
    rubrics = []

    def capture(model, state, questions, **kwargs):
        rubrics.append(questions[1]["criteria"])
        return original_decide(model, state, questions, **kwargs)

    monkeypatch.setattr(engine, "decide", capture)
    first = score_examples(FakeModel(), [Example("state", [question, noul])])[0][0]
    question.gold = 76
    second = score_examples(FakeModel(), [Example("state", [question, noul])])[0][0]
    assert rubrics == [{"false": "no refund", "true": "refund requested"}] * 2
    assert len(first) == 77
    assert first.sum() == pytest.approx(1)
    np.testing.assert_array_equal(first, second)
