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


def test_backend_batch_retains_each_complete_multislot_prompt(model, decision_request, monkeypatch):
    from s1.backends import GemmaBackend

    backend = GemmaBackend.__new__(GemmaBackend)
    backend.model = model
    changed = decision_request.model_copy(deep=True)
    changed.questions[0].instructions = "A different and longer first question"
    requests = [decision_request, changed]
    preparations = []
    prepare = model.prepare

    def record_prepare(request):
        inputs, slots, counts = prepare(request)
        preparations.append((inputs["input_ids"].clone(), slots.clone(), counts.clone()))
        return inputs, slots, counts

    monkeypatch.setattr(model, "prepare", record_prepare)
    references = [backend.predict_batch([request])[0] for request in requests]
    full_prompts = preparations[:]
    preparations.clear()
    actual = backend.predict_batch(requests)

    assert len(preparations) == len(full_prompts) == len(requests)
    for got, expected in zip(preparations, full_prompts):
        for got_tensor, expected_tensor in zip(got, expected):
            torch.testing.assert_close(got_tensor, expected_tensor, atol=0, rtol=0)
    for request, response, expected in zip(requests, actual, references):
        assert response["execution"]["independent_questions"] is False
        assert response["execution"]["batch_sizes"] == [len(requests)]
        for question in request.questions:
            torch.testing.assert_close(
                torch.tensor(list(response["answers"][question.id]["probabilities"].values())),
                torch.tensor(list(expected["answers"][question.id]["probabilities"].values())),
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
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
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


@pytest.mark.parametrize("modality", ["text", "image", "audio", "mixed"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_candidate_cache_reuses_rows_with_exact_inference_results(
    model, decision_request, dtype, modality
):
    model.lm.to(dtype=dtype)
    if modality in ("image", "mixed"):
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
        decision_request.media.append(ImageInput(data=base64.b64encode(buffer.getvalue()).decode()))
    if modality in ("audio", "mixed"):
        decision_request.media.append(AudioInput(samples=[0.1] * 16))
    reference = model.predict(decision_request)["answers"]
    assert model._candidate_cache is None
    model.enable_candidate_cache()
    assert model.predict(decision_request)["answers"] == reference
    first_rows = model._candidate_cache[1]
    assert first_rows[0].requires_grad is False
    assert model.predict(decision_request)["answers"] == reference
    assert model._candidate_cache[1] is first_rows
    model.enable_candidate_cache(False)
    assert model._candidate_cache is None


@pytest.mark.parametrize(
    "mutation",
    ["weight", "bias", "parameter", "head", "ids", "ids_tensor", "dtype", "storage"],
)
def test_candidate_cache_invalidates_changed_projection(model, decision_request, mutation):
    head = torch.nn.Linear(32, 256, bias=True)
    with torch.no_grad():
        head.weight.copy_(model.head.weight)
        head.bias.zero_()
    model.head = head.eval()
    model.enable_candidate_cache()
    model.predict(decision_request)
    original_rows = model._candidate_cache[1]
    with torch.no_grad():
        if mutation == "weight":
            model.head.weight[model.letters[0]].add_(1)
        elif mutation == "bias":
            model.head.bias[model.letters[0]].add_(3)
        elif mutation == "parameter":
            model.head.weight = torch.nn.Parameter(model.head.weight.detach() + 1)
        elif mutation == "head":
            model.head = torch.nn.Linear(32, 256, bias=True).eval()
        elif mutation == "ids":
            model.letters[0] = 1
        elif mutation == "ids_tensor":
            model.letters = model.letters.roll(1)
        elif mutation == "dtype":
            model.head.to(dtype=torch.float64)
        else:
            model.head.weight.data = model.head.weight.detach().clone() + 1
    with torch.inference_mode():
        actual = model.logits(decision_request)
        inputs, slots, counts = model.prepare(decision_request)
        hidden = model.backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state[
            0, slots
        ]
        expected = project_candidates(hidden, model.head, model.letters, counts, model.softcap)
    assert model._candidate_cache[1] is not original_rows
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_candidate_cache_bypasses_training_and_preserves_head_gradients(model, decision_request):
    model.enable_candidate_cache()
    model.predict(decision_request)
    model.lm.zero_grad(set_to_none=True)
    logits = model.logits(decision_request)
    assert model._candidate_cache is None
    logits[torch.isfinite(logits)].sum().backward()
    assert model.head.weight.grad is not None
    assert model.head.weight.grad[model.letters].abs().sum() > 0
    model.predict(decision_request)
    model.lm.train()
    with torch.no_grad():
        model.logits(decision_request)
    assert model._candidate_cache is None


def test_candidate_cache_does_not_persist_in_saved_checkpoint(model, decision_request, tmp_path):
    model.enable_candidate_cache()
    expected = model.predict(decision_request)["answers"]
    model.save(tmp_path)
    saved = Gemma4UnifiedForConditionalGeneration.from_pretrained(tmp_path)
    restored = UnifiedDecisionModel.from_components(saved, Processor())
    assert restored.candidate_cache_enabled is False
    assert restored._candidate_cache is None
    assert restored.predict(decision_request)["answers"] == expected
    explicit = UnifiedDecisionModel.from_components(saved, Processor(), enable_candidate_cache=True)
    assert explicit.predict(decision_request)["answers"] == expected
    assert explicit._candidate_cache is not None


def test_candidate_cache_does_not_reuse_unversioned_or_peft_weights(model, decision_request):
    model.enable_candidate_cache()
    with torch.inference_mode():
        model.head.weight = torch.nn.Parameter(model.head.weight.clone())
    model.predict(decision_request)
    assert model._candidate_cache is None
    # A PEFT head may have adapter-specific behavior beyond the base weight's
    # version counter. Opt-in caching does not add assumptions about those heads.
    model.lm.peft_config = {}
    model.head.weight = torch.nn.Parameter(model.head.weight.clone())
    model.predict(decision_request)
    assert model._candidate_cache is None


def test_candidate_cache_explicit_refresh_after_untracked_write(model, decision_request):
    model.enable_candidate_cache()
    model.predict(decision_request)
    rows = model._candidate_cache[1]
    # PyTorch intentionally does not increment the Parameter's version for a
    # .data write. The documented explicit refresh discards these stale rows.
    model.head.weight.data[model.letters[0]].add_(2)
    model.enable_candidate_cache()
    assert model._candidate_cache is None
    model.predict(decision_request)
    assert model._candidate_cache[1] is not rows
    torch.testing.assert_close(
        model._candidate_cache[1][0], model.head.weight[model.letters], atol=0, rtol=0
    )
    with pytest.raises(ValueError, match="boolean"):
        model.enable_candidate_cache("true")


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


@pytest.mark.parametrize(
    "device,precision,expected_dtype",
    [
        ("cpu", None, torch.float32),
        ("cpu", "float32", torch.float32),
        ("cuda:1", None, torch.bfloat16),
        ("cuda:1", "bfloat16", torch.bfloat16),
        ("cuda:1", "float32", torch.float32),
    ],
)
def test_constructor_forwards_explicit_precision_and_device(
    model, monkeypatch, device, precision, expected_dtype
):
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    loaded, initialized = [], []

    def load_model(name, **kwargs):
        loaded.append((name, kwargs))
        return model.lm.to(dtype=kwargs["dtype"])

    def initialize(self, lm, processor, selected_device, temperature, max_context):
        # Assert GPU constructor settings without claiming this CPU test performs
        # a CUDA transfer; real CUDA FP32 parity is an explicit live check.
        initialized.append((selected_device, lm.get_output_embeddings().weight.dtype))

    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: Processor())
    monkeypatch.setattr(AutoModelForMultimodalLM, "from_pretrained", load_model)
    monkeypatch.setattr(UnifiedDecisionModel, "_initialize", initialize)
    UnifiedDecisionModel(
        "owner/checkpoint", revision="a" * 40, device=device, precision=precision, temperature=1.0
    )
    assert loaded == [("owner/checkpoint", {"revision": "a" * 40, "dtype": expected_dtype})]
    assert initialized == [(device, expected_dtype)]


@pytest.mark.parametrize("precision", ["float16", "quantized", "unknown", "bfloat16"])
def test_constructor_rejects_unsupported_cpu_precision_before_loading(monkeypatch, precision):
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    def refuse_load(*args, **kwargs):
        pytest.fail("invalid precision must fail before processor/checkpoint access")

    monkeypatch.setattr(AutoProcessor, "from_pretrained", refuse_load)
    monkeypatch.setattr(AutoModelForMultimodalLM, "from_pretrained", refuse_load)
    with pytest.raises(ValueError, match="precision"):
        UnifiedDecisionModel("owner/checkpoint", device="cpu", precision=precision, temperature=1.0)


@pytest.mark.parametrize("dtype", [None, "auto", "float32", torch.float32])
def test_legacy_precision_and_compatible_dtype_can_coexist(model, monkeypatch, dtype):
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    calls = []

    def load_model(name, **kwargs):
        calls.append(kwargs)
        return model.lm

    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: Processor())
    monkeypatch.setattr(AutoModelForMultimodalLM, "from_pretrained", load_model)
    UnifiedDecisionModel(
        "owner/checkpoint", device="cpu", precision="float32", dtype=dtype, temperature=1.0
    )
    assert calls == [{"revision": None, "dtype": torch.float32}]


def test_conflicting_precision_and_dtype_fail_before_loading(monkeypatch):
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    def refuse_load(*args, **kwargs):
        pytest.fail("contradictory precision must fail before processor/checkpoint access")

    monkeypatch.setattr(AutoProcessor, "from_pretrained", refuse_load)
    monkeypatch.setattr(AutoModelForMultimodalLM, "from_pretrained", refuse_load)
    with pytest.raises(ValueError, match="must agree"):
        UnifiedDecisionModel(
            "owner/checkpoint", device="cuda", precision="bfloat16", dtype="float32"
        )


@pytest.mark.parametrize("quantization", ["int8", "nf4"])
@pytest.mark.parametrize(
    "device,dtype", [("mps", "bfloat16"), ("cpu", "bfloat16"), ("cuda", "float16")]
)
def test_dtype_does_not_bypass_quantization_policy(monkeypatch, quantization, device, dtype):
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    def refuse_load(*args, **kwargs):
        pytest.fail("invalid quantization runtime must fail before checkpoint access")

    monkeypatch.setattr(AutoProcessor, "from_pretrained", refuse_load)
    monkeypatch.setattr(AutoModelForMultimodalLM, "from_pretrained", refuse_load)
    with pytest.raises(ValueError, match="CUDA and bfloat16"):
        UnifiedDecisionModel(
            "owner/checkpoint", device=device, dtype=dtype, quantization=quantization
        )


def test_quantized_initialization_preserves_existing_placement(model, monkeypatch):
    def refuse_move(*args, **kwargs):
        pytest.fail("quantized modules must retain their explicit device-map placement")

    monkeypatch.setattr(model.lm, "to", refuse_move)
    restored = UnifiedDecisionModel.__new__(UnifiedDecisionModel)
    restored.quantization = "nf4"
    restored._initialize(model.lm, Processor(), "cpu", 1.0, 1024)
    assert restored.dtype == "float32"
    assert restored.attn_implementation == "sdpa"
    assert restored.device == "cpu"


def test_explicit_float32_constructor_loads_actual_tiny_checkpoint(
    model, decision_request, tmp_path, monkeypatch
):
    from transformers import AutoProcessor

    model.lm.to(dtype=torch.bfloat16).save_pretrained(tmp_path)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: Processor())
    loaded = UnifiedDecisionModel(str(tmp_path), device="cpu", precision="float32", temperature=1.0)
    assert loaded.head.weight.dtype == torch.float32
    assert all(parameter.dtype == torch.float32 for parameter in loaded.lm.parameters())
    assert loaded.head.weight.device.type == "cpu"
    assert torch.isfinite(loaded.logits(decision_request)[:, :2]).all()


@pytest.mark.integration
@pytest.mark.parametrize("quantization", ["int8", "nf4"])
def test_real_quantized_cuda_adapter_preserves_multimodal_candidate_readout(
    model, decision_request, tmp_path, monkeypatch, quantization
):
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("peft")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("requires CUDA with BF16 support; no checkpoint downloads")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

    from s1.evaluation.gemma import predict_batch

    model.lm.to(dtype=torch.bfloat16).save_pretrained(tmp_path / "base")
    adapted = get_peft_model(
        model.lm, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"])
    )
    # Exercise a nonzero adapter rather than the initial no-op LoRA weights.
    with torch.no_grad():
        for name, value in adapted.named_parameters():
            if "lora_B" in name:
                value.fill_(0.01)
    adapted.save_pretrained(tmp_path / "adapter")
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: Processor())
    loaded = UnifiedDecisionModel(
        str(tmp_path / "base"),
        adapter_path=str(tmp_path / "adapter"),
        device="cuda",
        precision="bfloat16",
        quantization=quantization,
        temperature=1.0,
    )
    assert loaded.quantization_details["quantized_linear_modules"] > 0
    assert loaded.head.weight.dtype == torch.bfloat16
    assert loaded.lm.peft_config
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
    image = ImageInput(data=base64.b64encode(buffer.getvalue()).decode())
    audio = AudioInput(samples=[0.1] * 16)
    loaded.reset_memory_peak()
    for media in ([], [image], [audio], [image, audio]):
        request = decision_request.model_copy(update={"media": media})
        inputs, slots, counts = loaded.prepare(request)
        with torch.inference_mode():
            native = loaded.lm(**inputs, use_cache=False).logits[0, slots][:, loaded.letters]
            selected = loaded.logits(request)
        for i, count in enumerate(counts):
            torch.testing.assert_close(selected[i, :count], native[i, :count].float())
        hook = loaded.head.register_forward_hook(
            lambda *args: pytest.fail("candidate inference called the full head")
        )
        try:
            responses = predict_batch(loaded, [request, request])
            assert responses[0]["answers"] == responses[1]["answers"]
        finally:
            hook.remove()
    receipt = loaded.memory_snapshot()
    assert receipt["peak_allocated_bytes"] >= receipt["allocated_bytes"] > 0
    assert receipt["model_footprint_bytes"] > 0
    with pytest.raises(ValueError, match="separately"):
        loaded.save(tmp_path / "unsafe-merge")


@pytest.mark.parametrize("attention", [None, "eager", "sdpa"])
def test_constructor_forwards_runtime_settings(model, monkeypatch, attention):
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    calls = []

    def load_model(name, **kwargs):
        calls.append(kwargs)
        model.lm.to(dtype=kwargs["dtype"])
        if "attn_implementation" in kwargs:
            model.lm.set_attn_implementation(kwargs["attn_implementation"])
        return model.lm

    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: Processor())
    monkeypatch.setattr(AutoModelForMultimodalLM, "from_pretrained", load_model)
    restored = UnifiedDecisionModel(
        "owner/checkpoint",
        revision="test",
        device="cpu",
        dtype="float16",
        attn_implementation=attention,
        temperature=1.0,
    )
    expected = {"revision": "test", "dtype": torch.float16}
    if attention is not None:
        expected["attn_implementation"] = attention
    assert calls == [expected]
    assert restored.dtype == "float16"
    assert restored.attn_implementation == (attention or "sdpa")


def test_from_components_preserves_dtype_and_backend_reports_runtime(model, monkeypatch):
    from importlib.metadata import version

    from s1 import unified
    from s1.backends import GemmaBackend

    model.lm.to(dtype=torch.bfloat16)
    restored = UnifiedDecisionModel.from_components(model.lm, Processor())
    assert restored.head.weight.dtype == torch.bfloat16
    assert restored.dtype == "bfloat16"
    monkeypatch.setattr(unified, "UnifiedDecisionModel", lambda *args, **kwargs: restored)
    backend = GemmaBackend("in-memory")
    assert backend.metadata["dtype"] == "bfloat16"
    assert backend.metadata["attn_implementation"] == "sdpa"
    assert backend.metadata["torch_version"] == version("torch")
    assert backend.metadata["transformers_version"] == version("transformers")


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

    model.lm.save_pretrained(tmp_path / "base")
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
    model.save_adapter(tmp_path / "decision-adapter")
    assert (tmp_path / "decision-adapter/adapter_model.safetensors").is_file()
    assert (tmp_path / "decision-adapter/s1_config.json").is_file()
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: Processor())
    separate = UnifiedDecisionModel(
        str(tmp_path / "base"), adapter_path=str(tmp_path / "decision-adapter"), device="cpu"
    )
    assert separate.temperature == model.temperature
    for key, answer in separate.predict(train.request)["answers"].items():
        assert answer["probabilities"] == pytest.approx(expected[key]["probabilities"], abs=1e-5)
    model.save(tmp_path)
    assert (tmp_path / "processor-test.json").is_file()
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: Processor())
    restored = UnifiedDecisionModel(str(tmp_path), device="cpu")
    assert restored.temperature == model.temperature
    for key, answer in restored.predict(train.request)["answers"].items():
        assert answer["probabilities"] == pytest.approx(expected[key]["probabilities"], abs=1e-5)


def test_matched_lora_curve_keeps_test_locked_and_saves_six_adapters(
    model, benchmark_case, tmp_path
):
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    from s1.evaluation.datasets import EvaluationCase
    from s1.evaluation.experiments import write_cases
    from s1.evaluation.supervision import prepare_support
    from s1.evaluation.training_curve import train_curve

    for split, count in (("train", 8), ("calibration", 2), ("test", 2)):
        cases = []
        for i in range(count):
            raw = benchmark_case.model_dump(mode="json")
            raw.update(id=f"{split}-{i}", group_id=f"{split}-{i}", split=split)
            raw["request"]["state"] = f"{split} independent document {i}"
            raw["request"]["questions"] = raw["request"]["questions"][:1]
            raw["gold"] = {"route": "billing" if i % 2 else "technical"}
            cases.append(EvaluationCase.model_validate(raw))
        write_cases(tmp_path / f"{split}.jsonl", cases)
    prepare_support(
        tmp_path / "train.jsonl",
        [tmp_path / "calibration.jsonl", tmp_path / "test.jsonl"],
        tmp_path / "support",
        shots=[1],
    )

    def factory(name, **kwargs):
        lm = Gemma4UnifiedForConditionalGeneration(model.lm.config)
        return UnifiedDecisionModel.from_components(
            get_peft_model(lm, LoraConfig(**kwargs["lora"])), Processor()
        )

    report = train_curve(
        tmp_path / "support/support.json",
        tmp_path / "calibration.jsonl",
        tmp_path / "curve",
        model_id="tiny-fixture",
        revision="a" * 40,
        steps=1,
        max_updates=6,
        model_factory=factory,
    )
    assert report["status"] == "completed" and len(report["runs"]) == 6
    for i in range(0, 6, 2):
        a, b = report["runs"][i : i + 2]
        assert a["train_ids"] == b["train_ids"]
        assert a["seed"] == b["seed"]
        assert a["brier_weight"] == 0 and b["brier_weight"] == 0.1
        assert a["adapter_sha256"] != b["adapter_sha256"]
    with pytest.raises(ValueError, match="calibration"):
        train_curve(
            tmp_path / "support/support.json",
            tmp_path / "test.jsonl",
            tmp_path / "leak",
            model_id="tiny-fixture",
            revision="a" * 40,
            steps=1,
            max_updates=6,
            model_factory=lambda *args, **kwargs: pytest.fail("must validate before model load"),
        )


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
