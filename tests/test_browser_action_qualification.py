"""BF16 qualification stays separate from NF4 and restores a genuine tiny PEFT model."""

import hashlib
import threading
from types import SimpleNamespace

import pytest
import test_browser_action_adapter as artifact_tests

from s1 import browser_action_adapter as artifact_api
from s1 import browser_action_qualification as qualification

artifact = artifact_tests.artifact


def test_normalization_preserves_hostname_and_all_non_transport_fields():
    left = {
        "model": "a",
        "state": {"page": {"url": "http://127.0.0.1:123/tasks/a?q=x"}, "elements": ["x"]},
    }
    right = {
        "model": "b",
        "state": {"page": {"url": "http://127.0.0.1:456/tasks/a?q=x"}, "elements": ["x"]},
    }
    assert qualification.payload_bytes(
        qualification.normalized_initial(left)
    ) == qualification.payload_bytes(qualification.normalized_initial(right))
    right["state"]["page"]["url"] = "http://localhost:456/tasks/a?q=x"
    assert qualification.normalized_initial(left) != qualification.normalized_initial(right)
    assert left["model"] == "a" and left["state"]["page"]["url"].startswith("http://127.0.0.1:123/")


def test_qualification_schema_failure_precedes_artifact_or_model_access(tmp_path, monkeypatch):
    path = tmp_path / "qualification.json"
    path.write_text('{"schema_version": true, "kind": "other"}')
    monkeypatch.setattr(
        artifact_api,
        "verify_action_adapter",
        lambda *_a, **_kw: pytest.fail("bad qualification reached adapter weights"),
    )
    with pytest.raises(ValueError, match="schema"):
        qualification.verify_bf16_action_qualification(tmp_path / "adapter", path)


@pytest.mark.parametrize("decode_failure", [False, True])
def test_decision_exit_guard_retains_primary_and_persistent_integrity_failure(
    monkeypatch, decode_failure
):
    guard = object.__new__(qualification._BF16Guard)
    guard.lock = threading.RLock()
    guard.integrity_errors = []
    guard.qualification = SimpleNamespace(qualification_sha256="test-qualification")
    guard.weight = SimpleNamespace(_version=0)
    guard.head = guard.backbone = object()
    guard.model = SimpleNamespace(temperature=qualification.TEMPERATURE)
    guard.parameters = []
    calls = []

    def check_boundary():
        calls.append(None)
        if len(calls) == 2:
            raise ValueError("exit guard invariant failed")

    monkeypatch.setattr(guard, "_check", check_boundary)
    primary = RuntimeError("decode failed")
    with pytest.raises(RuntimeError if decode_failure else ValueError) as caught:
        with guard.decision_context():
            if decode_failure:
                raise primary
    if decode_failure:
        assert caught.value is primary
        assert any("exit guard invariant failed" in note for note in primary.__notes__)
    else:
        assert str(caught.value) == "exit guard invariant failed"
    assert guard.integrity_errors == [
        {"type": "ValueError", "message": "exit guard invariant failed"}
    ]
    with pytest.raises(ValueError, match="retained integrity"):
        guard.check()


@pytest.mark.inference
@pytest.mark.parametrize("helper_failure", [False, True])
def test_real_tiny_bf16_fp32_peft_helper_is_exact_and_restored(
    artifact, monkeypatch, helper_failure
):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from s1 import browser_text

    torch.manual_seed(113)
    config = transformers.Gemma4UnifiedConfig(
        text_config={
            "vocab_size": 256,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "global_head_dim": 8,
            "layer_types": ["full_attention"],
            "max_position_embeddings": 128,
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
    lm = transformers.Gemma4UnifiedForConditionalGeneration(config).to(torch.bfloat16).eval()
    ids = torch.tensor([[20, 21, 22, 23]])

    def generated(actual):
        with torch.inference_mode():
            return actual.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=0,
            )

    baseline = generated(lm)
    model = SimpleNamespace(
        name=artifact_api.BASE_ID,
        revision=artifact_api.BASE_REVISION,
        quantization="none",
        quantization_details={"method": "none", "quantized_linear_modules": 0},
        lm=peft.get_peft_model(
            lm, peft.LoraConfig(**artifact_api.LORA, revision=artifact_api.BASE_REVISION)
        ),
        backbone=lm.model,
        head=lm.get_output_embeddings(),
        temperature=qualification.TEMPERATURE,
        max_context=8192,
        attn_implementation="sdpa",
        device="cpu",
    )
    # This is a genuine smaller BF16 model and FP32 adapter, with scaled test pins.
    monkeypatch.setattr(qualification, "TENSOR_COUNT", 4)
    monkeypatch.setattr(qualification, "HEAD_SHA256", qualification._head_hash(model.head.weight))
    verified = artifact_api.verify_action_adapter(artifact.path)
    declaration = {
        "runtime": qualification.runtime_profile(),
        "observed_diagnostics": {"scope": "tiny CPU test"},
    }
    blob = qualification.canonical(declaration)
    approved = qualification.VerifiedBF16ActionQualification(
        verified,
        blob,
        hashlib.sha256(blob).hexdigest(),
        declaration,
        qualification._artifact_identity(verified),
    )
    old_cudnn = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(False)
    try:
        metadata = qualification.attach_qualified_bf16_action_adapter(model, approved)
        assert (
            metadata["runtime_quantization"] == "none"
            and metadata["artifact_deployment_contract"] == "nf4"
        )
        assert all(
            p.dtype == (torch.float32 if ".lora_" in name else torch.bfloat16)
            for name, p in model.lm.named_parameters()
        )
        with model.action_adapter.decision_context():
            active = generated(model.lm)
        assert not torch.equal(active, baseline)
        observed = []

        def helper(actual, context):
            assert actual is model and context == {"goal": "generic input"}
            assert model.lm.get_model_status().enabled is False
            assert all(not p.requires_grad for p in model.lm.parameters())
            observed.append(generated(model.lm))
            if helper_failure:
                raise browser_text.FieldTextError(
                    "helper validation failed",
                    {"raw_token_ids": observed[-1].tolist(), "failure_stage": "validation"},
                )
            return "base text", {"raw_token_ids": observed[-1].tolist()}

        monkeypatch.setattr(browser_text, "generate_field_text", helper)
        if helper_failure:
            with pytest.raises(browser_text.FieldTextError) as caught:
                artifact_api.base_field_text(model, {"goal": "generic input"})
            assert (
                caught.value.generation["action_adapter_state"] == "disabled_for_pinned_base_helper"
            )
        else:
            _, evidence = artifact_api.base_field_text(model, {"goal": "generic input"})
            assert evidence["action_adapter_state"] == "disabled_for_pinned_base_helper"
        assert torch.equal(observed[0], baseline)
        assert (
            model.lm.get_model_status().enabled is True
            and model.lm.get_model_status().merged_adapters == []
        )
        assert all(not p.requires_grad for p in model.lm.parameters()) and not model.lm.training
        assert model.action_adapter.check(full=True)["success"]
        with model.action_adapter.decision_context():
            assert torch.equal(generated(model.lm), active)
        original_alpha = model.lm.peft_config["default"].lora_alpha
        model.lm.peft_config["default"].lora_alpha += 1
        with pytest.raises(ValueError, match="PEFT configuration"):
            model.action_adapter.check()
        model.lm.peft_config["default"].lora_alpha = original_alpha
        with pytest.raises(ValueError, match="retained integrity"):
            model.action_adapter.check()
        assert not torch.cuda.is_initialized()
    finally:
        torch.backends.cuda.enable_cudnn_sdp(old_cudnn)
