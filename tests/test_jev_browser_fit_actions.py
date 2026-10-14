"""Native action-token boundaries, genuine tiny gradients and frozen evidence."""

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import jev_browser_fit_actions as FIT
finally:
    sys.path.pop(0)

SPEC = importlib.util.spec_from_file_location(
    "action_collection_fixture", ROOT / "tests/test_jev_browser_fit_readout.py"
)
COLLECTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTION)


class Tokenizer:
    pad_token_id = 0
    unk_token_id = -1

    def __init__(self):
        self.vocab = {"<turn|>": 2}

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, -1)

    def encode(self, text):
        words = re.findall(r"<turn\|>|<assistant>|<user>|\w+|\s+|[^\w\s]", text)
        return [self.vocab.setdefault(word, len(self.vocab) + 3) for word in words]


class Processor:
    def __init__(self, torch):
        self.torch, self.tokenizer = torch, Tokenizer()
        self.calls, self.mutate, self.media = [], None, False

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        assert len(messages) == 1 and messages[0]["role"] == "user"
        self.calls.append(copy.deepcopy(messages))
        return "<user>" + messages[0]["content"][0]["text"] + "<assistant>"

    def __call__(self, *, text, return_tensors, add_special_tokens):
        assert return_tensors == "pt" and add_special_tokens is False
        ids = self.torch.tensor([self.tokenizer.encode(text)])
        result = {
            "input_ids": ids,
            "attention_mask": self.torch.ones_like(ids),
            "mm_token_type_ids": self.torch.zeros_like(ids),
        }
        # An actual preserved prompt mask, rather than assuming every input token is visible.
        result["attention_mask"][0, 0] = 0
        if text.endswith("<turn|>") and self.mutate:
            result[self.mutate][0, 1] += 1
        if self.media:
            result["pixel_values"] = self.torch.zeros(1, 1, 1)
        return result


@pytest.fixture
def tiny():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    if not hasattr(transformers, "Gemma4UnifiedConfig"):
        pytest.skip("native Gemma4 Unified is required")
    torch.manual_seed(5)
    config = transformers.Gemma4UnifiedConfig(
        text_config={
            "vocab_size": 512,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "layer_types": ["sliding_attention", "full_attention"] * 2,
            "num_kv_shared_layers": 2,
            "sliding_window": 32,
            "max_position_embeddings": 8192,
            "final_logit_softcapping": 3.0,
            "attention_k_eq_v": True,
        },
        vision_config={
            "patch_size": 2,
            "pooling_kernel_size": 1,
            "mm_embed_dim": 8,
            "output_proj_dims": 8,
            "mm_posemb_size": 4,
        },
        audio_config={"audio_embed_dim": 8},
        image_token_id=510,
        audio_token_id=511,
        video_token_id=509,
    )
    config._attn_implementation = "sdpa"
    base = transformers.Gemma4UnifiedForConditionalGeneration(config)
    base.generation_config.eos_token_id = [1, 2]
    lm = peft.get_peft_model(base, peft.LoraConfig(**FIT.LORA, revision=FIT.REVISION))
    processor = Processor(torch)
    return SimpleNamespace(
        lm=lm,
        backbone=base.model,
        head=base.get_output_embeddings(),
        processor=processor,
        tok=processor.tokenizer,
        device="cpu",
        max_context=8192,
        name=FIT.MODEL_ID,
        revision=FIT.REVISION,
        quantization="none",
    )


def rows(tmp_path, model, split="train", *, group=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = COLLECTION.collection(tmp_path, split, group=group or split)
    cases, _, snapshot = FIT.integrity._validated_snapshot(path, split)
    return FIT.compile_actions(model, cases, snapshot), path


def test_compile_exact_inference_prefix_only_model_alias_changed_and_gold_not_in_prompt(
    tmp_path, tiny
):
    compiled, path = rows(tmp_path, tiny)
    row = compiled[0]
    wire = json.loads((path.parent / "wire_requests.jsonl").read_text())
    message = tiny.processor.calls[0][0]["content"][0]["text"]
    from s1.jev_generated import INSTRUCTIONS

    original = copy.deepcopy(wire["payload"])
    original["model"] = FIT.IDENTITY
    assert message == INSTRUCTIONS + "\n" + FIT._encode(original).decode()
    assert "teacher_choice" not in message and "valid_next_targets" not in message
    assert row["record"]["target_json"] == '{"operation":"DONE","target":null}'
    assert row["record"]["target_token_ids"][-1] == 2
    p = row["record"]["prompt_tokens"]
    assert row["positions"].tolist() == list(range(p - 1, row["record"]["input_tokens"] - 1))
    assert row["inputs"]["attention_mask"][0, 0] == 0


def test_forced_singleton_active_target_is_compiled_from_audited_wire(tmp_path, tiny):
    path = COLLECTION.collection(tmp_path, "train")
    wire = json.loads((path.parent / "wire_requests.jsonl").read_text())
    wire["teacher_choice"] = {"operation": "CLICK", "target": "1"}
    wire["valid_next_targets"] = {"CLICK": ["1"]}
    cases, _, snapshot = FIT.integrity._validated_snapshot(path, "train")
    case = cases[0].model_copy(update={"gold": {"operation": "CLICK"}})
    # The public fit validates all wire/case/episode bytes first; this unit test
    # exercises the compiler's active forced head representation directly.
    snapshot["blobs"]["wire_requests.jsonl"] = FIT._encode(wire) + b"\n"
    compiled = FIT.compile_actions(tiny, [case], snapshot)
    assert compiled[0]["record"]["target_json"] == '{"operation":"CLICK","target":"1"}'


@pytest.mark.parametrize("field", ["input_ids", "attention_mask", "mm_token_type_ids"])
def test_any_native_prompt_tensor_boundary_change_rejects_before_forward(tmp_path, tiny, field):
    tiny.processor.mutate = field
    with pytest.raises(ValueError, match="prefix changed"):
        rows(tmp_path, tiny)


def test_context_and_media_fail_without_truncation(tmp_path, tiny):
    tiny.max_context = 100
    with pytest.raises(ValueError, match="budget"):
        rows(tmp_path, tiny)
    tiny.max_context, tiny.processor.media = 8192, True
    with pytest.raises(ValueError, match="text sequence"):
        rows(tmp_path / "other", tiny)


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_native_shifted_masked_ce_and_checkpoint_lora_gradient_parity(tmp_path, tiny, dtype_name):
    import torch
    from torch.nn.functional import cross_entropy

    dtype = getattr(torch, dtype_name)
    # Match deployment training: BF16 base, FP32 adapter matrices.
    tiny.lm.to(dtype=dtype)
    for name, parameter in tiny.lm.named_parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
            if ".lora_B." in name:
                torch.nn.init.normal_(parameter, std=0.01)
    compiled, _ = rows(tmp_path, tiny)
    row = compiled[0]
    tiny.lm.train()
    result = []
    for checkpoint in (False, True):
        tiny.lm.zero_grad(set_to_none=True)
        if checkpoint:
            tiny.lm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            tiny.lm.gradient_checkpointing_disable()
        loss, _ = FIT.action_loss(tiny, row)
        loss.backward()
        result.append(
            (
                loss.detach(),
                {
                    name: parameter.grad.detach().clone()
                    for name, parameter in tiny.lm.named_parameters()
                    if parameter.requires_grad
                },
            )
        )
    tolerance = 2e-5 if dtype == torch.float32 else 0.003
    torch.testing.assert_close(result[0][0], result[1][0], rtol=tolerance, atol=tolerance)
    for name, gradient in result[0][1].items():
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        torch.testing.assert_close(gradient, result[1][1][name], rtol=tolerance, atol=tolerance)
    tiny.lm.gradient_checkpointing_disable()
    with torch.no_grad():
        full = tiny.lm(**row["inputs"], use_cache=False).logits[0]
        manual = cross_entropy(full[row["positions"]].float(), row["targets"])
        labels = row["inputs"]["input_ids"].clone()
        labels[:, : row["record"]["prompt_tokens"]] = -100
        native = tiny.lm(**row["inputs"], labels=labels, use_cache=False).loss
    torch.testing.assert_close(manual, result[0][0], rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(native, manual, rtol=tolerance, atol=tolerance)
    assert not torch.cuda.is_initialized()


def test_genuine_lora_update_development_no_updates_frozen_head_and_trace(tmp_path, tiny):
    import torch

    train, _ = rows(tmp_path, tiny)
    development, _ = rows(tmp_path, tiny, "development")
    before = {
        name: parameter.detach().clone()
        for name, parameter in tiny.lm.named_parameters()
        if parameter.requires_grad
    }
    head = tiny.head.weight.detach().clone()
    output = tmp_path / "training"
    output.mkdir()
    result = FIT.train_actions(tiny, train, development, output)
    assert result["completed_updates"] == 1 and result["development_updates"] == 0
    assert len(result["trace"]["backbone_calls"]) == 2
    assert all(row["input_shape"][0] == 1 for row in result["trace"]["backbone_calls"])
    assert (
        sum(
            v
            for k, v in result["trace"]["decoder_layer_calls"].items()
            if k.startswith("backward:")
        )
        >= 4
    )
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in tiny.lm.named_parameters()
        if parameter.requires_grad
    )
    assert torch.equal(head, tiny.head.weight) and result["frozen_base_versions_unchanged"]
    assert result["original_head_sha256_before"] == result["original_head_sha256_after"]
    execution = result["execution"]
    assert execution["backbone_forward_calls"] == 2 and execution["max_forward_batch_size"] == 1
    assert execution["physical_input_positions"] == sum(
        row["record"]["input_tokens"] for row in train + development
    )
    assert execution["logical_input_tokens"] == execution["physical_input_positions"] - 2
    assert execution["target_projection_positions"] == sum(
        len(row["targets"]) for row in train + development
    )
    assert len((output / "updates.jsonl").read_text().splitlines()) == 1
    assert not tiny.lm.training and not tiny.lm.is_gradient_checkpointing


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_partial_update_evidence_preserves_primary_error_and_cleans_hooks(
    tmp_path, tiny, monkeypatch, exception
):
    train, _ = rows(tmp_path, tiny)
    development, _ = rows(tmp_path, tiny, "development")
    output = tmp_path / "training"
    output.mkdir()
    original = FIT.action_loss
    calls = []

    def fail(model, row):
        calls.append(row)
        if len(calls) == 2:
            raise exception("primary native failure")
        return original(model, row)

    monkeypatch.setattr(FIT, "action_loss", fail)
    with pytest.raises(exception, match="primary native failure") as caught:
        FIT.train_actions(tiny, train, development, output, steps=2)
    evidence = caught.value.action_training_evidence
    assert evidence["completed_updates"] == 1 and len(evidence["trace"]["backbone_calls"]) == 1
    assert json.loads((output / "training-failure.json").read_text())["completed_updates"] == 1
    assert not tiny.lm.training and not tiny.lm.is_gradient_checkpointing
    assert not tiny.backbone._forward_pre_hooks


def test_full_fit_freezes_inputs_adapter_only_and_final_manifest_receipt(
    tmp_path, tiny, monkeypatch
):
    import torch

    import s1.unified

    tiny.lm.to(dtype=torch.bfloat16)
    for parameter in tiny.lm.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    train = COLLECTION.collection(tmp_path, "train", group="training-only")
    development = COLLECTION.collection(tmp_path, "development", group="held-aside")
    monkeypatch.setattr(s1.unified, "UnifiedDecisionModel", lambda *args, **kwargs: tiny)
    output = tmp_path / "fit"
    receipt = FIT.fit(train_cases=train, development_cases=development, output_dir=output)
    assert receipt["stage"] == "complete" and receipt["full_lm_head_unchanged"]
    assert receipt["source_verification"]["success"]
    assert all(value["success"] for value in receipt["input_verification"].values())
    assert all(value["success"] for value in receipt["frozen_input_verification"].values())
    assert (output / "inputs/train/cases.jsonl").read_bytes() == train.read_bytes()
    manifest = json.loads((output / "adapter/manifest.json").read_text())
    assert set(path.name for path in (output / "adapter").iterdir()) == {
        "manifest.json",
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    assert manifest["training_receipt"]["sha256"] == FIT.integrity._sha(
        (output / "receipt.json").read_bytes()
    )
    assert manifest["provenance"]["train"]["ids"] != manifest["provenance"]["development"]["ids"]
    assert "scripts/jev_browser_fit_actions.py" in receipt["source_manifest"]["files"]
    with pytest.raises(FileExistsError):
        FIT.fit(train_cases=train, development_cases=development, output_dir=output)


def test_final_input_mismatch_writes_failed_receipt_and_never_valid_adapter(
    tmp_path, tiny, monkeypatch
):
    import torch

    import s1.unified

    tiny.lm.to(dtype=torch.bfloat16)
    for parameter in tiny.lm.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    train = COLLECTION.collection(tmp_path, "train", group="training-only")
    development = COLLECTION.collection(tmp_path, "development", group="held-aside")
    monkeypatch.setattr(s1.unified, "UnifiedDecisionModel", lambda *args, **kwargs: tiny)
    original = FIT._save_adapter

    def changed(model, directory):
        result = original(model, directory)
        train.write_bytes(train.read_bytes() + b" ")
        return result

    monkeypatch.setattr(FIT, "_save_adapter", changed)
    output = tmp_path / "fit"
    with pytest.raises(RuntimeError, match="collection changed"):
        FIT.fit(train_cases=train, development_cases=development, output_dir=output)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["stage"] == "failed" and not receipt["input_verification"]["train"]["success"]
    assert not (output / "adapter/manifest.json").exists()
    assert (output / "adapter/manifest.invalid.json").exists()


@pytest.mark.parametrize(
    "kwargs", [{"steps": 0}, {"steps": True}, {"learning_rate": float("nan")}, {"seed": False}]
)
def test_invalid_options_reject_before_output_or_model(tmp_path, kwargs):
    with pytest.raises(ValueError):
        FIT.fit(
            train_cases="missing",
            development_cases="missing",
            output_dir=tmp_path / "out",
            **kwargs,
        )
    assert not (tmp_path / "out").exists()


def test_finalization_interrupt_cannot_mask_primary(tmp_path, monkeypatch):
    primary = RuntimeError("original native failure")
    receipt = {"stage": "failed", "error": {"type": "RuntimeError", "message": str(primary)}}

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("secondary finalization interrupt")

    monkeypatch.setattr(FIT.integrity, "_finalize_fit", interrupted)
    FIT._finalize(tmp_path, receipt, primary_error=primary)
    assert receipt["stage"] == "failed" and receipt["error"]["message"] == str(primary)
    assert receipt["finalization_error"]["type"] == "KeyboardInterrupt"
    assert "Secondary finalization" in primary.__notes__[0]


def test_cleanup_and_resource_errors_keep_native_primary(tmp_path, tiny, monkeypatch):
    train, _ = rows(tmp_path, tiny)
    development, _ = rows(tmp_path, tiny, "development")
    output = tmp_path / "training"
    output.mkdir()

    def failure(*args):
        raise RuntimeError("primary forward failed")

    def resource_failure(*args):
        raise ValueError("resource probe failed")

    def cleanup_failure():
        raise KeyboardInterrupt("cleanup interrupted")

    monkeypatch.setattr(FIT, "action_loss", failure)
    monkeypatch.setattr(FIT, "_resources", resource_failure)
    monkeypatch.setattr(tiny.lm, "gradient_checkpointing_disable", cleanup_failure)
    with pytest.raises(RuntimeError, match="primary forward failed") as caught:
        FIT.train_actions(tiny, train, development, output)
    evidence = caught.value.action_training_evidence
    assert evidence["resource_error"]["type"] == "ValueError"
    assert evidence["cleanup_errors"][0]["type"] == "KeyboardInterrupt"
    assert not tiny.backbone._forward_pre_hooks


def test_text_decoder_structure_rejects_any_other_trainable_parameter(tmp_path, tiny):
    import torch

    tiny.lm.add_module("vision_q_proj", torch.nn.Linear(1, 1))
    train, _ = rows(tmp_path, tiny)
    development, _ = rows(tmp_path, tiny, "development")
    with pytest.raises(ValueError, match="q/v LoRA"):
        FIT.train_actions(tiny, train, development, tmp_path)


def test_manifest_publication_failure_is_failed_and_unloadable(tmp_path, tiny, monkeypatch):
    import torch

    import s1.unified

    tiny.lm.to(dtype=torch.bfloat16)
    for parameter in tiny.lm.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    train = COLLECTION.collection(tmp_path, "train", group="training-only")
    development = COLLECTION.collection(tmp_path, "development", group="held-aside")
    monkeypatch.setattr(s1.unified, "UnifiedDecisionModel", lambda *args, **kwargs: tiny)
    original = FIT._write

    def failed_write(path, value):
        if Path(path).parts[-2:] == ("adapter", "manifest.json"):
            Path(path).write_text('{"partial":')
            raise OSError("manifest publication failed")
        return original(path, value)

    monkeypatch.setattr(FIT, "_write", failed_write)
    output = tmp_path / "fit"
    with pytest.raises(OSError, match="manifest publication failed"):
        FIT.fit(train_cases=train, development_cases=development, output_dir=output)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["stage"] == "failed" and receipt["error"]["type"] == "OSError"
    assert (output / "adapter/manifest.invalid.json").read_text() == '{"partial":'
    assert not (output / "adapter/manifest.json").exists()
    assert receipt["source_verification"]["success"]


def test_model_load_interruption_has_staged_freezes_and_failed_receipt(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import s1.unified

    train = COLLECTION.collection(tmp_path, "train", group="training-only")
    development = COLLECTION.collection(tmp_path, "development", group="held-aside")
    output = tmp_path / "fit"

    def interrupted(*args, **kwargs):
        staged = json.loads((output / "receipt.json").read_text())
        assert staged["stage"] == "model_loading" and staged["source_manifest"]
        assert set(staged["input_manifests"]) == {"train", "development"}
        raise KeyboardInterrupt("native model load interrupted")

    monkeypatch.setattr(s1.unified, "UnifiedDecisionModel", interrupted)
    with pytest.raises(KeyboardInterrupt, match="native model load interrupted"):
        FIT.fit(train_cases=train, development_cases=development, output_dir=output)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["stage"] == "failed" and receipt["error"]["type"] == "KeyboardInterrupt"
    assert receipt["source_verification"]["success"]
    assert all(value["success"] for value in receipt["frozen_input_verification"].values())
