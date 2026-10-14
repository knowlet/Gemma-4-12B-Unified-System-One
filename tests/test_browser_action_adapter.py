"""Local artifact integrity and genuine tiny CPU PEFT/base isolation."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1 import browser_action_adapter as action
from s1.jev_generated import INSTRUCTIONS, POLICY


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(blob)
    return blob


def record(blob):
    return {"sha256": sha(blob), "size_bytes": len(blob)}


@pytest.fixture
def artifact(tmp_path):
    torch = pytest.importorskip("torch")
    safe = pytest.importorskip("safetensors.torch")
    from s1.contracts import DecisionRequest
    from s1.evaluation.datasets import request_fingerprint

    root, adapter = tmp_path / "training", tmp_path / "training/adapter"
    adapter.mkdir(parents=True)
    receipt = {
        "stage": "complete",
        "split_audit": {"valid": True},
        "full_lm_head_unchanged": True,
        "training": {
            "original_head_sha256_before": "1" * 64,
            "original_head_sha256_after": "1" * 64,
            "frozen_base_versions_unchanged": True,
        },
        "base_model_id": action.BASE_ID,
        "base_revision": action.BASE_REVISION,
        "training_quantization": "none",
        "training_dtype": "bfloat16",
        "deployment_quantization": "nf4",
        "policy": POLICY,
        "instructions_sha256": sha(INSTRUCTIONS.encode()),
    }
    source = b"# Unit test source evidence only\n"
    (root / "sources/s1").mkdir(parents=True)
    (root / "sources/s1/example.py").write_bytes(source)
    files = {"s1/example.py": {**record(source), "artifact_path": "sources/s1/example.py"}}
    identity = sha(
        json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )
    receipt["source_manifest"] = {"schema_version": 1, "files": files, "identity_sha256": identity}
    write(root / "sources/manifest.json", receipt["source_manifest"])

    def check(identity):
        return {
            "success": True,
            "identity_sha256": identity,
            "current_identity_sha256": identity,
            "changed": [],
            "missing": [],
            "added": [],
        }

    receipt["source_verification"] = check(identity)
    (
        receipt["input_manifests"],
        receipt["input_verification"],
        receipt["frozen_input_verification"],
    ) = {}, {}, {}
    provenance = {"source_scope": "development_only"}
    for split in ("train", "development"):
        request = {
            "state": {"observation": split},
            "questions": {
                "operation": {
                    "instructions": "Choose the next operation",
                    "criteria": {"CLICK": "Click", "DONE": "Complete"},
                },
            },
        }
        case = {
            "id": split + "-case",
            "split": split,
            "group_id": split + "-episode",
            "request": request,
            "gold": {"operation": "CLICK"},
        }
        cases = (json.dumps(case) + "\n").encode()
        blobs = {
            "cases.jsonl": cases,
            "wire_requests.jsonl": b"{}\n",
            "receipt.json": b"{}\n",
            "fixtures/0000.html": b"<p>Unit test</p>",
        }
        inputs = {label: record(blob) for label, blob in sorted(blobs.items())}
        identity = sha(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode())
        manifest = {"schema_version": 1, "files": inputs, "identity_sha256": identity}
        for label, blob in blobs.items():
            path = root / "inputs" / split / label
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        encoded = write(root / "inputs" / split / "input_manifest.json", manifest)
        receipt["input_manifests"][split] = {
            **manifest,
            "artifact_path": "inputs/" + split,
            "manifest_sha256": sha(encoded),
        }
        receipt["input_verification"][split] = check(identity)
        receipt["frozen_input_verification"][split] = check(identity)
        provenance[split] = {
            "ids": [case["id"]],
            "groups": [case["group_id"]],
            "request_fingerprints": [request_fingerprint(DecisionRequest.model_validate(request))],
        }
    peft = pytest.importorskip("peft")
    config = peft.LoraConfig(
        **copy.deepcopy(action.LORA),
        inference_mode=True,
        base_model_name_or_path=action.BASE_ID,
        revision=action.BASE_REVISION,
    ).to_dict()
    write(adapter / "adapter_config.json", config)
    torch.manual_seed(41)
    tensors = {}
    for module, output in (("q_proj", 32), ("v_proj", 16)):
        prefix = "base_model.model.model.language_model.layers.0.self_attn." + module
        tensors[prefix + ".lora_A.weight"] = torch.randn(8, 32) * 0.5
        tensors[prefix + ".lora_B.weight"] = torch.randn(output, 8) * 0.5
    (adapter / "adapter_model.safetensors").write_bytes(safe.save(tensors))
    manifest = {
        "schema_version": 1,
        "kind": "jev_action_lora_v1",
        "base_model_id": action.BASE_ID,
        "base_revision": action.BASE_REVISION,
        "training_quantization": "none",
        "training_dtype": "bfloat16",
        "deployment_quantization": "nf4",
        "policy": POLICY,
        "instructions_sha256": sha(INSTRUCTIONS.encode()),
        "inference_model_identity": "generated-actions-nf4",
        "lora": copy.deepcopy(action.LORA),
        "files": {name: record((adapter / name).read_bytes()) for name in action._FILES},
        "provenance": provenance,
        "training_receipt": {"path": "../receipt.json", "sha256": ""},
    }

    def seal():
        manifest["training_receipt"]["sha256"] = sha(write(root / "receipt.json", receipt))
        write(adapter / "manifest.json", manifest)

    seal()
    return SimpleNamespace(
        root=root,
        path=adapter,
        manifest=manifest,
        receipt=receipt,
        seal=seal,
        tensors=tensors,
        config=config,
    )


def test_valid_local_artifact_captures_weights_and_exact_provenance(artifact):
    verified = action.verify_action_adapter(artifact.path)
    metadata = verified.telemetry()
    assert metadata["train_cases"] == metadata["development_cases"] == 1
    assert metadata["tensor_count"] == 4
    assert metadata["weights_sha256"] == sha(verified.files["adapter_model.safetensors"])
    assert verified.manifest == artifact.manifest
    # Loading later consumes these captured bytes, not a newly modified origin.
    (artifact.path / "adapter_model.safetensors").write_bytes(b"later modified origin")
    assert sha(verified.files["adapter_model.safetensors"]) == metadata["weights_sha256"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("kind", "other"),
        ("base_model_id", "other"),
        ("base_revision", "b" * 40),
        ("policy", "other"),
        ("instructions_sha256", "0" * 64),
        ("deployment_quantization", "none"),
        ("training_quantization", "nf4"),
        ("training_dtype", "float32"),
        ("inference_model_identity", "other"),
        ("extra", True),
        ("provenance", None),
    ],
)
def test_manifest_failures_precede_weight_decoding(artifact, monkeypatch, key, value):
    artifact.manifest[key] = value
    artifact.seal()
    monkeypatch.setattr(
        action, "_tensor_bytes", lambda _: pytest.fail("invalid artifact decoded weights")
    )
    with pytest.raises(ValueError):
        action.verify_action_adapter(artifact.path)


@pytest.mark.parametrize(
    "options",
    [
        {"model_id": "other"},
        {"revision": "b" * 40},
        {"quantization": "none"},
    ],
)
def test_requested_base_pins_fail_before_weights(artifact, monkeypatch, options):
    monkeypatch.setattr(
        action, "_tensor_bytes", lambda _: pytest.fail("wrong base reached weights")
    )
    with pytest.raises(ValueError):
        action.verify_action_adapter(artifact.path, **options)


@pytest.mark.parametrize(
    "relative",
    [
        "adapter/adapter_model.safetensors",
        "adapter/adapter_config.json",
        "receipt.json",
        "sources/s1/example.py",
        "inputs/train/cases.jsonl",
        "inputs/development/fixtures/0000.html",
    ],
)
def test_every_consumed_file_hash_is_checked(artifact, relative):
    path = artifact.root / relative
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        action.verify_action_adapter(artifact.path)


@pytest.mark.parametrize(
    "key",
    [
        "full_lm_head_unchanged",
        "split_audit",
        "source_verification",
        "input_verification",
        "frozen_input_verification",
    ],
)
def test_incomplete_training_proof_is_rejected(artifact, key):
    artifact.receipt[key] = False if key == "full_lm_head_unchanged" else {}
    artifact.seal()
    with pytest.raises(ValueError):
        action.verify_action_adapter(artifact.path)


def test_actual_case_inventory_must_match_manifest(artifact):
    artifact.manifest["provenance"]["train"]["ids"] = ["fabricated-case"]
    artifact.seal()
    with pytest.raises(ValueError, match="actual frozen cases"):
        action.verify_action_adapter(artifact.path)


@pytest.mark.parametrize(
    "change",
    [
        "nan",
        "rank",
        "missing_pair",
        "lm_head",
        "image_q_proj",
        "audio_v_proj",
    ],
)
def test_only_finite_complete_text_decoder_lora_pairs_are_accepted(artifact, change):
    torch = pytest.importorskip("torch")
    safe = pytest.importorskip("safetensors.torch")
    values = {key: value.clone() for key, value in artifact.tensors.items()}
    first = next(iter(values))
    if change == "nan":
        values[first][0, 0] = float("nan")
    elif change == "rank":
        values[first] = torch.zeros(7, 32)
    elif change == "missing_pair":
        values.pop(first)
    elif change == "lm_head":
        values["base_model.model.lm_head.weight"] = torch.zeros(8, 32)
    else:
        tower = "vision_tower" if change == "image_q_proj" else "audio_tower"
        values[first.replace("model.language_model", tower)] = values.pop(first)
    blob = safe.save(values)
    (artifact.path / "adapter_model.safetensors").write_bytes(blob)
    artifact.manifest["files"]["adapter_model.safetensors"] = record(blob)
    artifact.seal()
    with pytest.raises(ValueError):
        action.verify_action_adapter(artifact.path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("target_modules", ["q_proj", "v_proj"]),
        ("modules_to_save", ["lm_head"]),
        ("bias", "all"),
        ("r", 4),
        ("use_dora", True),
        ("rank_pattern", {"q_proj": 9}),
        ("auto_mapping", {"parent_library": "external"}),
        ("unknown_option", True),
    ],
)
def test_peft_behavior_configuration_is_pinned(artifact, key, value):
    artifact.config[key] = value
    blob = write(artifact.path / "adapter_config.json", artifact.config)
    artifact.manifest["files"]["adapter_config.json"] = record(blob)
    artifact.seal()
    with pytest.raises(ValueError):
        action.verify_action_adapter(artifact.path)


@pytest.mark.inference
@pytest.mark.parametrize("helper_failure", [False, True])
def test_genuine_tiny_cpu_peft_actions_change_while_base_helper_is_exact_and_restored(
    artifact, monkeypatch, helper_failure
):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from s1 import browser_text
    from s1.browser_text import FieldTextError

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
    lm = transformers.Gemma4UnifiedForConditionalGeneration(config).eval()
    ids = torch.tensor([[20, 21, 22, 23]])

    def generated(model):
        with torch.inference_mode():
            return model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=0,
            )

    baseline = generated(lm)
    # Deployment descriptor pins are simulated; the random model stays CPU FP32.
    model = SimpleNamespace(
        name=action.BASE_ID,
        revision=action.BASE_REVISION,
        quantization="nf4",
        lm=lm,
        backbone=lm.model,
        head=lm.get_output_embeddings(),
        temperature=2.783,
        device="cpu",
    )
    head = model.head.weight.detach().clone()
    verified = action.verify_action_adapter(artifact.path)
    calls = []
    from peft import PeftModel

    original_loader = PeftModel.from_pretrained

    def local_loader(base, path, **kwargs):
        calls.append(kwargs)
        assert Path(path).is_dir() and not Path(path).is_symlink()
        assert (Path(path) / "adapter_model.safetensors").read_bytes() == verified.files[
            "adapter_model.safetensors"
        ]
        return original_loader(base, path, **kwargs)

    monkeypatch.setattr(PeftModel, "from_pretrained", local_loader)
    metadata = action.attach_action_adapter(model, verified)
    assert calls[0]["local_files_only"] is True and calls[0]["is_trainable"] is False
    assert calls[0]["torch_device"] == "cpu"
    assert metadata["all_parameters_frozen"] and all(
        not p.requires_grad for p in model.lm.parameters()
    )
    assert not model.lm.training and model.temperature == 2.783
    assert torch.equal(head, model.head.weight)
    with model.action_adapter.decision_context():
        active = generated(model.lm)
    assert not torch.equal(active, baseline)
    observed = []

    def helper(actual_model, context):
        assert actual_model is model and context == {"goal": "generic input"}
        assert model.lm.get_model_status().enabled is False
        observed.append(generated(model.lm))
        if helper_failure:
            raise FieldTextError("helper failed", {"raw_output": "actual base helper evidence"})
        return "base text", {"raw_output": "actual base helper evidence"}

    monkeypatch.setattr(browser_text, "generate_field_text", helper)
    if helper_failure:
        with pytest.raises(FieldTextError, match="helper failed") as caught:
            action.base_field_text(model, {"goal": "generic input"})
        assert caught.value.generation["action_adapter_state"] == "disabled_for_pinned_base_helper"
    else:
        text, evidence = action.base_field_text(model, {"goal": "generic input"})
        assert (
            text == "base text"
            and evidence["action_adapter_state"] == "disabled_for_pinned_base_helper"
        )
    assert torch.equal(observed[0], baseline)
    assert model.lm.get_model_status().enabled is True
    assert all(not p.requires_grad for p in model.lm.parameters()) and not model.lm.training
    with model.action_adapter.decision_context():
        assert torch.equal(generated(model.lm), active)
    assert torch.equal(head, model.head.weight)
    assert not torch.cuda.is_initialized()


@pytest.mark.inference
def test_genuine_released_trainer_artifact_passes_strict_loader_and_frozen_attach(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    import s1.unified

    source = Path(__file__).with_name("test_jev_browser_fit_actions.py")
    spec = importlib.util.spec_from_file_location("action_writer_contract_fixture", source)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    tiny = fixture.tiny.__wrapped__()
    tiny.lm.to(dtype=torch.bfloat16)
    for parameter in tiny.lm.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    # Reconstruct the exact untouched random base before the tiny LoRA update.
    base = tiny.lm.get_base_model()
    native = transformers.Gemma4UnifiedForConditionalGeneration(base.config).to(
        dtype=torch.bfloat16
    )
    original = {
        key.replace(".base_layer.", "."): value
        for key, value in base.state_dict().items()
        if ".lora_" not in key
    }
    native.load_state_dict(original)
    train = fixture.COLLECTION.collection(tmp_path, "train", group="training-only")
    development = fixture.COLLECTION.collection(tmp_path, "development", group="held-aside")
    monkeypatch.setattr(s1.unified, "UnifiedDecisionModel", lambda *args, **kwargs: tiny)
    output = tmp_path / "fit"
    receipt = fixture.FIT.fit(train_cases=train, development_cases=development, output_dir=output)
    assert receipt["stage"] == "complete" and receipt["training"]["completed_updates"] == 1
    verified = action.verify_action_adapter(output / "adapter")
    assert verified.manifest["files"] == receipt["adapter_files"]
    model = SimpleNamespace(
        name=action.BASE_ID,
        revision=action.BASE_REVISION,
        quantization="nf4",
        device="cpu",
        lm=native,
        backbone=native.model,
        head=native.get_output_embeddings(),
        temperature=2.783,
    )
    before = model.head.weight.detach().clone()
    metadata = action.attach_action_adapter(model, verified)
    assert (
        metadata["weights_sha256"]
        == receipt["adapter_files"]["adapter_model.safetensors"]["sha256"]
    )
    assert all(not parameter.requires_grad for parameter in model.lm.parameters())
    assert torch.equal(before, model.head.weight) and not torch.cuda.is_initialized()
