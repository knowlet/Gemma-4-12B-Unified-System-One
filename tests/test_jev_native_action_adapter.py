"""Pure native artifact checks with synthetic token traces, never model evidence.

The optional full179 fixture authenticates real historical data. Tensor decoding
is stubbed explicitly; these tests do not qualify a tokenizer, adapter or model.
"""

import builtins
import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1 import browser_action_adapter as action
from s1 import browser_native_action_data as data_api
from s1.browser_action_prompt import (
    NATIVE_FORMAT,
    NATIVE_KIND,
    expected_format,
    native_prompt,
    prompt_contract,
)
from s1.evaluation.datasets import request_fingerprint
from s1.jev_action_grammar import LegalActionTokenTrie
from s1.jev_generated import INSTRUCTIONS, POLICY, format_action_request

ROOT = Path(__file__).absolute().parents[1]


@pytest.fixture(autouse=True)
def no_framework_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "peft", "tokenizers", "safetensors"}:
            raise AssertionError("Pure artifact verifier imported a model framework: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def pin(blob):
    return {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = encode(value) + b"\n"
    path.write_bytes(blob)
    return blob


class RecordedTokenFixture:
    """Distinct short paths for fixture canonical actions; not native tokenization."""

    all_special_ids = [106]

    def __init__(self):
        self.tokens = {}

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        if text not in self.tokens:
            self.tokens[text] = [1000 + len(self.tokens), 9]
        return self.tokens[text]


def compiled(case, wire):
    payload = copy.deepcopy(wire["payload"])
    payload["model"] = "generated-actions-nf4"
    user_text, source, labels, representation = format_action_request(
        payload, payload["model"], prompt_format=NATIVE_FORMAT
    )
    tok = RecordedTokenFixture()
    grammar = LegalActionTokenTrie(labels, tok, prompt_length=10, eos_token_id=106)
    target = encode(wire["teacher_choice"]).decode()
    targets = tok.encode(target) + [106]
    ids = [1] * 10 + targets
    record = {
        "case_id": case.id,
        "split": case.split,
        "group_id": case.group_id,
        "request_fingerprint": request_fingerprint(case.request),
        "original_wire_sha256": pin(encode(wire["payload"]))["sha256"],
        "normalized_wire_sha256": pin(source)["sha256"],
        "original_model_alias": wire["payload"]["model"],
        "inference_model_identity": payload["model"],
        "normalization": "only payload.model replaced; all other original fields/order retained",
        "policy": POLICY,
        "prompt": native_prompt(user_text),
        "prompt_tokens": 10,
        "target_json": target,
        "target_token_ids": targets,
        "native_stop_token_id": 106,
        "input_tokens": len(ids),
        "logical_input_tokens": len(ids) - 1,
        "logits_positions": list(range(9, len(ids) - 1)),
        "processor_tensors": {
            "input_ids": [ids],
            "attention_mask": [[0] + [1] * (len(ids) - 1)],
            "mm_token_type_ids": [[0] * len(ids)],
        },
        "prompt_format": NATIVE_FORMAT,
        "prompt_representation": representation,
    }
    snapshot = {
        "record": record,
        "inputs": record["processor_tensors"],
        "positions": record["logits_positions"],
        "targets": targets,
    }
    allowed = [grammar(0, ids[:10] + targets[:i]) for i in range(len(targets))]
    plan = {
        "schema_id": "native-cfg-action-loss-plan-v1",
        "compiled_row_sha256": pin(encode(snapshot))["sha256"],
        "request_sha256": record["normalized_wire_sha256"],
        "target_ids": targets,
        "positions": record["logits_positions"],
        "allowed_ids": allowed,
        "decision_indices": [i for i, choices in enumerate(allowed) if len(choices) > 1],
        "canonical_inventory_sha256": grammar.telemetry()["canonical_inventory_sha256"],
        "encoded_inventory_sha256": pin(encode(grammar.encoded_actions))["sha256"],
        "legal_action_count": len(grammar.canonical_actions),
        "scope": "all original legal candidates; teacher-forced native token branches",
    }
    cfg = {
        "case_id": case.id,
        "plan": plan,
        "inventory": {
            "labels": labels,
            "canonical_actions": list(grammar.canonical_actions),
            "encoded_actions": list(map(list, grammar.encoded_actions)),
            "telemetry": grammar.telemetry(),
        },
    }
    return record, cfg, labels


@pytest.fixture
def example():
    from s1.browser_public_training import NativeWireRequest

    elements = [
        {
            "index": str(i),
            "label": "Open complete original catalog entry " + str(i),
            "value": "",
            "role": "button",
            "operations": ["CLICK"],
        }
        for i in range(80)
    ]
    payload = {
        "model": "old-observed-alias",
        "state": {"text": "one full original observation", "elements": elements},
        "questions": {
            "operation": {
                "instructions": "Choose",
                "criteria": {"CLICK": "Open", "DONE": "Complete"},
            },
            "click_target": {
                "instructions": "Choose",
                "criteria": {
                    e["index"]: {
                        "element": "[" + e["index"] + "] " + e["label"],
                        "current_value": "",
                        "role": "button",
                    }
                    for e in elements
                },
            },
        },
    }
    case = SimpleNamespace(
        id="fixture",
        split="train",
        group_id="fixture-group",
        gold={"operation": "CLICK", "click_target": "79"},
        request=NativeWireRequest.model_validate(payload),
    )
    wire = {
        "case_id": case.id,
        "payload": payload,
        "teacher_choice": {"operation": "CLICK", "target": "79"},
        "valid_next_targets": {"CLICK": ["79"]},
    }
    return case, wire


def test_explicit_contract_and_known_raw_compatibility():
    contract = prompt_contract()
    assert contract["selection_basis"] == "complete_user_text_utf8_bytes"
    assert (
        expected_format({"kind": NATIVE_KIND, "prompt_contract": contract}, NATIVE_FORMAT)
        == NATIVE_FORMAT
    )
    for kind in ("jev_action_lora_v1", "qualified_bf16_exposed_diagnostic_v1"):
        assert expected_format({"kind": kind}, "raw-json-v1") == "raw-json-v1"
        with pytest.raises(ValueError):
            expected_format({"kind": kind}, NATIVE_FORMAT)
    with pytest.raises(ValueError):
        expected_format({"kind": NATIVE_KIND, "prompt_contract": contract}, "raw-json-v1")
    with pytest.raises(ValueError):
        expected_format({}, "raw-json-v1")
    contract["requested_format"] = "raw-json-v1"
    with pytest.raises(ValueError):
        expected_format({"kind": NATIVE_KIND, "prompt_contract": contract}, NATIVE_FORMAT)


def test_recorded_full80_inventory_and_original_teacher(example):
    case, wire = example
    record, cfg, labels = compiled(case, wire)
    assert record["prompt_representation"]["format"] == NATIVE_FORMAT
    assert action._native_record(record, case, wire) == labels
    action._native_cfg(cfg, record, labels)
    assert cfg["plan"]["legal_action_count"] == 81


def test_requested_v2_raw_fallback_is_still_a_v2_artifact(example):
    from s1.browser_public_training import NativeWireRequest

    case, wire = example
    payload = copy.deepcopy(wire["payload"])
    payload["state"] = {"text": "short"}
    payload["questions"]["click_target"]["criteria"] = {"79": "Open"}
    wire = {**wire, "payload": payload}
    case = SimpleNamespace(**{**vars(case), "request": NativeWireRequest.model_validate(payload)})
    record, cfg, labels = compiled(case, wire)
    assert record["prompt_representation"]["format"] == "raw-json-v1"
    assert record["prompt_representation"]["requested_format"] == NATIVE_FORMAT
    action._native_record(record, case, wire)
    action._native_cfg(cfg, record, labels)


@pytest.mark.parametrize(
    "field",
    [
        "prompt",
        "prompt_format",
        "prompt_representation",
        "target_json",
        "original_wire_sha256",
        "logits_positions",
        "processor_tensors",
    ],
)
def test_record_corruption_rejects_before_tensor_decoding(example, field):
    case, wire = example
    record, _, _ = compiled(case, wire)
    if field == "processor_tensors":
        record[field]["attention_mask"][0][-1] = 0
    elif field == "logits_positions":
        record[field][0] += 1
    elif field == "prompt_representation":
        record[field]["criterion_refs"] = -1
    else:
        record[field] = "corrupt"
    with pytest.raises(ValueError):
        action._native_record(record, case, wire)


@pytest.mark.parametrize(
    "mutation", ["canonical", "encoded", "labels", "allowed", "snapshot", "telemetry"]
)
def test_cfg_teacher_inventory_and_snapshot_corruption_refuses(example, mutation):
    case, wire = example
    record, cfg, labels = compiled(case, wire)
    if mutation == "canonical":
        cfg["inventory"]["canonical_actions"].pop()
    elif mutation == "encoded":
        cfg["inventory"]["encoded_actions"][0][-1] = 0
    elif mutation == "labels":
        cfg["inventory"]["labels"]["click_target"].reverse()
    elif mutation == "allowed":
        cfg["plan"]["allowed_ids"][0].pop()
    elif mutation == "snapshot":
        cfg["plan"]["compiled_row_sha256"] = "0" * 64
    else:
        cfg["inventory"]["telemetry"]["max_new_tokens"] = 31
    with pytest.raises(ValueError):
        action._native_cfg(cfg, record, labels)


def check(identity):
    return {
        "success": True,
        "identity_sha256": identity,
        "current_identity_sha256": identity,
        "changed": [],
        "missing": [],
        "added": [],
    }


@pytest.fixture(scope="module")
def full179(tmp_path_factory):
    if not (ROOT / data_api.COMPONENT["path"]).is_file():
        pytest.skip("Ignored original historical DATA is not distributed")
    data = data_api.load_historical_data(ROOT)
    root = tmp_path_factory.mktemp("native-artifact")
    descriptor = data_api.freeze_historical_data(data, root / "inputs/historical")
    receipt = {
        "schema_version": 2,
        "kind": NATIVE_KIND,
        "stage": "complete",
        "steps": 142,
        "seed": 42,
        "learning_rate": 1e-4,
        "training_objective": "full-vocabulary-action-token-CE",
        "split_audit": {"valid": True},
        "full_lm_head_unchanged": True,
        "base_model_id": action.BASE_ID,
        "base_revision": action.BASE_REVISION,
        "training_quantization": "none",
        "training_dtype": "bfloat16",
        "deployment_quantization": "nf4",
        "policy": POLICY,
        "instructions_sha256": pin(INSTRUCTIONS.encode())["sha256"],
        "prompt_contract": prompt_contract(),
        "native_data": descriptor,
        "input_manifests": {},
        "input_verification": {},
        "frozen_input_verification": {},
        "compiled": {},
        "cfg_compiled": {},
    }
    files = {}
    required = {
        "s1/browser_action_adapter.py",
        "s1/browser_action_prompt.py",
        "s1/browser_native_action_data.py",
        "scripts/jev_browser_fit_native_actions.py",
        "scripts/jev_browser_fit_actions.py",
        "scripts/jev_cfg_action_loss.py",
        descriptor["component"]["path"],
    }
    for label in sorted(set(receipt["prompt_contract"]["sources"]) | required):
        actual = ROOT / ("src/" + label if label.startswith("s1/") else label)
        blob = (
            actual.read_bytes()
            if actual.is_file()
            else b"# Pure fixture source; no execution attested.\n"
        )
        path = root / "sources" / label
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        files[label] = {**pin(blob), "artifact_path": "sources/" + label}
    identity = pin(
        json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )["sha256"]
    receipt["source_manifest"] = {"schema_version": 1, "files": files, "identity_sha256": identity}
    write(root / "sources/manifest.json", receipt["source_manifest"])
    receipt["source_verification"] = check(identity)
    provenance = {"source_scope": "development_only"}
    records = {}
    for split in ("train", "development"):
        blobs = data_api.split_blobs(data, split)
        inputs = {label: pin(blob) for label, blob in blobs.items()}
        identity = pin(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode())["sha256"]
        manifest = {"schema_version": 1, "files": inputs, "identity_sha256": identity}
        for label, blob in blobs.items():
            path = root / "inputs" / split / label
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        encoded = write(root / "inputs" / split / "input_manifest.json", manifest)
        receipt["input_manifests"][split] = {
            **manifest,
            "manifest_sha256": pin(encoded)["sha256"],
            "artifact_path": "inputs/" + split,
        }
        receipt["input_verification"][split] = receipt["frozen_input_verification"][split] = check(
            identity
        )
        provenance[split] = data_api.inventory(blobs["cases.jsonl"], split)
        selected = [
            compiled(case, wire)
            for case, wire in zip(data["cases"], data["wires"], strict=True)
            if case.split == split
        ]
        records[split] = [row[0] for row in selected]
        for name, values in (
            ("compiled", records[split]),
            ("cfg-plans", [row[1] for row in selected]),
        ):
            blob = b"".join(encode(value) + b"\n" for value in values)
            (root / (name + "-" + split + ".jsonl")).write_bytes(blob)
            summary = {"examples": len(values), "sha256": pin(blob)["sha256"]}
            if name == "compiled":
                summary.update(
                    prompt_tokens=sum(row["prompt_tokens"] for row in values),
                    target_tokens=sum(len(row["target_token_ids"]) for row in values),
                )
            receipt["compiled" if name == "compiled" else "cfg_compiled"][split] = summary
    calls, updates, dev = [], [], []
    for split in ("train", "development"):
        for step, record in enumerate(records[split], 1):
            row = {
                "case_id": record["case_id"],
                "loss": 0.5,
                "correct_target_tokens": 0,
                "teacher_forced_exact_tokens": False,
                "target_tokens": len(record["target_token_ids"]),
                "input_tokens": record["input_tokens"],
                "logical_input_tokens": record["logical_input_tokens"],
            }
            if split == "train":
                row.update(
                    step=step, gradient_norm=0.1, forward_batch_size=1, backbone_forward_calls=1
                )
                updates.append(row)
            else:
                dev.append(row)
            calls.append(
                {
                    "split": split,
                    "step": step,
                    "phase": "forward" if split == "train" else "evaluation",
                    "case_id": record["case_id"],
                    "input_shape": [1, record["input_tokens"]],
                }
            )
    tensor_shapes = {
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": [8, 1],
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": [1, 8],
    }
    training = {
        "completed_updates": 142,
        "development_examples": 37,
        "development_updates": 0,
        "original_head_sha256_before": "1" * 64,
        "original_head_sha256_after": "1" * 64,
        "frozen_base_versions_unchanged": True,
        "trainable_parameters": 16,
        "trainable_names": [name.replace(".weight", ".default.weight") for name in tensor_shapes],
        "trace": {"backbone_calls": calls},
        "execution": {
            "backbone_forward_calls": 179,
            "max_forward_batch_size": 1,
            "physical_input_positions": sum(r["input_tokens"] for r in updates + dev),
            "logical_input_tokens": sum(r["logical_input_tokens"] for r in updates + dev),
            "target_projection_positions": sum(r["target_tokens"] for r in updates + dev),
        },
    }
    receipt["training"] = training
    write(root / "training.json", training)
    for label, values in (("updates.jsonl", updates), ("development.jsonl", dev)):
        (root / label).write_bytes(b"".join(encode(row) + b"\n" for row in values))
    config = {
        **action.LORA,
        "peft_type": "LORA",
        "base_model_name_or_path": action.BASE_ID,
        "revision": action.BASE_REVISION,
        "inference_mode": True,
    }
    write(root / "adapter/adapter_config.json", config)
    (root / "adapter/adapter_model.safetensors").write_bytes(b"PURE_TEST_NOT_REAL_TENSOR_BYTES")
    manifest = {
        "schema_version": 2,
        "kind": NATIVE_KIND,
        **{
            key: receipt[key]
            for key in (
                "base_model_id",
                "base_revision",
                "training_quantization",
                "training_dtype",
                "deployment_quantization",
                "policy",
                "instructions_sha256",
                "prompt_contract",
                "native_data",
            )
        },
        "inference_model_identity": "generated-actions-nf4",
        "lora": action.LORA,
        "files": {name: pin((root / "adapter" / name).read_bytes()) for name in action._FILES},
        "provenance": provenance,
        "training_receipt": {"path": "../receipt.json", "sha256": ""},
    }
    receipt.update(adapter_artifact="adapter", adapter_files=manifest["files"])
    manifest["training_receipt"]["sha256"] = pin(write(root / "receipt.json", receipt))["sha256"]
    write(root / "adapter/manifest.json", manifest)
    return root, {
        "tensor_count": 2,
        "parameters": 16,
        "stored_dtypes": ["torch.float32"],
        "tensor_shapes": tensor_shapes,
    }


@pytest.fixture
def artifact(full179, tmp_path, monkeypatch):
    original, tensors = full179
    root = tmp_path / "fit"
    shutil.copytree(original, root)
    monkeypatch.setattr(action, "_tensor_bytes", lambda _: copy.deepcopy(tensors))
    return root


def reseal(root, mutate):
    receipt = json.loads((root / "receipt.json").read_bytes())
    manifest = json.loads((root / "adapter/manifest.json").read_bytes())
    mutate(receipt, manifest)
    manifest["training_receipt"]["sha256"] = pin(write(root / "receipt.json", receipt))["sha256"]
    write(root / "adapter/manifest.json", manifest)


def test_full179_native_branch_is_distinct_and_relocatable(artifact):
    verified = action.verify_action_adapter(
        artifact / "adapter", expected_prompt_format=NATIVE_FORMAT
    )
    metadata = verified.telemetry()
    assert metadata["kind"] == NATIVE_KIND and metadata["prompt_format"] == NATIVE_FORMAT
    assert metadata["train_cases"] == 142 and metadata["development_cases"] == 37
    assert metadata["qualified"] is False and metadata["task_score"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "format",
        "objective",
        "descriptor",
        "development-update",
        "compiled-sha",
        "cfg-sha",
        "source-pin",
    ],
)
def test_resealed_native_contract_corruption_precedes_tensor_decode(
    artifact, monkeypatch, mutation
):
    def mutate(receipt, manifest):
        if mutation == "format":
            manifest["prompt_contract"]["requested_format"] = "raw-json-v1"
        elif mutation == "objective":
            receipt["training_objective"] = "conditional-CFG-loss"
        elif mutation == "descriptor":
            manifest["native_data"]["counts"]["train"] = 141
        elif mutation == "development-update":
            receipt["training"]["development_updates"] = 1
            write(artifact / "training.json", receipt["training"])
        elif mutation in {"compiled-sha", "cfg-sha"}:
            receipt["compiled" if mutation == "compiled-sha" else "cfg_compiled"]["train"][
                "sha256"
            ] = "0" * 64
        else:
            receipt["prompt_contract"]["sources"]["s1/jev_prompt.py"]["sha256"] = "0" * 64

    reseal(artifact, mutate)
    monkeypatch.setattr(
        action, "_tensor_bytes", lambda _: pytest.fail("Invalid profile reached tensor decoding")
    )
    with pytest.raises(ValueError):
        action.verify_action_adapter(artifact / "adapter")


def test_native_raw_mismatch_fails_before_tensor_decode(artifact, monkeypatch):
    monkeypatch.setattr(
        action, "_tensor_bytes", lambda _: pytest.fail("Wrong format reached tensors")
    )
    with pytest.raises(ValueError, match="prompt format"):
        action.verify_action_adapter(artifact / "adapter", expected_prompt_format="raw-json-v1")


@pytest.mark.parametrize(
    "mutation",
    ["duplicate-update", "gradient", "loss", "trace", "development-bool", "quality-claim"],
)
def test_actual_ledger_corruption_cannot_publish_native_complete(artifact, mutation):
    def mutate(receipt, _manifest):
        if mutation in {"duplicate-update", "gradient", "loss"}:
            rows = [
                json.loads(line) for line in (artifact / "updates.jsonl").read_bytes().splitlines()
            ]
            if mutation == "duplicate-update":
                rows[1]["case_id"] = rows[0]["case_id"]
                receipt["training"]["trace"]["backbone_calls"][1]["case_id"] = rows[0]["case_id"]
            else:
                rows[0]["gradient_norm" if mutation == "gradient" else "loss"] = -1.0
            (artifact / "updates.jsonl").write_bytes(b"".join(encode(row) + b"\n" for row in rows))
        elif mutation == "trace":
            receipt["training"]["trace"]["backbone_calls"][0]["input_shape"] = [1, 1]
        elif mutation == "development-bool":
            receipt["training"]["development_updates"] = False
        else:
            receipt["task_quality_attested"] = True
        write(artifact / "training.json", receipt["training"])

    reseal(artifact, mutate)
    with pytest.raises(ValueError):
        action.verify_action_adapter(artifact / "adapter")


def test_original_frozen_data_and_legacy_kind_bypass_refuse(artifact):
    manifest = json.loads((artifact / "adapter/manifest.json").read_bytes())
    first = next(iter(manifest["native_data"]["files"]))
    path = artifact / "inputs/historical" / first
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        action.verify_action_adapter(artifact / "adapter")
    manifest.update(schema_version=1, kind="jev_action_lora_v1")
    write(artifact / "adapter/manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest schema"):
        action.verify_action_adapter(artifact / "adapter")


def test_verified_raw_parent_can_initialize_distinct_native_child(artifact, tmp_path):
    from s1.contracts import DecisionRequest

    parent = tmp_path / "legacy-parent"
    shutil.copytree(artifact, parent)
    receipt = json.loads((parent / "receipt.json").read_bytes())
    manifest = json.loads((parent / "adapter/manifest.json").read_bytes())
    for key in ("prompt_contract", "native_data"):
        manifest.pop(key)
        receipt.pop(key)
    manifest.update(schema_version=1, kind="jev_action_lora_v1")
    for key in ("schema_version", "kind", "training_objective"):
        receipt.pop(key)
    receipt["training"].update(completed_updates=1, development_updates=0)
    for split in ("train", "development"):
        request = {
            "state": split,
            "questions": {
                "operation": {
                    "instructions": "Choose",
                    "criteria": {"CLICK": "Open", "DONE": "Complete"},
                }
            },
        }
        case = {
            "id": "legacy-" + split,
            "split": split,
            "group_id": "legacy-group-" + split,
            "request": request,
            "gold": {"operation": "CLICK"},
        }
        blobs = {"cases.jsonl": encode(case) + b"\n", "wire_requests.jsonl": b"{}\n"}
        entries = {label: pin(blob) for label, blob in blobs.items()}
        identity = pin(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode())[
            "sha256"
        ]
        inputs = {"schema_version": 1, "files": entries, "identity_sha256": identity}
        for label, blob in blobs.items():
            (parent / "inputs" / split / label).write_bytes(blob)
        blob = write(parent / "inputs" / split / "input_manifest.json", inputs)
        receipt["input_manifests"][split] = {
            **inputs,
            "manifest_sha256": pin(blob)["sha256"],
            "artifact_path": "inputs/" + split,
        }
        receipt["input_verification"][split] = receipt["frozen_input_verification"][split] = check(
            identity
        )
        manifest["provenance"][split] = {
            "ids": [case["id"]],
            "groups": [case["group_id"]],
            "request_fingerprints": [request_fingerprint(DecisionRequest.model_validate(request))],
        }
    manifest["training_receipt"]["sha256"] = pin(write(parent / "receipt.json", receipt))["sha256"]
    write(parent / "adapter/manifest.json", manifest)
    captured = action.capture_initial_action_adapter(
        parent / "adapter", artifact / "inputs/initial_action_adapter"
    )
    verified = captured["verified"]
    descriptor = captured["descriptor"]
    descriptor["initial_tensor_verification"] = {
        "success": True,
        "tensor_count": verified.tensor_metadata["tensor_count"],
        "tensor_shapes": verified.tensor_metadata["tensor_shapes"],
        "weights_sha256": verified.manifest["files"]["adapter_model.safetensors"]["sha256"],
        "all_tensors_exact": True,
        "all_trainables_qv_fp32": True,
        "frozen_base_versions_unchanged": True,
        "full_head_identity_version_preserved": True,
    }
    reseal(artifact, lambda receipt, _: receipt.update(initial_action_adapter=descriptor))
    child = action.verify_action_adapter(artifact / "adapter", expected_prompt_format=NATIVE_FORMAT)
    assert child.tensor_metadata["lineage_depth"] == 1
    assert child.telemetry()["prompt_format"] == NATIVE_FORMAT
