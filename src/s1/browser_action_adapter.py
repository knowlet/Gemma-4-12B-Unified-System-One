"""Verified local action LoRA; base field generation always disables the adapter.

Only recorded development training artifacts are accepted. Provenance is audited
before loading backbone weights and is never supplied to the action prompt.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .jev_generated import INSTRUCTIONS, POLICY

BASE_ID = "knowlet/Gemma-4-12B-Unified-System-One"
BASE_REVISION = "a66f836b56605039fe040f330180e336d19b3362"
LORA = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": r"model\.language_model\.layers\.[0-9]+\.self_attn\.(q_proj|v_proj)",
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
_FILES = {"adapter_config.json", "adapter_model.safetensors"}
_KEYS = {
    "schema_version",
    "kind",
    "base_model_id",
    "base_revision",
    "training_quantization",
    "training_dtype",
    "deployment_quantization",
    "policy",
    "instructions_sha256",
    "inference_model_identity",
    "lora",
    "files",
    "provenance",
    "training_receipt",
}


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _json(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate artifact JSON key")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("nonfinite artifact JSON")

    return json.loads(blob, object_pairs_hook=pairs, parse_constant=constant)


def _read(root, label, *, limit=256_000_000):
    if not isinstance(label, str):
        raise ValueError("adapter evidence path must be text")
    relative = Path(label)
    if relative.is_absolute() or ".." in relative.parts or "\\" in label:
        raise ValueError("adapter evidence path must stay inside its local directory")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("adapter evidence must not contain symlinks")
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError("adapter evidence file missing or exceeds its byte limit")
    return path.read_bytes()


def _record(root, label, record):
    if not isinstance(record, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256"))):
        raise ValueError("adapter evidence requires SHA256 records")
    blob = _read(root, label)
    if (
        type(record.get("size_bytes")) is not int
        or len(blob) != record["size_bytes"]
        or _sha(blob) != record["sha256"]
    ):
        raise ValueError(f"adapter evidence hash/size mismatch: {label}")
    return blob


def _inventory(blob, split):
    from .evaluation.datasets import EvaluationCase, request_fingerprint

    cases = [
        EvaluationCase.model_validate(_json(line)) for line in blob.splitlines() if line.strip()
    ]
    if not cases or any(case.split != split for case in cases):
        raise ValueError("adapter provenance requires exact training/development case splits")
    result = {
        "ids": sorted({case.id for case in cases}),
        "groups": sorted({case.group_id or case.id for case in cases}),
        "request_fingerprints": sorted({request_fingerprint(case.request) for case in cases}),
    }
    if len(result["ids"]) != len(cases):
        raise ValueError("adapter provenance contains duplicate case IDs")
    return result


def _training_evidence(root, receipt, provenance):
    training = receipt.get("training", {})
    if (
        receipt.get("stage") != "complete"
        or receipt.get("split_audit", {}).get("valid") is not True
        or receipt.get("full_lm_head_unchanged") is not True
        or training.get("frozen_base_versions_unchanged") is not True
        or not re.fullmatch(r"[0-9a-f]{64}", str(training.get("original_head_sha256_before")))
        or training.get("original_head_sha256_before") != training.get("original_head_sha256_after")
    ):
        raise ValueError("adapter requires a complete split-audited training receipt")
    sources = receipt.get("source_manifest", {})
    files = sources.get("files", {})
    identity = _sha(
        json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )
    verification = receipt.get("source_verification", {})
    if (
        not files
        or sources.get("schema_version") != 1
        or sources.get("identity_sha256") != identity
        or verification.get("success") is not True
        or verification.get("identity_sha256") != identity
        or verification.get("current_identity_sha256") != identity
        or any(verification.get(k) != [] for k in ("changed", "missing", "added"))
        or _json(_read(root, "sources/manifest.json", limit=4_000_000)) != sources
    ):
        raise ValueError("adapter training sources were not verified")
    for label, record in files.items():
        if record.get("artifact_path") != "sources/" + label or not label.endswith(".py"):
            raise ValueError("invalid adapter training source inventory")
        _record(root, record["artifact_path"], record)
    if (
        not isinstance(provenance, dict)
        or set(provenance) != {"source_scope", "train", "development"}
        or provenance["source_scope"] != "development_only"
    ):
        raise ValueError("adapter requires explicit development-only provenance")
    for split in ("train", "development"):
        record = receipt.get("input_manifests", {}).get(split, {})
        prefix = "inputs/" + split
        inputs = {k: record.get(k) for k in ("schema_version", "files", "identity_sha256")}
        entries = inputs["files"]
        if not isinstance(entries, dict) or not entries or record.get("artifact_path") != prefix:
            raise ValueError("adapter requires frozen training input inventories")
        encoded = _read(root, prefix + "/input_manifest.json", limit=4_000_000)
        expected = _sha(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode())
        if (
            inputs["schema_version"] != 1
            or inputs["identity_sha256"] != expected
            or _sha(encoded) != record.get("manifest_sha256")
            or _json(encoded) != inputs
        ):
            raise ValueError("adapter frozen input manifest differs from receipt")
        for kind in ("input_verification", "frozen_input_verification"):
            check = receipt.get(kind, {}).get(split, {})
            if (
                check.get("success") is not True
                or check.get("identity_sha256") != expected
                or check.get("current_identity_sha256") != expected
                or any(check.get(k) != [] for k in ("changed", "missing", "added"))
            ):
                raise ValueError("adapter training inputs were not verified")
        cases = None
        for label, entry in entries.items():
            blob = _record(root, prefix + "/" + label, entry)
            if label == "cases.jsonl":
                cases = blob
        if cases is None or _inventory(cases, split) != provenance[split]:
            raise ValueError("adapter provenance differs from actual frozen cases")
    if any(
        set(provenance["train"][key]) & set(provenance["development"][key])
        for key in ("ids", "groups", "request_fingerprints")
    ):
        raise ValueError("adapter training/development provenance overlaps")
    return identity


def _config(config):
    expected = {
        **LORA,
        "peft_type": "LORA",
        "base_model_name_or_path": BASE_ID,
        "revision": BASE_REVISION,
        "inference_mode": True,
    }
    for key, value in expected.items():
        actual = config.get(key)
        if actual != value:
            raise ValueError(f"adapter configuration pin mismatch: {key}")
    inert = {
        "modules_to_save": None,
        "use_dora": False,
        "use_rslora": False,
        "fan_in_fan_out": False,
        "rank_pattern": {},
        "alpha_pattern": {},
        "layers_to_transform": None,
        "layers_pattern": None,
        "target_parameters": None,
        "exclude_modules": None,
        "layer_replication": None,
        "trainable_token_indices": None,
        "lora_bias": False,
        "use_qalora": False,
        "ensure_weight_tying": False,
        "init_lora_weights": True,
        "loftq_config": {},
    }
    for key, value in inert.items():
        if config.get(key, value) != value:
            raise ValueError(f"unsupported adapter configuration: {key}")
    # PEFT metadata defaults are inert, but unknown executable/custom mappings
    # and all undeclared behavioral options are rejected.
    allowed = (
        set(expected)
        | set(inert)
        | {
            "auto_mapping",
            "peft_version",
            "megatron_config",
            "megatron_core",
            "qalora_group_size",
            "arrow_config",
            "runtime_config",
            "alora_invocation_tokens",
            "corda_config",
            "eva_config",
            "use_bdlora",
            "target_modules_mapping",
            "lora_ga_config",
            "velora_config",
            "monteclora_config",
            "kasa_config",
        }
    )
    if (
        set(config) - allowed
        or any(
            config.get(key) is not None
            for key in (
                "auto_mapping",
                "megatron_config",
                "arrow_config",
                "alora_invocation_tokens",
                "corda_config",
                "eva_config",
                "target_modules_mapping",
                "lora_ga_config",
                "velora_config",
                "monteclora_config",
                "kasa_config",
            )
        )
        or config.get("use_bdlora", False)
        or config.get("qalora_group_size", 16) != 16
        or config.get("runtime_config") not in (None, {"ephemeral_gpu_offload": False})
    ):
        raise ValueError("unsupported adapter configuration fields")


def _tensor_bytes(blob):
    import torch
    from safetensors.torch import load

    values = load(blob)
    pairs = {}
    for name, tensor in values.items():
        match = re.fullmatch(
            r"(base_model\.model\.model\.language_model\.layers\.[0-9]+\.self_attn\.(q_proj|v_proj))\.lora_([AB])\.weight",
            name,
        )
        if (
            not match
            or tensor.ndim != 2
            or tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("adapter weights must be finite q/v LoRA matrices only")
        key, _, side = match.groups()
        if tensor.shape[0 if side == "A" else 1] != 8 or min(tensor.shape) < 1:
            raise ValueError("adapter weight rank/shape mismatch")
        pairs.setdefault(key, set()).add(side)
    if not pairs or any(sides != {"A", "B"} for sides in pairs.values()):
        raise ValueError("adapter needs complete q/v LoRA matrix pairs")
    return {
        "tensor_count": len(values),
        "parameters": sum(t.numel() for t in values.values()),
        "stored_dtypes": sorted({str(t.dtype) for t in values.values()}),
        "tensor_shapes": {name: list(t.shape) for name, t in values.items()},
    }


@dataclass(frozen=True)
class VerifiedActionAdapter:
    manifest: dict
    manifest_sha256: str
    receipt_sha256: str
    files: dict
    tensor_metadata: dict

    def telemetry(self):
        m = self.manifest
        return {
            "kind": m["kind"],
            "manifest_sha256": self.manifest_sha256,
            "weights_sha256": m["files"]["adapter_model.safetensors"]["sha256"],
            "training_receipt_sha256": self.receipt_sha256,
            "policy": m["policy"],
            "base_model_id": m["base_model_id"],
            "base_revision": m["base_revision"],
            "training_quantization": m["training_quantization"],
            "training_dtype": m["training_dtype"],
            "deployment_quantization": m["deployment_quantization"],
            "train_cases": len(m["provenance"]["train"]["ids"]),
            "development_cases": len(m["provenance"]["development"]["ids"]),
            "source_scope": "development_only",
            **{k: v for k, v in self.tensor_metadata.items() if k != "tensor_shapes"},
        }


def _verify_action_adapter(path, *, model_id, revision, quantization):
    """Read/verify local bytes before any backbone is loaded; never uses the Hub."""
    configured = Path(path)
    if configured.is_symlink() or not configured.is_dir():
        raise ValueError("generated-action adapter must be an ordinary local directory")
    directory = configured.resolve()
    manifest_blob = _read(directory, "manifest.json", limit=4_000_000)
    m = _json(manifest_blob)
    pins = {
        "schema_version": 1,
        "kind": "jev_action_lora_v1",
        "base_model_id": BASE_ID,
        "base_revision": BASE_REVISION,
        "training_quantization": "none",
        "training_dtype": "bfloat16",
        "deployment_quantization": "nf4",
        "policy": POLICY,
        "instructions_sha256": _sha(INSTRUCTIONS.encode()),
        "inference_model_identity": "generated-actions-nf4",
        "lora": LORA,
    }
    if (
        not isinstance(m, dict)
        or set(m) != _KEYS
        or type(m["schema_version"]) is not int
        or any(m.get(k) != v for k, v in pins.items())
    ):
        raise ValueError("action adapter manifest schema/checkpoint/policy pin mismatch")
    if (model_id, revision, quantization) != (BASE_ID, BASE_REVISION, "nf4"):
        raise ValueError("action adapter requires the pinned NF4 deployment checkpoint")
    if (
        not isinstance(m["files"], dict)
        or set(m["files"]) != _FILES
        or {f.name for f in directory.iterdir()} != _FILES | {"manifest.json"}
    ):
        raise ValueError("action adapter must contain only the verified adapter files and manifest")
    files = {label: _record(directory, label, record) for label, record in m["files"].items()}
    _config(_json(files["adapter_config.json"]))
    record = m["training_receipt"]
    if (
        not isinstance(record, dict)
        or set(record) != {"path", "sha256"}
        or record["path"] != "../receipt.json"
    ):
        raise ValueError("action adapter must reference its adjacent training receipt")
    receipt_blob = _read(directory.parent, "receipt.json", limit=10_000_000)
    if _sha(receipt_blob) != record["sha256"]:
        raise ValueError("action adapter training receipt hash mismatch")
    receipt = _json(receipt_blob)
    if any(
        receipt.get(key) != value
        for key, value in pins.items()
        if key
        in (
            "base_model_id",
            "base_revision",
            "training_quantization",
            "training_dtype",
            "deployment_quantization",
            "policy",
            "instructions_sha256",
        )
    ):
        raise ValueError("adapter training receipt checkpoint/policy pins differ from manifest")
    _training_evidence(directory.parent, receipt, m["provenance"])
    tensor_metadata = _tensor_bytes(files["adapter_model.safetensors"])
    return VerifiedActionAdapter(
        copy.deepcopy(m), _sha(manifest_blob), _sha(receipt_blob), files, tensor_metadata
    )


def verify_action_adapter(path, *, model_id=BASE_ID, revision=BASE_REVISION, quantization="nf4"):
    try:
        return _verify_action_adapter(
            path, model_id=model_id, revision=revision, quantization=quantization
        )
    except (TypeError, KeyError, AttributeError, UnicodeError) as exc:
        raise ValueError("malformed generated-action adapter artifact") from exc


class _AttachedActionAdapter:
    def __init__(self, model, lm, metadata):
        self.model, self.lm, self.metadata = model, lm, metadata
        self.lock = threading.RLock()

    def _check(self):
        status = self.lm.get_model_status()
        if (
            self.model.lm is not self.lm
            or status.enabled is not True
            or status.merged_adapters != []
            or status.active_adapters != ["default"]
            or status.available_adapters != ["default"]
        ):
            raise ValueError("verified action adapter is not exclusively active")
        if any(p.requires_grad for p in self.lm.parameters()):
            raise ValueError("action inference parameters must remain frozen")

    @contextmanager
    def decision_context(self):
        with self.lock:
            self._check()
            yield

    @contextmanager
    def base_context(self):
        with self.lock:
            self._check()
            try:
                with self.lm.disable_adapter():
                    yield
            finally:
                # PEFT's enable_adapter_layers restoration can re-enable grads.
                self.lm.requires_grad_(False)
                self.lm.eval()


def attach_action_adapter(model, verified):
    """Load captured safetensor bytes locally, frozen; never merge or retune T."""
    from peft import PeftModel

    if not isinstance(verified, VerifiedActionAdapter):
        raise ValueError("action adapter must pass preflight verification")
    if (
        (model.name, model.revision, model.quantization) != (BASE_ID, BASE_REVISION, "nf4")
        or hasattr(model.lm, "peft_config")
        or hasattr(model, "action_adapter")
    ):
        raise ValueError("action adapter requires one unadapted pinned base model")
    for label, blob in verified.files.items():
        if _sha(blob) != verified.manifest["files"][label]["sha256"]:
            raise ValueError("verified adapter bytes changed after preflight")
    head, backbone, temperature = model.head, model.backbone, model.temperature
    version = head.weight._version
    with tempfile.TemporaryDirectory(prefix="jev-verified-action-") as temporary:
        for label, blob in verified.files.items():
            (Path(temporary) / label).write_bytes(blob)
        lm = PeftModel.from_pretrained(
            model.lm,
            temporary,
            is_trainable=False,
            local_files_only=True,
            autocast_adapter_dtype=True,
            torch_device=model.device,
        )
    lm.requires_grad_(False)
    lm.eval()
    loaded_shapes = {
        name.replace(".default.", "."): list(p.shape)
        for name, p in lm.named_parameters()
        if ".lora_" in name
    }
    if loaded_shapes != verified.tensor_metadata["tensor_shapes"]:
        raise ValueError("loaded action adapter tensor inventory differs from verified artifact")
    base = lm.get_base_model()
    if (
        base.get_output_embeddings() is not head
        or base.model is not backbone
        or head.weight._version != version
        or model.temperature != temperature
    ):
        raise ValueError("action adapter altered the original head/backbone/temperature descriptor")
    model.lm = lm
    metadata = {
        **verified.telemetry(),
        "all_parameters_frozen": True,
        "base_head_identity_version_preserved": True,
        "helper_policy": "adapter_disabled_pinned_base",
        "load_scope": "captured_verified_local_bytes_no_merge_no_remote",
    }
    model.action_adapter = _AttachedActionAdapter(model, lm, metadata)
    model.action_adapter._check()
    return copy.deepcopy(metadata)


def base_field_text(model, context):
    from .browser_text import generate_field_text

    adapter = getattr(model, "action_adapter", None)
    if adapter is None:
        return generate_field_text(model, context)
    with adapter.base_context():
        try:
            text, metadata = generate_field_text(model, context)
        except BaseException as exc:
            if isinstance(getattr(exc, "generation", None), dict):
                exc.generation["action_adapter_state"] = "disabled_for_pinned_base_helper"
            raise
        return text, {**metadata, "action_adapter_state": "disabled_for_pinned_base_helper"}
