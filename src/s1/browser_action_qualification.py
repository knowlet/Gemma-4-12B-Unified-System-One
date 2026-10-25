"""Explicit BF16 qualification of an unchanged, strictly verified NF4 artifact.

Qualification authenticates a particular exposed diagnostic and runtime profile;
it is neither promotion nor evidence of public or unseen-task performance.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from . import browser_action_adapter as artifact
from .browser_evidence import _source_paths

HEAD_SHA256 = "9619fd670a4074367c72f2cf09d777970ad1eafa511bff605a82c8558dcc24d4"
BASE_SHA256 = "11144621122123135eb79c9cfd0466a2c51f230109124aefe401cdbec8c71a53"
BASE_SIZE = 23_919_549_408
TENSOR_COUNT = 176
TEMPERATURE = 2.7830344470383452
PACKAGES = ("torch", "transformers", "peft", "safetensors", "tokenizers", "playwright", "uvicorn")
ARMS = ("candidate-original", "base13", "candidate13")
BASE_METADATA = {
    "artifact-manifest.json",
    "config.json",
    "generation_config.json",
    "s1_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
}
SCOPE = "original plus known exposed13 development diagnostics; no promotion or public/generalization claim"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def payload_bytes(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def file_record(path):
    path = Path(path)
    require(
        not path.is_symlink() and path.is_file() and path.stat().st_size <= 256_000_000,
        "qualification evidence must be an ordinary bounded file",
    )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def inventory(root):
    root = artifact._ordinary_directory(root)
    paths = sorted(root.rglob("*"))
    require(
        len(paths) <= 50_000 and not any(path.is_symlink() for path in paths),
        "qualification evidence inventory is excessive or symlinked",
    )
    return {
        path.relative_to(root).as_posix(): file_record(path) for path in paths if path.is_file()
    }


def source_inventory():
    paths = _source_paths()
    paths["scripts/jev_qualify_bf16_action_adapter.py"] = (
        Path(__file__).resolve().parents[2] / "scripts/jev_qualify_bf16_action_adapter.py"
    )
    return {label: file_record(path) for label, path in paths.items()}


def versions():
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in PACKAGES},
    }


def runtime_profile():
    return {
        "model": artifact.BASE_ID,
        "revision": artifact.BASE_REVISION,
        "quantization": "none",
        "compute_dtype": "bfloat16",
        "head_dtype": "bfloat16",
        "head_sha256": HEAD_SHA256,
        "temperature": TEMPERATURE,
        "base_sha256": BASE_SHA256,
        "base_size_bytes": BASE_SIZE,
        "tensor_count": TENSOR_COUNT,
        "adapter_dtype": "float32",
        "lora": artifact.LORA,
        "model_alias": "generated-actions-none",
        "policy": artifact.POLICY,
        "instructions_sha256": hashlib.sha256(artifact.INSTRUCTIONS.encode()).hexdigest(),
        "constraint": "legal_action_token_trie_v1",
        "max_context": 8192,
        "action_tokens": 32,
        "helper_tokens": 128,
        "policy_seconds": 20,
        "http_seconds": 25,
        "do_sample": False,
        "num_beams": 1,
        "attention": "sdpa",
        "cudnn_sdp_enabled": False,
        "offline": True,
        "helper": "same unadapted BF16 base; disable under shared lock then restore frozen/eval",
    }


def _read_json(root, label):
    return artifact._json(artifact._read(root, label))


def verify_audit_inputs(root, audit):
    require(
        file_record(root / "audit.py")["sha256"] == audit["script_sha256"],
        "qualified audit source differs",
    )
    for label, expected in audit["read_inputs"].items():
        require(
            Path(label).name != "model.safetensors", "base weights are not qualification evidence"
        )
        artifact._record(root / "audit-inputs", label, expected)


def normalized_initial(payload):
    """Normalize only transport model alias and a URL's ephemeral loopback port."""
    result = copy.deepcopy(payload)
    result["model"] = "<model-alias>"
    page = result["state"]["page"]
    page["url"] = re.sub(
        r"^(https?://(?:127\.0\.0\.1|localhost|\[::1\])):[0-9]+(?=/|$)", r"\1:<port>", page["url"]
    )
    return result


def campaign_facts(root, audit, pins):
    """Derive counts from raw summaries; never accept operator-supplied pass counts."""
    root = artifact._ordinary_directory(root)
    state = _read_json(root, "state.json")
    plan = _read_json(root, "plan.json")
    base_metadata = {
        Path(label).name: item
        for label, item in plan["source_input_pins"].items()
        if Path(label).name in BASE_METADATA
        and Path(label).parent.name == artifact.BASE_REVISION
        and "snapshots" in Path(label).parts
    }
    require(
        set(base_metadata) == BASE_METADATA, "audited processor/tokenizer metadata is incomplete"
    )
    require(
        state.get("stage") == "complete" and not state.get("error"),
        "qualification requires terminal complete campaign",
    )
    require(
        audit.get("evidence_checks_passed") is True
        and audit.get("all_infrastructure_complete") is True,
        "independent campaign audit did not pass",
    )
    require(
        audit.get("checks") and all(row.get("success") is True for row in audit["checks"]),
        "independent audit contains failed checks",
    )
    require(state == audit["controller_state"], "audited terminal controller state differs")
    require(
        hashlib.sha256(payload_bytes(plan)).hexdigest() == audit["plan_sha256"]
        and plan["controller_sha256"] == audit["controller_sha256"],
        "audited controller/plan identity differs",
    )
    require(
        plan["model"] == artifact.BASE_ID
        and plan["revision"] == artifact.BASE_REVISION
        and plan["quantization"] == "none"
        and plan["precision"] == "bfloat16",
        "diagnostic plan runtime differs",
    )
    require(
        plan["candidate"]["manifest_sha256"] == pins["manifest_sha256"]
        and plan["candidate"]["receipt_sha256"] == pins["training_receipt_sha256"]
        and plan["candidate"]["weights_sha256"] == pins["weights_sha256"]
        and plan["fits"]["child"]["inventory"] == pins["inventory"],
        "diagnostic plan qualified fit differs",
    )
    audited = {row["arm"]: row for row in audit["actual_arms"]}
    require(
        len(audit["actual_arms"]) == 3 and set(audited) == set(ARMS),
        "audit must cover exactly all three diagnostic arms",
    )
    packages = versions()
    require(
        packages["tokenizers"] == audit["tokenizers_version"],
        "qualified tokenizer package differs from audit",
    )
    actual, first = {}, {}
    for arm in ARMS:
        receipt_path = root / arm / "receipt.json"
        receipt = artifact._json(receipt_path.read_bytes())
        require(
            receipt["controller_sha256"] == audit["controller_sha256"]
            and receipt["plan_sha256"] == audit["plan_sha256"]
            and _read_json(root / arm, "prospective-plan.json") == plan,
            "arm controller/plan differs",
        )
        require(
            receipt["quantization"] == "none"
            and receipt["precision"] == "bfloat16"
            and receipt["quantization_details"] == {"method": "none", "quantized_linear_modules": 0}
            and receipt["model_reported"] == artifact.BASE_ID
            and receipt["revision_reported"] == artifact.BASE_REVISION
            and receipt["temperature"] == TEMPERATURE,
            "audited BF16 runtime differs",
        )
        require(
            all(
                receipt["versions"][name] == packages[name]
                for name in (
                    "torch",
                    "transformers",
                    "peft",
                    "safetensors",
                    "playwright",
                    "uvicorn",
                )
            ),
            "qualified model package versions differ from diagnostic",
        )
        metadata = receipt["diagnostic_action_adapter"]
        if arm == "base13":
            require(
                metadata is None and receipt["initial_invariants"]["unadapted_base"] is True,
                "diagnostic baseline acquired adapter",
            )
        else:
            require(
                all(
                    metadata[key] == pins[key]
                    for key in ("manifest_sha256", "training_receipt_sha256", "weights_sha256")
                )
                and metadata["backbone_quantization"] == "none"
                and metadata["production_attachment"] is False,
                "diagnostic candidate qualification binding differs",
            )
        require(
            inventory(root / arm / "fit-captures/child") == pins["inventory"],
            "diagnostic copied qualified fit differs",
        )
        require(
            all(
                receipt[key]["success"] is True and receipt[key]["full_head_sha256"] == HEAD_SHA256
                for key in ("initial_invariants", "final_invariants")
            ),
            "audited BF16 full head invariants differ",
        )
        require(
            receipt.get("stage") == "complete"
            and not receipt.get("error")
            and not receipt.get("finalization_errors")
            and receipt.get("helper_integrity_errors") == [],
            "diagnostic infrastructure/helper integrity failed",
        )
        require(
            audited[arm].get("infrastructure_complete") is True
            and audited[arm]["receipt_sha256"] == file_record(receipt_path)["sha256"],
            "audit receipt binding differs",
        )
        require(
            inventory(root / arm) == audited[arm]["inventory"],
            "audited complete arm inventory differs",
        )
        outcomes, requests = [], {}
        for suite in ["original"] if arm == "candidate-original" else ["ten", "challenges"]:
            base = root / arm / "benchmark" / suite
            summary = _read_json(base, "summary.json")
            require(summary in receipt["summaries"], "raw summary is absent from arm receipt")
            for outcome in summary["outcomes"]:
                require(
                    outcome["success"]
                    == (
                        outcome["goal_satisfied"]
                        and outcome["status"] == "done"
                        and outcome["error"] is None
                    ),
                    "strict outcome accounting differs",
                )
                outcomes.append({"case_id": outcome["case_id"], "success": outcome["success"]})
                calls = _read_json(base, "attempts/" + outcome["case_id"] + "/model_calls.json")
                policy = next(call for call in calls if call["kind"] == "policy")
                requests[suite + "/" + outcome["case_id"]] = policy["request"]
        count = 1 if arm == "candidate-original" else 13
        require(len(outcomes) == len(requests) == count, "diagnostic corpus differs")
        actual[arm] = {
            "attempts": count,
            "successes": sum(row["success"] for row in outcomes),
            "outcomes": outcomes,
        }
        require(
            actual[arm]["attempts"] == audited[arm]["attempts"]
            and actual[arm]["successes"] == audited[arm]["successes"],
            "audit outcome counts differ",
        )
        first[arm] = requests
    require(actual["candidate-original"]["successes"] == 1, "original diagnostic gate did not pass")
    require(
        actual["candidate13"]["successes"] == 13
        and all(row["success"] is True for row in actual["candidate13"]["outcomes"]),
        "candidate13 strict qualification gate did not pass",
    )
    require(
        set(first["base13"]) == set(first["candidate13"]) and len(first["base13"]) == 13,
        "paired diagnostic cases differ",
    )
    proof = audit["paired_initial_payloads_and_fixtures"]
    require(len(proof) == 13, "independent paired13 proof length differs")
    proof = {row["suite"] + "/" + row["case_id"]: row for row in proof}
    require(
        set(proof) == set(first["base13"]) and len(proof) == 13,
        "independent paired13 proof missing",
    )
    paired = {}
    for key, left in first["base13"].items():
        right, row = first["candidate13"][key], proof[key]
        normalized = payload_bytes(normalized_initial(left))
        require(
            normalized
            == payload_bytes(normalized_initial(right))
            == row["normalized_initial_payload_bytes"].encode()
            and row["normalized_initial_payload_byte_equal"] is True,
            "paired initial payload bytes differ",
        )
        digest = hashlib.sha256(normalized).hexdigest()
        require(
            digest == row["normalized_initial_payload_sha256"],
            "paired normalized payload hash differs",
        )
        require(
            row["raw_initial_payload_sha256"]
            == {
                arm: hashlib.sha256(payload_bytes(first[arm][key])).hexdigest()
                for arm in ("base13", "candidate13")
            },
            "paired raw payload hashes differ",
        )
        for filename, field in (("page.html", "fixture_html"), ("judge.json", "private_judge")):
            paths = [
                root / arm / "benchmark" / row["suite"] / "fixtures" / row["case_id"] / filename
                for arm in ("base13", "candidate13")
            ]
            require(
                paths[0].read_bytes() == paths[1].read_bytes()
                and row[field + "_byte_equal"] is True
                and file_record(paths[0])["sha256"] == row[field + "_sha256"],
                "paired fixture/judge bytes differ",
            )
        paired[key] = digest
    return {
        "scope": SCOPE,
        "attempts": 27,
        "distinct_paired_cases": 13,
        "original_gate_duplicate_attempts": 1,
        "arms": actual,
        "paired_initial_sha256": paired,
        "base_metadata": base_metadata,
    }


@dataclass(frozen=True)
class VerifiedBF16ActionQualification:
    artifact: artifact.VerifiedActionAdapter
    qualification_bytes: bytes
    qualification_sha256: str
    declaration: dict
    artifact_identity: str
    bundle_root: Path | None = None
    bundle_inventory: dict | None = None

    def telemetry(self):
        return {
            "kind": "qualified_bf16_exposed_diagnostic_v1",
            "qualification_sha256": self.qualification_sha256,
            "qualification_scope": SCOPE,
            "production_attachment": False,
            "tracked_cli_route_parity": "not attested by the historical diagnostic qualification",
            "artifact_deployment_contract": "nf4",
            "runtime_quantization": "none",
            "compute_precision": "bfloat16",
            "model_alias": "generated-actions-none",
            "manifest_sha256": self.artifact.manifest_sha256,
            "weights_sha256": self.artifact.manifest["files"]["adapter_model.safetensors"][
                "sha256"
            ],
            "training_receipt_sha256": self.artifact.receipt_sha256,
            "observed_diagnostics": copy.deepcopy(self.declaration["observed_diagnostics"]),
            "helper_policy": "adapter_disabled_pinned_bf16_base",
        }


def _verify_bf16_action_qualification(adapter_path, qualification_path):
    path = Path(qualification_path)
    root = artifact._ordinary_directory(path.parent)
    blob = artifact._read(root, path.name, limit=10_000_000)
    declaration = artifact._json(blob)
    require(
        set(declaration)
        == {
            "schema_version",
            "kind",
            "scope",
            "runtime",
            "runtime_sources",
            "versions",
            "artifact",
            "evidence",
            "observed_diagnostics",
        }
        and type(declaration["schema_version"]) is int
        and declaration["schema_version"] == 1
        and declaration["kind"] == "jev_action_bf16_qualification_v1",
        "BF16 qualification schema differs",
    )
    require(
        declaration["scope"] == SCOPE
        and canonical(declaration["runtime"]) == canonical(runtime_profile()),
        "BF16 qualification runtime/scope differs",
    )
    require(
        declaration["runtime_sources"] == source_inventory()
        and declaration["versions"] == versions(),
        "BF16 qualification source/package versions differ",
    )
    fit = artifact._ordinary_directory(Path(adapter_path).parent)
    require(
        inventory(fit) == declaration["artifact"]["inventory"],
        "qualified complete fit inventory differs",
    )
    evidence = root / "evidence"
    require(
        inventory(evidence) == declaration["evidence"]["inventory"],
        "qualified evidence inventory differs",
    )
    require(
        file_record(evidence / "audit.json")["sha256"] == declaration["evidence"]["audit_sha256"],
        "qualified independent audit hash differs",
    )
    audit = _read_json(evidence, "audit.json")
    verify_audit_inputs(evidence, audit)
    require(
        campaign_facts(evidence / "campaign", audit, declaration["artifact"])
        == declaration["observed_diagnostics"],
        "qualification diagnostic accounting differs",
    )
    verified = artifact.verify_action_adapter(adapter_path)
    pins = declaration["artifact"]
    require(
        verified.manifest_sha256 == pins["manifest_sha256"]
        and verified.receipt_sha256 == pins["training_receipt_sha256"]
        and verified.manifest["files"]["adapter_model.safetensors"]["sha256"]
        == pins["weights_sha256"],
        "qualification artifact binding differs",
    )
    require(
        verified.tensor_metadata["tensor_count"] == TENSOR_COUNT
        and verified.tensor_metadata["stored_dtypes"] == ["torch.float32"],
        "qualified adapter tensor inventory differs",
    )
    expected = {"qualification.json": file_record(path)}
    expected.update(
        {"fit/" + label: item for label, item in declaration["artifact"]["inventory"].items()}
    )
    expected.update(
        {"evidence/" + label: item for label, item in declaration["evidence"]["inventory"].items()}
    )
    require(
        path.name == "qualification.json" and inventory(root) == expected,
        "qualified bundle file inventory differs",
    )
    require(
        inventory(root / "fit") == declaration["artifact"]["inventory"],
        "qualified bundled fit differs",
    )
    return VerifiedBF16ActionQualification(
        verified,
        blob,
        hashlib.sha256(blob).hexdigest(),
        declaration,
        _artifact_identity(verified),
        root,
        expected,
    )


def verify_bf16_action_qualification(adapter_path, qualification_path):
    try:
        return _verify_bf16_action_qualification(adapter_path, qualification_path)
    except (KeyError, TypeError, AttributeError, StopIteration, UnicodeError) as exc:
        raise ValueError("malformed BF16 qualification") from exc


def capture_bf16_qualification(verified, destination):
    """Preserve and bind the complete qualification before backbone construction."""
    require(
        inventory(verified.bundle_root) == verified.bundle_inventory,
        "qualification bundle changed before capture",
    )
    destination = Path(destination)
    require(
        not destination.exists() and not destination.resolve().is_relative_to(verified.bundle_root),
        "qualification capture must be new and outside its origin",
    )
    shutil.copytree(verified.bundle_root, destination)
    return verify_bf16_qualification_capture(verified, destination)


def verify_bf16_qualification_capture(verified, destination):
    require(
        verified.declaration["runtime_sources"] == source_inventory()
        and verified.declaration["versions"] == versions(),
        "qualified runtime sources/packages changed",
    )
    require(
        inventory(verified.bundle_root) == verified.bundle_inventory
        and inventory(destination) == verified.bundle_inventory,
        "original/copied qualification evidence changed",
    )
    return {
        "success": True,
        "qualification_sha256": verified.qualification_sha256,
        "inventory_sha256": hashlib.sha256(canonical(verified.bundle_inventory)).hexdigest(),
        "files": verified.bundle_inventory,
    }


def _artifact_identity(verified):
    return hashlib.sha256(
        canonical(
            {
                "manifest": verified.manifest,
                "metadata": verified.tensor_metadata,
                "files": {
                    key: hashlib.sha256(value).hexdigest() for key, value in verified.files.items()
                },
            }
        )
    ).hexdigest()


def verify_bf16_base_checkpoint(verified):
    """Explicit local-only exact-byte check before a future backbone load."""
    from huggingface_hub import hf_hub_download

    for filename, expected in verified.declaration["observed_diagnostics"]["base_metadata"].items():
        metadata = Path(
            hf_hub_download(
                artifact.BASE_ID, filename, revision=artifact.BASE_REVISION, local_files_only=True
            )
        ).resolve()
        require(
            file_record(metadata) == expected,
            "qualified processor/tokenizer metadata differs: " + filename,
        )
    path = Path(
        hf_hub_download(
            artifact.BASE_ID,
            "model.safetensors",
            revision=artifact.BASE_REVISION,
            local_files_only=True,
        )
    )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    require(
        path.stat().st_size == BASE_SIZE and digest.hexdigest() == BASE_SHA256,
        "qualified base checkpoint bytes differ",
    )
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _head_hash(weight):
    import torch

    digest = hashlib.sha256()
    flat = weight.detach().flatten()
    for offset in range(0, flat.numel(), 1_000_000):
        digest.update(
            flat[offset : offset + 1_000_000].cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


class _BF16Guard:
    def __init__(self, model, qualification, values):
        self.model, self.qualification, self.values = model, qualification, values
        self.lm, self.head, self.backbone = model.lm, model.head, model.backbone
        self.weight = model.head.weight
        self.parameters = {
            name: (value, value._version, value.dtype)
            for name, value in model.lm.named_parameters()
        }
        self.configs = {
            name: copy.deepcopy(config.to_dict()) for name, config in model.lm.peft_config.items()
        }
        self.lock, self.integrity_errors = threading.RLock(), []
        self.metadata = qualification.telemetry()

    def _check(self, *, disabled=False):
        import torch

        model, status = self.model, self.lm.get_model_status()
        require(
            model.lm is self.lm
            and model.head is self.head
            and model.head.weight is self.weight
            and model.backbone is self.backbone,
            "qualified BF16 model/head/backbone identity changed",
        )
        require(
            (model.name, model.revision, model.quantization, model.max_context)
            == (artifact.BASE_ID, artifact.BASE_REVISION, "none", 8192),
            "qualified BF16 descriptor changed",
        )
        require(
            model.quantization_details == {"method": "none", "quantized_linear_modules": 0}
            and model.attn_implementation == "sdpa"
            and not torch.backends.cuda.cudnn_sdp_enabled(),
            "qualified BF16 quantization/attention changed",
        )
        require(
            model.temperature == TEMPERATURE and self.weight.dtype == torch.bfloat16,
            "qualified BF16 head dtype/temperature changed",
        )
        require(
            status.enabled is (False if disabled else True)
            and status.active_adapters == ["default"]
            and status.available_adapters == ["default"]
            and status.merged_adapters == []
            and set(self.lm.peft_config) == {"default"},
            "qualified BF16 PEFT status changed",
        )
        require(
            {name: config.to_dict() for name, config in self.lm.peft_config.items()}
            == self.configs,
            "qualified BF16 PEFT configuration changed",
        )
        current = dict(self.lm.named_parameters())
        require(
            set(current) == set(self.parameters)
            and all(
                current[name] is value
                and value._version == version
                and value.dtype == dtype
                and not value.requires_grad
                for name, (value, version, dtype) in self.parameters.items()
            ),
            "qualified BF16 parameters changed or became trainable",
        )
        require(
            not any(module.training for module in self.lm.modules()),
            "qualified BF16 entered training mode",
        )
        base = self.lm.get_base_model()
        require(
            base.get_output_embeddings() is self.head and base.model is self.backbone,
            "qualified BF16 base/head references changed",
        )

    def check(self, *, full=False):
        try:
            self._check()
            head_sha256 = None
            if full:
                import torch
                from peft import get_peft_model_state_dict

                loaded = get_peft_model_state_dict(
                    self.lm, adapter_name="default", save_embedding_layers=False
                )
                require(
                    set(loaded) == set(self.values)
                    and all(
                        value.dtype == torch.float32
                        and torch.equal(value.detach().cpu(), self.values[name])
                        for name, value in loaded.items()
                    ),
                    "qualified BF16 adapter tensors changed",
                )
                head_sha256 = _head_hash(self.weight)
                require(head_sha256 == HEAD_SHA256, "qualified BF16 full head hash changed")
            require(not self.integrity_errors, "qualified BF16 retained integrity failures")
            return {
                "success": True,
                "full_tensor_check": full,
                "qualification_sha256": self.qualification.qualification_sha256,
                "full_head_sha256": head_sha256,
                "head_version": self.weight._version,
                "head_identity": id(self.head),
                "head_weight_identity": id(self.weight),
                "backbone_identity": id(self.backbone),
                "temperature": self.model.temperature,
                "parameter_count": len(self.parameters),
                "all_parameter_versions_unchanged": True,
                "all_parameters_frozen_eval": True,
                "peft_enabled_default_unmerged": True,
                "exact_fp32_adapter_tensors": full,
            }
        except BaseException as exc:
            self.integrity_errors.append({"type": type(exc).__name__, "message": str(exc)})
            raise

    @contextmanager
    def decision_context(self):
        with self.lock:
            self.check()
            primary = None
            try:
                yield
            except BaseException as exc:
                primary = exc
                raise
            finally:
                try:
                    self.check()
                except BaseException as exc:
                    # check() retains the integrity failure even if decoding already failed.
                    if primary is not None:
                        primary.add_note("BF16 decision exit check also failed: " + repr(exc))
                    else:
                        raise

    @contextmanager
    def base_context(self):
        with self.lock:
            self.check()
            primary = None
            helper_error = None
            try:
                with self.lm.disable_adapter():
                    try:
                        self._check(disabled=True)
                    except BaseException as exc:
                        self.integrity_errors.append(
                            {"type": type(exc).__name__, "message": str(exc)}
                        )
                        raise
                    try:
                        yield
                    except BaseException as exc:
                        helper_error = exc
                        raise
            except BaseException as exc:
                # Ordinary helper validation failures are preserved by base_field_text.
                primary = exc
                if exc is not helper_error:
                    self.integrity_errors.append({"type": type(exc).__name__, "message": str(exc)})
                raise
            finally:
                try:
                    self.lm.requires_grad_(False)
                    self.lm.eval()
                    self.check()
                except BaseException as exc:
                    self.integrity_errors.append({"type": type(exc).__name__, "message": str(exc)})
                    if primary is not None:
                        primary.add_note("BF16 helper restoration also failed: " + repr(exc))
                    else:
                        raise


def attach_qualified_bf16_action_adapter(model, qualification):
    """Same exercised BF16 initializer, followed by frozen inference guards."""
    import torch
    from safetensors.torch import load

    require(
        isinstance(qualification, VerifiedBF16ActionQualification),
        "BF16 attachment requires verified qualification",
    )
    require(
        hashlib.sha256(qualification.qualification_bytes).hexdigest()
        == qualification.qualification_sha256
        and artifact._json(qualification.qualification_bytes) == qualification.declaration,
        "verified qualification was mutated",
    )
    require(
        _artifact_identity(qualification.artifact) == qualification.artifact_identity,
        "verified BF16 artifact was mutated",
    )
    require(
        qualification.declaration["runtime"] == runtime_profile(), "qualified BF16 runtime changed"
    )
    require(not hasattr(model, "action_adapter"), "qualified BF16 cannot replace an attachment")
    require(
        _head_hash(model.head.weight) == HEAD_SHA256 and model.temperature == TEMPERATURE,
        "qualified BF16 original head/temperature differs",
    )
    for config in model.lm.peft_config.values():
        config.base_model_name_or_path, config.revision = artifact.BASE_ID, artifact.BASE_REVISION
        checked = copy.deepcopy(config.to_dict())
        checked["inference_mode"] = True
        artifact._config(checked)
    try:
        initialization = artifact.initialize_training_action_adapter(model, qualification.artifact)
    finally:
        model.lm.requires_grad_(False)
        model.lm.eval()
    values = load(qualification.artifact.files["adapter_model.safetensors"])
    require(
        len(values) == TENSOR_COUNT
        and all(value.dtype == torch.float32 for value in values.values()),
        "qualified BF16 requires exact FP32 q/v inventory",
    )
    guard = _BF16Guard(model, qualification, values)
    guard.initial_invariants = guard.check(full=True)
    guard.initialization = initialization
    guard.metadata["all_parameters_frozen"] = True
    model.action_adapter = guard
    return copy.deepcopy(guard.metadata)
