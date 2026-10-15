#!/usr/bin/env python3
"""Fit the separately authenticated full actual07 public-native action corpus.

Defaults describe the next120-step development candidate. --run, reviewed source
handoff and fresh CPU/BF16 resource gates are mandatory. This command does not
qualify, attach, serve, benchmark or promote the resulting public artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import jev_public_action_artifact as artifact

ROOT = Path(__file__).resolve().parents[1]
BASE_BYTES = 23919549408
BASE_SHA = "11144621122123135eb79c9cfd0466a2c51f230109124aefe401cdbec8c71a53"
CACHE = ROOT / "artifacts/jev-local-hf/hub/models--knowlet--Gemma-4-12B-Unified-System-One/snapshots" / artifact.REVISION
MAX_RESOURCE_GATES = 512
# Seven setup gates, every compile, optimizer setup,22 dev forwards and four
# publication gates, including the final semantic child tensor verification.
FIXED_RESOURCE_GATES = 7 + sum(artifact.COUNTS.values()) + 1 + artifact.COUNTS["development"] + 4
MAX_TRAINING_STEPS = MAX_RESOURCE_GATES - FIXED_RESOURCE_GATES


def memory_observation():
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable"}:
            number, unit = value.split()
            artifact.require(unit == "kB", "Unexpected memory units")
            memory[name] = int(number) * 1024
    memory["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return memory


class ResourceGate:
    """Each observation is fsynced before its protected stage; no residency credit."""
    def __init__(self, path, expected_boot_id, *, observe=memory_observation):
        self.path, self.expected_boot_id, self.observe = artifact.ordinary(path), expected_boot_id, observe
        self.sequence, self.bytes = 0, 0
        artifact.require(bool(expected_boot_id) and not self.path.exists(), "New same-boot resource journal required")
        artifact.create(self.path, b"")

    def check(self, phase, scope):
        artifact.require(scope == artifact.resource_phase_scope(phase) and self.sequence < MAX_RESOURCE_GATES, "Bounded exact protected resource phase")
        record = {"sequence": self.sequence, "phase": phase, "scope": scope,
                  "UTC": datetime.now(UTC).isoformat(), "monotonic": time.monotonic(),
                  "expected_boot_id": self.expected_boot_id, "admitted": False,
                  "swap_or_resident_credit_bytes": 0}
        primary = None
        try:
            observation = self.observe()
            artifact.require(type(observation["MemTotal"]) is int and type(observation["MemAvailable"]) is int
                             and 0 < observation["MemTotal"] and 0 < observation["MemAvailable"], "Integer positive memory observation")
            reserve = max(24 * artifact.GIB, (observation["MemTotal"] + 4) // 5)
            load = 2 * BASE_BYTES + reserve
            required = {"cpu": reserve + 4 * artifact.GIB, "load": load,
                        "train": load + 30 * artifact.GIB + 256 * 1024**2}[scope]
            record.update(observation=observation, reserve_bytes=reserve, required_bytes=required)
            artifact.require(observation["boot_id"] == self.expected_boot_id
                and 0 < required <= observation["MemAvailable"] <= observation["MemTotal"],
                "Fresh same-boot " + scope + " memory admission refused")
            record["admitted"] = True
        except BaseException as exc:
            primary = exc
        record["error"] = None if primary is None else {"type": type(primary).__name__, "message": str(primary)}
        try:
            blob = artifact.encode(record) + b"\n"
            artifact.require(len(blob) <= 16384, "Resource journal row cap")
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
            try:
                artifact.require(os.fstat(fd).st_size == self.bytes, "Owned resource journal changed")
                artifact.require(os.write(fd, blob) == len(blob), "Short resource journal write")
                os.fsync(fd)
            finally:
                os.close(fd)
            self.sequence += 1
            self.bytes += len(blob)
        except BaseException as secondary:
            if primary is not None:
                primary.add_note("Resource refusal persistence failed: " + repr(secondary))
                raise primary from secondary
            raise
        if primary is not None:
            raise primary
        return record


def default_paths():
    runtime = ROOT / "artifacts/jev-runtime"
    admission = ROOT / "artifacts/jev-browser/20261016-public-native-training-07-admission-01"
    preflight = runtime / "public-native-action-fit-preflight-07-root-20261016-01"
    reference_root = runtime / "public-native-action-compiler-07-root-20261016-01"
    metadata = {
        "admission-receipt.json": admission / "receipt.json",
        "admission-provenance.json": admission / "admission-provenance.json",
        "admission-plan.json": admission / "plan.json",
        "preflight-receipt.json": preflight / "stage-01/preflight-receipt.json",
        "preflight-root-receipt.json": preflight / "root-receipt-01.json",
        "syntax-audit.json": runtime / "public-native-observation-gap-audit-20261016-01/actual07-H02-01/report.json",
        "compiler-root-receipt.json": reference_root / "root-receipt-01.json",
        "compiler-audit.json": runtime / "public-native-action-compiler-07-postrun-audit-20261016-01/report-01.json",
    }
    return {"public_root": ROOT / "artifacts/jev-browser/20261016-public-native-training-collection-07",
            "preflight_root": preflight, "admission_root": admission,
            "reference_compile": reference_root / "actual-compile-01", "metadata_paths": metadata}


def replace_json(path, value):
    temporary = path.with_name(path.name + "." + str(time.monotonic_ns()) + ".tmp")
    artifact.create(temporary, artifact.encode(value) + b"\n")
    os.replace(temporary, path)


def check_cap(output, *, pending_bytes=0, pending_files=0):
    sizes = artifact.tree_sizes(output)
    artifact.require(sum(sizes.values()) + pending_bytes + 64 * 1024**2 + 1024**2 <= artifact.GIB
                     and len(sizes) + pending_files + 256 + 8 <= artifact.MAX_FILES,
                     "Whole public/parent/source/compiled/adapter cap")
    return {"bytes": sum(sizes.values()), "files": len(sizes),
            "reserved_bytes": 64 * 1024**2 + 1024**2, "reserved_files": 264,
            "max_bytes": artifact.GIB, "max_files": artifact.MAX_FILES}


def _fresh_environment():
    artifact.require(not any(name.split(".")[0] in {"torch", "transformers", "s1", "peft", "tokenizers"}
                            for name in sys.modules), "Fresh pre-project/framework fitter process required")
    expected = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "HF_HUB_CACHE": str(ROOT / "artifacts/jev-local-hf/hub")}
    artifact.require(all(os.environ.get(k) == v for k, v in expected.items()), "Exact offline local-cache/two-thread environment required")


def _construct_model(seed):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.backends.cuda.enable_cudnn_sdp(False)
    model_type = artifact.trusted_module("s1.unified", "src/s1/unified.py").UnifiedDecisionModel
    return model_type(artifact.MODEL_ID, revision=artifact.REVISION, device="cuda", precision="bfloat16",
                      quantization="none", max_context=8192, temperature=artifact.TEMPERATURE,
                      attn_implementation="sdpa", lora={**artifact.LORA, "revision": artifact.REVISION})


def _model_check(model, kernel):
    import torch
    artifact.require((model.name, model.revision, model.quantization, model.temperature, model.max_context)
        == (artifact.MODEL_ID, artifact.REVISION, "none", artifact.TEMPERATURE, 8192)
        and model.head.weight.dtype == torch.bfloat16 and not model.head.weight.requires_grad
        and kernel._tensor_sha(model.head.weight) == artifact.HEAD_SHA,
        "Exact BF16 base/fullhead/temperature/context required")
    trainable = {name: value for name, value in model.lm.named_parameters() if value.requires_grad}
    artifact.require(len(trainable) == 176 and sum(value.numel() for value in trainable.values()) == 5193728
        and all(kernel.TRAINABLE_NAME.fullmatch(name) is not None and value.dtype == torch.float32
                and torch.isfinite(value).all() for name, value in trainable.items()), "Only176 finite FP32 q/v LoRA trainables")
    artifact.require(all(value.dtype == torch.bfloat16 for _, value in model.lm.named_parameters() if not value.requires_grad)
        and set(model.lm.peft_config) == {"default"}
        and model.lm.get_base_model().get_output_embeddings() is model.head
        and model.lm.get_base_model().model is model.backbone, "Original BF16 base/head and single default adapter identity")


def _parent_checks(captured):
    api = artifact.trusted_module("s1.browser_action_adapter", "src/s1/browser_action_adapter.py")
    for key in ("original_root", "frozen_root"):
        artifact.require(api.verify_initial_action_adapter_copy(captured[key], captured["descriptor"]["files"])["success"], "Original/copied full parent changed")


def publish_candidate(output, manifest, receipt, *, inputs, captured, gate):
    """Last operation is atomic marker replacement; no power-loss acknowledgement."""
    output = Path(output)
    inputs.check()
    _parent_checks(captured)
    receipt["output_inventory_excluding_receipt_and_manifest"] = {
        label: artifact.pin(output / label) for label in artifact.tree_sizes(output)
        if label not in {"receipt.json", "adapter/manifest.json"}}
    receipt["stage"], receipt["error"] = "complete", None
    receipt["final_bound"] = check_cap(output, pending_bytes=4 * 1024**2, pending_files=2)
    replace_json(output / "receipt.json", receipt)
    receipt_blob = artifact.read(output / "receipt.json")
    manifest["training_receipt"] = {"path": "../receipt.json", "sha256": artifact.sha(receipt_blob)}
    blob = artifact.encode(manifest) + b"\n"
    artifact._verify_public_material(output / "adapter", blob,
        expected_receipt_sha256=artifact.sha(receipt_blob), expected_source_handoff=inputs.source_handoff_sha256,
        gate=gate, _producer_journal=True)
    # Fresh semantic/tensor verification above appended genuine resource gates.
    # Seal those bytes now; no protected runtime operation occurs after sealing.
    receipt["output_inventory_excluding_receipt_and_manifest"] = {
        label: artifact.pin(output / label) for label in artifact.tree_sizes(output)
        if label not in {"receipt.json", "adapter/manifest.json"}}
    receipt["final_bound"] = check_cap(output, pending_bytes=4 * 1024**2, pending_files=2)
    replace_json(output / "receipt.json", receipt)
    receipt_blob = artifact.read(output / "receipt.json")
    manifest["training_receipt"] = {"path": "../receipt.json", "sha256": artifact.sha(receipt_blob)}
    blob = artifact.encode(manifest) + b"\n"
    artifact.validate_manifest(manifest)
    artifact.validate_training_receipt(receipt, manifest["provenance"], expected_tensor_shapes=manifest["tensor_metadata"]["tensor_shapes"])
    artifact.validate_resource_evidence(artifact.read(output / "resource-gates.jsonl"), steps=receipt["steps"])
    inputs.check()
    _parent_checks(captured)
    for label, entry in receipt["output_inventory_excluding_receipt_and_manifest"].items():
        artifact.require(artifact.pin(output / label) == entry, "Final sealed artifact bytes changed")
    artifact.require(artifact.read(output / "receipt.json") == receipt_blob, "Final receipt changed")
    check_cap(output, pending_bytes=len(blob), pending_files=1)
    marker = output / "adapter/manifest.json"
    artifact.require(not marker.exists(), "Valid marker must not already exist")
    temporary = output / ("adapter/manifest." + str(time.monotonic_ns()) + ".tmp")
    artifact.create(temporary, blob)
    # create() fsyncs the temporary and parent before this last operation.
    os.replace(temporary, marker)


def fit_public_actions(*, output, source_handoff, expected_source_handoff, expected_boot_id,
                       steps=120, learning_rate=5e-5, seed=42):
    artifact.require(type(steps) is int and steps > 0 and type(seed) is int
        and type(learning_rate) in (int, float) and math.isfinite(learning_rate) and learning_rate > 0,
        "Positive finite steps/LR and integer seed required")
    artifact.require(steps <= MAX_TRAINING_STEPS,
        "Public fit steps exceed complete resource journal capacity (maximum " + str(MAX_TRAINING_STEPS) + ")")
    _fresh_environment()
    output = artifact.ordinary(output)
    artifact.require(not output.exists(), "New independent public fit output required")
    paths = default_paths()
    parent_adapter = ROOT / "artifacts/jev-browser/20261015-action-lora-counterfactual-fit-01/adapter"
    for tree in (paths["public_root"], parent_adapter.parent, paths["preflight_root"], paths["reference_compile"]):
        tree = artifact.ordinary(tree)
        artifact.require(not output.is_relative_to(tree) and not tree.is_relative_to(output), "Fit output overlaps immutable input")
    output.mkdir(parents=True)
    receipt = {"schema_version": 1, "kind": artifact.KIND, "stage": "pending", "error": None,
               "scope": "Teacher-forced public development fitting diagnostics only",
               "qualified": False, "task_score": None, "steps": steps, "learning_rate": learning_rate,
               "seed": seed, "tokenizer_cache": str(CACHE), "temperature_before": artifact.TEMPERATURE,
               "model_loads": 0, "training_updates": 0}
    artifact.create(output / "receipt.json", artifact.encode(receipt) + b"\n")
    inputs, captured, model, hook = None, None, None, None
    start = time.monotonic()
    try:
        gate = ResourceGate(output / "resource-gates.jsonl", expected_boot_id)
        gate.check("before-any-project-framework-or-parent-payload", "cpu")
        handoff_blob = artifact.read(source_handoff)
        handoff = artifact.verify_source_handoff(handoff_blob, expected_sha256=expected_source_handoff)
        inputs = artifact.verify_public_fit_inputs(**paths, source_handoff_bytes=handoff_blob,
            expected_source_handoff=expected_source_handoff, expected_admission=artifact.FIXED_METADATA["admission-receipt.json"],
            expected_preflight=artifact.FIXED_METADATA["preflight-root-receipt.json"], tokenizer_cache=CACHE, gate=gate)
        meta = inputs.metadata()
        # Account for the full496 parent and every reference/source before copying.
        pending = sum(v["size_bytes"] for v in meta["parent_descriptor"]["files"].values())
        pending += sum(len(v) for _, v in inputs.public_items) + sum(len(v) for _, v in inputs.reference_items)
        pending += sum(p["size_bytes"] for p in handoff["files"].values()) + 32 * 1024**2
        check_cap(output, pending_bytes=pending, pending_files=701 + 496 + len(inputs.reference_items) + 100)
        artifact.copy_items(output / "inputs/public_collection", inputs.public_items)
        artifact.copy_items(output / "inputs/reference_compile", inputs.reference_items)
        artifact.copy_items(output / "inputs/metadata", tuple((k, v.encode()) for k, v in meta["metadata_blobs"].items()))
        artifact.create(output / "source-handoff.json", handoff_blob)
        artifact.copy_items(output / "sources", tuple((k, artifact.read(ROOT / k)) for k in handoff["files"]))
        receipt["versions"] = artifact.verify_reference_environment(inputs.reference_items)
        gate.check("immediately-before-legacy-parent-capture-and-Torch", "cpu")
        api = artifact.trusted_module("s1.browser_action_adapter", "src/s1/browser_action_adapter.py")
        captured = api.capture_initial_action_adapter(parent_adapter, output / "inputs/initial_action_adapter")
        verified_parent = artifact.validate_parent_capture(captured, meta["provenance"], expected_descriptor=meta["parent_descriptor"], gate=gate)
        check_cap(output)
        receipt["initial_action_adapter"] = captured["descriptor"]
        receipt["provenance"] = meta["provenance"]
        receipt["public_inputs"] = {"inventory": meta["public_inventory"], "observation_profile": artifact.parse(meta["provenance_bytes"])["observation_profile"],
                                    "metadata_files": meta["metadata_files"], "wire_identities": meta["wire_identities"]}
        # Neither CPU admission nor a successful earlier stage credits resident weights.
        gate.check("before-checkpoint-payload-hash-and-BF16-loader", "load")
        gate.check("before-any-BF16-model-construction", "train")
        checkpoint = (CACHE / "model.safetensors").resolve(strict=True)
        artifact.require(checkpoint.is_relative_to(CACHE.parents[1].resolve(strict=True))
            and artifact.pin(checkpoint, limit=BASE_BYTES) == {"sha256": BASE_SHA, "size_bytes": BASE_BYTES}, "Pinned full BF16 checkpoint bytes differ")
        receipt["base_checkpoint"] = dict(artifact.BASE_CHECKPOINT)
        kernel = artifact.trusted_module("jev_browser_fit_actions", "scripts/jev_browser_fit_actions.py")
        model = _construct_model(seed)
        receipt["model_loads"] = 1
        _model_check(model, kernel)
        artifact.require(artifact.verify_reference_environment(inputs.reference_items, model=model) == receipt["versions"], "Loaded processor runtime differs")
        for config in model.lm.peft_config.values():
            config.base_model_name_or_path, config.revision = artifact.MODEL_ID, artifact.REVISION
        gate.check("before-parent-injection", "train")
        captured["descriptor"]["initial_tensor_verification"] = api.initialize_training_action_adapter(model, verified_parent)
        rows, compiled_pins = artifact.compile_public_actions(model, inputs, output=output, gate=gate)
        _model_check(model, kernel)
        gate.check("immediately-before-unchanged-train_actions-optimizer", "train")
        # Native LM forward entry remains gated through train and22 eval-only rows.
        hook = model.lm.register_forward_pre_hook(lambda current, _a, _k: gate.check(
            "before-native-train-forward" if current.training else "before-native-development-forward", "train") and None, with_kwargs=True)
        receipt["training"] = kernel.train_actions(model, rows["train"], rows["development"], output,
            steps=steps, learning_rate=learning_rate, seed=seed)
        hook.remove()
        hook = None
        _model_check(model, kernel)
        artifact.require(artifact.verify_reference_environment(inputs.reference_items, model=model) == receipt["versions"], "Processor/cache environment changed after fitting")
        receipt["temperature_after"] = model.temperature
        receipt["training_updates"] = receipt["training"]["completed_updates"]
        files = kernel._save_adapter(model, output / "adapter")
        tensor_meta = artifact.tensor_metadata(artifact.read(output / "adapter/adapter_model.safetensors", 64 * 1024**2),
                                               verified_parent.tensor_metadata["tensor_shapes"], gate=gate)
        receipt["tensor_metadata"] = tensor_meta
        receipt["elapsed_seconds"] = time.monotonic() - start
        manifest = {"schema_version": 1, "kind": artifact.KIND, "base_model_id": artifact.MODEL_ID,
            "base_revision": artifact.REVISION, "base_checkpoint": dict(artifact.BASE_CHECKPOINT), "training_quantization": "none", "training_dtype": "bfloat16",
            "head_sha256": artifact.HEAD_SHA, "temperature": artifact.TEMPERATURE,
            "policy": "greedy_legal_action_v2", "instructions_sha256": "63874de795580b95edbcb1f7c6a6857e86f458dabba77e33814d8f08f158d68a",
            "inference_model_identity": artifact.IDENTITY, "lora": artifact.LORA, "files": files,
            "training_receipt": {}, "provenance": meta["provenance"], "public_inputs": receipt["public_inputs"],
            "parent": captured["descriptor"], "source_handoff": {"path": "source-handoff.json", "sha256": expected_source_handoff, "identity_sha256": handoff["identity_sha256"]},
            "compiled_files": compiled_pins, "tensor_metadata": tensor_meta,
            "deployment_contract": artifact.DEPLOYMENT, "qualification_required": artifact.QUALIFICATION_KIND}
        artifact.create(output / "adapter/manifest.pending.json", artifact.encode(manifest) + b"\n")
        publish_candidate(output, manifest, receipt, inputs=inputs, captured=captured, gate=gate)
        return receipt
    except BaseException as primary:
        receipt.update(stage="failed", error={"type": type(primary).__name__, "message": str(primary)},
                       elapsed_seconds=time.monotonic() - start)
        if hasattr(primary, "action_training_evidence"):
            receipt["partial_training"] = primary.action_training_evidence
        secondary_errors = []
        if model is not None:
            try:
                gate.check("before-bounded-partial-adapter-preservation", "cpu")
                partial = output / ("partial-adapter-" + str(time.monotonic_ns()))
                receipt["partial_adapter"] = {"path": partial.name, "files": kernel._save_adapter(model, partial)}
                artifact.create(partial / "manifest.invalid.json", artifact.encode({"kind": artifact.KIND, "stage": "failed", "error": receipt["error"]}) + b"\n")
            except BaseException as secondary:
                secondary_errors.append({"phase": "partial-adapter-preservation", "type": type(secondary).__name__, "message": str(secondary)})
        for label, operation in (("forward-hook-cleanup", (lambda: hook.remove()) if hook is not None else lambda: None),
                                 ("public-input-final-check", inputs.check if inputs is not None else lambda: None),
                                 ("parent-final-check", (lambda: _parent_checks(captured)) if captured is not None else lambda: None)):
            try:
                operation()
            except BaseException as secondary:
                secondary_errors.append({"phase": label, "type": type(secondary).__name__, "message": str(secondary)})
        receipt["finalization_errors"] = secondary_errors
        for error in secondary_errors:
            primary.add_note("Secondary fit finalization failure: " + json.dumps(error))
        try:
            replace_json(output / "receipt.json", receipt)
        except BaseException as secondary:
            primary.add_note("Failed receipt persistence: " + repr(secondary))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-handoff", required=True)
    parser.add_argument("--expected-source-handoff", required=True)
    parser.add_argument("--expected-boot-id", required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = vars(parser.parse_args(argv))
    artifact.require(args.pop("run"), "Explicit reviewed --run required; no execution by default")
    return fit_public_actions(**args)


if __name__ == "__main__":
    main()
