#!/usr/bin/env python3
"""Fit the full historical179 native population in the explicit V2 format.

Default dry. Reuses the original full-vocabulary action-token CE trainer and
BF16 resource formula. No browser score or hardware qualification follows from
an exported development artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"train": 142, "development": 37, "total": 179}
FORMAT = "shared-elements-v2"
MAX_STEPS = 280
OBJECTIVE = "full-vocabulary-action-token-CE"
# Exact trees referenced by the immutable45-file historical-data authority.
HISTORICAL_INPUT_TREES = (
    "artifacts/jev-browser/20261016-public-native-training-07-admission-01",
    "artifacts/jev-browser/20261016-public-native-training-collection-07",
    "artifacts/jev-runtime/generic-form-progress-candidates-20261019-01",
    "artifacts/jev-runtime/generic-form-progress-native-proof-20261019-01",
    "artifacts/jev-runtime/generic-menu-semantics-prototype-20261018-01",
    "artifacts/jev-runtime/owned-form-progress-collection-admission-20261019-01",
    "artifacts/jev-runtime/owned-generic-collection-admission-20261018-01",
    "artifacts/jev-runtime/owned-public-mixed-composition-20261018-01",
)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def _policy():
    import jev_browser_fit_public_actions

    return jev_browser_fit_public_actions


def _environment():
    _policy()._fresh_environment()


def _gate(path, boot_id):
    return _policy().ResourceGate(path, boot_id)


def _construct_model(seed):
    import torch

    from s1.browser_action_adapter import BASE_ID, BASE_REVISION, LORA
    from s1.unified import UnifiedDecisionModel

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.backends.cuda.enable_cudnn_sdp(False)
    return UnifiedDecisionModel(
        BASE_ID,
        revision=BASE_REVISION,
        device="cuda",
        precision="bfloat16",
        quantization="none",
        max_context=8192,
        temperature=_policy().artifact.TEMPERATURE,
        attn_implementation="sdpa",
        lora={**LORA, "revision": BASE_REVISION},
    )


def _check_model(model, kernel):
    _policy()._model_check(model, kernel)


def _checkpoint():
    policy = _policy()
    path = policy.artifact.resolve_base_checkpoint(policy.CACHE)
    require(
        policy.artifact.pin(path, limit=policy.BASE_BYTES)
        == {"sha256": policy.BASE_SHA, "size_bytes": policy.BASE_BYTES},
        "Pinned full BF16 checkpoint bytes differ",
    )
    return dict(policy.artifact.BASE_CHECKPOINT)


def _sources():
    labels = (
        "scripts/jev_browser_fit_native_actions.py",
        "scripts/jev_browser_fit_actions.py",
        "scripts/jev_browser_fit_public_actions.py",
        "scripts/jev_public_action_artifact.py",
        "scripts/jev_cfg_action_loss.py",
        "scripts/jev_browser_fit_readout.py",
        "scripts/jev_browser_training_data.py",
        "scripts/breakthrough_training.py",
        "artifacts/jev-runtime/v2-action-training-prepare-20261021-01/historical_data_projection.py",
    )
    return {label: ROOT / label for label in labels}


def _preflight_paths(output, historical_root, initial_action_adapter):
    for path in (output, historical_root):
        require(
            ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
            "Native fit paths must be ordinary and contain no symlinks or traversal",
        )
    require(historical_root.is_dir(), "Ordinary historical data root required")
    require(not output.exists(), "Fresh native fit output required")
    require(not historical_root.is_relative_to(output), "Output cannot contain historical input")
    require(
        not output.resolve().is_relative_to(ROOT / "src/s1"),
        "Output must not enter the captured source package",
    )
    require(
        not any(
            output.resolve().is_relative_to(historical_root / label)
            for label in HISTORICAL_INPUT_TREES
        ),
        "Output must not modify an immutable historical input tree",
    )
    if initial_action_adapter is not None:
        adapter = Path(initial_action_adapter).absolute()
        require(
            ".." not in adapter.parts
            and adapter.is_dir()
            and adapter.name == "adapter"
            and not any(p.is_symlink() for p in (adapter, *adapter.parents)),
            "Initial adapter must be an ordinary adjacent fit directory",
        )
        original = adapter.resolve().parent
        destination = output.resolve()
        require(
            not destination.is_relative_to(original) and not original.is_relative_to(destination),
            "Native output must not overlap the immutable initial parent fit",
        )


def _split_check(blobs, destination, expected):
    import jev_browser_fit_readout as integrity

    result = {
        "success": False,
        "changed": [],
        "missing": [],
        "added": [],
        "identity_sha256": expected["identity_sha256"],
    }
    if destination is None:
        actual = integrity._input_manifest(blobs)
    else:
        destination = Path(destination)
        require(
            not destination.is_symlink() and destination.is_dir(),
            "Ordinary frozen native split required",
        )
        entries = {p.name: p for p in destination.iterdir()}
        require(
            set(entries) == set(blobs) | {"input_manifest.json"}
            and all(p.is_file() and not p.is_symlink() for p in entries.values()),
            "Exact frozen native split files required",
        )
        require(
            integrity._json(entries["input_manifest.json"].read_bytes()) == expected,
            "Frozen native split manifest differs",
        )
        actual = integrity._input_manifest({name: entries[name].read_bytes() for name in blobs})
    result.update(current_identity_sha256=actual["identity_sha256"])
    old, current = expected["files"], actual["files"]
    result.update(
        changed=sorted(k for k in old.keys() & current.keys() if old[k] != current[k]),
        missing=sorted(old.keys() - current.keys()),
        added=sorted(current.keys() - old.keys()),
    )
    result["success"] = not any(result[k] for k in ("changed", "missing", "added"))
    return result


def _final_checks(output, receipt, data, sources, source_files, snapshots, captured):
    import jev_browser_fit_actions as kernel

    from s1 import browser_native_action_data as native
    from s1.browser_evidence import verify_runtime_sources

    errors = []
    if sources is not None:
        receipt["source_verification"] = verify_runtime_sources(
            sources, additional_files=source_files
        )
        if not receipt["source_verification"]["success"]:
            errors.append("Current execution sources changed")
    if data is not None and snapshots:
        try:
            receipt["historical_original_verification"] = native.verify_original(data)
            current = native.load_historical_data(data["root"])
            frozen = native.verify_frozen_historical_data(
                output / "inputs/historical", receipt["native_data"]
            )
            receipt["input_verification"], receipt["frozen_input_verification"] = {}, {}
            for split, snapshot in snapshots.items():
                original_check = _split_check(
                    native.split_blobs(current, split), None, snapshot["manifest"]
                )
                frozen_check = _split_check(
                    native.split_blobs(frozen, split),
                    output / "inputs" / split,
                    snapshot["manifest"],
                )
                receipt["input_verification"][split] = original_check
                receipt["frozen_input_verification"][split] = frozen_check
                if not original_check["success"] or not frozen_check["success"]:
                    errors.append("Native split changed: " + split)
        except BaseException as error:
            errors.append(type(error).__name__ + ": " + str(error))
    if captured is not None:
        checks = kernel._check_initial_snapshot(captured)
        receipt["initial_action_adapter"].update(checks)
        if not all(x["success"] for x in checks.values()):
            errors.append("Initial action adapter changed")
    require(not errors, "; ".join(errors))


def fit_native_actions(
    *,
    historical_root,
    output_dir,
    steps=None,
    learning_rate=1e-4,
    seed=42,
    prompt_format=FORMAT,
    initial_action_adapter=None,
    resource_profile=None,
):
    require(
        type(prompt_format) is str and prompt_format == FORMAT,
        "This distinct native profile requires explicit shared-elements-v2",
    )
    require(
        resource_profile in (None, "gx10-uma-native-40gib-v1"),
        "Explicit supported native resource profile required",
    )
    require(
        (steps is None or type(steps) is int and 1 <= steps <= MAX_STEPS)
        and type(seed) is int
        and type(learning_rate) in (int, float)
        and math.isfinite(learning_rate)
        and learning_rate > 0,
        "Bounded positive updates/learning rate and integer seed required",
    )
    output, historical_root = Path(output_dir).absolute(), Path(historical_root).absolute()
    _preflight_paths(output, historical_root, initial_action_adapter)
    _environment()
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema_version": 2,
        "kind": "jev_native_action_lora_v2",
        "stage": "setup",
        "scope": "Historical179 teacher-forced native development fitting; no browser score",
        "training_objective": OBJECTIVE,
        "steps": 142 if steps is None else steps,
        "learning_rate": learning_rate,
        "seed": seed,
        "qualified": False,
        "model_loads": 0,
        "model_load_started": False,
        "training_updates": 0,
        "training_update_count_scope": "Completed ledger records; in-flight effects may be unknown on failure",
    }
    data = sources = model = captured = hook = None
    snapshots, source_files, manifest = {}, _sources(), None
    primary, started = None, time.monotonic()
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if resource_profile is None:
            gate = _gate(output / "resource-gates.jsonl", boot_id)
        else:
            from s1 import browser_native_resources as resources

            gate = resources.Gate(output / "resource-gates.jsonl", boot_id)
            receipt["resource_profile"] = {
                "name": resources.PROFILE,
                "expected_unit": resources.EXPECTED_UNIT,
                "expected_boot_id": boot_id,
            }
        gate.check("before-any-project-framework-or-parent-payload", "cpu")
        if resource_profile is not None:
            receipt["resource_components"] = resources.capture_components(
                output / "inputs/resource-components"
            )
        import importlib.metadata

        import jev_browser_fit_actions as kernel
        import jev_cfg_action_loss as cfg

        from s1 import browser_action_adapter as adapters
        from s1 import browser_native_action_data as native
        from s1.browser_action_prompt import NATIVE_KIND, prompt_contract
        from s1.browser_evidence import capture_runtime_sources
        from s1.jev_generated import INSTRUCTIONS, POLICY, format_action_request

        receipt.update(
            base_model_id=kernel.MODEL_ID,
            base_revision=kernel.REVISION,
            training_quantization="none",
            training_dtype="bfloat16",
            deployment_quantization="nf4",
            lora=kernel.LORA,
            policy=POLICY,
            instructions_sha256=sha(INSTRUCTIONS.encode()),
            prompt_contract=prompt_contract(),
        )
        receipt["versions"] = {
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")
        }
        sources = capture_runtime_sources(output, additional_files=source_files)
        receipt["source_manifest"] = sources
        data = native.load_historical_data(historical_root)
        require(data["counts"] == COUNTS, "Complete historical179 population required")
        receipt["native_data"] = native.freeze_historical_data(data, output / "inputs/historical")
        cases = {
            split: [c for c in data["cases"] if c.split == split]
            for split in ("train", "development")
        }
        receipt["input_manifests"], receipt["provenance"] = {}, {"source_scope": "development_only"}
        for split in cases:
            blobs = native.split_blobs(data, split)
            snapshot = {"blobs": blobs, "manifest": kernel.integrity._input_manifest(blobs)}
            destination = output / "inputs" / split
            kernel.integrity._freeze_collection(snapshot, destination)
            snapshots[split] = snapshot
            receipt["input_manifests"][split] = {
                **snapshot["manifest"],
                "manifest_sha256": sha((destination / "input_manifest.json").read_bytes()),
                "artifact_path": "inputs/" + split,
                "preservation_timing": "Authenticated historical bytes frozen before model loading",
            }
            receipt["provenance"][split] = native.inventory(blobs["cases.jsonl"], split)
        overlap = [
            key
            for key in ("ids", "groups", "request_fingerprints")
            if set(receipt["provenance"]["train"][key])
            & set(receipt["provenance"]["development"][key])
        ]
        require(not overlap, "Native train/development overlap")
        receipt["split_audit"] = {
            "valid": True,
            "split_counts": {"train": 142, "development": 37},
            "overlaps": [],
        }
        if initial_action_adapter is not None:
            gate.check("immediately-before-legacy-parent-capture-and-Torch", "cpu")
            adapters.validate_initial_action_adapter_destination(
                initial_action_adapter, output / "inputs/initial_action_adapter"
            )
            captured = adapters.capture_initial_action_adapter(
                initial_action_adapter, output / "inputs/initial_action_adapter"
            )
            receipt["initial_action_adapter"] = captured["descriptor"]
            adapters.verify_initial_action_adapter_splits(
                captured["frozen_root"], receipt["provenance"]
            )
        receipt["stage"] = "model_loading"
        kernel._write(output / "receipt.json", receipt)
        gate.check("before-checkpoint-payload-hash-and-BF16-loader", "load")
        receipt["base_checkpoint"] = _checkpoint()
        gate.check("before-any-BF16-model-construction", "train")
        receipt["model_load_started"] = True
        model = _construct_model(seed)
        receipt["model_loads"] = 1
        if resource_profile is None:
            _check_model(model, kernel)
        else:
            gate.mark_model_loaded(model, lambda: _check_model(model, kernel))
        for configuration in model.lm.peft_config.values():
            configuration.base_model_name_or_path, configuration.revision = (
                kernel.MODEL_ID,
                kernel.REVISION,
            )
        if captured is not None:
            gate.check("before-parent-injection", "train")
            require(
                all(x["success"] for x in kernel._check_initial_snapshot(captured).values()),
                "Initial adapter changed before injection",
            )
            receipt["initial_action_adapter"]["initial_tensor_verification"] = (
                adapters.initialize_training_action_adapter(model, captured["verified"])
            )
        receipt["stage"] = "compiling"
        kernel._write(output / "receipt.json", receipt)
        rows, receipt["compiled"], receipt["cfg_compiled"] = {}, {}, {}
        by_id = {w["case_id"]: w for w in data["wires"]}
        for split, population in cases.items():
            rows[split] = []
            for index, case in enumerate(population):
                gate.check(f"before-BF16-processor-compile-{split}-{index}", "train")
                wire = by_id[case.id]
                compiled = kernel.compile_actions(
                    model,
                    [case],
                    {"blobs": {"wire_requests.jsonl": encode(wire) + b"\n"}},
                    identity=kernel.IDENTITY,
                    prompt_format=prompt_format,
                )
                require(len(compiled) == 1, "Every original native row must compile")
                row = compiled[0]
                payload = copy.deepcopy(wire["payload"])
                payload["model"] = kernel.IDENTITY
                plan = cfg.compile_action_plan(row, payload, model.tok, prompt_format=prompt_format)
                _, _, labels, _ = format_action_request(
                    payload, kernel.IDENTITY, prompt_format=prompt_format
                )
                tree = cfg.LegalActionTokenTrie(
                    labels,
                    model.tok,
                    prompt_length=row["record"]["prompt_tokens"],
                    eos_token_id=row["record"]["native_stop_token_id"],
                    max_new_tokens=32,
                )
                kernel._append(output / f"compiled-{split}.jsonl", row["record"])
                kernel._append(
                    output / f"cfg-plans-{split}.jsonl",
                    {
                        "case_id": case.id,
                        "plan": plan.evidence(),
                        "inventory": {
                            "labels": labels,
                            "canonical_actions": tree.canonical_actions,
                            "encoded_actions": tree.encoded_actions,
                            "telemetry": tree.telemetry(),
                        },
                    },
                )
                rows[split].append(row)
            receipt["compiled"][split] = {
                "examples": len(rows[split]),
                "sha256": sha((output / f"compiled-{split}.jsonl").read_bytes()),
                "prompt_tokens": sum(r["record"]["prompt_tokens"] for r in rows[split]),
                "target_tokens": sum(len(r["targets"]) for r in rows[split]),
            }
            receipt["cfg_compiled"][split] = {
                "examples": len(rows[split]),
                "sha256": sha((output / f"cfg-plans-{split}.jsonl").read_bytes()),
            }
        receipt["stage"] = "training"
        kernel._write(output / "receipt.json", receipt)
        gate.check("immediately-before-unchanged-train_actions-optimizer", "train")
        hook = model.lm.register_forward_pre_hook(
            lambda current, _args, _kwargs: (
                gate.check(
                    "before-native-train-forward"
                    if current.training
                    else "before-native-development-forward",
                    "train",
                )
                and None
            ),
            with_kwargs=True,
        )
        receipt["training"] = kernel.train_actions(
            model,
            rows["train"],
            rows["development"],
            output,
            steps=steps,
            learning_rate=learning_rate,
            seed=seed,
        )
        hook.remove()
        hook = None
        _check_model(model, kernel)
        if resource_profile is not None:
            gate.check("after-native-training", "train")
        receipt["training_updates"] = receipt["training"]["completed_updates"]
        files = kernel._save_adapter(model, output / "adapter")
        receipt.update(
            stage="complete",
            adapter_files=files,
            adapter_artifact="adapter",
            full_lm_head_unchanged=True,
        )
        manifest = {
            "schema_version": 2,
            "kind": NATIVE_KIND,
            **{
                k: receipt[k]
                for k in (
                    "base_model_id",
                    "base_revision",
                    "training_quantization",
                    "training_dtype",
                    "deployment_quantization",
                    "policy",
                    "instructions_sha256",
                    "lora",
                    "prompt_contract",
                    "native_data",
                    "provenance",
                )
            },
            "inference_model_identity": kernel.IDENTITY,
            "files": files,
        }
    except BaseException as error:
        primary = error
        receipt.update(stage="failed", error={"type": type(error).__name__, "message": str(error)})
        if hasattr(error, "action_training_evidence"):
            receipt["partial_training"] = error.action_training_evidence
            receipt["training_updates"] = error.action_training_evidence["completed_updates"]
            receipt["in_flight_update_effects_unknown"] = (
                error.action_training_evidence["active"].get("split") == "train"
            )
        if model is not None:
            try:
                receipt["partial_adapter_files"] = kernel._save_adapter(model, output / "adapter")
            except BaseException as secondary:
                receipt["partial_adapter_error"] = {
                    "type": type(secondary).__name__,
                    "message": str(secondary),
                }
        raise
    finally:
        try:
            if hook is not None:
                hook.remove()
            _final_checks(output, receipt, data, sources, source_files, snapshots, captured)
            if resource_profile is not None and receipt["stage"] == "complete":
                receipt["resource_evidence"] = gate.evidence(
                    steps=receipt["steps"], parent_present=captured is not None
                )
        except BaseException as secondary:
            receipt.update(
                stage="failed",
                finalization_error={"type": type(secondary).__name__, "message": str(secondary)},
            )
            if primary is not None:
                primary.add_note("Native finalization failed: " + repr(secondary))
            else:
                raise
        finally:
            receipt["elapsed_seconds"] = time.monotonic() - started
            pending = primary or sys.exception()
            try:
                from jev_browser_fit_actions import _write

                _write(output / "receipt.json", receipt)
            except BaseException as write_error:
                if pending is None:
                    raise
                pending.add_note("Could not preserve native receipt: " + repr(write_error))
    require(receipt["stage"] == "complete" and manifest is not None, "Native fit failed")
    manifest["training_receipt"] = {
        "path": "../receipt.json",
        "sha256": sha((output / "receipt.json").read_bytes()),
    }
    try:
        kernel._write(output / "adapter/manifest.json", manifest)
        adapters.verify_action_adapter(output / "adapter")
        _final_checks(output, receipt, data, sources, source_files, snapshots, captured)
    except BaseException as error:
        (output / "adapter/manifest.json").unlink(missing_ok=True)
        receipt.update(stage="failed", error={"type": type(error).__name__, "message": str(error)})
        try:
            kernel._write(output / "receipt.json", receipt)
        except BaseException as write_error:
            error.add_note("Could not preserve failed publication receipt: " + repr(write_error))
        raise
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--historical-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-format", choices=[FORMAT], default=FORMAT)
    parser.add_argument("--initial-action-adapter", type=Path)
    parser.add_argument("--resource-profile", choices=["gx10-uma-native-40gib-v1"])
    parser.add_argument("--run", action="store_true")
    arguments = vars(parser.parse_args(argv))
    if not arguments.pop("run"):
        print(
            json.dumps(
                {
                    "status": "DRY_NO_DATA_PROCESSOR_MODEL_OR_TRAINING",
                    "counts": COUNTS,
                    "prompt_format": FORMAT,
                    "objective": OBJECTIVE,
                }
            )
        )
        return 0
    if arguments["output_dir"] is None:
        parser.error("--run requires a fresh --output-dir")
    fit_native_actions(**arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
