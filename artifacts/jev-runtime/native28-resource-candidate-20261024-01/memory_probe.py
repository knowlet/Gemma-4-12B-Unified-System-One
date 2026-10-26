#!/usr/bin/env python3
"""HELD source candidate: full179 compilation and two original LoRA RAM updates.

Default dry; SOURCE_ONLY deliberately refuses every execution request. A future
root must finish the independent guardian, exact plan/source selection, checks
and freeze before changing that guard and reselecting the changed source.
No source qualification, fit claim or actual measurement has happened here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_ONLY = True
PROFILE = "gx10-uma-native-28gib-memory-diagnostic-v1"
RESOURCE_SCHEMA = "jev-native-resource-gate-gx10-uma-28gib-diagnostic-v1"
EXPECTED_UNIT = "/user.slice/user-1001.slice/user@1001.service/app.slice/jev-native28-memory-diagnostic-20261024-01.service"
PLAN_SCHEMA = "native28-two-update-memory-diagnostic-plan-v1"
RECEIPT_SCHEMA = "native28-two-update-memory-diagnostic-receipt-v1"
NATIVE = ROOT / "scripts/jev_browser_fit_native_actions.py"
NATIVE_SHA = "8f127b338d32b079e814090a828fd3a608271fac5a711216d131c52b509988d5"
KERNEL = ROOT / "scripts/jev_browser_fit_actions.py"
KERNEL_SHA = "709eb702ee1a926dc6bea3740d28e880246f1be944fdb6d06deb4bec915dd474"
PUBLIC = ROOT / "scripts/jev_browser_fit_public_actions.py"
PUBLIC_SHA = "a38bbab599f6f8ec8117d1b912e02913330ea7bb11cd67e4e6793ed80da01ca7"
RESOURCE = HERE / "resource_gate.py"
COUNTS = {"train": 142, "development": 37, "total": 179}
FORMAT = "shared-elements-v2"
STEPS, SEED, LEARNING_RATE = 2, 42, 1e-4
WALL_SECONDS = 10800
MODEL_MODULES = {"torch", "transformers", "s1", "peft", "tokenizers", "numpy"}


def require(value, message):
    if not value:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def small_source(path):
    path = Path(path)
    require(path.suffix == ".py" and not any(p.is_symlink() for p in (path, *path.parents)),
            "Ordinary selected Python source required")
    with path.open("rb") as stream:
        blob = stream.read(200001)
    require(0 < len(blob) <= 200000, "Bounded selected source required")
    blob.decode("utf-8")
    return blob


def source_ref(path):
    blob = small_source(path)
    return {"path": str(Path(path).relative_to(ROOT)), "size_bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest()}


def load_source(name, path, expected):
    blob = small_source(path)
    require(hashlib.sha256(blob).hexdigest() == expected, "Selected source bytes changed")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Selected module spec required")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(blob, str(path), "exec", dont_inherit=True), vars(module))
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def read_plan(path, expected):
    path = Path(path)
    require(path.suffix == ".json" and not any(p.is_symlink() for p in (path, *path.parents)),
            "Ordinary selected small plan required")
    with path.open("rb") as stream:
        blob = stream.read(1048577)
    require(0 < len(blob) <= 1048576 and type(expected) is str
            and re.fullmatch(r"[0-9a-f]{64}", expected) is not None
            and hashlib.sha256(blob).hexdigest() == expected, "Selected plan SHA differs")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate diagnostic plan field")
            result[key] = value
        return result

    plan = json.loads(blob, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError("Nonfinite diagnostic plan value: " + value)))
    require(type(plan) is dict and plan.get("schema") == PLAN_SCHEMA
            and plan.get("profile") == PROFILE and plan.get("executable") is True,
            "Explicit selected diagnostic plan required")
    require(plan.get("actor_sha256") == source_ref(Path(__file__))["sha256"]
            and plan.get("native_fitter_sha256") == NATIVE_SHA
            and plan.get("training_kernel_sha256") == KERNEL_SHA
            and plan.get("public_fitter_sha256") == PUBLIC_SHA,
            "Diagnostic/original composition pins differ")
    require(plan.get("counts") == COUNTS and type(plan.get("counts")) is dict
            and all(type(value) is int for value in plan["counts"].values())
            and type(plan.get("steps")) is int
            and plan["steps"] == STEPS and type(plan.get("seed")) is int
            and plan["seed"] == SEED and type(plan.get("learning_rate")) is float
            and plan["learning_rate"] == LEARNING_RATE
            and plan.get("prompt_format") == FORMAT
            and type(plan.get("max_context")) is int and plan["max_context"] == 8192
            and type(plan.get("whole_seconds")) is int and plan["whole_seconds"] == WALL_SECONDS,
            "Full population/fixed two-update diagnostic policy differs")
    require(plan.get("historical_root") == str(ROOT)
            and plan.get("normal_adapter_export") is False
            and plan.get("training_or_browser_qualified") is False,
            "Original historical input root and nonpublication policy required")
    require(plan.get("initial_action_adapter") is None
            or type(plan["initial_action_adapter"]) is str,
            "Selected parent must be absent or an explicit ordinary path")
    for key in ("resource_gate_sha256", "expected_runtime_source_identity_sha256"):
        require(type(plan.get(key)) is str
                and re.fullmatch(r"[0-9a-f]{64}", plan[key]) is not None,
                "Final selected source identities are still pending: " + key)
    require(plan.get("root_guardian_selection") is not None,
            "Independent root guardian selection is not implemented/selected")
    require(type(plan.get("expected_boot_id")) is str and bool(plan["expected_boot_id"]),
            "Explicit future same-boot selection required")
    require(plan.get("expected_unit") == EXPECTED_UNIT, "Exact separately owned diagnostic unit required")
    require(type(plan.get("runtime_python")) is str
            and plan["runtime_python"] == str(Path(sys.executable).resolve()),
            "Explicit selected interpreter identity required")
    return plan


def capture_probe_sources(output, plan):
    """Own source namespace; never pretend these are approved normal callers."""
    paths = [Path(__file__), RESOURCE, NATIVE, KERNEL, PUBLIC]
    blobs = [small_source(path) for path in paths]
    records = [{"path": str(path.relative_to(ROOT)), "size_bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest()}
               for path, blob in zip(paths, blobs, strict=True)]
    expected = [plan["actor_sha256"], plan["resource_gate_sha256"], NATIVE_SHA,
                KERNEL_SHA, PUBLIC_SHA]
    require([row["sha256"] for row in records] == expected,
            "Selected diagnostic/original source identities changed")
    directory = output / "diagnostic-sources"
    directory.mkdir(exist_ok=False)
    for index, (path, blob) in enumerate(zip(paths, blobs, strict=True)):
        with (directory / (str(index) + "-" + path.name)).open("xb") as stream:
            stream.write(blob)
    descriptor = {"scope": "separate diagnostic entry/resource and original composition sources",
                  "normal_named_caller_manifest_extended": False, "files": records}
    (directory / "descriptor.json").write_bytes(encode(descriptor) + b"\n")
    return descriptor


def verify_probe_sources(descriptor, output):
    require(type(descriptor) is dict and type(descriptor.get("files")) is list,
            "Diagnostic source descriptor required")
    paths = [Path(__file__), RESOURCE, NATIVE, KERNEL, PUBLIC]
    require([row["path"] for row in descriptor["files"]]
            == [str(path.relative_to(ROOT)) for path in paths],
            "Exact diagnostic source set/order required")
    require(all(source_ref(ROOT / row["path"]) == row for row in descriptor["files"]),
            "Diagnostic source changed after capture")
    directory = Path(output) / "diagnostic-sources"
    require((directory / "descriptor.json").read_bytes() == encode(descriptor) + b"\n",
            "Captured diagnostic descriptor changed")
    for index, (path, row) in enumerate(zip(paths, descriptor["files"], strict=True)):
        blob = small_source(directory / (str(index) + "-" + path.name))
        require(len(blob) == row["size_bytes"] and hashlib.sha256(blob).hexdigest() == row["sha256"],
                "Captured diagnostic source bytes changed")


def failure_record(error):
    record = {"type": type(error).__name__, "message": str(error)}
    if hasattr(error, "native28_resource_evidence"):
        record["native28_resource_evidence"] = error.native28_resource_evidence
    if hasattr(error, "action_training_evidence"):
        record["action_training_evidence"] = error.action_training_evidence
    return record


def install_forward_guard(model, gate):
    def before(current, _args, _kwargs):
        gate.check("before-native-train-forward" if current.training
                   else "before-native-development-forward", "train")
        return None
    return model.lm.register_forward_pre_hook(before, with_kwargs=True)


def compile_population(model, cases, wires, output, gate, kernel, cfg, formatter):
    """Original per-row compiler composition; all142/37, unchanged token/CFG math."""
    require(len(wires) == COUNTS["total"], "All original wire records required")
    by_id = {wire["case_id"]: wire for wire in wires}
    require(len(by_id) == COUNTS["total"], "Complete unique original wire population required")
    rows, compiled_meta, cfg_meta = {}, {}, {}
    for split, count in (("train", 142), ("development", 37)):
        population = cases[split]
        require(len(population) == count, "Complete original split must compile")
        rows[split] = []
        for index, case in enumerate(population):
            gate.check(f"before-BF16-processor-compile-{split}-{index}", "train")
            wire = by_id[case.id]
            compiled = kernel.compile_actions(
                model, [case], {"blobs": {"wire_requests.jsonl": encode(wire) + b"\n"}},
                identity=kernel.IDENTITY, prompt_format=FORMAT)
            require(len(compiled) == 1, "Every original native row must compile")
            row = compiled[0]
            payload = copy.deepcopy(wire["payload"])
            payload["model"] = kernel.IDENTITY
            action_plan = cfg.compile_action_plan(row, payload, model.tok, prompt_format=FORMAT)
            _, _, labels, _ = formatter(payload, kernel.IDENTITY, prompt_format=FORMAT)
            tree = cfg.LegalActionTokenTrie(
                labels, model.tok, prompt_length=row["record"]["prompt_tokens"],
                eos_token_id=row["record"]["native_stop_token_id"], max_new_tokens=32)
            kernel._append(output / f"compiled-{split}.jsonl", row["record"])
            kernel._append(output / f"cfg-plans-{split}.jsonl", {
                "case_id": case.id, "plan": action_plan.evidence(),
                "inventory": {"labels": labels, "canonical_actions": tree.canonical_actions,
                              "encoded_actions": tree.encoded_actions,
                              "telemetry": tree.telemetry()}})
            rows[split].append(row)
        compiled_meta[split] = {
            "examples": len(rows[split]),
            "sha256": hashlib.sha256((output / f"compiled-{split}.jsonl").read_bytes()).hexdigest(),
            "prompt_tokens": sum(row["record"]["prompt_tokens"] for row in rows[split]),
            "target_tokens": sum(len(row["targets"]) for row in rows[split]),
        }
        cfg_meta[split] = {
            "examples": len(rows[split]),
            "sha256": hashlib.sha256((output / f"cfg-plans-{split}.jsonl").read_bytes()).hexdigest(),
        }
    return rows, compiled_meta, cfg_meta


def _measure(plan, output, native, resources, *, began):
    """Composition only; SOURCE_ONLY currently prevents reaching this function."""
    output = Path(output).absolute()
    historical_root = Path(plan["historical_root"])
    parent = plan.get("initial_action_adapter")
    native._preflight_paths(output, historical_root, parent)
    require(output.is_relative_to(ROOT / "artifacts"), "Private artifact output required")
    native._environment()
    output.mkdir(parents=True, exist_ok=False)
    (output / ".gitignore").write_text("*\n")
    receipt = {
        "schema": RECEIPT_SCHEMA, "profile": PROFILE, "stage": "setup",
        "scope": "two-update RAM diagnostic; not full-training capacity, adapter or browser qualification",
        "qualified": False, "training_qualified": False, "browser_qualified": False,
        "normal_adapter_export": False, "counts": COUNTS, "steps": STEPS,
        "learning_rate": LEARNING_RATE, "seed": SEED, "model_loads": 0,
        "model_load_started": False, "training_updates": 0,
        "training_update_count_scope": "completed ledger records; in-flight effects unknown on failure",
        "root_guardian_selection": plan["root_guardian_selection"],
    }
    data = sources = model = captured = hook = gate = None
    snapshots, source_files = {}, native._sources()
    descriptor = None
    primary = None
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        require(boot_id == plan["expected_boot_id"], "Selected boot changed")
        gate = resources.Gate(output / "resource-gates.jsonl", boot_id)
        require(resources.PROFILE == PROFILE, "Selected diagnostic resource profile differs")
        receipt["resource_profile"] = {
            "name": resources.PROFILE, "expected_unit": resources.EXPECTED_UNIT,
            "expected_boot_id": boot_id, "cap_bytes_each": resources.CAP,
            "live_stop_bytes_each": resources.STOP,
            "combined_physical_cap_claimed": False,
        }
        gate.check("before-any-project-framework-or-parent-payload", "cpu")
        receipt["resource_components"] = resources.capture_components(output / "inputs/resource-components")
        descriptor = capture_probe_sources(output, plan)
        receipt["diagnostic_source_descriptor"] = descriptor

        import jev_browser_fit_actions as kernel
        import jev_cfg_action_loss as cfg
        from importlib import metadata
        from s1 import browser_action_adapter as adapters
        from s1 import browser_native_action_data as native_data
        from s1.browser_evidence import capture_runtime_sources
        from s1.jev_generated import format_action_request

        require(Path(kernel.__file__).resolve() == KERNEL, "Original trainer origin differs")
        require(source_ref(KERNEL)["sha256"] == KERNEL_SHA, "Original trainer changed before import")
        receipt.update(
            base_model_id=kernel.MODEL_ID, base_revision=kernel.REVISION,
            training_quantization="none", training_dtype="bfloat16",
            lora=kernel.LORA, prompt_format=FORMAT, max_context=8192,
            training_objective=native.OBJECTIVE,
            runtime_python=str(Path(sys.executable).resolve()),
            versions={name: metadata.version(name) for name in ("torch", "transformers", "peft")},
        )
        sources = capture_runtime_sources(output, additional_files=source_files)
        require(sources["identity_sha256"] == plan["expected_runtime_source_identity_sha256"],
                "Selected original runtime source inventory differs")
        receipt["source_manifest"] = sources
        data = native_data.load_historical_data(historical_root)
        require(data["counts"] == COUNTS, "Complete historical179 population required")
        receipt["native_data"] = native_data.freeze_historical_data(data, output / "inputs/historical")
        cases = {split: [case for case in data["cases"] if case.split == split]
                 for split in ("train", "development")}
        receipt["input_manifests"], receipt["provenance"] = {}, {"source_scope": "development_only"}
        for split in cases:
            blobs = native_data.split_blobs(data, split)
            snapshot = {"blobs": blobs, "manifest": kernel.integrity._input_manifest(blobs)}
            destination = output / "inputs" / split
            kernel.integrity._freeze_collection(snapshot, destination)
            snapshots[split] = snapshot
            receipt["input_manifests"][split] = {
                **snapshot["manifest"],
                "manifest_sha256": hashlib.sha256((destination / "input_manifest.json").read_bytes()).hexdigest(),
                "artifact_path": "inputs/" + split,
                "preservation_timing": "All original bytes frozen before model loading",
            }
            receipt["provenance"][split] = native_data.inventory(blobs["cases.jsonl"], split)
        overlaps = [key for key in ("ids", "groups", "request_fingerprints")
                    if set(receipt["provenance"]["train"][key])
                    & set(receipt["provenance"]["development"][key])]
        require(not overlaps, "Native train/development overlap")
        receipt["split_audit"] = {"valid": True, "split_counts": {"train": 142, "development": 37},
                                  "overlaps": []}
        if parent is not None:
            gate.check("immediately-before-legacy-parent-capture-and-Torch", "cpu")
            adapters.validate_initial_action_adapter_destination(parent, output / "inputs/initial_action_adapter")
            captured = adapters.capture_initial_action_adapter(parent, output / "inputs/initial_action_adapter")
            receipt["initial_action_adapter"] = captured["descriptor"]
            adapters.verify_initial_action_adapter_splits(captured["frozen_root"], receipt["provenance"])

        receipt["stage"] = "model_loading"
        kernel._write(output / "diagnostic-receipt.json", receipt)
        gate.check("before-checkpoint-payload-hash-and-BF16-loader", "load")
        receipt["base_checkpoint"] = native._checkpoint()
        gate.check("before-any-BF16-model-construction", "train")
        receipt["model_load_started"] = True
        model = native._construct_model(SEED)
        receipt["model_loads"] = 1
        gate.mark_model_loaded(model, lambda: native._check_model(model, kernel))
        for configuration in model.lm.peft_config.values():
            configuration.base_model_name_or_path, configuration.revision = kernel.MODEL_ID, kernel.REVISION
        if captured is not None:
            gate.check("before-parent-injection", "train")
            require(all(value["success"] for value in kernel._check_initial_snapshot(captured).values()),
                    "Initial adapter changed before injection")
            receipt["initial_action_adapter"]["initial_tensor_verification"] = (
                adapters.initialize_training_action_adapter(model, captured["verified"]))

        receipt["stage"] = "compiling"
        kernel._write(output / "diagnostic-receipt.json", receipt)
        rows, receipt["compiled"], receipt["cfg_compiled"] = compile_population(
            model, cases, data["wires"], output, gate, kernel, cfg, format_action_request)
        receipt["stage"] = "two_update_memory_measurement"
        kernel._write(output / "diagnostic-receipt.json", receipt)
        gate.check("immediately-before-unchanged-train_actions-optimizer", "train")
        hook = install_forward_guard(model, gate)
        receipt["training"] = kernel.train_actions(
            model, rows["train"], rows["development"], output,
            steps=STEPS, learning_rate=LEARNING_RATE, seed=SEED)
        hook.remove()
        hook = None
        native._check_model(model, kernel)
        gate.check("after-native-training", "train")
        receipt["training_updates"] = receipt["training"]["completed_updates"]
        require(receipt["training_updates"] == STEPS
                and receipt["training"]["development_examples"] == 37
                and receipt["training"]["development_updates"] == 0,
                "Original two-update/full-development measurement incomplete")
        receipt["full_lm_head_unchanged"] = True
        receipt["stage"] = "diagnostic_complete_pending_final_checks_and_external_closure"
    except BaseException as error:
        primary = error
        receipt.update(stage="failed_diagnostic_no_capacity_or_adapter_claim", error=failure_record(error))
        if hasattr(error, "action_training_evidence"):
            receipt["partial_training"] = error.action_training_evidence
            receipt["training_updates"] = error.action_training_evidence["completed_updates"]
            receipt["in_flight_update_effects_unknown"] = (
                error.action_training_evidence["active"].get("split") == "train")
        raise
    finally:
        final_errors = []

        def finalize_part(label, operation):
            try:
                operation()
            except BaseException as error:
                final_errors.append((label, error))

        try:
            if hook is not None:
                finalize_part("forward-hook-removal", hook.remove)
            if sources is not None or data is not None or captured is not None:
                finalize_part("original-data-and-runtime-sources", lambda: native._final_checks(
                    output, receipt, data, sources, source_files, snapshots, captured))
            if descriptor is not None:
                finalize_part("diagnostic-source-namespace", lambda: verify_probe_sources(descriptor, output))
            if primary is None and not final_errors and receipt["stage"] == "diagnostic_complete_pending_final_checks_and_external_closure":
                receipt["resource_evidence"] = gate.evidence(steps=STEPS, parent_present=captured is not None)
                require(time.monotonic() - began < WALL_SECONDS, "Whole diagnostic exceeded selected bound")
            if final_errors:
                receipt["finalization_errors"] = [
                    {"phase": label, **failure_record(error)} for label, error in final_errors]
                raise final_errors[0][1]
        except BaseException as secondary:
            receipt.update(stage="failed_diagnostic_no_capacity_or_adapter_claim",
                           finalization_error=failure_record(secondary))
            if primary is not None:
                primary.add_note("Diagnostic finalization also failed: " + repr(secondary))
            else:
                raise
        finally:
            receipt["elapsed_seconds"] = time.monotonic() - began
            pending = primary or sys.exception()
            try:
                (output / "diagnostic-receipt.json").write_bytes(encode(receipt) + b"\n")
            except BaseException as write_error:
                if pending is None:
                    raise
                pending.add_note("Could not preserve diagnostic receipt: " + repr(write_error))
    require(time.monotonic() - began < WALL_SECONDS, "Late diagnostic return refused")
    return receipt


def execute(plan_path, plan_sha256, output):
    began = time.monotonic()
    require(SOURCE_ONLY is False,
            "HELD SOURCE ONLY: guardian, source checks, exact plan and freeze are not implemented/selected")
    require(sys.flags.no_user_site and sys.dont_write_bytecode
            and not {name.split(".", 1)[0] for name in sys.modules} & MODEL_MODULES,
            "Fresh -B -s pre-project/framework process required")
    plan = read_plan(plan_path, plan_sha256)
    for path, expected in ((NATIVE, NATIVE_SHA), (KERNEL, KERNEL_SHA), (PUBLIC, PUBLIC_SHA)):
        require(source_ref(path)["sha256"] == expected, "Original composition source changed")
    require(source_ref(RESOURCE)["sha256"] == plan["resource_gate_sha256"], "New resource source changed")
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    native = load_source("_native28_original_fitter_helpers", NATIVE, NATIVE_SHA)
    resources = load_source("_native28_selected_resource_gate", RESOURCE, plan["resource_gate_sha256"])
    require(resources.PROFILE == PROFILE and resources.SCHEMA == RESOURCE_SCHEMA
            and resources.EXPECTED_UNIT == EXPECTED_UNIT
            and resources.CAP == 28 * 1024**3
            and resources.STOP == 28 * 1024**3 - 1024**3 // 2,
            "Mixed resource schema/unit/cap/stop refused")
    return _measure(plan, output, native, resources, began=began)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "HELD_DRY_NO_SOURCE_IMPORT_DATA_MODEL_OR_TRAINING",
                          "source_only": SOURCE_ONLY, "profile": PROFILE, "counts": COUNTS,
                          "steps": STEPS, "normal_adapter_export": False,
                          "training_browser_or_capacity_qualified": False}))
        return 0
    require(args.plan is not None and args.output is not None, "Explicit selected plan/output required")
    result = execute(args.plan, args.plan_sha256, args.output)
    print(json.dumps({"stage": result["stage"], "qualified": False,
                      "normal_adapter_export": False, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
