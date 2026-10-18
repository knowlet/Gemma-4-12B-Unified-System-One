#!/usr/bin/env python3
"""Local GB10 complete public120 or admitted mixed129 stage-bounded producer.

No execution occurs without --run and a fresh, externally pinned ready profile.
This source is independent of the frozen H13 diagnostic. New artifacts remain
unqualified until separate native inference and held-out browser evaluation.
"""

from __future__ import annotations

import argparse
import threading
import time
import uuid
from pathlib import Path

import jev_public_stage_action_artifact as artifact
import jev_stage_memory_profile_v2 as memory
from jev_stage_memory_profile_v2 import (
    MIB,
    append,
    backward_cache_release,
    counted_optimizer_step,
    native_cache_release,
    operator_call,
    original_decoder_layers,
    replace,
    require,
    storage_ledger,
    write,
)

ROOT = Path(__file__).resolve().parents[1]


def train_stage(
    torch, model, row_map, plans, *, mode, gate, kernel, fit_kernel, output, state, inputs_value
):
    parameters = [
        (name, value) for name, value in model.lm.named_parameters() if value.requires_grad
    ]
    require(
        len(parameters) == 176 and sum(p.numel() for _, p in parameters) == 5193728,
        "Exact original trainables",
    )
    frozen = {
        name: (value, value._version)
        for name, value in model.lm.named_parameters()
        if not value.requires_grad
    }
    optimizer = gate.phase(
        "optimizer-create-empty",
        lambda: torch.optim.AdamW([p for _, p in parameters], lr=5e-5, weight_decay=0.0),
    )
    require(len(optimizer.state) == 0, "Adam moments must start lazy")
    trace = {"native_forward_calls": [], "layers": {}}
    active = {"phase": "setup", "case_id": None}

    def forward(_module, _args, kwargs):
        gate.check("official-native-forward-entry", "train")
        trace["native_forward_calls"].append(
            {
                **active,
                "input_shape": list(kwargs["input_ids"].shape),
                "positions": kwargs["logits_to_keep"].tolist(),
                "use_cache": kwargs.get("use_cache"),
            }
        )

    def layer(index):
        def entered(_module, _args):
            key = active["phase"] + ":" + str(index)
            trace["layers"][key] = trace["layers"].get(key, 0) + 1

        return entered

    layers, architecture = original_decoder_layers(model)
    state["decoder_architecture"] = trace["decoder_architecture"] = architecture
    replace(output / "progress.json", state)
    hooks = [model.lm.register_forward_pre_hook(forward, with_kwargs=True)]
    hooks += [module.register_forward_pre_hook(layer(i)) for i, module in enumerate(layers)]
    try:
        model.lm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.lm.train()
        updates = inputs_value["ordered_updates"]
        require(
            len(updates) == artifact.COUNTS[inputs_value["regime"]]["train"],
            "Complete admitted one-pass update schedule",
        )
        torch.manual_seed(42)
        state["actual_train_order"] = updates
        state["actual_update_refs"] = []
        for update_ref in updates:
            row = row_map[(update_ref["component"], update_ref["row_id"])]
            active.update(phase="forward", case_id=row["record"]["case_id"])
            gate.phase("zero-grad-before-forward", lambda: optimizer.zero_grad(set_to_none=True))
            before = len(trace["native_forward_calls"])
            loss, metrics = gate.phase(
                "native-train-forward",
                lambda: native_cache_release(
                    torch,
                    model,
                    gate,
                    output,
                    lambda: operator_call(
                        torch,
                        output,
                        active["phase"],
                        lambda: fit_kernel.action_loss(
                            model, row, plan=plans[row["record"]["case_id"]], mode=mode
                        ),
                    ),
                    mode=mode,
                    case_id=row["record"]["case_id"],
                    step=state["completed_updates"] + 1,
                    phase="train-forward",
                ),
            )
            require(
                len(trace["native_forward_calls"]) == before + 1,
                "One original causal forward per row",
            )
            active["phase"] = "backward"
            backward = loss.backward
            gate.phase(
                "native-backward",
                lambda: backward_cache_release(
                    torch,
                    model,
                    gate,
                    output,
                    lambda: operator_call(torch, output, active["phase"], backward),
                    mode=mode,
                    case_id=row["record"]["case_id"],
                    step=state["completed_updates"] + 1,
                ),
            )
            state["actual_native_backward_calls"] += 1
            norm = gate.phase(
                "native-grad-clip",
                lambda: torch.nn.utils.clip_grad_norm_(
                    [p for _, p in parameters], 1.0, error_if_nonfinite=True
                ),
            )

            gate.phase("native-Adam-step", lambda: counted_optimizer_step(optimizer, state, output))
            state["synchronized_updates"] += 1
            state["actual_update_refs"].append(update_ref)
            replace(output / "progress.json", state)
            require(
                all(torch.isfinite(p).all() for _, p in parameters), "Finite updated original176"
            )
            require(
                all(v.grad is None for v, _ in frozen.values())
                and not any(p in optimizer.state for p, _ in frozen.values()),
                "No frozen gradients or Adam states",
            )
            append(
                output / "updates.jsonl",
                {
                    "step": state["completed_updates"],
                    "case_id": row["record"]["case_id"],
                    "loss": float(loss.detach()),
                    "gradient_norm": float(norm),
                    "metrics": metrics,
                    "storage": storage_ledger(torch, model, optimizer),
                },
                MIB,
            )
            replace(output / "progress.json", state)
            del loss
        model.lm.eval()
        model.lm.gradient_checkpointing_disable()
        with torch.no_grad():
            for development_ref in inputs_value["rows"]["development"]:
                row = row_map[(development_ref["component"], development_ref["row_id"])]
                active.update(phase="development", case_id=row["record"]["case_id"])
                loss, metrics = gate.phase(
                    "native-development-forward",
                    lambda: native_cache_release(
                        torch,
                        model,
                        gate,
                        output,
                        lambda: operator_call(
                            torch,
                            output,
                            active["phase"],
                            lambda: fit_kernel.action_loss(
                                model, row, plan=plans[row["record"]["case_id"]], mode=mode
                            ),
                        ),
                        mode=mode,
                        case_id=row["record"]["case_id"],
                        step=state["completed_updates"],
                        phase="dev-forward",
                    ),
                )
                append(
                    output / "development.jsonl",
                    {
                        "case_id": row["record"]["case_id"],
                        "loss": float(loss),
                        "metrics": metrics,
                        "storage": storage_ledger(torch, model, optimizer),
                    },
                    MIB,
                )
                del loss
        require(
            state["completed_updates"] == artifact.COUNTS[inputs_value["regime"]]["train"]
            and len(trace["native_forward_calls"])
            == sum(artifact.COUNTS[inputs_value["regime"]].values()),
            "Full admitted120/129 updates and22/27 development forwards",
        )
        require(
            all(v._version == version and not v.requires_grad for v, version in frozen.values()),
            "Frozen base versions unchanged",
        )
        gate.phase(
            "final-full-head-hash",
            lambda: require(
                kernel._tensor_sha(model.head.weight)
                == "9619fd670a4074367c72f2cf09d777970ad1eafa511bff605a82c8558dcc24d4",
                "Unchanged full head",
            ),
        )
        write(output / "native-call-trace.json", trace)
        return storage_ledger(torch, model, optimizer)
    finally:
        state["actual_native_forward_calls"] = len(trace["native_forward_calls"])
        for hook in hooks:
            hook.remove()


class AdmissionGate:
    """New owned admission uses the unchanged original startup CPU floor."""

    def __init__(self, original, output):
        self.original, self.output = original, output

    def check(self, phase, scope):
        if phase == "before-owned-collection-admission":
            require(scope == "cpu", "Owned admission remains CPU-only")
            result = self.original.check("before-public-project-admission", "cpu")
            append(
                self.output / "owned-admission-gates.jsonl",
                {
                    "phase": phase,
                    "scope": scope,
                    "original_gate_sequence": self.original.sequence - 1,
                    "delegated_phase": "before-public-project-admission",
                    "required_bytes": result["required_bytes"],
                    "resident_credit_bytes": 0,
                },
                MIB,
            )
            return result
        return self.original.check(phase, scope)


def cache_release_count(output, inputs_value):
    rows = [
        artifact.parse(line) for line in memory.read(output / "cache-releases.jsonl").splitlines()
    ]
    count = artifact.expected_counts(inputs_value["regime"])["cache_releases"]
    require(len(rows) == 2 * count, "Every fixed cache release has intent and completion")
    expected = []
    for step, ref in enumerate(inputs_value["ordered_updates"], 1):
        expected.extend([(ref["row_id"], step, "train-forward"), (ref["row_id"], step, "backward")])
    expected.extend(
        (ref["row_id"], len(inputs_value["ordered_updates"]), "dev-forward")
        for ref in inputs_value["rows"]["development"]
    )
    for i, identity in enumerate(expected):
        intent, complete = rows[2 * i : 2 * i + 2]
        require(
            intent["event"] == "intent"
            and complete["event"] == "complete"
            and complete["success"] is True
            and complete["API_returned"] is True
            and complete["error"] is None,
            "No failed/unfinished cache release",
        )
        for row in (intent, complete):
            require(
                (row["case_id"], row["step"], row["native_phase"]) == identity
                and row["policy"] == memory.CACHE_POLICY
                and row["mode"] == inputs_value["mode"],
                "Fixed release bound to exact actual native call",
            )
    return count


def worker(config_path):
    """Fresh authenticated owned unit; all framework and native work is lazy."""
    config_path = memory.ordinary(config_path)
    output = config_path.parent
    config = artifact.parse(memory.read(config_path))
    profile = memory.validate_stage_config(config, output)
    state = {
        "schema_id": memory.RECEIPT_SCHEMA,
        "profile_id": memory.PROFILE,
        "kind": config["artifact_kind"],
        "qualified": False,
        "stage": "pre-framework-barrier",
        "mode": config["mode"],
        "allocator_policy": memory.CACHE_POLICY,
        "source_union": config["source_inventory"],
        "model_constructor_attempts": 0,
        "model_loads": 0,
        "verified_model_loads": 0,
        "optimizer_step_attempts": 0,
        "completed_updates": 0,
        "synchronized_updates": 0,
        "actual_native_forward_calls": 0,
        "actual_native_backward_calls": 0,
        "actual_cache_releases": 0,
        "actual_update_refs": [],
        "development_updates": 0,
        "saved_adapter": {"status": "not-started", "files": None},
        "counter_scope": "Durable intent precedes native calls; returned and synchronized counts are lower bounds after failure, never a claim of zero unrecorded partial effects",
        "error": None,
    }
    stop, errors = threading.Event(), []
    ticker = threading.Thread(
        target=memory.heartbeat, args=(output, config, state, stop, errors), daemon=True
    )
    ticker.start()
    primary, tools, model = None, None, None
    started = time.monotonic()
    try:
        config, profile, ready, binding = memory.await_worker_release(config_path, errors)
        state["unit_binding"] = binding
        plan = artifact.check_plan(
            artifact.parse(artifact.read_ref(profile["input_plans"][config["mode"]]))
        )
        require(
            plan["mode"] == config["mode"]
            and artifact.KINDS[plan["regime"]] == config["artifact_kind"]
            and plan["source_union"] == config["source_inventory"],
            "Same separately pinned worker input/profile/source",
        )
        import jev_browser_fit_public_cfg_actions as producer
        import jev_public_cfg_action_artifact as public

        producer._fresh_environment()
        old = producer.ResourceGate(output / "original-startup-gates.jsonl", config["boot_id"])
        state["stage"] = "native-toolchain-verification"
        tools = memory.prepare_native_tools(
            output,
            binding,
            artifact.parse(artifact.read_ref(plan["runtime"]["native_toolchain"])),
            old,
        )
        state["native_toolchain_runtime"] = plan["runtime"]["native_toolchain"]
        state["stage"] = "complete-public-owned-admission"
        inputs = artifact.admit_inputs(plan, AdmissionGate(old, output))
        inputs_value = inputs.value()
        state["admitted_inputs_sha256"] = artifact.sha(artifact.encode(inputs_value))
        state["versions"] = public.verify_reference_environment(inputs._public.reference_items)
        artifact.verify_CPU_reference(
            profile["CPU_references"][config["mode"]], plan, descriptor_value=inputs_value
        )
        old.check("immediately-before-legacy-parent-capture-and-Torch", "cpu")
        api = public.trusted_module("s1.browser_action_adapter", "src/s1/browser_action_adapter.py")
        memory.arm_bound(output, pending=public.PARENT_TOTAL_BYTES + 24 * MIB)
        captured = api.capture_initial_action_adapter(
            ROOT / "artifacts/jev-browser/20261015-action-lora-counterfactual-fit-01/adapter",
            output / "initial-parent",
        )
        verified_parent = public.validate_parent_capture(
            captured,
            inputs._public.metadata()["provenance"],
            expected_descriptor=inputs._public.metadata()["parent_descriptor"],
            gate=old,
        )
        state["parent"] = {
            "files": public.PARENT_FILE_COUNT,
            "bytes": public.PARENT_TOTAL_BYTES,
            "historical_updates": 521,
            "descriptor": captured["descriptor"],
        }
        memory.apply_post_parent_file_limits()
        state["stage"] = "runtime-binary-verification"
        runtime = artifact.parse(artifact.read_ref(plan["runtime"]["allocator"]))
        dependency_runtime = artifact.parse(
            artifact.read_ref(plan["runtime"]["optional_dependency"])
        )
        memory.verify_allocator_files(runtime, dependency_runtime)
        import torch

        state["allocator_loaded_origins"] = memory.allocator_origins(torch, runtime)
        cap = memory.allocator_cap(torch, config["device_uuid"])
        state["allocator"] = cap
        state["optional_dependency_runtime"] = dependency_runtime
        gate = memory.StageGate(torch, output, config, binding, cap)
        state["stage"] = "full-checkpoint-hash"
        old.check("before-checkpoint-payload-hash-and-BF16-loader", "load")
        checkpoint = public.resolve_base_checkpoint(producer.CACHE)
        gate.phase(
            "full-original-checkpoint-hash",
            lambda: require(
                public.pin(checkpoint, limit=producer.BASE_BYTES) == public.BASE_CHECKPOINT,
                "Exact full base checkpoint",
            ),
        )
        old.check("before-any-BF16-model-construction", "train")
        kernel = public.trusted_module(
            "jev_browser_fit_actions", "scripts/jev_browser_fit_actions.py"
        )
        fit_kernel = public.trusted_module(
            "jev_browser_fit_cfg_actions", "scripts/jev_browser_fit_cfg_actions.py"
        )
        state["stage"] = "constructor"

        def construct():
            state["model_constructor_attempts"] += 1
            replace(output / "progress.json", state)
            result = producer._construct_model(42)
            state["model_loads"] += 1
            replace(output / "progress.json", state)
            memory.require_no_forbidden_modules()
            state["nonquantized_constructor"] = memory.require_unquantized_modules(result.lm)
            state["optional_dependency_origins"] = memory.optional_dependency_origins(
                dependency_runtime
            )
            return result

        model = gate.phase("full-original-constructor", construct)
        gate.loaded = True
        gate.phase("original-model-head-check", lambda: producer._model_check(model, kernel))
        state["verified_model_loads"] += 1
        replace(output / "progress.json", state)
        require(
            public.verify_reference_environment(inputs._public.reference_items, model=model)
            == state["versions"],
            "Original actual processor51 origins",
        )
        for adapter_config in model.lm.peft_config.values():
            adapter_config.base_model_name_or_path, adapter_config.revision = (
                public.MODEL_ID,
                public.REVISION,
            )
        state["stage"] = "parent-injection"
        state["parent_injection"] = gate.phase(
            "original-parent-injection",
            lambda: api.initialize_training_action_adapter(model, verified_parent),
        )
        state["stage"] = "complete-native-compilation"
        compiled = output / "compiled"
        compiled.mkdir(mode=0o700)
        row_map, plans = gate.phase(
            "complete-original-plus-owned-processor-compile",
            lambda: artifact.compile_inputs(model, inputs, compiled, gate),
        )
        artifact.verify_CPU_reference(
            profile["CPU_references"][config["mode"]],
            plan,
            compiled_root=compiled,
            descriptor_value=inputs_value,
        )
        gate.phase("postcompile-original-model-check", lambda: producer._model_check(model, kernel))
        state["stage"] = "complete-native-training-and-development"
        state["final_storage"] = train_stage(
            torch,
            model,
            row_map,
            plans,
            mode=config["mode"],
            gate=gate,
            kernel=kernel,
            fit_kernel=fit_kernel,
            output=output,
            state=state,
            inputs_value=inputs_value,
        )
        state["actual_cache_releases"] = cache_release_count(output, inputs_value)
        state["head_sha256"] = public.HEAD_SHA

        def final_model_check():
            producer._model_check(model, kernel)
            state["nonquantized_final"] = memory.require_unquantized_modules(model.lm)
            state["optional_dependency_origins_final"] = memory.optional_dependency_origins(
                dependency_runtime
            )

        gate.phase("final-original-model-check", final_model_check)
        gate.phase("final-original-parent-check", lambda: producer._parent_checks(captured))
        state["stage"] = "guarded-complete-adapter-save"
        artifact.guarded_save_adapter(model, inputs, output, gate, kernel, state)
        inputs.check()
        artifact.check_source_union(plan)
        state.update(
            peak_allocated_bytes=gate.peak_allocated,
            peak_reserved_bytes=gate.peak_reserved,
            allocator_peak_scope="Maximum of preserved phases and pre/post fixed cache-release peaks; no cache-release reset",
            stage="awaiting-native-work-final-capture",
        )
        memory.arm_bound(output)
        require(not errors, "Independent heartbeat failed")
        state["native_work_end_capture"] = memory.wait_native_work_capture(
            output, ready, binding, errors
        )
        state["stage"] = "complete_stage_fit_unqualified"
        artifact.verify_worker_receipt(state, inputs_value=inputs_value, config=config)
    except BaseException as exc:
        primary = exc
        state.update(
            error={"type": type(exc).__name__, "message": str(exc)},
            failed_stage=state["stage"],
            stage="failed_stage_memory",
        )
        raise
    finally:
        stop.set()
        ticker.join(timeout=2)
        if ticker.is_alive():
            errors.append(RuntimeError("Heartbeat thread closure unknown"))
        state["elapsed_seconds"] = time.monotonic() - started
        if tools is not None:
            state["native_toolchain_authorized_entries"] = tools.authorizations
            state["native_toolchain_prepared"] = tools.runtime is not None
        memory.finish_worker(output, state, primary=primary, cleanup_errors=errors)


def check_launch_profile(profile, *, gate=None):
    require(
        profile.get("schema_id") == "jev-stage-memory-profile-v2"
        and profile.get("profile_id") == memory.PROFILE
        and profile.get("cache_policy") == memory.CACHE_POLICY,
        "New v2 source/profile identity",
    )
    require(
        profile.get("regime") in artifact.COUNTS
        and profile.get("launch_status") == "reviewed-local-full-fit"
        and profile.get("evidence_budget", {}).get("status") == "reviewed-fits-unchanged-H13-caps",
        "Full-fit source/capacity/byte-budget review still pending; never auto-extend caps",
    )
    require(gate is not None, "Fresh controller CPU gate before compiled reference payload")
    gate.check("before-controller-CPU-reference-admission", "cpu")
    for key in ("capacity_review", "source_review", "evidence_budget_review"):
        artifact.read_ref(profile[key])
    require(
        set(profile["input_plans"]) == set(profile["CPU_references"]) == {"A", "C"},
        "Separate externally pinned A then C full-fit preparations",
    )
    for mode in ("A", "C"):
        plan = artifact.check_plan(artifact.parse(artifact.read_ref(profile["input_plans"][mode])))
        require(
            plan["mode"] == mode
            and plan["regime"] == profile["regime"]
            and plan["source_union"] == profile["source_inventory"],
            "Matching mode/data/source regime",
        )
        artifact.verify_CPU_reference(profile["CPU_references"][mode], plan)
    return profile


def campaign(profile_path, expected_profile_sha256, output, device_uuid):
    profile_path = memory.ordinary(profile_path)
    require(
        memory.file_pin(profile_path)["sha256"] == expected_profile_sha256,
        "Externally reviewed local full-fit profile SHA",
    )
    require(
        profile_path.is_relative_to(ROOT / "artifacts/jev-runtime"),
        "Owned reviewed profile namespace",
    )
    profile = artifact.parse(memory.read(profile_path))
    output = memory.ordinary(output)
    require(
        output.parent == profile_path.parent and not output.exists(),
        "Fresh direct-child owned campaign",
    )
    output.mkdir(mode=0o700)
    state = {
        "schema_id": artifact.RECEIPT_SCHEMA,
        "status": "running",
        "qualified": False,
        "regime": profile["regime"],
        "profile": artifact.reference(profile_path),
        "arms": {},
        "error": None,
    }
    primary = None
    try:
        startup_gate = artifact.CPUPrepareGate(output, memory.boot_id())
        profile = check_launch_profile(profile, gate=startup_gate)
        for mode in ("A", "C"):
            arm = output / mode
            arm.mkdir(mode=0o700)
            config = {
                "schema_id": "jev-stage-memory-config-v2",
                "profile_id": memory.PROFILE,
                "artifact_kind": artifact.KINDS[profile["regime"]],
                "mode": mode,
                "nonce": uuid.uuid4().hex,
                "boot_id": memory.boot_id(),
                "source_pin": memory.file_pin(Path(__file__)),
                "profile_pin": memory.file_pin(profile_path),
                "capsule_pin": memory.file_pin(Path(memory.__file__)),
                "worker_entry": str(Path(__file__).resolve()),
                "profile_path": str(profile_path),
                "source_inventory": profile["source_inventory"],
                "campaign_output": str(output),
                "campaign_root": str(output.parent),
                "device_uuid": device_uuid,
            }
            state["arms"][mode] = memory.arm_run(arm, config)
            # Every worker and its owned tool descendants are closed before
            # controller tensor loading and the next independent arm.
            validation = output / ("controller-validation-" + mode)
            validation.mkdir(mode=0o700)
            gate = artifact.CPUPrepareGate(validation, memory.boot_id())
            plan = artifact.parse(artifact.read_ref(profile["input_plans"][mode]))
            artifact.seal_closed_child(
                arm, config=config, plan=plan, CPU_ref=profile["CPU_references"][mode], gate=gate
            )
            manifest_sha = memory.file_pin(arm / "child-manifest.json")["sha256"]
            artifact.verify_completed_child(
                arm, expected_manifest_sha256=manifest_sha, plan=plan, gate=gate
            )
            state["arms"][mode] = {
                "closed_receipt": memory.file_pin(arm / "arm-receipt.json"),
                "child_manifest": memory.file_pin(arm / "child-manifest.json"),
                "success": True,
                "qualification": False,
            }
        state["status"] = "complete-local-stage-campaign-unqualified"
    except BaseException as exc:
        primary = exc
        state.update(
            status="failed-local-stage-campaign",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    finally:
        try:
            state["inventory_excluding_campaign_receipt"] = memory.final_inventory(
                output, reserved=4 * MIB
            )
            write(output / "campaign-receipt.json", state)
        except BaseException as exc:
            if primary is not None:
                primary.add_note("Campaign evidence finalization: " + str(exc))
            else:
                raise
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--profile-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device-uuid")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.worker is not None:
        worker(args.worker)
        return 0
    require(
        args.run
        and args.profile is not None
        and args.profile_sha256
        and args.output is not None
        and args.device_uuid,
        "Explicit reviewed full campaign command required",
    )
    campaign(args.profile, args.profile_sha256, args.output, args.device_uuid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
