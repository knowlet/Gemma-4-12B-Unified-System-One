#!/usr/bin/env python3
"""Dry-default, one-call frozen Edge final-feature/readout replay proof."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NATIVE = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261023-02"
NATIVE_SHA = "977004620b2fdf2d47c683fdea7adb418e9a74d8765040ceb014d3a54d4cc88b"
BASE_PLAN_SHA = "115b7850ac80b35aa0aacb5880644e1ee7ea79deb59beb748c7947c86d0492e8"
PROFILE = "decision3-edge-original-readout-feature-replay-cpu-2g-20261023-v1"
WALL_SECONDS = 120
CALL_SECONDS = 20
HEADS = ("envelope_record", "drawer_record", "catalog_record")
COUNTS = (2, 3, 4)
MODEL_MODULES = {"torch", "transformers", "safetensors", "numpy", "s1"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source(name, path, expected):
    path = Path(path)
    require(
        path.suffix == ".py" and path.stat().st_size < 100_000, "small Python dependency required"
    )
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "selected dependency SHA differs")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def authenticate_sources(helper, plan):
    for row in plan["source_records"]:
        require(
            Path(row["path"]).is_relative_to(ROOT) and row["size"] < 100_000,
            "small repository source reference required",
        )
        helper.check_record(row, full=True)


def load_plan(helper, path, expected):
    require(expected and sha(path) == expected, "selected replay plan SHA differs")
    plan = helper.read_json(path)
    require(
        plan["profile"] == PROFILE
        and plan["actor_sha256"] == sha(__file__)
        and plan["native_actor_sha256"] == NATIVE_SHA
        and plan["native_plan_sha256"] == BASE_PLAN_SHA
        and plan["native_plan_path"] == str(NATIVE / "edge-plan.json"),
        "replay/native source or profile differs",
    )
    require(
        plan["wall_seconds"] == WALL_SECONDS
        and plan["call_seconds"] == CALL_SECONDS
        and plan["requests"] == 1
        and plan["head_counts"] == list(COUNTS)
        and plan["absolute_tolerance"] == 1e-6
        and plan["relative_tolerance"] == 1e-5
        and plan["temperature"] == 1.0,
        "fixed replay shape/tolerances/budgets differ",
    )
    require(
        plan["owner_unit"] == "jev-d3-llama-cpp-cpu4-2g-long-20261023-06.service"
        and plan["owner_profile"] == "llama-cpp-cpu4-2g-long-command-owned-v1"
        and plan["memory_max_bytes"] == 2147483648
        and plan["reserve_bytes"] == 26119172916
        and plan["startup_available_bytes"] == 28266656564
        and plan["swap_max_bytes"] == 0
        and plan["owner_runtime_seconds"] == 10800
        and plan["cpu_affinity"] == [16, 17, 18, 19]
        and plan["cpu_quota_percent"] == 400,
        "original owner06 full-cap resource contract differs",
    )
    authenticate_sources(helper, plan)
    fixture = helper.read_json(HERE / "fixture.json")
    request = fixture["request"]
    require(
        fixture["semantic_expected_answers"] is None
        and list(request["questions"]) == list(HEADS)
        and [len(q["criteria"]) for q in request["questions"].values()] == list(COUNTS)
        and all(q["type"] == "choice" for q in request["questions"].values()),
        "predeclared complete source fixture differs",
    )
    return plan, request


def remove_handles(handles):
    primary = None
    for handle in list(reversed(handles)):
        try:
            handle.remove()
            handles.remove(handle)
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                primary.add_note("Additional observation hook cleanup failed: " + repr(error))
    if primary is not None:
        raise primary


def transaction(native, helper, output, stage, began, budget, operation, cleanup, finalize, demote):
    """Literal success boundary after evidence, cleanup and watchdog exit."""
    primary = None
    remaining = budget - (time.monotonic() - began)
    require(remaining > 0, stage + " preflight exhausted fixed budget")
    try:
        with native.DeadlineScope(helper.Deadline(remaining, output, stage)):
            try:
                result = operation()
            except BaseException as error:
                primary = error
                demote(error)
                raise
            finally:
                cleanup_error = None
                try:
                    cleanup()
                except BaseException as error:
                    if primary is None:
                        primary, cleanup_error = error, error
                        demote(error)
                    else:
                        primary.add_note("Observation cleanup also failed: " + repr(error))
                finalize(primary)
                if cleanup_error is not None:
                    raise cleanup_error
        elapsed = helper.ensure_elapsed(
            began, budget, stage + " after evidence/hook/watchdog cleanup"
        )
    except BaseException as outer:
        if primary is None:
            primary = outer
        elif primary is not outer:
            primary.add_note("Outer timing/cleanup also failed: " + repr(outer))
        demote(primary)
        finalize(primary)
        if primary is outer:
            raise
        raise primary
    return result, elapsed


def perform_call(native, helper, observer_module, runtime, torch, base, request, output, state):
    began = time.monotonic()
    handles = []
    row = {"completed": False, "status": "running", "feature_commit_status": "PENDING"}
    state["calls"].append(row)

    def demote(error):
        row.update(
            completed=False,
            status="failed_no_replay_claim",
            error={"type": type(error).__name__, "message": str(error)},
        )

    def finalize(primary):
        row["elapsed_before_last_state_write_seconds"] = time.monotonic() - began
        helper.finalize_state(output, state, primary)

    def operation():
        native.validate_runtime(runtime, torch, base["model"])
        before = observer_module.baseline_identity(runtime)
        require(
            runtime.encoder.language_model is runtime.backbone.language_model,
            "genuine publisher PromptEncoder/final-norm join differs",
        )

        def sink(kind, value):
            if kind == "raw_returned_before_validation":
                try:
                    helper.write_json(
                        output / "raw-native-response.json",
                        {"raw_response": value, "feature_commit_status": "PENDING"},
                    )
                except (TypeError, ValueError, OverflowError):
                    helper.write_json(
                        output / "raw-native-response.json",
                        {
                            "invalid_raw_diagnostic": helper.diagnostic(value),
                            "feature_commit_status": "PENDING",
                            "valid_response_claim": False,
                        },
                    )

        backend = native.NativeBatch4Backend(runtime, torch, base["model"], evidence_sink=sink)
        prepared, lengths = backend.prepare(**request)
        require(
            list(lengths) == list(HEADS)
            and set(prepared.sequences) == set(HEADS)
            and max(lengths.values()) <= 16384,
            "all three original full text prompts required; no truncation",
        )
        witness = observer_module.FinalNormObserver(torch, len(HEADS), max(lengths.values()))
        handles.append(
            runtime.backbone.language_model.layers[0].register_forward_pre_hook(witness.first_layer)
        )
        handles.append(runtime.backbone.language_model.norm.register_forward_hook(witness))
        row["input_tokens_per_question"] = lengths
        row["actual_partial_rows_expected"] = len(HEADS)
        row["full_source_fixture_request"] = request
        helper.write_json(output / "state.json", state)
        state["native_request_calls"] += 1
        raw = backend(**copy.deepcopy(request))
        row["native_contract"] = native.validate_smoke_answers(raw, request, lengths, base["model"])
        witness.require_single_pass()
        replayed = observer_module.replay_probabilities(runtime, torch, witness, COUNTS)
        row["replay"] = observer_module.compare_distributions(raw, request["questions"], replayed)
        witness.require_single_pass()
        require(
            observer_module.baseline_identity(runtime) == before,
            "live baseline tensor identity/version changed",
        )
        native.validate_runtime(runtime, torch, base["model"])
        row["baseline_tensor_identity_and_versions_unchanged"] = True
        row["first_layer_events"] = witness.first_layer_events
        row["final_norm_events"] = witness.norm_events
        row["extra_encoder_passes"] = len(witness.first_layer_events) - 1
        row["extra_final_norm_passes"] = len(witness.norm_events) - 1
        helper.write_json(
            output / "features-pending.json",
            {
                "status": "PENDING_EXTERNAL_TRANSACTION_ACCEPTANCE_NO_TRAINING_ELIGIBILITY",
                "plan_sha256": state["plan_sha256"],
                "ordered_heads": list(HEADS),
                "legal_counts": list(COUNTS),
                "dtype": "float32",
                "shape": [len(HEADS), 1024],
                "features": witness.features.cpu().tolist(),
                "replayed_probabilities": replayed,
                "training_eligible": False,
            },
        )
        row["completed"] = True
        row["status"] = "replay_candidate_pending_hook_and_watchdog_cleanup"
        return row

    result, elapsed = transaction(
        native,
        helper,
        output,
        "complete-prediction",
        began,
        CALL_SECONDS,
        operation,
        lambda: remove_handles(handles),
        finalize,
        demote,
    )
    result["complete_prediction_seconds"] = elapsed
    result["status"] = "replay_verified_pending_external_transaction_acceptance"
    return result


def execute(plan_path, expected, output):
    began = time.monotonic()
    require(
        sys.flags.no_user_site
        and sys.dont_write_bytecode
        and not MODEL_MODULES.intersection(sys.modules),
        "fresh -B -s process without model imports required",
    )
    native = load_source("_feature_replay_frozen_native02", NATIVE / "actor.py", NATIVE_SHA)
    helper = native.load_helper()
    plan, request = load_plan(helper, plan_path, expected)
    identity = helper.process_identity()
    limits = helper.owned_limits(identity)
    native.validate_owner(identity, limits, helper.START_ENV)
    gate = helper.host_gate()
    base = native.load_plan(helper, Path(plan["native_plan_path"]), BASE_PLAN_SHA)
    observer_module = load_source(
        "_feature_replay_observer", HERE / "observer.py", plan["observer_sha256"]
    )
    output = Path(output).resolve(strict=False)
    require(
        output.is_relative_to(HERE.parent.resolve()) and not output.exists(),
        "fresh resolved output in runtime tree required",
    )
    helper.ensure_elapsed(began, WALL_SECONDS, "whole replay entry/preflight")
    output.mkdir(parents=True, exist_ok=False)
    (output / ".gitignore").write_text("*\n")
    state = {
        "profile": PROFILE,
        "plan_sha256": expected,
        "status": "authenticating",
        "process": identity,
        "owned_limits": limits,
        "host_startup_admission": gate,
        "model_key": "edge",
        "model_loads": 0,
        "native_request_calls": 0,
        "calls": [],
        "completed": False,
        "feature_commit_status": "PENDING",
        "training_eligible": False,
        "benchmark_calls": 0,
        "browser_calls": 0,
        "benchmark_scores": None,
        "http_qualified": False,
        "browser_qualified": False,
        "model_qualified": False,
        "no_escape_audited": False,
        "whole_tree_audited": False,
        "continuation_qualified": False,
        "scope": "One original-readout replay proof, not semantic quality or corpus collection; root exit0 + original accepted owner/closure + full auth still required externally",
    }

    def demote(error):
        state.update(
            completed=False,
            status="failed_no_replay_claim",
            error={"type": type(error).__name__, "message": str(error)},
        )
        for row in state["calls"]:
            row["completed"] = False
            row["status"] = "failed_no_external_transaction_acceptance"

    def finalize(primary):
        state["elapsed_seconds"] = time.monotonic() - began
        helper.finalize_state(output, state, primary)

    def operation():
        runtime, torch = native.load_runtime(helper, base, output, state)
        perform_call(native, helper, observer_module, runtime, torch, base, request, output, state)
        require(
            state["model_loads"] == state["native_request_calls"] == len(state["calls"]) == 1,
            "one constructor and one original native call required",
        )
        native.authenticate(helper, base)
        authenticate_sources(helper, plan)
        require(sha(plan_path) == expected, "selected replay plan changed")
        native.validate_runtime(runtime, torch, base["model"])
        state["final_full_model_manifest_runtime_source_identities_unchanged"] = True
        state["after_final_auth_owned_counters"] = native.own_counters(identity)
        state["cuda_initialized_final"] = torch.cuda.is_initialized()
        state["completed"] = True
        state["status"] = (
            "original_readout_feature_replay_complete_pending_external_acceptance_no_training_claim"
        )
        return state

    result, elapsed = transaction(
        native,
        helper,
        output,
        "whole-feature-replay",
        began,
        WALL_SECONDS,
        operation,
        lambda: None,
        finalize,
        demote,
    )
    result["post_cleanup_elapsed_seconds"] = elapsed
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_default_no_runtime_or_payload_execution",
                    "profile": PROFILE,
                    "requests": 1,
                    "partial_rows": 3,
                    "legal_counts": list(COUNTS),
                    "whole_seconds_from_execute_entry": WALL_SECONDS,
                    "complete_prediction_seconds": CALL_SECONDS,
                    "feature_commit_status": "PENDING",
                    "training_eligible": False,
                }
            )
        )
        return
    require(args.plan is not None and args.output is not None, "explicit plan and output required")
    result = execute(args.plan, args.plan_sha256, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "post_cleanup_elapsed_seconds": result["post_cleanup_elapsed_seconds"],
                "output": str(args.output),
                "training_eligible": False,
            }
        )
    )


if __name__ == "__main__":
    main()
