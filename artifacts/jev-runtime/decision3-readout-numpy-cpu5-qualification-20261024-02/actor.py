#!/usr/bin/env python3
"""Source-only CPU5 successor: one-call nontraining NumPy zero-delta proof."""

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
SOURCE_ONLY = False
NATIVE = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261024-05"
NATIVE_SHA = "9a7e5af918c17f6824a720fb6d112ffa5fdac03493dde3efa176c7d50470a365"
NATIVE_HELPER_SHA = "4df54e06e8a795ebf5b300a3e2074319e781a278a2a870c21c2bd52897d7eb11"
BASE_PLAN_SHA = "095db88b78687d90fbed51aafd736c1b06fb1a582312de8dff97bf6109131e96"
OWNER_CONTROLLER_SHA = "81846efe2ee0357da07283c1a77a606283a920cbddef668c8f62def291057dee"
OWNER_FREEZE_SHA = "52ac647a80ca1ef0c05acc45bf90c990f3b753b66df216c87cfb2bdf495020ef"
CORE = ROOT / "artifacts/jev-runtime/decision3-cached-head-core-20261023-01/head_core.py"
CORE_SHA = "6761dd654481424f4d6c908a7c9fd2d43e4762eeb5420af9f22612ffe77657ed"
PROFILE = "decision3-edge-numpy-zero-delta-nontraining-cpu5-fast-4g-coexist-20261024-v1"
HOST_POLICY = "comfy-observed-cpu-coexist-v1"
WALL_SECONDS = 120
CALL_SECONDS = 20
HEADS = ("operation", "click_target", "type_text_target", "select_target")
COUNTS = (8, 1, 2, 3)
MODEL_MODULES = {"torch", "transformers", "safetensors", "numpy", "s1"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def require_released_source():
    require(
        SOURCE_ONLY is False,
        "SOURCE_ONLY CPU5 candidate: source checks/pins, selected plan, honest root prerequisite "
        "binding and fresh owned qualification remain pending",
    )


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
    expected = {
        HERE / "actor.py",
        HERE / "observer.py",
        HERE / "fixture.json",
        HERE / "numpy_probe.py",
        CORE,
        *(
            NATIVE / name
            for name in (
                "actor.py",
                "resource_helper.py",
                "source_audit.py",
                "edge-plan.json",
                "freeze.json",
            )
        ),
    }
    records = plan["source_records"]
    require(
        type(records) is list
        and len(records) == len(expected)
        and {Path(row["path"]) for row in records} == expected,
        "exact executed source projection required without missing or duplicate records",
    )
    for row in records:
        path = Path(row["path"])
        require(
            path.is_relative_to(ROOT)
            and path.resolve() == path
            and type(row["size"]) is int
            and 0 < row["size"] < 100_000,
            "small repository source reference required",
        )
        helper.check_record(row, full=True)


def load_plan(helper, path, expected):
    require(expected and sha(path) == expected, "selected replay plan SHA differs")
    plan = helper.read_json(path)
    require(
        plan.get("executable") is True
        and plan["schema"] == "decision3-owned-numpy-zero-delta-nontraining-plan-v1"
        and plan["profile"] == PROFILE
        and plan["actor_sha256"] == sha(__file__)
        and plan["native_actor_sha256"] == NATIVE_SHA
        and plan["native_helper_sha256"] == NATIVE_HELPER_SHA
        and plan["native_plan_sha256"] == BASE_PLAN_SHA
        and plan["owner_controller_sha256"] == OWNER_CONTROLLER_SHA
        and plan["owner_freeze_sha256"] == OWNER_FREEZE_SHA
        and plan["head_core_sha256"] == CORE_SHA
        and plan["fixture_sha256"] == sha(HERE / "fixture.json")
        and plan["native_plan_path"] == str(NATIVE / "edge-plan.json"),
        "replay/native source or profile differs",
    )
    require(
        plan["wall_seconds"] == WALL_SECONDS
        and plan["call_seconds"] == CALL_SECONDS
        and type(plan["requests"]) is int
        and plan["requests"] == 1
        and plan["head_counts"] == list(COUNTS)
        and all(type(count) is int for count in plan["head_counts"])
        and plan["absolute_tolerance"] == 1e-6
        and plan["relative_tolerance"] == 1e-5
        and plan["temperature"] == 1.0
        and plan["catalog_mode"] == "owned_actual_discovery_not_preselected"
        and plan["training_eligible"] is False,
        "fixed replay shape/tolerances/budgets differ",
    )
    require(
        plan["owner_unit"] == "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service"
        and plan["owner_profile"] == "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"
        and plan["host_observation_policy"] == HOST_POLICY
        and plan["memory_max_bytes"] == 4294967296
        and plan["reserve_bytes"] == 26119172916
        and plan["startup_available_bytes"] == 30414140212
        and plan["swap_max_bytes"] == 0
        and plan["owner_runtime_seconds"] == 10800
        and plan["cpu_affinity"] == [15, 16, 17, 18, 19]
        and plan["cpu_quota_percent"] == 500
        and type(plan["threads"]) is int
        and plan["threads"] == 5
        and type(plan["interop_threads"]) is int
        and plan["interop_threads"] == 1,
        "owner09 CPU5-coexist full-cap resource contract differs",
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


def perform_call(
    native, helper, observer_module, probe, runtime, torch, base, request, output, state
):
    began = time.monotonic()
    handles = []
    row = {"completed": False, "status": "running", "feature_commit_status": "PENDING_NONTRAINING"}
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
        # This is deliberately after the unchanged loader's no-preimport guard.
        np = probe.load_numpy(native, helper, state)
        initial_catalog = probe.catalog(np, native, helper, began)
        core = load_source("_qualified_numpy_head_core", CORE, CORE_SHA)
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
                        {"raw_response": value, "feature_commit_status": "PENDING_NONTRAINING"},
                    )
                except (TypeError, ValueError, OverflowError):
                    helper.write_json(
                        output / "raw-native-response.json",
                        {
                            "invalid_raw_diagnostic": helper.diagnostic(value),
                            "feature_commit_status": "PENDING_NONTRAINING",
                            "valid_response_claim": False,
                        },
                    )

        backend = native.NativeBatch4Backend(runtime, torch, base["model"], evidence_sink=sink)
        prepared, lengths = backend.prepare(**request)
        require(
            list(lengths) == list(HEADS)
            and set(prepared.sequences) == set(HEADS)
            and max(lengths.values()) <= 16384,
            "all four original full text prompts required; no truncation",
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
        row["native_contract"] = probe.validate_answers(
            native, raw, request, lengths, base["model"]
        )
        witness.require_single_pass()
        replayed = observer_module.replay_probabilities(runtime, torch, witness, COUNTS)
        row["replay"] = observer_module.compare_distributions(raw, request["questions"], replayed)
        row["numpy_zero_delta"] = probe.qualify(
            np,
            core,
            observer_module,
            native,
            helper,
            runtime,
            torch,
            witness,
            request["questions"],
            raw,
            COUNTS,
            output,
            began,
            initial_catalog,
        )
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
        # Final full authentication is inside the complete20s transaction too.
        native.authenticate(helper, base)
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
    require_released_source()
    require(
        sys.flags.no_user_site
        and sys.dont_write_bytecode
        and not MODEL_MODULES.intersection(sys.modules),
        "fresh -B -s process without model imports required",
    )
    native = load_source("_numpy_qualification_frozen_native05", NATIVE / "actor.py", NATIVE_SHA)
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
    probe = load_source(
        "_owned_numpy_qualification", HERE / "numpy_probe.py", plan["numpy_probe_sha256"]
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
        "selected_owner_host_observation_policy": HOST_POLICY,
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
        "feature_commit_status": "PENDING_NONTRAINING",
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
        "scope": "One nontraining NumPy/runtime discovery and same-call zeroDelta proof only. Root exit0, accepted owner09/current closure, full auth and independent actual audit required; no preselected full DSO closure, data collection, training or quality qualification. Native05 prior root review is a distinct prerequisite, not an independent audit of this candidate.",
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
        perform_call(
            native, helper, observer_module, probe, runtime, torch, base, request, output, state
        )
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
            "numpy_zero_delta_qualification_complete_pending_external_acceptance_nontraining"
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


def main(argv=None):
    require_released_source()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_default_no_model_or_job_execution",
                    "source_only": SOURCE_ONLY,
                    "profile": PROFILE,
                    "requests": 1,
                    "partial_rows": 4,
                    "legal_counts": list(COUNTS),
                    "whole_seconds_from_execute_entry": WALL_SECONDS,
                    "complete_prediction_seconds": CALL_SECONDS,
                    "owner_unit": "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service",
                    "cpu_affinity": [15, 16, 17, 18, 19],
                    "cpu_quota_percent": 500,
                    "threads": 5,
                    "interop_threads": 1,
                    "memory_max_bytes": 4294967296,
                    "feature_commit_status": "PENDING_NONTRAINING",
                    "labels": None,
                    "training_eligible": False,
                }
            )
        )
        return
    require_released_source()
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
