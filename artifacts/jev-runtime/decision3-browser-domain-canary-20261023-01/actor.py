#!/usr/bin/env python3
"""Dry-default six-request invented browser-domain diagnostic; no scores."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NATIVE_PATH = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261023-02/actor.py"
NATIVE_SHA = "977004620b2fdf2d47c683fdea7adb418e9a74d8765040ceb014d3a54d4cc88b"
NATIVE_PLAN = NATIVE_PATH.with_name("edge-plan.json")
NATIVE_PLAN_SHA = "115b7850ac80b35aa0aacb5880644e1ee7ea79deb59beb748c7947c86d0492e8"
PROFILE = "decision3-edge-invented-browser-domain-canary-cpu4-2g-v1"
CALL_SECONDS = 20
WALL_SECONDS = 120
CELLS = [
    (key, order)
    for key in ("actionable", "achieved", "unavailable")
    for order in ("original", "reversed")
]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_native():
    require(sha(NATIVE_PATH) == NATIVE_SHA, "frozen native02 source changed")
    spec = importlib.util.spec_from_file_location("_canary_pinned_native02", NATIVE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(
        compile(NATIVE_PATH.read_bytes(), str(NATIVE_PATH), "exec", dont_inherit=True),
        module.__dict__,
    )
    return module


def authenticate_sources(helper, plan):
    for row in plan["sources"]:
        path = Path(row["path"])
        require(
            path.is_absolute()
            and path.is_relative_to(ROOT)
            and ".." not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents))
            and (path.suffix in (".py", ".json", ".js") or path.name == "METADATA")
            and row["size_bytes"] < 200_000,
            "selected ordinary small ROOT source required",
        )
        require(
            path.stat().st_size == row["size_bytes"] and sha(path) == row["sha256"],
            "canary source changed",
        )
    require(sha(__file__) == plan["actor_sha256"], "canary actor changed")
    require(sha(HERE / "builder.py") == plan["builder_sha256"], "canary builder changed")
    require(sha(HERE / "fixtures.json") == plan["fixtures_sha256"], "invented fixture changed")


def load_plan(native, helper, path, expected):
    require(expected and sha(path) == expected, "explicit canary plan SHA required")
    plan = helper.read_json(path)
    require(
        plan["profile"] == PROFILE and plan["actor_sha256"] == sha(__file__),
        "canary identity changed",
    )
    require(
        plan["native_actor_sha256"] == NATIVE_SHA and plan["native_plan_sha256"] == NATIVE_PLAN_SHA,
        "frozen native dependency changed",
    )
    require(
        plan["cells"] == [list(v) for v in CELLS]
        and plan["calls"] == 6
        and plan["constructor_count"] == 1,
        "six predeclared cells/one constructor required",
    )
    require(
        plan["call_seconds"] == CALL_SECONDS
        and plan["wall_seconds"] == WALL_SECONDS
        and plan["browser_context_limit"] == 8192,
        "original request/whole budgets changed",
    )
    require(
        plan["controls"] == [] and plan["adaptive_controls"] is False,
        "no adaptive or additional controls selected",
    )
    require(
        plan["raw_actual_outputs_git_ignored"] is True and plan["official_tasks_used"] is False,
        "diagnostic data boundary changed",
    )
    base = native.load_plan(helper, NATIVE_PLAN, NATIVE_PLAN_SHA)
    for name in (
        "owner_unit",
        "owner_profile",
        "owner_runtime_seconds",
        "reserve_bytes",
        "memory_max_bytes",
        "startup_available_bytes",
        "swap_max_bytes",
        "cpu_affinity",
        "cpu_quota_percent",
        "threads",
        "batch_size",
        "runtime_python",
    ):
        require(plan[name] == base[name], "canary resource/owner join changed: " + name)
    authenticate_sources(helper, plan)
    return plan, base


class BrowserBridge:
    """Original native DTO API over one frozen already-loaded CPU backend."""

    def __init__(self, native, runtime, torch, model, provider, evidence):
        def sink(event, value):
            if event == "raw_returned_before_validation":
                evidence["raw_native_response"] = copy.deepcopy(value)
            elif event == "prepared":
                evidence["native_backend_preflight"] = copy.deepcopy(value)

        self.backend = native.NativeBatch4Backend(runtime, torch, model, evidence_sink=sink)
        self.spec, self.name = (
            provider.decision3_spec("d3-edge", device="cpu").model_copy(
                update={
                    "context_limit": 8192,
                    "notes": "Explicit fresh CPU native-batch4 diagnostic; no CUDA or HTTP qualification.",
                }
            ),
            "d3-edge",
        )
        self.provider, self.evidence = provider, evidence
        self.calls = 0

    def predict(self, request, *, deadline=None, evidence_sink=None):
        if deadline is not None:
            deadline()
        payload = self.provider.native_payload(request)
        submitted = self.evidence["submitted_request"]
        require(
            payload["state"] == submitted["state"]
            and list(payload["questions"]) == list(submitted["questions"]),
            "DTO changed submitted browser state/head order",
        )
        require(
            all(
                payload["questions"][key] == question
                and list(payload["questions"][key]["criteria"]) == list(question["criteria"])
                for key, question in submitted["questions"].items()
            ),
            "DTO changed actual submitted criteria/instructions/order",
        )
        _, lengths = self.backend.prepare(**payload)
        require(
            all(n <= 8192 for n in lengths.values()),
            "full browser context exceeds8192; no truncation",
        )
        self.evidence["native_payload"] = copy.deepcopy(payload)
        self.evidence["input_tokens_per_question"] = lengths
        self.calls += 1
        raw = self.backend(**payload)
        self.evidence["raw_native_response"] = copy.deepcopy(raw)
        if evidence_sink is not None:
            evidence_sink("native_raw_returned", {"native_payload": payload, "raw_response": raw})
        if deadline is not None:
            deadline()
        return raw


def profile_manifest():
    return {
        "profile": PROFILE,
        "scope": "self-authored browser-domain diagnostic; exact original predict body with fresh truthful CPU-only constructor",
        "candidate_cap": 255,
        "project_context_cap": 8192,
        "budget_seconds": 20,
        "native_max_batch_size": 4,
        "device": "cpu",
        "transport": "in_process_callback_not_HTTP",
        "HTTP_or_GPU_or_model_qualified": False,
        "all_original_heads_evaluated": True,
        "historical_batch8_reproduction_claim": False,
    }


def make_cpu_adapter(native, helper, bridge, predict):
    identity = helper.process_identity()
    native.validate_owner(identity, helper.owned_limits(identity), helper.START_ENV)
    bridge.backend.check_runtime()
    require(
        bridge.spec.device == "cpu"
        and bridge.spec.precision == "bfloat16"
        and bridge.spec.context_limit == 8192
        and bridge.spec.model_id == native.MODELS["edge"]["model_id"]
        and bridge.spec.revision == native.MODELS["edge"]["revision"],
        "truthful pinned CPU diagnostic spec required",
    )

    class CPUAdapter:
        pass

    CPUAdapter.predict = predict
    adapter = CPUAdapter()
    adapter.backend, adapter.identity, adapter.evidence_sink = bridge, "d3-edge", None
    return adapter


class RowWitness:
    """Observe actual partial batch shape; never pad or change tensor values."""

    def __init__(self, torch, rows):
        self.torch, self.rows, self.events = torch, rows, []

    def __call__(self, _module, args):
        require(args and len(args[0].shape) == 3, "actual decoder input required")
        hidden = args[0]
        require(
            hidden.shape[0] == self.rows
            and hidden.shape[2] == 1024
            and hidden.dtype == self.torch.bfloat16
            and hidden.device.type == "cpu",
            "actual unpadded-question-row CPU BF16 shape differs",
        )
        self.events.append(
            {
                "shape": list(hidden.shape),
                "dtype": str(hidden.dtype),
                "device": hidden.device.type,
                "inputs_or_outputs_modified": False,
            }
        )
        return None


def persist_local_evidence(helper, path, evidence, primary):
    try:
        try:
            helper.write_json(path, evidence)
        except (TypeError, ValueError, OverflowError):
            helper.write_json(
                path,
                {
                    "invalid_evidence_diagnostic": helper.diagnostic(evidence),
                    "valid_response_claim": False,
                },
            )
    except BaseException as secondary:
        if primary is None:
            raise
        primary.add_note("Raw toy evidence write also failed: " + repr(secondary))


def make_transport(builder, adapter, wire, order, evidence):
    def transport(url, key, original):
        require(
            url == "https://api.typesafe.ai/v1/systemone" and key == "in-process-canary",
            "original builder transport signature changed",
        )
        require("submitted_request" not in evidence, "exactly one native request per cell required")
        submitted = builder.submitted_request(original, order)
        evidence["original_constructed_request"] = copy.deepcopy(original)
        evidence["submitted_request"] = copy.deepcopy(submitted)
        legacy = adapter.predict(submitted)
        named = wire.named_browser_response(submitted, legacy)
        evidence["legacy_native_adapter_response"] = legacy
        evidence["named_inprocess_response"] = copy.deepcopy(named)
        return named

    return transport


def record_failure(helper, output, state, row, primary):
    row.update(
        completed=False,
        status="failed_no_request_completion",
        error={"type": type(primary).__name__, "message": str(primary)},
    )
    state["completed"] = False
    helper.finalize_state(output, state, primary)


def run_cell(native, helper, output, state, row, work):
    began = time.monotonic()
    primary = None
    try:
        with native.DeadlineScope(
            helper.Deadline(CALL_SECONDS, output, "canary-call-" + str(row["index"]))
        ):
            try:
                helper.write_json(output / "state.json", state)
                diagnostic = work()
                row["semantic_diagnostic"] = diagnostic
                row["status"] = "native_response_valid_pending_cleanup"
                row["durable_validation_seconds"] = helper.ensure_elapsed(
                    began, CALL_SECONDS, "canary request validation"
                )
                helper.write_json(output / "state.json", state)
                helper.ensure_elapsed(began, CALL_SECONDS, "canary request durable evidence")
            except BaseException as exc:
                primary = exc
                raise
        row["durable_validation_seconds"] = helper.ensure_elapsed(
            began, CALL_SECONDS, "canary request after deadline cleanup"
        )
        row.update(status="native_response_valid", completed=True)
    except BaseException as exc:
        if primary is None:
            primary = exc
        elif primary is not exc:
            primary.add_note("Request cleanup also failed: " + repr(exc))
        record_failure(helper, output, state, row, primary)
        raise primary


def finish_whole(native, helper, output, state, began, remaining, body):
    primary = None
    try:
        with native.DeadlineScope(helper.Deadline(remaining, output, "canary-whole")):
            try:
                body()
                state["status"] = "six_native_responses_complete_pending_cleanup_no_score"
                state["completed"] = True
            except BaseException as exc:
                primary = exc
                state.update(
                    completed=False,
                    status="failed_no_canary_completion",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
                raise
            finally:
                state["elapsed_seconds"] = time.monotonic() - began
                helper.finalize_state(output, state, primary)
        helper.ensure_elapsed(
            began, WALL_SECONDS, "whole canary after final auth/write/deadline cleanup"
        )
    except BaseException as exc:
        if primary is None:
            primary = exc
        elif primary is not exc:
            primary.add_note("Whole cleanup also failed: " + repr(exc))
        state.update(
            completed=False,
            status="failed_no_canary_completion",
            error={"type": type(primary).__name__, "message": str(primary)},
        )
        helper.finalize_state(output, state, primary)
        raise primary
    return state


def execute(path, expected, output):
    began = time.monotonic()
    require(sys.flags.no_user_site and sys.dont_write_bytecode, "fresh -B -s actor required")
    native = load_native()
    require(
        not native.MODEL_MODULES.intersection(sys.modules), "no preimported model libraries allowed"
    )
    helper = native.load_helper()
    plan, base = load_plan(native, helper, path, expected)
    identity = helper.process_identity()
    limits = helper.owned_limits(identity)
    native.validate_owner(identity, limits, helper.START_ENV)
    gate = helper.host_gate()
    helper.ensure_elapsed(began, WALL_SECONDS, "whole canary preflight")
    output = Path(output).resolve(strict=False)
    require(
        output.is_relative_to(HERE.parent.resolve()) and not output.exists(),
        "fresh resolved runtime output required",
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / ".gitignore").write_text("*\n", encoding="utf-8")
    state = {
        "profile": PROFILE,
        "plan_sha256": expected,
        "process": identity,
        "owned_limits": limits,
        "host_startup_admission": gate,
        "status": "authenticating",
        "model_loads": 0,
        "native_backend_requests_started": 0,
        "native_request_calls": 0,
        "native_request_calls_meaning": "fully returned and validated requests; failure does not imply no NN entry",
        "calls": [],
        "completed": False,
        "benchmark_scores": None,
        "browser_score": None,
        "model_qualified": False,
        "http_qualified": False,
        "whole_tree_no_escape_qualified": False,
        "semantic_diagnostics_are_self_authored_not_official_scores": True,
    }
    helper.write_json(output / "state.json", state)
    helper.ensure_elapsed(began, WALL_SECONDS, "whole canary preflight durable write")
    remaining = WALL_SECONDS - (time.monotonic() - began)
    require(remaining > 0, "preflight exhausted whole120")

    def body():
        state["before_load_owned_counters"] = native.own_counters(identity)
        runtime, torch = native.load_runtime(helper, base, output, state)
        builder = native.import_verified(
            "_canary_browser_builder", HERE / "builder.py", plan["builder_sha256"]
        )
        sys.path.insert(0, str(ROOT / "src"))
        provider = importlib.import_module("s1.evaluation.decision3")
        dto = importlib.import_module("s1.jev_decision3_native")
        require(
            Path(provider.__file__).resolve() == ROOT / "src/s1/evaluation/decision3.py"
            and Path(dto.__file__).resolve() == ROOT / "src/s1/jev_decision3_native.py",
            "actual native DTO/provider origin differs",
        )
        from importlib import metadata

        for package, version in plan["adapter_package_versions"].items():
            module = importlib.import_module(package)
            require(
                metadata.version(package) == version
                and Path(module.__file__).resolve() == helper.OVERLAY / package / "__init__.py",
                "native DTO dependency version/origin differs",
            )
        state["adapter_package_versions"] = plan["adapter_package_versions"]
        predict, predict_proof = builder.load_native_predict(
            plan["native_dto_source"], dto, PROFILE, profile_manifest
        )
        wire, wire_proof = builder.load_diagnostic_wire(
            plan["wire_source"], PROFILE, PROFILE + "-named-answers"
        )
        state["native_predict_exact_AST_proof"] = predict_proof
        state["wire_functions_exact_AST_proof"] = wire_proof
        fixtures = helper.read_json(HERE / "fixtures.json")
        for index, (key, order) in enumerate(CELLS):
            fixture = fixtures["cases"][key]
            row = {
                "index": index,
                "case": key,
                "order": order,
                "status": "running",
                "completed": False,
            }
            state["calls"].append(row)

            def work(fixture=fixture, order=order, index=index):
                row["before_owned_counters"] = native.own_counters(identity)
                evidence = {
                    "cell": [key, order],
                    "synthetic_observation_not_browser_extraction": True,
                }
                bridge = BrowserBridge(native, runtime, torch, base["model"], provider, evidence)
                adapter = make_cpu_adapter(native, helper, bridge, predict)
                transport = make_transport(builder, adapter, wire, order, evidence)
                choose, proof = builder.load_builder(
                    plan["browser_model_source"], plan["browser_questions_source"], transport
                )
                witness = RowWitness(
                    torch,
                    2
                    if any(action["kind"] == "click" for action in fixture["page"]["actions"])
                    else 1,
                )
                handle = runtime.backbone.language_model.layers[0].register_forward_pre_hook(
                    witness
                )
                primary = None
                try:
                    decision = choose(copy.deepcopy(fixture["page"]), fixtures["goal"], [])
                except BaseException as exc:
                    primary = exc
                    raise
                finally:
                    row["native_backend_request_entries"] = bridge.calls
                    state["native_backend_requests_started"] += bridge.calls
                    try:
                        handle.remove()
                    except BaseException as secondary:
                        if primary is None:
                            primary = secondary
                        else:
                            primary.add_note(
                                "Row observation cleanup also failed: " + repr(secondary)
                            )
                    evidence["actual_first_layer_events"] = witness.events
                    persist_local_evidence(
                        helper, output / ("raw-toy-cell-" + str(index) + ".json"), evidence, primary
                    )
                    if primary is not None:
                        raise primary
                require(bridge.calls == 1, "one genuine native backend request per cell required")
                require(
                    len(witness.events) == 1,
                    "one actual native partial-batch forward per request required",
                )
                state["native_request_calls"] += bridge.calls
                evidence["original_browser_mapping_decision"] = decision
                evidence["builder_AST_proof"] = proof
                helper.write_json(output / ("raw-toy-cell-" + str(index) + ".json"), evidence)
                row["question_head_count"] = len(evidence["submitted_request"]["questions"])
                row["question_head_order"] = list(evidence["submitted_request"]["questions"])
                row["actual_native_input_tokens_per_question"] = evidence[
                    "input_tokens_per_question"
                ]
                row["actual_request_sha256"] = hashlib.sha256(
                    json.dumps(
                        evidence["submitted_request"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ).hexdigest()
                row["raw_local_evidence_path"] = "raw-toy-cell-" + str(index) + ".json"
                row["native_max_batch_size"] = 4
                row["no_dummy_or_padding_question_rows"] = True
                row["actual_first_layer_events"] = witness.events
                native.validate_runtime(runtime, torch, base["model"])
                row["after_owned_counters"] = native.own_counters(identity)
                return builder.semantic_diagnostic(decision, fixture["expected"])

            run_cell(native, helper, output, state, row, work)
        require(
            state["model_loads"] == 1
            and state["native_request_calls"] == 6
            and all(row["completed"] for row in state["calls"]),
            "one constructor/six fully completed requests required",
        )
        native.authenticate(helper, base)
        native.validate_runtime(runtime, torch, base["model"])
        authenticate_sources(helper, plan)
        require(sha(path) == expected, "selected canary plan changed")
        state["final_full_model_manifest_runtime_source_identities_unchanged"] = True
        state["cuda_initialized_final"] = torch.cuda.is_initialized()
        state["after_final_auth_owned_counters"] = native.own_counters(identity)
        state["semantic_matches"] = sum(
            row["semantic_diagnostic"]["matches_self_authored_predicate"] for row in state["calls"]
        )
        state["semantic_cell_count"] = 6
        state["external_original_owner_closure_required_separately"] = True

    return finish_whole(native, helper, output, state, began, remaining, body)


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
                    "status": "dry_default_no_NN_or_job",
                    "profile": PROFILE,
                    "calls": 6,
                    "constructor_count": 1,
                    "request_seconds": 20,
                    "whole_seconds_from_execute_entry": 120,
                    "HTTP_or_official_score_claim": False,
                }
            )
        )
        return
    require(
        args.plan and args.output and args.plan_sha256, "explicit plan/SHA/fresh output required"
    )
    result = execute(args.plan, args.plan_sha256, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
