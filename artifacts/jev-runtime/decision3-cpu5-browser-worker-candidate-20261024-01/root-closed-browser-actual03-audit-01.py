"""Root read-only closed browser failure audit; no model/browser/scorer execution."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = Path(__file__).resolve().parent
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
NATIVE = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261024-05"
OUT = OWNER / "actual-03"
SPEC_SHA = "d5dfaaab9177b62be4d41ac5657668485ebf68953f57d08b26e3e6717d436d2f"
PLAN_SHA = "66a7fa4a88e18afc8baff5fc8d257880fa1d04ee967041c289f9d5ad9f731d7c"
UTIL_SHA = "af758e383c9f1108fe112098442936cd49cdba8bc91bf2e93b6a1b6024a84423"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def execute():
    utility_path = NATIVE / "root-closed-smoke-audit-01.py"
    raw = utility_path.read_bytes()
    require(digest(raw) == UTIL_SHA, "Selected pure stdlib audit utilities changed")
    spec = importlib.util.spec_from_file_location("_root_browser_closed_utilities", utility_path)
    utility = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(utility_path), "exec"), utility.__dict__)
    read, ref, authenticate = utility.read, utility.ref, utility.authenticate
    terminal_path = OWNER / "root-execution-cpu5-browser-03.json"
    terminal, receipt, closure = (
        read(terminal_path),
        read(OUT / "receipt.json"),
        read(OUT / "external-boundary-closure.json"),
    )
    require(
        terminal["exit_code"] == 1
        and receipt["completed"] is False
        and receipt["secondary_errors"] == [],
        "Preserve actual failed root/owner outcome",
    )
    require(
        closure["closure_mode"] == "authenticated_stop"
        and closure["closure_errors"] == []
        and all(
            closure[k] is True
            for k in (
                "whole_unit_exit_confirmed",
                "observed_cgroup_cohort_exit_confirmed",
                "post_stop_terminal_verified",
                "cgroup_absent",
                "sources_unchanged",
            )
        ),
        "Current original positive scoped closure required",
    )
    require(
        len(closure["two_process_scans"]) == 2
        and all(not x["matches"] and not x["unknown"] for x in closure["two_process_scans"]),
        "Two original empty cohort scans required",
    )
    require(
        all(
            closure[k] is False
            for k in (
                "no_escape_enforced",
                "whole_tree_exit_confirmed",
                "continuation_authorized",
                "model_qualified",
            )
        ),
        "Broad nonqualification flags retained",
    )
    cfg, admission = receipt["selected_config"], read(OUT / "admission.json")
    require(
        cfg == read(OUT / "sealed-config.json")
        and cfg["selection_sha256"] == SPEC_SHA
        and cfg["nonce"] == admission["nonce"] == closure["selection"]["nonce"]
        and admission["config_sha256"] == ref(OUT / "sealed-config.json")["sha256"]
        and admission["invocation_id"] == closure["invocation_id"],
        "Current config/admission/closure joins required",
    )
    spec_path, plan_path = HERE / "root-browser-command-spec-02.json", HERE / "plan-02.json"
    require(
        ref(spec_path)["sha256"] == SPEC_SHA and ref(plan_path)["sha256"] == PLAN_SHA,
        "Exact actual spec and plan required",
    )
    selected, plan, native_plan = read(spec_path), read(plan_path), read(NATIVE / "edge-plan.json")
    require(
        plan["native_plan"]
        == {
            "path": str(NATIVE / "edge-plan.json"),
            "sha256": ref(NATIVE / "edge-plan.json")["sha256"],
        }
        and plan["native_actor"]
        == {"path": str(NATIVE / "actor.py"), "sha256": ref(NATIVE / "actor.py")["sha256"]}
        and plan["owner_profile"] == cfg["profile"]
        and plan["owner_unit"] == cfg["unit"],
        "Exact selected native actor/plan and owner profile required",
    )
    base = [
        OWNER / "controller.py",
        spec_path,
        ROOT / "artifacts/jev-runtime/decision3-local-owner-terminal-20261022-01/owner_boundary.py",
        ROOT / "artifacts/jev-runtime/observer-system-services-20261022-01/comfy_observer.py",
        ROOT / "artifacts/jev-runtime/observer-system-services-20261022-01/system-services.json",
    ]
    expected = {r["path"]: r for r in [*selected["sources"], *(ref(p) for p in base)]}
    require(
        len(selected["sources"]) == 57
        and len(expected) == 62
        and expected == {r["path"]: r for r in cfg["sources"]},
        "Exact current57 plus5 source union required",
    )
    for row in (
        cfg["sources"]
        + selected["binary_stats"]
        + native_plan["model_files"]
        + native_plan["runtime_files"]
        + native_plan["owner_sources"]
        + [native_plan["manifest"]]
    ):
        authenticate(row)
    require(
        len(native_plan["model_files"]) == 20 and len(native_plan["runtime_files"]) == 28,
        "Original full model/runtime record count required",
    )
    command, started, intent = (
        read(OUT / "command-01-output.json")["result"],
        read(OUT / "command-01-started.json"),
        read(OUT / "command-01-intent.json"),
    )
    expected_argv = [v.replace("{OUTPUT}", str(OUT)) for v in selected["commands"][0]["argv"]]
    observed = started["observed_identity"]
    require(
        command["returncode"] == 1
        and command["argv"] == intent["argv"] == observed["argv"] == expected_argv
        and command["pid"] == started["pid"] == observed["process"]["pid"],
        "Exact failed actual child command required",
    )
    require(
        all(observed[k] == command["observed_identity"][k] for k in ("exe", "argv", "cwd"))
        and observed["cwd"] == str(ROOT)
        and observed["exe"] == str(Path(expected_argv[0]).resolve()),
        "Exact actual child executable and cwd required",
    )
    require(
        observed["process"]["pid"] != admission["actor"]["pid"]
        and all(
            observed["process"][k] == admission["actor"][k] == closure["actor"][k]
            for k in ("pgid", "sid", "uid", "cgroup")
        ),
        "Actual worker in current admitted cohort required",
    )
    wiring = OUT / "cpu-browser-wiring"
    worker_path = wiring / "worker-receipt.json"
    worker = read(worker_path)
    require(
        worker["success"] is False
        and worker["stage"] == "failed_or_pending_no_wiring_success"
        and worker["native_constructor_attempts"] == 1
        and worker["native_state"]["model_loads"] == 1
        and worker["final_native_authentication"] is True
        and worker["sources_unchanged"] is True
        and worker["server_thread_drained"] is True
        and worker["pending_native_or_server"] is False
        and worker["cleanup_errors"] == []
        and worker["helper_model_loaded"] is False
        and worker["helper_model_calls"] == 0,
        "One genuine native load and clean drained failure required",
    )
    require(
        all(
            worker["worker"]["process"][k] == observed["process"][k]
            for k in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")
        )
        and all(worker["worker"][k] == observed[k] for k in ("exe", "argv", "cwd"))
        and worker["worker_plan_sha256"] == PLAN_SHA
        and worker["owner"]["nonce"] == cfg["nonce"],
        "Current stable worker/source/nonce provenance required",
    )
    progress = sorted(wiring.glob("worker-receipt-*.json"))
    require(
        progress
        and read(progress[-1]) == worker
        and [p.name for p in progress]
        == [f"worker-receipt-{i:06d}.json" for i in range(1, len(progress) + 1)],
        "Exclusive complete numbered progress and final latest equality required",
    )
    thread = worker["server_thread"]
    require(
        thread["affinity"] == [15, 16, 17, 18, 19]
        and thread["intraop"] == 5
        and thread["interop"] == 1
        and thread["cuda_initialized"] is False,
        "Actual ASGI CPU5/1 identity required",
    )
    report_path = wiring / "browser/report.json"
    report = read(report_path)
    require(
        report["success"] is False
        and report["goal_satisfied"] is False
        and report["error"] is None
        and report["secondary_errors"] == []
        and report["final_status"] == "blocked"
        and report["source_unchanged"] is True
        and report["upstream_before"] == report["upstream_after"]
        and report["upstream_before"]["revision"] == "1231850a0bf1a0c0341fe408ef1668dbbfdfac46"
        and report["final_snapshot"]["text_calls"] == [],
        "Actual strict failed caller and unchanged upstream/helper0 required",
    )
    require(
        len(report["http"]) == len(report["records"]) == 4
        and all(x["kind"] == "policy" for x in report["records"]),
        "Four actual policy-only requests required",
    )
    directories = {p.name: p for p in (wiring / "server-evidence").iterdir() if p.is_dir()}
    require(
        set(directories) == {x["server_request_id"] for x in report["http"]},
        "All actual generated IDs exactly joined",
    )
    calls, operations = [], []
    for index, (http, policy) in enumerate(zip(report["http"], report["records"], strict=True)):
        directory = directories[http["server_request_id"]]
        exchange = read(directory / "exchange.json")
        request_bytes, response_bytes = (
            (directory / "request.bin").read_bytes(),
            (directory / "response.bin").read_bytes(),
        )
        request, response = json.loads(request_bytes), json.loads(response_bytes)
        require(
            exchange["status"] == http["status"] == 200
            and exchange["request_body_complete"] is True
            and exchange["body_sha256"] == digest(request_bytes) == http["request_sha256"]
            and exchange["response_sha256"] == digest(response_bytes) == http["response_sha256"]
            and exchange["observer_errors"] == []
            and exchange["primary_error"] is None
            and exchange["helper_calls"] == 0,
            "Complete actual HTTP wire bytes and valid200 evidence required",
        )
        require(
            request == policy["request"]
            and response == policy["response"]
            and response == read(directory / "candidate.json"),
            "Original actual request/response/candidate preserved",
        )
        events = [read(p) for p in sorted(directory.glob("event-*.json"))]
        names = [e["event"] for e in events]
        require(
            names
            == [
                "planned",
                "prepared",
                "actual_decoder_entered",
                "raw_returned_before_validation",
                "native_rows_completed",
                "backend_raw_returned",
                "completed",
            ],
            "Complete ordered genuine backend evidence required",
        )
        data = {e["event"]: e["data"] for e in events}
        rows, event, lengths, native_raw = (
            data["native_rows_completed"],
            data["actual_decoder_entered"],
            data["prepared"]["lengths"],
            data["raw_returned_before_validation"],
        )
        keys = list(request["questions"])
        require(
            set(lengths) == set(keys)
            and all(type(n) is int and 0 < n <= 16384 for n in lengths.values()),
            "Original complete rejection-only context lengths required",
        )
        require(
            keys == ["operation", "click_target"]
            and rows["observed_question_rows"] == 2
            and rows["observed_decoder_entries"] == 1
            and rows["dummy_questions_added"] == rows["helper_calls"] == 0
            and rows["native_raw_returned_and_validated"] is True
            and rows["question_order"] == keys,
            "Two genuine tail rows with no dummy/helper required",
        )
        require(
            event["shape"] == [2, max(lengths.values()), 1024]
            and event["dtype"] == "torch.bfloat16"
            and event["device"] == "cpu"
            and event["inputs_or_outputs_modified"] is False,
            "Actual unmodified BF16 CPU tail shape required",
        )
        require(
            native_raw == data["backend_raw_returned"] == data["completed"]["raw_response"]
            and list(native_raw["answers"]) == keys
            and native_raw["model"] == "d3-edge"
            and native_raw["usage"] == {"input_tokens": sum(lengths.values()), "output_tokens": 0},
            "Original complete native raw result/order/usage preserved",
        )
        for key in keys:
            answer = native_raw["answers"][key]
            ps = answer["probabilities"]
            require(
                list(ps) == list(request["questions"][key]["criteria"])
                and all(
                    type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1
                    for p in ps.values()
                )
                and abs(sum(ps.values()) - 1) <= 1e-6
                and answer["choice"]
                == max(ps, key=ps.__getitem__)
                == response["answers"][key]["choice"],
                "Complete native legal distribution and unchanged choice required",
            )
        operation = response["answers"]["operation"]["choice"]
        operations.append(operation)
        calls.append(
            {
                "index": index,
                "http_status": 200,
                "client_seconds": http["elapsed_ms"] / 1000,
                "actual_shape": event["shape"],
                "native_question_rows": 2,
                "dummy_questions": 0,
                "all_wire_hashes_and_native_distributions_joined": True,
            }
        )
    require(
        operations == ["CLICK", "CLICK", "CLICK", "BLOCKED"]
        and all(
            {"DONE", "BLOCKED"} <= set(x["request"]["questions"]["operation"]["criteria"])
            for x in report["records"]
        ),
        "Exact failure sequence with intact DONE/BLOCKED offerings required",
    )
    completion_in_inputs = [
        "Copper lamp armed" in x["request"]["state"]["page"]["text"] for x in report["records"]
    ]
    require(
        completion_in_inputs == [False, True, False, True],
        "Completion observation was actually delivered before repeat click and BLOCKED",
    )
    require(
        report["final_judge"]["clicks"]
        == [
            {"id": "copper", "trusted": True},
            {"id": "silver", "trusted": True},
            {"id": "copper", "trusted": True},
        ]
        and report["final_judge"]["active"] == "copper"
        and report["final_judge"]["visible"] is True,
        "Actual trusted DOM failure sequence required",
    )
    memory = receipt["state"]["last_live_properties"]
    require(
        receipt["state"]["preclose_exit_memory_authenticated"] is True
        and memory["kernel_memory_peak_bytes"]
        == int(receipt["state"]["preclose_exit_memory_raw"]["MemoryPeak"])
        and all(n == 0 for n in memory["kernel_memory_events"]["counters"].values()),
        "Authenticated own combined memory peak and zero events required",
    )
    result = {
        "schema": "root-closed-cpu5-real-browser-actual03-review-v1",
        "status": "EXECUTION_PATH_VERIFIED_STRICT_BROWSER_DIAGNOSTIC_FAIL",
        "root_exit_code": 1,
        "owner_completed": False,
        "original_positive_scoped_closure": True,
        "separate_reviewer_actual_bytes_read": False,
        "source_records_authenticated": 62,
        "model_records_authenticated": 20,
        "runtime_records_authenticated": 28,
        "model_loads": 1,
        "progress_snapshots": len(progress),
        "calls": calls,
        "operation_sequence": operations,
        "trusted_clicks": 3,
        "wrong_clicks": 1,
        "completion_text_reached_next_model_inputs": completion_in_inputs,
        "final_status": "blocked",
        "strict_success": False,
        "model_plus_browser_observed_unit_peak_bytes": memory["kernel_memory_peak_bytes"],
        "memory_events": memory["kernel_memory_events"]["counters"],
        "worker_failed_caller_did_not_auto_join_http": True,
        "all_four_http_native_joins_performed_by_this_root_closed_audit": True,
        "no_claims": [
            "official_browser_score",
            "quality_win",
            "training",
            "unseen_sites",
            "all_python_or_native_DSO_closure",
            "universal_no_escape_or_whole_tree",
        ],
        "refs": {
            "root_terminal": ref(terminal_path),
            "owner_receipt": ref(OUT / "receipt.json"),
            "closure": ref(OUT / "external-boundary-closure.json"),
            "worker": ref(worker_path),
            "caller_report": ref(report_path),
            "spec": ref(spec_path),
            "plan": ref(plan_path),
            "audit_source": ref(Path(__file__)),
        },
        "raw_states_responses_probabilities_features_or_model_bytes_included": False,
    }
    with (HERE / "root-closed-browser-actual03-review-01.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "operations": operations,
                "calls": calls,
                "peak_bytes": memory["kernel_memory_peak_bytes"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    if parser.parse_args().execute:
        execute()
    else:
        print(json.dumps({"status": "not_executed", "actual_read": False}))
