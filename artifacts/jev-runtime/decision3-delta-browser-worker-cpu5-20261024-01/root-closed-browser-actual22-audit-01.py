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
OUT = OWNER / "actual-22"
SPEC_SHA = "0ecf0256e3f9dd0c790df68326a9daf8fc41457305b3ebca3ec618ad72ee8381"
PLAN_SHA = "db62ad581f7556a70104117884603e2404477cbcfc9866e8807e8284374ea476"
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
    terminal_path = OWNER / "root-execution-delta-browser-22.json"
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
    spec_path, plan_path = HERE / "root-command-spec-02.json", HERE / "plan-01.json"
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
        len(selected["sources"]) == 58
        and len(expected) == 63
        and expected == {r["path"]: r for r in cfg["sources"]},
        "Exact current58 plus5 source union required",
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
    require(
        terminal["profile"] == "root-selected-delta-browser-cpu5-terminal-v1"
        and terminal["selection_sha256"] == SPEC_SHA
        and terminal["argv"]
        == [
            str(ROOT / ".venv/bin/python"),
            "-B",
            "-s",
            str(OWNER / "controller.py"),
            "--run",
            "--expected-entry-sha256",
            ref(OWNER / "controller.py")["sha256"],
            "--selection",
            str(spec_path),
            "--selection-sha256",
            SPEC_SHA,
            "--output",
            str(OUT),
        ],
        "exact root invocation",
    )
    wiring = OUT / "delta-browser-candidate"
    worker_path = wiring / "worker-receipt.json"
    worker = read(worker_path)
    require(
        worker["success"] is False
        and worker["stage"] == "failed_or_pending_no_wiring_success"
        and worker["native_state"]["model_loads"] == 1
        and worker["final_native_and_artifact_authentication"] is True
        and worker["sources_unchanged"] is True
        and worker["server_thread_drained"] is True
        and worker["pending_native_or_server"] is False
        and worker["helper_model_loaded"] is False
        and worker["helper_model_calls"] == 0
        and worker["publisher_native_numerics"] is False
        and worker["browser_qualified"] is False
        and worker["error"]["type"] == "AssertionError",
        "one genuine load, frozen artifact and clean drained policy failure",
    )
    require(
        len(worker["cleanup_errors"]) == 1
        and worker["cleanup_errors"][0]["stage"] == "unchanged_browser_report_and_http_closure"
        and worker["cleanup_errors"][0]["message"]
        == "unchanged strict caller/helper0 did not succeed",
        "expected strict policy failure only; no hidden cleanup/source/runtime error",
    )
    require(
        all(
            worker["worker"]["process"][k] == observed["process"][k]
            for k in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")
        )
        and all(worker["worker"][k] == observed[k] for k in ("exe", "argv", "cwd"))
        and worker["plan_sha256"] == PLAN_SHA,
        "same actual worker/plan",
    )
    progress = sorted(wiring.glob("worker-receipt-*.json"))
    require(
        progress
        and read(progress[-1]) == worker
        and [p.name for p in progress]
        == [f"worker-receipt-{i:06d}.json" for i in range(1, len(progress) + 1)],
        "exclusive numbered progress and latest equality",
    )
    for key in ("candidate_capability", "numeric_acceptance"):
        authenticate(plan[key])
    cap = read(Path(plan["candidate_capability"]["path"]))
    numeric = read(Path(plan["numeric_acceptance"]["path"]))
    training = read(Path(cap["root_acceptance_ref"]["path"]))
    require(
        cap["status"] == "ROOT_ACCEPTED_TRAIN_ONLY_READOUT"
        and cap["frozen_before_dev_sealed_benchmark_or_browser"] is True
        and cap["W0_merge_allowed"] is False
        and cap["alias"] == plan["model_alias"] == worker["candidate_alias"]
        and numeric["status"] == "accepted"
        and numeric["schema"] == "root-accepted-frozen-delta-cpu5-numeric-v1"
        and numeric["root_exit"] == 0
        and numeric["owner_completed"] is True
        and numeric["current_original_scoped_closure_positive"] is True
        and numeric["candidate_capability"] == plan["candidate_capability"]
        and numeric["delta_ref"] == cap["delta_ref"]
        and all(
            numeric["backend_source"][k] == plan["delta_backend"][k] for k in ("path", "sha256")
        ),
        "frozen candidate actual numeric qualification joins",
    )
    authenticate(cap["delta_ref"])
    authenticate(cap["root_acceptance_ref"])
    require(
        worker["artifact_proof"]
        == {
            "baseline_fp32_bytes_sha256": cap["baseline_fp32_bytes_sha256"],
            "delta_fp32_bytes_sha256": training["delta_statistics"]["FP32_raw_bytes_sha256"],
            "numeric_acceptance": plan["numeric_acceptance"],
        },
        "actual prebrowser original W0/true delta byte witnesses",
    )
    ns = worker["native_state"]
    require(
        ns["native_provenance"]["model_sha256"] == cap["model"]["model_sha256"]
        and ns["native_provenance"]["backbone_dtype"] == "bfloat16"
        and ns["native_provenance"]["readout_dtype"] == "float32"
        and ns["native_runtime_info"]["device"] == "cpu"
        and ns["actual_environment"]["OMP_NUM_THREADS"] == "5"
        and ns["actual_environment"]["CUDA_VISIBLE_DEVICES"] == "",
        "honest original CPU5/BF16/FP32 runtime",
    )
    report_path = wiring / "browser/report.json"
    report = read(report_path)
    require(
        ref(report_path)["sha256"] == worker["browser_report_sha256"]
        and report["success"] is False
        and report["goal_satisfied"] is False
        and report["error"] is None
        and report["secondary_errors"] == []
        and report["final_status"] == "blocked"
        and report["source_unchanged"] is True
        and report["upstream_before"] == report["upstream_after"]
        and report["upstream_before"]["revision"] == "1231850a0bf1a0c0341fe408ef1668dbbfdfac46"
        and report["final_snapshot"]["text_calls"] == []
        and report["requested_model"] == cap["alias"],
        "unchanged strict failed caller/helper0 and explicit candidate",
    )
    require(
        len(report["http"]) == len(report["records"]) == 4
        and all(x["kind"] == "policy" for x in report["records"]),
        "four genuine policy requests",
    )
    directories = {p.name: p for p in (wiring / "server-evidence").iterdir() if p.is_dir()}
    require(
        set(directories) == {x["server_request_id"] for x in report["http"]},
        "every server/client ID joined",
    )
    calls = []
    operations = []
    for index, (http, policy) in enumerate(zip(report["http"], report["records"], strict=True), 1):
        directory = directories[http["server_request_id"]]
        exchange = read(directory / "exchange.json")
        request_bytes = (directory / "request.bin").read_bytes()
        response_bytes = (directory / "response.bin").read_bytes()
        request = json.loads(request_bytes)
        response = json.loads(response_bytes)
        require(
            exchange["status"] == http["status"] == 200
            and exchange["request_body_complete"] is True
            and exchange["body_sha256"] == digest(request_bytes) == http["request_sha256"]
            and exchange["response_sha256"] == digest(response_bytes) == http["response_sha256"]
            and exchange["observer_errors"] == []
            and exchange["primary_error"] is None
            and exchange["helper_calls"] == 0
            and request == policy["request"]
            and response == policy["response"] == read(directory / "candidate.json"),
            "all actual wire bytes/client/server/candidate joined",
        )
        events = [read(p) for p in sorted(directory.glob("event-*.json"))]
        require(
            [e["event"] for e in events]
            == [
                "planned",
                "prepared",
                "chunk_returned",
                "raw_returned_before_validation",
                "completed",
                "candidate_raw_returned_before_alias_validation",
                "backend_raw_returned",
                "completed",
            ],
            "full candidate event sequence, preserving two distinct completed events",
        )
        (
            planned,
            prepared,
            chunk,
            prealias,
            backend_complete,
            candidate_raw,
            backend_raw,
            adapter_complete,
        ) = [e["data"] for e in events]
        raw = candidate_raw
        keys = list(request["questions"])
        lengths = prepared["lengths"]
        counts = [len(q["criteria"]) for q in request["questions"].values()]
        require(
            keys == ["operation", "click_target"]
            and list(lengths) == keys
            and all(type(n) is int and 0 < n <= 16384 for n in lengths.values())
            and prepared["original_question_order"] == keys
            and chunk["head_order"] == keys
            and chunk["ordered_criteria_ids"]
            == [list(q["criteria"]) for q in request["questions"].values()]
            and chunk["legal_counts"] == counts
            and chunk["rows"] == 2
            and chunk["padded_width"] == max(lengths.values())
            and chunk["first_layer_events"]
            == chunk["final_norm_events"]
            == [[2, max(lengths.values()), 1024]],
            "one same-forward real two-row tail without extra/dummy heads",
        )
        require(
            backend_complete == {"profile": cap["profile"], "raw_response": raw}
            and backend_raw == raw
            and prealias == {**raw, "model": "d3-edge"}
            and raw["model"] == response["model"] == cap["alias"]
            and list(raw["answers"]) == keys
            and raw["usage"]
            == response["usage"]
            == {"input_tokens": sum(lengths.values()), "output_tokens": 0},
            "adapted raw preserved before explicit candidate alias; not baseline control",
        )
        audit = response["decision3_native"]
        require(
            adapter_complete == {k: v for k, v in audit.items() if k != "elapsed_ms"}
            and audit["status"] == "completed"
            and audit["publisher_native_numerics"] is False
            and audit["candidate_alias"] == cap["alias"]
            and audit["original_request"] == request
            and audit["original_request_sha256"] == digest(request_bytes)
            and audit["native_question_rows"] == 2
            and audit["backend_predict_calls"] == 1
            and audit["raw_response"] == raw
            and audit["usage"] == raw["usage"],
            "original complete candidate DTO/native audit preserved",
        )
        ident = response["decision3_cpu_http"]["cpu_identity"]
        require(
            ident["candidate_capability"] == plan["candidate_capability"]
            and ident["numeric_acceptance"] == plan["numeric_acceptance"]
            and ident["publisher_native_numerics"] is False
            and ident["readout_adaptation_applied"] is True
            and ident["wire_copies_candidate_probabilities_without_change"] is True
            and ident["pid"] == observed["process"]["pid"]
            and ident["start_ticks"] == observed["process"]["start_ticks"]
            and ident["nonce"] == cfg["nonce"]
            and ident["invocation_id"] == admission["invocation_id"],
            "candidate provenance bound to actual current owned invocation",
        )
        max_error = 0.0
        for row, key in enumerate(keys):
            answer = raw["answers"][key]
            ps = answer["probabilities"]
            values = list(ps.values())
            count = counts[row]
            require(
                list(ps) == list(request["questions"][key]["criteria"])
                and len(values) == count
                and all(
                    type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in values
                )
                and abs(sum(values) - 1) <= 1e-6
                and answer["choice"]
                == max(ps, key=ps.__getitem__)
                == response["answers"][key]["choice"]
                and response["answers"][key]["probabilities"] == ps,
                "every ordered legal probability and chosen answer unchanged through HTTP",
            )
            entropy = -sum(p * math.log(p) for p in values if p > 0)
            conf = 1.0 if count < 2 else max(0.0, min(1.0, 1 - entropy / math.log(count)))
            require(abs(answer["confidence"] - conf) <= 1e-6, "raw original entropy confidence")
            projection = chunk["projection"]
            base_values = projection["base_legal_logits"][row]
            delta_values = projection["delta_legal_logits"][row]
            logits = projection["adapted_legal_logits"][row]
            require(
                len(base_values) == len(delta_values) == len(logits) == count
                and all(math.isfinite(x) for x in base_values + delta_values + logits)
                and all(
                    abs(a + b - c) <= 1e-6 + 1e-5 * abs(c)
                    for a, b, c in zip(base_values, delta_values, logits)
                ),
                "actual separate delta addition, no merged W0",
            )
            e = [math.exp(x - max(logits)) for x in logits]
            total = math.fsum(e)
            p64 = [x / total for x in e]
            require(
                all(abs(a - b) <= 1e-6 + 1e-5 * abs(b) for a, b in zip(p64, values))
                and max(range(count), key=lambda k: p64[k])
                == max(range(count), key=lambda k: values[k]),
                "root stdlib stable softmax of actual retained adapted logits",
            )
            max_error = max(max_error, max(abs(a - b) for a, b in zip(p64, values)))
        operations.append(response["answers"]["operation"]["choice"])
        calls.append(
            {
                "index": index,
                "http_status": 200,
                "client_seconds": http["elapsed_ms"] / 1000,
                "actual_shape": [2, max(lengths.values()), 1024],
                "native_question_rows": 2,
                "dummy_questions": 0,
                "all_wire_native_logits_and_distributions_joined": True,
                "root_FP64_softmax_max_probability_error": max_error,
            }
        )
    require(
        operations == ["CLICK", "CLICK", "CLICK", "BLOCKED"]
        and all(
            {"DONE", "BLOCKED"} <= set(x["request"]["questions"]["operation"]["criteria"])
            for x in report["records"]
        ),
        "exact failed policy sequence with intact completion choices",
    )
    completion = [
        "Copper lamp armed" in x["request"]["state"]["page"]["text"] for x in report["records"]
    ]
    require(
        completion == [False, True, False, True]
        and report["final_judge"]["clicks"]
        == [
            {"id": "copper", "trusted": True},
            {"id": "silver", "trusted": True},
            {"id": "copper", "trusted": True},
        ]
        and report["final_judge"]["active"] == "copper"
        and report["final_judge"]["visible"] is True,
        "trusted DOM sequence; completion text actually reached next inputs",
    )
    require(
        worker["http_evidence_closure"]["status"] == "all_actual_client_server_ids_and_bytes_joined"
        and len(worker["http_evidence_closure"]["requests"]) == 4,
        "original worker all HTTP closure retained",
    )
    memory = receipt["state"]["last_live_properties"]
    require(
        receipt["state"]["preclose_exit_memory_authenticated"] is True
        and memory["kernel_memory_peak_bytes"]
        == int(receipt["state"]["preclose_exit_memory_raw"]["MemoryPeak"])
        and all(n == 0 for n in memory["kernel_memory_events"]["counters"].values()),
        "owned combined memory peak/events",
    )
    require(
        0
        < worker["elapsed_before_final_write_seconds"]
        < command["elapsed_seconds"]
        < terminal["wall_seconds"]
        < 120,
        "whole120 worker/child/root bounds",
    )
    result = {
        "schema": "root-closed-frozen-delta-cpu5-browser-actual22-v1",
        "status": "CANDIDATE_EXECUTION_PATH_VERIFIED_STRICT_BROWSER_DIAGNOSTIC_FAIL",
        "authority": "root",
        "root_exit_code": 1,
        "owner_completed": False,
        "original_positive_scoped_closure": True,
        "independent_actual_review_claimed": False,
        "separate_reviewer_actual_bytes_read": False,
        "source_peer_review_scope": "source only",
        "source_records_authenticated": 63,
        "model_records_authenticated": 20,
        "runtime_records_authenticated": 28,
        "runtime_SHA_records": 22,
        "runtime_stat_only_records": 6,
        "model_loads": 1,
        "candidate_alias": cap["alias"],
        "candidate_capability": plan["candidate_capability"],
        "numeric_acceptance": plan["numeric_acceptance"],
        "calls": calls,
        "operation_sequence": operations,
        "trusted_clicks": 3,
        "wrong_clicks": 1,
        "completion_text_reached_next_model_inputs": completion,
        "final_status": "blocked",
        "strict_success": False,
        "browser_score": None,
        "benchmark_scores": None,
        "training_eligible": False,
        "exposed_diagnostic_may_not_enter_training": True,
        "same_sequence_as_prior_original_d3_diagnostic": True,
        "model_plus_browser_observed_unit_peak_bytes": memory["kernel_memory_peak_bytes"],
        "memory_events": memory["kernel_memory_events"]["counters"],
        "worker_seconds_before_final_write": worker["elapsed_before_final_write_seconds"],
        "root_seconds": terminal["wall_seconds"],
        "all_four_HTTP_candidate_joins_performed_by_root": True,
        "independent_extra_encoder_or_FP32_execution": False,
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
        "scope": "Real HTTP/choose/Agent/Chromium wiring and strict exposed two-control task failure. Full same candidate artifact and original byte/source/model identities checked; root joins all four adapted logits, legal native distributions and unchanged wire values. Independent FP64 witness here is softmax of actual retained logits, not new h/matmul replay. Not an official/unseen/browser-quality or benchmark win.",
    }
    with (HERE / "root-closed-browser-actual22-review-01.json").open("x") as stream:
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
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    if parser.parse_args().execute:
        execute()
