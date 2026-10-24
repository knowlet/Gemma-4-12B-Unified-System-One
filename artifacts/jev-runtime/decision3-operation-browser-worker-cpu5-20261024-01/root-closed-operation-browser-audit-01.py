"""Held root audit of the exposed operation-candidate browser diagnostic.

Source-only and unexecuted. Root must select exact terminal/spec/plan refs and
source counts in a released copy. No model/browser/scorer execution occurs in
the auditor; strict result and number of HTTP calls are derived, never fixed.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
NATIVE = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261024-05"
SOURCE_ONLY = False
SELECTED_REFS = {
    "terminal": {
        "path": "/home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/llama-cpp-owner-20261024-09/root-execution-operation-browser-30.json",
        "sha256": "408c0078bfc2e39ecc90e6c0b86f9a5f4c3611476daa04c8f561f88447a94eb8",
        "size_bytes": 1745,
    },
    "spec": {
        "path": "/home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/decision3-operation-browser-worker-cpu5-20261024-01/root-command-spec-01.json",
        "sha256": "5a3c22e73b20dcaaef3add41a99ae95bc58d835be8d5462389d0b4b97d041d60",
        "size_bytes": 13306,
    },
    "plan": {
        "path": "/home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/decision3-operation-browser-worker-cpu5-20261024-01/plan-01.json",
        "sha256": "65aaa9172ecce21bb892c3670ed217d9f95392882d31328d559df933db002930",
        "size_bytes": 13098,
    },
}
SPEC_SOURCE_COUNT = 58
SOURCE_UNION_COUNT = 63
UTIL_SHA = "af758e383c9f1108fe112098442936cd49cdba8bc91bf2e93b6a1b6024a84423"
WORKER_PROFILE = "decision3-dsl01-operation-cpu5-owned-two-control-browser-worker-v1"
ADAPTER_PROFILE = "jev-dsl01-operation-candidate-cpu5-browser-v1"
CANDIDATE_PROFILE = "decision3-dsl01-operation-readout-cpu5-4g-candidate-v1"
ALIAS = "d3-edge-dsl01-operation-delta-cpu5-candidate"
OWNER_PROFILE = "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"
UNIT = "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service"
DEV_FLAGS = {
    "dev_used_for_model_selection": True,
    "optimizer_input_dev": False,
    "optimizer_input_sealed": False,
    "candidate_selection_before_dev_or_sealed": False,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def release_gate():
    require(SOURCE_ONLY is False, "SOURCE_ONLY: operation browser audit is held")
    require(
        all(
            type(row) is dict
            and type(row.get("path")) is str
            and re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is not None
            and type(row.get("size_bytes")) is int
            and 0 < row["size_bytes"] <= 1048576
            for row in SELECTED_REFS.values()
        )
        and type(SPEC_SOURCE_COUNT) is int
        and SPEC_SOURCE_COUNT > 0
        and type(SOURCE_UNION_COUNT) is int
        and SOURCE_UNION_COUNT == SPEC_SOURCE_COUNT + 5,
        "new root-selected references/source counts remain pending",
    )


def selected(name, utility, supplied_path=None, supplied_sha=None):
    row = SELECTED_REFS[name]
    path = Path(row["path"])
    path = path if path.is_absolute() else ROOT / path
    require(
        path.is_relative_to(ROOT)
        and ".." not in path.parts
        and not any(p.is_symlink() for p in (path, *path.parents))
        and (supplied_path is None or supplied_path.absolute() == path)
        and (supplied_sha is None or supplied_sha == row["sha256"]),
        "exact root-selected reference required",
    )
    utility.authenticate(row)
    return path, utility.read(path), row["sha256"]


def strict_result(report):
    """Recompute the unchanged caller's exact fresh-DOM success predicate."""
    initial, final = report["initial_judge"], report["final_judge"]
    require(
        initial["active"] == "none"
        and initial["clicks"] == []
        and len(initial["buttons"]) == 2
        and {b["id"] for b in initial["buttons"]} == {"copper", "silver"}
        and all(b["visible"] is True and b["disabled"] is False for b in initial["buttons"]),
        "both original controls initially visible/enabled without a click",
    )
    goal = (
        final["url"] == initial["url"]
        and final["active"] == "copper"
        and final["text"] == "Copper lamp armed"
        and final["visible"] is True
        and final["last_trusted"] is True
        and final["clicks"] == [{"id": "copper", "trusted": True}]
    )
    success = bool(
        goal
        and report["final_status"] == "done"
        and report["error"] is None
        and report["secondary_errors"] == []
        and report["source_unchanged"] is True
    )
    require(
        report["goal_satisfied"] is goal
        and report["success"] is success
        and report["final_status"] == report["final_snapshot"]["status"],
        "reported goal/strict success must match the unchanged exact judge",
    )
    return goal, success


def projection_witness(head, count, base, delta, adapted, probabilities):
    require(
        type(count) is int
        and 1 <= count <= 255
        and len(base) == len(delta) == len(adapted) == len(probabilities) == count
        and all(
            type(v) in (int, float) and math.isfinite(v)
            for v in (*base, *delta, *adapted, *probabilities)
        ),
        "complete finite legal projection values required",
    )
    if head == "operation":
        require(
            all(abs(a + b - c) <= 1e-6 + 1e-5 * abs(c) for a, b, c in zip(base, delta, adapted)),
            "actual operation separate delta addition differs",
        )
        independent = [a + b for a, b in zip(base, delta)]
    else:
        require(
            all(v == 0 for v in delta)
            and all(struct.pack("<f", a) == struct.pack("<f", b) for a, b in zip(base, adapted)),
            "target delta or original FP32 logit bits changed",
        )
        independent = base
    exponentials = [math.exp(x - max(independent)) for x in independent]
    total = math.fsum(exponentials)
    reference = [x / total for x in exponentials]
    require(
        all(abs(a - b) <= 1e-6 + 1e-5 * abs(b) for a, b in zip(reference, probabilities))
        and max(range(count), key=reference.__getitem__)
        == max(range(count), key=probabilities.__getitem__),
        "FP64 operation-only addition/target-base softmax or ordered argmax differs",
    )
    return max(abs(a - b) for a, b in zip(reference, probabilities))


def execute(
    *, terminal=None, terminal_sha=None, spec=None, spec_sha=None, plan=None, plan_sha=None
):
    release_gate()
    utility_path = NATIVE / "root-closed-smoke-audit-01.py"
    raw = utility_path.read_bytes()
    require(digest(raw) == UTIL_SHA, "Selected pure stdlib audit utilities changed")
    utility_spec = importlib.util.spec_from_file_location(
        "_root_browser_closed_utilities", utility_path
    )
    utility = importlib.util.module_from_spec(utility_spec)
    exec(compile(raw, str(utility_path), "exec"), utility.__dict__)
    read, ref, authenticate = utility.read, utility.ref, utility.authenticate
    terminal_path, terminal, _ = selected("terminal", utility, terminal, terminal_sha)
    spec_path, selected_spec, SPEC_SHA = selected("spec", utility, spec, spec_sha)
    plan_path, plan, PLAN_SHA = selected("plan", utility, plan, plan_sha)
    require(
        type(terminal.get("argv")) is list and terminal["argv"].count("--output") == 1,
        "one exact owner output argument",
    )
    OUT = Path(terminal["argv"][terminal["argv"].index("--output") + 1])
    require(
        OUT.parent == OWNER and re.fullmatch(r"actual-[0-9]{2}", OUT.name) is not None,
        "selected owner branch",
    )
    terminal, receipt, closure = (
        read(terminal_path),
        read(OUT / "receipt.json"),
        read(OUT / "external-boundary-closure.json"),
    )
    require(
        type(terminal["exit_code"]) is int
        and terminal["exit_code"] in (0, 1)
        and type(receipt["completed"]) is bool
        and receipt["secondary_errors"] == [],
        "Preserve actual completed root/owner outcome; no assumed policy result",
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
        and cfg["profile"] == OWNER_PROFILE
        and cfg["unit"] == UNIT
        and cfg["output"] == str(OUT)
        and cfg["selection_sha256"] == SPEC_SHA
        and cfg["nonce"] == admission["nonce"] == closure["selection"]["nonce"]
        and admission["config_sha256"] == ref(OUT / "sealed-config.json")["sha256"]
        and admission["invocation_id"] == closure["invocation_id"],
        "Current config/admission/closure joins required",
    )
    require(
        ref(spec_path)["sha256"] == SPEC_SHA and ref(plan_path)["sha256"] == PLAN_SHA,
        "Exact actual spec and plan required",
    )
    native_plan = read(NATIVE / "edge-plan.json")
    selected_spec = read(spec_path)
    require(
        plan["schema"] == "owned-decision3-operation-cpu5-browser-worker-plan-v1"
        and plan["profile"] == WORKER_PROFILE
        and plan["only_head"] == "operation"
        and plan["diagnostic_only"] is True
        and plan["model_alias"] == ALIAS
        and plan["max_ticks"] == 16
        and plan["http_caller_timeout_seconds"] == 25
        and plan["http_cooperative_budget_seconds"] == 20
        and plan["whole_budget_seconds"] == 120
        and plan["helper_model"] is None
        and plan["owner_output"] == str(OUT),
        "new operation-only exposed diagnostic with unchanged budgets/helper0",
    )
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
    expected = {r["path"]: r for r in [*selected_spec["sources"], *(ref(p) for p in base)]}
    require(
        len(selected_spec["sources"]) == SPEC_SOURCE_COUNT
        and len(expected) == SOURCE_UNION_COUNT
        and expected == {r["path"]: r for r in cfg["sources"]},
        "Exact root-selected source union plus5 required",
    )
    for row in (
        cfg["sources"]
        + selected_spec["binary_stats"]
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
    expected_argv = [v.replace("{OUTPUT}", str(OUT)) for v in selected_spec["commands"][0]["argv"]]
    observed = started["observed_identity"]
    require(
        type(command["returncode"]) is int
        and command["returncode"] in (0, 1)
        and command["argv"] == intent["argv"] == observed["argv"] == expected_argv
        and command["pid"] == started["pid"] == observed["process"]["pid"],
        "Exact actual selected child command required",
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
        terminal["profile"] == "root-selected-operation-browser-cpu5-terminal-v1"
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
    wiring = OUT / "operation-browser-candidate"
    worker_path = wiring / "worker-receipt.json"
    worker = read(worker_path)
    require(
        worker["profile"] == WORKER_PROFILE
        and worker["candidate_alias"] == ALIAS
        and worker["only_adapted_head_id"] == "operation"
        and worker["diagnostic_only"] is True
        and all(worker[k] is value for k, value in DEV_FLAGS.items())
        and worker["new_unseen_evaluation_claim"] is False
        and worker["official_score_claim"] is False
        and worker["native_state"]["model_loads"] == 1
        and worker["final_native_and_artifact_authentication"] is True
        and worker["sources_unchanged"] is True
        and worker["server_thread_drained"] is True
        and worker["pending_native_or_server"] is False
        and worker["helper_model_loaded"] is False
        and worker["helper_model_calls"] == 0
        and worker["publisher_native_numerics"] is False
        and worker["browser_qualified"] is False,
        "one genuine load, accepted operation artifact, helper0 and clean drained runtime",
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
        cap["schema"] == "root-accepted-dsl01-operation-train-delta-v1"
        and cap["status"] == "ROOT_ACCEPTED_OPERATION_TRAIN_ONLY_READOUT"
        and cap["split"] == "train"
        and cap["training_eligible"] is True
        and cap["nontraining"] is False
        and cap["only_head"] == "operation"
        and cap["profile"] == CANDIDATE_PROFILE
        and all(cap[k] is value for k, value in DEV_FLAGS.items())
        and cap["W0_merge_allowed"] is False
        and cap["alias"] == plan["model_alias"] == worker["candidate_alias"] == ALIAS
        and cap["model"] == training["model"] == native_plan["model"]
        and training["schema"] == "root-accepted-dsl01-operation-trained-readout-v1"
        and training["status"] == "accepted"
        and training["authority"] == "root"
        and training["root_exit"] == 0
        and training["owner_completed"] is True
        and training["current_original_scoped_closure_positive"] is True
        and cap["delta_ref"] == training["delta_ref"]
        and cap["fit_capability"] == training["capability"]
        and cap["baseline_fp32_bytes_sha256"] == training["baseline_ref"]["FP32_bytes_sha256"]
        and numeric["status"] == "accepted"
        and numeric["schema"] == "root-accepted-operation-cpu5-numeric-v1"
        and numeric["authority"] == "root"
        and numeric["root_exit"] == 0
        and numeric["owner_completed"] is True
        and numeric["current_original_scoped_closure_positive"] is True
        and numeric["candidate_capability"] == plan["candidate_capability"]
        and numeric["candidate_profile"] == CANDIDATE_PROFILE
        and numeric["candidate_alias"] == ALIAS
        and numeric["only_adapted_head_id"] == "operation"
        and all(numeric[k] is value for k, value in DEV_FLAGS.items())
        and numeric["training_acceptance"] == cap["root_acceptance_ref"]
        and numeric["delta_ref"] == cap["delta_ref"]
        and numeric["baseline_fp32_bytes_sha256"] == cap["baseline_fp32_bytes_sha256"]
        and numeric["model"] == cap["model"]
        and numeric["browser_qualified"] is False
        and numeric["quality_or_win_claim"] is False
        and all(
            numeric["backend_source"][k] == plan["delta_backend"][k] for k in ("path", "sha256")
        ),
        "new operation candidate true numeric/training acceptance joins without shared-head qualification",
    )
    authenticate(cap["delta_ref"])
    authenticate(cap["root_acceptance_ref"])
    authenticate(cap["fit_capability"])
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
        and report["error"] is None
        and report["secondary_errors"] == []
        and report["source_unchanged"] is True
        and report["upstream_before"] == report["upstream_after"]
        and report["upstream_before"]["revision"] == "1231850a0bf1a0c0341fe408ef1668dbbfdfac46"
        and report["final_snapshot"]["text_calls"] == []
        and report["requested_model"] == cap["alias"]
        and report["max_ticks"] == 16
        and report["action_profile"] == "standard"
        and report["observation_profile"] == "native-1231850",
        "completed unchanged caller/helper0 and explicit candidate; no technical evidence gaps accepted",
    )
    goal_satisfied, strict_success = strict_result(report)
    require(
        worker["success"] is strict_success
        and worker["stage"]
        == (
            "complete_pending_original_external_closure"
            if strict_success
            else "failed_or_pending_no_wiring_success"
        )
        and receipt["completed"] is strict_success
        and terminal["exit_code"] == command["returncode"] == (0 if strict_success else 1),
        "strict browser result must agree with original worker/child/root outcome",
    )
    if strict_success:
        require(
            worker["error"] is None
            and worker["cleanup_errors"] == []
            and receipt["primary_error"] is None,
            "successful path has no concealed primary/cleanup error",
        )
    else:
        require(
            worker["error"]["type"] == "AssertionError"
            and len(worker["cleanup_errors"]) == 1
            and worker["cleanup_errors"][0]["stage"] == "unchanged_browser_report_and_http_closure"
            and worker["cleanup_errors"][0]["message"]
            == "unchanged strict caller/helper0 did not succeed"
            and receipt["primary_error"] is not None,
            "strict policy failure only; technical/cleanup/source failures are not qualified",
        )
    require(
        ref(ROOT / "tests/test_jev_native_http_browser.py")["sha256"]
        == report["test_source_sha256"]
        == plan["caller"]["sha256"]
        and ref(ROOT / "scripts/jev_browser_benchmark.py")["sha256"]
        == report["client_source_sha256"]
        and ref(wiring / "browser/fixture.html")["sha256"] == report["fixture_sha256"],
        "actual unchanged caller/harness/fixture source hashes retained",
    )
    require(
        len(report["http"]) == len(report["records"]) > 0
        and all(x["kind"] == "policy" for x in report["records"]),
        "every actual policy request has complete HTTP evidence; no fixed count",
    )
    require(
        all(
            type(x.get("server_request_id")) is str
            and re.fullmatch(r"[0-9a-f]{32}", x["server_request_id"]) is not None
            for x in report["http"]
        )
        and len({x["server_request_id"] for x in report["http"]}) == len(report["http"]),
        "unique generated server IDs required for all actual calls",
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
            and response == policy["response"] == read(directory / "candidate.json")
            and request_bytes == http["request_utf8"].encode("utf-8")
            and response_bytes == http["response_utf8"].encode("utf-8")
            and exchange["request_id"] == http["server_request_id"]
            and exchange["candidate_constructed"] is True
            and exchange["candidate_retained"] is True
            and exchange["network_delivery_guaranteed"] is False
            and 0 <= exchange["elapsed_ms_sampled_before_terminal_write"] < 20000
            and 0 < http["elapsed_ms"] < 25000,
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
            len(keys) == 2
            and set(keys) == {"operation", "click_target"}
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
            planned["original_request"] == request
            and planned["candidate_alias"] == ALIAS
            and planned["candidate_profile"] == ADAPTER_PROFILE
            and all(event["profile"] == ADAPTER_PROFILE for event in events),
            "full original planned request and exact operation adapter event identity",
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
            and audit["usage"] == raw["usage"]
            and audit["native_events"]
            == [{"event": e["event"], "evidence": e["data"]} for e in events[1:6]]
            and 0 < audit["elapsed_ms"] < 20000,
            "original complete candidate DTO/native audit preserved",
        )
        ident = response["decision3_cpu_http"]["cpu_identity"]
        require(
            ident["candidate_capability"] == plan["candidate_capability"]
            and ident["numeric_acceptance"] == plan["numeric_acceptance"]
            and ident["publisher_native_numerics"] is False
            and ident["profile"] == WORKER_PROFILE
            and ident["candidate_alias"] == ALIAS
            and ident["only_adapted_head_id"] == "operation"
            and ident["diagnostic_only"] is True
            and all(ident[k] is value for k, value in DEV_FLAGS.items())
            and ident["new_unseen_evaluation_claim"] is False
            and ident["readout_adaptation_applied"] is True
            and ident["wire_copies_candidate_probabilities_without_change"] is True
            and ident["pid"] == observed["process"]["pid"]
            and ident["start_ticks"] == observed["process"]["start_ticks"]
            and ident["nonce"] == cfg["nonce"]
            and ident["invocation_id"] == admission["invocation_id"],
            "candidate provenance bound to actual current owned invocation",
        )
        max_error = 0.0
        logged_argmax_counterfactuals = []
        projection = chunk["projection"]
        require(
            projection["ordered_head_ids"] == keys
            and projection["adapted_head_ids"] == ["operation"]
            and projection["preserved_head_ids"] == [key for key in keys if key != "operation"]
            and projection["target_logits_bitwise_unchanged"] is True,
            "actual operation row is selected by ID and target logits retain native bits",
        )
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
            base_values = projection["base_legal_logits"][row]
            delta_values = projection["delta_legal_logits"][row]
            logits = projection["adapted_legal_logits"][row]
            # These are retained FP32 projection values, not new h/W0/delta
            # array reads or a new encoder/matmul. Recompute only the actual
            # operation's addition in FP64; every target uses base directly.
            max_error = max(
                max_error, projection_witness(key, count, base_values, delta_values, logits, values)
            )
            criteria = list(request["questions"][key]["criteria"])
            base_choice = criteria[max(range(count), key=base_values.__getitem__)]
            logged_argmax_counterfactuals.append(
                {
                    "head": key,
                    "same_forward_base_logged_logit_argmax": base_choice,
                    "actual_adapted_argmax": answer["choice"],
                    "argmax_changed": base_choice != answer["choice"],
                    "scope": "retained same-forward logits, no original-policy rerun or extra encoder",
                }
            )
        operations.append(response["answers"]["operation"]["choice"])
        calls.append(
            {
                "index": index,
                "request_id": http["server_request_id"],
                "request_sha256": http["request_sha256"],
                "response_sha256": http["response_sha256"],
                "http_status": http["status"],
                "client_seconds": http["elapsed_ms"] / 1000,
                "actual_shape": [2, max(lengths.values()), 1024],
                "native_question_rows": 2,
                "dummy_questions": 0,
                "all_wire_native_logits_and_distributions_joined": True,
                "root_FP64_softmax_max_probability_error": max_error,
                "logged_logit_argmax_counterfactuals": logged_argmax_counterfactuals,
                "ordered_head_ids": keys,
                "legal_counts": counts,
                "only_adapted_head_id": "operation",
                "target_legal_logits_bitwise_unchanged": True,
                "gateway_terminal_sample_ms": exchange["elapsed_ms_sampled_before_terminal_write"],
                "timing_scope": "terminal sample is before writes; source final20 gate charges publication before ASGI start; client measures complete25s return",
            }
        )
    require(
        operations
        and all(
            {"DONE", "BLOCKED"} <= set(x["request"]["questions"]["operation"]["criteria"])
            and set(x["request"]["questions"]["operation"]["criteria"])
            <= {
                "CLICK",
                "TYPE_TEXT",
                "SELECT",
                "SCROLL_UP",
                "SCROLL_DOWN",
                "WAIT",
                "DONE",
                "BLOCKED",
            }
            for x in report["records"]
        )
        and (
            report["final_status"] not in ("done", "blocked")
            or operations[-1] == report["final_status"].upper()
        ),
        "actual policy sequence retains legal DONE/BLOCKED choices without repair",
    )
    completion = [
        "Copper lamp armed" in x["request"]["state"]["page"]["text"] for x in report["records"]
    ]
    require(
        worker["http_evidence_closure"]["status"] == "all_actual_client_server_ids_and_bytes_joined"
        and worker["http_evidence_closure"]["requests"]
        == [
            {
                "request_id": row["server_request_id"],
                "status": row["status"],
                "request_sha256": row["request_sha256"],
                "response_sha256": row["response_sha256"],
            }
            for row in report["http"]
        ],
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
        "schema": "root-closed-operation-cpu5-browser-diagnostic-v1",
        "status": "CANDIDATE_EXECUTION_PATH_VERIFIED_STRICT_BROWSER_DIAGNOSTIC_PASS"
        if strict_success
        else "CANDIDATE_EXECUTION_PATH_VERIFIED_STRICT_BROWSER_DIAGNOSTIC_FAIL",
        "authority": "root",
        "root_exit_code": terminal["exit_code"],
        "owner_completed": receipt["completed"],
        "original_positive_scoped_closure": True,
        "independent_actual_review_claimed": False,
        "separate_reviewer_actual_bytes_read": False,
        "source_peer_review_scope": "source only",
        "source_records_authenticated": SOURCE_UNION_COUNT,
        "model_records_authenticated": 20,
        "runtime_records_authenticated": 28,
        "runtime_SHA_records": 22,
        "runtime_stat_only_records": 6,
        "model_loads": 1,
        "candidate_alias": cap["alias"],
        "candidate_profile": CANDIDATE_PROFILE,
        "only_adapted_head_id": "operation",
        **DEV_FLAGS,
        "candidate_capability": plan["candidate_capability"],
        "numeric_acceptance": plan["numeric_acceptance"],
        "calls": calls,
        "operation_sequence": operations,
        "trusted_clicks": sum(row["trusted"] is True for row in report["final_judge"]["clicks"]),
        "wrong_clicks": sum(row["id"] != "copper" for row in report["final_judge"]["clicks"]),
        "completion_text_reached_next_model_inputs": completion,
        "final_status": report["final_status"],
        "goal_satisfied": goal_satisfied,
        "strict_success": strict_success,
        "browser_score": None,
        "benchmark_scores": None,
        "training_eligible": False,
        "exposed_diagnostic_may_not_enter_training": True,
        "new_unseen_evaluation_claim": False,
        "official_score_claim": False,
        "quality_or_win_claim": False,
        "model_plus_browser_observed_unit_peak_bytes": memory["kernel_memory_peak_bytes"],
        "memory_events": memory["kernel_memory_events"]["counters"],
        "worker_seconds_before_final_write": worker["elapsed_before_final_write_seconds"],
        "root_seconds": terminal["wall_seconds"],
        "all_actual_HTTP_candidate_joins_performed_by_root": True,
        "actual_http_calls": len(calls),
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
        "scope": "Real HTTP/choose/Agent/Chromium wiring with the unchanged strict exposed Copper diagnostic; result and call count derived from all report/server evidence. Same new operation artifact and original byte/source/model identities checked; root joins every retained operation-only projection, unchanged target logits, legal native distributions and unchanged wire values. FP64 witness adds retained delta only for actual operation IDs then applies stable softmax; it is not a new h/matmul replay. Successful200 publication is guarded by original source final20; terminal time sample precedes writes and does not measure network delivery. No shared-head qualification, official/unseen evaluation, quality or benchmark win.",
    }
    return result


def main():
    release_gate()  # Before parser or any source/artifact read.
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("terminal", "spec", "plan"):
        parser.add_argument("--" + name, type=Path)
        parser.add_argument("--" + name + "-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute(
        terminal=args.terminal,
        terminal_sha=args.terminal_sha256,
        spec=args.spec,
        spec_sha=args.spec_sha256,
        plan=args.plan,
        plan_sha=args.plan_sha256,
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "operations": result["operation_sequence"],
                "actual_http_calls": result["actual_http_calls"],
                "strict_success": result["strict_success"],
            }
        )
    )


if __name__ == "__main__":
    main()
