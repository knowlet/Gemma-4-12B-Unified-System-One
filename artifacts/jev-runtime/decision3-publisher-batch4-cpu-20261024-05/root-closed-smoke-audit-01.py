"""Read-only root audit of the closed CPU5 smoke; no model/scorer execution."""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = Path(__file__).resolve().parent
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
OUT = OWNER / "actual-01"
UNIT = "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service"
PROFILE = "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"
PLAN_SHA = "095db88b78687d90fbed51aafd736c1b06fb1a582312de8dff97bf6109131e96"
SPEC_SHA = "7ed862c88a81acb05416ebe3c3902073d028597f8ca5531c1dbdf65b0b29e2be"
CONTROLLER_SHA = "81846efe2ee0357da07283c1a77a606283a920cbddef668c8f62def291057dee"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha(path),
    }


def read(path):
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= 1048576,
        "Bounded original JSON required",
    )
    return json.loads(path.read_text())


def authenticate(row):
    p = Path(row["path"])
    p = p if p.is_absolute() else ROOT / p
    info = p.stat()
    size = row.get("size_bytes", row.get("size"))
    require(size is None or size == info.st_size, "Selected file size differs")
    if "stat" in row:
        require(
            all(type(v) is int and getattr(info, k) == v for k, v in row["stat"].items()),
            "Selected original stat identity differs",
        )
    if "sha256" in row:
        require(sha(p) == row["sha256"], "Selected file hash differs")
    else:
        require(bool(row.get("stat")), "Original stat-only scope required")


def execute():
    terminal_path = OWNER / "root-execution-native05-edge-smoke-01.json"
    require(
        terminal_path.is_file() and (OUT / "receipt.json").is_file(),
        "Original root and owner must be terminal first",
    )
    terminal, receipt = read(terminal_path), read(OUT / "receipt.json")
    closure = read(OUT / "external-boundary-closure.json")
    require(
        terminal["exit_code"] == 0
        and receipt["profile"] == PROFILE
        and receipt["completed"] is True
        and receipt["primary_error"] is None
        and receipt["secondary_errors"] == []
        and receipt["sources_unchanged"] is True
        and receipt["comfy_service_modified"] is False,
        "Original successful owner and unchanged sources required",
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
        )
        and len(closure["two_process_scans"]) == 2
        and all(not row["matches"] and not row["unknown"] for row in closure["two_process_scans"]),
        "Original current authenticated scoped closure required",
    )
    for k in (
        "no_escape_enforced",
        "whole_tree_exit_confirmed",
        "continuation_authorized",
        "model_qualified",
    ):
        require(closure[k] is False, "Original universal nonqualification flags required")
    config = receipt["selected_config"]
    require(
        config == read(OUT / "sealed-config.json")
        and config["profile"] == PROFILE
        and config["unit"] == UNIT
        and config["output"] == str(OUT)
        and config["selection_sha256"] == SPEC_SHA,
        "Exact current selected config required",
    )
    admission = read(OUT / "admission.json")
    require(
        admission["nonce"] == config["nonce"] == closure["selection"]["nonce"]
        and admission["invocation_id"] == closure["invocation_id"]
        and closure["selection"]["unit"] == UNIT
        and all(
            admission["actor"][k] == closure["actor"][k]
            for k in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")
        )
        and admission["config_sha256"] == sha(OUT / "sealed-config.json"),
        "Current admission/config/closure invocation differs",
    )
    spec_path, plan_path = HERE / "root-edge-smoke-command-spec-01.json", HERE / "edge-plan.json"
    require(
        sha(spec_path) == SPEC_SHA
        and sha(plan_path) == PLAN_SHA
        and sha(OWNER / "controller.py") == CONTROLLER_SHA,
        "Selected source/plan pins differ",
    )
    spec, plan = read(spec_path), read(plan_path)
    base = [
        OWNER / "controller.py",
        spec_path,
        ROOT / "artifacts/jev-runtime/decision3-local-owner-terminal-20261022-01/owner_boundary.py",
        ROOT / "artifacts/jev-runtime/observer-system-services-20261022-01/comfy_observer.py",
        ROOT / "artifacts/jev-runtime/observer-system-services-20261022-01/system-services.json",
    ]
    expected = {r["path"]: r for r in [*spec["sources"], *(ref(p) for p in base)]}
    require(
        len(spec["sources"]) == 50
        and len(expected) == 55
        and {r["path"]: r for r in config["sources"]} == expected,
        "Exact50 plus5 original source union required",
    )
    for row in config["sources"] + spec["binary_stats"]:
        authenticate(row)
    for row in [
        *plan["model_files"],
        *plan["runtime_files"],
        plan["manifest"],
        *plan["owner_sources"],
    ]:
        authenticate(row)
    require(
        len(plan["model_files"]) == 20 and len(plan["runtime_files"]) == 28,
        "Original model/runtime inventory differs",
    )
    expected_argv = [
        str(ROOT / ".venv/bin/python"),
        "-B",
        "-s",
        str(OWNER / "controller.py"),
        "--run",
        "--expected-entry-sha256",
        CONTROLLER_SHA,
        "--selection",
        str(spec_path),
        "--selection-sha256",
        SPEC_SHA,
        "--output",
        str(OUT),
    ]
    require(
        terminal["argv"] == expected_argv
        and terminal["profile"] == "root-selected-native05-cpu5-fast-edge-smoke-terminal-v1",
        "Exact root CPU5 invocation differs",
    )
    command = read(OUT / "command-01-output.json")["result"]
    started = read(OUT / "command-01-started.json")
    intent = read(OUT / "command-01-intent.json")
    require(
        command["returncode"] == 0
        and command["pid"] == started["pid"]
        and command["argv"]
        == [v.replace("{OUTPUT}", str(OUT)) for v in spec["commands"][0]["argv"]],
        "Original successful selected child command required",
    )
    observed = started["observed_identity"]
    child, wrapper = observed["process"], admission["actor"]
    require(
        child["pid"] == started["pid"] == command["pid"]
        and child["pid"] != wrapper["pid"]
        and all(child[k] == wrapper[k] for k in ("pgid", "sid", "uid", "cgroup"))
        and all(
            child[k] == command["observed_identity"]["process"][k]
            for k in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")
        )
        and all(observed[k] == command["observed_identity"][k] for k in ("exe", "argv", "cwd"))
        and observed["argv"] == intent["argv"] == command["argv"]
        and observed["cwd"]
        == intent["cwd"]
        == command["cwd"]
        == spec["commands"][0]["cwd"].replace("{OUTPUT}", str(OUT))
        and observed["exe"]
        == intent["resolved_executable"]
        == command["resolved_executable"]
        == str(Path(command["argv"][0]).resolve(strict=True)),
        "Current admitted wrapper cohort and exact child identity/exe/argv/cwd required",
    )
    state_path = OUT / "edge-probe/state.json"
    s = read(state_path)
    require(
        s["profile"] == plan["profile"]
        and s["plan_sha256"] == PLAN_SHA
        and s["model_key"] == "edge"
        and s["completed"] is True
        and s["status"] == "native_batch4_two_call_synthetic_probe_complete_no_benchmark_claim"
        and s["model_loads"] == 1
        and s["native_request_calls"] == 2
        and s["benchmark_calls"] == s["browser_calls"] == 0
        and s["cuda_initialized_final"] is False
        and s["initial_full_model_manifest_runtime_sources_authenticated"] is True
        and s["final_full_model_manifest_runtime_source_identities_unchanged"] is True,
        "One actual constructor/two native calls/final source authentication required",
    )
    proc = s["process"]
    child = started["observed_identity"]["process"]
    require(
        proc["pid"] == child["pid"]
        and proc["starttime_ticks"] == child["start_ticks"]
        and proc["pgid"] == child["pgid"]
        and proc["sid"] == child["sid"]
        and proc["cgroup"].strip() == "0::" + child["cgroup"]
        and proc["affinity"] == list(range(15, 20)),
        "Actual model child identity/affinity differs",
    )
    limits = s["owned_limits"]
    quota, period = map(int, limits["cpu.max"].split())
    require(
        limits["memory.max"] == "4294967296"
        and limits["memory.swap.max"] == "0"
        and period > 0
        and quota == 5 * period,
        "Actual CPU5/4GiB/swap0 limits differ",
    )
    require(
        all(
            s["actual_environment"][k] == "5"
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        ),
        "Actual CPU5 environment required",
    )
    require(
        all(
            type(v) in (int, float) and math.isfinite(v) and v > 0
            for v in (s["elapsed_seconds"], command["elapsed_seconds"], terminal["wall_seconds"])
        )
        and s["elapsed_seconds"] <= 120
        and s["elapsed_seconds"] <= command["elapsed_seconds"] <= terminal["wall_seconds"]
        and command["elapsed_seconds"] <= 10800,
        "Original actor120/owner10800 clocks and ordered enclosing scopes required",
    )
    request, lengths = plan["request"], s["input_tokens_per_question"]
    require(
        request == s["full_developer_request"]
        and len(request["questions"]) == 4
        and set(lengths) == set(request["questions"])
        and len(set(lengths.values())) > 1
        and all(type(n) is int and 0 < n <= 512 for n in lengths.values()),
        "Complete original synthetic request/context required",
    )
    provenance = s["native_provenance"]
    require(
        provenance["model_name"] == plan["model"]["name"]
        and provenance["batch_size"] == 4
        and provenance["model_sha256"] == plan["model"]["model_sha256"]
        and provenance["files_sha256"]["MODEL_MANIFEST.json"] == plan["model"]["manifest_sha256"],
        "Actual original model and manifest identity differs",
    )
    require(
        len(s["calls"]) == len(s["actual_first_layer_events"]) == 2, "Two actual events required"
    )
    calls = []
    for i, call in enumerate(s["calls"]):
        event, raw = call["actual_first_layer_event"], call["raw_response"]
        require(
            call["index"] == i
            and call["status"] == "validated"
            and call["completed"] is True
            and event == s["actual_first_layer_events"][i]
            and event["shape"] == [4, max(lengths.values()), 1024]
            and event["dtype"] == "torch.bfloat16"
            and event["device"] == "cpu"
            and event["inputs_or_outputs_modified"] is False
            and all(
                type(v) in (int, float) and math.isfinite(v) and v > 0
                for v in (call["validation_seconds"], call["durable_validation_seconds"])
            )
            and 0 < call["validation_seconds"] <= call["durable_validation_seconds"] <= 20,
            "Actual shape/dtype/unaltered forward and durable20-second scope required",
        )
        require(
            raw == read(OUT / "edge-probe" / f"call-{i}-raw.json")["raw_response"]
            and raw["model"] == plan["model"]["name"]
            and list(raw["answers"]) == list(request["questions"])
            and raw["usage"] == {"input_tokens": sum(lengths.values()), "output_tokens": 0},
            "Complete original raw response/order/token accounting differs",
        )
        for key, q in request["questions"].items():
            answer = raw["answers"][key]
            ps = answer["probabilities"]
            require(
                answer["type"] == "choice"
                and list(ps) == list(q["criteria"])
                and all(
                    type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                    for v in ps.values()
                )
                and abs(sum(ps.values()) - 1) <= 1e-6
                and answer["choice"] == max(ps, key=ps.__getitem__),
                "Complete legal native distribution required",
            )
            entropy = -sum(v * math.log(v) for v in ps.values() if v > 0)
            require(
                type(answer["confidence"]) in (int, float)
                and math.isfinite(answer["confidence"])
                and abs(answer["confidence"] - (1 - entropy / math.log(len(ps)))) <= 1e-6,
                "Original entropy confidence differs",
            )
        calls.append(
            {
                "index": i,
                "durable_seconds": call["durable_validation_seconds"],
                "shape": event["shape"],
            }
        )
    result = {
        "schema": "root-closed-native-batch4-cpu5-fast-smoke-review-v1",
        "status": "PASS_WITH_LIMITS_SYNTHETIC_SMOKE_ONLY",
        "reviewer": "root closed actual review",
        "separate_reviewer_actual_bytes_read": False,
        "full07_prerequisite_selected": False,
        "model_key": "edge",
        "model_revision": plan["model"]["revision"],
        "model_identity_sha256": plan["model"]["model_sha256"],
        "owner_unit": UNIT,
        "memory_max_bytes": 4294967296,
        "cpu_affinity": list(range(15, 20)),
        "threads": 5,
        "model_loads": 1,
        "calls": calls,
        "whole_root_seconds": terminal["wall_seconds"],
        "command_seconds": command["elapsed_seconds"],
        "actor_seconds": s["elapsed_seconds"],
        "post_final_auth_memory": s["after_final_auth_owned_counters"]["counters"],
        "selected_source_records_authenticated": 55,
        "model_records_authenticated": 20,
        "runtime_records_authenticated": 28,
        "runtime_scope": "selected SHA plus original stat-only records; no full DSO claim",
        "original_authenticated_scoped_closure": True,
        "whole_tree_exit_confirmed": False,
        "model_quality_browser_training_or_win_qualified": False,
        "qualification": {
            "smoke_complete": True,
            "external_authenticated_closure": True,
            "one_actual_constructor": True,
            "two_actual_four_row_events": True,
            "durable_call_20s": True,
            "whole_command_120s": command["elapsed_seconds"] <= 120,
            "final_full_model_and_sources_authenticated": True,
        },
        "artifact_refs": {
            "state": ref(state_path),
            "plan": ref(plan_path),
            "owner_receipt": ref(OUT / "receipt.json"),
            "root_execution": ref(terminal_path),
            "spec": ref(spec_path),
            "closure_observation": ref(OUT / "external-boundary-closure.json"),
            "audit_source": ref(Path(__file__)),
        },
        "raw_requests_answers_probabilities_features_or_model_bytes_included": False,
    }
    path = HERE / "root-closed-smoke-result-01.json"
    with path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "calls": calls, "review": ref(path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    if parser.parse_args().execute:
        execute()
    else:
        print(json.dumps({"status": "not_executed", "actual_evidence_read": False}))
