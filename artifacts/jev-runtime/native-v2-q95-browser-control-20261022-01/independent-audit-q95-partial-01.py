"""Read-only retained-evidence audit; no runtime/framework imports or HTTP calls."""

import collections
import datetime
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
RUN = HERE / "v2-q95-01"
BROWSER = RUN / "browser"
CONTROL = RUN / "q95-control"
PLAN_SHA = "f3f03d139e8441022b9558ce4dabe55a585e17e49807ad6a50ac5c3067e64756"
ACTOR = {"pid": 117747, "pgid": 117747, "sid": 117747, "start_ticks": 1780470}
cache = {}


def need(value, label):
    if not value:
        raise AssertionError(label)


def read(path):
    need(path.stat().st_size <= 32 * 1024**2, "Bounded retained metadata")
    return json.loads(path.read_bytes())


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def pin(path):
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    identity = (str(resolved), before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    if identity not in cache:
        h = hashlib.sha256()
        with resolved.open("rb") as stream:
            while chunk := stream.read(1024**2):
                h.update(chunk)
        after = resolved.stat()
        need(identity[1:] == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "Stable retained bytes")
        cache[identity] = {"sha256": h.hexdigest(), "size_bytes": before.st_size}
    return cache[identity]


def ref(path):
    return {"path": path.relative_to(ROOT).as_posix(), **pin(path)}


def checked_ref(value):
    path = ROOT / value["path"]
    need(path.is_relative_to(ROOT) and pin(path) == {k: value[k] for k in ("sha256", "size_bytes")}, "Exact selected retained reference bytes")
    return path


def main():
    started = time.monotonic()
    plan_path = RUN / "selected-plan.json"
    plan = read(plan_path)
    need(pin(plan_path)["sha256"] == PLAN_SHA, "Externally selected actual plan")
    need(len(plan["sources"]) == 893, "Exact selected893 refs")
    selection = read(checked_ref(plan["helper_selection"]))
    seed = read(checked_ref(selection["helper_seed_ref"]))
    helper_directory = Path(seed["service_directory"])
    external_path = RUN / "root-external-closure.json"
    external = read(external_path)
    helper_final_path = ROOT / "artifacts/jev-runtime/decision3-q95-owner-20261022-03/owner-03/owner-final.json"
    helper_final = read(helper_final_path)
    native_path, control_path = BROWSER / "receipt.json", CONTROL / "receipt.json"
    native, control = read(native_path), read(control_path)
    need(native["stage"] == control["stage"] == "failed", "Historical interruption preserved")
    need(external["plan_sha256"] == PLAN_SHA and external["actor"] == ACTOR and external["actor_read"]["actor"] == ACTOR, "Bound actual actor and plan")
    need(external["actor_read"]["nonce"] == plan["nonce"] and external["unit_readback_verified"] is True, "Bound nonce and kernel readback")
    need(external["reason"] == "owner_error" and external["primary_error"]["type"] == "ObservationError", "Original environmental primary preserved")
    need(external["cgroup_absent_after_stop"] is True and external["after_stop"]["MainPID"] == "0", "Native own cgroup and main process closure")
    need(len(external["invocation_id"]) == 32 and external["terminal_before_stop"]["InvocationID"] == external["invocation_id"] and external["terminal_before_stop"]["ExecMainPID"] == str(ACTOR["pid"]), "Owned terminal invocation and actor")
    need(len(external["two_proc_scans"]) == 2 and all(row["matches"] == row["unreadable"] == [] for row in external["two_proc_scans"]), "Native two clean process scans")
    need(external["changed_sources"] == [], "Root recorded all893 source closure")
    hc = helper_final["closure"]
    need(hc["whole_unit_exit_confirmed"] is True and hc["observed_cgroup_cohort_exit_confirmed"] is True and hc["cgroup_absent"] is True and hc["sources_unchanged"] is True and helper_final["full_population_unchanged"] is True, "Separate helper own unit/cohort/source closure")
    need(len(hc["two_process_scans"]) == 2 and all(row["matches"] == row["unknown"] == [] for row in hc["two_process_scans"]), "Helper two clean process scans")
    need(any(row["pid"] == seed["binding"]["pid"] and row["start_ticks"] == seed["binding"]["start_ticks"] for row in hc["tracked_cohort"]), "Selected helper child in actual closed cohort")
    actor_pids = {ACTOR["pid"], seed["binding"]["pid"], hc["actor"]["pid"]}
    need(all(not (Path("/proc") / str(pid)).exists() for pid in actor_pids), "Independent current actor/helper process absence")
    need(not (Path("/sys/fs/cgroup") / plan["cgroup"].lstrip("/")).exists() and not (Path("/sys/fs/cgroup") / hc["actor"]["cgroup"].lstrip("/")).exists(), "Independent current both cgroup absence")
    need(external["host_after"]["boot_id"] == plan["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip() and external["host_after"]["mem_total_bytes"] == plan["expected_memtotal"] and external["host_after"]["comfy_state"] == "absent_confirmed", "Recorded final known same host")
    need(native["server_thread_drained"] is True and native["cleanup_errors"] == [] and native["source_verification"]["success"] is True, "Native drain and runtime source closure")
    need(all(native["source_verification"][k] == [] for k in ("added", "changed", "missing")), "Native exact runtime source population")
    need(control["native_receipt"] == pin(native_path) and control["selection_sha256"] == plan["helper_selection"]["sha256"] and control["helper_binding"] == seed["binding"], "Control receipt original native two-field pin and exact helper binding")
    partial_attempts = []
    for group in ("known4", "public6"):
        for attempt in sorted((BROWSER / group / "results/attempts").glob("*")):
            outcome = attempt / "outcome.json"
            if outcome.exists():
                row = read(outcome)
                missing = [name for name in ("model_calls.json", "final_state.json", "states.json") if not (attempt / name).is_file()]
                partial_attempts.append({"group": group, "case_id": row["case_id"], "status": row["status"], "error_stage": row.get("error_stage"), "execution_error_present": row["error"] is not None, "reported_policy_calls": row["policy_calls"], "reported_text_calls": row["text_calls"], "durable_complete_attempt_bundle": not missing, "missing_files": missing, "outcome": ref(outcome)})
    need(len(partial_attempts) == 1 and partial_attempts[0]["missing_files"] == ["model_calls.json", "final_state.json", "states.json"], "One interrupted attempt, no complete durable bundle")
    need(all(not (BROWSER / group / "results/summary.json").exists() for group in ("known4", "public6")), "No completed group summary")
    need(control["group_indexes"] == [] and all(not (BROWSER / group / "external-q95-request-index.json").exists() for group in ("known4", "public6")), "Both missing completed-group external indexes declared")
    output_index_path = BROWSER / "output-index.json"
    output_index = read(output_index_path)
    actual = {}
    for path in BROWSER.rglob("*"):
        need(not path.is_symlink(), "Ordinary native output population")
        if path.is_file() and path != output_index_path:
            actual[path.relative_to(BROWSER).as_posix()] = pin(path)
    need(actual == output_index["files"] and len(actual) == output_index["file_count"] and sum(row["size_bytes"] for row in actual.values()) == output_index["total_bytes"], "Exact native output inventory")
    server_index_path = BROWSER / "server-index.json"
    server_index = read(server_index_path)
    need(server_index["all_requests_sealed"] is True and server_index["pending_request_ids"] == server_index["errors"] == [] and server_index["local_service_thread_drained"] is True, "Sealed native journals")
    server = {}
    for row in server_index["requests"]:
        path = BROWSER / "server" / row["path"]
        need(pin(path) == {"sha256": row["sha256"], "size_bytes": row["bytes"]}, "Native server journal pin")
        lines = [json.loads(line) for line in path.read_bytes().splitlines()]
        need(len(lines) == row["events"] and [x["sequence"] for x in lines] == list(range(len(lines))) and all(x["request_id"] == row["request_id"] for x in lines), "Native journal sequence and request identities")
        need(lines[-1]["event"] == row["last_event"] and row["sealed"] and not row["write_failed"], "Native terminal journal")
        server[row["request_id"]] = lines[-1]["data"].get("response")
    exchanges = sorted((BROWSER / "http").glob("*/exchange.json"))
    policies = sorted((CONTROL / "policy").glob("policy-*"))
    need(len(server) == len(exchanges) == len(policies) == 12, "Exact twelve policy candidate exchanges")
    policy_rows = []
    for number, (path, folder) in enumerate(zip(exchanges, policies)):
        row, candidate, selected, publication = read(path), read(folder / "candidate-admission.json"), read(folder / "selected-admission.json"), read(folder / "publication-attempt.json")
        native_request = path.parent / "request.bin"
        native_response = path.parent / "response.bin"
        need(row["path"] == "/v1/systemone" and row["status"] == 200 and pin(native_request) == row["request"] and pin(native_response) == row["response"], "Native retained request/body/status")
        need(row["api_request_id"] in server and read(native_response) == server[row["api_request_id"]], "Native HTTP/server semantic response join")
        need((folder / "request.bin").read_bytes() == native_request.read_bytes() and (folder / "native-candidate.bin").read_bytes() == native_response.read_bytes(), "Outer/native exact request and candidate bytes")
        need(candidate["candidate_sha256"] == pin(native_response)["sha256"] and candidate["sequence"] == selected["sequence"] == publication["sequence"] == number, "Outer candidate body hash and ordering")
        need(publication["selected_sha256"] == control["policy_candidates_and_publication_attempts"][number]["selected_sha256"] and publication["selected_status"] == control["policy_candidates_and_publication_attempts"][number]["selected_status"], "Control final publication ledger equality")
        if publication["selected_status"] == 200:
            need(publication["selected_sha256"] == pin(native_response)["sha256"] and candidate["error"] is None, "Timely admitted original candidate")
        policy_rows.append({"sequence": number, "native_candidate_status": row["status"], "outer_selected_status": publication["selected_status"], "native_exchange": ref(path), "candidate_admission": ref(folder / "candidate-admission.json"), "publication_attempt": ref(folder / "publication-attempt.json")})
    statuses = collections.Counter(row["outer_selected_status"] for row in policy_rows)
    need(statuses == {200: 11, 503: 1}, "Eleven admitted plus one secondary refusal")
    checks = []
    checked_auth_raw = set()
    for path in sorted((CONTROL / "helper").glob("helper-check-*.json")):
        value = read(path)
        need(value["binding"] == seed["binding"] and value["selection_sha256"] == plan["helper_selection"]["sha256"], "Every checker receipt exact same Q95 binding")
        admitted = value["helper_admitted"] and value["co_resident_admitted"]
        if admitted:
            auth = value["authentication"]
            need(auth["inference_requested"] is False and [x["status"] for x in auth["requests"]] == [500, 422] and value["elapsed_seconds"] <= 3, "Original no-inference selected auth and shared3s gate")
            checked_auth_raw.add(str(checked_ref(auth["requests"][1]["handler_evidence"])))
        checks.append({"phase": value["phase"], "admitted": bool(admitted), "error_type": value["error_type"], "error_code": value["error_code"], "elapsed_seconds": value["elapsed_seconds"], "receipt": ref(path)})
    need(len(checks) == 28 and sum(row["admitted"] for row in checks) == 27 and checks[-1]["phase"] == "after-native-policy-11" and checks[-1]["error_type"] == "FileNotFoundError", "Actual last secondary helper disappearance check")
    external_population = []
    for directory in sorted((helper_directory / "requests").iterdir()):
        raw_path, final_path = directory / "raw.json", directory / "outer-completion.json"
        raw, final = read(raw_path), read(final_path)
        need(raw["request_id"] == final["request_id"] == directory.name and raw["guard_restored"] is True and raw["server_terminal"] is True and raw["error"] is None and final["raw_sha256"] == pin(raw_path)["sha256"] and final["completed"] and final["final_gate_measured"], "Actual unpruned Q95 raw/final join")
        generation = raw.get("generation")
        row = {"request_id": directory.name, "kind": "field_text" if generation else "schema_authentication", "payload_sha256": raw["payload_sha256"], "selected_status": final["selected_status"], "outer_elapsed_ms": final["elapsed_ms"], "raw": ref(raw_path), "outer_completion": ref(final_path)}
        if generation:
            need(final["selected_status"] == 200 and final["outer_authorized"] is True and final["elapsed_ms"] < 20000 and generation["profile"] == seed["binding"]["profile"] and generation["success"] and generation["response_authorized"] and generation["finish_reason"] == "actual_EOS" and generation["seconds"] == 20 and generation["max_new_tokens"] == 128 and generation["deadline_exceeded"] is False and generation["adapter_restore_complete"] is True and generation["action_adapter_state"] == "disabled_for_pinned_base_helper", "Actual Q95 same-profile authorized timely text candidate")
            need(len(raw["forward_events"]) == len(generation["raw_token_ids"]) == len(generation["callback_events"]) == generation["usage"]["output_tokens"], "Actual text forward/token/callback accounting")
            row.update(output_tokens=len(generation["raw_token_ids"]), native_forwards=len(raw["forward_events"]), raw_generation_latency_ms=generation["latency_ms"], client_delivery_proven=False)
        else:
            need(str(raw_path) in checked_auth_raw and final["selected_status"] == 422 and raw["forward_events"] == [], "No-inference schema probe exact checked raw")
        external_population.append(row)
    need(len(external_population) == 28 and sum(row["kind"] == "field_text" for row in external_population) == 1 and len(checked_auth_raw) == 27, "Complete observed Q95 session population")
    small, large = 0, []
    for row in plan["sources"]:
        path = ROOT / row["path"]
        if row["size_bytes"] > 64 * 1024**2:
            need(path.stat().st_size == row["size_bytes"], "Declared large payload size")
            large.append(row)
        else:
            need(pin(path) == {k: row[k] for k in ("sha256", "size_bytes")}, "Independent small selected source/artifact hash")
            small += 1
    need(small == 892 and len(large) == 1, "892 independent hashes and one declared base payload")
    aliases = external["model_cache_aliases_after"]
    need(aliases == external["model_cache_aliases_before"] and len(aliases) == 11, "Root complete before/after aliases equal")
    for value in aliases:
        logical, resolved = ROOT / value["logical"]["path"], ROOT / value["resolved"]["path"]
        need(logical.resolve(strict=True) == resolved, "Independent current exact alias target")
        for link in value["links"]:
            need(os.readlink(ROOT / link["path"]) == link["link_text"], "Independent current alias chain")
        stat = resolved.stat()
        need(value["stat"] == {"device": stat.st_dev, "inode": stat.st_ino, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}, "Independent current alias target stat")
    result = {"schema_id": "gemma-q95-interrupted-independent-audit-v1", "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(), "status": "infrastructure_interrupted_owned_processes_closed", "evaluation_complete": False, "full_score": None, "planned_denominator": 10, "completed_groups": 0, "durable_complete_attempt_bundles": 0, "partial_attempts": partial_attempts, "primary_failure": {"type": "ObservationError", "reason": "Separate Q95/native owners conservatively refused unresolved UID0 Python cwd identity", "offenders": [{"pid": 120154, "start_ticks": 1800891}, {"pid": 120155, "start_ticks": 1800893}], "provenance_of_uid0_processes": "Not inferred by this audit; parent investigation owns attribution", "native_external_receipt": ref(external_path), "separate_helper_final": ref(helper_final_path)}, "secondary_helper_refusal": checks[-1], "native_policy": {"original_candidate_http200": 12, "outer_admitted_http200": 11, "outer_refused_http503": 1, "exact_native_outer_body_and_server_joins": True, "publication_attempts": policy_rows, "client_delivery_proven": False}, "helper_checks": {"count": len(checks), "admitted": 27, "refused": 1, "receipts": checks}, "q95_requests": {"observed_full_service_population": external_population, "schema_authentication_zero_forward_count": 27, "actual_field_text_generation_count": 1, "group_indexes_present": False, "client_model_calls_file_present": False, "per_client_text_request_join_proven": False, "scope": "Raw/final Q95 session population hashes only. Known4 was interrupted before either per-group external index or model_calls.json was written. One outcome reports1 text call, but client receipt/body join and delivered/typed text are unproven."}, "native_output_inventory": {"index": ref(output_index_path), "files": len(actual), "bytes": output_index["total_bytes"], "exact_population_and_hashes_verified": True}, "sources": {"selected_plan": ref(plan_path), "ordinary_selected_count": 893, "root_recorded_full_closure_unchanged": True, "independently_hashed_small_count": small, "large_payload_refs": large, "large_payload_full_rehash_independently_credited": False, "current_alias_target_chain_stat_count": len(aliases)}, "late_closure": {"native_actor": ACTOR, "native_invocation": external["invocation_id"], "native_unit": plan["unit"], "native_two_scans_empty": True, "helper_actor": hc["actor"], "helper_child": {k: seed["binding"][k] for k in ("pid", "start_ticks", "pgid", "sid")}, "helper_invocation": hc["invocation_id"], "helper_two_scans_empty": True, "both_selected_cgroups_currently_absent": True, "all_three_selected_processes_currently_absent": True, "no_universal_whole_tree_escape_claim": True}, "native_receipt": ref(native_path), "control_receipt": ref(control_path), "refs": {"server_index": ref(server_index_path), "public_judge_source": ref(ROOT / "scripts/jev_public_web_cases.py"), "heldout_judge_source": ref(ROOT / "scripts/jev_heldout_public_cases.py")}, "claim_limits": ["No completed10-task score or quality0 assigned", "No durable complete attempt bundle or successful DOM replay credited", "Server/native ASGI admission does not prove client response delivery", "One actual Q95 generation is not a proven client text/DOM effect join", "Source hashes and original qualifier receipts do not prove numerical argmax replay", "Native40 memcg does not claim process-native CUDA cap or universal no-escape"], "audit_execution": {"source": ref(Path(__file__)), "elapsed_seconds": time.monotonic() - started, "models_frameworks_HTTP_units_or_Comfy_calls": False, "raw_DOM_questions_answers_or_private_config_published": False}}
    path = HERE / "independent-audit-q95-partial-01.json"
    with path.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"audit": ref(path), "status": result["status"], "full_score": None, "complete_groups": 0, "complete_attempt_bundles": 0, "policy_admitted": 11, "policy_refused": 1, "actual_q95_text_generations": 1, "independent_small_hashes": small, "elapsed_seconds": time.monotonic() - started}))


if __name__ == "__main__":
    main()
