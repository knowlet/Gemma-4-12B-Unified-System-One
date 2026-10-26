import collections
import datetime
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

R = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
P = R / "artifacts/jev-runtime/native-v2-browser-execution-20261022-04"
D = P / "v2-01"
B = D / "browser"
PLAN_SHA = "73bff770e242b93d5b7b53d24ef933926bd63897a4707d6d2e2a0f06f4438b6d"
INVOCATION = "162ea2fa15394651861a59462050d4d3"
ACTOR = {"pid": 75166, "pgid": 75166, "sid": 75166, "start_ticks": 1175624}
started = time.monotonic()
cache = {}
checks = {}
refs = {}


def need(value, label):
    if not value:
        raise AssertionError(label)


def read(path):
    need(path.stat().st_size <= 32 * 1024**2, "Bounded metadata " + path.name)
    return json.loads(path.read_bytes())


def pin(path):
    actual = path.resolve()
    before = actual.stat()
    key = (
        str(actual),
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if key not in cache:
        h = hashlib.sha256()
        with actual.open("rb") as f:
            for block in iter(lambda: f.read(1024**2), b""):
                h.update(block)
        after = actual.stat()
        need(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "Stable bytes " + path.name,
        )
        cache[key] = {"sha256": h.hexdigest(), "size_bytes": before.st_size}
    return cache[key]


def ref(path):
    return {"path": str(path.relative_to(R)), **pin(path)}


def checked_source(name, expected):
    path = R / ("scripts/" + name + ".py")
    blob = path.read_bytes()
    need(hashlib.sha256(blob).hexdigest() == expected, "Pinned judge source " + name)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(blob, str(path), "exec", dont_inherit=True), module.__dict__)
    refs[name] = ref(path)
    return module


plan = read(D / "selected-plan.json")
need(pin(D / "selected-plan.json")["sha256"] == PLAN_SHA, "Externally selected plan")
need(len(plan["sources"]) == 785, "Exactly785 sources")
refs["plan"] = ref(D / "selected-plan.json")
public = checked_source(
    "jev_public_web_cases", "0ec9826d334cb2fbed24ade8780fa785291017d904b18233419ced760750bca3"
)
heldout = checked_source(
    "jev_heldout_public_cases", "339cddf596a25ab433ca66e23924631b6261e62efc06f351878d4ef5c3f6f8f9"
)
groups = []
all_rows = []
expected_ids = []
for group, module, count in [("known4", public, 4), ("public6", heldout, 6)]:
    selected = list(module.PUBLIC_WEB_CASES)
    need(len(selected) == count, "Fixed group selection")
    expected = [task + "-000" for task in selected]
    expected_ids += expected
    path = B / group / "results/summary.json"
    if not path.exists():
        continue
    summary = read(path)
    rows = summary["outcomes"]
    need([r["case_id"] for r in rows] == expected[: len(rows)], "Ordered fixed IDs")
    aggregate = collections.Counter()
    for row in rows:
        task = row["task_id"]
        need(task in selected and row["case_id"] == task + "-000", "Task/attempt identity")
        replay = module.validate_public_web_observation(task, row["verification"]["observed"])
        need(replay == row["verification"], "Exact recorded DOM judge replay")
        folder = B / group / "results/attempts" / row["case_id"]
        need(read(folder / "outcome.json") == row, "Outcome summary equality")
        state = read(folder / "final_state.json")
        need(
            (state.get("status") if isinstance(state, dict) else None) == row["status"],
            "Actual final state status",
        )
        done = bool(state and state["status"] == "done")
        success = bool(replay["success"] and done and row["error"] is None)
        premature = bool(done and replay["success"] is False and not replay.get("error"))
        need(type(row["success"]) is bool and row["success"] == success, "Strict done+DOM+noerror")
        need(
            type(row["goal_satisfied"]) is bool
            and row["goal_satisfied"] == bool(replay["success"]),
            "Goal accounting",
        )
        need(
            type(row["premature_done"]) is bool and row["premature_done"] == premature,
            "Premature DONE accounting",
        )
        aggregate.update(
            attempts=1,
            successes=int(success),
            goals_satisfied=int(replay["success"]),
            done=int(done),
            premature_done=int(premature),
            execution_errors=int(row["error"] is not None),
            policy_calls=row["policy_calls"],
            text_calls=row["text_calls"],
        )
        all_rows.append(row)
    need(summary["attempts"] == len(rows) == aggregate["attempts"], "Summary attempt accounting")
    for key in ["successes", "goals_satisfied", "policy_calls", "text_calls"]:
        need(summary[key] == aggregate[key], "Summary " + key)
    need(summary["premature_done_count"] == aggregate["premature_done"], "Summary premature DONE")
    need(
        summary["success_rate"] == (aggregate["successes"] / len(rows) if rows else 0),
        "Summary rate",
    )
    closure_path = B / group / "owned-client-closure.json"
    closure = read(closure_path) if closure_path.exists() else None
    client_closed = bool(closure and closure["confirmed"] is True and closure["remaining"] == [])
    groups.append(
        {
            "group": group,
            "fixed_denominator": count,
            "completed": len(rows),
            "aggregate": dict(aggregate),
            "summary": ref(path),
            "client_closure": ref(closure_path) if closure else None,
            "client_closure_confirmed": client_closed,
            "client_tracked_processes": len(closure["tracked_PID_start_ticks"])
            if closure
            else None,
        }
    )
ready = (B / "receipt.json").exists() and (D / "root-external-closure.json").exists()
if not ready:
    print(
        json.dumps(
            {
                "status": "running_or_final_evidence_pending",
                "groups": [
                    {
                        k: g[k]
                        for k in [
                            "group",
                            "fixed_denominator",
                            "completed",
                            "aggregate",
                            "client_closure_confirmed",
                        ]
                    }
                    for g in groups
                ],
                "final_receipt_present": (B / "receipt.json").exists(),
                "external_closure_present": (D / "root-external-closure.json").exists(),
            }
        )
    )
    sys.exit(0)
receipt = read(B / "receipt.json")
external = read(D / "root-external-closure.json")
refs["worker_receipt"] = ref(B / "receipt.json")
refs["external_closure"] = ref(D / "root-external-closure.json")
need([g["completed"] for g in groups] == [4, 6] and len(all_rows) == 10, "Complete fixed ten")
need(all(g["client_closure_confirmed"] for g in groups), "Both actual client closures")
need([r["case_id"] for r in all_rows] == expected_ids, "Exact full ten attempt IDs")
checks["fixed10_IDS_and_recorded_DOM_replay"] = True
need(
    receipt["stage"] == "complete_pending_external_closure"
    and receipt["attempts"] == receipt["planned_denominator"] == 10,
    "Completed worker receipt",
)
need(receipt["successes"] == sum(r["success"] for r in all_rows), "Worker success accounting")
need(
    receipt["seed"] == 20261014 and receipt["repeats"] == 1 and receipt["max_steps"] == 16,
    "Fixed budgets",
)
need(
    receipt["prompt_format"] == plan["prompt_format"] == "shared-elements-v2",
    "Selected V2 prompt profile",
)
need(
    receipt["cleanup_errors"] == [] and receipt["server_thread_drained"] is True,
    "Worker drained/cleanup",
)
need(receipt["source_verification"]["success"] is True, "Worker source closure")
need(
    all(receipt["source_verification"][k] == [] for k in ["changed", "missing", "added"]),
    "Exact worker source population closure",
)
need(
    receipt["source_verification"]["identity_sha256"]
    == receipt["source_verification"]["current_identity_sha256"],
    "Worker source identity unchanged",
)
need(
    [(g["group"], g["denominator"], g["summary"]["attempts"]) for g in receipt["groups"]]
    == [("known4", 4, 4), ("public6", 6, 6)],
    "Receipt fixed groups",
)
checks["worker_complete_and_group_receipts"] = True
idxpath = B / "output-index.json"
index = read(idxpath)
files = index["files"]
actual = {}
for p in B.rglob("*"):
    need(not p.is_symlink(), "No linked owned output")
    if p.is_file() and p != idxpath:
        actual[p.relative_to(B).as_posix()] = pin(p)
need(actual == files, "Complete exact output inventory no extra/missing files")
need(
    index["file_count"] == len(actual)
    and index["total_bytes"] == sum(p["size_bytes"] for p in actual.values()),
    "Index mechanical totals",
)
refs["output_index"] = ref(idxpath)
checks["all_output_index_hashes_and_exact_population"] = True
serverpath = B / "server-index.json"
server = read(serverpath)
need(
    server["all_requests_sealed"] is True
    and server["pending_request_ids"] == server["errors"] == []
    and server["local_service_thread_drained"] is True,
    "Sealed server journals",
)
request_ids = []
terminal_responses = {}
for row in server["requests"]:
    path = B / "server" / row["path"]
    need(
        path.parent == B / "server" and path.name == row["request_id"] + ".jsonl",
        "Server journal identity",
    )
    need(pin(path) == {"sha256": row["sha256"], "size_bytes": row["bytes"]}, "Server journal hash")
    lines = [json.loads(v) for v in path.read_bytes().splitlines()]
    need(
        row["sealed"] is True and row["write_failed"] is False and row["created"] is True,
        "Journal sealed/write result",
    )
    need(
        len(lines) == row["events"] and [v["sequence"] for v in lines] == list(range(len(lines))),
        "Journal sequence/events",
    )
    need(all(v["request_id"] == row["request_id"] for v in lines), "Journal request identity")
    need(
        lines[-1]["sequence"] == row["last_sequence"] and lines[-1]["event"] == row["last_event"],
        "Journal final event",
    )
    request_ids.append(row["request_id"])
    terminal_responses[row["request_id"]] = lines[-1]["data"].get("response")
need(len(request_ids) == len(set(request_ids)), "Unique server request IDs")
need(
    {p.name for p in (B / "server").glob("*.jsonl")} == {r["path"] for r in server["requests"]},
    "Exact server journal population",
)
refs["server_index"] = ref(serverpath)
checks["sealed_server_journal_hashes_sequences_and_population"] = True
http_paths = list((B / "http").glob("*/exchange.json"))
http_status = collections.Counter()
http_kinds = collections.Counter()
joined_ids = []
for path in http_paths:
    row = read(path)
    folder = path.parent
    need(
        pin(folder / "request.bin") == row["request"]
        and pin(folder / "response.bin") == row["response"],
        "HTTP request/response pins",
    )
    selected_id = row["api_request_id"] if row["path"] == "/v1/systemone" else row["request_id"]
    need(selected_id in request_ids, "HTTP exchange server journal join")
    need(
        json.loads((folder / "response.bin").read_bytes()) == terminal_responses[selected_id],
        "HTTP/server returned semantic body join",
    )
    joined_ids.append(selected_id)
    http_status[str(row["status"])] += 1
    http_kinds[row["path"]] += 1
need(
    len(http_paths) == len(request_ids) and set(joined_ids) == set(request_ids),
    "Exact HTTP/server request population join",
)
checks["HTTP_body_hashes_and_server_journal_join"] = True
need(
    external["plan_sha256"] == PLAN_SHA and external["invocation_id"] == INVOCATION,
    "External selected plan/Invocation",
)
need(
    external["reason"] == "terminal"
    and external["unit_readback_verified"] is True
    and external["primary_error"] is None,
    "External terminal success",
)
need(
    external["actor"] == ACTOR and external["actor_read"]["actor"] == ACTOR,
    "External actor identity",
)
need(
    external["actor_read"]["plan_sha256"] == PLAN_SHA
    and external["actor_read"]["nonce"] == plan["nonce"],
    "Actor selected plan/nonce",
)
terminal = external["terminal_before_stop"]
need(
    terminal["InvocationID"] == INVOCATION and terminal["ExecMainPID"] == str(ACTOR["pid"]),
    "Bound terminal identity",
)
need(
    all(
        terminal[k] == v
        for k, v in {
            "Result": "success",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "MainPID": "0",
        }.items()
    ),
    "Successful terminal process",
)
need(
    external["cgroup_absent_after_stop"] is True and external["after_stop"]["MainPID"] == "0",
    "External cgroup/process closure",
)
need(
    len(external["two_proc_scans"]) == 2
    and all(v["matches"] == v["unreadable"] == [] for v in external["two_proc_scans"]),
    "Two clean external proc scans",
)
need(external["changed_sources"] == [], "External source closure")
need(not (Path("/proc") / str(ACTOR["pid"])).exists(), "Independent current actor absence")
need(
    not (Path("/sys/fs/cgroup") / plan["cgroup"].lstrip("/")).exists(),
    "Independent current cgroup absence",
)
host = external["host_after"]
need(
    host["boot_id"]
    == plan["boot_id"]
    == Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    "Final same boot",
)
need(
    host["mem_total_bytes"] == external["selected_memtotal_bytes"] == plan["expected_memtotal"],
    "Final selected raw host total",
)
need(
    type(host["mem_available_bytes"]) is int
    and 0 < host["mem_available_bytes"] <= host["mem_total_bytes"],
    "Positive final host memory",
)
need(
    host["comfy_state"] in {"http_idle", "http_busy", "absent_confirmed"}
    and all(type(host[k]) is int and host[k] >= 0 for k in ["comfy_running", "comfy_pending"]),
    "Known final Comfy state",
)
checks["external_terminal_identity_unit_actor_cgroup_two_scans_and_host"] = True
source_bytes = 0
small_source_count = 0
large_payload_refs = []
for row in plan["sources"]:
    path = R / row["path"]
    if row["size_bytes"] > 64 * 1024**2:
        need(path.stat().st_size == row["size_bytes"], "Large payload selected size")
        large_payload_refs.append(row)
    else:
        need(
            pin(path) == {"sha256": row["sha256"], "size_bytes": row["size_bytes"]},
            "Exact small selected source/artifact hash " + row["path"],
        )
        small_source_count += 1
    source_bytes += row["size_bytes"]
checks["small_selected_source_artifact_full_hashes"] = True
checks["root_recorded_all785_source_closure_no_changes"] = True
aliases = external["model_cache_aliases_after"]
need(
    aliases == external["model_cache_aliases_before"] and len(aliases) == 11,
    "Recorded exact eleven before/after cache aliases",
)
need(
    small_source_count == 784 and len(large_payload_refs) == 1,
    "Exact independent784 small hashes and one declared large payload",
)
for row in aliases:
    logical = R / row["logical"]["path"]
    resolved = R / row["resolved"]["path"]
    need(logical.resolve() == resolved, "Current cache alias target")
    if row["logical"]["size_bytes"] <= 64 * 1024**2:
        need(
            pin(logical) == {k: row["logical"][k] for k in ["sha256", "size_bytes"]},
            "Current logical cache payload pin",
        )
        need(
            pin(resolved) == {k: row["resolved"][k] for k in ["sha256", "size_bytes"]},
            "Current resolved cache payload pin",
        )
    else:
        need(
            logical.stat().st_size
            == resolved.stat().st_size
            == row["logical"]["size_bytes"]
            == row["resolved"]["size_bytes"],
            "Large current payload size",
        )
    for link in row["links"]:
        need(os.readlink(R / link["path"]) == link["link_text"], "Current alias chain")
    s = resolved.stat()
    need(
        row["stat"]
        == {
            "device": s.st_dev,
            "inode": s.st_ino,
            "size_bytes": s.st_size,
            "mtime_ns": s.st_mtime_ns,
            "ctime_ns": s.st_ctime_ns,
        },
        "Current alias stat",
    )
checks["cache_alias_before_after_current_targets_stats_and_small_payloads"] = True
aggregate = collections.Counter()
for group in groups:
    aggregate.update(group["aggregate"])
raw_parent = R / "artifacts/jev-runtime/native-v2-browser-execution-20261022-03"
raw_audit_path = raw_parent / "independent-audit-raw03.json"
need(
    pin(raw_audit_path)["sha256"]
    == "0edb616db1f8774b5353dd2114667f1d693189a8fadda6284f7ee8aeb590a6dd",
    "Externally selected prior independent audit",
)
raw_audit = read(raw_audit_path)
need(
    raw_audit["evaluation_complete"] is True and raw_audit["fixed_denominator"] == 10,
    "Complete prior fixed ten",
)
raw_plan_path = raw_parent / "raw-03/selected-plan.json"
need(
    pin(raw_plan_path)["sha256"]
    == "e6aab745961255af77258d3cf6bcbbb516d3c8d9bc3507ab327fd99774d187d0",
    "Externally selected prior raw plan",
)
raw_plan = read(raw_plan_path)
raw_rows = []
for group in raw_audit["groups"]:
    source = R / group["summary"]["path"]
    need(
        pin(source) == {k: group["summary"][k] for k in ["sha256", "size_bytes"]},
        "Prior audited summary exact bytes",
    )
    summary = read(source)
    for row in summary["outcomes"]:
        judge = public if group["group"] == "known4" else heldout
        replay = judge.validate_public_web_observation(
            row["task_id"], row["verification"]["observed"]
        )
        need(replay == row["verification"], "Prior recorded DOM replay")
        need(
            row["success"]
            == bool(replay["success"] and row["status"] == "done" and row["error"] is None),
            "Prior strict success",
        )
        raw_rows.append(row)
need([row["case_id"] for row in raw_rows] == expected_ids, "Paired identical ordered fixed-ten IDs")
need(
    sum(row["success"] for row in raw_rows) == raw_audit["aggregate"]["successes"] == 2,
    "Prior aggregate success equality",
)


def compact_outcome(row):
    return {
        k: row[k]
        for k in [
            "success",
            "status",
            "goal_satisfied",
            "premature_done",
            "policy_calls",
            "text_calls",
        ]
    } | {"execution_error": row["error"] is not None}


paired_rows = [
    {
        "task_id": old["task_id"],
        "attempt_id": old["case_id"],
        "raw03": compact_outcome(old),
        "v2_01": compact_outcome(new),
    }
    for old, new in zip(raw_rows, all_rows)
]
need(
    all(old["task_id"] == new["task_id"] for old, new in zip(raw_rows, all_rows)),
    "Paired exact task IDs",
)
gains = [r["task_id"] for r in paired_rows if not r["raw03"]["success"] and r["v2_01"]["success"]]
losses = [r["task_id"] for r in paired_rows if r["raw03"]["success"] and not r["v2_01"]["success"]]
comparison = {
    "fixed_denominator": 10,
    "raw03_successes": sum(r["success"] for r in raw_rows),
    "v2_01_successes": sum(r["success"] for r in all_rows),
    "gains": gains,
    "losses": losses,
    "gains_count": len(gains),
    "losses_count": len(losses),
    "unchanged_success_task_ids": [
        r["task_id"] for r in paired_rows if r["raw03"]["success"] and r["v2_01"]["success"]
    ],
    "unchanged_failure_task_ids": [
        r["task_id"] for r in paired_rows if not r["raw03"]["success"] and not r["v2_01"]["success"]
    ],
    "task_outcomes": paired_rows,
    "raw03_profile": {
        k: raw_plan.get(k) for k in ["profile", "prompt_format", "adapter", "adapter_manifest"]
    },
    "v2_01_profile": {
        k: plan.get(k) for k in ["profile", "prompt_format", "adapter", "adapter_manifest"]
    },
    "scope": "Descriptive paired outcomes only. Adapter and observation serialization both changed; no controlled attribution and no current cross-model win claim.",
    "raw03_audit": ref(raw_audit_path),
    "raw03_selected_plan": ref(raw_plan_path),
}
checks["paired_exact_ten_prior_pins_DOM_and_outcomes"] = True
audit = {
    "schema_id": "jev-native-v2-independent-readonly-evidence-audit-v1",
    "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "complete_evaluation_external_closure_verified",
    "evaluation_complete": True,
    "fixed_denominator": 10,
    "aggregate": dict(aggregate),
    "success_rate": aggregate["successes"] / 10,
    "groups": groups,
    "expected_attempt_id_count": len(expected_ids),
    "ordered_expected_attempt_ids_sha256": hashlib.sha256(
        json.dumps(expected_ids, separators=(",", ":")).encode()
    ).hexdigest(),
    "checks": checks,
    "output_inventory": {"files": len(actual), "bytes": index["total_bytes"]},
    "server_journals": {
        "count": len(request_ids),
        "all_sealed": True,
        "all_hashes_sequences_verified": True,
    },
    "HTTP_exchanges": {
        "count": len(http_paths),
        "statuses": dict(http_status),
        "kinds": dict(http_kinds),
        "scope": "Stored server publication/body hashes and sealed journal join only; server evidence alone does not prove client response delivery.",
    },
    "selected_source_artifacts": {
        "count": 785,
        "logical_bytes": source_bytes,
        "independent_small_file_full_sha256_count": small_source_count,
        "large_payload_full_sha256_independently_verified": False,
        "large_payload_refs": large_payload_refs,
        "large_payload_scope": "Root owner records full785 unchanged source/artifact refs and before/after payload alias equality. Independent audit hashes784 small files and rechecks current exact alias chains, resolved targets, weight sizes and stat identities. No independent full23.9GB base-weight payload rehash is attempted or credited.",
    },
    "external_closure": {
        "unit": plan["unit"],
        "invocation_id": INVOCATION,
        "actor": ACTOR,
        "terminal_success": True,
        "two_clean_proc_scans": True,
        "independent_actor_and_cgroup_currently_absent": True,
        "cache_alias_count": len(aliases),
        "final_host_same_boot_total_known": True,
    },
    "refs": refs,
    "paired_raw03_comparison": comparison,
    "comparison_scope": "Paired fixed-ten comparison with raw03 changes both adapter and serialized observation format. Observed differences do not isolate either cause and do not establish cross-model superiority or Decision3 performance.",
    "audit_execution": {
        "read_only": True,
        "pinned_judges_compiled_from_verified_source": True,
        "models_GPU_browser_units_or_Comfy_mutation": False,
        "raw_DOM_requests_answers_or_questions_published": False,
        "elapsed_seconds": time.monotonic() - started,
        "auditor": {
            "path": str(Path(__file__).relative_to(R)),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "size_bytes": Path(__file__).stat().st_size,
        },
    },
}
path = P / "independent-audit-v2-01.json"
need(not path.exists(), "Fresh audit output")
with path.open("x") as stream:
    json.dump(audit, stream, ensure_ascii=False, allow_nan=False, indent=2)
    stream.write("\n")
print(
    json.dumps(
        {
            "status": audit["status"],
            "audit": ref(path),
            "aggregate": dict(aggregate),
            "checks": checks,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
)
