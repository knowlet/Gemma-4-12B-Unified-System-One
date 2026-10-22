import ast
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
P = R / "artifacts/jev-runtime/native-v2-feedback-independent-audit-20261022-01"
D = R / "artifacts/jev-runtime/browser-stale-feedback-bundle-20261022-01/v2-feedback-01"
B = D / "browser"
PLAN_SHA = "17627efbf0f1f2e61f7a233c503fd6d9b7ebae998e4bf11521590bab0bc90ea9"
INVOCATION = "0a9104b344394114b71da5f399efe6ac"
ACTOR = {"pid": 109805, "pgid": 109805, "sid": 109805, "start_ticks": 1665411}
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
need(len(plan["sources"]) == 790, "Exactly790 sources")
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
checks["root_recorded_all790_source_closure_no_changes"] = True
aliases = external["model_cache_aliases_after"]
need(
    aliases == external["model_cache_aliases_before"] and len(aliases) == 11,
    "Recorded exact eleven before/after cache aliases",
)
need(
    small_source_count == 789 and len(large_payload_refs) == 1,
    "Exact independent789 small hashes and one declared large payload",
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
raw_parent = R / "artifacts/jev-runtime/native-v2-browser-execution-20261022-04"
raw_audit_path = raw_parent / "independent-audit-v2-01.json"
need(
    pin(raw_audit_path)["sha256"]
    == "f6f15214f49cd4ec2602ba0bc265884106a605df9993065a26a365be0898747c",
    "Externally selected prior independent audit",
)
raw_audit = read(raw_audit_path)
need(
    raw_audit["evaluation_complete"] is True and raw_audit["fixed_denominator"] == 10,
    "Complete prior fixed ten",
)
raw_plan_path = raw_parent / "v2-01/selected-plan.json"
need(
    pin(raw_plan_path)["sha256"]
    == "73bff770e242b93d5b7b53d24ef933926bd63897a4707d6d2e2a0f06f4438b6d",
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
    sum(row["success"] for row in raw_rows) == raw_audit["aggregate"]["successes"] == 4,
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
        "control_v2": compact_outcome(old),
        "feedback_v2": compact_outcome(new),
    }
    for old, new in zip(raw_rows, all_rows)
]
need(
    all(old["task_id"] == new["task_id"] for old, new in zip(raw_rows, all_rows)),
    "Paired exact task IDs",
)
gains = [r["task_id"] for r in paired_rows if not r["control_v2"]["success"] and r["feedback_v2"]["success"]]
losses = [r["task_id"] for r in paired_rows if r["control_v2"]["success"] and not r["feedback_v2"]["success"]]
comparison = {
    "fixed_denominator": 10,
    "control_v2_successes": sum(r["success"] for r in raw_rows),
    "feedback_v2_successes": sum(r["success"] for r in all_rows),
    "gains": gains,
    "losses": losses,
    "gains_count": len(gains),
    "losses_count": len(losses),
    "unchanged_success_task_ids": [
        r["task_id"] for r in paired_rows if r["control_v2"]["success"] and r["feedback_v2"]["success"]
    ],
    "unchanged_failure_task_ids": [
        r["task_id"] for r in paired_rows if not r["control_v2"]["success"] and not r["feedback_v2"]["success"]
    ],
    "task_outcomes": paired_rows,
    "control_v2_profile": {
        k: raw_plan.get(k) for k in ["profile", "prompt_format", "adapter", "adapter_manifest"]
    },
    "feedback_v2_profile": {
        k: plan.get(k) for k in ["profile", "prompt_format", "adapter", "adapter_manifest"]
    },
    "scope": "Matched selected adapter, V2 observation, original NF4 helper and budgets; explicit feedback protocol only. Single paired run with live page/timing variation does not establish general causal improvement or cross-model superiority.",
    "control_v2_audit": ref(raw_audit_path),
    "control_v2_selected_plan": ref(raw_plan_path),
}
checks["paired_exact_ten_prior_pins_DOM_and_outcomes"] = True
need(plan["agent_protocol"] == receipt["agent_protocol"] == "typed-stale-action-feedback-v1",
     "Selected explicit feedback protocol")
for key in ("adapter", "trained_adapter_admission", "prompt_format", "expected_memtotal", "boot_id"):
    need(plan[key] == raw_plan[key], "Matched control selection " + key)
old_sources = {r["path"]: r for r in raw_plan["sources"]}
new_sources = {r["path"]: r for r in plan["sources"]}
need(len(old_sources) == 785 and len(new_sources) == 790,
     "Control785 and feedback790 exact populations")
need(all(new_sources.get(path) == selected for path, selected in old_sources.items()),
     "All785 control source refs unchanged")
checks["matched_control_adapter_format_host_and785_unchanged_refs"] = True


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def value_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def repeat_run(values):
    longest = current = 0
    prior = None
    for value in values:
        current = current + 1 if value == prior else 1
        longest = max(longest, current)
        prior = value
    return longest


http_semantic_bodies = collections.Counter()
for exchange in http_paths:
    event = read(exchange)
    folder = exchange.parent
    http_semantic_bodies[(value_sha(read(folder / "request.bin")),
                          value_sha(read(folder / "response.bin")), event["status"])] += 1
client_received = collections.Counter()
client_attempts = 0
feedback_rows = []
for group in ("known4", "public6"):
    bundle = B / group / "runtime-bundle"
    manifest = read(bundle / "bundle-manifest.json")
    need(manifest["agent_protocol"] == "typed-stale-action-feedback-v1",
         "Actual group feedback bundle label")
    need(pin(bundle / "action_feedback.py")["sha256"]
         == "79ccf2ccc2d88f9613c1320ee883ce9742b0a88c0535dbfae4de7c97ee428246",
         "Actual exact feedback utility")
    for name, selected in manifest["files"].items():
        need("/" not in name and pin(bundle / name) == selected,
             "Actual complete feedback bundle file hashes")
    parent_manifest = manifest["parent_v7_manifest"]
    install = b"\nimport action_feedback as feedback\nmodel_adapter = feedback.wrap_model_adapter(model_adapter)\n"
    runner = (bundle / "runner.py").read_bytes()
    need(runner.count(install) == 1, "Single exact actual runner feedback composition")
    inverse = runner.replace(install, b"", 1)
    need({"sha256": hashlib.sha256(inverse).hexdigest(), "size_bytes": len(inverse)}
         == parent_manifest["files"]["runner.py"], "Actual complete inverse V7 runner bytes")
    control_runner = raw_parent / "v2-01/browser" / group / "runtime-bundle/runner.py"
    need(inverse == control_runner.read_bytes()
         and ast.dump(ast.parse(inverse)) == ast.dump(ast.parse(control_runner.read_bytes())),
         "Actual matched control complete runner inverse bytes and AST")
    need(all(manifest["files"][name] == selected for name, selected
             in parent_manifest["files"].items() if name != "runner.py"),
         "Actual original V7 bundle files unchanged")
    refs[group + "_feedback_bundle_manifest"] = ref(bundle / "bundle-manifest.json")
    for outcome in [r for r in all_rows if r["case_id"] in {
            x["case_id"] for x in read(B / group / "results/summary.json")["outcomes"]}]:
        case = outcome["case_id"]
        path = B / group / "results/attempts" / case / "model_calls.json"
        model_calls = read(path)
        policy = [call for call in model_calls if call["kind"] == "policy"]
        need(len(policy) == outcome["policy_calls"], "Client policy count")
        for call in model_calls:
            for attempt in call.get("transport_attempts", []):
                client_attempts += 1
                need(attempt["request"] == call["request"], "Actual client attempt body")
                if type(attempt.get("http_status")) is int and "response" in attempt:
                    key = (value_sha(attempt["request"]), value_sha(attempt["response"]),
                           attempt["http_status"])
                    client_received[key] += 1
        records, requests_with_feedback, field_sizes = {}, 0, []
        semantic, without_feedback, decisions, operation_counts = [], [], [], collections.Counter()
        for call in policy:
            request = call["request"]
            state = request["state"]
            feedback = state.get("action_feedback", [])
            need(type(feedback) is list and len(feedback) <= 4, "Bounded actual feedback")
            if feedback:
                requests_with_feedback += 1
                field_sizes.append(len(feedback))
            ids = []
            for item in feedback:
                need(set(item) == {"attempt", "outcome", "phase", "operation", "target", "choice",
                                  "command_entry_page_fingerprint", "recorded_action_delta", "physical_input_outcome"},
                     "Actual typed feedback field population")
                need(type(item["attempt"]) is int and item["attempt"] > 0
                     and item["outcome"] == "StalePage" and item["phase"] in {"predict", "act"}
                     and type(item["recorded_action_delta"]) is int
                     and item["recorded_action_delta"] >= 0
                     and item["physical_input_outcome"] == "unknown",
                     "Honest actual typed rejection facts")
                for key in ("operation", "target", "choice", "command_entry_page_fingerprint"):
                    need(item[key] is None or type(item[key]) is str and len(item[key]) <= 128,
                         "Bounded non-exception feedback values")
                ids.append(item["attempt"])
                old = records.setdefault(item["attempt"], item)
                need(old == item, "Stable repeated rejection identity")
            need(ids == sorted(set(ids)), "Recent feedback attempt ordering")
            semantic.append(value_sha(request))
            stripped = {**request, "state": {k: v for k, v in state.items() if k != "action_feedback"}}
            without_feedback.append(value_sha(stripped))
            answer = call.get("response", {}).get("answers", {})
            operation = answer.get("operation", {}).get("choice")
            target_name = {"CLICK": "click_target", "TYPE_TEXT": "type_text_target",
                           "SELECT": "select_target"}.get(operation)
            decision = {"operation": operation,
                        "target": answer.get(target_name, {}).get("choice") if target_name else None}
            decisions.append(value_sha(decision))
            operation_counts[str(operation)] += 1
        old_path = raw_parent / "v2-01/browser" / group / "results/attempts" / case / "model_calls.json"
        old_policy = [c for c in read(old_path) if c["kind"] == "policy"]
        old_hashes = [value_sha(c["request"]) for c in old_policy]
        need(all("action_feedback" not in c["request"]["state"] for c in old_policy),
             "Control protocol had no rejection feedback")
        feedback_rows.append({"case_id": case, "policy_calls": len(policy),
                              "requests_with_feedback": requests_with_feedback,
                              "unique_stale_records_offered": len(records),
                              "recent_feedback_sizes": field_sizes,
                              "feedback_phases": dict(collections.Counter(x["phase"] for x in records.values())),
                              "recorded_action_deltas": dict(collections.Counter(str(x["recorded_action_delta"]) for x in records.values())),
                              "physical_input_outcome_always_unknown": True,
                              "request_sha_population": len(set(semantic)),
                              "longest_identical_request_run": repeat_run(semantic),
                              "longest_request_run_without_feedback_field": repeat_run(without_feedback),
                              "longest_same_decision_run": repeat_run(decisions),
                              "operation_counts": dict(operation_counts),
                              "control_policy_calls": len(old_policy),
                              "control_longest_identical_request_run": repeat_run(old_hashes),
                              "client_calls": ref(path), "control_client_calls": ref(old_path)})
need(all(client_received[key] <= http_semantic_bodies[key] for key in client_received),
     "Every recorded client HTTP body/status joins an actual server exchange")
need(sum(client_received.values()) == len(http_paths),
     "Complete actual client/server request response population")
checks["actual_client_response_delivery_records_join_server_bodies_and_statuses"] = True
checks["typed_actual_feedback_fields_and_original_control_absence"] = True
feedback_evidence = {"rows": feedback_rows,
                     "total_requests_with_feedback": sum(x["requests_with_feedback"] for x in feedback_rows),
                     "unique_stale_records_offered_total": sum(x["unique_stale_records_offered"] for x in feedback_rows),
                     "client_transport_attempts": client_attempts,
                     "recorded_client_responses_joined": sum(client_received.values()),
                     "scope": "Actual recorded model inputs and repeated decisions only. Feedback changes request bytes by design; byte variation alone does not prove progress. Physical input/guard branches remain unknown. Matched configuration does not control live-page variation."}
audit = {
    "schema_id": "jev-native-v2-feedback-independent-readonly-evidence-audit-v1",
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
        "count": 790,
        "logical_bytes": source_bytes,
        "independent_small_file_full_sha256_count": small_source_count,
        "large_payload_full_sha256_independently_verified": False,
        "large_payload_refs": large_payload_refs,
        "large_payload_scope": "Root owner records full790 unchanged source/artifact refs and before/after payload alias equality. Independent audit hashes789 small files and rechecks current exact alias chains, resolved targets, weight sizes and stat identities. No independent full23.9GB base-weight payload rehash is attempted or credited.",
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
    "feedback_evidence": feedback_evidence,
    "paired_control_v2_comparison": comparison,
    "comparison_scope": "Matched fixed-ten control comparison. Feedback is the declared configuration delta; live page/timing variation and one run per arm limit causal claims. No cross-model superiority claim.",
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
path = P / "independent-audit-feedback-01.json"
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
