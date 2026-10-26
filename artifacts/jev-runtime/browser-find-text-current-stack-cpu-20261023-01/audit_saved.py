"""Read saved self-generated evidence; only one exact oracle failure may replay."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
MAX_BYTES = 8 * 1024**2
KNOWN_ENTRY_SHA = "98c4ccf631b070b33d63037ce222fd824611d7c632790ff835c8a428ff36907d"
KNOWN_RECEIPT_SHA = "2d2153b752f06c4f7089ca4ac31b2eb367d28c9f3a5c0e59fe85e7ce307e006a"
KNOWN_MANIFEST_SHA = "b0044f72780b14ea9077fe45dd0f6627931fa0dce2499e9aeabd32f89df18b2a"
KNOWN_ERROR = {
    "type": "ValueError",
    "message": "Fake legal DONE follows actual successful FIND scroll",
    "notes": [],
}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def ordinary(path):
    path = Path(path).absolute()
    need(
        path.is_relative_to(ROOT)
        and ".." not in path.parts
        and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary confined saved path",
    )
    return path


def blob(path):
    with ordinary(path).open("rb") as stream:
        value = stream.read(MAX_BYTES + 1)
    need(len(value) <= MAX_BYTES, "Bounded saved evidence")
    return value


def load(path):
    return json.loads(blob(path))


def pin(path):
    value = blob(path)
    return {"sha256": sha(value), "size_bytes": len(value)}


def selected_path(row):
    relative = Path(row["path"])
    need(
        not relative.is_absolute() and ".." not in relative.parts,
        "Repo-relative selected saved reference",
    )
    path = ordinary(ROOT / relative)
    need(
        pin(path) == {k: row[k] for k in ("sha256", "size_bytes")}, "Exact selected saved reference"
    )
    return path


def strict_helper_context(request):
    source_path = ROOT / "scripts/jev_qualify_public_field_syntax_bf16.py"
    source = blob(source_path)
    need(
        sha(source) == "cbfb7bdd8301174a97ac18389a998ff3256ac32d6f7a770978f2f9b722c3b43b",
        "Exact original strict helper parser",
    )
    names = {"require", "canonical", "strict_json", "exact", "keys", "parse_helper_request"}
    nodes = [
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    need({n.name for n in nodes} == names, "Exact original parser definitions")
    namespace = {"json": json, "math": math, "IDENTITY": "generated-actions-none"}
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec", dont_inherit=True
        ),
        namespace,
    )
    questions = blob(ROOT / "artifacts/jev-runtime/upstream/jev_ultrafast/questions.py")
    need(
        sha(questions) == "66a7054b85040b949e1757311257ea0c5a843c635f9a40c99f6d15ad660fb422",
        "Exact canonical TEXT_VALUE source",
    )
    system = next(
        n.value.value
        for n in ast.parse(questions).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "TEXT_VALUE" for t in n.targets)
    )
    return namespace["parse_helper_request"](request, system_prompt=system)


def known_oracle_gate(output, receipt):
    """No general failed-receipt acceptance; authenticate the exact known47 files."""
    need(output == HERE / "actual-02", "Known oracle replay is restricted to actual02")
    need(pin(output / "receipt.json")["sha256"] == KNOWN_RECEIPT_SHA, "Exact known failed receipt")
    need(
        receipt["completed"] is False
        and receipt["error"] == KNOWN_ERROR
        and receipt["reviewed_source_sha256"] == KNOWN_ENTRY_SHA,
        "Only exact98c known DONE-oracle ValueError",
    )
    need(
        not receipt.get("persistence_errors")
        and not receipt.get("source_recheck_error")
        and not receipt.get("cleanup_errors"),
        "No other error may be admitted as oracle failure",
    )
    manifest_path = HERE / "attempt-02-immutable-files.json"
    need(
        pin(manifest_path)["sha256"] == KNOWN_MANIFEST_SHA,
        "Independently frozen known-attempt inventory",
    )
    manifest = load(manifest_path)
    need(
        manifest["schema_id"] == "find-current-stack-cpu-exact-oracle-attempt02-v1"
        and len(manifest["files"]) == 47,
        "Exact known original47 saved files",
    )
    expected = set()
    for row in manifest["files"]:
        path = selected_path(row)
        need(path.is_relative_to(output), "Known inventory confined to original actual02")
        expected.add(path.relative_to(output).as_posix())
    observed = set()
    for path in output.rglob("*"):
        need(not path.is_symlink(), "Ordinary original saved population")
        if path.is_file():
            observed.add(path.relative_to(output).as_posix())
    need(observed == expected, "Original known attempt population unchanged")
    archive = selected_path(manifest["actor_entry_archive"])
    need(
        archive == HERE / "run_fixture-attempt-02.py"
        and pin(archive) == {"sha256": KNOWN_ENTRY_SHA, "size_bytes": 54965},
        "Exact98c historical entry bytes, never executed from archive",
    )
    need(
        selected_path(manifest["actor_receipt"]) == output / "receipt.json",
        "Manifest binds exact original failed receipt",
    )
    selected_path(manifest["root_source_review"])
    return archive


def audit(output, *, replay_known_oracle=False):
    output = ordinary(output)
    need(output.parent == HERE, "Owned saved attempt only")
    receipt, ledger = load(output / "receipt.json"), load(output / "join-ledger.json")
    historical_entry = None
    if replay_known_oracle:
        historical_entry = known_oracle_gate(output, receipt)
    else:
        need(
            receipt["completed"] is True,
            "Failed actor receipt requires the exact independently selected oracle-replay gate",
        )
    need(receipt["fixture_only"] is True and receipt["quality_score"] is None, "Fixture scope only")
    need(
        receipt["real_model_calls"] == receipt["comfy_calls"] == 0
        and receipt["true_torch_imported"] is False
        and receipt["whole_tree_exit_claimed"] is False,
        "No model/Comfy/universal containment claim",
    )
    need(
        receipt["all_selected_sources_unchanged"]
        and receipt["environment_restored"]
        and receipt["browser_close_returned"]
        and receipt["closed_observed_browser_processes"],
        "Original source/environment/browser closure",
    )
    need(
        receipt["all_owned_server_threads_sockets_closed"]
        and not receipt["owned_threads_remaining"]
        and len(receipt["servers"]) == 3,
        "All three own accept threads and listeners closed",
    )
    servers = {}
    for row in receipt["servers"]:
        need(
            row["host"] == "127.0.0.1"
            and row["thread_joined"]
            and row["socket_fd_after_close"] == -1
            and row["after_close_probe"]["refused"]
            and not row.get("errors"),
            "Original loopback accept-thread/socket cleanup",
        )
        servers[row["name"]] = row["port"]
    need(
        set(servers) == {"actual-current-policy-asgi", "document", "strict-text-helper"},
        "Exact three own services",
    )
    policy_endpoint = f"http://127.0.0.1:{servers['actual-current-policy-asgi']}/v1/systemone"
    text_endpoint = f"http://127.0.0.1:{servers['strict-text-helper']}/chat/completions"
    document_url = f"http://127.0.0.1:{servers['document']}/owned"
    observed, closure = receipt["observed_browser_processes"], receipt["process_closure"]
    need(
        bool(observed)
        and len(observed) == len(closure)
        and len({(r["pid"], r["start_ticks"]) for r in observed}) == len(observed),
        "Exact nonempty observed process cohort",
    )
    for original, row in zip(observed, closure, strict=True):
        current = row["after"]
        absent = current is None or (
            current["pid"] == original["pid"] and current["start_ticks"] != original["start_ticks"]
        )
        need(
            row["before"] == original and row["same_process_absent"] is absent and absent,
            "Recomputed saved PID/start absence for each observed browser identity",
        )
    need(
        receipt["resources"]["memory_global_hard_cap_claimed"] is False,
        "Measured-only memory scope",
    )
    budget = receipt["client_budget_selection"]
    need(
        budget["selector_source_sha256"]
        == "f56f0800a981370f5a0e74d8722677721cb2043b03596c7779227502d4565ee1"
        and budget["canonical_cap_before_after"] == [60, 60]
        and budget["canonical_Agent_and_choose_unchanged"]
        and budget["harness_cap_override"] is False
        and budget["private_selected_cap"] == 16,
        "Official successor with unchanged canonical globals",
    )
    need(
        receipt["service_inverse"]["inverse_exact"]
        and receipt["service_inverse"]["declared_edits"] == 1,
        "Original c513 API-import inverse",
    )
    fake_final = receipt["fake_model_final"]
    need(
        fake_final
        == {
            "calls": 2,
            "contexts": ["enter", "exit", "enter", "exit"],
            "hook_count": 0,
            "all_numbers_fake": True,
        },
        "Exactly two simulated LM forwards and original N3 contexts/hooks",
    )
    document = load(output / "owned-document.json")
    before, after, final = (
        load(output / name)
        for name in ("before-state.json", "after-find-state.json", "final-state.json")
    )
    needle = document["needle"]
    need(
        needle not in before["page"]["text"]
        and needle in after["page"]["text"]
        and needle in final["page"]["text"],
        "Actual V7 visible observation transition",
    )
    need(
        final["status"] == "done"
        and final["decision"] is None
        and final["plan_index"] == 1
        and [h["operation"] for h in final["history"]] == ["FIND_TEXT"]
        and [d["operation"] for d in final["decisions"]] == ["FIND_TEXT", "DONE"]
        and len(final["text_calls"]) == 1,
        "Original DONE consumes terminal decision before history append",
    )
    need(
        after["status"] == "ready"
        and after["find_text_feedback"][-1]["status"] == "found"
        and after["find_text_feedback"][-1]["query"] == needle
        and after["find_text_feedback"][-1]["scroll_changed"] is True,
        "Actual FIND feedback",
    )
    need(
        [row["phase"] for row in ledger["viewport"]] == ["before_find", "after_find"],
        "Exact viewport phases",
    )
    view0, view1 = [r["value"] for r in ledger["viewport"]]
    need(
        view0["in_view"] is False and view1["in_view"] is True and view1["y"] > view0["y"],
        "Physical CDP scroll/visibility",
    )
    cdp = ledger["cdp"]
    need(cdp and not any("error" in r for r in cdp), "All retained actual CDP exchanges succeeded")
    for value in (view0, view1):
        need(
            any(
                r["method"] == "Runtime.evaluate"
                and r.get("response", {}).get("result", {}).get("value") == value
                for r in cdp
            ),
            "Viewport values join actual CDP response",
        )
    need(
        any(
            r["method"] == "Runtime.evaluate"
            and ".scrollIntoView(" in r["params"].get("expression", "")
            and r.get("response", {}).get("result", {}).get("value", {}).get("status") == "found"
            for r in cdp
            if isinstance(r.get("response", {}).get("result", {}).get("value"), dict)
        ),
        "Actual FIND scroll dispatch and response",
    )
    need(
        [r["kind"] for r in ledger["model_adapter"]] == ["policy", "text", "policy"],
        "Original adapter request order",
    )
    need(
        [r["url"] for r in ledger["real_http"]] == [policy_endpoint, text_endpoint, policy_endpoint]
        and all(
            r["method"] == "POST" and r["status"] == 200 and r["real_socket_transport"]
            for r in ledger["real_http"]
        ),
        "Three real owned loopback POSTs",
    )
    for row in ledger["real_http"]:
        parsed = urlsplit(row["url"])
        need(
            parsed.scheme == "http"
            and parsed.hostname == "127.0.0.1"
            and parsed.username is None
            and not parsed.query
            and not parsed.fragment,
            "Exact loopback endpoint authority",
        )
    policies = [r for r in ledger["model_adapter"] if r["kind"] == "policy"]
    joins = []
    returned_ids = [
        e["request_id"] for e in ledger["server_events"] if e["event"] == "server_returned"
    ]
    need(
        len(returned_ids) == len(policies) == 2 and len(set(returned_ids)) == 2,
        "Two independent original API request identities",
    )
    for index, record in enumerate(policies):
        directory = output / "policy-service/http" / f"{index:04d}"
        request_blob, response_blob = (
            blob(directory / "request.bin"),
            blob(directory / "response.bin"),
        )
        request, response, exchange = (
            json.loads(request_blob),
            json.loads(response_blob),
            load(directory / "exchange.json"),
        )
        http = ledger["real_http"][(0, 2)[index]]
        need(
            record["endpoint"] == policy_endpoint
            and record["request"] == request
            and record["response"] == response,
            "Client and actual ASGI body equality",
        )
        need(
            http["request_bytes_sha256"] == sha(request_blob)
            and http["response_bytes_sha256"] == sha(response_blob)
            and http["x_request_id"] == exchange["api_request_id"]
            and exchange["api_request_id"] in returned_ids,
            "Actual socket header/body/server-ID joins",
        )
        need(
            exchange["request"] == pin(directory / "request.bin")
            and exchange["response"] == pin(directory / "response.bin")
            and exchange["status"] == 200,
            "Original publication evidence hashes/status",
        )
        need(
            [a["http_status"] for a in record["transport_attempts"]] == [200]
            and record["transport_attempts"][0]["request"] == request
            and record["transport_attempts"][0]["response"] == response,
            "Original captured transport request/response",
        )
        need(
            list(request["questions"]) == record["runtime_question_ids"]
            and {k: list(v["criteria"]) for k, v in request["questions"].items()}
            == record["runtime_candidate_labels"],
            "Actual runtime head/candidate order",
        )
        operation = ("FIND_TEXT", "DONE")[index]
        need(
            response["answers"]["operation"]["choice"] == operation
            and set(response["answers"]) == {"operation"}
            and response["generation"]["decoded_action"]
            == {"operation": operation, "target": None},
            "Untargeted generated operation only; no all-head qualification",
        )
        need(
            response["generation"]["grammar"]["max_new_tokens"] == 32
            and response["generation"]["grammar"]["max_legal_actions"] == 1205,
            "Original grammar caps",
        )
        need(
            final["decisions"][index]["request"] == request
            and final["decisions"][index]["operation"] == operation,
            "Actual Agent decisions join sent policy bodies",
        )
        joins.append(
            {
                "index": index,
                "request_sha256": sha(request_blob),
                "response_sha256": sha(response_blob),
                "api_request_id": exchange["api_request_id"],
                "operation": operation,
            }
        )
    need(len(ledger["helper"]) == 1, "One strict helper request")
    helper, text_record, http = (
        ledger["helper"][0],
        ledger["model_adapter"][1],
        ledger["real_http"][1],
    )
    need(
        text_record["endpoint"] == text_endpoint
        and helper["request"] == text_record["request"]
        and helper["response"] == text_record["response"]
        and helper["schema_valid"]
        and helper["query"] == needle
        and helper["model_numbers_fake"],
        "Strict TEXT_VALUE helper body joins",
    )
    need(
        http["request_bytes_sha256"] == helper["request_bytes_sha256"]
        and json.loads(json.dumps(helper["response"]))
        == text_record["transport_attempts"][0]["response"],
        "Real helper socket and captured response",
    )
    need(
        strict_helper_context(helper["request"]) == helper["context"],
        "Original strict parser independently revalidates TEXT_VALUE request",
    )
    helper_response_bytes = json.dumps(
        helper["response"], ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode()
    need(
        http["response_bytes_sha256"] == sha(helper_response_bytes),
        "Owned helper encoded response joins actual received socket bytes",
    )
    need(
        helper["context"]["field"]["role"] == "browser_find_query"
        and helper["context"]["goal"] == document["goal"],
        "Strict helper generated goal/context",
    )
    need(
        [g["operation"] for g in ledger["fake_generations"]] == ["FIND_TEXT", "DONE"]
        and all(
            g["numerical_output_fake"]
            and not g["real_weights_loaded"]
            and all(t["accepted"] for t in g["allowed_token_checks"])
            for g in ledger["fake_generations"]
        ),
        "Simulated output consumes actual original legal trie",
    )
    need(
        not ledger["blocked_requests"] and ledger["documents"] and ledger["network"],
        "Owned document evidence",
    )
    need(
        all(
            r["path"] == "/owned" and r["sha256"] == sha(document["html_utf8"].encode())
            for r in ledger["documents"]
        ),
        "Generated HTML joins served document",
    )
    need(
        all(
            r["url"] == document_url for r in ledger["network"] if r["kind"] == "playwright-request"
        ),
        "Browser contacts only own document",
    )
    sources = load(output / "source-evidence.json")
    need(
        sources["fixture_source"]["sha256"]
        == sources["reviewed_fixture_sha256"]
        == sources["final_fixture_sha256"]
        == receipt["reviewed_source_sha256"],
        "Original reviewed self identity at entry/final closure",
    )
    historical_binding = None
    for row in [sources["fixture_source"], *sources["selected_sources"]]:
        if historical_entry is not None and row is sources["fixture_source"]:
            need(
                row["path"] == (HERE / "run_fixture.py").relative_to(ROOT).as_posix()
                and row["sha256"] == KNOWN_ENTRY_SHA
                and row["size_bytes"] == 54965,
                "Only exact98c historical self path may use archive",
            )
            need(
                pin(historical_entry) == {k: row[k] for k in ("sha256", "size_bytes")},
                "Archive equals original consumed source ref",
            )
            historical_binding = {
                "original_reference": row,
                "verified_archive": historical_entry.relative_to(ROOT).as_posix(),
            }
        else:
            selected_path(row)
    need(
        len(sources["selected_sources"]) == 48 and sources["all_selected_sources_unchanged"],
        "All48 original complete selected source pins still exact",
    )
    return {
        "schema_id": "find-current-stack-cpu-saved-audit-v2",
        "audit_passed": True,
        "functional_proof_passed": True,
        "actor_completed": receipt["completed"],
        "actor_error_preserved": receipt.get("error"),
        "mode": "KNOWN_EXACT_ORACLE_FAILURE_SAVED_REPLAY"
        if replay_known_oracle
        else "COMPLETED_ACTOR_SAVED_AUDIT",
        "original_receipt_sha256": pin(output / "receipt.json")["sha256"],
        "original_attempt_files_changed": False,
        "fixture_only": True,
        "quality_score": None,
        "policy_request_joins": joins,
        "strict_helper_join": True,
        "physical_viewport_before_after": [False, True],
        "scroll_y_before_after": [view0["y"], view1["y"]],
        "decisions": ["FIND_TEXT", "DONE"],
        "history_operations": ["FIND_TEXT"],
        "three_owned_listeners_accept_threads_closed": True,
        "observed_browser_pid_start_closure": len(closure),
        "whole_tree_claim": False,
        "all48_consumed_source_refs_rechecked": True,
        "historical_entry_binding": historical_binding,
        "observation_profile_errors_field": receipt.get(
            "observation_profile_errors", "NOT_PERSISTED_BEFORE_EXACT_ORACLE_FAILURE"
        ),
        "numerical_model_outputs": "ALL_SIMULATED_NOT_MODEL_QUALITY",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replay-known-oracle", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(audit(args.output, replay_known_oracle=args.replay_known_oracle), sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
