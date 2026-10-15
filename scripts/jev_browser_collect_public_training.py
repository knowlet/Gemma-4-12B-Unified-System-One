#!/usr/bin/env python3
"""Scripted, development-only genuine public native supervision; no model score.

Default is a source-bound dry-run. --collect requires the reviewed plan and
collector hashes. Native Agent/choose/snapshot/freshness/act remain unchanged;
private routes, labels, ownership and DOM proofs never enter a native payload.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.abc
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

from jev_browser_benchmark import (  # noqa: E402 - direct-script source bootstrap
    PlaywrightCDP,
    _finish_public_browser,
    _ObservationJournal,
    load_upstream,
    upstream_manifest,
)

from s1.browser_evidence import (  # noqa: E402
    _source_paths,
    capture_runtime_sources,
    verify_runtime_sources,
)
from s1.browser_public_training import NativeWireRequest, public_training_case  # noqa: E402

SCHEMA = "jev-public-native-training-collection-v3"
PROPOSAL_SHA = "e3c2c9833709f3d5c05a320c13403f46aa8b8938acaf7e867d52470928f4bdfa"
MODEL_ALIAS = "generated-actions-none"
FORBIDDEN = {"torch", "transformers", "peft", "bitsandbytes", "tensorflow", "jax", "safetensors"}
MAX_STEPS = BOOTSTRAP_STEPS = 16
SETTLE_SECONDS = 2.0
SETTLE_INTERVAL = 0.05
SETTLE_SAMPLES = 3
PASSIVE_STATE = """(() => ({ready_state:document.readyState,
time_origin:performance.timeOrigin,url:location.href,scroll_x:scrollX,scroll_y:scrollY,
width:innerWidth,height:innerHeight}))()"""
CHROMIUM = Path(
    "/home/albert_hsieh/.cache/ms-playwright/chromium_headless_shell-1208/chrome-linux/headless_shell"
)
DEFAULT_UPSTREAM = Path.cwd() / "artifacts/jev-runtime/upstream"
PARENT_FIT = Path.cwd() / "artifacts/jev-browser/20261015-action-lora-counterfactual-fit-01"
CACHE = (
    Path.cwd()
    / "artifacts/jev-local-hf/hub/models--knowlet--Gemma-4-12B-Unified-System-One/snapshots/a66f836b56605039fe040f330180e336d19b3362"
)
ASSET_SHA = {
    "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    "tokenizer_config.json": "794a39f8330ce05020774c70c091225bc5f031b9cacf41fc20e52eb54b4b52d8",
    "chat_template.jinja": "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
    "generation_config.json": "040d0ccb9e10df02df9738424c1ce6d20d5d15b09362da922093ccd97573dbb4",
}
NO_INPUT_DONE_RULE = {
    "rule_id": "done-fresh-false-no-input-v1",
    "operation": "DONE",
    "requires": "Strict true current preproof; exactly one matching native fresh before/after pair returning False without an exception; no act; byte-identical full history; native tick returns ready/decisionNone and its own new raw re-observation",
    "continuation": "Only the next ordinary native tick after the same bounded passive settling; no Agent state assignment, custom observe, mutation retry or budget extension",
    "fit_rows": "Retained excluded-no-input request, acceptedFalse; never a positive fit row. Revalidate original journals before skipping. All other unaccepted requests refuse serialization.",
    "accounting": "Every excluded DONE remains in resolved supervised16 budget, full request/compiled/raw/private/state/guard journals and all-native-call index. Accepted fit step IDs are dense and retain actual native request/compiled sequence links.",
}


class RejectModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise RuntimeError("Public collection forbids model/framework import: " + fullname)


class EpisodeInterrupted(BaseException):
    """Retain an actually started episode while propagating its interruption."""

    def __init__(self, primary, episode, result):
        self.primary, self.episode, self.result = primary, episode, result
        super().__init__(str(primary))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def navigation_read_error(exc):
    return type(exc).__name__ == "StalePage" and str(exc) in {
        "Document changed during evaluation",
        "Document is navigating",
        "Page did not settle",
    }


def passive_settle(browser, journal, *, clock=time.monotonic, sleep=time.sleep):
    """Goal-independent read-only pause; never refreshes or assigns native state."""
    started = clock()
    previous = None
    stable = 0
    for index in range(int(SETTLE_SECONDS / SETTLE_INTERVAL) + 1):
        try:
            sample = browser.evaluate(PASSIVE_STATE)
        except Exception as exc:
            journal.append(
                {
                    "phase": "sample-error",
                    "sample_index": index,
                    "elapsed_ms": (clock() - started) * 1000,
                    "error": error(exc),
                }
            )
            if not navigation_read_error(exc):
                raise
            sample = None
        else:
            journal.append(
                {
                    "phase": "sample",
                    "sample_index": index,
                    "elapsed_ms": (clock() - started) * 1000,
                    "observed": sample,
                }
            )
            require(
                type(sample) is dict
                and set(sample)
                == {"ready_state", "time_origin", "url", "scroll_x", "scroll_y", "width", "height"},
                "Malformed passive read; no hidden native refresh permitted",
            )
        complete = isinstance(sample, dict) and sample.get("ready_state") == "complete"
        stable = stable + 1 if complete and sample == previous else int(complete)
        previous = copy.deepcopy(sample)
        if clock() - started > SETTLE_SECONDS:
            break
        if stable >= SETTLE_SAMPLES:
            journal.append(
                {
                    "phase": "settled",
                    "elapsed_ms": (clock() - started) * 1000,
                    "stable_samples": stable,
                }
            )
            if clock() - started <= SETTLE_SECONDS:
                return
            break
        remaining = SETTLE_SECONDS - (clock() - started)
        if remaining <= 0:
            break
        sleep(min(SETTLE_INTERVAL, remaining))
    journal.append(
        {"phase": "timeout", "elapsed_ms": (clock() - started) * 1000, "stable_samples": stable}
    )
    raise ValueError(
        "Passive between-tick settling exceeded2s; no mutation retry or native refresh"
    )


def record(path):
    path = Path(path)
    require(not path.is_symlink(), "Evidence symlinks are forbidden")
    blob = path.read_bytes()
    return {"sha256": sha(blob), "size_bytes": len(blob)}


def write_bytes(path, blob):
    with Path(path).open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())


def write(path, value):
    write_bytes(path, encode(value) + b"\n")


def error(exc):
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "notes": list(getattr(exc, "__notes__", ())),
    }


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.touch(exist_ok=False)
        self.rows = []

    def append(self, value):
        row = {"sequence": len(self.rows), **copy.deepcopy(value)}
        with self.path.open("ab") as stream:
            stream.write(encode(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(row)
        return row["sequence"]


class Pages:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir()

    def store(self, html):
        require(isinstance(html, str), "Private HTML read did not return text")
        blob = html.encode()
        digest = sha(blob)
        path = self.root / (digest + ".html")
        if not path.exists():
            write_bytes(path, blob)
        require(
            record(path) == {"sha256": digest, "size_bytes": len(blob)},
            "Deduplicated HTML bytes changed",
        )
        require(
            len(blob) <= 5 * 1024**2,
            "Public HTML exceeds accepted five MiB limit; preserved outside fit rows",
        )
        return {"artifact_path": "pages/" + path.name, "sha256": digest, "size_bytes": len(blob)}


def private_modules():
    from s1 import browser_public_training_evidence as evidence
    from s1 import browser_public_training_tasks as tasks_module

    return tasks_module, evidence


def source_files():
    return {"scripts/jev_browser_collect_public_training.py": Path(__file__)}


def parent_tree_metadata(root=PARENT_FIT):
    root = Path(root)
    require(root.is_dir(), "Prospective existing child-fit evidence tree is missing")
    files = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Prospective parent evidence symlink is forbidden")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.stat().st_size
    require(
        files and len(files) <= 10000 and sum(files.values()) <= 1024**3,
        "Prospective parent already exceeds unchanged evidence cap",
    )
    return {
        "root": str(root.resolve()),
        "file_count": len(files),
        "size_bytes": sum(files.values()),
        "size_inventory_sha256": sha(encode(files)),
        "scope": "File-size projection only; no parent/base/specialist tensor bytes read, no fit parent choice or full content verification. Future fitter must independently authenticate complete recursive lineage.",
    }


def collection_plan(output, upstream, collection_id):
    tasks_module, evidence = private_modules()
    tasks = tasks_module.tasks()
    require(
        len(tasks) == 12 and len({t["goal_id"] for t in tasks}) == 12,
        "Exactly twelve prospective tasks required",
    )
    require(
        re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,79}", collection_id) is not None,
        "Collection ID must be a confined unique slug",
    )
    require(
        [t["split"] for t in tasks] == ["train"] * 8 + ["development"] * 4,
        "Fixed split/order changed",
    )
    for task in tasks:
        require(sha(task["goal"].encode()) == task["goal_sha256"], "Task goal bytes changed")
        require(
            task["bootstrap_target_path"] != task["target_path"],
            "Recovery must reach a different actual document",
        )
    sources = {
        str(p.resolve()): record(p)
        for p in (Path(__file__), Path(tasks_module.__file__), Path(evidence.__file__))
    }
    assets = {
        name: {
            "sha256": sha((CACHE / name).read_bytes()),
            "size_bytes": (CACHE / name).stat().st_size,
        }
        for name in ASSET_SHA
    }
    require(
        all(v["sha256"] == ASSET_SHA[k] for k, v in assets.items()),
        "Pinned cached tokenizer metadata differs",
    )
    upstream_source = upstream_manifest(upstream)
    require(
        upstream_source["source_sha256"]["jev_ultrafast/snapshot.js"]
        == "e50473501c8fb8e70f3b21866d987393e3f2315c639d638bd477d170e81ed78d"
        and upstream_source["source_sha256"]["jev_ultrafast/agent.py"]
        == "aaf8b271cb0ee6643fe7beb525328ab80de8568c7ac044d02b9c6721161e7c8b"
        and upstream_source["source_sha256"]["jev_ultrafast/browser.py"]
        == "ee08dafa93d731bd292aa0263a578a29c8001cf22eef3c59fe44a159649155ee"
        and upstream_source["source_sha256"]["jev_ultrafast/model.py"]
        == "a85ada8458d23642c5c8a4095acf0565e6268f81dcda26f6308942b653e9c9f3"
        and upstream_source["source_sha256"]["jev_ultrafast/questions.py"]
        == "66a7054b85040b949e1757311257ea0c5a843c635f9a40c99f6d15ad660fb422",
        "Unchanged native Agent/browser/choose/questions/snapshot source required",
    )
    plan = {
        "schema_version": 1,
        "schema_id": SCHEMA,
        "scope": "Scripted development-only real public-native teacher collection; no policy model inference, score, training updates or promotion",
        "collection_id": collection_id,
        "output": str(Path(output).resolve()),
        "collector_sha256": record(__file__)["sha256"],
        "proposal_sha256": PROPOSAL_SHA,
        "private_route_hint_revision": tasks_module.PRIVATE_ROUTE_HINT_REVISION,
        "upstream": upstream_source,
        "tasks": tasks,
        "ordered_episode_ids": [
            collection_id + "-" + t["goal_id"] + "-" + kind
            for t in tasks
            for kind in ("ordinary", "recovery")
        ],
        "supervised_decisions_per_episode": MAX_STEPS,
        "recovery_bootstrap_decisions_separate": BOOTSTRAP_STEPS,
        "no_input_done_recovery": copy.deepcopy(NO_INPUT_DONE_RULE),
        "passive_between_ticks": {
            "timeout_seconds": SETTLE_SECONDS,
            "interval_seconds": SETTLE_INTERVAL,
            "identical_complete_samples": SETTLE_SAMPLES,
            "JS": PASSIVE_STATE,
            "no_goal_or_agent_state_assignment": True,
        },
        "upstream_action_budget_unchanged": 60,
        "coverage": {
            "required_per_goal": [
                "ordinary_incomplete",
                "ordinary_DONE",
                "genuine_related_wrong_document_recovery_start",
                "recovery_incomplete",
                "recovered_DONE",
            ],
            "corpus_wide_observed_only": ["correct_document_but_position_incomplete"],
            "stop": "First true fresh private proof produces DONE immediately. No forced six states, scrolling after satisfaction or minimum48/24 claim.",
            "failed_pairs": "First prospective plan is all-or-none: any missing/failed episode, incomplete pair, serialization refusal, source error or projected evidence cap refusal quarantines all24 attempts and leaves both split cases/wires empty; partial collection is not complete.",
        },
        "source_pins": sources,
        "tokenizer_metadata": assets,
        "model_alias_actual_native_env": MODEL_ALIAS,
        "browser": {
            "executable": str(CHROMIUM),
            "headless": True,
            "fresh_Chrome_context_per_episode": True,
            "initial_URL_only": "Exact declared task start URL; every later navigation/scroll is an offered guarded native Agent action",
            "policy": "Frozen harness GET/HEAD, service workers blocked, CDP Document/root commits and passive top-document HTTP statuses; audited contacts are not all prevented",
        },
        "read_schedule": "Unchanged Agent/observe/fresh/act. Goal-independent bounded passive settling occurs between ticks only, without native re-observe or Agent state assignments. Each actual choose strict pre-proof completes the preceding nonterminal action by appending a transition; immediate post-action DOM reads/errors remain diagnostic. Accepted DONE retains immediate identical strict pre/post ownership. Only the explicit no-input DONE rule may retain an excluded decision and continue to the next ordinary tick. All phase-at-entry and resolved classifications, rejected calls and raw native observations retained; no mutation retries.",
        "native_post_action_read_errors": "Only source-defined navigation StalePage reads may defer after exactly one matching real history action, accepted native act and true execution freshness. Every such error is retained; mutation/action guard failures cannot defer or retry. The separate no-input DONE exclusion never creates an accepted transition.",
        "ownership_provenance": "Every ownership journal row detaches document_commit_count/document_request_count and owned_target_ids/owned_session_ids/root_frames at binding time, so admission can rederive the latest full-document loader from the exact final-ledger prefix. Fixed expected_ownership and native payload fields remain unchanged.",
        "limits": {
            "context": 8192,
            "new_tokens": 32,
            "targets_per_head": 400,
            "HTML_bytes": 5 * 1024**2,
            "projected_future_fit_bytes": 1024**3,
            "projected_future_fit_files": 10000,
        },
        "bundles": ["train", "development"],
        "source_modules": {
            "tasks": str(Path(tasks_module.__file__).resolve()),
            "private_judge": str(Path(evidence.__file__).resolve()),
        },
        "no_models": True,
        "no_fit_integration": True,
    }
    plan["prospective_existing_parent_metadata"] = parent_tree_metadata()
    plan["runtime_source_pins"] = {
        label: record(path) for label, path in _source_paths(source_files()).items()
    }
    return plan


def ownership(browser, transport, page):
    require(
        browser.target in transport.targets and browser.session in transport.sessions,
        "Unknown owned browser target/session",
    )
    root = transport.public_root_frames.get(browser.session)
    require(root is not None, "Missing owned root frame")
    commits = [
        r
        for r in transport.public_document_commits
        if r["session"] == browser.session
        and r["frame_id"] == root
        and r["source"] == "cdp_frame_commit"
        and not r["setup_blank"]
    ]
    require(commits and commits[-1]["loader_id"], "Missing full-document loader commit")
    require(
        page["url"]
        in {
            r["url"]
            for r in transport.public_document_commits
            if r["session"] == browser.session and r["frame_id"] == root
        },
        "Native document lacks owned CDP commit",
    )
    require(
        isinstance(page.get("page_key"), list) and len(page["page_key"]) == 7,
        "Native document key unavailable",
    )
    return {
        "target_id": browser.target,
        "session_id": browser.session,
        "root_frame_id": root,
        "loader_id": commits[-1]["loader_id"],
        "native_page_key": copy.deepcopy(page["page_key"]),
        "url": page["url"],
    }


def current_links(browser, page):
    nodes = [
        a["node"] for a in page["actions"] if a.get("kind") == "click" and a.get("role") == "link"
    ]
    expression = (
        "(nodes=>nodes.map(node=>{const e=window.__jevFast?.nodes.get(node);return {node,href:e?.href||null,label:e?.innerText||null}}))("
        + json.dumps(nodes)
        + ")"
    )
    rows = browser.evaluate(expression)
    require(
        isinstance(rows, list)
        and len(rows) == len(nodes)
        and [r.get("node") for r in rows] == nodes,
        "Offered link-node metadata changed",
    )
    return rows


def _native_choice(page, model, action):
    _elements, targets, controls = model.action_space(page["actions"])
    operation = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}.get(action["kind"])
    if operation:
        matches = [key for key, item in targets[operation].items() if item["id"] == action["id"]]
        require(len(matches) == 1, "Teacher target is not uniquely offered")
        return {"operation": operation, "target": matches[0]}
    matches = [key for key, item in controls.items() if item["id"] == action["id"]]
    require(len(matches) == 1, "Teacher control is not offered")
    return {"operation": matches[0], "target": None}


def scripted_teacher(task, page, payload, model, links, verdict, observed, *, bootstrap=False):
    """Private deterministic route matching; never creates an unobserved action."""
    if verdict["success"] and not bootstrap:
        return {"operation": "DONE", "target": None}
    current = urlsplit(page["url"])
    target = task["bootstrap_target_path"] if bootstrap else task["target_path"]
    routes = task["bootstrap_route_paths"] if bootstrap else task["route_paths"]
    offered = {
        a["node"]: a
        for a in page["actions"]
        if a.get("kind") == "click" and a.get("role") == "link"
    }
    allowed = set(task["allowed_top_level_hosts"])

    def find(path, fragment=None):
        matches = []
        for link in links:
            parsed = urlsplit(link.get("href") or "")
            if (
                parsed.scheme == "https"
                and parsed.hostname in allowed
                and parsed.path.rstrip("/") == path.rstrip("/")
                and not parsed.query
                and parsed.fragment == (fragment or "")
                and not (parsed.path == current.path and parsed.fragment == current.fragment)
                and link["node"] in offered
            ):
                matches.append(offered[link["node"]])
        return matches[0] if matches else None

    action = None
    if current.path.rstrip("/") != target.rstrip("/"):
        action = find(target)
        if action is None:
            # A class page may offer a backlink to its API category. Once a
            # waypoint is reached, only later waypoints advance this route.
            normalized_routes = [path.rstrip("/") for path in routes]
            position = (
                normalized_routes.index(current.path.rstrip("/"))
                if current.path.rstrip("/") in normalized_routes
                else -1
            )
            for path in reversed(routes[position + 1 :]):
                action = find(path)
                if action is not None:
                    break
        index_fragment = task.get(
            "bootstrap_index_section_fragment" if bootstrap else "index_section_fragment"
        )
        if (
            action is None
            and task["judge_kind"] == "mdn_method"
            and current.path.rstrip("/") == "/en-US/docs/Web/API"
            and current.fragment != index_fragment
        ):
            action = find(current.path, index_fragment)
        if (
            action is None
            and current.path.rstrip("/") not in {p.rstrip("/") for p in routes}
            and current.path.rstrip("/") != urlsplit(task["start_url"]).path.rstrip("/")
        ):
            action = find(urlsplit(task["start_url"]).path)
    elif not bootstrap and task.get("section_fragment"):
        action = find(target, task["section_fragment"])
        if (
            action is not None
            and urlsplit(
                links[next(i for i, r in enumerate(links) if r["node"] == action["node"])]["href"]
            ).fragment
            == current.fragment
        ):
            action = None
    if action is None:
        # Native scrolling only. Desired geometry comes from fresh private facts.
        target_facts = observed.get("target") or {}
        point = target_facts.get("heading") or target_facts.get("signature") or {}
        rect = point.get("rect", {}) if isinstance(point, dict) else {}
        direction = (
            "SCROLL_UP"
            if current.path.rstrip("/") == target.rstrip("/") and rect.get("y", 0) < 0
            else "SCROLL_DOWN"
        )
        controls = [a for a in page["actions"] if a["id"].upper() == direction]
        require(
            len(controls) == 1,
            "No offered native route/scroll can progress; no invented search/ENTER/Back",
        )
        action = controls[0]
    selected = _native_choice(page, model, action)
    require(
        selected["operation"] in payload["questions"]["operation"]["criteria"],
        "Chosen operation was not offered",
    )
    require(
        selected["operation"] not in {"TYPE_TEXT", "SELECT", "WAIT", "BLOCKED"},
        "Initial public teacher supports observed links/scrolls/DONE only",
    )
    return selected


def teacher_response(payload, selected):
    operation = selected["operation"]
    head = {
        "CLICK": "click_target",
        "TYPE_TEXT": "type_text_target",
        "SELECT": "select_target",
    }.get(operation)
    answers = {}
    for key, choice in [("operation", operation), *([(head, selected["target"])] if head else [])]:
        labels = payload["questions"][key]["criteria"]
        require(choice in labels, "Scripted label not in actual full request")
        answers[key] = {
            "choice": choice,
            "probabilities": {label: float(label == choice) for label in labels},
            "confidence": 1.0,
        }
    return {
        "answers": answers,
        "model": "scripted-public-native-development-teacher",
        "usage": {},
        "annotation": "scripted one-hot execution signal; not classifier probability or model score",
    }


def _no_input_done_evidence(episode, row, transition):
    """Read existing journals only; a failed mutation/exception never qualifies."""
    try:
        require(
            row["selected"] == {"operation": "DONE", "target": None}
            and row["bootstrap"] is False
            and transition["error"] is None
            and transition["immediate_diagnostic_error"] is None
            and transition["execution_accepted"] is False,
            "Only no-error unexecuted supervised DONE may be excluded",
        )
        pre = row["pre"]
        physical = episode.journals["private-dom"].rows[pre["sequence"]]
        require(
            pre["verdict"]["success"] is True
            and physical["strict"] is True
            and physical["observed"] == pre["observed"]
            and physical["verdict"] == pre["verdict"]
            and physical["expected_ownership"] == pre["binding"]
            and not any(key.endswith("_error") for key in physical),
            "Excluded DONE lacks original strict true preproof",
        )
        calls = episode.journals["all-native-calls"].rows
        request = calls[row["call_sequence"]]
        returned = [
            event
            for event in calls
            if event["phase"] == "teacher-returned"
            and event["request_sequence"] == row["call_sequence"]
        ]
        resolved = [
            event
            for event in calls
            if event["phase"] == "decision-resolved"
            and event["request_sequence"] == row["call_sequence"]
        ]
        require(
            request["phase"] == "request"
            and encode(request["payload"]) == encode(row["payload"])
            and request["wire_sha256"] == sha(encode(row["payload"]))
            and len(returned) == len(resolved) == 1
            and returned[0]["selected"] == row["selected"]
            and returned[0]["response"] == teacher_response(row["payload"], row["selected"])
            and returned[0]["private_precondition_sequence"] == pre["sequence"]
            and returned[0]["raw_observation_sequence"] == row["raw_sequence"]
            and returned[0]["compiled_sequence"] == row["compiled_sequence"]
            and returned[0]["execution_raw_observation_start"]
            == row["execution_raw_observation_start"]
            and resolved[0]["resolved_phase"] == "supervised"
            and resolved[0]["bootstrap"] is False
            and episode.journals["compiled-inputs"].rows[row["compiled_sequence"]]["wire_sha256"]
            == sha(encode(row["payload"])),
            "Excluded DONE request/teacher/compiler provenance differs",
        )
        before_sequence, after_sequence = (
            transition[key] for key in ("before_state_sequence", "after_state_sequence")
        )
        states = episode.journals["states"].rows
        require(
            type(before_sequence) is int
            and before_sequence == request["native_policy_call_index"] * 2
            and after_sequence == before_sequence + 1,
            "Excluded DONE before/after tick sequence differs",
        )
        before_row, after_row = states[before_sequence], states[after_sequence]
        before, after = before_row["state"], after_row["state"]
        require(
            before_row["phase"] == "before-tick"
            and after_row["phase"] == "after-tick"
            and encode(before["history"]) == encode(after["history"])
            and before["goal"] == after["goal"] == episode.task["goal"]
            and before["text_calls"] == after["text_calls"] == []
            and after["status"] == "ready"
            and after["decision"] is None
            and len(after["decisions"]) == len(before["decisions"]) + 1
            and after["decisions"][:-1] == before["decisions"]
            and after["decisions"][-1]["choice"] == "DONE"
            and after["decisions"][-1]["operation"] == "DONE"
            and after["decisions"][-1]["target"] is None
            and after["decisions"][-1]["fingerprint"] == row["native_fingerprint"]
            and encode(after["decisions"][-1]["request"]) == encode(row["payload"]),
            "Excluded DONE changed full history or lacks native ready/re-observed state",
        )
        start, end = (
            transition[key] for key in ("execution_guard_sequence_start", "guard_sequence_end")
        )
        guards = episode.journals["guarded-actions"].rows
        require(
            type(start) is int
            and type(end) is int
            and start == row["execution_guard_sequence_start"]
            and end == start + 2
            and 0 <= start < end <= len(guards),
            "Excluded DONE guard population differs",
        )
        tick_start = transition["guard_sequence_start"]
        require(
            type(tick_start) is int
            and 0 <= tick_start <= start
            and not any(event.get("native_method") == "act" for event in guards[tick_start:end]),
            "Excluded DONE tick contained an act attempt",
        )
        guard, result = guards[start:end]
        raw_pre = episode.raw_rows[row["raw_sequence"]]["response"]["result"]["value"]
        expected_page = {**raw_pre, "fingerprint": row["native_fingerprint"]}
        require(
            raw_pre["page_key"] == pre["binding"]["native_page_key"]
            and guard["phase"] == "before"
            and guard["native_method"] == "fresh"
            and guard["sequence"] == start
            and encode(guard["arguments"]) == encode([expected_page])
            and guard["keywords"] == {}
            and guard["target"] == pre["binding"]["target_id"]
            and guard["session"] == pre["binding"]["session_id"]
            and result["phase"] == "after"
            and result["native_method"] == "fresh"
            and result["sequence"] == start + 1
            and result["before_sequence"] == start
            and result["accepted"] is False
            and result.get("result") is False
            and "error" not in result,
            "Excluded DONE is not exactly one current native fresh-False pair without act/error",
        )
        raw_start, raw_end = (
            transition[key]
            for key in (
                "execution_raw_observation_start",
                "raw_observation_count_after_native_tick",
            )
        )
        require(
            type(raw_start) is int
            and type(raw_end) is int
            and raw_start == row["execution_raw_observation_start"]
            and row["raw_sequence"] < raw_start < raw_end <= len(episode.raw_rows),
            "Excluded DONE has no new native tick re-observation",
        )
        after_page = {key: value for key, value in after["page"].items() if key != "fingerprint"}
        matches = [
            index
            for index in range(raw_start, raw_end)
            if episode.raw_rows[index]["response"].get("result", {}).get("value") == after_page
        ]
        require(matches, "Native after-tick page lacks new detached raw snapshot")
        require(
            transition["selected"] == row["selected"]
            and transition["request_sequence"] == row["call_sequence"]
            and transition["bootstrap"] is False
            and transition["postproof_deferred"] is False
            and transition["native_status"] == "ready"
            and transition["before_history_length"] == len(before["history"])
            and transition["after_history_length"] == len(after["history"])
            and transition["before_fingerprint"] == before["page"]["fingerprint"]
            and transition["after_fingerprint"] == after["page"]["fingerprint"],
            "Excluded DONE transition differs from native state/cost references",
        )
        return {
            "rule_id": NO_INPUT_DONE_RULE["rule_id"],
            "before_state_sequence": before_sequence,
            "after_state_sequence": after_sequence,
            "execution_guard_sequence_start": start,
            "guard_sequence_end": end,
            "execution_raw_observation_start": raw_start,
            "raw_observation_count_after_native_tick": raw_end,
            "reobserved_raw_sequence": matches[-1],
            "unchanged_full_history_sha256": sha(encode(before["history"])),
        }
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return None


def _require_no_input_done_exclusion(episode, row):
    """Flags alone cannot discard a failed request from supervision."""
    transition = episode.journals["transitions"].rows[row["transition_sequence"]]
    evidence = _no_input_done_evidence(episode, row, transition)
    events = [
        event
        for event in episode.journals["all-native-calls"].rows
        if event["phase"] == "excluded-no-input"
        and event["request_sequence"] == row["call_sequence"]
    ]
    require(
        evidence is not None
        and row.get("excluded_no_input") is True
        and row.get("accepted") is False
        and transition.get("phase") == "excluded-no-input-DONE"
        and transition.get("accepted") is False
        and transition.get("excluded_no_input") is True
        and transition.get("exclusion_evidence") == evidence
        and transition.get("request_sequence") == row["call_sequence"]
        and transition.get("selected") == row["selected"]
        and len(events) == 1
        and type(row.get("exclusion_event_sequence")) is int
        and 0 <= row["exclusion_event_sequence"] < len(episode.journals["all-native-calls"].rows)
        and episode.journals["all-native-calls"].rows[row["exclusion_event_sequence"]] == events[0]
        and events[0]
        == {
            "sequence": row.get("exclusion_event_sequence"),
            "phase": "excluded-no-input",
            "request_sequence": row["call_sequence"],
            "transition_sequence": row["transition_sequence"],
            "rule_id": NO_INPUT_DONE_RULE["rule_id"],
            "evidence": evidence,
        },
        "Unverified excluded-no-input DONE; rejected transitions cannot be silently skipped",
    )
    return evidence


class NativeCompiler:
    def __init__(self):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        from s1.jev_action_grammar import LegalActionTokenTrie
        from s1.jev_generated import INSTRUCTIONS, POLICY, _request

        require(
            POLICY == "greedy_legal_action_v2"
            and sha(INSTRUCTIONS.encode())
            == "63874de795580b95edbcb1f7c6a6857e86f458dabba77e33814d8f08f158d68a",
            "Native policy instructions changed",
        )
        self.instructions, self.request, self.grammar = INSTRUCTIONS, _request, LegalActionTokenTrie
        self.raw = Tokenizer.from_str((CACHE / "tokenizer.json").read_text())
        self.config = json.loads((CACHE / "tokenizer_config.json").read_bytes())
        require(
            self.raw.padding is None
            and self.raw.truncation is None
            and self.raw.encode("<turn|>", add_special_tokens=False).ids == [106],
            "Native tokenizer truncation/padding/EOS changed",
        )
        tokenizer = self.raw
        generation = json.loads((CACHE / "generation_config.json").read_bytes())
        config = self.config

        class Encoder:
            all_special_ids = [
                x["id"]
                for x in json.loads((CACHE / "tokenizer.json").read_bytes())["added_tokens"]
                if x.get("special")
            ]
            bos_token_id = generation["bos_token_id"]
            eos_token_id = tokenizer.token_to_id(config["eos_token"])
            pad_token_id = generation["pad_token_id"]

            def encode(self, text, add_special_tokens=False):
                return tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

        self.encoder = Encoder()
        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )

        def reject(message):
            raise ValueError(message)

        env.globals["raise_exception"] = reject
        self.template = env.from_string((CACHE / "chat_template.jinja").read_text())

    def compile(self, payload):
        encoded, source, labels = self.request(payload, payload["model"])
        prompt = self.template.render(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": self.instructions + "\n" + encoded}],
                }
            ],
            bos_token=self.config["bos_token"],
            add_generation_prompt=True,
            enable_thinking=False,
            tools=None,
        )
        require(
            prompt.endswith("<|turn>model\n<|channel>thought\n<channel|>"),
            "Native prompt prefix differs",
        )
        ids = self.raw.encode(prompt, add_special_tokens=False).ids
        require(len(ids) + 32 <= 8192, "Full native prompt plus32 exceeds8192; none truncated")
        grammar = self.grammar(
            labels, self.encoder, prompt_length=len(ids), eos_token_id=106, max_new_tokens=32
        )
        return {
            "wire_sha256": sha(source),
            "prompt_sha256": sha(prompt.encode()),
            "prompt_token_ids_sha256": sha(encode(ids)),
            "prompt_tokens": len(ids),
            "context": 8192,
            "max_new_tokens": 32,
            "labels": labels,
            "grammar": grammar.telemetry(),
            "no_alias_or_payload_rewrite": True,
        }


class StatusTransport(PlaywrightCDP):
    def __init__(self, browser):
        super().__init__(browser)
        self.document_responses = []

    def _response(self, response):
        try:
            request = response.request
            if request.resource_type == "document" and request.frame.parent_frame is None:
                self.document_responses.append(
                    {"url": response.url, "status": response.status, "method": request.method}
                )
        except Exception as exc:
            self.public_observer_errors.append(
                {"source": "document_response_status", "error": error(exc)}
            )

    def cdp(self, method, session_id=None, **params):
        result = super().cdp(method, session_id, **params)
        if method == "Target.createTarget":
            self.targets[result["targetId"]].on("response", self._response)
        return result


class Episode:
    def __init__(self, bundle, folder, task, kind, plan, pages, compiler, evidence):
        (
            self.bundle,
            self.folder,
            self.task,
            self.kind,
            self.plan,
            self.pages,
            self.compiler,
            self.evidence,
        ) = bundle, folder, task, kind, plan, pages, compiler, evidence
        self.id = plan["collection_id"] + "-" + task["goal_id"] + "-" + kind
        folder.mkdir()
        self.journals = {
            name: Journal(folder / (name + ".jsonl"))
            for name in (
                "all-native-calls",
                "states",
                "private-dom",
                "ownership",
                "guarded-actions",
                "transitions",
                "compiled-inputs",
                "passive-settling",
            )
        }
        self.raw = _ObservationJournal(folder / "observations.jsonl", [])
        self.raw_rows = []
        self.pending = []
        self.agent = self.model = self.transport = None
        self.bootstrap = kind == "recovery"
        self.bootstrap_reached = False
        self.awaiting_post = None
        self.need_settle = False

    def raw_sink(self, event):
        self.raw.append(event)
        self.raw_rows.append(copy.deepcopy(event))

    def raw_sequence(self, page):
        matches = [
            i
            for i, r in enumerate(self.raw_rows)
            if isinstance((v := r["response"].get("result", {}).get("value")), dict)
            and "actions" in v
            and all(page.get(k) == value for k, value in v.items())
        ]
        require(matches, "Current native state has no exact detached raw snapshot")
        return matches[-1]

    def proof(self, phase, *, strict=True):
        page = self.agent.state["page"]
        bound = observed = verdict = html_ref = None
        failures = {}
        first_error = None
        exceptions = []
        context = {
            "document_commit_count": len(self.transport.public_document_commits),
            "document_request_count": len(self.transport.public_document_requests),
            "owned_target_ids": sorted(self.transport.targets),
            "owned_session_ids": sorted(self.transport.sessions),
            "root_frames": copy.deepcopy(self.transport.public_root_frames),
        }
        try:
            bound = ownership(self.agent.browser, self.transport, page)
        except Exception as exc:
            exceptions.append(exc)
            failures["binding_error"] = error(exc)
            first_error = exc
        own_seq = self.journals["ownership"].append(
            {"phase": phase, "binding": bound, **context, **failures}
        )
        try:
            observed = self.evidence.observe_public_training(self.agent.browser, self.task)
            require(isinstance(observed, dict), "Private DOM observer returned no object")
            observed["ownership"] = copy.deepcopy(bound)
        except Exception as exc:
            exceptions.append(exc)
            failures["observation_error"] = error(exc)
            if first_error is None:
                first_error = exc
        if observed is not None:
            try:
                verdict = self.evidence.validate_public_training_observation(
                    self.task, observed, expected_ownership=bound
                )
            except Exception as exc:
                exceptions.append(exc)
                failures["validation_error"] = error(exc)
                if first_error is None:
                    first_error = exc
        try:
            html = self.agent.browser.evaluate("document.documentElement.outerHTML")
            if isinstance(html, str):
                blob = html.encode()
                html_ref = {
                    "artifact_path": "pages/" + sha(blob) + ".html",
                    "sha256": sha(blob),
                    "size_bytes": len(blob),
                }
            html_ref = self.pages.store(html)
        except Exception as exc:
            exceptions.append(exc)
            failures["HTML_error"] = error(exc)
            if first_error is None:
                first_error = exc
        scope = {
            "public_scope_violations": copy.deepcopy(self.transport.public_navigation_violations),
            "public_observer_errors": copy.deepcopy(self.transport.public_observer_errors),
            "HTTP_document_failures": [
                copy.deepcopy(row)
                for row in self.transport.document_responses
                if row["status"] >= 400
            ],
        }
        # Actual failed facts and HTML/read errors are durable BEFORE refusal.
        seq = self.journals["private-dom"].append(
            {
                "phase": phase,
                "strict": strict,
                "observed": observed,
                "verdict": verdict,
                "expected_ownership": bound,
                "HTML": html_ref,
                **scope,
                **failures,
            }
        )
        fatal_errors = [exc for exc in exceptions if not navigation_read_error(exc)]
        if first_error is not None and (strict or fatal_errors):
            raise first_error if strict else fatal_errors[0]
        require(not any(scope.values()), "Observed public scope/HTTP/observer failure")
        if strict:
            require(
                all(
                    verdict.get("checks", {}).get(key) is True
                    for key in (
                        "trusted_expected_ownership",
                        "owned_context",
                        "same_native_document",
                        "viewport",
                        "document_ready",
                        "no_visible_access_blocker",
                        "no_access_error_title",
                    )
                ),
                "Fresh private observation ownership/access/document binding failed",
            )
        return {
            "sequence": seq,
            "ownership_sequence": own_seq,
            "observed": observed,
            "verdict": verdict,
            "binding": bound,
            "errors": failures,
        }

    def install_guards(self):
        browser = self.agent.browser
        for name in ("fresh", "act"):
            original = getattr(browser, name)

            def wrapped(*args, _name=name, _original=original, **kwargs):
                seq = self.journals["guarded-actions"].append(
                    {
                        "phase": "before",
                        "native_method": _name,
                        "target": browser.target,
                        "session": browser.session,
                        "arguments": copy.deepcopy(args),
                        "keywords": copy.deepcopy(kwargs),
                    }
                )
                try:
                    result = _original(*args, **kwargs)
                except BaseException as exc:
                    self.journals["guarded-actions"].append(
                        {
                            "phase": "after",
                            "native_method": _name,
                            "before_sequence": seq,
                            "accepted": False,
                            "error": error(exc),
                        }
                    )
                    raise
                self.journals["guarded-actions"].append(
                    {
                        "phase": "after",
                        "native_method": _name,
                        "before_sequence": seq,
                        "accepted": result is True if _name == "fresh" else True,
                        "result": result,
                    }
                )
                return result

            setattr(browser, name, wrapped)

    def resolved_counts(self):
        rows = [
            r for r in self.journals["all-native-calls"].rows if r["phase"] == "decision-resolved"
        ]
        return {
            phase: sum(r["resolved_phase"] == phase for r in rows)
            for phase in ("bootstrap", "supervised")
        }

    def complete_previous(self, pre, raw_sequence, request_sequence):
        previous = self.awaiting_post
        if previous is None:
            return
        require(
            self.agent.state["history"] == previous["executed_history"],
            "History changed between guarded execution and the next actual native request",
        )
        seq = self.journals["transitions"].append(
            {
                **previous["execution_record"],
                "phase": "completed-at-next-native-pre",
                "accepted": True,
                "postproof_deferred": False,
                "private_postcondition_sequence": pre["sequence"],
                "postcondition_native_request_sequence": request_sequence,
                "postcondition_raw_observation_sequence": raw_sequence,
                "after_fingerprint": self.agent.state["page"]["fingerprint"],
            }
        )
        previous.update(post=pre, accepted=True, transition_sequence=seq)
        self.awaiting_post = None

    def post(self, url, _key, payload):
        index = sum(r["phase"] == "request" for r in self.journals["all-native-calls"].rows)
        page = self.agent.state["page"]
        entry = {
            "native_policy_call_index": index,
            "episode_id": self.id,
            "phase_at_entry": "bootstrap" if self.bootstrap else "supervised",
            "payload": copy.deepcopy(payload),
            "wire_sha256": sha(encode(payload)),
            "page_fingerprint": page["fingerprint"],
            "history_length": len(self.agent.state["history"]),
        }
        seq = self.journals["all-native-calls"].append({"phase": "request", **entry})
        try:
            require(
                url == "https://api.typesafe.ai/v1/systemone",
                "Only intercepted native policy URL permitted",
            )
            require(
                set(payload) == {"model", "state", "questions"}
                and payload["model"] == MODEL_ALIAS
                and set(payload["state"]) == {"page", "elements", "recent_actions"},
                "Native payload boundary changed",
            )
            require(
                payload["state"]["page"] == {k: page[k] for k in ("url", "title", "text")}
                and all(
                    q["instructions"]["goal"] == self.task["goal"]
                    for q in payload["questions"].values()
                ),
                "Native goal/page changed",
            )
            require(
                page.get("omitted_actions") == 0
                and payload["state"]["elements"] == self.model.action_space(page["actions"])[0],
                "Native elements were omitted or changed",
            )
            require(
                payload["state"]["recent_actions"]
                == [
                    {k: h.get(k) for k in ("action", "kind", "text", "page_changed")}
                    for h in self.agent.state["history"][-10:]
                ],
                "Native history differs",
            )
            NativeWireRequest(payload)
            raw_seq = self.raw_sequence(page)
            pre = self.proof("before-native-call-" + str(index))
            self.complete_previous(pre, raw_seq, seq)
            boundary = self.bootstrap and urlsplit(pre["binding"]["url"]).path.rstrip(
                "/"
            ) == self.task["bootstrap_target_path"].rstrip("/")
            if boundary:
                require(
                    not pre["verdict"]["success"],
                    "Recovery bootstrap accidentally satisfies original goal",
                )
                self.bootstrap = False
                self.bootstrap_reached = True
                self.journals["all-native-calls"].append(
                    {
                        "phase": "bootstrap-boundary",
                        "request_sequence": seq,
                        "private_precondition_sequence": pre["sequence"],
                        "raw_observation_sequence": raw_seq,
                        "owned_binding": pre["binding"],
                        "original_goal_success": False,
                    }
                )
            resolved_phase = "bootstrap" if self.bootstrap else "supervised"
            limit = BOOTSTRAP_STEPS if self.bootstrap else MAX_STEPS
            require(
                self.resolved_counts()[resolved_phase] < limit,
                "Resolved native decision phase exceeded prospective16 budget",
            )
            self.journals["all-native-calls"].append(
                {
                    "phase": "decision-resolved",
                    "request_sequence": seq,
                    "phase_at_entry": entry["phase_at_entry"],
                    "resolved_phase": resolved_phase,
                    "bootstrap": self.bootstrap,
                    "private_precondition_sequence": pre["sequence"],
                }
            )
            compiled = self.compiler.compile(payload)
            compiled_seq = self.journals["compiled-inputs"].append(compiled)
            links = current_links(self.agent.browser, page)
            selected = scripted_teacher(
                self.task,
                page,
                payload,
                self.model,
                links,
                pre["verdict"],
                pre["observed"],
                bootstrap=self.bootstrap,
            )
            require(
                (selected["operation"] == "DONE") == bool(pre["verdict"]["success"])
                if not self.bootstrap
                else selected["operation"] != "DONE",
                "First true goal proof must immediately stop; continuing labels require unsatisfied proof",
            )
            response = teacher_response(payload, selected)
            self.pending.append(
                {
                    "payload": copy.deepcopy(payload),
                    "selected": selected,
                    "valid_next_targets": {
                        selected["operation"]: (
                            [selected["target"]] if selected["target"] is not None else []
                        )
                    },
                    "raw_sequence": raw_seq,
                    "pre": pre,
                    "call_sequence": seq,
                    "compiled_sequence": compiled_seq,
                    "bootstrap": self.bootstrap,
                    "execution_guard_sequence_start": len(self.journals["guarded-actions"].rows),
                    "execution_raw_observation_start": len(self.raw_rows),
                    "native_fingerprint": page["fingerprint"],
                }
            )
            self.journals["all-native-calls"].append(
                {
                    "phase": "teacher-returned",
                    "request_sequence": seq,
                    "selected": selected,
                    "response": response,
                    "resolved_phase": resolved_phase,
                    "bootstrap": self.bootstrap,
                    "private_precondition_sequence": pre["sequence"],
                    "raw_observation_sequence": raw_seq,
                    "compiled_sequence": compiled_seq,
                    "execution_raw_observation_start": len(self.raw_rows),
                    "offered_link_metadata": links,
                }
            )
            return response
        except BaseException as exc:
            self.journals["all-native-calls"].append(
                {"phase": "teacher-rejected", "request_sequence": seq, "error": error(exc)}
            )
            raise

    def tick(self):
        if self.need_settle:
            passive_settle(self.agent.browser, self.journals["passive-settling"])
            self.need_settle = False
        before_history = len(self.agent.state["history"])
        before_calls = len(self.pending)
        guards_start = len(self.journals["guarded-actions"].rows)
        before = copy.deepcopy(self.agent.snapshot())
        before_state_sequence = self.journals["states"].append(
            {"phase": "before-tick", "state": before, "bootstrap": self.bootstrap}
        )
        caught = None
        try:
            result = self.agent.command("tick")
        except BaseException as exc:
            caught = exc
            result = self.agent.snapshot()
        after_state_sequence = self.journals["states"].append(
            {"phase": "after-tick", "state": result, "bootstrap": self.bootstrap}
        )
        if len(self.pending) != before_calls + 1:
            self.journals["transitions"].append(
                {
                    "phase": "rejected-before-execution",
                    "accepted": False,
                    "bootstrap": self.bootstrap,
                    "error": error(caught) if caught else None,
                    "before_history_length": before_history,
                    "after_history_length": len(result["history"]),
                    "guard_sequence_start": guards_start,
                    "guard_sequence_end": len(self.journals["guarded-actions"].rows),
                }
            )
            if caught is not None:
                raise caught
            raise ValueError("Tick did not yield exactly one actual native policy request")
        row = self.pending[-1]
        raw_after_native_tick = len(self.raw_rows)
        selected = row["selected"]
        done = selected["operation"] == "DONE"
        added = result["history"][before_history:]
        execution_guards = self.journals["guarded-actions"].rows[
            row["execution_guard_sequence_start"] :
        ]
        matching_fresh_guard = any(
            event.get("phase") == "after"
            and event.get("native_method") == "fresh"
            and event.get("accepted") is True
            and (prior := self.journals["guarded-actions"].rows[event["before_sequence"]])[
                "arguments"
            ][0]["page_key"]
            == row["pre"]["binding"]["native_page_key"]
            and prior["arguments"][0]["fingerprint"] == row["native_fingerprint"]
            for event in execution_guards
        )
        rejected_guard = any(
            event.get("phase") == "after" and event.get("accepted") is False
            for event in execution_guards
        )
        act_accepted = any(
            event.get("phase") == "after"
            and event.get("native_method") == "act"
            and event.get("accepted") is True
            for event in execution_guards
        )
        execution_accepted = bool(
            matching_fresh_guard
            and not rejected_guard
            and (
                result["status"] == "done"
                and not added
                and not any(event.get("native_method") == "act" for event in execution_guards)
                if done
                else len(added) == 1
                and act_accepted
                and added[0]["operation"] == selected["operation"]
                and added[0]["target"] == selected["target"]
            )
            and (caught is None or (not done and navigation_read_error(caught)))
            and result["status"] != "blocked"
        )
        # An immediate navigation/animation view is diagnostic, never a positive
        # postcondition. Only the next actual choose pre-proof can accept it.
        diagnostic = None
        diagnostic_error = None
        try:
            diagnostic = self.proof("after-tick-diagnostic", strict=False)
        except BaseException as exc:
            diagnostic_error = exc
        record = {
            "phase": "executed-awaiting-native-postproof"
            if execution_accepted and not done
            else "terminal-or-rejected",
            "request_sequence": row["call_sequence"],
            "selected": selected,
            "bootstrap": row["bootstrap"],
            "before_history_length": before_history,
            "after_history_length": len(result["history"]),
            "accepted": False,
            "execution_accepted": execution_accepted,
            "postproof_deferred": execution_accepted and not done,
            "error": error(caught) if caught else None,
            "error_scope": "post-action-native-observation"
            if caught is not None and execution_accepted
            else "execution"
            if caught is not None
            else None,
            "immediate_diagnostic_error": error(diagnostic_error) if diagnostic_error else None,
            "immediate_private_diagnostic_sequence": diagnostic["sequence"] if diagnostic else None,
            "guard_sequence_start": guards_start,
            "execution_guard_sequence_start": row["execution_guard_sequence_start"],
            "guard_sequence_end": len(self.journals["guarded-actions"].rows),
            "private_precondition_sequence": row["pre"]["sequence"],
            "before_state_sequence": before_state_sequence,
            "after_state_sequence": after_state_sequence,
            "execution_raw_observation_start": row["execution_raw_observation_start"],
            "raw_observation_count_after_native_tick": raw_after_native_tick,
            "before_fingerprint": before["page"]["fingerprint"],
            "after_fingerprint": result["page"]["fingerprint"],
            "native_status": result["status"],
        }
        if done and execution_accepted and caught is None and diagnostic_error is None:
            try:
                post = self.proof("after-DONE-strict")
                accepted = bool(
                    row["pre"]["verdict"]["success"]
                    and post["verdict"]["success"]
                    and row["pre"]["binding"] == post["binding"]
                )
                record.update(
                    accepted=accepted,
                    private_postcondition_sequence=post["sequence"],
                    postproof_deferred=False,
                )
                row["post"] = post
            except BaseException as exc:
                diagnostic_error = exc
                record["strict_DONE_error"] = error(exc)
        elif execution_accepted and diagnostic_error is None:
            row["executed_history"] = copy.deepcopy(result["history"])
            row["execution_record"] = copy.deepcopy(record)
            self.awaiting_post = row
            self.need_settle = True
        excluded = _no_input_done_evidence(self, row, record)
        if excluded is not None:
            require(self.awaiting_post is None, "Excluded DONE left an uncompleted native action")
            record.update(
                phase="excluded-no-input-DONE", excluded_no_input=True, exclusion_evidence=excluded
            )
        transition = self.journals["transitions"].append(record)
        row.update(
            transition_sequence=transition,
            accepted=record["accepted"],
            immediate_diagnostic=diagnostic,
        )
        if excluded is not None:
            event_sequence = self.journals["all-native-calls"].append(
                {
                    "phase": "excluded-no-input",
                    "request_sequence": row["call_sequence"],
                    "transition_sequence": transition,
                    "rule_id": NO_INPUT_DONE_RULE["rule_id"],
                    "evidence": excluded,
                }
            )
            row.update(excluded_no_input=True, exclusion_event_sequence=event_sequence)
            _require_no_input_done_exclusion(self, row)
            self.need_settle = True
            return row
        if diagnostic_error is not None:
            if caught is not None:
                raise caught from diagnostic_error
            raise diagnostic_error
        if not execution_accepted:
            if caught is not None:
                raise caught
            raise ValueError("Native freshness/action guard rejected; no mutation retry or fit row")
        if done:
            require(row["accepted"], "Native DONE ownership/freshness guard rejected; no fit row")
        return row

    def execute(self):
        initial = self.proof("initial")
        require(
            not initial["verdict"]["success"],
            "Initial target goal already satisfied; no invented incomplete state",
        )
        while True:
            counts = self.resolved_counts()
            if self.bootstrap and counts["bootstrap"] >= BOOTSTRAP_STEPS:
                # The URL is only a candidate for the next actual choose. The
                # strict pre-proof alone resolves the boundary and accepts data.
                require(
                    urlsplit(self.agent.state["page"]["url"]).path.rstrip("/")
                    == self.task["bootstrap_target_path"].rstrip("/"),
                    "Guarded recovery bootstrap exceeded separate16 decisions",
                )
            if not self.bootstrap:
                require(
                    counts["supervised"] < MAX_STEPS,
                    "Scripted public trajectory exceeded16 supervised decisions includingDONE",
                )
            row = self.tick()
            if row["selected"]["operation"] == "DONE" and row["accepted"] is True:
                require(self.awaiting_post is None, "DONE left an uncompleted native transition")
                counts = self.resolved_counts()
                return {
                    "success": True,
                    "native_status": self.agent.state["status"],
                    "initial_private_success": initial["verdict"]["success"],
                    "bootstrap_reached": self.bootstrap_reached,
                    "bootstrap_decisions": counts["bootstrap"],
                    "supervised_decisions": counts["supervised"],
                    "history_actions": len(self.agent.state["history"]),
                }


def run_episode(bundle, task, kind, plan, pages, compiler, evidence, playwright, upstream):
    folder = bundle / "attempts" / (task["goal_id"] + "-" + kind)
    episode = Episode(bundle, folder, task, kind, plan, pages, compiler, evidence)
    result = {
        "episode_id": episode.id,
        "goal_id": task["goal_id"],
        "split": task["split"],
        "trajectory_kind": kind,
        "success": False,
        "fit_eligible": False,
        "scope": "Scripted native development collection, not model performance",
    }
    chrome = None
    primary = None
    cleanup = []
    started = time.perf_counter()
    try:
        chrome = playwright.chromium.launch(headless=True, executable_path=str(CHROMIUM))
        episode.transport = StatusTransport(chrome)
        episode.transport.public_navigation_hosts = tuple(task["allowed_top_level_hosts"])
        episode.transport.observation_expression = (
            Path(upstream) / "jev_ultrafast/snapshot.js"
        ).read_text()
        episode.transport.observation_sink = episode.raw_sink
        with load_upstream(upstream, episode.transport) as (agent_module, model_module):
            episode.model = model_module
            old_post = model_module.post_json
            model_module.post_json = episode.post

            def no_field_model(_context):
                raise ValueError(
                    "No field-writing model or invented field values in public native teacher"
                )

            old_field = model_module.field_text
            old_agent_field = agent_module.field_text
            model_module.field_text = agent_module.field_text = no_field_model
            body_error = None
            try:
                episode.agent = agent_module.Agent(
                    task["start_url"], task["goal"], screenshots=False
                )
                require(
                    episode.agent.state["goal"] == task["goal"]
                    and not episode.agent.state["history"],
                    "Fresh exact-goal native Agent required",
                )
                episode.install_guards()
                result.update(episode.execute())
            except BaseException as exc:
                body_error = exc
            finally:
                try:
                    if episode.agent is not None:
                        write(folder / "final_state.json", episode.agent.snapshot())
                except BaseException as exc:
                    cleanup.append({"stage": "final_native_snapshot_write", "error": error(exc)})
                    if body_error is None:
                        body_error = exc
                finally:
                    model_module.post_json = old_post
                    model_module.field_text = old_field
                    agent_module.field_text = old_agent_field
            if body_error is not None:
                raise body_error
    except BaseException as exc:
        primary = exc
        result["success"] = False
        result["error"] = error(exc)
    finally:
        if episode.agent is not None:
            try:
                episode.agent.close()
            except BaseException as exc:
                cleanup.append({"stage": "native_agent_close", "error": error(exc)})
        if chrome is not None:
            try:
                _finish_public_browser(chrome, episode.transport, folder, [])
            except BaseException as exc:
                cleanup.append({"stage": "owned_chrome_or_ledger_close", "error": error(exc)})
        if episode.transport is not None:
            transport = episode.transport
            write(folder / "document-response-statuses.json", transport.document_responses)
            result.update(
                public_scope_violations=transport.public_navigation_violations,
                public_observer_errors=transport.public_observer_errors,
                HTTP_document_failures=[
                    r for r in transport.document_responses if r["status"] >= 400
                ],
                browser_protocol_calls=transport.calls,
            )
            if (
                result["public_scope_violations"]
                or result["public_observer_errors"]
                or result["HTTP_document_failures"]
            ):
                result["success"] = False
        else:
            write(
                folder / "public-request-ledger.json",
                {"not_created": True, "error": result.get("error")},
            )
        if cleanup:
            result["success"] = False
        result.update(
            cleanup_errors=cleanup,
            bootstrap_reached=episode.bootstrap_reached,
            bootstrap_decisions=episode.resolved_counts()["bootstrap"],
            supervised_decisions=episode.resolved_counts()["supervised"],
            history_actions=len(episode.agent.state.get("history", [])) if episode.agent else 0,
            native_status=episode.agent.state.get("status") if episode.agent else None,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            native_observations=episode.raw.count,
            native_observation_write_ms=episode.raw.write_ms,
            actual_native_policy_calls=sum(
                r["phase"] == "request" for r in episode.journals["all-native-calls"].rows
            ),
            bootstrap_calls=sum(
                r["phase"] == "decision-resolved" and r["bootstrap"]
                for r in episode.journals["all-native-calls"].rows
            ),
            supervised_calls=sum(
                r["phase"] == "decision-resolved" and not r["bootstrap"]
                for r in episode.journals["all-native-calls"].rows
            ),
            unresolved_native_calls=sum(
                r["phase"] == "request" for r in episode.journals["all-native-calls"].rows
            )
            - sum(
                r["phase"] == "decision-resolved" for r in episode.journals["all-native-calls"].rows
            ),
            actual_call_phase_accounting="Immutable request.phase_at_entry; decision-resolved determines bootstrap/supervised budget. Rejected calls before resolution counted separately, never imputed labels.",
            actual_native_requests_by_entry_phase={
                phase: sum(
                    r["phase"] == "request" and r["phase_at_entry"] == phase
                    for r in episode.journals["all-native-calls"].rows
                )
                for phase in ("bootstrap", "supervised")
            },
            scripted_teacher_calls=sum(
                r["phase"] == "teacher-returned" for r in episode.journals["all-native-calls"].rows
            ),
            excluded_no_input_done_count=sum(
                r["phase"] == "excluded-no-input" for r in episode.journals["all-native-calls"].rows
            ),
            excluded_no_input_native_request_sequences=[
                r["request_sequence"]
                for r in episode.journals["all-native-calls"].rows
                if r["phase"] == "excluded-no-input"
            ],
            model_inference_calls=0,
        )
        write(folder / "attempt-result.json", result)
    if primary is not None and not isinstance(primary, Exception):
        raise EpisodeInterrupted(primary, episode, result) from primary
    return episode, result


def state_role(episode, row):
    complete = row["selected"]["operation"] == "DONE"
    if complete:
        return "ordinary_complete" if episode.kind == "ordinary" else "recovered_complete"
    right_document = urlsplit(row["pre"]["binding"]["url"]).path.rstrip("/") == episode.task[
        "target_path"
    ].rstrip("/")
    if episode.kind == "ordinary":
        return "correct_document_incomplete" if right_document else "ordinary_incomplete"
    return "recovered_document_incomplete" if right_document else "recovery_incomplete"


def pair_coverage(ordinary, recovery):
    def continuing(ep):
        return any(
            not r["bootstrap"]
            and r.get("accepted") is True
            and r["selected"]["operation"] != "DONE"
            and not r["pre"]["verdict"]["success"]
            for r in ep.pending
        )

    def final(ep):
        return bool(
            ep.pending
            and ep.pending[-1].get("accepted") is True
            and not ep.pending[-1].get("excluded_no_input")
            and ep.pending[-1]["selected"]["operation"] == "DONE"
            and ep.pending[-1]["pre"]["verdict"]["success"]
            and ep.pending[-1]["post"]["verdict"]["success"]
        )

    required = {
        "ordinary_incomplete": continuing(ordinary),
        "ordinary_DONE": final(ordinary),
        "genuine_related_wrong_document_recovery_start": recovery.bootstrap_reached,
        "recovery_incomplete": continuing(recovery),
        "recovered_DONE": final(recovery),
    }
    return {
        "required": required,
        "required_complete": all(required.values()),
        "observed_position_incomplete": any(
            state_role(ep, row) in {"correct_document_incomplete", "recovered_document_incomplete"}
            for ep in (ordinary, recovery)
            for row in ep.pending
            if not row["bootstrap"] and row.get("accepted")
        ),
    }


def serialize_episode(episode, relative_folder):
    """References hash SEALED whole journals; sequence addresses the exact row."""
    files = {name: record(episode.folder / (name + ".jsonl")) for name in episode.journals}
    raw = record(episode.folder / "observations.jsonl")
    ledger = record(episode.folder / "public-request-ledger.json")

    def ref(name, sequence):
        row = (
            raw
            if name == "observations"
            else ledger
            if name == "public-request-ledger"
            else files[name]
        )
        suffix = ".json" if name == "public-request-ledger" else ".jsonl"
        return {
            "artifact_path": relative_folder + "/" + name + suffix,
            "sha256": row["sha256"],
            "sequence": sequence,
        }

    cases = []
    wires = []
    publishable = []
    exclusions = set()
    exclusion_events = [
        event
        for event in episode.journals["all-native-calls"].rows
        if event["phase"] == "excluded-no-input"
    ]
    for row in episode.pending:
        if row.get("excluded_no_input") or any(
            event["request_sequence"] == row["call_sequence"] for event in exclusion_events
        ):
            _require_no_input_done_exclusion(episode, row)
            exclusions.add(row["call_sequence"])
            continue
        require(
            row.get("accepted") is True,
            "Rejected transition cannot become a public training case or disappear",
        )
        if not row["bootstrap"]:
            publishable.append(row)
    require(
        len(exclusion_events) == len(exclusions)
        and {event["request_sequence"] for event in exclusion_events} == exclusions,
        "Orphan or duplicate excluded native request event",
    )
    for index, row in enumerate(publishable):
        require(
            len(episode.raw_rows) == episode.raw.count
            and 0 <= row["raw_sequence"] < episode.raw.count,
            "Raw observation reference is outside the sealed journal",
        )
        raw_page = episode.raw_rows[row["raw_sequence"]]["response"]["result"]["value"]
        require(
            all(
                raw_page[key] == row["payload"]["state"]["page"][key]
                for key in ("url", "title", "text")
            )
            and raw_page["page_key"] == row["pre"]["binding"]["native_page_key"]
            and raw_page["omitted_actions"] == 0,
            "Raw observation and supervised native request/ownership differ",
        )
        for point in (row["pre"], row["post"]):
            physical = episode.journals["private-dom"].rows[point["sequence"]]
            require(
                physical["observed"] == point["observed"]
                and physical["verdict"] == point["verdict"]
                and physical["expected_ownership"] == point["binding"],
                "Private proof differs from the sealed journal row",
            )
        transition = episode.journals["transitions"].rows[row["transition_sequence"]]
        require(
            transition["accepted"] is True
            and transition["selected"] == row["selected"]
            and transition["request_sequence"] == row["call_sequence"],
            "Guarded transition reference differs from the real accepted call",
        )
        actual_call = episode.journals["all-native-calls"].rows[row["call_sequence"]]
        require(
            actual_call["phase"] == "request"
            and encode(actual_call["payload"]) == encode(row["payload"])
            and episode.journals["compiled-inputs"].rows[row["compiled_sequence"]]["wire_sha256"]
            == sha(encode(row["payload"])),
            "Actual native call/compiled input differs from the row",
        )
        refs = {
            "raw_observation": ref("observations", row["raw_sequence"]),
            "private_precondition": ref("private-dom", row["pre"]["sequence"]),
            "guarded_transition": ref("transitions", row["transition_sequence"]),
            "private_postcondition": ref("private-dom", row["post"]["sequence"]),
            "ownership": ref("ownership", row["pre"]["ownership_sequence"]),
            "public_ledger": ref("public-request-ledger", 0),
        }
        case_id = episode.id + "-step-" + f"{index:03d}"
        wire_id = case_id + "-wire"
        case = public_training_case(
            row["payload"],
            case_id=case_id,
            group_id="public-native-site:" + episode.task["site_group"],
            task_id=episode.task["goal_id"],
            split=episode.task["split"],
            selected=row["selected"],
            valid_targets=row["valid_next_targets"],
            wire_row_id=wire_id,
            episode_id=episode.id,
            pair_id=episode.plan["collection_id"] + "-" + episode.task["goal_id"],
            state_role=state_role(episode, row),
            proof_refs=refs,
        )
        cases.append(case.model_dump(mode="json"))
        wires.append(
            {
                "schema_id": "jev-public-native-action-wire-v1",
                "wire_row_id": wire_id,
                "case_id": case_id,
                "episode_id": episode.id,
                "pair_id": case.pair_id,
                "native_payload": copy.deepcopy(row["payload"]),
                "native_wire_sha256": sha(encode(row["payload"])),
                "teacher_choice": row["selected"],
                "valid_next_targets": row["valid_next_targets"],
                "native_policy_request_sequence": row["call_sequence"],
                "compiled_input_sequence": row["compiled_sequence"],
                "proof_refs": refs,
                "label_source": "scripted public-native development teacher; actual guarded action IDs; no model inference",
            }
        )
    # Final serialization must not append to any referenced evidence file.
    require(
        all(record(episode.folder / (name + ".jsonl")) == pin for name, pin in files.items()),
        "Episode journal changed during serialization",
    )
    require(
        record(episode.folder / "observations.jsonl") == raw
        and record(episode.folder / "public-request-ledger.json") == ledger,
        "Raw/public evidence changed during serialization",
    )
    return cases, wires


def inventory(root):
    return {
        path.relative_to(root).as_posix(): record(path)
        for path in sorted(Path(root).rglob("*"))
        if path.is_file()
    }


def write_jsonl(path, rows):
    write_bytes(path, b"".join(encode(row) + b"\n" for row in rows))


def verify_source_copies(bundle, manifest, upstream):
    errors = []
    if not isinstance(manifest, dict):
        return {"success": False, "errors": ["Missing source capture"]}
    for label, expected in manifest["files"].items():
        try:
            actual = record(bundle / expected["artifact_path"])
            require(
                actual == {key: expected[key] for key in ("sha256", "size_bytes")},
                "Copied runtime source differs",
            )
        except Exception as exc:
            errors.append({"path": label, "error": error(exc)})
    for label, digest in upstream["source_sha256"].items():
        try:
            require(
                record(bundle / "upstream" / label)["sha256"] == digest,
                "Copied upstream source differs",
            )
        except Exception as exc:
            errors.append({"path": label, "error": error(exc)})
    return {"success": not errors, "errors": errors}


def projected_fit_bound(bundles, cases, parent, *, publication_bytes=0, publication_files=0):
    entries = {split: inventory(bundle) for split, bundle in bundles.items()}
    bundle_bytes = sum(row["size_bytes"] for data in entries.values() for row in data.values())
    bundle_files = sum(len(data) for data in entries.values())
    # Existing compiler writes prompt/source/tensor rows. This deliberately
    # overestimates three text tensors plus positions/masks at full8192.
    compiled_upper = sum(
        4 * len(encode(case["request"])) + 64 * 8192 + 4096
        for rows in cases.values()
        for case in rows
    )
    reserved_bytes = 64 * 1024**2
    # Include staged files before publishing any eligible row. The reserve also
    # covers the two collection receipts, which contain the final inventories.
    projected_bytes = (
        parent["size_bytes"] + bundle_bytes + publication_bytes + compiled_upper + reserved_bytes
    )
    projected_files = parent["file_count"] + bundle_files + publication_files + 256
    return {
        "success": projected_bytes <= 1024**3 and projected_files <= 10000,
        "parent_metadata": parent,
        "both_full_bundle_bytes": bundle_bytes,
        "both_full_bundle_files": bundle_files,
        "staged_publication_bytes": publication_bytes,
        "staged_publication_files": publication_files,
        "compiled_rows_upper_estimate_bytes": compiled_upper,
        "fit_output_metadata_and_adapter_reserve_bytes": reserved_bytes,
        "fit_output_file_reserve": 256,
        "projected_complete_tree_bytes": projected_bytes,
        "projected_complete_tree_files": projected_files,
        "unchanged_limit_bytes": 1024**3,
        "unchanged_limit_files": 10000,
        "scope": "Conservative prospective tree estimate, not fitting or a guarantee. Full parent copy and both complete split bundles, including quarantined evidence, counted; no raw evidence omitted.",
    }


def collect(output, upstream, plan):
    """All24 declared attempt records; separate self-contained split bundles."""
    output = Path(output)
    require(not output.exists(), "Refuse to overwrite any collection output")
    output.mkdir(parents=True)
    write(output / "collection_plan.json", plan)
    plan_sha = sha(encode(plan))
    exclusions = {
        "schema_version": 1,
        "purpose": "exclusion-only site-family boundary; no held-out labels used",
        "excluded_site_families": [
            "docs.python.org",
            "science.nasa.gov",
            "arxiv.org",
            "wikipedia.org",
        ],
        "split_site_groups_disjoint": True,
    }
    write(output / "exclusions.json", exclusions)
    tasks_module, evidence = private_modules()
    del tasks_module
    bundles = {split: output / split for split in ("train", "development")}
    sources = {}
    pages = {}
    episodes = {}
    outcomes = []
    coverage = []
    primary = None
    finalization_errors = []
    old = {key: os.environ.get(key) for key in ("TYPESAFE_API_KEY", "TYPESAFE_MODEL")}
    os.environ.update(
        TYPESAFE_API_KEY="public-native-scripted-teacher-only", TYPESAFE_MODEL=MODEL_ALIAS
    )
    try:
        from playwright.sync_api import sync_playwright

        for split, bundle in bundles.items():
            bundle.mkdir()
            for name in ("attempts", "episodes", "quarantined"):
                (bundle / name).mkdir()
            pages[split] = Pages(bundle / "pages")
            write(bundle / "collection_plan.json", plan)
            write(bundle / "exclusions.json", exclusions)
            sources[split] = capture_runtime_sources(bundle, additional_files=source_files())
            require(
                {
                    label: {key: item[key] for key in ("sha256", "size_bytes")}
                    for label, item in sources[split]["files"].items()
                }
                == plan["runtime_source_pins"],
                "Captured runtime source inventory differs from reviewed plan",
            )
            native_dir = bundle / "upstream"
            native_dir.mkdir()
            for relative in plan["upstream"]["source_sha256"]:
                source = Path(upstream) / relative
                destination = native_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                write_bytes(destination, source.read_bytes())
        compiler = NativeCompiler()
        with sync_playwright() as playwright:
            for task in plan["tasks"]:
                for kind in ("ordinary", "recovery"):
                    require(
                        all(
                            verify_runtime_sources(manifest, additional_files=source_files())[
                                "success"
                            ]
                            for manifest in sources.values()
                        ),
                        "Source changed before next browser",
                    )
                    require(
                        all(
                            verify_source_copies(bundles[split], manifest, plan["upstream"])[
                                "success"
                            ]
                            for split, manifest in sources.items()
                        ),
                        "Preserved source copy differs before next browser",
                    )
                    try:
                        episode, result = run_episode(
                            bundles[task["split"]],
                            task,
                            kind,
                            plan,
                            pages[task["split"]],
                            compiler,
                            evidence,
                            playwright,
                            upstream,
                        )
                    except EpisodeInterrupted as interrupted:
                        episodes[(task["goal_id"], kind)] = (
                            interrupted.episode,
                            interrupted.result,
                        )
                        outcomes.append(interrupted.result)
                        raise interrupted.primary
                    episodes[(task["goal_id"], kind)] = (episode, result)
                    outcomes.append(result)
    except BaseException as exc:
        primary = exc
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    source_checks = {
        split: verify_runtime_sources(sources.get(split), additional_files=source_files())
        for split in bundles
    }
    for split in bundles:
        copied = verify_source_copies(bundles[split], sources.get(split), plan["upstream"])
        source_checks[split]["copied_sources"] = copied
        source_checks[split]["success"] = source_checks[split]["success"] and copied["success"]
    try:
        upstream_ok = upstream_manifest(upstream) == plan["upstream"]
        require(
            parent_tree_metadata() == plan["prospective_existing_parent_metadata"],
            "Parent size inventory changed during collection",
        )
    except Exception as exc:
        upstream_ok = False
        finalization_errors.append({"stage": "upstream_or_parent_metadata", "error": error(exc)})
    split_cases = {s: [] for s in bundles}
    split_wires = {s: [] for s in bundles}
    split_episodes = {s: [] for s in bundles}
    all_calls = {s: [] for s in bundles}
    complete_pair_population = True
    projection_rows = {s: [] for s in bundles}
    for task in plan["tasks"]:
        pair = [episodes.get((task["goal_id"], kind)) for kind in ("ordinary", "recovery")]
        complete_pair_population = bool(
            complete_pair_population
            and all(pair)
            and all(item[1]["success"] for item in pair)
            and pair_coverage(pair[0][0], pair[1][0])["required_complete"]
        )
        for item in pair:
            if item:
                projection_rows[task["split"]].extend(
                    {"request": row["payload"]} for row in item[0].pending if not row["bootstrap"]
                )
    before_serialization_projection = projected_fit_bound(
        bundles, projection_rows, plan["prospective_existing_parent_metadata"]
    )
    candidate_gate = bool(
        complete_pair_population
        and primary is None
        and upstream_ok
        and all(check["success"] for check in source_checks.values())
        and before_serialization_projection["success"]
    )
    # Seal and validate the entire prospective corpus before publishing even
    # the first fit row. One late serializer failure quarantines every pair.
    staged_pairs = {}
    pair_proofs = {}
    candidate_cases = {s: [] for s in bundles}
    candidate_wires = {s: [] for s in bundles}
    for task in plan["tasks"]:
        pair = [episodes.get((task["goal_id"], kind)) for kind in ("ordinary", "recovery")]
        proof = (
            pair_coverage(pair[0][0], pair[1][0])
            if all(pair)
            else {
                "required_complete": False,
                "required": {},
                "missing_attempt": True,
                "observed_position_incomplete": False,
            }
        )
        pair_proofs[task["goal_id"]] = proof
        if candidate_gate:
            try:
                serialized = [
                    serialize_episode(item[0], "episodes/" + item[0].folder.name) for item in pair
                ]
                staged_pairs[task["goal_id"]] = serialized
                for cases, wires in serialized:
                    candidate_cases[task["split"]].extend(cases)
                    candidate_wires[task["split"]].extend(wires)
            except Exception as exc:
                proof["serialization_error"] = error(exc)
                finalization_errors.append(
                    {"stage": "serialize_pair", "goal_id": task["goal_id"], "error": error(exc)}
                )

    # Forecast every still-unwritten row/index/episode file. Existing attempt
    # journals are already included in the tree inventory and will only move.
    publication_bytes = 0
    publication_files = 2  # split collection receipts (covered by byte reserve)
    for split in bundles:
        anticipated_episodes = []
        anticipated_calls = []
        for task in plan["tasks"]:
            if task["split"] != split:
                continue
            proof = pair_proofs[task["goal_id"]]
            for kind in ("ordinary", "recovery"):
                item = episodes.get((task["goal_id"], kind))
                if item is None:
                    continue
                episode, result = item
                projected_result = copy.deepcopy(result)
                projected_result.update(
                    fit_eligible=True,
                    attempted=True,
                    artifact_path="episodes/" + episode.folder.name,
                    pair_coverage=proof,
                )
                anticipated_episodes.append(projected_result)
                publication_bytes += len(encode(projected_result)) + 1
                publication_files += 1
                pin = record(episode.folder / "all-native-calls.jsonl")
                anticipated_calls.extend(
                    {
                        "episode_id": episode.id,
                        "artifact_path": projected_result["artifact_path"]
                        + "/all-native-calls.jsonl",
                        "sha256": pin["sha256"],
                        "sequence": row["sequence"],
                        "phase": row["phase"],
                    }
                    for row in episode.journals["all-native-calls"].rows
                )
        for rows in (
            candidate_cases[split],
            candidate_wires[split],
            anticipated_episodes,
            anticipated_calls,
        ):
            publication_bytes += sum(len(encode(row)) + 1 for row in rows)
            publication_files += 1
    staged_projection = projected_fit_bound(
        bundles,
        candidate_cases,
        plan["prospective_existing_parent_metadata"],
        publication_bytes=publication_bytes,
        publication_files=publication_files,
    )
    collection_gate = bool(
        candidate_gate
        and len(staged_pairs) == len(plan["tasks"])
        and not finalization_errors
        and staged_projection["success"]
    )
    for task in plan["tasks"]:
        pair = [episodes.get((task["goal_id"], kind)) for kind in ("ordinary", "recovery")]
        proof = pair_proofs[task["goal_id"]]
        eligible = collection_gate
        serialized = staged_pairs.get(task["goal_id"], [])
        coverage.append(
            {
                "goal_id": task["goal_id"],
                "split": task["split"],
                "fit_eligible_pair": eligible,
                **proof,
            }
        )
        for index, (kind, item) in enumerate(zip(("ordinary", "recovery"), pair, strict=True)):
            if item is None:
                missing = {
                    "episode_id": plan["collection_id"] + "-" + task["goal_id"] + "-" + kind,
                    "goal_id": task["goal_id"],
                    "split": task["split"],
                    "trajectory_kind": kind,
                    "success": False,
                    "fit_eligible": False,
                    "attempted": False,
                    "error": error(primary)
                    if primary
                    else {"type": "MissingAttempt", "message": "Not reached"},
                }
                outcomes.append(missing)
                split_episodes[task["split"]].append(missing)
                continue
            episode, result = item
            destination = (
                bundles[task["split"]]
                / ("episodes" if eligible else "quarantined")
                / episode.folder.name
            )
            episode.folder.rename(destination)
            episode.folder = destination
            result.update(
                fit_eligible=eligible,
                attempted=True,
                artifact_path=destination.relative_to(bundles[task["split"]]).as_posix(),
                pair_coverage=proof,
            )
            if eligible:
                cases, wires = serialized[index]
                split_cases[task["split"]].extend(cases)
                split_wires[task["split"]].extend(wires)
            write(destination / "episode.json", result)
            split_episodes[task["split"]].append(copy.deepcopy(result))
            pin = record(destination / "all-native-calls.jsonl")
            all_calls[task["split"]].extend(
                {
                    "episode_id": episode.id,
                    "artifact_path": result["artifact_path"] + "/all-native-calls.jsonl",
                    "sha256": pin["sha256"],
                    "sequence": row["sequence"],
                    "phase": row["phase"],
                }
                for row in episode.journals["all-native-calls"].rows
            )
    for split, bundle in bundles.items():
        # Even pre-browser infrastructure failure leaves all declared population
        # records and a terminal partial receipt; it never imputes an attempt.
        bundle.mkdir(exist_ok=True)
        write_jsonl(bundle / "cases.jsonl", split_cases[split])
        write_jsonl(bundle / "wire_requests.jsonl", split_wires[split])
        write_jsonl(bundle / "episodes.jsonl", split_episodes[split])
        write_jsonl(bundle / "all_native_calls.jsonl", all_calls[split])
    projected = staged_projection
    bundle_receipts = {}
    for split, bundle in bundles.items():
        complete = (
            bool(split_episodes[split])
            and all(r["fit_eligible"] for r in split_episodes[split])
            and source_checks[split]["success"]
            and upstream_ok
            and primary is None
            and projected["success"]
            and not finalization_errors
        )
        artifacts = inventory(bundle)
        receipt = {
            "schema_version": 1,
            "schema_id": SCHEMA,
            "stage": "complete" if complete else "partial",
            "scope": "Development-only genuine public-native scripted supervision; no model score",
            "split": split,
            "collection_plan_sha256": plan_sha,
            "cases_sha256": record(bundle / "cases.jsonl")["sha256"],
            "wire_sha256": record(bundle / "wire_requests.jsonl")["sha256"],
            "episode_inventory_sha256": record(bundle / "episodes.jsonl")["sha256"],
            "teacher_source_sha256": record(__file__)["sha256"],
            "private_judge_source_sha256": record(Path(evidence.__file__))["sha256"],
            "upstream": plan["upstream"],
            "native_snapshot_js_sha256": plan["upstream"]["source_sha256"][
                "jev_ultrafast/snapshot.js"
            ],
            "collector_runtime_source_manifest": sources.get(split),
            "source_verification": source_checks[split],
            "upstream_verification": upstream_ok,
            "ordered_episode_ids": [r["episode_id"] for r in split_episodes[split]],
            "attempted_episode_count": sum(r["attempted"] for r in split_episodes[split]),
            "successful_episode_count": sum(r["success"] for r in split_episodes[split]),
            "quarantined_episode_count": sum(not r["fit_eligible"] for r in split_episodes[split]),
            "fit_eligible_row_count": len(split_cases[split]),
            "all_actual_calls_index_sha256": record(bundle / "all_native_calls.jsonl")["sha256"],
            "coverage": [r for r in coverage if r["split"] == split],
            "observed_position_incomplete": any(
                r["observed_position_incomplete"] for r in coverage if r["split"] == split
            ),
            "missing_observed_only_coverage": []
            if any(r["observed_position_incomplete"] for r in coverage if r["split"] == split)
            else ["correct_document_but_position_incomplete"],
            "no_forced_state_count": True,
            "exclusion_manifest_sha256": record(bundle / "exclusions.json")["sha256"]
            if (bundle / "exclusions.json").exists()
            else None,
            "evidence_files": artifacts,
            "error": error(primary) if primary else None,
            "finalization_errors": finalization_errors,
            "projected_future_fit": projected,
            "all24_mandatory_pair_gate": collection_gate,
            "no_input_done_recovery": copy.deepcopy(NO_INPUT_DONE_RULE),
            "excluded_no_input_done_count": sum(
                r.get("excluded_no_input_done_count", 0) for r in split_episodes[split]
            ),
            "model_calls": 0,
            "training_updates": 0,
            "task_score": None,
        }
        write(bundle / "receipt.json", receipt)
        bundle_receipts[split] = {
            "path": split + "/receipt.json",
            **record(bundle / "receipt.json"),
            "stage": receipt["stage"],
            "fit_eligible_row_count": receipt["fit_eligible_row_count"],
        }
    complete = primary is None and all(r["stage"] == "complete" for r in bundle_receipts.values())
    receipt = {
        "schema_version": 1,
        "schema_id": SCHEMA,
        "stage": "complete" if complete else "partial",
        "scope": plan["scope"],
        "collection_plan_sha256": plan_sha,
        "ordered_episode_ids": plan["ordered_episode_ids"],
        "all24_attempt_records_retained": len(outcomes) == 24,
        "outcomes": outcomes,
        "coverage": coverage,
        "bundles": bundle_receipts,
        "source_verification": source_checks,
        "upstream_verification": upstream_ok,
        "projected_future_fit": projected,
        "before_serialization_projection": before_serialization_projection,
        "all24_mandatory_pair_gate": collection_gate,
        "no_input_done_recovery": copy.deepcopy(NO_INPUT_DONE_RULE),
        "excluded_no_input_done_count": sum(
            r.get("excluded_no_input_done_count", 0) for r in outcomes
        ),
        "error": error(primary) if primary else None,
        "finalization_errors": finalization_errors,
        "model_calls": 0,
        "training_updates": 0,
        "task_score": None,
        "fresh_transfer_seed": None,
    }
    write(output / "receipt.json", receipt)
    if primary:
        raise primary
    require(
        complete,
        "Public collection partial: failures/missing pair coverage quarantined; no complete corpus claim",
    )
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--collector-sha256")
    parser.add_argument("--plan-sha256")
    args = parser.parse_args(argv)
    sys.meta_path.insert(0, RejectModels())
    plan = collection_plan(args.output_dir, args.upstream, args.collection_id)
    digest = sha(encode(plan))
    if not args.collect:
        print(
            json.dumps(
                {
                    "mode": "source_only_dryrun",
                    "plan_sha256": digest,
                    "collector_sha256": plan["collector_sha256"],
                    "plan": plan,
                    "model_calls": 0,
                    "browser_calls": 0,
                    "network_calls": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    require(
        args.collector_sha256 == plan["collector_sha256"] and args.plan_sha256 == digest,
        "Root-reviewed exact collector and prospective plan hashes required before browser",
    )
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("HF_HUB_OFFLINE") == "1"
        and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "CUDA-hidden offline model-cache environment required",
    )
    require(
        not FORBIDDEN.intersection(name.split(".")[0] for name in sys.modules),
        "Model/framework import already present",
    )
    receipt = collect(args.output_dir, args.upstream, plan)
    print(
        json.dumps(
            {
                "stage": receipt["stage"],
                "bundles": receipt["bundles"],
                "model_calls": 0,
                "task_score": None,
            }
        )
    )


if __name__ == "__main__":
    main()
