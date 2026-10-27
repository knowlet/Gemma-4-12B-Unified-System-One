"""Recompute a pinned public Heaven run, reporting aggregates only.

This reads only the selected run evidence, selected official public checkout,
repository public code and config. It never executes a model, changes evidence,
or emits individual tasks, requests, responses, labels, or answer values.
A passing audit does not convert a pending result into an accepted score.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path

import run_jevbench_public as heaven

from s1.contracts import DecisionRequest
from s1.evaluation.decision3 import native_payload, normalize_native_response

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SHA = "206ca6745290cc4bc5c577d64addad7814381cdc6d336b5b7c09cb8992af8754"


def need(condition, category):
    if not condition:
        raise ValueError(category)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def same_json(left, right):
    """Match the official sorted-key JSON while preserving JSON scalar types."""
    return heaven._bytes(left) == heaven._bytes(right)


def same_ordered(left, right):
    """Match event JSON including mapping order and scalar types."""
    return encode(left) == encode(right)


def read(root, relative):
    path = root / relative
    need(path.resolve().is_relative_to(root.resolve()) and path.is_file(), "evidence_path")
    return path.read_bytes()


def read_json(root, relative):
    return json.loads(read(root, relative), parse_constant=lambda _: need(False, "nonfinite"))


def lines(root, relative):
    return [
        json.loads(x, parse_constant=lambda _: need(False, "nonfinite"))
        for x in read(root, relative).splitlines()
    ]


def read_pinned_journal(root, plan, expected_sha256):
    """Read the plan's in-root lease journal once, bound to inode and digest."""
    transport = plan.get("transport") if type(plan) is dict else None
    need(type(transport) is dict, "journal_identity")
    name = transport.get("journal")
    stats = transport.get("stat")
    need(type(stats) is dict, "journal_identity")
    identity = stats.get("journal")
    need(
        type(name) is str
        and type(identity) is dict
        and set(identity) == {"device", "inode"}
        and all(type(value) is int and value >= 0 for value in identity.values())
        and type(expected_sha256) is str
        and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
        "journal_identity",
    )
    root = Path(root)
    path = Path(name)
    need(
        root.is_absolute()
        and ".." not in root.parts
        and path.is_absolute()
        and path.is_relative_to(root)
        and path != root
        and ".." not in path.relative_to(root).parts,
        "evidence_path",
    )
    parts = path.relative_to(root).parts
    directories = (*root.parts[1:], *parts[:-1])
    dirs = []
    fd = None
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        dirs.append(os.open("/", directory_flags))
        for part in directories:
            dirs.append(os.open(part, directory_flags, dir_fd=dirs[-1]))
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dirs[-1])

        def linked_directories():
            root_info = os.stat("/", follow_symlinks=False)
            opened_root = os.fstat(dirs[0])
            need(
                stat.S_ISDIR(root_info.st_mode)
                and (root_info.st_dev, root_info.st_ino)
                == (opened_root.st_dev, opened_root.st_ino),
                "evidence_path",
            )
            for parent, child, part in zip(dirs[:-1], dirs[1:], directories, strict=True):
                current = os.stat(part, dir_fd=parent, follow_symlinks=False)
                opened = os.fstat(child)
                need(
                    stat.S_ISDIR(current.st_mode)
                    and (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino),
                    "evidence_path",
                )

        linked_directories()
        before = os.fstat(fd)
        need(
            stat.S_ISREG(before.st_mode)
            and {"device": before.st_dev, "inode": before.st_ino} == identity
            and before.st_size <= 16 * 1024**2,
            "journal_identity",
        )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(16 * 1024**2 + 1)
        after = os.fstat(fd)
        linked_directories()
        current = os.stat(parts[-1], dir_fd=dirs[-1], follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        need(
            tuple(getattr(before, key) for key in fields)
            == tuple(getattr(after, key) for key in fields)
            == tuple(getattr(current, key) for key in fields)
            and len(raw) == before.st_size,
            "journal_identity",
        )
        need(sha(raw) == expected_sha256, "lease_digest")
        return raw
    except OSError as error:
        raise ValueError("evidence_path") from error
    finally:
        if fd is not None:
            os.close(fd)
        for directory in reversed(dirs):
            os.close(directory)


def source_check(plan):
    allowed = (ROOT / "scripts", ROOT / "src/s1", ROOT / "configs")
    matching = unavailable = outside = 0
    for name, expected in plan["source_pins"].items():
        path = Path(name)
        if not any(path.is_relative_to(prefix) for prefix in allowed):
            outside += 1
            continue
        if not path.is_file() or not any(
            path.resolve().is_relative_to(p.resolve()) for p in allowed
        ):
            unavailable += 1
            continue
        need(sha(path.read_bytes()) == expected, "public_source_pin")
        matching += 1
    for path in (
        Path(heaven.__file__).absolute(),
        heaven.CONFIG.absolute(),
        Path(sys.modules[DecisionRequest.__module__].__file__).absolute(),
        Path(sys.modules[native_payload.__module__].__file__).absolute(),
    ):
        need(str(path) in plan["source_pins"], "audit_dependency_not_pinned")
    required = (
        "jev_decision3_jevbench_heaven_owner.py",
        "jev_decision3_jevbench_heaven_owner_release_20261027.py",
        "jev_decision3_jevbench_heaven_supervisor.py",
        "jev_decision3_jevbench_heaven_supervisor_release_20261027.py",
        "jev_decision3_jevbench_generic.py",
        "jev_gpu_forward_lease.py",
        "jev_decision3_candidate_browser_child.py",
        "jev_decision3_candidate_http_owner.py",
        "jev_stage_memory_profile_v2.py",
    )
    expected_paths = [ROOT / "scripts" / name for name in required]
    expected_paths.extend((Path(plan["generic_release"]), Path(plan["transport"]["lease_release"])))
    need(
        all(
            path.is_relative_to(ROOT / "scripts") and str(path) in plan["source_pins"]
            for path in expected_paths
        ),
        "audit_dependency_not_pinned",
    )
    pairs = [
        (expected_paths[0], expected_paths[1]),
        (expected_paths[2], expected_paths[3]),
        (expected_paths[4], Path(plan["generic_release"])),
        (expected_paths[5], Path(plan["transport"]["lease_release"])),
    ]
    for held, release in pairs:
        original, activated = held.read_bytes(), release.read_bytes()
        marker = b"\nSOURCE_ONLY = True\n"
        need(
            original.count(marker) == 1
            and activated == original.replace(marker, b"\nSOURCE_ONLY = False\n", 1),
            "release_source_difference",
        )
    need(unavailable == 0 and sha(heaven.CONFIG.read_bytes()) == CONFIG_SHA, "public_source_pin")
    return {"verified_public_files": matching, "outside_scope_not_read": outside}


def audit(run_root, official_root):
    run_root, official_root = (
        Path(run_root).resolve(strict=True),
        Path(official_root).resolve(strict=True),
    )
    plan_raw = read(run_root, "plan.json")
    plan = json.loads(plan_raw)
    sources = source_check(plan)
    tasks, official, config = heaven.load_public(official_root)
    need(len(tasks) == 231, "official_population")
    scorer = "worker/staging/public231"
    manifest = read_json(run_root, scorer + "/manifest.json")
    summary = read_json(run_root, scorer + "/summary.json")
    records = lines(run_root, scorer + "/records.jsonl")
    events = lines(run_root, "worker/generic-events.jsonl")
    terminal = read_json(run_root, "worker/terminal.json")
    hello = read_json(run_root, "worker/hello.json")
    go = read_json(run_root, "worker/go.json")
    closure = read_json(run_root, "supervisor/closure.json")
    supervision = lines(run_root, "supervisor/events.jsonl")
    evaluation = terminal["evaluation"]
    journal_bytes = read_pinned_journal(run_root, plan, evaluation["lease"]["sha256"])
    journal = [
        json.loads(x, parse_constant=lambda _: need(False, "nonfinite"))
        for x in journal_bytes.splitlines()
    ]
    stage = read_json(run_root, "worker/staging/status.json")
    quarantine = read_json(run_root, scorer + "/STATUS_NOT_ACCEPTED.json")
    need(
        stage.get("score_accepted") is False and quarantine.get("score_accepted") is False,
        "staging_not_quarantined",
    )
    need(
        same_json(manifest.get("dataset"), config)
        and manifest.get("status") == "completed"
        and manifest.get("n_attempted") == manifest.get("n_valid") == 231
        and len(records) == 231
        and len(events) == 924
        and len(journal) == 462,
        "incomplete_population",
    )
    need(
        hello.get("plan_sha256")
        == go.get("plan_sha256")
        == terminal.get("plan_sha256")
        == sha(plan_raw)
        and hello.get("invocation")
        == go.get("invocation")
        == terminal.get("invocation")
        == closure.get("invocation")
        and hello.get("unit") == plan.get("unit") == closure.get("unit"),
        "handshake_binding",
    )
    need(
        terminal.get("status") == closure.get("status") == "completed_pending_external_audit"
        and type(terminal.get("final_cuda_allocated")) is int
        and terminal.get("final_cuda_allocated") == 0
        and closure.get("cgroup_closure") == "absent"
        and closure.get("launch_succeeded") is True
        and terminal.get("model_score") is None,
        "cleanup_or_supervisor",
    )
    need(
        evaluation.get("status") == "completed_pending_external_audit"
        and evaluation.get("score_accepted") is False
        and same_json(evaluation.get("official_summary"), summary)
        and evaluation.get("calls") == 231
        and evaluation.get("delta_route") == "both_bypassed",
        "terminal_evaluation",
    )
    artifact_hashes = manifest["artifact_sha256"]
    need(len(artifact_hashes) == 233, "artifact_inventory")
    for name, digest in artifact_hashes.items():
        need(
            name in ("summary.json", "records.jsonl")
            or re.fullmatch(r"raw/[0-9]{4}-[0-9a-f]{64}\.json", name),
            "artifact_inventory",
        )
        need(sha(read(run_root, scorer + "/" + name)) == digest, "artifact_digest")
    expected_types = (
        "planned",
        "native_preflight",
        "native_raw_returned",
        "prepared_not_delivered",
    )
    for index, (task, row) in enumerate(zip(tasks, records, strict=True)):
        need(
            row.get("task_id") == task.id
            and type(row.get("task_index")) is int
            and row.get("task_index") == index
            and row.get("family") == task.family
            and row.get("split") == task.split
            and row.get("group") == task.group,
            "task_identity",
        )
        filename = f"raw/{index:04d}-{sha(task.id.encode())}.json"
        need(row.get("raw_path") == filename, "raw_path")
        raw_bytes = read(run_root, scorer + "/" + filename)
        raw = json.loads(raw_bytes)
        expected_request = {
            "state": task.state,
            "questions": {"decision": official.base.build_question(task)},
        }
        need(
            row.get("raw_sha256") == sha(raw_bytes)
            and raw.get("response_encoding") == "json"
            and raw.get("error") is None
            and same_json(raw.get("request"), expected_request)
            and row.get("request_sha256") == heaven._hash(heaven._bytes(expected_request)),
            "original_request",
        )
        per_call = events[index * 4 : index * 4 + 4]
        need(
            [event.get("event") for event in per_call] == list(expected_types)
            and all(
                type(event.get("call")) is int
                and event.get("call") == index + 1
                and event.get("route") == "jevbench-generic-native-v1"
                and event.get("delta_applied") == []
                and event.get("delta_bypassed") == ["operation", "click_target"]
                for event in per_call
            ),
            "event_order",
        )
        planned, preflight, returned, prepared = per_call
        request_sha = sha(encode(expected_request))
        need(
            encode(planned.get("request")) == encode(expected_request)
            and all(event.get("request_sha256") == request_sha for event in per_call),
            "event_request",
        )
        native = native_payload(DecisionRequest.model_validate(expected_request))
        one, two = preflight["evidence"], returned["evidence"]
        response = prepared["response"]
        need(
            one.get("raw_response") is None
            and same_ordered(one.get("preflight"), two.get("preflight"))
            and encode(one.get("native_payload"))
            == encode(native)
            == encode(two.get("native_payload"))
            and same_ordered(response.get("decision3_provider_evidence"), two)
            and prepared.get("response_sha256") == sha(encode(response))
            and same_json(response, raw.get("response")),
            "event_response",
        )
        reconstructed = normalize_native_response(
            DecisionRequest.model_validate(expected_request), two["raw_response"]
        )
        need(
            same_ordered(response.get("answers"), reconstructed.get("answers")),
            "native_answer_binding",
        )
        probs = heaven._native_probs(raw["response"], task)
        outcome = official.scoring.score_task(probs, task)
        need(
            same_json(row.get("official_task_outcome"), outcome)
            and same_json(row.get("probs_as_returned"), probs)
            and row.get("ok") is True
            and row.get("status") == "ok"
            and row.get("error") is None
            and all(
                same_json(row.get(k), outcome.get(k))
                for k in (
                    "valid",
                    "strict_valid",
                    "renormalized",
                    "correct",
                    "predicted",
                    "probs",
                    "ordinal_ev",
                )
            ),
            "official_item_score",
        )
        need(
            type(row.get("latency_s")) in (float, int)
            and math.isfinite(row["latency_s"])
            and row["latency_s"] >= 0,
            "invalid_latency",
        )
    recomputed = official.summarize.summarize(tasks, records)
    recomputed.update(
        dataset_label=config["dataset_label"],
        identity=manifest["identity"],
        source_revision=config["source_revision"],
        public_exposure=config["public_exposure"],
        official_composite=None,
        official_rank=None,
        leaderboard_status="not_submitted",
        sealed_evaluation="not_run",
    )
    need(same_json(recomputed, summary), "official_summary")
    need(
        same_json(
            evaluation.get("event_counts"),
            {**dict(collections.Counter(e["event"] for e in events)), "failed": 0},
        ),
        "event_counts",
    )
    previous_end = None
    for i in range(0, len(journal), 2):
        acquired, completed = journal[i : i + 2]
        need(
            acquired.get("event") == "acquired"
            and completed.get("event") == "completed"
            and acquired.get("kind") == completed.get("kind") == "decision3-gpu"
            and acquired.get("pid") == completed.get("pid") == hello["pid"]
            and acquired.get("nonce") == completed.get("nonce") == plan["transport"]["nonce"]
            and acquired.get("thread") == completed.get("thread")
            and type(acquired.get("monotonic_ns")) is int
            and type(completed.get("monotonic_ns")) is int
            and acquired["monotonic_ns"] <= completed["monotonic_ns"]
            and (previous_end is None or previous_end <= acquired["monotonic_ns"]),
            "lease_pairs",
        )
        previous_end = completed["monotonic_ns"]
    need(
        evaluation["lease"].get("own_completed_pairs") == 231
        and evaluation["lease"].get("sha256") == sha(journal_bytes),
        "lease_digest",
    )
    observations = [e for e in supervision if e.get("event") == "host"]
    binding = [e for e in supervision if e.get("event") == "binding_verified"]
    prelaunch = [e for e in supervision if e.get("event") == "prelaunch"]
    need(
        len(binding) == len(prelaunch) == 1
        and len(observations) > 0
        and binding[0].get("pid") == hello.get("pid")
        and binding[0].get("invocation") == hello.get("invocation")
        and binding[0].get("startticks") == hello.get("startticks")
        and binding[0].get("boot_id") == hello.get("boot_id")
        and all(e.get("running") == e.get("pending") == 0 for e in observations + prelaunch),
        "supervision",
    )
    observations_in_order = [e for e in supervision if e.get("event") in ("prelaunch", "host")]
    timestamps = [e.get("monotonic") for e in observations_in_order]
    need(
        all(type(value) in (int, float) and math.isfinite(value) for value in timestamps)
        and all(a <= b for a, b in zip(timestamps, timestamps[1:]))
        and all(
            type(e.get("available_bytes")) is int and e["available_bytes"] >= 34709107508
            for e in observations_in_order
        ),
        "supervision_resources",
    )
    selected = (
        "n_planned",
        "n_attempted",
        "n_valid",
        "n_correct",
        "accuracy",
        "macro_accuracy",
        "brier_mean",
        "ordinal_mae",
        "coverage",
        "schema_validity",
        "schema_validity_strict",
        "operational_success",
        "latency",
    )
    return {
        "audit": "pass_pending_external_acceptance",
        "official_revision": config["source_revision"],
        "dataset_sha256": config["canonical_dataset_sha256"],
        "plan_sha256": sha(plan_raw),
        "source_verification": sources,
        "raw_and_official_scores_verified": 231,
        "generic_events_verified": len(events),
        "lease_pairs_verified": len(journal) // 2,
        "host_observations": len(observations),
        "observed_memory_at_least_base_floor": True,
        "selected_supervisor_memory_floor": "not_recorded_in_closure",
        "max_host_observation_gap_s": max(b - a for a, b in zip(timestamps, timestamps[1:])),
        "supervisor_cgroup_absent": True,
        "cuda_allocated_at_worker_cleanup": terminal["final_cuda_allocated"],
        "summary": {key: summary[key] for key in selected},
        "latency_comparable": False,
        "latency_note": "full guarded callback; workload overlap is not verified and latency is not fairly comparable",
        "scope": "public 231 only, unsealed; no leaderboard rank or other benchmark score",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--official-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.run_root, args.official_root)
    except Exception as exc:
        # Do not print exceptions: they may contain request or answer content.
        print(f"audit failed ({type(exc).__name__}); no result", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
