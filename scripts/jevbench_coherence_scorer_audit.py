"""Replay a complete pinned JevBench coherence run from its captured responses.

The official SDK uses a new in-memory cache. This tool makes no model calls,
never changes the evidence, and prints aggregate results only. It does not read
unselected private sources or accept a pending score as an official ranking.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import jevbench_public_request_inventory as inventory
from jevbench_heaven_scorer_audit import (
    encode,
    need,
    public_pin_path,
    read,
    read_pinned_journal,
    release_layout,
    sha,
    verify_release_pair,
)

from s1.contracts import DecisionRequest
from s1.evaluation.decision3 import native_payload, normalize_native_response

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SHA = "ebacca34e3d4193419e43629f73fa5fed6617834415ff87ff6d12a09b6d3904a"
EVENT_TYPES = ("planned", "native_preflight", "native_raw_returned", "prepared_not_delivered")


def same_ordered(left, right):
    return encode(left) == encode(right)


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    def finite_float(value):
        result = float(value)
        need(math.isfinite(result), "nonfinite")
        return result

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda _: need(False, "nonfinite"),
        parse_float=finite_float,
    )


def read_json(root, name):
    return decode(read(root, name))


def lines(root, name):
    return [decode(line) for line in read(root, name).splitlines()]


def _source_check(plan):
    version, (owner_release, supervisor_release), generic_release, lease_release = release_layout(
        plan, "coherence", ROOT
    )
    allowed = (ROOT / "scripts", ROOT / "src/s1", ROOT / "configs")
    matched = outside = 0
    for name, expected in plan["source_pins"].items():
        path = public_pin_path(name, allowed, failure="source_pin")
        if path is None:
            outside += 1
            continue
        need(sha(path.read_bytes()) == expected, "source_pin")
        matched += 1
    names = (
        "jev_decision3_jevbench_coherence_owner.py",
        owner_release.name,
        "jev_decision3_jevbench_coherence_supervisor.py",
        supervisor_release.name,
        "jev_decision3_jevbench_generic.py",
        "jev_gpu_forward_lease.py",
        "jev_decision3_candidate_browser_child.py",
        "jev_decision3_candidate_http_owner.py",
        "jev_stage_memory_profile_v2.py",
        "jevbench_public_request_inventory.py",
    )
    paths = [ROOT / "scripts" / name for name in names]
    paths += [generic_release, lease_release]
    need(
        all(p.is_relative_to(ROOT / "scripts") and str(p) in plan["source_pins"] for p in paths),
        "source_pin_missing",
    )
    for held, release in (
        (paths[0], paths[1]),
        (paths[2], paths[3]),
        (paths[4], paths[-2]),
        (paths[5], paths[-1]),
    ):
        verify_release_pair(held, release, failure="release_source")
    for path in (
        inventory.COHERENCE_CONFIG.absolute(),
        Path(sys.modules[DecisionRequest.__module__].__file__).absolute(),
        Path(sys.modules[native_payload.__module__].__file__).absolute(),
    ):
        need(str(path) in plan["source_pins"], "source_pin_missing")
    config = inventory.COHERENCE_CONFIG.read_bytes()
    need(
        sha(config) == CONFIG_SHA
        and plan["source_pins"][str(inventory.COHERENCE_CONFIG.absolute())] == CONFIG_SHA,
        "official_config_pin",
    )
    result = {"public_files_verified": matched, "external_files_not_read": outside}
    if version == "v2":
        result["release_revision"] = version
    return result, decode(config)


class CapturedReplay:
    """Replay exact ordered request bytes and only their captured answers."""

    def __init__(self, captured):
        self.captured = captured
        self.count = 0

    def __call__(self, state, questions):
        need(
            self.count < len(self.captured)
            and encode({"state": state, "questions": questions}) == self.captured[self.count][0],
            "official_request_order",
        )
        answers = self.captured[self.count][1]
        self.count += 1
        return answers


def _latency_percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return (
        ordered[lo] if lo == hi else ordered[lo] * (hi - position) + ordered[hi] * (position - lo)
    )


def _events(root, timing, journal, hello, plan):
    events = lines(root, "worker/generic-events.jsonl")
    need(len(events) == 4992 and len(timing) == 1248 and len(journal) == 2496, "event_count")
    captured = []
    durations = []
    previous_end = None
    for index in range(1248):
        ev = events[4 * index : 4 * index + 4]
        need([x.get("event") for x in ev] == list(EVENT_TYPES), "event_sequence")
        need(
            all(
                type(x.get("call")) is int
                and x["call"] == index + 1
                and x.get("route") == "jevbench-generic-native-v1"
                and x.get("delta_applied") == []
                and x.get("delta_bypassed") == ["operation", "click_target"]
                for x in ev
            ),
            "route_or_bypass",
        )
        planned, preflight, returned, prepared = ev
        request, response = planned["request"], prepared["response"]
        request_sha = sha(encode(request))
        need(
            all(x.get("request_sha256") == request_sha for x in ev)
            and prepared.get("response_sha256") == sha(encode(response)),
            "event_digest",
        )
        model_request = DecisionRequest.model_validate(request)
        native = native_payload(model_request)
        one, two = preflight["evidence"], returned["evidence"]
        need(
            one.get("raw_response") is None
            and same_ordered(one.get("preflight"), two.get("preflight"))
            and same_ordered(one.get("native_payload"), native)
            and same_ordered(two.get("native_payload"), native)
            and same_ordered(response.get("decision3_provider_evidence"), two),
            "native_evidence",
        )
        checks = one.get("preflight")
        need(
            type(checks) is dict
            and type(checks.get("errors")) is dict
            and not checks["errors"]
            and type(checks.get("project_context_limit")) is int
            and checks["project_context_limit"] == 16384
            and type(checks.get("lengths")) is dict
            and set(checks["lengths"]) == set(request["questions"])
            and all(
                type(length) is int and 1 <= length <= 16384
                for length in checks["lengths"].values()
            )
            and type(two.get("raw_response")) is dict
            and two["raw_response"].get("model") == "d3-edge",
            "native_preflight",
        )
        reconstructed = normalize_native_response(model_request, two["raw_response"])
        need(same_ordered(response.get("answers"), reconstructed.get("answers")), "native_answers")
        tick = timing[index]
        start, end = tick.get("started_monotonic_ns"), tick.get("completed_monotonic_ns")
        need(
            type(tick.get("call")) is int
            and tick["call"] == index + 1
            and tick.get("request_sha256") == request_sha
            and type(start) is int
            and type(end) is int
            and 0 <= start <= end
            and (previous_end is None or previous_end <= start),
            "timing_evidence",
        )
        acquired, completed = journal[2 * index : 2 * index + 2]
        a, b = acquired.get("monotonic_ns"), completed.get("monotonic_ns")
        need(
            acquired.get("event") == "acquired"
            and completed.get("event") == "completed"
            and acquired.get("kind") == completed.get("kind") == "decision3-gpu"
            and type(acquired.get("pid")) is int
            and acquired.get("pid") == completed.get("pid") == hello["pid"]
            and acquired.get("nonce") == completed.get("nonce") == plan["transport"]["nonce"]
            and type(acquired.get("thread")) is int
            and acquired.get("thread") == completed.get("thread")
            and type(a) is int
            and type(b) is int
            and start <= a <= b <= end,
            "lease_evidence",
        )
        previous_end = end
        durations.append((end - start) / 1_000_000_000)
        captured.append((encode(request), response["answers"]))
    need(len({wire for wire, _ in captured}) == 1248, "unique_requests")
    return captured, durations, len(events)


def _supervisor(root, plan, plan_sha, terminal, hello, closure, timing):
    go = read_json(root, "worker/go.json")
    need(
        hello.get("plan_sha256") == go.get("plan_sha256") == terminal.get("plan_sha256") == plan_sha
        and hello.get("invocation")
        == go.get("invocation")
        == terminal.get("invocation")
        == closure.get("invocation")
        and hello.get("unit") == plan.get("unit") == closure.get("unit"),
        "binding",
    )
    need(
        terminal.get("status") == closure.get("status") == "completed_pending_external_audit"
        and type(terminal.get("final_cuda_allocated")) is int
        and terminal["final_cuda_allocated"] == 0
        and closure.get("cgroup_closure") == "absent"
        and closure.get("launch_succeeded") is True
        and (
            closure.get("closure") == "retained-unit"
            or (
                closure.get("closure") == "confirmed-unit-GC"
                and closure.get("stop_returned") is True
            )
        )
        and terminal.get("model_score") is None,
        "closure",
    )
    observations = lines(root, "supervisor/events.jsonl")
    pre = [x for x in observations if x.get("event") == "prelaunch"]
    bindings = [x for x in observations if x.get("event") == "binding_verified"]
    host = [x for x in observations if x.get("event") == "host"]
    need(len(pre) == len(bindings) == 1 and bool(host), "supervision")
    b = bindings[0]
    need(
        b.get("pid") == hello.get("pid")
        and b.get("startticks") == hello.get("startticks")
        and b.get("boot_id") == hello.get("boot_id")
        and b.get("invocation") == hello.get("invocation"),
        "binding",
    )
    samples = [x for x in observations if x.get("event") in ("prelaunch", "host")]
    times = [x.get("monotonic") for x in samples]
    need(
        all(type(t) in (int, float) and math.isfinite(t) for t in times)
        and all(a <= b for a, b in zip(times, times[1:]))
        and all(
            type(x.get("running")) is int
            and x["running"] == 0
            and type(x.get("pending")) is int
            and x["pending"] == 0
            and type(x.get("available_bytes")) is int
            and x["available_bytes"] >= 34709107508
            for x in samples
        ),
        "host_samples",
    )
    need(
        bool(timing)
        and host[0]["monotonic"] <= timing[0]["started_monotonic_ns"] / 1e9
        and timing[-1]["completed_monotonic_ns"] / 1e9 <= host[-1]["monotonic"],
        "host_sampling_does_not_span_callbacks",
    )
    return {
        "host_samples": len(host),
        "max_sample_gap_s": max(b - a for a, b in zip(times, times[1:])),
        "sampled_memory_above_base_floor": True,
        "host_samples_span_callbacks": True,
        "selected_floor": "not_recorded_in_closure",
        "cgroup_absent": True,
        "cuda_allocated_bytes": 0,
    }


def audit(run_root, official_root):
    root = Path(run_root).resolve(strict=True)
    official_root = Path(official_root)
    need(
        official_root.is_absolute() and official_root.resolve(strict=True) == official_root,
        "ordinary_official_root",
    )
    plan_bytes = read(root, "plan.json")
    plan = decode(plan_bytes)
    sources, config = _source_check(plan)
    stage = read_json(root, "worker/staging/status.json")
    quarantine = read_json(root, "worker/staging/coherence-mini/STATUS_NOT_ACCEPTED.json")
    terminal = read_json(root, "worker/terminal.json")
    hello = read_json(root, "worker/hello.json")
    closure = read_json(root, "supervisor/closure.json")
    need(
        stage.get("score_accepted") is False and quarantine.get("score_accepted") is False,
        "not_quarantined",
    )
    evaluation = terminal["evaluation"]
    need(
        evaluation.get("status") == "completed_pending_external_audit"
        and evaluation.get("score_accepted") is False
        and evaluation.get("model_score") is None
        and evaluation.get("cost_usd") is None
        and evaluation.get("delta_route") == "both_bypassed"
        and evaluation.get("calls") == 1248
        and evaluation.get("planned_trials") == 1200
        and evaluation.get("outcome_rows") == 1468
        and same_ordered(
            evaluation.get("event_counts"),
            {
                "planned": 1248,
                "native_preflight": 1248,
                "native_raw_returned": 1248,
                "prepared_not_delivered": 1248,
                "failed": 0,
            },
        ),
        "evaluation_identity",
    )
    report_bytes = read(root, "worker/staging/coherence-mini/report.json")
    need(sha(report_bytes) == evaluation.get("official_report_sha256"), "report_hash")
    recorded = decode(report_bytes)
    timing_bytes = read(root, "worker/callback-timing.jsonl")
    journal_bytes = read_pinned_journal(root, plan, evaluation["lease"]["sha256"])
    need(
        sha(timing_bytes) == evaluation["callback_timing"]["sha256"]
        and evaluation["callback_timing"]["count"] == 1248
        and sha(journal_bytes) == evaluation["lease"]["sha256"]
        and evaluation["lease"]["own_completed_pairs"] == 1248,
        "timing_or_lease_digest",
    )
    timing = [decode(line) for line in timing_bytes.splitlines()]
    journal = [decode(line) for line in journal_bytes.splitlines()]
    captured, durations, event_count = _events(root, timing, journal, hello, plan)
    supervisor = _supervisor(root, plan, sha(plan_bytes), terminal, hello, closure, timing)
    finder = inventory._verified_coherence(official_root, config, suite_name="jevbench-mini")
    sys.meta_path.insert(0, finder)
    try:
        from jevbench import evaluate
        from jevbench.backends import CallableBackend
        from jevbench.suites import load_suite

        suite = load_suite("jevbench-mini")
        need(len(suite.cases) == 240 and len(suite.plan) == 1200, "official_plan")
        replay = CapturedReplay(captured)
        report = evaluate(
            CallableBackend(replay, name=recorded["result"]["backend"]),
            suite=suite,
            cache=":memory:",
            concurrency=1,
            seed=0,
            repeats=1,
            trials=1,
            minimize=False,
            progress=False,
        )
        need(
            replay.count == 1248
            and report.result["requests"] == 1248
            and report.result["cache_hits"] == 193
            and not report.result["errors"]
            and not report.result["truncated_by_budget"]
            and not report.result["backend_down"]
            and len(report.result["bases"]) == 240
            and sum(report.result["planned_tests"].values()) == 1200
            and len(report.result["outcomes"]) == 1468,
            "replay_population",
        )
        # Per-run durations and Python/library provenance differ by replay environment.
        ignored_result = {"environment", "backend_seconds", "wall_seconds"}
        expected_result = {k: v for k, v in recorded["result"].items() if k not in ignored_result}
        replayed_result = {k: v for k, v in report.result.items() if k not in ignored_result}
        need(same_ordered(expected_result, replayed_result), "official_result_difference")
        need(
            same_ordered(recorded["settings"], report.settings)
            and type(report.settings.get("bootstrap")) is int
            and report.settings["bootstrap"] == 1000,
            "official_settings_difference",
        )
        actual_scores = report.scores
        need(
            type(actual_scores.get("bootstrap")) is int and actual_scores["bootstrap"] == 1000,
            "official_bootstrap_difference",
        )
        expected_scores = {k: v for k, v in recorded["scores"].items() if k != "environment"}
        replayed_scores = {k: v for k, v in actual_scores.items() if k != "environment"}
        need(same_ordered(expected_scores, replayed_scores), "official_scores_difference")
        inventory._check_coherence_files(official_root, config, suite_name="jevbench-mini")
        score = {
            "overall": actual_scores["overall"],
            "overall_ci": actual_scores["ci"]["overall"],
            "dimensions": actual_scores["pillars"],
            "n_cases": actual_scores["n_cases"],
            "n_checks": actual_scores["n_checks"],
            "n_outcomes": actual_scores["n_outcomes"],
            "incomplete_relations": len(actual_scores["incomplete"]),
            "bootstrap": actual_scores["bootstrap"],
        }
    finally:
        sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if name == "jevbench" or name.startswith("jevbench."):
                del sys.modules[name]
    return {
        "audit": "pass_pending_external_acceptance",
        "official_revision": config["source_revision"],
        "source_verification": sources,
        "plan_sha256": sha(plan_bytes),
        "raw_callback_responses_verified": len(captured),
        "events_verified": event_count,
        "lease_pairs_verified": len(journal) // 2,
        "official_trials_verified": 1200,
        "official_outcome_rows_verified": 1468,
        "fresh_replay_cache": ":memory:",
        "supervision": supervisor,
        "score": score,
        "callback_latency": {
            "n": len(durations),
            "p50_s": _latency_percentile(durations, 0.5),
            "p95_s": _latency_percentile(durations, 0.95),
            "method": "linear interpolation at (n-1)*q",
        },
        "latency_scope": evaluation["callback_timing"]["scope"],
        "latency_comparable": False,
        "latency_note": "guarded callback, CPU workload overlap is unknown; no fair speed comparison",
        "cost_usd": None,
        "scope": "official public mini suite only, no leaderboard rank",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--official-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        value = audit(args.run_root, args.official_root)
    except Exception as exc:
        # Error contents can include individual requests or answers.
        print(f"audit failed ({type(exc).__name__}); no result", file=sys.stderr)
        return 1
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
