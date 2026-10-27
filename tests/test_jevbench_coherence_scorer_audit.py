"""Synthetic events only; no public task text, private cases, or GPU."""

import copy
from types import SimpleNamespace

import jevbench_coherence_scorer_audit as audit
import pytest


def test_coherence_accepts_only_the_plan_bound_journal(tmp_path):
    journal = tmp_path / "forward.journal"
    raw = b'{"event":"synthetic"}\n'
    journal.write_bytes(raw)
    info = journal.stat()
    plan = {
        "transport": {
            "journal": str(journal),
            "stat": {"journal": {"device": info.st_dev, "inode": info.st_ino}},
        }
    }
    assert audit.read_pinned_journal(tmp_path, plan, audit.sha(raw)) == raw
    replacement = tmp_path / "replacement"
    replacement.write_bytes(raw)
    replacement.replace(journal)
    with pytest.raises(ValueError, match="journal_identity"):
        audit.read_pinned_journal(tmp_path, plan, audit.sha(raw))


def test_replay_requires_exact_order_and_json_types():
    first = {"state": {"x": 1}, "questions": {"a": {}, "b": {}}}
    second = {"state": "synthetic", "questions": {"next": {}}}
    replay = audit.CapturedReplay(
        [(audit.encode(first), {"synthetic": 1}), (audit.encode(second), {"synthetic": 2})]
    )
    with pytest.raises(ValueError, match="official_request_order"):
        replay({"x": True}, first["questions"])
    with pytest.raises(ValueError, match="official_request_order"):
        replay(first["state"], {"b": {}, "a": {}})
    assert replay.count == 0
    assert replay(first["state"], first["questions"]) == {"synthetic": 1}
    assert replay(second["state"], second["questions"]) == {"synthetic": 2}
    with pytest.raises(ValueError, match="official_request_order"):
        replay(second["state"], second["questions"])


def make_events():
    events, timing, lease = [], [], []
    for index in range(1248):
        request = {"state": f"synthetic-{index}", "questions": {"head": {"type": "noul"}}}
        request_sha = audit.sha(audit.encode(request))
        base = {
            "call": index + 1,
            "route": "jevbench-generic-native-v1",
            "delta_applied": [],
            "delta_bypassed": ["operation", "click_target"],
            "request_sha256": request_sha,
        }
        preflight = {
            "native_payload": request,
            "preflight": {"lengths": {"head": 1}, "errors": {}, "project_context_limit": 16384},
            "raw_response": None,
        }
        native_raw = {"model": "d3-edge", "answers": {"head": {"noul": 0.5}}}
        returned = {**preflight, "raw_response": native_raw}
        response = {"answers": native_raw["answers"], "decision3_provider_evidence": returned}
        events.extend(
            [
                {**base, "event": "planned", "request": request},
                {**base, "event": "native_preflight", "evidence": preflight},
                {**base, "event": "native_raw_returned", "evidence": returned},
                {
                    **base,
                    "event": "prepared_not_delivered",
                    "response": response,
                    "response_sha256": audit.sha(audit.encode(response)),
                },
            ]
        )
        start = 10 * index
        timing.append(
            {
                "call": index + 1,
                "request_sha256": request_sha,
                "started_monotonic_ns": start,
                "completed_monotonic_ns": start + 9,
            }
        )
        lease.extend(
            [
                {
                    "event": event,
                    "kind": "decision3-gpu",
                    "pid": 123,
                    "thread": 42,
                    "nonce": "synthetic",
                    "monotonic_ns": start + offset,
                }
                for event, offset in (("acquired", 2), ("completed", 7))
            ]
        )
    return events, timing, lease


@pytest.mark.parametrize("tamper", [None, "call", "native_bool", "answer", "timing", "lease"])
def test_events_bind_all_1248_requests_and_fail_closed(monkeypatch, tamper):
    events, timing, lease = make_events()
    if tamper == "call":
        events[0]["call"] = True
    elif tamper == "native_bool":
        events[1]["evidence"]["preflight"]["lengths"]["head"] = True
    elif tamper == "answer":
        events[2]["evidence"]["raw_response"]["answers"]["head"]["noul"] = 0.9
    elif tamper == "timing":
        timing[4]["request_sha256"] = "other"
    elif tamper == "lease":
        lease[9]["monotonic_ns"] = timing[4]["completed_monotonic_ns"] + 1
    # Ensure events hold independent wire objects as JSON actually does.
    raw = b"\n".join(audit.encode(event) for event in events) + b"\n"
    monkeypatch.setattr(audit, "read", lambda _root, _name: raw)
    monkeypatch.setattr(audit, "DecisionRequest", SimpleNamespace(model_validate=lambda x: x))
    monkeypatch.setattr(audit, "native_payload", lambda request: request)
    monkeypatch.setattr(
        audit, "normalize_native_response", lambda _request, raw: {"answers": raw["answers"]}
    )
    args = (None, timing, lease, {"pid": 123}, {"transport": {"nonce": "synthetic"}})
    if tamper is None:
        captured, durations, count = audit._events(*args)
        assert len(captured) == len(durations) == 1248 and count == 4992
    else:
        with pytest.raises(ValueError):
            audit._events(*args)


def test_percentile_uses_documented_linear_interpolation():
    values = [1.0, 2.0, 3.0, 4.0]
    before = copy.deepcopy(values)
    assert audit._latency_percentile(values, 0.5) == 2.5
    assert audit._latency_percentile(values, 0.95) == pytest.approx(3.85)
    assert values == before


@pytest.mark.parametrize(
    "failure", [None, "unknown_closure", "unproved_gc", "late_start", "early_end", "bool_queue"]
)
def test_supervisor_closure_and_sample_coverage(monkeypatch, failure):
    hello = {
        "pid": 123,
        "startticks": 7,
        "boot_id": "synthetic",
        "invocation": "x",
        "unit": "u",
        "plan_sha256": "digest",
    }
    plan = {"unit": "u"}
    terminal = {
        "plan_sha256": "digest",
        "invocation": "x",
        "status": "completed_pending_external_audit",
        "final_cuda_allocated": 0,
        "model_score": None,
    }
    closure = {
        "invocation": "x",
        "unit": "u",
        "status": "completed_pending_external_audit",
        "cgroup_closure": "absent",
        "launch_succeeded": True,
        "closure": "retained-unit",
    }
    go = {"plan_sha256": "digest", "invocation": "x"}
    samples = [
        {
            "event": "prelaunch",
            "monotonic": 1.0,
            "available_bytes": 34709107508,
            "running": 0,
            "pending": 0,
        },
        {
            "event": "binding_verified",
            "pid": 123,
            "startticks": 7,
            "boot_id": "synthetic",
            "invocation": "x",
        },
        {
            "event": "host",
            "monotonic": 2.0,
            "available_bytes": 34709107508,
            "running": 0,
            "pending": 0,
        },
        {
            "event": "host",
            "monotonic": 6.0,
            "available_bytes": 34709107508,
            "running": 0,
            "pending": 0,
        },
    ]
    timing = [{"started_monotonic_ns": 3_000_000_000, "completed_monotonic_ns": 5_000_000_000}]
    if failure == "unknown_closure":
        closure["closure"] = "unknown"
    elif failure == "unproved_gc":
        closure.update(closure="confirmed-unit-GC", stop_returned=False)
    elif failure == "late_start":
        timing[0]["started_monotonic_ns"] = 1_500_000_000
    elif failure == "early_end":
        timing[0]["completed_monotonic_ns"] = 6_500_000_000
    elif failure == "bool_queue":
        samples[-1]["running"] = False
    monkeypatch.setattr(audit, "read_json", lambda *_: go)
    monkeypatch.setattr(audit, "lines", lambda *_: samples)
    args = (None, plan, "digest", terminal, hello, closure, timing)
    if failure is None:
        assert audit._supervisor(*args)["host_samples_span_callbacks"] is True
    else:
        with pytest.raises(ValueError):
            audit._supervisor(*args)


def test_evidence_json_rejects_duplicate_keys_and_nonfinite():
    with pytest.raises(ValueError, match="duplicate_json_key"):
        audit.decode(b'{"nested":{"key":1,"key":2}}')
    with pytest.raises(ValueError, match="nonfinite"):
        audit.decode(b'{"value":NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        audit.decode(b'{"value":1e999}')
