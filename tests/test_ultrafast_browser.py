"""Outcome and durability tests; actual Chromium acceptance is a separate live run."""

from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "ultrafast_browser", Path(__file__).resolve().parents[1] / "scripts/ultrafast_browser.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_done_is_not_a_verifier_input():
    expected = {"hash": "#choices", "heading": "Article", "body_contains": ["Full article"]}
    assert not RUNNER.verify_final_state(expected, {"status": "done"})["verified_success"]
    assert not RUNNER.verify_final_state(
        expected, {"hash": "#choices", "heading": "Article", "body": "Matching link"}
    )["verified_success"]
    assert RUNNER.verify_final_state(
        expected, {"hash": "#choices", "heading": "Article", "body": "Full article"}
    )["verified_success"]


def test_matching_result_does_not_substitute_for_requested_filter():
    result = RUNNER.verify_final_state(
        {"category": "Design", "free_cancellation": True, "results": ["Casa Flora"]},
        {"category": "all", "free_cancellation": False, "results": ["Casa Flora"]},
    )
    assert not result["verified_success"]
    assert result["checks"]["results"]["passed"]
    assert not result["checks"]["free_cancellation"]["passed"]


@pytest.mark.parametrize(
    "expected", [{}, {"unknown": 1}, {"body_contains": "Article"}, {"body_contains": []}]
)
def test_invalid_verifier_fails_closed(expected):
    with pytest.raises(ValueError):
        RUNNER.verify_final_state(expected, {})


def test_missing_final_state_cannot_satisfy_null_predicate():
    assert not RUNNER.verify_final_state({"heading": None}, None)["verified_success"]
    assert not RUNNER.verify_final_state({"heading": None}, {})["verified_success"]


@pytest.mark.parametrize("revision,dirty", [("a" * 40, ""), (RUNNER.UPSTREAM_REVISION, " M file")])
def test_wrong_revision_or_dirty_upstream_rejected_before_import(
    monkeypatch, tmp_path, revision, dirty
):
    responses = iter([revision, dirty])
    monkeypatch.setattr(RUNNER.subprocess, "check_output", lambda *_args, **_kw: next(responses))
    with pytest.raises(ValueError, match="clean pinned"):
        RUNNER.load_upstream(tmp_path)


def fake_agent_module(*, final_state, raises=False, status="done"):
    instances = []

    class Browser:
        closed = False

        def evaluate(self, expression):
            assert expression == RUNNER.FINAL_STATE
            return final_state

        def call(self, method, **_params):
            assert method == "Page.captureScreenshot"
            return {"data": base64.b64encode(b"test-jpeg").decode()}

    class Agent:
        def __init__(self, url, goal, *, screenshots):
            self.url, self.goal = url, goal
            self.browser = Browser()
            self.state = {
                "status": "ready",
                "history": [],
                "decisions": [],
                "text_calls": [],
                "elapsed_ms": 0,
            }
            instances.append(self)

        def snapshot(self):
            return dict(self.state)

        def command(self, name):
            assert name == "tick"
            # Mutation can occur before observation fails. Keep that evidence.
            self.state["history"].append({"action": "click executed"})
            self.state["decisions"].append({"operation": "CLICK"})
            if raises:
                raise RuntimeError("observation interrupted after click")
            self.state["status"] = status

        def close(self):
            self.browser.closed = True

    return SimpleNamespace(Agent=Agent), instances


def task():
    return {
        "id": "case",
        "split": "development",
        "goal": "Open Article",
        "expected": {"hash": "#article", "heading": "Article"},
    }


def test_error_preserves_mutation_trace_and_runs_fresh_verifier(tmp_path):
    module, instances = fake_agent_module(final_state={"hash": "", "heading": "Home"}, raises=True)
    output = tmp_path / "run"
    summary = RUNNER.run_task(module, task(), "http://127.0.0.1/fixture", output)
    assert not summary["verified_success"]
    assert summary["termination"] == "exception"
    assert summary["error"]["message"] == "observation interrupted after click"
    assert summary["actions"] == 1
    assert instances[0].browser.closed
    rows = [json.loads(row) for row in (output / "snapshots.jsonl").read_text().splitlines()]
    assert rows[-1]["snapshot"]["history"] == [{"action": "click executed"}]
    assert json.loads((output / "final-state.json").read_text())["heading"] == "Home"
    assert instances[0].goal == "Open Article"  # Oracle never enters policy input.


def test_false_done_is_classified_and_cannot_overwrite_evidence(tmp_path):
    module, _ = fake_agent_module(final_state={"hash": "", "heading": "Home"})
    output = tmp_path / "run"
    summary = RUNNER.run_task(module, task(), "http://127.0.0.1/fixture", output)
    assert summary["agent_status"] == "done"
    assert summary["failure_class"] == "false_done"
    assert not summary["verified_success"]
    with pytest.raises(FileExistsError):
        RUNNER.run_task(module, task(), "http://127.0.0.1/fixture", output)


def test_decision_budget_leaves_final_state_and_marks_noncompletion(tmp_path):
    module, _ = fake_agent_module(final_state={"hash": "", "heading": "Home"}, status="ready")
    summary = RUNNER.run_task(
        module, task(), "http://127.0.0.1/fixture", tmp_path / "run", max_decisions=2
    )
    assert summary["decision_cycles"] == 2
    assert summary["failure_class"] == "budget_exhaustion"
    assert summary["termination"] == "decision_budget"
    assert summary["reported_cost_usd"] is None


def test_success_requires_exact_fresh_state_even_when_agent_finishes(tmp_path):
    module, _ = fake_agent_module(final_state={"hash": "#article", "heading": "Article"})
    summary = RUNNER.run_task(module, task(), "http://127.0.0.1/fixture", tmp_path / "run")
    assert summary["verified_success"]
    assert summary["failure_class"] is None
    assert (tmp_path / "run/final.jpg").read_bytes() == b"test-jpeg"


def test_done_after_timeout_retains_goal_evidence_but_fails_budget(tmp_path, monkeypatch):
    module, _ = fake_agent_module(final_state={"hash": "#article", "heading": "Article"})
    # start, pre-command budget check, final command completion, total wall
    clock = iter([0.0, 0.1, 2.0, 2.1])
    monkeypatch.setattr(RUNNER.time, "perf_counter", lambda: next(clock))
    summary = RUNNER.run_task(
        module, task(), "http://127.0.0.1/fixture", tmp_path / "run", task_timeout=1
    )
    assert summary["termination"] == "done"
    assert summary["goal_observed"]
    assert not summary["within_budget"]
    assert not summary["verified_success"]
    assert summary["failure_class"] == "budget_exhaustion"


def test_screenshot_failure_does_not_replace_fresh_dom_outcome(tmp_path):
    module, instances = fake_agent_module(final_state={"hash": "#article", "heading": "Article"})
    original = module.Agent.command

    def command(self, name):
        def unavailable(*_args, **_kwargs):
            raise RuntimeError("screenshot unavailable")

        original(self, name)
        self.browser.call = unavailable

    module.Agent.command = command
    summary = RUNNER.run_task(module, task(), "http://127.0.0.1/fixture", tmp_path / "run")
    assert summary["verified_success"]
    assert summary["verification_error"] is None
    assert summary["screenshot_error"]["message"] == "screenshot unavailable"
    assert instances[0].browser.closed


def test_timeout_does_not_erase_satisfied_final_state(tmp_path, monkeypatch):
    module, _ = fake_agent_module(final_state={"hash": "#article", "heading": "Article"})
    clock = iter([0.0, 2.0, 2.1, 2.2])
    monkeypatch.setattr(RUNNER.time, "perf_counter", lambda: next(clock))
    summary = RUNNER.run_task(
        module, task(), "http://127.0.0.1/fixture", tmp_path / "run", task_timeout=1
    )
    assert summary["termination"] == "task_timeout"
    assert summary["goal_observed"]
    assert not summary["verified_success"]
    assert summary["decision_cycles"] == 0


def test_transport_cleanup_closes_all_partial_targets_and_keeps_error_receipts(tmp_path):
    transport = RUNNER.IsolatedCDP(SimpleNamespace())
    transport.output = tmp_path
    transport.targets = {"first": None, "second": None}
    calls = []

    def close(method, *, targetId):
        assert method == "Target.closeTarget"
        calls.append(targetId)
        transport.targets.pop(targetId)
        if targetId == "first":
            raise RuntimeError("trace save failed")

    transport.call = close
    errors = transport.close_task()
    assert calls == ["first", "second"]
    assert not transport.targets
    assert errors == [{"target": "first", "type": "RuntimeError", "message": "trace save failed"}]
    assert json.loads((tmp_path / "cleanup-errors.json").read_text()) == errors


def test_manifest_has_fixed_population_and_distinct_reserved_tasks():
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/benchmarks/ultrafast-local.json"
        ).read_text()
    )
    assert manifest["upstream_revision"] == RUNNER.UPSTREAM_REVISION
    tasks = manifest["tasks"]
    assert len(tasks) == 6
    assert len({row["id"] for row in tasks}) == 6
    assert sum(row["split"] == "development" for row in tasks) == 3
    assert sum(row["split"] == "reserved_transfer" for row in tasks) == 3
    for row in tasks:
        assert row["goal"]
        RUNNER.verify_final_state(row["expected"], {})
