"""Outcome and durability tests; actual Chromium acceptance is a separate live run."""

from __future__ import annotations

import base64
import copy
import importlib.util
import json
import os
import py_compile
import sys
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


def test_matching_title_and_heading_do_not_substitute_for_destination_url():
    expected = {
        "url": "https://en.wikipedia.org/wiki/Graph_theory",
        "title": "Graph theory - Wikipedia",
        "heading": "Graph theory",
    }
    state = {**expected, "url": "https://en.wikipedia.org/w/index.php?search=Graph+theory"}
    result = RUNNER.verify_final_state(expected, state)
    assert not result["verified_success"]
    assert result["checks"]["title"]["passed"]
    assert result["checks"]["heading"]["passed"]
    assert not result["checks"]["url"]["passed"]
    assert RUNNER.verify_final_state(expected, expected)["verified_success"]


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


@pytest.fixture
def fake_upstream_sources(tmp_path, monkeypatch):
    sources = {
        "jev_ultrafast/__init__.py": b"from .agent import Agent\nfrom .browser import Browser\n",
        "jev_ultrafast/questions.py": b'VALUE = "verified"\n',
        "jev_ultrafast/browser.py": b"class Browser: pass\n",
        "jev_ultrafast/model.py": b"from .questions import VALUE\n",
        "jev_ultrafast/agent.py": b"from .model import VALUE\nclass Agent: pass\n",
        "jev_ultrafast/snapshot.js": b"() => ({})\n",
        "jev_ultrafast/static/fixture.html": b"<!doctype html><title>fixture</title>\n",
        "LICENSE": b"Fixture only\n",
    }
    for relative, raw in sources.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def git(args, **_kwargs):
        if "rev-parse" in args:
            return RUNNER.UPSTREAM_REVISION + "\n"
        if "status" in args:
            return ""
        return sources[args[-1].removeprefix("HEAD:")]

    monkeypatch.setattr(RUNNER.subprocess, "check_output", git)
    yield tmp_path, sources
    alias = "_s1_ultrafast_" + RUNNER.UPSTREAM_REVISION[:12]
    for name in tuple(sys.modules):
        if name == alias or name.startswith(alias + "."):
            del sys.modules[name]


def test_upstream_executes_verified_source_instead_of_timestamp_valid_pyc(fake_upstream_sources):
    root, sources = fake_upstream_sources
    path = root / "jev_ultrafast/questions.py"
    # Construct a bytecode cache with the same source size and mtime. Normal
    # timestamp-based loading would regard this different code as valid.
    path.write_bytes(b'VALUE = "poisoned"\n')
    old_stat = path.stat()
    py_compile.compile(str(path), doraise=True)
    cache = Path(importlib.util.cache_from_source(str(path)))
    assert b"poisoned" in cache.read_bytes()
    path.write_bytes(sources["jev_ultrafast/questions.py"])
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    assert path.stat().st_size == old_stat.st_size
    modules, identity = RUNNER.load_upstream(root)
    assert modules["model"].VALUE == "verified"
    assert modules["agent"].VALUE == "verified"
    assert identity["python_source_loading"] == "compile_verified_bytes_no_pyc"


def test_failed_verified_module_cleans_private_namespace(fake_upstream_sources):
    root, sources = fake_upstream_sources
    sources["jev_ultrafast/model.py"] = b"raise RuntimeError('fixture import failure')\n"
    (root / "jev_ultrafast/model.py").write_bytes(sources["jev_ultrafast/model.py"])
    with pytest.raises(RuntimeError, match="fixture import failure"):
        RUNNER.load_upstream(root)
    alias = "_s1_ultrafast_" + RUNNER.UPSTREAM_REVISION[:12]
    assert not any(name == alias or name.startswith(alias + ".") for name in sys.modules)


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


class StabilityClock:
    def __init__(self):
        self.value, self.pauses = 0.0, []

    def now(self):
        return self.value

    def sleep(self, seconds):
        assert seconds >= 0
        self.pauses.append(seconds)
        self.value += seconds


def semantic_marker(text="original", *, controls=None):
    return [
        1234.5,
        "https://example.test/",
        0,
        0,
        1120,
        780,
        "Title",
        text,
        [{"node": 1, "role": "button", "label": "Original", "kind": "click", "value": ""}],
        controls or [],
    ]


class MarkerBrowser:
    def __init__(self, clock, values, *, delay=0):
        self.clock, self.values, self.delay = clock, values, delay
        self.calls = []

    def evaluate(self, expression):
        assert expression == "original upstream MARKER"
        index = min(len(self.calls), len(self.values) - 1)
        self.calls.append(expression)
        self.clock.value += self.delay
        value = self.values[index]
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)


def stability_wait(values, *, deadline=10, delay=0):
    clock = StabilityClock()
    browser = MarkerBrowser(clock, values, delay=delay)
    result = RUNNER.wait_for_stable_observation(
        browser, "original upstream MARKER", deadline, clock=clock.now, sleep=clock.sleep
    )
    return result, clock, browser


def test_stability_uses_fixed_quiet_window_and_hash_only_trace():
    large_text = "original visible data " * 10000
    result, clock, browser = stability_wait([semantic_marker(large_text)])
    assert result["stable"] and result["status"] == "stable"
    assert result["elapsed_ms"] == pytest.approx(250)
    assert result["policy"]["quiet_ms"] == 250 and result["policy"]["poll_window_ms"] == 2000
    assert len(browser.calls) >= 2 and result["marker_changes"] == 0
    assert large_text not in json.dumps(result) and len(json.dumps(result)) < 10000
    assert all(len(sample["raw_marker_sha256"]) == 64 for sample in result["samples"])
    assert max(clock.pauses) <= 0.05


@pytest.mark.parametrize("field", [0, 1, 7, 8, 9])
def test_any_original_marker_semantic_change_restarts_quiet_time(field):
    first, changed = semantic_marker(), semantic_marker()
    changed[field] = {
        0: 9999,
        1: "https://example.test/new",
        7: "hydrated visible data",
        8: [{"node": 1, "role": "button", "expanded": "false"}],
        9: [[2, "new value", False, None, False, False]],
    }[field]
    result, _, _ = stability_wait([first, changed])
    assert result["stable"] and result["marker_changes"] == 1
    assert result["elapsed_ms"] == pytest.approx(300)
    assert result["samples"][1]["changed"] is True


def test_raw_hash_is_not_a_new_key_order_sensitive_comparator():
    first, reordered = semantic_marker(), semantic_marker()
    reordered[8][0] = dict(reversed(list(first[8][0].items())))
    assert first == reordered
    result, _, _ = stability_wait([first, reordered])
    assert result["stable"] and result["elapsed_ms"] == pytest.approx(250)
    assert result["marker_changes"] == 0
    assert result["samples"][0]["raw_marker_sha256"] != result["samples"][1]["raw_marker_sha256"]
    assert result["samples"][1]["changed"] is False


def test_unavailable_and_error_reads_never_form_a_stable_value_and_reset_quiet():
    result, _, _ = stability_wait(
        [semantic_marker(), None, OSError("read failed"), semantic_marker()]
    )
    assert result["stable"] and result["elapsed_ms"] == pytest.approx(400)
    assert [sample["valid"] for sample in result["samples"][:4]] == [True, False, False, True]
    assert result["samples"][1]["raw_marker_sha256"] is not None
    assert result["samples"][2]["raw_marker_sha256"] is None
    assert result["samples"][2]["error"]["message"] == "read failed"
    assert result["samples"][3]["quiet_ms"] == 0


@pytest.mark.parametrize("invalid", [None, "marker", {}, [], [None] * 10])
def test_invalid_marker_shapes_use_soft_window_without_becoming_stable(invalid):
    result, _, _ = stability_wait([invalid])
    assert not result["stable"] and result["status"] == "poll_window_elapsed"
    assert result["elapsed_ms"] == pytest.approx(2000)
    assert all(not sample["valid"] and sample["error"] for sample in result["samples"])


def test_blocking_transport_overrun_is_visible_and_does_not_claim_a_hard_cap():
    result, _, browser = stability_wait([semantic_marker()], delay=3)
    assert not result["stable"] and len(browser.calls) == 1
    assert result["elapsed_ms"] == 3000
    assert result["poll_window_overrun_ms"] == 1000
    assert result["task_deadline_overrun_ms"] == 0
    assert "no per-call timeout" in result["policy"]["window_semantics"]


def test_continuously_changing_valid_markers_stop_polling_at_two_seconds():
    result, _, browser = stability_wait([semantic_marker(str(i)) for i in range(100)])
    assert not result["stable"] and result["status"] == "poll_window_elapsed"
    assert result["elapsed_ms"] == pytest.approx(2000)
    assert result["marker_changes"] == len(browser.calls) - 1
    assert all(sample["valid"] and sample["error"] is None for sample in result["samples"])


def test_captured_done_markers_retain_real_hydration_difference_and_later_equality():
    witness = json.loads(
        (Path(__file__).parent / "fixtures/ultrafast_done_markers.json").read_text()
    )
    before, after = witness["markers"]["before_hydration"], witness["markers"]["after_hydration"]
    assert witness["upstream_revision"] == RUNNER.UPSTREAM_REVISION
    assert witness["zero_based_rows"] == {
        "37": "before_hydration",
        "38": "after_hydration",
        "39": "after_hydration",
        "41": "after_hydration",
    }
    assert before[:7] == after[:7] and before != after
    assert before[7] != after[7] and len(before[9]) == 10 and len(after[9]) == 18
    result, _, browser = stability_wait([before, before, after, after, after])
    assert result["stable"] and result["marker_changes"] == 1
    assert result["elapsed_ms"] == pytest.approx(350)
    # Quiet time cannot make a later changed page pass the original equality guard.
    page = {"marker": copy.deepcopy(after)}
    browser.values = [before]
    assert browser.evaluate("original upstream MARKER") != page["marker"]
    browser.values = [after]
    assert browser.evaluate("original upstream MARKER") == page["marker"]


def install_stability_agent(monkeypatch, *, marker_value=None, read_delay=0, inference_delay=0):
    clock = StabilityClock()
    module, instances = fake_agent_module(final_state={"hash": "#article", "heading": "Article"})
    original_init, original_command = module.Agent.__init__, module.Agent.command
    ticks = []

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.state["started_at"] = "original agent clock sentinel"
        original_evaluate = self.browser.evaluate
        marker_browser = MarkerBrowser(clock, [marker_value], delay=read_delay)

        def evaluate(expression):
            if expression == "original upstream MARKER":
                return marker_browser.evaluate(expression)
            return original_evaluate(expression)

        self.browser.evaluate = evaluate

    def command(self, name):
        ticks.append(clock.value)
        clock.value += inference_delay
        return original_command(self, name)

    module.Agent.__init__, module.Agent.command = initialize, command
    monkeypatch.setattr(RUNNER.time, "perf_counter", clock.now)
    monkeypatch.setattr(RUNNER.time, "sleep", clock.sleep)
    return module, instances, clock, ticks


def test_stability_wait_counts_in_original_budget_without_resetting_agent_clock(
    tmp_path, monkeypatch
):
    module, agents, _, ticks = install_stability_agent(
        monkeypatch, marker_value=semantic_marker(), inference_delay=0.6
    )
    summary = RUNNER.run_task(
        module,
        task(),
        "http://local",
        tmp_path / "run",
        task_timeout=1,
        stable_observation_marker="original upstream MARKER",
    )
    assert ticks == [pytest.approx(0.25)]
    assert summary["policy_wall_ms"] == pytest.approx(850)
    assert summary["stable_observation"]["elapsed_ms"] == pytest.approx(250)
    assert summary["verified_success"] and summary["decision_cycles"] == 1
    assert agents[0].state["started_at"] == "original agent clock sentinel"
    trace = [
        json.loads(line)
        for line in (tmp_path / "run/observation-stability.jsonl").read_text().splitlines()
    ]
    assert len(trace) == 1 and trace[0]["before_decision_cycle"] == 1
    assert trace[0]["policy_elapsed_ms_at_return"] == pytest.approx(250)
    assert task()["goal"] not in json.dumps(trace) and "expected" not in json.dumps(trace)


@pytest.mark.parametrize("read_delay,budget,overrun", [(0, 0.2, 0), (3, 1, 2000)])
def test_wait_or_late_cdp_response_exhausting_budget_dispatches_zero_new_ticks(
    tmp_path, monkeypatch, read_delay, budget, overrun
):
    module, agents, _, ticks = install_stability_agent(
        monkeypatch, marker_value=semantic_marker(), read_delay=read_delay
    )
    summary = RUNNER.run_task(
        module,
        task(),
        "http://local",
        tmp_path / "run",
        task_timeout=budget,
        stable_observation_marker="original upstream MARKER",
    )
    assert not ticks and summary["decision_cycles"] == 0
    assert summary["termination"] == "task_timeout" and summary["goal_observed"]
    assert not summary["within_budget"] and not summary["verified_success"]
    assert summary["stable_observation"]["task_deadline_overrun_ms"] == pytest.approx(overrun)
    assert agents[0].state["started_at"] == "original agent clock sentinel"


def test_unsettled_window_returns_to_original_tick_and_default_off_keeps_artifact_bytes(
    tmp_path, monkeypatch
):
    module, _, _, ticks = install_stability_agent(monkeypatch, marker_value=None)
    summary = RUNNER.run_task(
        module,
        task(),
        "http://local",
        tmp_path / "enabled",
        task_timeout=10,
        stable_observation_marker="original upstream MARKER",
    )
    assert ticks == [pytest.approx(2)] and summary["verified_success"]
    assert summary["stable_observation"]["stable_waits"] == 0
    assert summary["stable_observation"]["error_observations"] > 0
    module, _, _, ticks = install_stability_agent(monkeypatch, marker_value=semantic_marker())
    first, second = tmp_path / "off", tmp_path / "explicit-off"
    a = RUNNER.run_task(module, task(), "http://local", first)
    b = RUNNER.run_task(module, task(), "http://local", second, stable_observation_marker=None)
    assert a == b and "stable_observation" not in a
    assert ticks == [0, 0]
    assert not (first / "observation-stability.jsonl").exists()
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()
    }
