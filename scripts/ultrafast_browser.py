"""Unmodified Jev Ultrafast policy on isolated Chromium, with independent outcomes.

Default execution substitutes only the CDP transport. An opt-in diagnostic adds
an outer observation wait before each tick. The upstream snapshot, freshness
guards, operation/target selection and action executor remain pinned source bytes.
The local fixture is public integration screening, not a real-site leaderboard.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UPSTREAM_REVISION = "1231850a0bf1a0c0341fe408ef1668dbbfdfac46"
SOURCE_FILES = (
    "jev_ultrafast/__init__.py",
    "jev_ultrafast/agent.py",
    "jev_ultrafast/browser.py",
    "jev_ultrafast/model.py",
    "jev_ultrafast/questions.py",
    "jev_ultrafast/snapshot.js",
    "jev_ultrafast/static/fixture.html",
    "LICENSE",
)
STABLE_OBSERVATION_POLICY = {
    "name": "pre-tick-original-marker-quiet-v1",
    "quiet_ms": 250,
    "poll_window_ms": 2000,
    "poll_interval_ms": 50,
    "marker_source": "Unchanged pinned upstream Browser.MARKER",
    "comparison": "Original decoded marker equality; hashes are provenance only.",
    "window_semantics": "Soft polling window; synchronous CDP reads have no per-call timeout and may overrun.",
    "on_unsettled": "Record and continue the original tick unless the original task deadline has expired.",
    "budget_scope": "All waiting counts in the existing policy clock; neither task nor agent clocks are reset.",
    "freshness_guard_modified": False,
    "candidates_modified": False,
    "candidate_scope": "No candidate filtering, truncation, injection, or changed descriptions; normal observation can discover changed DOM candidates.",
    "observer_state_effects": "The original MARKER runs snapshot.js, which may initialize window.__jevFast, allocate bookkeeping node IDs, and refresh its pageKey/guard functions. No custom initialization or observer-cache reset is added.",
}

# This separate read runs after termination. Its output is never passed to the
# policy. Fixed selectors belong to the verifier, not the model/executor.
FINAL_STATE = """(() => ({
  url: location.href, hash: location.hash, title: document.title,
  heading: document.querySelector('h1')?.textContent ?? null,
  category: document.querySelector('#category')?.value ?? null,
  free_cancellation: document.querySelector('#free')?.checked ?? null,
  destination: document.querySelector('#destination')?.value ?? null,
  count: document.querySelector('#count')?.textContent ?? null,
  results: [...document.querySelectorAll('#results h2')].map(e=>e.textContent),
  filter_note: document.querySelector('.fixture-note')?.textContent ?? null,
  empty_message: document.querySelector('.empty')?.textContent ?? null,
  body: document.body.innerText
}))()"""


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def _marker_hash(marker):
    raw = json.dumps(marker, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _valid_marker(marker):
    """The pinned marker is a ten-item semantic array, never an unavailable read."""
    return (
        isinstance(marker, list)
        and len(marker) == 10
        and all(
            type(marker[index]) in (int, float) and math.isfinite(marker[index])
            for index in (0, 2, 3, 4, 5)
        )
        and all(isinstance(marker[index], str) for index in (1, 6, 7))
        and isinstance(marker[8], list)
        and all(isinstance(action, dict) for action in marker[8])
        and isinstance(marker[9], list)
        and all(isinstance(control, list) and len(control) == 6 for control in marker[9])
    )


def wait_for_stable_observation(browser, marker_expression, deadline, *, clock=None, sleep=None):
    """Read the original marker with browser/timing access only; never change policy state."""
    if not isinstance(marker_expression, str) or not marker_expression:
        raise ValueError("the original upstream marker expression is required")
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("a finite monotonic task deadline is required")
    clock, sleep = clock or time.perf_counter, sleep or time.sleep
    started = clock()
    window_deadline = min(started + STABLE_OBSERVATION_POLICY["poll_window_ms"] / 1000, deadline)
    result = {
        "policy": copy.deepcopy(STABLE_OBSERVATION_POLICY),
        "marker_expression_sha256": hashlib.sha256(marker_expression.encode()).hexdigest(),
        "started_monotonic": started,
        "task_deadline_monotonic": deadline,
        "scheduled_window_ms": max(0, window_deadline - started) * 1000,
        "remaining_task_ms_at_start": max(0, deadline - started) * 1000,
        "stable": False,
        "status": None,
        "marker_changes": 0,
        "samples": [],
    }
    previous, quiet_since = None, None
    while (before := clock()) < window_deadline:
        sample = {
            "started_elapsed_ms": (before - started) * 1000,
            "raw_marker_sha256": None,
            "marker_type": None,
            "valid": False,
            "changed": None,
            "error": None,
        }
        marker = None
        try:
            marker = browser.evaluate(marker_expression)
            sample["marker_type"] = type(marker).__name__
            sample["raw_marker_sha256"] = _marker_hash(marker)
            if not _valid_marker(marker):
                raise ValueError(
                    "upstream marker is unavailable or does not match its semantic-array contract"
                )
            sample["valid"] = True
        except Exception as exc:
            message = str(exc)
            sample["error"] = {"type": type(exc).__name__, "message": message[:1000]}
            if len(message) > 1000:
                sample["error"].update(
                    message_truncated=True,
                    message_sha256=hashlib.sha256(message.encode()).hexdigest(),
                )
        observed = clock()
        sample["returned_elapsed_ms"] = (observed - started) * 1000
        sample["read_validate_hash_ms"] = (observed - before) * 1000
        if sample["valid"]:
            sample["changed"] = None if previous is None else marker != previous
            if previous is None or sample["changed"]:
                if previous is not None:
                    result["marker_changes"] += 1
                quiet_since = observed
            previous = copy.deepcopy(marker)
            sample["quiet_ms"] = (observed - quiet_since) * 1000
        else:
            previous, quiet_since = None, None
            sample["quiet_ms"] = 0
        result["samples"].append(sample)
        # A blocking read can return after either soft window or task deadline.
        # Its result remains evidence, but cannot retroactively satisfy quiet time.
        if observed > window_deadline:
            break
        quiet_deadline = (
            quiet_since + STABLE_OBSERVATION_POLICY["quiet_ms"] / 1000
            if quiet_since is not None
            else None
        )
        if quiet_deadline is not None and observed >= quiet_deadline:
            result.update(stable=True, status="stable")
            break
        remaining = window_deadline - observed
        if remaining <= 0:
            break
        pause = min(STABLE_OBSERVATION_POLICY["poll_interval_ms"] / 1000, remaining)
        if quiet_deadline is not None:
            pause = min(pause, max(0, quiet_deadline - observed))
        sleep(pause)
    finished = clock()
    result.update(
        elapsed_ms=(finished - started) * 1000,
        poll_window_overrun_ms=max(0, finished - window_deadline) * 1000,
        task_deadline_overrun_ms=max(0, finished - deadline) * 1000,
        task_deadline_reached=finished >= deadline,
    )
    if result["status"] is None:
        result["status"] = "task_deadline" if finished >= deadline else "poll_window_elapsed"
    return result


def load_upstream(root):
    """Verify the exact checkout, then load it in a private module namespace."""
    root = Path(root).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION or dirty:
        raise ValueError("Jev Ultrafast must be the clean pinned upstream checkout")
    hashes, source_bytes = {}, {}
    for name in SOURCE_FILES:
        raw = (root / name).read_bytes()
        pinned = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{name}"])
        if raw != pinned:
            raise ValueError(f"upstream source mismatch: {name}")
        hashes[name] = hashlib.sha256(raw).hexdigest()
        source_bytes[name] = raw
    alias = "_s1_ultrafast_" + revision[:12]
    for name in tuple(sys.modules):
        if name == alias or name.startswith(alias + "."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        alias,
        root / "jev_ultrafast/__init__.py",
        submodule_search_locations=[str(root / "jev_ultrafast")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        # Build the private package in dependency order from the exact bytes
        # just checked against Git. SourceFileLoader could otherwise reuse an
        # unchecked timestamp-valid .pyc for a child imported by __init__.
        for child in ("questions", "browser", "model", "agent"):
            relative = f"jev_ultrafast/{child}.py"
            path = root / relative
            child_name = alias + "." + child
            child_spec = importlib.util.spec_from_file_location(child_name, path)
            child_module = importlib.util.module_from_spec(child_spec)
            sys.modules[child_name] = child_module
            exec(compile(source_bytes[relative], str(path), "exec"), child_module.__dict__)
            setattr(module, child, child_module)
        init_path = root / "jev_ultrafast/__init__.py"
        exec(
            compile(source_bytes["jev_ultrafast/__init__.py"], str(init_path), "exec"),
            module.__dict__,
        )
    except BaseException:
        for name in tuple(sys.modules):
            if name == alias or name.startswith(alias + "."):
                del sys.modules[name]
        raise
    return {
        "agent": sys.modules[alias + ".agent"],
        "browser": sys.modules[alias + ".browser"],
        "model": sys.modules[alias + ".model"],
    }, {
        "repository": "https://github.com/browser-use/jev-ultrafast",
        "revision": revision,
        "source_sha256": hashes,
        "policy_modified": False,
        "transport": "playwright-cdp",
        "python_source_loading": "compile_verified_bytes_no_pyc",
    }


@contextmanager
def fixture_server(root):
    """Serve only the reviewed fixture's static directory on an ephemeral loopback port."""
    directory = Path(root).resolve() / "jev_ultrafast/static"

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class IsolatedCDP:
    """Route the original executor's CDP calls to owned, disposable browser contexts."""

    def __init__(self, browser_module, output_root=None):
        self.module = browser_module
        self.output_root = Path(output_root) if output_root else None
        self.output = None
        self.targets = {}
        self.sessions = {}
        self.counter = 0

    def __enter__(self):
        from playwright.sync_api import sync_playwright

        self.playwright = sync_playwright().start()
        try:
            self.chromium = self.playwright.chromium.launch(headless=True)
        except BaseException:
            self.playwright.stop()
            raise
        self.original = self.module.cdp, self.module.ensure_daemon
        self.module.cdp = self.call
        self.module.ensure_daemon = lambda: None
        self.browser_version = self.chromium.version
        return self

    def set_output(self, output):
        if self.targets:
            raise ValueError("close the previous task browser before changing output")
        self.output = Path(output)

    def close_task(self):
        """Close any partially initialized owned targets before the next task.

        A Browser constructor may fail before an Agent exists to close it.
        Cleanup still writes that task's trace and never retries an input.
        """
        errors = []
        for target in tuple(self.targets):
            try:
                self.call("Target.closeTarget", targetId=target)
            except Exception as exc:
                errors.append({"target": target, "type": type(exc).__name__, "message": str(exc)})
        if errors and self.output and self.output.is_dir():
            _write(self.output / "cleanup-errors.json", errors)
        return errors

    def call(self, method, session_id=None, **params):
        started = time.perf_counter()
        record = {"method": method, "params": params, "session_id": session_id}
        try:
            if self.output is None or not self.output.is_dir():
                raise ValueError("set an existing task output before creating the browser")
            if method == "Target.createTarget":
                context = self.chromium.new_context(viewport={"width": 1120, "height": 780})
                context.tracing.start(screenshots=True, snapshots=True, sources=True)
                page = context.new_page()
                self.counter += 1
                target = f"isolated-{self.counter}"
                self.targets[target] = context, page
                result = {"targetId": target}
            elif method == "Target.attachToTarget":
                target = params["targetId"]
                context, page = self.targets[target]
                self.sessions[target] = context.new_cdp_session(page)
                result = {"sessionId": target}
            elif method == "Target.closeTarget":
                target = params["targetId"]
                context, _ = self.targets.pop(target)
                try:
                    context.tracing.stop(path=str(self.output / "trace.zip"))
                finally:
                    context.close()
                    self.sessions.pop(target, None)
                result = {"success": True}
            else:
                result = self.sessions[session_id].send(method, params)
            # Screenshots already have durable files/Playwright trace. Avoid
            # duplicating their base64 payload in every protocol log row.
            record["result"] = (
                {"data_omitted": "screenshot preserved separately"}
                if method == "Page.captureScreenshot"
                else result
            )
            return result
        except Exception as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            record["latency_ms"] = (time.perf_counter() - started) * 1000
            if self.output and self.output.is_dir():
                _append(self.output / "cdp.jsonl", record)

    def __exit__(self, *_args):
        try:
            self.close_task()
        finally:
            self.module.cdp, self.module.ensure_daemon = self.original
            self.chromium.close()
            self.playwright.stop()


def verify_final_state(expected, final_state):
    """Exact predeclared predicates over a fresh read, independent of agent DONE."""
    if not expected or not isinstance(expected, dict):
        raise ValueError("task needs nonempty independent final-state predicates")
    allowed = {
        "url",
        "hash",
        "title",
        "heading",
        "category",
        "free_cancellation",
        "destination",
        "count",
        "results",
        "filter_note",
        "empty_message",
        "body_contains",
    }
    if set(expected) - allowed:
        raise ValueError("unknown final-state predicate")
    checks = {}
    for key, wanted in expected.items():
        actual = final_state.get(key) if isinstance(final_state, dict) else None
        if key == "body_contains":
            if (
                not wanted
                or not isinstance(wanted, list)
                or not all(isinstance(s, str) and s for s in wanted)
            ):
                raise ValueError("body_contains must be a nonempty string list")
            actual = final_state.get("body", "") if isinstance(final_state, dict) else ""
            passed = all(value in actual for value in wanted)
        else:
            passed = isinstance(final_state, dict) and key in final_state and actual == wanted
        checks[key] = {"expected": wanted, "actual": actual, "passed": passed}
    return {"verified_success": all(row["passed"] for row in checks.values()), "checks": checks}


def run_task(
    agent_module,
    task,
    url,
    output,
    *,
    max_decisions=12,
    task_timeout=90,
    stable_observation_marker=None,
):
    """Persist all exits, and never feed the independent oracle into the agent.

    The timeout is checked between commands, not a hard interruption of an
    in-process inference. Its overrun remains visible in wall_ms.
    """
    if max_decisions < 1 or task_timeout <= 0:
        raise ValueError("positive task budgets required")
    if stable_observation_marker is not None and (
        not isinstance(stable_observation_marker, str) or not stable_observation_marker
    ):
        raise ValueError("the original upstream marker expression is required")
    verify_final_state(task["expected"], {})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "task.json", task)
    started = time.perf_counter()
    agent, snapshot, final_state = None, None, None
    error, verification_error, screenshot_error, close_error = None, None, None, None
    termination, cycles = "initialization", 0
    policy_wall_ms = None
    stability_records = []

    def save_snapshot(phase):
        nonlocal snapshot
        snapshot = agent.snapshot()
        _append(output / "snapshots.jsonl", {"phase": phase, "snapshot": snapshot})

    try:
        # Only goal and initial URL cross the policy boundary. task.expected
        # stays exclusively in this outer evaluation layer.
        agent = agent_module.Agent(url, task["goal"], screenshots=False)
        save_snapshot("initial")
        while snapshot["status"] not in {"done", "blocked"}:
            if cycles >= max_decisions:
                termination = "decision_budget"
                break
            if time.perf_counter() - started >= task_timeout:
                termination = "task_timeout"
                break
            if stable_observation_marker is not None:
                waiting = wait_for_stable_observation(
                    agent.browser, stable_observation_marker, started + task_timeout
                )
                waiting["before_decision_cycle"] = cycles + 1
                waiting["policy_elapsed_ms_at_return"] = (
                    waiting["started_monotonic"] - started
                ) * 1000 + waiting["elapsed_ms"]
                stability_records.append(waiting)
                _append(output / "observation-stability.jsonl", waiting)
                # Include the wait and its trace I/O in the unchanged task
                # budget. No new tick follows a late synchronous CDP return.
                if time.perf_counter() - started >= task_timeout:
                    termination = "task_timeout"
                    break
            cycles += 1
            agent.command("tick")
            save_snapshot("after_tick")
        else:
            termination = snapshot["status"]
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        termination = "exception"
    finally:
        # Freeze resource use before independent evaluation and cleanup. A
        # long final inference that returns DONE still exceeded its budget.
        policy_wall_ms = (time.perf_counter() - started) * 1000
        if agent is not None:
            try:
                save_snapshot("terminal")
            except Exception as exc:
                if error is None:
                    error = {"type": type(exc).__name__, "message": str(exc)}
            try:
                final_state = agent.browser.evaluate(FINAL_STATE)
                _write(output / "final-state.json", final_state)
            except Exception as exc:
                verification_error = {"type": type(exc).__name__, "message": str(exc)}
            try:
                screenshot = agent.browser.call("Page.captureScreenshot", format="jpeg", quality=80)
                (output / "final.jpg").write_bytes(base64.b64decode(screenshot["data"]))
            except Exception as exc:
                screenshot_error = {"type": type(exc).__name__, "message": str(exc)}
            finally:
                try:
                    agent.close()
                except Exception as exc:
                    close_error = {"type": type(exc).__name__, "message": str(exc)}
    verification = verify_final_state(task["expected"], final_state)
    goal_observed = verification["verified_success"] and verification_error is None
    within_budget = termination != "task_timeout" and policy_wall_ms <= task_timeout * 1000
    success = goal_observed and within_budget
    if success:
        failure_category = None
    elif not within_budget:
        failure_category = "budget_exhaustion"
    elif verification_error:
        failure_category = "verification_error"
    elif termination == "done":
        failure_category = "false_done"
    elif termination == "blocked":
        failure_category = "model_decision_blocked"
    elif termination in {"decision_budget", "task_timeout"}:
        failure_category = "budget_exhaustion"
    else:
        failure_category = "exception"
    summary = {
        "task_id": task["id"],
        "split": task["split"],
        "verified_success": success,
        "goal_observed": goal_observed,
        "within_budget": within_budget,
        "agent_status": snapshot.get("status") if snapshot else None,
        "termination": termination,
        "failure_class": failure_category,
        "wall_ms": (time.perf_counter() - started) * 1000,
        "policy_wall_ms": policy_wall_ms,
        "agent_elapsed_ms": snapshot.get("elapsed_ms") if snapshot else None,
        "decision_cycles": cycles,
        "decision_calls": len(snapshot.get("decisions", [])) if snapshot else 0,
        "text_calls": len(snapshot.get("text_calls", [])) if snapshot else 0,
        "actions": len(snapshot.get("history", [])) if snapshot else 0,
        "reported_cost_usd": None,
        "cost_status": "not_reported",
        "error": error,
        "verification_error": verification_error,
        "screenshot_error": screenshot_error,
        "close_error": close_error,
        "verification": verification,
        "timeout_semantics": "checked between commands; inference is not interrupted",
    }
    if stable_observation_marker is not None:
        summary["stable_observation"] = {
            "enabled": True,
            "policy": copy.deepcopy(STABLE_OBSERVATION_POLICY),
            "marker_expression_sha256": hashlib.sha256(
                stable_observation_marker.encode()
            ).hexdigest(),
            "wait_trace": "observation-stability.jsonl" if stability_records else None,
            "waits": len(stability_records),
            "stable_waits": sum(row["stable"] for row in stability_records),
            "elapsed_ms": math.fsum(row["elapsed_ms"] for row in stability_records),
            "poll_window_overrun_ms": math.fsum(
                row["poll_window_overrun_ms"] for row in stability_records
            ),
            "task_deadline_overrun_ms": math.fsum(
                row["task_deadline_overrun_ms"] for row in stability_records
            ),
            "error_observations": sum(
                sample["error"] is not None
                for row in stability_records
                for sample in row["samples"]
            ),
        }
    _write(output / "summary.json", summary)
    return summary
