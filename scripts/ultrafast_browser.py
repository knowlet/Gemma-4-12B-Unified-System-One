"""Unmodified Jev Ultrafast policy on isolated Chromium, with independent outcomes.

Only the CDP transport is substituted. The upstream snapshot, freshness guards,
operation/target selection and action executor remain the pinned source bytes.
The local fixture is public integration screening, not a real-site leaderboard.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
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
    hashes = {}
    for name in SOURCE_FILES:
        raw = (root / name).read_bytes()
        pinned = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{name}"])
        if raw != pinned:
            raise ValueError(f"upstream source mismatch: {name}")
        hashes[name] = hashlib.sha256(raw).hexdigest()
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
        spec.loader.exec_module(module)
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


def run_task(agent_module, task, url, output, *, max_decisions=12, task_timeout=90):
    """Persist all exits, and never feed the independent oracle into the agent.

    The timeout is checked between commands, not a hard interruption of an
    in-process inference. Its overrun remains visible in wall_ms.
    """
    if max_decisions < 1 or task_timeout <= 0:
        raise ValueError("positive task budgets required")
    verify_final_state(task["expected"], {})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "task.json", task)
    started = time.perf_counter()
    agent, snapshot, final_state = None, None, None
    error, verification_error, screenshot_error, close_error = None, None, None, None
    termination, cycles = "initialization", 0
    policy_wall_ms = None

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
    _write(output / "summary.json", summary)
    return summary
