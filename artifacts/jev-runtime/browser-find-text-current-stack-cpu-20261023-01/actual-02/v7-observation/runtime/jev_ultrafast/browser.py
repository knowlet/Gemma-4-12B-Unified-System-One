"""Observed actions through Browser Harness; one CDP session, no per-step subprocess."""

import hashlib
import json
import sys
import time
from pathlib import Path

from browser_harness.admin import ensure_daemon
from browser_harness.helpers import cdp

# Atomically read visible content and controls, preserving actual DOM node identity.
READ_STATE = Path(__file__).with_name("snapshot.js").read_text()
MARKER = f"(() => {{ const state={READ_STATE}; return state?.marker ?? null; }})()"

class StalePage(ValueError):
    """A decision no longer refers to the observed page."""


class Browser:
    def __init__(self, url):
        ensure_daemon()
        self.target = cdp("Target.createTarget", url="about:blank", background=True)["targetId"]
        self.session = cdp("Target.attachToTarget", targetId=self.target, flatten=True)["sessionId"]
        self.call("Emulation.setDeviceMetricsOverride", width=1120, height=780, deviceScaleFactor=1, mobile=False)
        # Keep rAF/menus rendering in an owned background tab, without activating the user's Chrome tab.
        self.call("Emulation.setFocusEmulationEnabled", enabled=True)
        self.call("Page.navigate", url=url)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.evaluate("document.readyState") == "complete":
                break
            time.sleep(0.02)

    def call(self, method, **params):
        return cdp(method, session_id=self.session, **params)

    def evaluate(self, expression):
        response = self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        if response.get("exceptionDetails"):
            raise StalePage("Document changed during evaluation")
        return response.get("result", {}).get("value")

    def observe(self, screenshot=True):
        _settle_started = time.monotonic()
        _settle_deadline = _settle_started + 5.0
        _settle_operations = []
        _settle_retries = 0
        _settle_closed = False
        _settle_ready_swallowed = False
        _settle_had_input = bool(getattr(self, "after_input", None))
        _settle_class = type(self)
        if getattr(_settle_class, "_observation_settling_fault", False):
            raise RuntimeError("Navigation settling diagnostic integrity failed")
        _settle_call_id = len(getattr(_settle_class, "observation_settling_records", ())) + 1

        def _settle_record(status, late_snapshot=False, error=None):
            elapsed = max(0.0, time.monotonic() - _settle_started)
            row = {
                "observe_call_id": _settle_call_id,
                "target_seconds": 5.0, "scope": "per_observe_call", "status": status,
                "started_monotonic": _settle_started, "deadline_monotonic": _settle_deadline,
                "elapsed_seconds": elapsed, "overshoot_seconds": max(0.0, elapsed - 5.0),
                "attempts": len(_settle_operations), "navigation_retries": _settle_retries,
                "operations": list(_settle_operations), "after_input_consumed": _settle_had_input,
                "baseline_after_input_runtime_error_swallowed": _settle_ready_swallowed,
                "late_snapshot": late_snapshot, "terminal_unknown": status != "success",
                "synchronous_cdp_hard_deadline": False,
                "operation_timing_scope": "native_observe_combined_not_individual_CDP",
                "error_type": type(error).__name__ if error is not None else None,
                "error": str(error) if error is not None else None,
            }
            self.observation_settling = row
            return row

        def _settle_finish(status, late_snapshot=False, error=None):
            nonlocal _settle_closed
            row = _settle_record(status, late_snapshot, error)
            encoded = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
            records = getattr(_settle_class, "observation_settling_records", ())
            # Charge an extra 1KiB for the final timeout/timing scalar updates.
            total = getattr(_settle_class, "_observation_settling_bytes", 0) + len(encoded) + 1024
            if len(records) >= 4096 or total > 64 * 1024 * 1024:
                _settle_class._observation_settling_fault = True
                raise RuntimeError("Navigation settling diagnostic bound exceeded")
            frozen = json.loads(encoded)
            elapsed = max(0.0, time.monotonic() - _settle_started)
            frozen.update(elapsed_seconds=elapsed, overshoot_seconds=max(0.0, elapsed - 5.0))
            self.observation_settling = frozen
            self.observation_settling_history = getattr(self, "observation_settling_history", ()) + (frozen,)
            _settle_class.observation_settling_records = records + (frozen,)
            _settle_class._observation_settling_bytes = total
            _settle_closed = True
            return frozen

        def _settle_timeout(late_snapshot=False, row=None):
            if row is None:
                row = _settle_finish("timeout", late_snapshot)
            elapsed = max(0.0, time.monotonic() - _settle_started)
            row.update(status="timeout", late_snapshot=late_snapshot, terminal_unknown=True,
                       elapsed_seconds=elapsed, overshoot_seconds=max(0.0, elapsed - 5.0))
            exc = StalePage("Navigation settling target exceeded: elapsed=%.6fs overshoot=%.6fs" % (
                row["elapsed_seconds"], row["overshoot_seconds"]))
            exc.observation_settling = json.loads(json.dumps(row))
            raise exc

        _settle_record("started")
        try:
            if getattr(self, "after_input", None):
                action, self.after_input = self.after_input, None
                # This is read-only and happens after execution was logged, even if navigation interrupts it.
                try:
                    self.call(
                        "Runtime.evaluate",
                        expression="""(action => new Promise(resolve => {
                          const field=window.__jevFast?.nodes.get(action.node);
                          const autocomplete=action.kind==='fill' && field?.getAttribute('role')==='combobox';
                          let frames=0, stopped=false;
                          const finish=()=>{stopped=true;resolve()};
                          setTimeout(finish,autocomplete ? 200 : 50);
                          const ready=()=>{
                            if (stopped) return;
                            const ids=(field?.getAttribute('aria-controls')||field?.getAttribute('aria-owns')||'')
                              .split(/\\s+/).filter(Boolean);
                            const roots=ids.length ? ids.map(id=>document.getElementById(id)).filter(Boolean) : [document];
                            const options=roots.flatMap(root=>[...root.querySelectorAll('[role="option"]')]);
                            if (++frames>=2 && (!autocomplete || options.some(e=>{
                              const r=e.getBoundingClientRect();
                              return r.width && r.height && r.bottom>0 && r.top<innerHeight &&
                                e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
                            }))) finish();
                            else requestAnimationFrame(ready);
                          };
                          requestAnimationFrame(ready);
                        }))(""" + json.dumps(action) + ")",
                        awaitPromise=True,
                        returnByValue=True,
                    )
                except RuntimeError:
                    _settle_ready_swallowed = True
                    pass
            while True:
                if time.monotonic() >= _settle_deadline:
                    _settle_timeout()
                if len(_settle_operations) >= 1024:
                    raise RuntimeError("Navigation settling native-operation bound exceeded")
                _operation_started = time.monotonic()
                _operation = {"started_monotonic": _operation_started,
                              "native_protocol_call_count": None,
                              "returned_snapshot_sha256": None, "null_navigation": False}
                try:
                    _settle_page = browser_operation(
                        {"operation": "observe", "session": self.session, "screenshot": screenshot}
                    )
                except BaseException as exc:
                    _operation_ended = time.monotonic()
                    _operation.update(ended_monotonic=_operation_ended,
                                      elapsed_seconds=max(0.0, _operation_ended - _operation_started),
                                      error_type=type(exc).__name__, error=str(exc))
                    _settle_operations.append(_operation)
                    if type(exc) is not StalePage or exc.args != ("Document is navigating",):
                        raise
                    _operation["null_navigation"] = True
                    _settle_retries += 1
                    _settle_record("transient_navigation", error=exc)
                    remaining = _settle_deadline - time.monotonic()
                    if remaining <= 0:
                        _settle_timeout()
                    time.sleep(min(0.02, remaining))
                    continue
                _operation_ended = time.monotonic()
                _operation.update(ended_monotonic=_operation_ended,
                                  elapsed_seconds=max(0.0, _operation_ended - _operation_started),
                                  error_type=None, error=None)
                _settle_operations.append(_operation)
                _operation["returned_snapshot_sha256"] = hashlib.sha256(json.dumps(
                    _settle_page, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
                if time.monotonic() >= _settle_deadline:
                    _settle_timeout(late_snapshot=True)
                _settle_final = _settle_finish("success")
                _settle_final_elapsed = max(0.0, time.monotonic() - _settle_started)
                _settle_final.update(elapsed_seconds=_settle_final_elapsed,
                                     overshoot_seconds=max(0.0, _settle_final_elapsed - 5.0))
                if _settle_final_elapsed >= 5.0:
                    _settle_timeout(late_snapshot=True, row=_settle_final)
                return _settle_page
        except BaseException as exc:
            if not _settle_closed:
                try:
                    _settle_finish("unexpected_error", error=exc)
                except BaseException as secondary:
                    _settle_class._observation_settling_fault = True
                    exc.add_note("Navigation diagnostic finalization failed: " + repr(secondary))
            raise

    def fresh(self, page, action=None):
        if action is not None and action["kind"] in {"click", "select"}:
            node = action["node"]
            if type(node) is not int:
                return False
            current = self.evaluate(
                "(() => { const c=window.__jevFast; "
                f"return c ? [c.pageKey(),c.guard(c.nodes.get({node}))] : null; }})()"
            )
            return current == [page["page_key"], page["guards"].get(str(node))]
        return self.evaluate(MARKER) == page["marker"]

    def act(self, action, page, text=None):
        if not self.fresh(page, action):
            raise StalePage("Page changed since this decision. Observe again.")
        if action["kind"] == "wait":
            time.sleep(0.1)
        result = browser_operation({"operation": "act", "session": self.session, "action": action, "text": text})
        self.after_input = action if action["kind"] != "wait" else None
        return result

    def close(self):
        if self.target:
            cdp("Target.closeTarget", targetId=self.target)
            self.target = None


def fingerprint(state):
    content = {k: state[k] for k in ("url", "text", "actions", "scroll")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def browser_operation(request):
    operation = request["operation"]
    session = request["session"]

    def call(method, **params):
        return cdp(method, session_id=session, **params)

    def evaluate(expression):
        result = call("Runtime.evaluate", expression=expression, returnByValue=True)
        if result.get("exceptionDetails"):
            if operation == "act" and request["action"]["kind"] == "select":
                raise RuntimeError("Dropdown execution was interrupted; inspect before retrying.")
            raise StalePage("Document changed during evaluation")
        return result.get("result", {}).get("value")

    if operation == "act":
        action = request["action"]
        kind = action["kind"]
        if kind == "scroll":
            call("Input.dispatchMouseEvent", type="mouseWheel", x=550, y=650, deltaX=0, deltaY=action["delta"])
        elif kind != "wait":
            if type(action["node"]) is not int:
                raise ValueError("Invalid observed node")
            # Code-owned node IDs refer to actual observed elements, never model-generated selectors.
            target = evaluate("""(action => {
              const e=window.__jevFast?.nodes.get(action.node);
              if (!e?.isConnected || e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]') ||
                  !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return null;
              if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true')) return null;
              const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2;
              if (!r.width || !r.height || x<0 || y<0 || x>=innerWidth || y>=innerHeight) return null;
              if (!e.contains(document.elementFromPoint(x,y))) return null;
              if (action.kind==='select') {
                if (e.tagName!=='SELECT' || ![...e.options].some(o=>o.value===action.value &&
                    !o.disabled && !o.closest('optgroup[disabled]'))) return null;
                e.value=action.value;
                e.dispatchEvent(new Event('input',{bubbles:true}));
                e.dispatchEvent(new Event('change',{bubbles:true}));
              }
              return {x,y};
            })(""" + json.dumps(action) + ")")
            if target is None:
                if kind == "select":
                    raise RuntimeError("Dropdown execution was not confirmed; inspect before retrying.")
                raise StalePage("Target changed or is covered. Observe again.")
            if kind != "select":
                x, y = target["x"], target["y"]
                for event in ("mousePressed", "mouseReleased"):
                    call("Input.dispatchMouseEvent", type=event, x=x, y=y, button="left", clickCount=1)
                if kind == "fill":
                    call(
                        "Input.dispatchKeyEvent",
                        type="keyDown",
                        key="a",
                        code="KeyA",
                        modifiers=4 if sys.platform == "darwin" else 2,
                        commands=["selectAll"],
                    )
                    call(
                        "Input.dispatchKeyEvent",
                        type="keyUp",
                        key="a",
                        code="KeyA",
                        modifiers=4 if sys.platform == "darwin" else 2,
                    )
                    call("Input.insertText", text=request["text"])
        return {"executed": action["id"]}

    info = evaluate(READ_STATE)
    if info is None:
        raise StalePage("Document is navigating")
    info["fingerprint"] = fingerprint(info)
    if request.get("screenshot", True):
        info["screenshot"] = call("Page.captureScreenshot", format="jpeg", quality=72)["data"]
    return info
