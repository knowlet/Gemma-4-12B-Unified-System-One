"""Real native-browser collection for independent owned link-navigation episodes.

Rows are quarantined pending a separate admission/fit workflow. The controlled
server and network guard operate only inside a fresh per-episode browser context.
Synthetic teacher choices are execution instructions, never model scores.
"""

from __future__ import annotations

import copy
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import jev_decision3_link_navigation_scenarios as selected_scenarios
import jev_decision3_text_history_collection as base

SCHEMA = "decision3-owned-link-navigation-collection-v1"
DEFAULT_SCENARIO_PROFILE = "link-navigation-v1"
MAX_TICKS = base.MAX_TICKS
_encode = base._encode
_action_space = base._action_space
_publisher_wire = base._publisher_wire
_publisher_rules = base._publisher_rules
_selected_action = base._selected_action
_teacher_response = base._teacher_response
_runtime_value = base._runtime_value
_latest_observation = base._latest_observation
_verify_freshness = base._verify_freshness
_native_action_expression = base._native_action_expression
_field_context = base._field_context
_NoHTTP = base._NoHTTP


def _scenario_module(profile):
    if profile != DEFAULT_SCENARIO_PROFILE:
        raise ValueError("Only the independent link-navigation profile is supported")
    return selected_scenarios


def _validate_scenario_spec(spec, profile):
    _scenario_module(profile)._check_spec(spec)


def _verified_training_row(*args, **kwargs):
    # Call only after the actual executor and transition have been verified.
    row = base.build_native_training_row(*args, **kwargs)
    if row["teacher_choice"]["operation"] == "CLICK":
        if "click_target" not in row["loss_mask"]:
            raise ValueError("Original wire must offer the native click target head")
        row["loss_mask"]["click_target"] = True
    return row


def _verify_executor(
    step, journal, upstream, read_state, *, scenario_profile=DEFAULT_SCENARIO_PROFILE
):
    selected = _scenario_module(scenario_profile)
    descriptor, action = step["teacher"], step["action"]
    operation = descriptor["operation"]
    calls = journal[step["tick_start"] : step["tick_end"]]
    if any(
        call.get("sequence") != index for index, call in enumerate(calls, start=step["tick_start"])
    ):
        raise ValueError("Native tick protocol sequence changed")
    if any("error" in call for call in calls):
        raise ValueError("A native tick with a failed protocol call is quarantined")
    fresh = step["fresh_calls"]
    if not fresh:
        raise ValueError("Every original freshness guard must have succeeded")
    for entry in fresh:
        if not step["tick_start"] <= entry["cdp_start"] < entry["cdp_end"] <= step["tick_end"]:
            raise ValueError("Freshness proof lies outside its original tick")
        _verify_freshness(entry, journal, read_state)
    inputs = [call for call in calls if call["method"].startswith("Input.")]
    decision = step["decision"]
    if (
        _encode(decision["request"]) != _encode(step["payload"])
        or decision["operation"] != operation
        or decision["target"] != step["target"]
        or decision["raw_answers"]
        != _teacher_response(step["payload"], operation, step["target"])["answers"]
    ):
        raise ValueError("Original choose returned a different full request or choice")
    if operation == "DONE":
        if (
            action is not None
            or step["acts"]
            or inputs
            or step["status"] != "done"
            or step["field_calls"]
        ):
            raise ValueError("DONE requires original terminal handling without browser input")
        if decision["choice"] != "DONE" or not selected.verify_goal(
            step["scenario"], step["after"]
        ):
            raise ValueError("DONE lacks independently checked goal completion")
        terminal_guards = [
            item
            for item in fresh
            if item.get("action") is None
            and item.get("page") == step["page"]
            and step["private_before_call"] < item["cdp_start"]
            and item["cdp_end"] == step["private_after_call"]
        ]
        if len(terminal_guards) != 1:
            raise ValueError("DONE requires its original post-decision terminal freshness guard")
        return {"native_terminal": True, "input_calls": 0}
    if operation == "BLOCKED" or action is None or len(step["acts"]) != 1:
        raise ValueError("One actual guarded original action is required")
    execution = step["acts"][0]
    if (
        execution.get("action") != action
        or execution.get("page") != step["page"]
        or execution.get("result") != {"executed": action["id"]}
        or decision["choice"] != action["id"]
        or "error" in execution
    ):
        raise ValueError("Actual original Browser.act did not execute the selected observed ID")
    if not (
        step["tick_start"]
        <= step["private_before_call"]
        < execution["cdp_start"]
        < execution["cdp_end"]
        <= step["observation_after_call"]
        < step["private_after_call"]
        < step["tick_end"]
    ):
        raise ValueError("Execution must follow its precondition and precede post-observation")
    matching_guards = [
        item
        for item in fresh
        if item.get("action") == action
        and item.get("page") == step["page"]
        and item["cdp_start"] == execution["cdp_start"]
        and item["cdp_end"] == execution["cdp_start"] + 1
    ]
    if len(matching_guards) != 1:
        raise ValueError("The actual action lacks its original immediate freshness guard")
    relevant = journal[execution["cdp_start"] : execution["cdp_end"]]
    if inputs != [call for call in relevant if call["method"].startswith("Input.")]:
        raise ValueError("Execution input must lie inside the actual Browser.act span")
    canonical_methods = {
        "wait": ["Runtime.evaluate"],
        "scroll": ["Runtime.evaluate", "Input.dispatchMouseEvent"],
        "select": ["Runtime.evaluate", "Runtime.evaluate"],
        "click": [
            "Runtime.evaluate",
            "Runtime.evaluate",
            "Input.dispatchMouseEvent",
            "Input.dispatchMouseEvent",
        ],
        "fill": [
            "Runtime.evaluate",
            "Runtime.evaluate",
            "Input.dispatchMouseEvent",
            "Input.dispatchMouseEvent",
            "Input.dispatchKeyEvent",
            "Input.dispatchKeyEvent",
            "Input.insertText",
        ],
    }
    if [call["method"] for call in relevant] != canonical_methods.get(action["kind"]):
        raise ValueError("Execution differs from the canonical guard/executor/input sequence")
    target_point = None
    if action["kind"] in {"click", "fill", "select"}:
        expression = _native_action_expression(upstream, action)
        evaluations = [
            call
            for call in relevant
            if call["method"] == "Runtime.evaluate"
            and call["params"].get("expression") == expression
        ]
        if len(evaluations) != 1 or evaluations[0]["sequence"] != execution["cdp_start"] + 1:
            raise ValueError("Action must join the exact canonical native DOM hit/executor")
        target_point = _runtime_value(journal, evaluations[0]["sequence"], expression)
        if (
            type(target_point) is not dict
            or set(target_point) != {"x", "y"}
            or any(type(value) not in (int, float) for value in target_point.values())
        ):
            raise ValueError("The original hit test must return its actual target coordinates")
    if action["kind"] != "fill" and step["field_calls"]:
        raise ValueError("Only the actual native fill action can invoke the field writer")
    if action["kind"] == "select":
        if inputs:
            raise ValueError("Pinned SELECT does not generate trusted CDP input")
        return {
            "native_select_executor_join": True,
            "select_input_change_events_trusted": False,
            "input_calls": 0,
        }
    if action["kind"] == "scroll":
        expected = {
            "type": "mouseWheel",
            "x": 550,
            "y": 650,
            "deltaX": 0,
            "deltaY": action["delta"],
        }
        if (
            len(inputs) != 1
            or inputs[0]["method"] != "Input.dispatchMouseEvent"
            or inputs[0]["params"] != expected
        ):
            raise ValueError("Scroll proof must retain the original native wheel delta")
    elif action["kind"] in {"click", "fill"}:
        mouse = [call for call in inputs if call["method"] == "Input.dispatchMouseEvent"]
        expected_mouse = [
            {"type": event, **target_point, "button": "left", "clickCount": 1}
            for event in ("mousePressed", "mouseReleased")
        ]
        if [call["params"] for call in mouse] != expected_mouse:
            raise ValueError("A trusted click must join the original CDP press and release")
        inserts = [call for call in inputs if call["method"] == "Input.insertText"]
        if action["kind"] == "fill":
            if (
                len(inserts) != 1
                or inserts[0]["params"] != {"text": descriptor["text"]}
                or execution["text"] != descriptor["text"]
            ):
                raise ValueError("Field proof must join the literal actually typed by native input")
            expected_field = {
                "context": _field_context(
                    step["scenario"]["goal"], action, step["page"], step["history_before"]
                ),
                "text": descriptor["text"],
            }
            if step["field_calls"] != [expected_field]:
                raise ValueError("Actual field literal/context evidence is incomplete")
            keys = [call["params"] for call in inputs if call["method"] == "Input.dispatchKeyEvent"]
            if len(keys) != 2 or keys[0].get("modifiers") not in {2, 4}:
                raise ValueError("Native fill must perform its original select-all keys")
            modifiers = keys[0]["modifiers"]
            if (
                keys
                != [
                    {
                        "type": "keyDown",
                        "key": "a",
                        "code": "KeyA",
                        "modifiers": modifiers,
                        "commands": ["selectAll"],
                    },
                    {"type": "keyUp", "key": "a", "code": "KeyA", "modifiers": modifiers},
                ]
                or len(inputs) != 5
            ):
                raise ValueError("Unexpected input outside the native fill executor")
        elif inserts:
            raise ValueError("CLICK cannot synthesize field input")
        elif len(inputs) != 2:
            raise ValueError("Unexpected input outside the native click executor")
    elif action["kind"] == "wait":
        if inputs:
            raise ValueError("WAIT cannot issue input")
    else:
        raise ValueError("Unsupported native executor kind")
    return {"native_guarded_action": True, "input_calls": len(inputs)}


def _verify_step(step, journal, upstream, read_state, *, scenario_profile=DEFAULT_SCENARIO_PROFILE):
    """Replay raw observation, native publisher, executor and independent DOM joins."""
    spec = step["scenario"]
    selected = _scenario_module(scenario_profile)
    _validate_scenario_spec(spec, scenario_profile)
    if spec["split"] != "train":
        raise ValueError("Only new train-only scenarios are allowed")
    if (
        _runtime_value(journal, step["private_before_call"], selected.private_state_expression())
        != step["before"]
    ):
        raise ValueError("Private precondition lacks its actual raw evaluation")
    if (
        _runtime_value(journal, step["private_after_call"], selected.private_state_expression())
        != step["after"]
    ):
        raise ValueError("Private postcondition lacks its actual raw evaluation")
    raw_page = _runtime_value(journal, step["observation_call"], read_state)
    if any(
        raw_page.get(key) != value for key, value in step["page"].items() if key != "fingerprint"
    ):
        raise ValueError("Original wire page lacks its original READ_STATE join")
    if (
        _latest_observation(journal[: step["private_before_call"]], read_state, step["page"])
        != step["observation_call"]
    ):
        raise ValueError("Wire observation is not the latest original observation")
    session = journal[step["observation_call"]]["session"]
    if not session or any(
        call["session"] != session for call in journal[step["tick_start"] : step["tick_end"]]
    ):
        raise ValueError("Native tick/private evidence crossed the owned browser session")
    if (
        not step["observation_call"]
        < step["private_before_call"]
        < step["private_after_call"]
        < step["tick_end"]
    ):
        raise ValueError("Observation and independent pre/postconditions have invalid ordering")
    if _encode(
        _publisher_wire(
            step["page"], spec["goal"], step["history_before"], _publisher_rules(upstream)
        )
    ) != _encode(step["payload"]):
        raise ValueError(
            "The full native publisher wire changed, including ordering or instructions"
        )
    descriptor = selected.teacher_step(spec, step["before"])
    if descriptor != step["teacher"]:
        raise ValueError("Teacher annotation differs from the authored independent route")
    target, action = _selected_action(step["page"], descriptor)
    if target != step["target"] or action != step["action"]:
        raise ValueError("Teacher did not choose the actual offered native target/control")
    before_events, after_events = step["before"]["events"], step["after"]["events"]
    if after_events[: len(before_events)] != before_events:
        raise ValueError("Owned event history was replaced")
    effect = selected.verify_transition(
        spec, step["before"], step["after"], descriptor, after_events[len(before_events) :]
    )
    execution = _verify_executor(
        step, journal, upstream, read_state, scenario_profile=scenario_profile
    )
    if effect.get("requires_native_select_executor_join") and not execution.get(
        "native_select_executor_join"
    ):
        raise ValueError("Synthetic SELECT events require their canonical executor proof")
    before_history, after_history = step["history_before"], step["history_after"]
    if descriptor["operation"] == "DONE":
        if after_history != before_history:
            raise ValueError("Native DONE must not append or mutate action history")
    else:
        if len(after_history) != len(before_history) + 1 or after_history[:-1] != before_history:
            raise ValueError("Original Agent action history was not appended exactly once")
        if (
            _latest_observation(
                journal[: step["private_after_call"]], read_state, step["page_after"]
            )
            != step["observation_after_call"]
        ):
            raise ValueError("Post-action page lacks its actual original native observation")
        expected_history = {
            "step": len(after_history),
            "action": action["label"],
            "kind": action["kind"],
            "choice": action["id"],
            "text": descriptor["text"],
            "operation": descriptor["operation"],
            "target": target,
            "page_changed": step["page_after"]["fingerprint"] != step["page"]["fingerprint"],
        }
        if any(after_history[-1].get(key) != value for key, value in expected_history.items()):
            raise ValueError(
                "Original Agent history differs from the actual selected action/effect"
            )
    return {"effect": effect, "execution": execution}


def _validate_origin(origin):
    try:
        parsed = urlsplit(origin)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is None
            or parsed.path
            or parsed.query
            or parsed.fragment
            or origin != f"http://127.0.0.1:{parsed.port}"
        ):
            raise ValueError("Expected exact HTTP loopback origin and port")
    except ValueError as exc:
        raise ValueError("Expected exact HTTP loopback origin and port") from exc


def _verify_document_urls(step, origin):
    _validate_origin(origin)
    for name, private in (("page", "before"), ("page_after", "after")):
        page, state = step[name], step[private]
        if page.get("url") != origin + state.get("pathname", ""):
            raise ValueError("Native observed document and private state URL differ")
    allowed = set(step["scenario"]["paths"].values())
    if step["before"]["pathname"] not in allowed or step["after"]["pathname"] not in allowed:
        raise ValueError("A step escaped its episode's authored documents")


@contextmanager
def fixture_server(spec):
    """Serve precisely one episode's authored documents without redirects."""
    selected_scenarios._check_spec(spec)
    pages = {
        path: html.encode("utf-8")
        for path, html in selected_scenarios.render_documents(spec).items()
    }

    class Handler(BaseHTTPRequestHandler):
        def respond(self, body):
            page = pages.get(self.path)
            self.send_response(200 if page is not None else 404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if body:
                self.wfile.write(page if page is not None else b"Not found")

        def do_GET(self):
            self.respond(True)

        def do_HEAD(self):
            self.respond(False)

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
        thread.join(timeout=2)


class EpisodeNetworkGuard:
    """Permit only sequential, authored main-document requests for one episode."""

    def __init__(self, spec, origin):
        selected_scenarios._check_spec(spec)
        _validate_origin(origin)
        self.origin = origin
        paths = spec["paths"]
        self.expected = {
            "landing": [paths["landing"], paths["workspace"], paths["detail"]],
            "workspace": [paths["workspace"], paths["detail"]],
            "complete": [paths["detail"]],
        }[spec["start_mode"]]
        self.requests = []
        self.violations = []

    def check(self, url, method, *, redirected=False, main_document=True):
        expected = (
            self.origin + self.expected[len(self.requests)]
            if len(self.requests) < len(self.expected)
            else None
        )
        # Chromium may fetch a missing favicon as a subresource. It cannot be a
        # navigation and the fixed loopback server responds with an empty 404.
        if (
            url == self.origin + "/favicon.ico"
            and method in {"GET", "HEAD"}
            and not redirected
            and not main_document
        ):
            return True
        if (
            type(url) is not str
            or method not in {"GET", "HEAD"}
            or redirected
            or not main_document
            or url != expected
        ):
            self.violations.append({"url": url, "method": method})
            return False
        self.requests.append(url)
        return True

    def route(self, route):
        request = route.request
        try:
            main = request.is_navigation_request() and request.frame.parent_frame is None
            allowed = self.check(
                request.url,
                request.method,
                redirected=request.redirected_from is not None,
                main_document=main,
            )
        except Exception:
            self.violations.append({"url": "unavailable", "method": "unavailable"})
            allowed = False
        if allowed:
            route.continue_()
        else:
            route.abort("blockedbyclient")

    def verify(self):
        if self.violations or self.requests != [self.origin + path for path in self.expected]:
            raise ValueError("Episode requests differ from the exact owned document sequence")


def collect_owned_episode(
    *, spec, episode_id, transport, agent_module, model_module, upstream, read_state
):
    """Run a real episode in a dedicated empty context, then quarantine its evidence.

    The caller supplies a RecordingCDP bound to a new Playwright context and the
    already verified native runtime. A separate admission step is still required.
    """
    context = transport.browser
    if context.pages or transport.targets:
        raise ValueError("An episode requires a fresh empty browser context")
    with fixture_server(spec) as origin:
        guard = EpisodeNetworkGuard(spec, origin)
        pages = []

        def on_page(page):
            pages.append(page)

        context.on("page", on_page)
        context.route("**/*", guard.route)
        try:
            trace = _collect_episode(
                spec=spec,
                episode_id=episode_id,
                origin=origin,
                transport=transport,
                agent_module=agent_module,
                model_module=model_module,
                upstream=upstream,
                read_state=read_state,
            )
            guard.verify()
            if len(pages) != 1 or context.pages or transport.targets:
                raise ValueError("Extra, unclosed or popup browser targets are forbidden")
            _verify_ownership(trace, transport.journal, origin, read_state)
            trace["network"] = copy.deepcopy(guard.requests)
            return trace
        finally:
            context.unroute("**/*", guard.route)
            context.remove_listener("page", on_page)


def _verify_ownership(trace, journal, origin, read_state):
    start, end = trace["cdp_start"], trace["cdp_end"]
    if not (type(start) is int and type(end) is int and 0 <= start < end <= len(journal)):
        raise ValueError("Invalid owned episode protocol span")
    calls = journal[start:end]
    if any("error" in call for call in calls):
        raise ValueError("Failed protocol call cannot be admitted")
    created, attached, closed = calls[0], calls[1], calls[-1]
    target = created.get("response", {}).get("targetId")
    session = attached.get("response", {}).get("sessionId")
    if (
        created["method"] != "Target.createTarget"
        or created["params"] != {"url": "about:blank", "background": True}
        or not target
        or attached["method"] != "Target.attachToTarget"
        or attached["params"] != {"targetId": target, "flatten": True}
        or not session
        or closed["method"] != "Target.closeTarget"
        or closed["params"] != {"targetId": target}
        or closed.get("response") != {"success": True}
        or any(call["session"] != session for call in calls[2:-1])
    ):
        raise ValueError("Target, session, or close ownership mismatch")
    navigations = [call for call in calls if call["method"] == "Page.navigate"]
    if len(navigations) != 1 or navigations[0]["params"] != {
        "url": origin + trace["scenario"]["start_path"]
    }:
        raise ValueError("Initial navigation differs from the owned document")
    steps = trace["steps"]
    if not trace["completed"] or not steps or steps[-1]["tick_end"] != end - 1:
        raise ValueError("Completed terminal step and closed episode required")
    if steps[0]["before"]["pathname"] != trace["scenario"]["start_path"]:
        raise ValueError("First state differs from the authored starting document")
    if steps[-1]["teacher"]["operation"] != "DONE":
        raise ValueError("A terminal DONE step is required")
    setup = journal[start + 2 : steps[0]["tick_start"]]
    expected_setup = [
        (
            "Emulation.setDeviceMetricsOverride",
            {"width": 1120, "height": 780, "deviceScaleFactor": 1, "mobile": False},
        ),
        ("Emulation.setFocusEmulationEnabled", {"enabled": True}),
        ("Page.navigate", {"url": origin + trace["scenario"]["start_path"]}),
    ]
    if [(call["method"], call["params"]) for call in setup[:3]] != expected_setup:
        raise ValueError("Unexpected native browser setup")
    reads = setup[3:]
    if not reads or any(
        call["method"] != "Runtime.evaluate"
        or call["params"].get("expression") not in {"document.readyState", read_state}
        for call in reads
    ):
        raise ValueError("Unexpected protocol execution before the first tick")
    for step in steps:
        if not start < step["tick_start"] < step["tick_end"] < end:
            raise ValueError("Step escaped owned protocol span")
        _verify_document_urls(step, origin)
    for previous, current in zip(steps, steps[1:]):
        if (
            previous["tick_end"] != current["tick_start"]
            or previous["teacher"]["operation"] == "DONE"
            or previous["after"] != current["before"]
        ):
            raise ValueError("Episode steps are not contiguous")


def _collect_episode(
    *,
    spec,
    episode_id,
    origin,
    transport,
    agent_module,
    model_module,
    upstream,
    read_state,
    scenario_profile=DEFAULT_SCENARIO_PROFILE,
):
    """Run original Agent ticks; retain failed attempts and never retry a stale action."""
    selected = _scenario_module(scenario_profile)
    _validate_scenario_spec(spec, scenario_profile)
    _validate_origin(origin)
    trace = {
        "schema_id": SCHEMA,
        "scenario_profile": scenario_profile,
        "episode_id": episode_id,
        "scenario": copy.deepcopy(spec),
        "steps": [],
        "rows": [],
        "completed": False,
        "status": "pending",
        "errors": [],
        "cdp_start": len(transport.journal),
    }
    agent = None
    saved_post, saved_model_field = model_module.post_json, model_module.field_text
    saved_agent_field, saved_client = agent_module.field_text, model_module.CLIENT
    pending = {}
    raw_fresh, raw_act = None, None
    settling_start = len(getattr(agent_module.Browser, "observation_settling_records", ()))
    try:
        model_module.CLIENT = _NoHTTP()
        agent = agent_module.Agent(origin + spec["start_path"], spec["goal"], screenshots=False)
        raw_fresh, raw_act = agent.browser.fresh, agent.browser.act

        def private():
            start = len(transport.journal)
            value = agent.browser.evaluate(selected.private_state_expression())
            if len(transport.journal) != start + 1:
                raise ValueError("A private proof must have one actual protocol evaluation")
            return value, start

        def fresh(page, action=None):
            entry = {
                "page": copy.deepcopy(page),
                "action": copy.deepcopy(action),
                "cdp_start": len(transport.journal),
            }
            pending.setdefault("fresh_calls", []).append(entry)
            try:
                result = raw_fresh(page, action)
                entry["result"] = result
                return result
            except BaseException as exc:
                entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
                raise
            finally:
                entry["cdp_end"] = len(transport.journal)

        def act(action, page, text=None):
            entry = {
                "action": copy.deepcopy(action),
                "page": copy.deepcopy(page),
                "text": text,
                "cdp_start": len(transport.journal),
            }
            pending.setdefault("acts", []).append(entry)
            try:
                result = raw_act(action, page, text=text)
                entry["result"] = copy.deepcopy(result)
                return result
            except BaseException as exc:
                entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
                raise
            finally:
                entry["cdp_end"] = len(transport.journal)

        def teacher_post(url, _key, payload):
            if url != "https://api.typesafe.ai/v1/systemone" or pending.get("policy_calls", 0):
                raise ValueError("Exactly one original policy publication is allowed per tick")
            pending["policy_calls"] = 1
            page, history = (
                copy.deepcopy(agent.state["page"]),
                copy.deepcopy(agent.state["history"]),
            )
            if model_module.action_space(page["actions"]) != _action_space(page["actions"]):
                raise ValueError("Pinned publisher action_space differs from independent replay")
            expected = _publisher_wire(page, spec["goal"], history, _publisher_rules(upstream))
            if _encode(payload) != _encode(expected):
                raise ValueError("The original publisher request was altered")
            before, before_call = private()
            descriptor = selected.teacher_step(spec, before)
            target, action = _selected_action(page, descriptor)
            pending.update(
                {
                    "payload": copy.deepcopy(payload),
                    "page": page,
                    "history_before": history,
                    "before": before,
                    "private_before_call": before_call,
                    "observation_call": _latest_observation(transport.journal, read_state, page),
                    "teacher": descriptor,
                    "target": target,
                    "action": copy.deepcopy(action),
                }
            )
            # Validate all heads, including unused and singleton heads, before act.
            base.build_native_training_row(
                payload,
                row_id="pending",
                scenario_id=spec["id"],
                ancestor_id=spec["ancestor_id"],
                episode_id=episode_id,
                step=len(trace["steps"]),
                operation=descriptor["operation"],
                target=target,
                proof_refs={"transition": "pending"},
            )
            return _teacher_response(payload, descriptor["operation"], target)

        def field_text(context):
            action, descriptor = pending.get("action"), pending.get("teacher", {})
            if not action or action["kind"] != "fill" or descriptor.get("operation") != "TYPE_TEXT":
                raise ValueError("A field writer requires the original selected fill action")
            expected = model_module.field_context(
                spec["goal"], action, agent.state["page"], agent.state["history"]
            )
            if context != expected or type(descriptor.get("text")) is not str:
                raise ValueError("Field literal must join the original native field context")
            pending.setdefault("field_calls", []).append(
                {"context": copy.deepcopy(context), "text": descriptor["text"]}
            )
            return descriptor["text"], {
                "model": "scripted-owned-field-literal",
                "latency_ms": 0,
                "usage": {},
            }

        agent.browser.fresh, agent.browser.act = fresh, act
        model_module.post_json = teacher_post
        model_module.field_text = agent_module.field_text = field_text
        for index in range(MAX_TICKS):
            pending.clear()
            pending.update(
                {
                    "tick_start": len(transport.journal),
                    "fresh_calls": [],
                    "acts": [],
                    "field_calls": [],
                }
            )
            try:
                agent.command("tick")
                if pending.get("policy_calls") != 1:
                    raise ValueError("Original Agent did not publish exactly one policy request")
                after, after_call = private()
                step = {
                    **copy.deepcopy(pending),
                    "index": index,
                    "scenario": copy.deepcopy(spec),
                    "after": after,
                    "private_after_call": after_call,
                    "tick_end": len(transport.journal),
                    "decision": copy.deepcopy(agent.state["decisions"][-1]),
                    "history_after": copy.deepcopy(agent.state["history"]),
                    "page_after": copy.deepcopy(agent.state["page"]),
                    "observation_after_call": _latest_observation(
                        transport.journal, read_state, agent.state["page"]
                    ),
                    "status": agent.state["status"],
                }
                _verify_document_urls(step, origin)
                step["proof"] = _verify_step(
                    step,
                    transport.journal,
                    upstream,
                    read_state,
                    scenario_profile=scenario_profile,
                )
                trace["steps"].append(step)
                trace["rows"].append(
                    _verified_training_row(
                        step["payload"],
                        row_id=episode_id + f".step-{index:02d}",
                        scenario_id=spec["id"],
                        ancestor_id=spec["ancestor_id"],
                        episode_id=episode_id,
                        step=index,
                        operation=step["teacher"]["operation"],
                        target=step["target"],
                        proof_refs={"transition": episode_id + f"#step-{index}"},
                    )
                )
                if agent.state["status"] == "done":
                    trace.update(completed=True, status="completed")
                    break
                if agent.state["status"] == "blocked":
                    raise ValueError(
                        "Original Agent blocked; do not convert failure to a teacher label"
                    )
            except BaseException:
                trace["failed_attempt"] = copy.deepcopy(pending)
                raise
        if not trace["completed"]:
            raise ValueError("Original 16-tick collection budget exhausted")
    except BaseException as exc:
        trace["status"] = "failed"
        trace["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        model_module.post_json, model_module.field_text = saved_post, saved_model_field
        agent_module.field_text, model_module.CLIENT = saved_agent_field, saved_client
        trace["agent_closed"] = agent is None
        if agent is not None:
            if raw_fresh is not None:
                agent.browser.fresh, agent.browser.act = raw_fresh, raw_act
            try:
                agent.close()
                trace["agent_closed"] = agent.browser.target is None
            except BaseException as exc:
                trace["errors"].append(
                    {"type": type(exc).__name__, "message": str(exc), "phase": "agent_close"}
                )
                trace.update(completed=False, status="failed")
        trace["observed_menu_counts"] = sorted(
            {len(row["payload"]["questions"]["operation"]["criteria"]) for row in trace["rows"]}
        )
        trace["cdp_end"] = len(transport.journal)
        if hasattr(agent_module.Browser, "observation_settling_records"):
            trace["observation_settling"] = {
                "records": copy.deepcopy(
                    agent_module.Browser.observation_settling_records[settling_start:]
                ),
                "fault": bool(getattr(agent_module.Browser, "_observation_settling_fault", False)),
            }
            if trace["observation_settling"]["fault"]:
                trace["errors"].append(
                    {"type": "ValueError", "message": "Original V7 settling diagnostics faulted"}
                )
                trace.update(completed=False, status="failed")
        trace["model_score"] = None
        trace["fit_eligible"] = False
    return trace
