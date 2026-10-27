"""Hermetic verifier tests; fabricated protocol entries are not collection evidence."""

import copy

import jev_decision3_link_navigation_collection as collection
import jev_decision3_link_navigation_scenarios as scenarios
import jev_decision3_text_history_collection as original
import pytest

ORIGIN = "http://127.0.0.1:45678"


def scene(mode="landing"):
    return next(spec for spec in scenarios.scenarios(101, {"0" * 64}) if spec["start_mode"] == mode)


def payload():
    return {
        "model": original.MODEL_ALIAS,
        "state": {"page": {"title": "Example"}},
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": "choose",
                "criteria": {"CLICK": "click", "DONE": "done", "BLOCKED": "blocked"},
            },
            "click_target": {
                "type": "choice",
                "instructions": "choose",
                "criteria": {"1": {"element": "first"}, "2": {"element": "target"}},
            },
        },
    }


def test_exact_document_sequence_and_optional_favicon():
    spec = scene()
    guard = collection.EpisodeNetworkGuard(spec, ORIGIN)
    assert guard.check(ORIGIN + spec["paths"]["landing"], "GET")
    assert guard.check(ORIGIN + "/favicon.ico", "GET", main_document=False)
    assert guard.check(ORIGIN + spec["paths"]["workspace"], "HEAD")
    assert guard.check(ORIGIN + spec["paths"]["detail"], "GET")
    guard.verify()
    assert guard.requests == [
        ORIGIN + spec["paths"][key] for key in ("landing", "workspace", "detail")
    ]
    for mode, path_names in (("workspace", ("workspace", "detail")), ("complete", ("detail",))):
        spec = scene(mode)
        guard = collection.EpisodeNetworkGuard(spec, ORIGIN)
        for path_name in path_names:
            assert guard.check(ORIGIN + spec["paths"][path_name], "GET")
        guard.verify()


@pytest.mark.parametrize(
    "kind",
    [
        "other_origin",
        "other_port",
        "decoy",
        "query",
        "fragment",
        "post",
        "redirect",
        "cross_episode",
        "popup",
        "favicon_navigation",
        "wrong_order",
        "encoded_path",
    ],
)
def test_network_guard_refuses_unsafe_or_unplanned_requests(kind):
    spec = scene()
    guard = collection.EpisodeNetworkGuard(spec, ORIGIN)
    url = ORIGIN + spec["start_path"]
    kwargs = {"method": "GET", "redirected": False, "main_document": True}
    if kind == "other_origin":
        url = url.replace("127.0.0.1", "localhost")
    elif kind == "other_port":
        url = url.replace("45678", "45679")
    elif kind == "decoy":
        url = ORIGIN + spec["decoy_paths"][0]
    elif kind == "query":
        url += "?x=1"
    elif kind == "fragment":
        url += "#section"
    elif kind == "post":
        kwargs["method"] = "POST"
    elif kind == "redirect":
        kwargs["redirected"] = True
    elif kind == "cross_episode":
        url = ORIGIN + scene("workspace")["start_path"]
    elif kind == "popup":
        kwargs["main_document"] = False
    elif kind == "favicon_navigation":
        url = ORIGIN + "/favicon.ico"
    elif kind == "wrong_order":
        url = ORIGIN + spec["paths"]["detail"]
    elif kind == "encoded_path":
        url = url.replace("/index", "/%69ndex")
    assert not guard.check(url, **kwargs)
    with pytest.raises(ValueError, match="requests differ"):
        guard.verify()


@pytest.mark.parametrize(
    "bad",
    [
        "http://localhost:1234",
        "https://127.0.0.1:1234",
        "http://127.0.0.1",
        "http://user@127.0.0.1:1234",
        "http://127.0.0.1:1234/",
        "http://127.0.0.1:1234?x=1",
    ],
)
def test_origin_requires_exact_loopback_authority(bad):
    with pytest.raises(ValueError):
        collection.EpisodeNetworkGuard(scene(), bad)


def test_native_click_row_preserves_offered_targets_and_quarantine():
    wire = payload()
    before = copy.deepcopy(wire)
    arguments = dict(
        row_id="row",
        scenario_id="scene",
        ancestor_id="ancestor",
        episode_id="episode",
        step=0,
        operation="CLICK",
        target="2",
        proof_refs={"transition": "episode#0"},
    )
    row = collection._verified_training_row(wire, **arguments)
    assert wire == before == row["payload"]
    assert row["gold"] == {"operation": "CLICK", "click_target": "2"}
    assert row["loss_mask"] == {"operation": True, "click_target": True}
    assert (
        row["fit_eligible"] is False and row["admission_status"] == "pending_independent_admission"
    )
    old = original.build_native_training_row(wire, **arguments)
    assert old["loss_mask"] == {"operation": True, "click_target": False}
    with pytest.raises(ValueError):
        collection._verified_training_row(wire, **(arguments | {"target": "3"}))
    done = collection._verified_training_row(
        wire, **(arguments | {"operation": "DONE", "target": None})
    )
    assert done["loss_mask"] == {"operation": True, "click_target": False}


def test_observed_menu_is_not_reordered_or_injected():
    spec = scene()
    label = spec["target_link"]
    actions = [
        {"id": "a1", "node": 31, "kind": "click", "label": "other"},
        {"id": "a2", "node": 42, "kind": "click", "label": label},
    ]
    page = {
        "url": ORIGIN + spec["start_path"],
        "title": "title",
        "text": "text",
        "actions": actions,
    }
    descriptor = {
        "operation": "CLICK",
        "kind": "click",
        "label": label,
        "action_id": None,
        "text": None,
    }
    target, action = collection._selected_action(page, descriptor)
    assert target == "2" and action == actions[1]
    wire = collection._publisher_wire(
        page, spec["goal"], [], {"NEXT_ACTION": "rules", "TARGET": "rules"}
    )
    assert list(wire["questions"]["click_target"]["criteria"]) == ["1", "2"]
    assert wire["questions"]["click_target"]["criteria"]["2"]["element"] == "[2] " + label
    page["actions"].append({"id": "a3", "node": 43, "kind": "click", "label": label})
    with pytest.raises(ValueError, match="one actual"):
        collection._selected_action(page, descriptor)


def click_executor(monkeypatch):
    spec = scene()
    action = {"id": "target-id", "node": 42, "kind": "click", "label": spec["target_link"]}
    page = {"page_key": "page-id", "guards": {"42": "guard"}}
    descriptor = {
        "operation": "CLICK",
        "kind": "click",
        "label": action["label"],
        "action_id": None,
        "text": None,
    }
    wire = payload()
    point = {"x": 1, "y": 2}
    freshness = "(() => { const c=window.__jevFast; return c ? [c.pageKey(),c.guard(c.nodes.get(42))] : null; })()"

    def call(seq, method, params, value=None):
        result = {"sequence": seq, "session": "session", "method": method, "params": params}
        result["response"] = {"result": {"value": value}} if method == "Runtime.evaluate" else {}
        return result

    journal = [
        call(0, "Runtime.evaluate", {"expression": "read"}, {}),
        call(1, "Runtime.evaluate", {"expression": "private"}, {}),
        call(2, "Runtime.evaluate", {"expression": freshness}, ["page-id", "guard"]),
        call(3, "Runtime.evaluate", {"expression": "target-expression"}, point),
        call(
            4,
            "Input.dispatchMouseEvent",
            {"type": "mousePressed", **point, "button": "left", "clickCount": 1},
        ),
        call(
            5,
            "Input.dispatchMouseEvent",
            {"type": "mouseReleased", **point, "button": "left", "clickCount": 1},
        ),
        call(6, "Runtime.evaluate", {"expression": "read"}, {}),
        call(7, "Runtime.evaluate", {"expression": "private"}, {}),
    ]
    decision = {
        "request": wire,
        "operation": "CLICK",
        "target": "2",
        "choice": "target-id",
        "raw_answers": collection._teacher_response(wire, "CLICK", "2")["answers"],
    }
    step = {
        "scenario": spec,
        "teacher": descriptor,
        "action": action,
        "target": "2",
        "payload": wire,
        "page": page,
        "tick_start": 0,
        "tick_end": 8,
        "private_before_call": 1,
        "observation_after_call": 6,
        "private_after_call": 7,
        "decision": decision,
        "field_calls": [],
        "fresh_calls": [
            {"action": action, "page": page, "cdp_start": 2, "cdp_end": 3, "result": True}
        ],
        "acts": [
            {
                "action": action,
                "page": page,
                "cdp_start": 2,
                "cdp_end": 6,
                "result": {"executed": "target-id"},
            }
        ],
    }
    monkeypatch.setattr(collection, "_native_action_expression", lambda *_: "target-expression")
    return step, journal


def test_click_executor_joins_fresh_native_target_and_real_mouse_sequence(monkeypatch):
    step, journal = click_executor(monkeypatch)
    assert collection._verify_executor(step, journal, "unused", "read")["native_guarded_action"]
    for mutation in ("guard", "point", "press", "choice"):
        changed_step, changed_journal = copy.deepcopy(step), copy.deepcopy(journal)
        if mutation == "guard":
            changed_journal[2]["response"]["result"]["value"][1] = "stale"
        elif mutation == "point":
            changed_journal[5]["params"]["x"] = 9
        elif mutation == "press":
            changed_journal[4]["params"]["type"] = "mouseMoved"
        else:
            changed_step["decision"]["choice"] = "some-other-id"
        with pytest.raises(ValueError):
            collection._verify_executor(changed_step, changed_journal, "unused", "read")


def test_pending_detail_cannot_be_labeled_done():
    spec = scene("workspace")
    state = {
        "pathname": spec["paths"]["detail"],
        "view": "detail",
        "ready": "complete",
        "heading": spec["record_label"],
        "status": spec["pending_text"],
        "count": 0,
        "links": [],
        "events": [],
    }
    assert scenarios.teacher_step(spec, state)["operation"] == "CLICK"
    assert not scenarios.verify_goal(spec, state)
    forged_done = {
        "operation": "DONE",
        "kind": "terminal",
        "label": None,
        "action_id": None,
        "text": None,
    }
    with pytest.raises(ValueError):
        scenarios.verify_transition(spec, state, state, forged_done, [])


def test_native_page_and_private_state_must_belong_to_same_owned_document():
    spec = scene()
    state = {"pathname": spec["paths"]["landing"]}
    after = {"pathname": spec["paths"]["workspace"]}
    step = {
        "scenario": spec,
        "before": state,
        "after": after,
        "page": {"url": ORIGIN + state["pathname"]},
        "page_after": {"url": ORIGIN + after["pathname"]},
    }
    collection._verify_document_urls(step, ORIGIN)
    for change in ("other_port", "state_path", "query"):
        modified = copy.deepcopy(step)
        if change == "other_port":
            modified["page_after"]["url"] = modified["page_after"]["url"].replace("45678", "45679")
        elif change == "state_path":
            modified["after"]["pathname"] = spec["decoy_paths"][0]
            modified["page_after"]["url"] = ORIGIN + modified["after"]["pathname"]
        else:
            modified["page_after"]["url"] += "?x=1"
        with pytest.raises(ValueError):
            collection._verify_document_urls(modified, ORIGIN)


def test_episode_ownership_rejects_foreign_session_target_and_setup():
    spec = scene("complete")
    state = {"pathname": spec["start_path"]}
    page = {"url": ORIGIN + spec["start_path"]}
    step = {
        "before": state,
        "after": state,
        "page": page,
        "page_after": page,
        "scenario": spec,
        "tick_start": 6,
        "tick_end": 7,
        "teacher": {"operation": "DONE"},
    }

    def row(method, params, session="owned", response=None):
        return {"method": method, "params": params, "session": session, "response": response or {}}

    journal = [
        row(
            "Target.createTarget",
            {"url": "about:blank", "background": True},
            None,
            {"targetId": "target"},
        ),
        row(
            "Target.attachToTarget",
            {"targetId": "target", "flatten": True},
            None,
            {"sessionId": "owned"},
        ),
        row(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1120, "height": 780, "deviceScaleFactor": 1, "mobile": False},
        ),
        row("Emulation.setFocusEmulationEnabled", {"enabled": True}),
        row("Page.navigate", {"url": ORIGIN + spec["start_path"]}),
        row("Runtime.evaluate", {"expression": "read"}),
        row("Runtime.evaluate", {"expression": "terminal"}),
        row("Target.closeTarget", {"targetId": "target"}, None, {"success": True}),
    ]
    trace = {"cdp_start": 0, "cdp_end": 8, "scenario": spec, "completed": True, "steps": [step]}
    collection._verify_ownership(trace, journal, ORIGIN, "read")
    for change in (
        "foreign_session",
        "foreign_target",
        "bad_setup",
        "bad_navigation",
        "early_done",
    ):
        t, j = copy.deepcopy(trace), copy.deepcopy(journal)
        if change == "foreign_session":
            j[6]["session"] = "other"
        elif change == "foreign_target":
            j[-1]["params"]["targetId"] = "other"
        elif change == "bad_setup":
            j[5]["params"]["expression"] = "unexpected-script"
        elif change == "bad_navigation":
            j[4]["params"]["url"] += "?query"
        else:
            t["steps"][0]["teacher"]["operation"] = "CLICK"
        with pytest.raises(ValueError):
            collection._verify_ownership(t, j, ORIGIN, "read")
