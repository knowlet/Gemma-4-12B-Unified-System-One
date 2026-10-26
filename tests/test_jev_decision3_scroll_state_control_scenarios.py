"""Invented DOM/event controls; these are not collected browser/model evidence."""

import copy
import importlib.util
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import pytest

_SOURCE = (
    Path(__file__).resolve().parents[1] / "scripts/jev_decision3_scroll_state_control_scenarios.py"
)
_SPEC = importlib.util.spec_from_file_location("scroll_state_controls", _SOURCE)
scene = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scene)
EXCLUSIONS = {"0" * 64}


def specs(seed=20261026):
    return scene.scenarios(seed, EXCLUSIONS)


def one(prefix):
    return next(spec for spec in specs() if spec["family"].startswith(prefix + "-options"))


def initial(spec):
    current = spec["initial"]
    modes = {"": scene.SELECT_PLACEHOLDER, **{o["value"]: o["label"] for o in spec["options"]}}
    block_ids = {
        "radio": [r["id"] for r in spec["radios"]],
        "text": ["entry-text"],
        "select": ["entry-select"],
        "submit": ["commit-control"],
        **{link["id"]: [link["id"]] for link in spec["navigation_links"]},
    }
    ids = [identity for token in spec["layout_order"] for identity in block_ids[token]]
    return {
        "url": "http://127.0.0.1/tasks/" + spec["id"],
        "title": spec["title"],
        "viewport_width": 1120,
        "viewport_height": 780,
        "document_height": 2600,
        "scroll_y": 0,
        "control_boxes": [
            {
                "id": identity,
                "left": 10 + 100 * (i % 4),
                "right": 90 + 100 * (i % 4),
                "top": 80 + 40 * (i // 4),
                "bottom": 110 + 40 * (i // 4),
                "visible": True,
            }
            for i, identity in enumerate(ids)
        ],
        "radios": [
            dict(r, name=scene.RADIO_GROUP, checked=r["value"] == current["radio_value"])
            for r in spec["radios"]
        ],
        "navigation": copy.deepcopy(spec["navigation_links"]),
        "commit_tag": "a" if spec["commit_kind"] == "link" else "button",
        "note_value": current["note_value"],
        "mode_value": current["mode_value"],
        "mode_label": modes[current["mode_value"]],
        "submitted": False,
        "submission_count": 0,
        "status": "No entry submitted.",
        "submitted_radio_value": "",
        "submitted_radio_label": "",
        "submitted_note": "",
        "submitted_mode_value": "",
        "submitted_mode_label": "",
        "events": [],
    }


def event(kind, target, trusted, value=None, checked=None):
    return {"type": kind, "target": target, "trusted": trusted, "value": value, "checked": checked}


def advance(spec, before, *, submit_blur=False):
    """Invent an effect from native control semantics; never a browser substitute."""
    step = scene.teacher_step(spec, before)
    after, events = copy.deepcopy(before), []
    op = step["operation"]
    if op == "TYPE_TEXT":
        after.update(
            note_value=spec["desired_text"], submitted=False, status="Edited; submit the entry."
        )
        events = [event("input", "entry-text", True, spec["desired_text"])]
    elif op == "SELECT":
        after.update(
            mode_value=spec["desired_option_value"],
            mode_label=spec["desired_option_label"],
            submitted=False,
            status="Edited; submit the entry.",
        )
        events = [
            event(kind, "entry-select", False, spec["desired_option_value"])
            for kind in ("input", "change")
        ]
    elif op == "CLICK" and step["label"] == spec["desired_radio_label"]:
        for radio in after["radios"]:
            radio["checked"] = radio["value"] == spec["desired_radio_value"]
        after.update(submitted=False, status="Edited; submit the entry.")
        selected = next(r for r in spec["radios"] if r["value"] == spec["desired_radio_value"])
        events = [
            event(kind, selected["id"], True, selected["value"], True)
            for kind in ("click", "input", "change")
        ]
    elif op == "CLICK":
        selected = next((r for r in after["radios"] if r["checked"]), None)
        count = before["submission_count"] + 1
        after.update(
            submitted=True,
            submission_count=count,
            status=f"Submitted entry {count}.",
            submitted_radio_value=selected["value"] if selected else "",
            submitted_radio_label=selected["label"] if selected else "",
            submitted_note=after["note_value"],
            submitted_mode_value=after["mode_value"],
            submitted_mode_label=after["mode_label"],
        )
        if submit_blur:
            events.append(event("change", "entry-text", True, before["note_value"]))
        events.append(
            event("click", "commit-control", True, "" if spec["commit_kind"] == "button" else None)
        )
    for index, item in enumerate(events, len(before["events"])):
        item["sequence"] = index
    after["events"].extend(copy.deepcopy(events))
    return step, after, events


def complete(spec):
    current = initial(spec)
    for _ in range(6):
        step, after, events = advance(spec, current)
        scene.verify_transition(spec, current, after, step, events)
        current = after
        if step["operation"] == "DONE":
            return current
    raise AssertionError("invented authored route did not finish")


class Nodes(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.nodes = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.nodes.append((tag, dict(attrs)))


def test_independent_namespace_twelve_scenes_sizes_and_detached_generation():
    first = specs()
    assert len(first) == 12 and scene.EXPECTED_MENU_COUNTS == (7,)
    assert scene.SCHEMA == "decision3-owned-scroll-state-control-scenarios-v1"
    assert scene.NAMESPACE == "decision3-owned-scroll-state-control-training-v1"
    assert len({s["id"] for s in first}) == len({s["ancestor_id"] for s in first}) == 12
    assert len({s["goal_sha256"] for s in first}) == 12
    assert Counter(len(s["options"]) for s in first) == {12: 4, 16: 4, 24: 4}
    assert all(s["split"] == "train" and s["expected_menu_counts"] == [7] for s in first)
    detached = specs()
    detached[0]["radios"][0]["label"] = "invented mutation"
    assert specs() == first and specs(20261027) != first


def test_all_explicit_routes_and_post_radio_editor_states_have_exact_prospective_counts():
    expected = {
        "checked-text": ["TYPE_TEXT", "CLICK", "DONE"],
        "checked-select": ["SELECT", "CLICK", "DONE"],
        "radio-text-select": ["CLICK", "TYPE_TEXT", "SELECT", "CLICK", "DONE"],
        "radio-select-text": ["CLICK", "SELECT", "TYPE_TEXT", "CLICK", "DONE"],
    }
    counts = Counter()
    for spec in specs():
        route = spec["family"].split("-options")[0]
        current, operations = initial(spec), []
        assert not scene.verify_goal(spec, current)
        for wanted in expected[route]:
            step, after, events = advance(spec, current)
            assert step["operation"] == wanted
            if wanted in {"TYPE_TEXT", "SELECT"}:
                assert any(
                    r["checked"] and r["value"] == spec["desired_radio_value"]
                    for r in current["radios"]
                )
                assert after["radios"] == current["radios"]
            proof = scene.verify_transition(spec, current, after, step, events)
            assert proof["positive_supervision"] is True
            assert proof["requires_canonical_native_executor_join"] is (wanted != "DONE")
            operations.append(wanted)
            current = after
        assert operations == expected[route] and scene.verify_goal(spec, current)
        counts.update(operations)
    assert counts == {"CLICK": 18, "TYPE_TEXT": 9, "SELECT": 9, "DONE": 12}
    assert sum(counts.values()) == 48


def test_render_has_real_tall_document_enabled_checked_radios_large_select_and_no_teacher_hints():
    for spec in specs():
        text = scene.render_fixture(spec)
        nodes = Nodes(text).nodes
        assert "overflow-y:auto" in text and "min-height:1800px" in text
        assert "overflow:hidden" not in text and "position:fixed" not in text
        assert text.index("</main>") < text.index("class='reference-notes'")
        radios = [a for tag, a in nodes if tag == "input" and a.get("type") == "radio"]
        assert len(radios) == len(spec["radios"])
        assert all("disabled" not in a and "hidden" not in a for a in radios)
        assert sum("checked" in a for a in radios) == int(
            spec["initial"]["radio_value"] is not None
        )
        options = [a for tag, a in nodes if tag == "option"]
        assert len(options) == spec["authored_option_count"] + 1
        assert sum("disabled" not in a for a in options) == spec["authored_option_count"]
        assert len([a for tag, a in nodes if a.get("class") == "navigation-link"]) == len(
            spec["navigation_links"]
        )
        for hidden_hint in (
            scene.NAMESPACE,
            spec["family"],
            "goal_order",
            "required_fields",
            "desired_radio_value",
            "teacher_step",
            "SCROLL_DOWN",
            "BACK",
        ):
            assert hidden_hint not in text


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_id", "other"),
        ("namespace", "other"),
        ("split", "dev"),
        ("expected_menu_counts", [8]),
        ("goal", "invented alternative"),
        ("authored_option_count", 8),
        ("seed", True),
    ],
)
def test_mutated_or_cross_profile_specs_cannot_render_or_teach(field, value):
    original = one("checked-text")
    changed = copy.deepcopy(original)
    changed[field] = value
    with pytest.raises(ValueError):
        scene.render_fixture(changed)
    with pytest.raises(ValueError):
        scene.teacher_step(changed, initial(original))


def test_exclusions_validation_and_collision_refusal():
    with pytest.raises(ValueError):
        scene.scenarios(20261026, set())
    with pytest.raises(ValueError):
        scene.scenarios(True, EXCLUSIONS)
    with pytest.raises(ValueError):
        scene.scenarios(20261026, {specs()[0]["goal_sha256"]})
    with pytest.raises(ValueError):
        scene.scenarios(20261026, {"not a hash"})


@pytest.mark.parametrize(
    "field,value",
    [
        ("document_height", 780),
        ("scroll_y", 1),
        ("scroll_y", False),
        ("viewport_width", True),
        ("viewport_height", 0),
    ],
)
def test_fresh_top_geometry_is_required_for_teacher_and_goal(field, value):
    spec = one("checked-text")
    state = initial(spec)
    state[field] = value
    with pytest.raises(ValueError):
        scene.teacher_step(spec, state)
    assert scene.verify_goal(spec, state) is False


@pytest.mark.parametrize(
    "corruption", ["missing", "order", "hidden", "offscreen", "nan", "identity", "extra"]
)
def test_all_actual_control_boxes_must_remain_complete_visible_and_ordered(corruption):
    spec = one("checked-select")
    state = initial(spec)
    boxes = state["control_boxes"]
    if corruption == "missing":
        boxes.pop()
    elif corruption == "order":
        boxes.reverse()
    elif corruption == "hidden":
        boxes[0]["visible"] = False
    elif corruption == "offscreen":
        boxes[0]["bottom"] = state["viewport_height"] + 1
    elif corruption == "nan":
        boxes[0]["left"] = float("nan")
    elif corruption == "identity":
        boxes[0]["id"] = "invented different control"
    else:
        boxes[0]["extra"] = True
    with pytest.raises(ValueError):
        scene.teacher_step(spec, state)


@pytest.mark.parametrize("corruption", ["string", "multiple", "wrong_value", "irrelevant_edit"])
def test_actual_radio_booleans_exclusivity_identity_and_preserved_controls(corruption):
    spec = one("checked-text")
    state = initial(spec)
    if corruption == "string":
        state["radios"][0]["checked"] = "false"
    elif corruption == "multiple":
        for r in state["radios"]:
            r["checked"] = True
    elif corruption == "wrong_value":
        state["radios"][0]["value"] = "invented value"
    else:
        state["mode_value"] = spec["desired_option_value"]
        if state["mode_value"] == spec["initial"]["mode_value"]:
            state["mode_value"] = ""
            state["mode_label"] = scene.SELECT_PLACEHOLDER
        else:
            state["mode_label"] = spec["desired_option_label"]
    with pytest.raises(ValueError):
        scene.teacher_step(spec, state)


@pytest.mark.parametrize(
    "corruption", ["untrusted", "target", "value", "extra", "prefix", "sequence"]
)
def test_text_effect_requires_native_input_and_unchanged_event_prefix(corruption):
    spec = one("checked-text")
    before = initial(spec)
    step, after, events = advance(spec, before)
    if corruption == "untrusted":
        events[0]["trusted"] = False
    elif corruption == "target":
        events[0]["target"] = "entry-select"
    elif corruption == "value":
        events[0]["value"] = "invented wrong text"
    elif corruption == "extra":
        events.append({**event("click", "commit-control", True), "sequence": 1})
    elif corruption == "sequence":
        events[0]["sequence"] = 7
    else:
        before["events"] = [{**event("click", "navigation-1", True), "sequence": 0}]
    after["events"] = copy.deepcopy(events)
    with pytest.raises(ValueError):
        scene.verify_transition(spec, before, after, step, events)


def test_select_requires_original_synthetic_pair_and_separate_native_executor_join():
    spec = one("checked-select")
    before = initial(spec)
    step, after, events = advance(spec, before)
    proof = scene.verify_transition(spec, before, after, step, events)
    assert proof["requires_native_select_executor_join"] is True
    assert proof["select_event_isTrusted"] is False
    for bad in ([events[0]], [{**row, "trusted": True} for row in events]):
        forged = {**after, "events": bad}
        with pytest.raises(ValueError):
            scene.verify_transition(spec, before, forged, step, bad)


@pytest.mark.parametrize("corruption", ["untrusted", "order", "missing", "value", "extra"])
def test_radio_progress_requires_exact_trusted_trio_and_cannot_supervise_repeat(corruption):
    spec = one("radio-text-select")
    before = initial(spec)
    step, after, events = advance(spec, before)
    scene.verify_transition(spec, before, after, step, events)
    if corruption == "untrusted":
        events[0]["trusted"] = False
    elif corruption == "order":
        events[0]["type"], events[1]["type"] = events[1]["type"], events[0]["type"]
    elif corruption == "missing":
        events.pop()
    elif corruption == "value":
        events[1]["value"] = "invented wrong radio"
    else:
        events.append({**event("click", "navigation-1", True), "sequence": 3})
    after["events"] = copy.deepcopy(events)
    with pytest.raises(ValueError):
        scene.verify_transition(spec, before, after, step, events)
    checked = initial(spec)
    for radio in checked["radios"]:
        radio["checked"] = radio["value"] == spec["desired_radio_value"]
    with pytest.raises(ValueError):
        scene.verify_transition(spec, checked, checked, step, [])


def test_submit_admits_only_exact_commit_or_preceding_unchanged_trusted_text_blur():
    spec = one("checked-text")
    before = advance(spec, initial(spec))[1]
    for blur in (False, True):
        step, after, events = advance(spec, before, submit_blur=blur)
        scene.verify_transition(spec, before, after, step, events)
        assert scene.verify_goal(spec, after)
    step, after, events = advance(spec, before, submit_blur=True)
    corruptions = (
        [{**events[0], "value": "wrong"}, events[1]],
        [{**events[0], "trusted": False}, events[1]],
        [{**events[0], "type": "input"}, events[1]],
        [{**events[0], "target": spec["radios"][0]["id"]}, events[1]],
        [events[1], events[0]],
        [events[0], events[0], events[1]],
    )
    for bad in corruptions:
        bad = copy.deepcopy(bad)
        for index, row in enumerate(bad, len(before["events"])):
            row["sequence"] = index
        forged = {**after, "events": [*before["events"], *bad]}
        with pytest.raises(ValueError):
            scene.verify_transition(spec, before, forged, step, bad)


@pytest.mark.parametrize(
    "field",
    [
        "submitted_radio_value",
        "submitted_radio_label",
        "submitted_note",
        "submitted_mode_value",
        "submitted_mode_label",
        "status",
        "submitted",
    ],
)
def test_done_rechecks_every_actual_saved_value_and_rejects_stale_or_premature_state(field):
    spec = one("radio-select-text")
    finished = complete(spec)
    changed = copy.deepcopy(finished)
    changed[field] = False if field == "submitted" else "invented wrong record"
    assert scene.verify_goal(spec, changed) is False
    done = scene.teacher_step(spec, finished)
    with pytest.raises(ValueError):
        scene.verify_transition(spec, finished, changed, done, [])
    edited = {**finished, "note_value": finished["note_value"] + " changed", "submitted": False}
    assert scene.verify_goal(spec, edited) is False
    with pytest.raises(ValueError):
        scene.verify_transition(spec, initial(spec), initial(spec), done, [])
