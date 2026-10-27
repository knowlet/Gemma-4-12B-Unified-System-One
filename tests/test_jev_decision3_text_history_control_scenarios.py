"""Invented controls only; no collected browser, policy or DEV evidence."""

import copy
import hashlib
import importlib.util
import json
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1] / "scripts/jev_decision3_text_history_control_scenarios.py"
)
SPEC = importlib.util.spec_from_file_location("text_history_controls", SOURCE)
scene = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scene)
EXCLUSIONS = {"0" * 64}


def specs(seed=20261027):
    return scene.scenarios(seed, EXCLUSIONS)


def one(family):
    return next(item for item in specs() if item["family"] == family + "-options12")


def initial(spec):
    current = spec["initial"]
    blocks = {
        "radio": [radio["id"] for radio in spec["radios"]],
        "text": ["draft-annotation"],
        "select": ["draft-tier"],
        "submit": ["record-draft"],
        **{link["id"]: [link["id"]] for link in spec["navigation_links"]},
    }
    ids = [identity for block in spec["layout_order"] for identity in blocks[block]]
    labels = {
        "": scene.SELECT_PLACEHOLDER,
        **{option["value"]: option["label"] for option in spec["options"]},
    }
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
                "left": 10 + 100 * (index % 4),
                "right": 90 + 100 * (index % 4),
                "top": 80 + 40 * (index // 4),
                "bottom": 110 + 40 * (index // 4),
                "visible": True,
            }
            for index, identity in enumerate(ids)
        ],
        "radios": [
            dict(radio, name=scene.RADIO_GROUP, checked=radio["value"] == current["radio_value"])
            for radio in spec["radios"]
        ],
        "navigation": copy.deepcopy(spec["navigation_links"]),
        "commit_tag": "a" if spec["commit_kind"] == "link" else "button",
        "note_value": current["note_value"],
        "mode_value": current["mode_value"],
        "mode_label": labels[current["mode_value"]],
        "text_refresh_count": 0,
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


def effect(spec, before, action):
    """Independent invented control effects; action is supplied by the test."""
    after, events = copy.deepcopy(before), []

    def event(kind, target, trusted, value=None, checked=None):
        events.append(
            {
                "sequence": len(before["events"]) + len(events),
                "type": kind,
                "target": target,
                "trusted": trusted,
                "value": value,
                "checked": checked,
            }
        )

    if action == "type":
        after["note_value"] = spec["desired_text"]
        event("input", "draft-annotation", True, spec["desired_text"])
        descriptor = {
            "operation": "TYPE_TEXT",
            "kind": "fill",
            "label": spec["field_label"],
            "action_id": None,
            "text": spec["desired_text"],
        }
    elif action == "radio":
        choice = next(
            radio for radio in spec["radios"] if radio["value"] == spec["desired_radio_value"]
        )
        for radio in after["radios"]:
            radio["checked"] = radio["value"] == choice["value"]
        for kind in ("click", "input", "change"):
            event(kind, choice["id"], True, choice["value"], True)
        descriptor = {
            "operation": "CLICK",
            "kind": "click",
            "label": choice["label"],
            "action_id": None,
            "text": None,
        }
    elif action == "select":
        after.update(
            mode_value=spec["desired_option_value"], mode_label=spec["desired_option_label"]
        )
        for kind in ("input", "change"):
            event(kind, "draft-tier", False, spec["desired_option_value"])
        descriptor = {
            "operation": "SELECT",
            "kind": "select",
            "label": spec["select_label"] + " → " + spec["desired_option_label"],
            "action_id": None,
            "text": None,
        }
    elif action == "submit":
        selected = next((radio for radio in after["radios"] if radio["checked"]), None)
        after.update(
            submitted=True,
            submission_count=before["submission_count"] + 1,
            submitted_radio_value=selected["value"] if selected else "",
            submitted_radio_label=selected["label"] if selected else "",
            submitted_note=after["note_value"],
            submitted_mode_value=after["mode_value"],
            submitted_mode_label=after["mode_label"],
        )
        after["status"] = f"Submitted entry {after['submission_count']}."
        event("click", "record-draft", True, "" if spec["commit_kind"] == "button" else None)
        descriptor = {
            "operation": "CLICK",
            "kind": "click",
            "label": spec["submit_label"],
            "action_id": None,
            "text": None,
        }
    elif action == "done":
        descriptor = {
            "operation": "DONE",
            "kind": "terminal",
            "label": None,
            "action_id": None,
            "text": None,
        }
    else:
        raise AssertionError("unknown invented native action")
    if action in ("type", "radio", "select"):
        after.update(submitted=False, status="Edited; submit the entry.")
        if spec["text_refresh"]["control"] == action and before["text_refresh_count"] == 0:
            after.update(note_value=spec["text_refresh"]["value"], text_refresh_count=1)
    after["events"].extend(events)
    return descriptor, after, events


class Nodes(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.nodes = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.nodes.append((tag, dict(attrs)))


def test_distinct_train_namespace_complete_factorial_and_prospective_counts():
    values = specs()
    assert len(values) == 33 and len(scene.FAMILIES) == 33
    assert scene.NAMESPACE == "decision3-owned-text-history-control-training-v1"
    assert all(value["split"] == "train" for value in values)
    assert (
        len({value["id"] for value in values})
        == len({value["ancestor_id"] for value in values})
        == 33
    )
    assert len({value["goal_sha256"] for value in values}) == 33
    assert Counter(len(value["options"]) for value in values) == {12: 11, 16: 11, 24: 11}
    assert {value["initial"]["note_value"] == value["desired_text"] for value in values} == {
        False,
        True,
    }
    for size in (12, 16, 24):
        assert {value["family"] for value in values if len(value["options"]) == size} == {
            f"{text}-{remaining}-options{size}"
            for text in ("empty", "wrong", "desired")
            for remaining in ("radio", "select", "submit")
        } | {f"recheck-radio-clears-options{size}", f"recheck-select-replaces-options{size}"}
    changed = specs()
    changed[0]["radios"][0]["label"] = "invented mutation"
    assert specs() == values and specs(20261028) != values


@pytest.mark.parametrize(
    "remaining,operation", [("radio", "CLICK"), ("select", "SELECT"), ("submit", "CLICK")]
)
def test_correct_visible_text_continues_remaining_goal_with_or_without_identical_input_history(
    remaining, operation
):
    spec = one("desired-" + remaining)
    for attempts in (0, 1, 4):
        state = initial(spec)
        # An invented history prefix supplies no authority to skip remaining work.
        state["events"] = [
            {
                "sequence": index,
                "type": "input",
                "target": "draft-annotation",
                "trusted": True,
                "value": spec["desired_text"],
                "checked": None,
            }
            for index in range(attempts)
        ]
        before = copy.deepcopy(state)
        step = scene.teacher_step(spec, state)
        assert step["operation"] == operation
        assert step["label"] == (
            spec["desired_radio_label"]
            if remaining == "radio"
            else spec["select_label"] + " → " + spec["desired_option_label"]
            if remaining == "select"
            else spec["submit_label"]
        )
        assert state == before and not scene.verify_goal(spec, state)


@pytest.mark.parametrize("text_mode", ["empty", "wrong"])
def test_empty_or_wrong_visible_text_requires_input_despite_same_prior_input(text_mode):
    spec = one(text_mode + "-submit")
    state = initial(spec)
    state["events"] = [
        {
            "sequence": 0,
            "type": "input",
            "target": "draft-annotation",
            "trusted": True,
            "value": spec["desired_text"],
            "checked": None,
        }
    ]
    assert state["note_value"] != spec["desired_text"]
    descriptor, after, events = effect(spec, state, "type")
    assert scene.teacher_step(spec, state) == descriptor
    proof = scene.verify_transition(spec, state, after, descriptor, events)
    assert proof["positive_supervision"] is True and proof["input_executed"] is True
    assert scene.teacher_step(spec, after)["label"] == spec["submit_label"]


@pytest.mark.parametrize(
    "family,control", [("recheck-radio-clears", "radio"), ("recheck-select-replaces", "select")]
)
def test_actual_input_then_control_refresh_then_same_input_again_has_complete_native_route(
    family, control
):
    spec, state = one(family), initial(one(family))
    actions = ["type", control, "type", "submit", "done"]
    operations = []
    for action in actions:
        descriptor, after, events = effect(spec, state, action)
        assert scene.teacher_step(spec, state) == descriptor
        proof = scene.verify_transition(spec, state, after, descriptor, events)
        assert proof["requires_canonical_native_executor_join"] is (action != "done")
        if action == control:
            assert state["note_value"] == spec["desired_text"]
            assert after["note_value"] != spec["desired_text"] and after["text_refresh_count"] == 1
            assert after["events"][0]["value"] == spec["desired_text"]
        operations.append(descriptor["operation"])
        state = after
    assert operations == [
        "TYPE_TEXT",
        "CLICK" if control == "radio" else "SELECT",
        "TYPE_TEXT",
        "CLICK",
        "DONE",
    ]
    assert scene.verify_goal(spec, state)


def test_redundant_input_premature_commit_done_and_unrelated_navigation_get_no_positive_annotation():
    spec = one("desired-select")
    before = initial(spec)
    for action in ("type", "submit", "done"):
        descriptor, after, events = effect(spec, before, action)
        with pytest.raises(ValueError):
            scene.verify_transition(spec, before, after, descriptor, events)
    descriptor = {
        "operation": "CLICK",
        "kind": "click",
        "label": spec["navigation_links"][0]["label"],
        "action_id": None,
        "text": None,
    }
    with pytest.raises(ValueError):
        scene.verify_transition(spec, before, before, descriptor, [])


def test_refresh_cannot_be_faked_repeated_or_used_to_change_unrelated_controls():
    spec = one("recheck-radio-clears")
    descriptor, typed, events = effect(spec, initial(spec), "type")
    scene.verify_transition(spec, initial(spec), typed, descriptor, events)
    descriptor, after, events = effect(spec, typed, "radio")
    for mutation in (
        "no-refresh",
        "extra-refresh",
        "wrong-refresh-value",
        "unrelated-select",
        "untrusted",
    ):
        changed, changed_events = copy.deepcopy(after), copy.deepcopy(events)
        if mutation == "no-refresh":
            changed["text_refresh_count"] = 0
        elif mutation == "extra-refresh":
            changed["text_refresh_count"] = 2
        elif mutation == "wrong-refresh-value":
            changed["note_value"] = spec["desired_text"]
        elif mutation == "unrelated-select":
            changed["mode_value"], changed["mode_label"] = "", scene.SELECT_PLACEHOLDER
        else:
            changed_events[0]["trusted"] = False
            changed["events"] = typed["events"] + changed_events
        with pytest.raises(ValueError):
            scene.verify_transition(spec, typed, changed, descriptor, changed_events)


def test_render_has_real_controls_full_options_current_values_and_no_private_teacher_metadata():
    for spec in specs():
        text = scene.render_fixture(spec)
        nodes = Nodes(text).nodes
        fields = [attrs for tag, attrs in nodes if tag == "input" and attrs.get("type") == "text"]
        assert fields == [
            {
                "id": "draft-annotation",
                "type": "text",
                "value": spec["initial"]["note_value"],
                "aria-label": spec["field_label"],
            }
        ]
        radios = [attrs for tag, attrs in nodes if tag == "input" and attrs.get("type") == "radio"]
        assert len(radios) == 3 and all(
            "disabled" not in attrs and "hidden" not in attrs for attrs in radios
        )
        options = [attrs for tag, attrs in nodes if tag == "option"]
        assert len(options) == len(spec["options"]) + 1
        assert [attrs["value"] for attrs in options if "disabled" not in attrs] == [
            o["value"] for o in spec["options"]
        ]
        assert "overflow-y:auto" in text and "min-height:1800px" in text
        for forbidden in (
            spec["family"],
            scene.NAMESPACE,
            "teacher_step",
            "goal_order",
            "required_fields",
            "action_history",
        ):
            assert forbidden not in text
        assert "text_refresh_count" in scene.private_state_expression()


def wire():
    return {
        "model": "invented-native-policy",
        "state": {
            "goal": "Invented calibration request",
            "elements": [
                {"role": "textbox", "id": "invented-annotation", "value": "invented-probe"}
            ],
            "recent_actions": [
                {"operation": "TYPE_TEXT", "text": "invented-probe", "page_changed": True}
            ],
        },
        "questions": {
            "select_target": {
                "type": "choice",
                "instructions": {"nested": [None, True]},
                "criteria": {f"tier-{i}": {"label": f"Tier {i}"} for i in range(24)},
            },
            "operation": {
                "type": "choice",
                "instructions": "Invented ordered choice",
                "criteria": {
                    name: {"preserve": name}
                    for name in (
                        "CLICK",
                        "TYPE_TEXT",
                        "SELECT",
                        "WAIT",
                        "DONE",
                        "BLOCKED",
                        "SCROLL_DOWN",
                    )
                },
            },
            "click_target": {
                "type": "choice",
                "instructions": "Invented controls",
                "criteria": {f"button-{i}": {"checked": False} for i in range(255)},
            },
            "type_text_target": {
                "type": "choice",
                "instructions": "Invented fields",
                "criteria": {"annotation": {"value": "invented-probe"}},
            },
        },
    }


def test_full_original_wire_preserves_history_instructions_all_heads255_choices_and_order():
    original = wire()
    before = copy.deepcopy(original)
    observed = copy.deepcopy(original)
    proof = scene.verify_native_wire_unchanged(original, observed)
    assert proof["ordered_head_ids"] == [
        "select_target",
        "operation",
        "click_target",
        "type_text_target",
    ]
    assert proof["candidate_counts"] == [24, 7, 255, 1]
    assert (
        proof["wire_sha256"]
        == hashlib.sha256(
            json.dumps(
                original, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
    )
    assert original == observed == before
    for mutation in (
        "history",
        "instruction",
        "head-order",
        "criterion-order",
        "prune",
        "criterion-value",
        "teacher",
    ):
        changed = copy.deepcopy(original)
        if mutation == "history":
            changed["state"]["recent_actions"] = []
        elif mutation == "instruction":
            changed["questions"]["select_target"]["instructions"]["nested"].reverse()
        elif mutation == "head-order":
            changed["questions"] = dict(reversed(list(changed["questions"].items())))
        elif mutation == "criterion-order":
            changed["questions"]["click_target"]["criteria"] = dict(
                reversed(list(changed["questions"]["click_target"]["criteria"].items()))
            )
        elif mutation == "prune":
            changed["questions"]["click_target"]["criteria"].pop("button-254")
        elif mutation == "criterion-value":
            changed["questions"]["click_target"]["criteria"]["button-0"]["checked"] = True
        else:
            changed["teacher_operation"] = "SELECT"
        with pytest.raises(ValueError):
            scene.verify_native_wire_unchanged(original, changed)


def test_exclusion_inventory_split_namespace_seed_and_goal_hash_cannot_be_bypassed():
    spec = one("desired-radio")
    with pytest.raises(ValueError):
        scene.scenarios(spec["seed"], {spec["goal_sha256"]})
    for excluded in ([], ["not-a-hash"], ["a" * 64, "a" * 64]):
        with pytest.raises(ValueError):
            scene.scenarios(20261027, excluded)
    for key, value in (
        ("split", "dev"),
        ("namespace", "invented-dev"),
        ("goal", "changed goal"),
        ("seed", True),
        ("expected_menu_counts", [6]),
        ("text_refresh", {"control": "radio", "value": ""}),
    ):
        changed = {**spec, key: value}
        with pytest.raises(ValueError):
            scene.render_fixture(changed)
        with pytest.raises(ValueError):
            scene.teacher_step(changed, initial(spec))
