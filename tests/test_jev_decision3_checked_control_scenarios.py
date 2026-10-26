"""Invented DOM/event controls only; these are not native browser or model evidence."""

import copy
import hashlib
import json
from html.parser import HTMLParser

import jev_decision3_checked_control_scenarios as fixtures
import pytest


class RadioControls(HTMLParser):
    def __init__(self):
        super().__init__()
        self.radios = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "input" and attributes.get("type") == "radio":
            self.radios.append(attributes)


def spec_for(family=0, seed=804):
    return fixtures.scenarios(seed, {"0" * 64})[family]


def initial_state(spec):
    return {
        "title": spec["title"],
        "url": "http://owned-training.invalid/fixture",
        "radios": [
            dict(choice, name=fixtures.RADIO_GROUP, checked=False) for choice in spec["radios"]
        ],
        "note_value": "",
        "mode_value": "",
        "mode_label": fixtures.SELECT_PLACEHOLDER,
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


def event(state, kind, target, value=None, *, trusted=True, checked=None):
    result = {
        "sequence": len(state["events"]),
        "type": kind,
        "target": target,
        "trusted": trusted,
        "value": value,
        "checked": checked,
    }
    state["events"].append(result)
    return result


def radio_after(spec, before):
    after = copy.deepcopy(before)
    choice = next(r for r in spec["radios"] if r["value"] == spec["desired_radio_value"])
    for radio in after["radios"]:
        radio["checked"] = radio["value"] == choice["value"]
    after.update(submitted=False, status="Edited; submit the entry.")
    for kind in ("click", "input", "change"):
        event(after, kind, choice["id"], choice["value"], checked=True)
    return after


def text_after(spec, before):
    after = copy.deepcopy(before)
    after.update(
        note_value=spec["desired_text"], submitted=False, status="Edited; submit the entry."
    )
    event(after, "input", "entry-text", spec["desired_text"])
    return after


def select_after(spec, before):
    after = copy.deepcopy(before)
    after.update(
        mode_value=spec["desired_option_value"],
        mode_label=spec["desired_option_label"],
        submitted=False,
        status="Edited; submit the entry.",
    )
    for kind in ("input", "change"):
        event(after, kind, "entry-select", spec["desired_option_value"], trusted=False)
    return after


def submit_after(spec, before):
    after = copy.deepcopy(before)
    after.update(
        submitted=True,
        submission_count=before["submission_count"] + 1,
        submitted_radio_value=spec["desired_radio_value"],
        submitted_radio_label=spec["desired_radio_label"],
        submitted_note=spec["desired_text"],
        submitted_mode_value=spec["desired_option_value"],
        submitted_mode_label=spec["desired_option_label"],
    )
    after["status"] = f"Submitted entry {after['submission_count']}."
    event(after, "click", "commit-control", "")
    return after


def invented_route(spec):
    states = [initial_state(spec)]
    for advance in (radio_after, text_after, select_after, submit_after):
        states.append(advance(spec, states[-1]))
    return states


def effect(spec, before, after, step=None):
    return fixtures.verify_transition(
        spec,
        before,
        after,
        fixtures.teacher_step(spec, before) if step is None else step,
        after["events"][len(before["events"]) :],
    )


def test_fresh_namespace_detached_specs_and_natural_menu_intent():
    specs = fixtures.scenarios(804, {"0" * 64})
    assert {spec["family"] for spec in specs} == {"radio-two", "radio-three"}
    assert len({spec["goal_sha256"] for spec in specs}) == 2
    assert all(spec["split"] == "train" and spec["expected_menu_counts"] == [6] for spec in specs)
    assert all(spec["namespace"] == fixtures.NAMESPACE for spec in specs)
    specs[0]["radios"][0]["label"] = "mutated detached spec"
    assert fixtures.scenarios(804, {"0" * 64})[0]["radios"][0]["label"] != "mutated detached spec"


@pytest.mark.parametrize("seed", [-1, True, 2**63, "804"])
def test_invalid_seed_is_refused(seed):
    with pytest.raises(ValueError, match="Seed"):
        fixtures.scenarios(seed, {"0" * 64})


def test_hash_exclusion_and_spec_identity_cannot_be_rewritten():
    spec = spec_for()
    with pytest.raises(ValueError, match="collides"):
        fixtures.scenarios(804, {spec["goal_sha256"]})
    for exclusions in ([], ["x" * 64], ["0" * 64, "0" * 64]):
        with pytest.raises(ValueError):
            fixtures.scenarios(804, exclusions)
    for key, value in (("split", "dev"), ("namespace", "foreign"), ("expected_menu_counts", [7])):
        altered = copy.deepcopy(spec)
        altered[key] = value
        with pytest.raises(ValueError):
            fixtures.render_fixture(altered)
    altered = copy.deepcopy(spec)
    altered["teacher_id"] = "never render"
    with pytest.raises(ValueError, match="Exact authored"):
        fixtures.render_fixture(altered)


def test_seeds_vary_visible_choices_order_and_desired_radio_position():
    specs = [spec_for(0, seed) for seed in range(16)]
    assert len({tuple(r["label"] for r in spec["radios"]) for spec in specs}) == 4
    assert {
        next(i for i, r in enumerate(spec["radios"]) if r["value"] == spec["desired_radio_value"])
        for spec in specs
    } == {0, 1}


@pytest.mark.parametrize("family", [0, 1])
def test_render_contains_all_enabled_alternatives_without_teacher_goal_or_desired_text(family):
    spec = spec_for(family)
    rendered = fixtures.render_fixture(spec)
    controls = RadioControls()
    controls.feed(rendered)
    assert len(controls.radios) == len(spec["radios"])
    for choice in spec["radios"]:
        assert f"aria-label='{choice['label']}'" in rendered
        assert f"value='{choice['value']}'" in rendered
    for forbidden in (
        spec["id"],
        spec["ancestor_id"],
        spec["goal"],
        spec["desired_text"],
        "desired_",
    ):
        assert forbidden not in rendered
    assert all("checked" not in radio and "disabled" not in radio for radio in controls.radios)
    assert "overflow:hidden" in rendered
    assert "checked:r.checked" in fixtures.private_state_expression()
    assert "summary.dataset.submitted='false'" in rendered


@pytest.mark.parametrize("family", [0, 1])
def test_invented_five_phase_route_preserves_native_join_requirements(family):
    spec = spec_for(family)
    states = invented_route(spec)
    assert [fixtures.teacher_step(spec, state)["operation"] for state in states] == [
        "CLICK",
        "TYPE_TEXT",
        "SELECT",
        "CLICK",
        "DONE",
    ]
    assert fixtures.teacher_step(spec, states[0])["label"] == spec["desired_radio_label"]
    assert fixtures.teacher_step(spec, states[3])["label"] == spec["submit_label"]
    for before, after in zip(states, states[1:]):
        proof = effect(spec, before, after)
        assert proof["positive_supervision"] is True
        assert proof["requires_canonical_native_executor_join"] is True
        assert proof["goal_verified_after"] is (after is states[-1])
        if proof["operation"] == "SELECT":
            assert proof["requires_native_select_executor_join"] is True
            assert proof["select_event_isTrusted"] is False
    final = copy.deepcopy(states[-1])
    proof = effect(spec, states[-1], final)
    assert proof["operation"] == "DONE" and proof["input_executed"] is False
    assert proof["requires_canonical_native_executor_join"] is False


@pytest.mark.parametrize(
    "malformation", ["string", "multiple", "value", "label", "group", "missing"]
)
def test_radio_oracle_refuses_string_false_multiple_checked_or_foreign_group(malformation):
    spec = spec_for()
    state = initial_state(spec)
    if malformation == "string":
        state["radios"][0]["checked"] = "false"
    elif malformation == "multiple":
        for radio in state["radios"]:
            radio["checked"] = True
    elif malformation == "missing":
        state["radios"].pop()
    else:
        state["radios"][0]["name" if malformation == "group" else malformation] = "foreign"
    assert fixtures.verify_goal(spec, state) is False
    with pytest.raises(ValueError, match="radio"):
        fixtures.teacher_step(spec, state)


def test_native_radio_change_can_exclude_a_previously_checked_other_choice():
    spec = spec_for()
    before = initial_state(spec)
    other = next(r for r in before["radios"] if r["value"] != spec["desired_radio_value"])
    other["checked"] = True
    after = radio_after(spec, before)
    assert effect(spec, before, after)["operation"] == "CLICK"
    assert sum(r["checked"] for r in after["radios"]) == 1


def test_reclicking_a_completed_radio_is_not_positive_supervision():
    spec = spec_for()
    before, after = invented_route(spec)[:2]
    first_step = fixtures.teacher_step(spec, before)
    assert fixtures.teacher_step(spec, after)["operation"] == "TYPE_TEXT"
    repeated = radio_after(spec, after)
    with pytest.raises(ValueError, match="explicit authored route"):
        effect(spec, after, repeated, first_step)


@pytest.mark.parametrize("kind", ["click", "input", "change"])
def test_forged_radio_event_is_refused_even_when_dom_looks_complete(kind):
    spec = spec_for()
    before, after = invented_route(spec)[:2]
    next(e for e in after["events"] if e["type"] == kind)["trusted"] = False
    with pytest.raises(ValueError, match="Trusted radio"):
        effect(spec, before, after)


def test_text_input_cannot_silently_change_the_completed_radio():
    spec = spec_for()
    states = invented_route(spec)
    after = copy.deepcopy(states[2])
    for radio in after["radios"]:
        radio["checked"] = False
    with pytest.raises(ValueError, match="preserve the completed radio"):
        effect(spec, states[1], after)


def test_select_requires_original_synthetic_input_change_and_real_dom_value():
    spec = spec_for()
    states = invented_route(spec)
    before, after = states[2:4]
    assert effect(spec, before, after)["requires_native_select_executor_join"] is True
    forged = copy.deepcopy(after)
    forged["events"][-1]["trusted"] = True
    with pytest.raises(ValueError, match="Canonical SELECT"):
        effect(spec, before, forged)
    wrong = copy.deepcopy(after)
    wrong["mode_label"] = "not the observed option"
    with pytest.raises(ValueError, match="selected option"):
        effect(spec, before, wrong)


@pytest.mark.parametrize(
    "name", ["submitted_radio_value", "submitted_note", "submitted_mode_label"]
)
def test_submit_flag_alone_cannot_replace_visible_recorded_values(name):
    spec = spec_for()
    before, after = invented_route(spec)[3:]
    after[name] = "wrong recorded value"
    assert fixtures.verify_goal(spec, after) is False
    with pytest.raises(ValueError, match="Trusted submit"):
        effect(spec, before, after)


def submit_after_text_blur(spec, before):
    after = submit_after(spec, before)
    after["events"] = copy.deepcopy(before["events"])
    event(after, "change", "entry-text", before["note_value"])
    event(after, "click", "commit-control", "")
    return after


@pytest.mark.parametrize("family", [0, 1])
def test_submit_accepts_exact_native_unchanged_text_blur_before_button_click(family):
    spec = spec_for(family)
    before = invented_route(spec)[3]
    after = submit_after_text_blur(spec, before)
    proof = effect(spec, before, after)
    assert before["note_value"] == after["note_value"] == spec["desired_text"]
    assert proof["appended_event_count"] == 2
    assert proof["requires_canonical_native_executor_join"] is True
    assert proof["goal_verified_after"] is True


@pytest.mark.parametrize(
    "corruption",
    [
        "wrong-value",
        "untrusted",
        "input-kind",
        "radio-target",
        "checked-blur",
        "wrong-click-value",
        "untrusted-click",
        "post-click",
        "duplicate-blur",
        "duplicate-click",
    ],
)
def test_submit_refuses_other_event_shapes_even_when_dom_goal_is_complete(corruption):
    spec = spec_for()
    before = invented_route(spec)[3]
    after = submit_after_text_blur(spec, before)
    appended = copy.deepcopy(after["events"][len(before["events"]) :])
    blur, click = appended
    if corruption == "wrong-value":
        blur["value"] = "different live input"
    elif corruption == "untrusted":
        blur["trusted"] = False
    elif corruption == "input-kind":
        blur["type"] = "input"
    elif corruption == "radio-target":
        blur["target"] = spec["radios"][0]["id"]
    elif corruption == "checked-blur":
        blur["checked"] = False
    elif corruption == "wrong-click-value":
        click["value"] = "foreign button value"
    elif corruption == "untrusted-click":
        click["trusted"] = False
    elif corruption == "post-click":
        appended = [click, blur]
    elif corruption == "duplicate-blur":
        appended = [blur, copy.deepcopy(blur), click]
    else:
        appended = [blur, click, copy.deepcopy(click)]
    for index, record in enumerate(appended, start=len(before["events"])):
        record["sequence"] = index
    after["events"] = copy.deepcopy(before["events"]) + appended
    assert fixtures.verify_goal(spec, after) is True
    with pytest.raises(ValueError, match="Trusted submit"):
        effect(spec, before, after)


def test_edited_values_invalidate_a_stale_submitted_record_and_premature_done():
    spec = spec_for()
    complete = invented_route(spec)[-1]
    done = fixtures.teacher_step(spec, complete)
    changed = copy.deepcopy(complete)
    changed["note_value"] = "edited after submission"
    changed["submitted"] = False
    assert fixtures.verify_goal(spec, changed) is False
    assert fixtures.teacher_step(spec, changed)["operation"] == "TYPE_TEXT"
    with pytest.raises(ValueError, match="explicit authored route"):
        effect(spec, changed, copy.deepcopy(changed), done)
    stale = copy.deepcopy(complete)
    stale["submitted"] = False
    assert fixtures.teacher_step(spec, stale)["label"] == spec["submit_label"]


def test_terminal_decision_cannot_hide_mutation_or_unassigned_input():
    spec = spec_for()
    complete = invented_route(spec)[-1]
    changed = copy.deepcopy(complete)
    event(changed, "click", "commit-control")
    with pytest.raises(ValueError, match="must not execute"):
        effect(spec, complete, changed)


@pytest.mark.parametrize("corruption", ["url", "prefix", "sequence", "target"])
def test_event_or_document_join_corruption_is_refused(corruption):
    spec = spec_for()
    states = invented_route(spec)
    before, after = states[1:3]
    if corruption == "url":
        after["url"] += "/foreign"
    elif corruption == "prefix":
        after["events"][0]["value"] = "changed prefix"
    elif corruption == "sequence":
        after["events"][-1]["sequence"] = True
    else:
        after["events"][-1]["target"] = "foreign-control"
    with pytest.raises(ValueError):
        effect(spec, before, after)


def test_transition_digests_bind_the_entire_invented_private_state():
    spec = spec_for()
    before, after = invented_route(spec)[:2]
    proof = effect(spec, before, after)
    encoded = json.dumps(
        before, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False
    )
    assert proof["before_sha256"] == hashlib.sha256(encoded.encode()).hexdigest()
