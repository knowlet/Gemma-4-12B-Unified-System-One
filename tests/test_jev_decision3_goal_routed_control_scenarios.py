"""Invented DOM/event controls only; no native browser or model result is supplied."""

import copy
import hashlib
import json
from collections import Counter, defaultdict
from html.parser import HTMLParser

import jev_decision3_goal_routed_control_scenarios as fixtures
import pytest


class Controls(HTMLParser):
    def __init__(self):
        super().__init__()
        self.radios = []
        self.order = []
        self.fields = {}
        self.options = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        identity = attributes.get("id")
        if tag == "fieldset":
            self.order.append("radio")
        if tag == "input" and attributes.get("type") == "radio":
            self.radios.append(attributes)
        if identity in {"entry-text", "entry-select", "commit-control"}:
            self.fields[identity] = attributes
            self.order.append(
                {"entry-text": "text", "entry-select": "select", "commit-control": "submit"}[
                    identity
                ]
            )
        if tag == "option":
            self.options.append(attributes)


def spec_for(family=0, seed=804):
    return fixtures.scenarios(seed, {"0" * 64})[family]


def initial_state(spec):
    initial = spec["initial"]
    modes = {"": fixtures.SELECT_PLACEHOLDER, **{o["value"]: o["label"] for o in spec["options"]}}
    return {
        "title": spec["title"],
        "url": "http://owned-training.invalid/fixture",
        "radios": [
            dict(
                choice,
                name=fixtures.RADIO_GROUP,
                checked=choice["value"] == initial["radio_value"],
            )
            for choice in spec["radios"]
        ],
        "note_value": initial["note_value"],
        "mode_value": initial["mode_value"],
        "mode_label": modes[initial["mode_value"]],
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


def advance(spec, before):
    """Invent an effect; acceptance still requires the separate transition validator."""
    after = copy.deepcopy(before)
    step = fixtures.teacher_step(spec, before)
    operation = step["operation"]
    if operation == "DONE":
        return after
    if operation == "TYPE_TEXT":
        after["note_value"] = spec["desired_text"]
        event(after, "input", "entry-text", spec["desired_text"])
    elif operation == "SELECT":
        after.update(
            mode_value=spec["desired_option_value"], mode_label=spec["desired_option_label"]
        )
        for kind in ("input", "change"):
            event(after, kind, "entry-select", spec["desired_option_value"], trusted=False)
    elif step["label"] == spec["desired_radio_label"]:
        choice = next(r for r in spec["radios"] if r["value"] == spec["desired_radio_value"])
        for radio in after["radios"]:
            radio["checked"] = radio["value"] == choice["value"]
        for kind in ("click", "input", "change"):
            event(after, kind, choice["id"], choice["value"], checked=True)
    else:
        assert operation == "CLICK" and step["label"] == spec["submit_label"]
        selected = next((r for r in after["radios"] if r["checked"]), None)
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
        event(after, "click", "commit-control", "")
        return after
    after.update(submitted=False, status="Edited; submit the entry.")
    return after


def invented_route(spec):
    states = [initial_state(spec)]
    for _ in range(16):
        if fixtures.teacher_step(spec, states[-1])["operation"] == "DONE":
            return states
        states.append(advance(spec, states[-1]))
    raise AssertionError("Invented route did not finish within the unchanged tick budget")


def effect(spec, before, after, step=None):
    return fixtures.verify_transition(
        spec,
        before,
        after,
        fixtures.teacher_step(spec, before) if step is None else step,
        after["events"][len(before["events"]) :],
    )


def test_namespace_detached_specs_closed_schema_and_exclusion_only_hashes():
    specs = fixtures.scenarios(804, {"0" * 64})
    assert fixtures.SCHEMA == "decision3-owned-goal-routed-control-scenarios-v1"
    assert fixtures.NAMESPACE == "decision3-owned-goal-routed-control-training-v1"
    assert fixtures.GOAL_NORMALIZATION == "NFKC-collapse-whitespace-v1"
    assert fixtures.EXPECTED_MENU_COUNTS == (6,)
    assert len(specs) == 12
    assert len({s["id"] for s in specs}) == len({s["goal_sha256"] for s in specs}) == 12
    assert all(s["split"] == "train" and s["expected_menu_counts"] == [6] for s in specs)
    assert all(s["family"] not in s["id"] and s["family"] not in s["ancestor_id"] for s in specs)
    with pytest.raises(ValueError, match="collides"):
        fixtures.scenarios(804, {specs[0]["goal_sha256"]})
    specs[0]["radios"][0]["label"] = "changed detached object"
    assert spec_for()["radios"][0]["label"] != "changed detached object"
    for exclusions in ([], ["x" * 64], ["0" * 64, "0" * 64]):
        with pytest.raises(ValueError):
            fixtures.scenarios(804, exclusions)


@pytest.mark.parametrize("seed", [-1, True, 2**63, "804"])
def test_invalid_seed_is_refused(seed):
    with pytest.raises(ValueError, match="Seed"):
        fixtures.scenarios(seed, {"0" * 64})


@pytest.mark.parametrize(
    "corruption", ["namespace", "split", "id", "initial", "required", "order", "goal", "extra"]
)
def test_authored_scene_cannot_be_rewritten_to_change_its_route(corruption):
    spec = spec_for(4)
    if corruption == "initial":
        spec["initial"]["note_value"] = spec["desired_text"]
    elif corruption == "required":
        spec["required_fields"]["radio"] = 0
    elif corruption == "order":
        spec["layout_order"] = list(reversed(spec["layout_order"]))
    elif corruption == "extra":
        spec["teacher_id"] = "not admitted"
    else:
        spec[corruption] = "foreign"
    with pytest.raises(ValueError):
        fixtures.render_fixture(spec)


def test_balanced_initial_operations_and_34_positive_rows_follow_independent_dom_requirements():
    specs = fixtures.scenarios(804, {"0" * 64})
    expected = ["CLICK"] * 4 + ["TYPE_TEXT"] * 4 + ["SELECT"] * 4
    initial_operations = []
    all_operations = []
    for spec, first in zip(specs, expected, strict=True):
        states = invented_route(spec)
        operations = [fixtures.teacher_step(spec, state)["operation"] for state in states]
        initial_operations.append(operations[0])
        assert operations[0] == first
        assert operations[-2:] == ["CLICK", "DONE"]
        assert len(operations) == (2 if spec["family"].startswith("ready-") else 3)
        assert not fixtures.verify_goal(spec, states[0])
        for before, after in zip(states, states[1:]):
            proof = effect(spec, before, after)
            assert proof["positive_supervision"] is True
            assert proof["requires_canonical_native_executor_join"] is True
            assert proof["goal_verified_after"] is (after is states[-1])
            if proof["operation"] == "SELECT":
                assert proof["requires_native_select_executor_join"] is True
                assert proof["select_event_isTrusted"] is False
        terminal = effect(spec, states[-1], copy.deepcopy(states[-1]))
        assert terminal["operation"] == "DONE" and terminal["input_executed"] is False
        assert terminal["requires_canonical_native_executor_join"] is False
        all_operations.extend(operations)
    assert Counter(initial_operations) == {"CLICK": 4, "TYPE_TEXT": 4, "SELECT": 4}
    assert Counter(all_operations) == {"CLICK": 14, "TYPE_TEXT": 4, "SELECT": 4, "DONE": 12}


def test_goal_routes_cover_irrelevant_unchecked_prechecked_and_already_requested_radios():
    specs = fixtures.scenarios(804, {"0" * 64})
    assert specs[4]["initial"]["radio_value"] is None
    assert specs[8]["initial"]["radio_value"] is None
    for index in (5, 9):
        assert specs[index]["required_fields"]["radio"] is False
        assert specs[index]["initial"]["radio_value"] is not None
    for index in (6, 7, 10, 11):
        spec = specs[index]
        assert spec["required_fields"]["radio"] is True
        assert spec["initial"]["radio_value"] == spec["desired_radio_value"]
        assert fixtures.teacher_step(spec, initial_state(spec))["operation"] != "CLICK"
    assert {len(spec["radios"]) for spec in specs} == {2, 3}


def test_dom_submit_position_goal_clause_order_defaults_and_target_position_vary_independently():
    by_layout = defaultdict(set)
    submit_before_radio = set()
    target_positions = set()
    initial_modes = set()
    for seed in range(64):
        ready = spec_for(2, seed)
        by_layout[tuple(ready["layout_order"])].add(tuple(ready["goal_order"]))
        submit_before_radio.add(
            ready["layout_order"].index("submit") < ready["layout_order"].index("radio")
        )
        radio = spec_for(0, seed)
        target_positions.add(
            next(
                i
                for i, r in enumerate(radio["radios"])
                if r["value"] == radio["desired_radio_value"]
            )
        )
        initial_modes.add(spec_for(4, seed)["initial"]["mode_value"])
    assert len(by_layout) > 8 and any(len(orders) > 1 for orders in by_layout.values())
    assert submit_before_radio == {False, True}
    assert target_positions == {0, 1}
    assert len(initial_modes) == 4


@pytest.mark.parametrize("family", range(12))
def test_render_keeps_all_controls_enabled_in_declared_order_and_no_hidden_supervision(family):
    spec = spec_for(family)
    rendered = fixtures.render_fixture(spec)
    controls = Controls()
    controls.feed(rendered)
    assert controls.order == spec["layout_order"]
    assert len(controls.radios) == len(spec["radios"])
    assert all("disabled" not in radio for radio in controls.radios)
    assert [r["value"] for r in controls.radios if "checked" in r] == (
        [] if spec["initial"]["radio_value"] is None else [spec["initial"]["radio_value"]]
    )
    assert controls.fields["entry-text"]["value"] == spec["initial"]["note_value"]
    assert all("disabled" not in attrs for attrs in controls.fields.values())
    assert [o["value"] for o in controls.options if "selected" in o] == [
        spec["initial"]["mode_value"]
    ]
    assert all("disabled" not in o for o in controls.options if o["value"])
    for forbidden in (
        spec["id"],
        spec["ancestor_id"],
        spec["family"],
        spec["goal"],
        "required_fields",
        "desired_",
    ):
        assert forbidden not in rendered
    script = rendered.split("<script>", 1)[1]
    assert spec["desired_text"] not in script
    assert spec["desired_radio_value"] not in script
    assert spec["desired_option_value"] not in script
    assert "summary.dataset.submitted='false'" in script
    assert "checked:r.checked" in fixtures.private_state_expression()
    assert "overflow:hidden" in rendered


@pytest.mark.parametrize(
    "malformation", ["string", "multiple", "value", "label", "group", "missing"]
)
def test_strict_radio_state_refuses_forgery_even_when_radio_is_irrelevant(malformation):
    spec = spec_for(4)
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


@pytest.mark.parametrize("kind", ["click", "input", "change"])
def test_radio_edit_requires_each_original_trusted_native_event(kind):
    spec = spec_for()
    before, after = invented_route(spec)[:2]
    next(e for e in after["events"] if e["type"] == kind)["trusted"] = False
    with pytest.raises(ValueError, match="Trusted radio"):
        effect(spec, before, after)


def test_repeated_radio_noop_and_irrelevant_radio_click_cannot_be_positive_labels():
    spec = spec_for()
    before, selected = invented_route(spec)[:2]
    radio_step = fixtures.teacher_step(spec, before)
    assert fixtures.teacher_step(spec, selected)["label"] == spec["submit_label"]
    with pytest.raises(ValueError, match="explicit authored route"):
        effect(spec, selected, copy.deepcopy(selected), radio_step)
    text_spec = spec_for(4)
    state = initial_state(text_spec)
    with pytest.raises(ValueError, match="explicit authored route"):
        effect(text_spec, state, copy.deepcopy(state), radio_step)


def test_text_requires_trusted_input_and_preserves_required_and_irrelevant_radios():
    for family in (4, 6):
        spec = spec_for(family)
        before, after = invented_route(spec)[:2]
        forged = copy.deepcopy(after)
        forged["events"][-1]["trusted"] = False
        with pytest.raises(ValueError, match="Trusted text"):
            effect(spec, before, forged)
        forged = copy.deepcopy(after)
        for radio in forged["radios"]:
            radio["checked"] = False
        if family == 4:
            forged["radios"][0]["checked"] = True
        with pytest.raises(ValueError):
            effect(spec, before, forged)


def test_select_requires_original_synthetic_events_real_value_and_unchanged_other_fields():
    spec = spec_for(8)
    before, after = invented_route(spec)[:2]
    assert effect(spec, before, after)["requires_native_select_executor_join"] is True
    for corruption in ("trust", "label", "text"):
        forged = copy.deepcopy(after)
        if corruption == "trust":
            forged["events"][-1]["trusted"] = True
        elif corruption == "label":
            forged["mode_label"] = "wrong visible option"
        else:
            forged["note_value"] = "changed irrelevant field"
        with pytest.raises(ValueError):
            effect(spec, before, forged)


@pytest.mark.parametrize("family", [4, 5, 6, 8, 9, 10])
def test_saved_record_must_match_all_actual_controls_including_irrelevant_or_empty_radio(family):
    spec = spec_for(family)
    before, complete = invented_route(spec)[-2:]
    assert fixtures.verify_goal(spec, complete) is True
    for name in (
        "submitted_radio_value",
        "submitted_radio_label",
        "submitted_note",
        "submitted_mode_value",
        "submitted_mode_label",
    ):
        forged = copy.deepcopy(complete)
        forged[name] = "wrong saved value"
        assert fixtures.verify_goal(spec, forged) is False
        with pytest.raises(ValueError, match="Trusted submit"):
            effect(spec, before, forged)


def submit_after_text_blur(spec, before):
    after = advance(spec, before)
    after["events"] = copy.deepcopy(before["events"])
    event(after, "change", "entry-text", before["note_value"])
    event(after, "click", "commit-control", "")
    return after


def test_submit_accepts_only_unchanged_trusted_text_blur_then_commit():
    spec = spec_for(4)
    before = invented_route(spec)[-2]
    after = submit_after_text_blur(spec, before)
    assert effect(spec, before, after)["appended_event_count"] == 2
    assert fixtures.verify_goal(spec, after) is True


@pytest.mark.parametrize(
    "corruption",
    [
        "wrong-value",
        "untrusted",
        "input-kind",
        "radio-target",
        "post-click",
        "duplicate",
        "checked",
    ],
)
def test_submit_refuses_other_blur_shapes_even_with_complete_actual_goal(corruption):
    spec = spec_for(4)
    before = invented_route(spec)[-2]
    after = submit_after_text_blur(spec, before)
    appended = copy.deepcopy(after["events"][len(before["events"]) :])
    blur, click = appended
    if corruption == "wrong-value":
        blur["value"] = "wrong"
    elif corruption == "untrusted":
        blur["trusted"] = False
    elif corruption == "input-kind":
        blur["type"] = "input"
    elif corruption == "radio-target":
        blur["target"] = spec["radios"][0]["id"]
    elif corruption == "post-click":
        appended = [click, blur]
    elif corruption == "duplicate":
        appended = [blur, copy.deepcopy(blur), click]
    else:
        blur["checked"] = False
    for index, record in enumerate(appended, start=len(before["events"])):
        record["sequence"] = index
    after["events"] = copy.deepcopy(before["events"]) + appended
    assert fixtures.verify_goal(spec, after) is True
    with pytest.raises(ValueError, match="Trusted submit"):
        effect(spec, before, after)


def test_fresh_done_refuses_stale_edits_missing_submission_and_hidden_input():
    spec = spec_for(6)
    complete = invented_route(spec)[-1]
    done = fixtures.teacher_step(spec, complete)
    changed = copy.deepcopy(complete)
    changed.update(note_value="edited after submission", submitted=False)
    assert fixtures.verify_goal(spec, changed) is False
    assert fixtures.teacher_step(spec, changed)["operation"] == "TYPE_TEXT"
    with pytest.raises(ValueError, match="explicit authored route"):
        effect(spec, changed, copy.deepcopy(changed), done)
    stale = copy.deepcopy(complete)
    stale["submitted"] = False
    assert fixtures.teacher_step(spec, stale)["label"] == spec["submit_label"]
    mutated = copy.deepcopy(complete)
    event(mutated, "click", "commit-control", "")
    with pytest.raises(ValueError, match="must not execute"):
        effect(spec, complete, mutated)


@pytest.mark.parametrize("corruption", ["url", "prefix", "sequence", "target"])
def test_event_prefix_full_sequence_and_document_identity_cannot_be_replaced(corruption):
    spec = spec_for()
    before, after = invented_route(spec)[-2:]
    if corruption == "url":
        after["url"] += "/foreign"
    elif corruption == "prefix":
        after["events"][0]["value"] = "changed prior event"
    elif corruption == "sequence":
        after["events"][-1]["sequence"] = True
    else:
        after["events"][-1]["target"] = "foreign-control"
    with pytest.raises(ValueError):
        effect(spec, before, after)


def test_proof_hash_binds_the_complete_invented_private_state():
    spec = spec_for(4)
    before, after = invented_route(spec)[:2]
    proof = effect(spec, before, after)
    encoded = json.dumps(
        before, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False
    )
    assert proof["before_sha256"] == hashlib.sha256(encoded.encode()).hexdigest()
