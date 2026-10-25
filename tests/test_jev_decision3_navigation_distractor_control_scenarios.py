"""Invented source/DOM/event controls; no browser/model qualification is claimed."""

import copy
from collections import Counter
from html.parser import HTMLParser

import jev_decision3_navigation_distractor_control_scenarios as fixtures
import pytest


class Controls(HTMLParser):
    def __init__(self):
        super().__init__()
        self.click_ids = []
        self.fields = {}
        self.anchors = []
        self.radios = []
        self.option_values = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        identity = attrs.get("id")
        if identity:
            self.fields[identity] = attrs
        if tag == "a":
            self.anchors.append(attrs)
            self.click_ids.append(identity)
        elif tag == "button":
            self.click_ids.append(identity)
        elif tag == "input" and attrs.get("type") in {"radio", "text"}:
            self.click_ids.append(identity)
            if attrs["type"] == "radio":
                self.radios.append(attrs)
        elif tag == "option":
            self.option_values.append(attrs.get("value"))


def specs(seed=918):
    return fixtures.scenarios(seed, {"0" * 64})


def initial(spec):
    state = {
        "title": spec["title"],
        "url": "http://owned-training.invalid/local-card",
        "radios": [
            dict(
                choice,
                name=fixtures.RADIO_GROUP,
                checked=choice["value"] == spec["initial"]["radio_value"],
            )
            for choice in spec["radios"]
        ],
        "navigation": copy.deepcopy(spec["navigation_links"]),
        "commit_tag": "a" if spec["commit_kind"] == "link" else "button",
        "note_value": spec["initial"]["note_value"],
        "mode_value": spec["initial"]["mode_value"],
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
    state["mode_label"] = {
        "": fixtures.SELECT_PLACEHOLDER,
        **{option["value"]: option["label"] for option in spec["options"]},
    }[state["mode_value"]]
    return state


def event(state, kind, target, value=None, *, trusted=True, checked=None):
    state["events"].append(
        {
            "sequence": len(state["events"]),
            "type": kind,
            "target": target,
            "trusted": trusted,
            "value": value,
            "checked": checked,
        }
    )


def advance(spec, before):
    after = copy.deepcopy(before)
    step = fixtures.teacher_step(spec, before)
    if step["operation"] == "DONE":
        return after
    if step["operation"] == "TYPE_TEXT":
        after["note_value"] = spec["desired_text"]
        event(after, "input", "entry-text", spec["desired_text"])
    elif step["operation"] == "SELECT":
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
        assert step["operation"] == "CLICK" and step["label"] == spec["submit_label"]
        radio = next((r for r in after["radios"] if r["checked"]), None)
        after.update(
            submitted=True,
            submission_count=before["submission_count"] + 1,
            submitted_radio_value=radio["value"] if radio else "",
            submitted_radio_label=radio["label"] if radio else "",
            submitted_note=after["note_value"],
            submitted_mode_value=after["mode_value"],
            submitted_mode_label=after["mode_label"],
        )
        after["status"] = f"Submitted entry {after['submission_count']}."
        event(after, "click", "commit-control", "" if spec["commit_kind"] == "button" else None)
        return after
    after.update(submitted=False, status="Edited; submit the entry.")
    return after


def route(spec):
    states = [initial(spec)]
    for _ in range(16):
        if fixtures.teacher_step(spec, states[-1])["operation"] == "DONE":
            return states
        states.append(advance(spec, states[-1]))
    raise AssertionError("Authored route exceeded original16 ticks")


def effect(spec, before, after, step=None):
    return fixtures.verify_transition(
        spec,
        before,
        after,
        fixtures.teacher_step(spec, before) if step is None else step,
        after["events"][len(before["events"]) :],
    )


def test_closed_deterministic_detached_train_namespace_and_exclusion_hashes():
    first = specs()
    assert len(first) == len(fixtures.FAMILIES) == 24
    assert first == specs()
    assert fixtures.SCHEMA == "decision3-owned-navigation-distractor-control-scenarios-v1"
    assert fixtures.NAMESPACE == "decision3-owned-navigation-distractor-control-training-v1"
    assert fixtures.EXPECTED_MENU_COUNTS == (6,)
    assert len({s["id"] for s in first}) == len({s["goal_sha256"] for s in first}) == 24
    assert all(s["split"] == "train" and s["expected_menu_counts"] == [6] for s in first)
    assert Counter(s["authored_click_candidates"] for s in first) == {14: 8, 18: 8, 24: 8}
    with pytest.raises(ValueError, match="collides"):
        fixtures.scenarios(918, {first[0]["goal_sha256"]})
    first[0]["navigation_links"][0]["label"] = "mutated detached object"
    assert first != specs()


@pytest.mark.parametrize("seed", [True, -1, 2**63, "918"])
def test_invalid_seed_is_not_an_authored_scene(seed):
    with pytest.raises(ValueError, match="Seed"):
        fixtures.scenarios(seed, {"0" * 64})


@pytest.mark.parametrize("exclusions", [[], ["x" * 64], ["0" * 64, "0" * 64]])
def test_exclusions_are_required_hashes_only(exclusions):
    with pytest.raises(ValueError):
        fixtures.scenarios(918, exclusions)


@pytest.mark.parametrize(
    "field",
    [
        "namespace",
        "goal",
        "initial",
        "navigation_links",
        "layout_order",
        "commit_kind",
        "authored_click_candidates",
    ],
)
def test_scene_rewrite_cannot_relabel_or_remove_distractors(field):
    spec = specs()[7]
    if field == "initial":
        spec[field]["note_value"] = spec["desired_text"]
    elif field == "navigation_links":
        spec[field].pop()
    elif field == "layout_order":
        spec[field] = list(reversed(spec[field]))
    else:
        spec[field] = "changed"
    with pytest.raises(ValueError):
        fixtures.render_fixture(spec)


@pytest.mark.parametrize("index", range(24))
def test_complete_dom_preserves_every_link_and_movable_control_without_teacher_annotations(index):
    spec = specs()[index]
    rendered = fixtures.render_fixture(spec)
    controls = Controls()
    controls.feed(rendered)
    assert len(controls.click_ids) == spec["authored_click_candidates"]
    assert len(set(controls.click_ids)) == len(controls.click_ids)
    expected = []
    for token in spec["layout_order"]:
        expected.extend(
            [r["id"] for r in spec["radios"]]
            if token == "radio"
            else ["entry-text"]
            if token == "text"
            else []
            if token == "select"
            else ["commit-control"]
            if token == "submit"
            else [token]
        )
    assert controls.click_ids == expected
    nav = {a["id"]: a for a in controls.anchors if a["id"] != "commit-control"}
    assert len(nav) == len(spec["navigation_links"])
    assert all(nav[row["id"]]["href"] == row["href"] for row in spec["navigation_links"])
    assert all(row["href"].startswith("/owned-navigation/") for row in spec["navigation_links"])
    assert all("disabled" not in attrs for attrs in controls.fields.values())
    assert controls.fields["entry-text"]["value"] == spec["initial"]["note_value"]
    assert controls.option_values == ["", *(o["value"] for o in spec["options"])]
    assert len(controls.radios) == len(spec["radios"])
    assert all("disabled" not in attrs for attrs in controls.radios)
    assert controls.fields["commit-control"].get("href") == (
        "#saved-record" if spec["commit_kind"] == "link" else None
    )
    for hidden in (
        spec["id"],
        spec["ancestor_id"],
        spec["family"],
        spec["goal"],
        "required_fields",
        "desired_",
        "click_band",
        "authored_click_candidates",
    ):
        assert hidden not in rendered
    script = rendered.split("<script>", 1)[1]
    assert spec["desired_text"] not in script
    assert spec["desired_radio_value"] not in script
    assert spec["desired_option_value"] not in script
    assert "event.preventDefault()" in script


def test_active_click_locations_cover_early_middle_late_and_helpful_links_are_not_forbidden():
    observed = Counter()
    helpful = set()
    positions = set()
    for spec in specs():
        controls = Controls()
        controls.feed(fixtures.render_fixture(spec))
        active_click = next(
            fixtures.teacher_step(spec, state)
            for state in route(spec)
            if fixtures.teacher_step(spec, state)["operation"] == "CLICK"
        )
        target = (
            next(r["id"] for r in spec["radios"] if r["value"] == spec["desired_radio_value"])
            if active_click["label"] == spec["desired_radio_label"]
            else "commit-control"
        )
        ordinal = controls.click_ids.index(target) + 1
        positions.add(ordinal)
        observed[spec["click_band"]] += 1
        helpful.add(spec["commit_kind"])
        if spec["click_band"] == "early":
            assert 1 <= ordinal <= 3
        elif spec["click_band"] == "late":
            assert spec["authored_click_candidates"] - 2 <= ordinal
        else:
            assert 4 <= ordinal < spec["authored_click_candidates"] - 2
    assert observed == {"early": 8, "middle": 8, "late": 8}
    assert min(positions) <= 2 and max(positions) > 20
    assert helpful == {"button", "link"}


def test_independent_domains_labels_defaults_and_permutations_vary_across_seeds():
    cards, fields, commits, layouts, choices = set(), set(), set(), set(), set()
    for seed in range(8):
        for spec in specs(seed):
            cards.add(spec["title"].rsplit(" ", 1)[0])
            fields.add(spec["field_label"])
            commits.add(spec["submit_label"])
            layouts.add(tuple(spec["layout_order"]))
            choices.add(tuple(r["label"] for r in spec["radios"]))
    assert len(cards) >= 6 and len(fields) >= 6 and len(commits) >= 4
    assert len(layouts) > 100 and len(choices) > 10


def test_all_authored_effects_and_fresh_done_use_native_identity_and_no_navigation():
    counts = Counter()
    for spec in specs():
        states = route(spec)
        assert not fixtures.verify_goal(spec, states[0])
        for before, after in zip(states, states[1:]):
            proof = effect(spec, before, after)
            assert proof["positive_supervision"] is True
            assert proof["requires_canonical_native_executor_join"] is True
            assert after["navigation"] == before["navigation"]
            counts[proof["operation"]] += 1
        done = effect(spec, states[-1], copy.deepcopy(states[-1]))
        assert done["operation"] == "DONE" and done["input_executed"] is False
        counts["DONE"] += 1
    assert counts == {"CLICK": 33, "TYPE_TEXT": 9, "SELECT": 6, "DONE": 24}


@pytest.mark.parametrize(
    "corruption",
    [
        "url",
        "navigation-label",
        "navigation-href",
        "commit-tag",
        "event-prefix",
        "untrusted",
        "target",
    ],
)
def test_navigation_or_control_identity_changes_cannot_be_positive_labels(corruption):
    spec = specs()[0]
    before, after = route(spec)[:2]
    if corruption == "url":
        after["url"] += "/account"
    elif corruption.startswith("navigation"):
        after["navigation"][0]["label" if corruption.endswith("label") else "href"] = "changed"
    elif corruption == "commit-tag":
        after["commit_tag"] = "foreign"
    elif corruption == "event-prefix":
        after["events"][0]["sequence"] = True
    elif corruption == "untrusted":
        after["events"][0]["trusted"] = False
    else:
        after["events"][0]["target"] = spec["navigation_links"][0]["id"]
    with pytest.raises(ValueError):
        effect(spec, before, after)


def test_irrelevant_navigation_click_and_repeated_satisfied_radio_are_not_teacher_steps():
    spec = specs()[0]
    before, selected = route(spec)[:2]
    wrong = dict(fixtures.teacher_step(spec, before), label=spec["navigation_links"][0]["label"])
    with pytest.raises(ValueError, match="explicit authored route"):
        effect(spec, before, selected, wrong)
    with pytest.raises(ValueError, match="explicit authored route"):
        effect(spec, selected, copy.deepcopy(selected), fixtures.teacher_step(spec, before))


def test_link_and_button_commit_allow_only_unchanged_trusted_text_blur_then_click():
    for kind in ("link", "button"):
        spec = next(s for s in specs() if s["commit_kind"] == kind and "text" in s["goal_order"])
        before, after = route(spec)[-2:]
        click = copy.deepcopy(after["events"][-1])
        after["events"] = copy.deepcopy(before["events"])
        event(after, "change", "entry-text", before["note_value"])
        click["sequence"] = len(after["events"])
        after["events"].append(click)
        assert effect(spec, before, after)["goal_verified_after"] is True
        changed = copy.deepcopy(after)
        changed["events"][-2]["value"] = "different field value"
        with pytest.raises(ValueError, match="Trusted submit"):
            effect(spec, before, changed)


def test_done_requires_actual_saved_record_and_never_executes_an_input():
    spec = specs()[7]
    complete = route(spec)[-1]
    assert fixtures.verify_goal(spec, complete)
    for field in ("submitted_note", "submitted_mode_value", "submitted_radio_value"):
        wrong = copy.deepcopy(complete)
        wrong[field] = "not the current DOM value"
        assert not fixtures.verify_goal(spec, wrong)
    changed = copy.deepcopy(complete)
    event(changed, "click", spec["navigation_links"][0]["id"])
    with pytest.raises(ValueError, match="must not execute"):
        effect(spec, complete, changed)
