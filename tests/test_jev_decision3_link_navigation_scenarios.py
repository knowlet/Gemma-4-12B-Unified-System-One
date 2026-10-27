"""Tests use authored DOM evidence; no browser or benchmark data is involved."""

import copy
from html.parser import HTMLParser

import pytest

from scripts import jev_decision3_link_navigation_scenarios as navigation


class Anchors(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.current = {"path": dict(attrs)["href"], "label": ""}

    def handle_data(self, data):
        if self.current is not None:
            self.current["label"] += data

    def handle_endtag(self, tag):
        if tag == "a":
            self.links.append(self.current)
            self.current = None


def fixtures(seed=41):
    return navigation.scenarios(seed, {"0" * 64})


def state(spec, view, completed=False, events=None):
    return {
        "pathname": spec["paths"][view],
        "view": view,
        "ready": "complete",
        "heading": {
            "landing": spec["title"],
            "workspace": spec["target_link"],
            "detail": spec["record_label"],
        }[view],
        "status": (spec["completed_text"] if completed else spec["pending_text"])
        if view == "detail"
        else None,
        "count": int(completed) if view == "detail" else None,
        "links": copy.deepcopy(spec[view + "_links"]) if view != "detail" else [],
        "events": list(events or []),
    }


def test_deterministic_train_only_independent_document_inventory():
    specs = fixtures()
    assert specs == fixtures()
    assert len(specs) == 27
    for key in ("id", "ancestor_id", "goal_sha256"):
        assert len({spec[key] for spec in specs}) == len(specs)
    all_paths = set()
    for spec in specs:
        assert spec["split"] == "train"
        assert (
            spec["start_path"]
            == spec["paths"][
                {"landing": "landing", "workspace": "workspace", "complete": "detail"}[
                    spec["start_mode"]
                ]
            ]
        )
        docs = navigation.render_documents(spec)
        assert len(docs) == 3 + 2 * (spec["density"] - 1)
        assert not all_paths.intersection(docs)
        all_paths.update(docs)
        for view in ("landing", "workspace"):
            parser = Anchors()
            parser.feed(docs[spec["paths"][view]])
            assert parser.links == spec[view + "_links"]
            assert len(parser.links) == spec["density"]
            assert all(link["path"] in docs for link in parser.links)
        assert spec["paths"]["workspace"] != spec["paths"]["detail"]
        detail = docs[spec["paths"]["detail"]]
        assert spec["action_label"] in detail
        if spec["start_mode"] == "complete":
            assert "data-count='1'>" + spec["completed_text"] in detail
        else:
            assert "data-count='0'>" + spec["pending_text"] in detail


def test_positions_vary_and_target_is_a_semantic_label():
    landing_positions = set()
    record_positions = set()
    for seed in range(6):
        for spec in fixtures(seed):
            landing = spec["landing_links"]
            records = spec["workspace_links"]
            landing_positions.add(
                next(i for i, link in enumerate(landing) if link["label"] == spec["target_link"])
            )
            record_positions.add(
                next(i for i, link in enumerate(records) if link["label"] == spec["record_label"])
            )
            assert sum(link["path"] == spec["paths"]["workspace"] for link in landing) == 1
            assert sum(link["path"] == spec["paths"]["detail"] for link in records) == 1
            assert len({link["label"] for link in landing}) == len(landing)
            assert len({link["label"] for link in records}) == len(records)
    assert len(landing_positions) >= 6
    assert len(record_positions) >= 6


def test_navigation_completion_and_done_for_every_authored_scenario():
    click = {"type": "click", "id": "complete-record", "trusted": True}
    for spec in fixtures():
        landing = state(spec, "landing")
        workspace = state(spec, "workspace")
        pending = state(spec, "detail")
        completed = state(spec, "detail", True, [click])
        for before, after, label in (
            (landing, workspace, spec["target_link"]),
            (workspace, pending, spec["record_label"]),
        ):
            step = navigation.teacher_step(spec, before)
            assert step["operation"] == "CLICK" and step["label"] == label
            evidence = navigation.verify_transition(spec, before, after, step, [])
            assert evidence["kind"] == "document_navigation"
            assert evidence["teacher_label"] == label
            assert evidence["source_path"] == before["pathname"]
            assert evidence["destination_path"] == after["pathname"]
            assert evidence["supervision_intent"] == {"operation": True, "click_target": True}
            assert before["pathname"] != after["pathname"]
            assert not navigation.verify_goal(spec, before)
        step = navigation.teacher_step(spec, pending)
        assert step["operation"] == "CLICK" and step["label"] == spec["action_label"]
        assert (
            navigation.verify_transition(spec, pending, completed, step, [click])["kind"]
            == "visible_completion"
        )
        assert navigation.verify_goal(spec, completed)
        done = navigation.teacher_step(spec, completed)
        assert done["operation"] == "DONE"
        terminal = navigation.verify_transition(spec, completed, completed, done, [])
        assert terminal["kind"] == "terminal"
        assert terminal["supervision_intent"] == {"operation": True, "click_target": False}


def test_reject_wrong_page_clicks_and_forged_or_missing_completion():
    spec = fixtures()[0]
    landing, workspace, pending = (state(spec, view) for view in ("landing", "workspace", "detail"))
    landing_step = navigation.teacher_step(spec, landing)
    with pytest.raises(ValueError):
        navigation.verify_transition(spec, landing, pending, landing_step, [])
    wrong_step = dict(landing_step, label=spec["landing_links"][0]["label"] + " wrong")
    with pytest.raises(ValueError):
        navigation.verify_transition(spec, landing, workspace, wrong_step, [])
    decoy = dict(workspace, pathname=spec["decoy_paths"][0])
    with pytest.raises(ValueError):
        navigation.teacher_step(spec, decoy)
    forged = dict(pending, status=spec["completed_text"])
    assert not navigation.verify_goal(spec, forged)
    assert not navigation.verify_goal(spec, pending)
    click = {"type": "click", "id": "complete-record", "trusted": False}
    untrusted = state(spec, "detail", True, [click])
    with pytest.raises(ValueError):
        navigation.verify_transition(
            spec, pending, untrusted, navigation.teacher_step(spec, pending), [click]
        )
    completed = state(spec, "detail", True)
    with pytest.raises(ValueError):
        navigation.verify_transition(
            spec, pending, completed, navigation.teacher_step(spec, pending), []
        )
    with pytest.raises(ValueError):
        navigation.verify_transition(
            spec, completed, completed, navigation.teacher_step(spec, completed), [click]
        )
    assert not navigation.verify_goal(spec, dict(completed, heading="Other record"))
    assert not navigation.verify_goal(spec, dict(completed, ready="loading"))


def test_exclusions_and_tampered_documents_are_rejected():
    specs = fixtures()
    with pytest.raises(ValueError, match="overlaps"):
        navigation.scenarios(41, {specs[0]["goal_sha256"]})
    for exclusions in (set(), {"not-a-hash"}, ["0" * 64, "0" * 64]):
        with pytest.raises(ValueError):
            navigation.scenarios(41, exclusions)
    for seed in (True, -1, 2**32, 1.5):
        with pytest.raises(ValueError):
            navigation.scenarios(seed, {"0" * 64})
    altered = copy.deepcopy(specs[0])
    altered["landing_links"][0]["path"] = "https://outside.example/"
    with pytest.raises(ValueError):
        navigation.render_documents(altered)
    altered = copy.deepcopy(specs[0])
    altered["goal"] += " changed"
    with pytest.raises(ValueError):
        navigation.render_documents(altered)
