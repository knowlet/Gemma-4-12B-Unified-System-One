"""Pure observed-state bookkeeping; no browser/model/framework is imported."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).with_name("observation_revisit.py")
SPEC = importlib.util.spec_from_file_location("revisit_pure_test", SOURCE)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def page(text="A"):
    url = "https://example.test/search?q=literal-id-42"
    actions = [
        {
            "id": "e1",
            "node": 1,
            "kind": "fill",
            "role": "searchbox",
            "label": "Search",
            "value": "e42",
            "rect": {"x": 1, "y": 2, "w": 30, "h": 20},
        },
        {
            "id": "e2",
            "node": 2,
            "kind": "click",
            "role": "button",
            "label": "Go",
            "name": "semantic-name",
            "value": "",
            "rect": {"x": 40, "y": 2, "w": 20, "h": 20},
        },
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
        {"id": "wait", "kind": "wait", "label": "Wait for the page to update"},
    ]
    return {
        "observation_profile": {
            "id": "composed-visible-hover-entries-settled-v8",
            "visited_nodes": 10,
        },
        "url": url,
        "title": "Search",
        "w": 800,
        "h": 600,
        "text": text,
        "scroll": {"y": 0, "height": 2000},
        "actions": actions,
        "page_key": [1000.5, url, 0, 0, 800, 600, [[1, "e42", False, None, False, False]]],
        "guards": {
            "1": [1, "searchbox", "Search", "e42", False],
            "2": [2, "button", "Go", "row A"],
        },
        "marker": [1000.5, url, 0, 0, 800, 600, "Search", text],
        "omitted_actions": 0,
        "fingerprint": "untrusted-old-fingerprint",
    }


def keys(value):
    return R.observation_signatures(value, include_projection=True)


def test_same_document_ABA_and_explicit_projection_default():
    tracker = R.ObservationRevisitTracker()
    assert tracker.observe(page(), observation_sequence=0)["observation_notes"] == []
    tracker.observe(page("B"), observation_sequence=1)
    result = tracker.observe(page(), observation_sequence=2)
    assert result["exact"]["previous_observation_sequences"] == [0]
    assert result["projection"] is None
    assert result["observation_notes"] == [
        "This document observation exactly matches retained observation sequences 0."
    ]


def test_reload_binding_and_geometry_are_projection_only_not_action_authority():
    first, second = page(), page()
    second["page_key"][0] += 500
    second["page_key"][6][0][0] = 101
    second["marker"][0] += 500
    second["actions"][0]["node"] = 101
    second["actions"][1]["node"] = 102
    second["actions"][0]["rect"]["x"] += 10
    second["guards"] = {
        "101": [101, "searchbox", "Search", "e42", False],
        "102": [102, "button", "Go", "row A"],
    }
    before = copy.deepcopy((first, second))
    tracker = R.ObservationRevisitTracker(include_projection=True)
    tracker.observe(first, observation_sequence=0)
    result = tracker.observe(second, observation_sequence=1)
    assert result["exact"]["previous_observation_sequences"] == []
    assert result["projection"]["previous_observation_sequences"] == [0]
    assert "excluding native binding IDs and geometry" in result["observation_notes"][0]
    assert "not complete state or action identity" in result["projection"]["scope"]
    assert (first, second) == before


@pytest.mark.parametrize(
    "change",
    [
        "url",
        "title",
        "text",
        "value",
        "checked",
        "selected",
        "editable",
        "scroll-y",
        "scroll-x",
        "height",
        "viewport",
        "order",
        "unknown",
        "name",
        "semantic-id",
    ],
)
def test_projection_keeps_observed_content_values_and_order(change):
    old, new = page(), page()
    if change == "url":
        new["url"] += "&page=2"
        new["page_key"][1] = new["url"]
    elif change in {"title", "text"}:
        new[change] += " changed"
    elif change in {"value", "checked", "selected", "unknown", "name"}:
        new["actions"][0][change] = "changed"
    elif change == "editable":
        new["page_key"][6][0][1] = "different off-viewport observed value"
    elif change == "scroll-y":
        new["scroll"]["y"] = new["page_key"][3] = 560
    elif change == "scroll-x":
        new["page_key"][2] = 20
    elif change == "height":
        new["scroll"]["height"] += 1
    elif change == "viewport":
        new["w"] = new["page_key"][4] = 900
    elif change == "order":
        new["actions"][:2] = new["actions"][1::-1]
        new["actions"][0]["id"], new["actions"][1]["id"] = "e1", "e2"
    else:
        old["actions"][0]["id"] = "actual-semantic-id"
        new["actions"][0]["id"] = "different-semantic-id"
    before = copy.deepcopy(new)
    assert all(a != b for a, b in zip(keys(old), keys(new)))
    assert new == before


@pytest.mark.parametrize("change", ["row-guard", "page-key", "rect", "marker"])
def test_exact_includes_guards_document_and_geometry_even_if_old_fingerprint_same(change):
    old, new = page(), page()
    if change == "row-guard":
        new["guards"]["2"][-1] = "different row destination/content"
    elif change == "page-key":
        new["page_key"][0] += 1
    elif change == "rect":
        new["actions"][1]["rect"]["y"] += 1
    else:
        new["marker"][-1] += " changed marker"
    assert old["fingerprint"] == new["fingerprint"]
    exact_old, projection_old = keys(old)
    exact_new, projection_new = keys(new)
    assert exact_old != exact_new
    assert projection_old == projection_new


def test_no_general_id_regex_or_value_normalization_and_future_fields_retained():
    old, new = page(), page()
    old["actions"][0].update(id="semantic-id", node="semantic-node", rect="semantic-rect")
    new["actions"][0].update(id="semantic-id", node="different-node", rect="semantic-rect")
    assert keys(old)[1] != keys(new)[1]
    new = copy.deepcopy(old)
    new["actions"][0]["value"] = "e43"
    assert keys(old)[1] != keys(new)[1]


@pytest.mark.parametrize(
    "column,replacement", [(1, "new query"), (2, True), (3, 2), (4, True), (5, True)]
)
def test_all_observed_editable_state_is_preserved_without_node_identity(column, replacement):
    old, new = page(), page()
    new["page_key"][6][0][column] = replacement
    assert keys(old)[1] != keys(new)[1]


def test_known_proxy_hover_bindings_are_optional_projection_not_exact_identity():
    old, new = page(), page()
    for value in (old, new):
        value["actions"][0].update(
            control_node=5, control_type="checkbox", control_guard=[5, "binding"]
        )
        value["actions"][1].update(affordance="css-hover-entry-v5", hover_owner=6)
    new["actions"][0].update(node=101, control_node=105, control_guard=[105, "new binding"])
    new["actions"][1].update(node=102, hover_owner=106)
    assert keys(old)[0] != keys(new)[0]
    assert keys(old)[1] == keys(new)[1]


def test_profile_diagnostics_are_exact_only_but_profile_identity_is_kept():
    old, new = page(), page()
    new["observation_profile"]["visited_nodes"] += 1
    assert keys(old)[0] != keys(new)[0]
    assert keys(old)[1] == keys(new)[1]
    new["observation_profile"]["id"] = "a different observation source"
    assert keys(old)[1] != keys(new)[1]


def test_existing_text_is_not_truncated_and_semantic_DOM_fields_are_preserved():
    old, new = page("x" * 10000 + "first"), page("x" * 10000 + "last")
    assert keys(old)[1] != keys(new)[1]
    old, new = page(), page()
    old["actions"][0]["dom_id"] = "e123"
    new["actions"][0]["dom_id"] = "e124"
    assert keys(old)[1] != keys(new)[1]


@pytest.mark.parametrize("flag", [1, None, "true"])
def test_projection_is_explicit_boolean(flag):
    with pytest.raises(ValueError):
        R.ObservationRevisitTracker(include_projection=flag)
    with pytest.raises(ValueError):
        R.observation_signatures(page(), include_projection=flag)


def test_window_eviction_is_idempotent_and_output_mutation_cannot_corrupt_tracker():
    tracker = R.ObservationRevisitTracker(window=2, include_projection=True)
    tracker.observe(page("A"), observation_sequence=0)
    tracker.observe(page("B"), observation_sequence=1)
    tracker.observe(page("C"), observation_sequence=2)
    assert (
        tracker.observe(page("A"), observation_sequence=3)["exact"][
            "previous_observation_sequences"
        ]
        == []
    )
    result = tracker.observe(page("A"), observation_sequence=4)
    assert result["exact"]["previous_observation_sequences"] == [3]
    result["exact"]["previous_observation_sequences"].append(999)
    assert tracker.observe(page("A"), observation_sequence=4)["exact"][
        "previous_observation_sequences"
    ] == [3]
    assert tracker.retained_count == 2
    with pytest.raises(ValueError, match="Sequence content changed"):
        tracker.observe(page("changed"), observation_sequence=4)
    with pytest.raises(ValueError, match="backwards"):
        tracker.observe(page(), observation_sequence=3)
    assert tracker.retained_count == 2


def test_maximum_window_and_long_sequences_remain_bounded_without_raw_pages():
    tracker = R.ObservationRevisitTracker(include_projection=True)
    for i in range(100):
        result = tracker.observe(page(), observation_sequence=R.MAX_SEQUENCE - 100 + i)
        assert tracker.retained_count <= 32
        assert len(result["exact"]["previous_observation_sequences"]) <= 32
        assert (
            len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode())
            <= R.MAX_METADATA_BYTES
        )
    assert all(type(x) in (str, int, type(None)) for entry in tracker._entries for x in entry)
    assert "example.test" not in repr(vars(tracker))


@pytest.mark.parametrize("window", [0, 33, True, 1.0, None])
def test_invalid_window_refused(window):
    with pytest.raises(ValueError):
        R.ObservationRevisitTracker(window=window)


@pytest.mark.parametrize("sequence", [-1, True, 1.0, None, R.MAX_SEQUENCE + 1])
def test_invalid_sequence_refused_without_recording(sequence):
    tracker = R.ObservationRevisitTracker()
    with pytest.raises(ValueError):
        tracker.observe(page(), observation_sequence=sequence)
    assert tracker.retained_count == 0


@pytest.mark.parametrize(
    "bad", ["missing", "nonfinite", "cycle", "key", "surrogate", "contradictory-scroll"]
)
def test_invalid_observation_refuses_atomically(bad):
    value = page()
    if bad == "missing":
        del value["guards"]
    elif bad == "nonfinite":
        value["unknown"] = float("nan")
    elif bad == "cycle":
        value["unknown"] = value
    elif bad == "key":
        value["guards"][1] = "not a JSON string key"
    elif bad == "surrogate":
        value["text"] = "\ud800"
    else:
        value["scroll"]["x"] = 12
    tracker = R.ObservationRevisitTracker(include_projection=True)
    with pytest.raises(ValueError):
        tracker.observe(value, observation_sequence=0)
    assert tracker.retained_count == 0
