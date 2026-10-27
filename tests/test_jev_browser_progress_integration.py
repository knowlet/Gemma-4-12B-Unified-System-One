"""Synthetic bounded visible snapshots; no private cases, browser, or model."""

import copy

import jev_browser_progress_integration as integration
import pytest

EPISODE = "a" * 32


@pytest.fixture(autouse=True)
def release_for_test(monkeypatch):
    monkeypatch.setattr(integration, "SOURCE_ONLY", False)


def page(
    *,
    url="https://example.invalid/a",
    text="Ready",
    node=1,
    fingerprint="f1",
    omitted=0,
    rect=0,
    controls=None,
):
    actions = (
        controls
        if controls is not None
        else [
            {
                "id": "e1",
                "node": node,
                "kind": "click",
                "label": "Open",
                "role": "button",
                "value": "",
                "rect": {"x": rect, "y": 0, "w": 10, "h": 10},
            },
        ]
    )
    guards = {
        str(action["node"]): [
            action["node"],
            action.get("role", "button"),
            action["label"],
            "",
            None,
            None,
            None,
            False,
            None,
            None,
            None,
            None,
            None,
            "scope",
        ]
        for action in actions
        if "node" in action
    }
    return {
        "url": url,
        "title": "Example",
        "text": text,
        "w": 800,
        "h": 600,
        "scroll": {"y": 0, "height": 800},
        "actions": actions,
        "guards": guards,
        "fingerprint": fingerprint,
        "page_key": [123, url],
        "omitted_actions": omitted,
        "marker": [123, url, 0, 0, 800, 600, "Example", text, [], []],
        "observation_profile": {
            "id": integration.PROFILE,
            "visited_nodes": 4,
            "node_limit": 20000,
            "depth_limit": 128,
            "node_limit_reached": False,
            "depth_limit_reached": False,
        },
    }


def entry(before, after, step, selected="e1"):
    candidate = next(action for action in before["actions"] if action["id"] == selected)
    return {
        "step": step,
        "choice": selected,
        "kind": candidate["kind"],
        "action": candidate["label"],
        "text": None,
        "page_changed": after["fingerprint"] != before["fingerprint"] if after else None,
        "url": after["url"] if after else before["url"],
    }


def execution(before, after, step, selected="e1"):
    return {
        "episode_id": EPISODE,
        "step": step,
        "choice": selected,
        "document": before["page_key"][0],
        "before_fingerprint": before["fingerprint"],
        "after_fingerprint": after["fingerprint"] if after else None,
        "status": "confirmed_executed",
    }


def committed(adapter, before, after, step, selected="e1"):
    return adapter.record_committed(
        episode_id=EPISODE,
        before=before,
        selected_id=selected,
        after=after,
        history_entry=entry(before, after, step, selected),
        execution=execution(before, after, step, selected),
    )


def annotate(adapter, request, before, after, step, selected="e1"):
    return adapter.annotate_next_policy(
        request,
        episode_id=EPISODE,
        current_page=after,
        history_entry=entry(before, after, step, selected),
        execution=execution(before, after, step, selected),
        freshness={
            "episode_id": EPISODE,
            "step": step,
            "document": after["page_key"][0],
            "fingerprint": after["fingerprint"],
            "status": "verified_current",
        },
    )


def body(snapshot=None, last=None):
    snapshot = snapshot or page()
    last = last or entry(snapshot, snapshot, 1)
    first = snapshot["actions"][0]
    element = {
        key: first[key]
        for key in ("role", "value", "checked", "selected", "expanded")
        if key in first
    }
    element.update(index="1", label=first["label"].split(" → ")[0], operations=["CLICK"])
    return {
        "model": "synthetic",
        "state": {
            "page": {key: snapshot[key] for key in ("url", "title", "text")},
            "elements": [element],
            "recent_actions": [
                {key: last.get(key) for key in ("action", "kind", "text", "page_changed")}
            ],
        },
        "questions": {"operation": {"criteria": {"a": "First", "b": "Second"}}},
    }


def test_gate_is_off_by_default(monkeypatch):
    monkeypatch.setattr(integration, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="independent review"):
        integration.BrowserProgressIntegration(episode_id=EPISODE)


def test_no_advisory_until_repeated_observed_projection_then_only_once():
    adapter = integration.BrowserProgressIntegration(episode_id=EPISODE)
    first, second, third = (
        page(fingerprint="1"),
        page(fingerprint="2", rect=1),
        page(fingerprint="3", rect=2),
    )
    assert not committed(adapter, first, second, 1)["repeated_transition"]
    request = body()
    assert adapter.annotate_next_policy(request) is request
    assert committed(adapter, second, third, 2)["repeated_transition"]
    original = copy.deepcopy(request)
    request = body(third, entry(second, third, 2))
    original = copy.deepcopy(request)
    changed = annotate(adapter, request, second, third, 2)
    assert request == original and changed["questions"] == original["questions"]
    assert list(changed["questions"]["operation"]["criteria"]) == ["a", "b"]
    hint = changed["state"][integration.KEY]
    assert hint["scope"] == integration.SCOPE and hint["repeated_transition"]
    assert hint["before_limits"] == {
        "text_at_cap": False,
        "guard_text_at_cap": False,
        "omitted_actions": 0,
    }
    assert adapter.annotate_next_policy(request) is request


def test_reordered_actions_do_not_hide_same_visible_projection():
    actions = [
        {"id": "e1", "node": 1, "kind": "click", "label": "Open", "role": "button"},
        {"id": "e2", "node": 2, "kind": "click", "label": "Other", "role": "button"},
    ]
    a = page(controls=actions)
    b = page(controls=list(reversed(actions)), fingerprint="f2")
    c = page(controls=actions, fingerprint="f3")
    adapter = integration.BrowserProgressIntegration(episode_id=EPISODE)
    committed(adapter, a, b, 1)
    assert committed(adapter, b, c, 2)["repeated_transition"]


def test_same_label_different_actual_node_and_late_visible_progress_do_not_match():
    adapter = integration.BrowserProgressIntegration(episode_id=EPISODE)
    first, other, loaded = (
        page(node=1),
        page(node=2, fingerprint="f2"),
        page(node=2, text="Loaded", fingerprint="f3"),
    )
    committed(adapter, first, first, 1)
    assert not committed(adapter, other, other, 2)["repeated_transition"]
    assert not committed(adapter, other, loaded, 3)["repeated_transition"]
    request = body()
    assert adapter.annotate_next_policy(request) is request


def test_spa_cycle_is_scoped_advisory_and_documents_do_not_conflate():
    adapter = integration.BrowserProgressIntegration(episode_id=EPISODE)
    a, b = (
        page(url="https://example.invalid/a", text="A"),
        page(url="https://example.invalid/b", text="B", fingerprint="b"),
    )
    committed(adapter, a, b, 1)
    committed(adapter, b, a, 2)
    assert committed(adapter, a, b, 3)["alternating_cycle"]
    assert annotate(adapter, body(b, entry(a, b, 3)), a, b, 3)["state"][integration.KEY][
        "alternating_cycle"
    ]
    adapter = integration.BrowserProgressIntegration(episode_id=EPISODE)
    changed_doc = copy.deepcopy(a)
    changed_doc["page_key"][0] = 999
    committed(adapter, a, a, 1)
    assert not committed(adapter, changed_doc, changed_doc, 2)["repeated_transition"]


def test_truncated_scope_is_unknown_and_unbound_action_breaks_chain():
    adapter = integration.BrowserProgressIntegration(episode_id=EPISODE)
    capped = page(text="x" * 6000, omitted=3)
    assert committed(adapter, capped, capped, 1)["observation"] == "unknown"
    assert committed(adapter, capped, capped, 2)["observation"] == "unknown"
    request = body(capped)
    assert adapter.annotate_next_policy(request) is request


@pytest.mark.parametrize("cap", ["text", "guard", "omitted"])
def test_individual_snapshot_caps_are_unknown(cap):
    adapter = integration.BrowserProgressIntegration(episode_id=EPISODE)
    current = page()
    if cap == "text":
        current["text"] = current["marker"][7] = "x" * 6000
    elif cap == "guard":
        current["guards"]["1"][-1] = "x" * 6000
    else:
        current["omitted_actions"] = 1
    assert committed(adapter, current, current, 1)["observation"] == "unknown"
    assert committed(adapter, current, current, 2)["observation"] == "unknown"
    request = body(current)
    assert adapter.annotate_next_policy(request) is request


@pytest.mark.parametrize(
    "fault", ["missing", "not_executed", "error", "wrong_step", "wrong_target"]
)
def test_owner_execution_attestation_is_required_and_failures_break_chain(fault):
    adapter = integration.BrowserProgressIntegration(episode_id=EPISODE)
    a = page()
    committed(adapter, a, a, 1)
    history = entry(a, a, 2)
    proof = execution(a, a, 2)
    if fault == "missing":
        proof = None
    elif fault == "not_executed":
        history["executed"] = False
    elif fault == "error":
        history["error"] = "synthetic error"
    elif fault == "wrong_step":
        proof["step"] = 99
    else:
        proof["choice"] = "other"
    result = adapter.record_committed(
        episode_id=EPISODE,
        before=a,
        selected_id="e1",
        after=a,
        history_entry=history,
        execution=proof,
    )
    assert result["observation"] == "unknown"
    assert not committed(adapter, a, a, 3)["repeated_transition"]


@pytest.mark.parametrize(
    "fault", ["episode", "document", "step", "missing", "execution", "freshness"]
)
def test_pending_hint_requires_same_owner_episode_document_step_and_execution(fault):
    adapter = integration.BrowserProgressIntegration(episode_id=EPISODE)
    a = page()
    committed(adapter, a, a, 1)
    committed(adapter, a, a, 2)
    request = body(a, entry(a, a, 2))
    current = copy.deepcopy(a)
    history = entry(a, a, 2)
    proof = execution(a, a, 2)
    episode = EPISODE
    freshness = {
        "episode_id": EPISODE,
        "step": 2,
        "document": a["page_key"][0],
        "fingerprint": a["fingerprint"],
        "status": "verified_current",
    }
    if fault == "episode":
        episode = "b" * 32
    elif fault == "document":
        current["page_key"][0] = 999
        freshness["document"] = 999
    elif fault == "step":
        history["step"] = 3
    elif fault == "missing":
        current = None
    elif fault == "execution":
        proof["status"] = "failed"
    else:
        freshness = None
    assert (
        adapter.annotate_next_policy(
            request,
            episode_id=episode,
            current_page=current,
            history_entry=history,
            execution=proof,
            freshness=freshness,
        )
        is request
    )
    assert integration.KEY not in request["state"]
    assert annotate(adapter, request, a, a, 2) is request


def test_missing_guard_or_unbound_action_breaks_chain():
    adapter = integration.BrowserProgressIntegration(episode_id=EPISODE)
    ordinary = page()
    broken = copy.deepcopy(ordinary)
    del broken["guards"]["1"]
    assert committed(adapter, ordinary, broken, 3)["observation"] == "unknown"
    assert not committed(adapter, ordinary, ordinary, 4)["repeated_transition"]
    fabricated = entry(ordinary, ordinary, 5)
    fabricated["choice"] = "other"
    assert (
        adapter.record_committed(
            episode_id=EPISODE,
            before=ordinary,
            selected_id="e1",
            after=ordinary,
            history_entry=fabricated,
            execution=execution(ordinary, ordinary, 5),
        )["observation"]
        == "unknown"
    )
    request = body()
    assert adapter.annotate_next_policy(request) is request


def test_missing_post_observation_and_history_gap_never_bridge():
    adapter = integration.BrowserProgressIntegration(episode_id=EPISODE)
    a = page()
    committed(adapter, a, a, 1)
    assert committed(adapter, a, None, 2)["observation"] == "unknown"
    assert not committed(adapter, a, a, 3)["repeated_transition"]
    assert committed(adapter, a, a, 5)["observation"] == "unknown"
    assert not committed(adapter, a, a, 6)["repeated_transition"]


def test_invalid_profile_and_pending_key_collision_fail_without_mutating_request():
    adapter = integration.BrowserProgressIntegration(episode_id=EPISODE)
    a = page()
    bad = copy.deepcopy(a)
    bad["observation_profile"]["node_limit_reached"] = True
    assert committed(adapter, a, bad, 1)["observation"] == "unknown"
    committed(adapter, a, a, 2)
    committed(adapter, a, a, 3)
    request = body()
    request["state"][integration.KEY] = {"original": True}
    original = copy.deepcopy(request)
    assert annotate(adapter, request, a, a, 3) is request
    assert request == original


@pytest.mark.parametrize("change", ["page", "elements", "last_action"])
def test_next_policy_must_match_observed_page_controls_and_last_action(change):
    adapter = integration.BrowserProgressIntegration(episode_id=EPISODE)
    a = page()
    committed(adapter, a, a, 1)
    committed(adapter, a, a, 2)
    request = body(a, entry(a, a, 2))
    if change == "page":
        request["state"]["page"]["text"] = "changed"
    elif change == "elements":
        request["state"]["elements"][0]["label"] = "different"
    else:
        request["state"]["recent_actions"][-1]["action"] = "different"
    original = copy.deepcopy(request)
    assert annotate(adapter, request, a, a, 2) is request
    assert request == original
    correct = body(a, entry(a, a, 2))
    assert annotate(adapter, correct, a, a, 2) is correct
