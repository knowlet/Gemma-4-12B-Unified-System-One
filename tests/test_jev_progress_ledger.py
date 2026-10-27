"""Synthetic, task-independent progress signals; no browser or model."""

import copy

import pytest

from s1.jev_progress_ledger import ProgressLedger


def state(
    *,
    location="https://example.invalid/a",
    visible="ready",
    controls=None,
    progress=None,
    complete=True,
):
    return {
        "location": location,
        "visible": visible,
        "controls": controls if controls is not None else {"primary": {"enabled": True}},
        "progress": progress if progress is not None else {},
        "complete": complete,
    }


def action(target="primary", *, kind="click", arguments=None):
    return {"kind": kind, "target": target, "arguments": arguments or {}}


def record(ledger, before, after, selected=None):
    return ledger.record(before=before, action=selected or action(), after=after, executed=True)


def test_repeated_noop_requires_same_executed_target_and_projection():
    ledger = ProgressLedger()
    a = state()
    assert not record(ledger, a, a).repeated_transition
    result = record(ledger, a, a)
    assert result.observation == "known" and result.repeated_transition
    assert not result.alternating_cycle
    assert not record(ledger, a, a, action("another-control")).repeated_transition


def test_order_and_unselected_timing_noise_do_not_change_projection():
    ledger = ProgressLedger()
    # The producer explicitly selects stable semantic fields; unrelated clock
    # observations never enter the projection. Control map order is irrelevant.
    clock_before = {"seconds": 100, "controls": {"a": 1, "b": 2}}
    clock_after = {"seconds": 101, "controls": {"b": 2, "a": 1}}
    a = state(controls=clock_before["controls"])
    b = state(controls=clock_after["controls"])
    before = copy.deepcopy((a, b))
    record(ledger, a, b)
    result = record(ledger, b, a)
    assert result.repeated_transition
    assert (a, b) == before
    assert clock_before["seconds"] != clock_after["seconds"]


def test_progress_counter_and_real_control_changes_break_noop():
    ledger = ProgressLedger()
    states = [state(progress={"cursor": index}) for index in range(4)]
    for left, right in zip(states, states[1:]):
        result = record(ledger, left, right, action("next"))
        assert not result.repeated_transition and not result.alternating_cycle
    changed = state(progress={"cursor": 3}, controls={"primary": {"checked": True}})
    assert not record(ledger, states[-1], changed).repeated_transition


def test_distinct_elements_with_same_display_label_remain_distinct():
    ledger = ProgressLedger()
    a = state(controls={"first": {"label": "Open"}, "second": {"label": "Open"}})
    record(ledger, a, a, action("first"))
    assert not record(ledger, a, a, action("second")).repeated_transition


def test_two_page_cycle_requires_same_state_transition_and_action():
    ledger = ProgressLedger()
    a = state(location="https://example.invalid/a", visible="A")
    b = state(location="https://example.invalid/b", visible="B")
    assert not record(ledger, a, b, action("to-b")).alternating_cycle
    assert not record(ledger, b, a, action("to-a")).alternating_cycle
    result = record(ledger, a, b, action("to-b"))
    assert result.repeated_transition and result.alternating_cycle


def test_cycle_with_a_different_target_is_not_repeated():
    ledger = ProgressLedger()
    a, b = state(visible="A"), state(visible="B")
    record(ledger, a, b, action("one"))
    record(ledger, b, a, action("back"))
    result = record(ledger, a, b, action("different"))
    assert not result.repeated_transition and not result.alternating_cycle


def test_late_change_and_unknown_observation_do_not_become_no_progress():
    ledger = ProgressLedger()
    waiting, loaded = state(visible="loading"), state(visible="loaded")
    record(ledger, waiting, waiting, action(kind="wait"))
    result = record(ledger, waiting, loaded, action(kind="wait"))
    assert not result.repeated_transition
    unknown = record(ledger, loaded, None)
    assert unknown.observation == "unknown"
    assert not unknown.repeated_transition and not unknown.alternating_cycle
    assert not record(ledger, loaded, loaded).repeated_transition


def test_unknown_or_discontinuous_chain_prevents_comparison_across_gap():
    for middle in (None, state(visible="different")):
        ledger = ProgressLedger()
        a = state()
        record(ledger, a, a)
        if middle is None:
            record(ledger, a, None)
            record(ledger, None, a)
        else:
            record(ledger, middle, middle)
        assert not record(ledger, a, a).repeated_transition


def test_incomplete_partial_values_are_unknown_and_cannot_trigger_repetition_or_cycle():
    ledger = ProgressLedger()
    a, b = state(visible="A"), state(visible="B")
    partial_b = state(visible="B", controls={"primary": {"value": None}}, complete=False)
    record(ledger, a, b, action("to-b"))
    partial = record(ledger, partial_b, a, action("to-a"))
    assert partial.observation == "unknown"
    assert not partial.repeated_transition and not partial.alternating_cycle
    again = record(ledger, a, b, action("to-b"))
    assert not again.repeated_transition and not again.alternating_cycle
    partial_noop = record(ledger, partial_b, partial_b)
    assert partial_noop.observation == "unknown"
    assert not partial_noop.repeated_transition


def test_complete_confirmed_null_remains_a_known_comparable_value():
    ledger = ProgressLedger()
    observed = state(controls={"field": {"value": None}}, progress={"selected": None})
    assert record(ledger, observed, observed).observation == "known"
    assert record(ledger, observed, observed).repeated_transition


def test_evicted_transitions_and_different_action_arguments_do_not_match():
    ledger = ProgressLedger(history_limit=3)
    a = state()
    record(ledger, a, a, action(arguments={"value": "first"}))
    for value in ("second", "third", "fourth"):
        assert not record(ledger, a, a, action(arguments={"value": value})).repeated_transition
    assert not record(ledger, a, a, action(arguments={"value": "first"})).repeated_transition


def test_invalid_or_unexecuted_input_never_enters_ledger():
    ledger = ProgressLedger()
    a = state()
    with pytest.raises(ValueError, match="executed"):
        ledger.record(before=a, action=action(), after=a, executed=False)
    with pytest.raises(ValueError, match="projection"):
        record(ledger, {"location": "x"}, a)
    with pytest.raises(ValueError, match="finite JSON"):
        record(ledger, state(progress={"nan": float("nan")}), a)
    assert record(ledger, a, a).sequence == 1
    assert (
        not ProgressLedger()
        .record(before=None, action=action(), after=None, executed=True)
        .repeated_transition
    )


def test_sensitive_input_is_not_retained_in_output_or_ledger():
    ledger = ProgressLedger()
    secret = "sensitive-synthetic-value"
    current = state(visible=secret, controls={"node": {"value": secret}})
    chosen = action("node", arguments={"value": secret})
    result = record(ledger, current, current, chosen)
    assert secret not in repr(result)
    assert secret not in repr(ledger._history)
