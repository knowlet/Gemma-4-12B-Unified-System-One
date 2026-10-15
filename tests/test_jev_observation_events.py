"""Profile failures survive native refreshes; navigation failures keep their meaning."""

import copy
import json

import pytest

from s1.jev_observation_events import ObservationProfileTracker

READ = "trusted READ_STATE"
MARKER = "trusted MARKER"
MANIFEST = "a" * 64


def tracker(**kwargs):
    return ObservationProfileTracker(
        read_state=READ, marker_expression=MARKER, manifest_sha256=MANIFEST, **kwargs
    )


def metadata(**kwargs):
    return {
        "id": "composed-visible-text-v2",
        "visited_nodes": 130,
        "node_limit": 20000,
        "depth_limit": 128,
        "node_limit_reached": False,
        "depth_limit_reached": False,
        **kwargs,
    }


def marker(text="Visible method(x)"):
    return [1234.5, "https://fixture.invalid/", 0, 0, 1120, 780, "Title", text, [], []]


def snapshot():
    return {
        "observation_profile": metadata(),
        "text": "Visible method(x)",
        "marker": marker(),
    }


def event(call, value, *, expression=READ):
    return {
        "protocol_call": call,
        "session": "owned-session",
        "expression": expression,
        "response": {"result": {"value": value}},
    }


def cap_event(call, *, expression=READ, encoded=None):
    if encoded is None:
        encoded = json.dumps(metadata(depth_limit_reached=True))
    return {
        "protocol_call": call,
        "session": "owned-session",
        "expression": expression,
        "response": {
            "exceptionDetails": {
                "text": "Uncaught",
                "exception": {
                    "description": "ComposedVisibleTextLimitError: COMPOSED_VISIBLE_TEXT_LIMIT "
                    + encoded
                    + "\n    at snapshot.js:1:1"
                },
            }
        },
    }


def test_cap_is_latched_across_good_native_refresh_and_marker():
    saved = []
    t = tracker(sink=saved.append)
    first = t.capture(cap_event(12))
    second = t.capture(event(15, snapshot()))
    third = t.capture(event(17, marker(), expression=MARKER))
    assert first["failure"]["kind"] == "cap"
    assert second["failure"] is third["failure"] is None
    assert [row["profile_failure_count"] for row in saved] == [1, 1, 1]
    assert t.failed and len(t.failures) == 1
    assert [row["expression_kind"] for row in saved] == ["READ_STATE", "READ_STATE", "MARKER"]
    assert all(row["manifest_sha256"] == MANIFEST for row in saved)


@pytest.mark.parametrize("expression", [READ, MARKER])
def test_source_defined_navigation_and_null_reads_are_not_caps(expression):
    t = tracker()
    t.capture(event(1, None, expression=expression))
    t.capture(
        {
            "protocol_call": 2,
            "session": "owned-session",
            "expression": expression,
            "error": {"type": "Error", "message": "Execution context was destroyed by navigation"},
        }
    )
    ordinary = cap_event(3, expression=expression)
    ordinary["response"]["exceptionDetails"]["exception"]["description"] = (
        "Error: Document changed\n at native.js:1"
    )
    t.capture(ordinary)
    assert not t.failed and t.failures == [] and len(t.rows) == 3
    assert all(row["failure"] is None for row in t.rows)


@pytest.mark.parametrize("expression", [READ, MARKER])
def test_named_cap_in_either_expression_is_preserved(expression):
    t = tracker()
    original = cap_event(1, expression=expression)
    saved = t.capture(original)
    assert saved["response"] == original["response"]
    assert saved["failure"]["kind"] == "cap"
    assert t.failed


@pytest.mark.parametrize(
    "encoded",
    [
        "not JSON",
        "{}",
        '{"id":"composed-visible-text-v2","id":"forged"}',
        json.dumps(metadata(depth_limit_reached=True, visited_nodes=True)),
        json.dumps(metadata(depth_limit_reached=False)),
        json.dumps(metadata(depth_limit_reached=True, depth_limit=129)),
    ],
)
def test_malformed_or_forged_cap_metadata_is_an_integrity_failure(encoded):
    row = tracker().capture(cap_event(1, encoded=encoded))
    assert row["failure"]["kind"] == "integrity"


@pytest.mark.parametrize(
    "change",
    [
        {"observation_profile": None},
        {"observation_profile": metadata(id="native-1231850")},
        {"observation_profile": metadata(depth_limit_reached=True)},
        {"observation_profile": metadata(visited_nodes=20001)},
        {"observation_profile": metadata(extra="not declared")},
        {"text": "marker no longer matches"},
        {"marker": []},
    ],
)
def test_wrong_profile_or_inconsistent_snapshot_never_becomes_eligible(change):
    t = tracker()
    bad = snapshot()
    bad.update(change)
    t.capture(event(1, bad))
    t.capture(event(2, snapshot()))
    assert t.failed and t.rows[0]["failure"]["kind"] == "integrity"
    assert t.rows[1]["profile_failure_count"] == 1


def test_marker_body_is_validated_without_rewriting_it():
    t = tracker()
    body = marker()
    assert t.capture(event(1, body, expression=MARKER))["response"]["result"]["value"] == body
    invalid = marker()
    invalid[1] = 123
    assert t.capture(event(2, invalid, expression=MARKER))["failure"]["kind"] == "integrity"


@pytest.mark.parametrize("expression", [READ, MARKER])
def test_native_marker_control_identity_list_is_required(expression):
    value = snapshot() if expression == READ else marker()
    (value["marker"] if expression == READ else value)[9] = {}
    row = tracker().capture(event(1, value, expression=expression))
    assert row["failure"]["kind"] == "integrity"


def test_envelopes_must_bind_the_exact_expression_session_and_protocol_order():
    t = tracker()
    t.capture(event(4, snapshot()))
    bads = [event(4, snapshot()), event(3, snapshot()), event(5, snapshot(), expression="other")]
    for key, value in [("session", ""), ("protocol_call", True)]:
        bad = event(5, snapshot())
        bad[key] = value
        bads.append(bad)
    for bad in bads:
        with pytest.raises(ValueError):
            t.capture(bad)
    assert len(t.rows) == 1


def test_sink_and_getters_cannot_mutate_saved_raw_payload_or_failures():
    def mutating_sink(row):
        row["response"].clear()
        row["failure"] = None

    t = tracker(sink=mutating_sink)
    original = cap_event(1)
    before = copy.deepcopy(original)
    returned = t.capture(original)
    original["response"].clear()
    returned["response"].clear()
    rows = t.rows
    rows.clear()
    failures = t.failures
    failures.clear()
    assert t.rows[0]["response"] == before["response"]
    assert t.failed and len(t.failures) == 1


def test_sink_failure_propagates_without_losing_the_detached_internal_row():
    def refusing_sink(_row):
        raise OSError("disk full")

    t = tracker(sink=refusing_sink)
    with pytest.raises(OSError, match="disk full"):
        t.capture(cap_event(1))
    assert len(t.rows) == 1 and t.failed


def test_replaying_saved_raw_events_reproduces_each_classification_exactly():
    t = tracker()
    for item in [event(1, snapshot()), cap_event(3, expression=MARKER), event(5, snapshot())]:
        t.capture(item)
    replay = tracker()
    expressions = {"READ_STATE": READ, "MARKER": MARKER}
    for row in t.rows:
        payload = "response" if "response" in row else "error"
        assert (
            replay.capture(
                {
                    "protocol_call": row["protocol_call"],
                    "session": row["session"],
                    "expression": expressions[row["expression_kind"]],
                    payload: row[payload],
                }
            )
            == row
        )


@pytest.mark.parametrize("selection", ["composed-visible-text-v2", "composed-visible-text-v3"])
@pytest.mark.parametrize("expression", [READ, MARKER])
def test_explicit_profile_cap_is_latched_through_matching_refresh(selection, expression):
    t = tracker(profile=selection)
    original = cap_event(
        1,
        expression=expression,
        encoded=json.dumps(metadata(id=selection, node_limit_reached=True)),
    )
    row = t.capture(original)
    assert row["failure"]["kind"] == "cap"
    good = snapshot()
    good["observation_profile"]["id"] = selection
    assert t.capture(event(2, good))["failure"] is None
    assert t.capture(event(3, marker(), expression=MARKER))["profile_failure_count"] == 1
    assert t.failed and t.rows[0]["response"] == original["response"]


@pytest.mark.parametrize("selection", ["composed-visible-text-v2", "composed-visible-text-v3"])
@pytest.mark.parametrize("payload", ["snapshot", "cap", "transport_cap"])
def test_another_explicit_profile_cannot_supply_snapshot_or_cap_metadata(selection, payload):
    other = (
        "composed-visible-text-v3"
        if selection == "composed-visible-text-v2"
        else "composed-visible-text-v2"
    )
    if payload == "snapshot":
        value = snapshot()
        value["observation_profile"]["id"] = other
        raw = event(1, value)
    else:
        encoded = json.dumps(metadata(id=other, node_limit_reached=True))
        raw = cap_event(1, encoded=encoded)
        if payload == "transport_cap":
            raw["error"] = {
                "type": "Error",
                "message": raw.pop("response")["exceptionDetails"]["exception"]["description"],
            }
    row = tracker(profile=selection).capture(raw)
    assert row["failure"]["kind"] == "integrity"


@pytest.mark.parametrize("selection", ["native-1231850", "composed-visible-text-v4", None, True])
def test_unknown_profile_is_refused_at_tracker_construction(selection):
    with pytest.raises(ValueError, match="Unknown explicit"):
        tracker(profile=selection)
