"""Pure verified-loader and CDP observer tests; no browser, model or network."""

import builtins
import copy
import hashlib
import importlib.util
import json
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1 import jev_observation_profile as profile
from s1.jev_observation_events import ObservationProfileTracker

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "observation_runtime_test_harness", ROOT / "scripts/jev_browser_benchmark.py"
)
H = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = H
SPEC.loader.exec_module(H)
HELPER_SHA = hashlib.sha256(Path(profile.__file__).read_bytes()).hexdigest()


@pytest.fixture
def canonical():
    configured = os.environ.get("JEV_UPSTREAM_PATH")
    if not configured:
        pytest.skip("Set JEV_UPSTREAM_PATH to the existing pinned checkout; never fetches")
    return Path(configured).resolve()


@pytest.fixture
def verified(tmp_path, canonical):
    result = profile.materialize_profile(
        canonical,
        tmp_path / "profile",
        expected_source_sha256=HELPER_SHA,
        profile=profile.COMPOSED_PROFILE,
    )
    return profile.verify_materialized_profile(
        tmp_path / "profile",
        expected_manifest_sha256=result["manifest_sha256"],
        expected_source_sha256=HELPER_SHA,
        profile=profile.COMPOSED_PROFILE,
    )


def test_default_preparation_never_imports_profile_or_checks_extra_sources(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "s1":
            pytest.fail("Default path imported the observation helper")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(H, "upstream_manifest", lambda *_: pytest.fail("Default added Git calls"))
    assert H.prepare_observation_profile("unused", H.NATIVE_OBSERVATION_PROFILE, None) is None


def test_default_loader_keeps_canonical_import_without_profile_checks(canonical, monkeypatch):
    monkeypatch.setattr(
        H, "_verify_observation_runtime", lambda *_: pytest.fail("Default verified a profile")
    )
    monkeypatch.setattr(
        H, "upstream_manifest", lambda *_: pytest.fail("Default loader added Git calls")
    )
    tr = H.PlaywrightCDP(None)
    before = sys.dont_write_bytecode
    with H.load_upstream(canonical, tr) as (agent, model):
        try:
            assert Path(agent.__file__).resolve() == canonical / "jev_ultrafast/agent.py"
            assert (
                sys.modules["jev_ultrafast.browser"].READ_STATE
                == (canonical / profile.SNAPSHOT).read_text()
            )
            assert sys.dont_write_bytecode is before
        finally:
            model.CLIENT.close()
    assert tr.observation_profile_expressions is None and tr.observation_profile_errors == []


@pytest.mark.parametrize(
    "selection,pin",
    [
        ("native", None),
        (H.NATIVE_OBSERVATION_PROFILE, HELPER_SHA),
        (H.COMPOSED_OBSERVATION_PROFILE, None),
        (H.COMPOSED_OBSERVATION_PROFILE, "0" * 64),
    ],
)
def test_bad_profile_or_pin_refuses_before_git_and_runtime(monkeypatch, selection, pin):
    monkeypatch.setattr(
        H, "upstream_manifest", lambda *_: pytest.fail("Bad profile/pin reached Git")
    )
    with pytest.raises(ValueError):
        H.prepare_observation_profile("unused", selection, pin)


@pytest.mark.parametrize("initial", [False, True])
def test_verified_native_import_uses_both_expressions_and_restores_bytecode(
    verified, canonical, initial, monkeypatch
):
    monkeypatch.setattr(sys, "dont_write_bytecode", initial)
    transport = H.PlaywrightCDP(None)
    previous = {"old-expression": "old"}
    transport.observation_profile_expressions = previous
    with H.load_upstream(canonical, transport, observation_runtime=verified) as (agent, model):
        try:
            native = sys.modules["jev_ultrafast.browser"]
            assert Path(agent.__file__) == verified.runtime_root / "jev_ultrafast/agent.py"
            assert (
                native.READ_STATE == verified.read_state
                and native.MARKER == verified.marker_expression
            )
            assert sys.dont_write_bytecode is initial
            assert transport.observation_profile_expressions == {
                verified.read_state: "READ_STATE",
                verified.marker_expression: "MARKER",
            }
        finally:
            model.CLIENT.close()
    assert sys.dont_write_bytecode is initial
    assert transport.observation_profile_expressions is previous
    assert not list(verified.path.rglob("__pycache__"))
    assert transport.observation_profile_errors == []


@pytest.mark.parametrize("initial", [False, True])
def test_import_failure_preserves_primary_restores_globals_and_latches(
    verified, canonical, initial, monkeypatch
):
    monkeypatch.setattr(sys, "dont_write_bytecode", initial)
    transport = H.PlaywrightCDP(None)
    before_path = list(sys.path)
    original = H.importlib.import_module

    def fail(name):
        if name == "jev_ultrafast.agent":
            assert sys.dont_write_bytecode is True
            raise RuntimeError("native import primary")
        return original(name)

    monkeypatch.setattr(H.importlib, "import_module", fail)
    with pytest.raises(RuntimeError, match="native import primary") as caught:
        with H.load_upstream(canonical, transport, observation_runtime=verified):
            pytest.fail("Import failure yielded")
    assert sys.dont_write_bytecode is initial and sys.path == before_path
    assert all(error["kind"] == "integrity" for error in transport.observation_profile_errors)
    assert any(error["phase"] == "native_import" for error in transport.observation_profile_errors)
    assert caught.value.__notes__


def test_copied_tree_corruption_refused_before_import(verified, canonical, monkeypatch):
    (verified.runtime_root / "jev_ultrafast/model.py").write_bytes(
        b"raise Exception('never execute')"
    )
    monkeypatch.setattr(
        H.importlib, "import_module", lambda *_: pytest.fail("Corrupt runtime imported")
    )
    transport = H.PlaywrightCDP(None)
    with pytest.raises(ValueError):
        with H.load_upstream(canonical, transport, observation_runtime=verified):
            pytest.fail("Corrupt runtime yielded")
    assert transport.observation_profile_errors[0]["phase"] == "before_import"


def test_boundary_source_change_cannot_mask_primary(verified, canonical):
    transport = H.PlaywrightCDP(None)
    with pytest.raises(RuntimeError, match="policy primary") as caught:
        with H.load_upstream(canonical, transport, observation_runtime=verified) as (_, model):
            model.CLIENT.close()
            (verified.path / "unexpected").write_text("retained")
            raise RuntimeError("policy primary")
    assert any(error["phase"] == "after_runtime" for error in transport.observation_profile_errors)
    assert any("final verification failed" in note for note in caught.value.__notes__)


class Session:
    def __init__(self, response=None, error=None, callback=None):
        self.response, self.error, self.callback = response, error, callback

    def send(self, method, params):
        if self.callback:
            self.callback()
        if self.error:
            raise self.error
        return self.response


def transport(response=None, error=None):
    result = H.PlaywrightCDP(None)
    result.sessions["s"] = Session(response, error)
    result.observation_expression = "READ"
    result.observation_profile_expressions = {"READ": "READ_STATE", "MARK": "MARKER"}
    return result


def test_profile_and_original_observation_sinks_keep_same_raw_response_and_entry_id():
    response = {"result": {"value": None}}
    tr = transport(response)
    profiles, snapshots = [], []
    tr.observation_profile_sink, tr.observation_sink = profiles.append, snapshots.append
    tr.sessions["s"].callback = lambda: setattr(tr, "calls", tr.calls + 7)
    assert tr.cdp("Runtime.evaluate", "s", expression="READ", returnByValue=True) is response
    assert profiles == [
        {"protocol_call": 1, "session": "s", "expression": "READ", "response": response}
    ]
    assert snapshots == [{"protocol_call": 1, "session": "s", "response": response}]
    profiles[0]["response"]["result"]["value"] = "caller mutation"
    assert snapshots[0]["response"] == response


def test_marker_is_separate_and_unknown_expressions_are_not_profile_events():
    tr = transport({"result": {"value": None}})
    rows, snapshots = [], []
    tr.observation_profile_sink, tr.observation_sink = rows.append, snapshots.append
    tr.cdp("Runtime.evaluate", "s", expression="MARK")
    tr.cdp("Runtime.evaluate", "s", expression="other")
    assert len(rows) == 1 and snapshots == [] and rows[0]["expression"] == "MARK"


def test_send_error_is_retained_without_snapshot_response():
    primary = RuntimeError("Execution context destroyed")
    tr = transport(error=primary)
    rows, snapshots = [], []
    tr.observation_profile_sink, tr.observation_sink = rows.append, snapshots.append
    with pytest.raises(RuntimeError) as caught:
        tr.cdp("Runtime.evaluate", "s", expression="READ")
    assert caught.value is primary and snapshots == []
    assert rows[0]["error"] == {"type": "RuntimeError", "message": str(primary)}


@pytest.mark.parametrize("send_failure", [False, True])
def test_sink_failure_latches_integrity_and_preserves_existing_primary(send_failure):
    primary = RuntimeError("original CDP failure")
    tr = transport({"result": {"value": None}}, primary if send_failure else None)

    def fail(_):
        raise OSError("journal persistence failed")

    tr.observation_profile_sink = fail
    with pytest.raises(RuntimeError if send_failure else OSError) as caught:
        tr.cdp("Runtime.evaluate", "s", expression="READ")
    if send_failure:
        assert caught.value is primary and "persistence" in caught.value.__notes__[0]
    assert tr.observation_profile_errors[0]["phase"] == "event_sink"


def snapshot_response():
    marker = [1.0, "about:blank", 0, 0, 1120, 780, "Owned", "Visible", [], []]
    return {
        "result": {
            "value": {
                "text": "Visible",
                "marker": marker,
                "observation_profile": {
                    "id": profile.COMPOSED_PROFILE,
                    "visited_nodes": 3,
                    "node_limit": 20000,
                    "depth_limit": 128,
                    "node_limit_reached": False,
                    "depth_limit_reached": False,
                },
            }
        }
    }


@pytest.mark.parametrize("failure", ["cap", "navigation", "sink", "boundary"])
def test_pure_benchmark_retains_profile_failure_after_good_refresh(
    tmp_path, canonical, monkeypatch, failure
):
    created = []

    class FakeChrome:
        def close(self):
            pass

    @contextmanager
    def playwright():
        yield SimpleNamespace(chromium=SimpleNamespace(launch=lambda **_: FakeChrome()))

    module = types.ModuleType("playwright.sync_api")
    module.sync_playwright = playwright
    monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", module)

    @contextmanager
    def server(_):
        yield "http://owned.invalid"

    monkeypatch.setattr(H, "fixture_server", server)
    monkeypatch.setattr(
        H,
        "make_cases",
        lambda *_a, **_k: [
            H.FixtureCase("unit", "filter_open", "Owned unit goal", "<p>Owned</p>", {})
        ],
    )
    monkeypatch.setattr(H, "verify_fixture", lambda *_: {"success": True})
    base_transport = H.PlaywrightCDP

    def make_transport(browser):
        tr = base_transport(browser)
        created.append(tr)
        return tr

    monkeypatch.setattr(H, "PlaywrightCDP", make_transport)
    if failure == "sink":

        def append(journal, event):
            raise OSError("profile journal failed")

        monkeypatch.setattr(H._ObservationProfileJournal, "append", append)

    @contextmanager
    def loader(_upstream, tr, *, observation_runtime):
        tr.observation_profile_expressions = {
            observation_runtime.read_state: "READ_STATE",
            observation_runtime.marker_expression: "MARKER",
        }

        class FakeAgent:
            def __init__(self, *_, **__):
                self.state = {"status": "done", "history": []}
                self.browser = SimpleNamespace(evaluate=lambda _: "<p>Owned</p>")
                metadata = {
                    "id": profile.COMPOSED_PROFILE,
                    "visited_nodes": 20000,
                    "node_limit": 20000,
                    "depth_limit": 128,
                    "node_limit_reached": True,
                    "depth_limit_reached": False,
                }
                description = (
                    "ComposedVisibleTextLimitError: COMPOSED_VISIBLE_TEXT_LIMIT "
                    + json.dumps(metadata)
                )
                if failure != "cap":
                    description = "Error: Execution context was destroyed"
                for response in (
                    {"exceptionDetails": {"exception": {"description": description}}},
                    snapshot_response(),
                ):
                    tr.sessions["s"] = Session(response)
                    tr.cdp("Runtime.evaluate", "s", expression=observation_runtime.read_state)

            def snapshot(self):
                return copy.deepcopy(self.state)

            def close(self):
                pass

        yield SimpleNamespace(Agent=FakeAgent), None
        if failure == "boundary":
            error = ValueError("final profile tree corrupted")
            H._profile_integrity_error(tr, "after_runtime", error)
            raise error

    monkeypatch.setattr(H, "load_upstream", loader)

    @contextmanager
    def adapter(*_a, **_k):
        yield [], []

    monkeypatch.setattr(H, "model_adapter", adapter)
    output = tmp_path / "benchmark"
    options = dict(
        upstream_path=canonical,
        endpoint="http://owned.invalid",
        model="fake-unit",
        output_dir=output,
        task_ids=["filter_open"],
        observation_profile=profile.COMPOSED_PROFILE,
        observation_profile_helper_sha256=HELPER_SHA,
    )
    if failure == "boundary":
        with pytest.raises(ValueError, match="final profile tree corrupted"):
            H.run_benchmark(**options)
        summary = json.loads((output / "summary.json").read_text())
    else:
        summary = H.run_benchmark(**options)
    assert summary["attempts"] == 1
    assert summary["observation_profile_failed"] is (failure != "navigation")
    assert summary["successes"] == int(failure == "navigation")
    assert summary["policy_calls"] == summary["text_calls"] == 0
    status = json.loads((output / "observation-profile-status.json").read_text())
    assert status["failed"] is (failure != "navigation")
    outcome = summary["outcomes"][0]
    if failure == "cap":
        rows = [
            json.loads(line)
            for line in (output / "attempts/unit/observation-profile-events.jsonl")
            .read_text()
            .splitlines()
        ]
        assert (
            len(rows) == 2
            and rows[-1]["failure"] is None
            and rows[-1]["profile_failure_count"] == 1
        )
        assert outcome["observation_profile_failures"][0]["kind"] == "cap"
    if failure == "sink":
        assert created[0].observation_profile_errors[0]["phase"] == "event_sink"


def test_shared_tracker_envelope_failure_is_persistently_poisoned_by_transport():
    tr = transport(snapshot_response())
    tracker = ObservationProfileTracker(
        read_state="READ", marker_expression="MARK", manifest_sha256="a" * 64
    )
    tr.observation_profile_sink = tracker.capture
    tr.calls = -1
    with pytest.raises(ValueError, match="identity"):
        tr.cdp("Runtime.evaluate", "s", expression="READ")
    assert tr.observation_profile_errors[0]["kind"] == "integrity"


def test_real_tracker_profile_writer_keeps_ordered_bytes_sequence_and_redaction(tmp_path):
    path = tmp_path / "profile.jsonl"
    journal = H._ObservationProfileJournal(path, ["secret-value"])
    tracker = ObservationProfileTracker(
        read_state="READ", marker_expression="MARK", manifest_sha256="a" * 64, sink=journal.append
    )
    response = snapshot_response()
    value = response["result"]["value"]
    value["text"] = value["marker"][7] = "Visible secret-value"
    tracker.capture(
        {"protocol_call": 1, "session": "s", "expression": "READ", "response": response}
    )
    tracker.capture(
        {
            "protocol_call": 2,
            "session": "s",
            "expression": "MARK",
            "response": {"result": {"value": value["marker"]}},
        }
    )
    expected = b"".join(
        json.dumps(H._redact(row, ["secret-value"]), ensure_ascii=False, allow_nan=False).encode()
        + b"\n"
        for row in tracker.rows
    )
    assert path.read_bytes() == expected and b"secret-value" not in expected
    rows = [json.loads(line) for line in expected.splitlines()]
    assert [list(row)[:2] for row in rows] == [["schema_id", "sequence"]] * 2
    assert [row["sequence"] for row in rows] == [1, 2] and journal.count == 2
    assert not tracker.failed


@pytest.mark.parametrize("mutation", ["reordered", "sequence"])
def test_profile_writer_refuses_reordered_or_wrong_sequence_without_appending(tmp_path, mutation):
    path = tmp_path / "profile.jsonl"
    journal = H._ObservationProfileJournal(path, [])
    tracker = ObservationProfileTracker(
        read_state="READ", marker_expression="MARK", manifest_sha256="a" * 64, sink=journal.append
    )
    row = tracker.capture(
        {"protocol_call": 1, "session": "s", "expression": "READ", "response": snapshot_response()}
    )
    before = path.read_bytes()
    row["sequence"] = 2
    if mutation == "reordered":
        reordered = {
            "sequence": row["sequence"],
            **{key: value for key, value in row.items() if key != "sequence"},
        }
        assert reordered == row and list(reordered) != list(row)
        row = reordered
    else:
        row["sequence"] = 0
    with pytest.raises(ValueError, match="row order or sequence"):
        journal.append(row)
    assert path.read_bytes() == before and journal.count == 1


def test_native_journal_still_owns_zero_based_sequence_and_original_format(tmp_path):
    path = tmp_path / "native.jsonl"
    journal = H._ObservationJournal(path, [])
    event = {"protocol_call": 3, "session": "s", "response": {"result": {"value": None}}}
    journal.append(event)
    expected = {"sequence": 0, **event}
    assert (
        path.read_bytes()
        == (json.dumps(expected, ensure_ascii=False, allow_nan=False) + "\n").encode()
    )
    assert list(json.loads(path.read_text()))[:2] == ["sequence", "protocol_call"]
