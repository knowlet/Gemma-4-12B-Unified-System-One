"""Transport and independent-judge checks; scripted policies are test-only diagnostics."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "jev_browser_benchmark", ROOT / "scripts/jev_browser_benchmark.py"
)
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


@pytest.mark.parametrize(
    "response",
    [
        {"result": {"value": {"url": "https://example.org", "actions": [{"id": 1}]}}},
        {"result": {"value": None}},
        {"exceptionDetails": {"text": "navigating"}},
    ],
)
def test_native_observation_capture_preserves_protocol_result_and_detaches(response):
    sent, captured = [], []

    def send(method, params):
        sent.append((method, params))
        return response

    transport = HARNESS.PlaywrightCDP(None)
    transport.sessions["owned"] = SimpleNamespace(send=send)
    transport.observation_expression = "pinned native snapshot"
    transport.observation_sink = captured.append
    params = {"expression": "pinned native snapshot", "returnByValue": True}
    actual = transport.cdp("Runtime.evaluate", session_id="owned", **params)
    assert actual is response
    assert sent == [("Runtime.evaluate", params)]
    assert captured == [{"protocol_call": 1, "session": "owned", "response": response}]
    response["upstream_added"] = "must not enter the earlier capture"
    assert "upstream_added" not in captured[0]["response"]
    transport.cdp("Runtime.evaluate", session_id="owned", expression="marker only")
    assert len(captured) == 1


def test_native_observation_journal_is_durable_redacted_and_refuses_overwrite(tmp_path):
    path = tmp_path / "observations.jsonl"
    journal = HARNESS._ObservationJournal(path, ["private-token"])
    journal.append({"response": {"url": "https://example.org/private-token"}})
    # Already readable before final-state or summary files exist.
    assert json.loads(path.read_text()) == {
        "sequence": 0,
        "response": {"url": "https://example.org/[REDACTED]"},
    }
    journal.append({"response": {"exceptionDetails": {"text": "navigating"}}})
    assert [json.loads(line)["sequence"] for line in path.read_text().splitlines()] == [0, 1]
    assert journal.count == 2 and journal.write_ms >= 0
    with pytest.raises(FileExistsError):
        HARNESS._ObservationJournal(path, [])


def test_native_observation_write_failure_propagates_without_a_second_browser_read():
    sent = []
    transport = HARNESS.PlaywrightCDP(None)
    transport.sessions["owned"] = SimpleNamespace(
        send=lambda method, params: sent.append((method, params)) or {"result": {"value": None}}
    )
    transport.observation_expression = "snapshot"

    def fail(_event):
        raise OSError("observation disk full")

    transport.observation_sink = fail
    with pytest.raises(OSError, match="observation disk full"):
        transport.cdp("Runtime.evaluate", session_id="owned", expression="snapshot")
    assert sent == [("Runtime.evaluate", {"expression": "snapshot"})]


@pytest.mark.parametrize(
    "method,url,navigation,parent,blocked",
    [
        ("GET", "https://arxiv.org/search/?query=attention", True, None, False),
        ("HEAD", "https://arxiv.org/abs/1706.03762", True, None, False),
        ("GET", "https://cdn.example.org/site.css", False, None, False),
        ("GET", "https://other.example.org/frame", True, object(), False),
        ("POST", "https://arxiv.org/feedback", False, None, True),
        ("PUT", "https://arxiv.org/feedback", True, None, True),
        ("GET", "https://other.example.org/article", True, None, True),
        ("GET", "http://arxiv.org/search/", True, None, True),
        ("GET", "https://user:password@arxiv.org/search/", True, None, True),
        ("GET", "https://arxiv.org:444/search/", True, None, True),
    ],
)
def test_public_route_retains_assets_and_bounds_read_only_navigation(
    method, url, navigation, parent, blocked
):
    decisions = []
    transport = HARNESS.PlaywrightCDP(None)
    route = SimpleNamespace(
        request=SimpleNamespace(
            method=method,
            url=url,
            is_navigation_request=lambda: navigation,
            frame=SimpleNamespace(parent_frame=parent),
        ),
        abort=lambda reason: decisions.append(("abort", reason)),
        continue_=lambda: decisions.append(("continue", None)),
    )
    transport._public_route(route, {"arxiv.org"})
    assert decisions == [("abort", "blockedbyclient") if blocked else ("continue", None)]
    assert bool(transport.public_request_blocks) is blocked
    if blocked:
        assert transport.public_request_blocks[0]["url"] == url


def test_public_context_blocks_service_worker_route_bypass_without_changing_default():
    created, routes = [], []

    def new_page(**options):
        created.append(options)
        return SimpleNamespace(
            context=SimpleNamespace(
                route=lambda pattern, callback: routes.append((pattern, callback)),
                on=lambda event, callback: None,
            ),
            goto=lambda _url: None,
        )

    transport = HARNESS.PlaywrightCDP(SimpleNamespace(new_page=new_page))
    transport.cdp("Target.createTarget", url="about:blank")
    assert created == [{}] and routes == []
    transport.public_navigation_hosts = ("arxiv.org",)
    transport.cdp("Target.createTarget", url="about:blank")
    assert created == [{}, {"service_workers": "block"}]
    assert len(routes) == 1 and routes[0][0] == "**/*"


def test_public_redirect_audit_retains_a_violating_hop_after_return_to_allowed_host():
    transport = HARNESS.PlaywrightCDP(None)
    previous = None
    for url in (
        "https://arxiv.org/search/",
        "https://outside.example.org/redirect",
        "https://arxiv.org/abs/1706.03762",
    ):
        request = SimpleNamespace(
            url=url,
            method="GET",
            frame=SimpleNamespace(parent_frame=None),
            is_navigation_request=lambda: True,
            redirected_from=previous,
        )
        transport._public_request_observed(request, {"arxiv.org"})
        previous = request
    assert len(transport.public_navigation_events) == 3
    assert transport.public_navigation_events[-1]["scope_violation"] is None
    assert len(transport.public_navigation_violations) == 1
    assert transport.public_navigation_violations[0]["url"].startswith("https://outside.")
    assert transport.public_request_blocks == []  # Audit is not contact prevention.


def test_public_early_navigation_without_frame_is_aborted_and_audited():
    class EarlyRequest:
        method = "GET"
        url = "https://arxiv.org/search/"
        redirected_from = None

        def is_navigation_request(self):
            return True

        @property
        def frame(self):
            raise RuntimeError("Frame is not yet created")

    transport = HARNESS.PlaywrightCDP(None)
    decisions = []
    request = EarlyRequest()
    transport._public_route(
        SimpleNamespace(
            request=request,
            abort=lambda reason: decisions.append(reason),
            continue_=lambda: pytest.fail("Unknown navigation frame must not be continued"),
        ),
        {"arxiv.org"},
    )
    transport._public_request_observed(request, {"arxiv.org"})
    assert decisions == ["blockedbyclient"]
    assert "metadata unavailable" in transport.public_request_blocks[0]["reason"]
    assert len(transport.public_navigation_violations) == 1
    assert transport.public_observer_errors == []


def test_public_observer_failure_is_explicit():
    transport = HARNESS.PlaywrightCDP(None)
    transport._public_request_observed(object(), {"arxiv.org"})
    assert transport.public_observer_errors[0]["type"] == "AttributeError"


def _document_transport():
    transport = HARNESS.PlaywrightCDP(None)
    transport.public_root_frames["owned"] = "root"
    return transport


def _document_request(url="https://arxiv.org/abs/1706.03762", **changes):
    return {
        "type": "Document",
        "frameId": "root",
        "requestId": "request-1",
        "loaderId": "loader-1",
        "request": {"url": url, "method": "GET"},
        **changes,
    }


def _document_commit(url="https://arxiv.org/abs/1706.03762", **changes):
    return {"frame": {"id": "root", "url": url, "loaderId": "loader-1", **changes}}


@pytest.mark.parametrize("public", [False, True])
def test_document_observer_setup_precedes_native_navigation_and_default_is_unchanged(public):
    events, callbacks = [], {}

    def send(method, params=None):
        events.append(("send", method, params))
        if method == "Page.getFrameTree":
            return {"frameTree": {"frame": {"id": "root"}}}
        if method in {"Network.enable", "Page.enable"}:
            assert transport.public_root_frames["session-1"] == "root"
        return {}

    def on(event, callback):
        events.append(("listen", event))
        callbacks[event] = callback

    session = SimpleNamespace(send=send, on=on)
    page = SimpleNamespace(context=SimpleNamespace(new_cdp_session=lambda _page: session))
    transport = HARNESS.PlaywrightCDP(None)
    transport.targets["target"] = page
    transport.public_navigation_hosts = ("arxiv.org",) if public else None
    attached = transport.cdp("Target.attachToTarget", targetId="target")
    transport.cdp("Page.navigate", session_id=attached["sessionId"], url="https://arxiv.org/")
    if public:
        assert [entry[:2] for entry in events] == [
            ("listen", "Network.requestWillBeSent"),
            ("listen", "Page.frameNavigated"),
            ("listen", "Page.navigatedWithinDocument"),
            ("send", "Page.getFrameTree"),
            ("send", "Network.enable"),
            ("send", "Page.enable"),
            ("send", "Page.navigate"),
        ]
        assert transport.calls == 5
        callbacks["Network.requestWillBeSent"](_document_request())
        callbacks["Page.frameNavigated"](_document_commit())
        callbacks["Page.navigatedWithinDocument"](
            {"frameId": "root", "url": "https://arxiv.org/abs/1706.03762#abstract"}
        )
        assert len(transport.public_document_requests) == 1
        assert len(transport.public_document_commits) == 2
    else:
        assert events == [("send", "Page.navigate", {"url": "https://arxiv.org/"})]
        assert transport.calls == 2
        assert transport.public_root_frames == {}


@pytest.mark.parametrize("method", ["Page.getFrameTree", "Network.enable", "Page.enable"])
def test_document_observer_setup_failure_propagates_before_navigation(method):
    sent = []

    def send(actual, _params=None):
        sent.append(actual)
        if actual == method:
            raise RuntimeError("observer setup failed")
        return {"frameTree": {"frame": {"id": "root"}}}

    transport = HARNESS.PlaywrightCDP(None)
    transport.public_navigation_hosts = ("arxiv.org",)
    session = SimpleNamespace(send=send, on=lambda *_: None)
    transport.targets["target"] = SimpleNamespace(
        context=SimpleNamespace(new_cdp_session=lambda _page: session)
    )
    with pytest.raises(RuntimeError, match="observer setup failed"):
        transport.cdp("Target.attachToTarget", targetId="target")
    assert "Page.navigate" not in sent


@pytest.mark.parametrize(
    "event",
    [
        _document_request(frameId="iframe"),
        _document_request(type="Prefetch"),
        _document_request(type="Image"),
    ],
)
def test_document_request_audit_excludes_other_resources_and_frames(event):
    transport = _document_transport()
    transport._public_document_request("owned", event, {"arxiv.org"})
    assert transport.public_document_requests == []
    assert transport.public_navigation_violations == []
    assert transport.public_observer_errors == []
    assert transport.public_document_started == set()


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_document_request_retains_native_ids_and_allowed_redirect(method):
    transport = _document_transport()
    event = _document_request(
        request={"url": "https://arxiv.org/abs/1706.03762", "method": method},
        redirectResponse={"url": "https://arxiv.org/search/", "status": 302},
    )
    transport._public_document_request("owned", event, {"arxiv.org"})
    assert transport.public_document_requests == [
        {
            "source": "cdp_document_request",
            "session": "owned",
            "request_id": "request-1",
            "loader_id": "loader-1",
            "frame_id": "root",
            "url": "https://arxiv.org/abs/1706.03762",
            "method": method,
            "redirected_from": "https://arxiv.org/search/",
            "redirect_status": 302,
            "scope_violation": None,
        }
    ]
    assert transport.public_document_started == {"owned"}
    assert transport.public_navigation_violations == []


@pytest.mark.parametrize(
    "url",
    [
        "https://outside.example.org/page",
        "http://arxiv.org/abs/1706.03762",
        "https://user:secret@arxiv.org/page",
        "https://arxiv.org:444/page",
        "https://arxiv.org:broken/page",
        "data:text/html,example",
        None,
    ],
)
def test_document_url_scope_rejects_invalid_or_undeclared_targets(url):
    assert HARNESS.PlaywrightCDP._public_document_url_scope(url, {"arxiv.org"})


def test_document_request_audits_method_target_and_redirect_source_together():
    transport = _document_transport()
    transport._public_document_request(
        "owned",
        _document_request(
            request={"url": "https://outside.example.org/page", "method": "POST"},
            redirectResponse={"url": "http://arxiv.org/search/", "status": 307},
        ),
        {"arxiv.org"},
    )
    record = transport.public_document_requests[0]
    assert record["method"] == "POST" and record["redirect_status"] == 307
    assert "outside declared HTTPS hosts" in record["scope_violation"]
    assert "only GET and HEAD" in record["scope_violation"]
    assert "redirected from outside" in record["scope_violation"]
    assert transport.public_navigation_violations == [record]
    # A later allowed URL cannot erase a violating redirect hop.
    transport._public_document_request("owned", _document_request(), {"arxiv.org"})
    assert len(transport.public_navigation_violations) == 1


def test_document_commit_tracks_root_replacement_fragment_and_iframe_separately():
    transport = _document_transport()
    transport._public_document_commit(
        "owned",
        _document_commit("https://outside.example.org/frame", parentId="root"),
        {"arxiv.org"},
    )
    transport._public_document_commit("owned", _document_commit(id="replacement"), {"arxiv.org"})
    transport._public_document_commit(
        "owned",
        {"frameId": "old-root", "url": "https://outside.example.org/"},
        {"arxiv.org"},
        within_document=True,
    )
    transport._public_document_commit(
        "owned",
        {"frameId": "replacement", "url": "https://arxiv.org/abs/1706.03762#abstract"},
        {"arxiv.org"},
        within_document=True,
    )
    assert transport.public_root_frames["owned"] == "replacement"
    assert [row["source"] for row in transport.public_document_commits] == [
        "cdp_frame_commit",
        "cdp_within_document",
    ]
    assert transport.public_document_commits[-1]["loader_id"] is None
    assert transport.public_navigation_violations == transport.public_observer_errors == []


def test_document_commit_exempts_only_pre_navigation_setup_blank():
    transport = _document_transport()
    transport._public_document_commit("owned", _document_commit("about:blank"), {"arxiv.org"})
    assert transport.public_document_commits[-1]["setup_blank"] is True
    assert transport.public_document_started == set()
    transport._public_document_request("owned", _document_request(), {"arxiv.org"})
    transport._public_document_commit("owned", _document_commit("about:blank"), {"arxiv.org"})
    assert transport.public_document_commits[-1]["setup_blank"] is False
    assert len(transport.public_navigation_violations) == 1


@pytest.mark.parametrize("within_document", [False, True])
def test_document_commit_catches_disallowed_activation_without_a_request(within_document):
    transport = _document_transport()
    url = "https://outside.example.org/prefetched"
    event = {"frameId": "root", "url": url} if within_document else _document_commit(url)
    transport._public_document_commit(
        "owned", event, {"arxiv.org"}, within_document=within_document
    )
    assert transport.public_document_requests == transport.public_navigation_events == []
    assert transport.public_document_commits[0]["url"] == url
    assert transport.public_navigation_violations == transport.public_document_commits


@pytest.mark.parametrize("observer", ["request", "commit", "within_document"])
def test_document_observer_malformed_events_are_explicit_errors(observer):
    transport = _document_transport()
    if observer == "request":
        transport._public_document_request("owned", {"type": "Document"}, {"arxiv.org"})
    else:
        transport._public_document_commit(
            "owned", {}, {"arxiv.org"}, within_document=observer == "within_document"
        )
    assert transport.public_document_requests == transport.public_document_commits == []
    assert transport.public_observer_errors[0]["type"] == "KeyError"


def test_public_request_ledger_survives_final_snapshot_and_close_failures(tmp_path):
    transport = HARNESS.PlaywrightCDP(None)
    transport.public_request_blocks.append({"url": "https://arxiv.org/feedback", "method": "POST"})
    primary = RuntimeError("final agent snapshot failed")

    def close():
        transport.public_navigation_events.append({"url": "https://arxiv.org/last-event"})
        transport.public_document_requests.append({"url": "https://arxiv.org/raw-request"})
        transport.public_document_commits.append({"url": "https://arxiv.org/raw-commit"})
        raise OSError("secondary browser close failed")

    HARNESS._finish_public_browser(
        SimpleNamespace(close=close), transport, tmp_path, [], primary_error=primary
    )
    ledger = json.loads((tmp_path / "public-request-ledger.json").read_bytes())
    assert ledger["blocks"] == transport.public_request_blocks
    assert ledger["navigation_events"][-1]["url"].endswith("last-event")
    assert ledger["document_requests"][-1]["url"].endswith("raw-request")
    assert ledger["document_commits"][-1]["url"].endswith("raw-commit")
    assert any("secondary browser close failed" in note for note in primary.__notes__)


def test_public_request_ledger_write_failure_preserves_primary_and_still_closes(
    tmp_path, monkeypatch
):
    primary, closed = RuntimeError("primary snapshot failure"), []

    def fail(*_args):
        raise OSError("request ledger write failure")

    monkeypatch.setattr(HARNESS, "_write_json", fail)
    HARNESS._finish_public_browser(
        SimpleNamespace(close=lambda: closed.append(True)),
        HARNESS.PlaywrightCDP(None),
        tmp_path,
        [],
        primary_error=primary,
    )
    assert closed == [True]
    assert any("request ledger write failure" in note for note in primary.__notes__)


@pytest.mark.parametrize(
    "settings",
    [
        {"suite_split": "holdout"},
        {"travel_challenges_only": True},
        {"travel_transfer_only": True},
        {"task_ids": []},
        {"task_ids": ["native_select"]},
        {"live_goal": "Open Spoon"},
        {"live_url": "https://en.wikipedia.org/wiki/Spork"},
    ],
)
def test_public_cases_reject_mixed_scope_before_browser_output(tmp_path, monkeypatch, settings):
    monkeypatch.setattr(HARNESS, "upstream_manifest", lambda _path: {})
    with pytest.raises(ValueError, match="Public cases"):
        HARNESS.run_benchmark(
            upstream_path="unused",
            endpoint="http://localhost/policy",
            model="diagnostic",
            output_dir=tmp_path / "run",
            public_case_ids=["public-arxiv-title-filter-results-v1"],
            **settings,
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("case_ids", [[], ["x", "x"], [None], "not-a-list"])
def test_public_cases_need_explicit_distinct_ids(tmp_path, monkeypatch, case_ids):
    monkeypatch.setattr(HARNESS, "upstream_manifest", lambda _path: {})
    with pytest.raises(ValueError, match="distinct explicit"):
        HARNESS.run_benchmark(
            upstream_path="unused",
            endpoint="http://localhost/policy",
            model="diagnostic",
            output_dir=tmp_path / "run",
            public_case_ids=case_ids,
        )
    assert not (tmp_path / "run").exists()


def test_variants_are_frozen_and_holdout_does_not_reuse_development():
    development = HARNESS.make_cases(seed=72)
    assert development == HARNESS.make_cases(seed=72)
    holdout = HARNESS.make_cases(seed=72, suite_split="holdout")
    assert {case.task_id for case in development} == set(HARNESS.SEEDED_TASK_IDS)
    assert {case.task_id for case in holdout} == set(HARNESS.SEEDED_TASK_IDS)
    assert set(HARNESS.TASK_IDS) - set(HARNESS.SEEDED_TASK_IDS) == {"upstream_travel"}
    assert {case.case_id for case in development}.isdisjoint(case.case_id for case in holdout)
    assert [case.expected for case in development] != [case.expected for case in holdout]
    assert [case.case_id for case in development] != [
        case.case_id for case in HARNESS.make_cases(72, 1)
    ]
    assert {case.task_id for case in development if case.needs_text} == {
        "search_filter_open",
        "autocomplete",
        "unapplied_search",
    }


ORIGINAL_FIXTURE_HASHES = {
    "filter_open": "feb9f8815c8665ad0a3d51ca8e358e2ea0e088f975ac8421f2b66334322f15c4",
    "search_filter_open": "66d2da8f3eee2eb1abcbb62f94fe2e5139f2879fb48879f8adde3cd55d054c9e",
    "already_checked": "c3e5a6534220246eb41240de68d4c8cb033f6b97470cb9a53a8059d71af0ad83",
    "native_select": "6f32e08608f78a13153b4979a6a54366c55a4767d9add786686005c33197e3f1",
    "autocomplete": "15e01385f9a4ece93f6dcc39c26eead545fec4afe168b2c09658343859ccc5b3",
    "submit_populated": "db05139fea206b132475d087e4846c5383b3ba271eb3b113bc141bdad222daf8",
    "premature_done": "d8c6e9fcada2bef9edc01070045c47b2cdc6931c32253bf79dfc74f543d3c4f1",
    "completion_toggle": "a19f426808f94bc5ea2ac8b5d33648ffce9f9353c71c80d8ba3d9e28ce880b8e",
    "upstream_travel": "e48e7d7b6225b215060692e0a28c3d0b59e465e4b94c1bc510d5f37ff0b8bfbe",
}


def test_original_seeded_fixture_bytes_stay_identical_to_recorded_comparison():
    cases = HARNESS.make_cases(seed=20261014)
    for case in cases:
        if case.task_id in ORIGINAL_FIXTURE_HASHES:
            assert (
                hashlib.sha256(case.html.encode()).hexdigest()
                == ORIGINAL_FIXTURE_HASHES[case.task_id]
            )


def test_unapplied_search_has_distinct_frozen_development_and_holdout_variants():
    development = HARNESS.make_unapplied_search_case(seed=72)
    assert development == HARNESS.make_unapplied_search_case(seed=72)
    holdout = HARNESS.make_unapplied_search_case(seed=72, suite_split="holdout")
    assert development.case_id != holdout.case_id
    assert development.html != holdout.html
    assert development.goal != holdout.goal
    assert development.needs_text and holdout.needs_text
    assert development.expected != holdout.expected


def test_unapplied_search_judge_checks_committed_state_independently_of_opened_details():
    case = HARNESS.make_unapplied_search_case(seed=72)
    browser = SimpleNamespace(
        evaluate=lambda _: {
            "result": {**case.expected, "opened_at_commit": 0},
            "result_visible": True,
            "typed_query": case.expected["query"],
            "applied": {
                "query": "",
                "category": "",
                "verified": False,
                "search_committed": False,
                "commit": 0,
            },
        }
    )
    verification = HARNESS.verify_fixture(browser, case)
    assert verification["success"] is False
    assert verification["mismatches"]["applied_search_committed"]["observed"] is False
    assert "opened_at_current_commit" in verification["mismatches"]


def test_unapplied_search_judge_respects_application_query_case_and_whitespace():
    case = HARNESS.make_unapplied_search_case(seed=72)
    query = "  " + case.expected["query"].swapcase() + "  "
    browser = SimpleNamespace(
        evaluate=lambda _: {
            "result": {**case.expected, "query": query, "opened_at_commit": 1},
            "result_visible": True,
            "typed_query": query,
            "applied": {**case.expected, "query": query, "commit": 1},
        }
    )
    assert HARNESS.verify_fixture(browser, case)["success"] is True


@pytest.mark.parametrize(
    ("query", "searched", "success"),
    [("  qUiTo  ", True, True), ("Quito", False, False), ("Porto", True, False)],
)
def test_travel_query_equivalence_still_requires_correct_city_and_submission(
    query, searched, success
):
    case = HARNESS.FixtureCase(
        case_id="travel-query-diagnostic",
        task_id="upstream_travel",
        goal="Search Quito and open Cedar Court",
        html="",
        expected={"name": "Cedar Court", "query": "Quito", "searched": True},
    )
    browser = SimpleNamespace(
        evaluate=lambda _: {
            "result": {"name": "Cedar Court", "query": query, "searched": searched},
            "result_visible": True,
        }
    )
    assert HARNESS.verify_fixture(browser, case)["success"] is success


def test_exposed_travel_fixture_is_not_loaded_into_holdout(tmp_path, monkeypatch):
    # Even a supplied upstream path must not inject the exposed source into holdout.
    holdout = HARNESS.make_cases(seed=72, suite_split="holdout", upstream_path=tmp_path)
    assert {case.task_id for case in holdout} == set(HARNESS.SEEDED_TASK_IDS)
    with pytest.raises(ValueError, match="pinned source"):
        source = tmp_path / HARNESS.UPSTREAM_TRAVEL_SOURCE
        source.parent.mkdir(parents=True)
        source.write_text("<html>modified fixture</html>")
        HARNESS.make_upstream_travel_case(tmp_path)
    monkeypatch.setattr(HARNESS, "upstream_manifest", lambda path: {})
    with pytest.raises(ValueError, match="development only"):
        HARNESS.run_benchmark(
            upstream_path=tmp_path,
            endpoint="http://localhost/model",
            model="test",
            output_dir=tmp_path / "run",
            task_ids=["upstream_travel"],
            suite_split="holdout",
        )


def test_travel_goal_challenges_change_only_requirements_and_have_neutral_ids(monkeypatch):
    base = HARNESS.FixtureCase(
        case_id="original",
        task_id="upstream_travel",
        goal="original goal",
        html="unchanged source\r\n",
        expected={"query": "Lisbon"},
        needs_text=True,
        query_string="scenario=travel",
        source={"sha256": "pinned", "scope": "exposed"},
    )
    monkeypatch.setattr(HARNESS, "make_upstream_travel_case", lambda _path, _repeat: base)
    cases = HARNESS.make_travel_goal_challenges("unused")
    assert [case.case_id for case in cases] == [
        "travel-goal-000-0",
        "travel-goal-000-1",
        "travel-goal-000-2",
    ]
    assert all(case.task_id == "upstream_travel" and case.html == base.html for case in cases)
    assert all(case.query_string == "scenario=travel" for case in cases)
    assert all(
        case.source == {**base.source, "scope": HARNESS.TRAVEL_CHALLENGE_SCOPE} for case in cases
    )
    assert [case.expected for case in cases] == [
        {
            "hash": "#the-glasshouse",
            "name": "The Glasshouse",
            "category": "Design",
            "free": True,
            "query": "Copenhagen",
            "searched": True,
        },
        {
            "hash": "#serra-lodge",
            "name": "Serra Lodge",
            "category": "Nature",
            "free": False,
            "query": "Lisbon",
            "searched": True,
        },
        {"hash": "#casa-flora", "name": "Casa Flora"},
    ]
    assert [case.needs_text for case in cases] == [True, True, False]
    assert [case.case_id for case in HARNESS.make_travel_goal_challenges("unused", 1)] == [
        "travel-goal-001-0",
        "travel-goal-001-1",
        "travel-goal-001-2",
    ]


@pytest.mark.parametrize(
    "settings",
    [
        {"suite_split": "holdout"},
        {"task_ids": ["upstream_travel"]},
        {"task_ids": []},
        {"live_url": "https://en.wikipedia.org/wiki/Spoon"},
        {"live_goal": "Open Spoon"},
        {"live_expected_title": "Spoon - Wikipedia"},
    ],
)
def test_travel_goal_challenges_refuse_mixed_or_holdout_configuration(
    tmp_path, monkeypatch, settings
):
    monkeypatch.setattr(HARNESS, "upstream_manifest", lambda _path: {})
    with pytest.raises(ValueError, match="Travel goal challenges"):
        HARNESS.run_benchmark(
            upstream_path="unused",
            endpoint="http://localhost/policy",
            model="diagnostic",
            output_dir=tmp_path / "run",
            travel_challenges_only=True,
            **settings,
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--suite-split", "holdout"],
        ["--tasks", "upstream_travel"],
        ["--live-url", "https://en.wikipedia.org/wiki/Spoon"],
        ["--live-goal", "Open Spoon"],
        ["--live-expected-title", "Spoon - Wikipedia"],
    ],
)
def test_benchmark_cli_refuses_challenges_with_fixture_or_live_options(monkeypatch, extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--upstream-path",
            "unused",
            "--endpoint",
            "http://localhost/policy",
            "--model",
            "diagnostic",
            "--output-dir",
            "unused",
            "--travel-challenges-only",
            *extra,
        ],
    )
    with pytest.raises(SystemExit) as caught:
        HARNESS.main()
    assert caught.value.code == 2


def test_benchmark_cli_forwards_optional_challenge_selector(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--upstream-path",
            "unused",
            "--endpoint",
            "http://localhost/policy",
            "--model",
            "diagnostic",
            "--output-dir",
            "unused",
            "--travel-challenges-only",
        ],
    )
    monkeypatch.setattr(
        HARNESS, "run_benchmark", lambda **kwargs: calls.append(kwargs) or {"diagnostic": True}
    )
    HARNESS.main()
    assert calls[0]["travel_challenges_only"] is True and calls[0]["task_ids"] is None


def test_simple_open_travel_goal_does_not_invent_search_or_filter_requirements(monkeypatch):
    base = HARNESS.FixtureCase(
        case_id="original",
        task_id="upstream_travel",
        goal="original",
        html="source",
        expected={},
        source={"scope": "original"},
    )
    monkeypatch.setattr(HARNESS, "make_upstream_travel_case", lambda _path, _repeat: base)
    simple = HARNESS.make_travel_goal_challenges("unused")[2]
    browser = SimpleNamespace(
        evaluate=lambda _code: {
            "result": {
                "hash": "#casa-flora",
                "name": "Casa Flora",
                "category": "All",
                "free": False,
                "query": "",
                "searched": False,
            },
            "result_visible": True,
        }
    )
    assert HARNESS.verify_fixture(browser, simple)["success"] is True


def test_transfer_catalogue_is_seeded_explicitly_derived_and_preserves_event_handlers(monkeypatch):
    initial = 'let query = "",\n        category = "all",\n        free = false,\n        searched = false;'
    page = initial + '\nconst places = [{name:"original"}];\nfunction travel(){nativeHandlers();}'
    base = HARNESS.FixtureCase(
        case_id="original",
        task_id="upstream_travel",
        goal="original",
        html=page,
        expected={},
        query_string="scenario=travel",
        source={"repository": "https://github.com/browser-use/jev-ultrafast"},
    )
    monkeypatch.setattr(HARNESS, "make_upstream_travel_case", lambda *_args: base)
    first = HARNESS.make_travel_transfer_cases("unused", seed=83)
    repeated = HARNESS.make_travel_transfer_cases("unused", seed=83)
    other = HARNESS.make_travel_transfer_cases("unused", seed=84)
    assert first == repeated and first != other
    assert len(first) == 5 and len({case.case_id for case in first}) == 5
    assert first[0].html == first[1].html == first[2].html == first[3].html
    assert first[4].html != first[0].html
    for case in first:
        assert case.source["parent_sha256"] == HARNESS.UPSTREAM_TRAVEL_SHA256
        assert case.source["sha256"] == hashlib.sha256(case.html.encode()).hexdigest()
        assert case.source["scope"] == HARNESS.TRAVEL_TRANSFER_SCOPE
        assert case.html.endswith("function travel(){nativeHandlers();}")
        assert "Casa Flora" not in case.goal and "Lisbon" not in case.goal
        assert "expected" not in case.html and "fixture_token" not in case.html
    assert first[3].expected.keys() == {"hash", "name"}
    assert first[0].expected["free"] is True
    assert first[1].expected["free"] is first[4].expected["free"] is False
    assert first[4].expected["searched"] is True
    assert [c.needs_text for c in first] == [True, False, True, False, False]


@pytest.mark.parametrize(
    "settings",
    [
        {"suite_split": "holdout"},
        {"task_ids": []},
        {"travel_challenges_only": True},
        {"live_goal": "Open a page"},
    ],
)
def test_transfer_scope_rejects_mixed_runs_before_creating_output(tmp_path, monkeypatch, settings):
    monkeypatch.setattr(HARNESS, "upstream_manifest", lambda _path: {})
    with pytest.raises(ValueError, match="Travel transfer"):
        HARNESS.run_benchmark(
            upstream_path="unused",
            endpoint="http://localhost/policy",
            model="diagnostic",
            output_dir=tmp_path / "run",
            travel_transfer_only=True,
            **settings,
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("index", range(5))
@pytest.mark.integration
def test_seeded_catalogue_transfer_is_reachable_without_changing_native_handlers(
    real_browser_environment, index
):
    """Scripted reachability is a judge diagnostic, never a model score."""
    upstream, chromium = real_browser_environment
    original = HARNESS.upstream_manifest(upstream)
    cases = HARNESS.make_travel_transfer_cases(upstream, seed=83)
    case = cases[index]
    transport = HARNESS.PlaywrightCDP(chromium)
    with HARNESS.fixture_server(cases) as base:
        with HARNESS.load_upstream(upstream, transport) as (agent_module, _model_module):
            agent = agent_module.Agent(base + case.url_suffix, case.goal, screenshots=False)

            def act(label, *, kind="click", text=None):
                observed = agent.browser.observe(screenshot=False)
                action = next(
                    a for a in observed["actions"] if a["label"] == label and a["kind"] == kind
                )
                agent.browser.act(action, observed, text=text)

            try:
                assert not HARNESS.verify_fixture(agent.browser, case)["success"]
                if index == 4:
                    actions = agent.state["page"]["actions"]
                    assert not any(
                        a["label"] == "Stay category → " + case.expected["category"]
                        for a in actions
                    )
                    before = HARNESS.verify_fixture(agent.browser, case)["observed"]
                    assert before["typed_destination"] == case.expected["query"]
                    assert before["result"]["category"] == case.expected["category"]
                    assert before["result"]["searched"] is False
                else:
                    if "query" in case.expected:
                        text = case.expected["query"]
                        if index == 0:
                            text = "  " + text.swapcase() + "  "
                        act("Destination", kind="fill", text=text)
                    if "category" in case.expected:
                        act("Stay category → " + case.expected["category"], kind="select")
                    if case.expected.get("free"):
                        act("Free cancellation")
                if case.expected.get("searched"):
                    act("Find stays")
                act("View " + case.expected["name"])
                verified = HARNESS.verify_fixture(agent.browser, case)
                assert verified["success"]
                if index in (1, 3):
                    assert verified["observed"]["result"]["searched"] is False
                if index in (1, 3, 4):
                    assert verified["observed"]["result"]["free"] is False
            finally:
                agent.close()
    assert HARNESS.upstream_manifest(upstream) == original


@pytest.mark.parametrize("index", range(3))
@pytest.mark.integration
def test_new_travel_requirements_are_reachable_through_guarded_actions(
    real_browser_environment, index
):
    """Test-only diagnostic scripts establish reachability, never model performance."""
    upstream, chromium = real_browser_environment
    cases = HARNESS.make_travel_goal_challenges(upstream)
    case = cases[index]
    original = (upstream / HARNESS.UPSTREAM_TRAVEL_SOURCE).read_bytes()
    assert all(candidate.html.encode() == original for candidate in cases)
    assert hashlib.sha256(original).hexdigest() == HARNESS.UPSTREAM_TRAVEL_SHA256
    transport = HARNESS.PlaywrightCDP(chromium)
    with HARNESS.fixture_server(cases) as base:
        with HARNESS.load_upstream(upstream, transport) as (agent_module, _model_module):
            agent = agent_module.Agent(base + case.url_suffix, case.goal, screenshots=False)

            def act(label, *, kind="click", text=None):
                page = agent.browser.observe(screenshot=False)
                action = next(
                    action
                    for action in page["actions"]
                    if action["label"] == label and action["kind"] == kind
                )
                agent.browser.act(action, page, text=text)

            try:
                assert HARNESS.verify_fixture(agent.browser, case)["success"] is False
                if index < 2:
                    act("Destination", kind="fill", text=case.expected["query"])
                    act("Stay category → " + case.expected["category"], kind="select")
                    if case.expected["free"]:
                        act("Free cancellation")
                    # Even the correct details cannot satisfy an unsubmitted query.
                    act("View " + case.expected["name"])
                    failed = HARNESS.verify_fixture(agent.browser, case)
                    assert failed["success"] is False
                    assert failed["observed"]["result"]["hash"] == case.expected["hash"]
                    assert failed["mismatches"]["searched"]["observed"] is False
                    act("← All stays")
                    act("Destination", kind="fill", text=case.expected["query"])
                    act("Find stays")
                act("View " + case.expected["name"])
                verified = HARNESS.verify_fixture(agent.browser, case)
                assert verified["success"] is True
                if index == 1:
                    assert verified["observed"]["result"]["free"] is False
                if index == 2:
                    assert verified["observed"]["result"]["searched"] is False
            finally:
                agent.close()


def test_endpoint_rejects_embedded_credentials():
    for endpoint in (
        "https://name:secret@example.com/v1",
        "https://example.com/v1?key=secret",
        "file:///tmp/model",
        "https://example.com/v1#key",
    ):
        with pytest.raises(ValueError):
            HARNESS._endpoint(endpoint)
    assert (
        HARNESS._endpoint("http://127.0.0.1:9845/v1/systemone/")
        == "http://127.0.0.1:9845/v1/systemone"
    )


def test_done_without_independent_result_fails():
    case = HARNESS.make_cases()[0]
    browser = SimpleNamespace(
        evaluate=lambda expression: {
            "result": None,
            "result_visible": False,
            "text": "All finished",
            "url": case.path,
        }
    )
    result = HARNESS.verify_fixture(browser, case)
    assert result["success"] is False
    assert "visible_result" in result["mismatches"]


@pytest.mark.parametrize(
    "status,error", [("blocked", None), ("ready", "Decision budget"), ("done", "Execution error")]
)
def test_satisfied_dom_with_budget_block_or_error_is_not_a_task_pass(status, error):
    assert HARNESS._task_success({"success": True}, {"status": status}, error) is False
    assert HARNESS._task_success({"success": True}, {"status": "done"}, None) is True


def test_technical_judge_failure_is_unknown_rather_than_policy_false_done():
    verification = {"success": None, "error": "Judge disconnected"}
    assert HARNESS._premature_done(verification, {"status": "done"}) is False
    assert HARNESS._task_success(verification, {"status": "done"}, "Judge disconnected") is False
    assert HARNESS._premature_done({"success": False}, {"status": "done"}) is True


def test_verifier_requires_saved_and_current_checkbox_state():
    case = next(case for case in HARNESS.make_cases() if case.task_id == "already_checked")
    browser = SimpleNamespace(
        evaluate=lambda expression: {
            "result": {"kind": "preferences", "Weekly newsletter": True, "Product tips": False},
            "result_visible": True,
            "fields": [
                {"label": "Weekly newsletter", "checked": False},
                {"label": "Product tips", "checked": False},
            ],
        }
    )
    result = HARNESS.verify_fixture(browser, case)
    assert result["success"] is False
    assert result["mismatches"]["current_Weekly newsletter"]["observed"] is False


def test_adapter_redirects_only_policy_and_preserves_actual_identity(monkeypatch):
    calls = []

    def original_post(url, key, body):
        calls.append((url, key, body))
        return {"model": "actual-model", "answers": {}, "authorization": key}

    def original_text(context):
        return "legacy", {"model": "legacy"}

    agent = SimpleNamespace(field_text=original_text)
    model = SimpleNamespace(post_json=original_post)
    monkeypatch.setenv("TYPESAFE_MODEL", "old-model")
    with HARNESS.model_adapter(
        agent,
        model,
        endpoint="http://127.0.0.1:123/v1/systemone",
        model="requested-model",
        api_key="never-log-this",
    ) as (records, secrets):
        response = model.post_json(
            "https://api.typesafe.ai/v1/systemone",
            "never-log-this",
            {"model": "ignored", "state": {"text": "observed"}},
        )
        assert response["model"] == "actual-model"
        model.post_json(
            "http://helper/v1/chat/completions", "never-log-this", {"model": "text-model"}
        )
        assert calls[0][0] == "http://127.0.0.1:123/v1/systemone"
        assert calls[0][2]["model"] == "requested-model"
        assert calls[1][0] == "http://helper/v1/chat/completions"
        assert "never-log-this" not in json.dumps(records)
        assert "never-log-this" in secrets
        with pytest.raises(ValueError, match="explicit text_endpoint"):
            agent.field_text({"goal": "type something"})
    assert model.post_json is original_post
    assert agent.field_text is original_text
    assert os.environ["TYPESAFE_MODEL"] == "old-model"


def test_text_callback_sees_only_upstream_context_and_requires_identity():
    context = {
        "goal": "Select a city",
        "field": {"label": "Destination"},
        "page": {"text": "visible suggestions"},
        "recent_actions": [],
    }
    received = []

    def callback(value):
        received.append(value)
        return "Generated city", {"model": "genuine-text-model", "usage": {"tokens": 5}}

    agent = SimpleNamespace(field_text=lambda _: None)
    model = SimpleNamespace(post_json=lambda *_: None)
    with HARNESS.model_adapter(
        agent,
        model,
        endpoint="http://localhost/model",
        model="policy",
        api_key=None,
        text_helper=callback,
    ) as (records, _secrets):
        text, metadata = agent.field_text(context)
    assert received == [context]
    assert text == "Generated city"
    assert metadata["model"] == "genuine-text-model"
    assert records[0]["request"] == context
    assert "expected" not in records[0]["request"]


def test_adapter_logs_failed_response_before_reraising():
    model = SimpleNamespace(post_json=lambda *_: {"answers": {}})
    agent = SimpleNamespace(field_text=lambda _: None)
    with HARNESS.model_adapter(
        agent, model, endpoint="http://localhost/model", model="policy", api_key=None
    ) as (records, _secrets):
        with pytest.raises(ValueError, match="identity"):
            model.post_json("https://api.typesafe.ai/v1/systemone", "ignored", {"state": {}})
        assert records[0]["response"] == {"answers": {}}
        assert "identity" in records[0]["error"]


@pytest.fixture
def pinned_upstream():
    if os.environ.get("JEV_BROWSER_TESTS") != "1":
        pytest.skip("Set JEV_BROWSER_TESTS=1 and JEV_UPSTREAM_PATH for real Chromium checks")
    pytest.importorskip("playwright.sync_api")
    upstream = Path(os.environ.get("JEV_UPSTREAM_PATH", "/tmp/s1-jev-ultrafast"))
    HARNESS.upstream_manifest(upstream)
    return upstream


@pytest.fixture
def real_browser_environment(pinned_upstream):
    from playwright.sync_api import sync_playwright

    upstream = pinned_upstream
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        try:
            yield upstream, browser
        finally:
            browser.close()


@contextmanager
def _local_prefetch_site(*, outside_host=False):
    """Actual local HTTP documents, including a scripted policy used only by tests."""
    hits, policy_payloads = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(
                {
                    "path": self.path,
                    "host": self.headers.get("Host"),
                    "purpose": self.headers.get("Sec-Purpose"),
                }
            )
            page = self.path.split("#", 1)[0]
            host = "localhost" if outside_host else "127.0.0.1"
            next_path = "/second" if page == "/first" else "/third"
            link = (
                f'<a href="http://{host}:{self.server.server_port}{next_path}">Next page</a>'
                if page != "/third"
                else '<a href="#section">Section</a><h2 id="section">Section</h2>'
            )
            rules = {
                "prefetch": [
                    {
                        "source": "document",
                        "where": {"selector_matches": "a"},
                        "eagerness": "conservative",
                    }
                ]
            }
            body = (
                f"<!doctype html><title>{page}</title><h1>{page}</h1>"
                "<style>a{display:block;width:200px;height:60px}</style>"
                f'{link}<script type="speculationrules">{json.dumps(rules)}</script>'
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Cache-Control", "max-age=600")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            policy_payloads.append(payload)
            page = urlsplit(payload["state"]["page"]["url"]).path
            operation = "DONE" if page == "/third" else "CLICK"
            choices = payload["questions"]["operation"]["criteria"]
            answers = {
                "operation": {
                    "choice": operation,
                    "confidence": 1.0,
                    "probabilities": {key: float(key == operation) for key in choices},
                }
            }
            if operation == "CLICK":
                criteria = payload["questions"]["click_target"]["criteria"]
                target = next(
                    key for key, value in criteria.items() if "Next page" in value["element"]
                )
                answers["click_target"] = {
                    "choice": target,
                    "confidence": 1.0,
                    "probabilities": {key: float(key == target) for key in criteria},
                }
            body = json.dumps({"model": "test-only-local-prefetch-policy", "answers": answers})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", hits, policy_payloads
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _allow_test_loopback_http(monkeypatch, base):
    """Normalize only the local test origin, retaining actual host scope decisions.

    Production remains strict HTTPS. Both permitted 127.0.0.1 and forbidden
    localhost are actual loopback hosts; their URLs stay unchanged in the ledgers.
    """
    port = urlsplit(base).port
    original_request = HARNESS.PlaywrightCDP._public_request_scope
    original_document = HARNESS.PlaywrightCDP._public_document_url_scope

    def normalize(url):
        parsed = urlsplit(url)
        assert parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}
        assert parsed.port == port
        return parsed._replace(scheme="https", netloc=parsed.hostname).geturl()

    def request_scope(request, hosts):
        proxy = SimpleNamespace(
            method=request.method,
            url=normalize(request.url),
            is_navigation_request=request.is_navigation_request,
            frame=request.frame,
        )
        return original_request(proxy, hosts)

    def document_scope(url, hosts):
        # The owned pre-navigation about:blank is handled by the observer itself.
        return original_document(normalize(url), hosts)

    monkeypatch.setattr(HARNESS.PlaywrightCDP, "_public_request_scope", staticmethod(request_scope))
    monkeypatch.setattr(
        HARNESS.PlaywrightCDP, "_public_document_url_scope", staticmethod(document_scope)
    )


def _observe_local_path(native, stale_page, path):
    deadline = time.monotonic() + 5
    while True:
        try:
            snapshot = native.observe(screenshot=False)
            if urlsplit(snapshot["url"]).path == path:
                return snapshot
        except stale_page:
            pass
        if time.monotonic() >= deadline:
            pytest.fail(f"Local native navigation did not settle on {path}")
        time.sleep(0.01)


@pytest.mark.integration
@pytest.mark.parametrize("outside_host", [False, True])
def test_real_native_prefetch_documents_are_captured_despite_missing_playwright_events(
    real_browser_environment, monkeypatch, outside_host
):
    upstream, chromium = real_browser_environment
    with _local_prefetch_site(outside_host=outside_host) as (base, hits, _payloads):
        _allow_test_loopback_http(monkeypatch, base)
        transport = HARNESS.PlaywrightCDP(chromium)
        transport.public_navigation_hosts = ("127.0.0.1",)
        with HARNESS.load_upstream(upstream, transport) as (agent_module, _model_module):
            native = agent_module.Browser(base + "/first")
            snapshots, navigation_entries = [], []
            try:
                for index, path in enumerate(("/first", "/second", "/third")):
                    snapshot = _observe_local_path(native, agent_module.StalePage, path)
                    snapshots.append(snapshot)
                    navigation_entries.append(
                        native.evaluate(
                            "({timeOrigin:performance.timeOrigin,"
                            "deliveryType:performance.getEntriesByType('navigation')[0].deliveryType})"
                        )
                    )
                    if index < 2:
                        action = next(
                            a
                            for a in snapshot["actions"]
                            if a["kind"] == "click" and a["label"] == "Next page"
                        )
                        native.act(action, snapshot)
                assert len({entry["timeOrigin"] for entry in navigation_entries}) == 3
                assert [entry["deliveryType"] for entry in navigation_entries[1:]] == [
                    "navigational-prefetch",
                    "navigational-prefetch",
                ]
                assert [
                    urlsplit(row["url"]).path for row in transport.public_navigation_events
                ] == ["/first"]
                assert [
                    urlsplit(row["url"]).path for row in transport.public_document_requests
                ] == ["/first", "/second", "/third"]
                commits = [
                    row for row in transport.public_document_commits if not row["setup_blank"]
                ]
                assert [urlsplit(row["url"]).path for row in commits] == [
                    "/first",
                    "/second",
                    "/third",
                ]
                assert all(row["method"] == "GET" for row in transport.public_document_requests)
                assert all(hit["purpose"] == "prefetch" for hit in hits if hit["path"] != "/first")
                assert transport.public_request_blocks == transport.public_observer_errors == []
                if outside_host:
                    assert len(transport.public_navigation_violations) == 4
                    assert all(
                        urlsplit(row["url"]).hostname == "localhost"
                        for row in transport.public_navigation_violations
                    )
                else:
                    assert transport.public_navigation_violations == []
                section = next(a for a in snapshots[-1]["actions"] if a["label"] == "Section")
                native.act(section, snapshots[-1])
                deadline = time.monotonic() + 5
                while not any(
                    row["source"] == "cdp_within_document"
                    for row in transport.public_document_commits
                ):
                    assert time.monotonic() < deadline
                    native.evaluate("location.href")
                assert transport.public_document_commits[-1]["url"].endswith("#section")
                assert transport.public_document_commits[-1]["source"] == "cdp_within_document"
                assert len(transport.public_document_requests) == 3
            finally:
                native.close()


@pytest.mark.integration
def test_real_runner_rejects_prefetched_outside_host_even_with_done_and_positive_judge(
    pinned_upstream, tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import jev_public_web_cases as public

    with _local_prefetch_site(outside_host=True) as (base, hits, payloads):
        _allow_test_loopback_http(monkeypatch, base)
        case = SimpleNamespace(
            case_id="test-local-prefetch",
            goal="Open Next page twice and stop.",
            start_url=base + "/first",
            allowed_top_level_hosts=("127.0.0.1",),
        )
        monkeypatch.setattr(public, "get_public_web_case", lambda _case_id: case)
        monkeypatch.setattr(public, "case_manifest", lambda _case: {"case_id": case.case_id})

        def positive_fresh_judge(browser, _case):
            observed = browser.evaluate(
                "({url:location.href,heading:document.querySelector('h1').innerText})"
            )
            return {
                "success": observed["heading"] == "/third",
                "observed": observed,
                "mismatches": {},
            }

        monkeypatch.setattr(public, "verify_public_web_case", positive_fresh_judge)
        output = tmp_path / "prefetched-outside-host"
        summary = HARNESS.run_benchmark(
            upstream_path=pinned_upstream,
            endpoint=base + "/policy",
            model="test-only-local-prefetch-policy",
            output_dir=output,
            public_case_ids=[case.case_id],
        )
        outcome = summary["outcomes"][0]
        assert outcome["status"] == "done" and outcome["goal_satisfied"] is True
        assert summary["successes"] == 0 and outcome["success"] is False
        assert outcome["error_stage"] == "public_navigation_scope"
        assert len(outcome["public_scope_violations"]) == 4
        assert outcome["public_request_blocks"] == outcome["public_observer_errors"] == []
        assert len(outcome["public_navigation_events"]) == 1
        assert len(outcome["public_document_requests"]) == 3
        assert any(
            hit["host"].startswith("localhost:") and hit["purpose"] == "prefetch" for hit in hits
        )
        ledger = json.loads((output / "public-request-ledger.json").read_bytes())
        assert ledger["document_requests"] == outcome["public_document_requests"]
        assert ledger["document_commits"] == outcome["public_document_commits"]
        assert payloads and all(
            "public_document" not in json.dumps(payload) for payload in payloads
        )


@pytest.mark.integration
def test_upstream_manifest_checks_executed_initializer(pinned_upstream, tmp_path):
    copied = tmp_path / "upstream"
    shutil.copytree(pinned_upstream, copied)
    manifest = HARNESS.upstream_manifest(copied)
    relative = "jev_ultrafast/__init__.py"
    assert (
        manifest["source_sha256"][relative]
        == hashlib.sha256((copied / relative).read_bytes()).hexdigest()
    )
    with (copied / relative).open("ab") as source:
        source.write(b"\n# changed executed package initializer\n")
    with pytest.raises(ValueError, match="local changes.*__init__"):
        HARNESS.upstream_manifest(copied)


@pytest.mark.integration
def test_all_original_nine_fixture_hashes_stay_unchanged(pinned_upstream):
    cases = HARNESS.make_cases(seed=20261014, upstream_path=pinned_upstream)
    assert len(cases) == 10
    original = {
        case.task_id: hashlib.sha256(case.html.encode()).hexdigest()
        for case in cases
        if case.task_id != "unapplied_search"
    }
    assert original == ORIGINAL_FIXTURE_HASHES


@pytest.mark.integration
def test_real_upstream_guarded_click_and_independent_dom_judge(
    real_browser_environment, monkeypatch
):
    upstream, chromium = real_browser_environment
    cases = HARNESS.make_cases()
    case = next(case for case in cases if case.task_id == "premature_done")
    transport = HARNESS.PlaywrightCDP(chromium)
    transport.observation_expression = (upstream / "jev_ultrafast/snapshot.js").read_text()
    observations = []
    transport.observation_sink = observations.append
    with HARNESS.fixture_server(cases) as base:
        with HARNESS.load_upstream(upstream, transport) as (agent_module, _model_module):
            agent = agent_module.Agent(base + case.path, case.goal)
            try:
                page = agent.state["page"]
                captured_page = observations[-1]["response"]["result"]["value"]
                assert "fingerprint" not in captured_page
                assert captured_page == {k: v for k, v in page.items() if k != "fingerprint"}
                assert HARNESS.verify_fixture(agent.browser, case)["success"] is False
                action = next(
                    action for action in page["actions"] if action["label"] == "Activate report"
                )
                assert agent.browser.act(action, page) == {"executed": action["id"]}
                assert HARNESS.verify_fixture(agent.browser, case)["success"] is True
                # The previous observation cannot authorize a second mutation.
                with pytest.raises(agent_module.StalePage):
                    agent.browser.act(action, page)
                chosen_pages = []

                def choose(fresh_page, _goal, _history):
                    chosen_pages.append(fresh_page)
                    return {"choice": "DONE"}

                monkeypatch.setattr(agent_module, "choose", choose)
                agent.command("predict")
                assert chosen_pages[0]["fingerprint"] != page["fingerprint"]
                refreshed = observations[-1]["response"]["result"]["value"]
                assert len(observations) >= 2
                assert refreshed == {k: v for k, v in chosen_pages[0].items() if k != "fingerprint"}
            finally:
                agent.close()


@pytest.mark.integration
@pytest.mark.parametrize("task_id", HARNESS.TASK_IDS)
def test_real_fixture_can_succeed_through_observed_upstream_actions(
    real_browser_environment, task_id
):
    """Fixture sanity only: prepared actions here are never benchmark policy inputs."""
    upstream, chromium = real_browser_environment
    cases = HARNESS.make_cases(seed=83, upstream_path=upstream)
    case = next(case for case in cases if case.task_id == task_id)
    transport = HARNESS.PlaywrightCDP(chromium)
    with HARNESS.fixture_server(cases) as base:
        with HARNESS.load_upstream(upstream, transport) as (agent_module, _model_module):
            agent = agent_module.Agent(base + case.url_suffix, case.goal)

            def act(label, *, kind="click", text=None):
                page = agent.browser.observe(screenshot=False)
                action = next(
                    action
                    for action in page["actions"]
                    if action["label"] == label and action["kind"] == kind
                )
                agent.browser.act(action, page, text=text)

            try:
                assert HARNESS.verify_fixture(agent.browser, case)["success"] is False
                if task_id in {"filter_open", "search_filter_open"}:
                    if case.needs_text:
                        act("Search records", kind="fill", text=case.expected["name"])
                    act("Region → " + case.expected["region"], kind="select")
                    act("Verified only")
                    act("Apply filters")
                    act("Open " + case.expected["name"])
                elif task_id == "already_checked":
                    act("Product tips")
                    act("Save preferences")
                elif task_id == "native_select":
                    act("Review mode → " + case.expected["mode"], kind="select")
                    act("Save review mode")
                elif task_id == "autocomplete":
                    act("Destination", kind="fill", text=case.expected["city"])
                    act(case.expected["city"])
                    act("Confirm destination")
                elif task_id == "submit_populated":
                    act("Submit enrollment")
                elif task_id == "premature_done":
                    act("Activate report")
                elif task_id == "completion_toggle":
                    clicks = [
                        action["label"]
                        for action in agent.browser.observe(screenshot=False)["actions"]
                        if action["kind"] == "click"
                    ]
                    assert clicks == ["Mark complete"]
                    act("Mark complete")
                elif task_id == "upstream_travel":
                    act("Destination", kind="fill", text="Lisbon")
                    act("Find stays")
                    act("Stay category → Design", kind="select")
                    act("Free cancellation")
                    act("View Casa Flora")
                elif task_id == "unapplied_search":
                    act("Search area", kind="fill", text=case.expected["query"])
                    act("Collection → " + case.expected["category"], kind="select")
                    act("Verified only")
                    act("Apply search")
                    act("Open " + case.expected["name"])
                assert HARNESS.verify_fixture(agent.browser, case)["success"] is True
                if task_id == "completion_toggle":
                    clicks = [
                        action["label"]
                        for action in agent.browser.observe(screenshot=False)["actions"]
                        if action["kind"] == "click"
                    ]
                    assert clicks == ["Reset completion"]
                    act("Reset completion")
                    assert HARNESS.verify_fixture(agent.browser, case)["success"] is False
                    clicks = [
                        action["label"]
                        for action in agent.browser.observe(screenshot=False)["actions"]
                        if action["kind"] == "click"
                    ]
                    assert clicks == ["Mark complete"]
            finally:
                agent.close()


@pytest.mark.integration
@pytest.mark.parametrize("suite_split", ["development", "holdout"])
def test_matching_record_does_not_prove_search_was_applied(real_browser_environment, suite_split):
    upstream, chromium = real_browser_environment
    case = HARNESS.make_unapplied_search_case(seed=91, suite_split=suite_split)
    labels = (
        ("Search area", "Collection", "Verified only", "Apply search")
        if suite_split == "development"
        else ("Location query", "Record group", "Certified only", "Update results")
    )
    query, category, checkbox, submit = labels
    transport = HARNESS.PlaywrightCDP(chromium)
    with HARNESS.fixture_server([case]) as base:
        with HARNESS.load_upstream(upstream, transport) as (agent_module, _model_module):
            agent = agent_module.Agent(base + case.path, case.goal)

            def act(label, *, kind="click", text=None):
                page = agent.browser.observe(screenshot=False)
                action = next(
                    action
                    for action in page["actions"]
                    if action["label"] == label and action["kind"] == kind
                )
                agent.browser.act(action, page, text=text)

            try:
                # The matching record is genuinely available before any search.
                assert any(
                    action["label"] == "Open " + case.expected["name"]
                    for action in agent.state["page"]["actions"]
                )
                act(query, kind="fill", text=case.expected["query"])
                act(category + " → " + case.expected["category"], kind="select")
                act(checkbox)
                act("Open " + case.expected["name"])
                verification = HARNESS.verify_fixture(agent.browser, case)
                assert verification["success"] is False
                assert verification["observed"]["result"]["name"] == case.expected["name"]
                assert verification["observed"]["typed_query"] == case.expected["query"]
                assert verification["observed"]["applied"]["query"] == ""
                assert verification["observed"]["applied"]["search_committed"] is False
                assert verification["observed"]["result"]["opened_at_commit"] == 0
                act(submit)
                assert HARNESS.verify_fixture(agent.browser, case)["success"] is False
                act("Open " + case.expected["name"])
                assert HARNESS.verify_fixture(agent.browser, case)["success"] is True
                # Re-applying invalidates an earlier detail view; it must be opened
                # under the current committed search rather than a stale receipt.
                act(submit)
                assert HARNESS.verify_fixture(agent.browser, case)["success"] is False
                act("Open " + case.expected["name"])
                assert HARNESS.verify_fixture(agent.browser, case)["success"] is True
            finally:
                agent.close()


@pytest.mark.integration
def test_original_travel_source_and_judge_require_search_submission(real_browser_environment):
    upstream, chromium = real_browser_environment
    case = HARNESS.make_upstream_travel_case(upstream)
    original = (upstream / HARNESS.UPSTREAM_TRAVEL_SOURCE).read_bytes()
    assert case.html.encode() == original
    assert hashlib.sha256(original).hexdigest() == HARNESS.UPSTREAM_TRAVEL_SHA256
    assert case.source["revision"] == HARNESS.UPSTREAM_REVISION
    transport = HARNESS.PlaywrightCDP(chromium)
    with HARNESS.fixture_server([case]) as base:
        with HARNESS.load_upstream(upstream, transport) as (agent_module, _model_module):
            agent = agent_module.Agent(base + case.url_suffix, case.goal)

            def act(label, *, kind="click", text=None):
                page = agent.browser.observe(screenshot=False)
                action = next(
                    action
                    for action in page["actions"]
                    if action["label"] == label and action["kind"] == kind
                )
                agent.browser.act(action, page, text=text)

            try:
                # Reproduce the historical omission: typed value and property selection
                # satisfy the appearance of the task, but the search was never submitted.
                act("Destination", kind="fill", text="Lisbon")
                act("Stay category → Design", kind="select")
                act("Free cancellation")
                act("View Casa Flora")
                verification = HARNESS.verify_fixture(agent.browser, case)
                assert verification["success"] is False
                assert verification["observed"]["result"]["hash"] == "#casa-flora"
                assert verification["observed"]["result"]["category"] == "Design"
                assert verification["observed"]["result"]["free"] is True
                assert verification["mismatches"]["query"]["observed"] == ""
                assert verification["mismatches"]["searched"]["observed"] is False
                act("← All stays")
                act("Destination", kind="fill", text="Lisbon")
                act("Find stays")
                act("View Casa Flora")
                assert HARNESS.verify_fixture(agent.browser, case)["success"] is True
            finally:
                agent.close()


@pytest.mark.integration
def test_real_runner_saves_false_done_requests_and_never_overwrites(
    pinned_upstream, tmp_path, monkeypatch
):
    upstream = pinned_upstream
    incoming = []

    class DonePolicy(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            incoming.append(body)
            choices = body["questions"]["operation"]["criteria"]
            operation = (
                "CLICK"
                if (
                    self.path.endswith("/activate")
                    or (
                        self.path.endswith("/complete")
                        and "is inactive." in body["state"]["page"]["text"]
                    )
                )
                else "DONE"
            )
            response = {
                "model": "test-only-always-done",
                "answers": {
                    "operation": {
                        "choice": operation,
                        "confidence": 1.0,
                        "probabilities": {key: float(key == operation) for key in choices},
                    }
                },
            }
            if operation == "CLICK":
                criteria = body["questions"]["click_target"]["criteria"]
                target = next(
                    key for key, value in criteria.items() if "Activate report" in value["element"]
                )
                response["answers"]["click_target"] = {
                    "choice": target,
                    "confidence": 1.0,
                    "probabilities": {key: float(key == target) for key in criteria},
                }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), DonePolicy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "false-done-run"
    arguments = {
        "upstream_path": upstream,
        "endpoint": f"http://127.0.0.1:{server.server_port}/v1/systemone",
        "model": "requested-diagnostic-policy",
        "output_dir": output,
        "task_ids": ["premature_done"],
    }
    try:
        summary = HARNESS.run_benchmark(**arguments)
        assert summary["attempts"] == 1
        assert summary["successes"] == 0
        assert summary["premature_done_count"] == 1
        assert incoming[0]["model"] == "requested-diagnostic-policy"
        assert "expected" not in incoming[0]
        attempt = next((output / "attempts").iterdir())
        calls = json.loads((attempt / "model_calls.json").read_text())
        assert calls[0]["response"]["model"] == "test-only-always-done"
        assert calls[0]["transport_attempts"][0]["http_status"] == 200
        assert (attempt / "final.html").is_file()
        assert (attempt / "states.json").is_file()
        observations = [
            json.loads(line) for line in (attempt / "observations.jsonl").read_text().splitlines()
        ]
        assert len(observations) == summary["outcomes"][0]["native_observations"] > 0
        assert [row["sequence"] for row in observations] == list(range(len(observations)))
        assert all("actions" in row["response"]["result"]["value"] for row in observations)
        source_hashes = summary["configuration"]["upstream"]["source_sha256"]
        assert len(source_hashes) == len(HARNESS.UPSTREAM_FILES)
        assert source_hashes[HARNESS.UPSTREAM_TRAVEL_SOURCE] == HARNESS.UPSTREAM_TRAVEL_SHA256
        with pytest.raises(FileExistsError):
            HARNESS.run_benchmark(**arguments)
        # A correct last mutation without a subsequent DONE is still a stopping failure.
        budget_summary = HARNESS.run_benchmark(
            **{
                **arguments,
                "endpoint": f"http://127.0.0.1:{server.server_port}/activate",
                "output_dir": tmp_path / "budget-run",
                "max_steps": 1,
            }
        )
        assert budget_summary["goals_satisfied"] == 1
        assert budget_summary["successes"] == 0
        assert budget_summary["outcomes"][0]["status"] == "ready"
        assert "budget" in budget_summary["outcomes"][0]["error"]
        historical_output = tmp_path / "historical-travel-run"
        historical_summary = HARNESS.run_benchmark(
            **{**arguments, "output_dir": historical_output, "task_ids": ["upstream_travel"]}
        )
        fixture = historical_summary["configuration"]["fixture_manifest"][0]
        assert fixture["source"]["sha256"] == HARNESS.UPSTREAM_TRAVEL_SHA256
        assert fixture["query_string"] == "scenario=travel"
        saved_source = historical_output / "fixtures" / fixture["case_id"] / "page.html"
        assert saved_source.read_bytes() == (upstream / HARNESS.UPSTREAM_TRAVEL_SOURCE).read_bytes()
        assert historical_summary["premature_done_count"] == 1

        challenge_arguments = {
            **arguments,
            "output_dir": tmp_path / "travel-goal-challenges",
            "travel_challenges_only": True,
        }
        challenge_arguments.pop("task_ids")
        challenge_start = len(incoming)
        challenge_summary = HARNESS.run_benchmark(**challenge_arguments)
        configuration = challenge_summary["configuration"]
        assert configuration["travel_challenges_only"] is True
        assert configuration["suite_scope"] == HARNESS.TRAVEL_CHALLENGE_SCOPE
        assert challenge_summary["attempts"] == challenge_summary["premature_done_count"] == 3
        assert challenge_summary["successes"] == 0
        assert [fixture["case_id"] for fixture in configuration["fixture_manifest"]] == [
            "travel-goal-000-0",
            "travel-goal-000-1",
            "travel-goal-000-2",
        ]
        assert all(
            fixture["html_sha256"] == HARNESS.UPSTREAM_TRAVEL_SHA256
            for fixture in configuration["fixture_manifest"]
        )
        assert len(incoming[challenge_start:]) == 3
        assert all("expected" not in payload for payload in incoming[challenge_start:])
        simple_judge = json.loads(
            (
                Path(challenge_arguments["output_dir"]) / "fixtures/travel-goal-000-2/judge.json"
            ).read_bytes()
        )
        assert simple_judge["expected"] == {"hash": "#casa-flora", "name": "Casa Flora"}

        with monkeypatch.context() as faults:

            def failed_judge(*_args):
                raise RuntimeError("diagnostic judge failure")

            faults.setattr(HARNESS, "verify_fixture", failed_judge)
            unknown = HARNESS.run_benchmark(
                **{**arguments, "output_dir": tmp_path / "judge-error-run"}
            )
            assert unknown["successes"] == unknown["premature_done_count"] == 0
            outcome = unknown["outcomes"][0]
            assert outcome["goal_satisfied"] is None
            assert outcome["error_stage"] == "verification"
            assert "diagnostic judge failure" in outcome["verification_error"]

        with monkeypatch.context() as faults:
            original_write = Path.write_text

            def failed_dom_write(self, *args, **kwargs):
                if self.name == "final.html":
                    raise PermissionError("diagnostic final HTML write failure")
                return original_write(self, *args, **kwargs)

            faults.setattr(Path, "write_text", failed_dom_write)
            capture = HARNESS.run_benchmark(
                **{
                    **arguments,
                    "endpoint": f"http://127.0.0.1:{server.server_port}/complete",
                    "output_dir": tmp_path / "dom-write-error-run",
                }
            )
            assert capture["goals_satisfied"] == 1
            assert capture["successes"] == capture["premature_done_count"] == 0
            outcome = capture["outcomes"][0]
            assert outcome["status"] == "done"
            assert outcome["verification"]["success"] is True
            assert outcome["verification_error"] is None
            assert outcome["error_stage"] == outcome["dom_error"]["stage"] == "dom_write"
            assert "diagnostic final HTML write failure" in outcome["dom_error"]["message"]
        original_load = HARNESS.load_upstream

        @contextmanager
        def failing_close_load(*args):
            with original_load(*args) as modules:
                original_close = modules[0].Agent.close

                def close_then_fail(self):
                    original_close(self)
                    raise RuntimeError("diagnostic close failure")

                modules[0].Agent.close = close_then_fail
                yield modules

        monkeypatch.setattr(HARNESS, "load_upstream", failing_close_load)
        close_summary = HARNESS.run_benchmark(
            **{
                **arguments,
                "output_dir": tmp_path / "close-error-run",
                "task_ids": ["premature_done", "completion_toggle"],
            }
        )
        assert close_summary["attempts"] == 2
        assert all(
            "diagnostic close failure" in outcome["close_error"]
            for outcome in close_summary["outcomes"]
        )
        assert len(list((tmp_path / "close-error-run" / "attempts").glob("*/outcome.json"))) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
