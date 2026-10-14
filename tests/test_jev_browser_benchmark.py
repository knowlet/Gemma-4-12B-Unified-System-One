"""Transport and independent-judge checks; scripted policies are test-only diagnostics."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "jev_browser_benchmark", ROOT / "scripts/jev_browser_benchmark.py"
)
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


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
def test_travel_query_equivalence_still_requires_correct_city_and_submission(query, searched, success):
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
def test_real_upstream_guarded_click_and_independent_dom_judge(real_browser_environment):
    upstream, chromium = real_browser_environment
    cases = HARNESS.make_cases()
    case = next(case for case in cases if case.task_id == "premature_done")
    transport = HARNESS.PlaywrightCDP(chromium)
    with HARNESS.fixture_server(cases) as base:
        with HARNESS.load_upstream(upstream, transport) as (agent_module, _model_module):
            agent = agent_module.Agent(base + case.path, case.goal)
            try:
                page = agent.state["page"]
                assert HARNESS.verify_fixture(agent.browser, case)["success"] is False
                action = next(
                    action for action in page["actions"] if action["label"] == "Activate report"
                )
                assert agent.browser.act(action, page) == {"executed": action["id"]}
                assert HARNESS.verify_fixture(agent.browser, case)["success"] is True
                # The previous observation cannot authorize a second mutation.
                with pytest.raises(agent_module.StalePage):
                    agent.browser.act(action, page)
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
