"""Diversified scripted supervision is development data, never a model score."""

import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_jev_browser_training_data import COLLECTOR


@pytest.fixture
def cases(monkeypatch):
    original = SimpleNamespace(
        html="unchanged exposed HTML", query_string="scenario=travel", source={"sha256": "parent"}
    )
    monkeypatch.setattr(COLLECTOR, "make_upstream_travel_case", lambda *_args: original)
    return COLLECTOR.make_training_travel_cases("unused")


def test_training_inventory_composes_fifteen_bounded_goal_families(cases, monkeypatch):
    assert len(cases) == 15
    assert all(case.html == "unchanged exposed HTML" for case in cases)
    assert all(case.case_id == f"training-travel-000-{i:02d}" for i, case in enumerate(cases))
    assert {case.expected["name"] for case in cases} == {
        "Casa Flora",
        "The Glasshouse",
        "Serra Lodge",
    }
    requirements = [
        {key for key in case.expected if key not in {"hash", "name"}} for case in cases[:5]
    ]
    assert requirements == [
        set(),
        {"query", "searched"},
        {"category"},
        {"query", "searched", "category"},
        {"query", "searched", "category", "free"},
    ]
    exposed_literals = {
        "Use the destination search and filters to find Design stays in Copenhagen with Free cancellation, then open The Glasshouse.",
        "Use the destination search and filters to find Nature stays in Lisbon without enabling Free cancellation, then open Serra Lodge.",
        "Open the details for Casa Flora.",
    }
    generated = [
        case
        for repeat in range(4)
        for case in COLLECTOR.make_training_travel_cases("unused", repeat)
    ]
    assert len({case.goal for case in generated}) == 60
    assert not exposed_literals & {case.goal for case in generated}
    assert {
        case.expected["free"]
        for case in generated
        if case.expected.get("name") == "The Glasshouse" and "free" in case.expected
    } == {False, True}
    assert all(
        case.expected["free"] is False
        for case in generated
        if case.expected["name"] == "Serra Lodge" and "free" in case.expected
    )


def test_static_phases_require_only_supplied_requirements(cases):
    action = COLLECTOR.TeachingAction
    assert COLLECTOR.development_phases(cases[0]) == [[action("View Casa Flora")]]
    assert COLLECTOR.development_phases(cases[1]) == [
        [action("Destination", "fill", "Lisbon")],
        [action("Find stays")],
        [action("View Casa Flora")],
    ]
    assert COLLECTOR.development_phases(cases[2]) == [
        [action("Stay category → Design", "select")],
        [action("View Casa Flora")],
    ]
    assert action("Free cancellation") not in [
        a for phase in COLLECTOR.development_phases(cases[14]) for a in phase
    ]


def test_dynamic_teacher_distinguishes_typed_applied_and_explicitly_off(cases):
    case = cases[9]  # Glasshouse, Copenhagen, Design, explicit off at repeat zero.
    observed = {
        "result": {"hash": "", "category": "Design", "free": False, "query": "", "searched": False},
        "typed_destination": "Copenhagen",
    }
    assert COLLECTOR._remaining_travel_actions(case, {"success": False, "observed": observed}) == [
        COLLECTOR.TeachingAction("Find stays")
    ]
    observed["result"].update(query="Copenhagen", searched=True, free=True)
    assert COLLECTOR._remaining_travel_actions(case, {"success": False, "observed": observed}) == [
        COLLECTOR.TeachingAction("Free cancellation")
    ]
    observed["result"]["free"] = False
    assert COLLECTOR._remaining_travel_actions(case, {"success": False, "observed": observed}) == [
        COLLECTOR.TeachingAction("View The Glasshouse")
    ]
    observed["result"]["hash"] = "#the-glasshouse"
    assert COLLECTOR._remaining_travel_actions(case, {"success": True, "observed": observed}) == []


@pytest.mark.parametrize(
    "options",
    [
        {"split": "development"},
        {"split": "calibration"},
        {"repeats": 5},
        {"travel_recovery_mode": "partial"},
        {"task_ids": ["completion_toggle"]},
    ],
)
def test_broad_collection_rejects_unsupported_modes_before_source_loading(tmp_path, options):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="Travel goal variants"):
        COLLECTOR.collect(
            upstream_path="unused", output_dir=output, travel_goal_variants=True, **options
        )
    assert not output.exists()


def test_direct_open_cannot_be_mislabeled_as_negative_recovery(cases):
    with pytest.raises(ValueError, match="full travel goal"):
        COLLECTOR.collect_episode(
            None,
            None,
            None,
            cases[0],
            rng=random.Random(0),
            split="train",
            episode_id="unused",
            rows=[],
            wires=[],
            dynamic_travel=True,
            recover_early_open=True,
            recovery_initialization=(),
        )


def upstream():
    if os.environ.get("JEV_BROWSER_TESTS") != "1":
        pytest.skip("Set JEV_BROWSER_TESTS=1 for guarded Chromium supervision checks")
    pytest.importorskip("playwright.sync_api")
    return Path(os.environ["JEV_UPSTREAM_PATH"])


@pytest.mark.integration
def test_real_broad_collection_freezes_inventory_and_keeps_direct_goals_minimal(tmp_path):
    from s1.evaluation.datasets import load_cases

    path = upstream()
    receipt = COLLECTOR.collect(
        upstream_path=path,
        output_dir=tmp_path / "broad",
        repeats=1,
        seed=91027,
        task_ids=["upstream_travel"],
        travel_goal_variants=True,
    )
    assert receipt["stage"] == "complete"
    assert receipt["training_goal_input_verification"]["success"]
    assert receipt["source_verification"]["success"] and receipt["upstream_verification"]["success"]
    assert len(receipt["episodes"]) == 18
    assert all(episode["verified_dom"]["success"] for episode in receipt["episodes"])
    output = tmp_path / "broad"
    inventory = json.loads((output / "training_goal_inventory.json").read_text())
    assert len(inventory["cases"]) == 15
    rows = load_cases(output / "cases.jsonl")
    wires = [json.loads(line) for line in (output / "wire_requests.jsonl").read_text().splitlines()]
    assert len(rows) == len(wires) == receipt["training_cases"]
    for i in (0, 5, 10):
        episode = next(
            e for e in receipt["episodes"] if e["episode_id"].startswith(f"train-91027-{i:04d}-")
        )
        assert episode["episode_id"].endswith("-fresh") and episode["steps"] == 2
        chosen = [row.gold["operation"] for row in rows if row.id.startswith(episode["episode_id"])]
        assert chosen == ["CLICK", "DONE"]
    assert sum(row.gold["operation"] == "DONE" for row in rows) == 18
    assert all(
        "expected" not in wire["payload"] and "gold" not in wire["payload"] for wire in wires
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "index,initialization",
    [
        (
            9,
            (
                COLLECTOR.TeachingAction("Stay category → Design", "select"),
                COLLECTOR.TeachingAction("Free cancellation"),
                COLLECTOR.TeachingAction("Destination", "fill", "Copenhagen"),
            ),
        ),
        (
            14,
            (
                COLLECTOR.TeachingAction("Stay category → Nature", "select"),
                COLLECTOR.TeachingAction("Free cancellation"),
            ),
        ),
        (4, 3),
        (14, 3),
    ],
)
def test_real_recovery_repairs_actual_off_state_and_submits_typed_query(
    monkeypatch, index, initialization
):
    from playwright.sync_api import sync_playwright

    path = upstream()
    case = COLLECTOR.make_training_travel_cases(path)[index]
    if isinstance(initialization, int):
        initialization = COLLECTOR._training_recovery_actions(case, initialization)
    monkeypatch.setenv("TYPESAFE_API_KEY", "scripted-training-test-placeholder")
    with COLLECTOR.fixture_server([case]) as base, sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        try:
            with COLLECTOR.load_upstream(path, COLLECTOR.PlaywrightCDP(browser)) as (
                agent_module,
                model_module,
            ):
                agent = agent_module.Agent(base + case.url_suffix, case.goal, screenshots=False)
                try:
                    rows, wires = [], []
                    result = COLLECTOR.collect_episode(
                        agent,
                        agent_module,
                        model_module,
                        case,
                        rng=random.Random(0),
                        split="train",
                        episode_id="train-91028-0000-upstream_travel-recovery",
                        rows=rows,
                        wires=wires,
                        dynamic_travel=True,
                        recover_early_open=True,
                        recovery_initialization=initialization,
                    )
                    assert result["verified_dom"]["success"]
                    assert not result["initial_verified_dom"]["success"]
                    initial = result["initial_verified_dom"]["observed"]["result"]
                    assert initial["free"] is not case.expected["free"]
                    assert (
                        result["verified_dom"]["observed"]["result"]["free"]
                        is case.expected["free"]
                    )
                    assert (
                        result["verified_dom"]["observed"]["result"]["query"]
                        == case.expected["query"]
                    )
                    assert result["verified_dom"]["observed"]["result"]["searched"] is True
                    assert rows[-1].gold["operation"] == "DONE"
                    assert sum(row.gold["operation"] == "DONE" for row in rows) == 1
                    history = agent.state["history"]
                    assert any(
                        item["action"] == "Free cancellation"
                        for item in history[len(initialization) :]
                    )
                    applied_already = (
                        initial["query"] == case.expected["query"] and initial["searched"]
                    )
                    repaired = history[len(result["initialization_history"]) :]
                    assert (
                        any(item["action"] == "Find stays" for item in repaired)
                        is not applied_already
                    )
                    if index == 14:
                        assert not result["initial_verified_dom"]["observed"]["result"]["hash"]
                        assert all(
                            item["action"] != "View Serra Lodge"
                            for item in result["initialization_history"]
                        )
                    submitted_rows = []
                    for wire in wires:
                        if wire["teacher_choice"]["operation"] != "CLICK":
                            continue
                        target = wire["payload"]["questions"]["click_target"]["criteria"][
                            wire["teacher_choice"]["target"]
                        ]
                        if target["element"].endswith("Find stays"):
                            submitted_rows.append(wire)
                    assert bool(submitted_rows) is not applied_already
                    for wire in submitted_rows:
                        field = next(
                            element
                            for element in wire["payload"]["state"]["elements"]
                            if element["role"] == "searchbox"
                        )
                        assert field["value"] == case.expected["query"]
                        assert wire["valid_next_targets"] == {
                            "CLICK": [wire["teacher_choice"]["target"]]
                        }
                finally:
                    agent.close()
        finally:
            browser.close()


@pytest.mark.integration
def test_positive_done_requires_fresh_dom_after_the_previous_action(monkeypatch):
    from playwright.sync_api import sync_playwright

    path = upstream()
    case = COLLECTOR.make_training_travel_cases(path)[0]
    monkeypatch.setenv("TYPESAFE_API_KEY", "scripted-training-test-placeholder")
    actual_verify = COLLECTOR.verify_fixture
    with COLLECTOR.fixture_server([case]) as base, sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        try:
            with COLLECTOR.load_upstream(path, COLLECTOR.PlaywrightCDP(browser)) as (
                agent_module,
                model_module,
            ):
                agent = agent_module.Agent(base + case.url_suffix, case.goal, screenshots=False)
                invalidated = []

                def verify_then_change(real_browser, fixture):
                    result = actual_verify(real_browser, fixture)
                    if result["success"] and not invalidated:
                        page = agent.state["page"]
                        back = COLLECTOR._available(page, COLLECTOR.TeachingAction("← All stays"))
                        real_browser.act(back, page)
                        invalidated.append(True)
                    return result

                monkeypatch.setattr(COLLECTOR, "verify_fixture", verify_then_change)
                rows, wires = [], []
                try:
                    with pytest.raises(ValueError, match="fresh satisfied DOM"):
                        COLLECTOR.collect_episode(
                            agent,
                            agent_module,
                            model_module,
                            case,
                            rng=random.Random(0),
                            split="train",
                            episode_id="train-91029-0000-upstream_travel-fresh",
                            rows=rows,
                            wires=wires,
                            dynamic_travel=True,
                        )
                    assert invalidated and all(row.gold["operation"] != "DONE" for row in rows)
                finally:
                    agent.close()
        finally:
            browser.close()
