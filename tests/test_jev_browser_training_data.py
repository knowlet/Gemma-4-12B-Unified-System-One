"""Supervised development trajectories stay outside benchmark policy inputs."""

import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1.browser_training_counterfactuals import (
    GOAL_NORMALIZATION,
    make_counterfactual_travel_episodes,
    normalized_goal_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    SPEC = importlib.util.spec_from_file_location(
        "jev_browser_training_data", ROOT / "scripts/jev_browser_training_data.py"
    )
    COLLECTOR = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = COLLECTOR
    SPEC.loader.exec_module(COLLECTOR)
finally:
    sys.path.pop(0)


def test_supervision_excludes_unused_target_heads_and_preserves_original_inputs():
    payload = {
        "model": "development-oracle",
        "state": {"page": {"text": "Search pending"}, "elements": []},
        "questions": {
            "operation": {
                "instructions": {"goal": "Find item"},
                "criteria": {"CLICK": "Click", "SELECT": "Select", "DONE": "Finished"},
            },
            "click_target": {
                "instructions": {"goal": "Find item", "operation": "CLICK"},
                "criteria": {"1": {"element": "Enable filter"}, "2": {"element": "Open result"}},
            },
            "select_target": {
                "instructions": {"goal": "Find item", "operation": "SELECT"},
                "criteria": {"3:1": "Choose region", "3:2": "Other region"},
            },
        },
    }
    before = json.loads(json.dumps(payload))
    row = COLLECTOR.training_case(
        payload,
        case_id="train-step",
        group_id="train-episode",
        task_id="filter_open",
        split="train",
        selected={"operation": "CLICK", "target": "1"},
        valid_targets={"CLICK": ["1"], "SELECT": ["3:1"]},
    )
    assert payload == before
    assert row.request.state == payload["state"]
    assert [question.id for question in row.request.questions] == ["operation", "click_target"]
    assert row.gold == {"operation": "CLICK", "click_target": "1"}
    assert row.soft_gold["operation"] == {"CLICK": 0.5, "SELECT": 0.5, "DONE": 0.0}
    assert row.soft_gold["click_target"] == {"1": 1.0, "2": 0.0}
    assert (
        json.loads(row.request.questions[0].instructions)
        == payload["questions"]["operation"]["instructions"]
    )
    serialized = row.request.model_dump_json()
    assert "gold" not in serialized and "train-episode" not in serialized


def test_collector_rejects_holdout_and_invalid_configuration_before_writing(tmp_path):
    for split, repeats in [("holdout", 1), ("train", 0), ("test", 1)]:
        with pytest.raises(ValueError, match="explicit fitting"):
            COLLECTOR.collect(
                upstream_path="unused", output_dir=tmp_path / split, split=split, repeats=repeats
            )
        assert not (tmp_path / split).exists()


@pytest.mark.parametrize(
    "options",
    [
        {"task_ids": []},
        {"task_ids": ["upstream_travel", "upstream_travel"]},
        {"task_ids": ["not_a_task"]},
        {"task_ids": ["upstream_travel"], "split": "development"},
        {"travel_recovery_mode": "unknown"},
        {"travel_recovery_mode": "partial", "split": "development"},
    ],
)
def test_collector_rejects_invalid_partial_recovery_options_before_sources(tmp_path, options):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        COLLECTOR.collect(upstream_path="unused", output_dir=output, **options)
    assert not output.exists()


def isolated_collection(monkeypatch, tmp_path, *, during_generation=None):
    """Exercise actual source capture/finalization without any browser or model."""
    source = tmp_path / "caller" / "jev_browser_training_data.py"
    source.parent.mkdir()
    source.write_bytes(Path(COLLECTOR.__file__).read_bytes())
    monkeypatch.setattr(COLLECTOR, "__file__", str(source))
    monkeypatch.setattr(COLLECTOR, "upstream_manifest", lambda _path: {"revision": "unit"})

    def cases(*_args):
        assert (
            tmp_path / "run" / "sources" / "scripts" / source.name
        ).read_bytes() == source.read_bytes()
        if during_generation:
            during_generation(source)
        return []

    monkeypatch.setattr(COLLECTOR, "make_cases", cases)
    # The capture must name the actual imported harness file rather than this
    # test's monkeypatched generator code filename.
    cases.__code__ = cases.__code__.replace(
        co_filename=str(ROOT / "scripts" / "jev_browser_benchmark.py")
    )
    monkeypatch.setattr(COLLECTOR, "fixture_server", lambda _cases: nullcontext("http://unit"))
    monkeypatch.setattr(COLLECTOR, "load_upstream", lambda *_args: nullcontext((None, None)))
    browser = SimpleNamespace(close=lambda: None)
    runtime = SimpleNamespace(chromium=SimpleNamespace(launch=lambda **_kwargs: browser))
    monkeypatch.setitem(
        sys.modules,
        "playwright.sync_api",
        SimpleNamespace(sync_playwright=lambda: nullcontext(runtime)),
    )
    return source, tmp_path / "run"


def test_collector_source_drift_writes_failed_receipt_and_raises(monkeypatch, tmp_path):
    source, output = isolated_collection(
        monkeypatch,
        tmp_path,
        during_generation=lambda path: path.write_bytes(path.read_bytes() + b"\n# changed\n"),
    )
    monkeypatch.setenv("TYPESAFE_API_KEY", "unit-key-must-be-restored")
    with pytest.raises(RuntimeError, match="runtime source changed"):
        COLLECTOR.collect(upstream_path="unused", output_dir=output, repeats=1)
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert receipt["stage"] == "failed"
    assert (
        receipt["source_preservation_timing"] == "before fixture generation and browser collection"
    )
    assert receipt["source_verification"]["changed"] == ["scripts/jev_browser_training_data.py"]
    assert (output / "sources" / "scripts" / source.name).read_bytes() != source.read_bytes()
    assert (output / "cases.jsonl").exists() and (output / "wire_requests.jsonl").exists()
    assert os.environ["TYPESAFE_API_KEY"] == "unit-key-must-be-restored"
    assert "unit-key-must-be-restored" not in (output / "receipt.json").read_text()


def test_collector_keeps_primary_failure_when_source_also_changes(monkeypatch, tmp_path):
    primary = LookupError("trajectory failed first")

    def fail(path):
        path.write_bytes(path.read_bytes() + b"\n# changed\n")
        raise primary

    _source, output = isolated_collection(monkeypatch, tmp_path, during_generation=fail)
    with pytest.raises(LookupError) as caught:
        COLLECTOR.collect(upstream_path="unused", output_dir=output, repeats=1)
    assert caught.value is primary
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert receipt["error"]["type"] == "LookupError"
    assert receipt["stage"] == "failed" and not receipt["source_verification"]["success"]


def counterfactual_exclusions(tmp_path, *, values=None):
    path = tmp_path / "excluded-goal-hashes.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "normalization": GOAL_NORMALIZATION,
                "goal_sha256": values or [normalized_goal_sha256("Synthetic excluded goal")],
            }
        )
    )
    return path


def counterfactual_plan(seed=20261006):
    return make_counterfactual_travel_episodes(
        seed=seed, excluded_goal_hashes=[normalized_goal_sha256("Synthetic excluded goal")]
    )


def test_counterfactual_plan_is_complete_balanced_feasible_and_paired():
    specs = counterfactual_plan()
    assert len(specs) == len({spec.case_id for spec in specs}) == 32
    assert len({spec.goal for spec in specs}) == 16
    assert Counter(spec.family for spec in specs) == {0: 8, 1: 8, 2: 8, 3: 8}
    assert Counter(spec.query_required for spec in specs) == {False: 16, True: 16}
    assert Counter(spec.free for spec in specs) == {False: 16, True: 16}
    assert {spec.property.name for spec in specs} == {"Casa Flora", "The Glasshouse", "Serra Lodge"}
    assert all(not spec.free or spec.property.free_available for spec in specs)
    pairs = defaultdict(list)
    for spec in specs:
        pairs[spec.pair_id].append(spec)
        if not spec.query_required:
            assert "query" not in spec.expected and "searched" not in spec.expected
        else:
            assert spec.expected["query"] == spec.property.city
            assert spec.expected["searched"] is True
        if spec.start == "detail" and not spec.free:
            assert spec.initialization == (
                ("Stay category → " + spec.property.category, "select", None),
                ("View " + spec.property.name, "click", None),
            )
    assert len(pairs) == 16
    for pair in pairs.values():
        assert {spec.query_required for spec in pair} == {True, False}
        assert pair[0].initialization == pair[1].initialization
        assert pair[0].property == pair[1].property
        assert pair[0].free == pair[1].free


def test_counterfactual_seed_and_exclusions_are_explicit_and_reproducible():
    assert counterfactual_plan() == counterfactual_plan()
    assert [s.case_id for s in counterfactual_plan()] != [
        s.case_id for s in counterfactual_plan(20261007)
    ]
    excluded_goal = counterfactual_plan()[0].goal
    assert normalized_goal_sha256("  " + excluded_goal.replace(" ", "\n") + "  ") == (
        normalized_goal_sha256(excluded_goal)
    )
    with pytest.raises(ValueError, match="overlaps"):
        make_counterfactual_travel_episodes(
            seed=20261006, excluded_goal_hashes=[normalized_goal_sha256(excluded_goal)]
        )


def test_counterfactual_wording_varies_search_scope_and_local_cancellation_negation():
    specs = counterfactual_plan()
    required_by_family = {
        spec.family: spec for spec in specs if spec.query_required and not spec.free
    }
    assert "using the destination search, without applying the free-cancellation filter" in (
        required_by_family[0].goal
    )
    assert "Use destination search to locate" in required_by_family[1].goal
    assert "Fill Destination" in required_by_family[2].goal
    assert "apply Find stays" in required_by_family[2].goal
    assert "with the destination controls" in required_by_family[3].goal
    assert "when the search is applied" in required_by_family[3].goal
    for spec in specs:
        if not spec.query_required and spec.family in (0, 1):
            assert "search" not in spec.goal.casefold()
            assert "destination" not in spec.goal.casefold()
        if not spec.query_required:
            assert "query" not in spec.expected and "searched" not in spec.expected


@pytest.mark.parametrize("seed", [None, True, -1, "20261006"])
def test_counterfactual_factory_rejects_invalid_seeds(seed):
    with pytest.raises(ValueError, match="explicit nonnegative"):
        counterfactual_plan(seed)


@pytest.mark.parametrize("hashes", [[], ["not-a-hash"], ["A" * 64], ["0" * 64, "0" * 64]])
def test_counterfactual_factory_rejects_invalid_exclusion_hashes(hashes):
    with pytest.raises(ValueError, match="exclusions"):
        make_counterfactual_travel_episodes(seed=20261006, excluded_goal_hashes=hashes)


@pytest.mark.parametrize(
    "overrides",
    [
        {"split": "development"},
        {"repeats": 2},
        {"counterfactual_seed": None},
        {"counterfactual_seed": 20261001},
        {"counterfactual_seed": True},
        {"counterfactual_exclusions": None},
        {"travel_goal_variants": True},
        {"travel_recovery_mode": "partial"},
        {"task_ids": ["filter_open"]},
        {"travel_counterfactual_goals": False},
    ],
)
def test_counterfactual_cli_configuration_fails_before_sources(tmp_path, overrides):
    options = {
        "travel_counterfactual_goals": True,
        "counterfactual_seed": 20261006,
        "counterfactual_exclusions": "unread-exclusion-input",
        "repeats": 1,
    }
    options.update(overrides)
    output = tmp_path / "never-created"
    with pytest.raises(ValueError):
        COLLECTOR.collect(upstream_path="unused", output_dir=output, **options)
    assert not output.exists()


def test_counterfactual_teacher_changes_with_goal_scope_on_the_same_actual_state():
    specs = [s for s in counterfactual_plan() if s.property.name == "Serra Lodge"]
    required = next(s for s in specs if s.query_required)
    optional = next(s for s in specs if not s.query_required)
    observed = {
        "result": {"hash": "", "category": "Nature", "free": False, "query": "", "searched": False},
        "typed_destination": "",
    }

    def case(spec):
        return SimpleNamespace(expected=spec.expected)

    first_required = COLLECTOR._remaining_travel_actions(
        case(required), {"success": False, "observed": observed}
    )
    first_optional = COLLECTOR._remaining_travel_actions(
        case(optional), {"success": False, "observed": observed}
    )
    assert first_required == [COLLECTOR.TeachingAction("Destination", "fill", "Lisbon")]
    assert first_optional == [COLLECTOR.TeachingAction("View Serra Lodge")]
    # Typed text is not an applied search; the same teacher must still submit.
    observed["typed_destination"] = "Lisbon"
    assert COLLECTOR._remaining_travel_actions(
        case(required), {"success": False, "observed": observed}
    ) == [COLLECTOR.TeachingAction("Find stays")]
    observed["result"]["hash"] = "#serra-lodge"
    assert COLLECTOR._remaining_travel_actions(
        case(required), {"success": False, "observed": observed}
    ) == [COLLECTOR.TeachingAction("← All stays")]
    assert (
        COLLECTOR._remaining_travel_actions(case(optional), {"success": True, "observed": observed})
        == []
    )
    # Optional search also accepts an already-applied destination: its profile
    # imposes no query/searched equality, rather than labeling such a state wrong.
    observed["result"].update(query="Lisbon", searched=True)
    assert all(observed["result"].get(k) == v for k, v in optional.expected.items() if k != "name")


@pytest.mark.parametrize(
    "query_required,operation,target", [(True, "CLICK", "back"), (False, "DONE", None)]
)
def test_same_guarded_early_open_teaches_back_or_done_without_policy_oracle_fields(
    monkeypatch, query_required, operation, target
):
    """Exercise the real initialization/teacher closure with a tiny fake browser."""
    import copy
    import random

    spec = next(
        s
        for s in counterfactual_plan()
        if s.family == 0
        and s.start == "detail"
        and s.query_required == query_required
        and s.property.name == "Serra Lodge"
    )
    case = SimpleNamespace(
        task_id="upstream_travel", expected=spec.expected, html="training-owned", goal=spec.goal
    )
    actions = [
        {"id": "select", "kind": "select", "label": "Stay category → Nature"},
        {"id": "view", "kind": "click", "label": "View Serra Lodge"},
        {"id": "back", "kind": "click", "label": "← All stays"},
    ]
    agent = SimpleNamespace(
        state={"page": {"fingerprint": "initial", "actions": actions}, "history": []}
    )
    actual_actions = []

    def act(action, _before, *, text=None):
        actual_actions.append(action["label"])
        agent.state["page"] = {"fingerprint": str(len(actual_actions)), "actions": actions}

    agent.browser = SimpleNamespace(
        act=act,
        observe=lambda **_kw: copy.deepcopy(agent.state["page"]),
        evaluate=lambda _expression: "<html>identical opened Serra Lodge</html>",
    )
    observation = {
        "result": {
            "hash": "#serra-lodge",
            "name": "Serra Lodge",
            "category": "Nature",
            "free": False,
            "query": "",
            "searched": False,
        },
        "typed_destination": None,
    }
    monkeypatch.setattr(
        COLLECTOR,
        "verify_fixture",
        lambda *_a: {"success": not query_required, "observed": observation},
    )
    module = SimpleNamespace(
        action_space=lambda _actions: (None, {"CLICK": {"back": actions[2]}}, None)
    )
    helper_module = SimpleNamespace()
    payloads = []

    def row(payload, **options):
        payloads.append(copy.deepcopy(payload))
        return SimpleNamespace(id=options["case_id"])

    monkeypatch.setattr(COLLECTOR, "training_case", row)

    class FirstPolicyChoice(Exception):
        pass

    response = []

    def command(_command):
        response.append(
            module.post_json(
                "https://api.typesafe.ai/v1/systemone",
                None,
                {
                    "model": "scripted-development-teacher",
                    "state": {
                        "page": copy.deepcopy(agent.state["page"]),
                        "history": copy.deepcopy(agent.state["history"]),
                    },
                    "questions": {
                        "operation": {
                            "instructions": {"goal": case.goal},
                            "criteria": {"CLICK": "Click", "DONE": "Finish"},
                        },
                        "click_target": {
                            "instructions": {"goal": case.goal},
                            "criteria": {"back": "← All stays"},
                        },
                    },
                },
            )
        )
        raise FirstPolicyChoice

    agent.command = command
    rows, wires = [], []
    with pytest.raises(FirstPolicyChoice):
        COLLECTOR.collect_episode(
            agent,
            helper_module,
            module,
            case,
            rng=random.Random(0),
            split="train",
            episode_id="unit-counterfactual",
            rows=rows,
            wires=wires,
            dynamic_travel=True,
            counterfactual_initialization=tuple(
                COLLECTOR.TeachingAction(*a) for a in spec.initialization
            ),
        )
    assert actual_actions == ["Stay category → Nature", "View Serra Lodge"]
    assert [h["action"] for h in agent.state["history"]] == actual_actions
    assert wires[0]["teacher_choice"] == {"operation": operation, "target": target}
    assert response[0]["answers"]["operation"]["choice"] == operation
    assert set(payloads[0]) == {"model", "state", "questions"}
    assert "expected" not in json.dumps(payloads[0])
    assert "teacher_choice" not in json.dumps(payloads[0])


def mock_counterfactual_collection(monkeypatch, tmp_path, *, drift=None, mismatch=False):
    """Validate scheduling/preservation with fakes: no Chromium, trajectories or models."""
    _source, output = isolated_collection(monkeypatch, tmp_path)
    excluded = counterfactual_exclusions(tmp_path)
    original = COLLECTOR.FixtureCase(
        case_id="original-training-page",
        task_id="upstream_travel",
        goal="unused original goal",
        html="<html>Owned exposed training HTML</html>",
        expected={},
        query_string="scenario=travel",
        source={"owned": True},
    )
    monkeypatch.setattr(COLLECTOR, "make_upstream_travel_case", lambda *_args: original)

    def forbidden_generator(*_args):
        raise AssertionError("Counterfactual mode called the evaluation-goal generator")

    forbidden_generator.__code__ = forbidden_generator.__code__.replace(
        co_filename=str(ROOT / "scripts" / "jev_browser_benchmark.py")
    )
    monkeypatch.setattr(COLLECTOR, "make_cases", forbidden_generator)
    agent_urls = []

    def agent(url, *_args, **_kwargs):
        agent_urls.append(url)
        return SimpleNamespace(url=url, close=lambda: None)

    served = []

    def serve(cases):
        served.extend(cases)
        return nullcontext("http://unit")

    monkeypatch.setattr(COLLECTOR, "fixture_server", serve)
    monkeypatch.setattr(
        COLLECTOR,
        "load_upstream",
        lambda *_args: nullcontext((SimpleNamespace(Agent=agent), None)),
    )
    calls = []

    def episode(_agent, _agent_module, _model_module, case, **options):
        inventory = json.loads((output / "training_goal_inventory.json").read_bytes())
        assert inventory["seed"] == 20261006
        assert inventory["episode_count"] == 32
        assert len(inventory["cases"]) == 32
        assert (output / "counterfactual_exclusions.json").read_bytes() == excluded.read_bytes()
        assert all("initialization" in entry for entry in inventory["cases"])
        assert len(served) == 1
        assert {entry["public_url_suffix"] for entry in inventory["cases"]} == {
            served[0].url_suffix
        }
        assert set(agent_urls) == {"http://unit" + served[0].url_suffix}
        assert all(
            token not in _agent.url
            for token in ("query0", "query1", "family", "listing", "detail", "free")
        )
        assert len(served[0].case_id) == 24
        assert options["dynamic_travel"] is True and not options["recover_early_open"]
        init = options["counterfactual_initialization"]
        history = [{"action": a.label, "kind": a.kind, "text": a.text} for a in init]
        calls.append((case, options))
        if drift and len(calls) == 32:
            target = excluded if drift == "original" else output / "counterfactual_exclusions.json"
            target.write_bytes(target.read_bytes() + b" ")
        result = {
            key: value for key, value in case.expected.items() if key not in {"query", "searched"}
        }
        result.update(query="", searched=False, hash="")
        options["wires"].append(
            {
                "payload": {
                    "model": "unit-teacher",
                    "state": {"page": {"url": _agent.url}, "recent_actions": history},
                    "questions": {
                        "operation": {
                            "instructions": {"goal": case.goal},
                            "criteria": {"DONE": "Finish"},
                        }
                    },
                }
            }
        )
        return {
            "episode_id": options["episode_id"],
            "initialization_history": history,
            "initialization_html_sha256": "drift" if mismatch and case.needs_text else "same-html",
            "initial_verified_dom": {"observed": {"result": result, "typed_destination": ""}},
        }

    monkeypatch.setattr(COLLECTOR, "collect_episode", episode)
    return output, excluded, calls


def test_counterfactual_only_schedule_preserves_inputs_before_mock_collection(
    monkeypatch, tmp_path
):
    output, excluded, calls = mock_counterfactual_collection(monkeypatch, tmp_path)
    receipt = COLLECTOR.collect(
        upstream_path="unused",
        output_dir=output,
        repeats=1,
        travel_counterfactual_goals=True,
        counterfactual_seed=20261006,
        counterfactual_exclusions=excluded,
    )
    assert receipt["stage"] == "complete" and len(calls) == 32
    assert (
        receipt["seed"] == receipt["effective_seed"] == receipt["counterfactual_seed"] == 20261006
    )
    assert receipt["ordinary_seed"] == 20261001
    assert receipt["counterfactual_pair_verification"]["pairs"] == 16
    assert receipt["counterfactual_public_fixture"]["served_pages"] == 1
    assert all("first_wire_goal_comparison_sha256" in e for e in receipt["episodes"])
    assert receipt["counterfactual_exclusion_verification"]["success"]
    assert receipt["training_goal_input_verification"]["success"]
    assert {case.html for case, _ in calls} == {"<html>Owned exposed training HTML</html>"}
    assert len({options["episode_id"] for _, options in calls}) == 32
    assert all("20261006" in options["episode_id"] for _, options in calls)
    assert "s1/browser_training_counterfactuals.py" in receipt["source_manifest"]["files"]


def test_counterfactual_payload_comparison_replaces_only_goals_and_retains_candidate_order():
    import copy

    payload = {
        "model": "native",
        "state": {"page": {"url": "http://unit/tasks/opaque"}, "recent_actions": []},
        "questions": {
            "operation": {
                "instructions": {"goal": "First goal", "rules": "Unchanged"},
                "criteria": {"CLICK": "Click", "DONE": "Finish"},
            },
            "click_target": {
                "instructions": {"goal": "First goal"},
                "criteria": {"2": "Second", "1": "First"},
            },
        },
    }
    original = copy.deepcopy(payload)
    baseline = COLLECTOR._counterfactual_comparison_bytes(payload, "First goal")
    assert payload == original
    changed = copy.deepcopy(payload)
    for question in changed["questions"].values():
        question["instructions"]["goal"] = "Second goal"
    assert COLLECTOR._counterfactual_comparison_bytes(changed, "Second goal") == baseline
    changed["state"]["page"]["url"] += "-query1"
    assert COLLECTOR._counterfactual_comparison_bytes(changed, "Second goal") != baseline
    changed = copy.deepcopy(payload)
    changed["questions"]["click_target"]["criteria"] = {"1": "First", "2": "Second"}
    assert COLLECTOR._counterfactual_comparison_bytes(changed, "First goal") != baseline


@pytest.mark.parametrize("drift", ["original", "frozen"])
def test_counterfactual_exclusion_drift_fails_and_preserves_receipt(monkeypatch, tmp_path, drift):
    output, excluded, _calls = mock_counterfactual_collection(monkeypatch, tmp_path, drift=drift)
    with pytest.raises(RuntimeError, match="exclusion input changed"):
        COLLECTOR.collect(
            upstream_path="unused",
            output_dir=output,
            repeats=1,
            travel_counterfactual_goals=True,
            counterfactual_seed=20261006,
            counterfactual_exclusions=excluded,
        )
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert receipt["stage"] == "failed"
    assert receipt["counterfactual_exclusion_verification"]["success"] is False


def test_counterfactual_pair_state_drift_fails_without_silent_pruning(monkeypatch, tmp_path):
    output, excluded, _calls = mock_counterfactual_collection(monkeypatch, tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="pair initialization differs"):
        COLLECTOR.collect(
            upstream_path="unused",
            output_dir=output,
            repeats=1,
            travel_counterfactual_goals=True,
            counterfactual_seed=20261006,
            counterfactual_exclusions=excluded,
        )
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert receipt["stage"] == "failed" and receipt["episodes"]


@pytest.mark.integration
@pytest.mark.parametrize("split,expected_episodes", [("train", 11), ("calibration", 9)])
def test_real_training_trajectories_use_guarded_actions_and_fresh_final_dom(
    tmp_path, split, expected_episodes
):
    if os.environ.get("JEV_BROWSER_TESTS") != "1":
        pytest.skip("Set JEV_BROWSER_TESTS=1 for guarded Chromium collection")
    pytest.importorskip("playwright.sync_api")
    from s1.evaluation.datasets import load_cases

    upstream = Path(os.environ.get("JEV_UPSTREAM_PATH", "/tmp/s1-jev-ultrafast"))
    before = COLLECTOR.upstream_manifest(upstream)
    output = tmp_path / split
    receipt = COLLECTOR.collect(
        upstream_path=upstream, output_dir=output, split=split, seed=91023, repeats=1
    )
    assert receipt["stage"] == "complete"
    assert receipt["source_verification"]["success"]
    assert receipt["upstream_verification"]["success"]
    assert (
        receipt["source_manifest"]["files"]["scripts/jev_browser_training_data.py"]["sha256"]
        == receipt["collector_sha256"]
    )
    assert len(receipt["episodes"]) == expected_episodes
    assert all(episode["verified_dom"]["success"] for episode in receipt["episodes"])
    assert COLLECTOR.upstream_manifest(upstream) == before
    rows = load_cases(output / "cases.jsonl")
    wires = [json.loads(line) for line in (output / "wire_requests.jsonl").read_text().splitlines()]
    assert len(rows) == len(wires) == receipt["training_cases"]
    assert all(row.split == split for row in rows)
    assert all(
        set(row.gold) <= {"operation", row.gold["operation"].lower() + "_target"} for row in rows
    )
    assert all(
        row.request.state == wire["payload"]["state"] for row, wire in zip(rows, wires, strict=True)
    )
    if split != "train":
        assert all(episode["task_id"] != "upstream_travel" for episode in receipt["episodes"])
    else:
        recovery = next(
            episode
            for episode in receipt["episodes"]
            if episode["initialization"] == "guarded early-open recovery"
        )
        first = next(row for row in rows if row.id == recovery["episode_id"] + "-step-000")
        assert first.gold["operation"] == "CLICK"
        assert "Casa Flora" in first.request.state["page"]["title"]
        assert "Destination anywhere" in first.request.state["page"]["text"]
    assert sum(row.gold["operation"] == "DONE" for row in rows) == expected_episodes
    with pytest.raises(FileExistsError):
        COLLECTOR.collect(upstream_path=upstream, output_dir=output, split=split, repeats=1)


@pytest.mark.integration
def test_partial_travel_recovery_collects_actual_missing_query_and_never_positive_false_done(
    tmp_path,
):
    if os.environ.get("JEV_BROWSER_TESTS") != "1":
        pytest.skip("Set JEV_BROWSER_TESTS=1 for guarded Chromium collection")
    pytest.importorskip("playwright.sync_api")
    from s1.evaluation.datasets import load_cases

    upstream = Path(os.environ.get("JEV_UPSTREAM_PATH", "/tmp/s1-jev-ultrafast"))
    output = tmp_path / "partial"
    receipt = COLLECTOR.collect(
        upstream_path=upstream,
        output_dir=output,
        seed=91024,
        repeats=15,
        task_ids=["upstream_travel"],
        travel_recovery_mode="partial",
    )
    assert receipt["stage"] == "complete" and receipt["source_verification"]["success"]
    assert receipt["upstream_verification"]["success"]
    assert len(receipt["episodes"]) == 30
    assert {episode["task_id"] for episode in receipt["episodes"]} == {"upstream_travel"}
    assert all(episode["verified_dom"]["success"] for episode in receipt["episodes"])
    rows = load_cases(output / "cases.jsonl")
    recovery = [e for e in receipt["episodes"] if e["episode_id"].endswith("-recovery")]
    configurations = {
        (tuple(e.get("initial_prerequisite_indices", [])), e.get("initial_search_submitted", False))
        for e in recovery
    }
    assert len(configurations) == 15
    assert ((0, 1, 2), True) not in configurations
    # Reproduce the actual learned policy failure: Design/free/submitted, no Lisbon.
    missing_query = next(
        e
        for e in recovery
        if e.get("initial_prerequisite_indices") == [1, 2] and e["initial_search_submitted"]
    )
    assert not missing_query["initial_verified_dom"]["success"]
    assert set(missing_query["initial_verified_dom"]["mismatches"]) == {"query"}
    first = next(r for r in rows if r.id == missing_query["episode_id"] + "-step-000")
    assert first.gold["operation"] == "CLICK"
    assert "Destination anywhere" in first.request.state["page"]["text"]
    assert "Design" in first.request.state["page"]["text"]
    assert "Free cancellation enabled" in first.request.state["page"]["text"]
    assert all(
        row.gold["operation"] == "CLICK" for row in rows if row.id.endswith("recovery-step-000")
    )
    assert sum(row.gold["operation"] == "DONE" for row in rows) == 30
    # Typing without submission is lost when the native detail/back path rerenders
    # the search. The recovery teacher must type again from that fresh observation.
    typed_unsubmitted = next(
        e
        for e in recovery
        if e.get("initial_prerequisite_indices") == [0, 1, 2] and not e["initial_search_submitted"]
    )
    episode_rows = [r for r in rows if r.id.startswith(typed_unsubmitted["episode_id"] + "-")]
    assert any(r.gold["operation"] == "TYPE_TEXT" for r in episode_rows)
    assert receipt["scope"] == "development-only scripted training collection; no model evaluation"
