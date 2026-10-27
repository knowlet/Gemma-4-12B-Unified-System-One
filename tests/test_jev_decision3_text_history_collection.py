"""Hermetic profile-boundary tests; fabricated inputs are not collection evidence."""

import copy
from pathlib import Path
from types import SimpleNamespace

import jev_decision3_text_history_collection as collection
import jev_decision3_text_history_control_scenarios as scenarios
import pytest

PROFILE = collection.TEXT_HISTORY_CONTROL_PROFILE


def exclusions(path, **updates):
    value = {
        "schema_id": collection.EXCLUSION_SCHEMA,
        "normalization": collection.NORMALIZATION,
        "goal_sha256": ["0" * 64],
        "scenario_ids": [],
        "ancestor_ids": [],
    }
    value.update(updates)
    path.write_bytes(collection._encode(value))
    return path


def prospective_arguments(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    monkeypatch.setattr(collection.benchmark, "upstream_manifest", lambda _: {"test": True})
    monkeypatch.setattr(collection, "_prepare_observation_profile", lambda *_: None)
    monkeypatch.setattr(
        collection.benchmark,
        "fixture_server",
        lambda *_: pytest.fail("prospective plan launched a fixture server"),
    )
    return {
        "output_dir": tmp_path / "unwritten-output",
        "upstream": upstream,
        "collection_id": "text-history-hermetic",
        "seed": 20261027,
        "exclusion_manifest_path": exclusions(tmp_path / "exclusions.json"),
        "scenario_profile": PROFILE,
    }


def test_profile_checks_module_origin_schema_namespace_and_natural_menu_contract(monkeypatch):
    assert collection._scenario_module(PROFILE) is scenarios
    assert collection._scenario_definition(PROFILE) == (
        "jev_decision3_text_history_control_scenarios",
        scenarios.SCHEMA,
        scenarios.NAMESPACE,
        (7,),
    )
    for value in (None, "unknown", "text-history-control-v2"):
        with pytest.raises(ValueError, match="Unknown"):
            collection._scenario_definition(value)
    base = {
        "__file__": scenarios.__file__,
        "SCHEMA": scenarios.SCHEMA,
        "NAMESPACE": scenarios.NAMESPACE,
        "GOAL_NORMALIZATION": scenarios.GOAL_NORMALIZATION,
        "EXPECTED_MENU_COUNTS": (7,),
    }
    for change in (
        {"__file__": "/tmp/foreign-scenarios.py"},
        {"SCHEMA": "foreign"},
        {"NAMESPACE": "foreign"},
        {"GOAL_NORMALIZATION": "foreign"},
        {"EXPECTED_MENU_COUNTS": (6,)},
    ):
        spoof = SimpleNamespace(**(base | change))
        with monkeypatch.context() as scope:
            scope.setattr(collection.importlib, "import_module", lambda _: spoof)
            with pytest.raises(ValueError, match="origin, namespace, schema or menu"):
                collection._scenario_module(PROFILE)
    collection._assert_menu_coverage({7}, PROFILE)
    for counts in (set(), {6}, {7, 8}):
        with pytest.raises(ValueError, match="genuine K7"):
            collection._assert_menu_coverage(counts, PROFILE)


def test_prospective_plan_binds_new_sources_train_population_and_exclusions(tmp_path, monkeypatch):
    args = prospective_arguments(tmp_path, monkeypatch)
    plan = collection.collection_plan(**args)
    assert not args["output_dir"].exists()
    assert plan["split"] == "train" and len(plan["episodes"]) == 33
    assert plan["expected_menu_counts"] == [7]
    assert all(e["scenario"]["split"] == "train" for e in plan["episodes"])
    assert all(e["intended_menu_counts"] == [7] for e in plan["episodes"])
    assert all("observed_menu_counts" not in e for e in plan["episodes"])
    assert plan["model_score"] is None and plan["fit_eligible"] is False
    sources = plan["sources"]
    collector_name = "scripts/jev_decision3_text_history_collection.py"
    fixture_name = "scripts/jev_decision3_text_history_control_scenarios.py"
    assert sources[collector_name]["sha256"] == collection._sha(
        Path(collection.__file__).read_bytes()
    )
    assert sources[fixture_name]["sha256"] == collection._sha(Path(scenarios.__file__).read_bytes())
    assert "scripts/jev_decision3_affordance_collection.py" not in sources
    assert plan["collector_sha256"] == sources[collector_name]["sha256"]
    assert collection._digest(plan) == collection._digest(collection.collection_plan(**args))
    for field, replacement in (
        ("scenario_schema_id", "foreign"),
        ("scenario_namespace", "foreign"),
        ("expected_menu_counts", [6]),
        ("scenario_profile", "unknown"),
    ):
        changed = copy.deepcopy(plan)
        changed[field] = replacement
        with pytest.raises(ValueError):
            collection._plan_scenarios(changed)
    for field, replacement in (
        ("split", "dev"),
        ("schema_id", "foreign"),
        ("expected_menu_counts", [8]),
    ):
        changed = copy.deepcopy(plan["episodes"][0]["scenario"])
        changed[field] = replacement
        with pytest.raises(ValueError):
            collection._validate_scenario_spec(changed, PROFILE)


@pytest.mark.parametrize("field", ["goal_sha256", "scenario_ids", "ancestor_ids"])
def test_excluded_goal_scenario_and_ancestor_are_refused_before_writes(
    tmp_path, monkeypatch, field
):
    args = prospective_arguments(tmp_path, monkeypatch)
    scene = scenarios.scenarios(args["seed"], {"0" * 64})[0]
    value = {
        "goal_sha256": scene["goal_sha256"],
        "scenario_ids": scene["id"],
        "ancestor_ids": scene["ancestor_id"],
    }[field]
    exclusions(args["exclusion_manifest_path"], **{field: [value]})
    with pytest.raises(ValueError):
        collection.collection_plan(**args)
    assert not args["output_dir"].exists()


def test_native_publisher_history_is_ordered_and_keeps_only_original_last_ten():
    history = [
        {
            "action": f"control-{i}",
            "kind": "fill",
            "text": f"value-{i}",
            "page_changed": i % 2 == 0,
            "private_teacher": "never published",
        }
        for i in range(12)
    ]
    page = {"url": "http://127.0.0.1/test", "title": "Example", "text": "Example", "actions": []}
    before = copy.deepcopy(history)
    wire = collection._publisher_wire(
        page, "Example goal", history, {"NEXT_ACTION": "rules", "TARGET": "rules"}
    )
    assert wire["state"]["recent_actions"] == [
        {name: item[name] for name in ("action", "kind", "text", "page_changed")}
        for item in history[-10:]
    ]
    assert "private_teacher" not in collection._encode(wire).decode()
    assert history == before


def test_full_wire_and_history_are_sealed_with_operation_only_loss_mask():
    payload = {
        "model": collection.MODEL_ALIAS,
        "state": {
            "page": {"title": "Invented"},
            "recent_actions": [{"action": "input", "text": "draft", "page_changed": True}],
        },
        "questions": {
            "click_target": {
                "type": "choice",
                "instructions": "Choose",
                "criteria": {"2": {"element": "Button"}},
            },
            "operation": {
                "type": "choice",
                "instructions": "Choose",
                "criteria": {
                    name: name
                    for name in (
                        "CLICK",
                        "TYPE_TEXT",
                        "SELECT",
                        "SCROLL_DOWN",
                        "WAIT",
                        "DONE",
                        "BLOCKED",
                    )
                },
            },
            "select_target": {
                "type": "choice",
                "instructions": "Choose",
                "criteria": {f"3:{i}": {"element": f"Tier {i}"} for i in range(1, 25)},
            },
            "type_text_target": {
                "type": "choice",
                "instructions": "Choose",
                "criteria": {"1": {"element": "Note", "current_value": "draft"}},
            },
        },
    }
    before = collection._encode(payload)
    row = collection.build_native_training_row(
        payload,
        row_id="row",
        scenario_id="scene",
        ancestor_id="ancestor",
        episode_id="episode",
        step=1,
        operation="CLICK",
        target="2",
        proof_refs={"transition": "invented"},
    )
    assert collection._encode(row["payload"]) == before == collection._encode(payload)
    assert row["loss_mask"] == {head: head == "operation" for head in payload["questions"]}
    assert (
        row["fit_eligible"] is False and row["admission_status"] == "pending_independent_admission"
    )
    assert all(
        name not in row["payload"] for name in ("teacher_choice", "gold", "loss_mask", "proof_refs")
    )


def test_bad_execution_pins_fail_before_output_creation(tmp_path, monkeypatch):
    args = prospective_arguments(tmp_path, monkeypatch)
    plan = collection.collection_plan(**args)
    monkeypatch.setattr(collection, "_replan", lambda *_: pytest.fail("unexpected replan"))
    with pytest.raises(ValueError, match="collector SHA256"):
        collection.collect(plan, collector_sha256="0" * 64, plan_sha256=collection._digest(plan))
    with pytest.raises(ValueError, match="prospective plan changed"):
        collection.collect(plan, collector_sha256=plan["collector_sha256"], plan_sha256="0" * 64)
    assert not args["output_dir"].exists()


def test_step_replay_refuses_history_removal_rewrite_or_false_input_effect(monkeypatch):
    """Exercise a hermetic verifier seam, never a real collection or admission."""
    spec = scenarios.scenarios(20261027, {"0" * 64})[0]
    before_history = [{"action": "field", "kind": "fill", "text": "same", "page_changed": True}]
    action = {"id": "native-action", "label": "field", "kind": "fill"}
    descriptor = {
        "operation": "TYPE_TEXT",
        "kind": "fill",
        "label": "field",
        "action_id": None,
        "text": "same",
    }
    before, after = {"events": []}, {"events": []}
    page, page_after = {"fingerprint": "first"}, {"fingerprint": "second"}
    payload = {"synthetic": "only for verifier control"}
    appended = {
        "step": 2,
        "action": "field",
        "kind": "fill",
        "choice": "native-action",
        "text": "same",
        "operation": "TYPE_TEXT",
        "target": "1",
        "page_changed": True,
    }
    step = {
        "scenario": spec,
        "private_before_call": 2,
        "private_after_call": 4,
        "observation_call": 1,
        "observation_after_call": 3,
        "tick_start": 1,
        "tick_end": 5,
        "before": before,
        "after": after,
        "page": page,
        "page_after": page_after,
        "payload": payload,
        "history_before": before_history,
        "history_after": [*before_history, appended],
        "teacher": descriptor,
        "target": "1",
        "action": action,
    }
    journal = [{"session": "owned"} for _ in range(5)]
    monkeypatch.setattr(
        collection, "_runtime_value", lambda _j, index, _e: {1: {}, 2: before, 4: after}[index]
    )
    monkeypatch.setattr(
        collection, "_latest_observation", lambda _j, _e, state: 1 if state is page else 3
    )
    monkeypatch.setattr(collection, "_publisher_rules", lambda _: {})
    monkeypatch.setattr(collection, "_publisher_wire", lambda _p, _g, _h, _r: payload)
    monkeypatch.setattr(scenarios, "teacher_step", lambda *_: descriptor)
    monkeypatch.setattr(scenarios, "verify_transition", lambda *_: {})
    monkeypatch.setattr(collection, "_selected_action", lambda *_: ("1", action))
    monkeypatch.setattr(collection, "_verify_executor", lambda *_a, **_k: {})
    collection._verify_step(step, journal, "upstream", "read", scenario_profile=PROFILE)
    for history in (
        before_history,
        [{**before_history[0], "text": "rewritten"}, appended],
        [before_history[0], {**appended, "page_changed": False}],
    ):
        changed = {**step, "history_after": history}
        with pytest.raises(ValueError, match="history"):
            collection._verify_step(changed, journal, "upstream", "read", scenario_profile=PROFILE)
