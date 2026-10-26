"""Invented profile/wire boundaries; no browser, model or collected evidence."""

import copy
import json
from pathlib import Path
from unittest import mock

import jev_decision3_affordance_collection as collection
import jev_decision3_affordance_scenarios as original
import pytest

CHECKED = collection.CHECKED_CONTROL_PROFILE
DEFAULT = collection.DEFAULT_SCENARIO_PROFILE
BINDINGS = ("scenario_profile", "scenario_schema_id", "scenario_namespace", "expected_menu_counts")


def prospective(tmp_path, monkeypatch, *, profile=CHECKED):
    root = tmp_path.resolve(strict=True)
    upstream = root / "invented-upstream"
    upstream.mkdir(exist_ok=True)
    excluded = root / "invented-exclusions.json"
    excluded.write_bytes(
        collection._encode(
            {
                "schema_id": collection.EXCLUSION_SCHEMA,
                "normalization": collection.NORMALIZATION,
                "goal_sha256": ["0" * 64],
                "scenario_ids": [],
                "ancestor_ids": [],
            }
        )
    )
    monkeypatch.setattr(
        collection.benchmark,
        "upstream_manifest",
        lambda _path: {
            "revision": collection.benchmark.UPSTREAM_REVISION,
            "source_sha256": {"model.py": "1" * 64},
        },
    )
    monkeypatch.setattr(collection.benchmark, "prepare_observation_profile", lambda *_args: None)
    return collection.collection_plan(
        output_dir=root / "uncreated-output",
        upstream=upstream,
        collection_id="invented-checked-control",
        seed=812,
        exclusion_manifest_path=excluded,
        scenario_profile=profile,
    )


def fail_runtime(*_args, **_kwargs):
    raise AssertionError("profile refusal must precede runtime/teacher/DOM work")


def write_admission_metadata(root, plan, *, receipt_profile=CHECKED):
    receipt = {
        "schema_id": collection.SCHEMA,
        "plan_sha256": collection._digest(plan),
        "completed": True,
        "status": "completed_pending_admission",
        "scenario_profile": receipt_profile,
    }
    (root / "plan.json").write_bytes(collection._encode(plan))
    (root / "receipt.json").write_bytes(collection._encode(receipt))
    return collection._digest(receipt), collection._digest(plan)


def test_checked_plan_binds_new_population_and_replans_without_output(tmp_path, monkeypatch):
    plan = prospective(tmp_path, monkeypatch)
    selected = collection._scenario_module(CHECKED)
    assert plan["scenario_profile"] == CHECKED
    assert plan["scenario_schema_id"] == selected.SCHEMA
    assert plan["scenario_namespace"] == selected.NAMESPACE
    assert plan["expected_menu_counts"] == [6]
    assert len(plan["episodes"]) == 2
    assert {item["scenario"]["family"] for item in plan["episodes"]} == {
        "radio-two",
        "radio-three",
    }
    assert all(item["scenario"]["namespace"] == selected.NAMESPACE for item in plan["episodes"])
    assert all(item["intended_menu_counts"] == [6] for item in plan["episodes"])
    assert collection._encode(collection._replan(plan)) == collection._encode(plan)
    assert not Path(plan["output_dir"]).exists()
    assert plan["model_score"] is None and plan["fit_eligible"] is False


def test_original_profile_keeps_its_exact_population_and_k_bins(tmp_path, monkeypatch):
    plan = prospective(tmp_path, monkeypatch, profile=DEFAULT)
    assert collection._scenario_module() is original
    assert plan["expected_menu_counts"] == [4, 7, 8]
    assert len(plan["episodes"]) == 4
    assert {tuple(item["intended_menu_counts"]) for item in plan["episodes"]} == {
        (4,),
        (7,),
        (7, 8),
    }
    assert "scripts/jev_decision3_checked_control_scenarios.py" not in plan["sources"]
    collection._assert_menu_coverage({4, 7, 8})
    collection._assert_menu_coverage({6}, CHECKED)
    with pytest.raises(ValueError, match="K4/K7/K8"):
        collection._assert_menu_coverage({6})
    with pytest.raises(ValueError, match="K6"):
        collection._assert_menu_coverage({4, 6, 7, 8}, CHECKED)


def test_selected_module_and_registry_source_are_in_the_authenticated_inventory(
    tmp_path, monkeypatch
):
    plan = prospective(tmp_path, monkeypatch)
    selected_path = "scripts/jev_decision3_checked_control_scenarios.py"
    assert {
        selected_path,
        "scripts/jev_decision3_affordance_collection.py",
        "scripts/jev_decision3_affordance_scenarios.py",
    } <= set(plan["sources"])
    collection._assert_sources(plan["sources"], CHECKED)
    for mode in ("missing", "hash"):
        changed = copy.deepcopy(plan["sources"])
        if mode == "missing":
            del changed[selected_path]
        else:
            changed[selected_path]["sha256"] = "f" * 64
        with pytest.raises(ValueError, match="source inventory changed"):
            collection._assert_sources(changed, CHECKED)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scenario_profile", DEFAULT),
        ("scenario_profile", "unknown-profile"),
        ("scenario_namespace", original.NAMESPACE),
        ("scenario_schema_id", original.SCHEMA),
        ("expected_menu_counts", [4, 7, 8]),
    ],
)
def test_replan_and_collect_reject_profile_tampering_before_runtime(
    tmp_path, monkeypatch, field, value
):
    plan = prospective(tmp_path, monkeypatch)
    plan[field] = value
    monkeypatch.setattr(collection, "_materialize_profile", fail_runtime)
    monkeypatch.setattr(collection.benchmark, "fixture_server", fail_runtime)
    with pytest.raises(ValueError, match="[Pp]rofile"):
        collection._replan(plan)
    with pytest.raises(ValueError, match="[Pp]rofile"):
        collection.collect(
            plan,
            collector_sha256=collection._sha(Path(collection.__file__).read_bytes()),
            plan_sha256=collection._digest(plan),
        )
    assert not Path(plan["output_dir"]).exists()


def test_missing_profile_never_admits_new_namespace_as_a_legacy_default(tmp_path, monkeypatch):
    plan = prospective(tmp_path, monkeypatch)
    for keys in (("scenario_profile",), BINDINGS):
        changed = copy.deepcopy(plan)
        for key in keys:
            del changed[key]
        with pytest.raises(ValueError, match="[Pp]rofile"):
            collection._plan_scenarios(changed)
    old = prospective(tmp_path, monkeypatch, profile=DEFAULT)
    for key in BINDINGS:
        del old[key]
    assert collection._plan_scenarios(old) == (DEFAULT, original)


@pytest.mark.parametrize("tamper", ["namespace", "profile", "missing-profile", "receipt"])
def test_independent_admission_rejects_profile_mismatch_before_source_or_dom_replay(
    tmp_path, monkeypatch, tamper
):
    plan = prospective(tmp_path, monkeypatch)
    receipt_profile = CHECKED
    if tamper == "namespace":
        plan["scenario_namespace"] = original.NAMESPACE
    elif tamper == "profile":
        plan["scenario_profile"] = DEFAULT
    elif tamper == "missing-profile":
        for key in BINDINGS:
            del plan[key]
    else:
        receipt_profile = DEFAULT
    receipt_sha, plan_sha = write_admission_metadata(
        tmp_path.resolve(strict=True), plan, receipt_profile=receipt_profile
    )
    monkeypatch.setattr(collection, "_assert_sources", fail_runtime)
    monkeypatch.setattr(collection, "_verify_step", fail_runtime)
    with pytest.raises(ValueError, match="[Pp]rofile"):
        collection.admit_training_collection(
            tmp_path.resolve(strict=True),
            expected_receipt_sha256=receipt_sha,
            expected_plan_sha256=plan_sha,
        )


def test_wrong_profile_is_refused_before_episode_or_step_touches_protocol(tmp_path, monkeypatch):
    plan = prospective(tmp_path, monkeypatch)
    spec = plan["episodes"][0]["scenario"]
    monkeypatch.setattr(collection, "_runtime_value", fail_runtime)
    with pytest.raises(ValueError, match="[Pp]rofile"):
        collection._verify_step({"scenario": spec}, [], Path("/unused"), "unused")
    with pytest.raises(ValueError, match="[Pp]rofile"):
        collection.collect_episode(
            spec=spec,
            episode_id="invented",
            origin="unused",
            transport=None,
            agent_module=None,
            model_module=None,
            upstream=Path("/unused"),
            read_state="unused",
        )


def test_unknown_profile_is_refused_before_planning_any_files(monkeypatch):
    monkeypatch.setattr(collection, "_prospective_output", fail_runtime)
    with pytest.raises(ValueError, match="Unknown explicit scenario profile"):
        collection.collection_plan(
            output_dir="unused",
            upstream="unused",
            collection_id="invented",
            seed=1,
            exclusion_manifest_path="unused",
            scenario_profile="unselected-profile",
        )


def test_k6_native_row_keeps_all_heads_criteria_order_and_teacher_outside_wire():
    menus = ["CLICK", "TYPE_TEXT", "SELECT", "WAIT", "DONE", "BLOCKED"]
    payload = {
        "model": collection.MODEL_ALIAS,
        "state": {"page": {"title": "Invented controls", "text": "Ordinary visible fields"}},
        "questions": {
            "type_text_target": {
                "type": "choice",
                "instructions": "visible field",
                "criteria": {"5": {"element": "Note", "current_value": ""}},
            },
            "click_target": {
                "type": "choice",
                "instructions": "visible control",
                "criteria": {
                    "7": {"element": "Recorded control"},
                    "1": {"element": "Other control"},
                },
            },
            "operation": {
                "type": "choice",
                "instructions": {"goal": "invented only"},
                "criteria": {label: label for label in menus},
            },
            "select_target": {
                "type": "choice",
                "instructions": "visible choices",
                "criteria": {"4:2": {"element": "Mode B"}, "4:1": {"element": "Mode A"}},
            },
        },
    }
    original_bytes = collection._encode(payload)
    for operation, target in (
        ("CLICK", "7"),
        ("TYPE_TEXT", "5"),
        ("SELECT", "4:2"),
        ("DONE", None),
    ):
        row = collection.build_native_training_row(
            payload,
            row_id="invented.step-00",
            scenario_id="invented-checked",
            ancestor_id="invented-ancestor",
            episode_id="invented",
            step=0,
            operation=operation,
            target=target,
            proof_refs={"transition": "invented#step-0"},
        )
        assert collection._encode(row["payload"]) == original_bytes
        assert list(row["payload"]["questions"]) == list(payload["questions"])
        assert row["candidate_labels"]["operation"] == menus
        assert row["candidate_labels"]["click_target"] == ["7", "1"]
        assert row["candidate_labels"]["select_target"] == ["4:2", "4:1"]
        assert row["loss_mask"] == {head: head == "operation" for head in payload["questions"]}
        assert row["fit_eligible"] is False
        assert b"teacher_choice" not in collection._encode(row["payload"])
    assert collection._encode(payload) == original_bytes


def test_cli_forwards_explicit_scenario_profile_without_collecting(capsys):
    with mock.patch.object(collection, "collection_plan", return_value={"invented": True}) as plan:
        assert (
            collection.main(
                [
                    "--output-dir",
                    "unused",
                    "--upstream",
                    "unused",
                    "--collection-id",
                    "invented",
                    "--seed",
                    "1",
                    "--exclusion-manifest",
                    "unused",
                    "--scenario-profile",
                    CHECKED,
                    "--observation-profile",
                    collection.benchmark.NATIVE_OBSERVATION_PROFILE,
                ]
            )
            == 0
        )
    assert plan.call_args.kwargs["scenario_profile"] == CHECKED
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
