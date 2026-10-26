"""Profile isolation and admission boundaries; no browser or model execution."""

import copy
from pathlib import Path

import jev_decision3_affordance_collection as collection
import pytest

PROFILE = collection.GOAL_ROUTED_CONTROL_PROFILE
BINDINGS = ("scenario_profile", "scenario_schema_id", "scenario_namespace", "expected_menu_counts")


def prospective(tmp_path, monkeypatch, *, profile=PROFILE):
    root = tmp_path.resolve(strict=True)
    upstream = root / "invented-upstream"
    upstream.mkdir(exist_ok=True)
    exclusion = root / "invented-exclusions.json"
    exclusion.write_bytes(
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
        lambda _: {
            "revision": collection.benchmark.UPSTREAM_REVISION,
            "source_sha256": {"model.py": "1" * 64},
        },
    )
    monkeypatch.setattr(collection.benchmark, "prepare_observation_profile", lambda *_: None)
    return collection.collection_plan(
        output_dir=root / "uncreated-output",
        upstream=upstream,
        collection_id="invented-goal-routed",
        seed=913,
        exclusion_manifest_path=exclusion,
        scenario_profile=profile,
    )


def unexpected_runtime(*_args, **_kwargs):
    raise AssertionError("A mismatched profile reached browser or protocol work")


def test_new_plan_authenticates_its_own_population_without_creating_output(tmp_path, monkeypatch):
    plan = prospective(tmp_path, monkeypatch)
    module = collection._scenario_module(PROFILE)
    assert plan["scenario_profile"] == PROFILE
    assert plan["scenario_schema_id"] == module.SCHEMA
    assert plan["scenario_namespace"] == module.NAMESPACE
    assert plan["expected_menu_counts"] == [6]
    assert len(plan["episodes"]) == 12
    assert all(item["scenario"]["split"] == "train" for item in plan["episodes"])
    assert all(item["scenario"]["namespace"] == module.NAMESPACE for item in plan["episodes"])
    assert all(item["intended_menu_counts"] == [6] for item in plan["episodes"])
    assert collection._encode(collection._replan(plan)) == collection._encode(plan)
    assert not Path(plan["output_dir"]).exists()
    assert plan["model_score"] is None and plan["fit_eligible"] is False
    collection._assert_sources(plan["sources"], PROFILE)
    selected = "scripts/jev_decision3_goal_routed_control_scenarios.py"
    assert selected in plan["sources"]
    changed = copy.deepcopy(plan["sources"])
    changed[selected]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source inventory changed"):
        collection._assert_sources(changed, PROFILE)


@pytest.mark.parametrize(
    "tamper",
    ["checked-profile", "checked-namespace", "checked-schema", "missing-profile", "legacy"],
)
def test_cross_profile_or_missing_bindings_fail_before_browser_work(tmp_path, monkeypatch, tamper):
    plan = prospective(tmp_path, monkeypatch)
    checked = collection._scenario_module(collection.CHECKED_CONTROL_PROFILE)
    if tamper == "checked-profile":
        plan["scenario_profile"] = collection.CHECKED_CONTROL_PROFILE
    elif tamper == "checked-namespace":
        plan["scenario_namespace"] = checked.NAMESPACE
    elif tamper == "checked-schema":
        plan["scenario_schema_id"] = checked.SCHEMA
    elif tamper == "missing-profile":
        del plan["scenario_profile"]
    else:
        for key in BINDINGS:
            del plan[key]
    monkeypatch.setattr(collection, "_materialize_profile", unexpected_runtime)
    monkeypatch.setattr(collection.benchmark, "fixture_server", unexpected_runtime)
    with pytest.raises(ValueError, match="[Pp]rofile"):
        collection._replan(plan)
    with pytest.raises(ValueError, match="[Pp]rofile"):
        collection.collect(
            plan,
            collector_sha256=collection._sha(Path(collection.__file__).read_bytes()),
            plan_sha256=collection._digest(plan),
        )
    assert not Path(plan["output_dir"]).exists()


def test_new_scene_cannot_be_admitted_through_old_profile_protocol(tmp_path, monkeypatch):
    spec = prospective(tmp_path, monkeypatch)["episodes"][0]["scenario"]
    monkeypatch.setattr(collection, "_runtime_value", unexpected_runtime)
    for old in (collection.DEFAULT_SCENARIO_PROFILE, collection.CHECKED_CONTROL_PROFILE):
        with pytest.raises(ValueError, match="[Pp]rofile"):
            collection._verify_step(
                {"scenario": spec}, [], Path("/unused"), "unused", scenario_profile=old
            )


def test_older_profiles_keep_their_existing_populations(tmp_path, monkeypatch):
    for profile, episodes, counts in (
        (collection.DEFAULT_SCENARIO_PROFILE, 4, [4, 7, 8]),
        (collection.CHECKED_CONTROL_PROFILE, 2, [6]),
    ):
        plan = prospective(tmp_path, monkeypatch, profile=profile)
        assert len(plan["episodes"]) == episodes
        assert plan["expected_menu_counts"] == counts
        assert "scripts/jev_decision3_goal_routed_control_scenarios.py" not in plan["sources"]
