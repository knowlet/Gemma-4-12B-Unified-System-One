"""Hermetic native-wire/proof boundaries; real browser collection is opt-in.

Fake protocol records below test rejection rules only. They are not collected
browser evidence, a model score, an admission receipt or optimizer input.
"""

import copy
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jev_decision3_affordance_collection as collection
import jev_decision3_affordance_scenarios as fixtures
import pytest


def native_payload(*, menu_count=8, width=1):
    menus = {
        4: ["CLICK", "WAIT", "DONE", "BLOCKED"],
        7: ["CLICK", "TYPE_TEXT", "SELECT", "SCROLL_DOWN", "WAIT", "DONE", "BLOCKED"],
        8: ["CLICK", "TYPE_TEXT", "SELECT", "SCROLL_DOWN", "SCROLL_UP", "WAIT", "DONE", "BLOCKED"],
    }
    # Deliberately reorder heads and target keys: serialized order is part of
    # publisher fidelity, rather than a set that a projection may reconstruct.
    questions = {
        "click_target": {
            "type": "choice",
            "instructions": {"goal": "new fixture only"},
            "criteria": {
                str(index): {"element": f"Control {index}"} for index in range(width, 0, -1)
            },
        },
        "operation": {
            "type": "choice",
            "instructions": {"goal": "new fixture only"},
            "criteria": {label: label for label in menus[menu_count]},
        },
    }
    if menu_count != 4:
        questions["select_target"] = {
            "type": "choice",
            "instructions": ["select an actual observed option"],
            "criteria": {"3:2": {"element": "Delivery → Review"}},
        }
        questions["type_text_target"] = {
            "type": "choice",
            "instructions": "actual observed field",
            "criteria": {"2": {"element": "Entry", "current_value": ""}},
        }
    return {
        "model": collection.MODEL_ALIAS,
        "state": {
            "page": {"title": "Authored training surface", "text": "ordinary visible content"}
        },
        "questions": questions,
    }


def row(payload, *, operation="CLICK", target="1", **kwargs):
    metadata = {
        "row_id": "fresh-episode.step-00",
        "scenario_id": "fresh-owned-control",
        "ancestor_id": "fresh-owned-family",
        "episode_id": "fresh-episode",
        "step": 0,
        "operation": operation,
        "target": target,
        "proof_refs": {"transition": "fresh-episode#step-0"},
    }
    metadata.update(kwargs)
    return collection.build_native_training_row(payload, **metadata)


@pytest.mark.parametrize("menu_count", [4, 7, 8])
def test_complete_native_wire_order_and_singletons_survive_annotation(menu_count):
    payload = native_payload(menu_count=menu_count)
    original = collection._encode(payload)
    annotated = row(payload)
    assert collection._encode(annotated["payload"]) == original
    assert list(annotated["payload"]["questions"]) == list(payload["questions"])
    assert annotated["candidate_labels"]["click_target"] == ["1"]
    assert annotated["loss_mask"] == {head: head == "operation" for head in payload["questions"]}
    assert annotated["fit_eligible"] is False
    assert annotated["admission_status"] == "pending_independent_admission"
    assert collection._encode(payload) == original
    payload["state"]["page"]["text"] = "mutation after annotation"
    assert collection._encode(annotated["payload"]) == original


@pytest.mark.parametrize("width", [53, 255])
def test_all_real_native255_target_keys_are_retained_without_classifier52_projection(width):
    payload = native_payload(width=width)
    annotated = row(payload, target=str(width))
    assert annotated["candidate_labels"]["click_target"] == [
        str(index) for index in range(width, 0, -1)
    ]
    assert collection._encode(annotated["payload"]) == collection._encode(payload)


def test_256_target_payload_is_refused_intact_instead_of_pruned():
    payload = native_payload(width=256)
    before = collection._encode(payload)
    with pytest.raises(ValueError, match="255|candidate"):
        row(payload)
    assert collection._encode(payload) == before
    assert len(payload["questions"]["click_target"]["criteria"]) == 256


@pytest.mark.parametrize("operation", ["WAIT", "SCROLL_DOWN", "SCROLL_UP", "DONE"])
def test_targetless_supervision_uses_null_and_has_no_synthetic_target(operation):
    payload = native_payload()
    annotated = row(payload, operation=operation, target=None)
    assert annotated["teacher_choice"] == {"operation": operation, "target": None}
    assert annotated["gold"] == {"operation": operation}
    assert annotated["valid_next_targets"] == {operation: []}
    assert collection._encode(annotated["payload"]) == collection._encode(payload)
    assert operation.lower() + "_target" not in annotated["payload"]["questions"]
    with pytest.raises(ValueError, match="targetless"):
        row(payload, operation=operation, target="1")


@pytest.mark.parametrize(
    "operation,target", [("CLICK", None), ("CLICK", "999"), ("TYPE_TEXT", "1"), ("SELECT", "3")]
)
def test_targeted_operations_require_their_actual_native_key(operation, target):
    with pytest.raises(ValueError, match="actual offered target"):
        row(native_payload(), operation=operation, target=target)


def test_blocked_remains_offered_but_has_no_positive_training_label():
    payload = native_payload()
    assert "BLOCKED" in payload["questions"]["operation"]["criteria"]
    with pytest.raises(ValueError, match="BLOCKED"):
        row(payload, operation="BLOCKED", target=None)
    with pytest.raises(ValueError, match="independently justified"):
        row(payload, valid_next_targets={"CLICK": ["1"], "BLOCKED": []})


def test_teacher_literal_gold_loss_and_private_proof_stay_outside_native_payload():
    payload = native_payload()
    annotated = row(
        payload,
        operation="TYPE_TEXT",
        target="2",
        proof_refs={"transition": "private-proof-identity"},
    )
    assert annotated["gold"] == {"operation": "TYPE_TEXT", "type_text_target": "2"}
    assert annotated["loss_mask"]["type_text_target"] is False
    assert annotated["loss_mask"]["select_target"] is False
    wire = collection._encode(annotated["payload"])
    for name in ("teacher_choice", "gold", "loss_mask", "proof_refs", "private-proof-identity"):
        assert name.encode() not in wire
    assert wire == collection._encode(payload)


def test_alias_rewrite_and_simultaneous_done_mutation_supervision_are_refused():
    payload = native_payload()
    payload["model"] = "different-native-alias"
    with pytest.raises(ValueError, match="alias"):
        row(payload)
    with pytest.raises(ValueError, match="Completion"):
        row(
            native_payload(),
            operation="DONE",
            target=None,
            valid_next_targets={"DONE": [], "CLICK": ["1"]},
        )


def test_freshness_is_rederived_from_canonical_raw_guard_not_saved_true():
    action = {"kind": "click", "node": 17}
    page = {"page_key": "owned-page", "guards": {"17": "actual-guard"}}
    expression = "(() => { const c=window.__jevFast; return c ? [c.pageKey(),c.guard(c.nodes.get(17))] : null; })()"
    journal = [
        {
            "sequence": 0,
            "method": "Runtime.evaluate",
            "session": "owned-session",
            "params": {"expression": expression, "returnByValue": True},
            "response": {"result": {"value": ["owned-page", "actual-guard"]}},
        }
    ]
    entry = {"cdp_start": 0, "cdp_end": 1, "action": action, "page": page, "result": True}
    collection._verify_freshness(entry, journal, "canonical-read-state")
    changed = copy.deepcopy(journal)
    changed[0]["response"]["result"]["value"][1] = "different-guard"
    with pytest.raises(ValueError, match="freshness response"):
        collection._verify_freshness(entry, changed, "canonical-read-state")
    changed = copy.deepcopy(journal)
    changed[0]["params"]["expression"] = "forged read"
    with pytest.raises(ValueError, match="expression"):
        collection._verify_freshness(entry, changed, "canonical-read-state")


def test_marker_freshness_is_tied_to_the_reviewed_observer_expression():
    read_state = "canonical-read-state"
    expression = "(() => { const state=canonical-read-state; return state?.marker ?? null; })()"
    journal = [
        {
            "sequence": 0,
            "method": "Runtime.evaluate",
            "session": "owned-session",
            "params": {"expression": expression},
            "response": {"result": {"value": "marker-a"}},
        }
    ]
    entry = {
        "cdp_start": 0,
        "cdp_end": 1,
        "action": {"kind": "scroll"},
        "page": {"marker": "marker-a"},
        "result": True,
    }
    collection._verify_freshness(entry, journal, read_state)
    with pytest.raises(ValueError, match="expression"):
        collection._verify_freshness(entry, journal, "different observer")


def executor_click_fixture(monkeypatch):
    action = {"id": "a17", "node": 17, "kind": "click", "label": "Activate"}
    page = {"page_key": "owned-page", "guards": {"17": "actual-guard"}}
    guard = "(() => { const c=window.__jevFast; return c ? [c.pageKey(),c.guard(c.nodes.get(17))] : null; })()"
    monkeypatch.setattr(collection, "_native_action_expression", lambda *_: "canonical-hit")
    point = {"x": 10, "y": 20}
    records = [
        ("Runtime.evaluate", {"expression": "private-before"}, None),
        ("Runtime.evaluate", {"expression": guard}, ["owned-page", "actual-guard"]),
        ("Runtime.evaluate", {"expression": "canonical-hit"}, point),
        (
            "Input.dispatchMouseEvent",
            {"type": "mousePressed", **point, "button": "left", "clickCount": 1},
            None,
        ),
        (
            "Input.dispatchMouseEvent",
            {"type": "mouseReleased", **point, "button": "left", "clickCount": 1},
            None,
        ),
        ("Runtime.evaluate", {"expression": "canonical-read-state"}, {}),
        ("Runtime.evaluate", {"expression": "private-after"}, None),
    ]
    journal = [
        {
            "sequence": index,
            "method": method,
            "session": "owned-session",
            "params": params,
            "response": {"result": {"value": value}},
        }
        for index, (method, params, value) in enumerate(records)
    ]
    payload = native_payload(menu_count=4)
    step = {
        "teacher": {"operation": "CLICK"},
        "action": action,
        "page": page,
        "tick_start": 0,
        "tick_end": len(journal),
        "payload": payload,
        "target": "1",
        "private_before_call": 0,
        "observation_after_call": 5,
        "private_after_call": 6,
        "fresh_calls": [
            {"page": page, "action": action, "cdp_start": 1, "cdp_end": 2, "result": True}
        ],
        "acts": [
            {
                "action": action,
                "page": page,
                "result": {"executed": "a17"},
                "cdp_start": 1,
                "cdp_end": 5,
            }
        ],
        "field_calls": [],
        "decision": {
            "request": payload,
            "operation": "CLICK",
            "target": "1",
            "choice": "a17",
            "raw_answers": collection._teacher_response(payload, "CLICK", "1")["answers"],
        },
    }
    return step, journal


def test_executor_accepts_guard_hit_and_input_inside_original_act(monkeypatch):
    step, journal = executor_click_fixture(monkeypatch)
    assert collection._verify_executor(step, journal, Path("/unused"), "canonical-read-state") == {
        "native_guarded_action": True,
        "input_calls": 2,
    }


def test_executor_rejects_tick_inputs_outside_the_actual_act_span(monkeypatch):
    step, journal = executor_click_fixture(monkeypatch)
    step["acts"][0]["cdp_end"] = 3
    with pytest.raises(ValueError, match="Execution|inside|canonical|sequence"):
        collection._verify_executor(step, journal, Path("/unused"), "canonical-read-state")


def test_executor_rejects_input_before_the_action_freshness_guard(monkeypatch):
    step, journal = executor_click_fixture(monkeypatch)
    journal[1:5] = [journal[3], journal[4], journal[1], journal[2]]
    for index, call in enumerate(journal):
        call["sequence"] = index
    step["fresh_calls"][0].update(cdp_start=3, cdp_end=4)
    with pytest.raises(ValueError, match="guard|canonical|sequence"):
        collection._verify_executor(step, journal, Path("/unused"), "canonical-read-state")


def test_done_requires_its_post_decision_terminal_guard(monkeypatch):
    monkeypatch.setattr(collection.fixtures, "verify_goal", lambda *_: True)
    payload = native_payload(menu_count=4)
    marker = "(() => { const state=canonical-read-state; return state?.marker ?? null; })()"
    journal = [
        {
            "sequence": index,
            "method": "Runtime.evaluate",
            "session": "owned-session",
            "params": {"expression": expression},
            "response": {"result": {"value": "marker-a"}},
        }
        for index, expression in enumerate([marker, "private-before", marker, "private-after"])
    ]
    page = {"marker": "marker-a"}
    before = {"page": page, "action": None, "cdp_start": 0, "cdp_end": 1, "result": True}
    terminal = {**before, "cdp_start": 2, "cdp_end": 3}
    step = {
        "teacher": {"operation": "DONE"},
        "action": None,
        "page": page,
        "tick_start": 0,
        "tick_end": 4,
        "payload": payload,
        "target": None,
        "private_before_call": 1,
        "private_after_call": 3,
        "fresh_calls": [before, terminal],
        "acts": [],
        "field_calls": [],
        "scenario": {},
        "after": {},
        "status": "done",
        "decision": {
            "request": payload,
            "operation": "DONE",
            "target": None,
            "choice": "DONE",
            "raw_answers": collection._teacher_response(payload, "DONE", None)["answers"],
        },
    }
    assert collection._verify_executor(step, journal, Path("/unused"), "canonical-read-state") == {
        "native_terminal": True,
        "input_calls": 0,
    }
    step["fresh_calls"] = [before]
    with pytest.raises(ValueError, match="terminal.*guard"):
        collection._verify_executor(step, journal, Path("/unused"), "canonical-read-state")


def ownership_fixture():
    origin = "http://127.0.0.1:45123"
    url = origin + "/tasks/fresh-owned-control"
    records = [
        (
            "Target.createTarget",
            None,
            {"url": "about:blank", "background": True},
            {"targetId": "target-1"},
        ),
        (
            "Target.attachToTarget",
            None,
            {"targetId": "target-1", "flatten": True},
            {"sessionId": "session-1"},
        ),
        (
            "Emulation.setDeviceMetricsOverride",
            "session-1",
            {"width": 1120, "height": 780, "deviceScaleFactor": 1, "mobile": False},
            {},
        ),
        ("Emulation.setFocusEmulationEnabled", "session-1", {"enabled": True}, {}),
        ("Page.navigate", "session-1", {"url": url}, {}),
        (
            "Runtime.evaluate",
            "session-1",
            {"expression": "document.readyState"},
            {"result": {"value": "complete"}},
        ),
        (
            "Runtime.evaluate",
            "session-1",
            {"expression": "canonical-read-state"},
            {"result": {"value": {}}},
        ),
        ("Runtime.evaluate", "session-1", {"expression": "tick-private-before"}, {}),
        ("Runtime.evaluate", "session-1", {"expression": "tick-fresh"}, {}),
        ("Runtime.evaluate", "session-1", {"expression": "tick-private-after"}, {}),
        ("Target.closeTarget", None, {"targetId": "target-1"}, {"success": True}),
    ]
    journal = [
        {
            "sequence": index,
            "method": method,
            "session": session,
            "params": params,
            "response": response,
        }
        for index, (method, session, params, response) in enumerate(records)
    ]
    trace = {
        "cdp_start": 0,
        "cdp_end": 11,
        "scenario": {"id": "fresh-owned-control"},
        "steps": [
            {
                "tick_start": 7,
                "tick_end": 10,
                "teacher": {"operation": "DONE"},
                "before": {"url": url},
                "after": {"url": url},
            }
        ],
    }
    return trace, journal, origin


def test_owned_episode_lifecycle_and_tick_spans_refuse_reused_done_and_unassigned_input():
    trace, journal, origin = ownership_fixture()
    assert collection._verify_episode_ownership(trace, journal, origin, "canonical-read-state") == (
        0,
        11,
    )
    duplicated = copy.deepcopy(trace)
    duplicated["steps"].append(copy.deepcopy(duplicated["steps"][0]))
    with pytest.raises(ValueError, match="advance once"):
        collection._verify_episode_ownership(duplicated, journal, origin, "canonical-read-state")
    changed = copy.deepcopy(journal)
    changed[6]["method"] = "Input.insertText"
    changed[6]["params"] = {"text": "unassigned field mutation"}
    with pytest.raises(ValueError, match="Unassigned execution"):
        collection._verify_episode_ownership(trace, changed, origin, "canonical-read-state")
    changed = copy.deepcopy(journal)
    changed[8]["session"] = "different-session"
    with pytest.raises(ValueError, match="owned session"):
        collection._verify_episode_ownership(trace, changed, origin, "canonical-read-state")


def test_native_action_space_preserves_order_singletons_and_real_select_option_keys():
    actions = [
        {"id": "a", "kind": "click", "node": 9, "label": "Activate", "role": "button"},
        {"id": "b", "kind": "fill", "node": 10, "label": "Entry", "value": ""},
        {"id": "c", "kind": "click", "node": 10, "label": "Open Entry", "value": ""},
        {
            "id": "d",
            "kind": "select",
            "node": 11,
            "label": "Delivery → Draft",
            "value": "draft",
            "current_value": "draft",
        },
        {
            "id": "e",
            "kind": "select",
            "node": 11,
            "label": "Delivery → Review",
            "value": "review",
            "current_value": "draft",
        },
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
        {"id": "wait", "kind": "wait", "label": "Wait"},
    ]
    before = copy.deepcopy(actions)
    elements, targets, controls = collection._action_space(actions)
    assert list(targets) == ["CLICK", "TYPE_TEXT", "SELECT"]
    assert list(targets["CLICK"]) == ["1", "2"]
    assert list(targets["TYPE_TEXT"]) == ["2"]
    assert list(targets["SELECT"]) == ["3:1", "3:2"]
    assert elements[2]["options"][1]["value"] == "review"
    assert list(controls) == ["SCROLL_DOWN", "WAIT"]
    descriptor = {
        "operation": "SELECT",
        "kind": "select",
        "label": "Delivery → Review",
        "action_id": None,
        "text": None,
    }
    target, selected = collection._selected_action({"actions": actions}, descriptor)
    assert target == "3:2" and selected["id"] == "e"
    assert actions == before


def exclusions(tmp_path, **changes):
    data = {
        "schema_id": collection.EXCLUSION_SCHEMA,
        "normalization": collection.NORMALIZATION,
        "goal_sha256": ["0" * 64],
        "scenario_ids": [],
        "ancestor_ids": [],
    }
    data.update(changes)
    path = tmp_path / "reviewed-hash-only-exclusions.json"
    path.write_bytes(collection._encode(data))
    return path


def plan_arguments(tmp_path):
    upstream = tmp_path / "canonical-upstream"
    upstream.mkdir()
    return {
        "output_dir": tmp_path / "fresh-output",
        "upstream": upstream,
        "collection_id": "new-affordances-source-test",
        "seed": 804,
        "exclusion_manifest_path": exclusions(tmp_path),
    }


def fake_upstream(monkeypatch):
    monkeypatch.setattr(
        collection.benchmark,
        "upstream_manifest",
        lambda _path: {
            "revision": collection.benchmark.UPSTREAM_REVISION,
            "source_sha256": {"model.py": "1" * 64},
        },
    )
    monkeypatch.setattr(collection.benchmark, "prepare_observation_profile", lambda *_args: None)


def test_dry_run_binds_sources_new_scenarios_goal_exclusions_and_intended_not_observed_k(
    tmp_path, monkeypatch
):
    fake_upstream(monkeypatch)
    arguments = plan_arguments(tmp_path)
    monkeypatch.setattr(
        collection.benchmark,
        "fixture_server",
        lambda *_args: pytest.fail("dry-run started browser/server"),
    )
    prospective = collection.collection_plan(**arguments)
    assert not arguments["output_dir"].exists()
    assert prospective["split"] == "train"
    assert prospective["admission_status"] == "not_collected"
    assert prospective["model_score"] is None and prospective["fit_eligible"] is False
    assert len(prospective["episodes"]) == 4
    assert {tuple(item["intended_menu_counts"]) for item in prospective["episodes"]} == {
        (4,),
        (7,),
        (7, 8),
    }
    assert "observed_menu_counts" not in prospective
    assert all(item["goal_sha256"] != "0" * 64 for item in prospective["episodes"])
    assert all(item["scenario"]["split"] == "train" for item in prospective["episodes"])
    assert collection._digest(prospective) == collection._digest(
        collection.collection_plan(**arguments)
    )


def test_held_out_goal_or_ancestor_rejected_before_any_collection(tmp_path, monkeypatch):
    fake_upstream(monkeypatch)
    arguments = plan_arguments(tmp_path)
    spec = fixtures.scenarios(arguments["seed"], {"0" * 64})[0]
    arguments["exclusion_manifest_path"] = exclusions(tmp_path, ancestor_ids=[spec["ancestor_id"]])
    with pytest.raises(ValueError, match="held-out ancestor"):
        collection.collection_plan(**arguments)
    assert not arguments["output_dir"].exists()


@pytest.mark.parametrize(
    "change",
    [
        {"schema_id": "owned-semantic-menu-collection-admission-v1"},
        {"goal_sha256": []},
        {"private_goals": ["never accept raw held-out content"]},
    ],
)
def test_exclusion_input_accepts_only_reviewed_nonempty_hash_schema(tmp_path, change):
    with pytest.raises(ValueError):
        collection.exclusion_manifest(exclusions(tmp_path, **change))


def test_source_change_and_native_alias_are_bound_by_plan(tmp_path, monkeypatch):
    fake_upstream(monkeypatch)
    arguments = plan_arguments(tmp_path)
    prospective = collection.collection_plan(**arguments)
    assert prospective["model_alias"] == collection.MODEL_ALIAS
    changed = copy.deepcopy(prospective["sources"])
    next(iter(changed.values()))["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source inventory changed"):
        collection._assert_sources(changed)


def test_primary_comparison_settled_v7_profile_and_helper_are_explicit_plan_inputs(
    tmp_path, monkeypatch
):
    fake_upstream(monkeypatch)
    arguments = plan_arguments(tmp_path)
    calls = []

    def prepare(upstream, profile, helper):
        calls.append((upstream, profile, helper))
        return SimpleNamespace(
            manifest=lambda: {"profile": profile, "helper_source_sha256": helper}
        )

    monkeypatch.setattr(collection, "_prepare_observation_profile", prepare)
    prospective = collection.collection_plan(
        **arguments,
        observation_profile=collection.SETTLED_V7_PROFILE,
        observation_helper_sha256="1" * 64,
    )
    assert calls[0][1:] == (collection.SETTLED_V7_PROFILE, "1" * 64)
    assert prospective["observation_profile"] == collection.SETTLED_V7_PROFILE
    assert prospective["observation_helper_sha256"] == "1" * 64
    assert "scripts/jev_observation_profile_v7.py" in prospective["sources"]
    changed = copy.deepcopy(prospective)
    changed["observation_helper_sha256"] = "2" * 64
    assert collection._digest(changed) != collection._digest(prospective)


def test_explicit_collection_pin_failure_happens_before_output_or_runtime(tmp_path, monkeypatch):
    fake_upstream(monkeypatch)
    arguments = plan_arguments(tmp_path)
    prospective = collection.collection_plan(**arguments)
    monkeypatch.setattr(
        collection, "_replan", lambda *_args: pytest.fail("bad plan pin reached runtime replan")
    )
    with pytest.raises(ValueError, match="collector SHA256"):
        collection.collect(
            prospective, collector_sha256="0" * 64, plan_sha256=collection._digest(prospective)
        )
    with pytest.raises(ValueError, match="prospective plan changed"):
        collection.collect(
            prospective,
            collector_sha256=collection._sha(Path(collection.__file__).read_bytes()),
            plan_sha256="0" * 64,
        )
    assert not arguments["output_dir"].exists()


def test_projection_requires_new_admission_handle_and_preserves_sealed_native_bytes():
    annotated = row(native_payload())
    encoded = collection._encode(annotated)
    receipt, plan = "1" * 64, "2" * 64
    seal = collection._digest({"rows": [encoded.decode()], "receipt": receipt, "plan": plan})
    # This private sentinel is used only to unit-test the immutable projection
    # seam. It supplies no evidence and must never be written as an admission.
    handle = collection.AdmittedAffordanceCollection(
        (encoded,), receipt, plan, seal, collection._ADMISSION_PROOF
    )
    projected = collection.compile_wire_projection(handle)
    assert collection._encode(projected[0]["payload"]) == collection._encode(annotated["payload"])
    assert projected[0]["loss_mask"]["click_target"] is False
    with pytest.raises(ValueError, match="independently admitted"):
        collection.compile_wire_projection(SimpleNamespace(_rows=(encoded,)))
    with pytest.raises(ValueError, match="alias"):
        collection.compile_wire_projection(handle, model_alias="different-model")
    changed = copy.deepcopy(annotated)
    changed["payload"]["questions"]["operation"]["criteria"] = dict(
        reversed(list(changed["payload"]["questions"]["operation"]["criteria"].items()))
    )
    with pytest.raises(ValueError, match="row bytes"):
        collection.compile_wire_projection(replace(handle, _rows=(collection._encode(changed),)))


@pytest.mark.parametrize(
    "schema,split,completed",
    [
        ("foreign", "train", True),
        (collection.SCHEMA, "development", True),
        (collection.SCHEMA, "train", False),
    ],
)
def test_admission_refuses_foreign_nontrain_and_incomplete_receipts_without_browser(
    tmp_path, schema, split, completed
):
    plan = {"schema_id": schema, "split": split}
    plan_sha = collection._digest(plan)
    receipt = {
        "schema_id": collection.SCHEMA,
        "plan_sha256": plan_sha,
        "completed": completed,
        "status": "completed_pending_admission",
    }
    (tmp_path / "plan.json").write_bytes(collection._encode(plan))
    (tmp_path / "receipt.json").write_bytes(collection._encode(receipt))
    with pytest.raises(ValueError, match="Foreign|Incomplete"):
        collection.admit_training_collection(
            tmp_path,
            expected_receipt_sha256=collection._digest(receipt),
            expected_plan_sha256=plan_sha,
        )


def test_raw_json_duplicate_keys_and_symlink_evidence_are_refused(tmp_path):
    evidence = tmp_path / "duplicate.json"
    evidence.write_text('{"schema_id":"first","schema_id":"second"}')
    with pytest.raises(ValueError, match="Duplicate"):
        collection._plain_json(evidence)
    link = tmp_path / "link.json"
    link.symlink_to(evidence)
    with pytest.raises(ValueError, match="ordinary file"):
        collection._plain_json(link)
    with pytest.raises(ValueError, match="escapes"):
        collection._confined_file(tmp_path, "../outside.json")


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("S1_RUN_AFFORDANCE_BROWSER_COLLECTION") != "1",
    reason="explicit fresh loopback collection only after resource diagnostics close",
)
def test_genuine_owned_browser_collection_and_independent_admission(tmp_path):
    """Actual browser proof, separate from hermetic fake-wire tests above.

    Requires reviewed hash-only exclusions, the exact original upstream checkout
    and an explicit Chromium executable. This does no model inference or fit.
    """
    upstream = os.environ["S1_AFFORDANCE_UPSTREAM"]
    chromium = os.environ["S1_AFFORDANCE_CHROMIUM"]
    exclusions_path = os.environ["S1_AFFORDANCE_EXCLUSIONS"]
    profile = os.environ.get(
        "S1_AFFORDANCE_OBSERVATION_PROFILE", collection.benchmark.NATIVE_OBSERVATION_PROFILE
    )
    helper = os.environ.get("S1_AFFORDANCE_OBSERVATION_HELPER_SHA256")
    prospective = collection.collection_plan(
        output_dir=tmp_path / "fresh-native-affordance-collection",
        upstream=upstream,
        collection_id="explicit-native-affordance-browser-proof",
        seed=804,
        exclusion_manifest_path=exclusions_path,
        chromium=chromium,
        observation_profile=profile,
        observation_helper_sha256=helper,
    )
    receipt = collection.collect(
        prospective,
        collector_sha256=prospective["collector_sha256"],
        plan_sha256=collection._digest(prospective),
    )
    assert receipt["completed"], receipt["errors"]
    output = Path(prospective["output_dir"])
    admitted = collection.admit_training_collection(
        output,
        expected_receipt_sha256=collection._sha((output / "receipt.json").read_bytes()),
        expected_plan_sha256=collection._digest(prospective),
    )
    projected = collection.compile_wire_projection(admitted)
    assert {len(item["payload"]["questions"]["operation"]["criteria"]) for item in projected} == {
        4,
        7,
        8,
    }
    assert all(item["fit_eligible"] is False for item in projected)
    assert receipt["model_calls"] == 0 and receipt["model_score"] is None
