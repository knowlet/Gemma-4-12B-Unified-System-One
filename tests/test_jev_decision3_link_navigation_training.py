"""Synthetic dual-head merge contracts; no model, browser, or private data."""

import copy

import jev_decision3_link_navigation_features as features
import jev_decision3_link_navigation_pipeline as pipeline
import jev_decision3_link_navigation_training as training
import jev_decision3_text_history_collection as base
import numpy as np
import pytest


def old_dataset(monkeypatch):
    baseline = np.zeros((255, 1024), dtype=np.float32)
    baseline.flags.writeable = False
    monkeypatch.setattr(training, "W0_SHA256", training._sha(baseline.tobytes()))
    old, evidence = {}, {}
    for head, rows in training.OLD_ROWS.items():
        x = np.zeros((rows, 1024), dtype=np.float32)
        y = np.zeros((rows, 255), dtype=np.float32)
        y[:, 0] = 1
        counts = np.full(rows, 2, dtype=np.int64)
        for item in (x, y, counts):
            item.flags.writeable = False
        mapping = [{"prefix": head, "index": index} for index in range(rows)]
        old[head] = {
            "features": x,
            "targets": y,
            "counts": counts,
            "baseline": baseline,
            "row_map": mapping,
            "row_mapping_sha256": training._sha(training._encode(mapping)),
        }
        inventory = [training._sha(f"{head}-{index}".encode()) for index in range(rows)]
        evidence[head] = {
            "wire_sha256_ordered": inventory,
            "wire_inventory_sha256": training._sha(training._encode(inventory)),
            "head_id": head,
            "rows": rows,
            "row_map": copy.deepcopy(mapping),
            "row_mapping_sha256": old[head]["row_mapping_sha256"],
            "frozen_input_sha256": {
                key: training._sha(old[head][key].tobytes())
                for key in ("features", "targets", "counts", "baseline")
            },
        }
    return old, evidence


def synthetic_collection():
    blobs = []
    for index in range(training.NEW_ROWS):
        done = index % 3 == 0
        singleton = index % 3 == 1
        labels = {"t1": "First"} if singleton else {"t2": "Other", "t1": "First"}
        wire = {
            "model": features.ALIAS,
            "state": {"page": index},
            "questions": {
                "operation": {
                    "type": "choice",
                    "instructions": "Choose action",
                    "criteria": {"CLICK": "Click", "DONE": "Done", "WAIT": "Wait"},
                },
                "click_target": {
                    "type": "choice",
                    "instructions": "Choose target",
                    "criteria": labels,
                },
            },
        }
        operation = "DONE" if done else "CLICK"
        target = None if done else "t1"
        valid = (
            {"DONE": []}
            if done
            else {"CLICK": list(labels), "WAIT": []}
            if not singleton
            else {"CLICK": ["t1"]}
        )
        row = {
            "schema_id": base.ROW_SCHEMA,
            "split": "train",
            "id": f"case-{index}",
            "payload": wire,
            "teacher_choice": {"operation": operation, "target": target},
            "valid_next_targets": valid,
            "loss_mask": {"operation": True, "click_target": not done},
            "wire_sha256": training._sha(training._encode(wire)),
            "fit_eligible": False,
        }
        blobs.append(base._encode(row))
    receipt, plan = "a" * 64, "b" * 64
    seal = base._digest(
        {"rows": [blob.decode() for blob in blobs], "receipt": receipt, "plan": plan}
    )
    return pipeline.AdmittedLinkCollection(
        tuple(blobs), receipt, plan, seal, pipeline._ADMISSION_PROOF
    )


def synthetic_shards(admitted, baseline):
    shards = features.feature_only_shards(admitted, shard_size=40)
    for shard in shards:
        cases = []
        for item in shard.pop("wires"):
            wire = item["payload"]
            heads = list(wire["questions"])
            criteria = [list(wire["questions"][head]["criteria"]) for head in heads]
            values = np.zeros((len(heads), 1024), dtype=np.float32)
            values[:, 0] = item["global_ordinal"]
            answers = {}
            for head, labels in zip(heads, criteria, strict=True):
                answers[head] = {
                    "type": "choice",
                    "choice": labels[0],
                    "probabilities": {name: 1.0 / len(labels) for name in labels},
                    "confidence": 1.0 if len(labels) == 1 else 0.0,
                }
            raw = {"model": "d3-edge", "answers": answers}
            parsed, _, _, _ = training.parse_native_payload(wire, model_identity=features.ALIAS)
            cases.append(
                {
                    "global_ordinal": item["global_ordinal"],
                    "case_id": item["case_id"],
                    "wire_sha256": item["wire_sha256"],
                    "head_order": heads,
                    "criteria_order": criteria,
                    "native_payload_sha256": training._sha(
                        training._encode(training.native_payload(parsed))
                    ),
                    "features": values,
                    "features_fp32_sha256": training._sha(values.tobytes()),
                    "preflight": {
                        "errors": [],
                        "project_context_limit": 16384,
                        "lengths": {head: 10 for head in heads},
                    },
                    "raw_response": raw,
                    "raw_response_sha256": training._sha(training._encode(raw)),
                }
            )
        shard.update(
            status="captured_pending_independent_fp64_audit",
            model_score=None,
            model_id=features.MODEL_ID,
            revision=features.REVISION,
            manifest_sha256=features.MANIFEST_SHA256,
            model_sha256=features.MODEL_SHA256,
            baseline=baseline,
            baseline_sha256=training.W0_SHA256,
            cases=cases,
        )
    return shards


@pytest.fixture
def inputs(monkeypatch):
    old, accepted = old_dataset(monkeypatch)
    admitted = synthetic_collection()
    shards = synthetic_shards(admitted, old["operation"]["baseline"])
    return old, accepted, admitted, shards


def test_complete_masked_append_preserves_full_old_prefix(inputs):
    old, accepted, admitted, shards = inputs
    result = training.merge(old, accepted, admitted, shards)
    assert result["status"] == "merged_pending_external_admission"
    assert (
        result["fit_eligible"] is False
        and result["fit_authorized"] is False
        and result["scores"] is None
    )
    assert result["new_rows"] == {"operation": 72, "click_target": 48}
    for head, count in training.OLD_ROWS.items():
        for key in ("features", "targets", "counts"):
            assert result[head][key][:count].tobytes() == old[head][key].tobytes()
            assert not result[head][key].flags.writeable
        assert result[head]["row_map"][:count] == old[head]["row_map"]
    op = result["operation"]["targets"]
    assert op[393, :3].tolist() == [0, 1, 0]  # DONE
    assert op[395, :3].tolist() == [0.5, 0, 0.5]  # CLICK or WAIT
    click = result["click_target"]["targets"]
    assert click[65, :2].tolist() == [1, 0]  # singleton
    assert click[66, :2].tolist() == [0.5, 0.5]


@pytest.mark.parametrize(
    "corruption",
    [
        "head_order",
        "criteria_order",
        "case_order",
        "wire_sha",
        "feature_sha",
        "raw_sha",
        "nonfinite",
        "preflight",
        "baseline",
        "coverage",
        "response",
    ],
)
def test_feature_evidence_drift_refused(inputs, corruption):
    old, accepted, admitted, shards = inputs
    case = shards[0]["cases"][0]
    if corruption == "head_order":
        case["head_order"].reverse()
    elif corruption == "criteria_order":
        case["criteria_order"][0].reverse()
    elif corruption == "case_order":
        shards[0]["cases"][0], shards[0]["cases"][1] = shards[0]["cases"][1], shards[0]["cases"][0]
    elif corruption == "wire_sha":
        case["wire_sha256"] = "f" * 64
    elif corruption == "feature_sha":
        case["features_fp32_sha256"] = "f" * 64
    elif corruption == "raw_sha":
        case["raw_response_sha256"] = "f" * 64
    elif corruption == "nonfinite":
        case["features"][0, 0] = np.nan
    elif corruption == "preflight":
        case["preflight"]["lengths"]["operation"] = 16385
    elif corruption == "baseline":
        shards[0]["baseline"] = np.ones((255, 1024), dtype=np.float32)
    elif corruption == "coverage":
        shards[1]["start"] -= 1
    else:
        case["raw_response"]["answers"]["operation"]["confidence"] = 0.5
        case["raw_response_sha256"] = training._sha(training._encode(case["raw_response"]))
    with pytest.raises((ValueError, KeyError)):
        training.merge(old, accepted, admitted, shards)


def test_duplicate_wire_rejected_even_with_distinct_case_ids(inputs, monkeypatch):
    old, accepted, admitted, shards = inputs
    rows = pipeline.compile_wire_projection(admitted)
    rows[1]["wire_sha256"] = rows[0]["wire_sha256"]
    monkeypatch.setattr(pipeline, "compile_wire_projection", lambda _: rows)
    with pytest.raises(ValueError, match="duplicate"):
        training.merge(old, accepted, admitted, shards)


@pytest.mark.parametrize(
    "problem", ["unoffered", "duplicate", "wrong_mask", "done_mix", "bad_type"]
)
def test_invalid_supervision_refused(inputs, monkeypatch, problem):
    old, accepted, admitted, shards = inputs
    rows = pipeline.compile_wire_projection(admitted)
    row = rows[0] if problem == "done_mix" else rows[1]
    if problem == "unoffered":
        row["valid_next_targets"] = {"CLICK": ["missing"]}
    elif problem == "duplicate":
        row["valid_next_targets"] = {"CLICK": ["t1", "t1"]}
    elif problem == "wrong_mask":
        row["loss_mask"]["click_target"] = True
        row["teacher_choice"] = {"operation": "DONE", "target": None}
        row["valid_next_targets"] = {"DONE": []}
    elif problem == "done_mix":
        row["valid_next_targets"]["CLICK"] = ["t1"]
    else:
        row["loss_mask"]["operation"] = 1
    monkeypatch.setattr(pipeline, "compile_wire_projection", lambda _: rows)
    with pytest.raises(ValueError):
        training.merge(old, accepted, admitted, shards)


def test_old_prefix_pin_or_bounds_mismatch_refused(inputs):
    old, accepted, admitted, shards = inputs
    accepted["operation"]["frozen_input_sha256"]["features"] = "f" * 64
    with pytest.raises(ValueError, match="old array"):
        training.merge(old, accepted, admitted, shards)


def test_missing_old_wire_inventory_and_cross_batch_duplicate_fail_closed(inputs):
    old, accepted, admitted, shards = inputs
    saved = accepted["operation"].pop("wire_sha256_ordered")
    with pytest.raises(ValueError, match="wire inventory"):
        training.merge(old, accepted, admitted, shards)
    accepted["operation"]["wire_sha256_ordered"] = saved
    saved[0] = pipeline.compile_wire_projection(admitted)[0]["wire_sha256"]
    accepted["operation"]["wire_inventory_sha256"] = training._sha(training._encode(saved))
    with pytest.raises(ValueError, match="duplicates"):
        training.merge(old, accepted, admitted, shards)


def test_fp64_replay_refuses_self_consistent_but_wrong_native_probability(inputs):
    old, accepted, admitted, shards = inputs
    case = shards[0]["cases"][0]
    answer = case["raw_response"]["answers"]["operation"]
    answer["probabilities"] = {"CLICK": 0.6, "DONE": 0.3, "WAIT": 0.1}
    answer["confidence"] = 1.0 + sum(
        value * training.math.log(value) for value in answer["probabilities"].values()
    ) / training.math.log(3)
    case["raw_response_sha256"] = training._sha(training._encode(case["raw_response"]))
    with pytest.raises(ValueError, match="FP64 replay"):
        training.merge(old, accepted, admitted, shards)


def test_old_candidate_count_above_capacity_is_refused(inputs):
    old, accepted, admitted, shards = inputs
    counts = old["operation"]["counts"].copy()
    counts[0] = 256
    counts.flags.writeable = False
    old["operation"]["counts"] = counts
    accepted["operation"]["frozen_input_sha256"]["counts"] = training._sha(counts.tobytes())
    with pytest.raises(ValueError, match="counts out of range"):
        training.merge(old, accepted, admitted, shards)


@pytest.mark.parametrize("errors", [{}, []], ids=["empty-dict", "empty-list"])
def test_preflight_accepts_both_publisher_empty_error_containers(inputs, errors):
    old, accepted, admitted, shards = inputs
    for shard in shards:
        for case in shard["cases"]:
            case["preflight"]["errors"] = errors.copy()
    result = training.merge(old, accepted, admitted, shards)
    assert result["new_rows"] == {"operation": 72, "click_target": 48}
    assert result["fit_eligible"] is False


@pytest.mark.parametrize("errors", [{"operation": "error"}, ["error"], None, "", False, 0])
def test_preflight_rejects_nonempty_or_wrong_error_container(inputs, errors):
    old, accepted, admitted, shards = inputs
    shards[0]["cases"][0]["preflight"]["errors"] = errors
    with pytest.raises(ValueError, match="preflight"):
        training.merge(old, accepted, admitted, shards)
