"""In-memory, quarantined merge of admitted link rows with the 393/65 prefixes.

This module starts no model or optimizer and reads no files. The caller must
independently authenticate the old prefix evidence, captured shards, scoped
process closure and all provenance before granting any fitting capability.
Recomputing the native probabilities here is not an independent process audit.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re

import jev_decision3_link_navigation_features as features
import jev_decision3_link_navigation_pipeline as pipeline
import numpy as np

from s1.evaluation.decision3 import native_payload
from s1.jev_decision3_native import parse_native_payload

WIDTH = 1024
CAPACITY = 255
OLD_ROWS = {"operation": 393, "click_target": 65}
NEW_ROWS = 72
HEADS = tuple(OLD_ROWS)
W0_SHA256 = features.W0_SHA256


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _mapping_encode(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _array(value, shape, dtype, *, readonly=False):
    if not isinstance(value, np.ndarray):
        _need(
            hasattr(value, "detach")
            and str(value.device) == "cpu"
            and not value.requires_grad
            and value.is_contiguous(),
            "CPU detached contiguous tensor required",
        )
        value = value.detach().numpy()
    _need(
        type(value) is np.ndarray
        and value.shape == shape
        and value.dtype == dtype
        and value.flags.c_contiguous
        and (not readonly or not value.flags.writeable)
        and bool(np.isfinite(value).all()),
        "exact finite contiguous array geometry required",
    )
    return value


def _bytes(value):
    return value.tobytes(order="C")


def _pin(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _old_prefix(old, accepted):
    _need(
        type(old) is dict and type(accepted) is dict, "old datasets and selected evidence required"
    )
    _need(set(old) >= set(HEADS) and set(accepted) == set(HEADS), "both exact heads required")
    snapshots = {}
    for head, rows in OLD_ROWS.items():
        data, seal = old[head], accepted[head]
        _need(type(data) is dict and type(seal) is dict, "head inputs and hashes required")
        _need(seal.get("head_id") == head and seal.get("rows") == rows, "old head identity differs")
        arrays = {}
        for key, shape, dtype in (
            ("features", (rows, WIDTH), np.float32),
            ("targets", (rows, CAPACITY), np.float32),
            ("counts", (rows,), np.int64),
            ("baseline", (CAPACITY, WIDTH), np.float32),
        ):
            arr = _array(data[key], shape, dtype, readonly=True)
            digest = _sha(_bytes(arr))
            _need(seal.get("frozen_input_sha256", {}).get(key) == digest, "old array bytes differ")
            arrays[key] = arr
        _need(_sha(_bytes(arrays["baseline"])) == W0_SHA256, "old baseline differs")
        mapping = data["row_map"]
        _need(type(mapping) is list and len(mapping) == rows, "complete old row map required")
        old_wires = seal.get("wire_sha256_ordered")
        _need(
            type(old_wires) is list
            and len(old_wires) == rows
            and all(_pin(value) for value in old_wires)
            and seal.get("wire_inventory_sha256") == _sha(_encode(old_wires)),
            "complete externally authenticated old per-row wire inventory required",
        )
        for entry, expected_wire in zip(mapping, old_wires, strict=True):
            recorded, _ = _known_old_identities(entry)
            _need(not recorded or recorded == {expected_wire}, "old row and wire inventory differ")
        mapping_sha = data.get("row_mapping_sha256")
        _need(
            mapping_sha in {_sha(_encode(mapping)), _sha(_mapping_encode(mapping))}
            and mapping_sha == seal.get("row_mapping_sha256")
            and _encode(mapping) == _encode(seal.get("row_map")),
            "old row map or ordering differs",
        )
        counts, targets = arrays["counts"], arrays["targets"]
        _need(
            bool(((counts >= 1) & (counts <= CAPACITY)).all()), "old candidate counts out of range"
        )
        for index, count in enumerate(counts):
            target = targets[index]
            _need(
                bool((target >= 0).all())
                and bool((target[int(count) :] == 0).all())
                and math.isclose(float(target.sum(dtype=np.float64)), 1.0, abs_tol=1e-6),
                "old target distribution is invalid",
            )
        snapshots[head] = {
            **arrays,
            "row_map": copy.deepcopy(mapping),
            "wire_sha256_ordered": list(old_wires),
        }
    _need(
        _bytes(snapshots["operation"]["baseline"]) == _bytes(snapshots["click_target"]["baseline"]),
        "old shared baseline differs",
    )
    return snapshots


def _known_old_identities(value):
    """Collect old full-request and wire hashes using their shared compact encoding."""
    wires, cases = set(), set()

    def visit(item):
        if type(item) is dict:
            for key in ("wire_sha256", "request_sha256"):
                if key in item:
                    _need(_pin(item[key]), "invalid old full-request identity")
                    wires.add(item[key])
            if type(item.get("case_id")) is str:
                cases.add(item["case_id"])
            for nested in item.values():
                visit(nested)
        elif type(item) is list:
            for nested in item:
                visit(nested)

    visit(value)
    return wires, cases


def _validate_supervision(row, heads, criteria):
    mask, teacher, accepted = row["loss_mask"], row["teacher_choice"], row["valid_next_targets"]
    _need(
        type(mask) is dict and list(mask) == heads and all(type(v) is bool for v in mask.values()),
        "original head masks differ",
    )
    _need(
        mask.get("operation") is True
        and all(not value for name, value in mask.items() if name not in HEADS),
        "unsupported loss mask",
    )
    _need(
        type(teacher) is dict and set(teacher) == {"operation", "target"},
        "actual teacher choice required",
    )
    operation, target = teacher["operation"], teacher["target"]
    _need(
        type(operation) is str and operation != "BLOCKED" and operation in criteria["operation"],
        "invalid operation",
    )
    _need(
        type(accepted) is dict and operation in accepted and bool(accepted),
        "accepted operations required",
    )
    targets = {"CLICK": "click_target", "TYPE_TEXT": "type_text_target", "SELECT": "select_target"}
    for op, options in accepted.items():
        _need(
            type(op) is str
            and op != "BLOCKED"
            and op in criteria["operation"]
            and type(options) is list,
            "invalid accepted operation",
        )
        field = targets.get(op)
        if field:
            _need(
                field in criteria
                and options
                and all(type(v) is str and v in criteria[field] for v in options)
                and len(set(options)) == len(options),
                "invalid accepted target",
            )
        else:
            _need(not options, "targetless operation has targets")
    _need(
        "DONE" not in accepted or list(accepted) == ["DONE"],
        "DONE cannot supervise another operation",
    )
    actual_head = targets.get(operation)
    _need(
        (actual_head is None and target is None)
        or (actual_head is not None and type(target) is str and target in accepted[operation]),
        "actual teacher target differs",
    )
    _need(
        not mask.get("click_target", False) or operation == "CLICK",
        "click mask requires actual click",
    )
    result = {}
    for head, valid in (("operation", list(accepted)), ("click_target", accepted.get("CLICK", []))):
        if not mask.get(head, False):
            continue
        ordered = criteria[head]
        _need(valid and all(value in ordered for value in valid), "active head target not offered")
        distribution = np.zeros(CAPACITY, dtype=np.float32)
        for value in valid:
            distribution[ordered.index(value)] = 1.0 / len(valid)
        result[head] = distribution
    return result


def _native_audit(case, row, baseline):
    wire = row["payload"]
    _need(
        type(wire) is dict
        and set(wire) == {"model", "state", "questions"}
        and wire["model"] == features.ALIAS,
        "complete original wire required",
    )
    _need(
        _sha(_encode(wire)) == row["wire_sha256"] == case["wire_sha256"],
        "original wire bytes differ",
    )
    questions = wire["questions"]
    _need(
        type(questions) is dict and 1 <= len(questions) <= 4 and "operation" in questions,
        "ordered native heads required",
    )
    heads = list(questions)
    criteria = {}
    for head, question in questions.items():
        _need(
            type(head) is str
            and type(question) is dict
            and set(question) == {"type", "criteria", "instructions"}
            and question["type"] == "choice",
            "native choice required",
        )
        values = question["criteria"]
        _need(
            type(values) is dict
            and 1 <= len(values) <= CAPACITY
            and all(type(k) is str and k.strip() for k in values),
            "native criteria required",
        )
        criteria[head] = list(values)
    _need(
        case["head_order"] == heads and case["criteria_order"] == [criteria[h] for h in heads],
        "ordered head or criteria mismatch",
    )
    parsed, _, _, _ = parse_native_payload(wire, model_identity=features.ALIAS)
    expected_native = native_payload(parsed)
    _need(
        _sha(_encode(expected_native)) == case["native_payload_sha256"],
        "native payload digest differs",
    )
    preflight = case["preflight"]
    _need(
        type(preflight) is dict
        and type(preflight.get("errors")) in (dict, list)
        and not preflight["errors"]
        and preflight.get("project_context_limit") == 16384
        and type(preflight.get("lengths")) is dict
        and list(preflight["lengths"]) == heads
        and all(type(x) is int and 0 < x <= 16384 for x in preflight["lengths"].values()),
        "original preflight differs",
    )
    matrix = _array(case["features"], (len(heads), WIDTH), np.float32)
    _need(_sha(_bytes(matrix)) == case["features_fp32_sha256"], "same-call feature bytes differ")
    raw = case["raw_response"]
    _need(
        type(raw) is dict
        and raw.get("model") == "d3-edge"
        and _sha(_encode(raw)) == case["raw_response_sha256"]
        and type(raw.get("answers")) is dict
        and list(raw["answers"]) == heads,
        "raw native response differs",
    )
    logits = matrix.astype(np.float64) @ baseline.astype(np.float64).T
    _need(bool(np.isfinite(logits).all()), "nonfinite FP64 logits")
    max_error = 0.0
    for index, head in enumerate(heads):
        labels, answer = criteria[head], raw["answers"][head]
        _need(
            type(answer) is dict
            and answer.get("type") == "choice"
            and not answer.get("error")
            and type(answer.get("probabilities")) is dict
            and list(answer["probabilities"]) == labels,
            "native answer differs",
        )
        probabilities = list(answer["probabilities"].values())
        _need(
            all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in probabilities)
            and math.isclose(sum(probabilities), 1, abs_tol=1e-6),
            "native probabilities invalid",
        )
        _need(
            answer.get("choice")
            == max(answer["probabilities"], key=answer["probabilities"].__getitem__),
            "native choice differs",
        )
        legal = logits[index, : len(labels)]
        exponentials = np.exp(legal - legal.max())
        expected = exponentials / exponentials.sum()
        error = max(abs(float(x) - float(y)) for x, y in zip(expected, probabilities, strict=True))
        max_error = max(max_error, error)
        _need(
            error <= 1e-5 and labels[int(np.argmax(expected))] == answer["choice"],
            "native FP64 replay differs",
        )
        entropy = -sum(v * math.log(v) for v in probabilities if v > 0)
        confidence = (
            1.0 if len(labels) == 1 else max(0.0, min(1.0, 1.0 - entropy / math.log(len(labels))))
        )
        actual = answer.get("confidence")
        _need(
            type(actual) in (int, float)
            and math.isfinite(actual)
            and abs(actual - confidence) <= 1e-5,
            "native confidence differs",
        )
    return heads, criteria, matrix, max_error


def merge(old, accepted_old, admitted, shards):
    """Check every source join, replay each head, then append by the original masks.

    Inputs are memory objects. The result is quarantined and is not an audit
    admission, a training authorization or model-quality evidence.
    """
    prefix = _old_prefix(old, accepted_old)
    rows = pipeline.compile_wire_projection(admitted)
    _need(
        type(rows) is list and len(rows) == NEW_ROWS, "complete original 72-row collection required"
    )
    _need(type(shards) is list and bool(shards), "complete ordered shards required")
    identities = [
        {"global_ordinal": i, "case_id": row["case_id"], "wire_sha256": row["wire_sha256"]}
        for i, row in enumerate(rows)
    ]
    _need(
        len({item["case_id"] for item in identities}) == NEW_ROWS
        and len({item["wire_sha256"] for item in identities}) == NEW_ROWS
        and all(_pin(item["wire_sha256"]) for item in identities),
        "duplicate original case or wire",
    )
    old_wires, old_cases = _known_old_identities([prefix[head]["row_map"] for head in HEADS])
    old_wires.update(value for head in HEADS for value in prefix[head]["wire_sha256_ordered"])
    _need(
        not any(
            item["wire_sha256"] in old_wires or item["case_id"] in old_cases for item in identities
        ),
        "new case or wire duplicates an identified old row",
    )
    identity_sha = _sha(_encode(identities))
    cursor, material, worst = 0, {head: [] for head in HEADS}, 0.0
    baseline = prefix["operation"]["baseline"]
    evidence = []
    for index, shard in enumerate(shards):
        _need(
            type(shard) is dict
            and shard.get("schema") == features.SCHEMA
            and shard.get("status") == "captured_pending_independent_fp64_audit"
            and shard.get("fit_eligible") is False
            and shard.get("model_score") is None,
            "pending native capture required",
        )
        _need(
            shard.get("index") == index
            and shard.get("start") == cursor
            and type(shard.get("stop")) is int
            and cursor < shard["stop"] <= NEW_ROWS
            and shard.get("total") == NEW_ROWS
            and shard.get("ordered_identity_sha256") == identity_sha
            and shard.get("receipt_sha256") == admitted.receipt_sha256
            and shard.get("plan_sha256") == admitted.plan_sha256,
            "shard coverage or collection identity differs",
        )
        _need(
            shard.get("model_id") == features.MODEL_ID
            and shard.get("revision") == features.REVISION
            and shard.get("manifest_sha256") == features.MANIFEST_SHA256
            and shard.get("model_sha256") == features.MODEL_SHA256
            and shard.get("baseline_sha256") == W0_SHA256,
            "pinned publisher identity differs",
        )
        _need(
            _bytes(_array(shard["baseline"], (CAPACITY, WIDTH), np.float32)) == _bytes(baseline),
            "same W0 bytes required",
        )
        cases = shard.get("cases")
        _need(
            type(cases) is list and len(cases) == shard["stop"] - cursor and len(cases) <= 40,
            "shard cases differ",
        )
        for case in cases:
            row = rows[cursor]
            _need(
                row.get("fit_eligible") is False
                and case.get("global_ordinal") == cursor
                and case.get("case_id") == row["case_id"],
                "original row sequence differs",
            )
            heads, criteria, matrix, error = _native_audit(case, row, baseline)
            labels = _validate_supervision(row, heads, criteria)
            worst = max(worst, error)
            for head, target in labels.items():
                pos = heads.index(head)
                material[head].append(
                    (
                        matrix[pos].copy(),
                        target,
                        len(criteria[head]),
                        {
                            "case_id": row["case_id"],
                            "wire_sha256": row["wire_sha256"],
                            "global_ordinal": cursor,
                            "head_id": head,
                            "criteria_order": list(criteria[head]),
                            "loss_mask": copy.deepcopy(row["loss_mask"]),
                        },
                    )
                )
            evidence.append(
                {
                    "global_ordinal": cursor,
                    "case_id": row["case_id"],
                    "wire_sha256": row["wire_sha256"],
                    "features_sha256": case["features_fp32_sha256"],
                    "raw_response_sha256": case["raw_response_sha256"],
                }
            )
            cursor += 1
    _need(cursor == NEW_ROWS, "missing feature rows")
    result = {}
    for head in HEADS:
        initial = prefix[head]
        added = material[head]
        x = np.concatenate(
            (
                initial["features"],
                np.asarray([item[0] for item in added], dtype=np.float32).reshape(-1, WIDTH),
            )
        )
        y = np.concatenate(
            (
                initial["targets"],
                np.asarray([item[1] for item in added], dtype=np.float32).reshape(-1, CAPACITY),
            )
        )
        counts = np.concatenate(
            (initial["counts"], np.asarray([item[2] for item in added], dtype=np.int64))
        )
        mapping = copy.deepcopy(initial["row_map"]) + [item[3] for item in added]
        _need(
            all(
                _bytes(value[: OLD_ROWS[head]]) == _bytes(initial[key])
                for key, value in (("features", x), ("targets", y), ("counts", counts))
            )
            and _encode(mapping[: OLD_ROWS[head]]) == _encode(initial["row_map"]),
            "old prefix changed",
        )
        for value in (x, y, counts):
            value.flags.writeable = False
        result[head] = {
            "features": x,
            "targets": y,
            "counts": counts,
            "baseline": baseline,
            "row_map": mapping,
            "row_mapping_sha256": _sha(_mapping_encode(mapping)),
        }
    _old_prefix(old, accepted_old)
    return {
        **result,
        "baseline": baseline,
        "status": "merged_pending_external_admission",
        "fit_eligible": False,
        "fit_authorized": False,
        "scores": None,
        "new_rows": {head: len(material[head]) for head in HEADS},
        "ordered_identity_sha256": identity_sha,
        "source_evidence_sha256": _sha(_encode(evidence)),
        "prior_wire_dedup_coverage": "complete caller-supplied per-row inventories; external provenance verification required",
        "max_fp64_probability_error": worst,
    }
