"""Hard, soft and ordinal metrics with explicit sample counts and tie conventions."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from s1.contracts import validate_distribution

NLL_FLOOR = 1e-12


def _mean(values):
    return float(np.mean(values)) if values else None


def binary_metrics(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    order = np.argsort(-scores, kind="stable")
    tp = fp = 0
    previous_recall = previous_fpr = 0.0
    ap = auc = 0.0
    for value in sorted(set(scores.tolist()), reverse=True):
        block = scores[order] == value
        tp += int(labels[order][block].sum())
        fp += int((~labels[order][block]).sum())
        recall = tp / positives if positives else 0
        fpr = fp / negatives if negatives else 0
        ap += (recall - previous_recall) * tp / (tp + fp)
        auc += (fpr - previous_fpr) * (recall + previous_recall) / 2
        previous_recall, previous_fpr = recall, fpr
    predicted = scores > 0.5  # Matches first-maximum [false, true] ties.
    return {
        "n": len(scores),
        "positives": positives,
        "negatives": negatives,
        "auroc": auc if positives and negatives else None,
        "average_precision": ap if positives else None,
        "fpr": float((predicted & ~labels).sum()) / negatives if negatives else None,
        "fnr": float((~predicted & labels).sum()) / positives if positives else None,
        "threshold": 0.5,
        "positive_label": "true",
        "tie_label": "false",
    }


def risk_coverage(confidence, correct):
    """Expected prefix risk under a random ordering inside equal-confidence blocks."""
    ordered = sorted(zip(confidence, correct), reverse=True)
    risks, accepted = [], 0
    expected_correct = 0.0
    for value in sorted(set(confidence), reverse=True):
        block = [float(ok) for conf, ok in ordered if conf == value]
        rate = float(np.mean(block))
        for offset in range(1, len(block) + 1):
            risks.append(1 - (expected_correct + offset * rate) / (accepted + offset))
        accepted += len(block)
        expected_correct += sum(block)
    return {
        "n": len(correct),
        "aurc": _mean(risks),
        "points": [
            {"coverage": (i + 1) / len(risks), "risk": risk} for i, risk in enumerate(risks)
        ],
        "accuracy_at_80": 1 - risks[max(1, round(0.8 * len(risks))) - 1] if risks else None,
        "accuracy_at_50": 1 - risks[max(1, round(0.5 * len(risks))) - 1] if risks else None,
        "tie_policy": "expected_within_equal_confidence",
    }


def _macro_f1(rows):
    labels = sorted(
        {r["gold"] for r in rows}
        | {r["actual_label"] for r in rows}
        | {label for r in rows for label in r.get("labels", r.get("probabilities") or {})}
    )
    values = []
    for label in labels:
        tp = sum(r["gold"] == label == r["actual_label"] for r in rows)
        fp = sum(r["actual_label"] == label != r["gold"] for r in rows)
        fn = sum(r["gold"] == label != r["actual_label"] for r in rows)
        values.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return _mean(values)


def quality_metrics(records):
    rows = [r for r in records if r["status"] == "ok"]
    probabilistic, hard_nll, brier, confidence, correct = [], [], [], [], []
    soft_nll, soft_brier, soft_kl, teacher_agreement = [], [], [], []
    ordinal_error, rps, mae = [], [], []
    binary_scores, binary_gold = [], []
    zero_gold = 0
    for row in rows:
        dist = row.get("probabilities")
        if dist is None:
            continue
        labels = row.get("labels") or list(dist)
        if set(dist) != set(labels) or row["gold"] not in labels:
            raise ValueError("prediction probabilities/gold do not match labels")
        p = np.asarray(validate_distribution([dist[key] for key in labels], len(labels)))
        y = labels.index(row["gold"])
        onehot = np.eye(len(p))[y]
        hard_nll.append(float(-np.log(max(NLL_FLOOR, p[y]))))
        zero_gold += int(p[y] == 0)
        brier.append(float(np.square(p - onehot).sum()))
        confidence.append(float(p.max()))
        correct.append(row["actual_label"] == row["gold"])
        probabilistic.append(row)
        target = row.get("soft_gold")
        if target is not None:
            if set(target) != set(labels):
                raise ValueError("soft target labels differ from prediction")
            q = np.asarray(validate_distribution([target[key] for key in labels], len(labels)))
            logp = np.log(np.maximum(p, NLL_FLOOR))
            soft_nll.append(float(-(q * logp).sum()))
            soft_brier.append(float(np.square(p - q).sum()))
            soft_kl.append(float((q * (np.log(np.maximum(q, NLL_FLOOR)) - logp)).sum()))
            teacher_agreement.append(int(p.argmax()) == int(q.argmax()))
        if row["type"] == "noul":
            binary_scores.append(float(dist["true"]))
            binary_gold.append(row["gold"] == "true")
        if row["type"] == "score":
            ordinal_error.append(abs(int(p.argmax()) - y))
            rps.append(float(np.square(np.cumsum(p - onehot)[:-1]).sum() / (len(p) - 1)))
            if "score" in row and "gold_score" in row:
                mae.append(abs(row["score"] - row["gold_score"]))
    bins, ece = [], 0.0
    for index in range(15):
        selected = [i for i, conf in enumerate(confidence) if min(int(conf * 15), 14) == index]
        mean_conf = _mean([confidence[i] for i in selected])
        accuracy = _mean([correct[i] for i in selected])
        if selected:
            ece += len(selected) / len(confidence) * abs(mean_conf - accuracy)
        bins.append(
            {
                "lower": index / 15,
                "upper": (index + 1) / 15,
                "n": len(selected),
                "confidence": mean_conf,
                "accuracy": accuracy,
            }
        )
    task_rows, schema_rows, requests = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        if row.get("task_id"):
            task_rows[row["task_id"]].append(row)
            schema_rows[
                (row["task_id"], row.get("schema_id") or row["question_id"], row["type"])
            ].append(row)
    for row in records:
        requests[row["request_id"]].append(row)
    valid_requests = [
        group for group in requests.values() if all(r["status"] == "ok" for r in group)
    ]
    critical = [
        r
        for r in records
        if r.get("critical") and r["eligibility"] == "eligible" and r["status"] != "not_run"
    ]
    slices = {}
    for field in ("task_id", "language", "type"):
        values = sorted({str(row[field]) for row in rows if row.get(field) is not None})
        slices[field] = {
            value: {
                "n": len(selected := [r for r in rows if str(r.get(field)) == value]),
                "accuracy": _mean([r["actual_label"] == r["gold"] for r in selected]),
            }
            for value in values
        }
    return {
        "n": len(rows),
        "probability_n": len(probabilistic),
        "nll": _mean(hard_nll),
        "nll_floor": NLL_FLOOR,
        "zero_gold_probability_n": zero_gold,
        "brier_sum": _mean(brier),
        "ece_15": ece if confidence else None,
        "reliability": bins,
        "selective": risk_coverage(confidence, correct),
        "noul": binary_metrics(binary_scores, binary_gold),
        "score": {
            "n": len(rps),
            "mae_n": len(mae),
            "mae": _mean(mae),
            "ordinal_error": _mean(ordinal_error),
            "rps_normalized": _mean(rps),
        },
        "soft_targets": {
            "n": len(soft_nll),
            "cross_entropy": _mean(soft_nll),
            "brier_sum": _mean(soft_brier),
            "kl": _mean(soft_kl),
            "teacher_argmax_agreement": _mean(teacher_agreement),
        },
        "task_macro_accuracy": _mean(
            [_mean([r["actual_label"] == r["gold"] for r in group]) for group in task_rows.values()]
        ),
        "schema_macro_f1": [
            {
                "task_id": key[0],
                "schema_id": key[1],
                "type": key[2],
                "n": len(group),
                "macro_f1": _macro_f1(group),
            }
            for key, group in sorted(schema_rows.items())
        ],
        "all_fields_correct": {
            "valid_requests": len(valid_requests),
            "conditional_accuracy": _mean(
                [all(r["actual_label"] == r["gold"] for r in group) for group in valid_requests]
            ),
        },
        "critical_fields": {
            "attempted_n": len(critical),
            "error_rate": _mean(
                [r["status"] != "ok" or r["actual_label"] != r["gold"] for r in critical]
            ),
        },
        "slices": slices,
    }
