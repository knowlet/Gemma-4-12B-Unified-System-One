"""Aggregate diagnostics over externally admitted TRAIN-only operation rows.

This pure in-memory helper opens no files, trains no model and runs no GPU. The
caller is responsible for authenticating a complete TRAIN-only source, array
hashes and the candidate. A caller-supplied flag is an assertion, not proof of
data provenance. Outputs are training diagnostics, never benchmark scores.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np


def _need(value, message):
    if not value:
        raise ValueError(message)


def _criteria(entry):
    _need(type(entry) is dict, "training row metadata required")
    while "source_row" in entry:
        entry = entry["source_row"]
        _need(type(entry) is dict, "nested training row metadata required")
    values = entry.get("criteria_order", entry.get("ordered_criteria_ids"))
    _need(
        type(values) is list
        and values
        and len(set(values)) == len(values)
        and all(type(value) is str and value for value in values),
        "unique original ordered operation IDs required",
    )
    _need(entry.get("head_id") == "operation", "operation rows only")
    return values


def analyze_train(*, features, targets, counts, baseline, delta, row_map, old_rows, train_only):
    """Compare original/candidate FP64 masked projections without per-row output."""
    _need(train_only is True, "explicit externally verified TRAIN-only assertion required")
    _need(type(features) is np.ndarray and features.ndim == 2, "training feature matrix required")
    n, width = features.shape
    _need(
        type(old_rows) is int and 0 <= old_rows <= n and n > 0, "training split boundary required"
    )
    _need(
        type(targets) is np.ndarray
        and type(counts) is np.ndarray
        and type(baseline) is np.ndarray
        and type(delta) is np.ndarray
        and targets.ndim == 2
        and targets.shape[0] == n
        and counts.shape == (n,)
        and baseline.shape == delta.shape == (targets.shape[1], width)
        and type(row_map) is list
        and len(row_map) == n,
        "aligned complete training matrices required",
    )
    _need(
        all(np.isfinite(value).all() for value in (features, targets, baseline, delta))
        and np.issubdtype(counts.dtype, np.integer)
        and np.all(counts >= 1)
        and np.all(counts <= targets.shape[1]),
        "finite matrices and legal operation counts required",
    )
    orders = [_criteria(entry) for entry in row_map]
    _need(
        all(len(menu) == int(count) for menu, count in zip(orders, counts, strict=True)),
        "original operation counts differ",
    )
    tail = np.arange(targets.shape[1])[None, :] >= counts[:, None]
    _need(
        (targets >= 0).all()
        and (targets[tail] == 0).all()
        and np.allclose(targets.sum(axis=1, dtype=np.float64), 1, atol=1e-6, rtol=0),
        "complete original soft labels and zero illegal tail required",
    )
    x = features.astype(np.float64)
    original = x @ baseline.astype(np.float64).T
    candidate = original + x @ delta.astype(np.float64).T
    original[tail] = candidate[tail] = -np.inf
    valid = targets > 0
    original_choice = np.argmax(original, axis=1)
    candidate_choice = np.argmax(candidate, axis=1)
    indices = np.arange(n)
    positions = defaultdict(Counter)
    predicted = {"original": Counter(), "candidate": Counter()}
    teacher = Counter()
    by_teacher = defaultdict(
        lambda: {
            "rows": 0,
            "original_hits": 0,
            "candidate_hits": 0,
            "original_click": 0,
            "candidate_click": 0,
        }
    )
    groups = {"old": [], "new": []}
    for i, menu in enumerate(orders):
        for j, name in enumerate(menu):
            positions[j][name] += 1
        chosen_original, chosen_candidate = menu[original_choice[i]], menu[candidate_choice[i]]
        predicted["original"][chosen_original] += 1
        predicted["candidate"][chosen_candidate] += 1
        groups["old" if i < old_rows else "new"].append(i)
        accepted = [menu[index] for index in np.flatnonzero(valid[i])]
        if len(accepted) == 1:
            name = accepted[0]
            teacher[name] += 1
            row = by_teacher[name]
            row["rows"] += 1
            row["original_hits"] += int(valid[i, original_choice[i]])
            row["candidate_hits"] += int(valid[i, candidate_choice[i]])
            row["original_click"] += int(chosen_original == "CLICK")
            row["candidate_click"] += int(chosen_candidate == "CLICK")
    ambiguous = {index for index, values in positions.items() if len(values) > 1}

    def metrics(selected):
        if not selected:
            return {
                "rows": 0,
                "original_hits": None,
                "candidate_hits": None,
                "original_ce": None,
                "candidate_ce": None,
            }
        selected = np.asarray(selected, dtype=np.int64)
        result = {"rows": int(len(selected))}
        for label, logits, choice in (
            ("original", original, original_choice),
            ("candidate", candidate, candidate_choice),
        ):
            subset = logits[selected]
            maximum = np.max(subset, axis=1)
            logsum = maximum + np.log(np.exp(subset - maximum[:, None]).sum(axis=1))
            safe_logits = np.where(np.isfinite(subset), subset, 0.0)
            ce = logsum - np.sum(targets[selected].astype(np.float64) * safe_logits, axis=1)
            result[label + "_hits"] = int(valid[selected, choice[selected]].sum())
            result[label + "_ce"] = float(np.mean(ce))
        return result

    _need(
        all(math.isfinite(value) for value in original[~tail])
        and all(math.isfinite(value) for value in candidate[~tail]),
        "nonfinite projections refused",
    )
    return {
        "scope": "externally admitted TRAIN-only; not a benchmark or browser score",
        "rows": n,
        "groups": {name: metrics(rows) for name, rows in groups.items()},
        "candidate_count_groups": {
            str(k): metrics(np.flatnonzero(counts == k).tolist())
            for k in sorted(set(map(int, counts)))
        },
        "single_label_teacher_counts": dict(sorted(teacher.items())),
        "multi_label_rows": int(sum(len(np.flatnonzero(valid[i])) > 1 for i in indices)),
        "single_label_metrics": dict(sorted(by_teacher.items())),
        "predicted_operation_counts": {
            name: dict(sorted(values.items())) for name, values in predicted.items()
        },
        "ordinal_operation_counts": {
            str(index): dict(sorted(values.items())) for index, values in sorted(positions.items())
        },
        "ambiguous_ordinal_count": len(ambiguous),
        "rows_with_ambiguous_ordinal": sum(
            any(index in ambiguous for index in range(len(menu))) for menu in orders
        ),
    }
