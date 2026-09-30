"""Paired source-group bootstrap; repeated slots never become independent units."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .artifacts import read_records


def cluster_interval(values, groups, *, seed=0, resamples=2000, confidence=0.95):
    if len(values) != len(groups) or resamples < 100 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap inputs")
    clusters = defaultdict(list)
    for value, group in zip(values, groups):
        if not np.isfinite(value):
            raise ValueError("bootstrap values must be finite")
        clusters[group].append(float(value))
    keys = sorted(clusters)
    result = {
        "estimate": float(np.mean(values)) if len(values) else None,
        "groups": len(keys),
        "observations": len(values),
        "confidence": confidence,
        "seed": seed,
        "resamples": resamples,
        "interval": None,
        "status": "inconclusive",
    }
    if len(keys) < 2:
        return result
    totals = np.array([sum(clusters[key]) for key in keys])
    counts = np.array([len(clusters[key]) for key in keys])
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(resamples):
        sample = rng.integers(0, len(keys), len(keys))
        estimates.append(float(totals[sample].sum() / counts[sample].sum()))
    alpha = (1 - confidence) / 2
    result.update(interval=np.quantile(estimates, [alpha, 1 - alpha]).tolist(), status="estimated")
    return result


def _key(row):
    return row["case_id"], row["question_id"], row["repetition"]


def _population_identity(keys, index):
    """Pin decision slots and source groups without including observed answers."""
    members = [[*key, index[key]["group_id"]] for key in sorted(keys)]
    return hashlib.sha256(
        json.dumps(members, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def paired_comparison(left, right, *, seed=0, resamples=2000):
    indices = []
    for rows in (left, right):
        index = {_key(row): row for row in rows}
        if len(index) != len(rows):
            raise ValueError("duplicate decision identities")
        indices.append(index)
    a, b = indices
    common = sorted(a.keys() & b.keys())
    for key in common:
        if any(
            a[key].get(field) != b[key].get(field)
            for field in ("gold", "group_id", "task_id", "schema_id", "type")
        ):
            raise ValueError("paired observations disagree on gold/group/task metadata")
    successful = [key for key in common if a[key]["status"] == b[key]["status"] == "ok"]
    jointly_eligible = [
        key for key in common if a[key]["eligibility"] == b[key]["eligibility"] == "eligible"
    ]
    eligible = [
        key
        for key in jointly_eligible
        if a[key]["status"] != "not_run" and b[key]["status"] != "not_run"
    ]
    population = a | b
    population_evidence = {
        "decisions": len(population),
        "groups": len({row["group_id"] for row in population.values()}),
        "identity_sha256": _population_identity(population, population),
        "jointly_eligible_decisions": len(jointly_eligible),
        "jointly_eligible_groups": len({a[key]["group_id"] for key in jointly_eligible}),
        "jointly_eligible_identity_sha256": _population_identity(jointly_eligible, a),
        "eligibility_mismatches": sum(
            a[key]["eligibility"] != b[key]["eligibility"] for key in common
        ),
        "left_only": len(a.keys() - b.keys()),
        "right_only": len(b.keys() - a.keys()),
        "left_not_run": sum(row["status"] == "not_run" for row in left),
        "right_not_run": sum(row["status"] == "not_run" for row in right),
    }

    def interval(keys):
        values = [
            int(b[key]["status"] == "ok" and b[key]["actual_label"] == b[key]["gold"])
            - int(a[key]["status"] == "ok" and a[key]["actual_label"] == a[key]["gold"])
            for key in keys
        ]
        return cluster_interval(
            values, [a[key]["group_id"] for key in keys], seed=seed, resamples=resamples
        )

    return {
        "direction": "right_minus_left",
        "left_decisions": len(left),
        "right_decisions": len(right),
        "common_decisions": len(common),
        "left_only": len(a.keys() - b.keys()),
        "right_only": len(b.keys() - a.keys()),
        "population": population_evidence,
        "conditional_accuracy_delta": interval(successful),
        "operational_accuracy_delta": interval(eligible),
        "left_valid": sum(r["status"] == "ok" for r in left),
        "right_valid": sum(r["status"] == "ok" for r in right),
        "multiple_comparisons_adjusted": False,
    }


def compare_runs(directories, *, seed=0, resamples=2000):
    entries, fingerprints, views = [], set(), set()
    for number, directory in enumerate(directories):
        path = Path(directory)
        manifest = json.loads((path / "run_manifest.json").read_text())
        if manifest["status"] in ("running", "interrupted"):
            raise ValueError("formal comparison requires a finished run")
        fingerprints.add(manifest["plan"]["dataset_sha256"])
        rows = read_records(path, "predictions")
        for cell in manifest["plan"]["models"]:
            state = manifest["execution"][cell["model_id"]]["status"]
            if state not in ("completed", "completed_with_errors"):
                continue
            suite = cell["identity"]["suite"]
            views.add((suite["track"], suite["information_view"]))
            subset = [row for row in rows if row["model_id"] == cell["model_id"]]
            expected = (
                sum(cell["coverage"]["decisions"].values())
                * cell["identity"]["profile"]["repetitions"]
            )
            if len(subset) != expected:
                raise ValueError("comparison requires complete decision records")
            entries.append((f"{number}:{cell['model_id']}", subset, cell["experiment_id"]))
    if len(fingerprints) != 1 or len(views) != 1 or len(entries) < 2:
        raise ValueError(
            "comparison requires two executed models on identical data and information views"
        )
    baseline = entries[0]
    return {
        "schema_version": 2,
        "dataset_sha256": fingerprints.pop(),
        "baseline": baseline[0],
        "comparisons": [
            {
                "left": baseline[0],
                "right": name,
                "left_experiment_id": baseline[2],
                "right_experiment_id": identity,
                **paired_comparison(baseline[1], rows, seed=seed, resamples=resamples),
            }
            for name, rows, identity in entries[1:]
        ],
    }
