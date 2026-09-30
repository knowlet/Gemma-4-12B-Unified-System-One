"""Offline workload preparation and paired behavioral comparisons."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from .artifacts import read_records, write_json
from .datasets import EvaluationCase, fingerprint, load_cases
from .statistics import cluster_interval


def write_cases(path, cases):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for case in cases:
            stream.write(case.model_dump_json(exclude_none=True) + "\n")


def perturb_dataset(source, output, *, seed=0):
    """Preserve labels/IDs/groups while permuting both questions and Choice options."""
    from s1.training import permute_request

    rng = random.Random(seed)
    cases = load_cases(source)
    for case in cases:
        case.request = permute_request(case.request, rng)
        rng.shuffle(case.request.questions)
    write_cases(output, cases)
    return {
        "cases": len(cases),
        "dataset_sha256": fingerprint(cases),
        "seed": seed,
        "transformation": "question_and_choice_order",
    }


def prepare_sweep(output, *, seed=0, cases_per_cell=8):
    """One-axis N/K/length/reuse stress fixtures; never a representative quality suite."""
    if cases_per_cell < 1:
        raise ValueError("cases_per_cell must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cells = []
    axes = {
        "n": [1, 2, 4, 8, 16, 32, 64],
        "k": [2, 4, 8, 16, 32, 52],
        "state_chars": [128, 512, 2048, 8192, 16384],
        "reuse_percent": [0, 50, 95],
    }
    for axis, values in axes.items():
        for value in values:
            settings = {"n": 1, "k": 4, "state_chars": 128, "reuse_percent": 0, axis: value}
            cases = []
            reused = round(cases_per_cell * settings["reuse_percent"] / 100)
            for index in range(cases_per_cell):
                source = 0 if index < reused else index
                state = f"Source {source}; select option-0. "
                state += "Neutral background. " * max(
                    0, (settings["state_chars"] - len(state)) // 20
                )
                questions = [
                    {
                        "id": f"q{q}",
                        "type": "choice",
                        "instructions": "Select the named option.",
                        "criteria": {f"option-{k}": f"Option {k}" for k in range(settings["k"])},
                    }
                    for q in range(settings["n"])
                ]
                cases.append(
                    EvaluationCase.model_validate(
                        {
                            "id": f"case-{index}",
                            "group_id": f"source-{source}",
                            "task_id": "synthetic-shape",
                            "schema_id": f"k-{settings['k']}",
                            "language": "en",
                            "split": "test",
                            "request": {"state": state, "questions": questions},
                            "gold": {q["id"]: "option-0" for q in questions},
                        }
                    )
                )
            name = f"{axis}-{value}"
            write_cases(output / f"{name}.jsonl", cases)
            cells.append(
                {
                    "id": name,
                    **settings,
                    "dataset": f"{name}.jsonl",
                    "dataset_sha256": fingerprint(cases),
                    "cases": len(cases),
                }
            )
    manifest = {
        "schema_version": 2,
        "kind": "synthetic_systems_workloads",
        "seed": seed,
        "cells": cells,
        "batch_sizes": [1, 2, 4, 8, 16, 32],
        "concurrency": [1, 4, 16, 64],
        "length_unit": "characters, not tokenizer tokens",
        "result_cache": False,
        "quality_claim": False,
    }
    write_json(output / "sweep.json", manifest)
    return manifest


def behavior_comparison(left_directory, right_directory, left_model, right_model, *, seed=0):
    """Explicitly allows paired counterfactual inputs; reports failures and coverage."""

    def selected(path, model):
        records = [r for r in read_records(path, "predictions") if r["model_id"] == model]
        keys = {(r["case_id"], r["question_id"], r["repetition"]): r for r in records}
        if not records or len(keys) != len(records):
            raise ValueError("missing model or duplicate decision records")
        return keys

    left, right = selected(left_directory, left_model), selected(right_directory, right_model)
    if left.keys() != right.keys():
        raise ValueError("behavior comparison requires identical paired decision IDs")
    deltas, flips, probability_deltas, groups = [], [], [], []
    for key in sorted(left):
        a, b = left[key], right[key]
        if a["group_id"] != b["group_id"] or a["gold"] != b["gold"]:
            raise ValueError("order/retention comparisons require matched golds and groups")
        if a["status"] != "ok" or b["status"] != "ok":
            continue
        if set(a["labels"]) != set(b["labels"]):
            raise ValueError("candidate label sets differ")
        flips.append(a["actual_label"] != b["actual_label"])
        deltas.append(int(b["actual_label"] == b["gold"]) - int(a["actual_label"] == a["gold"]))
        groups.append(a["group_id"])
        if a["probabilities"] is not None and b["probabilities"] is not None:
            probability_deltas.append(
                max(
                    abs(a["probabilities"][label] - b["probabilities"][label])
                    for label in a["labels"]
                )
            )
    return {
        "paired_decisions": len(left),
        "both_valid": len(deltas),
        "not_both_valid": len(left) - len(deltas),
        "choice_flip_rate": float(np.mean(flips)) if flips else None,
        "max_probability_delta": max(probability_deltas) if probability_deltas else None,
        "accuracy_delta": cluster_interval(deltas, groups, seed=seed) if deltas else None,
    }
