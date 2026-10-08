"""Offline outcome audit for the eight prospectively fixed research profiles.

No model invocation, fitting, recipe selection, rank, or formal acceptance is
performed. Missing/failed populations remain visible. Official public and
coherence source roots enable the existing independent raw-answer replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "3a8dcceba9fbc0c7ce31bddbb85c0d10acdfa855ddf237f4d5e81982ec6a926d"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_jevbench_public import load_public  # noqa: E402
from summarize_jevbench_campaign import (  # noqa: E402
    _metric_equal,
    audit_public,
    paired_accuracy,
    replay_coherence,
)

from s1.evaluation.datasets import load_cases  # noqa: E402
from s1.evaluation.statistics import cluster_interval  # noqa: E402


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def inventory(directory):
    root = Path(directory)
    return {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def expected_populations():
    return {
        "synthetic_calibration": 192,
        "synthetic_test": 384,
        "public_per_policy": 231,
        "release_boolq": 256,
        "release_native": 52,
        "historic_text": 128,
        "native_fixture_cases": 8,
        "native_fixture_questions": 29,
        "coherence_cases_per_policy": 240,
        "coherence_checks_per_policy": 50,
        "coherence_planned_tests_per_policy": 1200,
        "coherence_cached_answers_per_policy": 1248,
    }


def guarded(action):
    try:
        return action()
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return {"status": "integrity_failed", "complete": False, "error": str(exc)}


def index_rows(rows):
    result = {}
    for row in rows:
        key = row["case_id"], row["question_id"]
        require(key not in result, "duplicate case/question record")
        result[key] = row
    return result


def row_metrics(row, temperature):
    """Recompute one soft-target decision without trusting saved aggregates."""
    require(
        type(temperature) in (int, float) and math.isfinite(temperature) and temperature > 0,
        "positive finite temperature required",
    )
    logits, truth = np.asarray(row["logits"], dtype=float), np.asarray(row["target"], dtype=float)
    require(
        logits.ndim == truth.ndim == 1 and len(logits) == len(truth) == len(row["labels"]),
        "candidate logits, labels and targets differ",
    )
    require(
        np.isfinite(logits).all()
        and np.isfinite(truth).all()
        and np.all(truth >= 0)
        and math.isclose(float(truth.sum()), 1.0, abs_tol=1e-9),
        "invalid soft target or logits",
    )
    logits = logits / temperature
    shifted = logits - logits.max()
    logp = shifted - np.log(np.exp(shifted).sum())
    probs = np.exp(logp)
    entropy = -sum(p * math.log(p) for p in truth if p > 0)
    result = {
        "soft_ce": float(-(truth * logp).sum()),
        "kl": float(-(truth * logp).sum() - entropy),
        "brier": float(np.square(probs - truth).sum()),
        "oracle_argmax_match": float(int(probs.argmax()) == int(truth.argmax())),
    }
    if row["type"] == "score":
        levels = np.asarray(row["score_values"], dtype=float)
        require(
            levels.shape == probs.shape
            and np.isfinite(levels).all()
            and np.all(np.diff(levels) > 0),
            "invalid ordinal values",
        )
        result["expected_score_mae"] = float(abs(np.dot(probs - truth, levels)))
    return result


def soft_summary(rows, temperatures):
    result = {}
    for kind in ("all", "choice", "noul", "score"):
        subset = rows if kind == "all" else [r for r in rows if r["type"] == kind]
        values = [row_metrics(row, temperatures[row["type"]]) for row in subset]
        result[kind] = {"n": len(values)}
        for metric in sorted({k for value in values for k in value}):
            result[kind][metric] = float(np.mean([v[metric] for v in values if metric in v]))
    return result


def validate_soft_records(rows, cases, split):
    expected = {(c.id, q.id): (c, q) for c in cases for q in c.request.questions}
    observed = index_rows(rows)
    require(set(observed) == set(expected), f"incomplete or unexpected {split} population")
    for key, row in observed.items():
        case, question = expected[key]
        require(
            row["group_id"] == (case.group_id or case.id)
            and row["split"] == split
            and row["type"] == question.type
            and row["labels"] == question.labels(),
            "record source/split/type/label identity differs from frozen data",
        )
        require(
            row["target"] == [case.soft_gold[question.id][label] for label in question.labels()],
            "record soft targets differ from frozen data",
        )
        values = question.score_values() if question.type == "score" else None
        require(row["score_values"] == values, "record ordinal values differ from frozen data")
        row_metrics(row, 1.0)
    return observed


def audit_synthetic(directory, calibration_cases, test_cases):
    root = Path(directory)
    cal, heldout = read(root / "calibration.json"), read(root / "heldout.json")
    validate_soft_records(cal["records"], calibration_cases, "calibration")
    rows = heldout["records"]
    validate_soft_records(rows, test_cases, "test")
    temperatures = {}
    for kind in ("choice", "noul", "score"):
        selected = [row for row in cal["records"] if row["type"] == kind]
        require(bool(selected), "calibration does not cover all types")
        grid = np.geomspace(0.1, 10.0, 81)
        losses = [
            np.mean([row_metrics(row, float(t))["soft_ce"] for row in selected]) for t in grid
        ]
        expected = float(grid[int(np.argmin(losses))])
        saved = cal["temperatures"][kind]
        require(
            saved["temperature"] == expected and saved["n"] == len(selected),
            "calibration temperature/population differs from fixed soft-CE grid",
        )
        require(
            math.isclose(
                saved["soft_ce_before"],
                float(np.mean([row_metrics(row, 1.0)["soft_ce"] for row in selected])),
                abs_tol=1e-12,
            )
            and math.isclose(saved["soft_ce_after"], float(min(losses)), abs_tol=1e-12),
            "saved calibration losses differ",
        )
        require(
            saved["boundary"] == (expected in (float(grid[0]), float(grid[-1]))),
            "calibration boundary status differs",
        )
        temperatures[kind] = expected
    raw, calibrated = (
        soft_summary(rows, {k: 1.0 for k in temperatures}),
        soft_summary(rows, temperatures),
    )
    require(
        _metric_equal(heldout["raw_metrics"], raw)
        and _metric_equal(heldout["calibrated_metrics"], calibrated),
        "saved heldout metrics differ",
    )
    family_by_id = {case.id: case.task_id for case in test_cases}
    by_template = {}
    for family in sorted(set(family_by_id.values())):
        selected = [row for row in rows if family_by_id[row["case_id"]] == family]
        by_template[family] = soft_summary(selected, temperatures)["all"]
    return {
        "status": "verified",
        "complete": True,
        "n_calibration": len(cal["records"]),
        "n_test": len(rows),
        "temperatures": cal["temperatures"],
        "raw": raw,
        "calibrated": calibrated,
        "by_template": by_template,
        "records": rows,
        "scope": "Known finite-world probability targets; argmax is oracle match, not realized-event accuracy",
    }


def paired_soft(
    reference, candidate, reference_t, candidate_t, *, expected_keys=None, seed=42, samples=2000
):
    a, b = index_rows(reference), index_rows(candidate)
    expected = set(a) if expected_keys is None else set(expected_keys)
    common = expected & set(a) & set(b)
    population = {
        "expected": len(expected),
        "reference": len(a),
        "candidate": len(b),
        "paired": len(common),
        "reference_missing": len(expected - set(a)),
        "candidate_missing": len(expected - set(b)),
        "unexpected": len((set(a) | set(b)) - expected),
    }
    if set(a) != expected or set(b) != expected:
        return {
            "status": "inconclusive",
            "population": population,
            "reason": "complete declared heldout population is required",
            "metrics": {},
        }
    for key in expected:
        require(
            all(
                a[key][field] == b[key][field]
                for field in ("group_id", "split", "type", "labels", "target", "score_values")
            ),
            "paired source/target/label metadata differ",
        )
    output = {}
    for kind in ("all", "choice", "noul", "score"):
        keys = sorted(key for key in expected if kind == "all" or a[key]["type"] == kind)
        groups = [a[key]["group_id"] for key in keys]
        values = [
            (
                row_metrics(a[key], reference_t[a[key]["type"]]),
                row_metrics(b[key], candidate_t[b[key]["type"]]),
            )
            for key in keys
        ]
        metrics = sorted(
            {metric for pair in values for value in pair for metric in value}
            - {"oracle_argmax_match"}
        )
        output[kind] = {}
        for metric in metrics:
            selected = [
                (pair, group)
                for pair, group in zip(values, groups, strict=True)
                if metric in pair[0] and metric in pair[1]
            ]
            interval = cluster_interval(
                [pair[1][metric] - pair[0][metric] for pair, _ in selected],
                [group for _, group in selected],
                seed=seed,
                resamples=samples,
            )
            output[kind][metric] = {
                **interval,
                "direction": "candidate minus reference; lower is better",
            }
    return {
        "status": "computed",
        "population": population,
        "metrics": output,
        "scope": "Paired source-group descriptive intervals, no formal pass or multiplicity-adjusted claim",
    }


def answer_probabilities(answer, question):
    require(isinstance(answer, dict) and answer.get("type") == question.type, "answer type differs")
    probs = answer.get("probabilities")
    require(
        isinstance(probs, dict) and set(probs) == set(question.labels()),
        "answer label support differs",
    )
    require(
        all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in probs.values())
        and math.isclose(sum(probs.values()), 1.0, abs_tol=1e-6),
        "invalid native distribution",
    )
    best = max(question.labels(), key=probs.__getitem__)
    if question.type == "choice":
        require(
            answer.get("choice") == best,
            "native choice contradicts distribution",
        )
    if question.type == "noul":
        require(
            math.isclose(answer.get("noul", math.nan), probs["true"], abs_tol=1e-6),
            "native noul contradicts distribution",
        )
    if question.type == "score":
        expected = sum(
            probs[label] * value
            for label, value in zip(question.labels(), question.score_values(), strict=True)
        )
        require(
            math.isclose(answer.get("score", math.nan), expected, abs_tol=1e-6),
            "native score contradicts expected score",
        )
        require(answer.get("level") == best, "native ordinal level contradicts distribution")
    return probs


def audit_regression_records(saved, cases):
    expected = {case.id: case for case in cases}
    rows = saved["records"]
    require(len({r["case_id"] for r in rows}) == len(rows), "duplicate regression case")
    require({r["case_id"] for r in rows} == set(expected), "incomplete regression case population")
    decisions, correct = [], 0
    for row in rows:
        case = expected[row["case_id"]]
        require(row["gold"] == case.gold, "regression gold differs from pinned data")
        answers = row["response"]["answers"]
        require(
            set(answers) == {q.id for q in case.request.questions},
            "regression question population differs",
        )
        for question in case.request.questions:
            probs = answer_probabilities(answers[question.id], question)
            predicted = max(question.labels(), key=probs.__getitem__)
            hit = predicted == case.gold[question.id]
            correct += hit
            decisions.append(
                {
                    "case_id": case.id,
                    "question_id": question.id,
                    "group_id": case.group_id or case.id,
                    "correct": hit,
                    "predicted": predicted,
                    "gold": case.gold[question.id],
                    "probabilities": probs,
                }
            )
    require(
        saved["n_cases"] == len(cases)
        and saved["n_questions"] == len(decisions)
        and saved["n_correct"] == correct
        and math.isclose(saved["accuracy"], correct / len(decisions), abs_tol=1e-12),
        "regression counters differ from raw answers",
    )
    return {
        "status": "verified",
        "complete": True,
        "n_cases": len(cases),
        "n_questions": len(decisions),
        "n_correct": correct,
        "accuracy": correct / len(decisions),
        "records": decisions,
    }


def audit_native(path, cases):
    rows = read(path)
    expected = {case.id: case for case in cases}
    require(
        len({r["case_id"] for r in rows}) == len(rows)
        and {r["case_id"] for r in rows} == set(expected),
        "native fixture population differs",
    )
    decisions = []
    for row in rows:
        case = expected[row["case_id"]]
        require(
            row["modalities"] == [m.type for m in case.request.media],
            "native modality identity differs",
        )
        require(
            set(row["answers"]) == {q.id for q in case.request.questions},
            "native question population differs",
        )
        for question in case.request.questions:
            probs = answer_probabilities(row["answers"][question.id], question)
            predicted = max(question.labels(), key=probs.__getitem__)
            decisions.append(
                {
                    "case_id": case.id,
                    "question_id": question.id,
                    "group_id": case.group_id or case.id,
                    "predicted": predicted,
                    "gold": case.gold[question.id],
                    "correct": predicted == case.gold[question.id],
                    "probabilities": probs,
                }
            )
    return {
        "status": "verified",
        "complete": True,
        "n_cases": len(rows),
        "n_questions": len(decisions),
        "records": decisions,
        "scope": "Native fixture contract/regression; separate from complete release native52",
    }


def paired_native(reference, candidate, *, seed=42, samples=2000):
    a, b = index_rows(reference), index_rows(candidate)
    if set(a) != set(b):
        return {
            "status": "inconclusive",
            "reference": len(a),
            "candidate": len(b),
            "paired": len(set(a) & set(b)),
            "reason": "native population mismatch",
        }
    keys = sorted(a)
    require(
        all(
            a[k]["group_id"] == b[k]["group_id"]
            and a[k]["gold"] == b[k]["gold"]
            and set(a[k]["probabilities"]) == set(b[k]["probabilities"])
            for k in keys
        ),
        "native paired source/gold/labels differ",
    )
    return {
        "status": "computed",
        "n_paired": len(keys),
        "label_flips": sum(a[k]["predicted"] != b[k]["predicted"] for k in keys),
        "max_probability_drift": max(
            (
                abs(a[k]["probabilities"][label] - b[k]["probabilities"][label])
                for k in keys
                for label in a[k]["probabilities"]
            ),
            default=0.0,
        ),
        "accuracy_delta": cluster_interval(
            [int(b[k]["correct"]) - int(a[k]["correct"]) for k in keys],
            [a[k]["group_id"] for k in keys],
            seed=seed,
            resamples=samples,
        ),
    }


def audit_coherence(path, cache_root, *, source_root=None, backend=None):
    report = read(path)
    scores, result = report["scores"], report["result"]
    require(backend is None or result["backend"] == backend, "coherence backend identity differs")
    checks = scores.get("checks", {})
    paths = sorted(Path(cache_root).glob("*.json"))
    hashes = {p.name: digest(p) for p in paths if p.is_file() and not p.is_symlink()}
    for p in paths:
        require(
            len(p.stem) == 64 and all(c in "0123456789abcdef" for c in p.stem), "invalid cache name"
        )
        json.dumps(read(p), allow_nan=False)
    complete = (
        scores.get("n_cases") == 240
        and scores.get("n_checks") == 50
        and len(checks) == 50
        and not scores.get("incomplete")
        and not result.get("errors")
        and sum(c["planned"] for c in checks.values()) == 1200
        and all(c["n"] == c["planned"] and c["coverage"] == 1.0 for c in checks.values())
        and len(hashes) == 1248
    )
    replay = (
        replay_coherence(report, cache_root, source_root)
        if source_root is not None and complete
        else None
    )
    bases = []
    for case in result.get("bases", {}).values():
        for distribution in case.get("base", {}).values():
            p = np.asarray(list(distribution.values()), dtype=float)
            require(
                p.ndim == 1
                and len(p) > 0
                and np.isfinite(p).all()
                and np.all(p >= 0)
                and math.isclose(float(p.sum()), 1.0, abs_tol=1e-6),
                "invalid coherence base distribution",
            )
            bases.append(
                {
                    "max_probability": float(p.max()),
                    "normalized_entropy": -sum(v * math.log(v) for v in p if v > 0)
                    / math.log(len(p))
                    if len(p) > 1
                    else 0.0,
                }
            )
    return {
        "status": "verified_raw_replay" if replay else "saved_only_not_officially_replayed",
        "complete": complete,
        "sha256": digest(path),
        "overall": scores.get("overall"),
        "ci95": (scores.get("ci") or {}).get("overall"),
        "dimensions": scores.get("pillars"),
        "n_cases": scores.get("n_cases"),
        "n_checks": scores.get("n_checks"),
        "checks": checks,
        "planned_tests": sum(c["planned"] for c in checks.values()),
        "outcome_rows": len(result.get("outcomes", [])),
        "outcome_denominator_scope": "Raw outcome rows may repeat trial/case slots; official planned/check support is the denominator",
        "errors": len(result.get("errors", [])),
        "incomplete": scores.get("incomplete"),
        "raw_cache_files": len(hashes),
        "raw_cache_sha256": hashes,
        "replay": replay,
        "base_distribution_stats": {
            "n": len(bases),
            **{
                key: float(np.mean([b[key] for b in bases]))
                for key in ("max_probability", "normalized_entropy")
            },
        }
        if bases
        else {"n": 0},
    }


def coherence_effect(calibrated, unit):
    if not calibrated.get("complete") or not unit.get("complete"):
        return {"status": "inconclusive", "reason": "both complete coherence policies required"}
    require(set(calibrated["checks"]) == set(unit["checks"]), "coherence check plans differ")
    return {
        "status": "computed",
        "direction": "calibrated minus unit replay",
        "overall_delta": calibrated["overall"] - unit["overall"],
        "dimension_deltas": {
            k: calibrated["dimensions"][k] - unit["dimensions"][k] for k in calibrated["dimensions"]
        },
        "check_deltas": {
            k: calibrated["checks"][k]["score"] - unit["checks"][k]["score"]
            for k in calibrated["checks"]
        },
        "base_distribution_deltas": {
            k: calibrated["base_distribution_stats"][k] - unit["base_distribution_stats"][k]
            for k in ("max_probability", "normalized_entropy")
        },
        "scope": "Temperature effects including LOG/MEA and uniformizing; not a new independent model run or formal improvement gate",
    }


def audit_public_profile(path, public_source):
    if public_source is None:
        return {
            "status": "not_audited",
            "complete": False,
            "reason": "pinned public source root required",
        }
    tasks, official, config = public_source
    result = audit_public(path, tasks, official, config)
    summary = result["summary"]
    require(
        summary.get("n_planned") == summary.get("n_attempted") == len(tasks),
        "public total population differs",
    )
    errors = sum(row.get("ok") is not True for row in result["records"])
    false_accept = false_reject = no_gold = yes_gold = 0
    for task, row in zip(tasks, result["records"], strict=True):
        if task.question["type"] == "noul":
            no_gold += task.expected == "no"
            yes_gold += task.expected == "yes"
            false_accept += task.expected == "no" and row.get("predicted") == "yes"
            false_reject += task.expected == "yes" and row.get("predicted") == "no"
    return {
        "status": "verified_official_raw",
        "complete": True,
        "summary": summary,
        "by_type": result["by_type"],
        "errors": errors,
        "n_valid": summary["n_valid"],
        "false_accepts": {"count": false_accept, "gold_no": no_gold},
        "false_rejects": {"count": false_reject, "gold_yes": yes_gold},
        "records": result["records"],
        "manifest": result["manifest"],
        "raw_answers": result["raw_answers"],
        "raw_runtime": {
            row["task_id"]: read(Path(path) / row["raw_path"])
            .get("response", {})
            .get("runtime", {})
            for row in result["records"]
            if row.get("ok") is True
        },
        "verified_files": result["verified_files"],
    }


def public_replay_identity(native, replay):
    if not native.get("complete") or not replay.get("complete"):
        return {"status": "inconclusive", "reason": "both audited public policies required"}
    require(
        [r["task_id"] for r in native["records"]] == [r["task_id"] for r in replay["records"]],
        "public replay task identity differs",
    )
    manifest = replay["manifest"]
    require(
        "replay" in manifest["identity"].get("execution", ""),
        "replay lacks timing scope declaration",
    )
    count, failed = 0, 0
    for index, row in enumerate(replay["records"]):
        task_id = row["task_id"]
        if row.get("ok") is not True or native["records"][index].get("ok") is not True:
            failed += 1
            continue
        first, second = native["raw_runtime"][task_id], replay["raw_runtime"][task_id]
        require(
            first.get("raw_logits") == second.get("raw_logits") and bool(first.get("raw_logits")),
            "public replay changed raw candidate logits/labels",
        )
        require(
            second.get("calibration_replay") is True and second.get("forward_calls") == 0,
            "public replay unexpectedly invoked native model",
        )
        for raw in second["raw_logits"]:
            answer = replay["raw_answers"][index][raw["id"]]
            t = manifest["identity"]["calibration"][answer["type"]]
            values = np.asarray(raw["logits"], dtype=float) / t
            values = np.exp(values - values.max())
            values /= values.sum()
            require(
                set(answer["probabilities"]) == set(raw["labels"])
                and all(
                    math.isclose(answer["probabilities"][label], float(p), abs_tol=1e-6)
                    for label, p in zip(raw["labels"], values, strict=True)
                ),
                "public replay probabilities do not match fixed temperature/raw logits",
            )
        count += 1
    return {
        "status": "verified_same_raw_logits" if not failed else "inconclusive_failed_answers",
        "n_tasks": count,
        "failed_or_unmatched_tasks": failed,
        "timing_scope": "CPU raw-logit replay; not GPU model inference latency",
    }


def audit_stage(path, stage, protocol, protocol_hash):
    if path is None or not (Path(path) / "receipt.json").is_file():
        return {"status": "not_run", "complete": False, "stage": stage, "profiles": []}
    root, receipt = Path(path), read(Path(path) / "receipt.json")
    require(
        receipt["stage"] == stage and receipt["prospective_protocol"] == protocol,
        "stage/prospective protocol identity differs",
    )
    require(
        receipt["source_files"]["prospective_protocol"] == protocol_hash,
        "stage protocol hash differs",
    )
    require(
        {
            "campaign",
            "breakthrough_prompt.py",
            "breakthrough_training.py",
            "data_generator",
            "public_runner",
            "release_data_loader",
            "checkpoint_identity_helper",
            "prospective_protocol",
            "s1/unified.py",
            "s1/contracts.py",
        }
        <= set(receipt["source_files"]),
        "stage source inventory lacks required executed components",
    )
    for name, expected in receipt["source_files"].items():
        relative = Path("sources") / f"{name}.txt"
        resolved = (root / relative).resolve()
        require(
            resolved.is_relative_to(root.resolve()) and resolved.is_file(),
            "executed source missing or path unsafe",
        )
        require(digest(resolved) == expected, "executed source hash differs")
    expected_data = read(ROOT / protocol["data"]["synthetic"]["directory"] / "manifest.json")
    require(receipt["dataset"] == expected_data, "stage dataset manifest differs")
    recipe = receipt["training_recipe"]
    require(
        all(
            recipe.get(k) == value
            for k, value in {
                "steps": 128,
                "seed": 42,
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_gradient_accumulation": 8,
                "lora_lr": 1e-5,
                "head_lr": 1e-4,
                "head_batch": 64,
            }.items()
        ),
        "stage training recipe differs",
    )
    declared = [row["name"] for row in receipt["profiles"]]
    expected_profiles = next(s["profiles"] for s in protocol["stages"] if s["id"] == stage)
    require(
        len(set(declared)) == len(declared)
        and set(declared) <= {p["id"] for p in expected_profiles},
        "stage duplicate or undeclared profiles",
    )
    return {
        "status": receipt["status"],
        "complete": receipt["status"] == "completed",
        "stage": stage,
        "error": receipt.get("error"),
        "gpu": receipt.get("gpu"),
        "versions": receipt.get("versions"),
        "declared_completed_profiles": declared,
        "source_hashes_verified": True,
        "source_files": receipt["source_files"],
        "receipt_sha256": digest(root / "receipt.json"),
    }


def audit_identity(receipt, declaration, stage_result, protocol):
    identity = receipt["identity"]
    expected = protocol["weights"][declaration["weights"]]
    require(
        identity.get("model_id") == expected["model_id"]
        and identity.get("revision") == expected["revision"]
        and identity.get("variant") == declaration["id"],
        "profile model/revision/variant differs",
    )
    require(
        identity.get("source_files") == stage_result.get("source_files"),
        "child source hashes differ from stage",
    )
    require(
        identity.get("coherence_commit") == protocol["data"]["coherence"]["revision"]
        and identity.get("benchmarkheaven_commit") == protocol["data"]["public231"]["revision"],
        "profile benchmark revisions differ",
    )
    checkpoint = identity["checkpoint_files"]
    files = checkpoint["files"]
    require(
        isinstance(files, list)
        and len(files) >= 2
        and len({p["path"] for p in files}) == len(files),
        "invalid checkpoint file inventory",
    )
    for file in files:
        require(
            Path(file["path"]).name == file["path"]
            and re.fullmatch(r"[0-9a-f]{64}", file["sha256"])
            and type(file["size_bytes"]) is int
            and file["size_bytes"] > 0,
            "invalid checkpoint file identity",
        )
    expected_hash = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(checkpoint["sha256"] == expected_hash, "checkpoint metadata inventory digest differs")
    processor = identity["processor_files"]
    require(
        isinstance(processor, dict)
        and "config.json" in processor
        and "tokenizer_config.json" in processor,
        "native processor/tokenizer inventory missing",
    )
    for name, file in processor.items():
        require(
            Path(name).name == name
            and re.fullmatch(r"[0-9a-f]{64}", file["sha256"])
            and type(file["size_bytes"]) is int
            and file["size_bytes"] > 0,
            "invalid processor identity",
        )
    return {
        "status": "metadata_verified",
        "model_id": expected["model_id"],
        "revision": expected["revision"],
        "checkpoint_inventory_sha256": expected_hash,
        "processor_files": processor,
        "scope": "Recorded immutable revision and self-consistent runtime hashes; raw HF weight/processor files are not rehashed by this offline outcome analyzer",
    }


def audit_training(root, kind, train_cases, receipt):
    if kind not in ("head", "attention_lora"):
        return {"status": "not_run", "scope": "Declared non-training control"}
    root = Path(root)
    if kind == "head":
        saved = read(root / "head-training.json")
        require(
            saved["steps"] == 128
            and saved["seed"] == 42
            and saved["batch_size"] == 64
            and saved["learning_rate"] == 1e-4
            and len(saved["losses"]) == 128,
            "head recipe/update count differs",
        )
        require(all(math.isfinite(v) for v in saved["losses"]), "nonfinite head losses")
        file = root / "decision-head.pt"
        expected = receipt["identity"]["research_head_sha256"]
        if file.is_file():
            require(digest(file) == expected, "trained head binary hash differs")
        return {
            "status": "recipe_verified",
            "steps": 128,
            "binary_sha256": expected,
            "binary_status": "verified" if file.is_file() else "not_downloaded",
            "scope": "CPU final head; absence of exported binary is not weight verification",
        }
    directory = root / "mixed-lora"
    saved = read(directory / "training.json")
    updates = [
        json.loads(line)
        for line in (directory / "updates.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ids = [case_id for row in updates for case_id in row["case_ids"]]
    require(
        saved["status"] == "completed"
        and saved["steps"] == 128
        and saved["seed"] == 42
        and saved["gradient_accumulation"] == 8
        and saved["learning_rate"] == 1e-5
        and saved["n_training_calls"]
        == saved["n_unique_training_cases"]
        == len(train_cases)
        == 1024,
        "LoRA recipe/population differs",
    )
    require(
        len(updates) == 128
        and [r["step"] for r in updates] == list(range(1, 129))
        and all(len(r["case_ids"]) == 8 and math.isfinite(r["loss"]) for r in updates)
        and Counter(ids) == Counter(case.id for case in train_cases),
        "LoRA updates do not cover train exactly once",
    )
    artifacts = {}
    for name, declared in saved["adapter_files"].items():
        require(Path(name).name == name, "unsafe adapter file path")
        file = directory / "adapter" / name
        if file.is_file():
            require(
                digest(file) == declared["sha256"] and file.stat().st_size == declared["bytes"],
                "adapter binary hash/size differs",
            )
        artifacts[name] = {**declared, "status": "verified" if file.is_file() else "not_downloaded"}
    return {
        "status": "recipe_verified",
        "updates": len(updates),
        "training_cases": len(ids),
        "adapter_files": artifacts,
        "scope": "Final update only; no public checkpoint selection",
    }


def analyze_profile(root, declaration, stage_result, data, public_source, coherence_root):
    name = declaration["id"]
    directory = Path(root) / name if root is not None else None
    if directory is None or not directory.is_dir():
        return {
            "status": "not_run",
            "execution_status": "not_run",
            "complete": False,
            "profile": name,
            "expected_populations": expected_populations(),
            "stage_status": stage_result["status"],
            "error": stage_result.get("error"),
        }
    receipt = read(directory / "receipt.json") if (directory / "receipt.json").is_file() else {}
    result = {
        "profile": name,
        "stage_status": stage_result["status"],
        "receipt_present": bool(receipt),
        "declared_completed": name in stage_result.get("declared_completed_profiles", []),
        "identity": receipt.get("identity"),
        "raw_files_sha256": inventory(directory),
        "expected_populations": expected_populations(),
    }
    result["execution_status"] = (
        "completed"
        if result["declared_completed"]
        else ("failed" if stage_result["status"] == "failed" else "unconfirmed_or_in_progress")
    )
    result["identity_integrity"] = guarded(
        lambda: audit_identity(receipt, declaration, stage_result, data["protocol"])
    )
    result["observed_public_raw_files"] = {
        policy: sum(name.startswith(f"{folder}/raw/") for name in result["raw_files_sha256"])
        for policy, folder in (
            ("calibrated", "public231"),
            ("unit", "public231-unit"),
            ("published_global", "public231-published_global"),
        )
    }
    result["synthetic"] = guarded(
        lambda: audit_synthetic(directory, data["calibration"], data["test"])
    )
    result["public"] = {
        policy: guarded(
            lambda folder=folder: audit_public_profile(directory / folder, public_source)
        )
        for policy, folder in (
            ("calibrated", "public231"),
            ("unit", "public231-unit"),
            ("published_global", "public231-published_global"),
        )
    }
    result["public_replay"] = {
        policy: guarded(
            lambda policy=policy: public_replay_identity(
                result["public"]["calibrated"], result["public"][policy]
            )
        )
        for policy in ("unit", "published_global")
    }
    regression = (
        read(directory / "regression.json") if (directory / "regression.json").is_file() else {}
    )
    result["regression"] = {
        name: guarded(lambda name=name: audit_regression_records(regression[name], cases))
        for name, cases in data["release"].items()
    }
    result["native_fixture"] = guarded(
        lambda: audit_native(directory / "native-fixture.json", data["native"])
    )
    revision = (receipt.get("identity") or {}).get("revision", "missing-receipt")
    result["coherence"] = {
        "calibrated": guarded(
            lambda: audit_coherence(
                directory / "coherence-mini.json",
                directory / "coherence-cache",
                source_root=coherence_root,
                backend=f"{name}@{revision}",
            )
        ),
        "unit": guarded(
            lambda: audit_coherence(
                directory / "coherence-unit.json",
                directory / "coherence-cache-unit",
                source_root=coherence_root,
                backend=f"{name}-unit-replay@{revision}",
            )
        ),
    }
    result["coherence_effect"] = guarded(
        lambda: coherence_effect(result["coherence"]["calibrated"], result["coherence"]["unit"])
    )
    result["training"] = guarded(
        lambda: audit_training(directory.parent, declaration["training"], data["train"], receipt)
    )
    populations = [
        result["synthetic"],
        *result["public"].values(),
        *result["regression"].values(),
        result["native_fixture"],
        *result["coherence"].values(),
    ]
    result["complete"] = (
        bool(receipt)
        and result["declared_completed"]
        and stage_result.get("source_hashes_verified") is True
        and all(p.get("complete") for p in populations)
        and all(
            r.get("status") == "verified_same_raw_logits" for r in result["public_replay"].values()
        )
        and result["training"].get("status") in ("not_run", "recipe_verified")
        and result["identity_integrity"].get("status") == "metadata_verified"
    )
    result["status"] = "complete_outcome_audit" if result["complete"] else "incomplete_or_failed"
    result["official_replay_complete"] = result["complete"] and all(
        c.get("status") == "verified_raw_replay" for c in result["coherence"].values()
    )
    return result


def strip_records(value):
    if isinstance(value, dict):
        return {
            key: strip_records(item)
            for key, item in value.items()
            if key not in {"records", "raw_answers", "raw_runtime", "manifest"}
        }
    if isinstance(value, list):
        return [strip_records(item) for item in value]
    return value


def analyze(
    ablate=None,
    train=None,
    *,
    protocol_path=None,
    public_root=None,
    coherence_root=None,
    historical_jev=None,
    seed=42,
    samples=2000,
):
    protocol_path = Path(
        protocol_path or ROOT / "configs/experiments/jevbench-breakthrough-20261008.json"
    )
    protocol, protocol_hash = read(protocol_path), digest(protocol_path)
    require(protocol_hash == PROTOCOL_SHA256, "prospectively frozen protocol bytes differ")
    synthetic = ROOT / protocol["data"]["synthetic"]["directory"]
    require(
        digest(synthetic / "manifest.json") == protocol["data"]["synthetic"]["manifest_sha256"],
        "frozen synthetic manifest differs",
    )
    for filename, expected in protocol["data"]["synthetic"]["files"].items():
        require(
            Path(filename).name == filename and digest(synthetic / filename) == expected,
            "frozen synthetic file differs",
        )
    release_root = ROOT / protocol["data"]["release_regression"]["directory"]
    require(
        digest(release_root / "manifest.json")
        == protocol["data"]["release_regression"]["manifest_sha256"],
        "frozen release manifest differs",
    )
    release_files = {
        "test": "test.jsonl",
        "media": "media-test.jsonl",
        "regression_text": "regression-text.jsonl",
    }
    for filename in release_files.values():
        require(
            digest(release_root / filename)
            == protocol["data"]["release_regression"]["evaluation_files"][filename]["file_sha256"],
            "frozen release regression bytes differ",
        )
    native_path = ROOT / protocol["data"]["native_fixture"]["path"]
    require(
        digest(native_path) == protocol["data"]["native_fixture"]["file_sha256"],
        "frozen fixture differs",
    )
    data = {
        split: load_cases(synthetic / f"{split}.jsonl")
        for split in ("train", "calibration", "test")
    }
    data["release"] = {
        name: load_cases(release_root / filename) for name, filename in release_files.items()
    }
    data["native"] = load_cases(native_path)
    data["protocol"] = protocol
    public_contract = ROOT / protocol["data"]["public231"]["source_contract"]
    require(
        digest(public_contract) == protocol["data"]["public231"]["source_contract_sha256"],
        "frozen official public source contract differs",
    )
    public_source = load_public(public_root) if public_root is not None else None
    stages, profiles = {}, {}
    roots = {"ablate": ablate, "train": train}
    for stage in protocol["stages"]:
        name, root = stage["id"], roots[stage["id"]]
        stages[name] = guarded(
            lambda name=name, root=root: audit_stage(root, name, protocol, protocol_hash)
        )
        for declaration in stage["profiles"]:
            profiles[declaration["id"]] = {
                "profile": declaration["id"],
                "expected_populations": expected_populations(),
                **guarded(
                    lambda declaration=declaration: analyze_profile(
                        root, declaration, stages[name], data, public_source, coherence_root
                    )
                ),
            }
    comparisons = []
    control = "released-user-control"
    pairs = [(control, name, "released_control") for name in profiles if name != control]
    pairs += [
        (reference, name, "training_primary")
        for name, reference in protocol["evaluation"]["primary_training_comparators"].items()
    ]
    expected = {(c.id, q.id) for c in data["test"] for q in c.request.questions}
    for reference, candidate, role in pairs:
        a, b = profiles[reference], profiles[candidate]
        row = {
            "reference": reference,
            "candidate": candidate,
            "role": role,
            "hardware": {
                "reference_gpu": stages["train"].get("gpu")
                if reference.startswith("released-user-")
                else stages["ablate"].get("gpu"),
                "candidate_gpu": stages["train"].get("gpu")
                if candidate.startswith("released-user-")
                else stages["ablate"].get("gpu"),
            },
            "latency_scope": "No paired speedup claim; stages may use different A100 boards",
        }
        sa, sb = a.get("synthetic", {}), b.get("synthetic", {})
        if sa.get("complete") and sb.get("complete"):
            ta = {k: v["temperature"] for k, v in sa["temperatures"].items()}
            tb = {k: v["temperature"] for k, v in sb["temperatures"].items()}
            row["synthetic"] = {
                "calibrated": guarded(
                    lambda: paired_soft(
                        sa["records"],
                        sb["records"],
                        ta,
                        tb,
                        expected_keys=expected,
                        seed=seed,
                        samples=samples,
                    )
                ),
                "unit": guarded(
                    lambda: paired_soft(
                        sa["records"],
                        sb["records"],
                        {k: 1.0 for k in ta},
                        {k: 1.0 for k in tb},
                        expected_keys=expected,
                        seed=seed,
                        samples=samples,
                    )
                ),
            }
        else:
            row["synthetic"] = {
                "status": "inconclusive",
                "reason": "both full fresh populations required",
            }
        pa, pb = (
            a.get("public", {}).get("calibrated", {}),
            b.get("public", {}).get("calibrated", {}),
        )
        row["public"] = (
            guarded(
                lambda: paired_accuracy(
                    pa["records"], pb["records"], public_source[0], seed=seed, samples=samples
                )
            )
            if public_source is not None and pa.get("complete") and pb.get("complete")
            else {"status": "inconclusive"}
        )
        row["regression"] = {}
        for dataset in release_files:
            ra, rb = (
                a.get("regression", {}).get(dataset, {}),
                b.get("regression", {}).get(dataset, {}),
            )
            row["regression"][dataset] = (
                guarded(
                    lambda ra=ra, rb=rb: paired_native(
                        ra["records"], rb["records"], seed=seed, samples=samples
                    )
                )
                if ra.get("complete") and rb.get("complete")
                else {"status": "inconclusive"}
            )
        na, nb = a.get("native_fixture", {}), b.get("native_fixture", {})
        row["native_fixture"] = (
            guarded(lambda: paired_native(na["records"], nb["records"], seed=seed, samples=samples))
            if na.get("complete") and nb.get("complete")
            else {"status": "inconclusive"}
        )
        comparisons.append(row)
    historical = {
        "status": "not_requested",
        "scope": "Historical Jev-Omni only; no new heldout/training/control or same-cell speedup claim",
    }
    if historical_jev is not None:
        audited = guarded(
            lambda: audit_public_profile(Path(historical_jev) / "public231", public_source)
        )
        historical.update(
            {
                "status": audited["status"],
                "public": strip_records(audited),
                "raw_files_sha256": inventory(Path(historical_jev) / "public231"),
                "comparisons": [],
            }
        )
        if audited.get("complete"):
            for name, profile in profiles.items():
                current = profile.get("public", {}).get("calibrated", {})
                if current.get("complete"):
                    historical["comparisons"].append(
                        {
                            "candidate": name,
                            "reference": "historical-jev-omni",
                            "result": guarded(
                                lambda current=current: paired_accuracy(
                                    audited["records"],
                                    current["records"],
                                    public_source[0],
                                    seed=seed,
                                    samples=samples,
                                )
                            ),
                            "scope": "Historical public-only descriptive comparison; different model/prompt/hardware/configuration may confound",
                        }
                    )
    complete = all(p.get("complete") for p in profiles.values()) and all(
        s.get("complete") for s in stages.values()
    )
    return {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "protocol_sha256": protocol_hash,
        "analyzer_sha256": digest(__file__),
        "status": "complete_outcome_audit" if complete else "incomplete_or_not_run",
        "stages": stages,
        "profile_counts": dict(Counter(p["status"] for p in profiles.values())),
        "execution_profile_counts": dict(
            Counter(p.get("execution_status", "unconfirmed") for p in profiles.values())
        ),
        "declared_profiles": len(profiles),
        "profiles": strip_records(profiles),
        "comparisons": comparisons,
        "historical_jev_omni": historical,
        "official_raw_replay_complete": complete
        and all(p.get("official_replay_complete") for p in profiles.values()),
        "formal_acceptance": "not_assessed",
        "rank": None,
        "cost_usd": None,
        "leaderboard_status": "not_submitted",
        "sealed_evaluation": "not_run",
        "statistics": {"seed": seed, "resamples": samples, "confidence": 0.95},
        "excluded_from_declared_profiles": ["local MPS smoke", "prior public campaign models"],
        "execution_status_evidence": {
            "path": "artifacts/jevbench/breakthrough-20261008/execution-status.json",
            "sha256": digest(
                ROOT / "artifacts/jevbench/breakthrough-20261008/execution-status.json"
            ),
            "record": read(ROOT / "artifacts/jevbench/breakthrough-20261008/execution-status.json"),
        }
        if (ROOT / "artifacts/jevbench/breakthrough-20261008/execution-status.json").is_file()
        else None,
        "scope": "Offline descriptive audit of complete declared populations; no fitting, checkpoint selection, default change or publication",
    }


def markdown(report):
    lines = [
        "# Breakthrough outcome audit",
        "",
        f"Status: `{report['status']}`; {report['declared_profiles']} declared profiles.",
        "",
        "Formal acceptance is not assessed; official rank/cost are unknown, sealed evaluation is not run.",
        "",
        "| Profile | Status | Fresh test | Public | BoolQ / native / historic | Coherence calibrated / unit |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for name, row in report["profiles"].items():
        synthetic = row.get("synthetic", {})
        public = row.get("public", {}).get("calibrated", {}).get("summary", {})
        regression = row.get("regression", {})
        cohorts = " / ".join(
            f"{regression[k].get('n_correct', '?')}/{regression[k].get('n_questions', '?')}"
            if regression.get(k, {}).get("complete")
            else "incomplete"
            for k in ("test", "media", "regression_text")
        )
        coherence = row.get("coherence", {})
        coh = " / ".join(
            f"{coherence[k]['overall']:.4f}"
            if coherence.get(k, {}).get("complete")
            else "incomplete"
            for k in ("calibrated", "unit")
        )
        lines.append(
            f"| {name} | {row['status']} | {synthetic.get('n_test', 0)} | {public.get('n_correct', '?')}/{public.get('n_planned', '?')} | {cohorts} | {coh} |"
        )
    lines += [
        "",
        "The JSON retains per-type/template metrics, paired source-group intervals, all raw hashes, errors and unexecuted populations. Coherence outcome rows are not independent denominators. Raw-logit replay latency is CPU replay, not GPU latency. Historical Jev-Omni is reported separately.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablate", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--public-root", type=Path)
    parser.add_argument("--coherence-root", type=Path)
    parser.add_argument("--historical-jev", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(
        not args.output.exists() and not args.output.with_suffix(".md").exists(),
        "refusing to overwrite analysis output",
    )
    report = analyze(
        args.ablate,
        args.train,
        protocol_path=args.protocol,
        public_root=args.public_root,
        coherence_root=args.coherence_root,
        historical_jev=args.historical_jev,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    args.output.with_suffix(".md").write_text(markdown(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "profile_counts": report["profile_counts"],
                "output": str(args.output),
            }
        )
    )
    return (
        0
        if report["status"] == "complete_outcome_audit" and report["official_raw_replay_complete"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
