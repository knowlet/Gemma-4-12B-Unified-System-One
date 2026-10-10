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
import random
import re
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "3a8dcceba9fbc0c7ce31bddbb85c0d10acdfa855ddf237f4d5e81982ec6a926d"
PUBLIC_REPLAY_EXECUTION = "raw-logit calibration replay; recorded durations are CPU replay only, NOT model inference latency"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_jevbench_public import load_public  # noqa: E402
from summarize_jevbench_campaign import (  # noqa: E402
    COHERENCE_SETTINGS,
    _bytes,
    _freeze_coherence_source,
    _metric_equal,
    audit_public,
    paired_accuracy,
    replay_coherence,
)

from s1.contracts import DecisionRequest, answer_from_probabilities  # noqa: E402
from s1.evaluation.datasets import fingerprint, load_cases, request_fingerprint  # noqa: E402
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


def arithmetic_bytes(value):
    """Serialization used by the independently archived CPU arithmetic proof."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def load_arithmetic_proof(path, proof_sha256, verifier_sha256, roots, protocol):
    """Bind optional same-environment evidence; default local arithmetic stays strict.

    The two explicit digests are review inputs, not values inferred from the
    certificate. They bind the independently executed verifier and output bytes.
    Every original input is also checked here; vector checks occur per profile.
    """
    path = Path(path)
    require(
        all(
            isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)
            for v in (proof_sha256, verifier_sha256)
        )
        and path.is_file()
        and not path.is_symlink()
        and 0 < path.stat().st_size <= 128 * 1024 * 1024
        and digest(path) == proof_sha256,
        "CPU arithmetic proof requires independently reviewed output/source SHA256",
    )
    source = path.parent / "verifier.py.txt"
    require(
        source.is_file()
        and not source.is_symlink()
        and source.stat().st_size <= 1024 * 1024
        and digest(source) == verifier_sha256,
        "CPU arithmetic verifier bytes differ",
    )
    proof = read(path)
    require(
        proof.get("schema_version") == 1
        and proof.get("status") == "completed"
        and isinstance(proof.get("run"), str)
        and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", proof["run"])
        and proof.get("verifier_source_sha256") == verifier_sha256
        and proof.get("original_image_id") == "im-hWNAPF8eRYjs4tA8VG582u",
        "CPU arithmetic proof run/source/image differs",
    )
    env = proof.get("environment", {})
    require(
        env.get("python") == "3.12.10"
        and env.get("torch") == "2.10.0"
        and env.get("numpy") == "2.5.3"
        and env.get("machine") in {"x86_64", "amd64"}
        and str(env.get("platform", "")).startswith("Linux-")
        and env.get("visible_gpu_count") == 0,
        "CPU arithmetic pinned environment differs",
    )
    execution = proof.get("execution", {})
    require(
        all(
            execution.get(k) == v
            for k, v in {
                "cpu_only": True,
                "no_model_call": True,
                "volume_read_only": True,
                "cpu": 2,
                "memory_mib": 4096,
                "timeout_seconds": 600,
                "network_blocked": True,
                "model_forwards": 0,
                "training_updates": 0,
                "hf_downloads": 0,
                "volume_writes": 0,
            }.items()
        ),
        "CPU arithmetic execution scope differs",
    )
    require(
        proof.get("public_replay") == "not_replayed; already independently strict-verified locally",
        "CPU proof scope must preserve separately verified local public population",
    )
    expected_members, expected_profiles, sources = {}, {}, {}
    for stage in protocol["stages"]:
        stage_id = stage["id"]
        root = Path(roots[stage_id])
        receipt = read(root / "receipt.json")
        require(
            receipt["status"] == "completed"
            and receipt["run"] == proof["run"]
            and receipt["stage"] == stage_id,
            "CPU proof requires completed original stages",
        )
        require(
            receipt["versions"]["torch"] == env["torch"]
            and receipt["versions"]["numpy"] == env["numpy"],
            "CPU/GPU arithmetic versions differ",
        )
        sources[stage_id] = receipt["source_files"]
        names = {"receipt.json"} | {f"sources/{name}.txt" for name in receipt["source_files"]}
        for declared in stage["profiles"]:
            profile = declared["id"]
            expected_profiles[profile] = stage_id
            names |= {
                f"{profile}/{name}"
                for name in (
                    "receipt.json",
                    "calibration.json",
                    "raw-logits.json",
                    "coherence-mini.json",
                    "coherence-unit.json",
                )
            }
            for folder in ("coherence-cache", "coherence-cache-unit"):
                cache = root / profile / folder
                paths = list(cache.iterdir())
                require(
                    len(paths) == 1248
                    and all(
                        p.is_file()
                        and not p.is_symlink()
                        and re.fullmatch(r"[0-9a-f]{64}\.json", p.name)
                        for p in paths
                    ),
                    "CPU proof requires full original cache inventories",
                )
                names |= {f"{profile}/{folder}/{p.name}" for p in paths}
        for name in sorted(names):
            member = root / name
            require(
                member.is_file()
                and not member.is_symlink()
                and member.resolve().is_relative_to(root.resolve()),
                "unsafe CPU proof input",
            )
            expected_members[f"{stage_id}/{name}"] = {
                "sha256": digest(member),
                "size_bytes": member.stat().st_size,
            }
    require(
        set(sources) == {"ablate", "train"} and len(expected_profiles) == 8,
        "CPU proof requires both stages and all eight profiles",
    )
    require(
        proof.get("source_files") == sources and proof.get("input_members") == expected_members,
        "CPU arithmetic source/input byte inventory differs",
    )
    manifest = path.parent / "input-manifest.json"
    app_file = path.parent / "modal-app.json"
    require(
        manifest.is_file()
        and not manifest.is_symlink()
        and read(manifest) == expected_members
        and digest(manifest)
        == hashlib.sha256(arithmetic_bytes(expected_members)).hexdigest()
        == proof.get("input_manifest_sha256"),
        "CPU arithmetic input manifest differs",
    )
    require(app_file.is_file() and not app_file.is_symlink(), "CPU arithmetic app receipt missing")
    app = read(app_file)
    require(
        app.get("image_id") == proof["original_image_id"]
        and app.get("verifier_source_sha256") == verifier_sha256
        and isinstance(app.get("app_id"), str)
        and app["app_id"].startswith("ap-")
        and isinstance(app.get("function_id"), str)
        and app["function_id"].startswith("fu-"),
        "CPU arithmetic app/source/image receipt differs",
    )
    require(
        isinstance(proof.get("profiles"), dict)
        and set(proof["profiles"]) == set(expected_profiles),
        "CPU arithmetic full eight-profile population differs",
    )
    for name, stage_id in expected_profiles.items():
        profile = proof["profiles"][name]
        prefix = f"{stage_id}/{name}/"
        require(
            profile.get("stage") == stage_id
            and profile.get("public") == {}
            and profile.get("file_sha256")
            == {
                key.removeprefix(prefix): value["sha256"]
                for key, value in expected_members.items()
                if key.startswith(prefix)
            },
            "CPU arithmetic profile byte inventory or scope differs",
        )
        require(
            profile.get("identity")
            == read(Path(roots[stage_id]) / name / "receipt.json")["identity"],
            "CPU arithmetic child identity differs",
        )
    return {
        "proof": proof,
        "metadata": {
            "status": "input_bytes_verified",
            "proof_sha256": proof_sha256,
            "verifier_source_sha256": verifier_sha256,
            "path": str(path),
            "input_manifest_sha256": digest(manifest),
            "modal_app_sha256": digest(app_file),
            "input_files": len(expected_members),
            "input_bytes": sum(v["size_bytes"] for v in expected_members.values()),
            "environment": env,
            "execution": execution,
            "public_scope": proof["public_replay"],
            "scope": "Reviewed source/output digests, pinned CPU environment and complete original byte inventories; expected vectors independently checked for all coherence requests below. This is separate from strict local macOS parity.",
        },
    }


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


def calibration_grid_match(saved_temperature, grid, losses):
    """Bind the minimizing grid index despite one binary64 libm rounding ULP.

    The frozen x86 and local ARM NumPy geomspace can differ by one ULP at an
    interior point. This never admits a different grid index or an endpoint
    change; the caller also replays both saved losses at the executed value.
    """
    require(
        type(saved_temperature) in (int, float)
        and math.isfinite(saved_temperature)
        and saved_temperature > 0,
        "calibration temperature/population differs from fixed soft-CE grid",
    )
    index = int(np.argmin(losses))
    expected = float(grid[index])
    nearest = min(range(len(grid)), key=lambda i: abs(float(grid[i]) - saved_temperature))
    delta = abs(saved_temperature - expected)
    require(
        nearest == index
        and delta <= math.ulp(expected)
        and math.nextafter(expected, -math.inf)
        <= saved_temperature
        <= math.nextafter(expected, math.inf)
        and (index not in (0, len(grid) - 1) or delta == 0),
        "calibration temperature/population differs from fixed soft-CE grid",
    )
    return {
        "minimizing_grid_index": index,
        "local_grid_temperature": expected,
        "executed_temperature": saved_temperature,
        "absolute_difference": delta,
        "float64_ulp_bound": math.ulp(expected),
    }


def audit_synthetic(directory, calibration_cases, test_cases, *, receipt=None):
    root = Path(directory)
    cal, heldout = read(root / "calibration.json"), read(root / "heldout.json")
    if receipt is not None:
        require(
            receipt["identity"].get("calibration") == cal["temperatures"],
            "child identity/calibration file differ",
        )
    validate_soft_records(cal["records"], calibration_cases, "calibration")
    rows = heldout["records"]
    validate_soft_records(rows, test_cases, "test")
    temperatures, grid_verification = {}, {}
    for kind in ("choice", "noul", "score"):
        selected = [row for row in cal["records"] if row["type"] == kind]
        require(bool(selected), "calibration does not cover all types")
        grid = np.geomspace(0.1, 10.0, 81)
        losses = [
            np.mean([row_metrics(row, float(t))["soft_ce"] for row in selected]) for t in grid
        ]
        saved = cal["temperatures"][kind]
        matched = calibration_grid_match(saved["temperature"], grid, losses)
        require(
            saved["n"] == len(selected),
            "calibration temperature/population differs from fixed soft-CE grid",
        )
        saved_loss = float(
            np.mean([row_metrics(row, saved["temperature"])["soft_ce"] for row in selected])
        )
        require(
            math.isclose(
                saved["soft_ce_before"],
                float(np.mean([row_metrics(row, 1.0)["soft_ce"] for row in selected])),
                abs_tol=1e-12,
                rel_tol=0,
            )
            and math.isclose(saved["soft_ce_after"], saved_loss, abs_tol=1e-12, rel_tol=0)
            and math.isclose(saved_loss, float(min(losses)), abs_tol=1e-12, rel_tol=0),
            "saved calibration losses differ",
        )
        require(
            saved["boundary"] == (matched["minimizing_grid_index"] in (0, len(grid) - 1)),
            "calibration boundary status differs",
        )
        temperatures[kind] = saved["temperature"]
        grid_verification[kind] = {**matched, "loss_at_executed_temperature": saved_loss}
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
        "calibration_grid_verification": grid_verification,
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


def audit_coherence(path, cache_root, *, source_root=None, backend=None, receipt=None):
    report = read(path)
    scores, result = report["scores"], report["result"]
    require(backend is None or result["backend"] == backend, "coherence backend identity differs")
    if receipt is not None:
        require(
            receipt.get("coherence")
            == {
                "overall": scores.get("overall"),
                "interval": (scores.get("ci") or {}).get("overall"),
                "dimensions": scores.get("pillars"),
                "errors": result.get("errors"),
                "scores": scores,
            },
            "child coherence receipt/report differ",
        )
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


def coherence_effect(calibrated, unit, linkage=None):
    if not calibrated.get("complete") or not unit.get("complete"):
        return {"status": "inconclusive", "reason": "both complete coherence policies required"}
    if (
        not linkage
        or linkage.get("status")
        not in ("verified_same_raw_logits", "verified_same_raw_logits_pinned_cpu")
        or linkage.get("complete") is not True
    ):
        return {
            "status": "inconclusive_calibration_linkage",
            "reason": "same raw logits and both complete temperature transforms must be verified",
        }
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


def audit_public_identity(result, receipt, policy, protocol):
    """Bind a verified bundle to this child, permitting only fixed replay fields."""
    identity = receipt["identity"]
    if policy == "calibrated":
        expected = identity
    else:
        require(policy in ("unit", "published_global"), "unknown public replay policy")
        temperature = protocol["calibration"][f"{policy}_temperature"]
        expected = {
            **identity,
            "variant": f"{identity['variant']}-{policy}",
            "calibration": {kind: temperature for kind in ("choice", "noul", "score")},
            "execution": PUBLIC_REPLAY_EXECUTION,
        }
    require(
        result["manifest"]["identity"] == expected,
        "child receipt/public manifest identity differs",
    )
    if policy == "calibrated":
        require(receipt.get("public") == result["summary"], "child receipt/public summary differs")
    return {"status": "verified_child_identity", "policy": policy}


def audit_public_profile(path, public_source, *, receipt=None, policy="calibrated", protocol=None):
    if public_source is None:
        return {
            "status": "not_audited",
            "complete": False,
            "reason": "pinned public source root required",
        }
    tasks, official, config = public_source
    result = audit_public(path, tasks, official, config)
    identity_integrity = (
        audit_public_identity(result, receipt, policy, protocol) if receipt is not None else None
    )
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
        "child_identity_integrity": identity_integrity,
        "summary": summary,
        "by_type": result["by_type"],
        "errors": errors,
        "n_valid": summary["n_valid"],
        "false_accepts": {"count": false_accept, "gold_no": no_gold},
        "false_rejects": {"count": false_reject, "gold_yes": yes_gold},
        "records": result["records"],
        "manifest": result["manifest"],
        "raw_answers": result["raw_answers"],
        "raw_requests": {
            task.id: {
                "state": task.state,
                "questions": {"decision": official.base.build_question(task)},
            }
            for task in tasks
        },
        "raw_runtime": {
            row["task_id"]: read(Path(path) / row["raw_path"])
            .get("response", {})
            .get("runtime", {})
            for row in result["records"]
            if row.get("ok") is True
        },
        "verified_files": result["verified_files"],
    }


def verify_temperature_answers(request, raw_rows, answers, temperatures):
    """Match every native field to the recorded logits, without any model call."""
    import torch

    require(
        isinstance(raw_rows, list)
        and [row.get("id") for row in raw_rows] == [q.id for q in request.questions]
        and isinstance(answers, dict)
        and set(answers) == {q.id for q in request.questions},
        "raw-logit/answer full question population or order differs",
    )
    maximum = 0.0
    for question, raw in zip(request.questions, raw_rows, strict=True):
        require(
            raw.get("labels") == question.labels(), "raw-logit ordered labels differ from request"
        )
        logits = np.asarray(raw.get("logits"), dtype=float)
        require(
            logits.ndim == 1
            and len(logits) == len(question.labels())
            and np.isfinite(logits).all(),
            "raw-logit candidate values must be aligned and finite",
        )
        temperature = temperatures[question.type]
        require(
            type(temperature) in (int, float) and math.isfinite(temperature) and temperature > 0,
            "positive finite type temperature required",
        )
        # The frozen Predictor constructs CPU FP32 tensors and softmax. Reproduce
        # that arithmetic, allowing one FP32 epsilon across local torch versions.
        transformed = torch.tensor(raw["logits"], dtype=torch.float32) / temperature
        require(bool(torch.isfinite(transformed).all()), "nonfinite FP32 temperature transform")
        expected = transformed.softmax(-1).tolist()
        answer = answers[question.id]
        actual = answer_probabilities(answer, question)
        values = [actual[label] for label in question.labels()]
        require(
            answer == answer_from_probabilities(question, values),
            "native answer fields differ from probabilities/request",
        )
        error = max(
            abs(observed - predicted) for observed, predicted in zip(values, expected, strict=True)
        )
        maximum = max(maximum, error)
        require(
            error <= float(np.finfo(np.float32).eps),
            "probabilities do not match type temperature/raw logits",
        )
    return {"questions": len(request.questions), "max_probability_error": maximum}


def compare_pinned_cpu_answers(request, raw_rows, answers, temperatures, evidence):
    """Check complete byte-bound expected vectors; never trust certificate booleans."""
    minimal = set(evidence) == {"raw_key", "cache_file", "expected_answers"}
    require(
        (minimal or evidence.get("raw_rows") == raw_rows)
        and isinstance(raw_rows, list)
        and [r.get("id") for r in raw_rows] == [q.id for q in request.questions],
        "CPU proof raw question population or logits differ",
    )
    expected = evidence.get("expected_answers")
    rows = evidence.get("questions")
    require(
        (
            minimal
            or (
                isinstance(rows, list)
                and [r.get("id") for r in rows] == [q.id for q in request.questions]
            )
        )
        and isinstance(expected, dict)
        and isinstance(answers, dict)
        and set(expected) == set(answers) == {q.id for q in request.questions},
        "CPU proof expected answer population/order differs",
    )
    maximum, mismatch_count, field_count, count = 0.0, 0, 0, 0
    for index, (question, raw) in enumerate(zip(request.questions, raw_rows, strict=True)):
        labels = question.labels()
        require(
            raw.get("labels") == labels
            and len(raw.get("logits", [])) == len(labels)
            and all(type(v) in (int, float) and math.isfinite(v) for v in raw["logits"]),
            "CPU proof raw candidate labels/values differ",
        )
        if not minimal:
            row = rows[index]
            require(
                row.get("type") == question.type
                and row.get("labels") == labels
                and row.get("temperature") == temperatures[question.type],
                "CPU proof primitive labels or temperature differ",
            )
        reference = answer_probabilities(expected[question.id], question)
        predicted = [reference[label] for label in labels]
        require(
            (
                minimal
                or (
                    row.get("expected_probabilities") == predicted
                    and row.get("expected_answer") == expected[question.id]
                )
            )
            and expected[question.id] == answer_from_probabilities(question, predicted),
            "CPU proof expected native fields differ from full vectors",
        )
        observed = answer_probabilities(answers[question.id], question)
        values = [observed[label] for label in labels]
        require(
            answers[question.id] == answer_from_probabilities(question, values),
            "native answer fields differ from probabilities/request",
        )
        error = max(abs(x - y) for x, y in zip(values, predicted, strict=True))
        require(
            error <= float(np.finfo(np.float32).eps), "CPU proof temperature probabilities differ"
        )
        maximum = max(maximum, error)
        mismatch_count += sum(x != y for x, y in zip(values, predicted, strict=True))
        field_count += answers[question.id] != expected[question.id]
        count += len(labels)
    return {
        "questions": len(request.questions),
        "probabilities": count,
        "max_abs_error": maximum,
        "mismatch_count": mismatch_count,
        "native_field_mismatch_count": field_count,
        "max_probability_error": maximum,
    }


def audit_pinned_cpu_calibration(directory, evidence, receipt):
    cal = read(Path(directory) / "calibration.json")
    proof = evidence.get("calibration", {})
    rows = cal["records"]
    require(
        len(rows) == 192
        and proof.get("cases") == proof.get("record_count") == 192
        and proof.get("records") == rows
        and cal["temperatures"] == receipt["identity"]["calibration"],
        "CPU proof full calibration records/identity differ",
    )
    require(
        proof.get("record_identity_order_sha256")
        == hashlib.sha256(
            arithmetic_bytes(
                [[r["case_id"], r["question_id"], r["type"], r["labels"]] for r in rows]
            )
        ).hexdigest(),
        "CPU proof calibration record order differs",
    )
    grid = proof.get("grid")
    local_grid = np.geomspace(0.1, 10.0, 81)
    require(
        isinstance(grid, list)
        and len(grid) == 81
        and all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in grid)
        and grid[0] == float(local_grid[0])
        and grid[-1] == float(local_grid[-1])
        and all(
            abs(t - float(v)) <= math.ulp(float(v))
            and math.nextafter(float(v), -math.inf) <= t <= math.nextafter(float(v), math.inf)
            for t, v in zip(grid, local_grid, strict=True)
        ),
        "CPU proof fixed calibration grid differs",
    )
    require(
        set(proof.get("types", {})) == {"choice", "noul", "score"},
        "CPU proof calibration primitive population differs",
    )
    for kind, type_evidence in proof["types"].items():
        selected = [row for row in rows if row["type"] == kind]
        losses = [float(np.mean([row_metrics(r, t)["soft_ce"] for r in selected])) for t in grid]
        declared = type_evidence.get("losses")
        require(
            type_evidence.get("n") == len(selected) == 64
            and isinstance(declared, list)
            and len(declared) == 81
            and all(
                type(v) in (int, float)
                and math.isfinite(v)
                and math.isclose(v, actual, abs_tol=1e-12, rel_tol=0)
                for v, actual in zip(declared, losses, strict=True)
            ),
            "CPU proof full calibration grid losses differ",
        )
        index = int(np.argmin(losses))
        saved = cal["temperatures"][kind]
        require(
            type_evidence.get("argmin_index") == index
            and grid[index] == saved["temperature"]
            and type_evidence.get("saved_metadata")
            == type_evidence.get("expected_metadata")
            == saved
            and type_evidence.get("exact_metadata_match") is True,
            "CPU proof selected calibration index/executed metadata differ",
        )
    return {
        "status": "verified_pinned_cpu_grid",
        "complete": True,
        "calibration_cases": 192,
        "grid_points_per_type": 81,
        "scope": "Complete saved records, exact executed grid index/temperature/metadata and independently recomputed full soft-CE grid; no fitting or recipe selection",
    }


def public_calibration_identity(native, receipt=None):
    if not native.get("complete"):
        return {
            "status": "inconclusive",
            "complete": False,
            "reason": "audited public policy required",
        }
    records = native["records"]
    require(
        len(records) == 231 and len({r["task_id"] for r in records}) == 231,
        "public calibration requires complete 231 task population",
    )
    calibration = native["manifest"]["identity"]["calibration"]
    if receipt is not None:
        require(
            calibration == receipt["identity"]["calibration"],
            "native public/child fitted temperatures differ",
        )
    require(
        set(calibration) == {"choice", "noul", "score"},
        "fitted temperature primitive population differs",
    )
    temperatures = {kind: value["temperature"] for kind, value in calibration.items()}
    valid = {row["task_id"] for row in records if row.get("ok") is True}
    require(
        set(native["raw_runtime"]) == valid
        and set(native["raw_requests"]) == {row["task_id"] for row in records}
        and len(native["raw_answers"]) == 231,
        "public calibration runtime/request/answer population differs",
    )
    count, maximum = 0, 0.0
    for index, row in enumerate(records):
        if row.get("ok") is not True:
            continue
        task_id = row["task_id"]
        request = DecisionRequest(**native["raw_requests"][task_id])
        runtime = native["raw_runtime"][task_id]
        require(
            runtime.get("calibration_replay") is False
            and runtime.get("forward_calls") == len(request.questions)
            and runtime.get("sequential_independent_questions") is True,
            "native public forward/replay declaration differs",
        )
        verified = verify_temperature_answers(
            request, runtime.get("raw_logits"), native["raw_answers"][index], temperatures
        )
        count += 1
        maximum = max(maximum, verified["max_probability_error"])
    return {
        "status": "verified_fitted_temperature" if count == 231 else "inconclusive_failed_answers",
        "complete": count == 231,
        "expected_tasks": 231,
        "verified_tasks": count,
        "failed_or_unmatched_tasks": 231 - count,
        "temperatures": temperatures,
        "max_probability_error": maximum,
        "probability_absolute_tolerance": float(np.finfo(np.float32).eps),
        "scope": "Complete native public answers, fitted type temperatures, full raw-logit question/label identities and derived answer fields; CPU FP32 reconstruction, no model call",
    }


def public_replay_identity(native, replay):
    if not native.get("complete") or not replay.get("complete"):
        return {"status": "inconclusive", "reason": "both audited public policies required"}
    require(
        [r["task_id"] for r in native["records"]] == [r["task_id"] for r in replay["records"]],
        "public replay task identity differs",
    )
    calibration = public_calibration_identity(native)
    if calibration.get("complete") is not True:
        return {
            "status": "inconclusive_native_calibration",
            "complete": False,
            "native_calibration": calibration,
        }
    require(
        set(replay["raw_runtime"])
        == {row["task_id"] for row in replay["records"] if row.get("ok") is True}
        and set(replay["raw_requests"]) == set(native["raw_requests"])
        and len(replay["raw_answers"]) == 231,
        "public replay runtime/request/answer population differs",
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
        require(
            native["raw_requests"][task_id] == replay["raw_requests"][task_id],
            "public replay changed request",
        )
        verify_temperature_answers(
            DecisionRequest(**native["raw_requests"][task_id]),
            second["raw_logits"],
            replay["raw_answers"][index],
            manifest["identity"]["calibration"],
        )
        count += 1
    return {
        "status": "verified_same_raw_logits" if not failed else "inconclusive_failed_answers",
        "complete": not failed,
        "n_tasks": count,
        "failed_or_unmatched_tasks": failed,
        "timing_scope": "CPU raw-logit replay; not GPU model inference latency",
        "native_calibration": calibration,
    }


COHERENCE_LINKAGE_PLAN = r"""
import hashlib, json, socket, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import jevbench as jb
def deny_network(*args, **kwargs):
    raise RuntimeError("offline coherence linkage cannot use network")
socket.create_connection = deny_network
data = json.loads(Path(sys.argv[2]).read_text())
requests, missing = [], set()
def answer(state, questions):
    blob = json.dumps({"b": data["backend"], "s": state, "q": questions, "r": 0},
                      ensure_ascii=False, separators=(",", ":"))
    filename = hashlib.sha256(blob.encode()).hexdigest() + ".json"
    requests.append({"state": state, "questions": questions, "cache_file": filename})
    if filename not in data["answers"]:
        missing.add(filename)
        raise RuntimeError("missing original cached answer: " + filename)
    return data["answers"][filename]
settings = data["settings"]
report = jb.evaluate(jb.from_callable(answer, name=data["backend"]), suite="jevbench-mini",
                     cache=":memory:", concurrency=1, progress=False,
                     tolerance=settings["tolerance"], min_n=settings["min_n"],
                     bootstrap=settings["bootstrap"])
print(json.dumps({"requests": requests, "missing": sorted(missing),
                  "errors": report.result.get("errors", [])}, ensure_ascii=False, allow_nan=False))
"""


def coherence_linkage_requests(report, cache, source_root):
    require(report.get("settings") == COHERENCE_SETTINGS, "coherence linkage settings differ")
    with tempfile.TemporaryDirectory(prefix="s1-temperature-linkage-") as temporary:
        directory = Path(temporary)
        source = _freeze_coherence_source(source_root, directory)
        inputs = directory / "inputs.json"
        inputs.write_bytes(
            _bytes(
                {
                    "backend": report["result"]["backend"],
                    "settings": report["settings"],
                    "answers": cache,
                }
            )
        )
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    COHERENCE_LINKAGE_PLAN,
                    str(directory / "src"),
                    str(inputs),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
        except subprocess.SubprocessError as exc:
            raise AuditError(
                "offline coherence linkage request reconstruction did not complete"
            ) from exc
        require(
            result.returncode == 0,
            "offline coherence linkage request reconstruction failed: " + result.stderr[-1000:],
        )
        rebuilt = json.loads(result.stdout)
    require(
        not rebuilt["missing"] and not rebuilt["errors"],
        "coherence linkage request reconstruction is incomplete",
    )
    return rebuilt["requests"], source


def audit_coherence_linkage(
    directory, calibrated, unit, receipt, *, source_root=None, cpu_evidence=None
):
    root = Path(directory)
    raw_file = root / "raw-logits.json"
    if (
        source_root is None
        or not raw_file.is_file()
        or raw_file.is_symlink()
        or any(
            policy.get("status") != "verified_raw_replay" or policy.get("complete") is not True
            for policy in (calibrated, unit)
        )
    ):
        return {
            "status": "calibration_linkage_unverified",
            "complete": False,
            "expected_requests": 1248,
            "reason": "pinned source, raw-logits.json and both complete official cache replays are required",
            "scope": "No temperature attribution from saved policy scores alone",
        }
    reports = {
        policy: read(root / filename)
        for policy, filename in (
            ("calibrated", "coherence-mini.json"),
            ("unit", "coherence-unit.json"),
        )
    }
    identity = receipt["identity"]
    require(
        reports["calibrated"]["result"]["backend"]
        == f"{identity['variant']}@{identity['revision']}"
        and reports["unit"]["result"]["backend"]
        == f"{identity['variant']}-unit-replay@{identity['revision']}",
        "coherence temperature linkage backend/child identity differs",
    )
    caches, cache_hashes = {}, {}
    for policy, folder, audited in (
        ("calibrated", "coherence-cache", calibrated),
        ("unit", "coherence-cache-unit", unit),
    ):
        require(
            digest(
                root / ("coherence-mini.json" if policy == "calibrated" else "coherence-unit.json")
            )
            == audited["sha256"],
            "coherence report changed since official replay",
        )
        paths = list((root / folder).iterdir())
        require(
            all(
                path.is_file()
                and not path.is_symlink()
                and re.fullmatch(r"[0-9a-f]{64}\.json", path.name)
                for path in paths
            ),
            "unsafe coherence linkage cache entry",
        )
        cache_hashes[policy] = {path.name: digest(path) for path in paths}
        require(
            cache_hashes[policy] == audited["raw_cache_sha256"],
            "coherence cache changed since official replay",
        )
        caches[policy] = {path.name: read(path) for path in paths}
        require(
            len(caches[policy]) == 1248,
            "coherence linkage requires each full 1248 cache population",
        )
    plan, source = coherence_linkage_requests(
        reports["calibrated"], caches["calibrated"], source_root
    )
    require(len(plan) == 1248, "coherence linkage request plan population differs")
    require(
        hashlib.sha256(_bytes([row["cache_file"] for row in plan])).hexdigest()
        == calibrated["replay"]["ordered_request_fingerprints_sha256"],
        "coherence linkage request plan differs from official replay",
    )
    raw = read(raw_file)
    require(
        isinstance(raw, dict) and all(re.fullmatch(r"[0-9a-f]{64}", key) for key in raw),
        "invalid raw-logits request inventory",
    )
    temperatures = {
        kind: value["temperature"] for kind, value in receipt["identity"]["calibration"].items()
    }
    require(
        set(temperatures) == {"choice", "noul", "score"},
        "coherence fitted primitive temperatures differ",
    )
    if cpu_evidence is not None:
        require(
            set(cpu_evidence.get("coherence", {})) == {"calibrated", "unit"},
            "CPU proof full coherence policy population differs",
        )
        for policy in ("calibrated", "unit"):
            policy_evidence = cpu_evidence["coherence"][policy]
            require(
                policy_evidence.get("backend") == reports[policy]["result"]["backend"]
                and policy_evidence.get("cache_sha256") == cache_hashes[policy]
                and isinstance(policy_evidence.get("requests"), list)
                and len(policy_evidence["requests"]) == 1248,
                "CPU proof coherence backend/cache/request population differs",
            )
    totals = {
        policy: {
            "questions": 0,
            "probabilities": 0,
            "max_abs_error": 0.0,
            "mismatch_count": 0,
            "native_field_mismatch_count": 0,
        }
        for policy in ("calibrated", "unit")
    }
    files, raw_keys, questions, maximum = {"calibrated": [], "unit": []}, [], 0, 0.0
    for index, row in enumerate(plan):
        request = DecisionRequest(state=row["state"], questions=row["questions"])
        key = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        require(key in raw, "coherence request lacks raw logits")
        raw_keys.append(key)
        for policy, temperature in (
            ("calibrated", temperatures),
            ("unit", {kind: 1.0 for kind in temperatures}),
        ):
            blob = json.dumps(
                {
                    "b": reports[policy]["result"]["backend"],
                    "s": row["state"],
                    "q": row["questions"],
                    "r": 0,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            name = hashlib.sha256(blob.encode()).hexdigest() + ".json"
            files[policy].append(name)
            require(
                name in caches[policy], "coherence policy lacks reconstructed request/cache key"
            )
            if cpu_evidence is None:
                verified = verify_temperature_answers(
                    request, raw[key], caches[policy][name], temperature
                )
            else:
                evidence = cpu_evidence["coherence"][policy]["requests"][index]
                minimal = set(evidence) == {"raw_key", "cache_file", "expected_answers"}
                require(
                    (
                        minimal
                        or (
                            evidence.get("state") == row["state"]
                            and evidence.get("request_questions") == row["questions"]
                            and evidence.get("request")
                            == {"state": row["state"], "questions": row["questions"]}
                        )
                    )
                    and evidence.get("raw_key") == key
                    and evidence.get("cache_file") == name,
                    "CPU proof official ordered request/cache/raw identities differ",
                )
                verified = compare_pinned_cpu_answers(
                    request, raw[key], caches[policy][name], temperature, evidence
                )
                for metric in totals[policy]:
                    totals[policy][metric] = (
                        max(totals[policy][metric], verified[metric])
                        if metric == "max_abs_error"
                        else totals[policy][metric] + verified[metric]
                    )
            maximum = max(maximum, verified["max_probability_error"])
            if policy == "calibrated":
                questions += verified["questions"]
    require(len(set(raw_keys)) == 1248, "coherence native raw-logit request keys are not unique")
    for policy, audited in (("calibrated", calibrated), ("unit", unit)):
        require(
            set(files[policy]) == set(caches[policy]),
            "coherence linkage has missing/unexpected request population",
        )
        require(
            hashlib.sha256(_bytes(files[policy])).hexdigest()
            == audited["replay"]["ordered_request_fingerprints_sha256"],
            "coherence policy request order differs from official replay",
        )
        if cpu_evidence is not None:
            evidence = cpu_evidence["coherence"][policy]
            require(
                evidence.get("ordered_cache_files") == files[policy]
                and evidence.get("ordered_cache_sha256")
                == hashlib.sha256(arithmetic_bytes(files[policy])).hexdigest()
                and evidence.get("ordered_raw_keys_sha256")
                == hashlib.sha256(arithmetic_bytes(raw_keys)).hexdigest()
                and all(evidence.get(k) == v for k, v in totals[policy].items()),
                "CPU proof request order or independently recomputed counters differ",
            )
    return {
        "status": "verified_same_raw_logits"
        if cpu_evidence is None
        else "verified_same_raw_logits_pinned_cpu",
        "complete": True,
        "expected_requests": 1248,
        "verified_requests_per_policy": 1248,
        "verified_question_records_per_policy": questions,
        "raw_logits_sha256": digest(raw_file),
        "raw_logits_saved_requests": len(raw),
        "used_raw_request_keys_sha256": hashlib.sha256(_bytes(raw_keys)).hexdigest(),
        "source": source,
        "request_plan_sha256": hashlib.sha256(_bytes(plan)).hexdigest(),
        "cache_inventory_sha256": {
            policy: hashlib.sha256(_bytes(hashes)).hexdigest()
            for policy, hashes in cache_hashes.items()
        },
        "temperatures": {"calibrated": temperatures, "unit": {kind: 1.0 for kind in temperatures}},
        "max_probability_error": maximum,
        "probability_absolute_tolerance": float(np.finfo(np.float32).eps),
        "pinned_cpu_comparison": totals if cpu_evidence is not None else None,
        "scope": "Pinned official full request/cache plan and each native answer recomputed from the same recorded raw logits at fitted type-T or unit-T; CPU FP32 only, no model call. Saved raw-logits.json may include non-coherence requests. Optional pinned Linux CPU evidence verifies archived full expected vectors separately from strict local macOS parity.",
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


def training_data_identity(cases):
    require(
        len(cases) == 1024
        and len({case.id for case in cases}) == 1024
        and all(case.split == "train" for case in cases)
        and sum(len(case.request.questions) for case in cases) == 1024,
        "training reference must contain 1024 unique frozen train cases/questions",
    )
    return {
        "cases": len(cases),
        "question_records": 1024,
        "dataset_sha256": fingerprint(cases),
        "ordered_requests_sha256": hashlib.sha256(
            json.dumps([request_fingerprint(case.request) for case in cases]).encode()
        ).hexdigest(),
        "type_counts": dict(Counter(q.type for case in cases for q in case.request.questions)),
        "scope": "Reference cases loaded from the verified prospective synthetic manifest",
    }


def load_research_tensors(path, *, max_bytes):
    """Inspect small torch.save research artifacts on CPU; never load base weights."""
    import torch

    path = Path(path)
    require(
        path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= max_bytes,
        "research tensor file is missing, unsafe or exceeds the size limit",
    )
    # The frozen app uses modern torch.save ZIP archives. Bound uncompressed storage
    # as well as the file size before weights_only parsing allocates any tensors.
    try:
        with zipfile.ZipFile(path) as archive:
            require(
                sum(entry.file_size for entry in archive.infolist()) <= max_bytes,
                "research tensor archive exceeds the uncompressed size limit",
            )
        return torch.load(path, map_location="cpu", weights_only=True)
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError(
            f"cannot safely read research tensor artifact: {type(exc).__name__}"
        ) from exc


def finite_fp32_tensor(value, shape, message):
    import torch

    require(
        torch.is_tensor(value)
        and value.device.type == "cpu"
        and value.dtype == torch.float32
        and tuple(value.shape) == tuple(shape)
        and bool(torch.isfinite(value).all()),
        message,
    )


def inspect_head_binary(path):
    import torch

    head = load_research_tensors(path, max_bytes=8 * 1024 * 1024)
    require(
        isinstance(head, dict) and set(head) == {"weight", "bias"}, "head tensor schema differs"
    )
    weight = head["weight"]
    require(
        torch.is_tensor(weight)
        and weight.ndim == 2
        and weight.shape[0] == 52
        and 0 < weight.shape[1] <= 16384,
        "head must contain 52 bounded candidate rows",
    )
    finite_fp32_tensor(weight, weight.shape, "head weight must be finite CPU FP32")
    finite_fp32_tensor(head["bias"], (52,), "head bias must be 52 finite CPU FP32 values")
    return {
        "status": "verified",
        "weight_shape": list(weight.shape),
        "bias_shape": [52],
        "dtype": "float32",
    }


def inspect_train_features(path, cases, *, hidden_size=None):
    import torch

    saved = load_research_tensors(path, max_bytes=128 * 1024 * 1024)
    require(
        isinstance(saved, dict) and set(saved) == {"features", "records"},
        "training feature artifact schema differs",
    )
    rows, features = saved["records"], saved["features"]
    require(isinstance(rows, list) and len(rows) == 1024, "training feature row count differs")
    validate_soft_records(rows, cases, "train")
    require(
        [(row["case_id"], row["question_id"]) for row in rows]
        == [(case.id, q.id) for case in cases for q in case.request.questions],
        "training feature row order differs from frozen train data",
    )
    require(
        torch.is_tensor(features)
        and features.ndim == 2
        and features.shape[0] == 1024
        and 0 < features.shape[1] <= 16384
        and (hidden_size is None or features.shape[1] == hidden_size),
        "training feature shape differs from rows/final head",
    )
    finite_fp32_tensor(features, features.shape, "training features must be finite CPU FP32")
    return {
        "status": "verified",
        "sha256": digest(path),
        "shape": list(features.shape),
        "dtype": "float32",
        "frozen_rows_verified": len(rows),
        "scope": "Exact source/group/split/type/label/soft-target/ordinal identities and order; native feature computation is not numerically rerun",
    }


def inspect_adapter_binary(directory, trainable_parameters):
    """Verify final saved adapters without constructing or loading the 12B model."""
    from safetensors import safe_open

    config = read(Path(directory) / "adapter_config.json")
    projections = {"q_proj", "k_proj", "v_proj", "o_proj"}
    require(
        config.get("peft_type") == "LORA"
        and config.get("r") == 8
        and config.get("lora_alpha") == 16
        and set(config.get("target_modules", [])) == projections
        and config.get("bias") == "none"
        and config.get("lora_dropout", 0) == 0
        and config.get("use_dora", False) is False
        and config.get("use_rslora", False) is False
        and not config.get("rank_pattern")
        and not config.get("alpha_pattern")
        and not config.get("modules_to_save")
        and not config.get("lora_bias", False),
        "final adapter config differs from fixed attention LoRA recipe",
    )
    path = Path(directory) / "adapter_model.safetensors"
    require(0 < path.stat().st_size <= 512 * 1024 * 1024, "adapter exceeds the size limit")
    pairs, shapes, count, found = {}, {}, 0, set()
    try:
        with safe_open(path, framework="pt", device="cpu") as saved:
            for name in saved.keys():
                match = re.fullmatch(
                    r"(?P<module>.+\.(?P<projection>q_proj|k_proj|v_proj|o_proj))\.lora_(?P<side>A|B)\.weight",
                    name,
                )
                require(match is not None, "unexpected non-attention adapter tensor")
                shape = saved.get_slice(name).get_shape()
                side, module = match["side"], match["module"]
                require(
                    len(shape) == 2
                    and all(0 < axis <= 65536 for axis in shape)
                    and shape[0 if side == "A" else 1] == 8,
                    "adapter tensor rank/shape differs",
                )
                require(saved.get_slice(name).get_dtype() == "F32", "adapter tensor must be FP32")
                tensor = saved.get_tensor(name)
                finite_fp32_tensor(tensor, shape, "adapter tensor must be finite CPU FP32")
                pairs.setdefault(module, set()).add(side)
                shapes[name] = shape
                count += tensor.numel()
                found.add(match["projection"])
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError(f"cannot safely read final adapter tensors: {type(exc).__name__}") from exc
    require(
        bool(pairs)
        and found == projections
        and all(sides == {"A", "B"} for sides in pairs.values()),
        "final attention adapter A/B pairs or projection support are incomplete",
    )
    require(
        type(trainable_parameters) is int and trainable_parameters == count,
        "final adapter parameter count differs from training receipt",
    )
    return {
        "status": "verified",
        "tensors": len(shapes),
        "tensor_shapes": shapes,
        "parameters": count,
        "dtype": "float32",
        "scope": "Final unmerged attention A/B tensors and config; no requirement that unused media branches received gradients; no optimizer replay",
    }


def audit_training(root, kind, train_cases, receipt):
    if kind not in ("head", "attention_lora"):
        return {"status": "not_run", "complete": True, "scope": "Declared non-training control"}
    root = Path(root)
    data_identity = training_data_identity(train_cases)
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
        require(
            bool(re.fullmatch(r"[0-9a-f]{64}", expected)), "invalid trained head binary identity"
        )
        present = file.is_file() and not file.is_symlink()
        if present:
            require(digest(file) == expected, "trained head binary hash differs")
        tensors = inspect_head_binary(file) if present else {"status": "not_downloaded"}
        feature_file = root / "train-features.pt"
        feature_present = feature_file.is_file() and not feature_file.is_symlink()
        features = (
            inspect_train_features(
                feature_file,
                train_cases,
                hidden_size=tensors["weight_shape"][1] if present else None,
            )
            if feature_present
            else {"status": "not_downloaded"}
        )
        complete = present and feature_present
        return {
            "status": "recipe_verified" if complete else "incomplete_missing_binary",
            "complete": complete,
            "recipe_verified": True,
            "steps": 128,
            "training_receipt_sha256": digest(root / "head-training.json"),
            "training_data": data_identity,
            "binary_sha256": expected,
            "binary_status": "verified" if present else "not_downloaded",
            "final_tensors": tensors,
            "features": features,
            "finite_loss_updates": 128,
            "gradient_trace": {"status": "unrecorded", "recorded_updates": 0},
            "sampled_batch_ids": {"status": "unrecorded"},
            "scope": "CPU final head and complete frozen feature population. Frozen helper enforces finite losses/gradient clipping/weights, but no raw gradient norms or sampled batch IDs were recorded; no optimizer replay",
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
    expected_order = list(train_cases)
    random.Random(42).shuffle(expected_order)
    require(
        ids == [case.id for case in expected_order]
        and all(row.get("case_id") == row["case_ids"][-1] for row in updates),
        "LoRA seeded micro-batch order/last case identity differs",
    )
    require(saved.get("losses") == updates, "LoRA final loss trace differs from update log")
    require(
        all(
            type(row.get("gradient_norm")) in (int, float)
            and math.isfinite(row["gradient_norm"])
            and row["gradient_norm"] >= 0
            for row in updates
        ),
        "LoRA gradient norms must be recorded finite nonnegative values",
    )
    require(
        isinstance(saved["adapter_files"], dict)
        and "adapter_model.safetensors" in saved["adapter_files"]
        and "adapter_config.json" in saved["adapter_files"],
        "final adapter weight/config inventory missing",
    )
    require(
        receipt["identity"].get("research_adapter_files") == saved["adapter_files"],
        "child identity/final adapter inventory differ",
    )
    artifacts = {}
    for name, declared in saved["adapter_files"].items():
        require(Path(name).name == name, "unsafe adapter file path")
        require(
            bool(re.fullmatch(r"[0-9a-f]{64}", declared["sha256"]))
            and type(declared["bytes"]) is int
            and declared["bytes"] > 0,
            "invalid final adapter file identity",
        )
        file = directory / "adapter" / name
        present = file.is_file() and not file.is_symlink()
        if present:
            require(
                digest(file) == declared["sha256"] and file.stat().st_size == declared["bytes"],
                "adapter binary hash/size differs",
            )
        artifacts[name] = {**declared, "status": "verified" if present else "not_downloaded"}
    complete = all(file["status"] == "verified" for file in artifacts.values())
    tensors = (
        inspect_adapter_binary(directory / "adapter", saved.get("trainable_parameters"))
        if complete
        else {"status": "not_downloaded"}
    )
    return {
        "status": "recipe_verified" if complete else "incomplete_missing_binary",
        "complete": complete,
        "recipe_verified": True,
        "updates": len(updates),
        "training_cases": len(ids),
        "training_data": data_identity,
        "training_receipt_sha256": digest(directory / "training.json"),
        "update_log_sha256": digest(directory / "updates.jsonl"),
        "update_order": "verified_seed42_one_epoch_accum8",
        "gradient_trace": {
            "status": "verified_finite_recorded_norms",
            "recorded_updates": len(updates),
            "minimum": min(row["gradient_norm"] for row in updates),
            "maximum": max(row["gradient_norm"] for row in updates),
            "scope": "Recorded total pre-clipping norms, not per-parameter gradients or a requirement that every media adapter was used",
        },
        "adapter_files": artifacts,
        "final_tensors": tensors,
        "scope": "Final update 128 only; full recorded update/data-identity trace and final tensor schema, no optimizer replay or public checkpoint selection",
    }


def analyze_profile(
    root, declaration, stage_result, data, public_source, coherence_root, cpu_evidence=None
):
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
        lambda: audit_synthetic(directory, data["calibration"], data["test"], receipt=receipt)
    )
    result["public"] = {
        policy: guarded(
            lambda folder=folder, policy=policy: audit_public_profile(
                directory / folder,
                public_source,
                receipt=receipt,
                policy=policy,
                protocol=data["protocol"],
            )
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
    result["public_calibration_linkage"] = guarded(
        lambda: public_calibration_identity(result["public"]["calibrated"], receipt)
    )
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
                receipt=receipt,
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
    result["coherence_calibration_linkage"] = guarded(
        lambda: audit_coherence_linkage(
            directory,
            result["coherence"]["calibrated"],
            result["coherence"]["unit"],
            receipt,
            source_root=coherence_root,
        )
    )
    if cpu_evidence is not None:
        strict = result["coherence_calibration_linkage"]
        result["pinned_cpu_calibration"] = guarded(
            lambda: audit_pinned_cpu_calibration(directory, cpu_evidence, receipt)
        )
        alternate = guarded(
            lambda: audit_coherence_linkage(
                directory,
                result["coherence"]["calibrated"],
                result["coherence"]["unit"],
                receipt,
                source_root=coherence_root,
                cpu_evidence=cpu_evidence,
            )
        )
        result["coherence_calibration_linkage"] = {
            **alternate,
            "strict_local_replay": strict,
            "verification_runtime": "separate pinned Linux x86 CPU proof; strict local macOS result preserved",
        }
    result["coherence_effect"] = guarded(
        lambda: coherence_effect(
            result["coherence"]["calibrated"],
            result["coherence"]["unit"],
            result["coherence_calibration_linkage"],
        )
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
        and result["public_calibration_linkage"].get("complete") is True
        and result["coherence_calibration_linkage"].get("complete") is True
        and result["training"].get("status") in ("not_run", "recipe_verified")
        and result["training"].get("complete") is True
        and result["identity_integrity"].get("status") == "metadata_verified"
        and (cpu_evidence is None or result["pinned_cpu_calibration"].get("complete") is True)
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
            if key not in {"records", "raw_answers", "raw_runtime", "raw_requests", "manifest"}
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
    arithmetic_proof=None,
    arithmetic_proof_sha256=None,
    arithmetic_verifier_sha256=None,
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
    require(
        arithmetic_proof is not None
        or (arithmetic_proof_sha256 is None and arithmetic_verifier_sha256 is None),
        "CPU arithmetic SHA inputs require proof path",
    )
    arithmetic = (
        load_arithmetic_proof(
            arithmetic_proof, arithmetic_proof_sha256, arithmetic_verifier_sha256, roots, protocol
        )
        if arithmetic_proof is not None
        else None
    )
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
                        root,
                        declaration,
                        stages[name],
                        data,
                        public_source,
                        coherence_root,
                        arithmetic["proof"]["profiles"][declaration["id"]]
                        if arithmetic is not None
                        else None,
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
    arithmetic_metadata = None
    if arithmetic is not None:
        linked = [p.get("coherence_calibration_linkage", {}) for p in profiles.values()]
        verified = all(
            p.get("pinned_cpu_calibration", {}).get("complete") is True
            and p.get("coherence_calibration_linkage", {}).get("status")
            == "verified_same_raw_logits_pinned_cpu"
            and p["coherence_calibration_linkage"].get("complete") is True
            for p in profiles.values()
        )
        exact = (
            all(
                not policy["mismatch_count"] and not policy["native_field_mismatch_count"]
                for row in linked
                for policy in row.get("pinned_cpu_comparison", {}).values()
            )
            if verified
            else None
        )
        require(
            not verified or arithmetic["proof"].get("exact_arithmetic_match") is exact,
            "CPU proof headline differs from independently verified complete vectors",
        )
        arithmetic_metadata = {
            **arithmetic["metadata"],
            "status": "all_expected_vectors_verified"
            if verified
            else "incomplete_vector_verification",
            "complete": verified,
            "exact_expected_vectors": exact,
            "verified_requests": sum(
                2 * row.get("verified_requests_per_policy", 0) for row in linked
            ),
            "verified_question_records": sum(
                2 * row.get("verified_question_records_per_policy", 0) for row in linked
            ),
            "strict_local_status_counts": dict(
                Counter(
                    row.get("strict_local_replay", {}).get("status", "missing") for row in linked
                )
            ),
            "strict_local_parity": "Preserved per profile; pinned CPU verification does not turn local macOS numeric mismatches into strict parity",
        }
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
        "pinned_cpu_arithmetic": arithmetic_metadata,
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
        "The JSON retains per-type/template metrics, paired source-group intervals, all raw hashes, errors and unexecuted populations. Native public and both coherence temperature policies require full raw-logit linkage for a complete profile; missing linkage prevents temperature attribution. Coherence outcome rows are not independent denominators. Raw-logit replay latency is CPU replay, not GPU latency. Historical Jev-Omni is reported separately. Native media52/fixture answers have no saved logits, so their temperature linkage is not claimed.",
        "Optional independently archived Linux x86 CPU arithmetic evidence is explicitly separate from strict local macOS parity. Its complete expected vectors are checked against original raw logits, fitted temperatures, official request/cache order, source/input byte inventories and native answer fields; original strict results remain in the JSON.",
        "",
    ]
    proof = report.get("pinned_cpu_arithmetic")
    if proof is not None:
        lines += [
            f"Pinned CPU arithmetic: `{proof['status']}`; {proof['verified_requests']} verified requests and {proof['verified_question_records']} question records. Proof SHA256 `{proof['proof_sha256']}`; verifier SHA256 `{proof['verifier_source_sha256']}`.",
            "",
            f"Strict local coherence linkage status counts remain `{proof['strict_local_status_counts']}`. Pinned CPU evidence verifies the original Linux x86 arithmetic separately; it does not establish strict macOS parity.",
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
    parser.add_argument("--arithmetic-proof", type=Path)
    parser.add_argument("--arithmetic-proof-sha256")
    parser.add_argument("--arithmetic-verifier-sha256")
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
        arithmetic_proof=args.arithmetic_proof,
        arithmetic_proof_sha256=args.arithmetic_proof_sha256,
        arithmetic_verifier_sha256=args.arithmetic_verifier_sha256,
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
