#!/usr/bin/env python3
"""Recompute a compact release report from complete, provenance-bound observations.

No models are loaded. CUDA request latency and MLX prepared-forward latency are
reported separately; their ratio is deliberately not interpreted as a speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

FILES = {
    "train": "train.jsonl",
    "calibration": "calibration.jsonl",
    "test": "test.jsonl",
    "media": "media-test.jsonl",
    "regression_text": "regression-text.jsonl",
}
COUNTS = {"train": 2048, "calibration": 256, "test": 256, "media": 52, "regression_text": 128}
POPULATIONS = {
    "fresh_boolq": ("test", None),
    "old_boolq": ("regression_text", None),
    "media_all": ("media", None),
    "mnist": ("media", "mnist"),
    "fsdd": ("media", "fsdd"),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _finite(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("measurement must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError("measurement is outside its valid range")
    return float(value)


def load_population(directory, expected_counts=None):
    from s1.evaluation.datasets import audit_splits, fingerprint, load_cases, request_fingerprint

    root = Path(directory)
    manifest = read_json(root / "manifest.json")
    fixed_release = expected_counts is None
    expected_counts = COUNTS if fixed_release else expected_counts
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha256(root / name) != expected:
            raise ValueError("frozen dataset file checksum mismatch")
    data, identities = {}, {}
    for role, name in FILES.items():
        cases = load_cases(root / name)
        receipt = manifest["datasets"][name]
        expected_split = role if role in ("train", "calibration") else "test"
        values = {
            "cases": len(cases),
            "questions": sum(len(c.request.questions) for c in cases),
            "file_sha256": sha256(root / name),
            "dataset_sha256": fingerprint(cases),
            "case_ids": [c.id for c in cases],
            "group_ids": [c.group_id or c.id for c in cases],
            "request_sha256s": [request_fingerprint(c.request) for c in cases],
            "split": expected_split,
        }
        if (
            len(cases) != expected_counts[role]
            or any(c.split != expected_split for c in cases)
            or any(receipt.get(key) != value for key, value in values.items())
        ):
            raise ValueError(f"{role}: frozen population receipt mismatch")
        data[role], identities[role] = cases, values
    if not audit_splits(root / name for name in FILES.values())["valid"]:
        raise ValueError("frozen populations overlap across fitting/test splits")
    if len({case.group_id or case.id for case in data["test"]}) != len(data["test"]):
        raise ValueError("fresh BoolQ test requires unique source groups")
    if fixed_release and (
        sum(case.task_id == "mnist" for case in data["media"]) != 32
        or sum(case.task_id == "fsdd" for case in data["media"]) != 20
    ):
        raise ValueError("media population requires exactly 32 MNIST and 20 FSDD cases")
    return data, identities, sha256(root / "manifest.json")


def decision_index(cases):
    return {
        (case.id, question.id): {
            "case_id": case.id,
            "question_id": question.id,
            "group_id": case.group_id or case.id,
            "task_id": case.task_id,
            "type": question.type,
            "labels": question.labels(),
            "gold": case.gold[question.id],
            "gold_index": question.labels().index(case.gold[question.id]),
        }
        for case in cases
        for question in case.request.questions
    }


def _complete_keys(rows, expected, *, id_key="case_id"):
    index = {}
    for row in rows:
        key = row.get(id_key), row.get("question_id")
        if key in index:
            raise ValueError("duplicate decision identity")
        index[key] = row
    if index.keys() != expected.keys():
        raise ValueError("complete exact decision population required; no eligible subset")
    return index


def _probabilities(row, expected):
    values = row.get("probabilities")
    if not isinstance(values, dict) or list(values) != expected["labels"]:
        raise ValueError("candidate labels or ordering differ from frozen population")
    probabilities = np.array([_finite(value) for value in values.values()])
    if (probabilities > 1).any() or not np.isclose(probabilities.sum(), 1, atol=1e-6, rtol=0):
        raise ValueError("invalid probability distribution")
    if row.get("gold") != expected["gold"] or row.get("type") != expected["type"]:
        raise ValueError("gold label or question type differs from frozen population")
    return probabilities.tolist()


def validate_cuda(report, cases, dataset_identity, run_manifest, backend, temperature):
    from s1.benchmark import fingerprint

    expected = decision_index(cases)
    if (
        report.get("backend") != backend
        or report.get("coverage") != 1
        or report.get("counts") != {"ok": len(expected)}
        or report.get("warmup_errors")
        or report.get("case_ids") != [case.id for case in cases]
        or report.get("dataset_sha256") != fingerprint(cases)
    ):
        raise ValueError("CUDA report has incomplete coverage or mismatched population")
    metadata = report.get("metadata", {})
    if any(
        metadata.get(key) != value
        for key, value in {
            "identity": run_manifest["identity"],
            "base_model": run_manifest["recipe"]["model"],
            "base_revision": run_manifest["recipe"]["revision"],
            "temperature": temperature,
            "release_dataset_sha256": dataset_identity["dataset_sha256"],
        }.items()
    ):
        raise ValueError("CUDA report provenance or calibration differs")
    observed = _complete_keys(report.get("records", []), expected)
    rows = []
    for key, record in expected.items():
        value = observed[key]
        if value.get("status") != "ok":
            raise ValueError("every declared CUDA decision must succeed")
        rows.append(
            {
                **record,
                "probabilities": _probabilities(value, record),
                "latency_samples_ms": [_finite(value.get("latency_ms"))],
            }
        )
    return rows


def validate_calibration(report, cases):
    from s1.training import fit_temperature

    expected = decision_index(cases)
    records = _complete_keys(report.get("records", []), expected)
    logits, golds = [], []
    for key, truth in expected.items():
        record = records[key]
        values = record.get("logits")
        if (
            record.get("labels") != truth["labels"]
            or record.get("gold_index") != truth["gold_index"]
            or not isinstance(values, list)
            or len(values) != len(truth["labels"])
            or not np.isfinite(np.asarray(values, dtype=float)).all()
        ):
            raise ValueError(
                "calibration logits/golds differ from complete frozen calibration split"
            )
        logits.append(values)
        golds.append(truth["gold_index"])
    fitted = fit_temperature(logits, golds)
    if report.get("n") != len(expected) or any(
        not math.isclose(report.get(key, float("nan")), fitted[key], rel_tol=1e-10, abs_tol=1e-12)
        for key in ("temperature", "nll_before", "nll_after")
    ):
        raise ValueError("recorded temperature fit differs from calibration observations")
    return fitted


def validate_mlx_provenance(report, manifest, artifacts, dataset_manifest_sha256):
    if report.get("status") != "ok" or report.get("release_validation_complete") is not True:
        raise ValueError("MLX report must have completed its full release population")
    if any(
        report.get(key) != value
        for key, value in {
            "source_model": manifest["recipe"]["model"],
            "source_revision": manifest["recipe"]["revision"],
            "prompt_version": 1,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "source_s1_config_sha256": artifacts["merged"]["s1_config.json"],
        }.items()
    ):
        raise ValueError("MLX source checkpoint, sidecar or frozen dataset identity differs")
    files = report.get("source_checkpoint_files", [])
    expected = {
        name: value
        for name, value in artifacts["merged"].items()
        if name == "config.json"
        or name.endswith(".safetensors")
        or name.endswith(".safetensors.index.json")
    }
    names = [record.get("path") for record in files]
    if len(set(names)) != len(names) or set(names) != set(expected) or not expected:
        raise ValueError(
            "MLX source checkpoint inventory must include every CUDA config/index/shard"
        )
    for record in files:
        if (
            record.get("sha256") != expected[record["path"]]
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] <= 0
        ):
            raise ValueError("MLX source checkpoint file differs from CUDA artifact receipt")
    if files != sorted(files, key=lambda row: row["path"]) or digest(files) != report.get(
        "source_checkpoint_sha256"
    ):
        raise ValueError("MLX source checkpoint digest differs from exhaustive file inventory")
    measurement = report.get("measurement", {})
    if measurement.get("not_end_to_end_request_latency") is not True:
        raise ValueError("MLX latency scope must identify prepared-forward measurement")


def validate_mlx_split(report, role, cases):
    from s1.evaluation.datasets import request_fingerprint

    expected = decision_index(cases)
    case_lookup = {case.id: case for case in cases}
    records = report.get("cases", {}).get(role, [])
    if len(records) != len(cases) or {record.get("id") for record in records} != set(case_lookup):
        raise ValueError(f"MLX {role}: exact complete case population required")
    temperature = _finite(report["calibration"]["temperature"], positive=True)
    repeat = report["measurement"].get("repeat_per_case")
    if type(repeat) is not int or repeat < 1:
        raise ValueError("MLX repetition count must be positive")
    rows, calibration_records = [], []
    for record in records:
        case = case_lookup[record["id"]]
        if (
            record.get("status") != "ok"
            or record.get("split") != case.split
            or record.get("group_id") != (case.group_id or case.id)
            or record.get("request_sha256") != request_fingerprint(case.request)
            or set(record.get("answers", {})) != {q.id for q in case.request.questions}
        ):
            raise ValueError(f"MLX {role}: status, request/group or question identity differs")
        samples = record.get("prepared_forward_samples_ms", [])
        if len(samples) != repeat:
            raise ValueError("MLX latency sample population is incomplete")
        samples = [_finite(value) for value in samples]
        for question in case.request.questions:
            truth = expected[(case.id, question.id)]
            answer = record["answers"][question.id]
            if answer.get("labels") != truth["labels"]:
                raise ValueError("MLX candidate label ordering differs")
            probabilities = _probabilities(answer, truth)
            logits = np.asarray(answer.get("raw_logits"), dtype=float)
            if logits.shape != (len(probabilities),) or not np.isfinite(logits).all():
                raise ValueError("MLX raw candidate logits must be complete and finite")
            scaled = logits / temperature
            softmax = np.exp(scaled - scaled.max())
            softmax /= softmax.sum()
            if not np.allclose(probabilities, softmax, atol=1e-10, rtol=1e-10):
                raise ValueError(
                    "MLX probabilities do not use the reported calibration temperature"
                )
            if answer.get("prediction") != truth["labels"][int(softmax.argmax())]:
                raise ValueError("MLX prediction differs from calibrated probabilities")
            rows.append({**truth, "probabilities": probabilities, "latency_samples_ms": samples})
            calibration_records.append(
                {
                    "case_id": case.id,
                    "question_id": question.id,
                    "labels": truth["labels"],
                    "gold_index": truth["gold_index"],
                    "logits": logits.tolist(),
                }
            )
    _complete_keys(rows, expected)
    summary = report.get("summary", {}).get(role, {})
    if any(
        summary.get(key) != value
        for key, value in {
            "expected_cases": len(cases),
            "successful_cases": len(cases),
            "case_coverage": 1,
            "expected_decisions": len(expected),
            "scored_decisions": len(expected),
            "decision_coverage": 1,
        }.items()
    ):
        raise ValueError("MLX declared coverage differs from complete record population")
    if role == "calibration":
        validate_calibration({**report["calibration"], "records": calibration_records}, cases)
    return rows


def summarize_rows(rows, runtime):
    from s1.metrics import summarize

    width = max(len(row["probabilities"]) for row in rows)
    padded = np.zeros((len(rows), width))
    for i, row in enumerate(rows):
        padded[i, : len(row["probabilities"])] = row["probabilities"]
    metrics = summarize(
        padded, [row["gold_index"] for row in rows], [len(row["labels"]) for row in rows]
    )
    # Count request latency once per case, even if a request has several questions.
    requests = {}
    for row in rows:
        previous = requests.setdefault(row["case_id"], row["latency_samples_ms"])
        if previous != row["latency_samples_ms"]:
            raise ValueError("questions in the same request have inconsistent latency samples")
    samples = [value for values in requests.values() for value in values]
    return {
        "status": "complete",
        "cases": len(requests),
        "decisions": len(rows),
        "coverage": 1.0,
        "metrics": {
            "accuracy": metrics["acc"],
            "nll": metrics["nll"],
            "brier": metrics["brier"],
            "ece_15": metrics["ece"],
        },
        "latency_ms": {
            "scope": "CUDA end-to-end backend request"
            if runtime == "cuda"
            else "MLX synchronized prepared tensors to candidate logits",
            "requests": len(requests),
            "samples": len(samples),
            "p50": float(np.percentile(samples, 50)),
            "p95": float(np.percentile(samples, 95)),
        },
    }


def paired_accuracy(left, right, *, seed=42, resamples=10000):
    from s1.evaluation.statistics import cluster_interval

    a = {(row["case_id"], row["question_id"]): row for row in left}
    b = {(row["case_id"], row["question_id"]): row for row in right}
    if len(a) != len(left) or len(b) != len(right) or a.keys() != b.keys():
        raise ValueError("paired accuracy requires the exact complete population")
    changes, groups = [], []
    for key, before in a.items():
        after = b[key]
        if any(before[field] != after[field] for field in ("labels", "gold", "group_id", "type")):
            raise ValueError("paired decision labels/golds/groups differ")
        changes.append(
            int(np.argmax(after["probabilities"]) == after["gold_index"])
            - int(np.argmax(before["probabilities"]) == before["gold_index"])
        )
        groups.append(before["group_id"])
    interval = cluster_interval(changes, groups, seed=seed, resamples=resamples)
    return {
        "status": interval["status"],
        "direction": "target minus reference",
        "decisions": len(changes),
        "coverage": 1.0,
        "delta": interval["estimate"],
        "ci_95": interval["interval"],
        "source_groups": interval["groups"],
        "method": "paired source-group percentile bootstrap",
        "seed": seed,
        "resamples": resamples,
    }


def build_summary(
    run_dir, dataset_dir, mlx_report=None, *, expected_counts=None, seed=42, resamples=10000
):
    root = Path(run_dir)
    data, identities, dataset_sha = load_population(dataset_dir, expected_counts)
    manifest, artifacts = read_json(root / "manifest.json"), read_json(root / "artifacts.json")
    identity = manifest["identity"]
    if digest({key: manifest[key] for key in ("recipe", "datasets", "implementation")}) != identity:
        raise ValueError("release manifest identity checksum differs")
    if artifacts.get("identity") != identity:
        raise ValueError("artifact receipt belongs to another run")
    for stage in ("train", "evaluate"):
        receipt = read_json(root / stage / "receipt.json")
        if receipt.get("status") != "completed" or receipt.get("identity") != identity:
            raise ValueError(f"{stage} requires a complete matching run receipt")
    for role, observed in identities.items():
        if any(
            manifest["datasets"][role].get(key) != observed[key]
            for key in ("cases", "dataset_sha256", "file_sha256")
        ):
            raise ValueError("run dataset identity differs from frozen population")
    if manifest["datasets"].get("preparation_manifest_sha256") != dataset_sha:
        raise ValueError("run preparation manifest differs from frozen dataset")
    calibration = {
        "base_cuda": validate_calibration(
            read_json(root / "evaluate/base-calibration.json"), data["calibration"]
        ),
        "trained_cuda": validate_calibration(
            read_json(root / "train/postmerge-calibration.json"), data["calibration"]
        ),
    }
    sidecar_path = root / "merged/s1_config.json"
    sidecar = read_json(sidecar_path)
    if sha256(sidecar_path) != artifacts["merged"]["s1_config.json"] or any(
        sidecar.get(key) != value
        for key, value in {
            "source_model": manifest["recipe"]["model"],
            "source_revision": manifest["recipe"]["revision"],
            "training_recipe_sha256": identity,
            "prompt_version": 1,
            "temperature": calibration["trained_cuda"]["temperature"],
            "calibration_dataset_sha256": identities["calibration"]["dataset_sha256"],
        }.items()
    ):
        raise ValueError(
            "merged checkpoint sidecar differs from trained calibration or artifact receipt"
        )
    observations = {}
    inputs = {
        "run_manifest_sha256": sha256(root / "manifest.json"),
        "artifacts_receipt_sha256": sha256(root / "artifacts.json"),
        "cuda_reports": {},
    }
    for kind in ("base", "trained"):
        backend = f"{kind}_cuda"
        observations[backend] = {}
        for role in ("test", "media", "regression_text"):
            path = root / "evaluate" / f"{kind}-{role}.json"
            observations[backend][role] = validate_cuda(
                read_json(path),
                data[role],
                identities[role],
                manifest,
                kind,
                calibration[backend]["temperature"],
            )
            inputs["cuda_reports"][path.name] = sha256(path)
    mlx = read_json(mlx_report) if mlx_report is not None else None
    if mlx is not None:
        validate_mlx_provenance(mlx, manifest, artifacts, dataset_sha)
        validate_mlx_split(mlx, "calibration", data["calibration"])
        calibration["trained_mlx"] = {
            key: mlx["calibration"][key] for key in ("temperature", "n", "nll_before", "nll_after")
        }
        observations["trained_mlx"] = {
            role: validate_mlx_split(mlx, role, data[role]) for role in ("test", "media")
        }
        inputs["mlx_report_sha256"] = sha256(mlx_report)
    else:
        calibration["trained_mlx"] = {"status": "not_run"}
    populations = {}
    for label, (role, task) in POPULATIONS.items():
        selected = [case for case in data[role] if task is None or case.task_id == task]
        if not selected:
            raise ValueError(f"missing declared population {label}")
        keys = set(decision_index(selected))
        backends = {}
        for backend in ("base_cuda", "trained_cuda", "trained_mlx"):
            if role not in observations.get(backend, {}):
                backends[backend] = {"status": "not_run", "reason": "no complete report provided"}
                continue
            rows = [
                row
                for row in observations[backend][role]
                if (row["case_id"], row["question_id"]) in keys
            ]
            backends[backend] = summarize_rows(rows, "mlx" if backend.endswith("mlx") else "cuda")
        populations[label] = {
            "cases": len(selected),
            "decisions": len(keys),
            "source_groups": len({case.group_id or case.id for case in selected}),
            "dataset_sha256": identities[role]["dataset_sha256"],
            "backends": backends,
        }
    comparisons = {
        "base_to_trained_cuda": paired_accuracy(
            observations["base_cuda"]["test"],
            observations["trained_cuda"]["test"],
            seed=seed,
            resamples=resamples,
        )
    }
    comparisons["trained_cuda_to_mlx"] = (
        paired_accuracy(
            observations["trained_cuda"]["test"],
            observations["trained_mlx"]["test"],
            seed=seed,
            resamples=resamples,
        )
        if mlx is not None
        else {"status": "not_run"}
    )
    return {
        "schema_version": 1,
        "status": "complete" if mlx else "cuda_complete_mlx_not_run",
        "release_identity": identity,
        "source_model": manifest["recipe"]["model"],
        "source_revision": manifest["recipe"]["revision"],
        "dataset_manifest_sha256": dataset_sha,
        "inputs": inputs,
        "calibration": calibration,
        "populations": populations,
        "fresh_boolq_paired_accuracy": comparisons,
        "metric_definitions": {
            "nll_probability_floor": 1e-12,
            "ece_equal_width_bins": 15,
            "brier": "sum over all legal candidates; mean over decisions",
        },
        "scope": "descriptive fixed-population results; no test-selected acceptance gate",
        "latency_caveat": "CUDA includes backend preprocessing and response construction; MLX uses prepared tensors. These timings do not establish a cross-runtime speedup.",
        "mlx_quantization": mlx.get("quantization") if mlx else None,
    }


def markdown(summary):
    lines = [
        "# Release evaluation",
        "",
        "Metrics use the complete frozen populations. Each runtime fits temperature on the separate 256-case calibration split.",
        "",
        "| Population | Backend | Cases | Accuracy | NLL | Brier | ECE-15 | p50 / p95 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, population in summary["populations"].items():
        for backend, row in population["backends"].items():
            if row["status"] == "not_run":
                lines.append(
                    f"| {name} | {backend} | {population['cases']} | not_run | — | — | — | — |"
                )
                continue
            m, latency = row["metrics"], row["latency_ms"]
            lines.append(
                f"| {name} | {backend} | {row['cases']} | {m['accuracy']:.2%} | {m['nll']:.4f} | {m['brier']:.4f} | {m['ece_15']:.4f} | {latency['p50']:.2f} / {latency['p95']:.2f} |"
            )
    lines.extend(
        [
            "",
            summary["latency_caveat"],
            "",
            "Fresh BoolQ paired accuracy differences (target minus reference; 95% source-group bootstrap CI):",
            "",
        ]
    )
    for name, comparison in summary["fresh_boolq_paired_accuracy"].items():
        if comparison["status"] == "not_run":
            lines.append(f"- {name}: not_run.")
        else:
            low, high = comparison["ci_95"]
            lines.append(
                f"- {name}: {100 * comparison['delta']:+.2f} percentage points [{100 * low:+.2f}, {100 * high:+.2f}], n={comparison['decisions']}; {comparison['resamples']:,} resamples, seed {comparison['seed']}."
            )
    lines.extend(
        [
            "",
            "Calibration temperatures: "
            + "; ".join(
                f"{name}={value['temperature']:.6f}"
                if "temperature" in value
                else f"{name}=not_run"
                for name, value in summary["calibration"].items()
            )
            + ".",
            "",
            summary["scope"] + ".",
            "",
            f"Release identity: `{summary['release_identity']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mlx-report", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/release/datasets"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = build_summary(args.run_dir, args.dataset_dir, args.mlx_report)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        (args.output / "README.md").write_text(markdown(summary))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"release summary: {exc}\n")
    print(f"{summary['status']}: {args.output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    raise SystemExit(main())
