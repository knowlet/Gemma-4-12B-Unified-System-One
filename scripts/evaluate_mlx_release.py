#!/usr/bin/env python3
"""Calibrate one MLX export on 256 held-out cases, then evaluate 256 + 52 cases.

All provenance, split and input checks precede MLX initialization. The evaluator
uses raw legal candidate logits and the repository's NumPy temperature fitter;
it never fits to test labels or copies the source checkpoint's temperature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

EXPECTED_CASES = {"calibration": 256, "test": 256, "media": 52}
DATASET_NAMES = {
    "calibration": "calibration.jsonl",
    "test": "test.jsonl",
    "media": "media-test.jsonl",
}


@lru_cache(maxsize=1)
def shared_scripts():
    directory = Path(__file__).resolve().parent
    return (
        runpy.run_path(str(directory / "verify_mlx_export.py")),
        runpy.run_path(str(directory / "prepare_mlx_validation.py")),
    )


def _digest(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def read_pinned_json(path, expected_sha256):
    raw = Path(path).read_bytes()
    if not _digest(expected_sha256) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"manifest SHA256 mismatch: {Path(path).name}")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("manifest must be a JSON object")
    return result


def validate_manifests(manifests, dataset_manifest, *, expected_counts=None):
    """Bind captured cases to the pinned data receipt and audit all split identities."""
    expected_counts = EXPECTED_CASES if expected_counts is None else expected_counts
    if set(manifests) != set(EXPECTED_CASES):
        raise ValueError("calibration, test and media manifests are all required")
    if dataset_manifest.get("split_audit", {}).get("valid") is not True:
        raise ValueError("release dataset split audit must be valid")
    datasets = dataset_manifest.get("datasets", {})
    train = datasets.get("train.jsonl")
    if not train or train.get("split") != "train":
        raise ValueError("dataset receipt must include the training split for leakage checks")
    identity = None
    for role, manifest in manifests.items():
        expected_split = "calibration" if role == "calibration" else "test"
        if manifest.get("split") != expected_split or manifest.get("prompt_version") != 1:
            raise ValueError(f"{role}: incorrect split or prompt version")
        source = (
            manifest.get("source_model"),
            manifest.get("source_revision"),
            manifest.get("source_checkpoint_sha256"),
            tuple(manifest.get("letters", [])),
            manifest.get("source_s1_config_sha256"),
        )
        if not source[0] or not source[1] or not _digest(source[2]) or not _digest(source[4]):
            raise ValueError("captured inputs require a hashed source checkpoint identity")
        if identity is not None and source != identity:
            raise ValueError("captured manifests refer to different checkpoints or candidate ids")
        identity = source
        letters = source[3]
        if (
            len(letters) != 52
            or len(set(letters)) != 52
            or any(type(token) is not int or token < 0 for token in letters)
        ):
            raise ValueError("capture must contain 52 distinct nonnegative candidate token ids")
        cases = manifest.get("cases", [])
        receipt = datasets.get(DATASET_NAMES[role], {})
        if len(cases) != expected_counts[role] or receipt.get("cases") != expected_counts[role]:
            raise ValueError(f"{role}: expected exactly {expected_counts[role]} cases")
        if receipt.get("split") != expected_split:
            raise ValueError(f"{role}: dataset receipt split differs")
        for capture_key, receipt_key in (
            ("dataset_sha256", "dataset_sha256"),
            ("dataset_file_sha256", "file_sha256"),
        ):
            if not _digest(manifest.get(capture_key)) or manifest[capture_key] != receipt.get(
                receipt_key
            ):
                raise ValueError(f"{role}: captured dataset fingerprint differs from receipt")
        for case_key, receipt_key in (
            ("id", "case_ids"),
            ("group_id", "group_ids"),
            ("request_sha256", "request_sha256s"),
        ):
            values = [case.get(case_key) for case in cases]
            if values != receipt.get(receipt_key):
                raise ValueError(f"{role}: captured {case_key} identities differ from receipt")
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{role}: invalid {case_key} identity")
            if case_key != "group_id" and len(set(values)) != len(values):
                raise ValueError(f"{role}: duplicate {case_key} identities")
        question_count = 0
        for case in cases:
            if case.get("split") != expected_split or not _digest(case.get("request_sha256")):
                raise ValueError(f"{role}: invalid case split/request fingerprint")
            if not _digest(case.get("tensor_file_sha256")):
                raise ValueError(f"{role}: captured tensor SHA256 is required")
            questions, slots, counts = (
                case.get("questions", []),
                case.get("slots", []),
                case.get("nopts", []),
            )
            if (
                not 1 <= len(questions) <= 64
                or len(slots) != len(questions)
                or len(counts) != len(questions)
            ):
                raise ValueError(f"{role}: invalid question/slot/option counts")
            if any(type(slot) is not int or slot < 0 for slot in slots) or slots != sorted(
                set(slots)
            ):
                raise ValueError(f"{role}: slots must be nonnegative and strictly increasing")
            ids = [question.get("id") for question in questions]
            if (
                any(not isinstance(value, str) or not value for value in ids)
                or len(set(ids)) != len(ids)
                or set(case.get("gold", {})) != set(ids)
            ):
                raise ValueError(f"{role}: question ids and gold labels must match exactly")
            for question, count in zip(questions, counts):
                labels = question.get("labels", [])
                if (
                    type(count) is not int
                    or not 2 <= count <= 52
                    or len(labels) != count
                    or len(set(labels)) != count
                    or any(not isinstance(label, str) or not label for label in labels)
                    or case["gold"][question["id"]] not in labels
                    or question.get("type") not in ("choice", "noul", "score")
                ):
                    raise ValueError(f"{role}: invalid candidate labels/gold/type")
            question_count += len(questions)
        if question_count != receipt.get("questions"):
            raise ValueError(f"{role}: decision count differs from data receipt")
    # Include training and historical held-out data from the receipt. Group
    # repetition within a split is permitted; crossings between splits are not.
    for key in ("case_ids", "group_ids", "request_sha256s"):
        seen = {}
        for name, receipt in datasets.items():
            values = receipt.get(key)
            if not isinstance(values, list) or len(values) != receipt.get("cases"):
                raise ValueError(f"{name}: incomplete leakage-audit identities")
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name}: malformed leakage-audit identities")
            for value in values:
                previous = seen.setdefault(value, receipt["split"])
                if previous != receipt["split"]:
                    raise ValueError(f"split leakage detected in {key}")
    return {
        "source_model": identity[0],
        "source_revision": identity[1],
        "source_checkpoint_sha256": identity[2],
        "source_s1_config_sha256": identity[4],
        "prompt_version": 1,
        "train_cases_audited": train["cases"],
    }


def validate_conversion(model_path, conversion, identity):
    """Verify actual exported bytes against the pinned conversion receipt."""
    verifier, preparation = shared_scripts()
    if conversion.get("source_checkpoint_sha256") != identity["source_checkpoint_sha256"]:
        raise ValueError("conversion source checkpoint differs from captured inputs")
    actual = preparation["checkpoint_identity"](model_path)
    if conversion.get("files") != actual["files"]:
        raise ValueError("exported config/index/shard hashes differ from conversion receipt")
    if conversion.get("export_checkpoint_sha256") not in (None, actual["sha256"]):
        raise ValueError("conversion export checkpoint digest is inconsistent")
    config = json.loads((Path(model_path) / "config.json").read_text())
    softcap = verifier["validate_model_config"](config)
    if not (config.get("quantization") or config.get("quantization_config")):
        raise ValueError("release calibration requires an explicitly quantized MLX export")
    return actual, config, softcap


def validate_dataset_files(manifests, dataset_manifest, directory):
    """Bind golds and candidate ordering to the actual frozen, hash-pinned datasets."""
    from s1.evaluation.datasets import fingerprint, load_cases, request_fingerprint

    _, preparation = shared_scripts()
    roles = {filename: role for role, filename in DATASET_NAMES.items()}
    for filename, receipt in dataset_manifest["datasets"].items():
        if Path(filename).name != filename or not filename.endswith(".jsonl"):
            raise ValueError("release dataset paths must be local JSONL filenames")
        path = Path(directory) / filename
        if preparation["file_sha256"](path) != receipt.get("file_sha256"):
            raise ValueError(f"{filename}: dataset file SHA256 mismatch")
        cases = load_cases(path)
        if fingerprint(cases) != receipt.get("dataset_sha256"):
            raise ValueError(f"{filename}: canonical dataset fingerprint mismatch")
        if len(cases) != receipt.get("cases") or sum(
            len(case.request.questions) for case in cases
        ) != receipt.get("questions"):
            raise ValueError(f"{filename}: dataset case/question counts differ")
        for field, actual in (
            ("case_ids", [case.id for case in cases]),
            ("group_ids", [case.group_id or case.id for case in cases]),
            ("request_sha256s", [request_fingerprint(case.request) for case in cases]),
        ):
            if actual != receipt.get(field):
                raise ValueError(f"{filename}: dataset {field} identities differ")
        if any(case.split != receipt.get("split") for case in cases):
            raise ValueError(f"{filename}: dataset split differs")
        role = roles.get(filename)
        if role is not None:
            captured = manifests[role]["cases"]
            if len(captured) != len(cases):
                raise ValueError(f"{role}: incomplete captured cases")
            for record, case in zip(captured, cases):
                expected_questions = [
                    {"id": question.id, "type": question.type, "labels": question.labels()}
                    for question in case.request.questions
                ]
                if record["id"] != case.id or record["gold"] != case.gold:
                    raise ValueError(f"{role}: captured case ids/gold differ from frozen dataset")
                if record["questions"] != expected_questions:
                    raise ValueError(
                        f"{role}: captured question/label ordering differs from dataset"
                    )
        del cases


def probabilities_from_logits(rows, temperature):
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be positive and finite")
    probabilities = []
    for row in rows:
        values = np.asarray(row, dtype=np.float64)
        if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("raw legal logits must be finite vectors")
        values = values / temperature
        exp = np.exp(values - values.max())
        probabilities.append(exp / exp.sum())
    return probabilities


def fit_calibration(records, expected_cases=256):
    """Use exactly the complete calibration records; never accept a test record."""
    from s1.training import fit_temperature

    if len(records) != expected_cases or any(
        record.get("status") != "ok" or record.get("split") != "calibration" for record in records
    ):
        raise ValueError("temperature fitting requires complete calibration-only case records")
    logits, golds = [], []
    for record in records:
        for answer in record["answers"].values():
            logits.append(answer["raw_logits"])
            golds.append(answer["labels"].index(answer["gold"]))
    return {
        "method": "s1.training.fit_temperature; 241 log-spaced temperatures [0.05,20] plus1",
        "input": "raw legal selected logits after native softcap, before any temperature",
        "case_count": len(records),
        **fit_temperature(logits, golds),
    }


def apply_temperature(records, temperature):
    for record in records:
        if record["status"] != "ok":
            continue
        for answer in record["answers"].values():
            probability = probabilities_from_logits([answer["raw_logits"]], temperature)[0]
            answer["probabilities"] = dict(zip(answer["labels"], probability.tolist()))
            answer["prediction"] = answer["labels"][int(probability.argmax())]


def summarize_split(records, manifest):
    from s1.metrics import summarize

    rows, golds, counts, latencies = [], [], [], []
    successes = [record for record in records if record["status"] == "ok"]
    for record in successes:
        latencies.extend(record["prepared_forward_samples_ms"])
        for answer in record["answers"].values():
            if "probabilities" not in answer:
                continue
            rows.append([answer["probabilities"][label] for label in answer["labels"]])
            golds.append(answer["labels"].index(answer["gold"]))
            counts.append(len(answer["labels"]))
    padded = np.zeros((len(rows), max(counts, default=0)))
    for index, values in enumerate(rows):
        padded[index, : len(values)] = values
    metrics = summarize(padded, golds, counts) if rows else {"n": 0}
    if "acc" in metrics:
        metrics["accuracy"] = metrics.pop("acc")
    expected = sum(len(case["questions"]) for case in manifest["cases"])
    return {
        "expected_cases": len(manifest["cases"]),
        "successful_cases": len(successes),
        "case_coverage": len(successes) / len(manifest["cases"]),
        "expected_decisions": expected,
        "scored_decisions": len(rows),
        "decision_coverage": len(rows) / expected,
        "metrics": metrics,
        "prepared_forward_latency_ms": {
            "samples": len(latencies),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        }
        if latencies
        else None,
    }


def collect_split(model, mx, role, directory, manifest, records, *, warmup, repeat, softcap):
    verifier, _ = shared_scripts()
    for index, case in enumerate(manifest["cases"]):
        record = {
            "id": case["id"],
            "split": case["split"],
            "status": "error",
            "group_id": case["group_id"],
            "request_sha256": case["request_sha256"],
            "tensor_file_sha256": case["tensor_file_sha256"],
            "answers": {},
        }
        records.append(record)
        try:
            arrays, digest = verifier["load_case_arrays"](directory, case)
            if digest != case["tensor_file_sha256"]:
                raise ValueError("prepared tensors changed after preflight")
            record["input_tokens"] = int(arrays["input_ids"].shape[-1])
            if index == 0:
                for _ in range(warmup):
                    verifier["selected_logits"](model, arrays, case, manifest["letters"], softcap)
            samples, first, drift = [], None, 0.0
            for _ in range(repeat):
                mx.synchronize()
                started = time.perf_counter()
                logits = verifier["selected_logits"](
                    model, arrays, case, manifest["letters"], softcap
                )
                mx.synchronize()
                samples.append((time.perf_counter() - started) * 1000)
                if first is None:
                    first = logits
                else:
                    drift = max(
                        drift, max(float(np.abs(a - b).max()) for a, b in zip(first, logits))
                    )
            del arrays
            if len(first) != len(case["questions"]):
                raise ValueError("selected logits question count differs")
            for question, count, row in zip(case["questions"], case["nopts"], first):
                if row.shape != (count,) or not np.isfinite(row).all():
                    raise ValueError("invalid raw legal candidate logits")
                record["answers"][question["id"]] = {
                    "type": question["type"],
                    "labels": question["labels"],
                    "gold": case["gold"][question["id"]],
                    "raw_logits": row.tolist(),
                }
            record.update(
                status="ok", prepared_forward_samples_ms=samples, max_repeat_logit_drift=drift
            )
        except Exception as exc:
            record["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        if index == 0 or (index + 1) % 16 == 0 or index + 1 == len(manifest["cases"]):
            print(f"{role}: {index + 1}/{len(manifest['cases'])}", file=sys.stderr, flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    for role in EXPECTED_CASES:
        parser.add_argument(f"--{role}-inputs", type=Path, required=True)
        parser.add_argument(f"--{role}-manifest-sha256", required=True)
    for kind in ("dataset", "conversion"):
        parser.add_argument(f"--{kind}-manifest", type=Path, required=True)
        parser.add_argument(f"--{kind}-manifest-sha256", required=True)
    parser.add_argument("--warmup", type=int, default=1, help="Warmups of first case in each split")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    # Only NumPy-based s1 modules are imported; Torch is unnecessary in the MLX environment.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    args = parse_args(argv)
    verifier, _ = shared_scripts()
    report = {
        "schema_version": 1,
        "status": "error",
        "release_validation_complete": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model.resolve()),
        "library_versions": verifier["_library_versions"](),
        "calibration": None,
        "cases": {role: [] for role in EXPECTED_CASES},
        "measurement": {
            "scope": "synchronized prepared tensors to raw legal logits on CPU",
            "includes": "MLX array transfer, backbone, selected head rows, native softcap, CPU copy",
            "excludes": "preprocessing, file I/O, model loading, NumPy softmax and calibration",
            "warmup": "first case of each split only; other shape compilation may affect latency",
            "warmup_per_split": args.warmup,
            "repeat_per_case": args.repeat,
            "not_end_to_end_request_latency": True,
        },
        "metric_definitions": {"nll_probability_floor": 1e-12, "ece_equal_width_bins": 15},
    }
    stage, manifests, temperature = "preflight", {}, None
    try:
        if args.warmup < 0 or args.repeat < 1:
            raise ValueError("warmup must be nonnegative and repeat must be positive")
        directories = {role: getattr(args, f"{role}_inputs") for role in EXPECTED_CASES}
        pins = {role: getattr(args, f"{role}_manifest_sha256") for role in EXPECTED_CASES}
        manifests = {
            role: read_pinned_json(directory / "manifest.json", pins[role])
            for role, directory in directories.items()
        }
        dataset = read_pinned_json(args.dataset_manifest, args.dataset_manifest_sha256)
        identity = validate_manifests(manifests, dataset)
        validate_dataset_files(manifests, dataset, args.dataset_manifest.parent)
        conversion = read_pinned_json(args.conversion_manifest, args.conversion_manifest_sha256)
        exported, config, softcap = validate_conversion(args.model, conversion, identity)
        report.update(identity)
        report.update(
            source_checkpoint_files=manifests["calibration"]["source_checkpoint_files"],
            input_manifest_sha256=pins,
            dataset_manifest_sha256=args.dataset_manifest_sha256,
            conversion_manifest_sha256=args.conversion_manifest_sha256,
            export_checkpoint_sha256=exported["sha256"],
            export_files=exported["files"],
            quantization=config.get("quantization", config.get("quantization_config")),
        )
        base_path = args.model / "base_s1_config.json"
        source_sidecar = base_path if base_path.exists() else args.model / "s1_config.json"
        source_sidecar_bytes = source_sidecar.read_bytes()
        if hashlib.sha256(source_sidecar_bytes).hexdigest() != identity["source_s1_config_sha256"]:
            raise ValueError("source s1_config.json bytes differ from captured merged checkpoint")
        base_sidecar = json.loads(source_sidecar_bytes)
        for key in ("source_model", "source_revision", "prompt_version"):
            if base_sidecar.get(key) != identity[key]:
                raise ValueError(f"model sidecar {key} differs from captured checkpoint identity")
        for role, manifest in manifests.items():
            for case in manifest["cases"]:
                arrays, digest = verifier["load_case_arrays"](directories[role], case)
                del arrays
                if digest != case["tensor_file_sha256"]:
                    raise ValueError(f"{role}: prepared tensor SHA256 mismatch")
        stage = "load_mlx_model"
        import mlx.core as mx
        from mlx_vlm import load

        started = time.perf_counter()
        model, processor = load(str(args.model), strict=True)
        mx.synchronize()
        report["load_time_ms"] = (time.perf_counter() - started) * 1000
        report.update(
            verifier["validate_loaded_model"](model, processor, manifests["calibration"]["letters"])
        )
        for role in EXPECTED_CASES:
            stage = f"collect_{role}"
            collect_split(
                model,
                mx,
                role,
                directories[role],
                manifests[role],
                report["cases"][role],
                warmup=args.warmup,
                repeat=args.repeat,
                softcap=softcap,
            )
            if role == "calibration":
                stage = "fit_calibration"
                report["calibration"] = fit_calibration(report["cases"][role])
                temperature = report["calibration"]["temperature"]
            apply_temperature(report["cases"][role], temperature)
        report["summary"] = {
            role: summarize_split(report["cases"][role], manifests[role]) for role in EXPECTED_CASES
        }
        if any(
            value["decision_coverage"] != 1 or value["case_coverage"] != 1
            for value in report["summary"].values()
        ):
            raise ValueError("complete calibration and held-out coverage is required")
        stage = "save_calibrated_sidecar"
        calibrated = {
            **base_sidecar,
            "temperature": temperature,
            "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
            "export_checkpoint_sha256": exported["sha256"],
            "calibration_dataset_sha256": manifests["calibration"]["dataset_sha256"],
            "mlx_calibration": {
                **report["calibration"],
                "split": "calibration",
                "runtime": "mlx-vlm",
                "manifest_sha256": pins["calibration"],
                "dataset_manifest_sha256": args.dataset_manifest_sha256,
                "source_temperature": base_sidecar.get("temperature"),
            },
        }
        if not base_path.exists():
            base_path.write_bytes(source_sidecar_bytes)
        serialized = (json.dumps(calibrated, indent=2, allow_nan=False) + "\n").encode()
        temporary = args.model / ".s1_config.calibrated.tmp"
        temporary.write_bytes(serialized)
        temporary.replace(args.model / "s1_config.json")
        report["calibrated_s1_config_sha256"] = hashlib.sha256(serialized).hexdigest()
        report["status"] = "ok"
        report["release_validation_complete"] = True
    except Exception as exc:
        report["failure"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
    if manifests and all(role in manifests for role in EXPECTED_CASES):
        try:
            if temperature is not None:
                for records in report["cases"].values():
                    apply_temperature(records, temperature)
            report["summary"] = {
                role: summarize_split(report["cases"][role], manifests[role])
                for role in EXPECTED_CASES
                if manifests[role].get("cases")
            }
        except Exception as exc:
            report["summary_failure"] = {"type": type(exc).__name__, "message": str(exc)}
            report["status"] = "error"
            report["release_validation_complete"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['status']}: {args.output}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
