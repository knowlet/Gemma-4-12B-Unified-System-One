"""Collect immutable campaign receipts and paired comparisons across retry runs.

Selection is chronological, never based on accuracy: the latest completed attempt
wins; if none completed, the latest compatible attempt remains explicitly failed
or incomplete. Every attempted run is retained in the audit index. Incompatible
pre-execution failures may be indexed but never selected or assigned scores.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from s1.evaluation.comparison import DATASET_PINS
from s1.evaluation.statistics import paired_comparison

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = (
    "gemma-base-bf16",
    *(f"gemma-ce128-s{seed}-{mode}" for seed in range(3) for mode in ("bf16", "int8", "nf4")),
    "laya-general",
    "decider-local",
    "kev-local",
    "agentjev-local",
)


RECOVERY_MODELS = tuple(f"gemma-ce128-s{seed}-nf4-recovered" for seed in range(3))
OPTIONAL_MODELS = ("clef-local",)
RECOVERY_TRAIN_SHA256 = "4bc27d96d998ef02137e58e2c8b5d4f928dafc9db03476d436a13acddf35decb"
RECOVERY_CALIBRATION_SHA256 = "2d919497859b47f55aaaf6e67a63e8f2141877f77cacc0625dd59d3e67add9b8"


def verify_recovery_pair(left, right):
    metadata = right.get("recovery") or {}
    if any(
        left.get(key) is None or left.get(key) != right.get(key)
        for key in ("checkpoint", "revision")
    ):
        raise ValueError("recovery comparison requires the same frozen base checkpoint")
    if (
        not left.get("adapter_sha256")
        or metadata.get("parent_adapter_sha256") != left["adapter_sha256"]
    ):
        raise ValueError("recovery parent adapter identity does not match the original NF4 row")
    if not right.get("adapter_sha256"):
        raise ValueError("recovery child adapter identity is missing")
    if not (
        metadata.get("method") == "fixed_nf4_qlora_continuation"
        and metadata.get("additional_optimizer_updates") == 100
        and metadata.get("train_dataset_sha256") == RECOVERY_TRAIN_SHA256
        and metadata.get("calibration_dataset_sha256") == RECOVERY_CALIBRATION_SHA256
        and metadata.get("no_test_fitting") is True
        and metadata.get("exploratory") is True
        and metadata.get("recipe_fixed_before_execution") is True
    ):
        raise ValueError("recovery lineage or fixed training-data recipe is unverified")


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path, output):
    return os.path.relpath(Path(path).resolve(), Path(output).resolve().parent)


def timestamp(row, campaign):
    if isinstance(row.get("started_at_unix"), (int, float)):
        return row["started_at_unix"], "receipt.started_at_unix"
    value = campaign.get("generated_at")
    if value:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("campaign chronology requires an explicit timezone")
        return parsed.timestamp(), "campaign.generated_at"
    raise ValueError("attempt requires started_at_unix or timezone-qualified generated_at")


def observed(row):
    return any(
        (row.get(name) or {}).get("recorded", 0) > 0
        or (row.get(name) or {}).get("accuracy") is not None
        for name in ("boolq", "media")
    ) or bool(row.get("load"))


def compatible_dataset(dataset):
    return (
        dataset.get("sha256") == DATASET_PINS["text-test.jsonl"][1]
        and dataset.get("cases") == DATASET_PINS["text-test.jsonl"][0]
        and dataset.get("media_sha256") == DATASET_PINS["media-test.jsonl"][1]
    )


def artifact_path(source, model, path):
    root = source.parent / model
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("receipt artifact escapes its model directory")
    return result


def verify_recovery_training_evidence(attempt, test_records, output):
    """Check observed training evidence, not merely a planned recovery recipe."""
    row = attempt["row"]
    artifact = (row.get("artifacts") or {}).get("training")
    if not artifact:
        raise ValueError("recovery training receipt is missing")
    path = artifact_path(attempt["source"], row["model_id"], artifact)
    if not path.exists():
        raise ValueError("recovery training receipt is unavailable")
    expected = (row.get("artifact_sha256") or {}).get(artifact)
    if not expected or expected != digest(path):
        raise ValueError("recovery training receipt hash differs from its recorded artifact")
    proof = read_json(path)
    if (
        proof.get("status") != "completed"
        or proof.get("actual_optimizer_updates") != 100
        or proof.get("adapter_sha256") != row.get("adapter_sha256")
        or proof.get("recovery") != row.get("recovery")
        or proof.get("base_model") != row.get("checkpoint")
        or proof.get("base_revision") != row.get("revision")
    ):
        raise ValueError(
            "recovery training receipt does not confirm the evaluated adapter and recipe"
        )
    training_ids = proof.get("observed_training_ids") or []
    training_hashes = proof.get("observed_training_request_sha256") or []
    if len(training_ids) != 100 or len(set(training_ids)) != 100 or len(training_hashes) != 100:
        raise ValueError("recovery training evidence does not contain 100 distinct observed cases")
    calibration = proof.get("calibration_reference") or []
    calibration_hashes = {case.get("request_sha256") for case in calibration}
    test_hashes = {record["request_sha256"] for record in test_records.values()}
    if (
        len(calibration) != 16
        or len(calibration_hashes) != 16
        or None in calibration_hashes
        or set(training_hashes) & calibration_hashes
        or (set(training_hashes) | calibration_hashes) & test_hashes
    ):
        raise ValueError(
            "recovery training/calibration and held-out requests overlap or are incomplete"
        )
    frozen_before = proof.get("frozen_parameter_sample_sha256_before")
    if not frozen_before or frozen_before != proof.get("frozen_parameter_sample_sha256_after"):
        raise ValueError("recovery frozen-weight sample audit failed")
    if (
        not proof.get("lora_sha256_before")
        or not proof.get("lora_sha256_after")
        or proof.get("lora_sha256_before") == proof.get("lora_sha256_after")
    ):
        raise ValueError("recovery LoRA update evidence is missing")
    return {
        "training_receipt": relative(path, output),
        "training_receipt_sha256": expected,
        "actual_optimizer_updates": proof["actual_optimizer_updates"],
        "observed_distinct_training_cases": len(set(training_ids)),
        "observed_calibration_cases": len(calibration),
        "held_out_request_overlap": False,
        "frozen_parameter_sample_unchanged": True,
    }


def load_prediction_evidence(attempt, output):
    """Join recorded request hashes from the actual runner plan, never current inputs."""
    row = attempt["row"]
    path = artifact_path(
        attempt["source"],
        row["model_id"],
        (row.get("artifacts") or {}).get("predictions", "boolq/predictions.jsonl"),
    )
    manifest_path = path.parent / "run_manifest.json"
    if not path.exists() or not manifest_path.exists():
        raise ValueError("saved BoolQ predictions or run manifest unavailable")
    artifact_root = attempt["source"].parent / row["model_id"]
    recorded_hashes = row.get("artifact_sha256") or {}
    required_artifacts = {
        str(artifact.relative_to(artifact_root.resolve())) for artifact in (path, manifest_path)
    }
    if not required_artifacts.issubset(recorded_hashes):
        raise ValueError("receipt is missing BoolQ prediction or manifest artifact hashes")
    for name, expected_hash in recorded_hashes.items():
        artifact = artifact_path(attempt["source"], row["model_id"], name)
        if not artifact.is_file() or digest(artifact) != expected_hash:
            raise ValueError(f"receipt artifact hash mismatch: {row['model_id']}/{name}")
    manifest = read_json(manifest_path)
    plan = manifest["plan"]
    if plan["dataset_sha256"] != DATASET_PINS["text-test.jsonl"][1]:
        raise ValueError("saved prediction manifest has a different BoolQ dataset")
    if manifest.get("status") not in ("completed", "completed_with_errors"):
        raise ValueError("saved BoolQ manifest is not finished")
    cells = [cell for cell in plan["models"] if cell["model_id"] == row["model_id"]]
    if len(cells) != 1:
        raise ValueError("saved prediction manifest model identity is ambiguous")
    model = cells[0].get("identity", {}).get("model", {})
    if model.get("id") != row["model_id"]:
        raise ValueError("receipt model ID differs from the saved model identity")
    for receipt_key, plan_key in (
        ("checkpoint", "model_id"),
        ("revision", "revision"),
        ("precision", "precision"),
        ("quantization", "quantization"),
        ("adapter_sha256", "decision_adapter_sha256"),
    ):
        if (
            receipt_key not in row
            or plan_key not in model
            or row[receipt_key] != model[plan_key]
            or (receipt_key != "adapter_sha256" and not row[receipt_key])
        ):
            raise ValueError(f"receipt {receipt_key} differs from the saved model identity")
    execution = manifest.get("execution", {}).get(row["model_id"], {})
    if execution.get("status") not in ("completed", "completed_with_errors"):
        raise ValueError("saved BoolQ model execution is not finished")
    telemetry = execution.get("telemetry") or {}
    for key in ("revision", "precision", "quantization"):
        if telemetry.get(key) != model[key]:
            raise ValueError(f"runtime {key} differs from the saved model identity")
    planned = {case["case_id"]: case for case in cells[0]["cases"]}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != DATASET_PINS["text-test.jsonl"][0]:
        raise ValueError("paired comparison requires all 128 recorded BoolQ decisions")
    indices = {}
    for source in records:
        record = dict(source)
        key = (record["case_id"], record["question_id"], record["repetition"])
        if key in indices or record["model_id"] != row["model_id"] or record["repetition"] != 0:
            raise ValueError("duplicate, repeated, or mismatched prediction identity")
        case = planned.get(record["case_id"])
        if case is None or not case.get("request_sha256"):
            raise ValueError("prediction has no recorded request identity")
        if record.get("request_sha256") not in (None, case["request_sha256"]):
            raise ValueError("prediction request hash differs from its own manifest")
        if record["group_id"] != case["group_id"]:
            raise ValueError("prediction source group differs from its manifest")
        if (
            record["status"] not in ("ok", "error", "timeout")
            or record["eligibility"] != "eligible"
        ):
            raise ValueError(
                "paired comparison requires attempted, eligible full-population records"
            )
        record["request_sha256"] = case["request_sha256"]
        indices[key] = record
    valid = [r for r in indices.values() if r["status"] == "ok"]
    correct = sum(r["actual_label"] == r["gold"] for r in valid)
    quality = row.get("boolq") or {}
    if (quality.get("recorded"), quality.get("valid"), quality.get("correct")) != (
        len(indices),
        len(valid),
        correct,
    ):
        raise ValueError("saved predictions disagree with receipt decision counts")
    if quality.get("accuracy") != correct / 128:
        raise ValueError("saved predictions disagree with receipt operational accuracy")
    return indices, {
        "predictions": relative(path, output),
        "predictions_sha256": digest(path),
        "manifest": relative(manifest_path, output),
        "manifest_sha256": digest(manifest_path),
    }


def compare_selected(selected, output, *, seed, resamples):
    pairs = [
        (f"gemma-ce128-s{s}-bf16", f"gemma-ce128-s{s}-{mode}", "same_seed_quantization")
        for s in range(3)
        for mode in ("int8", "nf4")
    ] + [
        ("gemma-base-bf16", name, "same_input_generalist")
        for name in ("laya-general", "decider-local", "kev-local", "agentjev-local")
    ]
    pairs.extend(
        ("gemma-base-bf16", name, "same_input_generalist")
        for name in OPTIONAL_MODELS
        if name in selected
    )
    if any(name in selected for name in RECOVERY_MODELS):
        pairs.extend(
            (
                f"gemma-ce128-s{s}-nf4",
                f"gemma-ce128-s{s}-nf4-recovered",
                "exploratory_extra_training_recovery",
            )
            for s in range(3)
        )
    cache, comparisons = {}, []
    for left_id, right_id, kind in pairs:
        entry = {"left": left_id, "right": right_id, "kind": kind}
        if kind == "exploratory_extra_training_recovery":
            entry["interpretation"] = (
                "includes 100 additional training updates; not a quantization-only comparison"
            )
        if left_id not in selected or right_id not in selected:
            comparisons.append(
                {**entry, "status": "not_available", "reason": "model attempt missing"}
            )
            continue
        try:
            if kind == "same_seed_quantization":
                left_row, right_row = selected[left_id]["row"], selected[right_id]["row"]
                if any(
                    left_row.get(key) is None or left_row.get(key) != right_row.get(key)
                    for key in ("checkpoint", "revision", "adapter_sha256")
                ):
                    raise ValueError(
                        "same-seed comparison requires identical checkpoint and adapter identities"
                    )
                mode = right_id.rsplit("-", 1)[-1]
                if not (
                    mode in ("int8", "nf4")
                    and left_row.get("precision") == right_row.get("precision") == "bfloat16"
                    and left_row.get("quantization") == "none"
                    and right_row.get("quantization") == mode
                ):
                    raise ValueError(
                        "same-seed comparison requires an unquantized BF16 reference and "
                        "the candidate's declared INT8/NF4 mode with BF16 computation"
                    )
            if kind == "exploratory_extra_training_recovery":
                verify_recovery_pair(selected[left_id]["row"], selected[right_id]["row"])
            for name in (left_id, right_id):
                if name not in cache:
                    cache[name] = load_prediction_evidence(selected[name], output)
            left, left_evidence = cache[left_id]
            right, right_evidence = cache[right_id]
            if left.keys() != right.keys() or any(
                left[k]["request_sha256"] != right[k]["request_sha256"] for k in left
            ):
                raise ValueError("paired recorded request identities differ")
            if kind == "exploratory_extra_training_recovery":
                entry["training_evidence"] = verify_recovery_training_evidence(
                    selected[right_id], right, output
                )
            result = paired_comparison(
                list(left.values()), list(right.values()), seed=seed, resamples=resamples
            )
            comparisons.append(
                {
                    **entry,
                    "status": "computed",
                    **result,
                    "left_evidence": left_evidence,
                    "right_evidence": right_evidence,
                    "left_gpu": selected[left_id]["row"].get("gpu"),
                    "right_gpu": selected[right_id]["row"].get("gpu"),
                }
            )
        except (ValueError, KeyError, TypeError, OSError) as exc:
            comparisons.append({**entry, "status": "not_available", "reason": str(exc)})
    return comparisons


def collect(paths, output, *, campaign_id="matched-comparison", seed=20260930, resamples=2000):
    if resamples < 100:
        raise ValueError("at least 100 bootstrap resamples required")
    paths = [Path(path).resolve() for path in paths]
    if len(set(paths)) != len(paths):
        raise ValueError("campaign input paths must be unique")
    attempts, sources = [], []
    selected, dataset = {}, None
    for source in paths:
        snapshot = source.read_bytes()
        campaign = json.loads(snapshot)
        compatible = compatible_dataset(campaign["dataset"])
        if compatible and dataset is None:
            dataset = copy.deepcopy(campaign["dataset"])
        models = [row["model_id"] for row in campaign["runs"]]
        if len(set(models)) != len(models):
            raise ValueError("one campaign may not contain duplicate model attempts")
        sources.append(
            {
                "campaign_id": campaign["campaign_id"],
                "path": relative(source, output),
                "sha256": hashlib.sha256(snapshot).hexdigest(),
                "status": campaign.get("status"),
                "dataset_compatible": compatible,
            }
        )
        for row in campaign["runs"]:
            if not compatible and observed(row):
                raise ValueError(
                    "cannot combine evaluated attempts with different dataset identities"
                )
            for key, pin in (("boolq", "text-test.jsonl"), ("media", "media-test.jsonl")):
                actual = (row.get(key) or {}).get("dataset_sha256")
                if actual and actual != DATASET_PINS[pin][1]:
                    raise ValueError("receipt quality dataset differs from the shared dataset")
            when, chronology = timestamp(row, campaign)
            attempt = {
                "id": f"{campaign['campaign_id']}:{row['model_id']}",
                "source": source,
                "campaign_id": campaign["campaign_id"],
                "row": copy.deepcopy(row),
                "dataset_compatible": compatible,
                "timestamp": when,
                "chronology": chronology,
            }
            attempt["row"]["configured_gpu_tier"] = campaign.get("policy", {}).get("gpu")
            attempts.append(attempt)
    if len({a["id"] for a in attempts}) != len(attempts):
        raise ValueError("campaign IDs must identify unique attempt sets")
    for model in sorted({a["row"]["model_id"] for a in attempts}):
        compatible = [
            a for a in attempts if a["row"]["model_id"] == model and a["dataset_compatible"]
        ]
        completed = [a for a in compatible if a["row"]["status"] == "completed"]
        candidates = completed or compatible
        if candidates:
            # ID is a stable tie-breaker only, never score or file modification time.
            selected[model] = max(candidates, key=lambda a: (a["timestamp"], a["id"]))
    expected = (
        EXPECTED_MODELS
        + (RECOVERY_MODELS if any(name in selected for name in RECOVERY_MODELS) else ())
        + tuple(
            name for name in OPTIONAL_MODELS if any(a["row"]["model_id"] == name for a in attempts)
        )
    )
    runs, audit = [], []
    for model in (*expected, *sorted(set(selected) - set(expected))):
        if model not in selected:
            continue
        attempt = selected[model]
        row = attempt["row"]
        if row["status"] == "completed" or observed(row):
            # A failed pairing must not leave a corrupted standalone score publishable.
            records, _ = load_prediction_evidence(attempt, output)
            if model in RECOVERY_MODELS:
                verify_recovery_training_evidence(attempt, records, output)
        row["selected_attempt"] = attempt["id"]
        row["source_campaign"] = attempt["campaign_id"]
        row["source_campaign_path"] = relative(attempt["source"], output)
        runs.append(row)
    for attempt in sorted(attempts, key=lambda a: (a["timestamp"], a["id"])):
        row = attempt["row"]
        receipt = artifact_path(attempt["source"], row["model_id"], "receipt.json")
        chosen = selected.get(row["model_id"]) is attempt
        audit.append(
            {
                "attempt_id": attempt["id"],
                "model_id": row["model_id"],
                "campaign_id": attempt["campaign_id"],
                "status": row["status"],
                "timestamp": attempt["timestamp"],
                "chronology": attempt["chronology"],
                "selected": chosen,
                "dataset_compatible": attempt["dataset_compatible"],
                "error": row.get("error"),
                "selection_reason": "latest completed attempt"
                if chosen and row["status"] == "completed"
                else "latest compatible attempt; none completed"
                if chosen
                else "incompatible pre-execution attempt"
                if not attempt["dataset_compatible"]
                else "retained superseded or unsuccessful attempt",
                "source_campaign": relative(attempt["source"], output),
                "receipt": relative(receipt, output) if receipt.exists() else None,
                "receipt_sha256": digest(receipt) if receipt.exists() else None,
            }
        )
    missing = sorted(set(expected) - set(selected))
    unresolved = [r["model_id"] for r in runs if r["status"] != "completed"]
    return {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "generated_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
        "status": "incomplete" if missing or unresolved else "completed",
        "dataset": dataset
        or {
            "sha256": DATASET_PINS["text-test.jsonl"][1],
            "cases": 128,
            "media_sha256": DATASET_PINS["media-test.jsonl"][1],
        },
        "expected_models": list(expected),
        "missing_models": missing,
        "unresolved_models": unresolved,
        "runs": runs,
        "sources": sources,
        "attempts": audit,
        "selection_policy": "latest status=completed attempt by receipt start time (campaign timestamp fallback), otherwise latest compatible attempt; never by accuracy",
        "comparisons": compare_selected(selected, output, seed=seed, resamples=resamples),
        "statistical_policy": {
            "unit": "source group",
            "resamples": resamples,
            "seed": seed,
            "confidence": 0.95,
            "multiple_comparisons_adjusted": False,
            "population": "same 128 BoolQ examples; complete attempted records and matching request hashes",
            "interpretation": "descriptive paired screening intervals; not deployment acceptance",
            "latency": "per model and per workload only; p99 values are not averaged across seeds",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/validation/2026-10-01/comparison-summary.json"
    )
    parser.add_argument("--campaign-id", default="20261001-matched-comparison")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    result = collect(
        args.campaigns,
        args.output,
        campaign_id=args.campaign_id,
        resamples=args.resamples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "selected_models": len(result["runs"]),
                "attempts": len(result["attempts"]),
                "computed_comparisons": sum(
                    c["status"] == "computed" for c in result["comparisons"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
