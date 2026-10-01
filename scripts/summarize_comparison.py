"""Render a comparison from recorded campaign and historical evidence.

The report never substitutes nominal weight sizes for measured VRAM, or historical
base-model service latency for a fine-tuned or quantized model's latency. A blocked
campaign can be reported without manufacturing evaluation results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORICAL = ROOT / "docs/validation/2026-10-01/live-summary.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _number(value, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "not measured"


def _gb(value) -> str:
    return _number(value / 1_000_000_000) if isinstance(value, (int, float)) else "not measured"


def _memory(value) -> str:
    if not isinstance(value, (int, float)):
        return "not measured"
    return f"{value / 1_000_000_000:.2f} ({value / 2**30:.2f})"


def _pct(value) -> str:
    return f"{100 * value:.2f}%" if isinstance(value, (int, float)) else "not measured"


def _link(path: Path, output: Path, label: str) -> str:
    return f"[{label}]({os.path.relpath(path, output.parent)})"


def _observation(row: dict) -> dict:
    observed = row["observed"]
    reported = row.get("reported", {})
    count = observed["recorded_decisions"]
    return {
        "expected": count,
        "recorded": count,
        "valid": observed["valid_decisions"],
        "correct": observed["correct"],
        "accuracy": observed["operational_accuracy"],
        "mean_ms": 1000 * reported["wall_seconds"] / count
        if reported.get("wall_seconds") is not None and count
        else None,
        "dataset_sha256": row.get("dataset_sha256"),
        "prediction_sha256": row.get("prediction_sha256"),
        "task_observations": row.get("task_observations", {}),
    }


def historical_rows(historical: dict) -> list[dict]:
    """Use only numeric observations present in the saved October 1 summary."""
    quality = historical.get("quality", [])
    lookup = {(q["stage"], q["model"]): q for q in quality}
    stages = historical.get("latest_stages", {})
    rows = []
    singles = [
        ("Gemma 12B base", "training-full", "base:text", "base:media", "BF16", 24),
        (
            "Gemma 12B base (parity control)",
            "checkpoint-fp32",
            "public-text-quality:gemma-g4",
            "public-media-quality:gemma-g4",
            "FP32",
            48,
        ),
        ("Laya general", "laya", "laya-general", None, "recorded SDK runtime", None),
    ]
    for name, stage, text_id, media_id, precision, nominal in singles:
        text = lookup.get((stage, text_id))
        if text is None:
            continue
        media = lookup.get((stage, media_id))
        hardware_stage = "services" if stage == "laya" else stage
        rows.append(
            {
                "model_id": name,
                "precision": precision,
                "boolq": _observation(text),
                "media": _observation(media) if media else None,
                "media_status": "unsupported native media" if stage == "laya" else "measured",
                "nominal_weight_gb": nominal,
                "gpu": stages.get(hardware_stage, {}).get("gpu", "not recorded"),
                "stage": stage,
                "job_url": stages.get(hardware_stage, {}).get("job_url"),
                "seeds": 1,
            }
        )
    grouped = defaultdict(list)
    for q in quality:
        if q["stage"] not in ("training-full", "baselines-full") or "shots" not in q:
            continue
        if q["stage"] == "training-full" and not q["model"].endswith(":text"):
            continue
        grouped[(q["stage"], q["method"], q["shots"])].append(q)
    for (stage, method, shots), group in sorted(grouped.items()):
        group.sort(key=lambda q: q["seed"])
        observations = [_observation(q) for q in group]
        accuracies = [q["accuracy"] for q in observations]
        means = [q["mean_ms"] for q in observations if q["mean_ms"] is not None]
        media = lookup.get((stage, f"shots-{shots}-seed-0-{method}:media"))
        gemma = stage == "training-full"
        rows.append(
            {
                "model_id": f"{'Gemma 12B + ' if gemma else ''}{method}, {shots}/class",
                "precision": "BF16" if gemma else "CPU runtime",
                "boolq": {
                    "expected": observations[0]["expected"],
                    "accuracy": statistics.mean(accuracies),
                    "accuracy_min": min(accuracies),
                    "accuracy_max": max(accuracies),
                    "mean_ms_min": min(means) if means else None,
                    "mean_ms_max": max(means) if means else None,
                    "seeds": [q["seed"] for q in group],
                    "observations": observations,
                },
                "media": _observation(media) if media else None,
                "media_status": "seed 0 only" if media else "not measured",
                "nominal_weight_gb": 24 if gemma else None,
                "gpu": stages.get(stage, {}).get("gpu") or "CPU",
                "stage": stage,
                "job_url": stages.get(stage, {}).get("job_url"),
                "seeds": len(group),
            }
        )
    return rows


def _validate_observation(value: dict, expected: int, label: str) -> dict:
    """Recompute operational accuracy and reject ambiguous/contradictory counts."""
    result = dict(value)
    denominator = result.get("expected", expected)
    if denominator != expected:
        raise ValueError(f"{label}: expected {expected} decisions, got {denominator}")
    result["expected"] = denominator
    counts = [result.get(key) for key in ("correct", "valid", "recorded")]
    if any(v is not None and (not isinstance(v, int) or v < 0) for v in counts):
        raise ValueError(f"{label}: counts must be nonnegative integers")
    if all(v is not None for v in counts):
        correct, valid, recorded = counts
        if not correct <= valid <= recorded <= denominator:
            raise ValueError(f"{label}: inconsistent decision counts")
        accuracy = correct / denominator
        supplied = result.get("accuracy")
        if supplied is not None and not math.isclose(supplied, accuracy, abs_tol=1e-9):
            raise ValueError(f"{label}: accuracy must retain the full expected denominator")
        result["accuracy"] = accuracy
    elif result.get("accuracy") is not None:
        raise ValueError(f"{label}: accuracy requires correct, valid, and recorded counts")
    return result


def seed_aggregates(runs: list[dict]) -> list[dict]:
    """Require every predeclared seed; never average only the seeds that succeeded."""
    by_id = {run["model_id"]: run for run in runs}
    if not any(name.startswith("gemma-ce128-s") for name in by_id):
        return []
    groups = []
    modes = ["bf16", "int8", "nf4"]
    if any(name.endswith("-nf4-recovered") for name in by_id):
        modes.append("nf4-recovered")
    for mode in modes:
        members = [by_id.get(f"gemma-ce128-s{seed}-{mode}") for seed in range(3)]
        completed = [
            seed
            for seed, member in enumerate(members)
            if member
            and member["boolq"].get("valid") == 128
            and member["boolq"].get("accuracy") is not None
        ]
        hardware = {member.get("gpu") for member in members if member}
        tiers = {member.get("configured_gpu_tier") for member in members if member}
        comparable_hardware = (len(hardware) == 1 and None not in hardware) or (
            len(tiers) == 1 and None not in tiers
        )
        complete = len(completed) == 3 and comparable_hardware
        status = (
            "complete"
            if complete
            else (
                "hardware_mismatch" if len(completed) == 3 and len(hardware) > 1 else "incomplete"
            )
        )
        scores = [member["boolq"]["accuracy"] for member in members] if complete else []
        peaks = (
            [member["memory"].get("peak_allocated_bytes") for member in members] if complete else []
        )
        group = {
            "format": mode,
            "status": status,
            "observed_gpus": sorted(value for value in hardware if value),
            "configured_gpu_tiers": sorted(value for value in tiers if value),
            "expected_seeds": [0, 1, 2],
            "completed_seeds": completed,
            "missing_seeds": sorted(set(range(3)) - set(completed)),
            "accuracy_mean": statistics.mean(scores) if scores else None,
            "accuracy_min": min(scores) if scores else None,
            "accuracy_max": max(scores) if scores else None,
            "worst_peak_allocated_bytes": max(peaks)
            if peaks and all(p is not None for p in peaks)
            else None,
            "paired_mean_delta_vs_bf16_pp": None,
            "paired_mean_delta_vs_original_nf4_pp": None,
            "additional_training": mode == "nf4-recovered",
            "all_run_stages_completed": complete
            and all(m.get("status") == "completed" for m in members),
            "accuracy_memory_target_status": "not_applicable" if mode == "bf16" else "not_measured",
        }
        if complete and mode != "bf16":
            targets = [member["acceptance"].get("status") for member in members]
            if all(t in ("passed", "failed") for t in targets):
                group["accuracy_memory_target_status"] = (
                    "passed" if all(t == "passed" for t in targets) else "failed"
                )
            reference_mode = "nf4" if mode == "nf4-recovered" else "bf16"
            baseline = [by_id.get(f"gemma-ce128-s{seed}-{reference_mode}") for seed in range(3)]
            if all(
                base
                and base["boolq"].get("valid") == 128
                and base["boolq"].get("accuracy") is not None
                and (
                    base.get("gpu") == member.get("gpu")
                    or (
                        base.get("configured_gpu_tier") is not None
                        and base.get("configured_gpu_tier") == member.get("configured_gpu_tier")
                    )
                )
                and (
                    mode != "nf4-recovered"
                    or (
                        base.get("adapter_sha256") is not None
                        and member.get("recovery", {}).get("parent_adapter_sha256")
                        == base["adapter_sha256"]
                    )
                )
                for base, member in zip(baseline, members)
            ):
                delta_key = (
                    "paired_mean_delta_vs_original_nf4_pp"
                    if mode == "nf4-recovered"
                    else "paired_mean_delta_vs_bf16_pp"
                )
                group[delta_key] = statistics.mean(
                    100 * (member["boolq"]["accuracy"] - base["boolq"]["accuracy"])
                    for base, member in zip(baseline, members)
                )
        groups.append(group)
    return groups


def build_summary(historical: dict, campaign: dict | None) -> dict:
    datasets = historical.get("data", {}).get("datasets", {})
    text = datasets.get("text-test.jsonl", {})
    media = datasets.get("media-test.jsonl", {})
    expected = text.get("cases", 128)
    media_expected = media.get("cases", 52)
    historical_hash = text.get("dataset_sha256")
    campaign = campaign or {}
    dataset = campaign.get("dataset", {})
    # A preflight may have no materialized dataset. Measured rows require its hash.
    actual_hash = dataset.get("sha256") or dataset.get("dataset_sha256")
    if actual_hash and historical_hash and actual_hash != historical_hash:
        raise ValueError("campaign BoolQ dataset hash differs from the historical fixed 128 cases")
    runs = []
    for source in campaign.get("runs", []):
        run = {
            key: source.get(key)
            for key in (
                "model_id",
                "checkpoint",
                "revision",
                "precision",
                "quantization",
                "adapter_sha256",
                "gpu",
                "configured_gpu_tier",
                "status",
                "reason",
            )
        }
        run["reason"] = source.get("reason") or source.get("error")
        run["configured_gpu_tier"] = source.get("configured_gpu_tier") or campaign.get(
            "policy", {}
        ).get("gpu")
        run["boolq"] = _validate_observation(source.get("boolq") or {}, expected, run["model_id"])
        if run["boolq"].get("accuracy") is not None and not actual_hash:
            raise ValueError("measured campaign BoolQ results require the dataset hash")
        run["media"] = _validate_observation(
            source.get("media") or {}, media_expected, f"{run['model_id']} media"
        )
        media_hash = dataset.get("media_sha256")
        if run["media"].get("accuracy") is not None:
            if not media_hash:
                raise ValueError("measured campaign media results require the media dataset hash")
            if media_hash != media.get("dataset_sha256"):
                raise ValueError("campaign media dataset hash differs from the historical 52 cases")
        run["memory"] = {
            key: (source.get("memory") or {}).get(key)
            for key in (
                "model_footprint_bytes",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "device_used_bytes",
                "gpu_total_bytes",
                "cuda_peak_scope",
                "process_peak_rss_bytes",
            )
        }
        if run["memory"]["process_peak_rss_bytes"] is None:
            run["memory"]["process_peak_rss_bytes"] = (source.get("memory") or {}).get(
                "host_rss_bytes"
            )
        run["acceptance"] = source.get("acceptance") or {}
        run["recovery"] = source.get("recovery") or {}
        run["telemetry"] = source.get("telemetry") or {}
        run["load"] = source.get("load") or []
        run["artifacts"] = source.get("artifacts") or {}
        runs.append(run)
    return {
        "schema_version": 1,
        "campaign_id": campaign.get("campaign_id", "no campaign supplied"),
        "status": campaign.get("status", "not_run"),
        "generated_at": campaign.get("generated_at"),
        "dataset": {
            "sha256": historical_hash,
            "cases": expected,
            "media_sha256": dataset.get("media_sha256") or media.get("dataset_sha256"),
            "historical_media_sha256": media.get("dataset_sha256"),
            "media_exact_historical_match": dataset.get("media_sha256")
            == media.get("dataset_sha256")
            if dataset.get("media_sha256")
            else None,
            "media_comparability": dataset.get("media_comparability"),
            "media_cases": media_expected,
            "campaign_match": actual_hash == historical_hash if actual_hash else None,
        },
        "runs": runs,
        "seed_aggregates": seed_aggregates(runs),
        "comparisons": campaign.get("comparisons", []),
        "attempts": campaign.get("attempts", []),
        "selection_policy": campaign.get("selection_policy"),
        "missing_models": campaign.get("missing_models", []),
        "historical": historical_rows(historical),
        "historical_load": historical.get("load", {}),
        "historical_batch_checks": historical.get("batch_checks", {}),
        "historical_comparisons": [
            c
            for c in historical.get("comparisons", [])
            if c.get("left") == "training-full:base:text"
            and c.get("right") in ("training-full:shots-128-seed-0-ce:text", "laya:laya-general")
        ],
    }


def _accuracy(value: dict | None) -> str:
    if not value or value.get("accuracy") is None:
        return "not measured"
    result = _pct(value["accuracy"])
    if value.get("correct") is not None:
        result += f" ({value['correct']}/{value['expected']}; valid {value['valid']})"
    if value.get("accuracy_min") is not None:
        result += f" [{_pct(value['accuracy_min'])}, {_pct(value['accuracy_max'])}]"
    return result


def _table(headers: list[str], rows: list[list]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(_cell(v) for v in row) + " |" for row in rows),
        "",
    ]


def practical_takeaways(summary: dict) -> list[str]:
    runs = {row["model_id"]: row for row in summary["runs"]}
    groups = {row["format"]: row for row in summary["seed_aggregates"]}
    points = []
    bf16 = groups.get("bf16", {})
    base, decider, laya = (
        runs.get(name, {}) for name in ("gemma-base-bf16", "decider-local", "laya-general")
    )
    if bf16.get("accuracy_mean") is not None:
        points.append(
            f"- Gemma CE-128 BF16 records {_pct(bf16['accuracy_mean'])} mean BoolQ accuracy "
            f"across all three seeds, with worst measured CUDA allocation "
            f"{_memory(bf16['worst_peak_allocated_bytes'])} GB (GiB). It accepts raw images "
            "and audio, but media accuracy varies by task and high concurrency adds queueing."
        )
    if all(row.get("boolq", {}).get("accuracy") is not None for row in (base, decider)):
        points.append(
            f"- Decider records {_pct(decider['boolq']['accuracy'])}, compared with "
            f"Gemma base {_pct(base['boolq']['accuracy'])}"
            + (
                f" and CE-128 BF16 mean {_pct(bf16['accuracy_mean'])}"
                if bf16.get("accuracy_mean") is not None
                else ""
            )
            + f". Its peak CUDA allocation is {_memory(decider['memory']['peak_allocated_bytes'])} "
            "GB (GiB); native image/audio inputs are unsupported. These are point estimates, "
            "not proof of a population-level accuracy ordering."
        )
    if laya.get("boolq", {}).get("accuracy") is not None:
        observed_times = [
            row["boolq"]["mean_ms"]
            for row in runs.values()
            if row["boolq"].get("mean_ms") is not None
        ]
        fastest = observed_times and laya["boolq"].get("mean_ms") == min(observed_times)
        points.append(
            f"- Laya records {_pct(laya['boolq']['accuracy'])}, "
            f"{_number(laya['boolq'].get('mean_ms'))} ms serial mean latency"
            + (" (the lowest measured mean in the selected rows)" if fastest else "")
            + f", and {_memory(laya['memory']['peak_allocated_bytes'])} GB (GiB) peak allocation. "
            "This is a small, fast text-only option with lower BoolQ accuracy than Gemma CE-128."
        )
    for mode in ("int8", "nf4", "nf4-recovered"):
        group = groups.get(mode, {})
        if group.get("accuracy_mean") is None:
            continue
        members = [runs[f"gemma-ce128-s{seed}-{mode}"] for seed in range(3)]
        timings = [row["boolq"].get("mean_ms") for row in members]
        latency = (
            f" Serial mean latency spans {min(timings):.2f}–{max(timings):.2f} ms across the recorded runs."
            if all(value is not None for value in timings)
            else ""
        )
        point = (
            f"- {'Exploratory recovered NF4' if mode == 'nf4-recovered' else mode.upper()} "
            f"records {_pct(group['accuracy_mean'])} mean accuracy "
            f"[{_pct(group['accuracy_min'])}, {_pct(group['accuracy_max'])}], with worst "
            f"peak CUDA allocation {_memory(group['worst_peak_allocated_bytes'])} GB (GiB)."
            + latency
        )
        if mode == "int8":
            point += " The current runtime exceeds the strict 12 GB allocation target; INT8 is not a measured latency improvement over BF16 here."
        elif mode == "nf4":
            if group["accuracy_mean"] <= 0.89:
                point += " The original quantization-only method misses the requested >89–90% accuracy target."
            fsdd = [row["media"].get("by_task", {}).get("fsdd", {}) for row in members]
            if all(task.get("correct") == 1 and task.get("expected") == 20 for task in fsdd):
                point += (
                    " FSDD falls to 1/20 for every seed, so the media regression is substantial."
                )
        else:
            point += " This separate method adds 100 fixed training updates per seed."
            below = [seed for seed, row in enumerate(members) if row["boolq"]["accuracy"] <= 0.89]
            if group["accuracy_mean"] >= 0.9 and below:
                point += f" Its mean crosses 90%, but seeds {below} remain at or below 89%, so it does not establish an all-seed quality pass."
            if all(
                row["media"].get("by_task", {}).get("fsdd", {}).get("correct") == 1
                for row in members
            ):
                point += " The continuation does not recover audio accuracy: FSDD remains 1/20 for every seed."
        if group.get("accuracy_memory_target_status") == "failed":
            point += " The combined per-seed accuracy/memory acceptance target fails."
        points.append(point)
    if not points:
        points.append(
            "- Current measured comparisons are unavailable. Historical figures remain labeled separately; missing runs do not establish a ranking."
        )
    points.append(
        "- With 128 BoolQ cases, one decision changes accuracy by 0.78125 percentage points. "
        "Passing a point threshold is small-sample screening, not a guarantee for other tasks, "
        "distributions, or production load. Peak allocation alone is not a minimum GPU capacity."
    )
    return points


def render(
    summary: dict, historical: dict, historical_path: Path, campaign_path: Path | None, output: Path
) -> str:
    dataset = summary["dataset"]
    lines = [
        "# Accuracy, memory, and latency comparison",
        "",
        "This report is generated from saved evidence. New campaign rows and historical Modal "
        "results retain their own model, precision, hardware, and workload identities.",
        "",
        f"Campaign: `{summary['campaign_id']}`; status: **{summary['status']}**.",
        "No missing or blocked result is counted as a successful evaluation.",
        "",
        "## Shared evaluation contract",
        "",
        f"BoolQ uses the same {dataset['cases']} fixed validation cases, with dataset SHA-256 "
        f"`{dataset['sha256']}`. Errors and rejected requests remain in the full denominator. "
        "Repeated seeds reuse this test set; they are not additional independent test examples.",
        f"Native media uses {dataset['media_cases']} examples: 32 MNIST images and 20 FSDD "
        "recordings (six audio speaker groups). Text-only models have unsupported native media. "
        "OCR/ASR pipelines must be measured separately with preprocessing time and memory. "
        "The historical M2 source-label oracle is diagnostic and is never ranked as media accuracy.",
        "",
        "GB means decimal bytes / 1,000,000,000; GiB means bytes / 1,073,741,824. Model "
        "footprint, CUDA peak allocated, CUDA peak "
        "reserved, process-lifetime peak RSS, and GPU capacity are different quantities. The "
        "registered model tensor footprint can exclude auxiliary quantizer state; CUDA peaks "
        "include resident weights and the measured workload. Device used is an instantaneous "
        "device-wide observation, includes CUDA context and potentially other processes, and "
        "is not a per-process peak. Allocation below 8 GiB does not prove deployment on an "
        "8 GiB GPU: reserved memory and device usage may be higher. Acceptance caps use "
        "decimal GB, so an 8.344 GB allocation fails an 8.000 GB cap even though it is 7.771 GiB. "
        "Historical "
        "24 GB / 48 GB figures are nominal 12B weight estimates, not measured deployment minima. "
        "Quantization also retains some layers, scales, adapters, and activations at higher precision.",
        "",
        "The configured Modal tier is A100-80GB, but the allocator may provide PCIe or SXM "
        "variants. The table preserves each observed GPU SKU. Accuracy aggregates preserve "
        "that hardware list; latency remains per model/workload and is not pooled across SKUs.",
        "",
        "## Current campaign: identical BoolQ, native media, and memory harness",
        "",
    ]
    current = []
    for run in summary["runs"]:
        quality, mem = run["boolq"], run["memory"]
        media = run["media"]
        media_text = _accuracy(media)
        if media.get("native_status") in ("unsupported", "not_supported"):
            media_text = "unsupported native media"
        closed_loop = {
            cell.get("concurrency"): cell
            for cell in run["load"]
            if cell.get("arrival") == "closed_loop"
        }
        current.append(
            [
                f"{run['model_id']} / "
                + (
                    f"{run['quantization']} / "
                    if run.get("quantization") not in (None, "none")
                    else ""
                )
                + (run.get("precision") or "not recorded"),
                run.get("status") or "not_run",
                _accuracy(quality),
                media_text,
                _memory(mem["model_footprint_bytes"]),
                f"{_memory(mem['peak_allocated_bytes'])} / {_memory(mem['peak_reserved_bytes'])}",
                _memory(mem["device_used_bytes"]),
                f"{_number(quality.get('mean_ms'))} / {_number(quality.get('p99_ms'))}",
                f"{_number(closed_loop.get(1, {}).get('p99_ms'))} / "
                f"{_number(closed_loop.get(16, {}).get('p99_ms'))}",
                run.get("gpu") or "not measured",
            ]
        )
    lines.extend(
        _table(
            [
                "Model / format",
                "Run status",
                "BoolQ accuracy",
                "Native media accuracy",
                "Recorded tensor footprint GB (GiB)",
                "CUDA peak allocated / reserved GB (GiB)",
                "Device used GB (GiB)",
                "BoolQ mean / p99 ms",
                "Load C1 / C16 p99 ms",
                "GPU",
            ],
            current,
        )
    )
    if not current:
        lines.extend(["No campaign receipts supplied; the table contains no new measurements.", ""])
    lines.extend(
        [
            "Decider and Kev use eager PyTorch reference execution; their live worker logs report "
            "missing optional causal-conv1d and flash-linear-attention kernels. Published CUDA-graph "
            "or compiled GPU timings describe different runtimes. AgentJev uses the pinned current "
            "coding checkpoint, not the older typed-decision checkpoint. INT8 timing includes "
            "the logging behavior of its recorded run. Early verbose cast-warning attempts remain "
            "in the audit trail; newer receipts identify the once-per-process logging filter. "
            "The filter changes logging only, and the reruns preserve accuracy. Runtime and "
            "hardware variation prevent attributing a timing difference solely to logging.",
            "",
            "The selected Decider and Kev rows use corrected full-module tensor footprints. "
            "Earlier partial-module footprints remain only in superseded attempts in the audit "
            "index; the first Kev footprint excluded its small FP32 pointer head. CUDA peaks "
            "cover the full process in both versions. Treat measured CUDA allocation and "
            "device usage as separate deployment-memory observations.",
            "",
        ]
    )
    for run in summary["runs"]:
        if run.get("reason"):
            lines.append(f"- `{_cell(run['model_id'])}`: {_cell(run['reason'])}.")
    media_rows = []
    for run in summary["runs"]:
        media = run["media"]
        if media.get("accuracy") is None:
            continue
        scores = []
        for task in ("mnist", "fsdd"):
            observed = media.get("by_task", {}).get(task, {})
            count, correct = observed.get("expected"), observed.get("correct")
            scores.append(
                f"{correct}/{count} ({100 * correct / count:.2f}%)"
                if count and correct is not None
                else "not measured"
            )
        media_rows.append(
            [
                run["model_id"],
                *scores,
                _accuracy(media),
                f"{_number(media.get('mean_ms'))} / {_number(media.get('p99_ms'))}",
            ]
        )
    if media_rows:
        lines.extend(["## Current native-media retention", ""])
        lines.extend(
            _table(
                ["Model", "MNIST", "FSDD", "All 52 media cases", "Media mean / p99 ms"],
                media_rows,
            )
        )
        lines.extend(
            [
                "These are the same raw images and recordings in every current model row. "
                "Text-only providers remain unsupported for native media; no oracle "
                "description, OCR, or ASR output is inserted into these scores.",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "Accuracy targets (>89% or ≥90%) and memory targets (~12 GB or 6–8 GB) are acceptance "
            "goals. A format name, successful model load, or nominal bit count does not establish "
            "that a target was met. Each quantized CE adapter must be evaluated in its own row.",
            "",
            "## Current campaign load results",
            "",
            "Load percentiles include the workload identified below; they are separate from serial "
            "BoolQ evaluation time. Successful-request percentiles exclude failures, which remain "
            "visible in request counts. Small request counts provide screening estimates of p99.",
            "The current harness runs one model in one process and serializes inference with a "
            "service lock, without batching or request-result caching. Loopback HTTP excludes "
            "WAN latency. Each model receives six cells of 128 requests (768 total): closed-loop "
            "concurrency 1, 4, 16, and 64, plus fixed and Poisson arrivals at 5 requests/second "
            "with at most 16 outstanding requests. The latency SLO is 1,000 ms. These measurements "
            "include this runtime's queueing; they do not estimate an optimized provider's "
            "batched throughput.",
            "",
        ]
    )
    load_rows = []
    for run in summary["runs"]:
        for load in run["load"]:
            load_rows.append(
                [
                    run["model_id"],
                    load.get("arrival", load.get("load_mode", "not recorded")),
                    load.get("concurrency", "not recorded"),
                    f"{load.get('successful', 'not recorded')} / {load.get('requests', 'not recorded')}",
                    _number(load.get("p50_ms")),
                    _number(load.get("p95_ms")),
                    _number(load.get("p99_ms")),
                    load.get("slo_misses", "not recorded"),
                ]
            )
    lines.extend(
        _table(
            [
                "Model",
                "Arrival",
                "Concurrency",
                "Successful / requests",
                "p50 ms",
                "p95 ms",
                "p99 ms",
                "SLO misses",
            ],
            load_rows,
        )
    )
    if not load_rows:
        lines.extend(["No new completed load measurements are recorded.", ""])
    lines.extend(
        [
            "## Quantization and target checks",
            "",
            "The implemented paths use bitsandbytes INT8 or NF4 with BF16 surrounding layers "
            "and NF4 compute. The INT8 kernel internally casts activations to FP16 before "
            "quantizing them. The tied "
            "input embedding / LM head and media projections remain dense because the candidate "
            "readout indexes floating-point head rows. Existing CE-128 adapters are reused across "
            "the original three formats and all three predeclared seeds; no seed is selected "
            "using test accuracy. Any NF4 recovery extension is labeled separately because "
            "it adds training updates and changes adapter weights. "
            "CPU offload is disabled. FP8 is not an implemented path in this campaign. "
            "The [quantized inference guide](quantization.md) documents local CLI and Modal "
            "serving flags.",
            "",
            "The pinned checkpoint contains 11,959,730,176 parameters: 10,899,947,520 eligible "
            "linear parameters and 1,059,782,656 parameters retained at BF16, including a "
            "1,006,632,960-parameter tied embedding/head counted once. Therefore the raw "
            "parameter-only floors are **23.919 GB BF16**, **13.020 GB INT8**, and **7.570 GB NF4**. "
            "These are derived counts, not GPU measurements; quantizer metadata, adapters, "
            "activations and workspaces add memory. A strict 12 GB INT8 allocation target cannot "
            "be met by this linear-only conversion. The NF4 8 GB target has little room beyond "
            "its raw parameters and must be checked against live peak allocation. See the "
            "[metadata-only sizing receipt](validation/2026-10-01/quantization-sizing.json).",
            "",
        ]
    )
    targets = []
    for run in summary["runs"]:
        if run.get("quantization") not in ("int8", "nf4"):
            continue
        target = run["acceptance"]
        targets.append(
            [
                run["model_id"],
                target.get("status", "not_measured"),
                _accuracy(run["boolq"]),
                _memory(run["memory"]["peak_allocated_bytes"]),
                _gb(target.get("peak_allocated_cap_bytes")),
                target.get("accuracy_at_least_90pct", "not measured"),
                target.get("memory_target_pass", "not measured"),
            ]
        )
    lines.extend(
        _table(
            [
                "Quantized model",
                "Accuracy/memory target status",
                "BoolQ",
                "Peak allocated GB (GiB)",
                "Target cap GB",
                "Accuracy ≥90%",
                "Memory target met",
            ],
            targets,
        )
    )
    aggregates = []
    for group in summary["seed_aggregates"]:
        score = _pct(group["accuracy_mean"])
        if group["accuracy_mean"] is not None:
            score += f" [{_pct(group['accuracy_min'])}, {_pct(group['accuracy_max'])}]"
        recovered = group["format"] == "nf4-recovered"
        delta = (
            group["paired_mean_delta_vs_original_nf4_pp"]
            if recovered
            else group["paired_mean_delta_vs_bf16_pp"]
        )
        reference = "original NF4" if recovered else "BF16"
        aggregates.append(
            [
                group["format"],
                group["status"],
                str(group["completed_seeds"]),
                score,
                _memory(group["worst_peak_allocated_bytes"]),
                f"{delta:+.3f} vs {reference}" if delta is not None else "not measured",
                group["accuracy_memory_target_status"],
                group["all_run_stages_completed"],
                "; ".join(group["observed_gpus"]),
            ]
        )
    lines.extend(
        _table(
            [
                "CE-128 format",
                "3-seed evidence",
                "Completed seeds",
                "Mean accuracy [min, max]",
                "Worst peak allocated GB (GiB)",
                "Paired mean delta, pp / reference",
                "Accuracy/memory target",
                "All run stages complete",
                "Observed GPU SKUs",
            ],
            aggregates,
        )
    )
    lines.extend(
        [
            "Three-seed aggregates require all predeclared seeds (0, 1, 2), complete BoolQ "
            "coverage, and matching GPU identity or configured GPU tier. Observed SKUs remain "
            "explicit. Missing seeds are never dropped from an average. "
            "Memory uses the worst peak across seeds, and paired deltas compare the same seed "
            "against its newly measured BF16 row (or original NF4 for the extra-training "
            "recovery). These are three models evaluated on the same "
            "128 examples, not 384 independent test examples. Accuracy/memory target status "
            "is separate from completion of every load stage and its latency SLOs.",
            "",
        ]
    )
    recovery_runs = [run for run in summary["runs"] if run["model_id"].endswith("-nf4-recovered")]
    recovery_complete = len(recovery_runs) == 3 and all(
        run["boolq"].get("valid") == 128 for run in recovery_runs
    )
    lines.extend(
        [
            "**Exploratory NF4 recovery:** an additional fixed 100-update QLoRA continuation "
            "is a separate method, using the 256-case training pool and 16-case calibration "
            "split. The parent CE adapter has 100 optimizer updates; the fixed continuation "
            "adds 100 at batch size 1 (100 training cases, less than one epoch), for a "
            "200-update recipe. Its adapter weights differ from the original CE adapter, and it must not "
            "be described as a controlled quantization-only comparison. The original NF4 scores "
            "and every seed remain visible.",
            "All three recovery quality receipts are available in this snapshot."
            if recovery_complete
            else "Recovery evidence is pending or incomplete in this snapshot; "
            "no three-seed recovery accuracy or target pass is claimed.",
            "",
        ]
    )
    if summary["comparisons"]:
        lines.extend(
            [
                "## Current paired BoolQ comparisons",
                "",
                "Deltas are candidate minus reference, in percentage points. Every computed "
                "interval uses the same 128 recorded requests and source-group bootstrap; the "
                "95% intervals are descriptive and unadjusted for multiple comparisons. Same-seed "
                "quantization comparisons additionally require identical checkpoint and adapter "
                "identities. Explicit recovery pairs permit a changed adapter only when "
                "its recorded parent digest and fixed training-data recipe are verified; "
                "these compare extra training, not quantization alone. Missing or mismatched "
                "evidence does not produce an interval.",
                "",
            ]
        )
        comparison_rows = []
        for comparison in summary["comparisons"]:
            delta = comparison.get("operational_accuracy_delta", {})
            interval = delta.get("interval")
            estimate = delta.get("estimate")
            comparison_rows.append(
                [
                    comparison["left"],
                    comparison["right"],
                    comparison["status"],
                    f"{100 * estimate:+.3f}" if estimate is not None else "not measured",
                    f"[{100 * interval[0]:+.3f}, {100 * interval[1]:+.3f}]"
                    if interval
                    else "not measured",
                    delta.get("groups", "not measured"),
                ]
            )
        lines.extend(
            _table(
                [
                    "Reference",
                    "Candidate",
                    "Comparison status",
                    "Accuracy delta, pp",
                    "95% CI, pp",
                    "Source groups",
                ],
                comparison_rows,
            )
        )
    lines.extend(
        [
            "Target checks apply to complete 128-case BoolQ observations and measured CUDA "
            "allocation. They do not certify the total capacity of a smaller GPU, reserve space "
            "for another process, or imply that load-latency SLOs passed. Tiny random-weight CUDA "
            "tests verify conversion/readout/adapter/media execution but do not establish full "
            "12B accuracy or memory acceptance.",
            "",
            "## Historical Modal A100 results (2026-10-01)",
            "",
        ]
    )
    historical_table = []
    for row in summary["historical"]:
        quality = row["boolq"]
        if quality.get("mean_ms") is not None:
            latency = _number(quality["mean_ms"])
        elif quality.get("mean_ms_min") is not None:
            latency = f"{quality['mean_ms_min']:.2f}–{quality['mean_ms_max']:.2f}"
        else:
            latency = "not recorded"
        media_text = _accuracy(row["media"]) if row["media"] else row["media_status"]
        if row["media"] and row["media_status"] == "seed 0 only":
            media_text += "; seed 0 only"
        weights = (
            f"≈{row['nominal_weight_gb']} GB nominal" if row["nominal_weight_gb"] else "unknown"
        )
        historical_table.append(
            [
                f"{row['model_id']} ({row['precision']})",
                _accuracy(quality),
                row["seeds"],
                media_text,
                weights,
                "not recorded",
                latency,
                row["gpu"],
            ]
        )
    lines.extend(
        _table(
            [
                "Model",
                "BoolQ accuracy [seed min, max]",
                "Seeds",
                "Native media accuracy",
                "Weight size (estimate only)",
                "Peak VRAM",
                "BoolQ wall/decision ms",
                "Hardware",
            ],
            historical_table,
        )
    )
    lines.extend(
        [
            "Historical CE latency is total held-out evaluation wall time divided by 128 decisions, "
            "shown per seed as a range. It is not a service p99 and must not be assigned to a "
            "quantized model. The 91.15% CE-128 result averages three seeds; seed 0 is 116/128 "
            "(90.625%). The media retention measurements for trained models use seed 0 only.",
            "",
            "Historical base BF16 media accuracy is 28/32 (87.50%) on MNIST and 6/20 (30.00%) on "
            "FSDD. These are digit recognition subsets, not broad visual/audio capability tests. "
            "CE-128 seed 0 improves BoolQ by 3.906 percentage points versus base, with descriptive "
            "paired 95% source-group CI [-1.562, +9.375] pp; this does not establish a statistically "
            "accepted improvement. Laya's point difference versus base is -4.688 pp with CI "
            "[-13.301, +3.926] pp.",
            "",
            "## Historical latency and load, by workload",
            "",
        ]
    )
    old_load = []
    for stage, item in summary["historical_load"].items():
        for load in item.get("cells", []):
            systems = load["systems"]
            rate = load.get("offered_requests_per_second")
            old_load.append(
                [
                    stage,
                    load["id"],
                    f"{load['successful']} / {load['requests']}",
                    _number(rate, 3)
                    if rate is not None
                    else (
                        "closed loop" if load.get("load_mode") == "closed_loop" else "not recorded"
                    ),
                    _number(systems.get("successful_p99_ms"), 3),
                    systems.get("deadline_misses", "not recorded"),
                ]
            )
    lines.extend(
        _table(
            [
                "Stage",
                "Workload",
                "Successful / requests",
                "Offered requests/s",
                "p99 ms",
                "1,000 ms SLO misses",
            ],
            old_load,
        )
    )
    sweeps = []
    for stage, item in summary["historical_batch_checks"].items():
        for sweep in item.get("runtime_sweeps", []):
            if sweep["id"] not in ("g0-g4-contract", "native-batch-1"):
                continue
            model = sweep.get("models", {}).get("gemma-g4", {})
            systems = model.get("systems", {})
            sweeps.append(
                [
                    stage,
                    item.get("precision"),
                    sweep["id"],
                    _number(systems.get("successful_p99_ms"), 3),
                ]
            )
    lines.extend(_table(["Stage", "Precision", "Direct G4 workload", "p99 ms"], sweeps))
    lines.extend(
        [
            "The legacy 10,000-request service result is 105.027 ms p99 (block-bootstrap 95% CI "
            "102.413–106.950 ms). Its offered rate and actual peak inflight were not recorded. "
            "The later stress harness measured queueing: closed-loop concurrency 16 reached "
            "1,564.742 ms p99 and concurrency 64 reached 6,692.645 ms, with SLO misses. "
            "The legacy result therefore does not imply 105 ms p99 at genuine concurrency 16. "
            "All these service tests used loopback HTTP and do not include WAN latency.",
            "",
            "BF16 native-batch and question-independence parity checks failed in the historical "
            "run; the FP32 control passed. Direct-call timing and accuracy depend on precision, "
            "batch shape, context, and number of candidates. All original findings remain in the "
            "historical evidence.",
            "",
            "## Practical strengths and limits",
            "",
            *practical_takeaways(summary),
            "",
            "## Evidence and reproduction",
            "",
            f"Historical source: {_link(historical_path, output, 'live-summary.json')} "
            f"(SHA-256 `{_hash(historical_path)}`). The "
            "[full historical report](validation/2026-10-01/live-validation.md) includes prediction "
            "hashes, model revisions, Modal jobs, calibration, learning curves, and provenance. "
            "The [September 30 smoke report](validation/2026-09-30/README.md) used A100 40GB "
            "and five fixture decisions; those smoke results are not an accuracy ranking.",
            "",
        ]
    )
    if campaign_path:
        lines.extend(
            [
                f"Campaign source: {_link(campaign_path, output, 'campaign receipt')} "
                f"(SHA-256 `{_hash(campaign_path)}`).",
                "",
            ]
        )
    if summary["attempts"]:
        selected_count = sum(attempt.get("selected", False) for attempt in summary["attempts"])
        lines.extend(
            [
                f"The combined receipt retains **{len(summary['attempts'])} attempts**, with "
                f"**{selected_count} selected model rows**. Selection uses the latest completed "
                "attempt by recorded start time, or the latest compatible incomplete attempt "
                "when none completed; it never selects by test accuracy. Superseded failures "
                "and incompatible pre-execution attempts remain in its audit index.",
                "",
            ]
        )
    identities = []
    for run in summary["runs"]:
        identities.append(
            [
                run["model_id"],
                run.get("checkpoint") or "not recorded",
                run.get("revision") or "not recorded",
                _memory(run["memory"]["process_peak_rss_bytes"]),
                run["telemetry"].get("context_limit", "not recorded"),
                run["telemetry"].get("context_policy", "not recorded"),
            ]
        )
    lines.extend(
        _table(
            [
                "Model",
                "Checkpoint",
                "Pinned revision",
                "Process peak RSS GB (GiB)",
                "Context limit",
                "Context policy",
            ],
            identities,
        )
    )
    lines.extend(
        [
            "Follow the [original-data preparation and restoration steps]"
            "(modal.md#reproduce-the-october-1-matched-comparison) before launching a fresh run. "
            "The restored native media is byte-identical to the archived dataset; its "
            "[recovery audit](validation/2026-10-01/media-rebuild.json) retains the reconstruction "
            "diagnostics without changing the benchmark input identity.",
            "",
            "```bash",
            "uv run --no-sync modal run apps/modal/compare.py --run your-unique-run",
            "# Exploratory additional-training method; reports all three seeds separately.",
            "uv run --no-sync modal run apps/modal/recover_nf4.py --run your-recovery-run",
            "uv run --no-sync python scripts/collect_comparison.py \\",
            "  artifacts/comparison/your-unique-run/campaign.json \\",
            "  artifacts/comparison/your-recovery-run/campaign.json \\",
            "  --output docs/validation/2026-10-01/comparison-summary.json",
            "uv run --no-sync python scripts/summarize_comparison.py \\",
            "  --historical docs/validation/2026-10-01/live-summary.json \\",
            "  --campaign docs/validation/2026-10-01/comparison-summary.json \\",
            "  --output docs/comparison.md",
            "```",
            "",
            "For split campaigns or retries, pass all campaign JSON paths to the collector; "
            "it retains their attempt history and never selects a model by test score. "
            "The renderer rejects changed BoolQ/media dataset hashes and inconsistent accuracy "
            "denominators. Missing measurements remain explicit. See the campaign runner's "
            "`--help` for live execution options; generating this document does not launch GPU jobs.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/comparison.md")
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    historical = _read(args.historical)
    campaign = _read(args.campaign) if args.campaign else None
    summary = build_summary(historical, campaign)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(summary, historical, args.historical, args.campaign, args.output))
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
