"""Offline E0–E7 tooling acceptance; no pretrained weights or network calls.

Produces inspectable fixtures/artifacts, not a model-quality benchmark result.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from s1.evaluation.artifacts import write_json
from s1.evaluation.datasets import EvaluationCase
from s1.evaluation.experiments import perturb_dataset, prepare_sweep, write_cases
from s1.evaluation.multimodal import prepare_media_views
from s1.evaluation.registry import Registry
from s1.evaluation.reports import build_report
from s1.evaluation.runner import execute
from s1.evaluation.statistics import compare_runs
from s1.evaluation.supervision import fit_baseline, prepare_support
from s1.evaluation.workflow import execute_workflows


def run(output):
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    configs = root / "configs/benchmarks"
    registry = Registry.load(
        configs / "models.toml", configs / "suites.toml", configs / "profiles.toml"
    )
    profile = registry.profiles["text-smoke"]
    models = dict(registry.models)
    models["uniform-control"] = models["uniform"].model_copy(update={"id": "uniform-control"})
    registry = replace(registry, models=models)
    runs = []
    for mode in ("closed_loop", "fixed", "poisson"):
        selected = profile.model_copy(
            update={
                "models": ("uniform", "uniform-control"),
                "load_mode": mode,
                "concurrency": 2,
                "arrival_rate": 1000.0,
                "slo_ms": 100.0,
                "budget": profile.budget.model_copy(update={"max_requests": 22}),
            }
        )
        configured = replace(registry, profiles={profile.id: selected})
        directory = output / mode
        result = execute(
            configured,
            profile.id,
            output=directory,
            lockfile=root / "uv.lock",
            enable_models=["uniform", "uniform-control"],
        )
        assert result["run_status"] == "completed"
        assert all(
            m["records_complete"] and m["denominators"]["valid_decisions"] == 18
            for m in result["models"].values()
        )
        runs.append(directory)
    write_json(output / "comparison.json", compare_runs([runs[0]], seed=0, resamples=200))
    perturb_dataset(
        root / "examples/benchmarks/text-contract.jsonl", output / "reordered.jsonl", seed=3
    )
    prepare_sweep(output / "sweep", cases_per_cell=2)
    for split, count in (("train", 16), ("calibration", 4), ("test", 4)):
        cases = [
            EvaluationCase.model_validate(
                {
                    "id": f"{split}-{i}",
                    "group_id": f"{split}-{i}",
                    "split": split,
                    "task_id": "synthetic-specialist",
                    "schema_id": "sentiment",
                    "language": "en",
                    "request": {
                        "state": f"{'happy' if i % 2 else 'sad'} fixture {split} {i}",
                        "questions": [
                            {
                                "id": "q",
                                "type": "choice",
                                "instructions": "Classify sentiment",
                                "criteria": {"negative": "sad", "positive": "happy"},
                            }
                        ],
                    },
                    "gold": {"q": "positive" if i % 2 else "negative"},
                }
            )
            for i in range(count)
        ]
        write_cases(output / f"{split}.jsonl", cases)
    support = prepare_support(
        output / "train.jsonl",
        [output / "calibration.jsonl", output / "test.jsonl"],
        output / "support",
        shots=[2, 4],
    )
    fitted = fit_baseline(
        output / "support" / support["cells"][0]["dataset"], output / "prior", method="prior"
    )
    assert len(fitted["support_ids"]) == 4
    media_rows = []
    for index, sample in enumerate((0.25, -0.25)):
        media_rows.append(
            {
                "case": {
                    "id": f"audio-{index}",
                    "group_id": "audio-pair",
                    "split": "test",
                    "request": {
                        "state": "Observe the signal",
                        "media": [{"type": "audio", "samples": [sample]}],
                        "questions": [
                            {
                                "id": "q",
                                "type": "choice",
                                "instructions": "Signal sign?",
                                "criteria": {
                                    "positive": "positive",
                                    "negative": "negative",
                                    "unknown": "insufficient information",
                                },
                            }
                        ],
                    },
                    "gold": {"q": "positive" if sample > 0 else "negative"},
                },
                "missing_media_gold": {"q": "unknown"},
                "verified_text": "Positive signal" if sample > 0 else "Negative signal",
            }
        )
    bundle = output / "media-bundle.jsonl"
    bundle.write_text("\n".join(json.dumps(r) for r in media_rows) + "\n")
    prepare_media_views(bundle, output / "media-views")
    workflow = output / "workflow"
    result = execute_workflows(
        registry,
        root / "examples/benchmarks/workflows.jsonl",
        "uniform",
        "uniform",
        output=workflow,
        lockfile=root / "uv.lock",
        enable_models=["uniform"],
    )
    assert result["status"] == "completed"
    assert result["policies"]["rules_first"]["completion_rate"] == 1
    report = build_report([*runs, workflow], output / "report")
    receipt = {
        "status": "passed",
        "kind": "offline_fixture_acceptance",
        "model_quality_evidence": False,
        "run_directories": [str(p) for p in runs],
        "report": report,
        "live_models": "not_run; requires pinned weights/credentials/hardware and representative data",
    }
    write_json(output / "acceptance.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
