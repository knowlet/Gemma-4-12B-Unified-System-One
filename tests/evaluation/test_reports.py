import json
from pathlib import Path

from s1.evaluation.gates import assess_gate
from s1.evaluation.registry import Registry
from s1.evaluation.reports import build_report
from s1.evaluation.runner import execute
from s1.evaluation.statistics import paired_comparison
from s1.evaluation.workflow import execute_workflows

ROOT = Path(__file__).resolve().parents[2]


def test_standalone_report_contains_observed_quality_and_resettable_workflow(tmp_path):
    config = ROOT / "configs/benchmarks"
    registry = Registry.load(
        config / "models.toml", config / "suites.toml", config / "profiles.toml"
    )
    execute(
        registry,
        "text-smoke",
        output=tmp_path / "run",
        lockfile=ROOT / "uv.lock",
        enable_models=["uniform"],
    )
    workflow = execute_workflows(
        registry,
        ROOT / "examples/benchmarks/workflows.jsonl",
        "uniform",
        "uniform",
        output=tmp_path / "workflow",
        lockfile=ROOT / "uv.lock",
        enable_models=["uniform"],
    )
    assert workflow["status"] == "completed"
    assert workflow["policies"]["always_small"]["false_completions"] == 6
    assert workflow["policies"]["rules_first"]["completion_rate"] == 1
    result = build_report([tmp_path / "run", tmp_path / "workflow"], tmp_path / "report")
    assert result["groups"] == result["workflow_runs"] == 1
    html = (tmp_path / "report/report.html").read_text()
    assert "<svg" in html and "Closed-loop replay" in html
    assert "<script" not in html and "https://" not in html
    assert "18/18" in (tmp_path / "report/report.md").read_text()


def test_gate_requires_sufficient_source_groups(tmp_path):
    spec = {
        "left_experiment_id": "a" * 64,
        "right_experiment_id": "b" * 64,
        "accuracy_margin": 0.01,
        "min_groups": 30,
    }
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps(spec))
    comparison = tmp_path / "comparison.json"

    def assess(groups, interval):
        rows = [
            {
                "case_id": str(index),
                "question_id": "q",
                "repetition": 0,
                "group_id": str(index),
                "gold": "a",
                "actual_label": "a",
                "status": "ok",
                "eligibility": "eligible",
            }
            for index in range(groups)
        ]
        comparison.write_text(
            json.dumps(
                {
                    "comparisons": [
                        {
                            **spec,
                            "population": paired_comparison(rows, rows, resamples=100)[
                                "population"
                            ],
                            "operational_accuracy_delta": {
                                "groups": groups,
                                "observations": groups,
                                "interval": interval,
                            },
                        }
                    ]
                }
            )
        )
        return assess_gate(comparison, gate)["status"]

    assert assess(2, [0, 0]) == "inconclusive"
    assert assess(100, [-0.001, 0.02]) == "passed"
    assert assess(100, [-0.02, -0.015]) == "failed"
    assert assess(100, [-0.02, 0.01]) == "inconclusive"
