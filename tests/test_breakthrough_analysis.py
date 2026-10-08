"""Independent soft metrics, full populations and offline analysis failure states."""

import copy
import hashlib
import json
import math
import runpy
from pathlib import Path

import pytest

from s1.contracts import answer_from_probabilities
from s1.evaluation.datasets import EvaluationCase

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def analyzer():
    return runpy.run_path(str(ROOT / "scripts/analyze_breakthrough.py"))


def row(case="first", group="world-a", kind="choice", target=None, probs=None):
    labels = ["a", "b"] if kind != "score" else ["0", "2", "5"]
    target = target or ([0.2, 0.8] if kind != "score" else [0.2, 0.2, 0.6])
    probs = probs or ([0.5, 0.5] if kind != "score" else [0.3, 0.4, 0.3])
    return {
        "case_id": case,
        "question_id": "answer",
        "group_id": group,
        "type": kind,
        "split": "test",
        "labels": labels,
        "target": target,
        "logits": [math.log(p) for p in probs],
        "score_values": [0, 2, 5] if kind == "score" else None,
    }


def case(name="c1", *, questions=1, split="test"):
    return EvaluationCase.model_validate(
        {
            "id": name,
            "split": split,
            "group_id": f"source-{name}",
            "request": {
                "state": {},
                "questions": [
                    {
                        "id": f"q{i}",
                        "type": "choice",
                        "instructions": "Select.",
                        "criteria": {"a": "first", "b": "second"},
                    }
                    for i in range(questions)
                ],
            },
            "gold": {f"q{i}": "a" for i in range(questions)},
            "soft_gold": {f"q{i}": {"a": 0.75, "b": 0.25} for i in range(questions)},
        }
    )


def test_soft_metrics_have_known_probability_and_uneven_score_mean(analyzer):
    value = analyzer["row_metrics"](row(), 1.0)
    assert value["soft_ce"] == pytest.approx(math.log(2))
    assert value["kl"] == pytest.approx(math.log(2) + 0.2 * math.log(0.2) + 0.8 * math.log(0.8))
    assert value["brier"] == pytest.approx(0.18)
    ordinal = analyzer["row_metrics"](row(kind="score"), 1.0)
    assert ordinal["expected_score_mae"] == pytest.approx(
        abs((0.4 * 2 + 0.3 * 5) - (0.2 * 2 + 0.6 * 5))
    )
    assert "accuracy" not in ordinal
    with pytest.raises(ValueError, match="invalid soft target"):
        analyzer["row_metrics"](row(target=[0.1, 0.1]), 1.0)
    with pytest.raises(ValueError, match="temperature"):
        analyzer["row_metrics"](row(), 0)


def test_paired_loss_intervals_use_source_groups_and_full_population(analyzer):
    reference = [row(case=str(i), group=f"world-{i // 2}") for i in range(4)]
    candidate = [row(case=str(i), group=f"world-{i // 2}", probs=[0.2, 0.8]) for i in range(4)]
    temps = {"choice": 1.0, "noul": 1.0, "score": 1.0}
    result = analyzer["paired_soft"](reference, candidate, temps, temps, samples=100)
    brier = result["metrics"]["choice"]["brier"]
    assert brier["groups"] == 2 and brier["observations"] == 4
    assert brier["estimate"] == pytest.approx(-0.18)
    assert brier["interval"] == pytest.approx([-0.18, -0.18])
    incomplete = analyzer["paired_soft"](reference, candidate[:3], temps, temps, samples=100)
    assert (
        incomplete["status"] == "inconclusive"
        and incomplete["population"]["candidate_missing"] == 1
    )
    assert not incomplete["metrics"]
    duplicate = candidate + [candidate[0]]
    with pytest.raises(ValueError, match="duplicate"):
        analyzer["paired_soft"](reference, duplicate, temps, temps, samples=100)
    changed = copy.deepcopy(candidate)
    changed[0]["target"] = [0.4, 0.6]
    with pytest.raises(ValueError, match="metadata"):
        analyzer["paired_soft"](reference, changed, temps, temps, samples=100)


def test_native_audit_counts_question_slots_and_rejects_missing_or_wrong_answers(
    analyzer, tmp_path
):
    cases = [case("c1", questions=2), case("c2", questions=3)]
    saved = [
        {
            "case_id": c.id,
            "modalities": [],
            "answers": {
                q.id: answer_from_probabilities(q, [0.75, 0.25]) for q in c.request.questions
            },
        }
        for c in cases
    ]
    path = tmp_path / "native.json"
    path.write_text(json.dumps(saved))
    result = analyzer["audit_native"](path, cases)
    assert result["n_cases"] == 2 and result["n_questions"] == 5
    assert result["complete"]
    baseline = result["records"]
    changed = copy.deepcopy(baseline)
    changed[0].update(predicted="b", correct=False, probabilities={"a": 0.2, "b": 0.8})
    paired = analyzer["paired_native"](baseline, changed, samples=100)
    assert paired["n_paired"] == 5 and paired["label_flips"] == 1
    assert paired["max_probability_drift"] == pytest.approx(0.55)
    saved[0]["answers"]["q0"]["choice"] = "b"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="contradicts"):
        analyzer["audit_native"](path, cases)
    saved[0]["answers"].pop("q0")
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="question population"):
        analyzer["audit_native"](path, cases)


def test_regression_recomputes_full_population_and_saved_counters(analyzer):
    cases = [case("c1"), case("c2")]
    rows = [
        {
            "case_id": c.id,
            "gold": c.gold,
            "response": {
                "answers": {
                    q.id: answer_from_probabilities(q, [0.75, 0.25]) for q in c.request.questions
                }
            },
        }
        for c in cases
    ]
    saved = {"records": rows, "n_cases": 2, "n_questions": 2, "n_correct": 2, "accuracy": 1.0}
    assert analyzer["audit_regression_records"](saved, cases)["n_correct"] == 2
    changed = copy.deepcopy(saved)
    changed["n_correct"] = 1
    with pytest.raises(ValueError, match="counters"):
        analyzer["audit_regression_records"](changed, cases)
    with pytest.raises(ValueError, match="population"):
        analyzer["audit_regression_records"](saved, cases + [case("missing")])


def test_native_tie_uses_frozen_question_order_after_sorted_json(analyzer, tmp_path):
    sample = EvaluationCase.model_validate(
        {
            "id": "tie",
            "split": "test",
            "request": {
                "state": {},
                "questions": [
                    {
                        "id": "q",
                        "type": "choice",
                        "instructions": "Select.",
                        "criteria": {"z": "first", "a": "second"},
                    }
                ],
            },
            "gold": {"q": "z"},
        }
    )
    q = sample.request.questions[0]
    saved = [
        {
            "case_id": sample.id,
            "modalities": [],
            "answers": {q.id: answer_from_probabilities(q, [0.5, 0.5])},
        }
    ]
    path = tmp_path / "native.json"
    path.write_text(json.dumps(saved, sort_keys=True))
    result = analyzer["audit_native"](path, [sample])
    assert result["records"][0]["predicted"] == "z" and result["records"][0]["correct"]


def test_soft_record_identity_checks_frozen_targets_and_groups(analyzer):
    sample = case()
    q = sample.request.questions[0]
    record = {
        "case_id": sample.id,
        "question_id": q.id,
        "group_id": sample.group_id,
        "split": "test",
        "type": q.type,
        "labels": q.labels(),
        "score_values": None,
        "target": [0.75, 0.25],
        "logits": [0.0, 0.0],
    }
    assert analyzer["validate_soft_records"]([record], [sample], "test")
    altered = {**record, "group_id": "forged-group"}
    with pytest.raises(ValueError, match="source"):
        analyzer["validate_soft_records"]([altered], [sample], "test")
    with pytest.raises(ValueError, match="targets"):
        analyzer["validate_soft_records"]([{**record, "target": [0.25, 0.75]}], [sample], "test")


def test_synthetic_calibration_and_heldout_metrics_are_recomputed(analyzer, tmp_path):
    populations = {}
    for split in ("calibration", "test"):
        cases, rows = [], []
        for kind, labels, probabilities in (
            ("choice", ["a", "b"], [0.75, 0.25]),
            ("noul", ["false", "true"], [0.25, 0.75]),
            ("score", ["0", "2", "5"], [0.1, 0.3, 0.6]),
        ):
            name = f"{split}-{kind}"
            sample = EvaluationCase.model_validate(
                {
                    "id": name,
                    "group_id": name,
                    "task_id": f"template-{split}-{kind}",
                    "split": split,
                    "request": {
                        "state": {},
                        "questions": [
                            {
                                "id": "q",
                                "type": kind,
                                "instructions": "Report known frequencies.",
                                "criteria": {label: label for label in labels},
                            }
                        ],
                    },
                    "gold": {"q": labels[probabilities.index(max(probabilities))]},
                    "soft_gold": {"q": dict(zip(labels, probabilities, strict=True))},
                }
            )
            cases.append(sample)
            rows.append(
                {
                    "case_id": name,
                    "group_id": name,
                    "split": split,
                    "type": kind,
                    "question_id": "q",
                    "labels": labels,
                    "target": probabilities,
                    "logits": [math.log(p) for p in probabilities],
                    "score_values": [0, 2, 5] if kind == "score" else None,
                }
            )
        populations[split] = cases, rows
    rows = populations["calibration"][1]
    temperatures = {
        r["type"]: {
            "temperature": 1.0,
            "n": 1,
            "boundary": False,
            "soft_ce_before": -sum(p * math.log(p) for p in r["target"]),
            "soft_ce_after": -sum(p * math.log(p) for p in r["target"]),
        }
        for r in rows
    }
    entropy = {r["type"]: -sum(p * math.log(p) for p in r["target"]) for r in rows}
    metrics = {
        kind: {
            "n": 3 if kind == "all" else 1,
            "soft_ce": sum(entropy.values()) / 3 if kind == "all" else entropy[kind],
            "kl": 0.0,
            "brier": 0.0,
            "oracle_argmax_match": 1.0,
            **({"expected_score_mae": 0.0} if kind in ("all", "score") else {}),
        }
        for kind in ("all", "choice", "noul", "score")
    }
    (tmp_path / "calibration.json").write_text(
        json.dumps({"records": rows, "temperatures": temperatures})
    )
    (tmp_path / "heldout.json").write_text(
        json.dumps(
            {
                "records": populations["test"][1],
                "raw_metrics": metrics,
                "calibrated_metrics": metrics,
            }
        )
    )
    result = analyzer["audit_synthetic"](
        tmp_path, populations["calibration"][0], populations["test"][0]
    )
    assert result["complete"] and result["n_test"] == 3
    assert result["calibrated"]["score"]["expected_score_mae"] == pytest.approx(0)
    temperatures["choice"]["temperature"] = 2.0
    (tmp_path / "calibration.json").write_text(
        json.dumps({"records": rows, "temperatures": temperatures})
    )
    with pytest.raises(ValueError, match="fixed soft-CE grid"):
        analyzer["audit_synthetic"](tmp_path, populations["calibration"][0], populations["test"][0])


def test_replay_requires_identical_logits_no_forward_and_temperature_probabilities(analyzer):
    runtime = {
        "raw_logits": [{"id": "decision", "labels": ["a", "b"], "logits": [0.0, math.log(3)]}]
    }
    native = {
        "complete": True,
        "records": [{"task_id": "t1", "ok": True}],
        "raw_runtime": {"t1": runtime},
        "raw_answers": [{"decision": {"type": "choice", "probabilities": {"a": 0.25, "b": 0.75}}}],
    }
    replay = copy.deepcopy(native)
    replay["manifest"] = {
        "identity": {"execution": "raw-logit replay; CPU only", "calibration": {"choice": 1.0}}
    }
    replay["raw_runtime"]["t1"].update(calibration_replay=True, forward_calls=0)
    assert (
        analyzer["public_replay_identity"](native, replay)["status"] == "verified_same_raw_logits"
    )
    replay["raw_runtime"]["t1"]["forward_calls"] = 1
    with pytest.raises(ValueError, match="invoked"):
        analyzer["public_replay_identity"](native, replay)
    replay["raw_runtime"]["t1"]["forward_calls"] = 0
    replay["raw_answers"][0]["decision"]["probabilities"] = {"a": 0.5, "b": 0.5}
    with pytest.raises(ValueError, match="probabilities"):
        analyzer["public_replay_identity"](native, replay)


def test_coherence_uses_official_check_support_not_outcome_row_count(analyzer, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for i in range(1248):
        (cache / (hashlib.sha256(str(i).encode()).hexdigest() + ".json")).write_text("{}")
    scores = {
        "n_cases": 240,
        "n_checks": 50,
        "overall": 0.5,
        "pillars": {"LOG": 0.7, "MEA": 0.3},
        "incomplete": [],
        "checks": {
            f"check{i}": {"n": 24, "planned": 24, "coverage": 1.0, "score": 0.5} for i in range(50)
        },
    }
    result = {
        "backend": "profile@pin",
        "errors": [],
        "outcomes": [{}] * 1468,
        "bases": {"c1": {"base": {"q1": {"a": 0.25, "b": 0.75}}}},
    }
    path = tmp_path / "coherence.json"
    path.write_text(json.dumps({"scores": scores, "result": result}))
    audited = analyzer["audit_coherence"](path, cache, backend="profile@pin")
    assert audited["complete"] and audited["planned_tests"] == 1200
    assert audited["outcome_rows"] == 1468 and audited["raw_cache_files"] == 1248
    assert audited["status"] == "saved_only_not_officially_replayed"
    unit = copy.deepcopy(audited)
    unit["overall"] = 0.4
    unit["dimensions"]["MEA"] = 0.2
    effect = analyzer["coherence_effect"](audited, unit)
    assert effect["overall_delta"] == pytest.approx(0.1)
    assert effect["dimension_deltas"]["MEA"] == pytest.approx(0.1)
    next(cache.iterdir()).unlink()
    partial = analyzer["audit_coherence"](path, cache)
    assert not partial["complete"] and partial["raw_cache_files"] == 1247


def test_unexecuted_plan_stays_eight_not_run_and_no_formal_pass(analyzer):
    report = analyzer["analyze"]()
    assert report["status"] == "incomplete_or_not_run"
    assert report["declared_profiles"] == 8 and report["profile_counts"] == {"not_run": 8}
    assert report["execution_profile_counts"] == {"not_run": 8}
    assert not report["official_raw_replay_complete"]
    assert report["formal_acceptance"] == "not_assessed" and report["rank"] is None
    assert all(row["synthetic"]["status"] == "inconclusive" for row in report["comparisons"])
    assert "Historical Jev-Omni is reported separately" in analyzer["markdown"](report)


def test_failed_stage_and_source_tampering_remain_visible(analyzer, tmp_path):
    protocol_path = ROOT / "configs/experiments/jevbench-breakthrough-20261008.json"
    protocol = json.loads(protocol_path.read_text())
    stage = tmp_path / "ablate"
    sources = stage / "sources"
    sources.mkdir(parents=True)
    names = {
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
    hashes = {}
    for name in names:
        path = sources / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (
            protocol_path.read_bytes()
            if name == "prospective_protocol"
            else b"# simulated failed-stage fixture\n"
        )
        path.write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    saved = {
        "status": "failed",
        "error": "simulated before first model result",
        "stage": "ablate",
        "prospective_protocol": protocol,
        "source_files": hashes,
        "profiles": [],
        "dataset": json.loads(
            (ROOT / protocol["data"]["synthetic"]["directory"] / "manifest.json").read_text()
        ),
        "training_recipe": {
            "steps": 128,
            "seed": 42,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_gradient_accumulation": 8,
            "lora_lr": 1e-5,
            "head_lr": 1e-4,
            "head_batch": 64,
        },
    }
    (stage / "receipt.json").write_text(json.dumps(saved))
    report = analyzer["analyze"](ablate=stage)
    assert report["stages"]["ablate"]["status"] == "failed"
    assert report["stages"]["ablate"]["error"] == saved["error"]
    assert report["profile_counts"] == {"not_run": 8}
    (sources / "campaign.txt").write_text("tampered")
    changed = analyzer["analyze"](ablate=stage)
    assert changed["stages"]["ablate"]["status"] == "integrity_failed"
    assert "hash differs" in changed["stages"]["ablate"]["error"]
    assert not changed["official_raw_replay_complete"]


def test_analysis_output_cannot_overwrite_existing_evidence(analyzer, tmp_path):
    output = tmp_path / "existing.json"
    output.write_text('{"keep":true}')
    with pytest.raises(ValueError, match="overwrite"):
        analyzer["main"](["--output", str(output)])
    assert json.loads(output.read_text()) == {"keep": True}
