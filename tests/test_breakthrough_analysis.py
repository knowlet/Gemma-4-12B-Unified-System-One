"""Independent soft metrics, full populations and offline analysis failure states."""

import copy
import hashlib
import json
import math
import random
import runpy
import subprocess
import zipfile
from pathlib import Path

import pytest

from s1.contracts import DecisionRequest, answer_from_probabilities
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
    receipt = {"identity": {"calibration": copy.deepcopy(temperatures)}}
    assert analyzer["audit_synthetic"](
        tmp_path, populations["calibration"][0], populations["test"][0], receipt=receipt
    )["complete"]
    receipt["identity"]["calibration"]["choice"]["n"] = 2
    with pytest.raises(ValueError, match="identity/calibration"):
        analyzer["audit_synthetic"](
            tmp_path, populations["calibration"][0], populations["test"][0], receipt=receipt
        )
    temperatures["score"]["temperature"] = math.nextafter(1.0, math.inf)
    (tmp_path / "calibration.json").write_text(
        json.dumps({"records": rows, "temperatures": temperatures})
    )
    replayed = analyzer["audit_synthetic"](
        tmp_path,
        populations["calibration"][0],
        populations["test"][0],
        receipt={"identity": {"calibration": copy.deepcopy(temperatures)}},
    )
    assert replayed["complete"]
    assert replayed["calibration_grid_verification"]["score"]["minimizing_grid_index"] == 40
    assert (
        replayed["calibration_grid_verification"]["score"]["executed_temperature"]
        == temperatures["score"]["temperature"]
    )
    tampered = copy.deepcopy(temperatures)
    tampered["score"]["soft_ce_after"] += 2e-12
    (tmp_path / "calibration.json").write_text(
        json.dumps({"records": rows, "temperatures": tampered})
    )
    with pytest.raises(ValueError, match="saved calibration losses"):
        analyzer["audit_synthetic"](tmp_path, populations["calibration"][0], populations["test"][0])
    temperatures["choice"]["temperature"] = 2.0
    (tmp_path / "calibration.json").write_text(
        json.dumps({"records": rows, "temperatures": temperatures})
    )
    with pytest.raises(ValueError, match="fixed soft-CE grid"):
        analyzer["audit_synthetic"](tmp_path, populations["calibration"][0], populations["test"][0])


def test_calibration_grid_accepts_one_interior_ulp_but_preserves_grid_selection(analyzer):
    import numpy as np

    grid = np.geomspace(0.1, 10.0, 81)
    # The archived x86 score temperature and ARM replay occupy adjacent binary64
    # values at the same interior grid point. The minimizing index remains 32.
    losses = np.square(np.log(grid) - np.log(grid[32]))
    expected = float(grid[32])
    saved = math.nextafter(expected, -math.inf)
    result = analyzer["calibration_grid_match"](saved, grid, losses)
    assert result["minimizing_grid_index"] == 32
    assert result["executed_temperature"] == saved
    assert result["absolute_difference"] <= math.ulp(expected)
    for changed in (
        math.nextafter(saved, -math.inf),
        float(grid[31]),
        float(grid[33]),
        expected * 1.000001,
        float("nan"),
    ):
        with pytest.raises(ValueError, match="fixed soft-CE grid"):
            analyzer["calibration_grid_match"](changed, grid, losses)
    # At a power-of-two boundary, two smaller binary64 steps still fit within
    # math.ulp(1); require adjacent representations as well as the absolute bound.
    losses = np.square(np.log(grid))
    twice = math.nextafter(math.nextafter(1.0, -math.inf), -math.inf)
    with pytest.raises(ValueError, match="fixed soft-CE grid"):
        analyzer["calibration_grid_match"](twice, grid, losses)
    losses = np.arange(len(grid), dtype=float)
    with pytest.raises(ValueError, match="fixed soft-CE grid"):
        analyzer["calibration_grid_match"](math.nextafter(float(grid[0]), math.inf), grid, losses)


def temperature_fixture(request, temperatures):
    torch = pytest.importorskip("torch")
    rows, answers = [], {}
    for question in request.questions:
        logits = torch.arange(len(question.labels()), dtype=torch.float32).tolist()
        rows.append(
            {
                "id": question.id,
                "labels": question.labels(),
                "logits": logits,
                "sequence_tokens": 20,
            }
        )
        values = (torch.tensor(logits) / temperatures[question.type]).softmax(-1).tolist()
        answers[question.id] = answer_from_probabilities(question, values)
    return rows, answers


def linkage_questions(index):
    kind = ("choice", "noul", "score")[index % 3]
    criteria = (
        {"a": "first", "b": "second"}
        if kind == "choice"
        else (["low", "middle", "high"] if kind == "score" else None)
    )
    questions = {"decision": {"type": kind, "instructions": "Select.", "criteria": criteria}}
    if index == 0:
        questions["additional"] = {"type": "noul", "instructions": "Allow?", "criteria": None}
    return questions


@pytest.fixture
def public_temperature_evidence():
    temperatures = {"choice": 2.0, "noul": 0.5, "score": 1.5}
    native = {
        "complete": True,
        "records": [],
        "raw_runtime": {},
        "raw_answers": [],
        "raw_requests": {},
        "manifest": {
            "identity": {
                "calibration": {
                    kind: {"temperature": value, "n": 64} for kind, value in temperatures.items()
                }
            }
        },
    }
    for index in range(231):
        # Public tasks each have exactly one official named decision question.
        payload = {
            "state": str(index),
            "questions": {"decision": linkage_questions(index)["decision"]},
        }
        request = DecisionRequest(**payload)
        rows, answers = temperature_fixture(request, temperatures)
        task_id = f"t{index}"
        native["records"].append({"task_id": task_id, "ok": True})
        native["raw_requests"][task_id] = payload
        native["raw_runtime"][task_id] = {
            "raw_logits": rows,
            "calibration_replay": False,
            "forward_calls": 1,
            "sequential_independent_questions": True,
            "prompt": "user_question",
        }
        native["raw_answers"].append(answers)
    replay = copy.deepcopy(native)
    replay["manifest"]["identity"] = {
        "execution": "raw-logit replay; CPU only",
        "calibration": {kind: 1.0 for kind in temperatures},
    }
    for index, row in enumerate(replay["records"]):
        task_id = row["task_id"]
        replay["raw_runtime"][task_id].update(calibration_replay=True, forward_calls=0)
        _, answers = temperature_fixture(
            DecisionRequest(**replay["raw_requests"][task_id]), {kind: 1.0 for kind in temperatures}
        )
        replay["raw_answers"][index] = answers
    return (
        native,
        replay,
        {"identity": {"calibration": copy.deepcopy(native["manifest"]["identity"]["calibration"])}},
    )


def test_replay_requires_identical_logits_no_forward_and_temperature_probabilities(
    analyzer, public_temperature_evidence
):
    native, replay, _ = public_temperature_evidence
    assert (
        analyzer["public_replay_identity"](native, replay)["status"] == "verified_same_raw_logits"
    )
    replay["raw_runtime"]["t0"]["forward_calls"] = 1
    with pytest.raises(ValueError, match="invoked"):
        analyzer["public_replay_identity"](native, replay)
    replay["raw_runtime"]["t0"]["forward_calls"] = 0
    question = DecisionRequest(**replay["raw_requests"]["t0"]).questions[0]
    replay["raw_answers"][0]["decision"] = answer_from_probabilities(question, [0.5, 0.5])
    with pytest.raises(ValueError, match="probabilities"):
        analyzer["public_replay_identity"](native, replay)


def test_native_public_fitted_temperature_is_independently_reconstructed(
    analyzer, public_temperature_evidence
):
    native, replay, receipt = public_temperature_evidence
    result = analyzer["public_calibration_identity"](native, receipt)
    assert result["complete"] and result["verified_tasks"] == 231
    assert result["max_probability_error"] == 0.0
    question = DecisionRequest(**native["raw_requests"]["t0"]).questions[0]
    native["raw_answers"][0]["decision"] = answer_from_probabilities(question, [0.5, 0.5])
    with pytest.raises(ValueError, match="type temperature/raw logits"):
        analyzer["public_calibration_identity"](native, receipt)
    with pytest.raises(ValueError, match="type temperature/raw logits"):
        analyzer["public_replay_identity"](native, replay)


def test_global_public_replay_reconstructs_every_fixed_temperature_answer(
    analyzer, public_temperature_evidence
):
    native, replay, _ = public_temperature_evidence
    temperatures = {kind: 2.7830344470383452 for kind in ("choice", "noul", "score")}
    replay["manifest"]["identity"]["calibration"] = temperatures
    for index, row in enumerate(replay["records"]):
        _, replay["raw_answers"][index] = temperature_fixture(
            DecisionRequest(**replay["raw_requests"][row["task_id"]]), temperatures
        )
    verified = analyzer["public_replay_identity"](native, replay)
    assert verified["complete"] and verified["n_tasks"] == 231


@pytest.mark.parametrize(
    "changed", ["population", "raw_question", "raw_labels", "fitted_identity", "derived_field"]
)
def test_native_public_temperature_linkage_rejects_self_consistent_scope_tampering(
    analyzer, public_temperature_evidence, changed
):
    native, replay, receipt = public_temperature_evidence
    if changed == "population":
        native["records"].pop()
    elif changed == "raw_question":
        native["raw_runtime"]["t0"]["raw_logits"].append(
            copy.deepcopy(native["raw_runtime"]["t0"]["raw_logits"][0])
        )
    elif changed == "raw_labels":
        native["raw_runtime"]["t0"]["raw_logits"][0]["labels"].reverse()
    elif changed == "fitted_identity":
        receipt["identity"]["calibration"]["choice"]["temperature"] = 1.0
    else:
        native["raw_answers"][0]["decision"]["margin"] = 0.0
    with pytest.raises(
        ValueError, match="231 task|population|ordered labels|fitted temperatures|answer fields"
    ):
        analyzer["public_calibration_identity"](native, receipt)
    if changed == "derived_field":
        with pytest.raises(ValueError, match="answer fields"):
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
    receipt = {
        "coherence": {
            "overall": scores["overall"],
            "interval": None,
            "dimensions": copy.deepcopy(scores["pillars"]),
            "errors": [],
            "scores": copy.deepcopy(scores),
        }
    }
    audited = analyzer["audit_coherence"](path, cache, backend="profile@pin", receipt=receipt)
    assert audited["complete"] and audited["planned_tests"] == 1200
    assert audited["outcome_rows"] == 1468 and audited["raw_cache_files"] == 1248
    assert audited["status"] == "saved_only_not_officially_replayed"
    unit = copy.deepcopy(audited)
    unit["overall"] = 0.4
    unit["dimensions"]["MEA"] = 0.2
    assert (
        analyzer["coherence_effect"](audited, unit)["status"] == "inconclusive_calibration_linkage"
    )
    effect = analyzer["coherence_effect"](
        audited, unit, {"status": "verified_same_raw_logits", "complete": True}
    )
    assert effect["overall_delta"] == pytest.approx(0.1)
    assert effect["dimension_deltas"]["MEA"] == pytest.approx(0.1)
    next(cache.iterdir()).unlink()
    partial = analyzer["audit_coherence"](path, cache)
    assert not partial["complete"] and partial["raw_cache_files"] == 1247


@pytest.fixture
def coherence_temperature_evidence(analyzer, tmp_path, monkeypatch):
    source = tmp_path / "official-fixture"
    package = source / "src/jevbench"
    package.mkdir(parents=True)
    package.joinpath("__init__.py").write_text(
        "from types import SimpleNamespace\n"
        "def from_callable(fn, name=None): return SimpleNamespace(fn=fn, name=name)\n"
        + "def questions(index):\n"
        + "    kind=('choice','noul','score')[index%3]\n"
        + "    criteria={'a':'first','b':'second'} if kind=='choice' else (['low','middle','high'] if kind=='score' else None)\n"
        + "    result={'decision':{'type':kind,'instructions':'Select.','criteria':criteria}}\n"
        + "    if index==0: result['additional']={'type':'noul','instructions':'Allow?','criteria':None}\n"
        + "    return result\n"
        + "def evaluate(model, **kwargs):\n"
        + "    for index in range(1248): model.fn(str(index),questions(index))\n"
        + "    return SimpleNamespace(result={'errors':[]})\n"
    )
    subprocess.run(["git", "init", "-q", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "add", "src"], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Frozen local fixture",
        ],
        check=True,
        capture_output=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setitem(
        analyzer["_freeze_coherence_source"].__globals__, "COHERENCE_REVISION", revision
    )
    temperatures = {"choice": 2.0, "noul": 0.5, "score": 1.5}
    receipt = {
        "identity": {
            "variant": "profile",
            "revision": "pin",
            "calibration": {
                kind: {"temperature": value, "n": 64} for kind, value in temperatures.items()
            },
        }
    }
    raw, policies, keys = {}, {}, {}
    for policy, folder, backend, filename in (
        ("calibrated", "coherence-cache", "profile@pin", "coherence-mini.json"),
        ("unit", "coherence-cache-unit", "profile-unit-replay@pin", "coherence-unit.json"),
    ):
        cache = tmp_path / folder
        cache.mkdir()
        keys[policy] = []
        hashes = {}
        for index in range(1248):
            state, questions = str(index), linkage_questions(index)
            request = DecisionRequest(state=state, questions=questions)
            rows, answers = temperature_fixture(
                request,
                temperatures if policy == "calibrated" else {kind: 1.0 for kind in temperatures},
            )
            raw[hashlib.sha256(request.model_dump_json().encode()).hexdigest()] = rows
            blob = json.dumps(
                {"b": backend, "s": state, "q": questions, "r": 0},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            name = hashlib.sha256(blob.encode()).hexdigest() + ".json"
            path = cache / name
            path.write_text(json.dumps(answers, sort_keys=True))
            keys[policy].append(name)
            hashes[name] = analyzer["digest"](path)
        report = {"settings": analyzer["COHERENCE_SETTINGS"], "result": {"backend": backend}}
        path = tmp_path / filename
        path.write_text(json.dumps(report))
        policies[policy] = {
            "status": "verified_raw_replay",
            "complete": True,
            "sha256": analyzer["digest"](path),
            "raw_cache_sha256": hashes,
            "replay": {
                "ordered_request_fingerprints_sha256": hashlib.sha256(
                    analyzer["_bytes"](keys[policy])
                ).hexdigest()
            },
        }
    (tmp_path / "raw-logits.json").write_text(json.dumps(raw, sort_keys=True))
    return {
        "root": tmp_path,
        "source": source,
        "receipt": receipt,
        "policies": policies,
        "keys": keys,
    }


def run_coherence_linkage(analyzer, evidence):
    return analyzer["audit_coherence_linkage"](
        evidence["root"],
        evidence["policies"]["calibrated"],
        evidence["policies"]["unit"],
        evidence["receipt"],
        source_root=evidence["source"],
    )


def test_coherence_links_complete_batched_question_populations_to_same_raw_logits(
    analyzer, coherence_temperature_evidence
):
    result = run_coherence_linkage(analyzer, coherence_temperature_evidence)
    assert result["complete"] and result["status"] == "verified_same_raw_logits"
    assert result["verified_requests_per_policy"] == 1248
    assert result["verified_question_records_per_policy"] == 1249
    assert result["max_probability_error"] == 0.0
    assert result["temperatures"]["calibrated"]["noul"] == 0.5
    assert result["temperatures"]["unit"] == {"choice": 1.0, "noul": 1.0, "score": 1.0}


@pytest.mark.parametrize(
    "changed",
    [
        "missing_raw",
        "raw_values",
        "raw_labels",
        "unit_answer",
        "missing_unit",
        "extra_unit",
        "fitted_temperature",
        "wrong_request_order",
    ],
)
def test_coherence_linkage_rejects_complete_and_partial_self_consistent_tampering(
    analyzer, coherence_temperature_evidence, changed
):
    evidence = coherence_temperature_evidence
    root, policies = evidence["root"], evidence["policies"]
    raw_path = root / "raw-logits.json"
    raw = json.loads(raw_path.read_text())
    request = DecisionRequest(state="0", questions=linkage_questions(0))
    raw_key = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
    if changed == "missing_raw":
        del raw[raw_key]
    elif changed == "raw_values":
        raw[raw_key][0]["logits"] = [0.0, 0.0]
    elif changed == "raw_labels":
        raw[raw_key][0]["labels"].reverse()
    elif changed in ("unit_answer", "missing_unit", "extra_unit"):
        name = evidence["keys"]["unit"][0]
        path = root / "coherence-cache-unit" / name
        if changed == "unit_answer":
            answers = json.loads(path.read_text())
            answers["decision"] = answer_from_probabilities(request.questions[0], [0.5, 0.5])
            path.write_text(json.dumps(answers))
            policies["unit"]["raw_cache_sha256"][name] = analyzer["digest"](path)
        elif changed == "missing_unit":
            path.unlink()
            del policies["unit"]["raw_cache_sha256"][name]
        else:
            path = root / "coherence-cache-unit" / ("0" * 64 + ".json")
            path.write_text("{}")
            policies["unit"]["raw_cache_sha256"][path.name] = analyzer["digest"](path)
    elif changed == "fitted_temperature":
        evidence["receipt"]["identity"]["calibration"]["choice"]["temperature"] = 1.0
    else:
        policies["unit"]["replay"]["ordered_request_fingerprints_sha256"] = "0" * 64
    raw_path.write_text(json.dumps(raw, sort_keys=True))
    with pytest.raises(
        ValueError,
        match="lacks raw logits|temperature/raw logits|ordered labels|full 1248|order differs",
    ):
        run_coherence_linkage(analyzer, evidence)


def test_coherence_without_frozen_logit_evidence_is_explicitly_unverified(analyzer, tmp_path):
    complete = {"status": "verified_raw_replay", "complete": True}
    result = analyzer["audit_coherence_linkage"](tmp_path, complete, complete, {}, source_root=ROOT)
    assert result["status"] == "calibration_linkage_unverified" and not result["complete"]
    assert result["expected_requests"] == 1248
    assert "No temperature attribution" in result["scope"]


@pytest.fixture
def pinned_cpu_temperature_evidence(analyzer, coherence_temperature_evidence):
    evidence = coherence_temperature_evidence
    raw = json.loads((evidence["root"] / "raw-logits.json").read_text())
    fitted = {
        k: v["temperature"] for k, v in evidence["receipt"]["identity"]["calibration"].items()
    }
    output = {"coherence": {}}
    for policy, folder, backend in (
        ("calibrated", "coherence-cache", "profile@pin"),
        ("unit", "coherence-cache-unit", "profile-unit-replay@pin"),
    ):
        ts = fitted if policy == "calibrated" else {k: 1.0 for k in fitted}
        rows, raw_keys, questions, probabilities = [], [], 0, 0
        for index, name in enumerate(evidence["keys"][policy]):
            state, request_questions = str(index), linkage_questions(index)
            request = DecisionRequest(state=state, questions=request_questions)
            key = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
            answers = json.loads((evidence["root"] / folder / name).read_text())
            metadata = []
            for question in request.questions:
                values = [
                    answers[question.id]["probabilities"][label] for label in question.labels()
                ]
                metadata.append(
                    {
                        "id": question.id,
                        "type": question.type,
                        "labels": question.labels(),
                        "temperature": ts[question.type],
                        "expected_probabilities": values,
                        "expected_answer": answers[question.id],
                    }
                )
                questions += 1
                probabilities += len(values)
            raw_keys.append(key)
            rows.append(
                {
                    "raw_key": key,
                    "cache_file": name,
                    "state": state,
                    "request_questions": request_questions,
                    "request": {"state": state, "questions": request_questions},
                    "raw_rows": raw[key],
                    "questions": metadata,
                    "expected_answers": answers,
                }
            )
        output["coherence"][policy] = {
            "backend": backend,
            "cache_sha256": evidence["policies"][policy]["raw_cache_sha256"],
            "ordered_cache_files": evidence["keys"][policy],
            "ordered_cache_sha256": hashlib.sha256(
                analyzer["arithmetic_bytes"](evidence["keys"][policy])
            ).hexdigest(),
            "ordered_raw_keys_sha256": hashlib.sha256(
                analyzer["arithmetic_bytes"](raw_keys)
            ).hexdigest(),
            "requests": rows,
            "questions": questions,
            "probabilities": probabilities,
            "max_abs_error": 0.0,
            "mismatch_count": 0,
            "native_field_mismatch_count": 0,
        }
    return evidence, output


def run_pinned_coherence(analyzer, evidence, proof):
    return analyzer["audit_coherence_linkage"](
        evidence["root"],
        evidence["policies"]["calibrated"],
        evidence["policies"]["unit"],
        evidence["receipt"],
        source_root=evidence["source"],
        cpu_evidence=proof,
    )


def test_pinned_cpu_full_vectors_do_not_relax_default_local_arithmetic(
    analyzer, pinned_cpu_temperature_evidence, monkeypatch
):
    evidence, proof = pinned_cpu_temperature_evidence

    def unavailable(*args, **kwargs):
        raise ValueError("strict local numeric mismatch remains visible")

    monkeypatch.setitem(
        analyzer["audit_coherence_linkage"].__globals__, "verify_temperature_answers", unavailable
    )
    with pytest.raises(ValueError, match="strict local numeric mismatch"):
        run_coherence_linkage(analyzer, evidence)
    checked = run_pinned_coherence(analyzer, evidence, proof)
    assert checked["complete"] and checked["status"] == "verified_same_raw_logits_pinned_cpu"
    assert checked["verified_requests_per_policy"] == 1248
    assert checked["verified_question_records_per_policy"] == 1249
    assert checked["pinned_cpu_comparison"]["calibrated"]["mismatch_count"] == 0
    for policy in proof["coherence"].values():
        policy["requests"] = [
            {k: r[k] for k in ("raw_key", "cache_file", "expected_answers")}
            for r in policy["requests"]
        ]
    minimal = run_pinned_coherence(analyzer, evidence, proof)
    assert (
        minimal["complete"] and minimal["pinned_cpu_comparison"] == checked["pinned_cpu_comparison"]
    )
    proof["coherence"]["unit"]["requests"][0]["raw_key"] = "0" * 64
    with pytest.raises(ValueError, match="ordered request/cache/raw identities"):
        run_pinned_coherence(analyzer, evidence, proof)


@pytest.mark.parametrize(
    "changed", ["population", "order", "raw", "temperature", "vector", "field", "counter"]
)
def test_pinned_cpu_requires_full_bound_vectors_and_native_fields(
    analyzer, pinned_cpu_temperature_evidence, changed
):
    evidence, proof = pinned_cpu_temperature_evidence
    policy = proof["coherence"]["calibrated"]
    row = policy["requests"][0]
    if changed == "population":
        policy["requests"].pop()
    elif changed == "order":
        policy["requests"][:2] = reversed(policy["requests"][:2])
    elif changed == "raw":
        row["raw_rows"][0]["logits"][0] += 0.001
    elif changed == "temperature":
        row["questions"][0]["temperature"] = 1.0
    elif changed == "vector":
        request = DecisionRequest(state=row["state"], questions=row["request_questions"])
        answer = answer_from_probabilities(request.questions[0], [0.5, 0.5])
        row["expected_answers"]["decision"] = answer
        row["questions"][0]["expected_answer"] = answer
        row["questions"][0]["expected_probabilities"] = [0.5, 0.5]
    elif changed == "field":
        row["expected_answers"]["decision"]["margin"] = 0.0
    else:
        policy["questions"] -= 1
    with pytest.raises(ValueError, match="CPU proof"):
        run_pinned_coherence(analyzer, evidence, proof)


def test_cpu_proof_review_digests_cannot_be_inferred_from_certificate(analyzer, tmp_path):
    proof = tmp_path / "proof.json"
    proof.write_text("{}")
    with pytest.raises(ValueError, match="reviewed output/source SHA256"):
        analyzer["load_arithmetic_proof"](proof, None, None, {}, {})
    with pytest.raises(ValueError, match="reviewed output/source SHA256"):
        analyzer["load_arithmetic_proof"](proof, "0" * 64, "0" * 64, {}, {})
    source = tmp_path / "verifier.py.txt"
    source.write_text('print("bounded verifier")')
    with pytest.raises(ValueError, match="verifier bytes differ"):
        analyzer["load_arithmetic_proof"](proof, analyzer["digest"](proof), "0" * 64, {}, {})


def test_cpu_proof_binds_all_inputs_environment_and_child_identity(analyzer, tmp_path):
    # A complete eight-profile byte inventory, without any model/media loading.
    roots, stages, members, sources, profiles = {}, [], {}, {}, {}
    for stage_index, stage in enumerate(("ablate", "train")):
        root = tmp_path / stage
        roots[stage] = root
        (root / "sources/s1").mkdir(parents=True)
        source = root / "sources/s1/contracts.py.txt"
        source.write_text("frozen fixture contract source")
        sources[stage] = {"s1/contracts.py": analyzer["digest"](source)}
        receipt = {
            "status": "completed",
            "stage": stage,
            "run": "fixed-run",
            "versions": {"torch": "2.10.0", "numpy": "2.5.3"},
            "source_files": sources[stage],
        }
        (root / "receipt.json").write_text(json.dumps(receipt))
        declarations = []
        for index in range(4):
            name = f"profile-{4 * stage_index + index}"
            declarations.append({"id": name})
            directory = root / name
            directory.mkdir()
            identity = {"variant": name, "calibration": {"choice": {"temperature": 1.0}}}
            for filename in (
                "receipt.json",
                "calibration.json",
                "raw-logits.json",
                "coherence-mini.json",
                "coherence-unit.json",
            ):
                (directory / filename).write_text(
                    json.dumps({"identity": identity}) if filename == "receipt.json" else "{}"
                )
            for folder in ("coherence-cache", "coherence-cache-unit"):
                cache = directory / folder
                cache.mkdir()
                for case_index in range(1248):
                    (
                        cache / (hashlib.sha256(str(case_index).encode()).hexdigest() + ".json")
                    ).write_text("{}")
            profiles[name] = {"stage": stage, "identity": identity, "public": {}}
        stages.append({"id": stage, "profiles": declarations})
        for path in sorted(root.rglob("*")):
            if path.is_file():
                members[f"{stage}/{path.relative_to(root).as_posix()}"] = {
                    "sha256": analyzer["digest"](path),
                    "size_bytes": path.stat().st_size,
                }
    members = dict(sorted(members.items()))
    for name, profile in profiles.items():
        prefix = f"{profile['stage']}/{name}/"
        profile["file_sha256"] = {
            key.removeprefix(prefix): value["sha256"]
            for key, value in members.items()
            if key.startswith(prefix)
        }
    folder = tmp_path / "proof"
    folder.mkdir()
    verifier = folder / "verifier.py.txt"
    verifier.write_text("reviewed CPU-only verifier fixture")
    verifier_sha = analyzer["digest"](verifier)
    manifest = folder / "input-manifest.json"
    manifest.write_bytes(analyzer["arithmetic_bytes"](members))
    (folder / "modal-app.json").write_text(
        json.dumps(
            {
                "app_id": "ap-fixture",
                "function_id": "fu-fixture",
                "image_id": "im-hWNAPF8eRYjs4tA8VG582u",
                "verifier_source_sha256": verifier_sha,
            }
        )
    )
    proof = {
        "schema_version": 1,
        "status": "completed",
        "run": "fixed-run",
        "verifier_source_sha256": verifier_sha,
        "original_image_id": "im-hWNAPF8eRYjs4tA8VG582u",
        "source_files": sources,
        "input_members": members,
        "input_manifest_sha256": analyzer["digest"](manifest),
        "environment": {
            "python": "3.12.10",
            "torch": "2.10.0",
            "numpy": "2.5.3",
            "machine": "x86_64",
            "platform": "Linux-fixture",
            "visible_gpu_count": 0,
        },
        "execution": {
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
        },
        "public_replay": "not_replayed; already independently strict-verified locally",
        "profiles": profiles,
    }
    path = folder / "proof.json"

    def verify(value):
        path.write_bytes(analyzer["arithmetic_bytes"](value))
        return analyzer["load_arithmetic_proof"](
            path, analyzer["digest"](path), verifier_sha, roots, {"stages": stages}
        )

    result = verify(proof)
    assert result["metadata"]["input_files"] == 20012
    changed = copy.deepcopy(proof)
    changed["environment"]["torch"] = "2.8.0"
    with pytest.raises(ValueError, match="pinned environment"):
        verify(changed)
    changed = copy.deepcopy(proof)
    changed["profiles"]["profile-0"]["identity"] = changed["profiles"]["profile-1"]["identity"]
    with pytest.raises(ValueError, match="child identity differs"):
        verify(changed)
    (roots["train"] / "profile-7/raw-logits.json").write_text('{"altered":true}')
    with pytest.raises(ValueError, match="input byte inventory differs"):
        verify(proof)


def test_cpu_calibration_proof_replays_full_grid_and_exact_executed_metadata(analyzer, tmp_path):
    import numpy as np

    rows = []
    for kind in ("choice", "noul", "score"):
        for index in range(64):
            r = row(
                case=f"{kind}-{index}",
                kind=kind,
                probs=[0.2, 0.8] if kind != "score" else [0.2, 0.2, 0.6],
            )
            r["split"] = "calibration"
            rows.append(r)
    grid = np.geomspace(0.1, 10.0, 81).tolist()
    types, temperatures = {}, {}
    for kind in ("choice", "noul", "score"):
        selected = [r for r in rows if r["type"] == kind]
        losses = [
            float(np.mean([analyzer["row_metrics"](r, t)["soft_ce"] for r in selected]))
            for t in grid
        ]
        index = int(np.argmin(losses))
        saved = {
            "n": 64,
            "temperature": grid[index],
            "boundary": index in (0, 80),
            "soft_ce_before": losses[40],
            "soft_ce_after": losses[index],
        }
        temperatures[kind] = saved
        types[kind] = {
            "n": 64,
            "argmin_index": index,
            "losses": losses,
            "expected_metadata": copy.deepcopy(saved),
            "saved_metadata": copy.deepcopy(saved),
            "exact_metadata_match": True,
        }
    (tmp_path / "calibration.json").write_text(
        json.dumps({"records": rows, "temperatures": temperatures})
    )
    proof = {
        "calibration": {
            "cases": 192,
            "record_count": 192,
            "records": copy.deepcopy(rows),
            "grid": grid,
            "types": types,
            "record_identity_order_sha256": hashlib.sha256(
                analyzer["arithmetic_bytes"](
                    [[r["case_id"], r["question_id"], r["type"], r["labels"]] for r in rows]
                )
            ).hexdigest(),
        }
    }
    receipt = {"identity": {"calibration": temperatures}}
    assert analyzer["audit_pinned_cpu_calibration"](tmp_path, proof, receipt)["complete"]
    for changed in ("record_order", "grid_loss", "selected_index", "expected_metadata"):
        altered = copy.deepcopy(proof)
        cal = altered["calibration"]
        if changed == "record_order":
            cal["records"][:2] = reversed(cal["records"][:2])
        elif changed == "grid_loss":
            cal["types"]["score"]["losses"][32] += 2e-12
        elif changed == "selected_index":
            cal["types"]["score"]["argmin_index"] += 1
        else:
            cal["types"]["score"]["expected_metadata"]["temperature"] = grid[41]
        with pytest.raises(ValueError, match="CPU proof"):
            analyzer["audit_pinned_cpu_calibration"](tmp_path, altered, receipt)


@pytest.mark.parametrize("policy", ["calibrated", "unit", "published_global"])
def test_public_identity_binds_child_and_only_fixed_replay_extensions(analyzer, policy):
    protocol = json.loads(
        (ROOT / "configs/experiments/jevbench-breakthrough-20261008.json").read_text()
    )
    identity = {
        "model_id": "released-model",
        "revision": "a" * 40,
        "variant": "released-current",
        "gpu": "recorded board",
        "source_files": {"campaign": "frozen-source"},
        "processor_files": {"tokenizer.json": "frozen-tokenizer"},
        "calibration": {
            kind: {"temperature": 2.0, "n": 64} for kind in ("choice", "noul", "score")
        },
        "execution": "native independent",
    }
    summary = {"identity": copy.deepcopy(identity), "n_correct": 197, "n_planned": 231}
    receipt = {"identity": copy.deepcopy(identity), "public": copy.deepcopy(summary)}
    expected = copy.deepcopy(identity)
    if policy != "calibrated":
        expected.update(
            variant=f"released-current-{policy}",
            calibration={
                kind: protocol["calibration"][f"{policy}_temperature"]
                for kind in ("choice", "noul", "score")
            },
            execution=analyzer["PUBLIC_REPLAY_EXECUTION"],
        )
    result = {
        "manifest": {"identity": expected},
        "summary": {**summary, "identity": copy.deepcopy(expected)},
    }
    assert (
        analyzer["audit_public_identity"](result, receipt, policy, protocol)["status"]
        == "verified_child_identity"
    )
    # A complete, internally consistent bundle for another checkpoint/profile
    # must still fail when placed under this child's directory.
    swapped = copy.deepcopy(result)
    swapped["manifest"]["identity"]["revision"] = "b" * 40
    swapped["summary"]["identity"] = swapped["manifest"]["identity"]
    with pytest.raises(ValueError, match="receipt/public manifest"):
        analyzer["audit_public_identity"](swapped, receipt, policy, protocol)
    for key, altered in (
        ("gpu", "other-board"),
        ("source_files", {"campaign": "other-source"}),
        ("calibration", {"choice": 99.0}),
        ("variant", "other-profile"),
        ("execution", "other execution scope"),
    ):
        changed = copy.deepcopy(result)
        changed["manifest"]["identity"][key] = altered
        with pytest.raises(ValueError, match="receipt/public manifest"):
            analyzer["audit_public_identity"](changed, receipt, policy, protocol)
    if policy == "calibrated":
        receipt["public"]["n_correct"] = 198
        with pytest.raises(ValueError, match="receipt/public summary"):
            analyzer["audit_public_identity"](result, receipt, policy, protocol)


@pytest.mark.parametrize("field", ["overall", "interval", "dimensions", "errors", "scores"])
def test_coherence_child_receipt_cannot_disagree_with_saved_official_report(
    analyzer, tmp_path, field
):
    scores = {"overall": 0.75, "ci": {"overall": [0.7, 0.8]}, "pillars": {"LOG": 1.0, "MEA": 0.5}}
    report = {"scores": scores, "result": {"backend": "profile@revision", "errors": []}}
    path = tmp_path / "coherence.json"
    path.write_text(json.dumps(report))
    receipt = {
        "coherence": {
            "overall": 0.75,
            "interval": [0.7, 0.8],
            "dimensions": {"LOG": 1.0, "MEA": 0.5},
            "errors": [],
            "scores": copy.deepcopy(scores),
        }
    }
    receipt["coherence"][field] = {"tampered": True}
    with pytest.raises(ValueError, match="coherence receipt/report"):
        analyzer["audit_coherence"](path, tmp_path / "cache", receipt=receipt)


@pytest.fixture(scope="module")
def training_cases():
    return [case(f"train-{index}", split="train") for index in range(1024)]


@pytest.fixture
def head_evidence(tmp_path, training_cases):
    torch = pytest.importorskip("torch")
    rows = []
    for source in training_cases:
        question = source.request.questions[0]
        rows.append(
            {
                "case_id": source.id,
                "group_id": source.group_id,
                "split": "train",
                "type": question.type,
                "question_id": question.id,
                "labels": question.labels(),
                "target": [source.soft_gold[question.id][label] for label in question.labels()],
                "score_values": None,
                "logits": [0.0, 0.0],
                "sequence_tokens": 12,
            }
        )
    torch.save({"features": torch.zeros(1024, 4), "records": rows}, tmp_path / "train-features.pt")
    head = tmp_path / "decision-head.pt"
    torch.save({"weight": torch.zeros(52, 4), "bias": torch.zeros(52)}, head)
    receipt = {"identity": {"research_head_sha256": hashlib.sha256(head.read_bytes()).hexdigest()}}
    training = {
        "steps": 128,
        "seed": 42,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "losses": [0.5] * 128,
    }
    (tmp_path / "head-training.json").write_text(json.dumps(training))
    return receipt


def test_trained_head_requires_downloaded_binary_and_matching_hash(
    analyzer, tmp_path, training_cases, head_evidence
):
    receipt = head_evidence
    head = tmp_path / "decision-head.pt"
    payload = head.read_bytes()
    head.unlink()
    missing = analyzer["audit_training"](tmp_path, "head", training_cases, receipt)
    assert missing["recipe_verified"] and missing["binary_status"] == "not_downloaded"
    assert missing["status"] == "incomplete_missing_binary" and not missing["complete"]
    head.write_bytes(payload)
    complete = analyzer["audit_training"](tmp_path, "head", training_cases, receipt)
    assert complete["complete"] and complete["binary_status"] == "verified"
    assert complete["features"]["frozen_rows_verified"] == 1024
    assert complete["final_tensors"]["weight_shape"] == [52, 4]
    assert complete["training_data"]["cases"] == 1024
    assert complete["finite_loss_updates"] == 128
    assert complete["gradient_trace"]["status"] == "unrecorded"
    assert complete["sampled_batch_ids"]["status"] == "unrecorded"
    head.write_bytes(payload + b"changed")
    with pytest.raises(ValueError, match="binary hash"):
        analyzer["audit_training"](tmp_path, "head", training_cases, receipt)


def test_head_feature_binary_is_required_for_complete_training_audit(
    analyzer, tmp_path, training_cases, head_evidence
):
    (tmp_path / "train-features.pt").unlink()
    result = analyzer["audit_training"](tmp_path, "head", training_cases, head_evidence)
    assert result["binary_status"] == "verified"
    assert result["features"]["status"] == "not_downloaded" and not result["complete"]


@pytest.mark.parametrize("changed", ["group", "target", "order", "nonfinite", "precision", "shape"])
def test_head_feature_source_and_tensor_tampering_is_rejected(
    analyzer, tmp_path, training_cases, head_evidence, changed
):
    torch = pytest.importorskip("torch")
    path = tmp_path / "train-features.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if changed == "group":
        saved["records"][0]["group_id"] = "calibration-source"
    elif changed == "target":
        saved["records"][0]["target"] = [0.25, 0.75]
    elif changed == "order":
        saved["records"][0], saved["records"][1] = saved["records"][1], saved["records"][0]
    elif changed == "nonfinite":
        saved["features"][0, 0] = float("nan")
    elif changed == "precision":
        saved["features"] = saved["features"].half()
    else:
        saved["features"] = torch.zeros(1024, 5)
    torch.save(saved, path)
    with pytest.raises(ValueError, match="frozen data|row order|finite CPU FP32|shape differs"):
        analyzer["audit_training"](tmp_path, "head", training_cases, head_evidence)


@pytest.mark.parametrize("changed", ["nonfinite", "precision", "shape"])
def test_self_consistent_head_hash_cannot_hide_invalid_final_tensors(
    analyzer, tmp_path, training_cases, head_evidence, changed
):
    torch = pytest.importorskip("torch")
    path = tmp_path / "decision-head.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if changed == "nonfinite":
        saved["bias"][0] = float("inf")
    elif changed == "precision":
        saved["weight"] = saved["weight"].half()
    else:
        saved["weight"] = torch.zeros(51, 4)
    torch.save(saved, path)
    head_evidence["identity"]["research_head_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="finite CPU FP32|52 bounded"):
        analyzer["audit_training"](tmp_path, "head", training_cases, head_evidence)


def test_research_tensor_read_is_bounded_before_cpu_allocation(analyzer, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    path = tmp_path / "compressed.pt"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        stream.writestr("storage", b"x" * 10000)
    assert path.stat().st_size < 1024
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: pytest.fail("must reject before load")
    )
    with pytest.raises(ValueError, match="uncompressed size"):
        analyzer["load_research_tensors"](path, max_bytes=1024)


@pytest.fixture
def lora_evidence(tmp_path, training_cases):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    order = list(training_cases)
    random.Random(42).shuffle(order)
    updates = []
    for index in range(128):
        ids = [source.id for source in order[index * 8 : (index + 1) * 8]]
        updates.append(
            {
                "step": index + 1,
                "case_id": ids[-1],
                "case_ids": ids,
                "loss": 0.5,
                "gradient_norm": 2.0,
            }
        )
    directory = tmp_path / "mixed-lora"
    adapter = directory / "adapter"
    adapter.mkdir(parents=True)
    tensors = {}
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        module = f"base_model.model.language_model.layers.0.self_attn.{projection}"
        tensors[f"{module}.lora_A.weight"] = torch.zeros(8, 4)
        tensors[f"{module}.lora_B.weight"] = torch.zeros(4, 8)
    save_file(tensors, str(adapter / "adapter_model.safetensors"))
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "bias": "none",
                "lora_dropout": 0.0,
            }
        )
    )
    declared = {
        file.name: {
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "bytes": file.stat().st_size,
        }
        for file in adapter.iterdir()
    }
    saved = {
        "status": "completed",
        "steps": 128,
        "seed": 42,
        "gradient_accumulation": 8,
        "learning_rate": 1e-5,
        "n_training_calls": 1024,
        "n_unique_training_cases": 1024,
        "trainable_parameters": sum(tensor.numel() for tensor in tensors.values()),
        "losses": updates,
        "adapter_files": declared,
    }
    (directory / "training.json").write_text(json.dumps(saved))
    (directory / "updates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in updates))
    receipt = {"identity": {"research_adapter_files": copy.deepcopy(declared)}}
    return saved, receipt


def test_lora_requires_final_weight_and_config_and_child_inventory_alignment(
    analyzer, tmp_path, training_cases, lora_evidence
):
    directory = tmp_path / "mixed-lora"
    adapter = directory / "adapter"
    saved, receipt = lora_evidence
    file = adapter / "adapter_model.safetensors"
    payload = file.read_bytes()
    file.unlink()
    missing = analyzer["audit_training"](tmp_path, "attention_lora", training_cases, receipt)
    assert not missing["complete"] and missing["status"] == "incomplete_missing_binary"
    assert missing["adapter_files"]["adapter_model.safetensors"]["status"] == "not_downloaded"
    file.write_bytes(payload)
    result = analyzer["audit_training"](tmp_path, "attention_lora", training_cases, receipt)
    assert result["complete"]
    assert result["gradient_trace"]["recorded_updates"] == 128
    assert result["gradient_trace"]["maximum"] == 2.0  # Norm is recorded before clipping.
    assert result["training_data"]["cases"] == 1024
    assert result["final_tensors"]["parameters"] == saved["trainable_parameters"]
    receipt["identity"]["research_adapter_files"]["adapter_config.json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity/final adapter"):
        analyzer["audit_training"](tmp_path, "attention_lora", training_cases, receipt)


@pytest.mark.parametrize(
    "changed",
    ["norm_missing", "norm_nan", "norm_negative", "trace", "order", "last_case", "repeat"],
)
def test_lora_update_trace_and_exact_train_coverage_are_verified(
    analyzer, tmp_path, training_cases, lora_evidence, changed
):
    saved, receipt = lora_evidence
    updates = copy.deepcopy(saved["losses"])
    if changed == "norm_missing":
        del updates[0]["gradient_norm"]
    elif changed == "norm_nan":
        updates[0]["gradient_norm"] = float("nan")
    elif changed == "norm_negative":
        updates[0]["gradient_norm"] = -1.0
    elif changed == "trace":
        updates[0]["loss"] = 0.6
    elif changed == "order":
        updates[0]["case_ids"][0], updates[0]["case_ids"][1] = (
            updates[0]["case_ids"][1],
            updates[0]["case_ids"][0],
        )
    elif changed == "last_case":
        updates[0]["case_id"] = "unrelated"
    else:
        updates[0]["case_ids"][0] = updates[0]["case_ids"][1]
    if changed != "trace":
        saved["losses"] = updates
    directory = tmp_path / "mixed-lora"
    (directory / "training.json").write_text(json.dumps(saved))
    (directory / "updates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in updates))
    with pytest.raises(ValueError, match="norms|loss trace|micro-batch|exactly once"):
        analyzer["audit_training"](tmp_path, "attention_lora", training_cases, receipt)


@pytest.mark.parametrize(
    "changed", ["nonfinite", "precision", "rank", "pair", "extra", "config", "parameters"]
)
def test_rehashed_lora_files_cannot_hide_invalid_final_checkpoint(
    analyzer, tmp_path, training_cases, lora_evidence, changed
):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    saved, receipt = lora_evidence
    directory = tmp_path / "mixed-lora"
    adapter = directory / "adapter"
    path = adapter / "adapter_model.safetensors"
    tensors = safetensors.load_file(str(path))
    key = next(name for name in tensors if "lora_A" in name)
    if changed == "nonfinite":
        tensors[key][0, 0] = float("nan")
    elif changed == "precision":
        tensors[key] = tensors[key].half()
    elif changed == "rank":
        tensors[key] = torch.zeros(7, 4)
    elif changed == "pair":
        del tensors[key]
    elif changed == "extra":
        tensors["unrelated.weight"] = torch.zeros(1)
    elif changed == "config":
        config = json.loads((adapter / "adapter_config.json").read_text())
        config["lora_alpha"] = 32
        (adapter / "adapter_config.json").write_text(json.dumps(config))
    else:
        saved["trainable_parameters"] += 1
    safetensors.save_file(tensors, str(path))
    for file in adapter.iterdir():
        saved["adapter_files"][file.name] = {
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "bytes": file.stat().st_size,
        }
    receipt["identity"]["research_adapter_files"] = copy.deepcopy(saved["adapter_files"])
    (directory / "training.json").write_text(json.dumps(saved))
    with pytest.raises(
        ValueError,
        match="finite CPU FP32|FP32|rank/shape|A/B pairs|non-attention|config differs|parameter count",
    ):
        analyzer["audit_training"](tmp_path, "attention_lora", training_cases, receipt)


def test_final_tensor_inspection_accepts_actual_local_peft_export_with_unused_media(
    analyzer, tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")

    class TinyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self, name, torch.nn.Linear(4, 4, bias=False))

        def forward(self, values):
            return self.o_proj(self.v_proj(values))

    class TinyNative(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = transformers.PretrainedConfig()
            self.language_model = TinyAttention()
            self.vision_tower = TinyAttention()

        def forward(self, values):
            return self.language_model(values)

    model = peft.get_peft_model(
        TinyNative(),
        peft.LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none"
        ),
    )
    model(torch.ones(1, 4)).sum().backward()
    assert all(parameter.grad is None for parameter in model.vision_tower.parameters())
    model.save_pretrained(tmp_path)
    parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    result = analyzer["inspect_adapter_binary"](tmp_path, parameters)
    assert result["status"] == "verified" and result["tensors"] == 16
    assert result["parameters"] == parameters
    assert any("vision_tower" in name for name in result["tensor_shapes"])


@pytest.mark.parametrize("gap", ["trained_binary", "public_calibration", "coherence_calibration"])
def test_profile_requires_binary_and_complete_native_temperature_linkages(
    analyzer, monkeypatch, tmp_path, gap
):
    directory = tmp_path / "trained"
    directory.mkdir()
    (directory / "receipt.json").write_text(json.dumps({"identity": {"revision": "a" * 40}}))
    namespace = analyzer["analyze_profile"].__globals__
    for name in (
        "audit_synthetic",
        "audit_public_profile",
        "audit_regression_records",
        "audit_native",
        "audit_coherence",
    ):
        monkeypatch.setitem(namespace, name, lambda *args, **kwargs: {"complete": True})
    monkeypatch.setitem(namespace, "audit_identity", lambda *args: {"status": "metadata_verified"})
    monkeypatch.setitem(
        namespace, "public_replay_identity", lambda *args: {"status": "verified_same_raw_logits"}
    )
    monkeypatch.setitem(
        namespace,
        "public_calibration_identity",
        lambda *args: {"complete": gap != "public_calibration"},
    )
    monkeypatch.setitem(
        namespace,
        "audit_coherence_linkage",
        lambda *args, **kwargs: {
            "complete": gap != "coherence_calibration",
            "status": "verified_same_raw_logits"
            if gap != "coherence_calibration"
            else "calibration_linkage_unverified",
        },
    )
    monkeypatch.setitem(
        namespace,
        "audit_training",
        lambda *args: {
            "status": "incomplete_missing_binary" if gap == "trained_binary" else "recipe_verified",
            "recipe_verified": True,
            "complete": gap != "trained_binary",
            "binary_status": "not_downloaded" if gap == "trained_binary" else "verified",
        },
    )
    stage = {
        "status": "completed",
        "source_hashes_verified": True,
        "declared_completed_profiles": ["trained"],
    }
    data = {"calibration": [], "test": [], "train": [], "release": {}, "native": [], "protocol": {}}
    result = analyzer["analyze_profile"](
        tmp_path, {"id": "trained", "training": "head"}, stage, data, None, None
    )
    assert result["execution_status"] == "completed"
    assert not result["complete"] and result["status"] == "incomplete_or_failed"
    if gap == "trained_binary":
        assert result["training"]["binary_status"] == "not_downloaded"
    else:
        assert result[f"{gap}_linkage"]["complete"] is False


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
