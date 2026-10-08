"""Public screening integrity and delegation to the pinned official scorer."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_jevbench_public", ROOT / "scripts/run_jevbench_public.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def make_tasks():
    return [
        SimpleNamespace(
            id="noul-1",
            family="policy",
            split="public",
            group=None,
            state={"document": "public content"},
            question={"type": "noul", "instructions": "Is it allowed?"},
            labels=["no", "yes"],
            expected="yes",
            provenance={},
        ),
        SimpleNamespace(
            id="choice-1",
            family="routing",
            split="public",
            group=None,
            state="route this",
            question={
                "type": "choice",
                "instructions": "Route",
                "criteria": {"B": "second", "A": "first"},
            },
            labels=["B", "A"],
            expected="B",
            provenance={},
        ),
    ]


@pytest.fixture
def fake_official(tmp_path):
    """Minimal test interface, not a substitute benchmark/scoring algorithm."""
    root = tmp_path / "official"
    sources = {
        "jevbench/__init__.py": "MARKER = 'verified fixture package'\n",
        "jevbench/tasks.py": "import json\nfrom types import SimpleNamespace\ndef load_jsonl(path):\n return [SimpleNamespace(**json.loads(x)) for x in open(path) if x.strip()]\ndef dataset_hash(tasks):\n return 'fixture-canonical'\n",
        "jevbench/scoring.py": "CALLS=[]\ndef score_task(probs, task):\n CALLS.append((probs,task.id))\n return {'valid':True,'strict_valid':True,'correct':False,'renormalized':False,'predicted':'fixture','probs':probs}\n",
        "jevbench/metrics.py": "MARKER = 'fixture metrics'\n",
        "jevbench/summarize.py": "def summarize(tasks, records):\n return {'n_valid':sum(r['valid'] for r in records),'complete':len(tasks)==len(records),'n_attempted':len(records),'price_per_1000_decisions_usd':None}\n",
        "jevbench/adapters/base.py": "def build_question(task):\n return dict(task.question)\n",
    }
    for name, source in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    tasks = make_tasks()
    data = root / "datasets/public/original.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text("".join(json.dumps(vars(task)) + "\n" for task in tasks))
    config = {
        "dataset_label": "fixture-only",
        "source_revision": "a" * 40,
        "expected_tasks": len(tasks),
        "canonical_dataset_sha256": "fixture-canonical",
        "ordered_ids_sha256": digest(
            json.dumps([t.id for t in tasks], separators=(",", ":")).encode()
        ),
        "task_files": [
            {
                "path": "datasets/public/original.jsonl",
                "count": len(tasks),
                "sha256": digest(data.read_bytes()),
            }
        ],
        "source_files": {name: digest(source.encode()) for name, source in sources.items()},
        "public_exposure": "public fixture, not sealed",
    }
    return root, config


def test_namespace_isolation_ignores_unrelated_installed_jevbench(fake_official, monkeypatch):
    root, config = fake_official
    unrelated = SimpleNamespace(marker="another publisher's incompatible package")
    monkeypatch.setitem(sys.modules, "jevbench", unrelated)
    tasks, official, _ = RUNNER.load_public(root / "datasets/public", config=config)
    assert [task.id for task in tasks] == ["noul-1", "choice-1"]
    assert sys.modules["jevbench"] is unrelated
    assert official.scoring.__name__.startswith("_s1_public_jevbench_")


@pytest.mark.parametrize("relative", ["jevbench/scoring.py", "datasets/public/original.jsonl"])
def test_mutated_source_or_task_bytes_rejected_before_execution(fake_official, relative):
    root, config = fake_official
    path = root / relative
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        RUNNER.load_public(root, config=config)


def test_id_order_checked_even_when_order_independent_dataset_hash_matches(fake_official):
    root, config = fake_official
    data = root / "datasets/public/original.jsonl"
    lines = data.read_text().splitlines()
    data.write_text("\n".join(reversed(lines)) + "\n")
    config["task_files"][0]["sha256"] = digest(data.read_bytes())
    with pytest.raises(ValueError, match="ordering"):
        RUNNER.load_public(root, config=config)


@pytest.mark.parametrize("change", ["duplicate", "private", "incomplete"])
def test_population_integrity_cannot_be_hidden_by_refreshed_file_hash(fake_official, change):
    root, config = fake_official
    data = root / "datasets/public/original.jsonl"
    rows = [json.loads(x) for x in data.read_text().splitlines()]
    if change == "duplicate":
        rows[1]["id"] = rows[0]["id"]
    elif change == "private":
        rows[1]["split"] = "private"
    else:
        rows.pop()
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config["task_files"][0]["sha256"] = digest(data.read_bytes())
    with pytest.raises(ValueError):
        RUNNER.load_public(root, config=config)


def test_raw_evidence_full_population_and_official_scorer_own_outcomes(
    fake_official, monkeypatch, tmp_path
):
    root, config = fake_official
    tasks, official, _ = RUNNER.load_public(root, config=config)
    monkeypatch.setattr(RUNNER, "load_public", lambda *_: (tasks, official, config))
    requests = []

    def predict(state, questions):
        requests.append(copy.deepcopy({"state": state, "questions": questions}))
        assert set(questions) == {"decision"}
        assert "expected" not in state if isinstance(state, dict) else True
        if questions["decision"]["type"] == "noul":
            state["document"] = "mutated by model callback"
            return {"decision": {"type": "noul", "noul": 0.9}}
        return {
            "answers": {
                "decision": {"type": "choice", "choice": "B", "probabilities": {"B": 0.8, "A": 0.2}}
            }
        }

    output = tmp_path / "run"
    summary = RUNNER.run_public(
        predict, root, output, {"model_id": "fixture", "revision": "b" * 40}
    )
    rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    assert [row["task_id"] for row in rows] == [task.id for task in tasks]
    assert official.scoring.CALLS == [
        ({"yes": 0.9, "no": 1 - 0.9}, "noul-1"),
        ({"B": 0.8, "A": 0.2}, "choice-1"),
    ]
    # Fixture scorer returns false deliberately: the runner must not replace it
    # with its own gold/argmax calculation.
    assert all(row["correct"] is False for row in rows)
    assert all(row["cost_usd"] is None for row in rows)
    raw = json.loads((output / rows[0]["raw_path"]).read_text())
    assert raw["request"] == requests[0]
    assert digest((output / rows[0]["raw_path"]).read_bytes()) == rows[0]["raw_sha256"]
    assert summary["complete"] and summary["official_rank"] is None
    assert summary["sealed_evaluation"] == "not_run"
    manifest = json.loads((output / "manifest.json").read_text())
    for path, expected in manifest["artifact_sha256"].items():
        assert digest((output / path).read_bytes()) == expected


def test_failed_and_malformed_answers_remain_in_population(fake_official, monkeypatch, tmp_path):
    root, config = fake_official
    tasks, official, _ = RUNNER.load_public(root, config=config)
    monkeypatch.setattr(RUNNER, "load_public", lambda *_: (tasks, official, config))

    def predict(state, questions):
        if isinstance(state, dict):
            raise RuntimeError("fixture inference failed")
        return {
            "decision": {"type": "choice", "choice": "unknown", "probabilities": {"unknown": 1.0}}
        }

    output = tmp_path / "failed"
    result = RUNNER.run_public(predict, root, output, {"model_id": "fixture", "revision": "c" * 40})
    rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    assert len(rows) == len(tasks) and result["complete"]
    assert result["n_valid"] == 0
    assert all(row["status"] == "failed" and row["correct"] is False for row in rows)
    assert official.scoring.CALLS == []
    assert json.loads((output / "manifest.json").read_text())["status"] == "completed_with_errors"
    assert (
        json.loads((output / rows[1]["raw_path"]).read_text())["response"]["decision"]["choice"]
        == "unknown"
    )


@pytest.mark.parametrize("answer", [True, float("nan"), -0.1, 1.1])
def test_noul_rejects_boolean_nonfinite_and_out_of_range(answer):
    with pytest.raises(ValueError):
        RUNNER._native_probs({"decision": {"type": "noul", "noul": answer}}, make_tasks()[0])


def test_output_collision_does_not_overwrite_saved_evidence(fake_official, monkeypatch, tmp_path):
    root, config = fake_official
    tasks, official, _ = RUNNER.load_public(root, config=config)
    monkeypatch.setattr(RUNNER, "load_public", lambda *_: (tasks, official, config))
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("existing result")
    with pytest.raises(FileExistsError):
        RUNNER.run_public(
            lambda *_: pytest.fail("must not execute"),
            root,
            output,
            {"model_id": "fixture", "revision": "d" * 40},
        )
    assert marker.read_text() == "existing result"


def test_reported_usage_is_preserved_without_inventing_token_counts_or_cost(
    fake_official, monkeypatch, tmp_path
):
    root, config = fake_official
    tasks, official, _ = RUNNER.load_public(root, config=config)
    monkeypatch.setattr(RUNNER, "load_public", lambda *_: (tasks, official, config))
    usage = {"input_tokens": 42, "output_tokens": 0, "input_tokens_details": {"cached_tokens": 12}}

    def predict(state, questions):
        if isinstance(state, dict):
            return {"answers": {"decision": {"type": "noul", "noul": 0.7}}, "usage": usage}
        return {
            "decision": {"type": "choice", "choice": "B", "probabilities": {"B": 0.7, "A": 0.3}}
        }

    output = tmp_path / "usage"
    summary = RUNNER.run_public(
        predict, root, output, {"model_id": "fixture", "revision": "e" * 40}
    )
    rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    assert rows[0]["usage"] == usage
    assert rows[0]["usage_status"] == "reported_unverified"
    assert rows[1]["usage"] is None and rows[1]["usage_status"] == "not_reported"
    assert all(row["cost_usd"] is None for row in rows)
    assert summary["price_per_1000_decisions_usd"] is None
    preserved, status = RUNNER._reported_usage({"usage": usage})
    preserved["input_tokens_details"]["cached_tokens"] = 999
    assert usage["input_tokens_details"]["cached_tokens"] == 12
    assert status == "reported_unverified"


@pytest.mark.parametrize(
    "usage",
    [
        [],
        {"input_tokens": True},
        {"input_tokens": -1},
        {"total_tokens": 1.5},
        {"other": float("nan")},
    ],
)
def test_invalid_usage_stays_unknown_and_does_not_synthesize_bill(usage):
    assert RUNNER._reported_usage({"usage": usage}) == (None, "invalid_not_used")


def test_official_scorer_exception_preserves_malformed_raw_answer_and_remaining_population(
    fake_official, monkeypatch, tmp_path
):
    root, config = fake_official
    tasks, official, _ = RUNNER.load_public(root, config=config)
    tasks.reverse()  # Put the malformed item first to verify continuation.
    monkeypatch.setattr(RUNNER, "load_public", lambda *_: (tasks, official, config))
    original_score = official.scoring.score_task

    def score(probs, task):
        if task.question["type"] == "choice":
            # The real pinned scorer reports unexpected keys using sorted().
            sorted(set(probs) - set(task.labels))
        return original_score(probs, task)

    monkeypatch.setattr(official.scoring, "score_task", score)

    def predict(state, questions):
        if isinstance(state, str):
            return {
                "decision": {
                    "type": "choice",
                    "choice": "B",
                    "probabilities": {"A": 0.5, "unknown": 0.25, 1: 0.25},
                }
            }
        return {"decision": {"type": "noul", "noul": 0.8}}

    output = tmp_path / "scorer-failure"
    summary = RUNNER.run_public(
        predict, root, output, {"model_id": "fixture", "revision": "f" * 40}
    )
    rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    assert summary["complete"] and len(rows) == 2
    assert rows[0]["status"] == "failed" and rows[0]["correct"] is False
    assert rows[0]["schema_error"] == "official_scorer_exception: TypeError"
    assert rows[0]["error"]["phase"] == "official_scoring"
    assert rows[1]["status"] == "ok" and rows[1]["valid"] is True
    raw = json.loads((output / rows[0]["raw_path"]).read_text())
    assert raw["response_encoding"] == "python_repr_not_json"
    assert "'unknown': 0.25, 1: 0.25" in raw["response"]
    assert json.loads((output / "manifest.json").read_text())["status"] == "completed_with_errors"
