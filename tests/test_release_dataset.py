"""Release data leakage/determinism checks without network or model inference."""

import json
import runpy
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def script():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/prepare_release_dataset.py")
    )


def rows(count=20):
    return [
        {"passage": f"passage-{i}", "question": f"question-{i}", "answer": bool(i % 2)}
        for i in range(count)
    ]


def test_selection_is_balanced_disjoint_and_deterministic(script):
    source = rows()
    source.append({**source[0], "question": "same passage another question"})
    excluded = {script["passage_group"](source[1]["passage"])}
    kwargs = dict(source_split="validation", split="test", count=8, forbidden=excluded, seed=7)
    chosen = script["select_rows"](source, **kwargs)
    assert chosen == script["select_rows"](source, **kwargs)
    assert sum(row["answer"] for _, row in chosen) == 4
    groups = {script["passage_group"](row["passage"]) for _, row in chosen}
    assert len(groups) == 8
    assert not groups & excluded
    following = script["select_rows"](
        source, **{**kwargs, "split": "calibration", "count": 4, "forbidden": groups | excluded}
    )
    assert not groups & {script["passage_group"](row["passage"]) for _, row in following}


def test_selection_keeps_source_labels_and_content(script):
    chosen = script["select_rows"](
        rows(), source_split="train", split="train", count=4, forbidden=set(), seed=3
    )
    cases, provenance = script["make_cases"](chosen, source_split="train", split="train")
    for (index, source), case, record in zip(chosen, cases, provenance):
        assert case.id == f"boolq-train-{index}"
        assert case.request.state == {"passage": source["passage"], "question": source["question"]}
        assert case.gold["answer"] == str(source["answer"]).lower()
        assert record["source_row_sha256"] == script["sha256"](script["canonical"](source))


@pytest.mark.parametrize("count", [0, 1, 3, 100])
def test_insufficient_or_unbalanced_counts_rejected(script, count):
    with pytest.raises(ValueError):
        script["select_rows"](
            rows(), source_split="train", split="train", count=count, forbidden=set(), seed=0
        )


def test_prior_group_scan_includes_archives_and_distinguishes_training(script, tmp_path):
    for name, split, state in (
        ("current.jsonl", "train", {"passage": "training", "question": "question"}),
        ("archive-v1/old.jsonl", "test", "old test passage"),
        ("archive-v2/calibration.jsonl", "calibration", {"passage": "calibration"}),
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"id": f"boolq-{split}", "split": split, "request": {"state": state}}) + "\n"
        )
    all_groups, protected, files = script["prior_groups"](tmp_path)
    assert len(all_groups) == 3
    assert len(protected) == 2
    assert script["passage_group"]("training") not in protected
    assert script["passage_group"]("old test passage") in protected
    assert len(files) == 3


def test_frozen_file_cannot_be_overwritten(script, tmp_path):
    path = tmp_path / "frozen.json"
    script["write_immutable"](path, b"original")
    script["write_immutable"](path, b"original")
    with pytest.raises(FileExistsError, match="frozen"):
        script["write_immutable"](path, b"changed")
    assert path.read_bytes() == b"original"
