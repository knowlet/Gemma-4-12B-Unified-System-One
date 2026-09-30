import json

import pytest

from s1.evaluation.baselines import BaselineBackend
from s1.evaluation.contracts import ModelSpec
from s1.evaluation.datasets import EvaluationCase
from s1.evaluation.experiments import write_cases
from s1.evaluation.supervision import fit_baseline, prepare_support


def examples(split, count):
    return [
        EvaluationCase.model_validate(
            {
                "id": f"{split}-{i}",
                "group_id": f"{split}-group-{i}",
                "split": split,
                "request": {
                    "state": f"{'good wonderful' if i % 2 else 'bad awful'} {split} {i}",
                    "questions": [
                        {
                            "id": "sentiment",
                            "type": "choice",
                            "instructions": "Classify sentiment",
                            "criteria": {"bad": "negative", "good": "positive"},
                        }
                    ],
                },
                "gold": {"sentiment": "good" if i % 2 else "bad"},
            }
        )
        for i in range(count)
    ]


def test_support_curves_are_nested_matched_seeded_and_disjoint(tmp_path):
    write_cases(tmp_path / "train.jsonl", examples("train", 20))
    write_cases(tmp_path / "test.jsonl", examples("test", 4))
    report = prepare_support(
        tmp_path / "train.jsonl", [tmp_path / "test.jsonl"], tmp_path / "supports", shots=[2, 4]
    )
    assert len(report["cells"]) == 6
    for offset in (0, 2, 4):
        assert set(report["cells"][offset]["support_ids"]) < set(
            report["cells"][offset + 1]["support_ids"]
        )
    assert report["cells"][0]["support_ids"] != report["cells"][2]["support_ids"]
    with pytest.raises(ValueError, match="disjoint"):
        duplicated = examples("train", 20)[0].model_copy(update={"split": "test"})
        write_cases(tmp_path / "leak.jsonl", [duplicated])
        prepare_support(
            tmp_path / "train.jsonl", [tmp_path / "leak.jsonl"], tmp_path / "bad", shots=[1]
        )


@pytest.mark.parametrize("method", ["prior", "tfidf"])
def test_baseline_fit_and_inference_with_candidate_reordering(tmp_path, method):
    if method == "tfidf":
        pytest.importorskip("sklearn")
    write_cases(tmp_path / "train.jsonl", examples("train", 20))
    report = fit_baseline(tmp_path / "train.jsonl", tmp_path / "model", method=method)
    spec = ModelSpec(
        id="classifier",
        adapter=method,
        artifact_file=report["artifact_file"],
        artifact_sha256=report["artifact_sha256"],
    )
    backend = BaselineBackend(spec)
    case = examples("test", 1)[0]
    result = backend.predict(case.request)["answers"]["sentiment"]
    case.request.questions[0].criteria = {"good": "positive", "bad": "negative"}
    reordered = backend.predict(case.request)["answers"]["sentiment"]
    assert result["probabilities"] == reordered["probabilities"]
    if method == "tfidf":
        assert result["choice"] == "bad"
    artifact = json.loads((tmp_path / "model/classifier.json").read_text())
    assert len(artifact["train_ids"]) == 20
    assert len(artifact["train_groups"]) == 20
    write_cases(tmp_path / "test.jsonl", examples("test", 4))
    with pytest.raises(ValueError, match="train split"):
        fit_baseline(tmp_path / "test.jsonl", tmp_path / "leak", method=method)


def test_support_refuses_insufficient_distinct_groups_before_writing(tmp_path):
    write_cases(tmp_path / "train.jsonl", examples("train", 4))
    write_cases(tmp_path / "test.jsonl", examples("test", 2))
    with pytest.raises(ValueError, match="insufficient"):
        prepare_support(
            tmp_path / "train.jsonl", [tmp_path / "test.jsonl"], tmp_path / "support", shots=[8]
        )
    assert not (tmp_path / "support").exists()
