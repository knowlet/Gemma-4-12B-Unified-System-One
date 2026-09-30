import json

import pytest

from s1.evaluation.datasets import load_cases
from s1.evaluation.importing import import_dataset


def test_import_preserves_public_source_and_labels(tmp_path):
    source = tmp_path / "boolq.jsonl"
    source.write_text(
        json.dumps({"passage": "A passage", "question": "A question?", "answer": True})
    )
    report = import_dataset(
        source,
        tmp_path / "test.jsonl",
        format="boolq",
        source_url="https://example.invalid/data",
        revision="abc123",
        license="fixture",
        original_split="validation",
        split="test",
    )
    assert report["original_split"] == "validation"
    assert report["evaluation_split"] == "test"
    assert report["source_sha256"]
    assert load_cases(tmp_path / "test.jsonl")[0].gold == {"answer": "true"}


def test_unlabeled_ocnli_and_high_cardinality_are_rejected(tmp_path):
    for kind, row in [
        ("ocnli", {"sentence1": "前提", "sentence2": "假設", "label": "-"}),
        (
            "choice",
            {
                "state": "intent",
                "question": "which?",
                "answer": "0",
                "options": {str(i): str(i) for i in range(60)},
            },
        ),
    ]:
        source = tmp_path / f"{kind}.jsonl"
        source.write_text(json.dumps(row))
        with pytest.raises(ValueError):
            import_dataset(
                source,
                tmp_path / f"{kind}-out.jsonl",
                format=kind,
                source_url="https://example.invalid/data",
                revision="fixture",
                license="fixture",
                original_split="test",
                split="test",
            )
        assert not (tmp_path / f"{kind}-out.jsonl").exists()
