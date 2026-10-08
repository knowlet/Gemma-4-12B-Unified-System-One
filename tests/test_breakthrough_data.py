"""Independent truth, leakage, heldout templates and immutable data checks."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest

from s1.evaluation.datasets import audit_splits, load_cases, request_fingerprint


@pytest.fixture(scope="module")
def script():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/prepare_breakthrough_data.py")
    )


def test_exact_conditional_logic_and_ordinal_expectation(script):
    boolean = {"kind": "boolean_table", "counts": [1, 2, 3, 4]}
    assert script["oracle"](boolean, "x_given_y")["probabilities"]["true"] == pytest.approx(2 / 3)
    assert script["oracle"](boolean, "x_implies_y")["probabilities"]["true"] == pytest.approx(0.7)
    assert script["oracle"](boolean, "x_xor_y")["probabilities"]["true"] == pytest.approx(0.5)
    ordinal = script["oracle"]({"kind": "ordinal", "labels": [0, 2, 5], "counts": [1, 2, 3]})
    assert ordinal["expected_score"] == pytest.approx(19 / 6)
    assert ordinal["most_likely_label"] == "5"
    choice = script["oracle"](
        {"kind": "categorical", "labels": ["a", "b", "c", "d"], "counts": [2, 0, 3, 5]}
    )
    assert choice["probabilities"] == {"a": 0.2, "b": 0.0, "c": 0.3, "d": 0.5}


def test_frozen_data_is_deterministic_disjoint_balanced_and_template_heldout(script, tmp_path):
    destinations = [tmp_path / "first", tmp_path / "second"]
    for output in destinations:
        script["prepare"](output, train=30, calibration=12, test=24)
    left, right = destinations
    assert {p.name: p.read_bytes() for p in left.iterdir()} == {
        p.name: p.read_bytes() for p in right.iterdir()
    }
    paths = [left / f"{split}.jsonl" for split in ("train", "calibration", "test")]
    assert audit_splits(paths)["valid"]
    groups, requests, families = {}, {}, {}
    for split, path in zip(("train", "calibration", "test"), paths, strict=True):
        cases = load_cases(path)
        groups[split] = {case.group_id for case in cases}
        requests[split] = {request_fingerprint(case.request) for case in cases}
        families[split] = {case.task_id for case in cases}
        assert {case.request.questions[0].type for case in cases} == {"choice", "noul", "score"}
        assert {
            len(case.request.questions[0].labels())
            for case in cases
            if case.request.questions[0].type == "choice"
        } == {2, 4, 8}
        for case in cases:
            q = case.request.questions[0]
            target = case.soft_gold[q.id]
            assert sum(target.values()) == pytest.approx(1)
            assert set(target) == set(q.labels())
            assert target[case.gold[q.id]] == max(target.values())
            assert "soft_gold" not in json.dumps(case.request.model_dump())
    assert (
        not groups["train"] & groups["calibration"]
        | groups["train"] & groups["test"]
        | groups["calibration"] & groups["test"]
    )
    assert (
        not requests["train"] & requests["calibration"]
        | requests["train"] & requests["test"]
        | requests["calibration"] & requests["test"]
    )
    assert families["test"].isdisjoint(families["train"] | families["calibration"])
    manifest = json.loads((left / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((left / name).read_bytes()).hexdigest() == expected


def test_oracle_receipts_match_inference_count_tables(script, tmp_path):
    output = tmp_path / "data"
    script["prepare"](output, train=30, calibration=12, test=24)
    cases = {
        case.id: case
        for split in ("train", "calibration", "test")
        for case in load_cases(output / f"{split}.jsonl")
    }
    records = json.loads((output / "provenance.json").read_text())["selected_source_records"]
    for record in records:
        case = cases[record["case_id"]]
        expected = record["oracle"]
        assert case.soft_gold["answer"] == expected["probabilities"]
        assert request_fingerprint(case.request) == record["request_sha256"]
        state = case.request.state
        kind = case.request.questions[0].type
        if kind == "choice":
            observed = dict.fromkeys(expected["labels"], 0)
            if "bag_inventory" in state:
                observed.update(state["bag_inventory"])
            else:
                rows = next(value for value in state.values() if isinstance(value, list))
                for row in rows:
                    observed[row["category"]] += row.get("residents", row.get("items"))
            assert list(observed.values()) == expected["integer_masses"]
        if kind == "score":
            rows = next(value for value in state.values() if isinstance(value, list))
            observed = dict.fromkeys(expected["labels"], 0)
            for row in rows:
                if "numeric_score" in row:
                    label, mass = row["numeric_score"], row["observations"]
                else:
                    rule = next(
                        rule
                        for rule in state["score_band_rules"]
                        if rule["measurement_min"] <= row["measurement"] <= rule["measurement_max"]
                    )
                    label, mass = rule["assigned_score"], row["frequency"]
                observed[str(label)] += mass
            assert list(observed.values()) == expected["integer_masses"]
            assert sum(float(label) * mass for label, mass in observed.items()) / sum(
                observed.values()
            ) == pytest.approx(expected["expected_score"])
        if kind == "noul":
            rows = next(value for value in state.values() if isinstance(value, list))
            observed = {"false": 0, "true": 0}
            operation = record["operation"]
            for row in rows:
                attributes = row.get("record_attributes", row)
                x, y = (attributes[state["symbols"][symbol]] for symbol in ("X", "Y"))
                if operation == "x_given_y" and not y:
                    continue
                if operation == "not_x_given_not_y" and y:
                    continue
                if operation in {"x", "x_given_y"}:
                    result = x
                elif operation in {"not_x", "not_x_given_not_y"}:
                    result = not x
                elif operation == "x_and_y":
                    result = all((x, y))
                elif operation == "x_or_y":
                    result = any((x, y))
                elif operation == "x_xor_y":
                    result = sum((x, y)) == 1
                else:
                    assert operation == "x_implies_y"
                    result = not (x and not y)
                observed[str(result).lower()] += row.get("records", row.get("individuals"))
            assert list(observed.values()) == expected["integer_masses"]
            assert case.soft_gold["answer"] == {
                label: mass / sum(observed.values()) for label, mass in observed.items()
            }


def test_forbidden_request_identity_is_excluded_without_reading_content(script, tmp_path):
    first = tmp_path / "first"
    script["prepare"](first, train=12, calibration=6, test=12)
    forbidden = request_fingerprint(load_cases(first / "train.jsonl")[0].request)
    output = tmp_path / "excluded"
    manifest = script["prepare"](
        output, train=12, calibration=6, test=12, forbidden_request_sha256s=[forbidden]
    )
    assert manifest["forbidden_request_count"] == 1
    assert all(forbidden not in item["request_sha256s"] for item in manifest["datasets"].values())
    with pytest.raises(ValueError, match="SHA256"):
        script["prepare"](tmp_path / "bad", forbidden_request_sha256s=["benchmark content"])


def test_frozen_output_and_invalid_recipe_are_rejected(script, tmp_path):
    output = tmp_path / "data"
    script["prepare"](output, train=12, calibration=6, test=12)
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="frozen"):
        script["prepare"](output, seed=999, train=12, calibration=6, test=12)
    assert (output / "manifest.json").read_bytes() == before
    with pytest.raises(ValueError, match="three"):
        script["prepare"](tmp_path / "invalid", train=2)
