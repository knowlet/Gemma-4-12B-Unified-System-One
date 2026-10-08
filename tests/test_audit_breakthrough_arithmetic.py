"""Small CPU arithmetic checks; no model, campaign artifacts or cloud calls."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest

from s1.contracts import DecisionRequest

SCRIPT = Path(__file__).parents[1] / "scripts/audit_breakthrough_arithmetic.py"
SPEC = importlib.util.spec_from_file_location("audit_breakthrough_arithmetic", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

TEMPERATURES = {"choice": 2.0, "noul": 1.0, "score": 2.0}

# Exact CPU float32 softmax values for [0, 2, 4] / 2 and [1, 0, -1] / 2.
NUMERIC_P = [0.09003057330846786, 0.2447284758090973, 0.6652409434318542]
ORDINAL_P = [0.5064803957939148, 0.30719590187072754, 0.18632373213768005]


@pytest.fixture
def arithmetic_request():
    return DecisionRequest.model_validate(
        {
            "state": "An artificial arithmetic fixture, with no media.",
            "questions": [
                {
                    "id": "route",
                    "type": "choice",
                    "instructions": "Choose a route.",
                    "criteria": {"a": "A", "b": "B", "c": "C", "d": "D"},
                },
                {"id": "valid", "type": "noul", "instructions": "Is it valid?"},
                {
                    "id": "numeric",
                    "type": "score",
                    "instructions": "Score it using the numeric levels.",
                    "criteria": {"10": "Low", "20": "Medium", "40": "High"},
                },
                {
                    "id": "ordinal",
                    "type": "score",
                    "instructions": "Score it using the listed levels.",
                    "criteria": ["Low", "Medium", "High"],
                },
            ],
        }
    )


@pytest.fixture
def arithmetic_rows(arithmetic_request):
    logits = [[0.0] * 4, [0.0, 0.0], [0.0, 2.0, 4.0], [1.0, 0.0, -1.0]]
    return [
        {"id": question.id, "labels": question.labels(), "logits": values}
        for question, values in zip(arithmetic_request.questions, logits, strict=True)
    ]


@pytest.fixture
def expected_answers():
    return {
        "route": {
            "id": "route",
            "type": "choice",
            "probabilities": {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
            "confidence": 0.25,
            "margin": 0.0,
            "choice": "a",
        },
        "valid": {
            "id": "valid",
            "type": "noul",
            "probabilities": {"false": 0.5, "true": 0.5},
            "confidence": 0.5,
            "margin": 0.0,
            "noul": 0.5,
        },
        "numeric": {
            "id": "numeric",
            "type": "score",
            "probabilities": dict(zip(["10", "20", "40"], NUMERIC_P, strict=True)),
            "confidence": NUMERIC_P[2],
            "margin": NUMERIC_P[2] - NUMERIC_P[1],
            "level": "40",
            "score": 32.404512986540794,
        },
        "ordinal": {
            "id": "ordinal",
            "type": "score",
            "probabilities": dict(zip(["0", "1", "2"], ORDINAL_P, strict=True)),
            "confidence": ORDINAL_P[0],
            "margin": ORDINAL_P[0] - ORDINAL_P[1],
            "level": "0",
            "score": 0.6798433661460876,
        },
    }


def test_digest_hashes_exact_bytes():
    assert audit.digest(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert (
        audit.digest(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert audit.digest(b"abc\n") != audit.digest(b"abc")


def test_cpu_fp32_probabilities_preserve_all_typed_derived_fields(
    arithmetic_request, arithmetic_rows, expected_answers
):
    rows_before, temperatures_before = copy.deepcopy(arithmetic_rows), dict(TEMPERATURES)
    result = audit.cpu_answer(arithmetic_request, arithmetic_rows, TEMPERATURES)
    assert result["answers"] == expected_answers
    assert result["question_records"] == [
        {
            "id": question.id,
            "type": question.type,
            "labels": question.labels(),
            "temperature": TEMPERATURES[question.type],
            "expected_probabilities": list(expected_answers[question.id]["probabilities"].values()),
            "expected_answer": expected_answers[question.id],
        }
        for question in arithmetic_request.questions
    ]
    assert arithmetic_rows == rows_before and TEMPERATURES == temperatures_before


def test_choice_and_noul_use_their_own_temperature(arithmetic_request):
    data = arithmetic_request.model_dump()
    data["questions"] = data["questions"][:2]
    data["questions"][0]["criteria"] = {"a": "A", "b": "B"}
    request = DecisionRequest.model_validate(data)
    rows = [
        {"id": "route", "labels": ["a", "b"], "logits": [0.0, 2.0]},
        {"id": "valid", "labels": ["false", "true"], "logits": [0.0, 1.0]},
    ]
    result = audit.cpu_answer(request, rows, {"choice": 2.0, "noul": 0.5, "score": 2.0})
    assert result["answers"] == {
        "route": {
            "id": "route",
            "type": "choice",
            "probabilities": {"a": 0.2689414322376251, "b": 0.7310585975646973},
            "confidence": 0.7310585975646973,
            "margin": 0.7310585975646973 - 0.2689414322376251,
            "choice": "b",
        },
        "valid": {
            "id": "valid",
            "type": "noul",
            "probabilities": {"false": 0.11920291185379028, "true": 0.8807970285415649},
            "confidence": 0.8807970285415649,
            "margin": 0.8807970285415649 - 0.11920291185379028,
            "noul": 0.8807970285415649,
        },
    }


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "extra",
        "reordered",
        "duplicate_id",
        "wrong_id",
        "wrong_labels",
        "label_order",
        "width",
    ],
)
def test_cpu_answer_rejects_partial_or_misaligned_population(
    arithmetic_request, arithmetic_rows, bad
):
    if bad == "missing":
        arithmetic_rows.pop()
    elif bad == "extra":
        arithmetic_rows.append(copy.deepcopy(arithmetic_rows[-1]))
    elif bad == "reordered":
        arithmetic_rows[0], arithmetic_rows[1] = arithmetic_rows[1], arithmetic_rows[0]
    elif bad == "duplicate_id":
        arithmetic_rows[1]["id"] = arithmetic_rows[0]["id"]
    elif bad == "wrong_id":
        arithmetic_rows[0]["id"] = "other"
    elif bad == "wrong_labels":
        arithmetic_rows[0]["labels"][0] = "other"
    elif bad == "label_order":
        arithmetic_rows[0]["labels"].reverse()
    else:
        arithmetic_rows[0]["logits"].pop()
    with pytest.raises(ValueError):
        audit.cpu_answer(arithmetic_request, arithmetic_rows, TEMPERATURES)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 1e100, "not-a-number", None])
def test_cpu_answer_rejects_nonfinite_or_invalid_fp32_logits(
    arithmetic_request, arithmetic_rows, bad
):
    arithmetic_rows[0]["logits"][0] = bad
    with pytest.raises(ValueError):
        audit.cpu_answer(arithmetic_request, arithmetic_rows, TEMPERATURES)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, -math.inf, "not-a-number", None])
def test_cpu_answer_rejects_invalid_temperature(arithmetic_request, arithmetic_rows, bad):
    temperatures = {**TEMPERATURES, "score": bad}
    with pytest.raises(ValueError):
        audit.cpu_answer(arithmetic_request, arithmetic_rows, temperatures)


def test_cpu_answer_requires_temperature_for_every_executed_type(
    arithmetic_request, arithmetic_rows
):
    with pytest.raises(ValueError):
        audit.cpu_answer(arithmetic_request, arithmetic_rows, {"choice": 2.0, "noul": 1.0})


def test_exact_comparison_accounts_for_every_question_and_probability(
    arithmetic_request, expected_answers
):
    assert audit.compare_answers(
        arithmetic_request, expected_answers, copy.deepcopy(expected_answers)
    ) == {
        "questions": 4,
        "probabilities": 12,
        "max_abs_error": 0.0,
        "mismatch_count": 0,
        "native_field_mismatch_count": 0,
    }


def test_one_ulp_probability_difference_is_retained_as_evidence(
    arithmetic_request, expected_answers
):
    actual = copy.deepcopy(expected_answers)
    higher, lower = math.nextafter(0.25, math.inf), math.nextafter(0.25, -math.inf)
    actual["route"]["probabilities"].update(a=higher, b=lower)
    actual["route"].update(confidence=higher, margin=higher - 0.25)
    result = audit.compare_answers(arithmetic_request, expected_answers, actual)
    assert result == {
        "questions": 4,
        "probabilities": 12,
        "max_abs_error": higher - 0.25,
        "mismatch_count": 2,
        "native_field_mismatch_count": 1,
    }
    assert 0 < result["max_abs_error"] < 1e-12


@pytest.mark.parametrize(
    ("question", "field", "value"),
    [
        ("route", "confidence", math.nextafter(0.25, math.inf)),
        ("route", "margin", math.nextafter(0.0, math.inf)),
        ("route", "choice", "b"),
        ("valid", "noul", math.nextafter(0.5, math.inf)),
        ("numeric", "level", "20"),
        ("numeric", "score", math.nextafter(32.404512986540794, math.inf)),
    ],
)
def test_native_field_difference_is_not_hidden_by_equal_probabilities(
    arithmetic_request, expected_answers, question, field, value
):
    actual = copy.deepcopy(expected_answers)
    actual[question][field] = value
    result = audit.compare_answers(arithmetic_request, expected_answers, actual)
    assert result["questions"] == 4 and result["probabilities"] == 12
    assert result["mismatch_count"] == 0 and result["max_abs_error"] == 0
    assert result["native_field_mismatch_count"] == 1


@pytest.mark.parametrize("side", ["expected", "actual"])
@pytest.mark.parametrize("bad", ["missing", "extra"])
def test_comparison_rejects_partial_or_extra_answer_population(
    arithmetic_request, expected_answers, side, bad
):
    expected, actual = copy.deepcopy(expected_answers), copy.deepcopy(expected_answers)
    target = expected if side == "expected" else actual
    if bad == "missing":
        target.pop("ordinal")
    else:
        target["extra"] = copy.deepcopy(target["route"])
    with pytest.raises(ValueError):
        audit.compare_answers(arithmetic_request, expected, actual)


@pytest.mark.parametrize("bad", ["id", "type", "missing_label", "extra_label", "label_order"])
def test_comparison_rejects_misaligned_actual_answer(arithmetic_request, expected_answers, bad):
    actual = copy.deepcopy(expected_answers)
    if bad == "id":
        actual["route"]["id"] = "other"
    elif bad == "type":
        actual["route"]["type"] = "score"
    elif bad == "missing_label":
        actual["route"]["probabilities"].pop("a")
    elif bad == "extra_label":
        actual["route"]["probabilities"]["other"] = 0.0
    else:
        actual["route"]["probabilities"] = dict(
            reversed(list(actual["route"]["probabilities"].items()))
        )
    with pytest.raises(ValueError):
        audit.compare_answers(arithmetic_request, expected_answers, actual)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.01, 1.01, "not-a-number", None])
def test_comparison_rejects_invalid_actual_probabilities(arithmetic_request, expected_answers, bad):
    actual = copy.deepcopy(expected_answers)
    actual["route"]["probabilities"]["a"] = bad
    with pytest.raises(ValueError):
        audit.compare_answers(arithmetic_request, expected_answers, actual)


def test_comparison_rejects_nonunit_probability_mass(arithmetic_request, expected_answers):
    actual = copy.deepcopy(expected_answers)
    actual["route"]["probabilities"]["a"] = 0.1
    with pytest.raises(ValueError):
        audit.compare_answers(arithmetic_request, expected_answers, actual)


@pytest.fixture
def bound_member(tmp_path):
    base, name, raw = tmp_path / "authorized", "ablate/sources/example.py.txt", b"# exact source\n"
    path = base / name
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    identity = {"size_bytes": len(raw), "sha256": audit.digest(raw)}
    return base, name, raw, identity


def test_bound_read_returns_exact_authorized_regular_bytes(bound_member):
    base, name, raw, identity = bound_member
    assert audit.read_bound_member(base, name, identity) == raw


def test_bound_read_allows_system_symlink_ancestor_above_authorized_base(tmp_path, bound_member):
    physical_base, name, raw, identity = bound_member
    system_ancestor = tmp_path / "system-ancestor"
    system_ancestor.symlink_to(physical_base.parent, target_is_directory=True)
    authorized_base = system_ancestor / physical_base.name
    assert system_ancestor.is_symlink() and not authorized_base.is_symlink()
    assert audit.read_bound_member(authorized_base, name, identity) == raw


@pytest.mark.parametrize("location", ["member", "below_base", "base"])
def test_bound_read_rejects_symlink_at_or_below_authorized_base(tmp_path, bound_member, location):
    base, name, raw, identity = bound_member
    path = base / name
    if location == "base":
        linked_base = tmp_path / "linked-base"
        linked_base.symlink_to(base, target_is_directory=True)
        base = linked_base
    elif location == "below_base":
        outside = tmp_path / "outside-sources"
        outside.mkdir()
        (outside / path.name).write_bytes(raw)
        path.unlink()
        path.parent.rmdir()
        path.parent.symlink_to(outside, target_is_directory=True)
    else:
        outside = tmp_path / "outside-source.txt"
        outside.write_bytes(raw)
        path.unlink()
        path.symlink_to(outside)
    with pytest.raises(ValueError) as error:
        audit.read_bound_member(base, name, identity)
    assert name in str(error.value)


@pytest.mark.parametrize("field", ["size_bytes", "sha256"])
def test_bound_read_rejects_wrong_identity_and_identifies_member(bound_member, field):
    base, name, raw, identity = bound_member
    identity[field] = len(raw) + 1 if field == "size_bytes" else audit.digest(b"different source\n")
    with pytest.raises(ValueError) as error:
        audit.read_bound_member(base, name, identity)
    assert name in str(error.value)
    assert ("byte size" if field == "size_bytes" else "hash") in str(error.value)


@pytest.mark.parametrize(
    "name",
    [
        "/ablate/receipt.json",
        "ablate/../train/receipt.json",
        "ablate//receipt.json",
        "ablate/./receipt.json",
        "ablate\\receipt.json",
        "ablate/C:receipt.json",
        "other/receipt.json",
    ],
)
def test_bound_read_rejects_unsafe_or_unauthorized_member_path(bound_member, name):
    base, _, _, identity = bound_member
    with pytest.raises(ValueError):
        audit.read_bound_member(base, name, identity)
