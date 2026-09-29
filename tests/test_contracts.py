import math

import pytest
from pydantic import ValidationError

from s1.contracts import DecisionRequest, Question, answer_from_probabilities


@pytest.mark.parametrize(
    "criteria", [[], ["x"], ["x", "x"], ["", "b"], [str(i) for i in range(53)]]
)
def test_bad_options_rejected(criteria):
    with pytest.raises(ValidationError):
        Question(id="q", instructions="pick", criteria=criteria)


def test_aliases_and_ids(request_data):
    request_data["questions"]["route"]["options"] = request_data["questions"]["route"].pop(
        "criteria"
    )
    req = DecisionRequest.model_validate(request_data)
    assert req.questions[0].labels() == ["billing", "technical"]
    request_data["questions"]["route"]["id"] = "different"
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(request_data)


def test_duplicate_ids(decision_request):
    data = decision_request.model_dump()
    data["questions"].append(data["questions"][0])
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(data)


def test_numeric_score_and_confidence(decision_request):
    answer = answer_from_probabilities(decision_request.questions[2], [0.25, 0.75])
    assert answer["score"] == 17.5
    assert answer["level"] == "20"
    assert answer["confidence"] == 0.75
    assert answer["margin"] == 0.5


@pytest.mark.parametrize("values", [[math.nan, 1], [math.inf, 0], [0.2, 0.2], [-0.1, 1.1]])
def test_invalid_probabilities(decision_request, values):
    with pytest.raises(ValueError):
        answer_from_probabilities(decision_request.questions[0], values)


@pytest.mark.parametrize("criteria", [{"NaN": "low", "20": "high"}, {"20": "high", "10": "low"}])
def test_bad_score_levels(criteria):
    with pytest.raises(ValidationError):
        Question(id="s", type="score", instructions="rank", criteria=criteria)


def test_media_validation(request_data):
    request_data["media"] = [{"type": "audio", "samples": [float("nan")]}]
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(request_data)
    request_data["media"] = [{"type": "image", "url": "https://example.org/image.png"}]
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(request_data)
