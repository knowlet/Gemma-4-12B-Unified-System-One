import pytest

from s1.contracts import DecisionRequest


@pytest.fixture
def request_data():
    return {
        "state": "I was charged twice. Please refund the duplicate.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "payments", "technical": "bugs"},
            },
            "refund": {"type": "noul", "instructions": "Is a refund requested?"},
            "urgency": {
                "type": "score",
                "instructions": "How urgent?",
                "criteria": {"10": "low", "20": "high"},
            },
        },
    }


@pytest.fixture
def decision_request(request_data):
    return DecisionRequest.model_validate(request_data)


@pytest.fixture
def benchmark_case(request_data):
    from s1.benchmark import Case

    return Case.model_validate(
        {
            "id": "ticket-1",
            "request": request_data,
            "gold": {"route": "billing", "refund": True, "urgency": "10"},
        }
    )
