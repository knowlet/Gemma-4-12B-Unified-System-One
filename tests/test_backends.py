import httpx
import pytest

from s1.backends import (
    HTTPBackend,
    LayaBackend,
    UniformBackend,
    UnsupportedRequest,
    normalize_response,
)
from s1.contracts import AudioInput


def test_http_contract_and_no_retry(decision_request):
    calls = []

    def respond(request):
        import json

        payload = json.loads(request.content)
        assert set(payload["questions"]) == {"route", "refund", "urgency"}
        assert request.headers["Authorization"] == "Bearer test-only"
        calls.append(request)
        return httpx.Response(503, json={"error": "unavailable"})

    backend = HTTPBackend(
        "https://test.invalid/v1/systemone",
        token="test-only",
        client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    with pytest.raises(httpx.HTTPStatusError):
        backend.predict(decision_request)
    assert len(calls) == 1


def test_laya_normalizes_numeric_scores_and_rounding(decision_request):
    class Agent:
        revision = "test-snapshot"

        def predict(self, state, questions, **kwargs):
            assert questions["urgency"]["criteria"] == ["low", "high"]
            assert kwargs["max_len"] == 1024
            return {
                "answers": {
                    "route": {"probabilities": {"billing": 0.3333, "technical": 0.6666}},
                    "refund": {"noul": 0.9},
                    "urgency": {"probabilities": {"0": 0.2, "1": 0.8}},
                }
            }

    backend = LayaBackend(agent=Agent(), max_len=1024)
    assert backend.metadata["revision"] == "test-snapshot"
    result = backend.predict(decision_request)["answers"]
    assert result["urgency"]["score"] == 18
    assert result["refund"]["probabilities"] == pytest.approx({"false": 0.1, "true": 0.9})
    assert sum(result["route"]["probabilities"].values()) == pytest.approx(1)
    decision_request.media.append(AudioInput(samples=[0.1]))
    with pytest.raises(UnsupportedRequest):
        backend.predict(decision_request)


def test_never_fabricate_missing_distributions(decision_request):
    result = UniformBackend().predict(decision_request)
    del result["answers"]["route"]["probabilities"]
    with pytest.raises(KeyError):
        normalize_response(decision_request, result)


def test_mismatched_response_ids(decision_request):
    result = UniformBackend().predict(decision_request)
    result["answers"]["extra"] = result["answers"]["route"]
    with pytest.raises(ValueError):
        normalize_response(decision_request, result)
