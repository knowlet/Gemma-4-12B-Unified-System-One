import json

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
from s1.errors import (
    BackendResponseError,
    BackendTimeoutError,
    BackendTransportError,
    RequestValidationError,
)


@pytest.mark.parametrize("status", [301, 307, 400, 422, 429, 500, 503, 504])
def test_http_contract_and_no_retry(decision_request, status):
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        assert set(payload["questions"]) == {"route", "refund", "urgency"}
        assert payload["state"] == decision_request.state
        assert request.headers["Authorization"] == "Bearer test-only"
        calls.append(request)
        return httpx.Response(
            status, json={"error": "unavailable"}, headers={"Location": "/redirect"}
        )

    with httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
        backend = HTTPBackend("https://test.invalid/v1/systemone", token="test-only", client=client)
        with pytest.raises(BackendResponseError, match=f"HTTP {status}") as error:
            backend.predict(decision_request)
    assert isinstance(error.value.__cause__, httpx.HTTPStatusError)
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
    with pytest.raises(RequestValidationError) as error:
        backend.predict(decision_request)
    assert isinstance(error.value, UnsupportedRequest)


def test_never_fabricate_missing_distributions(decision_request):
    result = UniformBackend().predict(decision_request)
    del result["answers"]["route"]["probabilities"]
    with pytest.raises(BackendResponseError, match="probabilities must be an object"):
        normalize_response(decision_request, result)


def test_mismatched_response_ids(decision_request):
    result = UniformBackend().predict(decision_request)
    result["answers"]["extra"] = result["answers"]["route"]
    with pytest.raises(BackendResponseError, match="question ids"):
        normalize_response(decision_request, result)


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (httpx.ConnectTimeout, BackendTimeoutError),
        (httpx.ReadTimeout, BackendTimeoutError),
        (httpx.WriteTimeout, BackendTimeoutError),
        (httpx.PoolTimeout, BackendTimeoutError),
        (httpx.ConnectError, BackendTransportError),
        (httpx.ReadError, BackendTransportError),
        (httpx.WriteError, BackendTransportError),
        (httpx.RemoteProtocolError, BackendTransportError),
        (httpx.DecodingError, BackendResponseError),
    ],
)
def test_http_classifies_failures_without_retry(decision_request, failure, expected):
    calls = []

    def respond(request):
        calls.append(request)
        assert set(request.extensions["timeout"].values()) == {0.25}
        raise failure("private upstream message", request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        backend = HTTPBackend("https://test.invalid/decide", client=client, timeout=0.25)
        with pytest.raises(expected) as error:
            backend.predict(decision_request)
    assert type(error.value) is expected
    assert isinstance(error.value.__cause__, failure)
    assert "private upstream message" not in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize("content", [b"not json", b'{"answers":', b"", b"\xff"])
def test_http_rejects_malformed_json(decision_request, content):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, content=content)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        backend = HTTPBackend("https://test.invalid/decide", client=client)
        with pytest.raises(BackendResponseError, match="invalid JSON") as error:
            backend.predict(decision_request)
    assert isinstance(error.value.__cause__, ValueError)
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("path", "value"),
    [
        pytest.param((), None, id="null-response"),
        pytest.param((), [], id="list-response"),
        pytest.param((), "response", id="string-response"),
        pytest.param((), 1, id="numeric-response"),
        pytest.param((), {}, id="missing-answers"),
        pytest.param(("answers",), None, id="null-answers"),
        pytest.param(("answers",), "answers", id="string-answers"),
        pytest.param(("answers",), 1, id="numeric-answers"),
        pytest.param(("answers",), {}, id="missing-question-ids"),
        pytest.param(("answers",), [None], id="null-list-answer"),
        pytest.param(("answers",), [[]], id="list-list-answer"),
        pytest.param(("answers",), [{}], id="missing-list-id"),
        pytest.param(("answers",), [{"id": []}], id="unhashable-list-id"),
        pytest.param(("answers",), [{"id": 1}], id="numeric-list-id"),
        pytest.param(("answers",), [{"id": ""}], id="empty-list-id"),
        pytest.param(("answers",), [{"id": "route"}, {"id": "route"}], id="duplicate-list-id"),
        pytest.param(("answers", "route"), None, id="null-answer"),
        pytest.param(("answers", "route"), [], id="list-answer"),
        pytest.param(("answers", "route"), "answer", id="string-answer"),
        pytest.param(("answers", "route", "id"), "other", id="conflicting-id"),
        pytest.param(("answers", "route", "id"), [], id="malformed-id"),
        pytest.param(("answers", "route", "type"), "score", id="wrong-type"),
        pytest.param(("answers", "route", "type"), None, id="null-type"),
        pytest.param(("answers", "route", "probabilities"), None, id="null-distribution"),
        pytest.param(("answers", "route", "probabilities"), [0.5, 0.5], id="list-distribution"),
        pytest.param(("answers", "route", "probabilities"), {"billing": 1.0}, id="missing-label"),
        pytest.param(
            ("answers", "route", "probabilities"),
            {"billing": 0.5, "technical": 0.5, "extra": 0},
            id="extra-label",
        ),
        pytest.param(
            ("answers", "route", "probabilities"),
            {"billing": 0.4, "technical": 0.4},
            id="missing-mass",
        ),
        pytest.param(
            ("answers", "route", "probabilities"),
            {"billing": 0.6, "technical": 0.6},
            id="excess-mass",
        ),
        pytest.param(
            ("answers", "route", "probabilities"),
            {"billing": 0, "technical": 0},
            id="zero-mass",
        ),
        pytest.param(("answers", "urgency"), {}, id="missing-score-distribution"),
        pytest.param(("answers", "refund"), {}, id="missing-noul"),
        pytest.param(("answers", "refund", "probabilities"), [], id="invalid-noul-distribution"),
        pytest.param(("answers", "refund", "noul"), 0.9, id="conflicting-noul-distribution"),
    ],
)
def test_http_rejects_malformed_response_shapes(decision_request, path, value):
    response = UniformBackend().predict(decision_request)
    if path:
        target = response
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    else:
        response = value
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json=response)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        backend = HTTPBackend("https://test.invalid/decide", client=client)
        with pytest.raises(BackendResponseError):
            backend.predict(decision_request)
    assert len(calls) == 1


@pytest.mark.parametrize("question", ["route", "refund", "urgency"])
@pytest.mark.parametrize(
    "value",
    [None, True, "0.5", "bad", [], {}, -0.1, 1.1, float("nan"), float("inf"), 10**400],
    ids=[
        "null",
        "bool",
        "numeric-string",
        "string",
        "list",
        "object",
        "negative",
        "over-one",
        "nan",
        "inf",
        "overflow",
    ],
)
def test_rejects_invalid_probability_values(decision_request, question, value):
    response = UniformBackend().predict(decision_request)
    answer = response["answers"][question]
    if question == "refund":
        answer["noul"] = value
    else:
        label = next(iter(answer["probabilities"]))
        answer["probabilities"][label] = value
    with pytest.raises(BackendResponseError):
        normalize_response(decision_request, response)


@pytest.mark.parametrize("listed", [False, True])
def test_http_accepts_both_answer_shapes(decision_request, listed):
    response = UniformBackend().predict(decision_request)
    response["model"] = "test-model"
    if listed:
        response["answers"] = list(reversed(response["answers"].values()))
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response))
    ) as client:
        result = HTTPBackend("https://test.invalid/decide", client=client).predict(decision_request)
    assert result["model"] == "test-model"
    assert list(result["answers"]) == [q.id for q in decision_request.questions]
    assert result["answers"]["urgency"]["score"] == 15


@pytest.mark.parametrize("choice", [None, [], {}, 1, "unknown", "billing"])
def test_rejects_illegal_or_nonmaximizing_choice(decision_request, choice):
    response = UniformBackend().predict(decision_request)
    response["answers"]["route"].update(
        choice=choice, probabilities={"billing": 0.25, "technical": 0.75}
    )
    with pytest.raises(BackendResponseError, match="choice"):
        normalize_response(decision_request, response)


@pytest.mark.parametrize(
    ("probabilities", "choice", "expected"),
    [
        ({"billing": 0.25, "technical": 0.75}, "technical", "technical"),
        ({"billing": 0.5, "technical": 0.5}, "technical", "technical"),
        ({"billing": 0.5, "technical": 0.5}, "billing", "billing"),
        ({"billing": 0.25, "technical": 0.75}, None, "technical"),
        ({"technical": 0.5, "billing": 0.5}, None, "billing"),
    ],
)
def test_choice_contract_survives_repeated_normalization(
    decision_request, probabilities, choice, expected
):
    response = UniformBackend().predict(decision_request)
    response["answers"]["route"] = {"probabilities": probabilities}
    if choice is not None:
        response["answers"]["route"]["choice"] = choice
    result = normalize_response(decision_request, response)
    answer = result["answers"]["route"]
    assert answer["choice"] == expected
    assert list(answer["probabilities"]) == ["billing", "technical"]
    assert answer["confidence"] == max(probabilities.values())
    assert normalize_response(decision_request, result) == result
    assert response["answers"]["route"].get("choice") == choice


def test_near_tie_is_not_a_maximizing_choice(decision_request):
    response = UniformBackend().predict(decision_request)
    response["answers"]["route"].update(
        choice="billing", probabilities={"billing": 0.499999999, "technical": 0.500000001}
    )
    with pytest.raises(BackendResponseError, match="maximize"):
        normalize_response(decision_request, response)


def test_benchmark_scores_distribution_argmax_independently_of_tied_choice(benchmark_case):
    from s1.benchmark import evaluate

    class TiedBackend(UniformBackend):
        def predict(self, request):
            response = super().predict(request)
            response["answers"]["route"]["choice"] = "technical"
            return response

    backend = TiedBackend()
    response = normalize_response(benchmark_case.request, backend.predict(benchmark_case.request))
    assert response["answers"]["route"]["choice"] == "technical"
    report = evaluate(backend, [benchmark_case])
    assert report["coverage"] == 1
    assert report["metrics"]["acc"] == pytest.approx(2 / 3)


def test_http_unsupported_media_does_not_send_request(decision_request):
    calls = []
    decision_request.media.append(AudioInput(samples=[0.1]))
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: calls.append(request))
    ) as client:
        backend = HTTPBackend("https://test.invalid/decide", client=client)
        with pytest.raises(UnsupportedRequest):
            backend.predict(decision_request)
    assert calls == []
