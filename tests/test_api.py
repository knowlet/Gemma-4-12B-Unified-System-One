import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from s1.api import create_app
from s1.backends import HTTPBackend, UniformBackend, UnsupportedRequest
from s1.errors import (
    BackendResponseError,
    BackendTimeoutError,
    BackendTransportError,
    RequestValidationError,
)


@pytest.fixture(params=["/v1/systemone", "/decide"])
def endpoint(request):
    return request.param


def test_auth_and_validation(request_data, endpoint):
    client = TestClient(create_app(UniformBackend(), api_key="test-key"))
    assert client.get("/healthz").status_code == 200
    assert client.post(endpoint, json=request_data).status_code == 401
    headers = {"Authorization": "Bearer test-key"}
    response = client.post(endpoint, json=request_data, headers=headers)
    assert response.status_code == 200
    assert response.json()["answers"]["refund"]["noul"] == 0.5
    request_data["questions"]["route"]["criteria"] = []
    assert client.post(endpoint, json=request_data, headers=headers).status_code == 422


def test_malformed_named_question_is_client_error(request_data, endpoint):
    client = TestClient(create_app(UniformBackend()))
    request_data["questions"]["route"] = None
    assert client.post(endpoint, json=request_data).status_code == 422


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (RequestValidationError("invalid image or context"), 422),
        (UnsupportedRequest("unsupported media"), 422),
        (BackendResponseError("invalid distribution"), 502),
        (BackendTransportError("backend unavailable"), 502),
        (BackendTimeoutError("backend timed out"), 504),
    ],
)
def test_explicit_error_provenance(request_data, endpoint, failure, status):
    class BrokenBackend:
        name = "broken"

        def predict(self, request):
            raise failure

    with TestClient(create_app(BrokenBackend())) as client:
        response = client.post(endpoint, json=request_data)
    assert response.status_code == status
    assert response.json() == {"detail": str(failure)}


@pytest.mark.parametrize("failure", [ValueError("internal value bug"), RuntimeError("model bug")])
def test_internal_errors_are_not_client_errors(request_data, endpoint, failure):
    class BrokenBackend:
        name = "broken"

        def predict(self, request):
            raise failure

    with TestClient(create_app(BrokenBackend()), raise_server_exceptions=False) as client:
        response = client.post(endpoint, json=request_data)
    assert response.status_code == 500
    assert str(failure) not in response.text


def test_internal_normalization_value_error_is_not_misclassified(
    monkeypatch, request_data, endpoint
):
    def broken_answer(*args, **kwargs):
        raise ValueError("internal normalization bug")

    # Build the backend response before replacing the normalization helper.
    from s1.contracts import DecisionRequest

    result = UniformBackend().predict(DecisionRequest.model_validate(request_data))

    class Backend:
        name = "test"

        def predict(self, request):
            return result

    monkeypatch.setattr("s1.backends.answer_from_probabilities", broken_answer)
    with TestClient(create_app(Backend()), raise_server_exceptions=False) as client:
        response = client.post(endpoint, json=request_data)
    assert response.status_code == 500


@pytest.mark.parametrize(
    "result",
    [None, [], {}, {"answers": None}, {"answers": [None]}, {"answers": [{"id": []}]}],
)
def test_backend_response_shapes_are_gateway_errors(request_data, endpoint, result):
    class Backend:
        name = "malformed"

        def predict(self, request):
            return result

    with TestClient(create_app(Backend())) as client:
        response = client.post(endpoint, json=request_data)
    assert response.status_code == 502
    assert isinstance(response.json()["detail"], str)


@pytest.mark.parametrize(
    ("upstream_status", "body", "status", "detail"),
    [
        (200, b"not json", 502, "backend returned invalid JSON"),
        (200, b"\xff", 502, "backend returned invalid JSON"),
        (200, b"null", 502, "backend response must be an object"),
        (200, b'{"answers": null}', 502, "response answers must be an object or list"),
        (204, b"", 502, "backend returned invalid JSON"),
        (307, b"", 502, "backend returned HTTP 307"),
        (422, b"upstream error", 502, "backend returned HTTP 422"),
        (503, b"upstream error", 502, "backend returned HTTP 503"),
        (504, b"upstream timeout", 502, "backend returned HTTP 504"),
    ],
)
def test_http_response_failures_are_structured_gateway_errors(
    request_data, endpoint, upstream_status, body, status, detail
):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(upstream_status, content=body, headers={"Location": "/redirect"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as upstream:
        backend = HTTPBackend("https://test.invalid/decide", client=upstream)
        with TestClient(create_app(backend)) as client:
            response = client.post(endpoint, json=request_data)
    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("failure", "status", "detail"),
    [
        (httpx.ConnectTimeout, 504, "backend request timed out"),
        (httpx.ReadTimeout, 504, "backend request timed out"),
        (httpx.WriteTimeout, 504, "backend request timed out"),
        (httpx.PoolTimeout, 504, "backend request timed out"),
        (httpx.ConnectError, 502, "backend transport failed"),
        (httpx.RemoteProtocolError, 502, "backend transport failed"),
        (httpx.DecodingError, 502, "backend response could not be decoded"),
    ],
)
def test_http_transport_failures_are_structured_without_retry(
    request_data, endpoint, failure, status, detail
):
    calls = []

    def respond(request):
        calls.append(request)
        raise failure("private upstream details", request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as upstream:
        backend = HTTPBackend("https://test.invalid/decide", client=upstream)
        with TestClient(create_app(backend)) as client:
            response = client.post(endpoint, json=request_data)
    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert len(calls) == 1


@pytest.mark.parametrize(
    "answer",
    [
        None,
        {},
        {"probabilities": []},
        {"probabilities": {"billing": "bad", "technical": 0.5}},
        {"probabilities": {"billing": 0.2, "technical": 0.2}},
        {"probabilities": {"billing": 0.2, "technical": 0.8}, "choice": "billing"},
        {"probabilities": {"billing": 0.2, "technical": 0.8}, "choice": "unknown"},
        {"probabilities": {"billing": 0.2, "technical": 0.8}, "choice": None},
    ],
)
def test_invalid_upstream_answers_are_gateway_errors(
    request_data, decision_request, endpoint, answer
):
    result = UniformBackend().predict(decision_request)
    result["answers"]["route"] = answer
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=result))
    ) as upstream:
        backend = HTTPBackend("https://test.invalid/decide", client=upstream)
        with TestClient(create_app(backend)) as client:
            response = client.post(endpoint, json=request_data)
    assert response.status_code == 502
    assert isinstance(response.json()["detail"], str)


@pytest.mark.parametrize("listed", [False, True])
@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ({"probabilities": {"billing": 0.5, "technical": 0.5}, "choice": "technical"}, "technical"),
        ({"probabilities": {"billing": 0.2, "technical": 0.8}, "choice": "technical"}, "technical"),
        ({"probabilities": {"billing": 0.2, "technical": 0.8}}, "technical"),
        ({"probabilities": {"technical": 0.5, "billing": 0.5}}, "billing"),
    ],
)
def test_http_and_api_preserve_or_derive_choice(
    request_data, decision_request, endpoint, listed, answer, expected
):
    result = UniformBackend().predict(decision_request)
    result["answers"]["route"] = {"id": "route", **answer}
    if listed:
        result["answers"] = list(result["answers"].values())
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=result))
    ) as upstream:
        backend = HTTPBackend("https://test.invalid/decide", client=upstream)
        with TestClient(create_app(backend)) as client:
            response = client.post(endpoint, json=request_data)
    assert response.status_code == 200
    normalized = response.json()["answers"]["route"]
    assert normalized["choice"] == expected
    assert list(normalized["probabilities"]) == ["billing", "technical"]
