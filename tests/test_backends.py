import json
import sys
from types import SimpleNamespace

import httpx
import pytest

from s1.backends import (
    GemmaBackend,
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


@pytest.fixture
def gemma_runtime(monkeypatch):
    calls = []
    model = SimpleNamespace(
        name="test/model",
        revision="pinned",
        temperature=2.5,
        device="cpu",
        dtype="float32",
        attn_implementation="sdpa",
        max_context=1024,
        head=SimpleNamespace(weight=SimpleNamespace(dtype="torch.float32")),
        quantization="none",
        quantization_details={},
    )

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return model

    monkeypatch.setitem(
        sys.modules,
        "s1.unified",
        SimpleNamespace(DEFAULT_MODEL="test/model", UnifiedDecisionModel=load),
    )
    monkeypatch.setattr("s1.backends.version", lambda package: "test")
    return model, calls


@pytest.mark.parametrize("mode", [None, "causal_multislot", "independent"])
def test_gemma_question_mode_is_recorded_without_reaching_model_loader(gemma_runtime, mode):
    model, calls = gemma_runtime
    options = {} if mode is None else {"question_mode": mode}
    backend = GemmaBackend(revision="pinned", temperature=None, **options)
    assert calls == [("test/model", {"revision": "pinned", "temperature": None})]
    assert backend.question_mode == (mode or "causal_multislot")
    assert backend.metadata["question_mode"] == backend.question_mode
    assert backend.metadata["temperature"] == model.temperature == 2.5


def test_invalid_gemma_question_mode_fails_before_loading(gemma_runtime):
    _, calls = gemma_runtime
    with pytest.raises(ValueError, match="question_mode"):
        GemmaBackend(question_mode="generated")
    assert calls == []


@pytest.mark.parametrize("entry", ["predict", "single_batch", "batch"])
def test_gemma_independent_preserves_native_request_and_response(
    monkeypatch, gemma_runtime, decision_request, entry
):
    model, _ = gemma_runtime
    model.predict = lambda request: pytest.fail("independent mode must use native question prompts")
    backend = GemmaBackend(question_mode="independent")
    decision_request.media.append(AudioInput(samples=[0.1, 0.2]))
    original = decision_request.model_dump()
    requests = [decision_request]
    if entry == "batch":
        other = decision_request.model_copy(deep=True)
        other.state = "another state"
        other.questions.reverse()
        requests.append(other)
    responses = [UniformBackend().predict(request) for request in requests]
    execution = {
        "forward_calls": 1,
        "batch_sizes": [sum(len(request.questions) for request in requests)],
        "sequence_tokens": [7, 11, 13] * len(requests),
        "scope": "whole_adapter_batch",
        "readout": "candidate",
        "independent_questions": True,
    }
    for response in responses:
        response["execution"] = execution
        response["answers"]["route"]["probabilities"] = {"billing": 0.125, "technical": 0.875}
        response["answers"]["route"]["choice"] = "technical"
    calls = []

    def native_batch(actual_model, actual_requests, **kwargs):
        calls.append((actual_model, actual_requests, kwargs))
        return responses

    monkeypatch.setitem(
        sys.modules, "s1.evaluation.gemma", SimpleNamespace(predict_batch=native_batch)
    )
    if entry == "predict":
        actual = backend.predict(decision_request)
        assert actual is responses[0]
    else:
        actual = backend.predict_batch(requests)
        assert actual is responses
    assert len(calls) == 1
    actual_model, actual_requests, options = calls[0]
    assert actual_model is model
    assert len(actual_requests) == len(requests)
    assert all(actual is expected for actual, expected in zip(actual_requests, requests))
    assert options == {"independent": True, "readout": "candidate"}
    assert decision_request.model_dump() == original
    assert model.temperature == 2.5


@pytest.mark.parametrize("mode", [None, "causal_multislot"])
def test_gemma_default_keeps_serial_single_request_and_causal_batch(
    monkeypatch, gemma_runtime, decision_request, mode
):
    model, _ = gemma_runtime
    response = UniformBackend().predict(decision_request)
    serial = []
    model.predict = lambda request: serial.append(request) or response
    backend = GemmaBackend(**({"question_mode": mode} if mode else {}))
    batches = []

    def native_batch(actual_model, requests, **kwargs):
        batches.append((actual_model, requests, kwargs))
        return [response] * len(requests)

    monkeypatch.setitem(
        sys.modules, "s1.evaluation.gemma", SimpleNamespace(predict_batch=native_batch)
    )
    assert backend.predict(decision_request) is response
    assert backend.predict_batch([decision_request]) == [response]
    assert serial == [decision_request, decision_request]
    assert batches == []
    requests = [decision_request, decision_request.model_copy(deep=True)]
    assert backend.predict_batch(requests) == [response, response]
    assert batches == [(model, requests, {"independent": False, "readout": "candidate"})]
    assert len(serial) == 2


@pytest.mark.parametrize("entry", ["predict", "single_batch", "batch"])
@pytest.mark.parametrize("failure_kind", ["import", "inference"])
def test_gemma_independent_errors_do_not_fall_back_to_causal(
    monkeypatch, gemma_runtime, decision_request, entry, failure_kind
):
    model, _ = gemma_runtime
    model.predict = lambda request: pytest.fail("must not fall back to causal inference")
    backend = GemmaBackend(question_mode="independent")
    failure = RuntimeError("native inference failed")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setitem(
        sys.modules,
        "s1.evaluation.gemma",
        None if failure_kind == "import" else SimpleNamespace(predict_batch=fail),
    )
    expected = ModuleNotFoundError if failure_kind == "import" else RuntimeError
    with pytest.raises(expected) as caught:
        if entry == "predict":
            backend.predict(decision_request)
        else:
            backend.predict_batch([decision_request] * (2 if entry == "batch" else 1))
    if failure_kind == "inference":
        assert caught.value is failure


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
