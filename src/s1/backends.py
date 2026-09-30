"""Explicit adapters; backend errors are never replaced with another model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from importlib.metadata import version
from typing import Protocol

import httpx

from .contracts import DecisionRequest, answer_from_probabilities
from .errors import (
    BackendResponseError,
    BackendTimeoutError,
    BackendTransportError,
    RequestValidationError,
)


class UnsupportedRequest(RequestValidationError):
    """This backend does not support the supplied request."""


class Backend(Protocol):
    name: str

    def predict(self, request: DecisionRequest) -> dict: ...


_ROUNDING_TOLERANCE = 0.000051


def _probability(value):
    # Do not coerce booleans, numeric strings, or arbitrary objects into numbers.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendResponseError("backend probabilities must be numbers in [0, 1]")
    try:
        value = float(value)
    except OverflowError as exc:
        raise BackendResponseError("backend probabilities must be finite") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise BackendResponseError("backend probabilities must be finite numbers in [0, 1]")
    return value


def _distribution(dist, labels):
    if not isinstance(dist, Mapping):
        raise BackendResponseError("response probabilities must be an object")
    if set(dist) != set(labels):
        raise BackendResponseError("response probability labels differ from request")
    values = [_probability(dist[label]) for label in labels]
    total = sum(values)
    # Laya serializes each probability to four decimal places. Only repair
    # bounded rounding error; missing mass or invalid values are failures.
    if total <= 0 or abs(total - 1) > len(values) * _ROUNDING_TOLERANCE:
        raise BackendResponseError("invalid backend probability distribution")
    return [value / total for value in values]


def normalize_response(request, response, *, ordinal_scores=False):
    """Validate backend shapes and rebuild derived fields from probabilities.

    Answers may be keyed objects or a list with unique string IDs. Choice/score
    distributions must contain exactly the requested labels and numeric values;
    Noul requires a numeric P(true), with any supplied distribution agreeing.
    A supplied choice must be a legal label at the distribution's exact maximum.
    Preserve it even in a tie; an absent choice uses the first maximum in request
    label order. Benchmark metrics independently use that ordered distribution's
    argmax, so a preserved tied choice need not be the benchmark's tie winner.
    """
    if not isinstance(response, Mapping):
        raise BackendResponseError("backend response must be an object")
    answers = response.get("answers")
    if isinstance(answers, list):
        by_id = {}
        for answer in answers:
            if not isinstance(answer, Mapping):
                raise BackendResponseError("each response answer must be an object")
            answer_id = answer.get("id")
            if not isinstance(answer_id, str) or not answer_id:
                raise BackendResponseError("each listed answer needs a nonempty string id")
            if answer_id in by_id:
                raise BackendResponseError("duplicate answer ids")
            by_id[answer_id] = answer
        answers = by_id
    if not isinstance(answers, Mapping):
        raise BackendResponseError("response answers must be an object or list")
    if set(answers) != {q.id for q in request.questions}:
        raise BackendResponseError("response question ids differ from the request")
    normalized = {}
    for q in request.questions:
        answer = answers[q.id]
        if not isinstance(answer, Mapping):
            raise BackendResponseError("each response answer must be an object")
        if "id" in answer and answer["id"] != q.id:
            raise BackendResponseError("response answer id differs from its dictionary key")
        if answer.get("type", q.type) != q.type:
            raise BackendResponseError("response question type differs from request")
        if q.type == "noul":
            p = _probability(answer.get("noul"))
            values = [1 - p, p]
            if "probabilities" in answer:
                supplied = _distribution(answer["probabilities"], q.labels())
                if any(
                    abs(actual - expected) > len(values) * _ROUNDING_TOLERANCE
                    for actual, expected in zip(supplied, values)
                ):
                    raise BackendResponseError("response noul differs from its distribution")
        else:
            keys = (
                [str(i) for i in range(len(q.labels()))]
                if ordinal_scores and q.type == "score"
                else q.labels()
            )
            values = _distribution(answer.get("probabilities"), keys)
        normalized[q.id] = answer_from_probabilities(q, values)
        if q.type == "choice" and "choice" in answer:
            choice = answer["choice"]
            labels = q.labels()
            if not isinstance(choice, str) or choice not in labels:
                raise BackendResponseError("response choice is not a legal label")
            if values[labels.index(choice)] != max(values):
                raise BackendResponseError("response choice does not maximize its distribution")
            normalized[q.id]["choice"] = choice
    return {**response, "answers": normalized}


class UniformBackend:
    name = "uniform"
    metadata = {"kind": "sanity-baseline", "model": "uniform"}

    def predict(self, request):
        return {
            "answers": {
                q.id: answer_from_probabilities(q, [1 / len(q.labels())] * len(q.labels()))
                for q in request.questions
            }
        }


class LayaBackend:
    def __init__(
        self,
        model="convaiinnovations/laya",
        *,
        revision=None,
        device=None,
        subfolder=None,
        max_len=512,
        agent=None,
    ):
        self.name = f"laya:{model}" + (f"/{subfolder}" if subfolder else "")
        self.max_len = max_len
        if max_len < 1:
            raise ValueError("max_len must be positive")
        if agent is None:
            import laya

            kwargs = {"device": device, "revision": revision}
            if subfolder:
                kwargs["subfolder"] = subfolder
            agent = laya.load(model, **kwargs)
        self.agent = agent
        self.metadata = {
            "model": model,
            "revision": getattr(agent, "revision", None) or revision,
            "subfolder": subfolder,
            "device": str(getattr(agent, "device", device)),
            "max_len": max_len,
            "temperature": getattr(agent, "temperature", None),
            "temperature_by_options": getattr(agent, "temperature_by_options", None),
        }
        try:
            self.metadata["runtime_version"] = version("laya")
        except ModuleNotFoundError:
            pass

    def predict(self, request):
        if request.media:
            raise UnsupportedRequest("Laya supports text only; media was not discarded")
        questions = request.named_questions_payload()
        for q in request.questions:
            if q.type == "score":
                questions[q.id]["criteria"] = q.descriptions()
        response = self.agent.predict(request.state, questions, max_len=self.max_len)
        return normalize_response(request, response, ordinal_scores=True)


class HTTPBackend:
    """Call an explicitly configured Jev-compatible endpoint (no retries)."""

    def __init__(
        self, url, *, name="http", token=None, timeout=120, client=None, supports_media=False
    ):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if httpx.URL(url).username or httpx.URL(url).password:
            raise ValueError("use a bearer token rather than credentials in the endpoint URL")
        self.name, self.url, self.supports_media = name, url, supports_media
        self.timeout = timeout
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=False)
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.metadata = {
            "kind": "http",
            "endpoint": str(httpx.URL(url).copy_with(query=None)),
            "timeout": timeout,
        }

    def predict(self, request):
        if request.media and not self.supports_media:
            raise UnsupportedRequest("endpoint has not declared support for this media contract")
        payload = {"state": request.state, "questions": request.named_questions_payload()}
        if request.media:
            payload["media"] = [m.model_dump() for m in request.media]
        return normalize_response(request, self._post(payload))

    def _post(self, payload):
        try:
            response = self.client.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
                follow_redirects=False,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise BackendTimeoutError("backend request timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise BackendResponseError(f"backend returned HTTP {exc.response.status_code}") from exc
        except httpx.DecodingError as exc:
            raise BackendResponseError("backend response could not be decoded") from exc
        except httpx.RequestError as exc:
            raise BackendTransportError("backend transport failed") from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise BackendResponseError("backend returned invalid JSON") from exc
        return data

    def close(self):
        self.client.close()


class GemmaBackend:
    def __init__(self, model=None, **kwargs):
        from .unified import DEFAULT_MODEL, UnifiedDecisionModel

        self.model = UnifiedDecisionModel(model or DEFAULT_MODEL, **kwargs)
        self.name = f"gemma:{self.model.name}"
        self.metadata = {
            "model": self.model.name,
            "revision": self.model.revision,
            "temperature": self.model.temperature,
            "device": self.model.device,
            "max_context": self.model.max_context,
        }

    def predict(self, request):
        return self.model.predict(request)
