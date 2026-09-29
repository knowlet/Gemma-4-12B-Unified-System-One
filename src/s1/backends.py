"""Explicit adapters; backend errors are never replaced with another model."""

from __future__ import annotations

import math
from importlib.metadata import version
from typing import Protocol

import httpx

from .contracts import DecisionRequest, answer_from_probabilities


class UnsupportedRequest(ValueError):
    pass


class Backend(Protocol):
    name: str

    def predict(self, request: DecisionRequest) -> dict: ...


def normalize_response(request, response, *, ordinal_scores=False):
    answers = response["answers"]
    if isinstance(answers, list):
        if len({a["id"] for a in answers}) != len(answers):
            raise ValueError("duplicate answer ids")
        answers = {a["id"]: a for a in answers}
    if set(answers) != {q.id for q in request.questions}:
        raise ValueError("response question ids differ from the request")
    normalized = {}
    for q in request.questions:
        answer = answers[q.id]
        if answer.get("type", q.type) != q.type:
            raise ValueError("response question type differs from request")
        if q.type == "noul":
            p = float(answer["noul"])
            values = [1 - p, p]
        else:
            dist = answer["probabilities"]
            keys = (
                [str(i) for i in range(len(q.labels()))]
                if ordinal_scores and q.type == "score"
                else q.labels()
            )
            if set(dist) != set(keys):
                raise ValueError("response probability labels differ from request")
            values = [float(dist[k]) for k in keys]
        total = sum(values)
        # Laya serializes each probability to four decimal places. Only repair
        # bounded rounding error; missing mass or invalid values are failures.
        if (
            not all(math.isfinite(x) and 0 <= x <= 1 for x in values)
            or abs(total - 1) > len(values) * 0.000051
        ):
            raise ValueError("invalid backend probability distribution")
        normalized[q.id] = answer_from_probabilities(q, [x / total for x in values])
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
        response = self.client.post(self.url, json=payload, headers=self.headers)
        response.raise_for_status()
        return normalize_response(request, response.json())

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
