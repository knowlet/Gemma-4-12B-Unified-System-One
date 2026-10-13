"""HTTP compatibility and an optional, offline pinned-upstream round trip."""

import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from s1.api import create_app
from s1.backends import UniformBackend
from s1.errors import (
    BackendResponseError,
    BackendTimeoutError,
    BackendTransportError,
    RequestValidationError,
)

UPSTREAM_REVISION = "1231850a0bf1a0c0341fe408ef1668dbbfdfac46"


class CapturingBackend(UniformBackend):
    name = "gemma-native"

    def __init__(self):
        self.calls = []

    def predict(self, request):
        self.calls.append(request)
        return super().predict(request)


@pytest.fixture
def wire_payload():
    return {
        "model": "gemma-browser",
        "state": {"page": {"text": "Destination Lisbon"}, "elements": []},
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": {"goal": "Find Lisbon stays", "rules": "Submit first"},
                "criteria": {"CLICK": "Click observed target", "DONE": "Goal visibly satisfied"},
            },
            "click_target": {
                "type": "choice",
                "instructions": {"goal": "Find Lisbon stays", "operation": "CLICK"},
                "criteria": {
                    "2": {"element": "[2] Find stays", "current_value": "", "role": "button"}
                },
            },
        },
    }


@pytest.fixture
def upstream_model(monkeypatch):
    # Supplying a local, already-fetched checkout keeps default tests offline.
    configured = os.environ.get("JEV_UPSTREAM_PATH")
    if not configured:
        pytest.skip("set JEV_UPSTREAM_PATH to an existing pinned Jev checkout")
    checkout = Path(configured)
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == UPSTREAM_REVISION
    for filename in ("model.py", "questions.py"):
        relative = "jev_ultrafast/" + filename
        original = subprocess.check_output(["git", "-C", str(checkout), "show", "HEAD:" + relative])
        assert (checkout / relative).read_bytes() == original
    name = "_s1_jev_api_upstream_test"
    package = types.ModuleType(name)
    package.__path__ = [str(checkout / "jev_ultrafast")]
    monkeypatch.setitem(sys.modules, name, package)
    # Load actual model.py and questions.py without browser-harness/Chrome or
    # importing upstream's package initializer, which connects browser modules.
    try:
        yield importlib.import_module(name + ".model")
    finally:
        for module in list(sys.modules):
            if module.startswith(name + "."):
                monkeypatch.delitem(sys.modules, module, raising=False)


def test_actual_upstream_choose_round_trip(upstream_model, monkeypatch):
    backend = CapturingBackend()
    app = create_app(
        backend, api_key="test-key", jev_compatible=True, model_identity="gemma-browser"
    )
    page = {
        "url": "http://fixture.invalid/hotels",
        "title": "Hotel search",
        "text": "Lisbon · Design · Free cancellation",
        "actions": [
            {
                "id": "click-search",
                "kind": "click",
                "node": "button-1",
                "label": "Find stays",
                "role": "button",
            },
            {
                "id": "fill-destination",
                "kind": "fill",
                "node": "field-1",
                "label": "Destination",
                "role": "searchbox",
                "value": "Lisbon",
            },
            {
                "id": "select-design",
                "kind": "select",
                "node": "select-1",
                "label": "Stay category → Design",
                "role": "combobox",
                "value": "Design",
                "current_value": "all",
            },
            {
                "id": "select-nature",
                "kind": "select",
                "node": "select-1",
                "label": "Stay category → Nature",
                "role": "combobox",
                "value": "Nature",
                "current_value": "all",
            },
            {"id": "WAIT", "kind": "wait", "label": "Wait for loading"},
        ],
    }
    wire_calls = []
    with TestClient(app) as client:

        def local_post(url, key, body):
            wire_calls.append(body)
            assert url == "https://api.typesafe.ai/v1/systemone"
            response = client.post(
                "/v1/systemone", json=body, headers={"Authorization": f"Bearer {key}"}
            )
            assert response.status_code == 200, response.text
            return response.json()

        monkeypatch.setattr(upstream_model, "post_json", local_post)
        monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
        monkeypatch.setenv("TYPESAFE_MODEL", "gemma-browser")
        goal = "Find Design stays in Lisbon with Free cancellation, then open Casa Flora."
        decision = upstream_model.choose(page, goal, [])
    assert len(wire_calls) == len(backend.calls) == 1
    assert decision["operation"] == "CLICK"
    assert decision["choice"] == "click-search"
    assert decision["target"] == "1"
    assert decision["target_probabilities"] == {"1": 1.0}
    assert decision["model"] == "gemma-native"
    native = backend.calls[0]
    assert native.state == wire_calls[0]["state"]
    assert [q.id for q in native.questions] == ["operation", "select_target"]
    for question in native.questions:
        original = wire_calls[0]["questions"][question.id]
        assert json.loads(question.instructions) == original["instructions"]
        assert json.loads(question.instructions)["goal"] == goal
        if question.id == "select_target":
            assert {key: json.loads(value) for key, value in question.criteria.items()} == original[
                "criteria"
            ]
    assert set(decision["raw_answers"]) == set(wire_calls[0]["questions"])


def test_opt_in_systemone_accepts_wire_schema_but_decide_stays_strict(wire_payload, request_data):
    backend = CapturingBackend()
    with TestClient(
        create_app(backend, jev_compatible=True, model_identity="gemma-browser")
    ) as client:
        assert client.post("/v1/systemone", json=wire_payload).status_code == 200
        assert client.post("/decide", json=wire_payload).status_code == 422
        assert client.post("/decide", json=request_data).status_code == 200
    assert len(backend.calls) == 2


def test_wire_schema_is_rejected_without_explicit_opt_in(wire_payload):
    backend = CapturingBackend()
    with TestClient(create_app(backend)) as client:
        assert client.post("/v1/systemone", json=wire_payload).status_code == 422
    assert not backend.calls


def test_jev_authentication_precedes_model_call(wire_payload):
    backend = CapturingBackend()
    app = create_app(
        backend, api_key="test-key", jev_compatible=True, model_identity="gemma-browser"
    )
    with TestClient(app) as client:
        for headers in ({}, {"Authorization": "Bearer wrong-key"}):
            assert (
                client.post("/v1/systemone", json=wire_payload, headers=headers).status_code == 401
            )
        assert not backend.calls
        response = client.post(
            "/v1/systemone", json=wire_payload, headers={"Authorization": "Bearer test-key"}
        )
    assert response.status_code == 200
    assert response.json()["answers"]["click_target"]["choice"] == "2"
    assert response.json()["model"] == backend.name


def test_jev_model_mismatch_returns_client_error_without_inference(wire_payload):
    backend = CapturingBackend()
    with TestClient(
        create_app(backend, jev_compatible=True, model_identity="other-model")
    ) as client:
        response = client.post("/v1/systemone", json=wire_payload)
    assert response.status_code == 422
    assert "does not switch models" in response.json()["detail"]
    assert not backend.calls


@pytest.mark.parametrize("payload", [None, [], "page", {}, {"state": "page", "questions": "bad"}])
def test_jev_malformed_objects_are_client_errors(payload):
    backend = CapturingBackend()
    with TestClient(create_app(backend, jev_compatible=True)) as client:
        assert client.post("/v1/systemone", json=payload).status_code == 422
    assert not backend.calls


def test_jev_singleton_operation_is_rejected(wire_payload):
    wire_payload["questions"]["operation"]["criteria"] = {"DONE": "Complete"}
    backend = CapturingBackend()
    with TestClient(
        create_app(backend, jev_compatible=True, model_identity="gemma-browser")
    ) as client:
        assert client.post("/v1/systemone", json=wire_payload).status_code == 422
    assert not backend.calls


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (RequestValidationError("unsupported context"), 422),
        (BackendResponseError("invalid model output"), 502),
        (BackendTransportError("provider unavailable"), 502),
        (BackendTimeoutError("provider timed out"), 504),
    ],
)
def test_jev_backend_failure_statuses(wire_payload, failure, status):
    class Broken(CapturingBackend):
        def predict(self, request):
            raise failure

    with TestClient(
        create_app(Broken(), jev_compatible=True, model_identity="gemma-browser")
    ) as client:
        response = client.post("/v1/systemone", json=wire_payload)
    assert response.status_code == status
    assert response.json() == {"detail": str(failure)}


def test_jev_invalid_distribution_is_gateway_error(wire_payload):
    class Broken(CapturingBackend):
        def predict(self, request):
            result = super().predict(request)
            result["answers"]["operation"]["probabilities"] = {"CLICK": 0.2, "DONE": 0.2}
            return result

    with TestClient(
        create_app(Broken(), jev_compatible=True, model_identity="gemma-browser")
    ) as client:
        response = client.post("/v1/systemone", json=wire_payload)
    assert response.status_code == 502


def test_jev_internal_errors_are_not_misclassified(wire_payload):
    class Broken(CapturingBackend):
        def predict(self, request):
            raise ValueError("internal implementation failure")

    with TestClient(
        create_app(Broken(), jev_compatible=True, model_identity="gemma-browser"),
        raise_server_exceptions=False,
    ) as client:
        response = client.post("/v1/systemone", json=wire_payload)
    assert response.status_code == 500
    assert "internal implementation failure" not in response.text


@pytest.mark.parametrize(
    ("failure_type", "status"),
    [(RequestValidationError, 422), (BackendResponseError, 502), (RuntimeError, 502)],
)
def test_jev_failure_preserves_incurred_generation_evidence(wire_payload, failure_type, status):
    generation = {
        "raw_output": "Pending: submit the visible changes.",
        "model": "gemma-native",
        "revision": "a" * 40,
        "usage": {"input_tokens": 120, "output_tokens": 8},
        "latency_ms": 25.0,
    }
    failure = failure_type("candidate readout failed after generation")
    failure.generation = generation

    class Broken(CapturingBackend):
        def predict(self, request):
            raise failure

    with TestClient(
        create_app(Broken(), jev_compatible=True, model_identity="gemma-browser")
    ) as client:
        response = client.post("/v1/systemone", json=wire_payload)
    assert response.status_code == status
    assert response.json() == {"detail": {"message": str(failure), "generation": generation}}


def test_hierarchical_endpoint_retains_every_original_target(wire_payload):
    wire_payload["questions"]["click_target"]["criteria"] = {
        str(index): {"element": f"[{index}] Item {index}"} for index in range(1, 81)
    }
    backend = CapturingBackend()
    with TestClient(
        create_app(
            backend,
            jev_compatible=True,
            jev_hierarchical=True,
            model_identity="gemma-browser",
        )
    ) as client:
        response = client.post("/v1/systemone", json=wire_payload)
        assert client.post("/decide", json=wire_payload).status_code == 422
    assert response.status_code == 200, response.text
    result = response.json()
    assert list(result["answers"]["click_target"]["probabilities"]) == list(
        wire_payload["questions"]["click_target"]["criteria"]
    )
    assert sum(result["answers"]["click_target"]["probabilities"].values()) == pytest.approx(1.0)
    assert result["hierarchy"]["backend_predict_calls"] == len(backend.calls) == 6
    assert result["hierarchy"]["status"] == "completed"


def test_hierarchy_requires_explicit_compatible_endpoint():
    with pytest.raises(ValueError, match="explicit compatible"):
        create_app(CapturingBackend(), jev_hierarchical=True)


def test_hierarchy_batching_requires_explicit_hierarchy():
    with pytest.raises(ValueError, match="batching requires hierarchical"):
        create_app(CapturingBackend(), jev_compatible=True, jev_hierarchical_batch=True)


def test_hierarchy_api_uses_the_ordered_batch_backend(wire_payload):
    wire_payload["questions"]["click_target"]["criteria"] = {
        str(index): {"element": f"[{index}] Item {index}"} for index in range(1, 81)
    }

    class BatchBackend(CapturingBackend):
        def __init__(self):
            super().__init__()
            self.batches = []

        def predict_batch(self, requests):
            self.batches.append(requests)
            return [UniformBackend.predict(self, request) for request in requests]

    backend = BatchBackend()
    with TestClient(
        create_app(
            backend,
            jev_compatible=True,
            jev_hierarchical=True,
            jev_hierarchical_batch=True,
            model_identity="gemma-browser",
        )
    ) as client:
        response = client.post("/v1/systemone", json=wire_payload)
    assert response.status_code == 200, response.text
    hierarchy = response.json()["hierarchy"]
    assert hierarchy["backend_predict_batch_calls"] == len(backend.batches) == 1
    assert hierarchy["backend_predict_calls"] == len(backend.calls) == 0
    assert len(backend.batches[0]) == 6
    assert hierarchy["stage_batching"]["used"] is True
    assert len(response.json()["answers"]["click_target"]["probabilities"]) == 80


def test_hierarchy_failure_keeps_prior_generation_and_full_stage_trace(wire_payload):
    wire_payload["questions"]["click_target"]["criteria"] = {
        str(index): {"element": f"[{index}] Item {index}"} for index in range(1, 54)
    }
    generation = {"raw_output": "Pending: choose a target.", "output_tokens": 8}

    class Broken(CapturingBackend):
        def predict(self, request):
            self.calls.append(request)
            if len(self.calls) == 3:
                error = BackendResponseError("candidate scoring failed")
                error.generation = generation
                raise error
            return UniformBackend.predict(self, request)

    backend = Broken()
    with TestClient(
        create_app(
            backend,
            jev_compatible=True,
            jev_hierarchical=True,
            model_identity="gemma-browser",
        )
    ) as client:
        response = client.post("/v1/systemone", json=wire_payload)
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["generation"] == generation
    assert detail["hierarchy"]["backend_predict_calls"] == 3
    assert detail["hierarchy"]["status"] == "failed"
    assert detail["hierarchy"]["stages"][2]["error_generation"] == generation
    assert "answers" not in detail
