"""Serving the actual Jev wire request through the public CLI boundary."""

import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from s1.cli import main


@pytest.mark.parametrize("compatible", [False, True])
def test_cli_serves_structured_jev_request_only_when_enabled(monkeypatch, compatible):
    payload = {
        "model": "browser-policy",
        "state": {"page": {"text": "Task awaiting confirmation"}},
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": {"goal": "Confirm task", "rules": "Stop only once confirmed"},
                "criteria": {"CLICK": "Click a control", "DONE": "Every requirement is visible"},
            },
            "click_target": {
                "type": "choice",
                "instructions": {"goal": "Confirm task", "operation": "CLICK"},
                "criteria": {"1": {"element": "[1] Confirm", "current_value": ""}},
            },
        },
    }
    observed = []

    def run(application, *, host, port):
        assert host == "127.0.0.1"
        with TestClient(application) as client:
            response = client.post("/v1/systemone", json=payload)
            observed.append(response)
            assert client.post("/decide", json=payload).status_code == 422

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))
    monkeypatch.delenv("S1_API_KEY", raising=False)
    arguments = ["serve", "--backend", "uniform"]
    if compatible:
        arguments += ["--jev-compatible", "--jev-model-identity", "browser-policy"]
    assert main(arguments) == 0
    assert observed[0].status_code == (200 if compatible else 422)
    if compatible:
        assert observed[0].json()["answers"]["click_target"]["choice"] == "1"
        assert observed[0].json()["model"] == "uniform"


@pytest.mark.parametrize("option", ["--jev-hierarchical", "--jev-batch-stages"])
def test_cli_rejects_missing_hierarchy_prerequisites_before_backend_loading(option):
    with pytest.raises(SystemExit) as error:
        main(["serve", "--backend", "uniform", option])
    assert error.value.code == 2
