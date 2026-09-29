import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from s1.api import create_app
from s1.backends import UniformBackend


def test_auth_and_validation(request_data):
    client = TestClient(create_app(UniformBackend(), api_key="test-key"))
    assert client.get("/healthz").status_code == 200
    assert client.post("/v1/systemone", json=request_data).status_code == 401
    headers = {"Authorization": "Bearer test-key"}
    for path in ("/v1/systemone", "/decide"):
        response = client.post(path, json=request_data, headers=headers)
        assert response.status_code == 200
        assert response.json()["answers"]["refund"]["noul"] == 0.5
    request_data["questions"]["route"]["criteria"] = []
    assert client.post("/decide", json=request_data, headers=headers).status_code == 422


def test_malformed_named_question_is_client_error(request_data):
    client = TestClient(create_app(UniformBackend()))
    request_data["questions"]["route"] = None
    assert client.post("/decide", json=request_data).status_code == 422
