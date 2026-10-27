"""Private capture uses fake HTTPX transport, never a browser or real network."""

import hashlib
import json
from types import SimpleNamespace

import httpx
import jev_browser_benchmark as runner
import jev_browser_private_transport_capture as capture
import pytest


@pytest.fixture(autouse=True)
def reviewed_source(monkeypatch):
    monkeypatch.setattr(capture, "SOURCE_ONLY", False)


def make_model(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = SimpleNamespace(CLIENT=client)

    def post_json(url, key, body):
        for _ in range(3):
            response = client.post(url, json=body, headers={"Authorization": "Bearer " + key})
            if response.status_code in (429, 503, 529):
                continue
            return response.json()
        raise RuntimeError("exhausted")

    model.post_json = post_json
    return model


def test_pinned_runner_opt_in_captures_each_retry_without_headers_or_mutation():
    network = []

    def handler(request):
        network.append(request.content)
        if len(network) == 1:
            return httpx.Response(503, content=b'{"retry":true}')
        return httpx.Response(200, content=b'{"model":"actual","answers":{}}')

    model = make_model(handler)
    agent = SimpleNamespace(field_text=lambda _: None)
    before_file = hashlib.sha256(open(runner.__file__, "rb").read()).hexdigest()
    before_adapter, before_post = runner.model_adapter, model.CLIENT.post
    captured = []
    with capture.attach_to_runner(runner, sink=captured.append):
        with runner.model_adapter(
            agent,
            model,
            endpoint="http://localhost/model",
            model="candidate",
            api_key="private-token-123",
        ) as (records, _):
            result = model.post_json(
                "https://api.typesafe.ai/v1/systemone",
                "private-token-123",
                {"state": {"text": "synthetic"}},
            )
    assert result == {"model": "actual", "answers": {}}
    assert len(network) == len(captured) == 2
    assert [event["sequence"] for event in captured] == [1, 2]
    assert [event["http_status"] for event in captured] == [503, 200]
    assert [event["request_bytes"] for event in captured] == network
    assert captured[0]["response_bytes"] == b'{"retry":true}'
    assert captured[1]["response_bytes"] == b'{"model":"actual","answers":{}}'
    assert all(event["request_source"] == "response-attached-request" for event in captured)
    assert all("headers" not in event and "url" not in event for event in captured)
    assert all(b"private-token-123" not in event["request_bytes"] for event in captured)
    assert len(records[0]["transport_attempts"]) == 2
    assert "private-token-123" not in json.dumps(records)
    assert runner.model_adapter is before_adapter and model.CLIENT.post == before_post
    assert hashlib.sha256(open(runner.__file__, "rb").read()).hexdigest() == before_file
    model.CLIENT.close()


def test_transport_exception_is_preserved_and_attached_request_is_not_delivery_proof():
    captured, failure = [], []

    def handler(request):
        error = httpx.ConnectError("private exception detail", request=request)
        failure.append(error)
        raise error

    model = make_model(handler)
    before = model.CLIENT.post
    with capture.capture_post(model.CLIENT, sink=captured.append, secrets=("not-in-body",)):
        with pytest.raises(httpx.ConnectError) as caught:
            model.CLIENT.post("https://example.invalid", json={"sample": "invented"})
    assert caught.value is failure[0]
    assert len(captured) == 1 and captured[0]["status"] == "transport_exception"
    assert captured[0]["request_source"] == "exception-attached-request-not-delivery-proof"
    assert captured[0]["request_bytes"] == b'{"sample":"invented"}'
    assert captured[0]["response_bytes"] is None and captured[0]["exception_type"] == "ConnectError"
    assert "private exception detail" not in repr(captured)
    assert model.CLIENT.post == before
    model.CLIENT.close()


def test_capture_failure_preserves_transport_exception():
    original_error = httpx.ConnectError("original")

    def failed(*_args, **_kwargs):
        raise original_error

    client = SimpleNamespace(post=failed)

    def bad_sink(_event):
        raise RuntimeError("private details")

    with capture.capture_post(client, sink=bad_sink):
        with pytest.raises(httpx.ConnectError) as caught:
            client.post("https://example.invalid")
    assert caught.value is original_error
    assert caught.value.__notes__ == ["Private transport capture failed: RuntimeError"]
    assert client.post is failed


def test_exception_with_response_captures_bodies_and_preserves_exception():
    request = httpx.Request("POST", "https://example.invalid", content=b'"synthetic"')
    response = httpx.Response(503, request=request, content=b'{"error":"unavailable"}')
    original_error = httpx.HTTPStatusError("status", request=request, response=response)

    def failed(*_args, **_kwargs):
        raise original_error

    client, captured = SimpleNamespace(post=failed), []
    with capture.capture_post(client, sink=captured.append):
        with pytest.raises(httpx.HTTPStatusError) as caught:
            client.post("ignored")
    assert caught.value is original_error
    assert captured[0]["status"] == "transport_exception"
    assert captured[0]["http_status"] == 503
    assert captured[0]["request_bytes"] == b'"synthetic"'
    assert captured[0]["response_bytes"] == b'{"error":"unavailable"}'


def test_async_sink_is_refused_before_successful_response_returns():
    async def async_sink(_event):
        return None

    model = make_model(lambda _request: httpx.Response(200, content=b"{}"))
    with capture.capture_post(model.CLIENT, sink=async_sink):
        with pytest.raises(RuntimeError, match="Private transport capture failed"):
            model.CLIENT.post("https://example.invalid", json={"synthetic": True})
    model.CLIENT.close()


@pytest.mark.parametrize("problem", ["sink", "missing_request", "credential_body"])
def test_successful_transport_without_safe_capture_aborts_before_action(problem):
    captured, sent = [], []

    def handler(request):
        sent.append(request.content)
        if problem == "credential_body":
            return httpx.Response(200, content=b'{"model":"private-token-123"}')
        return httpx.Response(200, content=b'{"model":"actual"}')

    model = make_model(handler)
    if problem == "missing_request":
        model.CLIENT.post = lambda *_args, **_kwargs: httpx.Response(200, content=b"{}")

    def sink(event):
        if problem == "sink":
            raise RuntimeError("do not log this")
        captured.append(event)

    with capture.capture_post(model.CLIENT, sink=sink, secrets=("private-token-123",)):
        with pytest.raises(
            RuntimeError, match="Private transport capture failed; no action executed"
        ):
            model.CLIENT.post("https://example.invalid", json={"state": "synthetic"})
    assert captured == []
    assert len(sent) <= 1
    model.CLIENT.close()


def test_no_install_means_no_capture_and_source_gate():
    assert runner.model_adapter is not None
    captured = []
    client = SimpleNamespace(post=lambda *_args, **_kwargs: "response")
    assert client.post("untouched") == "response" and captured == []
    capture.SOURCE_ONLY = True
    try:
        with pytest.raises(ValueError, match="independent review"):
            with capture.capture_post(client, sink=captured.append):
                pass
    finally:
        capture.SOURCE_ONLY = False


@pytest.mark.parametrize(
    "location", ["request-value", "request-key", "response-value", "response-key", "duplicate-key"]
)
def test_partially_escaped_credentials_never_reach_private_sink(location):
    secret = "private-token-123"
    escaped = b"private-t\\u006fken-123"
    request_body = b'{"safe":"ok"}'
    response_body = b'{"safe":"ok"}'
    if location == "request-value":
        request_body = b'{"value":"' + escaped + b'"}'
    elif location == "request-key":
        request_body = b'{"' + escaped + b'":"value"}'
    elif location == "response-value":
        response_body = b'{"value":"' + escaped + b'"}'
    elif location == "response-key":
        response_body = b'{"' + escaped + b'":"value"}'
    else:
        response_body = b'{"duplicate":"' + escaped + b'","duplicate":"safe"}'
    assert secret.encode() not in request_body + response_body
    captured = []
    model = make_model(lambda _request: httpx.Response(200, content=response_body))
    with capture.capture_post(model.CLIENT, sink=captured.append, secrets=(secret,)):
        with pytest.raises(RuntimeError, match="Private transport capture failed"):
            model.CLIENT.post("https://example.invalid", content=request_body)
    assert captured == []
    model.CLIENT.close()


@pytest.mark.parametrize("side", ["request", "response"])
def test_non_json_body_is_refused_before_it_can_reach_sink(side):
    model = make_model(
        lambda _request: httpx.Response(200, content=b"plain" if side == "response" else b"{}")
    )
    captured = []
    with capture.capture_post(model.CLIENT, sink=captured.append, secrets=("private-token-123",)):
        with pytest.raises(RuntimeError, match="Private transport capture failed"):
            model.CLIENT.post(
                "https://example.invalid", content=b"plain" if side == "request" else b"{}"
            )
    assert captured == []
    model.CLIENT.close()


def test_follow_redirects_fails_closed_without_recording_final_only():
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(302, headers={"Location": "/next"}, content=b"{}")
        return httpx.Response(200, content=b"{}")

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    captured = []
    with capture.capture_post(client, sink=captured.append):
        with pytest.raises(RuntimeError, match="Private transport capture failed"):
            client.post("https://example.invalid/start", json={"synthetic": True})
    assert len(seen) == 2 and captured == []
    client.close()


def test_exception_response_with_redirect_preserves_exception_and_refuses_partial_capture():
    request = httpx.Request("POST", "https://example.invalid", content=b"{}")
    previous = httpx.Response(302, request=request, content=b"{}")
    response = httpx.Response(503, request=request, content=b"{}", history=[previous])
    original_error = httpx.HTTPStatusError("status", request=request, response=response)

    def failed(*_args, **_kwargs):
        raise original_error

    client, captured = SimpleNamespace(post=failed), []
    with capture.capture_post(client, sink=captured.append):
        with pytest.raises(httpx.HTTPStatusError) as caught:
            client.post("ignored")
    assert caught.value is original_error and captured == []
    assert caught.value.__notes__ == ["Private transport capture failed: ValueError"]


@pytest.mark.parametrize("target", ["client", "runner"])
@pytest.mark.parametrize("primary_error", [False, True])
def test_cleanup_conflict_preserves_primary_and_never_overwrites_external_change(
    target, primary_error
):
    def replacement(*_a, **_k):
        return None

    client = SimpleNamespace(post=lambda *_a, **_k: None)
    original_adapter = runner.model_adapter
    context = (
        capture.capture_post(client, sink=lambda _event: None)
        if target == "client"
        else capture.attach_to_runner(runner, sink=lambda _event: None)
    )
    container, attribute = (client, "post") if target == "client" else (runner, "model_adapter")
    error = KeyError("primary")
    expected = KeyError if primary_error else RuntimeError
    try:
        with pytest.raises(expected) as caught:
            with context:
                setattr(container, attribute, replacement)
                if primary_error:
                    raise error
        assert getattr(container, attribute) is replacement
        if primary_error:
            assert caught.value is error
            assert error.__notes__ == ["Private transport cleanup failed: RuntimeError"]
    finally:
        runner.model_adapter = original_adapter


@pytest.mark.parametrize("side", ["request", "response"])
def test_oversized_body_never_reaches_private_sink(side):
    request_body = b"x" * (capture.MAX_REQUEST_BYTES + 1) if side == "request" else b"{}"
    response_body = b"x" * (capture.MAX_RESPONSE_BYTES + 1) if side == "response" else b"{}"
    captured = []
    model = make_model(lambda _request: httpx.Response(200, content=response_body))
    with capture.capture_post(model.CLIENT, sink=captured.append):
        with pytest.raises(RuntimeError, match="Private transport capture failed"):
            model.CLIENT.post("https://example.invalid", content=request_body)
    assert captured == []
    model.CLIENT.close()


def test_oversized_exception_request_keeps_original_transport_exception():
    request = httpx.Request(
        "POST", "https://example.invalid", content=b"x" * (capture.MAX_REQUEST_BYTES + 1)
    )
    error = httpx.ConnectError("original", request=request)

    def fail(*_args, **_kwargs):
        raise error

    captured = []
    client = SimpleNamespace(post=fail)
    with capture.capture_post(client, sink=captured.append):
        with pytest.raises(httpx.ConnectError) as caught:
            client.post("ignored")
    assert caught.value is error
    assert error.__notes__ == ["Private transport capture failed: ValueError"]
    assert captured == []


def test_exception_request_with_escaped_credential_preserves_error_without_saving():
    request = httpx.Request("POST", "https://example.invalid", content=b'{"x":"s\\u0065cret123"}')
    error = httpx.ConnectError("original", request=request)

    def fail(*_args, **_kwargs):
        raise error

    client, captured = SimpleNamespace(post=fail), []
    with capture.capture_post(client, sink=captured.append, secrets=("secret123",)):
        with pytest.raises(httpx.ConnectError) as caught:
            client.post("ignored")
    assert caught.value is error
    assert error.__notes__ == ["Private transport capture failed: ValueError"]
    assert captured == []


@pytest.mark.parametrize("body", [b'{"number":NaN}', b'{"number":Infinity}'])
def test_nonfinite_json_is_refused_before_capture(body):
    captured = []
    model = make_model(lambda _request: httpx.Response(200, content=body))
    with capture.capture_post(model.CLIENT, sink=captured.append):
        with pytest.raises(RuntimeError, match="Private transport capture failed"):
            model.CLIENT.post("https://example.invalid", json={"state": "synthetic"})
    assert captured == []
    model.CLIENT.close()
