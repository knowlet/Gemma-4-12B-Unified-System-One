"""Opt-in private capture for the unchanged pinned Jev browser runner.

This source is held pending independent review. Caller owns/authenticates the
runner, process, HTTPX client, private sink, secrets and resource supervision.
The sink receives sensitive bodies, never headers, URLs, or credential values.
HTTPX response.content is decoded content, not raw HTTP transfer framing.
"""

from __future__ import annotations

import functools
import inspect
import json
import threading
from contextlib import contextmanager

SOURCE_ONLY = True
MAX_REQUEST_BYTES = 1_000_000
MAX_RESPONSE_BYTES = 16 * 1024**2
_RUNNER_LOCK = threading.Lock()
_CLIENT_LOCK = threading.Lock()


def _need(value, message):
    if not value:
        raise ValueError(message)


def _gate():
    _need(SOURCE_ONLY is False, "held private transport capture requires independent review")


def _check_body(body, secrets):
    """Refuse configured credentials in raw or JSON-decoded keys and values.

    Preserve duplicate object keys during scanning; a later duplicate must not
    hide an earlier secret. Nonempty non-JSON bodies cannot be reliably checked
    for encoded credentials and are therefore refused instead of recorded.
    """
    if body is None or body == b"":
        return
    for secret in secrets:
        if secret:
            markers = {secret.encode("utf-8"), json.dumps(secret, ensure_ascii=True)[1:-1].encode()}
            _need(
                not any(marker in body for marker in markers),
                "configured credential in body; private capture refused",
            )
    try:

        def invalid(_value):
            raise ValueError("nonfinite JSON value")

        value = json.loads(body, object_pairs_hook=lambda pairs: pairs, parse_constant=invalid)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ValueError(
            "non-JSON body cannot be safely checked; private capture refused"
        ) from None
    pending = [value]
    while pending:
        value = pending.pop()
        if type(value) in (list, tuple):
            pending.extend(value)
        elif type(value) is str:
            _need(
                not any(secret and secret in value for secret in secrets),
                "configured credential in decoded body; private capture refused",
            )


def _event(*, sink, sequence, response, error, secrets):
    """Emit bodies observed by HTTPX; an exception request is not delivery proof."""
    request, response_bytes = None, None
    source = "unavailable"
    if response is not None:
        _need(
            type(response.history) is list and not response.history,
            "redirected responses cannot be attributed to one captured request",
        )
        request = response.request
        response_bytes = response.content
        _need(type(response_bytes) is bytes, "HTTPX response content bytes required")
        source = "response-attached-request"
    elif error is not None:
        try:
            request = error.request
        except (AttributeError, RuntimeError):
            request = None
        if request is not None:
            source = "exception-attached-request-not-delivery-proof"
    request_bytes = None
    if request is not None:
        try:
            request_bytes = request.content
        except (AttributeError, RuntimeError):
            if response is not None:
                raise ValueError("HTTPX request content bytes unavailable") from None
            source = "unavailable"
        if request_bytes is not None:
            _need(type(request_bytes) is bytes, "HTTPX request content bytes required")
    if response is not None:
        _need(type(request_bytes) is bytes, "HTTPX request content bytes unavailable")
    _need(
        request_bytes is None or len(request_bytes) <= MAX_REQUEST_BYTES,
        "private request body limit exceeded",
    )
    _need(
        response_bytes is None or len(response_bytes) <= MAX_RESPONSE_BYTES,
        "private response body limit exceeded",
    )
    _check_body(request_bytes, secrets)
    _check_body(response_bytes, secrets)
    result = sink(
        {
            "sequence": sequence,
            "status": "transport_exception" if error is not None else "response",
            "request_source": source,
            "request_bytes": request_bytes,
            "response_bytes": response_bytes,
            "http_status": response.status_code if response is not None else None,
            "exception_type": type(error).__name__ if error is not None else None,
            "scope": "httpx-observed content; response may be decoded",
        }
    )
    if inspect.iscoroutine(result):
        result.close()
        raise ValueError("private sink must complete synchronously")
    _need(not inspect.isawaitable(result), "private sink must complete synchronously")


@contextmanager
def capture_post(client, *, sink, secrets=()):
    """Wrap one isolated HTTPX client; preserve each call, retry and exception.

    A successful response without complete bytes or a failed sink aborts before
    the caller can act. If the transport itself raises, that original exception
    survives even if its private capture also fails. No retries are added.
    """
    _gate()
    _need(
        callable(sink) and callable(getattr(client, "post", None)),
        "client and private sink required",
    )
    _need(
        type(secrets) in (tuple, list) and all(type(value) is str for value in secrets),
        "explicit credential strings required",
    )
    secrets = tuple(secrets)
    _need(_CLIENT_LOCK.acquire(blocking=False), "capture requires one isolated client")
    try:
        original = client.post
    except BaseException:
        _CLIENT_LOCK.release()
        raise
    in_call = threading.Lock()
    sequence = 0

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        nonlocal sequence
        _need(in_call.acquire(blocking=False), "concurrent captured requests are unsupported")
        try:
            sequence += 1
            try:
                response = original(*args, **kwargs)
            except BaseException as error:
                try:
                    try:
                        error_response = getattr(error, "response", None)
                    except (AttributeError, RuntimeError):
                        error_response = None
                    _event(
                        sink=sink,
                        sequence=sequence,
                        response=error_response,
                        error=error,
                        secrets=secrets,
                    )
                except BaseException as capture_error:
                    try:
                        error.add_note(
                            "Private transport capture failed: " + type(capture_error).__name__
                        )
                    except BaseException:
                        pass
                raise
            try:
                _event(sink=sink, sequence=sequence, response=response, error=None, secrets=secrets)
            except Exception:
                raise RuntimeError("Private transport capture failed; no action executed") from None
            return response
        finally:
            in_call.release()

    primary = None
    try:
        client.post = wrapped
        yield
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            if client.post is not wrapped:
                raise RuntimeError("HTTP client post changed during private capture")
            client.post = original
        except BaseException as error:
            if primary is None:
                raise
            try:
                primary.add_note("Private transport cleanup failed: " + type(error).__name__)
            except BaseException:
                pass
        finally:
            _CLIENT_LOCK.release()


@contextmanager
def attach_to_runner(runner, *, sink):
    """Temporarily inject capture before the pinned runner's own post wrapper.

    Run this inside the same pinned child process as ``runner.run_benchmark``.
    The source-pinning owner must authenticate this module and runner itself.
    Environment-derived/default model credentials are added only to the
    in-memory body refusal check; the sink never receives them.
    """
    _gate()
    _need(
        callable(sink) and callable(getattr(runner, "model_adapter", None)),
        "runner and private sink required",
    )
    _need(_RUNNER_LOCK.acquire(blocking=False), "capture requires one isolated runner")
    try:
        original = runner.model_adapter
    except BaseException:
        _RUNNER_LOCK.release()
        raise

    @contextmanager
    @functools.wraps(original)
    def injected(agent_module, model_module, **kwargs):
        client = getattr(model_module, "CLIENT", None)
        _need(client is not None, "pinned upstream HTTPX client required")
        secrets = (
            kwargs.get("api_key") or "local-benchmark",
            kwargs.get("text_api_key") or "local-benchmark",
            runner.os.environ.get("TYPESAFE_API_KEY", ""),
            runner.os.environ.get("TEXT_MODEL_API_KEY", ""),
        )
        with capture_post(client, sink=sink, secrets=secrets):
            with original(agent_module, model_module, **kwargs) as result:
                yield result

    primary = None
    try:
        runner.model_adapter = injected
        yield
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            if runner.model_adapter is not injected:
                raise RuntimeError("runner adapter changed during private capture")
            runner.model_adapter = original
        except BaseException as error:
            if primary is None:
                raise
            try:
                primary.add_note("Private transport cleanup failed: " + type(error).__name__)
            except BaseException:
                pass
        finally:
            _RUNNER_LOCK.release()
