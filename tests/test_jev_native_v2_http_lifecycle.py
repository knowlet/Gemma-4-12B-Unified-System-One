"""Real socket ingress with synthetic descriptors; no decoder, model or GPU runs."""

import http.client
import json
import socket
import struct
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import jev_native_v2_browser as driver
import pytest
import uvicorn
from test_jev_native_v2_browser import model_fixture

from s1.browser_server_evidence import ServerEvidenceJournal


@contextmanager
def service(tmp_path, *, near_deadline):
    # Charge simulated prior queue time to the original budget. Only the
    # driver's existing clock seam changes; asyncio and sockets use real time.
    calls = 0

    def clock():
        nonlocal calls
        calls += 1
        return 0.0 if calls == 1 or not near_deadline else 19.0

    phases, partial_received = [], threading.Event()
    model = model_fixture()
    driver.CurrentNF4HelperGuard(model)
    journal = ServerEvidenceJournal(tmp_path / "server")
    inner = driver.make_application(
        model,
        None,
        journal,
        SimpleNamespace(check=phases.append),
        tmp_path,
        "synthetic-test-token",
        "raw-json-v1",
        "synthetic system",
        clock=clock,
    )

    async def observed(scope, receive, send):
        async def observe_receive():
            message = await receive()
            if message["type"] == "http.request" and message.get("more_body"):
                partial_received.set()
            return message

        await inner(scope, observe_receive, send)

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    server = uvicorn.Server(
        uvicorn.Config(
            observed,
            log_level="critical",
            lifespan="on",
            http="h11",
            timeout_graceful_shutdown=3,
        )
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
    try:
        thread.start()
        deadline = time.monotonic() + 5
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started, "Real loopback server failed to start"
        yield listener.getsockname(), partial_received, journal, phases
    finally:
        server.should_exit = True
        thread.join(5)
        listener.close()
        assert not thread.is_alive(), "Real loopback server failed to stop"
        assert not server.server_state.tasks, "ASGI request tasks were not released"
        assert not server.server_state.connections, "HTTP connections were not released"


def receive_response(connection):
    with http.client.HTTPResponse(connection) as response:
        response.begin()
        return response.status, response.read()


def assert_healthy(address):
    with socket.create_connection(address, timeout=3) as connection:
        connection.sendall(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        status, body = receive_response(connection)
    assert status == 200 and json.loads(body)["status"] == "ok"


@pytest.mark.parametrize("termination", ["timeout", "disconnect", "reset"])
def test_partial_body_releases_serial_lane_and_preserves_refusal(tmp_path, termination):
    with service(tmp_path, near_deadline=termination == "timeout") as (
        address,
        received,
        journal,
        phases,
    ):
        with socket.create_connection(address, timeout=3) as partial:
            partial.sendall(
                b"POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\n"
                b"Content-Length: 100\r\nContent-Type: application/json\r\n"
                b"Authorization: Bearer synthetic-test-token\r\n\r\n{"
            )
            assert received.wait(3), "Real server did not receive the partial body"
            if termination == "timeout":
                # Keep the incomplete client connected. It must receive the
                # refusal without finishing its body or releasing the socket.
                status, body = receive_response(partial)
                assert status == 504
                assert json.loads(body)["detail"]["type"] == "TimeoutError"
                assert "answers" not in json.loads(body)
            else:
                if termination == "reset":
                    partial.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                partial.close()
            assert_healthy(address)

        exchange = json.loads((tmp_path / "http/0000/exchange.json").read_bytes())
        assert exchange["status"] == (504 if termination == "timeout" else 502)
        assert exchange["error"]["type"] == (
            "TimeoutError" if termination == "timeout" else "ClientDisconnect"
        )
        assert exchange["request"] is None and exchange["api_request_id"] is None
        assert exchange["client_delivery"].startswith("unknown")
        refusal = json.loads((tmp_path / "http/0000/response.bin").read_bytes())
        assert refusal["detail"] == exchange["error"]
        assert not (tmp_path / "http/0000/candidate-response.bin").exists()
        assert not journal.snapshot()["requests"]
        # Only the later health check reaches dispatch/resource checks.
        assert phases == ["before-http-request", "after-http-request"]


def test_complete_body_within_budget_reaches_real_service(tmp_path):
    with service(tmp_path, near_deadline=False) as (address, _received, journal, phases):
        with socket.create_connection(address, timeout=3) as connection:
            connection.sendall(
                b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n"
                b"Connection: close\r\n\r\n{}"
            )
            status, body = receive_response(connection)
        assert status == 200 and json.loads(body)["status"] == "ok"
        assert (tmp_path / "http/0000/request.bin").read_bytes() == b"{}"
        assert not journal.snapshot()["requests"]
        assert phases == ["before-http-request", "after-http-request"]
