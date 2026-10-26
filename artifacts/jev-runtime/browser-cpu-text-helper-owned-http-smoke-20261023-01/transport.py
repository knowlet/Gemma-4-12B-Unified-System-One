"""One future owned HTTP probe. Importing this module performs no HTTP/runtime I/O."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
import socket
import sys
import threading
import time
import types
from pathlib import Path

SERVER_SETTINGS = {
    "host": "127.0.0.1",
    "port": 0,
    "http": "h11",
    "loop": "asyncio",
    "ws": "none",
    "lifespan": "off",
    "proxy_headers": False,
    "interface": "asgi3",
    "workers": 1,
    "limit_max_requests": 1,
    "timeout_graceful_shutdown": 1,
    "timeout_keep_alive": 0,
    "access_log": False,
    "log_config": None,
}


def need(value, message):
    if not value:
        raise ValueError(message)


def load_stack(manifest, check_ref):
    """Authenticate selected critical sources, then bind actual package identities."""
    need(manifest["schema"] == "owned-helper-one-shot-http-stack-v1", "HTTP stack schema")
    need(manifest["server_settings"] == SERVER_SETTINGS, "Exact explicit HTTP server policy")
    for row in manifest["sources"]:
        check_ref(row)
    for name, expected in manifest["modules"].items():
        found = importlib.util.find_spec(name)
        need(
            found is not None and str(Path(found.origin).resolve()) == expected["origin"],
            "Selected HTTP package origin: " + name,
        )
    modules = {name: importlib.import_module(name) for name in manifest["modules"]}
    observed = {}
    for name, module in modules.items():
        version = importlib.metadata.version(name) if name == "click" else module.__version__
        row = {"origin": str(Path(module.__file__).resolve()), "version": version}
        need(row == manifest["modules"][name], "Actual selected HTTP identity: " + name)
        observed[name] = row
    return modules["uvicorn"], observed


def load_upstream(manifest, check_ref):
    """Load the two exact files without running upstream Agent/Browser __init__."""
    package_name = "_owned_one_shot_http_upstream"
    need(package_name not in sys.modules, "Fresh isolated upstream client package")
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    loaded = {}
    for name in ("questions.py", "model.py"):
        row = manifest["upstream"][name]
        blob = check_ref(row)
        label = package_name + "." + Path(name).stem
        module = types.ModuleType(label)
        module.__file__, module.__package__ = row["path"], package_name
        sys.modules[label] = module
        exec(compile(blob, row["path"], "exec", dont_inherit=True), vars(module))
        loaded[name] = module
    return loaded["model.py"]


class OneRequest:
    """Only one request reaches the unchanged production application/native arm."""

    def __init__(self, app):
        self.app, self.calls, self.extra = app, 0, 0

    async def __call__(self, scope, receive, send):
        need(scope["type"] == "http", "HTTP only; lifespan and websockets disabled")
        self.calls += 1
        if self.calls != 1:
            self.extra += 1
            body = b'{"detail":"One-shot request already consumed"}'
            await send({"type": "http.response.start", "status": 409, "headers": []})
            await send({"type": "http.response.body", "body": body, "more_body": False})
            return
        await self.app(scope, receive, send)


def observe_client(model, endpoint, key, retain, clock=time.monotonic):
    """Forward the original synchronous post unchanged; record its complete result."""
    client, attempts = model.CLIENT, []
    original = client.post
    need(
        client.timeout.as_dict() == {k: 25 for k in ("connect", "read", "write", "pool")}
        and client._transport._pool._http2 is True,
        "Original upstream Client HTTP2/timeout25 unchanged",
    )

    def post(url, **kwargs):
        need(url == endpoint + "/chat/completions", "Exact one-shot loopback destination")
        need(kwargs.get("headers") == {"Authorization": "Bearer " + key}, "Original bearer")
        record = {"index": len(attempts) + 1, "started_monotonic": clock()}
        attempts.append(record)
        primary = None
        try:
            response = original(url, **kwargs)
            # Original Client.post is non-streaming: it already collected this body.
            body = response.content
            need(type(body) is bytes and len(body) <= 1024 * 1024, "Bounded complete HTTP body")
            record.update(
                full_body_received_monotonic=clock(),
                http_status=response.status_code,
                http_version=response.http_version,
                body_base64=base64.b64encode(body).decode(),
                body_sha256=hashlib.sha256(body).hexdigest(),
                body_size_bytes=len(body),
            )
            return response
        except BaseException as exc:
            primary = exc
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            record["finished_monotonic"] = clock()
            try:
                record["private_evidence_ref"] = retain(record)
            except BaseException as secondary:
                if primary is not None:
                    primary.add_note("HTTP attempt evidence also failed: " + repr(secondary))
                else:
                    raise

    client.post = post
    return attempts, lambda: setattr(client, "post", original)


async def run_one(
    app,
    model,
    uvicorn,
    settings,
    context,
    key,
    retain,
    check,
    *,
    clock=time.monotonic,
    socket_factory=socket.socket,
):
    """Run one real loopback HTTP exchange in future execution, then close it."""
    loop = asyncio.get_running_loop()
    need(settings == SERVER_SETTINGS, "No automatic HTTP backend or expanded request policy")
    need(type(loop).__module__.startswith("asyncio."), "Explicit stdlib asyncio loop")
    wrapped = OneRequest(app)
    listener = socket_factory(socket.AF_INET, socket.SOCK_STREAM)
    server = task = restore = None
    primary = cleanup_error = None
    result = None
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.setblocking(False)
        host, port = listener.getsockname()
        need(host == "127.0.0.1" and type(port) is int and 0 < port < 65536, "Real loopback port")
        options = dict(settings)
        need(options.pop("host") == host and options.pop("port") == 0, "Fixed random-port policy")
        config = uvicorn.Config(wrapped, host=host, port=port, **options)
        config.load()
        need(
            config.http_protocol_class.__module__ == "uvicorn.protocols.http.h11_impl"
            and config.ws_protocol_class is None
            and config.lifespan_class.__module__ == "uvicorn.lifespan.off",
            "Explicit h11/no-websocket/no-lifespan branches",
        )
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve(sockets=[listener]))
        while not server.started:
            check("HTTP-startup")
            need(not task.done(), "Server terminated before ready")
            await asyncio.sleep(0.005)
        ready = clock()
        endpoint = "http://127.0.0.1:" + str(port) + "/v1"
        os.environ.update(
            TEXT_MODEL_BASE_URL=endpoint,
            TEXT_MODEL_API_KEY=key,
            TEXT_MODEL="Qwen/Qwen2.5-0.5B-Instruct",
            TEXT_MODEL_REASONING="none",
        )
        attempts, restore = observe_client(model, endpoint, key, retain, clock)
        caller_thread = threading.get_ident()

        def call_original():
            need(threading.get_ident() != caller_thread, "Sync client must leave server event loop")
            return model.field_text(context)

        started = clock()
        value, metadata = await asyncio.to_thread(call_original)
        returned = clock()
        check("HTTP-original-client-returned")
        need(
            wrapped.calls == 1
            and wrapped.extra == 0
            and len(attempts) == 1
            and attempts[0].get("http_status") == 200
            and attempts[0].get("http_version") == "HTTP/1.1",
            "One complete native HTTP/1.1 request; retry success unqualified",
        )
        full_received = attempts[0]["full_body_received_monotonic"]
        result = {
            "value": value,
            "upstream_metadata": metadata,
            "endpoint": endpoint,
            "ready_monotonic": ready,
            "client_started_monotonic": started,
            "client_full_body_received_monotonic": full_received,
            "client_returned_monotonic": returned,
            "client_full_body_elapsed_seconds": full_received - started,
            "client_elapsed_lt20_observation_only": full_received - started < 20,
            "client_flag_changes_no_score_or_refusal_gate": True,
            "server_request_count": wrapped.calls,
            "extra_requests": wrapped.extra,
            "attempts": [{k: v for k, v in row.items() if k != "body_base64"} for row in attempts],
            "server_settings": settings,
        }
    except BaseException as exc:
        primary = exc
        raise
    finally:
        # Neither cleanup nor client completion changes the original parent deadline.
        if server is not None:
            server.should_exit = True
        operations = []
        if restore is not None:
            operations.append(restore)
        operations.append(model.CLIENT.close)
        for operation in operations:
            try:
                operation()
            except BaseException as secondary:
                if primary is not None:
                    primary.add_note("HTTP cleanup also failed: " + repr(secondary))
                elif cleanup_error is None:
                    cleanup_error = secondary
                else:
                    cleanup_error.add_note("Additional HTTP cleanup failed: " + repr(secondary))
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=2)
            except BaseException as secondary:
                if primary is not None:
                    primary.add_note("Server shutdown also failed: " + repr(secondary))
                elif cleanup_error is None:
                    cleanup_error = secondary
                else:
                    cleanup_error.add_note("Server shutdown also failed: " + repr(secondary))
        try:
            listener.close()
        except BaseException as secondary:
            if primary is not None:
                primary.add_note("Listener close also failed: " + repr(secondary))
            elif cleanup_error is None:
                cleanup_error = secondary
            else:
                cleanup_error.add_note("Listener close also failed: " + repr(secondary))
        if cleanup_error is not None:
            raise cleanup_error
    check("HTTP-after-server-client-cleanup")
    return result
