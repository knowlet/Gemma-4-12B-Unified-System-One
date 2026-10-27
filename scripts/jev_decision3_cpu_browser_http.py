"""Held CPU browser HTTP core; caller owns supervision and hard cancellation.

The loopback service checks this thread's five-CPU affinity, but cannot prove
CPU quota, memory limits, source authenticity, process cleanup, client receipt,
or cancellation of a blocked native forward. All are external responsibilities.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

SOURCE_ONLY = True
MAX_BODY = 1_000_000
MAX_RESPONSE = 16 * 1024**2
TIMEOUT_SECONDS = 20.0
AFFINITY = frozenset(range(15, 20))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _parse(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            _need(key not in value, "duplicate JSON key")
            value[key] = item
        return value

    def reject_constant(_value):
        raise ValueError("nonfinite JSON")

    def finite(value):
        number = float(value)
        _need(math.isfinite(number), "nonfinite JSON")
        return number

    return json.loads(
        raw, object_pairs_hook=pairs, parse_float=finite, parse_constant=reject_constant
    )


def _encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _affinity():
    _need(os.sched_getaffinity(0) == AFFINITY, "five-CPU affinity required")


class CpuBrowserHTTP:
    """Prepare one request at a time; the caller owns the adapter and listener."""

    def __init__(self, adapter, *, adapter_module, token, max_requests, event_sink, enabled=False):
        _need(SOURCE_ONLY is False and enabled is True, "reviewed CPU HTTP opt-in required")
        _need(
            getattr(adapter_module, "SOURCE_ONLY", None) is False
            and isinstance(adapter, adapter_module.CpuBrowserAdapter)
            and not adapter.failed,
            "explicit released CPU browser adapter required",
        )
        _need(
            type(token) is str and re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token),
            "private token required",
        )
        _need(
            type(max_requests) is int and 1 <= max_requests <= 16, "bounded request count required"
        )
        _need(callable(event_sink), "external event sink required")
        _affinity()
        self.adapter, self.token, self.max_requests, self.event_sink = (
            adapter,
            token,
            max_requests,
            event_sink,
        )
        self._state_lock = threading.Lock()
        self.failed = False
        self.active = False
        self.pending = None
        self.calls = 0

    def fail(self):
        """Latch failure; this cannot stop a running native forward."""
        with self._state_lock:
            self.failed = True

    def _check(self, started):
        with self._state_lock:
            _need(not self.failed, "service failed; external owner cleanup required")
        now = time.monotonic()
        _need(
            type(started) in (int, float)
            and math.isfinite(started)
            and 0 <= now - started < TIMEOUT_SECONDS,
            "cooperative HTTP deadline exceeded; no hard cancellation",
        )
        _affinity()

    def prepare(self, body, *, started):
        """Produce bounded response bytes but make no delivery claim."""
        with self._state_lock:
            _need(not self.failed, "service failed; external owner cleanup required")
            if self.active or self.pending is not None or self.calls >= self.max_requests:
                self.failed = True
                raise ValueError("single request or bounded request count exceeded")
            self.active = True
            self.calls += 1
            call = self.calls
        try:
            self._check(started)
            _need(
                type(body) is bytes and 0 < len(body) <= MAX_BODY, "bounded request bytes required"
            )
            payload = _parse(body)
            _need(type(payload) is dict, "request object required")
            self._check(started)
            value = self.adapter.predict(payload)
            self._check(started)
            response = _encode(value)
            _need(0 < len(response) <= MAX_RESPONSE, "bounded response bytes required")
            receipt = {
                "call": call,
                "request_sha256": hashlib.sha256(body).hexdigest(),
                "response_sha256": hashlib.sha256(response).hexdigest(),
                "status": "prepared_not_delivered",
            }
            self.event_sink(dict(receipt))
            self._check(started)
            with self._state_lock:
                _need(not self.failed, "service failed; external owner cleanup required")
                self.active = False
                self.pending = dict(receipt)
                return response, receipt
        except BaseException:
            with self._state_lock:
                self.failed = True
                self.active = False
            raise

    def delivery(self, receipt, *, delivered, started):
        """Record a local socket write; never assert the client received it."""
        try:
            self._check(started)
            _need(type(delivered) is bool and delivered, "socket write not completed")
            with self._state_lock:
                _need(
                    not self.failed and type(self.pending) is dict and receipt == self.pending,
                    "matching pending response required",
                )
            self.event_sink(
                {**receipt, "status": "socket_write_returned_client_receipt_unverified"}
            )
            self._check(started)
            with self._state_lock:
                _need(not self.failed and self.pending == receipt, "delivery state changed")
                self.pending = None
        except BaseException:
            self.fail()
            raise


def loopback_server(service, *, enabled=False):
    """Bind a loopback socket only; caller must serve, stop and close it."""
    _need(SOURCE_ONLY is False and enabled is True, "reviewed CPU HTTP opt-in required")
    _need(isinstance(service, CpuBrowserHTTP), "explicit CPU HTTP service required")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def handle(self):
            self._started = time.monotonic()
            self._delivered = False

            def expire():
                service.fail()
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

            timer = threading.Timer(TIMEOUT_SECONDS, expire)
            timer.daemon = True
            timer.start()
            try:
                super().handle()
            finally:
                timer.cancel()
                timer.join(timeout=1.0)
                if timer.is_alive() or not self._delivered:
                    service.fail()

        def do_POST(self):
            self.close_connection = True
            started_response = False
            try:
                service._check(self._started)
                _need(self.path == "/v1/systemone", "endpoint path required")
                supplied = self.headers.get_all("Authorization", [])
                _need(
                    len(supplied) == 1
                    and supplied[0].isascii()
                    and secrets.compare_digest(supplied[0], "Bearer " + service.token),
                    "authorization failed",
                )
                lengths = self.headers.get_all("Content-Length", [])
                _need(
                    len(lengths) == 1
                    and lengths[0].isascii()
                    and lengths[0].isdigit()
                    and not self.headers.get_all("Transfer-Encoding", []),
                    "unambiguous content length required",
                )
                size = int(lengths[0])
                _need(0 < size <= MAX_BODY, "bounded body required")
                parts, remaining = [], size
                while remaining:
                    service._check(self._started)
                    left = TIMEOUT_SECONDS - (time.monotonic() - self._started)
                    self.connection.settimeout(left)
                    chunk = self.rfile.read(min(remaining, 65536))
                    _need(bool(chunk), "incomplete body")
                    parts.append(chunk)
                    remaining -= len(chunk)
                response, receipt = service.prepare(b"".join(parts), started=self._started)
                service._check(self._started)
                self.connection.settimeout(TIMEOUT_SECONDS - (time.monotonic() - self._started))
                started_response = True
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
                self.wfile.flush()
                service.delivery(receipt, delivered=True, started=self._started)
                self._delivered = True
            except Exception:
                service.fail()
                if not started_response:
                    try:
                        self.send_response(400)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                    except OSError:
                        pass

    class ExclusiveLoopbackServer(ThreadingMixIn, HTTPServer):
        # Non-daemon handler threads must be closed by the external owner.
        daemon_threads = False
        block_on_close = True

        def __init__(self):
            self._connection_lock = threading.Lock()
            self._connection_active = False
            super().__init__(("127.0.0.1", 0), Handler)

        def process_request(self, connection, client_address):
            with self._connection_lock:
                if self._connection_active:
                    service.fail()
                    self.shutdown_request(connection)
                    return
                with service._state_lock:
                    refused = service.failed
                if refused:
                    self.shutdown_request(connection)
                    return
                self._connection_active = True
            try:
                super().process_request(connection, client_address)
            except BaseException:
                with self._connection_lock:
                    self._connection_active = False
                service.fail()
                self.shutdown_request(connection)
                raise

        def process_request_thread(self, connection, client_address):
            try:
                super().process_request_thread(connection, client_address)
            finally:
                with self._connection_lock:
                    self._connection_active = False

    return ExclusiveLoopbackServer()
