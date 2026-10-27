"""Source-only loopback HTTP wiring for a caller-owned candidate backend.

The caller supplies the already loaded, lease-wrapped backend, independent owner
identity checks, same-call witness receipt inspection, and private event sink.
The listener does not launch models, own GPU resources, authenticate source
bytes, score tasks, or hard-cancel a blocked forward. A separate supervising
owner must bind and monitor its process, GPU lease and resource lifecycle.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import jev_decision3_browser_wire as wire
from jev_decision3_browser_worker import DeadlineBackend

from s1.jev_decision3_native import JevDecision3NativeAdapter

SOURCE_ONLY = True
MAX_BODY = 1_000_000
MAX_RESPONSE = 16 * 1024**2
TIMEOUT_SECONDS = 20.0


def _need(value, message):
    if not value:
        raise ValueError(message)


def _gate():
    _need(SOURCE_ONLY is False, "held candidate HTTP source requires independent review")


def _encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def finite(value):
        number = float(value)
        _need(math.isfinite(number), "nonfinite JSON number")
        return number

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_float=finite,
        parse_constant=lambda _: _need(False, "nonfinite JSON value"),
    )


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


class CandidateHTTP:
    """Request binding only; all authentication of injected objects is external."""

    def __init__(
        self,
        backend,
        *,
        model_identity,
        token,
        max_requests,
        owner_guard,
        witness_snapshot,
        witness_verify,
        event_sink,
        clock=time.monotonic,
    ):
        _gate()
        _need(
            callable(getattr(backend, "predict", None))
            and getattr(backend, "spec", None) is not None
            and getattr(backend, "lease", None) is not None
            and callable(getattr(backend.lease, "check", None)),
            "caller-owned lease-wrapped backend required",
        )
        _need(type(model_identity) is str and bool(model_identity.strip()), "model alias required")
        _need(
            type(token) is str and re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token),
            "private header-safe token required",
        )
        _need(
            type(max_requests) is int and 1 <= max_requests <= 16, "bounded request count required"
        )
        _need(
            all(
                callable(item)
                for item in (owner_guard, witness_snapshot, witness_verify, event_sink, clock)
            ),
            "owner identity, witness and event callbacks required",
        )
        self.backend, self.model_identity, self.token = backend, model_identity, token
        self.max_requests = max_requests
        self.owner_guard, self.witness_snapshot, self.witness_verify = (
            owner_guard,
            witness_snapshot,
            witness_verify,
        )
        self.event_sink, self.clock = event_sink, clock
        self.calls, self.failed = 0, False
        self.pending = None
        self.lock = threading.Lock()
        self.identity = self._authenticate("initialized")

    def _authenticate(self, phase):
        self.backend.lease.check()
        identity = copy.deepcopy(self.owner_guard(self.backend, phase))
        _need(
            type(identity) is dict and bool(identity), "external owner identity evidence required"
        )
        _encode(identity)
        return identity

    def handle(self, body, *, started):
        """Prepare a response; delivery is separately recorded by the handler."""
        _gate()
        _need(self.lock.acquire(blocking=False), "only one candidate call may run")
        row = None
        try:
            _need(not self.failed, "prior candidate call failed; owner shutdown required")
            _need(self.pending is None, "prior HTTP delivery is still pending")
            _need(
                type(body) is bytes and 0 < len(body) <= MAX_BODY,
                "bounded original HTTP body required",
            )
            _need(self.calls < self.max_requests, "candidate request limit exceeded")
            self.calls += 1
            row = {
                "call": self.calls,
                "http_body_sha256": _sha(body),
                "started_monotonic": started,
                "status": "started",
            }

            def check():
                _need(not self.failed, "candidate service failed")
                elapsed = self.clock() - started
                _need(0 <= elapsed < TIMEOUT_SECONDS, "HTTP cooperative deadline exceeded")

            check()
            before = self.witness_snapshot()
            _need(
                type(before) is list and len(before) <= 32,
                "bounded ordered witness receipt snapshot required",
            )
            before = copy.deepcopy(before)
            _encode(before)
            identity = self._authenticate("before-request")
            _need(_encode(identity) == _encode(self.identity), "owner identity changed")
            payload = _parse(body)
            _need(
                type(payload) is dict and payload.get("model") == self.model_identity,
                "model alias differs",
            )
            check()
            # Outer DeadlineBackend carries the HTTP-entry timer through lock wait,
            # complete candidate forward, witness callbacks and the CUDA fence.
            adapter = JevDecision3NativeAdapter(
                DeadlineBackend(self.backend, check), model_identity=self.model_identity
            )
            legacy = adapter.predict(payload)
            check()
            after = self.witness_snapshot()
            _need(
                type(after) is list
                and len(after) == len(before) + 1
                and _encode(after[:-1]) == _encode(before),
                "one fresh same-call witness receipt required",
            )
            receipt = copy.deepcopy(after[-1])
            _need(type(receipt) is dict and bool(receipt), "complete witness receipt required")
            _need(
                self.witness_verify(
                    copy.deepcopy(receipt), copy.deepcopy(payload), copy.deepcopy(legacy)
                )
                is True,
                "external witness/request/response verification required",
            )
            check()
            result = wire.named_browser_response(payload, legacy)
            response = _encode(result)
            _need(0 < len(response) <= MAX_RESPONSE, "bounded converted HTTP response required")
            check()
            _need(
                _encode(self._authenticate("after-request")) == _encode(self.identity),
                "owner identity changed",
            )
            row.update(
                status="prepared",
                canonical_request_sha256=_sha(_encode(payload)),
                response_sha256=_sha(response),
                witness_receipt=receipt,
                owner_identity=identity,
                scope="prepared response; independent numerical audit and actual delivery pending",
            )
            self.event_sink(copy.deepcopy(row))
            check()
            self.pending = copy.deepcopy(row)
            return response, row
        except BaseException as error:
            self.failed = True
            if row is not None:
                row.update(status="failed", error_type=type(error).__name__)
                try:
                    self.event_sink(copy.deepcopy(row))
                except BaseException as secondary:
                    error.add_note("failure evidence sink also failed: " + repr(secondary))
            raise
        finally:
            self.lock.release()

    def delivery(self, row, *, delivered):
        """Record local socket write outcome; it does not prove client receipt."""
        _gate()
        _need(self.lock.acquire(blocking=False), "candidate call or delivery is active")
        try:
            _need(
                type(delivered) is bool
                and type(row) is dict
                and row.get("status") == "prepared"
                and self.pending is not None
                and _encode(self.pending) == _encode(row),
                "matching pending delivery required",
            )
            result = copy.deepcopy(row)
            elapsed = self.clock() - row["started_monotonic"]
            timely = 0 <= elapsed < TIMEOUT_SECONDS
            result["delivery_elapsed_seconds"] = elapsed
            result["status"] = (
                "socket_write_returned"
                if delivered and timely
                else "socket_write_returned_late"
                if delivered
                else "socket_write_failed"
            )
            if not delivered or not timely:
                self.failed = True
            try:
                self.event_sink(result)
                finalized_elapsed = self.clock() - row["started_monotonic"]
                if delivered and timely and not 0 <= finalized_elapsed < TIMEOUT_SECONDS:
                    self.failed = True
                    late = copy.deepcopy(result)
                    late["status"] = "delivery_evidence_late"
                    late["evidence_elapsed_seconds"] = finalized_elapsed
                    self.event_sink(late)
                self.pending = None
            except BaseException:
                self.failed = True
                raise
        finally:
            self.lock.release()


def loopback_server(service):
    """Create one synchronous IPv4 loopback listener; caller owns lifecycle."""
    _gate()
    _need(isinstance(service, CandidateHTTP), "explicit candidate service required")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def handle(self):
            # Includes request line and headers, even under continuously slow
            # input. Closing this socket does not cancel an already running GPU
            # call; the external owner still supervises the process.
            self._http_started = service.clock()

            def expire_socket():
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

            timer = threading.Timer(TIMEOUT_SECONDS, expire_socket)
            timer.daemon = True
            timer.start()
            try:
                super().handle()
            finally:
                timer.cancel()
                timer.join(timeout=1.0)
                if timer.is_alive():
                    service.failed = True

        def do_POST(self):
            started = self._http_started
            if self.path != "/v1/systemone":
                self.send_error(404)
                return
            supplied = self.headers.get_all("Authorization", [])
            if (
                len(supplied) != 1
                or not supplied[0].isascii()
                or not secrets.compare_digest(supplied[0], "Bearer " + service.token)
            ):
                self.send_error(401)
                return
            prepared = None
            write_returned = False
            try:
                lengths = self.headers.get_all("Content-Length", [])
                _need(
                    len(lengths) == 1
                    and lengths[0].isascii()
                    and lengths[0].isdigit()
                    and not self.headers.get_all("Transfer-Encoding", []),
                    "one unambiguous content length required",
                )
                size = int(lengths[0])
                _need(0 < size <= MAX_BODY, "bounded HTTP body required")
                chunks = []
                remaining_bytes = size
                while remaining_bytes:
                    remaining_time = TIMEOUT_SECONDS - (service.clock() - started)
                    _need(remaining_time > 0, "HTTP read deadline exceeded")
                    self.connection.settimeout(remaining_time)
                    chunk = self.rfile.read(min(65536, remaining_bytes))
                    _need(bool(chunk), "incomplete HTTP body")
                    chunks.append(chunk)
                    remaining_bytes -= len(chunk)
                response, prepared = service.handle(b"".join(chunks), started=started)
                remaining_time = TIMEOUT_SECONDS - (service.clock() - started)
                _need(remaining_time > 0, "HTTP delivery deadline exceeded")
                self.connection.settimeout(remaining_time)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
                self.wfile.flush()
                write_returned = True
                service.delivery(prepared, delivered=True)
            except (Exception, socket.timeout):
                service.failed = True
                if prepared is not None and not write_returned:
                    try:
                        service.delivery(prepared, delivered=False)
                    except Exception:
                        pass
                elif prepared is None:
                    try:
                        self.send_error(400, "Request failed; no action authorized")
                    except OSError:
                        pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    return server
