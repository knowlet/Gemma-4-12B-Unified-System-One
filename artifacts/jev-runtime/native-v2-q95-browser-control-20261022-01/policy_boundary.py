"""Whole20s policy publication boundary; original native candidate stays intact."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from pathlib import Path

SECONDS = 20.0
MAX_BODY = 1_000_000


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write(path, blob):
    with Path(path).open("xb") as stream:
        stream.write(blob)
        stream.flush()
        import os

        os.fsync(stream.fileno())


class ControlState:
    def __init__(self):
        self._lock = threading.Lock()
        self.failure = None
        self.requests = []

    def fail(self, error):
        with self._lock:
            self.failure = self.failure or {
                "type": type(error).__name__,
                "message": "Selected Gemma/Q95 admission refused",
            }

    def healthy(self):
        with self._lock:
            if self.failure is not None:
                raise RuntimeError("Earlier Gemma/Q95 control refusal remains latched")


class PolicyBoundary:
    """No mutation of native candidates; only admitted timely bytes may publish."""

    def __init__(self, app, checker, output, token, control, *, clock=None):
        self.app, self.checker, self.output = app, checker, Path(output)
        self.token, self.control = token, control
        self.clock, self.serial, self.sequence = clock or time.monotonic, asyncio.Lock(), 0

    def __getattr__(self, name):
        return getattr(self.app, name)

    def deadline(self, started):
        if self.clock() - started >= SECONDS:
            raise TimeoutError("Whole Gemma/Q95 policy20s exceeded")

    async def __call__(self, scope, receive, send):
        try:
            return await self.dispatch(scope, receive, send)
        except BaseException as error:
            self.control.fail(error)
            raise

    async def dispatch(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope.get("path") == "/chat/completions":
            # The original prepared NF4 helper is retained as unused evidence.
            # This control must never silently fall back to it.
            self.control.fail(RuntimeError("Unselected local NF4 text route was requested"))
            body = encode({"detail": "This arm selects its separately owned Q95 text helper"})
            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        # The original app remains the sole bearer-auth authority, including
        # duplicate-header semantics. Every policy POST crosses this boundary.
        if scope.get("path") != "/v1/systemone" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        started = self.clock()
        async with self.serial:
            number = self.sequence
            self.sequence += 1
            folder = self.output / f"policy-{number:04d}"
            folder.mkdir(mode=0o700)
            received, messages, before, after = [], [], None, None
            error = None
            candidate_status, candidate_body, candidate_headers = None, b"", []

            async def tracked_receive():
                message = await receive()
                if message["type"] == "http.request":
                    received.append(message.get("body", b""))
                    if sum(map(len, received)) > MAX_BODY:
                        raise ValueError("Bounded original policy request")
                return message

            async def retain(message):
                if message["type"] == "http.response.start":
                    if any(m["type"] == "http.response.start" for m in messages):
                        raise ValueError("One original native response start")
                elif message["type"] == "http.response.body":
                    if (
                        sum(len(m.get("body", b"")) for m in messages)
                        + len(message.get("body", b""))
                        > MAX_BODY
                    ):
                        raise ValueError("Bounded original native candidate")
                else:
                    raise ValueError("Original policy HTTP messages only")
                messages.append(dict(message))

            try:
                self.control.healthy()
                self.deadline(started)
                before = self.checker(self.checker.binding, f"before-native-policy-{number}")
                self.deadline(started)
                await self.app(scope, tracked_receive, retain)
                starts = [m for m in messages if m["type"] == "http.response.start"]
                bodies = [m for m in messages if m["type"] == "http.response.body"]
                if len(starts) != 1 or not bodies or bodies[-1].get("more_body", False):
                    raise ValueError("Complete original native HTTP candidate")
                candidate_status = starts[0]["status"]
                candidate_headers = starts[0].get("headers", [])
                candidate_body = b"".join(m.get("body", b"") for m in bodies)
                write(folder / "native-candidate.bin", candidate_body)
                after = self.checker(self.checker.binding, f"after-native-policy-{number}")
                self.deadline(started)
            except BaseException as cause:
                error = {
                    "type": type(cause).__name__,
                    "message": "Selected Gemma/Q95 policy admission refused",
                }
                self.control.fail(cause)
            status = (
                candidate_status
                if error is None
                else (504 if error["type"] == "TimeoutError" else 503)
            )
            body = candidate_body if error is None else encode({"detail": error})
            request_blob = b"".join(received)
            if request_blob:
                write(folder / "request.bin", request_blob)
            record = {
                "sequence": number,
                "candidate_status": candidate_status,
                "candidate_sha256": hashlib.sha256(candidate_body).hexdigest()
                if candidate_status is not None
                else None,
                "request_sha256": hashlib.sha256(request_blob).hexdigest()
                if request_blob
                else None,
                "helper_before": before,
                "helper_after": after,
                "error": error,
                "whole_seconds": SECONDS,
                "native_candidate_mutated": False,
                "client_delivery_proven": False,
            }
            write(folder / "candidate-admission.json", encode(record))
            # All native body/header construction and candidate evidence precede
            # this final cooperative publication clock. Never release late200.
            if status == 200:
                try:
                    self.deadline(started)
                except TimeoutError as cause:
                    self.control.fail(cause)
                    status, error = (
                        504,
                        {"type": "TimeoutError", "message": "Whole Gemma/Q95 policy20s exceeded"},
                    )
                    body = encode({"detail": error})
            headers = [
                (k, v)
                for k, v in candidate_headers
                if k.lower() not in {b"content-length", b"content-type"}
            ]
            headers += [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ]
            selected = {
                **record,
                "selected_status": status,
                "selected_sha256": hashlib.sha256(body).hexdigest(),
                "error": error,
                "elapsed_seconds": self.clock() - started,
            }
            write(folder / "selected-admission.json", encode(selected))
            refusal = {}
            for code, kind in ((503, "RuntimeError"), (504, "TimeoutError")):
                payload = encode(
                    {"detail": {"type": kind, "message": "Final Gemma/Q95 publication refused"}}
                )
                refusal[code] = (
                    payload,
                    hashlib.sha256(payload).hexdigest(),
                    [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(payload)).encode()),
                    ],
                )
            ledger = {
                "sequence": number,
                "candidate_status": candidate_status,
                "selected_status": status,
                "selected_sha256": hashlib.sha256(body).hexdigest(),
                "helper_before": before,
                "helper_after": after,
                "scope": "ASGI admission/publication attempt; client delivery unproven",
                "client_delivery_proven": False,
            }
            self.control.requests.append(ledger)
            start_message = {"type": "http.response.start", "status": status, "headers": headers}
            refusal_starts = {
                code: {"type": "http.response.start", "status": code, "headers": values[2]}
                for code, values in refusal.items()
            }
            # Linearize the final healthy/deadline admission with fail(). There
            # is no hashing, persistence, header construction or user callback
            # between commit and ASGI response.start. Release the thread lock
            # before awaiting: ASGI flow control may yield to another request.
            # A later failure cannot retract a committed response start; this
            # records ASGI admission, never proof of socket/client delivery.
            with self.control._lock:
                if status == 200:
                    try:
                        if self.control.failure is not None:
                            raise RuntimeError("Earlier Gemma/Q95 control refusal remains latched")
                        self.deadline(started)
                    except (RuntimeError, TimeoutError) as cause:
                        self.control.failure = self.control.failure or {
                            "type": type(cause).__name__,
                            "message": "Selected Gemma/Q95 admission refused",
                        }
                        status = 504 if isinstance(cause, TimeoutError) else 503
                        body, digest, headers = refusal[status]
                        ledger.update(selected_status=status, selected_sha256=digest)
                        start_message = refusal_starts[status]
            await send(start_message)
            await send({"type": "http.response.body", "body": body})
            write(
                folder / "publication-attempt.json",
                encode(
                    {
                        **ledger,
                        "client_delivery_proven": False,
                        "elapsed_seconds": self.clock() - started,
                    }
                ),
            )
