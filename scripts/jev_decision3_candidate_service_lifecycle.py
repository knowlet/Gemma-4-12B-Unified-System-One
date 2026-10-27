"""Source-only lifecycle for a caller-owned persistent candidate service.

The caller authenticates the supplied HTTP module, loaded backend, GPU lease,
witnesses and supervisor. This facade owns only a loopback listener and its
request thread. Stop cannot cancel a blocked GPU forward: a separate supervising
process must enforce resource limits and terminate only its own invocation.
"""

from __future__ import annotations

import copy
import errno
import math
import threading

SOURCE_ONLY = True
MAX_REQUESTS = 16


def _need(value, message):
    if not value:
        raise ValueError(message)


def _gate():
    _need(SOURCE_ONLY is False, "held service lifecycle requires independent review")


def _close_listener(server):
    server.server_close()
    _need(server.fileno() == -1, "listener socket closure unconfirmed")


class CandidateServiceLifecycle:
    """One listener for up to 16 calls; cleanup follows verified thread closure.

    This is not a GPU owner or a hard-cancellation mechanism. ``supervisor`` is
    cooperative; a separate process must independently enforce the host budget.
    ``cleanup`` must close only resources the caller owns, and return nonempty
    JSON evidence. A blocked thread or unconfirmed lease must prevent cleanup.
    """

    def __init__(
        self,
        http,
        service,
        *,
        supervisor,
        closure_guard,
        cleanup,
        event_sink,
        poll_seconds=0.1,
    ):
        _gate()
        _need(
            getattr(http, "SOURCE_ONLY", None) is False
            and isinstance(service, http.CandidateHTTP)
            and callable(getattr(http, "loopback_server", None)),
            "independently authenticated released HTTP service required",
        )
        _need(
            type(service.max_requests) is int
            and 1 <= service.max_requests <= MAX_REQUESTS
            and service.calls == 0
            and service.failed is False
            and service.pending is None,
            "fresh service with bounded request count required",
        )
        _need(
            all(callable(item) for item in (supervisor, closure_guard, cleanup, event_sink)),
            "external supervisor, closure guard, cleanup and event sink required",
        )
        _need(
            type(poll_seconds) in (int, float)
            and math.isfinite(poll_seconds)
            and 0 < poll_seconds <= 0.5,
            "bounded listener poll interval required",
        )
        self.http, self.service = http, service
        self.supervisor, self.closure_guard = supervisor, closure_guard
        self.cleanup, self.event_sink = cleanup, event_sink
        self.poll_seconds = float(poll_seconds)
        self.lock = threading.RLock()
        self.stop_requested = threading.Event()
        self.done = threading.Event()
        self.server = self.thread = None
        self.status = "created"
        self.reason = self.failure = self.cleanup_evidence = None
        self.cleanup_attempted = False
        self.cleanup_finished = False

    def _emit(self, event):
        self.http._encode(event)
        self.event_sink(copy.deepcopy(event))

    def start(self):
        _gate()
        with self.lock:
            _need(self.status == "created", "single start required")
            server = None
            try:
                self.supervisor()
                server = self.http.loopback_server(self.service)
                self.server = server
                host, port = server.server_address
                _need(
                    host == "127.0.0.1" and type(port) is int and 0 < port <= 65535,
                    "loopback listener required",
                )
                server.timeout = self.poll_seconds
                self.thread = threading.Thread(
                    target=self._serve, name="candidate-service", daemon=True
                )
                self.status = "running"
                self._emit(
                    {
                        "event": "listening",
                        "address": host,
                        "port": port,
                        "max_requests": self.service.max_requests,
                    }
                )
                self.thread.start()
                return host, port
            except BaseException as error:
                self.stop_requested.set()
                if server is not None:
                    try:
                        _close_listener(server)
                    except BaseException as close_error:
                        error.add_note("Listener close also failed: " + repr(close_error))
                self.status = "failed"
                self.failure = "start_failed"
                if self.thread is None or self.thread.ident is None:
                    self.thread = None
                    self.done.set()
                raise

    def _serve(self):
        reason = "requested"
        try:
            while not self.stop_requested.is_set():
                self.supervisor()
                if self.service.failed:
                    reason = "request_failed"
                    break
                if self.service.calls >= self.service.max_requests:
                    reason = "request_limit"
                    break
                try:
                    self.server.handle_request()
                except OSError as error:
                    if (
                        self.stop_requested.is_set()
                        and self.server.fileno() == -1
                        and error.errno in (errno.EBADF, errno.EINVAL)
                    ):
                        break
                    raise
                self.supervisor()
                if self.service.failed:
                    reason = "request_failed"
                    break
        except BaseException as error:
            reason = "supervisor_or_listener_failed"
            self.failure = type(error).__name__
        finally:
            try:
                _close_listener(self.server)
            except BaseException as error:
                reason = "listener_close_failed"
                self.failure = type(error).__name__
            with self.lock:
                self.reason = reason
                self.status = (
                    "listener_closed" if reason in ("requested", "request_limit") else "failed"
                )
                event = {
                    "event": "listener_closed",
                    "reason": reason,
                    "calls": self.service.calls,
                    "request_failed": self.service.failed,
                    "failure_type": self.failure,
                }
            try:
                self._emit(event)
            except BaseException as error:
                with self.lock:
                    self.failure = type(error).__name__
                    self.status = "failed"
            finally:
                self.done.set()

    def snapshot(self):
        with self.lock:
            return {
                "status": self.status,
                "reason": self.reason,
                "request_calls": self.service.calls,
                "request_failed": self.service.failed,
                "failure_type": self.failure,
                "thread_alive": bool(self.thread and self.thread.is_alive()),
                "cleanup_attempted": self.cleanup_attempted,
                "cleanup_finished": self.cleanup_finished,
                "cleanup_evidence": copy.deepcopy(self.cleanup_evidence),
            }

    def stop(self, *, timeout_seconds=3.0):
        """Stop accepting, join the listener and invoke caller cleanup once.

        The timeout bounds only the listener thread join. Lock acquisition,
        closure guard, cleanup and event sink are not hard-timed. An independent
        process owner must enforce the full deadline. A join timeout leaves
        cleanup unattempted; re-observe or terminate that exact invocation.
        """
        _gate()
        _need(
            type(timeout_seconds) in (int, float)
            and math.isfinite(timeout_seconds)
            and 0 < timeout_seconds <= 30,
            "bounded listener join timeout required",
        )
        with self.lock:
            _need(self.status != "created", "service has not started")
            if self.cleanup_attempted:
                _need(self.cleanup_finished, "prior cleanup failed")
                return self.snapshot()
            self.stop_requested.set()
            thread = self.thread
            server = self.server
        if server is not None:
            _close_listener(server)
        if thread is not None:
            _need(thread is not threading.current_thread(), "listener cannot join itself")
            thread.join(timeout_seconds)
            if thread.is_alive():
                raise TimeoutError(
                    "listener still running; cleanup and process closure unconfirmed"
                )
        with self.lock:
            if self.cleanup_attempted:
                _need(self.cleanup_finished, "prior cleanup failed")
                return self.snapshot()
            _need(
                self.done.is_set() and self.service.pending is None,
                "listener or delivery closure unconfirmed",
            )
            self.closure_guard()
            self.cleanup_attempted = True
            try:
                evidence = self.cleanup()
                _need(
                    type(evidence) is dict and bool(evidence),
                    "explicit owned cleanup evidence required",
                )
                self.http._encode(evidence)
                self.cleanup_evidence = copy.deepcopy(evidence)
                self._emit({"event": "cleanup_completed", "evidence": self.cleanup_evidence})
                self.cleanup_finished = True
            except BaseException as error:
                self.failure = type(error).__name__
                self.status = "failed"
                raise
            if self.status != "failed":
                self.status = "stopped"
            return self.snapshot()
