"""Held generic JevBench callback for a caller-owned native Decision3 provider.

The caller authenticates sources, provider, lease, GPU and supervision. This
module does not load models, touch files or score a benchmark. Generic route
selection is explicit: browser head IDs never activate a delta here. Its
DecisionRequest contract is narrower than the official JevBench schema;
unsupported inputs fail, and complete suite compatibility is not established.
The original callback input is recorded separately from the native provider's
typed serialization, which can legitimately transform criteria representations.
For example, choice lists become mappings and score mappings become lists;
this module does not establish equivalence for a complete official population.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading

from s1.contracts import DecisionRequest
from s1.evaluation.decision3 import Decision3Backend, native_payload, normalize_native_response

SOURCE_ONLY = True
ROUTE = "jevbench-generic-native-v1"
BROWSER_DELTAS = ("operation", "click_target")


def _gate():
    if SOURCE_ONLY is not False:
        raise ValueError("held generic adapter requires independent review")


def _encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def _sha(value):
    return hashlib.sha256(_encode(value)).hexdigest()


class GenericJevBenchAdapter:
    """A serial native callback; caller owns all resources and deadlines.

    `event_sink` receives private request/response evidence. A prepared event
    records completion of the backend and fence, not delivery to the official
    scorer. The outer owner must bind the returned value and official outcome.
    """

    def __init__(
        self, base, *, lease_module, lease, torch, deadline_factory, identity_guard, event_sink
    ):
        _gate()
        if (
            getattr(lease_module, "SOURCE_ONLY", None) is not False
            or not callable(getattr(lease_module, "SerialBackend", None))
            or not all(callable(x) for x in (deadline_factory, identity_guard, event_sink))
        ):
            raise ValueError("reviewed lease and owner callbacks required")
        if type(base) is not Decision3Backend or getattr(base.spec, "adapter", None) != "decision3":
            raise ValueError("original native Decision3 provider required")
        self.base, self.lease = base, lease
        self.serial = lease_module.SerialBackend(base, lease, torch=torch)
        if self.serial.backend is not base or self.serial.lease is not lease:
            raise ValueError("original provider and shared lease required")
        self.deadline_factory, self.identity_guard, self.event_sink = (
            deadline_factory,
            identity_guard,
            event_sink,
        )
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._calls = 0
        self._failed = False

    def _identity(self):
        if self.serial.backend is not self.base or self.serial.lease is not self.lease:
            raise ValueError("provider or lease changed")
        value = self.identity_guard(self.base, self.lease)
        if type(value) is not dict or not value:
            raise ValueError("external owner identity required")
        return _encode(value)

    def _emit(self, event):
        _encode(event)
        self.event_sink(copy.deepcopy(event))

    def predict(self, state, questions):
        _gate()
        with self._state_lock:
            if not self._lock.acquire(blocking=False):
                self._failed = True
                raise ValueError("generic adapter cannot run concurrently")
        owns_lock = True
        call = None
        try:
            if self._failed:
                raise ValueError("generic adapter already failed")
            self._calls += 1
            call = self._calls
            original = copy.deepcopy({"state": state, "questions": questions})
            if type(questions) is not dict or not questions:
                raise ValueError("original named question map required")
            request_sha = _sha(original)
            deadline = self.deadline_factory()
            if not callable(deadline):
                raise ValueError("per-call deadline required")
            deadline()
            identity = self._identity()
            request = DecisionRequest.model_validate(copy.deepcopy(original))
            if [q.id for q in request.questions] != list(questions):
                raise ValueError("original ordered question IDs required")
            expected_native = _encode(native_payload(request))
            base_event = {
                "route": ROUTE,
                "call": call,
                "request_sha256": request_sha,
                "delta_applied": [],
                "delta_bypassed": list(BROWSER_DELTAS),
            }
            self._emit({**base_event, "event": "planned", "request": original})
            events = []

            def evidence(event, data):
                if (
                    len(events) >= 2
                    or event != ("native_preflight", "native_raw_returned")[len(events)]
                ):
                    raise ValueError("ordered native provider evidence required")
                events.append(copy.deepcopy(data))
                self._emit({**base_event, "event": event, "evidence": data})

            deadline()
            response = self.serial.predict(request, deadline=deadline, evidence_sink=evidence)
            if (
                len(events) != 2
                or type(response) is not dict
                or type(response.get("answers")) is not dict
            ):
                raise ValueError("complete native provider response and evidence required")
            if any(_encode(row.get("native_payload")) != expected_native for row in events) or (
                _encode(events[1]) != _encode(response.get("decision3_provider_evidence"))
                or events[0].get("raw_response") is not None
                or _encode(events[0].get("preflight")) != _encode(events[1].get("preflight"))
            ):
                raise ValueError("native request or response evidence differs")
            expected_answers = normalize_native_response(request, events[1].get("raw_response"))[
                "answers"
            ]
            if _encode(expected_answers) != _encode(response["answers"]):
                raise ValueError("normalized response differs from native evidence")
            if list(response["answers"]) != list(questions):
                raise ValueError("native response question IDs or order changed")
            if self._identity() != identity:
                raise ValueError("owner identity changed during call")
            deadline()
            with self._state_lock:
                if self._failed:
                    raise ValueError("generic adapter failed during call")
            self._emit(
                {
                    **base_event,
                    "event": "prepared_not_delivered",
                    "response": response,
                    "response_sha256": _sha(response),
                }
            )
            deadline()
            # Check the latch and release atomically with respect to another
            # caller's nonblocking admission attempt. A rejected overlap must
            # never let this in-flight response reach the scorer as success.
            with self._state_lock:
                if self._failed:
                    raise ValueError("generic adapter failed during call")
                self._lock.release()
                owns_lock = False
                return response
        except BaseException as error:
            with self._state_lock:
                self._failed = True
            if call is not None:
                try:
                    self._emit(
                        {
                            "route": ROUTE,
                            "call": call,
                            "event": "failed",
                            "error_type": type(error).__name__,
                            "delta_applied": [],
                            "delta_bypassed": list(BROWSER_DELTAS),
                        }
                    )
                except BaseException as secondary:
                    if secondary is not error:
                        error.add_note("failure evidence also failed: " + repr(secondary))
            raise
        finally:
            if owns_lock:
                with self._state_lock:
                    self._lock.release()

    def answers(self, state, questions):
        """Official coherence callable shape: named answers only."""
        return self.predict(state, questions)["answers"]
