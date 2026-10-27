"""Held CPU Decision3 browser response adapter, without a service or owner.

The caller owns a source-authenticated CPU publisher runtime, exclusive access,
resource admission, request delivery and hard process cancellation. This module
makes no claim about those properties, numeric equivalence, or browser success.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time

import jev_decision3_browser_wire as wire

from s1.errors import BackendTimeoutError
from s1.evaluation.decision3 import CONTEXT_LIMIT, native_payload, normalize_native_response
from s1.jev_decision3_native import PROFILE, Decision3NativeRequest, parse_native_payload
from s1.jev_generated import OPERATIONS, TARGETS, _request
from s1.jev_hierarchy import _criteria

SOURCE_ONLY = True
BUDGET_SECONDS = 20.0
EXPECTED_PUBLISHER = "d3-edge"


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _parse_request(payload, identity):
    """Use the native parser; repair its representative-only singleton validation.

    The temporary extra label is used only for the existing outer validator.
    All original heads are validated independently and only original data reach
    the publisher; no answer or candidate is added to the real request.
    """
    if type(payload) is dict:
        _need(len(_encode(payload)) <= 1_000_000, "bounded original request required")
    questions = payload.get("questions") if type(payload) is dict else None
    if type(questions) is not dict:
        return parse_native_payload(payload, model_identity=identity)
    singletons = []
    for key, raw in questions.items():
        if type(raw) is not dict:
            return parse_native_payload(payload, model_identity=identity)
        alias, _original, labels = _criteria(raw, key)
        if labels is not None and len(labels) == 1:
            singletons.append((key, alias, labels))
    if not singletons:
        return parse_native_payload(payload, model_identity=identity)
    original = copy.deepcopy(payload)
    source = _encode(original)
    _need(len(source) <= 1_000_000, "bounded original request required")
    offered = {}
    for key, raw in questions.items():
        _need(type(raw) is dict and raw.get("type", "choice") == "choice", "choice heads required")
        _alias, _value, labels = _criteria(raw, key)
        _need(
            type(labels) is dict
            and 1 <= len(labels) <= (len(OPERATIONS) if key == "operation" else 255)
            and all(
                len(label) <= 128 and len(description) <= 16000
                for label, description in labels.items()
            ),
            "complete bounded original options required",
        )
        offered[key] = labels
    _need(
        "operation" in offered
        and not set(offered["operation"]) - OPERATIONS
        and not (set(offered) - {"operation", *TARGETS.values()})
        and all(
            (operation in offered["operation"]) == (target in offered)
            for operation, target in TARGETS.items()
        ),
        "operation and target heads must agree",
    )
    representative = copy.deepcopy(original)
    for key, alias, labels in singletons:
        extra = (
            ("WAIT" if "WAIT" not in labels else "DONE")
            if key == "operation"
            else "__cpu_validation_placeholder__"
        )
        if extra in labels and key != "operation":
            extra = "__cpu_validation_placeholder_2__"
        _need(extra not in labels, "distinct temporary validation label required")
        criteria = representative["questions"][key][alias]
        if type(criteria) is dict:
            criteria[extra] = "synthetic validation placeholder"
        else:
            _need(type(criteria) is list, "supported criteria required")
            criteria.append(extra)
    _request(representative, identity)
    data = copy.deepcopy(original)
    converted = []
    for key, raw in data["questions"].items():
        entry = copy.deepcopy(raw)
        entry["id"] = key
        for alias in ("options", "levels"):
            if alias in entry:
                entry["criteria"] = entry.pop(alias)
        converted.append(entry)
    data["questions"] = converted
    parsed = Decision3NativeRequest.model_validate(data)
    return parsed, original, source, {key: list(labels) for key, labels in offered.items()}


def _runtime(runtime):
    _need(
        str(getattr(runtime, "device", "")) == "cpu"
        and type(getattr(runtime, "model_name", None)) is str
        and runtime.model_name == EXPECTED_PUBLISHER
        and type(getattr(runtime, "batch_size", None)) is int
        and runtime.batch_size == 8
        and getattr(runtime, "fast", None) is None
        and getattr(runtime, "permutation_average", None) is False
        and callable(getattr(runtime, "prepare", None))
        and callable(getattr(runtime, "system_one", None)),
        "explicit publisher CPU runtime required",
    )


def _preflight(prepared, request):
    ids = [question.id for question in request.questions]
    _need(
        type(getattr(prepared, "keys", None)) is list
        and prepared.keys == ids
        and type(getattr(prepared, "errors", None)) is dict
        and not prepared.errors
        and type(getattr(prepared, "questions", None)) is dict
        and set(prepared.questions) == set(ids)
        and type(getattr(prepared, "sequences", None)) is dict
        and set(prepared.sequences) == set(ids)
        and not getattr(prepared, "images", None)
        and not getattr(prepared, "videos", None)
        and not getattr(prepared, "reversed_sequences", None),
        "complete ordered text-only preflight required",
    )
    lengths = {}
    for question in request.questions:
        sequence = prepared.sequences[question.id]
        item = prepared.questions[question.id]
        _need(
            type(sequence) is list
            and 1 <= len(sequence) <= CONTEXT_LIMIT
            and all(type(token) is int and 0 <= token < 2**63 for token in sequence)
            and getattr(item, "kind", None) == "choice"
            and type(getattr(item, "keys", None)) is list
            and item.keys == question.labels(),
            "complete native head and bounded context required",
        )
        lengths[question.id] = len(sequence)
    return lengths


def _raw_answers(raw, request, lengths):
    ids = [question.id for question in request.questions]
    _need(
        type(raw) is dict
        and raw.get("model") == EXPECTED_PUBLISHER
        and type(raw.get("model")) is str
        and type(raw.get("answers")) is dict
        and list(raw["answers"]) == ids
        and type(raw.get("usage")) is dict
        and type(raw["usage"].get("input_tokens")) is int
        and raw["usage"]["input_tokens"] == sum(lengths.values())
        and type(raw["usage"].get("output_tokens")) is int
        and raw["usage"]["output_tokens"] == 0,
        "complete publisher identity, ordered answers and usage required",
    )
    for question in request.questions:
        answer = raw["answers"][question.id]
        labels = question.labels()
        _need(
            type(answer) is dict
            and set(answer) == {"type", "choice", "probabilities", "confidence"}
            and answer["type"] == "choice"
            and type(answer["probabilities"]) is dict
            and list(answer["probabilities"]) == labels,
            "complete ordered publisher choice required",
        )
        values = list(answer["probabilities"].values())
        _need(
            all(
                type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                for value in values
            )
            and math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6)
            and answer["choice"] == labels[values.index(max(values))]
            and type(answer["confidence"]) in (int, float)
            and math.isfinite(answer["confidence"])
            and 0 <= answer["confidence"] <= 1,
            "valid complete publisher probabilities required",
        )
        confidence = (
            1.0
            if len(values) < 2
            else max(
                0.0,
                min(
                    1.0,
                    1
                    + sum(value * math.log(value) for value in values if value > 0)
                    / math.log(len(values)),
                ),
            )
        )
        _need(
            math.isclose(answer["confidence"], confidence, rel_tol=1e-12, abs_tol=1e-12),
            "publisher confidence differs from probabilities",
        )


class CpuBrowserAdapter:
    """One CPU runtime; caller guarantees exclusive access outside this object."""

    def __init__(self, runtime, *, model_identity, enabled=False, clock=time.monotonic):
        _need(SOURCE_ONLY is False and enabled is True, "reviewed CPU adapter opt-in required")
        _runtime(runtime)
        _need(
            type(model_identity) is str and model_identity.strip() and len(model_identity) <= 256,
            "explicit browser model alias required",
        )
        _need(callable(clock), "monotonic clock required")
        self.runtime, self.model_identity, self.clock = runtime, model_identity, clock
        self.lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.failed = False

    def predict(self, payload):
        with self._state_lock:
            _need(not self.failed, "adapter failed; external owner cleanup required")
            if not self.lock.acquire(blocking=False):
                self.failed = True
                raise ValueError("exclusive adapter request required; adapter failed")
        released = False
        try:
            start = self.clock()
            _need(
                type(start) in (int, float) and math.isfinite(start),
                "valid monotonic time required",
            )

            def check():
                with self._state_lock:
                    _need(not self.failed, "adapter failed; external owner cleanup required")
                now = self.clock()
                _need(
                    type(now) in (int, float) and math.isfinite(now) and now >= start,
                    "valid monotonic time required",
                )
                if now - start >= BUDGET_SECONDS:
                    raise BackendTimeoutError(
                        "cooperative 20-second deadline exceeded; forward is not hard-cancelled; "
                        "external owner must control the process"
                    )

            check()
            _runtime(self.runtime)
            parsed, original, source, _labels = _parse_request(payload, self.model_identity)
            original_bytes = _encode(original)
            _need(
                hashlib.sha256(original_bytes).digest() == hashlib.sha256(source).digest(),
                "original request serialization differs",
            )
            native = native_payload(parsed)
            native_bytes = _encode(native)
            check()
            prepare_input = copy.deepcopy(native)
            prepared = self.runtime.prepare(**prepare_input)
            _need(_encode(prepare_input) == native_bytes, "publisher mutated preparation input")
            lengths = _preflight(prepared, parsed)
            check()
            forward_input = copy.deepcopy(native)
            raw = self.runtime.system_one(**forward_input)
            _need(_encode(forward_input) == native_bytes, "publisher mutated forward input")
            check()
            _runtime(self.runtime)
            _raw_answers(raw, parsed, lengths)
            normalized = normalize_native_response(parsed, raw)
            check()
            legacy = {
                **normalized,
                "model": self.model_identity,
                "answers": [copy.deepcopy(normalized["answers"][q.id]) for q in parsed.questions],
                "decision3_native": {
                    "version": PROFILE,
                    "original_request": original,
                    "original_request_sha256": hashlib.sha256(original_bytes).hexdigest(),
                    "raw_response": copy.deepcopy(raw),
                    "status": "completed",
                },
                "decision3_cpu": {
                    "profile": "decision3-cpu-browser-response-v1",
                    "device": "cpu",
                    "native_head_lengths": lengths,
                    "timeout": "20 seconds cooperative; no hard cancellation",
                    "scope": "caller authenticates runtime and owns resource supervision; browser success unverified",
                },
            }
            response = wire.named_browser_response(original, legacy)
            check()
            # Commit the response while excluding an overlapping caller from
            # changing failure state between the final check and lock release.
            with self._state_lock:
                _need(not self.failed, "adapter failed; external owner cleanup required")
                self.lock.release()
                released = True
                return response
        except BaseException:
            with self._state_lock:
                self.failed = True
            raise
        finally:
            if not released:
                self.lock.release()
