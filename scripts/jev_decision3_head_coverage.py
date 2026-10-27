"""Track explicit Decision3 head participation without changing benchmark calls.

Generic and browser are externally selected routes, not classifications based
on question names. Browser participation requires same-call adapter evidence
from the caller. This module neither scores answers nor authenticates that
external evidence; preserve the official scorer and independently audit it.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import threading

HEADS = ("operation", "click_target")
ROUTES = ("generic", "browser")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _encode(value):
    pending = [(value, False)]
    active = set()
    while pending:
        item, leaving = pending.pop()
        if leaving:
            active.remove(id(item))
            continue
        if type(item) in (dict, list):
            _need(id(item) not in active, "cyclic JSON values are forbidden")
            active.add(id(item))
            pending.append((item, True))
            if type(item) is dict:
                _need(all(type(key) is str for key in item), "JSON object keys must be strings")
                children = item.values()
            else:
                children = item
            pending.extend((child, False) for child in children)
        else:
            _need(
                item is None or type(item) in (str, bool, int, float), "strict JSON values required"
            )
    try:
        return json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError, UnicodeError, RecursionError, OverflowError) as exc:
        raise ValueError("finite UTF-8 JSON is required") from exc


def _parse(payload):
    if type(payload) is dict:
        return payload
    _need(type(payload) in (str, bytes), "ordered JSON packet required")

    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(_value):
        raise ValueError("nonfinite JSON is forbidden")

    try:
        return json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid)
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON packet") from exc


def inspect_request(state, questions):
    """Return an ordered structural view without normalizing the original packet."""
    _need(type(questions) is dict and bool(questions), "ordered named questions required")
    wire_bytes = _encode({"state": state, "questions": questions})
    entries = []
    for key, question in questions.items():
        _need(
            type(key) is str and type(question) is dict,
            "named question required",
        )
        kind = question.get("type", "choice")
        _need(kind in ("choice", "score", "noul"), "unknown native question type")
        _need(
            "instructions" in question and type(question["instructions"]) in (str, dict, list),
            "original instructions required",
        )
        fields = [name for name in ("criteria", "options", "levels") if name in question]
        _need(len(fields) <= 1, "ambiguous criteria aliases")
        if not fields:
            _need(kind == "noul", "choice or score criteria required")
            selection = {"field": None, "kind": "none", "items": []}
        else:
            field = fields[0]
            criteria = question[field]
            if criteria is None:
                _need(kind == "noul", "only noul accepts null criteria")
                selection = {"field": field, "kind": "null", "items": None}
            elif type(criteria) is dict:
                _need(
                    (bool(criteria) or kind == "noul")
                    and all(
                        type(label) is str and (kind == "choice" or bool(label.strip()))
                        for label in criteria
                    ),
                    "nonempty ordered criterion mapping required",
                )
                if kind == "noul":
                    _need(
                        set(criteria) in (set(), {"false", "true"}),
                        "noul criteria must be empty or contain false and true",
                    )
                selection = {
                    "field": field,
                    "kind": "mapping",
                    "items": [[label, value] for label, value in criteria.items()],
                }
            else:
                _need(
                    kind in ("choice", "score")
                    and type(criteria) is list
                    and bool(criteria)
                    and all(
                        type(value) is str and (kind == "score" or bool(value.strip()))
                        for value in criteria
                    )
                    and (kind != "choice" or len(set(criteria)) == len(criteria)),
                    "nonempty ordered choice or score list required",
                )
                selection = {"field": field, "kind": "list", "items": criteria}
        # The detached copy prevents a caller from changing the audit view later.
        entries.append({"id": key, "type": kind, "criteria": json.loads(_encode(selection))})
    return {
        "question_ids": list(questions),
        "ordered_questions": entries,
        "request_sha256": hashlib.sha256(wire_bytes).hexdigest(),
    }


def inspect_packet(packet):
    """Inspect a text packet without conflating canonical JSON with wire bytes.

    ``packet_sha256`` is the historical alias for ``packet_canonical_sha256``.
    ``packet_wire_sha256`` hashes exact bytes (UTF-8 for strings), or is None
    for a dict whose original serialization is unavailable. Pinned views include
    this field, so different encodings of equivalent JSON cannot share a pin.
    """
    value = _parse(packet)
    _need(
        type(value) is dict
        and {"state", "questions"} <= set(value)
        and set(value) <= {"model", "state", "questions", "media"},
        "exact original packet shape required",
    )
    if "model" in value:
        _need(
            type(value["model"]) is str and bool(value["model"].strip()),
            "model alias must be nonempty",
        )
    if "media" in value:
        _need(type(value["media"]) is list and not value["media"], "only empty media is supported")
    view = inspect_request(value["state"], value["questions"])
    canonical_sha = hashlib.sha256(_encode(value)).hexdigest()
    view["packet_sha256"] = canonical_sha
    view["packet_canonical_sha256"] = canonical_sha
    wire_bytes = packet.encode("utf-8") if type(packet) is str else packet
    view["packet_wire_sha256"] = (
        hashlib.sha256(wire_bytes).hexdigest() if type(wire_bytes) is bytes else None
    )
    return view


def _validate_receipt(view, receipt):
    _need(
        type(receipt) is dict
        and set(receipt) == {"question_ids", "ordered_questions", "adapted_rows"},
        "complete same-call browser evidence required",
    )
    _need(
        _encode(receipt["question_ids"]) == _encode(view["question_ids"])
        and _encode(receipt["ordered_questions"]) == _encode(view["ordered_questions"]),
        "browser head or original criteria order differs",
    )
    rows = receipt["adapted_rows"]
    _need(
        type(rows) is dict and list(rows) == list(HEADS), "ordered adapted head evidence required"
    )
    applied = []
    for head in HEADS:
        positions = rows[head]
        _need(
            type(positions) is list
            and all(type(index) is int and index >= 0 for index in positions),
            "invalid adapted row positions",
        )
        if positions:
            _need(
                head in view["question_ids"] and positions == [view["question_ids"].index(head)],
                "adapted row does not match original head",
            )
            _need(
                view["ordered_questions"][positions[0]]["type"] == "choice",
                "browser delta requires original Choice head",
            )
            applied.append(head)
    return applied


class HeadCoverage:
    """Thread-safe call/participation counters separate from benchmark scores."""

    def __init__(self):
        self._lock = threading.Lock()
        self._routes = {route: {"started": 0, "completed": 0, "failed": 0} for route in ROUTES}
        self._requested = dict.fromkeys(HEADS, 0)
        self._applied = dict.fromkeys(HEADS, 0)
        self._browser_no_delta = 0
        self._rejected = 0

    def snapshot(self):
        with self._lock:
            return {
                "schema": "decision3-head-coverage-v1",
                "routes": {key: dict(value) for key, value in self._routes.items()},
                "requested_heads": dict(self._requested),
                "applied_heads": dict(self._applied),
                "generic_bypass_requests": self._routes["generic"]["started"],
                "generic_bypass_completed": self._routes["generic"]["completed"],
                "browser_completed_without_adapted_head": self._browser_no_delta,
                "rejected_before_prediction": self._rejected,
                "model_score": None,
                "participation_basis": "caller-supplied same-call evidence; independently verify adapter witness",
            }

    def _start(self, view, route):
        with self._lock:
            self._routes[route]["started"] += 1
            for head in HEADS:
                if head in view["question_ids"]:
                    self._requested[head] += 1

    def _finish(self, route, applied, failed):
        with self._lock:
            self._routes[route]["failed" if failed else "completed"] += 1
            if not failed:
                for head in applied:
                    self._applied[head] += 1
                if route == "browser" and not applied:
                    self._browser_no_delta += 1

    def _before(self, state, questions, route, packet=None, expected_views=None):
        try:
            _need(route in ROUTES, "explicit generic or browser route required")
            view = inspect_request(state, questions) if packet is None else inspect_packet(packet)
            if expected_views is not None:
                expected = expected_views.get(view.get("packet_sha256", view["request_sha256"]))
                _need(
                    expected is not None and _encode(expected) == _encode(view),
                    "original pinned packet or order differs",
                )
        except (TypeError, ValueError):
            with self._lock:
                self._rejected += 1
            raise
        self._start(view, route)
        return view

    def wrap_sync(self, predict, *, route, evidence=None, packet=False, expected_views=None):
        """Wrap a synchronous ``predict(state, questions)`` or ``predict(packet)``."""
        _need(
            callable(predict) and not inspect.iscoroutinefunction(predict),
            "synchronous predictor required",
        )
        _need(route in ROUTES and type(packet) is bool, "explicit route and packet mode required")
        _need(
            (route == "generic" and evidence is None)
            or (route == "browser" and callable(evidence)),
            "browser requires explicit same-call evidence; generic forbids it",
        )

        _need(
            expected_views is None or type(expected_views) is dict,
            "optional pinned request views must be a mapping",
        )
        if expected_views is not None:
            expected_views = json.loads(_encode(expected_views))

        @functools.wraps(predict)
        def call(*args):
            _need(len(args) == (1 if packet else 2), "original predictor arity required")
            view = (
                self._before(None, None, route, args[0], expected_views)
                if packet
                else self._before(args[0], args[1], route, expected_views=expected_views)
            )
            try:
                result = predict(*args)
                if inspect.iscoroutine(result):
                    result.close()
                    raise ValueError("synchronous predictor returned coroutine")
                _need(not inspect.isawaitable(result), "synchronous predictor returned awaitable")
                after = inspect_packet(args[0]) if packet else inspect_request(args[0], args[1])
                _need(after == view, "predictor changed original request or its order")
                if route == "generic":
                    applied = []
                else:
                    frozen_result = _encode(result)
                    applied = _validate_receipt(view, evidence(result))
                    _need(
                        _encode(result) == frozen_result, "evidence reader changed model response"
                    )
                    after_evidence = (
                        inspect_packet(args[0]) if packet else inspect_request(args[0], args[1])
                    )
                    _need(after_evidence == view, "evidence reader changed original request")
            except BaseException:
                self._finish(route, (), True)
                raise
            self._finish(route, applied, False)
            return result

        return call

    def wrap_async(self, predict, *, route, evidence=None, packet=False, expected_views=None):
        """Wrap an async callable; each concurrent invocation retains its own view."""
        _need(callable(predict), "async predictor required")
        _need(route in ROUTES and type(packet) is bool, "explicit route and packet mode required")
        _need(
            (route == "generic" and evidence is None)
            or (route == "browser" and callable(evidence)),
            "browser requires explicit same-call evidence; generic forbids it",
        )

        _need(
            expected_views is None or type(expected_views) is dict,
            "optional pinned request views must be a mapping",
        )
        if expected_views is not None:
            expected_views = json.loads(_encode(expected_views))

        @functools.wraps(predict)
        async def call(*args):
            _need(len(args) == (1 if packet else 2), "original predictor arity required")
            view = (
                self._before(None, None, route, args[0], expected_views)
                if packet
                else self._before(args[0], args[1], route, expected_views=expected_views)
            )
            try:
                pending = predict(*args)
                _need(inspect.isawaitable(pending), "async predictor must return awaitable")
                result = await pending
                after = inspect_packet(args[0]) if packet else inspect_request(args[0], args[1])
                _need(after == view, "predictor changed original request or its order")
                if route == "generic":
                    applied = []
                else:
                    frozen_result = _encode(result)
                    receipt = evidence(result)
                    if inspect.isawaitable(receipt):
                        receipt = await receipt
                    applied = _validate_receipt(view, receipt)
                    _need(
                        _encode(result) == frozen_result, "evidence reader changed model response"
                    )
                    after_evidence = (
                        inspect_packet(args[0]) if packet else inspect_request(args[0], args[1])
                    )
                    _need(after_evidence == view, "evidence reader changed original request")
            except BaseException:
                self._finish(route, (), True)
                raise
            self._finish(route, applied, False)
            return result

        return call
