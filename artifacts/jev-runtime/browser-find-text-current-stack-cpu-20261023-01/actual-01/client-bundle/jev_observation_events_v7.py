"""Record explicit profile reads without changing native observations or retries."""

from __future__ import annotations

import copy
import json
import re
from hashlib import sha256

PROFILE = "composed-visible-text-v2"
PRUNED_PROFILE = "composed-visible-text-settled-v7"
SCHEMA = "jev-observation-profile-event-v1"
CAP_PREFIX = "COMPOSED_VISIBLE_TEXT_LIMIT "
META_KEYS = {
    "id",
    "visited_nodes",
    "node_limit",
    "depth_limit",
    "node_limit_reached",
    "depth_limit_reached",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _metadata_valid(value, *, exhausted, profile):
    return (
        type(value) is dict
        and set(value) == META_KEYS
        and value["id"] == profile
        and type(value["visited_nodes"]) is int
        and 1 <= value["visited_nodes"] <= 20000
        and type(value["node_limit"]) is int
        and value["node_limit"] == 20000
        and type(value["depth_limit"]) is int
        and value["depth_limit"] == 128
        and type(value["node_limit_reached"]) is bool
        and type(value["depth_limit_reached"]) is bool
        and (value["node_limit_reached"] or value["depth_limit_reached"]) is exhausted
    )


def _failure(kind, reason):
    return {"kind": kind, "reason": reason}


def _cap_failure(description, *, profile):
    if type(description) is not str:
        return None
    first_line = description.split("\n", 1)[0]
    if CAP_PREFIX not in first_line:
        return None
    prefix, encoded = first_line.split(CAP_PREFIX, 1)
    if prefix.strip() not in {"", "ComposedVisibleTextLimitError:"}:
        return _failure("integrity", "Malformed composed limit exception name")
    try:

        def object_pairs(items):
            _require(len({key for key, _ in items}) == len(items), "Duplicate limit metadata")
            return dict(items)

        metadata = json.loads(encoded, object_pairs_hook=object_pairs)
    except (TypeError, ValueError):
        return _failure("integrity", "Malformed composed limit exception metadata")
    if not _metadata_valid(metadata, exhausted=True, profile=profile):
        return _failure("integrity", "Invalid composed limit exception metadata")
    return _failure("cap", "Composed text traversal exhausted its node or depth limit")


def _marker_valid(value):
    return (
        type(value) is list
        and len(value) == 10
        and all(type(value[index]) in {int, float} for index in (0, 2, 3, 4, 5))
        and all(type(value[index]) is str for index in (1, 6, 7))
        and type(value[8]) is list
        and type(value[9]) is list
    )


def _response_failure(kind, response, *, profile):
    details = response.get("exceptionDetails")
    if details:
        if type(details) is not dict:
            return _failure("integrity", "Malformed Runtime.evaluate exception details")
        exception = details.get("exception", {})
        description = exception.get("description") if type(exception) is dict else None
        # Ordinary navigation/context exceptions keep their native treatment.
        return _cap_failure(description or details.get("text"), profile=profile)
    result = response.get("result")
    if type(result) is not dict or "value" not in result:
        return _failure("integrity", "Profile read lacks a by-value result")
    value = result["value"]
    if value is None:
        # The unchanged snapshot explicitly returns null while body is absent.
        return None
    if kind == "MARKER":
        if not _marker_valid(value):
            return _failure("integrity", "Profile marker has an invalid native shape")
        return None
    if type(value) is not dict or not _metadata_valid(
        value.get("observation_profile"), exhausted=False, profile=profile
    ):
        return _failure("integrity", "Successful snapshot has invalid profile metadata")
    if (
        type(value.get("text")) is not str
        or not _marker_valid(value.get("marker"))
        or value["marker"][7] != value["text"]
    ):
        return _failure("integrity", "Successful snapshot text and marker disagree")
    return None


class ObservationProfileTracker:
    """Detach and optionally persist every effective READ_STATE/MARKER reply.

    Malformed envelopes and sink failures propagate to the transport, which
    must retain a persistent integrity error. Confirmed profile failures below
    remain latched even when a later native refresh succeeds. This class issues
    no browser calls, inputs, retries or model requests.
    """

    def __init__(
        self,
        *,
        read_state,
        marker_expression,
        manifest_sha256,
        profile=PROFILE,
        sink=None,
    ):
        _require(
            type(profile) is str and profile in {PROFILE, PRUNED_PROFILE},
            "Unknown explicit observation profile",
        )
        _require(
            type(manifest_sha256) is str and re.fullmatch("[0-9a-f]{64}", manifest_sha256),
            "Explicit profile manifest SHA256 required",
        )
        _require(
            type(read_state) is type(marker_expression) is str
            and read_state
            and marker_expression
            and read_state != marker_expression,
            "Distinct trusted observation expressions required",
        )
        _require(sink is None or callable(sink), "Profile sink must be callable")
        self._expressions = {read_state: "READ_STATE", marker_expression: "MARKER"}
        self._manifest_sha256 = manifest_sha256
        self._profile = profile
        self._sink = sink
        self._rows = []
        self._failures = []

    @property
    def rows(self):
        return copy.deepcopy(self._rows)

    @property
    def failures(self):
        return copy.deepcopy(self._failures)

    @property
    def failed(self):
        return bool(self._failures)

    def capture(self, event):
        _require(type(event) is dict, "Profile event must be an ordinary object")
        payload_keys = {"response", "error"} & set(event)
        _require(
            len(payload_keys) == 1
            and set(event) == {"protocol_call", "session", "expression", *payload_keys},
            "Profile event envelope differs",
        )
        _require(
            type(event["protocol_call"]) is int
            and event["protocol_call"] > 0
            and (not self._rows or event["protocol_call"] > self._rows[-1]["protocol_call"])
            and type(event["session"]) is str
            and event["session"]
            and type(event["expression"]) is str
            and event["expression"] in self._expressions,
            "Profile event identity differs",
        )
        payload_key = next(iter(payload_keys))
        payload = event[payload_key]
        _require(type(payload) is dict, "Profile event payload must be an ordinary object")
        if payload_key == "error":
            _require(
                set(payload) == {"type", "message"}
                and all(type(value) is str for value in payload.values()),
                "Profile transport error differs",
            )
        # Refuse unjournalable objects before classification or state changes.
        json.dumps(payload, allow_nan=False)
        payload = copy.deepcopy(payload)
        expression = event["expression"]
        kind = self._expressions[expression]
        failure = (
            _response_failure(kind, payload, profile=self._profile)
            if payload_key == "response"
            else _cap_failure(payload["message"], profile=self._profile)
        )
        sequence = len(self._rows) + 1
        if failure is not None:
            self._failures.append({"sequence": sequence, **failure})
        row = {
            "schema_id": SCHEMA,
            "sequence": sequence,
            "protocol_call": event["protocol_call"],
            "session": event["session"],
            "expression_kind": kind,
            "expression_sha256": sha256(expression.encode()).hexdigest(),
            "manifest_sha256": self._manifest_sha256,
            payload_key: payload,
            "failure": failure,
            "profile_failure_count": len(self._failures),
        }
        self._rows.append(copy.deepcopy(row))
        if self._sink is not None:
            self._sink(copy.deepcopy(row))
        return copy.deepcopy(row)
