"""Aggregate DONE-review reconciliation over caller-supplied private evidence.

No files, models, browser, or network are accessed. The caller is responsible for
securely loading and authenticating all sources and independently checking the
service witness and numerical audit. Matching supplied data does not prove its
origin, completeness, browser success, or model quality.
"""

import collections
import hashlib
import json
import math
import re

_LIMIT = 4 * 1024 * 1024


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _json_value(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        _need(math.isfinite(value), "nonfinite JSON")
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    _need(type(value) is dict and all(type(k) is str for k in value), "invalid JSON value")
    for item in value.values():
        _json_value(item)


def _encode(value, *, sorted_keys):
    _json_value(value)
    raw = json.dumps(
        value, sort_keys=sorted_keys, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    _need(len(raw) <= _LIMIT, "JSON exceeds limit")
    return raw


def _key(request, response):
    _need(type(request) is dict and type(response) is dict, "JSON objects required")
    return (_sha(_encode(request, sorted_keys=True)), _sha(_encode(response, sorted_keys=True)))


def _parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def finite(value):
        number = float(value)
        _need(math.isfinite(number), "nonfinite JSON")
        return number

    def invalid(_):
        raise ValueError("nonfinite JSON")

    _need(type(raw) is bytes and len(raw) <= _LIMIT, "bounded bytes required")
    value = json.loads(raw, object_pairs_hook=pairs, parse_float=finite, parse_constant=invalid)
    _json_value(value)
    return value


def _groups(entries):
    result = collections.defaultdict(list)
    for entry in entries:
        result[entry[1]].append(entry)
    return result


def reconcile(*, events, runner_records, captures, client_closure, bindings, service_events):
    """Return only aggregate counts for supplied records, with unresolved kept unknown.

    Each capture is {'record': record, 'request_bytes': bytes, 'response_bytes':
    bytes}; exception captures omit the two byte fields. Inputs must be ordered.
    The caller must authenticate every source and establish capture completeness;
    a matching result only establishes consistency of the supplied objects.
    """
    for item in (events, runner_records, captures, bindings, service_events):
        _need(type(item) is list and len(item) <= 1024, "bounded evidence lists required")
    _need(
        type(client_closure) is dict
        and client_closure.get("confirmed") is True
        and client_closure.get("remaining") == []
        and type(client_closure.get("tracked_PID_start_ticks")) is dict,
        "confirmed client closure required",
    )
    counts = {
        "review_call": 0,
        "replan_call": 0,
        "review_failed": 0,
        "replan_failed": 0,
        "done_supported": 0,
        "replanned": 0,
        "review_limit_blocked": 0,
        "unrecognized": 0,
    }
    event_entries = []
    for index, event in enumerate(events):
        _need(type(event) is dict and type(event.get("status")) is str, "invalid event")
        status = event["status"]
        if status in counts and status != "unrecognized":
            counts[status] += 1
        else:
            counts["unrecognized"] += 1
        if status in ("review_call", "replan_call"):
            _need(
                set(event) == {"status", "request_sha256", "response_sha256"}
                and _digest(event["request_sha256"])
                and _digest(event["response_sha256"]),
                "invalid successful review event",
            )
            event_entries.append(
                (index, (event["request_sha256"], event["response_sha256"]), status)
            )

    runners, runner_errors, runner_unusable = [], 0, 0
    for index, record in enumerate(runner_records):
        _need(type(record) is dict, "invalid runner record")
        if record.get("kind") != "policy":
            continue
        failed = record.get("error") is not None
        if failed:
            runner_errors += 1
        try:
            key = _key(record.get("request"), record.get("response"))
        except (ValueError, TypeError, OverflowError, UnicodeError, RecursionError):
            runner_unusable += 1
            continue
        runners.append((index, key, not failed))

    captured, exceptions, non_success, unparseable = [], 0, 0, 0
    capture_pid = None
    for index, item in enumerate(captures, 1):
        _need(type(item) is dict and type(item.get("record")) is dict, "invalid capture")
        record = item["record"]
        pid = record.get("pid")
        _need(
            type(record.get("sequence")) is int
            and record["sequence"] == index
            and type(pid) is int
            and pid > 0
            and (capture_pid is None or capture_pid == pid)
            and type(client_closure["tracked_PID_start_ticks"].get(str(pid))) is int
            and client_closure["tracked_PID_start_ticks"][str(pid)] > 0,
            "capture sequence or client PID differs",
        )
        capture_pid = pid
        if record.get("status") == "transport_exception":
            _need(
                type(record.get("exception_type")) is str and bool(record["exception_type"]),
                "exception type required",
            )
            exceptions += 1
            continue
        _need(
            record.get("status") == "response"
            and record.get("request_source") == "response-attached-request"
            and type(record.get("http_status")) is int,
            "invalid capture response",
        )
        raws = []
        for field, limit in (("request_bytes", 1_000_000), ("response_bytes", 16 * 1024**2)):
            desc, raw = record.get(field), item.get(field)
            _need(
                type(desc) is dict
                and set(desc) == {"sha256", "size_bytes"}
                and _digest(desc["sha256"])
                and type(desc["size_bytes"]) is int
                and type(raw) is bytes
                and len(raw) <= limit
                and len(raw) == desc["size_bytes"]
                and _sha(raw) == desc["sha256"],
                "capture bytes differ from receipt",
            )
            raws.append(raw)
        if record["http_status"] != 200:
            non_success += 1
            continue
        try:
            request, response = map(_parse, raws)
            key = _key(request, response)
        except (ValueError, TypeError, OverflowError, UnicodeError, RecursionError):
            unparseable += 1
            continue
        captured.append((index, key, raws[0], raws[1], request, response))

    _need(
        all(type(e) is dict and type(e.get("call")) is int for e in service_events),
        "invalid service event",
    )
    service = []
    prior_call = 0
    for binding in bindings:
        _need(
            type(binding) is dict
            and binding.get("schema") == "decision3-semantic-browser-binding-v1"
            and binding.get("status") == "prepared_not_delivered"
            and type(binding.get("call")) is int
            and binding["call"] > prior_call
            and _digest(binding.get("http_body_sha256"))
            and _digest(binding.get("response_sha256")),
            "invalid ordered service binding",
        )
        prior_call = binding["call"]
        try:
            key = _key(binding.get("original_request"), binding.get("named_response"))
        except (ValueError, TypeError, OverflowError, UnicodeError, RecursionError):
            continue
        related = [(i, e) for i, e in enumerate(service_events) if e.get("call") == binding["call"]]
        delivered = (
            len(related) == 2
            and [e.get("status") for _, e in related] == ["prepared", "socket_write_returned"]
            and all(
                e.get("http_body_sha256") == binding["http_body_sha256"]
                and e.get("response_sha256") == binding["response_sha256"]
                and e.get("witness_receipt") == binding.get("witness_receipt")
                for _, e in related
            )
            and type(binding.get("witness_receipt")) is dict
        )
        service.append(
            (binding["call"], key, binding, delivered, related[-1][0] if delivered else -1)
        )

    event_map, runner_map, capture_map, service_map = map(
        _groups, (event_entries, runners, captured, service)
    )
    matched = {"review_call": 0, "replan_call": 0}
    unresolved = {"review_call": 0, "replan_call": 0}
    previous = (-1, -1, -1, -1)
    for _, key, status in event_entries:
        groups = (event_map[key], runner_map[key], capture_map[key], service_map[key])
        if any(len(group) != 1 for group in groups):
            unresolved[status] += 1
            continue
        runner, capture, service_item = groups[1][0], groups[2][0], groups[3][0]
        binding = service_item[2]
        if (
            not runner[2]
            or not service_item[3]
            or _sha(capture[2]) != binding["http_body_sha256"]
            or _sha(capture[3]) != binding["response_sha256"]
            or _encode(capture[4], sorted_keys=False)
            != _encode(binding["original_request"], sorted_keys=False)
            or _encode(capture[5], sorted_keys=False)
            != _encode(binding["named_response"], sorted_keys=False)
        ):
            unresolved[status] += 1
            continue
        current = (runner[0], capture[0], service_item[0], service_item[4])
        _need(all(a < b for a, b in zip(previous, current)), "successful call order differs")
        previous = current
        matched[status] += 1
    incomplete = (
        any(unresolved.values())
        or counts["review_failed"]
        or counts["replan_failed"]
        or counts["review_limit_blocked"]
        or counts["unrecognized"]
        or runner_errors
        or runner_unusable
        or exceptions
        or non_success
        or unparseable
        or len(bindings) != sum(item[3] for item in service)
    )
    return {
        "status": (
            "unresolved"
            if incomplete
            else "no_success_events_observed"
            if not event_entries
            else "supplied_evidence_consistent"
        ),
        "observed_events": counts,
        "matched_events": matched,
        "unresolved_events": unresolved,
        "observed_runner_policy_errors": runner_errors,
        "observed_unusable_runner_policy_records": runner_unusable,
        "observed_capture_transport_exceptions": exceptions,
        "observed_capture_non_200_responses": non_success,
        "observed_capture_unparseable_responses": unparseable,
        "observed_bindings_without_delivery_evidence": len(bindings)
        - sum(item[3] for item in service),
        "source_origin_verified": False,
        "source_completeness_verified": False,
        "numerical_witness_verified": False,
        "semantic_delta_execution_verified": False,
        "unmatched_capture_attribution": "unknown",
        "browser_success": None,
    }
