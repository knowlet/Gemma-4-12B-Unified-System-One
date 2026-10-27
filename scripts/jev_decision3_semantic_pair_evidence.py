"""Reconcile one owned semantic browser service with private HTTPX capture.

The calling actor must authenticate the service, capture process and releases.
This routine only validates supplied bytes and replays each persisted witness;
it neither runs a model nor establishes a browser task outcome.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

SOURCE_ONLY = True


def _need(value, message):
    if not value:
        raise ValueError(message)


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(path, limit, *, allow_empty=False):
    path = Path(path)
    _need(path.is_absolute() and path.resolve() == path, "ordinary absolute evidence path required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        _need(
            stat.S_ISREG(before.st_mode)
            and (allow_empty or before.st_size > 0)
            and before.st_size <= limit,
            "bounded evidence required",
        )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        after = os.fstat(fd)
        _need(
            len(raw) == before.st_size
            and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "evidence changed while reading",
        )
        return raw
    finally:
        os.close(fd)


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _need(key not in result, "duplicate evidence key")
            result[key] = value
        return result

    def reject(_):
        raise ValueError("nonfinite evidence value")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject)


def verify(
    *,
    service_dir,
    capture_dir,
    plan,
    terminal,
    audit_module,
    binding_module,
    baseline_sha256,
    runner_policy_calls,
    runner_text_calls,
):
    """Independently replay witnesses and reconcile ordered HTTPX-observed bytes.

    Unmatched captured calls have unknown attribution: capture omits endpoint
    data, so they cannot be assumed to be text calls. Exact bytes, not filesystem
    names or a success report, determine each match. Actor/capture provenance,
    GPU exclusion and DOM judge execution must be checked by the caller.
    """
    _need(SOURCE_ONLY is False, "held semantic pair evidence requires independent review")
    _need(_digest(baseline_sha256), "externally pinned baseline required")
    service_dir, capture_dir = Path(service_dir), Path(capture_dir)
    for path in (service_dir, capture_dir):
        _need(
            path.is_absolute() and path.is_dir() and path.resolve() == path,
            "ordinary evidence directory required",
        )
    closure = _json(_read(capture_dir.parent / "owned-client-closure.json", 1024**2))
    _need(
        type(closure) is dict
        and closure.get("confirmed") is True
        and closure.get("remaining") == []
        and type(closure.get("tracked_PID_start_ticks")) is dict,
        "confirmed owned client process-group closure required",
    )
    _need(
        type(terminal) is dict and type(terminal.get("service")) is dict, "child terminal required"
    )
    service = terminal["service"]
    count = service.get("request_calls")
    bindings, receipts, reported = (
        service.get(k) for k in ("private_bindings", "witness_receipts", "numeric_audits")
    )
    _need(
        type(count) is int
        and 1 <= count <= plan["max_requests"]
        and all(
            type(rows) is list and len(rows) == count for rows in (bindings, receipts, reported)
        ),
        "one binding, witness and numerical result per call required",
    )
    _need(
        type(runner_policy_calls) is int
        and runner_policy_calls == count
        and type(runner_text_calls) is int
        and runner_text_calls >= 0,
        "runner policy and text counters required",
    )
    attempts = sorted(capture_dir.iterdir())
    _need(1 <= len(attempts) <= 64, "bounded nonempty client capture required")
    captures = []
    transport_exceptions = 0
    non_success_responses = 0
    capture_pid = None
    for index, directory in enumerate(attempts, 1):
        _need(
            directory.name == f"attempt-{index:04d}"
            and directory.is_dir()
            and directory.resolve() == directory,
            "ordered ordinary capture directories required",
        )
        record = _json(_read(directory / "record.json", 1024**2))
        _need(
            type(record) is dict
            and record.get("sequence") == index
            and type(record.get("sequence")) is int,
            "capture sequence differs",
        )
        pid = record.get("pid")
        _need(
            type(pid) is int
            and pid > 0
            and (capture_pid is None or capture_pid == pid)
            and type(closure["tracked_PID_start_ticks"].get(str(pid))) is int
            and closure["tracked_PID_start_ticks"][str(pid)] > 0,
            "capture PID must belong to the closed owned client process group",
        )
        capture_pid = pid
        if record.get("status") == "transport_exception":
            _need(
                type(record.get("exception_type")) is str and bool(record["exception_type"]),
                "capture exception type required",
            )
            transport_exceptions += 1
            continue
        _need(record.get("status") == "response", "known capture outcome required")
        _need(
            record.get("request_source") == "response-attached-request"
            and type(record.get("http_status")) is int,
            "captured response provenance required",
        )
        if not 200 <= record["http_status"] < 300:
            non_success_responses += 1
        pair = []
        for key, limit in (("request_bytes", 1_000_000), ("response_bytes", 16 * 1024**2)):
            ref = record.get(key)
            _need(
                type(ref) is dict
                and set(ref) == {"sha256", "size_bytes"}
                and _digest(ref["sha256"])
                and type(ref["size_bytes"]) is int,
                "capture digest and size required",
            )
            raw = _read(directory / (key + ".bin"), limit, allow_empty=True)
            _need(
                len(raw) == ref["size_bytes"] and _sha(raw) == ref["sha256"],
                "captured bytes changed",
            )
            pair.append(raw)
        captures.append((index, record["http_status"], *pair))
    events = [
        _json(line) for line in _read(service_dir / "http-events.jsonl", 8 * 1024**2).splitlines()
    ]
    _need(len(events) == count * 2, "complete per-call HTTP events required")
    expected_delta = {key: value["fp32_sha256"] for key, value in plan["deltas"].items()}
    previous = 0
    matches = []
    matched_hashes = set()
    for i, (ref, receipt, numeric) in enumerate(zip(bindings, receipts, reported, strict=True), 1):
        _need(
            type(ref) is dict
            and set(ref) == {"path", "sha256", "size_bytes"}
            and ref["path"] == f"binding-{i:04d}.json"
            and _digest(ref["sha256"])
            and type(ref["size_bytes"]) is int,
            "ordered binding receipt required",
        )
        raw = _read(service_dir / ref["path"], 64 * 1024**2)
        _need(len(raw) == ref["size_bytes"] and _sha(raw) == ref["sha256"], "binding bytes changed")
        binding = _json(raw)
        _need(
            type(receipt) is dict
            and receipt.get("directory") == str(service_dir / "witnesses" / f"call-{i - 1:04d}")
            and _digest(receipt.get("manifest_sha256"))
            and _digest(receipt.get("witness_sha256"))
            and receipt.get("delta_sha256") == expected_delta,
            "ordered pinned witness required",
        )
        _need(binding.get("witness_receipt") == receipt, "binding and witness differ")
        first, second = events[2 * (i - 1) : 2 * i]
        _need(
            type(first) is dict
            and type(second) is dict
            and first.get("status") == "prepared"
            and second.get("status") == "socket_write_returned"
            and first.get("call") == second.get("call") == i
            and type(first.get("call")) is int
            and type(second.get("call")) is int
            and first.get("witness_receipt") == second.get("witness_receipt") == receipt
            and first.get("http_body_sha256")
            == second.get("http_body_sha256")
            == binding.get("http_body_sha256")
            and first.get("response_sha256")
            == second.get("response_sha256")
            == binding.get("response_sha256"),
            "ordered delivery evidence differs",
        )
        possible = [
            (seq, status, req, resp)
            for seq, status, req, resp in captures
            if seq > previous
            and _sha(req) == binding.get("http_body_sha256")
            and _sha(resp) == binding.get("response_sha256")
        ]
        _need(
            bool(possible) and possible[0][1] == 200,
            "ordered successful captured exchange required",
        )
        sequence, _, request, response = possible[0]
        _need(
            binding_module.verify_client_exchange(binding, request, response).get("status")
            == "provided_bytes_match",
            "client and service bytes differ",
        )
        replay = audit_module.audit_directory(
            receipt["directory"],
            expected_file_sha256=receipt["manifest_sha256"],
            expected_witness_sha256=receipt["witness_sha256"],
            expected_delta_sha256=expected_delta,
            expected_names_sha256=plan["vocabulary"]["names_sha256"],
        )
        _need(
            type(replay) is dict
            and replay == numeric
            and replay.get("baseline_sha256") == baseline_sha256
            and replay.get("status") == "numeric_consistency_only"
            and replay.get("browser_qualified") is False
            and replay.get("http_qualified") is False,
            "independent FP64 replay differs",
        )
        previous = sequence
        matches.append(sequence)
        matched_hashes.add((_sha(request), _sha(response)))
    _need(
        [seq for seq, _, req, resp in captures if (_sha(req), _sha(resp)) in matched_hashes]
        == matches,
        "unmatched or reordered identical client exchange",
    )
    return {
        "status": "provided_bytes_and_fp64_match",
        "model_calls": count,
        "capture_events": len(attempts),
        "capture_pid": capture_pid,
        "capture_pid_recorded_in_client_closure": True,
        "runner_policy_calls": runner_policy_calls,
        "runner_text_calls": runner_text_calls,
        "unmatched_capture_events": len(attempts) - count,
        "unmatched_capture_attribution": "unknown",
        "observed_transport_exceptions": transport_exceptions,
        "observed_non_success_responses": non_success_responses,
        "capture_covers_all_runner_transport_attempts": None,
        "matched_sequences": matches,
        "model_score": None,
        "browser_success": None,
        "scope": "supplied bytes only; actor provenance, resource supervision and DOM judge remain external",
    }
