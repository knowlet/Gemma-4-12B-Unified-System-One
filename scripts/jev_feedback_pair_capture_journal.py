"""Source-only structure verifier for an actor's private capture journal.

This validates only the supplied journal and its referenced body bytes. It
cannot establish where evidence originated, whether capture was complete,
whether a request was delivered, whether a task succeeded, or whether case
selection was blind. Body lengths are observed, not predeclared by the actor.
Requests over 1 MB are outside this verifier's scope and remain unverified.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

SOURCE_ONLY = True
_MAX_EVENTS = 64
_MAX_DIRECTORY_ENTRIES = 1024
_MAX_LINE = 65536
_MAX_JOURNAL = _MAX_EVENTS * (_MAX_LINE + 1)
_MAX_BODY = {"request": 1_000_000, "response": 16 * 1024**2}
_EVENT_FIELDS = {
    "sequence",
    "status",
    "request_source",
    "http_status",
    "exception_type",
    "scope",
    "request_file",
    "request_sha256",
    "response_file",
    "response_sha256",
}
_LABELS = {"policy_endpoint", "text_endpoint", "shared_endpoint", "unknown"}
_SOURCES = {
    "response-attached-request",
    "exception-attached-request-not-delivery-proof",
    "unavailable",
}


def _need(value, message):
    if not value:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


def _decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _need(key not in result, "duplicate journal JSON key")
            result[key] = value
        return result

    def reject(_value):
        raise ValueError("nonfinite journal JSON value")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject)


def _read_at(directory_fd, name, limit):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    try:
        before = os.fstat(fd)
        _need(
            stat.S_ISREG(before.st_mode)
            and before.st_uid == os.getuid()
            and before.st_mode & 0o077 == 0
            and before.st_nlink == 1
            and before.st_size <= limit,
            "bounded private regular evidence required",
        )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        after = os.fstat(fd)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _need(
            len(raw) == before.st_size
            and _identity(before) == _identity(after) == _identity(named),
            "evidence identity changed while reading",
        )
        return raw, _identity(before)
    finally:
        os.close(fd)


def verify(capture_dir, *, journal_sha256):
    """Check the supplied journal and body files, returning structural observations.

    The caller provides the journal digest; it is a byte comparison, not evidence
    of who supplied it or of the number of calls the original process observed.
    Files outside the capture namespace remain untouched. No body bytes, paths,
    or request details are returned.
    """
    _need(SOURCE_ONLY is False, "held capture journal verifier requires independent review")
    _need(
        type(journal_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", journal_sha256),
        "explicit journal SHA256 required",
    )
    _need(type(capture_dir) is str, "absolute private directory required")
    root = Path(capture_dir)
    _need(
        root.is_absolute() and ".." not in root.parts and root.resolve() == root,
        "ordinary absolute private directory required",
    )
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        initial_dir = os.fstat(directory_fd)
        _need(
            stat.S_ISDIR(initial_dir.st_mode)
            and initial_dir.st_uid == os.getuid()
            and initial_dir.st_mode & 0o077 == 0,
            "owned private directory required",
        )
        raw, journal_identity = _read_at(directory_fd, "capture-events.jsonl", _MAX_JOURNAL)
        _need(_sha(raw) == journal_sha256, "journal SHA256 differs")
        _need(not raw or raw.endswith(b"\n"), "incomplete journal line")
        lines = raw.split(b"\n")[:-1] if raw else []
        _need(len(lines) <= _MAX_EVENTS, "bounded capture event count required")
        expected = set()
        identities = {"capture-events.jsonl": journal_identity}
        observations = []
        attribution = None
        for sequence, line in enumerate(lines, 1):
            _need(0 < len(line) <= _MAX_LINE, "bounded journal line required")
            record = _decode(line)
            _need(type(record) is dict, "capture event must be an object")
            has_label = "destination_match" in record
            if attribution is None:
                attribution = has_label
            _need(has_label == attribution, "mixed destination attribution schema")
            _need(
                set(record) == _EVENT_FIELDS | ({"destination_match"} if has_label else set())
                and type(record["sequence"]) is int
                and record["sequence"] == sequence
                and record["status"] in {"response", "transport_exception"}
                and record["request_source"] in _SOURCES
                and type(record["scope"]) is str
                and len(record["scope"]) <= 4096,
                "ordered exact capture event required",
            )
            if has_label:
                _need(
                    type(record["destination_match"]) is str
                    and record["destination_match"] in _LABELS,
                    "fixed destination label required",
                )
            status, http, error = record["status"], record["http_status"], record["exception_type"]
            _need(http is None or type(http) is int, "invalid HTTP status")
            _need(
                (status == "response" and type(http) is int and error is None)
                or (
                    status == "transport_exception" and type(error) is str and 0 < len(error) <= 512
                ),
                "capture status and exception differ",
            )
            lengths = {}
            for part, limit in _MAX_BODY.items():
                name, digest = record[part + "_file"], record[part + "_sha256"]
                if name is None:
                    _need(digest is None, "missing body must have no digest")
                    lengths[part] = None
                    continue
                _need(
                    type(name) is str
                    and name == f"capture-{sequence:04d}-{part}.bin"
                    and name not in expected
                    and type(digest) is str
                    and re.fullmatch(r"[0-9a-f]{64}", digest),
                    "unique canonical body reference required",
                )
                expected.add(name)
                body, identities[name] = _read_at(directory_fd, name, limit)
                _need(_sha(body) == digest, "body SHA256 differs")
                lengths[part] = len(body)
            if status == "response":
                _need(
                    record["request_source"] == "response-attached-request"
                    and all(lengths[part] is not None for part in _MAX_BODY),
                    "response requires attached request and both bodies",
                )
            observations.append(
                {
                    "sequence": sequence,
                    "status": status,
                    "request_source": record["request_source"],
                    "http_status": http,
                    "request_size_bytes": lengths["request"],
                    "response_size_bytes": lengths["response"],
                    "destination_match": record.get("destination_match"),
                }
            )
        actual = set()
        with os.scandir(directory_fd) as entries:
            for entry_count, entry in enumerate(entries, 1):
                _need(entry_count <= _MAX_DIRECTORY_ENTRIES, "bounded directory entries required")
                if entry.name.startswith("capture-"):
                    actual.add(entry.name)
        _need(actual == expected | {"capture-events.jsonl"}, "missing or orphan capture file")
        for name, identity in identities.items():
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            _need(stat.S_ISREG(info.st_mode) and _identity(info) == identity, "evidence replaced")
        end_dir = os.fstat(directory_fd)
        current_dir = os.stat(root, follow_symlinks=False)
        _need(
            _identity(initial_dir) == _identity(end_dir) == _identity(current_dir),
            "capture directory replaced",
        )
        return {
            "status": "supplied_capture_journal_structure_verified",
            "events": observations,
            "journal_sha256": journal_sha256,
            "predeclared_body_length_verified": False,
            "source_authenticity_verified": False,
            "delivery_verified": False,
            "capture_completeness_verified": False,
            "task_success": None,
            "blind_selection_verified": False,
        }
    finally:
        os.close(directory_fd)
