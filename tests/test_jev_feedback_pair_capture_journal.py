"""Synthetic journal verification; no browser, model or benchmark cases."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import jev_feedback_pair_capture_journal as journal
import pytest


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def capture(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "SOURCE_ONLY", False)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)

    def write(events):
        raw = b"".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for event in events
        )
        path = root / "capture-events.jsonl"
        path.write_bytes(raw)
        path.chmod(0o600)
        return sha(raw)

    def body(seq, part, data):
        name = f"capture-{seq:04d}-{part}.bin"
        path = root / name
        path.write_bytes(data)
        path.chmod(0o600)
        return name, sha(data)

    def record(seq, status="response", code=200, label="unknown", request=b"{}", response=b"{}"):
        rec = {
            "sequence": seq,
            "status": status,
            "request_source": (
                "response-attached-request"
                if status == "response"
                else "exception-attached-request-not-delivery-proof"
                if request is not None
                else "unavailable"
            ),
            "http_status": code if status == "response" else None,
            "exception_type": None if status == "response" else "SyntheticError",
            "scope": "synthetic",
        }
        if label is not None:
            rec["destination_match"] = label
        for part, data in (("request", request), ("response", response)):
            if data is None:
                rec[part + "_file"] = rec[part + "_sha256"] = None
            else:
                rec[part + "_file"], rec[part + "_sha256"] = body(seq, part, data)
        return rec

    return root, write, record


def test_all_events_remain_in_order_including_unknown_exception_non200(capture):
    root, write, record = capture
    events = [
        record(1, label="policy_endpoint", request=b'{"x":1}', response=b"{}"),
        record(2, code=503, label="text_endpoint"),
        record(3, status="transport_exception", label="unknown", request=b"{}", response=None),
        record(
            4, status="transport_exception", label="shared_endpoint", request=None, response=None
        ),
    ]
    digest = write(events)
    # Other actor output is outside the capture namespace.
    (root / "runner").mkdir()
    (root / "review-events.jsonl").write_text("synthetic")
    result = journal.verify(str(root), journal_sha256=digest)
    assert [item["sequence"] for item in result["events"]] == [1, 2, 3, 4]
    assert [item["status"] for item in result["events"]] == [
        "response",
        "response",
        "transport_exception",
        "transport_exception",
    ]
    assert [item["destination_match"] for item in result["events"]] == [
        "policy_endpoint",
        "text_endpoint",
        "unknown",
        "shared_endpoint",
    ]
    assert result["events"][1]["http_status"] == 503
    assert result["events"][2]["response_size_bytes"] is None
    assert result["events"][2]["request_source"] == "exception-attached-request-not-delivery-proof"
    assert result["events"][3]["request_source"] == "unavailable"
    assert result["events"][0]["request_size_bytes"] == len(b'{"x":1}')
    assert result["predeclared_body_length_verified"] is False
    assert result["source_authenticity_verified"] is False
    assert result["delivery_verified"] is False
    assert result["capture_completeness_verified"] is False
    assert result["task_success"] is None and result["blind_selection_verified"] is False


def test_legacy_events_and_empty_journal_are_structurally_valid(capture):
    root, write, record = capture
    digest = write([record(1, label=None)])
    result = journal.verify(str(root), journal_sha256=digest)
    assert result["events"][0]["destination_match"] is None
    for path in root.glob("capture-*.bin"):
        path.unlink()
    result = journal.verify(str(root), journal_sha256=write([]))
    assert result["events"] == [] and result["capture_completeness_verified"] is False


@pytest.mark.parametrize("sequence", [0, 2, True])
def test_rejects_missing_or_wrong_sequence(capture, sequence):
    root, write, record = capture
    event = record(1)
    event["sequence"] = sequence
    with pytest.raises(ValueError, match="ordered exact"):
        journal.verify(str(root), journal_sha256=write([event]))


def test_rejects_duplicate_sequence_and_mixed_attribution(capture):
    root, write, record = capture
    first, second = record(1), record(2)
    second["sequence"] = 1
    with pytest.raises(ValueError, match="ordered exact"):
        journal.verify(str(root), journal_sha256=write([first, second]))
    second["sequence"] = 2
    second.pop("destination_match")
    with pytest.raises(ValueError, match="mixed destination"):
        journal.verify(str(root), journal_sha256=write([first, second]))


@pytest.mark.parametrize(
    "action", ["orphan", "missing", "changed", "wrong_sha", "symlink", "hardlink"]
)
def test_rejects_invalid_body_inventory(capture, tmp_path, action):
    root, write, record = capture
    event = record(1)
    path = root / event["request_file"]
    if action == "orphan":
        orphan = root / "capture-0002-response.bin"
        orphan.write_bytes(b"{}")
        orphan.chmod(0o600)
    elif action == "missing":
        path.unlink()
    elif action == "changed":
        path.write_bytes(b"changed")
    elif action == "wrong_sha":
        event["request_sha256"] = "0" * 64
    elif action == "symlink":
        path.unlink()
        target = tmp_path / "target"
        target.write_bytes(b"{}")
        path.symlink_to(target)
    else:
        os.link(path, tmp_path / "hardlink")
    with pytest.raises((ValueError, OSError)):
        journal.verify(str(root), journal_sha256=write([event]))


@pytest.mark.parametrize("name", ["capture-events.jsonl", "capture-0001-request.bin"])
def test_rejects_file_replacement_after_read(capture, monkeypatch, name):
    root, write, record = capture
    digest = write([record(1)])
    original = journal._read_at
    replaced = []

    def replacing(fd, filename, limit):
        result = original(fd, filename, limit)
        if filename == name:
            path = root / filename
            other = root / "temporary"
            other.write_bytes(path.read_bytes())
            other.chmod(0o600)
            os.replace(other, path)
            replaced.append(True)
        return result

    monkeypatch.setattr(journal, "_read_at", replacing)
    with pytest.raises(ValueError, match="evidence replaced|capture directory replaced"):
        journal.verify(str(root), journal_sha256=digest)
    assert replaced == [True]


def test_rejects_wrong_journal_hash_and_partial_line(capture):
    root, write, record = capture
    digest = write([record(1)])
    with pytest.raises(ValueError, match="journal SHA256"):
        journal.verify(str(root), journal_sha256="0" * 64)
    path = root / "capture-events.jsonl"
    partial = path.read_bytes().removesuffix(b"\n")
    path.write_bytes(partial)
    with pytest.raises(ValueError, match="incomplete journal line"):
        journal.verify(str(root), journal_sha256=sha(partial))
    assert digest != sha(partial)


def test_default_gate_refuses(capture, monkeypatch):
    root, write, _ = capture
    digest = write([])
    monkeypatch.setattr(journal, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="independent review"):
        journal.verify(str(root), journal_sha256=digest)


@pytest.mark.parametrize("mutation", ["extra", "label", "missing_sha", "response_missing"])
def test_rejects_unsupported_schema_or_missing_response_body(capture, mutation):
    root, write, record = capture
    event = record(1)
    if mutation == "extra":
        event["unexpected"] = "synthetic"
    elif mutation == "label":
        event["destination_match"] = "unrecognized"
    elif mutation == "missing_sha":
        event["request_sha256"] = None
    else:
        (root / event["response_file"]).unlink()
        event["response_file"] = event["response_sha256"] = None
    with pytest.raises(ValueError):
        journal.verify(str(root), journal_sha256=write([event]))


def test_rejects_duplicate_json_keys(capture):
    root, write, record = capture
    write([record(1)])
    path = root / "capture-events.jsonl"
    raw = path.read_bytes().replace(b'"sequence":1', b'"sequence":1,"sequence":1')
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="duplicate journal JSON key"):
        journal.verify(str(root), journal_sha256=sha(raw))


def test_rejects_body_larger_than_request_limit(capture):
    root, write, record = capture
    event = record(1, request=b"x" * 1_000_001)
    with pytest.raises(ValueError, match="bounded private"):
        journal.verify(str(root), journal_sha256=write([event]))


def test_refuses_nonprivate_evidence(capture):
    root, write, record = capture
    event = record(1)
    digest = write([event])
    path = root / event["request_file"]
    path.chmod(0o644)
    with pytest.raises(ValueError, match="bounded private"):
        journal.verify(str(root), journal_sha256=digest)


def test_fifo_body_refused_without_blocking(capture):
    root, write, record = capture
    event = record(1)
    body = root / event["request_file"]
    body.unlink()
    os.mkfifo(body, 0o600)
    digest = write([event])
    script = """\
import sys
sys.path.insert(0, 'scripts')
import jev_feedback_pair_capture_journal as journal
journal.SOURCE_ONLY = False
try:
    journal.verify(sys.argv[1], journal_sha256=sys.argv[2])
except ValueError as error:
    assert 'bounded private regular' in str(error)
else:
    raise AssertionError('FIFO unexpectedly accepted')
"""
    subprocess.run(
        [sys.executable, "-B", "-s", "-c", script, str(root), digest],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        timeout=3,
    )


def test_directory_entry_limit_refuses_without_ignoring_non_capture_files(capture, monkeypatch):
    root, write, record = capture
    digest = write([record(1)])
    (root / "other-one").touch()
    (root / "other-two").touch()
    monkeypatch.setattr(journal, "_MAX_DIRECTORY_ENTRIES", 4)
    with pytest.raises(ValueError, match="bounded directory entries"):
        journal.verify(str(root), journal_sha256=digest)
