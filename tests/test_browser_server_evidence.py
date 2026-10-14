"""Durable local request evidence without model loading or environment discovery."""

import hashlib
import json

import pytest

from s1.browser_server_evidence import ServerEvidenceJournal, json_evidence


def event(sequence, name, *, request_id="a" * 32, data=None):
    return {
        "request_id": request_id,
        "sequence": sequence,
        "event": name,
        "recorded_unix": 100 + sequence,
        "elapsed_ms": sequence * 2.0,
        "data": data or {},
    }


def test_pending_snapshot_then_terminal_hashes_actual_owned_bytes(tmp_path):
    journal = ServerEvidenceJournal(tmp_path / "journal")
    payload = {
        "request_id": "../../not-a-path",
        "output_dir": "/untrusted",
        "criteria": ["all", "candidates"],
    }
    journal(event(0, "request_started", data={"payload": payload}))
    pending = journal.snapshot()
    assert pending["pending_request_ids"] == ["a" * 32]
    assert not pending["all_requests_sealed"]
    assert pending["requests"][0]["server_status"] == "in_flight"
    result = {"answers": {"target": {"probabilities": {"first": 0.3, "second": 0.7}}}}
    journal(event(1, "server_returned", data={"response": result}))
    result["answers"].clear()
    snapshot = journal.snapshot()
    row = snapshot["requests"][0]
    assert snapshot["pending_request_ids"] == [] and snapshot["all_requests_sealed"]
    assert row["server_status"] == "server_returned" and row["sealed"]
    blob = (journal.directory / row["path"]).read_bytes()
    assert row["sha256"] == hashlib.sha256(blob).hexdigest() and row["bytes"] == len(blob)
    written = [json.loads(line) for line in blob.splitlines()]
    assert written[0]["data"]["payload"] == payload
    assert written[1]["data"]["response"]["answers"]["target"]["probabilities"] == {
        "first": 0.3,
        "second": 0.7,
    }
    assert pending["requests"][0]["server_status"] == "in_flight"
    assert list(tmp_path.iterdir()) == [journal.directory]
    assert "client" in snapshot["scope"] and "unknown" in snapshot["scope"]


@pytest.mark.parametrize("request_id", ["../../escape", "/absolute", "a" * 31, "A" * 32, None])
def test_request_values_cannot_choose_a_path(tmp_path, request_id):
    journal = ServerEvidenceJournal(tmp_path / "journal")
    with pytest.raises(ValueError, match="server"):
        journal(event(0, "request_started", request_id=request_id))
    assert list(journal.directory.iterdir()) == []
    assert len(journal.snapshot()["errors"]) == 1


def test_existing_journal_and_unowned_request_file_are_never_overwritten(tmp_path):
    directory = tmp_path / "journal"
    journal = ServerEvidenceJournal(directory)
    with pytest.raises(FileExistsError):
        ServerEvidenceJournal(directory)
    path = directory / ("a" * 32 + ".jsonl")
    path.write_bytes(b"unowned earlier data\n")
    for sequence in (0, 1):
        with pytest.raises(FileExistsError):
            journal(event(sequence, "request_started"))
    assert path.read_bytes() == b"unowned earlier data\n"
    snapshot = journal.snapshot()
    assert snapshot["requests"][0]["created"] is False
    assert not snapshot["all_requests_sealed"]
    assert len(snapshot["errors"]) == 2


def test_actual_partial_write_bytes_remain_visible_and_never_seal(tmp_path, monkeypatch):
    import s1.browser_server_evidence as evidence

    journal = ServerEvidenceJournal(tmp_path / "journal")
    original = evidence.os.fsync
    calls = 0

    def fail_once(fd):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("diagnostic disk sync failure")
        original(fd)

    monkeypatch.setattr(evidence.os, "fsync", fail_once)
    with pytest.raises(OSError, match="disk sync"):
        journal(event(0, "request_started"))
    journal(
        event(
            1,
            "server_failed",
            data={"error": {"type": "RuntimeError", "message": "actual failure"}},
        )
    )
    snapshot = journal.snapshot()
    row = snapshot["requests"][0]
    blob = (journal.directory / row["path"]).read_bytes()
    assert len(blob.splitlines()) == 2 and row["events"] == 1
    assert row["sha256"] == hashlib.sha256(blob).hexdigest() and row["bytes"] == len(blob)
    assert row["successful_event_bytes"] < row["bytes"]
    assert row["server_status"] == "server_failed" and not row["sealed"]
    assert not snapshot["all_requests_sealed"] and len(snapshot["errors"]) == 1


def test_event_order_is_checked_without_replacing_existing_data(tmp_path):
    journal = ServerEvidenceJournal(tmp_path / "journal")
    journal(event(0, "request_started"))
    with pytest.raises(ValueError, match="sequence order"):
        journal(event(0, "server_returned"))
    assert len((journal.directory / ("a" * 32 + ".jsonl")).read_text().splitlines()) == 1
    assert not journal.snapshot()["all_requests_sealed"]


def test_failed_first_event_encoding_cannot_be_erased_by_later_terminal(tmp_path):
    journal = ServerEvidenceJournal(tmp_path / "journal")
    with pytest.raises(UnicodeEncodeError):
        journal(event(0, "request_started", data={"unencodable": "\ud800"}))
    journal(event(1, "server_returned", data={"response": {"actual": True}}))
    snapshot = journal.snapshot()
    assert snapshot["requests"][0]["server_status"] == "server_returned"
    assert snapshot["requests"][0]["write_failed"]
    assert not snapshot["requests"][0]["sealed"] and len(snapshot["errors"]) == 1


def test_json_evidence_detaches_and_retains_invalid_output():
    value = {"nested": [float("nan"), {1: b"not JSON"}]}
    detached = json_evidence(value)
    value["nested"].clear()
    assert detached == {
        "nested": [
            {"nonfinite_float": "nan"},
            {"non_json_mapping": [{"key": 1, "value": {"non_json_type": "bytes"}}]},
        ]
    }
    json.dumps(detached, allow_nan=False)
