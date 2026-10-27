"""Synthetic client and witness reconciliation; no browser or GPU."""

import copy
import hashlib
import json
from types import SimpleNamespace

import jev_decision3_semantic_pair_evidence as evidence
import pytest


def _bytes(value):
    return json.dumps(value, separators=(",", ":")).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _fixture(tmp_path):
    service, capture = tmp_path / "service", tmp_path / "capture"
    service.mkdir()
    capture.mkdir()
    (tmp_path / "owned-client-closure.json").write_bytes(
        _bytes({"confirmed": True, "remaining": [], "tracked_PID_start_ticks": {"1234": 42}})
    )
    expected = {"operation": "a" * 64, "click_target": "b" * 64}
    baseline = "c" * 64
    plan = {
        "max_requests": 2,
        "deltas": {k: {"fp32_sha256": v} for k, v in expected.items()},
        "vocabulary": {"names_sha256": "d" * 64},
    }
    records, receipts, audits, events = [], [], [], []
    for i in range(1, 3):
        request, response = _bytes({"seq": i}), _bytes({"answer": i})
        directory = capture / f"attempt-{i:04d}"
        directory.mkdir()
        for name, data in (("request_bytes", request), ("response_bytes", response)):
            (directory / f"{name}.bin").write_bytes(data)
        (directory / "record.json").write_bytes(
            _bytes(
                {
                    "sequence": i,
                    "pid": 1234,
                    "status": "response",
                    "request_source": "response-attached-request",
                    "http_status": 200,
                    "request_bytes": {"sha256": _sha(request), "size_bytes": len(request)},
                    "response_bytes": {"sha256": _sha(response), "size_bytes": len(response)},
                }
            )
        )
        receipt = {
            "directory": str(service / "witnesses" / f"call-{i - 1:04d}"),
            "manifest_sha256": "e" * 64,
            "witness_sha256": "f" * 64,
            "delta_sha256": expected,
            "scope": "pending external owner and numerical audit",
        }
        binding = {
            "schema": "decision3-semantic-browser-binding-v1",
            "status": "prepared_not_delivered",
            "http_body_sha256": _sha(request),
            "response_sha256": _sha(response),
            "witness_receipt": receipt,
        }
        raw = _bytes(binding)
        name = f"binding-{i:04d}.json"
        (service / name).write_bytes(raw)
        records.append({"path": name, "sha256": _sha(raw), "size_bytes": len(raw)})
        receipts.append(receipt)
        audits.append(
            {
                "status": "numeric_consistency_only",
                "baseline_sha256": baseline,
                "browser_qualified": False,
                "http_qualified": False,
            }
        )
        for status in ("prepared", "socket_write_returned"):
            events.append(
                {
                    "status": status,
                    "call": i,
                    "witness_receipt": receipt,
                    "http_body_sha256": _sha(request),
                    "response_sha256": _sha(response),
                }
            )
    (service / "http-events.jsonl").write_bytes(b"\n".join(map(_bytes, events)) + b"\n")
    terminal = {
        "service": {
            "request_calls": 2,
            "private_bindings": records,
            "witness_receipts": receipts,
            "numeric_audits": audits,
        }
    }
    replayed = []

    def audit(path, **kwargs):
        replayed.append((path, kwargs))
        return copy.deepcopy(audits[len(replayed) - 1])

    binding = SimpleNamespace(verify_client_exchange=lambda *_: {"status": "provided_bytes_match"})
    args = dict(
        service_dir=service,
        capture_dir=capture,
        plan=plan,
        terminal=terminal,
        audit_module=SimpleNamespace(audit_directory=audit),
        binding_module=binding,
        baseline_sha256=baseline,
        runner_policy_calls=2,
        runner_text_calls=0,
    )
    return args, replayed


def test_two_requests_replayed_and_matched(tmp_path, monkeypatch):
    args, replayed = _fixture(tmp_path)
    with pytest.raises(ValueError, match="independent review"):
        evidence.verify(**args)
    monkeypatch.setattr(evidence, "SOURCE_ONLY", False)
    result = evidence.verify(**args)
    assert result["model_calls"] == 2 and result["matched_sequences"] == [1, 2]
    assert result["browser_success"] is None and result["model_score"] is None
    assert len(replayed) == 2
    assert replayed[0][1]["expected_delta_sha256"] == {
        "operation": "a" * 64,
        "click_target": "b" * 64,
    }
    assert replayed[0][1]["expected_names_sha256"] == "d" * 64


@pytest.mark.parametrize(
    "mutation", ["bytes", "receipt", "baseline", "audit", "count", "status", "order"]
)
def test_inconsistent_evidence_fails_closed(tmp_path, monkeypatch, mutation):
    args, _ = _fixture(tmp_path)
    monkeypatch.setattr(evidence, "SOURCE_ONLY", False)
    if mutation == "bytes":
        (args["capture_dir"] / "attempt-0001" / "request_bytes.bin").write_bytes(b"changed")
    elif mutation == "receipt":
        args["terminal"]["service"]["witness_receipts"][0]["directory"] += "-other"
    elif mutation == "baseline":
        args["baseline_sha256"] = "0" * 64
    elif mutation == "audit":
        args["audit_module"].audit_directory = lambda *_args, **_kwargs: {"status": "not-audit"}
    elif mutation == "count":
        args["terminal"]["service"]["request_calls"] = 1
    elif mutation == "status":
        path = args["capture_dir"] / "attempt-0001" / "record.json"
        record = json.loads(path.read_bytes())
        record["http_status"] = 500
        path.write_bytes(_bytes(record))
    elif mutation == "order":
        first = args["capture_dir"] / "attempt-0001"
        second = args["capture_dir"] / "attempt-0002"
        temp = args["capture_dir"] / "temp"
        first.rename(temp)
        second.rename(first)
        temp.rename(second)
    with pytest.raises((ValueError, KeyError)):
        evidence.verify(**args)


def test_identical_repeated_exchange_is_matched_in_order_and_extra_copy_rejected(
    tmp_path, monkeypatch
):
    args, _ = _fixture(tmp_path)
    monkeypatch.setattr(evidence, "SOURCE_ONLY", False)
    first = args["capture_dir"] / "attempt-0001"
    second = args["capture_dir"] / "attempt-0002"
    req, resp = (
        (first / "request_bytes.bin").read_bytes(),
        (first / "response_bytes.bin").read_bytes(),
    )
    for name, raw in (("request_bytes", req), ("response_bytes", resp)):
        (second / f"{name}.bin").write_bytes(raw)
    record = json.loads((second / "record.json").read_bytes())
    for name, raw in (("request_bytes", req), ("response_bytes", resp)):
        record[name] = {"sha256": _sha(raw), "size_bytes": len(raw)}
    (second / "record.json").write_bytes(_bytes(record))
    path = args["service_dir"] / "binding-0002.json"
    binding = json.loads(path.read_bytes())
    binding["http_body_sha256"], binding["response_sha256"] = _sha(req), _sha(resp)
    raw = _bytes(binding)
    path.write_bytes(raw)
    args["terminal"]["service"]["private_bindings"][1].update(sha256=_sha(raw), size_bytes=len(raw))
    path = args["service_dir"] / "http-events.jsonl"
    events = [json.loads(line) for line in path.read_bytes().splitlines()]
    for event in events[2:]:
        event["http_body_sha256"], event["response_sha256"] = _sha(req), _sha(resp)
    path.write_bytes(b"\n".join(map(_bytes, events)) + b"\n")
    assert evidence.verify(**args)["matched_sequences"] == [1, 2]
    third = args["capture_dir"] / "attempt-0003"
    third.mkdir()
    for name in ("request_bytes.bin", "response_bytes.bin"):
        (third / name).write_bytes((second / name).read_bytes())
    record["sequence"] = 3
    (third / "record.json").write_bytes(_bytes(record))
    args["audit_module"].audit_directory = lambda *_a, **_k: args["terminal"]["service"][
        "numeric_audits"
    ][0]
    with pytest.raises(ValueError, match="unmatched"):
        evidence.verify(**args)


def test_unmatched_transport_exception_is_counted_without_guessing_its_model(tmp_path, monkeypatch):
    args, _ = _fixture(tmp_path)
    monkeypatch.setattr(evidence, "SOURCE_ONLY", False)
    third = args["capture_dir"] / "attempt-0003"
    third.mkdir()
    (third / "record.json").write_bytes(
        _bytes(
            {
                "sequence": 3,
                "pid": 1234,
                "status": "transport_exception",
                "exception_type": "TimeoutException",
            }
        )
    )
    args["runner_text_calls"] = 1
    result = evidence.verify(**args)
    assert result["observed_transport_exceptions"] == 1
    assert result["unmatched_capture_events"] == 1
    assert result["unmatched_capture_attribution"] == "unknown"
    assert result["capture_covers_all_runner_transport_attempts"] is None
    assert result["runner_policy_calls"] == 2 and result["runner_text_calls"] == 1


def test_unmatched_empty_response_remains_unknown_without_fabricating_success(
    tmp_path, monkeypatch
):
    args, _ = _fixture(tmp_path)
    monkeypatch.setattr(evidence, "SOURCE_ONLY", False)
    third = args["capture_dir"] / "attempt-0003"
    third.mkdir()
    request, response = b'{"other":true}', b""
    (third / "request_bytes.bin").write_bytes(request)
    (third / "response_bytes.bin").write_bytes(response)
    (third / "record.json").write_bytes(
        _bytes(
            {
                "sequence": 3,
                "pid": 1234,
                "status": "response",
                "request_source": "response-attached-request",
                "http_status": 503,
                "request_bytes": {"sha256": _sha(request), "size_bytes": len(request)},
                "response_bytes": {"sha256": _sha(response), "size_bytes": 0},
            }
        )
    )
    result = evidence.verify(**args)
    assert result["observed_non_success_responses"] == 1
    assert result["unmatched_capture_events"] == 1
    assert result["unmatched_capture_attribution"] == "unknown"


@pytest.mark.parametrize("fault", ["untracked_pid", "changed_pid", "open_group", "missing_closure"])
def test_capture_pid_must_be_in_closed_client_process_group(tmp_path, monkeypatch, fault):
    args, _ = _fixture(tmp_path)
    monkeypatch.setattr(evidence, "SOURCE_ONLY", False)
    closure = tmp_path / "owned-client-closure.json"
    record_path = args["capture_dir"] / "attempt-0002" / "record.json"
    if fault == "untracked_pid":
        closure.write_bytes(
            _bytes({"confirmed": True, "remaining": [], "tracked_PID_start_ticks": {"999": 42}})
        )
    elif fault == "changed_pid":
        record = json.loads(record_path.read_bytes())
        record["pid"] = 999
        record_path.write_bytes(_bytes(record))
    elif fault == "open_group":
        closure.write_bytes(
            _bytes(
                {
                    "confirmed": True,
                    "remaining": [{"pid": 1}],
                    "tracked_PID_start_ticks": {"1234": 42},
                }
            )
        )
    else:
        closure.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        evidence.verify(**args)
