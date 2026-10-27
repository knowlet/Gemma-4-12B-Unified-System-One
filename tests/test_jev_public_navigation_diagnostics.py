"""Synthetic scope diagnostics must not reveal request or observer contents."""

import json
import types
from types import SimpleNamespace

import jev_browser_benchmark as harness
import jev_public_navigation_diagnostics as diagnostics
import pytest


def test_diagnostics_distinguish_scope_observer_and_block_without_raw_values():
    transport = harness.PlaywrightCDP(None)
    transport.public_root_frames["session-secret"] = "frame-secret"
    transport._public_document_request(
        "session-secret",
        {
            "type": "Document",
            "frameId": "frame-secret",
            "requestId": "request-secret",
            "loaderId": "loader-secret",
            "request": {"url": "https://secret-outside.example/path", "method": "POST"},
            "redirectResponse": {"url": "http://secret-outside.example/path", "status": 307},
        },
        {"allowed.example"},
    )
    transport._public_document_commit(
        "session-secret",
        {"frame": {"id": "frame-secret", "loaderId": "loader-secret", "url": "http://secret"}},
        {"allowed.example"},
    )
    # An observer exception can carry sensitive text and is not a scope violation.
    transport._public_document_commit("session-secret", {"error": "secret"}, {"allowed.example"})
    aborted = []
    transport._public_route(
        SimpleNamespace(
            request=SimpleNamespace(url="https://secret", method="POST"),
            abort=lambda code: aborted.append(code),
        ),
        {"allowed.example"},
    )

    result = diagnostics._public_scope_diagnostics(
        transport.public_navigation_violations,
        transport.public_observer_errors,
        transport.public_request_blocks,
    )

    assert result["status"] == "available"
    assert result["coverage"] == "unverified"
    assert result["count_unit"] == "records_not_unique_requests"
    assert result["violation_records"] == 2
    assert result["violation_sources"]["document_request"] == 1
    assert result["violation_sources"]["frame_commit"] == 1
    assert result["violation_reasons"]["document_scope"] == 2
    assert result["violation_reasons"]["document_method"] == 1
    assert result["violation_reasons"]["document_redirect_scope"] == 1
    assert result["observer_error_records"] == 1
    assert result["observer_error_sources"]["document_commit_observer"] == 1
    assert result["blocked_request_records"] == 1
    assert result["blocked_request_reasons"]["request_method"] == 1
    assert aborted == ["blockedbyclient"]
    assert "secret" not in json.dumps(result)


def test_unattributed_scope_and_observer_failure_are_not_treated_as_complete():
    result = diagnostics._public_scope_diagnostics(
        [{"scope_violation": "public navigation frame metadata unavailable"}],
        [{"type": "ValueError", "message": "sensitive goal and URL"}],
        [],
    )
    assert result["status"] == "incomplete"
    assert result["violation_sources"]["unattributed"] == 1
    assert result["violation_reasons"]["request_frame_unavailable"] == 1
    assert result["observer_error_sources"]["unattributed"] == 1
    assert "sensitive" not in json.dumps(result)


def test_malformed_and_unknown_records_are_counted_without_retaining_text():
    result = diagnostics._public_scope_diagnostics(
        [
            {"source": "private-source", "scope_violation": "private-reason"},
            {"source": "cdp_document_request", "scope_violation": None},
            None,
        ],
        [{"source": "private-source", "message": "private-message"}, None],
        [{"reason": "private-reason"}, {"reason": "x" * 513}],
    )
    assert result["status"] == "incomplete"
    assert result["violation_records"] == 3
    assert result["violation_sources"]["unknown"] == 2
    assert result["violation_reasons"]["unknown"] == 3
    assert result["observer_error_records"] == 2
    assert result["observer_error_sources"]["unknown"] == 2
    assert result["blocked_request_reasons"]["unknown"] == 2
    assert "private" not in json.dumps(result)


def test_missing_signal_stream_is_unknown_not_zero():
    result = diagnostics._public_scope_diagnostics(None, [], [])
    assert result["status"] == "incomplete"
    assert result["violation_records"] is None
    assert result["violation_sources"] is None
    assert result["violation_reasons"] is None
    assert result["observer_error_records"] == 0
    assert result["blocked_request_records"] == 0


def test_same_record_with_multiple_reasons_is_one_record_not_one_navigation():
    result = diagnostics._public_scope_diagnostics(
        [
            {
                "source": "cdp_document_request",
                "scope_violation": "public document outside declared HTTPS hosts; "
                "public document permits only GET and HEAD requests",
            }
        ],
        [],
        [],
    )
    assert result["violation_records"] == 1
    assert result["violation_reasons"]["document_scope"] == 1
    assert result["violation_reasons"]["document_method"] == 1
    assert result["count_unit"] == "records_not_unique_requests"


def test_diagnostics_do_not_change_success_or_premature_done_semantics():
    before = harness._task_success({"success": True}, {"status": "done"}, "scope failure")
    result = diagnostics._public_scope_diagnostics([], [{"message": "sensitive"}], [])
    after = harness._task_success({"success": True}, {"status": "done"}, "scope failure")
    assert before is False and after is False
    assert result["observer_error_records"] == 1
    assert result["violation_records"] == 0


def _runner(result):
    module = types.ModuleType("synthetic_public_runner")
    module.result = result
    exec("def run_benchmark(*args, **kwargs):\n    return result\n", module.__dict__)
    return module


def test_opt_in_aggregates_without_exposing_case_or_changing_runner_result(monkeypatch):
    monkeypatch.setattr(diagnostics, "SOURCE_ONLY", False)
    result = {
        "outcomes": [
            {
                "case_id": "sensitive-case",
                "goal": "sensitive-goal",
                "success": False,
                "error_stage": "public_navigation_scope",
                "public_scope_violations": [
                    {
                        "source": "cdp_document_request",
                        "scope_violation": "public document outside declared HTTPS hosts",
                        "url": "https://sensitive/path",
                    }
                ],
                "public_observer_errors": [],
                "public_request_blocks": [],
            },
            {"case_id": "missing-public-data", "success": True},
        ]
    }
    runner = _runner(result)
    original = runner.run_benchmark
    with diagnostics.attach_to_runner(runner, enabled=True) as reports:
        assert runner.run_benchmark() is result
        assert result["outcomes"][0]["success"] is False
        assert result["outcomes"][1]["success"] is True
    assert runner.run_benchmark is original
    assert len(reports) == 1
    report = reports[0]
    assert report["status"] == "incomplete"
    assert report["coverage"] == report["source_provenance"] == "unverified"
    assert report["attempts"] == 2 and report["incomplete_attempts"] == 1
    assert report["totals_complete"] is False
    assert report["missing_streams"] == {
        "public_scope_violations": 1,
        "public_observer_errors": 1,
        "public_request_blocks": 1,
    }
    assert report["totals_observed"]["violation_records"] == 1
    assert report["totals_observed"]["violation_reasons"]["document_scope"] == 1
    assert "sensitive" not in json.dumps(report)
    assert "missing-public-data" not in json.dumps(report)


def test_oversized_input_is_incomplete_without_scanning_or_leaking():
    outcomes = [{"case_id": "secret"}] * (diagnostics._MAX_OUTCOMES + 1)
    report = diagnostics.summarize_runner_result({"outcomes": outcomes})
    assert report["status"] == "incomplete" and report["reason"] == "outcomes_limit"
    assert "secret" not in json.dumps(report)

    rows = [{"url": "https://secret"}] * (diagnostics._MAX_RECORDS + 1)
    result = {"outcomes": [{"public_scope_violations": rows}]}
    report = diagnostics.summarize_runner_result(result)
    assert report["status"] == "incomplete" and report["reason"] == "records_limit"
    assert "secret" not in json.dumps(report)
    item = diagnostics._public_scope_diagnostics(rows, [], [])
    assert item["status"] == "incomplete" and item["violation_records"] is None


def test_oversized_source_label_stays_unknown_without_retaining_text():
    result = diagnostics._public_scope_diagnostics(
        [
            {
                "source": "sensitive" * 1000,
                "scope_violation": "public document outside declared HTTPS hosts",
            }
        ],
        [],
        [],
    )
    assert result["status"] == "incomplete"
    assert result["violation_sources"]["unknown"] == 1
    assert "sensitive" not in json.dumps(result)


def test_disabled_and_held_gate_do_not_replace_runner(monkeypatch):
    runner = _runner({"outcomes": []})
    original = runner.run_benchmark
    with diagnostics.attach_to_runner(runner) as reports:
        assert reports == [] and runner.run_benchmark is original
    monkeypatch.setattr(diagnostics, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="independent review"):
        with diagnostics.attach_to_runner(runner, enabled=True):
            pass
    assert runner.run_benchmark is original
    with pytest.raises(ValueError, match="explicit diagnostic opt-in"):
        with diagnostics.attach_to_runner(runner, enabled="yes"):
            pass


def test_runner_exception_is_preserved_without_recording_its_message(monkeypatch):
    monkeypatch.setattr(diagnostics, "SOURCE_ONLY", False)
    runner = types.ModuleType("failing_synthetic_runner")
    exec("def run_benchmark():\n    raise RuntimeError('secret message')\n", runner.__dict__)
    original = runner.run_benchmark
    with diagnostics.attach_to_runner(runner, enabled=True) as reports:
        with pytest.raises(RuntimeError, match="secret message"):
            runner.run_benchmark()
    assert runner.run_benchmark is original
    assert reports[0]["reason"] == "runner_raised"
    assert "secret" not in json.dumps(reports)


def test_diagnostic_failure_does_not_change_original_result(monkeypatch):
    monkeypatch.setattr(diagnostics, "SOURCE_ONLY", False)
    runner = _runner({"outcomes": [{"success": False}]})
    original = runner.run_benchmark

    def fail(_):
        raise RuntimeError("sensitive diagnostic error")

    monkeypatch.setattr(diagnostics, "summarize_runner_result", fail)
    with diagnostics.attach_to_runner(runner, enabled=True) as reports:
        assert runner.run_benchmark() is runner.result
    assert runner.run_benchmark is original
    assert reports[0]["reason"] == "summary_failed"
    assert "sensitive" not in json.dumps(reports)


def test_runner_callback_replacement_is_not_overwritten(monkeypatch):
    monkeypatch.setattr(diagnostics, "SOURCE_ONLY", False)
    runner = _runner({"outcomes": []})

    def replacement():
        return None

    with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
        with diagnostics.attach_to_runner(runner, enabled=True):
            runner.run_benchmark = replacement
    assert runner.run_benchmark is replacement
