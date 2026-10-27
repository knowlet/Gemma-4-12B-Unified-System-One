"""Opt-in, privacy-preserving summaries of runner navigation evidence.

Record counts are not unique requests, complete observation, or adjudication.
This held module must be reviewed before it is enabled by a private owner.
"""

from __future__ import annotations

import threading
import types
from contextlib import contextmanager

_PUBLIC_SCOPE_REASONS = {
    "public task permits only GET and HEAD requests": "request_method",
    "public navigation frame metadata unavailable": "request_frame_unavailable",
    "public task top-level navigation outside declared HTTPS hosts": "request_top_level_scope",
    "public document outside declared HTTPS hosts": "document_scope",
    "public document permits only GET and HEAD requests": "document_method",
    "public document redirected from outside declared HTTPS hosts": "document_redirect_scope",
}
_PUBLIC_SCOPE_SOURCES = {
    "cdp_document_request": "document_request",
    "cdp_frame_commit": "frame_commit",
    "cdp_within_document": "within_document",
    "cdp_document_commit": "document_commit_observer",
}
_MAX_OUTCOMES = 256
_MAX_RECORDS = 8192
_MAX_SOURCE_CHARS = 128


def _public_scope_diagnostics(violations, observer_errors, blocks):
    """Count fixed scope signals without returning URLs or exception contents.

    A single navigation may appear in several observer streams. These counts
    represent records, not unique requests, and do not establish coverage.
    An absent list or malformed record stays unknown rather than becoming zero.
    Request blocks are reported separately: they do not all invalidate a task.
    """

    def sources(records):
        if type(records) is not list or len(records) > _MAX_RECORDS:
            return None
        counts = dict.fromkeys((*_PUBLIC_SCOPE_SOURCES.values(), "unattributed", "unknown"), 0)
        for record in records:
            if type(record) is not dict:
                counts["unknown"] += 1
            elif "source" not in record:
                counts["unattributed"] += 1
            else:
                source = record["source"]
                label = (
                    _PUBLIC_SCOPE_SOURCES.get(source)
                    if type(source) is str and len(source) <= _MAX_SOURCE_CHARS
                    else None
                )
                counts[label or "unknown"] += 1
        return counts

    def reasons(records, field):
        if type(records) is not list or len(records) > _MAX_RECORDS:
            return None
        counts = dict.fromkeys((*_PUBLIC_SCOPE_REASONS.values(), "unknown"), 0)
        for record in records:
            reason = record.get(field) if type(record) is dict else None
            if type(reason) is not str or len(reason) > 512:
                counts["unknown"] += 1
                continue
            # CDP document records can contain several fixed reasons. Reject
            # malformed or unknown parts without retaining any input text.
            parts = reason.split("; ")
            if not 1 <= len(parts) <= 3 or any(part not in _PUBLIC_SCOPE_REASONS for part in parts):
                counts["unknown"] += 1
                continue
            for label in {_PUBLIC_SCOPE_REASONS[part] for part in parts}:
                counts[label] += 1
        return counts

    lists = (violations, observer_errors, blocks)
    violation_sources = sources(violations)
    violation_reasons = reasons(violations, "scope_violation")
    observer_sources = sources(observer_errors)
    block_reasons = reasons(blocks, "reason")
    complete = all(type(rows) is list and len(rows) <= _MAX_RECORDS for rows in lists)
    known = (
        complete
        and not any(
            counts["unknown"]
            for counts in (violation_sources, violation_reasons, observer_sources, block_reasons)
        )
        and not (violation_sources["unattributed"] or observer_sources["unattributed"])
    )
    return {
        "schema": "public-scope-diagnostics-v1",
        "status": "available" if known else "incomplete",
        "coverage": "unverified",
        "count_unit": "records_not_unique_requests",
        "violation_records": len(violations)
        if type(violations) is list and len(violations) <= _MAX_RECORDS
        else None,
        "violation_sources": violation_sources,
        "violation_reasons": violation_reasons,
        "observer_error_records": len(observer_errors)
        if type(observer_errors) is list and len(observer_errors) <= _MAX_RECORDS
        else None,
        "observer_error_sources": observer_sources,
        "blocked_request_records": len(blocks)
        if type(blocks) is list and len(blocks) <= _MAX_RECORDS
        else None,
        "blocked_request_reasons": block_reasons,
    }


SOURCE_ONLY = True


def _unavailable(reason):
    return {
        "schema": "public-scope-summary-v1",
        "status": "incomplete",
        "reason": reason,
        "coverage": "unverified",
        "source_provenance": "unverified",
        "attempts": None,
    }


def summarize_runner_result(result):
    """Produce one aggregate without case identity or raw event contents.

    This checks only the supplied in-memory runner result. It cannot establish
    source provenance, completeness, task success, or unique navigation count.
    Non-public or malformed outcomes are incomplete rather than zero events.
    """
    if type(result) is not dict or type(result.get("outcomes")) is not list:
        return _unavailable("missing_outcomes")
    outcomes = result["outcomes"]
    if not outcomes:
        return _unavailable("empty_outcomes")
    if len(outcomes) > _MAX_OUTCOMES:
        return _unavailable("outcomes_limit")
    stream_keys = (
        "public_scope_violations",
        "public_observer_errors",
        "public_request_blocks",
    )
    total_records = 0
    for outcome in outcomes:
        if type(outcome) is dict:
            for key in stream_keys:
                rows = outcome.get(key)
                if type(rows) is list:
                    total_records += len(rows)
                    if total_records > _MAX_RECORDS:
                        return _unavailable("records_limit")
    totals = {
        "violation_records": 0,
        "observer_error_records": 0,
        "blocked_request_records": 0,
        "violation_sources": dict.fromkeys(
            (*_PUBLIC_SCOPE_SOURCES.values(), "unattributed", "unknown"), 0
        ),
        "violation_reasons": dict.fromkeys((*_PUBLIC_SCOPE_REASONS.values(), "unknown"), 0),
        "observer_error_sources": dict.fromkeys(
            (*_PUBLIC_SCOPE_SOURCES.values(), "unattributed", "unknown"), 0
        ),
        "blocked_request_reasons": dict.fromkeys((*_PUBLIC_SCOPE_REASONS.values(), "unknown"), 0),
    }
    incomplete = 0
    missing_streams = dict.fromkeys(stream_keys, 0)
    for outcome in outcomes:
        for key in stream_keys:
            if type(outcome) is not dict or type(outcome.get(key)) is not list:
                missing_streams[key] += 1
        if type(outcome) is dict:
            item = _public_scope_diagnostics(
                outcome.get("public_scope_violations"),
                outcome.get("public_observer_errors"),
                outcome.get("public_request_blocks"),
            )
        else:
            item = _public_scope_diagnostics(None, None, None)
        incomplete += item["status"] != "available"
        for key, total in totals.items():
            value = item[key]
            if type(total) is dict and type(value) is dict:
                for label in total:
                    total[label] += value[label]
            elif type(total) is int and type(value) is int:
                totals[key] += value
    return {
        "schema": "public-scope-summary-v1",
        "status": "available" if incomplete == 0 else "incomplete",
        "coverage": "unverified",
        "source_provenance": "unverified",
        "count_unit": "records_not_unique_requests",
        "attempts": len(outcomes),
        "incomplete_attempts": incomplete,
        "totals_complete": incomplete == 0,
        "missing_streams": missing_streams,
        "totals_observed": totals,
    }


@contextmanager
def attach_to_runner(runner, *, enabled=False):
    """Collect opt-in aggregate results without changing the runner result.

    Collected reports are private caller-owned memory, not authenticated
    receipts. Their lifetime, persistence and any publication are the caller's
    responsibility. The original judge, output and errors are not rewritten.
    """
    if type(enabled) is not bool:
        raise ValueError("explicit diagnostic opt-in required")
    reports = []
    if not enabled:
        yield reports
        return
    if SOURCE_ONLY is not False:
        raise ValueError("held navigation diagnostics require independent review")
    if type(runner) is not types.ModuleType:
        raise ValueError("runner module required")
    original = getattr(runner, "run_benchmark", None)
    if not callable(original) or getattr(original, "__globals__", None) is not runner.__dict__:
        raise ValueError("original runner callback required")
    owner_thread = threading.get_ident()
    active = False

    def wrapped(*args, **kwargs):
        nonlocal active
        if threading.get_ident() != owner_thread or runner.run_benchmark is not wrapped or active:
            raise ValueError("serial diagnostic ownership required")
        active = True
        try:
            try:
                result = original(*args, **kwargs)
            except BaseException:
                reports.append(_unavailable("runner_raised"))
                raise
            try:
                report = summarize_runner_result(result)
            except Exception:
                # Diagnostics must never turn a successful benchmark into a
                # failure, or disclose the exception's potentially private text.
                report = _unavailable("summary_failed")
            reports.append(report)
            return result
        finally:
            active = False

    runner.run_benchmark = wrapped
    primary = None
    try:
        yield reports
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if runner.run_benchmark is wrapped:
            runner.run_benchmark = original
        elif primary is None:
            raise RuntimeError("navigation diagnostics cleanup unconfirmed")
        else:
            primary.add_note("navigation diagnostics cleanup unconfirmed")
