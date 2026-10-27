"""Only loopback pages and synthetic model responses; no real model or cases."""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jev_observation_profile_v7_links as links
import jev_v7_links_loopback_smoke as smoke
import pytest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "artifacts/jev-runtime/upstream"
CHROME = Path("/home/albert_hsieh/.cache/ms-playwright/chromium-1208/chrome-linux/chrome")
TOKEN = "a" * 32


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        records = []
        status = 200
        body_override = None

        def log_message(self, *_):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            request = json.loads(raw)
            self.records.append(request)
            if self.body_override is not None:
                body = self.body_override
            elif self.status != 200:
                body = b'{"synthetic_error":"unavailable"}'
            else:
                answers = {}
                for key, question in request["questions"].items():
                    labels = list(question["criteria"])
                    selected = "CLICK" if key == "operation" else labels[0]
                    answers[key] = {
                        "choice": selected,
                        "confidence": 1.0,
                        "probabilities": {k: float(k == selected) for k in labels},
                    }
                body = json.dumps({"model": "synthetic", "answers": answers}).encode()
            self.send_response(self.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield http, Handler
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=3)


def run(
    server,
    monkeypatch,
    *,
    event_sink=lambda _: None,
    wall_seconds=30,
    supervisor_check=lambda: None,
):
    if not CHROME.is_file():
        pytest.skip("Local Chromium executable unavailable")
    monkeypatch.setattr(smoke, "SOURCE_ONLY", False)
    return smoke.run(
        upstream_root=str(UPSTREAM),
        links_sha256=hashlib.sha256(Path(links.__file__).read_bytes()).hexdigest(),
        endpoint=f"http://127.0.0.1:{server.server_port}/decision",
        token=TOKEN,
        model_alias="synthetic",
        chromium_executable=str(CHROME),
        wall_seconds=wall_seconds,
        supervisor_check=supervisor_check,
        event_sink=event_sink,
    )


def test_real_chromium_fake_endpoint_preserves_paired_requests(server, monkeypatch):
    http, handler = server
    result = run(http, monkeypatch)
    assert len(handler.records) == 4
    assert result["status"] == "synthetic_smoke_pending_external_audit"
    assert result["clicks_executed"] == 0
    assert result["task_success"] is None
    assert [row["profile"] for row in result["trials"]] == ["v7", "v7-links", "v7-links", "v7"]
    assert all(
        row["status"] == "response_validated" and row["response_complete"]
        for row in result["trials"]
    )
    for old, new in (
        (handler.records[0], handler.records[1]),
        (handler.records[3], handler.records[2]),
    ):
        left = old["questions"]["click_target"]["criteria"]
        right = new["questions"]["click_target"]["criteria"]
        assert list(left) == list(right) == ["1", "2"]
        assert all("link_destination" not in value for value in left.values())
        assert all("link_destination" in value for value in right.values())
        assert all("query" not in value["link_destination"] for value in right.values())
    assert TOKEN not in json.dumps(result)


def test_http_failure_body_and_denominator_are_preserved(server, monkeypatch):
    http, handler = server
    handler.status = 503
    result = run(http, monkeypatch)
    assert len(result["trials"]) == 4
    assert result["status"] == "synthetic_smoke_incomplete"
    assert all(row["status"] == "failed" and row["http_status"] == 503 for row in result["trials"])
    assert all(
        bytes.fromhex(row["response_bytes_hex"]) == b'{"synthetic_error":"unavailable"}'
        for row in result["trials"]
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:1/",
        "http://localhost:1/",
        "http://u:p@127.0.0.1:1/",
        "http://127.0.0.1:1/?x=1",
    ],
)
def test_endpoint_requires_explicit_credential_free_loopback(url):
    with pytest.raises(ValueError):
        smoke._endpoint(url)


def test_setup_failure_retains_predeclared_denominator(server, monkeypatch):
    if not CHROME.is_file():
        pytest.skip("Local Chromium executable unavailable")
    monkeypatch.setattr(smoke, "SOURCE_ONLY", False)
    http, _ = server
    result = smoke.run(
        upstream_root=str(UPSTREAM),
        links_sha256="0" * 64,
        endpoint=f"http://127.0.0.1:{http.server_port}/decision",
        token=TOKEN,
        model_alias="synthetic",
        chromium_executable=str(CHROME),
        wall_seconds=10,
        supervisor_check=lambda: None,
        event_sink=lambda _: None,
    )
    assert result["status"] == "synthetic_smoke_incomplete"
    assert result["fatal_error_type"] == "ValueError"
    assert len(result["trials"]) == result["expected_trials"] == 4
    assert all(row["status"] == "not_started" for row in result["trials"])


def test_nonfinite_response_is_preserved_as_failure(server, monkeypatch):
    http, handler = server
    handler.body_override = b'{"value":NaN}'
    result = run(http, monkeypatch)
    assert result["status"] == "synthetic_smoke_incomplete"
    assert all(row["status"] == "failed" and row["response_complete"] for row in result["trials"])
    assert all(
        bytes.fromhex(row["response_bytes_hex"]) == handler.body_override
        for row in result["trials"]
    )


def test_escaped_bearer_echo_is_not_persisted(server, monkeypatch):
    http, handler = server
    handler.body_override = b'{"value":"' + b"\\u0061" * 32 + b'"}'
    result = run(http, monkeypatch)
    assert result["status"] == "synthetic_smoke_incomplete"
    assert all(
        row["response_rejected_secret"] and not row["response_complete"] for row in result["trials"]
    )
    assert all("response_bytes_hex" not in row for row in result["trials"])
    assert TOKEN not in json.dumps(result)


def test_oversize_response_is_recorded_as_incomplete(server, monkeypatch):
    http, handler = server
    handler.status = 503
    monkeypatch.setattr(smoke, "MAX_RESPONSE", 8)
    result = run(http, monkeypatch)
    assert result["status"] == "synthetic_smoke_incomplete"
    assert all(
        row["status"] == "failed" and not row["response_complete"] for row in result["trials"]
    )
    assert all("response_prefix_bytes_retained_before_overflow" in row for row in result["trials"])


def test_capture_sink_observes_each_request_response_and_completion(server, monkeypatch):
    http, _ = server
    events = []
    result = run(http, monkeypatch, event_sink=events.append)
    assert result["status"] == "synthetic_smoke_pending_external_audit"
    assert [event["phase"] for event in events].count("planned") == 1
    assert [event["phase"] for event in events].count("request_prepared") == 4
    assert [event["phase"] for event in events].count("response_captured") == 4
    assert [event["phase"] for event in events].count("trial_finished") == 4
    for event in events:
        if event["phase"] == "response_captured":
            assert event["data"]["response_complete"]
            assert (
                json.loads(bytes.fromhex(event["data"]["response_bytes_hex"]))["model"]
                == "synthetic"
            )
    assert TOKEN not in json.dumps(events)


def test_sink_failure_stops_before_first_request(server, monkeypatch):
    http, handler = server

    def reject_request(event):
        if event["phase"] == "request_prepared":
            raise ValueError("private sink failure")

    result = run(http, monkeypatch, event_sink=reject_request)
    assert result["status"] == "synthetic_smoke_incomplete"
    assert result["fatal_error_type"] == "CaptureSinkError"
    assert len(handler.records) == 0
    assert [row["status"] for row in result["trials"]] == [
        "failed",
        "not_started",
        "not_started",
        "not_started",
    ]


def test_expired_cooperative_wall_does_not_hide_planned_trials(server, monkeypatch):
    http, handler = server
    events = []
    result = run(http, monkeypatch, event_sink=events.append, wall_seconds=0.000001)
    assert result["status"] == "synthetic_smoke_incomplete"
    assert len(result["trials"]) == 4
    assert not handler.records
    assert events[0]["phase"] == "planned"
    assert result["hard_deadline"] == "external_supervisor_required"


def test_held_source_cannot_invoke(monkeypatch):
    monkeypatch.setattr(smoke, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="independent review"):
        smoke.run(
            upstream_root=str(UPSTREAM),
            links_sha256="0" * 64,
            endpoint="http://127.0.0.1:1/",
            token=TOKEN,
            model_alias="synthetic",
            chromium_executable=str(CHROME),
            wall_seconds=1,
            supervisor_check=lambda: None,
            event_sink=lambda _: None,
        )


def test_upstream_fifo_is_rejected_without_blocking(tmp_path):
    import os
    import subprocess
    import sys

    directory = tmp_path / "jev_ultrafast"
    directory.mkdir()
    os.mkfifo(directory / "model.py")
    code = (
        "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[2]); "
        "import jev_v7_links_loopback_smoke as s; "
        's._source(Path(sys.argv[1]), "jev_ultrafast/model.py")'
    )
    child = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), str(ROOT / "scripts")],
        capture_output=True,
        timeout=3,
        check=False,
    )
    assert child.returncode != 0
    assert b"bounded ordinary upstream source required" in child.stderr


def test_transient_supervisor_refusal_aborts_all_later_calls(server, monkeypatch):
    http, handler = server
    calls = []
    events = []

    def guard():
        calls.append(len(calls) + 1)
        if len(calls) == 4:
            raise RuntimeError("transient private guard failure")

    result = run(http, monkeypatch, event_sink=events.append, supervisor_check=guard)
    assert result["status"] == "synthetic_smoke_incomplete"
    assert result["fatal_error_type"] == "SupervisorGuardError"
    assert len(calls) == 4
    assert len(handler.records) == 1
    assert [row["status"] for row in result["trials"]] == [
        "response_validated",
        "failed",
        "not_started",
        "not_started",
    ]
    assert events[-1]["phase"] == "trial_finished"
    assert events[-1]["data"]["error_type"] == "SupervisorGuardError"
    assert "transient private guard failure" not in json.dumps(result)


def test_false_supervisor_result_refuses_before_browser(server, monkeypatch):
    http, handler = server
    result = run(http, monkeypatch, supervisor_check=lambda: False)
    assert result["status"] == "synthetic_smoke_incomplete"
    assert result["fatal_error_type"] == "SupervisorGuardError"
    assert not handler.records
    assert all(row["status"] == "not_started" for row in result["trials"])


@pytest.mark.parametrize(
    "body",
    [
        b'{"model":"synthetic","model":"other","answers":{}}',
        b'{"model":"synthetic","answers":{"operation":{"choice":"DONE","choice":"CLICK"}}}',
    ],
)
def test_duplicate_json_keys_rejected_and_raw_response_preserved(server, monkeypatch, body):
    http, handler = server
    handler.body_override = body
    result = run(http, monkeypatch)
    assert result["status"] == "synthetic_smoke_incomplete"
    assert all(row["status"] == "failed" and row["response_complete"] for row in result["trials"])
    assert all(bytes.fromhex(row["response_bytes_hex"]) == body for row in result["trials"])
    assert all("complete_answers" not in row for row in result["trials"])


@pytest.mark.parametrize(
    "body",
    [
        b'{"value":' + b"[" * 1500 + b"0" + b"]" * 1500 + b"}",
        b'{"value":"\\ud800"}',
    ],
)
def test_json_recursion_and_lone_surrogate_preserve_raw_failure(server, monkeypatch, body):
    http, handler = server
    handler.body_override = body
    events = []
    result = run(http, monkeypatch, event_sink=events.append)
    assert result["status"] == "synthetic_smoke_incomplete"
    assert all(row["status"] == "failed" and row["response_complete"] for row in result["trials"])
    assert all(bytes.fromhex(row["response_bytes_hex"]) == body for row in result["trials"])
    assert sum(event["phase"] == "response_captured" for event in events) == 4


@pytest.mark.parametrize("component", ["httpx_client", "browser_context", "browser"])
def test_cleanup_failure_keeps_guard_error_and_unstarted_trials(server, monkeypatch, component):
    from playwright.sync_api import Browser, BrowserContext

    http, handler = server
    calls = []
    events = []

    def guard():
        calls.append(1)
        if len(calls) == 4:
            raise RuntimeError("synthetic guard failure")

    owner = {
        "httpx_client": smoke.httpx.Client,
        "browser_context": BrowserContext,
        "browser": Browser,
    }[component]
    original = owner.close
    failures = []

    def fail_after_close(instance, *args, **kwargs):
        result = original(instance, *args, **kwargs)
        if not failures:
            failures.append(True)
            raise RuntimeError("synthetic private cleanup failure")
        return result

    monkeypatch.setattr(owner, "close", fail_after_close)
    result = run(http, monkeypatch, event_sink=events.append, supervisor_check=guard)
    assert result["status"] == "synthetic_smoke_incomplete"
    assert result["fatal_error_type"] == "SupervisorGuardError"
    assert {"component": component, "error_type": "RuntimeError"} in result["cleanup_errors"]
    assert [row["status"] for row in result["trials"]] == [
        "response_validated",
        "failed",
        "not_started",
        "not_started",
    ]
    assert len(handler.records) == 1
    assert "synthetic private cleanup failure" not in json.dumps(result)
