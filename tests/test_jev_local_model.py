"""Invalid optional policies fail before any checkpoint or browser is loaded."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    SPEC = importlib.util.spec_from_file_location(
        "jev_local_model", ROOT / "scripts/jev_local_model.py"
    )
    DRIVER = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(DRIVER)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize(
    "options",
    [
        ["--suite-split", "holdout"],
        ["--travel-challenges-only"],
        ["--travel-transfer-only"],
        ["--tasks", "native_select"],
        ["--live-only"],
        ["--live-url", "https://en.wikipedia.org/wiki/Spork"],
        ["--live-goal", "Open Spoon"],
        ["--live-expected-title", "Spoon"],
        ["--public-cases", "x", "x"],
    ],
)
def test_public_case_scope_rejected_before_loading_upstream_or_weights(
    tmp_path, monkeypatch, options
):
    monkeypatch.setattr(
        DRIVER,
        "upstream_manifest",
        lambda _path: pytest.fail("Invalid public scope must fail before upstream or model access"),
    )
    output = tmp_path / "run"
    with pytest.raises(SystemExit) as caught:
        DRIVER.main(
            [
                "--upstream-path",
                "unused",
                "--output-dir",
                str(output),
                "--public-cases",
                "public-arxiv-title-filter-results-v1",
                *options,
            ]
        )
    assert caught.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize(
    "options",
    [
        ["--decision-readout", "unused"],
        ["--decision-readout", "unused", "--prompt-mode", "reasoned"],
        ["--decision-readout", "unused", "--prompt-mode", "grounded", "--competitor", "decider"],
        ["--batch-hierarchy-stages"],
        ["--prefix-cache"],
        ["--prefix-cache", "--prompt-mode", "reasoned"],
        ["--prefix-cache", "--prompt-mode", "grounded", "--competitor", "jev-omni"],
        ["--prefix-suffix-batch-size", "2", "--prompt-mode", "grounded"],
        ["--prefix-suffix-batch-size", "3"],
        ["--generated-actions", "--prompt-mode", "reasoned"],
        ["--generated-actions", "--prompt-mode", "user"],
        ["--generated-actions", "--prompt-mode", "grounded"],
        ["--generated-actions", "--analysis-style", "compact"],
        ["--generated-actions", "--hierarchical-candidates"],
        ["--generated-actions", "--batch-hierarchy-stages"],
        ["--generated-actions", "--decision-readout", "unused"],
        ["--generated-actions", "--prefix-cache"],
        ["--generated-actions", "--prefix-suffix-batch-size", "2"],
        ["--generated-actions", "--competitor", "decider"],
        ["--generated-actions", "--competitor", "jev-omni"],
        ["--generated-action-adapter", "unused"],
        ["--generated-action-grammar"],
        ["--generated-actions", "--generated-action-adapter", "unused"],
    ],
)
def test_incompatible_modes_fail_before_public_model_loading(tmp_path, monkeypatch, options):
    def unexpected(_path):
        pytest.fail("Upstream or weights must not be touched for invalid configuration")

    monkeypatch.setattr(DRIVER, "upstream_manifest", unexpected)
    output = tmp_path / "attempt"
    with pytest.raises(SystemExit) as exc:
        DRIVER.main(["--upstream-path", "unused", "--output-dir", str(output), *options])
    assert exc.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize(
    "options",
    [
        ["--tasks", "filter_open"],
        ["--suite-split", "holdout"],
        ["--live-url", "https://example.test/travel"],
        ["--live-goal", "Find a stay"],
        ["--live-expected-title", "Stay details"],
        ["--live-only"],
        ["--travel-transfer-only"],
        [
            "--live-url",
            "https://example.test/travel",
            "--live-goal",
            "Find a stay",
            "--live-expected-title",
            "Stay details",
            "--live-only",
        ],
    ],
)
def test_travel_challenges_reject_incompatible_scope_before_upstream_or_weights(
    tmp_path, monkeypatch, options
):
    def unexpected(_path):
        pytest.fail("Invalid challenge scope must fail before reading upstream or loading weights")

    monkeypatch.setattr(DRIVER, "upstream_manifest", unexpected)
    output = tmp_path / "attempt"
    with pytest.raises(SystemExit) as caught:
        DRIVER.main(
            [
                "--upstream-path",
                "unused",
                "--output-dir",
                str(output),
                "--travel-challenges-only",
                *options,
            ]
        )
    assert caught.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("mode", [None, "challenges", "transfer", "public"])
@pytest.mark.parametrize("policy", ["native", "generated", "adapted", "grammar"])
@pytest.mark.parametrize(
    "journal_mode", ["off", "completed", "pending", "observer_failure", "snapshot_failure"]
)
def test_travel_modes_default_and_flags_are_recorded_and_forwarded_without_real_runtime(
    tmp_path, monkeypatch, mode, journal_mode, policy
):
    import json
    from types import SimpleNamespace

    from s1 import (
        api,
        browser_action_adapter,
        browser_evidence,
        browser_policy,
        browser_text,
        unified,
    )
    from s1.browser_server_evidence import ServerEvidenceJournal

    if journal_mode == "snapshot_failure":

        def broken_snapshot(_journal):
            raise OSError("diagnostic index failure")

        monkeypatch.setattr(ServerEvidenceJournal, "snapshot", broken_snapshot)

    output = tmp_path / "run"
    generated = policy != "native"
    adapter_calls, helper_calls = [], []
    verified = SimpleNamespace(telemetry=lambda: {"manifest_sha256": "verified-test-bytes"})

    def verify_adapter(path, **pins):
        assert path == Path("test-adapter")
        assert not model_calls
        adapter_calls.append(("verify", pins))
        return verified

    def attach_adapter(model, artifact):
        assert artifact is verified and len(model_calls) == 1
        adapter_calls.append(("attach", model))
        return {"manifest_sha256": "verified-test-bytes", "all_parameters_frozen": True}

    def base_helper(model, context):
        helper_calls.append((model.name, context))
        return "base helper result", {"action_adapter_state": "disabled_for_pinned_base_helper"}

    monkeypatch.setattr(browser_action_adapter, "verify_action_adapter", verify_adapter)
    monkeypatch.setattr(browser_action_adapter, "attach_action_adapter", attach_adapter)
    monkeypatch.setattr(browser_action_adapter, "base_field_text", base_helper)
    initial_receipts, benchmark_calls, model_calls, sockets, threads = [], [], [], [], []
    application = SimpleNamespace(
        state=SimpleNamespace(request_evidence_errors=[], request_evidence_timings=[])
    )
    sinks, app_options = [], []

    def capture(destination, **_kwargs):
        if mode == "public":
            assert _kwargs["additional_files"]["scripts/jev_public_web_cases.py"] == (
                ROOT / "scripts/jev_public_web_cases.py"
            )
        initial_receipts.append(json.loads((destination / "receipt.json").read_text()))
        return {"identity_sha256": "test-source-identity"}

    monkeypatch.setattr(browser_evidence, "capture_runtime_sources", capture)
    monkeypatch.setattr(
        browser_evidence,
        "verify_runtime_sources",
        lambda *_args, **_kwargs: {"success": True},
    )
    monkeypatch.setattr(DRIVER, "upstream_manifest", lambda _path: {"revision": "test-upstream"})
    monkeypatch.setattr(DRIVER.metadata, "version", lambda _name: "test-version")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_name=lambda: "mock GPU; no device access",
                max_memory_allocated=lambda: 0,
            ),
            backends=SimpleNamespace(
                cuda=SimpleNamespace(
                    enable_cudnn_sdp=lambda _enabled: None,
                    cudnn_sdp_enabled=lambda: False,
                )
            ),
        ),
    )

    def model(name, **options):
        model_calls.append((name, options))
        return SimpleNamespace(
            name=name,
            revision=options["revision"],
            temperature=2.0,
            quantization_details={"mode": options["quantization"], "test_boundary": True},
        )

    monkeypatch.setattr(unified, "UnifiedDecisionModel", model)
    monkeypatch.setattr(
        browser_policy,
        "BrowserDecisionBackend",
        lambda _model, **_kwargs: SimpleNamespace(prompt_variant="native"),
    )

    def create_app(*_args, **kwargs):
        sinks.append(kwargs["request_evidence_sink"])
        app_options.append(kwargs)
        return application

    monkeypatch.setattr(api, "create_app", create_app)
    monkeypatch.setattr(
        browser_text,
        "generate_field_text",
        lambda *_args, **_kwargs: pytest.fail("The forwarding test must not generate text"),
    )

    class Listener:
        def __init__(self):
            self.closed = False
            sockets.append(self)

        def bind(self, address):
            assert address == ("127.0.0.1", 0)

        def getsockname(self):
            return ("127.0.0.1", 32123)

        def close(self):
            self.closed = True

    class Server:
        def __init__(self, config):
            assert config is application
            self.started = False
            self.should_exit = False

        def run(self, *, sockets):
            assert len(sockets) == 1
            if journal_mode != "off":
                assert (
                    json.loads((output / "receipt.json").read_text())["server_evidence"][
                        "artifact_path"
                    ]
                    == "server-evidence"
                )
            self.started = True

    class Thread:
        def __init__(self, *, target, kwargs, daemon):
            assert daemon is True
            self.target, self.kwargs, self.joined = target, kwargs, False
            threads.append(self)

        def start(self):
            self.target(**self.kwargs)

        def join(self, *, timeout):
            assert timeout == 10
            self.joined = True

        def is_alive(self):
            return journal_mode == "pending"

    monkeypatch.setattr(DRIVER, "socket", SimpleNamespace(socket=Listener))
    monkeypatch.setattr(DRIVER, "threading", SimpleNamespace(Thread=Thread))
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(Config=lambda application, **_kwargs: application, Server=Server),
    )

    def benchmark(**kwargs):
        benchmark_calls.append(kwargs)
        if policy == "adapted":
            assert kwargs["text_helper"]({"goal": "preserved user goal"})[0] == "base helper result"
        if journal_mode != "off":
            sink = sinks[0]
            assert sink is not None
            sink({"request_id": "b" * 32, "sequence": 0, "event": "request_started", "data": {}})
            application.state.request_evidence_timings.append({"elapsed_ms": 1.25})
            if journal_mode == "snapshot_failure":
                raise RuntimeError("actual benchmark failure before index failure")
            if journal_mode != "pending":
                sink(
                    {
                        "request_id": "b" * 32,
                        "sequence": 1,
                        "event": "server_returned",
                        "data": {"response": "mock computed result"},
                    }
                )
                application.state.request_evidence_timings.append({"elapsed_ms": 2.5})
            if journal_mode == "observer_failure":
                application.state.request_evidence_errors.append(
                    {"type": "OSError", "message": "diagnostic observer failure"}
                )
        return {"scope": "mock forwarding only", "attempts": 0}

    monkeypatch.setattr(DRIVER, "run_benchmark", benchmark)
    options = (
        ["--public-cases", "public-arxiv-title-filter-results-v1"]
        if mode == "public"
        else [f"--travel-{mode}-only", "--suite-split", "development"]
        if mode
        else []
    )
    if journal_mode != "off":
        options.append("--server-evidence")
    if generated:
        options.append("--generated-actions")
    if policy == "grammar":
        options.append("--generated-action-grammar")
    if policy == "adapted":
        options.extend(["--generated-action-adapter", "test-adapter"])
    arguments = ["--upstream-path", "unused", "--output-dir", str(output), *options]
    if journal_mode in {"pending", "observer_failure", "snapshot_failure"}:
        with pytest.raises(SystemExit) as error:
            DRIVER.main(arguments)
        assert error.value.code == 1
    else:
        DRIVER.main(arguments)
    assert len(initial_receipts) == len(benchmark_calls) == len(model_calls) == 1
    assert initial_receipts[0]["travel_challenges_only"] is (mode == "challenges")
    assert initial_receipts[0]["generated_actions"] is generated
    assert app_options[0]["jev_generated_actions"] is generated
    assert app_options[0]["jev_generated_action_grammar"] is (policy == "grammar")
    assert initial_receipts[0]["generated_action_grammar"] is (policy == "grammar")
    assert benchmark_calls[0]["model"] == ("generated-actions-nf4" if generated else "native-nf4")
    assert benchmark_calls[0]["travel_challenges_only"] is (mode == "challenges")
    assert initial_receipts[0]["travel_transfer_only"] is (mode == "transfer")
    assert benchmark_calls[0]["travel_transfer_only"] is (mode == "transfer")
    assert benchmark_calls[0]["suite_split"] == "development"
    assert benchmark_calls[0]["task_ids"] is None
    assert benchmark_calls[0]["live_url"] is None
    assert benchmark_calls[0]["public_case_ids"] == (
        ["public-arxiv-title-filter-results-v1"] if mode == "public" else None
    )
    if mode == "public":
        assert initial_receipts[0]["public_case_ids"] == ["public-arxiv-title-filter-results-v1"]
    receipt = json.loads((output / "receipt.json").read_text())
    if policy == "adapted":
        assert [call[0] for call in adapter_calls] == ["verify", "attach"]
        assert "adapter_path" not in model_calls[0][1]
        assert helper_calls == [(model_calls[0][0], {"goal": "preserved user goal"})]
        assert receipt["decision_model"]["action_adapter"]["all_parameters_frozen"] is True
    else:
        assert not adapter_calls and not helper_calls
    if generated:
        assert receipt["decision_model"]["policy"] == "greedy_legal_action_v2"
        assert receipt["decision_model"]["decoding_constraint"] == (
            "legal_action_token_trie_v1" if policy == "grammar" else "none"
        )
        assert receipt["decision_model"]["max_new_tokens"] == 32
        assert receipt["decision_model"]["budget_seconds"] == 20
    assert receipt["stage"] == (
        "browser_benchmark" if journal_mode == "snapshot_failure" else "complete"
    )
    assert receipt["travel_challenges_only"] is (mode == "challenges")
    assert receipt["travel_transfer_only"] is (mode == "transfer")
    if journal_mode == "snapshot_failure":
        assert receipt["error"] == {
            "type": "RuntimeError",
            "message": "actual benchmark failure before index failure",
        }
        assert receipt["cleanup_errors"] == [
            {"stage": "server_evidence", "message": "diagnostic index failure"}
        ]
        assert receipt["server_evidence"]["artifact_path"] == "server-evidence"
        assert "index_path" not in receipt["server_evidence"]
        assert sockets[0].closed is True and threads[0].joined is True
        return
    assert receipt["summary"] == {"scope": "mock forwarding only", "attempts": 0}
    assert "error" not in receipt
    if journal_mode == "off":
        assert sinks == [None]
        assert "server_evidence" not in receipt and not (output / "server-evidence").exists()
        assert "cleanup_errors" not in receipt
    else:
        evidence = receipt["server_evidence"]
        assert evidence["index_path"] == "server-evidence/index.json"
        index = json.loads((output / evidence["index_path"]).read_text())
        assert index["requests"] == evidence["requests"]
        assert index["pending_request_ids"] == (["b" * 32] if journal_mode == "pending" else [])
        assert index["all_requests_sealed"] is (journal_mode == "completed")
        assert evidence["observer_calls"] == (1 if journal_mode == "pending" else 2)
        assert evidence["observer_elapsed_ms"] == (1.25 if journal_mode == "pending" else 3.75)
        if journal_mode in {"pending", "observer_failure"}:
            assert receipt["cleanup_errors"][0]["stage"] == (
                "thread_shutdown" if journal_mode == "pending" else "server_evidence"
            )
        else:
            assert "cleanup_errors" not in receipt
        if journal_mode == "observer_failure":
            assert evidence["observer_errors"] == application.state.request_evidence_errors
    assert sockets[0].closed is True and threads[0].joined is True
