"""Invalid optional policies fail before any checkpoint or browser is loaded."""

import importlib.util
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    # main() imports this sibling lazily, after the temporary script path is gone.
    importlib.import_module("jev_public_web_cases")
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
        [],
        ["--quantization", "nf4"],
        ["--quantization", "none", "--generated-actions"],
        [
            "--quantization",
            "none",
            "--generated-actions",
            "--generated-action-grammar",
        ],
    ],
)
def test_bf16_qualification_invalid_flags_precede_evidence_or_model_access(
    tmp_path, monkeypatch, options
):
    monkeypatch.setattr(
        DRIVER,
        "upstream_manifest",
        lambda *_a: pytest.fail("invalid BF16 flags reached upstream/model"),
    )
    with pytest.raises(SystemExit) as caught:
        DRIVER.main(
            [
                "--upstream-path",
                "unused",
                "--output-dir",
                str(tmp_path / "run"),
                "--generated-action-qualification",
                "unused",
                "--generated-action-adapter",
                "unused",
                *options,
            ]
        )
    assert caught.value.code == 2
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "options, expected",
    [
        (
            ["--public-cases", "public-python-counter-most-common-v1"],
            {"public_case_ids": ["public-python-counter-most-common-v1"]},
        ),
        (
            ["--seed", "71", "--repeats", "2", "--suite-split", "holdout"],
            {"seed": 71, "repeats": 2, "suite_split": "holdout"},
        ),
        (["--tasks", "native_select"], {"tasks": ["native_select"]}),
        (["--travel-transfer-only"], {"travel_transfer_only": True}),
        (
            [
                "--live-url",
                "https://en.wikipedia.org/wiki/Spork",
                "--live-goal",
                "Open Spoon",
                "--live-expected-title",
                "Spoon",
                "--live-only",
            ],
            {"live_only": True, "tasks": [], "live_goal": "Open Spoon"},
        ),
    ],
)
def test_qualified_diagnostic_inputs_record_actual_scope_without_model_access(
    tmp_path, monkeypatch, options, expected
):
    from s1 import browser_action_qualification, browser_evidence

    historical = {"scope": "original plus known exposed13 diagnostics only"}
    verified = SimpleNamespace(
        artifact=SimpleNamespace(telemetry=lambda: {"manifest_sha256": "test-fit"}),
        telemetry=lambda: {"observed_diagnostics": historical},
    )
    monkeypatch.setattr(
        browser_action_qualification, "verify_bf16_action_qualification", lambda *_args: verified
    )
    monkeypatch.setattr(DRIVER, "upstream_manifest", lambda *_args: {"revision": "test"})

    def stop_before_dependencies(*_args, **_kwargs):
        raise RuntimeError("CPU test stops before model imports")

    monkeypatch.setattr(browser_evidence, "capture_runtime_sources", stop_before_dependencies)
    monkeypatch.setattr(
        browser_action_qualification, "verify_bf16_qualification_capture", lambda *_args: {}
    )
    output = tmp_path / "run"
    with pytest.raises(SystemExit) as caught:
        DRIVER.main(
            [
                "--upstream-path",
                "unused",
                "--output-dir",
                str(output),
                "--generated-action-qualification",
                "unused",
                "--generated-action-adapter",
                "unused",
                "--quantization",
                "none",
                "--generated-actions",
                "--generated-action-grammar",
                "--server-evidence",
                *options,
            ]
        )
    assert caught.value.code == 1
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["error"]["message"] == "CPU test stops before model imports"
    assert receipt["bf16_qualification"]["observed_diagnostics"] == historical
    assert receipt["bf16_evaluation"].items() >= expected.items()
    assert "no quality guarantee" in receipt["bf16_evaluation"]["scope"]
    assert receipt["stage"] == "failed"


@pytest.mark.parametrize("failure", ["file_fsync", "replace", "directory_fsync"])
def test_qualified_receipt_write_failure_preserves_previous_bytes(tmp_path, monkeypatch, failure):
    path = tmp_path / "receipt.json"
    previous = b'{"stage":"browser_benchmark","retained":true}\n'
    path.write_bytes(previous)
    fsync = DRIVER.os.fsync

    def fail_fsync(descriptor):
        is_directory = stat.S_ISDIR(DRIVER.os.fstat(descriptor).st_mode)
        if (failure == "directory_fsync") is is_directory:
            raise OSError(failure)
        return fsync(descriptor)

    def fail_replace(*_args):
        raise OSError("replace")

    if failure == "replace":
        monkeypatch.setattr(DRIVER.os, "replace", fail_replace)
    else:
        monkeypatch.setattr(DRIVER.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match=failure):
        DRIVER._write_receipt(path, {"stage": "complete"}, qualified=True)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_qualified_receipt_fsyncs_file_and_directory_and_legacy_uses_write_text(
    tmp_path, monkeypatch
):
    path = tmp_path / "receipt.json"
    observed = []
    fsync = DRIVER.os.fsync

    def record_fsync(descriptor):
        observed.append(stat.S_ISDIR(DRIVER.os.fstat(descriptor).st_mode))
        return fsync(descriptor)

    monkeypatch.setattr(DRIVER.os, "fsync", record_fsync)
    DRIVER._write_receipt(path, {"stage": "dependencies"}, qualified=True)
    DRIVER._write_receipt(path, {"stage": "complete"}, qualified=True)
    assert observed == [False, True, False, True]
    assert json.loads(path.read_text()) == {"stage": "complete"}
    assert list(tmp_path.iterdir()) == [path]

    def forbidden_replace(*_args):
        pytest.fail("legacy receipt write reached atomic replacement")

    monkeypatch.setattr(DRIVER.os, "replace", forbidden_replace)
    DRIVER._write_receipt(path, {"legacy": True})
    assert path.read_text() == '{\n  "legacy": true\n}\n'
    assert observed == [False, True, False, True]


def test_qualified_failed_rollback_retains_backup_and_reports_all_failures(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    previous = b'{"stage":"browser_benchmark"}\n'
    path.write_bytes(previous)
    primary = RuntimeError("model primary")
    receipt = {"stage": "complete", "error": {"type": "RuntimeError", "message": str(primary)}}
    replace, fsync = DRIVER.os.replace, DRIVER.os.fsync
    replacements = []

    def replace_then_fail_rollback(source, destination):
        replacements.append((source, destination))
        if len(replacements) == 2:
            raise OSError("rollback replacement failed")
        return replace(source, destination)

    def fail_directory_fsync(descriptor):
        if stat.S_ISDIR(DRIVER.os.fstat(descriptor).st_mode):
            raise OSError("directory fsync failed")
        return fsync(descriptor)

    monkeypatch.setattr(DRIVER.os, "replace", replace_then_fail_rollback)
    monkeypatch.setattr(DRIVER.os, "fsync", fail_directory_fsync)
    retained = DRIVER._finish_qualified_receipt(path, receipt, primary)
    assert retained is primary and receipt["stage"] == "failed"
    backups = list(tmp_path.glob(".receipt.json.previous.*"))
    assert len(backups) == 1 and backups[0].read_bytes() == previous
    notes = receipt["receipt_write_error"]["notes"]
    assert any("rollback replacement failed" in note for note in notes)
    assert any(str(backups[0]) in note for note in notes)
    assert any("rollback replacement failed" in note for note in primary.__notes__)
    assert json.loads(path.read_text())["stage"] == "failed"


@pytest.mark.parametrize("earlier_failure", ["model", "helper", "cleanup", "none"])
def test_qualified_final_persistence_retains_earlier_failure_and_progress(
    tmp_path, monkeypatch, earlier_failure
):
    from s1.browser_text import FieldTextError

    path = tmp_path / "receipt.json"
    previous = b'{"stage":"browser_benchmark"}\n'
    path.write_bytes(previous)
    primary = (
        FieldTextError("helper failed", {"failure_stage": "validation"})
        if earlier_failure == "helper"
        else RuntimeError(f"{earlier_failure} failed")
        if earlier_failure != "none"
        else None
    )
    receipt = {"stage": "complete"}
    if earlier_failure == "cleanup":
        receipt["cleanup_errors"] = [{"stage": "thread_shutdown", "message": str(primary)}]
    elif primary is not None:
        receipt["error"] = {"type": type(primary).__name__, "message": str(primary)}

    def fail_replace(*_args):
        raise OSError("final receipt replace failed")

    monkeypatch.setattr(DRIVER.os, "replace", fail_replace)
    retained = DRIVER._finish_qualified_receipt(path, receipt, primary)
    assert path.read_bytes() == previous
    assert receipt["stage"] == "failed"
    assert receipt["receipt_write_error"] == {
        "type": "OSError",
        "message": "final receipt replace failed",
    }
    if primary is not None:
        assert retained is primary
        assert any("persistence also failed" in note for note in primary.__notes__)
        if earlier_failure == "helper":
            assert retained.generation == {"failure_stage": "validation"}
        if earlier_failure == "cleanup":
            assert receipt["cleanup_errors"][0]["message"] == str(primary)
        else:
            assert receipt["error"]["message"] == str(primary)
    else:
        assert isinstance(retained, OSError)


@pytest.mark.parametrize("helper_failure", [False, True])
def test_qualified_cli_prints_final_write_error_and_reraises_primary(
    tmp_path, monkeypatch, capsys, helper_failure
):
    from s1 import browser_action_qualification, browser_evidence
    from s1.browser_text import FieldTextError

    primary = (
        FieldTextError("helper primary", {"failure_stage": "validation"})
        if helper_failure
        else RuntimeError("model primary")
    )
    verified = SimpleNamespace(artifact=SimpleNamespace(telemetry=lambda: {}), telemetry=lambda: {})
    monkeypatch.setattr(
        browser_action_qualification, "verify_bf16_action_qualification", lambda *_args: verified
    )
    monkeypatch.setattr(DRIVER, "upstream_manifest", lambda *_args: {})
    monkeypatch.setattr(
        browser_action_qualification, "verify_bf16_qualification_capture", lambda *_args: {}
    )

    def fail_computation(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(browser_evidence, "capture_runtime_sources", fail_computation)
    write = DRIVER._write_receipt
    previous = []

    def write_then_fail(path, receipt, *, qualified=False):
        if previous:
            raise OSError("final receipt persistence failed")
        write(path, receipt, qualified=qualified)
        previous.append(path.read_bytes())

    monkeypatch.setattr(DRIVER, "_write_receipt", write_then_fail)
    output = tmp_path / "run"
    with pytest.raises(type(primary)) as caught:
        DRIVER.main(
            [
                "--upstream-path",
                "unused",
                "--output-dir",
                str(output),
                "--generated-action-qualification",
                "unused",
                "--generated-action-adapter",
                "unused",
                "--quantization",
                "none",
                "--generated-actions",
                "--generated-action-grammar",
                "--server-evidence",
            ]
        )
    assert caught.value is primary
    printed = json.loads(capsys.readouterr().out)
    assert printed["stage"] == "failed"
    assert printed["error"]["message"] == str(primary)
    assert printed["receipt_write_error"]["message"] == "final receipt persistence failed"
    assert printed["cleanup_errors"][0]["stage"] == "bf16_qualification_finalization"
    assert (output / "receipt.json").read_bytes() == previous[0]


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
