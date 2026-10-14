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
        ["--decision-readout", "unused"],
        ["--decision-readout", "unused", "--prompt-mode", "reasoned"],
        ["--decision-readout", "unused", "--prompt-mode", "grounded", "--competitor", "decider"],
        ["--batch-hierarchy-stages"],
        ["--prefix-cache"],
        ["--prefix-cache", "--prompt-mode", "reasoned"],
        ["--prefix-cache", "--prompt-mode", "grounded", "--competitor", "jev-omni"],
        ["--prefix-suffix-batch-size", "2", "--prompt-mode", "grounded"],
        ["--prefix-suffix-batch-size", "3"],
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


@pytest.mark.parametrize("enabled", [False, True])
def test_travel_challenges_default_and_flag_are_recorded_and_forwarded_without_real_runtime(
    tmp_path, monkeypatch, enabled
):
    import json
    from types import SimpleNamespace

    from s1 import api, browser_evidence, browser_policy, browser_text, unified

    output = tmp_path / "run"
    initial_receipts, benchmark_calls, model_calls, sockets, threads = [], [], [], [], []

    def capture(destination, **_kwargs):
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
    monkeypatch.setattr(api, "create_app", lambda *_args, **_kwargs: "mock application")
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
            assert config == "mock application"
            self.started = False
            self.should_exit = False

        def run(self, *, sockets):
            assert len(sockets) == 1
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
            return False

    monkeypatch.setattr(DRIVER, "socket", SimpleNamespace(socket=Listener))
    monkeypatch.setattr(DRIVER, "threading", SimpleNamespace(Thread=Thread))
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(Config=lambda application, **_kwargs: application, Server=Server),
    )

    def benchmark(**kwargs):
        benchmark_calls.append(kwargs)
        return {"scope": "mock forwarding only", "attempts": 0}

    monkeypatch.setattr(DRIVER, "run_benchmark", benchmark)
    options = ["--travel-challenges-only", "--suite-split", "development"] if enabled else []
    DRIVER.main(["--upstream-path", "unused", "--output-dir", str(output), *options])
    assert len(initial_receipts) == len(benchmark_calls) == len(model_calls) == 1
    assert initial_receipts[0]["travel_challenges_only"] is enabled
    assert benchmark_calls[0]["travel_challenges_only"] is enabled
    assert benchmark_calls[0]["suite_split"] == "development"
    assert benchmark_calls[0]["task_ids"] is None
    assert benchmark_calls[0]["live_url"] is None
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["stage"] == "complete"
    assert receipt["travel_challenges_only"] is enabled
    assert receipt["summary"] == {"scope": "mock forwarding only", "attempts": 0}
    assert "error" not in receipt and "cleanup_errors" not in receipt
    assert sockets[0].closed is True and threads[0].joined is True
