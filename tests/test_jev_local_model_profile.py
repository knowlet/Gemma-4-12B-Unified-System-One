"""Profile CLI validation/forwarding with fake dependencies, zero model execution."""

import builtins
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def driver(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "profile_cli_test_harness", ROOT / "scripts/jev_browser_benchmark.py"
    )
    harness = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = harness
    spec.loader.exec_module(harness)
    monkeypatch.setitem(sys.modules, "jev_browser_benchmark", harness)
    spec = importlib.util.spec_from_file_location(
        "profile_cli_test_driver", ROOT / "scripts/jev_local_model.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result, harness


@pytest.mark.parametrize(
    "flags",
    [
        ["--observation-profile", "native"],
        ["--observation-profile-helper-sha256", "a" * 64],
        ["--observation-profile", "composed-visible-text-v2"],
        ["--observation-profile", "composed-visible-text-v3"],
        [
            "--observation-profile",
            "composed-visible-text-v2",
            "--observation-profile-helper-sha256",
            "0" * 64,
        ],
        [
            "--observation-profile",
            "composed-visible-text-v3",
            "--observation-profile-helper-sha256",
            "0" * 64,
        ],
    ],
)
def test_invalid_observation_options_precede_sources_models_and_output(
    driver, monkeypatch, tmp_path, flags
):
    cli, harness = driver
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {"torch", "transformers", "peft", "s1.unified", "s1.browser_action_adapter"}:
            pytest.fail("Invalid observation flags reached model dependencies")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(
        harness, "upstream_manifest", lambda *_: pytest.fail("Invalid pin reached canonical Git")
    )
    monkeypatch.setattr(
        cli, "upstream_manifest", lambda *_: pytest.fail("Invalid options reached model setup")
    )
    with pytest.raises(SystemExit) as caught:
        cli.main(["--upstream-path", "unused", "--output-dir", str(tmp_path / "run"), *flags])
    assert caught.value.code == 2 and not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "selection,profile_failure",
    [
        ("native-1231850", False),
        ("composed-visible-text-v2", False),
        ("composed-visible-text-v2", True),
        ("composed-visible-text-v3", False),
        ("composed-visible-text-v3", True),
    ],
)
def test_profile_arguments_reach_benchmark_and_failures_remain_in_receipt(
    driver, monkeypatch, tmp_path, selection, profile_failure
):
    cli, _ = driver
    composed = selection != "native-1231850"
    calls, validated = [], []

    def prepare(path, selection, pin):
        validated.append((path, selection, pin))
        return (
            SimpleNamespace(materialized_manifest=lambda: {"schema": "unit-manifest"})
            if composed
            else None
        )

    monkeypatch.setattr(cli, "prepare_observation_profile", prepare)
    monkeypatch.setattr(cli, "upstream_manifest", lambda *_: {"revision": "unit"})

    def module(name, **values):
        result = types.ModuleType(name)
        vars(result).update(values)
        monkeypatch.setitem(sys.modules, name, result)

    module(
        "s1.browser_evidence",
        capture_runtime_sources=lambda *_a, **_k: {"unit": True},
        verify_runtime_sources=lambda *_a, **_k: {"success": True},
    )
    module("s1.browser_server_evidence", ServerEvidenceJournal=object)
    module(
        "torch",
        cuda=SimpleNamespace(get_device_name=lambda: "fake-device", max_memory_allocated=lambda: 0),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(enable_cudnn_sdp=lambda _: None, cudnn_sdp_enabled=lambda: False)
        ),
    )

    class Server:
        def __init__(self, _):
            self.started = True

        def run(self, **_):
            pass

    module("uvicorn", Config=lambda *a, **k: None, Server=Server)
    module("s1.api", create_app=lambda *_a, **_k: SimpleNamespace())
    module(
        "s1.browser_action_adapter",
        base_field_text=lambda *_: pytest.fail("Fake test invoked helper"),
    )
    module(
        "s1.browser_text", generate_field_text=lambda *_: pytest.fail("Fake test invoked helper")
    )

    class Model:
        def __init__(self, name, **options):
            self.name, self.revision = name, options["revision"]
            self.temperature, self.quantization_details = 1.0, {}

    module("s1.unified", UnifiedDecisionModel=Model)
    module(
        "s1.browser_policy",
        BrowserDecisionBackend=lambda *_a, **_k: SimpleNamespace(prompt_variant="unit"),
    )

    class Thread:
        def __init__(self, **_):
            pass

        def start(self):
            pass

        def join(self, **_):
            pass

        def is_alive(self):
            return False

    class Socket:
        def bind(self, *_):
            pass

        def getsockname(self):
            return "127.0.0.1", 1

        def close(self):
            pass

    monkeypatch.setattr(cli.threading, "Thread", Thread)
    monkeypatch.setattr(cli.socket, "socket", Socket)
    monkeypatch.setattr(cli.metadata, "version", lambda _: "unit-version")

    def benchmark(**kwargs):
        calls.append(kwargs)
        return {"observation_profile_failed": profile_failure} if composed else {"attempts": 0}

    monkeypatch.setattr(cli, "run_benchmark", benchmark)
    flags = ["--upstream-path", "unit-source", "--output-dir", str(tmp_path / "run")]
    if composed:
        flags += [
            "--observation-profile",
            selection,
            "--observation-profile-helper-sha256",
            "a" * 64,
        ]
    if profile_failure:
        with pytest.raises(SystemExit) as caught:
            cli.main(flags)
        assert caught.value.code == 1
    else:
        cli.main(flags)
    assert len(calls) == len(validated) == 1
    if composed:
        assert calls[0]["observation_profile"] == selection
        assert calls[0]["observation_profile_helper_sha256"] == "a" * 64
    else:
        assert (
            "observation_profile" not in calls[0]
            and "observation_profile_helper_sha256" not in calls[0]
        )
    receipt = json.loads((tmp_path / "run/receipt.json").read_text())
    assert ("observation_profile" in receipt) is composed
    if profile_failure:
        assert (
            receipt["summary"]["observation_profile_failed"]
            and "observation profile" in receipt["error"]["message"]
        )
        assert receipt["stage"] != "complete"
