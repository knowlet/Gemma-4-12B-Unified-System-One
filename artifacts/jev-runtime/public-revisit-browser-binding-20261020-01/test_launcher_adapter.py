"""Source preparation and argv tests, with framework/browser/network blocked."""

import builtins
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location(
    "tested_revisit_launcher_adapter", HERE / "launcher_adapter.py"
)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


@pytest.fixture(autouse=True)
def no_model_or_browser(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "transformers",
            "numpy",
            "tokenizers",
            "playwright",
            "selenium",
            "httpx",
            "requests",
        }:
            raise AssertionError("Source-only adapter attempted runtime import: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.mark.parametrize("group", ["known4", "public6"])
def test_two_argument_API_preserves_complete_case_budget_and_seed(group, monkeypatch):
    real = a.launcher()
    calls = []

    def two_arguments(arguments, case_module):
        calls.append((arguments[:], case_module))
        return real._arguments(arguments, case_module)

    monkeypatch.setattr(a, "launcher", lambda: SimpleNamespace(_arguments=two_arguments))
    source_group, forwarded = a.source_arguments(group)
    assert len(calls) == 1 and calls[0][1] == a.GROUPS[group]
    plan = json.loads(a.selected(a.CASE_PLAN))
    baseline = source_group["argv_template"]
    expected = baseline[baseline.index("--") + 1 :]
    expected[expected.index("--observation-profile") + 1] = a.PROFILE
    expected[expected.index("--observation-profile-helper-sha256") + 1] = a.PINS[a.HELPER][0]
    expected[expected.index("--output-dir") + 1] = "<REVISIT_GROUP_RESULTS>"
    if group == "public6":
        expected += ["--public-cases", *source_group["selected_case_ids_in_order"]]
    assert forwarded == expected
    assert len(source_group["selected_case_ids_in_order"]) == (4 if group == "known4" else 6)
    assert source_group in plan["groups_in_execution_order"]


@pytest.mark.parametrize("group", ["known4", "public6"])
def test_concrete_forwarding_only_replaces_owned_endpoint_and_output(group):
    results = HERE / ("not-created-" + group) / "results"
    _source, template = a.source_arguments(group)
    args = a.forward_arguments(
        group,
        policy="http://127.0.0.1:12345/v1/systemone",
        helper="http://127.0.0.1:12345",
        results=results,
    )
    expected = template[:]
    for flag, value in (
        ("--endpoint", "http://127.0.0.1:12345/v1/systemone"),
        ("--text-endpoint", "http://127.0.0.1:12345"),
        ("--output-dir", str(results)),
    ):
        expected[expected.index(flag) + 1] = value
    assert args == expected and not results.parent.exists()


@pytest.mark.parametrize(
    "helper,policy",
    [
        ("https://127.0.0.1:12345", "https://127.0.0.1:12345/v1/systemone"),
        ("http://localhost:12345", "http://localhost:12345/v1/systemone"),
        (
            "http://user:password@127.0.0.1:12345",
            "http://user:password@127.0.0.1:12345/v1/systemone",
        ),
        ("http://127.0.0.1:12345", "http://127.0.0.1:12346/v1/systemone"),
    ],
)
def test_different_or_credential_bearing_endpoint_rejected(helper, policy):
    with pytest.raises(RuntimeError):
        a.forward_arguments("known4", helper=helper, policy=policy, results=HERE / "never-created")


@pytest.mark.parametrize("group", ["known4", "public6"])
def test_actual_source_bundle_preparation_never_claims_admission(group, monkeypatch):
    with tempfile.TemporaryDirectory(dir=HERE) as temp:
        monkeypatch.setattr(a, "HERE", Path(temp))
        output = Path(temp) / "prepared"
        receipt = a.prepare_source(group, output)
        assert receipt["profile"] == a.PROFILE and receipt["launcher_API_positional_arguments"] == 2
        assert (
            receipt["benchmark_ready"]
            is receipt["capacity_selected"]
            is receipt["service_bound"]
            is False
        )
        assert receipt["browser_model_HTTP_calls"] == 0
        assert receipt["old_V8_Q95_admits_new_prompt"] is False
        manifest = json.loads((output / "runtime-bundle/bundle-manifest.json").read_bytes())
        a.launcher().verify_bundle(output / "runtime-bundle", manifest)
        assert manifest["profile"] == a.PROFILE and manifest["case_module"] == a.GROUPS[group]
        assert manifest["old_V8_or_Q95_qualification_covers_new_prompt"] is False
        assert (output / "adapter-source-preparation.json").is_file()
        with pytest.raises(ValueError, match="Fresh"):
            a.prepare_source(group, output)


def test_source_drift_rejected_before_import(monkeypatch):
    original = a.read

    def drift(path, limit=1024**2):
        blob = original(path, limit)
        return blob + b"\n" if str(path).endswith(a.LAUNCHER) else blob

    monkeypatch.setattr(a, "read", drift)
    with pytest.raises(ValueError, match="dependency differs"):
        a.launcher()


def test_dry_cli_creates_nothing_and_has_no_run_flag(capsys):
    assert a.main(["--group", "known4"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "DRY_SOURCE_ARGUMENTS_ONLY"
    assert result["benchmark_ready"] is result["output_created"] is False
    assert result["denominator"] == 4 and result["browser_model_HTTP_calls"] == 0
    with pytest.raises(SystemExit):
        a.main(["--group", "known4", "--run"])


@pytest.mark.parametrize("group", ["known4", "public6"])
def test_standalone_cli_without_ambient_scripts_path(group):
    # pytest's configured scripts path must not mask the standalone entry API.
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL") if k in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    result = subprocess.run(
        [sys.executable, "-B", "-s", str(HERE / "launcher_adapter.py"), "--group", group],
        cwd=a.ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["benchmark_ready"] is receipt["output_created"] is False
    assert receipt["denominator"] == (4 if group == "known4" else 6)
    assert receipt["launcher_API_positional_arguments"] == 2
