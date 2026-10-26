"""New runner source/interface and fake-launch tests; never launch a benchmark."""

import ast
import builtins
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
SOURCE = ROOT / "scripts/jev_browser_revisit_benchmark.py"
SPEC = importlib.util.spec_from_file_location("pure_revisit_runner_subject", SOURCE)
L = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = L
SPEC.loader.exec_module(L)
P = L.PROFILE_MODULE
CANONICAL = (
    ROOT
    / "artifacts/jev-runtime/public-native-hover-settled-v8-20261019-01/profile-01/parent_v5/parent/canonical"
)


@pytest.fixture(autouse=True)
def no_actor_or_child(monkeypatch):
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        if name.split(".", 1)[0] in {
            "torch",
            "transformers",
            "httpx",
            "ctypes",
            "playwright",
            "selenium",
            "modal",
        }:
            raise AssertionError("Forbidden actor dependency: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)
    monkeypatch.setattr(
        L.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("No real benchmark child allowed"),
    )


@pytest.mark.parametrize("case_module", ["original", "heldout"])
def test_both_bundles_are_closed_and_exact_small_derivatives(tmp_path, case_module):
    output, manifest = L.prepare_bundle(tmp_path / "runner", case_module=case_module)
    L.verify_bundle(output, manifest)
    name, expected = L.BASES[case_module]
    base = ROOT / "artifacts/jev-runtime/public-native-hover-settled-v8-20261019-01" / name
    assert (
        hashlib.sha256((output / "parent-bundle-manifest.json").read_bytes()).hexdigest()
        == expected
    )
    runner = (output / "runner.py").read_bytes()
    inverse = (
        runner.replace(
            b"import jev_observation_profile_revisit as profile",
            b"import jev_observation_profile_v8 as profile",
        )
        .replace(
            (f'PRUNED_OBSERVATION_PROFILE = "{L.PROFILE}"').encode(),
            b'PRUNED_OBSERVATION_PROFILE = "composed-visible-hover-entries-settled-v8"',
        )
        .replace(
            b"from jev_observation_events_revisit import ObservationProfileTracker",
            b"from jev_observation_events_v8 import ObservationProfileTracker",
        )
    )
    assert inverse == (base / "runner.py").read_bytes()
    assert ast.dump(ast.parse(inverse), include_attributes=False) == ast.dump(
        ast.parse((base / "runner.py").read_bytes()), include_attributes=False
    )
    events = (output / "jev_observation_events_revisit.py").read_bytes()
    assert (
        events.replace(
            (f'PRUNED_PROFILE = "{L.PROFILE}"').encode(),
            b'PRUNED_PROFILE = "composed-visible-hover-entries-settled-v8"',
        )
        == (base / "jev_observation_events_v8.py").read_bytes()
    )
    assert manifest["artifact_files_including_manifest"] == len(manifest["files"]) + 1 <= 64
    assert manifest["old_V8_or_Q95_qualification_covers_new_prompt"] is False
    assert (output / "launcher.py").read_bytes() == SOURCE.read_bytes()
    assert not (output / "jev_browser_profile_v8_benchmark.py").exists()
    assert (
        manifest["profile_helper_sha256"]
        == hashlib.sha256((output / "jev_observation_profile_revisit.py").read_bytes()).hexdigest()
    )


def test_new_handle_satisfies_actual_generic_benchmark_profile_seams(tmp_path, monkeypatch):
    output, _manifest = L.prepare_bundle(tmp_path / "runner")
    nodes = {
        n.name: n
        for n in ast.parse((output / "runner.py").read_bytes()).body
        if isinstance(n, ast.FunctionDef)
    }
    names = [
        "prepare_observation_profile",
        "_verify_observation_runtime",
        "_observation_profile_identity",
        "_assert_observation_imports",
    ]
    namespace = {
        "Path": Path,
        "hashlib": hashlib,
        "sys": sys,
        "NATIVE_OBSERVATION_PROFILE": "native-1231850",
        "COMPOSED_OBSERVATION_PROFILE": "composed-visible-text-v2",
        "PRUNED_OBSERVATION_PROFILE": P.PROFILE,
        "upstream_manifest": lambda _root: {"source_sha256": dict(P.CANONICAL_SHA256)},
    }
    monkeypatch.setitem(sys.modules, "jev_observation_profile_revisit", P)
    exec(
        compile(
            ast.Module(body=[nodes[name] for name in names], type_ignores=[]),
            "<actual-generic-runner-function-AST-only>",
            "exec",
        ),
        namespace,
    )
    bundle = namespace["prepare_observation_profile"](CANONICAL, P.PROFILE, L.PROFILE_SHA)
    result = P.materialize_profile(
        CANONICAL, tmp_path / "profile", expected_source_sha256=L.PROFILE_SHA
    )
    runtime = P.verify_materialized_profile(
        tmp_path / "profile",
        expected_manifest_sha256=result["manifest_sha256"],
        expected_source_sha256=L.PROFILE_SHA,
    )
    verified = namespace["_verify_observation_runtime"](CANONICAL, runtime)
    identity = namespace["_observation_profile_identity"](verified)
    assert (
        identity["id"] == P.PROFILE
        and identity["runtime_source_sha256"] == bundle.manifest()["runtime_source_sha256"]
    )
    for label in runtime.manifest()["runtime_source_sha256"]:
        if label.endswith(".py"):
            name = label.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
            fake = SimpleNamespace(__file__=str(runtime.runtime_root / label))
            if name == "jev_ultrafast.browser":
                fake.READ_STATE, fake.MARKER = runtime.read_state, runtime.marker_expression
            monkeypatch.setitem(sys.modules, name, fake)
    namespace["_assert_observation_imports"](runtime)


@pytest.mark.parametrize("change", ["source", "extra", "symlink", "manifest"])
def test_runner_tamper_and_population_change_refuse(tmp_path, change):
    output, manifest = L.prepare_bundle(tmp_path / "runner")
    if change == "source":
        (output / "runner.py").write_bytes(b"# changed\n")
    elif change == "extra":
        (output / "unexpected").mkdir()
    elif change == "symlink":
        (output / "runner.py").unlink()
        (output / "runner.py").symlink_to(output / "launcher.py")
    else:
        (output / "bundle-manifest.json").write_text("{}\n")
    with pytest.raises(ValueError):
        L.verify_bundle(output, manifest)


def test_default_is_prepare_only_and_run_without_both_pins_never_starts(tmp_path, capsys):
    assert L.main(["--bundle-dir", str(tmp_path / "dry")]) == 0
    assert json.loads(capsys.readouterr().out)["model_or_browser_executed"] is False
    with pytest.raises(SystemExit) as refused:
        L.main(
            [
                "--bundle-dir",
                str(tmp_path / "refused"),
                "--run",
                "--profile-source-sha256",
                L.PROFILE_SHA,
            ]
        )
    assert refused.value.code == 2


def run_args(path):
    return [
        "--bundle-dir",
        str(path),
        "--run",
        "--profile-source-sha256",
        L.PROFILE_SHA,
        "--launcher-source-sha256",
        hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "--",
        "--upstream-path",
        "unchanged-upstream",
        "--endpoint",
        "http://127.0.0.1:9999",
        "--model",
        "fake-policy",
        "--output-dir",
        "unchanged-output",
    ]


def test_explicit_run_wiring_is_exercised_with_fake_child_only(tmp_path, monkeypatch):
    calls = []

    def fake(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(L.subprocess, "run", fake)
    output = tmp_path / "runner"
    assert L.main(run_args(output)) == 7
    assert len(calls) == 1 and calls[0][0][:3] == [sys.executable, "-B", "-s"]
    assert calls[0][0][3] == str(output / "runner.py")
    assert L.PROFILE in calls[0][0] and L.PROFILE_SHA in calls[0][0]
    receipt = json.loads((output / "launch-receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["returncode"] == 7
    assert receipt["child_exit_is_task_success"] is False


def test_fake_child_primary_is_preserved_if_final_integrity_also_fails(tmp_path, monkeypatch):
    primary = RuntimeError("fake child primary")
    output = tmp_path / "runner"

    def fake(*_args, **_kwargs):
        (output / "runner.py").write_bytes(b"# tampered during fake child\n")
        raise primary

    monkeypatch.setattr(L.subprocess, "run", fake)
    with pytest.raises(RuntimeError) as captured:
        L.main(run_args(output))
    assert captured.value is primary
    assert any("closure evidence" in note for note in primary.__notes__)


def test_heldout_arguments_preserve_original_fixed_caps_and_require_new_identity():
    base = [
        "--upstream-path",
        "unchanged-upstream",
        "--endpoint",
        "http://127.0.0.1:9999",
        "--model",
        "fake-policy",
        "--output-dir",
        "unchanged-output",
    ]
    forwarded = L._arguments(base, "heldout")
    assert forwarded[: len(base)] == base
    assert forwarded[forwarded.index("--observation-profile") + 1] == L.PROFILE
    cases = forwarded[forwarded.index("--public-cases") + 1 :]
    assert len(cases) == 6 and len(set(cases)) == 6
    for added in (
        ["--repeats", "2"],
        ["--max-steps", "17"],
        ["--observation-profile", L.PARENT.PROFILE],
        ["--observation-profile-helper-sha256", "0" * 64],
        ["--public-cases", cases[0], cases[0]],
    ):
        with pytest.raises(SystemExit):
            L._arguments(base + added, "heldout")
