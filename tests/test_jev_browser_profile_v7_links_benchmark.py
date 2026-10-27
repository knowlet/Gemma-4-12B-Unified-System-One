"""Source-only V7 links runner tests; no browser or model execution."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jev_browser_profile_v7_links_benchmark as launcher
import pytest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "artifacts/jev-runtime/upstream"


@pytest.mark.parametrize("cases", ["original", "heldout"])
def test_bundle_has_link_profile_only_and_reversible_runner(tmp_path, cases):
    path, manifest = launcher.prepare_bundle(tmp_path / "runner", case_module=cases)
    launcher.verify_bundle(path, manifest)
    assert manifest["profile"] == launcher.PROFILE
    assert (
        manifest["profile_helper_sha256"]
        == hashlib.sha256((path / "jev_observation_profile_v7_links.py").read_bytes()).hexdigest()
    )
    assert (
        hashlib.sha256((path / "jev_observation_profile_v7.py").read_bytes()).hexdigest()
        == launcher.V7_SHA256
    )
    assert not manifest["model_executed"]
    assert not manifest["runtime_qualification_attested"]
    original = (path / "original_jev_browser_benchmark.py").read_bytes()
    derived = (path / "runner.py").read_bytes()
    inverse = derived.replace(b"                        _settling_primary = None\n", b"")
    inverse = inverse.replace(
        b"                        _settling_cursor = profile.v7.settling_cursor(agent_module.Browser)\n",
        b"",
    )
    inverse = inverse.replace(b"                            _settling_primary = exc\n", b"")
    sidecar = (
        b"                            profile.v7.write_settling_sidecar(agent_module.Browser, _settling_cursor,\n"
        b'                                attempt / "navigation-settling.json", _write_json, secrets,\n'
        b"                                _settling_primary if _settling_primary is not None else sys.exception(), _redact)\n"
    )
    assert inverse.count(sidecar) == 1
    inverse = inverse.replace(sidecar, b"")
    for before, after in (
        (launcher.NEW_IMPORT, launcher.OLD_IMPORT),
        (launcher.NEW_CHOICE, launcher.OLD_CHOICE),
        (launcher.NEW_EVENTS_IMPORT, launcher.OLD_EVENTS_IMPORT),
    ):
        inverse = inverse.replace(before, after)
    if cases == "heldout":
        inverse = inverse.replace(launcher.NEW_CASE, launcher.OLD_CASE, 1)
    assert inverse == original
    assert ast.dump(ast.parse(inverse)) == ast.dump(ast.parse(original))
    assert (
        b'PRUNED_PROFILE = "composed-visible-text-settled-v7-links"'
        in (path / "jev_observation_events_v7_links.py").read_bytes()
    )


def test_arguments_require_the_separate_profile_and_pin():
    expected = "a" * 64
    result = launcher._arguments([], "original", expected)
    assert result == [
        "--observation-profile",
        launcher.PROFILE,
        "--observation-profile-helper-sha256",
        expected,
    ]
    with pytest.raises(SystemExit):
        launcher._arguments(
            ["--observation-profile", "composed-visible-text-settled-v7"], "original", expected
        )
    with pytest.raises(SystemExit):
        launcher._arguments(["--observation-profile-helper-sha256", "b" * 64], "original", expected)


def test_bundle_rejects_dependency_drift_and_extra_file(tmp_path):
    path, manifest = launcher.prepare_bundle(tmp_path / "runner")
    dependency = path / "jev_observation_profile_v7.py"
    dependency.write_bytes(dependency.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        launcher.verify_bundle(path, manifest)
    other, other_manifest = launcher.prepare_bundle(tmp_path / "other")
    (other / "unknown").write_text("unknown")
    with pytest.raises(ValueError):
        launcher.verify_bundle(other, other_manifest)


CHILD = r"""
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

bundle, upstream, profile_root, expected = sys.argv[1:]
sys.path.insert(0, bundle)
spec = importlib.util.spec_from_file_location("link_test_runner", Path(bundle) / "runner.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
profile = importlib.import_module("jev_observation_profile_v7_links")
prepared = runner.prepare_observation_profile(upstream, profile.PROFILE, expected)
assert prepared.manifest()["profile"] == profile.PROFILE
result = profile.materialize_profile(upstream, profile_root, expected_source_sha256=expected)
verified = profile.verify_materialized_profile(profile_root,
    expected_manifest_sha256=result["manifest_sha256"], expected_source_sha256=expected)
assert runner._verify_observation_runtime(upstream, verified).manifest()["profile"] == profile.PROFILE
transport = SimpleNamespace(cdp=lambda *a, **kw: None, observation_profile_errors=[])
with runner.load_upstream(upstream, transport, observation_runtime=verified) as (agent, model):
    assert Path(model.__file__).resolve() == verified.runtime_root / "jev_ultrafast/model.py"
    assert model.action_space([]) == ([], {}, {})
    assert transport.observation_profile_expressions[verified.read_state] == "READ_STATE"
assert transport.observation_profile_expressions is None
assert not transport.observation_profile_errors
print(json.dumps({"profile": profile.PROFILE, "verified": True}))
"""


def test_actual_runner_imports_and_verifies_prepared_runtime_without_browser(tmp_path):
    path, manifest = launcher.prepare_bundle(tmp_path / "runner")
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-s",
            "-c",
            CHILD,
            str(path),
            str(UPSTREAM),
            str(tmp_path / "profile"),
            manifest["profile_helper_sha256"],
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"profile": launcher.PROFILE, "verified": True}
