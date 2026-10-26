"""Pure v8 source/closure tests; no browser, model, tokenizer or framework."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jev_browser_profile_v5_benchmark as v5_launcher
import jev_browser_profile_v7_benchmark as v7_launcher
import jev_browser_profile_v8_benchmark as launcher
import jev_observation_profile_v8 as profile
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = hashlib.sha256(Path(profile.__file__).read_bytes()).hexdigest()
UPSTREAM = Path(os.environ.get("JEV_UPSTREAM_PATH", ROOT / "artifacts/jev-runtime/upstream"))
COMPOSITION = (
    ROOT / "artifacts/jev-runtime/public-affordance-v5-v7-composition-20261019-01/derived-source-01"
)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


@pytest.fixture(scope="module")
def canonical():
    if not UPSTREAM.is_dir():
        pytest.skip("Existing pinned canonical source checkout required; never downloads")
    return profile.read_canonical_sources(UPSTREAM)


@pytest.fixture(scope="module")
def bundle(canonical):
    return profile.build_profile(canonical, expected_source_sha256=SOURCE_SHA)


def isolated(directory, expression):
    code = (
        "import sys,json;sys.path.insert(0,sys.argv[1]);import jev_observation_profile_v8 as p;"
        + expression
    )
    return subprocess.run(
        [sys.executable, "-I", "-B", "-s", "-c", code, str(directory)],
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_exact_reviewed_composition_runtime_and_both_parent_inverses(bundle):
    if not COMPOSITION.is_dir():
        pytest.skip("Saved reviewed composition reference is local integration evidence")
    proof = COMPOSITION / "source-proof.json"
    assert (
        sha(proof.read_bytes())
        == "12c737f55296bc0e5f04d17972cdaa0185990ba3fc88de057adbba3c30720354"
    )
    assert all(
        blob == (COMPOSITION / "runtime" / label).read_bytes()
        for label, blob in bundle.runtime_files.items()
    )
    old = bundle._v5_bundle
    before = ("id:'" + profile._V5.PROFILE + "'").encode()
    after = ("id:'" + profile.PROFILE + "'").encode()
    assert (
        bundle.runtime_files[profile.SNAPSHOT].replace(after, before, 1)
        == old.runtime_files[profile.SNAPSHOT]
    )
    native_v7 = profile._V7.build_profile(
        dict(bundle.canonical_files), expected_source_sha256=profile.V7_SOURCE_SHA256
    )
    browser = bundle.runtime_files[profile.BROWSER]
    proxy = profile._V5._PARENT.PROXY_EXECUTOR.encode()
    assert browser.replace(proxy, b"", 1) == native_v7.runtime_files[profile.BROWSER]
    assert profile._V7._observe_source(browser.decode()) == profile._V7._observe_source(
        native_v7.runtime_files[profile.BROWSER].decode()
    )
    assert bundle.manifest()["offered_capacity"]["appended_proxies_total_bound_attested"] is False
    assert (
        bundle.manifest()["offered_capacity"][
            "universal_400head_8192context_qualification_attested"
        ]
        is False
    )


def test_materialized_complete_source_closure_and_rehashed_runtime_refusal(tmp_path, canonical):
    target = tmp_path / "profile"
    result = profile.materialize_profile(UPSTREAM, target, expected_source_sha256=SOURCE_SHA)
    handle = profile.verify_materialized_profile(
        target,
        expected_manifest_sha256=result["manifest_sha256"],
        expected_source_sha256=SOURCE_SHA,
    )
    manifest = handle.manifest()
    assert {*manifest["artifact_files"], "profile-manifest.json"} == {
        p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()
    }
    assert handle.read_state == (handle.runtime_root / profile.SNAPSHOT).read_text()
    expression = (
        "r=p.verify_materialized_profile(str(p.Path(sys.argv[1]).parent),expected_manifest_sha256='"
        + result["manifest_sha256"]
        + "',expected_source_sha256='"
        + SOURCE_SHA
        + "');print(json.dumps({'profile':r.manifest()['profile'],'origins':[m.__file__ for m in [p,p._V5,p._V7,p._V5._PARENT,p._V7._PARENT]],'frameworks':[n for n in sys.modules if n.split('.')[0] in {'torch','transformers','playwright','httpx'}]}))"
    )
    child = isolated(target / "sources", expression)
    assert child.returncode == 0, child.stderr
    observed = json.loads(child.stdout)
    assert observed["profile"] == profile.PROFILE and observed["frameworks"] == []
    assert len(observed["origins"]) == 5 and all(
        Path(path).parent == target / "sources" for path in observed["origins"]
    )
    browser = handle.runtime_root / profile.BROWSER
    browser.write_bytes(browser.read_bytes() + b"\n# unauthorized runtime\n")
    label = "runtime/" + profile.BROWSER
    manifest["artifact_files"][label] = {
        "sha256": sha(browser.read_bytes()),
        "size_bytes": browser.stat().st_size,
    }
    changed = profile._encode(manifest) + b"\n"
    (target / "profile-manifest.json").write_bytes(changed)
    with pytest.raises(ValueError, match="source/metadata"):
        profile.verify_materialized_profile(
            target, expected_manifest_sha256=sha(changed), expected_source_sha256=SOURCE_SHA
        )


@pytest.mark.parametrize("kind", ["extra", "directory", "symlink"])
def test_complete_population_refuses_extras_and_aliases(tmp_path, canonical, kind):
    target = tmp_path / "profile"
    result = profile.materialize_profile(UPSTREAM, target, expected_source_sha256=SOURCE_SHA)
    if kind == "extra":
        (target / "extra.py").write_text("pass\n")
    elif kind == "directory":
        (target / "unknown").mkdir()
    else:
        (target / "alias").symlink_to(target / "profile-manifest.json")
    with pytest.raises(ValueError):
        profile.verify_materialized_profile(
            target,
            expected_manifest_sha256=result["manifest_sha256"],
            expected_source_sha256=SOURCE_SHA,
        )


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_adjacent_parent_copies_cannot_fall_back_to_repository(tmp_path, mutation):
    for name, blob in profile._dependency_sources().items():
        (tmp_path / name).write_bytes(blob)
    (tmp_path / "jev_observation_profile_v8.py").write_bytes(Path(profile.__file__).read_bytes())
    dependency = tmp_path / "jev_observation_profile_v7.py"
    if mutation == "missing":
        dependency.unlink()
    else:
        dependency.write_bytes(dependency.read_bytes() + b"\n# source drift\n")
    child = isolated(tmp_path, "print(p.PROFILE)")
    assert child.returncode != 0 and "adjacent dependency" in child.stderr


@pytest.mark.parametrize("case_module", ["original", "heldout"])
def test_launcher_original_inverse_and_v5_v7_tracker_deltas(tmp_path, case_module):
    b8, manifest = launcher.prepare_bundle(tmp_path / "v8", case_module=case_module)
    b7, _ = v7_launcher.prepare_bundle(tmp_path / "v7", case_module=case_module)
    current = (b8 / "runner.py").read_bytes()
    inverse = (
        current.replace(launcher.NEW_IMPORT, v7_launcher.NEW_IMPORT)
        .replace(launcher.NEW_CHOICE, v7_launcher.NEW_CHOICE)
        .replace(launcher.NEW_EVENTS_IMPORT, v7_launcher.NEW_EVENTS_IMPORT)
    )
    original = (b7 / "runner.py").read_bytes()
    assert inverse == original and ast.dump(
        ast.parse(inverse), include_attributes=False
    ) == ast.dump(ast.parse(original), include_attributes=False)
    events = (b8 / "jev_observation_events_v8.py").read_bytes()
    assert (
        events.replace(launcher.NEW_EVENTS_CHOICE, v7_launcher.NEW_EVENTS_CHOICE).replace(
            launcher.HOVER_CAP_REPLACEMENT, launcher.HOVER_CAP_ANCHOR, 1
        )
        == (b7 / "jev_observation_events_v7.py").read_bytes()
    )
    parent = (b8 / "jev_observation_events_v3_parent.py").read_bytes()
    v5_events = parent.replace(
        v5_launcher.OLD_EVENTS_CHOICE, v5_launcher.NEW_EVENTS_CHOICE
    ).replace(v5_launcher.HOVER_CAP_ANCHOR, v5_launcher.HOVER_CAP_REPLACEMENT, 1)
    assert events.replace(launcher.NEW_EVENTS_CHOICE, v5_launcher.NEW_EVENTS_CHOICE) == v5_events
    assert (
        manifest["source_delta"]["inverse_bytes_equal"]
        and manifest["source_delta"]["tracker_inverse_bytes_equal"]
    )
    assert manifest["runtime_qualification_attested"] is False
    child = isolated(
        b8, "print(json.dumps([m.__file__ for m in [p,p._V5,p._V7,p._V5._PARENT,p._V7._PARENT]]))"
    )
    assert child.returncode == 0, child.stderr
    assert all(Path(path).parent == b8 for path in json.loads(child.stdout))


def test_explicit_v8_identity_and_unchanged_heldout_decision_budget():
    with pytest.raises(ValueError, match="Explicit v8"):
        profile.build_profile(
            {}, expected_source_sha256=SOURCE_SHA, profile="composed-visible-text-settled-v7"
        )
    with pytest.raises(SystemExit):
        launcher._arguments(
            ["--observation-profile", "composed-visible-text-settled-v7"], "original", SOURCE_SHA
        )
    with pytest.raises(SystemExit):
        launcher._arguments(
            [
                "--upstream-path",
                "unused",
                "--endpoint",
                "unused",
                "--model",
                "unused",
                "--output-dir",
                "unused",
                "--max-steps",
                "17",
            ],
            "heldout",
            SOURCE_SHA,
        )


def test_v8_sidecar_identity_and_preserved_primary_error():
    browser = SimpleNamespace(observation_settling_records=({"terminal_unknown": False},))
    written = []
    result = profile.write_settling_sidecar(
        browser, 0, Path("unused"), lambda *args: written.append(args), []
    )
    assert (
        result["profile"] == profile.PROFILE
        and result["schema_id"] == "jev-v8-native-observation-settling-sidecar-v1"
    )
    assert len(written) == 1 and result["task_success_attested"] is False
    primary = ValueError("original failure")

    def failed(*args):
        raise OSError("evidence writer failed")

    with pytest.raises(ValueError) as captured:
        profile.write_settling_sidecar(browser, 0, Path("unused"), failed, [], primary=primary)
    assert captured.value is primary and browser._observation_settling_fault and primary.__notes__
