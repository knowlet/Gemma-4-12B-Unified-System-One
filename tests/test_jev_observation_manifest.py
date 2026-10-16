"""Bounded pure source/copy/verification tests; no browser, model or network."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from s1 import jev_observation_profile as p

HELPER_SHA = hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest()
MANIFEST_KEYS = {
    "schema",
    "profile",
    "canonical_contract_revision",
    "canonical_Git_verification_performed_by_this_library",
    "canonical_Git_prerequisite",
    "canonical_source_sha256",
    "runtime_source_sha256",
    "helper_source_sha256",
    "runtime_root",
    "canonical_copy_root",
    "helper_copy",
    "derivation",
    "limits",
    "semantics",
    "not_attested_by_this_library",
    "artifact_files",
}
CANONICAL_FILES = {
    "jev_ultrafast/__init__.py",
    "jev_ultrafast/agent.py",
    "jev_ultrafast/browser.py",
    "jev_ultrafast/model.py",
    "jev_ultrafast/questions.py",
    "jev_ultrafast/snapshot.js",
    "jev_ultrafast/static/fixture.html",
}
ARTIFACT_FILES = {
    prefix + "/" + label for prefix in ("canonical", "runtime") for label in CANONICAL_FILES
} | {"sources/jev_observation_profile.py"}
CANONICAL_SNAPSHOT_SHA = "e50473501c8fb8e70f3b21866d987393e3f2315c639d638bd477d170e81ed78d"
COMPOSED_SNAPSHOT_SHA = "5327d4ab5d615ac0804f4913bb86bb59b383b11ff80f2408eb4aa565243f9dd5"


def canonical_path():
    configured = os.environ.get("JEV_UPSTREAM_PATH")
    if not configured:
        pytest.skip("Set JEV_UPSTREAM_PATH to an existing pinned checkout; never fetches")
    return Path(configured)


@pytest.fixture(autouse=True)
def require_configured_upstream():
    canonical_path()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


@pytest.fixture
def canonical():
    return p.read_canonical_sources(canonical_path())


def build(canonical, profile=p.NATIVE_PROFILE):
    return p.build_profile(canonical, expected_source_sha256=HELPER_SHA, profile=profile)


def materialize(tmp_path, profile=p.COMPOSED_PROFILE):
    output = tmp_path / "profile"
    result = p.materialize_profile(
        canonical_path(), output, expected_source_sha256=HELPER_SHA, profile=profile
    )
    return output, result


def verify(output, result, profile=p.COMPOSED_PROFILE):
    return p.verify_materialized_profile(
        output,
        expected_manifest_sha256=result["manifest_sha256"],
        expected_source_sha256=HELPER_SHA,
        profile=profile,
    )


@pytest.mark.parametrize("profile", [p.NATIVE_PROFILE, p.COMPOSED_PROFILE])
def test_complete_manifest_matches_published_contract(canonical, profile):
    bundle = build(canonical, profile)
    manifest = bundle.materialized_manifest()
    assert set(manifest) == MANIFEST_KEYS and len(MANIFEST_KEYS) == 16
    assert set(bundle.manifest()) == MANIFEST_KEYS - {"artifact_files"}
    assert manifest["schema"] == "explicit-snapshot-profile-v1"
    assert manifest["profile"] == profile
    assert manifest["canonical_contract_revision"] == "1231850a0bf1a0c0341fe408ef1668dbbfdfac46"
    assert manifest["runtime_root"] == "runtime"
    assert manifest["canonical_copy_root"] == "canonical"
    assert manifest["helper_copy"] == "sources/jev_observation_profile.py"
    assert manifest["canonical_Git_verification_performed_by_this_library"] is False
    assert set(manifest["canonical_source_sha256"]) == CANONICAL_FILES
    assert set(manifest["runtime_source_sha256"]) == CANONICAL_FILES
    assert set(manifest["artifact_files"]) == ARTIFACT_FILES and len(ARTIFACT_FILES) == 15
    assert all(set(pin) == {"sha256", "size_bytes"} for pin in manifest["artifact_files"].values())
    helper = Path(p.__file__).read_bytes()
    assert manifest["helper_source_sha256"] == HELPER_SHA
    assert manifest["artifact_files"]["sources/jev_observation_profile.py"] == {
        "sha256": HELPER_SHA,
        "size_bytes": len(helper),
    }
    assert (
        manifest["canonical_source_sha256"]["jev_ultrafast/snapshot.js"] == CANONICAL_SNAPSHOT_SHA
    )
    runtime_sha = CANONICAL_SNAPSHOT_SHA if profile == p.NATIVE_PROFILE else COMPOSED_SNAPSHOT_SHA
    assert manifest["runtime_source_sha256"]["jev_ultrafast/snapshot.js"] == runtime_sha
    assert (
        manifest["artifact_files"]["canonical/jev_ultrafast/snapshot.js"]["sha256"]
        == CANONICAL_SNAPSHOT_SHA
    )
    assert manifest["artifact_files"]["runtime/jev_ultrafast/snapshot.js"]["sha256"] == runtime_sha


def test_complete_manifest_is_pure_and_detached(canonical, monkeypatch):
    bundle = build(canonical, p.COMPOSED_PROFILE)
    expected = bundle.materialized_manifest()

    def no_io(*_, **__):
        raise AssertionError("Complete manifest must use captured immutable bytes only")

    monkeypatch.setattr(Path, "read_bytes", no_io)
    monkeypatch.setattr(Path, "open", no_io)
    detached = bundle.materialized_manifest()
    detached["artifact_files"].clear()
    detached["runtime_source_sha256"][p.SNAPSHOT] = "caller-mutated"
    assert bundle.materialized_manifest() == expected


@pytest.mark.parametrize("profile", [p.NATIVE_PROFILE, p.COMPOSED_PROFILE])
def test_positive_materialize_and_verify_exact_immutable_identity(tmp_path, canonical, profile):
    output, result = materialize(tmp_path, profile)
    bundle = build(canonical, profile)
    expected_blob = p._encode(bundle.materialized_manifest()) + b"\n"
    assert (output / "profile-manifest.json").read_bytes() == expected_blob
    assert result["manifest_sha256"] == sha(expected_blob)
    handle = verify(output, result, profile)
    assert handle.path == output and handle.runtime_root == output / "runtime"
    assert handle.manifest_sha256 == result["manifest_sha256"]
    assert handle.manifest() == bundle.materialized_manifest()
    assert (
        handle.read_state == bundle.read_state
        and handle.marker_expression == bundle.marker_expression
    )
    assert dict(handle.bundle.canonical_files) == canonical
    assert dict(handle.bundle.runtime_files) == dict(bundle.runtime_files)
    assert handle.manifest()["canonical_Git_verification_performed_by_this_library"] is False
    assert not (output / ".profile-manifest.pending").exists()
    with pytest.raises(FrozenInstanceError):
        handle.manifest_sha256 = "new"
    with pytest.raises(TypeError):
        handle.bundle.runtime_files[p.SNAPSHOT] = b"new"
    detached = handle.manifest()
    detached["artifact_files"].clear()
    assert len(handle.manifest()["artifact_files"]) == 15


def test_native_default_runtime_remains_exact_all_seven(tmp_path, canonical):
    output, result = materialize(tmp_path, p.NATIVE_PROFILE)
    handle = verify(output, result, p.NATIVE_PROFILE)
    assert dict(handle.bundle.runtime_files) == canonical
    assert handle.manifest()["derivation"] is handle.manifest()["limits"] is None
    assert "observation_profile:textProfile" not in handle.read_state


@pytest.mark.parametrize("label", sorted(ARTIFACT_FILES))
def test_every_source_copy_is_authenticated_even_if_manifest_rehashed(tmp_path, label):
    output, result = materialize(tmp_path)
    target = output / label
    target.write_bytes(target.read_bytes() + b"\n# malicious data, never executable\n")
    manifest = json.loads((output / "profile-manifest.json").read_bytes())
    manifest["artifact_files"][label] = {
        "sha256": sha(target.read_bytes()),
        "size_bytes": target.stat().st_size,
    }
    blob = p._encode(manifest) + b"\n"
    (output / "profile-manifest.json").write_bytes(blob)
    # Caller-provided artifact hash is not authority for canonical/derived/helper bytes.
    result["manifest_sha256"] = sha(blob)
    with pytest.raises(ValueError, match="Canonical source mismatch|metadata/profile|bytes differ"):
        verify(output, result)


@pytest.mark.parametrize(
    "label",
    [
        "runtime/jev_ultrafast/snapshot.js",
        "sources/jev_observation_profile.py",
        "profile-manifest.json",
    ],
)
def test_missing_source_or_manifest_refused(tmp_path, label):
    output, result = materialize(tmp_path)
    (output / label).unlink()
    with pytest.raises(ValueError, match="Missing materialized"):
        verify(output, result)


@pytest.mark.parametrize("kind", ["extra-file", "empty-dir", "nested-dir", "pending-file"])
def test_extra_files_and_unknown_empty_nested_directories_refused(tmp_path, kind):
    output, result = materialize(tmp_path)
    if kind == "extra-file":
        (output / "runtime/jev_ultrafast/unreviewed.py").write_bytes(
            b"raise Exception('never import')"
        )
    elif kind == "empty-dir":
        (output / "unused").mkdir()
    elif kind == "nested-dir":
        (output / "runtime/jev_ultrafast/unknown/nested").mkdir(parents=True)
    else:
        (output / ".profile-manifest.pending").write_bytes(b"partial")
    with pytest.raises(ValueError, match="Unknown"):
        verify(output, result)


@pytest.mark.parametrize("kind", ["file", "directory", "root", "ancestor"])
def test_symlink_source_runtime_root_and_parent_refused(tmp_path, kind):
    output, result = materialize(tmp_path)
    if kind == "file":
        path = output / "runtime/jev_ultrafast/model.py"
        path.unlink()
        path.symlink_to(output / "canonical/jev_ultrafast/model.py")
    elif kind == "directory":
        path = output / "runtime/jev_ultrafast"
        shutil.rmtree(path)
        path.symlink_to(output / "canonical/jev_ultrafast", target_is_directory=True)
    elif kind == "root":
        alias = tmp_path / "root-alias"
        alias.symlink_to(output, target_is_directory=True)
        output = alias
    else:
        alias = tmp_path / "parent-alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        output = alias / "profile"
    with pytest.raises(ValueError, match="symlink|Ordinary canonical"):
        verify(output, result)


def test_special_file_refused_without_opening_FIFO(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO regression requires local mkfifo support")
    output, result = materialize(tmp_path)
    path = output / "runtime/jev_ultrafast/model.py"
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(ValueError, match="Ordinary profile file"):
        verify(output, result)


@pytest.mark.parametrize(
    "field",
    [
        "profile",
        "canonical_Git_verification_performed_by_this_library",
        "runtime_root",
        "helper_source_sha256",
        "runtime_source_sha256",
        "limits",
        "unknown",
    ],
)
def test_rehashed_tampered_manifest_metadata_is_not_authority(tmp_path, field):
    output, result = materialize(tmp_path)
    manifest = json.loads((output / "profile-manifest.json").read_bytes())
    manifest[field] = "attacker-chosen"
    blob = p._encode(manifest) + b"\n"
    (output / "profile-manifest.json").write_bytes(blob)
    result["manifest_sha256"] = sha(blob)
    with pytest.raises(ValueError, match="metadata/profile differs"):
        verify(output, result)


def test_wrong_manifest_pin_and_unknown_profile_refused(tmp_path):
    output, result = materialize(tmp_path)
    with pytest.raises(ValueError, match="manifest hash differs"):
        verify(output, {**result, "manifest_sha256": "0" * 64})
    for wrong in (None, "A" * 64, "bad"):
        with pytest.raises(ValueError, match="Explicit reviewed materialized"):
            verify(output, {**result, "manifest_sha256": wrong})
    with pytest.raises(ValueError, match="Unknown explicit"):
        verify(output, result, "native")
    with pytest.raises(ValueError, match="metadata/profile"):
        verify(output, result, p.NATIVE_PROFILE)


def test_unreviewed_mutable_verifier_refused_before_artifact_read(tmp_path, monkeypatch):
    output, result = materialize(tmp_path)
    copy_path = tmp_path / "changed-helper.py"
    copy_path.write_bytes(Path(p.__file__).read_bytes() + b"\n# changed verifier\n")
    monkeypatch.setattr(p, "__file__", str(copy_path))

    def no_read(*_, **__):
        raise AssertionError("Do not read artifacts before authenticating verifier")

    monkeypatch.setattr(p, "_capture_materialized_tree", no_read)
    with pytest.raises(ValueError, match="Reviewed helper source changed"):
        verify(output, result)


def test_closed_inventory_change_after_bytes_checked_refused(tmp_path, monkeypatch):
    output, result = materialize(tmp_path)
    original = p._capture_materialized_tree

    def changing(root, *, read_contents):
        if not read_contents:
            path = root / "runtime/jev_ultrafast/model.py"
            os.utime(path, ns=(0, 0))
        return original(root, read_contents=read_contents)

    monkeypatch.setattr(p, "_capture_materialized_tree", changing)
    with pytest.raises(ValueError, match="closed inventory changed"):
        verify(output, result)


def test_artifact_helper_python_is_never_executed(tmp_path):
    output, result = materialize(tmp_path)
    marker = tmp_path / "must-not-exist"
    malicious = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n".encode()
    (output / "sources/jev_observation_profile.py").write_bytes(malicious)
    with pytest.raises(ValueError, match="bytes differ"):
        verify(output, result)
    assert not marker.exists()


@pytest.mark.parametrize("cap", ["file", "total", "count"])
def test_bounded_file_total_and_count_caps(tmp_path, monkeypatch, cap):
    output, result = materialize(tmp_path)
    if cap == "file":
        (output / "runtime/jev_ultrafast/model.py").write_bytes(b"x" * (p.MAX_SOURCE_BYTES + 1))
    elif cap == "total":
        monkeypatch.setattr(p, "MAX_MATERIALIZED_BYTES", 1)
    else:
        monkeypatch.setattr(p, "MAX_MATERIALIZED_FILES", 15)
    with pytest.raises(ValueError, match="exceeds bound"):
        verify(output, result)


def test_read_only_verifier_does_not_call_Git_or_execute_runtime(tmp_path, monkeypatch):
    output, result = materialize(tmp_path)
    before = {
        str(x.relative_to(output)): sha(x.read_bytes()) for x in output.rglob("*") if x.is_file()
    }

    def no_execution(*_, **__):
        raise AssertionError("Pure verification must not launch a subprocess")

    monkeypatch.setattr(subprocess, "run", no_execution)
    monkeypatch.setattr(importlib.util, "spec_from_file_location", no_execution)
    loaded_before = set(sys.modules)
    handle = verify(output, result)
    after = {
        str(x.relative_to(output)): sha(x.read_bytes()) for x in output.rglob("*") if x.is_file()
    }
    assert (
        before == after
        and handle.manifest()["canonical_Git_verification_performed_by_this_library"] is False
    )
    assert not (
        {"torch", "transformers", "playwright", "jev_ultrafast"}
        & (set(sys.modules) - loaded_before)
    )


def test_verified_v2_JS_and_marker_parse_without_browser(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node unavailable; no JS parse claim")
    output, result = materialize(tmp_path)
    handle = verify(output, result)
    script = "const vm=require('vm');for(const s of JSON.parse(require('fs').readFileSync(0,'utf8')))new vm.Script(s);process.stdout.write('parsed2');"
    completed = subprocess.run(
        [node, "-e", script],
        input=json.dumps([handle.read_state, handle.marker_expression]),
        text=True,
        capture_output=True,
        check=True,
    )
    assert completed.stdout == "parsed2"
