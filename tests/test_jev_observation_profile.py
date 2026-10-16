"""Pure source/copy tests only; no Chrome, model, Torch or public requests."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from s1 import jev_observation_profile as p

HELPER_SHA = hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest()


@pytest.fixture
def canonical():
    configured = os.environ.get("JEV_UPSTREAM_PATH")
    if not configured:
        pytest.skip("set JEV_UPSTREAM_PATH to an existing pinned Jev checkout; never fetches")
    return p.read_canonical_sources(configured)


def build(canonical, profile=p.NATIVE_PROFILE):
    return p.build_profile(canonical, profile=profile, expected_source_sha256=HELPER_SHA)


def copied_canonical(tmp_path, canonical):
    directory = tmp_path / "canonical"
    for label, blob in canonical.items():
        path = directory / label
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    return directory


def test_native_default_is_byte_exact_all_seven(canonical):
    bundle = p.build_profile(canonical, expected_source_sha256=HELPER_SHA)
    assert dict(bundle.runtime_files) == dict(bundle.canonical_files) == canonical
    manifest = bundle.manifest()
    assert manifest["profile"] == p.NATIVE_PROFILE
    assert manifest["runtime_source_sha256"] == manifest["canonical_source_sha256"]
    assert manifest["limits"] is manifest["derivation"] is None
    assert manifest["canonical_Git_verification_performed_by_this_library"] is False
    assert "observation_profile:textProfile" not in bundle.read_state


@pytest.mark.parametrize(
    "alias",
    ["native", "canonical", "composed-visible-text-v1-experimental", " composed-visible-text-v2"],
)
def test_profile_aliases_are_rejected_without_fallback(canonical, alias):
    with pytest.raises(ValueError, match="Unknown explicit"):
        build(canonical, alias)


@pytest.mark.parametrize("label", list(p.CANONICAL_SHA256))
def test_every_canonical_source_hash_is_required(canonical, label):
    canonical[label] += b"\n"
    with pytest.raises(ValueError, match="Canonical source mismatch"):
        build(canonical, p.COMPOSED_PROFILE)


def test_missing_extra_or_mutable_source_refused(canonical):
    for mutated in (
        {k: v for k, v in canonical.items() if k != p.SNAPSHOT},
        {**canonical, "extra.py": b"x"},
    ):
        with pytest.raises(ValueError, match="Exactly seven"):
            build(mutated)
    canonical[p.SNAPSHOT] = bytearray(canonical[p.SNAPSHOT])
    with pytest.raises(ValueError, match="immutable source bytes"):
        build(canonical)


def test_unreviewed_helper_pin_refused(canonical):
    with pytest.raises(ValueError, match="Reviewed helper source changed"):
        p.build_profile(canonical, expected_source_sha256="0" * 64)
    with pytest.raises(ValueError, match="Explicit reviewed"):
        p.build_profile(canonical, expected_source_sha256=None)


def test_composed_delta_is_generic_walker_and_limit_throw_only(canonical):
    bundle = build(canonical, p.COMPOSED_PROFILE)
    assert [k for k in canonical if canonical[k] != bundle.runtime_files[k]] == [p.SNAPSHOT]
    native, derived = canonical[p.SNAPSHOT].decode(), bundle.read_state
    begin, end = "  const words=[], walker=", "  const text=words.join"
    assert derived[: derived.index("  const words=[], range=")] == native[: native.index(begin)]
    expected_tail = native[native.index(end) :].replace(
        "  return {url:location.href,title:document.title,w:innerWidth,h:innerHeight,text,",
        "  return {observation_profile:textProfile,url:location.href,title:document.title,w:innerWidth,h:innerHeight,text,",
    )
    assert derived[derived.index(end) :] == expected_tail
    assert (
        derived.index("throw error")
        < derived.index(end)
        < derived.index("return {observation_profile")
    )
    assert "composed-visible-text-v1-experimental" not in derived
    assert "COMPOSED_VISIBLE_TEXT_LIMIT " in bundle.marker_expression
    assert (
        bundle.marker_expression
        == f"(() => {{ const state={derived}; return state?.marker ?? null; }})()"
    )


def test_walker_is_exact_reviewed_parent_with_two_declared_changes():
    suffix = "  if (textProfile.node_limit_reached || textProfile.depth_limit_reached) {\n    const error=new Error('COMPOSED_VISIBLE_TEXT_LIMIT '+JSON.stringify(textProfile));\n    error.name='ComposedVisibleTextLimitError';throw error;\n  }\n"
    assert p.REPLACEMENT.endswith(suffix)
    parent = p.REPLACEMENT[: -len(suffix)].replace(
        p.COMPOSED_PROFILE, "composed-visible-text-v1-experimental"
    )
    # Held corrected H02 walker literal, independent of ignored local artifacts.
    assert (
        hashlib.sha256(parent.encode()).hexdigest()
        == "c11f1d18575273a1a9bc648dcdbffbd4e708f19b5ccd16757f8ada14ce060595"
    )


def test_bundle_detaches_caller_and_manifest_mutations(canonical):
    bundle = build(canonical, p.COMPOSED_PROFILE)
    before = bundle.read_state
    canonical[p.SNAPSHOT] = b"caller changed"
    manifest = bundle.manifest()
    manifest["runtime_source_sha256"][p.SNAPSHOT] = "wrong"
    assert (
        bundle.read_state == before
        and bundle.manifest()["runtime_source_sha256"][p.SNAPSHOT] != "wrong"
    )
    with pytest.raises(TypeError):
        bundle.runtime_files[p.SNAPSHOT] = b"mutated"


@pytest.mark.parametrize("profile", [p.NATIVE_PROFILE, p.COMPOSED_PROFILE])
def test_new_materialization_copies_complete_canonical_and_runtime(tmp_path, canonical, profile):
    origin = copied_canonical(tmp_path, canonical)
    destination = tmp_path / "profile"
    bundle = build(canonical, profile)
    result = p.materialize_profile(
        origin, destination, profile=profile, expected_source_sha256=HELPER_SHA
    )
    manifest = json.loads((destination / "profile-manifest.json").read_bytes())
    assert (
        result["manifest_sha256"]
        == hashlib.sha256((destination / "profile-manifest.json").read_bytes()).hexdigest()
    )
    assert manifest["profile"] == profile and manifest["helper_source_sha256"] == HELPER_SHA
    assert len(manifest["artifact_files"]) == 15
    assert {
        f.relative_to(destination).as_posix() for f in destination.rglob("*") if f.is_file()
    } == {*manifest["artifact_files"], "profile-manifest.json"}
    for label, pin in manifest["artifact_files"].items():
        blob = (destination / label).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == pin["sha256"] and len(blob) == pin["size_bytes"]
    for label in canonical:
        assert (
            (origin / label).read_bytes()
            == (destination / "canonical" / label).read_bytes()
            == canonical[label]
        )
        assert (destination / "runtime" / label).read_bytes() == bundle.runtime_files[label]
    browser_source = (destination / "runtime/jev_ultrafast/browser.py").read_text()
    assert 'READ_STATE = Path(__file__).with_name("snapshot.js").read_text()' in browser_source
    assert (
        'MARKER = f"(() => {{ const state={READ_STATE}; return state?.marker ?? null; }})()"'
        in browser_source
    )
    assert not (destination / ".profile-manifest.pending").exists()


def test_output_reuse_and_overlap_never_overwrite(tmp_path, canonical):
    origin = copied_canonical(tmp_path, canonical)
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "old"
    marker.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="never overwrite"):
        p.materialize_profile(origin, destination, expected_source_sha256=HELPER_SHA)
    assert marker.read_bytes() == b"preserve"
    with pytest.raises(ValueError, match="overlap"):
        p.materialize_profile(origin, origin / "new", expected_source_sha256=HELPER_SHA)
    assert not (origin / "new").exists()


def test_bad_source_or_pin_creates_no_output(tmp_path, canonical):
    origin = copied_canonical(tmp_path, canonical)
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="helper source changed"):
        p.materialize_profile(origin, destination, expected_source_sha256="0" * 64)
    assert not destination.exists()
    (origin / p.SNAPSHOT).write_bytes(b"unreviewed")
    with pytest.raises(ValueError, match="Canonical source mismatch"):
        p.materialize_profile(origin, destination, expected_source_sha256=HELPER_SHA)
    assert not destination.exists()


def test_symlink_source_or_output_parent_refused(tmp_path, canonical):
    origin = copied_canonical(tmp_path, canonical)
    alias = tmp_path / "alias"
    alias.symlink_to(origin, target_is_directory=True)
    with pytest.raises(ValueError, match="Ordinary canonical"):
        p.materialize_profile(alias, tmp_path / "out", expected_source_sha256=HELPER_SHA)
    with pytest.raises(ValueError, match="Ordinary canonical"):
        p.materialize_profile(origin, alias / "out", expected_source_sha256=HELPER_SHA)


def test_actual_derived_read_state_and_marker_parse_in_node(canonical):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node unavailable; no browser syntax claim")
    bundle = build(canonical, p.COMPOSED_PROFILE)
    code = "const vm=require('vm');for(const s of JSON.parse(require('fs').readFileSync(0,'utf8')))new vm.Script(s);process.stdout.write('parsed2');"
    result = subprocess.run(
        [node, "-e", code],
        input=json.dumps([bundle.read_state, bundle.marker_expression]),
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout == "parsed2"


def test_cli_default_and_explicit_profile_are_forwarded(monkeypatch, capsys):
    calls = []

    def materialize(root, output, **kwargs):
        calls.append((root, output, kwargs))
        return {"profile": kwargs["profile"], "browser_model_calls": 0}

    monkeypatch.setattr(p, "materialize_profile", materialize)
    common = [
        "--canonical-root",
        "/tmp/canonical",
        "--output",
        "/tmp/new-profile",
        "--source-sha256",
        "a" * 64,
    ]
    assert p.main(common) == 0
    assert json.loads(capsys.readouterr().out)["profile"] == p.NATIVE_PROFILE
    assert calls[-1] == (
        Path("/tmp/canonical"),
        Path("/tmp/new-profile"),
        {"profile": p.NATIVE_PROFILE, "expected_source_sha256": "a" * 64},
    )
    assert p.main([*common, "--profile", p.COMPOSED_PROFILE]) == 0
    assert json.loads(capsys.readouterr().out)["profile"] == p.COMPOSED_PROFILE


def test_cli_rejects_unknown_profile_before_reading_sources(monkeypatch):
    monkeypatch.setattr(
        p,
        "materialize_profile",
        lambda *_args, **_kwargs: pytest.fail("unknown profile reached source reads"),
    )
    with pytest.raises(SystemExit) as error:
        p.main(
            [
                "--canonical-root",
                "/tmp/no-source",
                "--output",
                "/tmp/no-output",
                "--source-sha256",
                "a" * 64,
                "--profile",
                "composed",
            ]
        )
    assert error.value.code == 2
