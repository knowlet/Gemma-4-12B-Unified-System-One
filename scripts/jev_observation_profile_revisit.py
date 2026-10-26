"""Materialize a separately named V8 plus bounded revisit metadata experiment.

Source preparation and verification import only stdlib source builders. This
profile is not covered by old V8/Q95 capacity or browser-quality evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
PROFILE = "composed-visible-hover-entries-settled-revisit-v1"
V8_SHA = "9d264939f805af7443e765906df4828534429872fc35563fa3a45ad383e60ba8"
DEPENDENCIES = {
    "jev_observation_profile_v8.py": (V8_SHA, "scripts/jev_observation_profile_v8.py"),
    "jev_profile_tree_capture.py": (
        "1cf4f46178ea7761209ba6abe87d708209d128b44e1fa41c658edcd7ff7c753d",
        "scripts/jev_profile_tree_capture.py",
    ),
    "revisit_integration.py": (
        "081310eee8e20908c3131f89d8c3e2f7a2870c9a0e04e7788c6fe44bdd62e9c8",
        "artifacts/jev-runtime/generic-state-revisit-integration-20261020-01/revisit_integration.py",
    ),
    "observation_revisit.py": (
        "80249cdd386619b9121160c94d3a071f586cfa40d75977a7e751138175a6b5d0",
        "artifacts/jev-runtime/generic-state-revisit-progress-20261020-01/observation_revisit.py",
    ),
}


def _dependency(name):
    sha, repository_path = DEPENDENCIES[name]
    adjacent = Path(__file__).absolute().with_name(name)
    path = adjacent if adjacent.exists() or adjacent.is_symlink() else ROOT / repository_path
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Ordinary exact dependency path required")
    if not path.is_file() or not 0 < path.stat().st_size <= 1024**2:
        raise ValueError("Bounded exact dependency required")
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != sha:
        raise ValueError("Pinned revisit dependency differs: " + name)
    return path, blob


def _load(name):
    path, blob = _dependency(name)
    key = "_jev_revisit_" + name.removesuffix(".py")
    if key in sys.modules:
        result = sys.modules[key]
        if Path(result.__file__).absolute() != path:
            raise ValueError("Loaded exact dependency origin differs")
    else:
        spec = importlib.util.spec_from_file_location(key, path)
        result = importlib.util.module_from_spec(spec)
        sys.modules[key] = result
        try:
            spec.loader.exec_module(result)
        except BaseException:
            sys.modules.pop(key, None)
            raise
    if _dependency(name)[1] != blob:
        raise ValueError("Dependency changed during import")
    return result


V8 = _load("jev_observation_profile_v8.py")
TREE = _load("jev_profile_tree_capture.py")
TRANSFORM = _load("revisit_integration.py")
MAX_FILES, MAX_TOTAL_BYTES, MAX_SOURCE_BYTES = 64, 16 * 1024**2, 1024**2
CANONICAL_SHA256, CANONICAL_REVISION = V8.CANONICAL_SHA256, V8.CANONICAL_REVISION
SNAPSHOT = V8.SNAPSHOT
read_canonical_sources = V8.read_canonical_sources
settling_cursor = V8.settling_cursor


def _sources(expected):
    helper = V8._read_pinned(__file__, expected)
    dependencies = {name: _dependency(name)[1] for name in DEPENDENCIES}
    dependencies.update(V8._dependency_sources())
    dependencies["jev_observation_profile_revisit.py"] = helper
    return dependencies


@dataclass(frozen=True)
class ProfileBundle:
    runtime_files: object
    _files: object
    _manifest_bytes: bytes

    def manifest(self):
        return json.loads(self._manifest_bytes)

    def _materialized_sources(self):
        return dict(self._files)

    def materialized_manifest(self):
        value = self.manifest()
        value["artifact_files"] = {
            label: {"sha256": V8._sha(blob), "size_bytes": len(blob)}
            for label, blob in sorted(self._files.items())
        }
        return value

    @property
    def read_state(self):
        return self.runtime_files[SNAPSHOT].decode()

    @property
    def marker_expression(self):
        return f"(() => {{ const state={self.read_state}; return state?.marker ?? null; }})()"


@dataclass(frozen=True)
class VerifiedMaterializedProfile:
    path: Path
    bundle: ProfileBundle
    manifest_sha256: str
    _manifest_bytes: bytes

    def manifest(self):
        return json.loads(self._manifest_bytes)

    @property
    def runtime_root(self):
        return self.path / "runtime"

    @property
    def read_state(self):
        return self.bundle.read_state

    @property
    def marker_expression(self):
        return self.bundle.marker_expression


def build_profile(canonical_sources, *, expected_source_sha256, profile=PROFILE):
    V8._require(profile == PROFILE and type(profile) is str, "Explicit revisit profile required")
    sources = _sources(expected_source_sha256)
    base = V8.build_profile(canonical_sources, expected_source_sha256=V8_SHA)
    runtime = dict(base.runtime_files)
    transformed, proof = TRANSFORM.build_runtime_sources(
        {
            "agent.py": runtime["jev_ultrafast/agent.py"],
            "model.py": runtime["jev_ultrafast/model.py"],
            "observation_revisit.py": sources["observation_revisit.py"],
        }
    )
    runtime.update({"jev_ultrafast/" + name: blob for name, blob in transformed.items()})
    old, new = ("id:'" + V8.PROFILE + "'").encode(), ("id:'" + PROFILE + "'").encode()
    V8._require(runtime[SNAPSHOT].count(old) == 1, "Exact parent snapshot identity")
    runtime[SNAPSHOT] = runtime[SNAPSHOT].replace(old, new, 1)
    V8._require(
        runtime[SNAPSHOT].replace(new, old, 1) == base.runtime_files[SNAPSHOT],
        "Snapshot changed beyond identity",
    )
    files = {"parent_v8/" + name: blob for name, blob in base._materialized_sources().items()}
    files["parent_v8/profile-manifest.json"] = V8._encode(base.materialized_manifest()) + b"\n"
    files.update({"runtime/" + name: blob for name, blob in runtime.items()})
    files.update({"sources/" + name: blob for name, blob in sources.items()})
    manifest = {
        "schema": "explicit-observation-runtime-profile-revisit-v1",
        "profile": PROFILE,
        "parent_profile": V8.PROFILE,
        "canonical_revision": CANONICAL_REVISION,
        "canonical_source_sha256": dict(CANONICAL_SHA256),
        "runtime_source_sha256": {name: V8._sha(blob) for name, blob in runtime.items()},
        "helper_source_sha256": expected_source_sha256,
        "runtime_root": "runtime",
        "artifact_file_count_including_manifest": len(files) + 1,
        "runtime_file_count": len(runtime),
        "limits": {
            "files": MAX_FILES,
            "total_bytes": MAX_TOTAL_BYTES,
            "file_bytes": MAX_SOURCE_BYTES,
        },
        "integration_proof": proof,
        "snapshot_identity_only_delta": True,
        "fresh_capacity_and_browser_evaluation_required": True,
        "old_V8_or_Q95_qualification_covers_new_prompt": False,
    }
    V8._require(len(files) + 1 <= MAX_FILES, "New profile file bound exceeded")
    bundle = ProfileBundle(MappingProxyType(runtime), MappingProxyType(files), V8._encode(manifest))
    final = V8._encode(bundle.materialized_manifest()) + b"\n"
    V8._require(
        all(0 < len(blob) <= MAX_SOURCE_BYTES for blob in [*files.values(), final])
        and sum(map(len, files.values())) + len(final) <= MAX_TOTAL_BYTES,
        "New profile byte bound exceeded",
    )
    return bundle


def _labels(manifest_name="profile-manifest.json"):
    return {
        *("parent_v8/" + name for name in V8._materialized_labels()),
        *("runtime/" + name for name in CANONICAL_SHA256),
        "runtime/jev_ultrafast/observation_revisit.py",
        *(
            "sources/" + name
            for name in (
                *DEPENDENCIES,
                *V8._dependency_sources(),
                "jev_observation_profile_revisit.py",
            )
        ),
        manifest_name,
    }


def _capture(path, *, read_contents, manifest_name="profile-manifest.json"):
    return TREE.capture_tree(
        path,
        labels=_labels(manifest_name),
        read_contents=read_contents,
        helpers=V8,
        MAX_SOURCE_BYTES=MAX_SOURCE_BYTES,
        MAX_FILES=MAX_FILES,
        MAX_TOTAL_BYTES=MAX_TOTAL_BYTES,
    )


def _fsync_directories(path):
    for directory in sorted(
        [path, *(p for p in path.rglob("*") if p.is_dir())],
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def materialize_profile(canonical_root, output, *, expected_source_sha256, profile=PROFILE):
    canonical_root, output = V8._ordinary_path(canonical_root), V8._ordinary_path(output)
    V8._require(
        not output.exists()
        and not output.is_relative_to(canonical_root)
        and not canonical_root.is_relative_to(output),
        "Fresh disjoint profile output required",
    )
    bundle = build_profile(
        read_canonical_sources(canonical_root),
        expected_source_sha256=expected_source_sha256,
        profile=profile,
    )
    output.mkdir(parents=True, exist_ok=False)
    files = bundle._materialized_sources()
    for name, blob in sorted(files.items()):
        V8._write_exclusive(output / name, blob)
    final = V8._encode(bundle.materialized_manifest()) + b"\n"
    V8._write_exclusive(output / ".profile-manifest.pending", final)
    blobs, first_files, first_dirs = _capture(
        output, read_contents=True, manifest_name=".profile-manifest.pending"
    )
    V8._require(blobs == {**files, ".profile-manifest.pending": final}, "Prepared profile differs")
    _fsync_directories(output)
    V8._require(
        _sources(expected_source_sha256)
        == {
            name.removeprefix("sources/"): blob
            for name, blob in files.items()
            if name.startswith("sources/")
        },
        "Sources changed during preparation",
    )
    _, last_files, last_dirs = _capture(
        output, read_contents=False, manifest_name=".profile-manifest.pending"
    )
    V8._require(
        first_files == last_files and first_dirs == last_dirs, "Prepared profile population changed"
    )
    os.replace(output / ".profile-manifest.pending", output / "profile-manifest.json")
    _fsync_directories(output)
    return {
        "output": str(output),
        "profile": PROFILE,
        "runtime_root": str(output / "runtime"),
        "manifest_sha256": V8._sha(final),
        "manifest_size_bytes": len(final),
        "artifact_files": len(files) + 1,
        "browser_model_calls": 0,
    }


def verify_materialized_profile(
    path, *, expected_manifest_sha256, expected_source_sha256, profile=PROFILE
):
    root = V8._ordinary_path(path)
    blobs, first_files, first_dirs = _capture(root, read_contents=True)
    final = blobs["profile-manifest.json"]
    V8._require(
        type(expected_manifest_sha256) is str and V8._sha(final) == expected_manifest_sha256,
        "Explicit materialized manifest SHA differs",
    )
    canonical = {
        name: blobs["parent_v8/parent_v5/parent/canonical/" + name] for name in CANONICAL_SHA256
    }
    bundle = build_profile(
        canonical, expected_source_sha256=expected_source_sha256, profile=profile
    )
    V8._require(
        final == V8._encode(bundle.materialized_manifest()) + b"\n"
        and blobs == {**bundle._materialized_sources(), "profile-manifest.json": final},
        "Materialized source/metadata differs",
    )
    _, last_files, last_dirs = _capture(root, read_contents=False)
    V8._require(
        first_files == last_files and first_dirs == last_dirs, "Verified profile population changed"
    )
    _sources(expected_source_sha256)
    return VerifiedMaterializedProfile(root, bundle, V8._sha(final), final)


def write_settling_sidecar(
    browser_class, cursor, path, writer, secrets, primary=None, redactor=None
):
    def new_identity_writer(destination, value, selected_secrets):
        V8._require(value["profile"] == V8.PROFILE, "Parent settling identity differs")
        value.update(
            profile=PROFILE, schema_id="jev-revisit-native-observation-settling-sidecar-v1"
        )
        rendered = redactor(value, list(selected_secrets)) if redactor is not None else value
        V8._require(
            len((json.dumps(rendered, indent=2, ensure_ascii=False) + "\n").encode())
            <= V8.MAX_SIDECAR_BYTES,
            "New identity sidecar exceeds original bound",
        )
        return writer(destination, value, selected_secrets)

    return V8.write_settling_sidecar(
        browser_class, cursor, path, new_identity_writer, secrets, primary, redactor
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args(argv)
    result = materialize_profile(
        args.canonical_root, args.output, expected_source_sha256=args.source_sha256
    )
    verify_materialized_profile(
        args.output,
        expected_manifest_sha256=result["manifest_sha256"],
        expected_source_sha256=args.source_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
