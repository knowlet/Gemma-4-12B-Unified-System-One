"""Source-authenticated V7 variant exposing bounded observed link destinations.

This is a distinct observation profile, not a V7 benchmark result. Only the
snapshot and model request construction are changed; no browser/model is run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import jev_observation_profile_v7 as v7

PROFILE = "composed-visible-text-settled-v7-links"
V7_SHA256 = "d9f4236d1d46c9472c3d6c1551342639ecb870e6c0dd419246927645410ea868"
SNAPSHOT = v7.SNAPSHOT
MODEL = "jev_ultrafast/model.py"
CANONICAL_SHA256 = v7.CANONICAL_SHA256
CANONICAL_REVISION = v7.CANONICAL_REVISION

# The bounded side table is populated only after marker/guards and action IDs
# have been computed. It is never placed in actions, which feed fingerprint().
SNAPSHOT_ANCHOR = "  return {observation_profile:textProfile,url:location.href"
SNAPSHOT_ADDITION = r"""  const link_destinations=Object.create(null);
  for (const action of actions) {
    if (action.kind!=='click' || !action.node) continue;
    const element=cache.nodes.get(action.node);
    if (!element || element.tagName!=='A' || !element.hasAttribute('href')) continue;
    try {
      const destination=new URL(element.getAttribute('href'),document.baseURI);
      if (!['http:','https:'].includes(destination.protocol) ||
          !destination.hostname || destination.origin.length>256) continue;
      const completePath=destination.pathname;
      let path=completePath.slice(0,256);
      if (path.length<completePath.length) path=path.replace(/%(?:[0-9a-f])?$/i,'');
      link_destinations[action.id]={origin:destination.origin,path,
        truncated:path.length<completePath.length};
    } catch (_) { /* An invalid href supplies no destination. */ }
  }
"""
MODEL_REPLACEMENTS = (
    ("def action_space(actions):", "def action_space(actions, link_destinations=None):"),
    (
        '            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])',
        '            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])\n'
        '            if link_destinations is not None and action["id"] in link_destinations:\n'
        '                element["link_destination"] = link_destinations[action["id"]]',
    ),
    (
        '    elements, targets, controls = action_space(state["actions"])',
        '    link_destinations = state.get("link_destinations", {})\n'
        '    elements, targets, controls = action_space(state["actions"], link_destinations)',
    ),
    (
        '                    **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},',
        '                    **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},\n'
        '                    **({"link_destination": link_destinations[a["id"]]}\n'
        '                       if operation == "CLICK" and a["id"] in link_destinations else {}),',
    ),
)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Pinned link profile anchor changed")
    return source.replace(old, new, 1)


def _source_pin(expected):
    blob = v7._ordinary_path(__file__).read_bytes()
    if not isinstance(expected, str) or len(expected) != 64 or _sha(blob) != expected:
        raise ValueError("Reviewed link profile source changed")
    v7._source_pin(V7_SHA256)
    return blob


def _derive_snapshot(source):
    original = source.decode("utf-8")
    before = "id:'" + v7.PROFILE + "'"
    after = "id:'" + PROFILE + "'"
    result = _replace_once(original, before, after)
    result = _replace_once(result, SNAPSHOT_ANCHOR, SNAPSHOT_ADDITION + SNAPSHOT_ANCHOR)
    result = _replace_once(
        result,
        "return {observation_profile:textProfile,url:location.href",
        "return {link_destinations,observation_profile:textProfile,url:location.href",
    )
    inverse = _replace_once(
        result,
        "return {link_destinations,observation_profile:textProfile,url:location.href",
        "return {observation_profile:textProfile,url:location.href",
    )
    inverse = _replace_once(inverse, SNAPSHOT_ADDITION, "")
    inverse = _replace_once(inverse, after, before)
    if inverse != original:
        raise ValueError("Snapshot inverse differs from V7")
    return result.encode("utf-8")


def _derive_model(source):
    original = source.decode("utf-8")
    result = original
    for old, new in MODEL_REPLACEMENTS:
        result = _replace_once(result, old, new)
    inverse = result
    for old, new in reversed(MODEL_REPLACEMENTS):
        inverse = _replace_once(inverse, new, old)
    if inverse != original:
        raise ValueError("Model inverse differs from V7")
    return result.encode("utf-8")


@dataclass(frozen=True)
class ProfileBundle:
    canonical_files: object
    runtime_files: object
    _helper_source: bytes
    _v7_bundle: object

    @property
    def read_state(self):
        return self.runtime_files[SNAPSHOT].decode("utf-8")

    @property
    def marker_expression(self):
        return f"(() => {{ const state={self.read_state}; return state?.marker ?? null; }})()"

    def manifest(self):
        return {
            "schema": "explicit-observation-runtime-profile-v7-links-v1",
            "profile": PROFILE,
            "canonical_contract_revision": CANONICAL_REVISION,
            "canonical_source_sha256": dict(CANONICAL_SHA256),
            "runtime_source_sha256": {k: _sha(v) for k, v in self.runtime_files.items()},
            "helper_source_sha256": _sha(self._helper_source),
            "parent_v7_helper_sha256": V7_SHA256,
            "parent_v7_runtime_source_sha256": {
                k: _sha(v) for k, v in self._v7_bundle.runtime_files.items()
            },
            "changed_runtime_files": [MODEL, SNAPSHOT],
            "inverse_bytes_equal": True,
            "link_destination": {
                "schemes": ["http", "https"],
                "max_origin_characters": 256,
                "max_path_characters": 256,
                "credentials_query_fragment_excluded": True,
                "path_may_contain_sensitive_information": True,
            },
            "scope": "Different observation profile; not an original V7 benchmark score.",
        }

    def _materialized_sources(self):
        files = self._v7_bundle._materialized_sources()
        files.update({"runtime/" + key: value for key, value in self.runtime_files.items()})
        files["sources/jev_observation_profile_v7_links.py"] = self._helper_source
        return files

    def materialized_manifest(self):
        manifest = self.manifest()
        manifest["artifact_files"] = {
            key: {"sha256": _sha(blob), "size_bytes": len(blob)}
            for key, blob in sorted(self._materialized_sources().items())
        }
        return manifest


def read_canonical_sources(root):
    return v7.read_canonical_sources(root)


def build_profile(canonical_sources, *, expected_source_sha256, profile=PROFILE):
    if profile != PROFILE:
        raise ValueError("Explicit V7 link profile required")
    helper = _source_pin(expected_source_sha256)
    parent = v7.build_profile(canonical_sources, expected_source_sha256=V7_SHA256)
    runtime = dict(parent.runtime_files)
    runtime[SNAPSHOT] = _derive_snapshot(runtime[SNAPSHOT])
    runtime[MODEL] = _derive_model(runtime[MODEL])
    if any(runtime[k] != parent.runtime_files[k] for k in runtime if k not in {MODEL, SNAPSHOT}):
        raise ValueError("Unrelated V7 source changed")
    return ProfileBundle(parent.canonical_files, MappingProxyType(runtime), helper, parent)


_ordinary_path = v7._ordinary_path
_write_exclusive = v7._write_exclusive
_encode = v7._encode
_require = v7._require
MAX_SOURCE_BYTES = v7.MAX_SOURCE_BYTES
MAX_TOTAL_BYTES = v7.MAX_TOTAL_BYTES
MAX_FILES = v7.MAX_FILES


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


def materialize_profile(canonical_root, output, *, expected_source_sha256, profile=PROFILE):
    canonical_root, output = _ordinary_path(canonical_root), _ordinary_path(output)
    _require(
        not output.exists()
        and not output.is_relative_to(canonical_root)
        and not canonical_root.is_relative_to(output),
        "New disjoint profile output required",
    )
    bundle = build_profile(
        read_canonical_sources(canonical_root),
        expected_source_sha256=expected_source_sha256,
        profile=profile,
    )
    output.mkdir(parents=True, exist_ok=False)
    files = bundle._materialized_sources()
    for label, blob in sorted(files.items()):
        _write_exclusive(output / label, blob)
    final = _encode(bundle.materialized_manifest()) + b"\n"
    _write_exclusive(output / ".profile-manifest.pending", final)
    captured, first_files, first_dirs = _capture_tree(
        output, read_contents=True, manifest_name=".profile-manifest.pending"
    )
    _require(
        captured[".profile-manifest.pending"] == final
        and all(captured[k] == v for k, v in files.items()),
        "Prepared profile bytes differ",
    )
    for directory in sorted(
        [output, *(p for p in output.rglob("*") if p.is_dir())],
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    _require(
        _source_pin(expected_source_sha256) == bundle._helper_source,
        "Helper changed during materialization",
    )
    _, last_files, last_dirs = _capture_tree(
        output, read_contents=False, manifest_name=".profile-manifest.pending"
    )
    _require(
        first_files == last_files and first_dirs == last_dirs, "Prepared profile population changed"
    )
    os.replace(output / ".profile-manifest.pending", output / "profile-manifest.json")
    return {
        "output": str(output),
        "profile": PROFILE,
        "manifest_sha256": _sha(final),
        "manifest_size_bytes": len(final),
        "runtime_root": str(output / "runtime"),
        "browser_model_calls": 0,
    }


def _stat_identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _materialized_labels():
    return {
        *("canonical/" + k for k in CANONICAL_SHA256),
        *("runtime/" + k for k in CANONICAL_SHA256),
        "sources/jev_observation_profile_v7.py",
        "sources/" + v7.PARENT_COPY,
        "sources/jev_observation_profile_v7_links.py",
        "profile-manifest.json",
    }


def _capture_tree(root, *, read_contents, manifest_name="profile-manifest.json"):
    labels = _materialized_labels() - {"profile-manifest.json"} | {manifest_name}
    expected_dirs = {
        str(p) for label in labels for p in PurePosixPath(label).parents if str(p) != "."
    }
    files, directories, blobs, total = {}, {}, {}, 0

    def visit(label, descriptor):
        nonlocal total
        before = os.fstat(descriptor)
        _require(stat.S_ISDIR(before.st_mode), "Ordinary profile directory required")
        directories[label] = _stat_identity(before)
        with os.scandir(descriptor) as entries:
            for entry in entries:
                relative = label + "/" + entry.name if label else entry.name
                observed = entry.stat(follow_symlinks=False)
                _require(not stat.S_ISLNK(observed.st_mode), "Profile symlink refused")
                if stat.S_ISDIR(observed.st_mode):
                    _require(relative in expected_dirs, "Unknown profile directory")
                    child = os.open(
                        entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                    )
                    try:
                        _require(
                            _stat_identity(os.fstat(child)) == _stat_identity(observed),
                            "Directory changed before capture",
                        )
                        visit(relative, child)
                    finally:
                        os.close(child)
                    continue
                _require(
                    stat.S_ISREG(observed.st_mode) and relative in labels,
                    "Unknown or nonordinary profile file",
                )
                _require(0 < observed.st_size <= MAX_SOURCE_BYTES, "Profile file exceeds bound")
                files[relative] = _stat_identity(observed)
                total += observed.st_size
                _require(
                    len(files) <= MAX_FILES and total <= MAX_TOTAL_BYTES,
                    "Complete profile exceeds bound",
                )
                if read_contents:
                    child = os.open(
                        entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor
                    )
                    with os.fdopen(child, "rb") as stream:
                        opened = os.fstat(stream.fileno())
                        _require(
                            stat.S_ISREG(opened.st_mode)
                            and _stat_identity(opened) == files[relative],
                            "File changed before capture",
                        )
                        blob = stream.read(MAX_SOURCE_BYTES + 1)
                        _require(
                            len(blob) == opened.st_size
                            and _stat_identity(os.fstat(stream.fileno())) == files[relative],
                            "File changed during capture",
                        )
                        blobs[relative] = blob
        _require(
            _stat_identity(os.fstat(descriptor)) == directories[label],
            "Directory changed during capture",
        )

    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        visit("", descriptor)
        _require(
            _stat_identity(root.stat(follow_symlinks=False)) == directories[""],
            "Profile root changed",
        )
    finally:
        os.close(descriptor)
    _require(
        set(files) == labels and set(directories) == {"", *expected_dirs},
        "Complete profile population differs",
    )
    return blobs, files, directories


def verify_materialized_profile(
    path, *, expected_manifest_sha256, expected_source_sha256, profile=PROFILE
):
    helper = _source_pin(expected_source_sha256)
    _require(
        type(expected_manifest_sha256) is str
        and re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256),
        "Explicit manifest SHA required",
    )
    _require(type(profile) is str and profile == PROFILE, "Explicit V7 link profile required")
    root = _ordinary_path(path)
    _require(root.is_dir(), "Materialized profile directory required")
    blobs, first_files, first_dirs = _capture_tree(root, read_contents=True)
    manifest_blob = blobs["profile-manifest.json"]
    _require(_sha(manifest_blob) == expected_manifest_sha256, "Materialized manifest hash differs")
    canonical = {label: blobs["canonical/" + label] for label in CANONICAL_SHA256}
    bundle = build_profile(
        canonical, expected_source_sha256=expected_source_sha256, profile=profile
    )
    _require(
        bundle._helper_source == helper
        and manifest_blob == _encode(bundle.materialized_manifest()) + b"\n",
        "Materialized source/metadata differs",
    )
    _require(
        all(blobs[label] == blob for label, blob in bundle._materialized_sources().items()),
        "Materialized canonical/runtime/helper source differs",
    )
    _, final_files, final_dirs = _capture_tree(root, read_contents=False)
    _require(
        first_files == final_files and first_dirs == final_dirs, "Profile closed inventory changed"
    )
    _require(
        _source_pin(expected_source_sha256) == helper, "Reviewed helper changed during verification"
    )
    return VerifiedMaterializedProfile(root, bundle, _sha(manifest_blob), manifest_blob)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            materialize_profile(
                args.canonical_root,
                args.output,
                expected_source_sha256=args.source_sha256,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
