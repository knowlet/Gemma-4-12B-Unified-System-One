"""Explicit source-authenticated v5 hover discovery plus v7 observe settling.

Runtime bytes equal the reviewed source composition. This prepares sources only;
unbounded appended label proxies require fresh request capacity/refusal checks.
No model, browser, tokenizer or collector is imported by this library.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

PROFILE = "composed-visible-hover-entries-settled-v8"
V5_SOURCE_SHA256 = "7847b2f9eecb3f9d492ccfdf8415c52632b99ac6ead6df560c49c3caf004e141"
V7_SOURCE_SHA256 = "d9f4236d1d46c9472c3d6c1551342639ecb870e6c0dd419246927645410ea868"
V4_SOURCE_SHA256 = "f96fb0553c4c8ef1a163633577bfb3ddbb03ddbc1ff6644c36d8fbe2c06d4b43"
V3_SOURCE_SHA256 = "8d199ac1564af05892e9b21e225d1fdf500f1953840855c902160ed8d56df89d"
MAX_SOURCE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 16 * MAX_SOURCE_BYTES
MAX_FILES = 64
MAX_SIDECAR_BYTES = 8 * 1024 * 1024


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _encode(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ordinary_path(path):
    path = Path(path).absolute()
    _require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary canonical paths required",
    )
    return path


def _read_pinned(path, expected):
    path = _ordinary_path(path)
    _require(
        path.is_file() and 0 < path.stat().st_size <= MAX_SOURCE_BYTES,
        "Bounded adjacent dependency required",
    )
    blob = path.read_bytes()
    _require(_sha(blob) == expected, "Pinned adjacent dependency changed")
    return blob


def _load_dependency(filename, expected):
    path = _ordinary_path(Path(__file__).with_name(filename))
    _read_pinned(path, expected)
    name = "_jev_v8_" + path.stem
    if name in sys.modules:
        module = sys.modules[name]
        _require(_ordinary_path(module.__file__) == path, "Loaded dependency origin differs")
        _read_pinned(module.__file__, expected)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        _read_pinned(path, expected)
        return module
    except BaseException:
        sys.modules.pop(name, None)
        raise


_V5 = _load_dependency("jev_observation_profile_v5.py", V5_SOURCE_SHA256)
_V7 = _load_dependency("jev_observation_profile_v7.py", V7_SOURCE_SHA256)
CANONICAL_SHA256 = _V5.CANONICAL_SHA256
CANONICAL_REVISION = _V5.CANONICAL_REVISION
SNAPSHOT, BROWSER = _V5.SNAPSHOT, _V5.BROWSER

EXPECTED_RUNTIME = MappingProxyType(
    {
        "jev_ultrafast/__init__.py": "60f025b8d24465e2c0f7a3547d43b8020fd8843de7b6cb3f892ef63f5548cf63",
        "jev_ultrafast/agent.py": "aaf8b271cb0ee6643fe7beb525328ab80de8568c7ac044d02b9c6721161e7c8b",
        "jev_ultrafast/browser.py": "ed44fd48509d60a0731b64effc0ccfb428d36dcf1e148fbb9a190896cb966dcd",
        "jev_ultrafast/model.py": "a85ada8458d23642c5c8a4095acf0565e6268f81dcda26f6308942b653e9c9f3",
        "jev_ultrafast/questions.py": "66a7054b85040b949e1757311257ea0c5a843c635f9a40c99f6d15ad660fb422",
        "jev_ultrafast/snapshot.js": "44a3b5df762bae7ffe505314fd18ba3171e6d54f55bc6af963337b5dc6b3bd04",
        "jev_ultrafast/static/fixture.html": "e48e7d7b6225b215060692e0a28c3d0b59e465e4b94c1bc510d5f37ff0b8bfbe",
    }
)


def _dependency_sources():
    directory = Path(__file__).absolute().parent
    v4 = _ordinary_path(directory / "jev_observation_profile_v4.py")
    v3 = _ordinary_path(_V5._PARENT._parent_path())
    _require(_ordinary_path(_V5._PARENT.__file__) == v4, "Adjacent v4 parent origin differs")
    _require(
        _ordinary_path(_V5._PARENT._PARENT.__file__) == v3
        and _ordinary_path(_V7._PARENT.__file__) == v3,
        "Shared v3 parent origin differs",
    )
    return {
        "jev_observation_profile_v5.py": _read_pinned(
            directory / "jev_observation_profile_v5.py", V5_SOURCE_SHA256
        ),
        "jev_observation_profile_v4.py": _read_pinned(v4, V4_SOURCE_SHA256),
        "jev_observation_profile_v7.py": _read_pinned(
            directory / "jev_observation_profile_v7.py", V7_SOURCE_SHA256
        ),
        "jev_observation_profile_v3_parent.py": _read_pinned(v3, V3_SOURCE_SHA256),
    }


_dependency_sources()


def _source_pin(expected):
    _require(
        type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected),
        "Explicit reviewed v8 helper SHA required",
    )
    helper = _read_pinned(__file__, expected)
    _dependency_sources()
    return helper


def read_canonical_sources(root):
    return _V5.read_canonical_sources(root)


def _derive_snapshot(source):
    before = ("id:'" + _V5.PROFILE + "'").encode()
    after = ("id:'" + PROFILE + "'").encode()
    _require(source.count(before) == 1, "Pinned v5 snapshot identity anchor")
    result = source.replace(before, after, 1)
    _require(result.replace(after, before, 1) == source, "Nonidentity snapshot change")
    return result


@dataclass(frozen=True)
class ProfileBundle:
    canonical_files: object
    runtime_files: object
    _manifest_bytes: bytes
    _helper_source: bytes
    _v5_bundle: object
    _dependencies: object

    def manifest(self):
        return json.loads(self._manifest_bytes)

    def _materialized_sources(self):
        files = {
            "parent_v5/" + label: blob
            for label, blob in self._v5_bundle._materialized_sources().items()
        }
        files["parent_v5/profile-manifest.json"] = (
            _encode(self._v5_bundle.materialized_manifest()) + b"\n"
        )
        files.update({"runtime/" + label: blob for label, blob in self.runtime_files.items()})
        files.update({"sources/" + label: blob for label, blob in self._dependencies.items()})
        files["sources/jev_observation_profile_v8.py"] = self._helper_source
        return files

    def materialized_manifest(self):
        value = self.manifest()
        value["artifact_files"] = {
            label: {"sha256": _sha(blob), "size_bytes": len(blob)}
            for label, blob in sorted(self._materialized_sources().items())
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
    _require(type(profile) is str and profile == PROFILE, "Explicit v8 profile required")
    helper = _source_pin(expected_source_sha256)
    dependencies = _dependency_sources()
    old = _V5.build_profile(
        canonical_sources, expected_source_sha256=V5_SOURCE_SHA256, profile=_V5.PROFILE
    )
    v7 = _V7.build_profile(
        canonical_sources, expected_source_sha256=V7_SOURCE_SHA256, profile=_V7.PROFILE
    )
    canonical, runtime = dict(old.canonical_files), dict(old.runtime_files)
    runtime[SNAPSHOT] = _derive_snapshot(runtime[SNAPSHOT])
    runtime[BROWSER], observe_proof = _V7._derive_browser(runtime[BROWSER])
    _require(
        _V7._observe_source(runtime[BROWSER].decode())
        == _V7._observe_source(v7.runtime_files[BROWSER].decode()),
        "Observe bytes differ from v7",
    )
    proxy = _V5._PARENT.PROXY_EXECUTOR.encode()
    _require(
        runtime[BROWSER].count(proxy) == 1
        and runtime[BROWSER].replace(proxy, b"", 1) == v7.runtime_files[BROWSER],
        "Nonproxy browser delta relative to v7",
    )
    _require(
        {label: _sha(blob) for label, blob in runtime.items()} == dict(EXPECTED_RUNTIME),
        "Reviewed complete runtime7 bytes differ",
    )
    _require(
        all(
            runtime[label] == canonical[label]
            for label in runtime
            if label not in {BROWSER, SNAPSHOT}
        ),
        "Unrelated native runtime source differs",
    )
    manifest = {
        "schema": "explicit-observation-runtime-profile-v8",
        "profile": PROFILE,
        "canonical_contract_revision": CANONICAL_REVISION,
        "canonical_Git_verification_performed_by_this_library": False,
        "canonical_Git_prerequisite": "Caller retains strict benchmark upstream_manifest checks.",
        "canonical_source_sha256": dict(CANONICAL_SHA256),
        "runtime_source_sha256": dict(EXPECTED_RUNTIME),
        "helper_source_sha256": _sha(helper),
        "dependency_source_sha256": {label: _sha(blob) for label, blob in dependencies.items()},
        "runtime_root": "runtime",
        "canonical_copy_root": "parent_v5/parent/canonical",
        "helper_copy": "sources/jev_observation_profile_v8.py",
        "derivation": {
            "candidate_parent_profile": _V5.PROFILE,
            "settling_parent_profile": _V7.PROFILE,
            "snapshot_inverse_bytes_equal_v5_except_profile_literal": True,
            "observe_bytes_equal_v7": True,
            "v7_browser_delta_only_inherited_v4_proxy_validation": True,
            "browser_operation_equal_v5_v4": True,
            "v7_observe_proof_relative_to_v5": observe_proof,
            "other_five_runtime_files_canonical": True,
            "native_CLICK_event_semantics_changed": False,
            "additional_markers_or_candidate_reordering": False,
            "task_specific_rules": False,
        },
        "limits": old.manifest()["limits"],
        "navigation_settling": v7.manifest()["navigation_settling"],
        "offered_capacity": {
            "native_action_truncation": 250,
            "native_omitted_count_and_order_retained": True,
            "proxy_and_hover_actions_appended_after_native_cap": True,
            "appended_proxies_total_bound_attested": False,
            "universal_400head_8192context_qualification_attested": False,
            "fresh_actual_payload_capacity_and_edge_refusal_qualification_required": True,
            "candidate_pruning_or_reordering_added": False,
        },
        "scope": "Explicit new observation condition; source-only materialization, not v7 serving qualification or model quality.",
        "known_inherited_limitations": old.manifest()["known_inherited_limitations"],
        "not_attested_by_this_library": [
            "browser/model execution",
            "unseen task performance",
            "fresh payload token capacity",
            "serving qualification",
            "universal CSS coverage",
            "runtime stale-mutation validation",
        ],
    }
    return ProfileBundle(
        MappingProxyType(canonical),
        MappingProxyType(runtime),
        _encode(manifest),
        helper,
        old,
        MappingProxyType(dependencies),
    )


def settling_cursor(browser_class):
    _require(
        not getattr(browser_class, "_observation_settling_fault", False),
        "Settling diagnostics failed",
    )
    return len(getattr(browser_class, "observation_settling_records", ()))


def write_settling_sidecar(
    browser_class, cursor, path, writer, secrets, primary=None, redactor=None
):
    """Export diagnostic-only rows; a secondary write failure cannot replace a primary."""
    try:
        records = getattr(browser_class, "observation_settling_records", ())
        _require(type(cursor) is int and 0 <= cursor <= len(records), "Invalid settling cursor")
        _require(
            not getattr(browser_class, "_observation_settling_fault", False),
            "Settling diagnostics incomplete",
        )
        selected = records[cursor:]
        value = {
            "schema_id": "jev-v8-native-observation-settling-sidecar-v1",
            "profile": PROFILE,
            "timing_scope": "per_native_observe_operation_combined_not_individual_CDP",
            "records": selected,
            "complete": True,
            "terminal_unknown": any(row["terminal_unknown"] for row in selected),
            "task_success_attested": False,
        }
        _require(not secrets or callable(redactor), "Exact sidecar redactor required")
        rendered = redactor(value, list(secrets)) if redactor is not None else value
        # Match the pinned benchmark writer, including indentation and newline.
        encoded = (json.dumps(rendered, indent=2, ensure_ascii=False) + "\n").encode()
        _require(len(encoded) <= MAX_SIDECAR_BYTES, "Settling sidecar byte bound exceeded")
        writer(path, value, secrets)
        return value
    except BaseException as exc:
        browser_class._observation_settling_fault = True
        if primary is not None:
            primary.add_note("Navigation settling sidecar failed: " + repr(exc))
            raise primary
        raise


def _materialized_labels():
    return {
        *("parent_v5/" + label for label in _V5._materialized_labels()),
        *("runtime/" + label for label in CANONICAL_SHA256),
        "sources/jev_observation_profile_v8.py",
        "sources/jev_observation_profile_v5.py",
        "sources/jev_observation_profile_v4.py",
        "sources/jev_observation_profile_v7.py",
        "sources/jev_observation_profile_v3_parent.py",
        "profile-manifest.json",
    }


def _write_exclusive(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())


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
    _require(type(profile) is str and profile == PROFILE, "Explicit v8 profile required")
    root = _ordinary_path(path)
    _require(root.is_dir(), "Materialized profile directory required")
    blobs, first_files, first_dirs = _capture_tree(root, read_contents=True)
    manifest_blob = blobs["profile-manifest.json"]
    _require(_sha(manifest_blob) == expected_manifest_sha256, "Materialized manifest hash differs")
    canonical = {label: blobs["parent_v5/parent/canonical/" + label] for label in CANONICAL_SHA256}
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--profile", choices=[PROFILE], default=PROFILE)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            materialize_profile(
                args.canonical_root,
                args.output,
                expected_source_sha256=args.source_sha256,
                profile=args.profile,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
