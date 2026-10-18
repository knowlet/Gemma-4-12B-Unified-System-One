"""An independent, authenticated navigation-settling ablation over pinned v4.

Pinned v4 control discovery and executor bytes remain the parent contract.
Only Browser.observe and one reversible snapshot profile identity are derived.
This source-only builder never starts a browser, policy, model or network client.
"""

from __future__ import annotations

import argparse
import ast
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

PROFILE = "composed-visible-controls-settled-v6"
PARENT_SOURCE_SHA256 = "f96fb0553c4c8ef1a163633577bfb3ddbb03ddbc1ff6644c36d8fbe2c06d4b43"
GRANDPARENT_SOURCE_SHA256 = "8d199ac1564af05892e9b21e225d1fdf500f1953840855c902160ed8d56df89d"
PARENT_COPY = "jev_observation_profile_v4.py"
GRANDPARENT_COPY = "jev_observation_profile_v3_parent.py"
SNAPSHOT = "jev_ultrafast/snapshot.js"
BROWSER = "jev_ultrafast/browser.py"
MAX_SOURCE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 16 * MAX_SOURCE_BYTES
MAX_FILES = 32


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


def _parent_path():
    # The same adjacent closure works in scripts, isolated runner and materialization.
    return _ordinary_path(Path(__file__).absolute().with_name(PARENT_COPY))


def _parent_source():
    path = _parent_path()
    _require(
        path.is_file() and path.stat().st_size <= MAX_SOURCE_BYTES, "Pinned v4 parent file required"
    )
    blob = path.read_bytes()
    _require(_sha(blob) == PARENT_SOURCE_SHA256, "Pinned v4 parent source changed")
    return path, blob


def _load_parent():
    path, _ = _parent_source()
    name = "_jev_observation_profile_v6_parent_f96fb0553c4c"
    existing = sys.modules.get(name)
    if existing is not None:
        _require(
            _sha(Path(existing.__file__).read_bytes()) == PARENT_SOURCE_SHA256,
            "Loaded parent source changed",
        )
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_PARENT = _load_parent()
CANONICAL_SHA256 = _PARENT.CANONICAL_SHA256
CANONICAL_REVISION = _PARENT.CANONICAL_REVISION


def _grandparent_source():
    path, blob = _PARENT._parent_source()
    _require(_sha(blob) == GRANDPARENT_SOURCE_SHA256, "Pinned v3 grandparent changed")
    return path, blob


def _source_pin(expected):
    _require(
        type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected),
        "Explicit reviewed v6 helper SHA256 required",
    )
    blob = _ordinary_path(__file__).read_bytes()
    _require(
        0 < len(blob) <= MAX_SOURCE_BYTES and _sha(blob) == expected,
        "Reviewed v6 helper source changed",
    )
    _parent_source()
    _grandparent_source()
    return blob


def read_canonical_sources(root):
    return _PARENT.read_canonical_sources(root)


# Only the observed document-navigation null is transient. The timeout is per
# observe call, including the inherited post-input readiness read and screenshot.
SETTLING_PREAMBLE = """        _settle_started = time.monotonic()
        _settle_deadline = _settle_started + 2.0
        _settle_attempts, _settle_navigation_retries = 0, 0

        def _settle_record(status, late_snapshot=False):
            elapsed = max(0.0, time.monotonic() - _settle_started)
            telemetry = {
                "target_seconds": 2.0, "scope": "per_observe_call", "status": status,
                "attempts": _settle_attempts, "navigation_retries": _settle_navigation_retries,
                "elapsed_seconds": elapsed, "overshoot_seconds": max(0.0, elapsed - 2.0),
                "late_snapshot": late_snapshot, "synchronous_cdp_hard_deadline": False,
            }
            self.observation_settling = telemetry
            return telemetry

        def _settle_timeout(late_snapshot=False):
            telemetry = _settle_record("timeout", late_snapshot)
            exc = StalePage("Navigation settling target exceeded: elapsed=%.6fs overshoot=%.6fs" % (
                telemetry["elapsed_seconds"], telemetry["overshoot_seconds"]))
            exc.observation_settling = dict(telemetry)
            raise exc

        _settle_record("started")
"""
NATIVE_RETRY = """        for attempt in range(10):
            try:
                return browser_operation(
                    {"operation": "observe", "session": self.session, "screenshot": screenshot}
                )
            except StalePage:
                if attempt == 9:
                    raise
                time.sleep(0.02)
        raise StalePage("Page did not settle")
"""
SETTLED_RETRY = """        while True:
            if time.monotonic() >= _settle_deadline:
                _settle_timeout()
            _settle_attempts += 1
            try:
                _settle_page = browser_operation(
                    {"operation": "observe", "session": self.session, "screenshot": screenshot}
                )
            except StalePage as exc:
                if type(exc) is not StalePage or exc.args != ("Document is navigating",):
                    _settle_record("unexpected_error")
                    raise
                _settle_navigation_retries += 1
                _settle_record("transient_navigation")
                remaining = _settle_deadline - time.monotonic()
                if remaining <= 0:
                    _settle_timeout()
                time.sleep(min(0.02, remaining))
                continue
            except BaseException:
                _settle_record("unexpected_error")
                raise
            telemetry = _settle_record("success")
            if telemetry["elapsed_seconds"] >= 2.0:
                _settle_timeout(late_snapshot=True)
            return _settle_page
"""


def _replace_once(source, before, after):
    _require(source.count(before) == 1, "Pinned derivation anchor changed")
    return source.replace(before, after, 1)


def _derive_snapshot(v4):
    before = "id:'" + _PARENT.PROFILE + "'"
    after = "id:'" + PROFILE + "'"
    derived = _replace_once(v4.decode(), before, after).encode()
    _require(
        derived.replace(after.encode(), before.encode(), 1) == v4,
        "Snapshot contains a nonidentity delta",
    )
    return derived


def _observe_source(source):
    tree = ast.parse(source)
    classes = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Browser"
    ]
    _require(len(classes) == 1, "One native Browser required")
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "observe"
    ]
    _require(len(methods) == 1, "One native observation method required")
    node = methods[0]
    return "".join(source.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno])


def _derive_browser(v4):
    source = v4.decode()
    before = _observe_source(source)
    after = _replace_once(
        before,
        "    def observe(self, screenshot=True):\n",
        "    def observe(self, screenshot=True):\n" + SETTLING_PREAMBLE,
    )
    after = _replace_once(
        after,
        '                self.call(\n                    "Runtime.evaluate",\n',
        '                _settle_ready = self.call(\n                    "Runtime.evaluate",\n',
    )
    after = _replace_once(
        after,
        "            except RuntimeError:\n                pass\n",
        '                if _settle_ready.get("exceptionDetails"):\n'
        '                    raise StalePage("Document changed during evaluation")\n'
        "            except BaseException:\n"
        '                _settle_record("unexpected_error")\n'
        "                raise\n",
    )
    after = _replace_once(after, NATIVE_RETRY, SETTLED_RETRY)
    derivative = _replace_once(source, before, after)
    inverse = _replace_once(derivative, after, before)
    _require(
        inverse == source
        and ast.dump(ast.parse(inverse), include_attributes=False)
        == ast.dump(ast.parse(source), include_attributes=False),
        "Browser contains a nonobserver source/AST delta",
    )
    _require(_observe_source(derivative) == after, "Observation method span changed")
    return derivative.encode(), {
        "changed_method": "Browser.observe",
        "inverse_bytes_equal": True,
        "inverse_AST_equal": True,
        "only_observe_delta": True,
        "parent_observe_sha256": _sha(before.encode()),
        "derived_observe_sha256": _sha(after.encode()),
    }


@dataclass(frozen=True)
class ProfileBundle:
    canonical_files: object
    runtime_files: object
    _manifest_bytes: bytes
    _helper_source: bytes
    _parent_source: bytes
    _grandparent_source: bytes

    def manifest(self):
        return json.loads(self._manifest_bytes)

    def _materialized_sources(self):
        files = {"canonical/" + k: v for k, v in self.canonical_files.items()}
        files.update({"runtime/" + k: v for k, v in self.runtime_files.items()})
        files["sources/jev_observation_profile_v6.py"] = self._helper_source
        files["sources/" + PARENT_COPY] = self._parent_source
        files["sources/" + GRANDPARENT_COPY] = self._grandparent_source
        return files

    def materialized_manifest(self):
        manifest = self.manifest()
        manifest["artifact_files"] = {
            label: {"sha256": _sha(blob), "size_bytes": len(blob)}
            for label, blob in sorted(self._materialized_sources().items())
        }
        return manifest

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
    _require(type(profile) is str and profile == PROFILE, "Explicit v6 profile required")
    helper = _source_pin(expected_source_sha256)
    _, parent_source = _parent_source()
    _, grandparent_source = _grandparent_source()
    v4 = _PARENT.build_profile(
        canonical_sources, expected_source_sha256=PARENT_SOURCE_SHA256, profile=_PARENT.PROFILE
    )
    canonical, runtime = dict(v4.canonical_files), dict(v4.runtime_files)
    runtime[SNAPSHOT] = _derive_snapshot(runtime[SNAPSHOT])
    runtime[BROWSER], browser_proof = _derive_browser(runtime[BROWSER])
    _require(
        all(runtime[k] == v4.runtime_files[k] for k in runtime if k not in {SNAPSHOT, BROWSER}),
        "Unrelated v4 runtime source changed",
    )
    manifest = {
        "schema": "explicit-observation-runtime-profile-v6",
        "profile": PROFILE,
        "canonical_contract_revision": CANONICAL_REVISION,
        "canonical_Git_verification_performed_by_this_library": False,
        "canonical_Git_prerequisite": "Caller must retain strict benchmark upstream_manifest checks.",
        "canonical_source_sha256": dict(CANONICAL_SHA256),
        "runtime_source_sha256": {label: _sha(blob) for label, blob in runtime.items()},
        "helper_source_sha256": _sha(helper),
        "parent_helper_source_sha256": PARENT_SOURCE_SHA256,
        "grandparent_helper_source_sha256": GRANDPARENT_SOURCE_SHA256,
        "runtime_root": "runtime",
        "canonical_copy_root": "canonical",
        "helper_copy": "sources/jev_observation_profile_v6.py",
        "parent_helper_copy": "sources/" + PARENT_COPY,
        "grandparent_helper_copy": "sources/" + GRANDPARENT_COPY,
        "derivation": {
            "parent_profile": _PARENT.PROFILE,
            "parent_snapshot_sha256": _sha(v4.runtime_files[SNAPSHOT]),
            "parent_browser_sha256": _sha(v4.runtime_files[BROWSER]),
            "snapshot_profile_literal_substitutions": 1,
            "snapshot_inverse_bytes_equal": True,
            "browser_observe": browser_proof,
            "all_other_v4_runtime_bytes_equal": True,
            "task_specific_rules": False,
        },
        "limits": v4.manifest()["limits"],
        "navigation_settling": {
            "target_seconds": 2.0,
            "scope": "per_observe_call",
            "transient": "exact native StalePage('Document is navigating') only",
            "unexpected_errors": "propagate immediately; never retry or suppress",
            "fast_path": "first complete valid native snapshot returns immediately",
            "late_snapshot": "refused with elapsed/overshoot in timeout exception and Browser.observation_settling",
            "synchronous_cdp_hard_deadline": False,
            "agent_second_observe_can_accumulate_another_call": True,
            "model_or_action_budget_change": False,
        },
        "scope": "Navigation ablation over interface-exposed v4; not blind heldout, serving or model-performance qualification.",
        "not_attested_by_this_library": [
            "browser/model execution",
            "prepared token capacity",
            "serving qualification",
            "training/task performance",
        ],
    }
    return ProfileBundle(
        MappingProxyType(canonical),
        MappingProxyType(runtime),
        _encode(manifest),
        helper,
        parent_source,
        grandparent_source,
    )


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


def _materialized_labels():
    return {
        *("canonical/" + k for k in CANONICAL_SHA256),
        *("runtime/" + k for k in CANONICAL_SHA256),
        "sources/jev_observation_profile_v6.py",
        "sources/" + PARENT_COPY,
        "sources/" + GRANDPARENT_COPY,
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
    _require(type(profile) is str and profile == PROFILE, "Explicit v6 profile required")
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
