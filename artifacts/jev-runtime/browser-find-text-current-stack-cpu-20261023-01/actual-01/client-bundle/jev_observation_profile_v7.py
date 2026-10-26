"""An independent, authenticated navigation-settling ablation over pinned v3.

Pinned v3 discovery, native executor and Agent bytes remain the parent contract.
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

PROFILE = "composed-visible-text-settled-v7"
PARENT_SOURCE_SHA256 = "8d199ac1564af05892e9b21e225d1fdf500f1953840855c902160ed8d56df89d"
PARENT_COPY = "jev_observation_profile_v3_parent.py"
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
    adjacent = Path(__file__).absolute().with_name(PARENT_COPY)
    if adjacent.exists() or adjacent.is_symlink():
        return _ordinary_path(adjacent)
    return _ordinary_path(
        Path(__file__).absolute().parents[1] / "src/s1/jev_observation_profile.py"
    )


def _parent_source():
    path = _parent_path()
    _require(
        path.is_file() and path.stat().st_size <= MAX_SOURCE_BYTES, "Pinned v3 parent file required"
    )
    blob = path.read_bytes()
    _require(_sha(blob) == PARENT_SOURCE_SHA256, "Pinned v3 parent source changed")
    return path, blob


def _load_parent():
    path, _ = _parent_source()
    name = "_jev_observation_profile_v7_parent_8d199ac1564a"
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


def _source_pin(expected):
    _require(
        type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected),
        "Explicit reviewed v7 helper SHA256 required",
    )
    blob = _ordinary_path(__file__).read_bytes()
    _require(
        0 < len(blob) <= MAX_SOURCE_BYTES and _sha(blob) == expected,
        "Reviewed v7 helper source changed",
    )
    _parent_source()
    return blob


def read_canonical_sources(root):
    return _PARENT.read_canonical_sources(root)


# Only the original bodyless-document null is transient. All native CDP work
# remains inside the unmodified browser_operation; timings below combine its
# READ_STATE, screenshot and transport/evidence work, never individual CDP calls.
TARGET_SECONDS = 5.0
MAX_DIAGNOSTIC_RECORDS = 4096
MAX_DIAGNOSTIC_BYTES = 64 * 1024 * 1024
MAX_SIDECAR_BYTES = 8 * 1024 * 1024
SETTLING_PREAMBLE = """        _settle_started = time.monotonic()
        _settle_deadline = _settle_started + 5.0
        _settle_operations = []
        _settle_retries = 0
        _settle_closed = False
        _settle_ready_swallowed = False
        _settle_had_input = bool(getattr(self, "after_input", None))
        _settle_class = type(self)
        if getattr(_settle_class, "_observation_settling_fault", False):
            raise RuntimeError("Navigation settling diagnostic integrity failed")
        _settle_call_id = len(getattr(_settle_class, "observation_settling_records", ())) + 1

        def _settle_record(status, late_snapshot=False, error=None):
            elapsed = max(0.0, time.monotonic() - _settle_started)
            row = {
                "observe_call_id": _settle_call_id,
                "target_seconds": 5.0, "scope": "per_observe_call", "status": status,
                "started_monotonic": _settle_started, "deadline_monotonic": _settle_deadline,
                "elapsed_seconds": elapsed, "overshoot_seconds": max(0.0, elapsed - 5.0),
                "attempts": len(_settle_operations), "navigation_retries": _settle_retries,
                "operations": list(_settle_operations), "after_input_consumed": _settle_had_input,
                "baseline_after_input_runtime_error_swallowed": _settle_ready_swallowed,
                "late_snapshot": late_snapshot, "terminal_unknown": status != "success",
                "synchronous_cdp_hard_deadline": False,
                "operation_timing_scope": "native_observe_combined_not_individual_CDP",
                "error_type": type(error).__name__ if error is not None else None,
                "error": str(error) if error is not None else None,
            }
            self.observation_settling = row
            return row

        def _settle_finish(status, late_snapshot=False, error=None):
            nonlocal _settle_closed
            row = _settle_record(status, late_snapshot, error)
            encoded = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
            records = getattr(_settle_class, "observation_settling_records", ())
            # Charge an extra 1KiB for the final timeout/timing scalar updates.
            total = getattr(_settle_class, "_observation_settling_bytes", 0) + len(encoded) + 1024
            if len(records) >= 4096 or total > 64 * 1024 * 1024:
                _settle_class._observation_settling_fault = True
                raise RuntimeError("Navigation settling diagnostic bound exceeded")
            frozen = json.loads(encoded)
            elapsed = max(0.0, time.monotonic() - _settle_started)
            frozen.update(elapsed_seconds=elapsed, overshoot_seconds=max(0.0, elapsed - 5.0))
            self.observation_settling = frozen
            self.observation_settling_history = getattr(self, "observation_settling_history", ()) + (frozen,)
            _settle_class.observation_settling_records = records + (frozen,)
            _settle_class._observation_settling_bytes = total
            _settle_closed = True
            return frozen

        def _settle_timeout(late_snapshot=False, row=None):
            if row is None:
                row = _settle_finish("timeout", late_snapshot)
            elapsed = max(0.0, time.monotonic() - _settle_started)
            row.update(status="timeout", late_snapshot=late_snapshot, terminal_unknown=True,
                       elapsed_seconds=elapsed, overshoot_seconds=max(0.0, elapsed - 5.0))
            exc = StalePage("Navigation settling target exceeded: elapsed=%.6fs overshoot=%.6fs" % (
                row["elapsed_seconds"], row["overshoot_seconds"]))
            exc.observation_settling = json.loads(json.dumps(row))
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
            if len(_settle_operations) >= 1024:
                raise RuntimeError("Navigation settling native-operation bound exceeded")
            _operation_started = time.monotonic()
            _operation = {"started_monotonic": _operation_started,
                          "native_protocol_call_count": None,
                          "returned_snapshot_sha256": None, "null_navigation": False}
            try:
                _settle_page = browser_operation(
                    {"operation": "observe", "session": self.session, "screenshot": screenshot}
                )
            except BaseException as exc:
                _operation_ended = time.monotonic()
                _operation.update(ended_monotonic=_operation_ended,
                                  elapsed_seconds=max(0.0, _operation_ended - _operation_started),
                                  error_type=type(exc).__name__, error=str(exc))
                _settle_operations.append(_operation)
                if type(exc) is not StalePage or exc.args != ("Document is navigating",):
                    raise
                _operation["null_navigation"] = True
                _settle_retries += 1
                _settle_record("transient_navigation", error=exc)
                remaining = _settle_deadline - time.monotonic()
                if remaining <= 0:
                    _settle_timeout()
                time.sleep(min(0.02, remaining))
                continue
            _operation_ended = time.monotonic()
            _operation.update(ended_monotonic=_operation_ended,
                              elapsed_seconds=max(0.0, _operation_ended - _operation_started),
                              error_type=None, error=None)
            _settle_operations.append(_operation)
            _operation["returned_snapshot_sha256"] = hashlib.sha256(json.dumps(
                _settle_page, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            if time.monotonic() >= _settle_deadline:
                _settle_timeout(late_snapshot=True)
            _settle_final = _settle_finish("success")
            _settle_final_elapsed = max(0.0, time.monotonic() - _settle_started)
            _settle_final.update(elapsed_seconds=_settle_final_elapsed,
                                 overshoot_seconds=max(0.0, _settle_final_elapsed - 5.0))
            if _settle_final_elapsed >= 5.0:
                _settle_timeout(late_snapshot=True, row=_settle_final)
            return _settle_page
"""


def _replace_once(source, before, after):
    _require(source.count(before) == 1, "Pinned derivation anchor changed")
    return source.replace(before, after, 1)


def _derive_snapshot(v3):
    before = "id:'" + _PARENT.PRUNED_PROFILE + "'"
    after = "id:'" + PROFILE + "'"
    derived = _replace_once(v3.decode(), before, after).encode()
    _require(
        derived.replace(after.encode(), before.encode(), 1) == v3,
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


def _derive_browser(v3):
    source = v3.decode()
    before = _observe_source(source)
    body = before.split("\n", 1)[1]
    body = _replace_once(body, NATIVE_RETRY, SETTLED_RETRY)
    body = _replace_once(
        body,
        "            except RuntimeError:\n                pass\n",
        "            except RuntimeError:\n                _settle_ready_swallowed = True\n                pass\n",
    )
    after = "    def observe(self, screenshot=True):\n" + SETTLING_PREAMBLE
    after += "        try:\n" + "".join(
        "    " + line if line.strip() else line for line in body.splitlines(keepends=True)
    )
    after += """        except BaseException as exc:
            if not _settle_closed:
                try:
                    _settle_finish("unexpected_error", error=exc)
                except BaseException as secondary:
                    _settle_class._observation_settling_fault = True
                    exc.add_note("Navigation diagnostic finalization failed: " + repr(secondary))
            raise
"""
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
        "native_browser_operation_unchanged": True,
        "baseline_after_input_RuntimeError_swallow_preserved": True,
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

    def manifest(self):
        return json.loads(self._manifest_bytes)

    def _materialized_sources(self):
        files = {"canonical/" + k: v for k, v in self.canonical_files.items()}
        files.update({"runtime/" + k: v for k, v in self.runtime_files.items()})
        files["sources/jev_observation_profile_v7.py"] = self._helper_source
        files["sources/" + PARENT_COPY] = self._parent_source
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
    _require(type(profile) is str and profile == PROFILE, "Explicit v7 profile required")
    helper = _source_pin(expected_source_sha256)
    _, parent_source = _parent_source()
    v3 = _PARENT.build_profile(
        canonical_sources,
        expected_source_sha256=PARENT_SOURCE_SHA256,
        profile=_PARENT.PRUNED_PROFILE,
    )
    canonical, runtime = dict(v3.canonical_files), dict(v3.runtime_files)
    runtime[SNAPSHOT] = _derive_snapshot(runtime[SNAPSHOT])
    runtime[BROWSER], browser_proof = _derive_browser(runtime[BROWSER])
    _require(
        all(runtime[k] == v3.runtime_files[k] for k in runtime if k not in {SNAPSHOT, BROWSER}),
        "Unrelated v3 runtime source changed",
    )
    manifest = {
        "schema": "explicit-observation-runtime-profile-v7",
        "profile": PROFILE,
        "canonical_contract_revision": CANONICAL_REVISION,
        "canonical_Git_verification_performed_by_this_library": False,
        "canonical_Git_prerequisite": "Caller must retain strict benchmark upstream_manifest checks.",
        "canonical_source_sha256": dict(CANONICAL_SHA256),
        "runtime_source_sha256": {label: _sha(blob) for label, blob in runtime.items()},
        "helper_source_sha256": _sha(helper),
        "parent_helper_source_sha256": PARENT_SOURCE_SHA256,
        "runtime_root": "runtime",
        "canonical_copy_root": "canonical",
        "helper_copy": "sources/jev_observation_profile_v7.py",
        "parent_helper_copy": "sources/" + PARENT_COPY,
        "derivation": {
            "parent_profile": _PARENT.PRUNED_PROFILE,
            "parent_snapshot_sha256": _sha(v3.runtime_files[SNAPSHOT]),
            "parent_browser_sha256": _sha(v3.runtime_files[BROWSER]),
            "snapshot_profile_literal_substitutions": 1,
            "snapshot_inverse_bytes_equal": True,
            "browser_observe": browser_proof,
            "all_other_v3_runtime_bytes_equal": True,
            "task_specific_rules": False,
        },
        "limits": v3.manifest()["limits"],
        "navigation_settling": {
            "target_seconds": 5.0,
            "scope": "per_observe_call",
            "transient": "exact native StalePage('Document is navigating') only",
            "unexpected_errors": "new loop fails immediately; baseline after_input RuntimeError swallow preserved",
            "fast_path": "first complete valid native snapshot returns immediately",
            "late_snapshot": "refused with elapsed/overshoot in timeout exception and Browser.observation_settling",
            "synchronous_cdp_hard_deadline": False,
            "agent_second_observe_can_accumulate_another_call": True,
            "model_or_action_budget_change": False,
            "timing_scope": "native observe operation combined; no individual CDP timing claim",
            "baseline_after_input_RuntimeError_swallow_preserved": True,
            "max_operations_per_call": 1024,
            "max_records_per_runtime": MAX_DIAGNOSTIC_RECORDS,
            "max_diagnostic_charge_bytes": MAX_DIAGNOSTIC_BYTES,
            "charge_padding_per_record_bytes": 1024,
            "max_sidecar_bytes_per_attempt": MAX_SIDECAR_BYTES,
        },
        "scope": "Navigation ablation over fixed v3; not blind heldout, serving or model-performance qualification.",
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
            "schema_id": "jev-v7-native-observation-settling-sidecar-v1",
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
        "sources/jev_observation_profile_v7.py",
        "sources/" + PARENT_COPY,
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
    _require(type(profile) is str and profile == PROFILE, "Explicit v7 profile required")
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
