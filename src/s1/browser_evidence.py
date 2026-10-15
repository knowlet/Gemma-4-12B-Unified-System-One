"""Preserve the actual imported S1 sources used by a browser evaluation.

Public manifests use relative labels only. The capture scope is Python sources
inside this imported package and explicitly named benchmark entrypoints; it
never walks the repository root, model cache, configuration, or environment.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

_DEFAULT_CALLERS = (
    "scripts/jev_browser_benchmark.py",
    "scripts/jev_local_model.py",
    "scripts/jev_public_web_cases.py",
)
_ALLOWED_CALLERS = frozenset(
    (
        *_DEFAULT_CALLERS,
        "apps/modal/jev_browser.py",
        "scripts/jev_browser_fit_readout.py",
        "scripts/jev_browser_fit_actions.py",
        "scripts/jev_browser_training_data.py",
        "scripts/breakthrough_training.py",
    )
)


def _source_paths(additional_files=None):
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    paths = {
        "s1/" + path.relative_to(package).as_posix(): path
        for path in package.rglob("*.py")
        if path.is_file()
    }
    for label in _DEFAULT_CALLERS:
        path = root / label
        if path.is_file():
            paths[label] = path
    if additional_files is not None:
        if not isinstance(additional_files, Mapping):
            raise TypeError("additional_files must map approved relative labels to actual files")
        for label, source in additional_files.items():
            if label not in _ALLOWED_CALLERS:
                raise ValueError("additional source label is not an approved benchmark caller")
            path = Path(source)
            if path.name != Path(label).name:
                raise ValueError(f"additional source must have the named caller filename: {label}")
            paths[label] = path
    for label, path in paths.items():
        if path.is_symlink():
            raise ValueError(f"source symlinks are outside the capture scope: {label}")
    return dict(sorted(paths.items()))


def _record(blob, label):
    return {
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
        "artifact_path": "sources/" + label,
    }


def _identity(files):
    serialized = json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def capture_runtime_sources(output_dir, *, additional_files=None):
    """Copy exact source bytes before loading weights and return their identity.

    ``additional_files`` is an optional mapping from the approved caller labels
    to actual paths, allowing a frozen checkout or Modal entrypoint to override
    the default repository-relative lookup. Supply the same mapping to verify.
    Existing ``sources`` directories are refused rather than overwritten.
    """
    paths = _source_paths(additional_files)
    blobs = {}
    for label, path in paths.items():
        if not path.is_file():
            raise ValueError(f"configured benchmark source is missing: {label}")
        blobs[label] = path.read_bytes()
    files = {label: _record(blob, label) for label, blob in blobs.items()}
    manifest = {
        "schema_version": 1,
        "identity_sha256": _identity(files),
        "scope": "all imported s1 package Python sources and named benchmark callers",
        "files": files,
    }
    destination = Path(output_dir) / "sources"
    destination.mkdir(parents=True, exist_ok=False)
    for label, blob in blobs.items():
        target = destination / label
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def verify_runtime_sources(manifest, *, additional_files=None):
    """Compare current source identity to the pre-load capture, including file set.

    Returns ``success`` plus changed/missing/added relative labels; read failures
    or manifest mutation are explicit failures. Check this in the caller's
    finally block before reporting a campaign's source identity as unchanged.
    """
    result = {"success": False, "changed": [], "missing": [], "added": []}
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or not isinstance(manifest.get("files"), dict)
    ):
        return {**result, "error": "Invalid runtime source manifest"}
    expected = manifest["files"]
    if manifest.get("identity_sha256") != _identity(expected):
        return {**result, "error": "Runtime source manifest identity was modified"}
    try:
        paths = _source_paths(additional_files)
        current = {
            label: _record(path.read_bytes(), label)
            for label, path in paths.items()
            if path.is_file()
        }
    except (OSError, TypeError, ValueError) as exc:
        return {**result, "error": f"{type(exc).__name__} while reading configured runtime sources"}
    changed = sorted(
        label for label in expected.keys() & current.keys() if expected[label] != current[label]
    )
    missing = sorted(expected.keys() - current.keys())
    added = sorted(current.keys() - expected.keys())
    return {
        "success": not changed and not missing and not added,
        "identity_sha256": manifest["identity_sha256"],
        "current_identity_sha256": _identity(current),
        "changed": changed,
        "missing": missing,
        "added": added,
    }
