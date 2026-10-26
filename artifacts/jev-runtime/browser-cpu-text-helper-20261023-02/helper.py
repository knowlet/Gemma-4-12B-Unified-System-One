"""Hash-bound source successor: exact Transformer critical-source root binding.

Native generation, strict protocol, evidence and deadline behavior are reused
unchanged from frozen01. Importing either layer imports stdlib only. No server,
model, owner or network is started here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import weakref
from pathlib import Path

BASE_DIRECTORY = Path(__file__).resolve().parent.with_name("browser-cpu-text-helper-20261023-01")
BASE_SOURCE_SHA256 = "a312dba1fabd173e1ce502f6896431a7cc3d6b4979a7eae49c2b1807d343b0fa"
CRITICAL_RELATIVE_PATHS = (
    "models/qwen2/modeling_qwen2.py",
    "generation/utils.py",
    "modeling_utils.py",
    "cache_utils.py",
    "tokenization_utils_base.py",
    "models/auto/tokenization_auto.py",
)


def _read_base():
    path = BASE_DIRECTORY / "helper.py"
    if not path.is_file() or path.is_symlink() or path.stat().st_size != 31089:
        raise ValueError("Exact ordinary frozen01 source required")
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != BASE_SOURCE_SHA256:
        raise ValueError("Frozen01 source digest drift")
    spec = importlib.util.spec_from_file_location("_cpu_text_helper_frozen01_for02", path)
    module = importlib.util.module_from_spec(spec)
    # Execute the verified bytes themselves; do not reopen source through a loader.
    exec(compile(blob, str(path), "exec"), module.__dict__)
    return module


_BASE = _read_base()
_ACCEPTED = weakref.WeakSet()
MODEL_ID, REVISION, PROFILE = _BASE.MODEL_ID, _BASE.REVISION, _BASE.PROFILE
TEXT_VALUE, Refusal = _BASE.TEXT_VALUE, _BASE.Refusal
DirectoryJournal, Deadline = _BASE.DirectoryJournal, _BASE.Deadline


def _complete_owner_pin(entry):
    _BASE.require(
        type(entry) is dict
        and type(entry.get("path")) is str
        and Path(entry["path"]).is_absolute()
        and type(entry.get("size_bytes")) is int
        and entry["size_bytes"] >= 0
        and type(entry.get("sha256")) is str
        and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None,
        "Completed immutable external owner ticket/verifier pins required",
    )


def validate_selection(selection):
    """Strict metadata gate before delegating any future owner/payload/import."""
    _BASE.validate_selection(selection)
    origin = selection["module_origins"]["transformers"]
    _BASE.require(
        type(origin) is str and Path(origin).is_absolute(), "Absolute selected Transformer origin"
    )
    origin = Path(origin).resolve()
    _BASE.require(origin.name == "__init__.py", "Selected Transformer package origin required")
    root = origin.parent
    paths = set()
    for entry in selection["runtime_files"]:
        _BASE.require(
            type(entry) is dict
            and type(entry.get("path")) is str
            and Path(entry["path"]).is_absolute(),
            "Absolute pinned source path required",
        )
        paths.add(Path(entry["path"]).resolve())
    for relative in CRITICAL_RELATIVE_PATHS:
        expected = root / relative
        resolved = expected.resolve()
        _BASE.require(
            resolved == expected and resolved.is_relative_to(root) and resolved in paths,
            "Critical Transformer pin must resolve inside the selected package root: " + relative,
        )
    _complete_owner_pin(selection["owner"]["ticket"])
    _complete_owner_pin(selection["owner"]["verifier"])
    return selection


def load_runtime(selection_path, expected_selection_sha256):
    path = Path(selection_path)
    _BASE.require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024 * 1024,
        "Bounded ordinary completed runtime descriptor required",
    )
    blob = path.read_bytes()
    _BASE.require(
        _BASE.digest(blob) == expected_selection_sha256,
        "Immutable successor runtime selection required",
    )
    validate_selection(_BASE.strict_json(blob))
    # The base rereads/verifies the same exact SHA before its own owner, file,
    # import-origin and native-runtime checks. No base globals are replaced.
    runtime = _BASE.load_runtime(str(path), expected_selection_sha256)
    _ACCEPTED.add(runtime)
    return runtime


def make_application(runtime, journal, api_key):
    _BASE.require(
        runtime in _ACCEPTED, "Production runtime must pass successor root-bound load_runtime"
    )
    return _BASE.make_application(runtime, journal, api_key)


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "profile": PROFILE,
                "source_revision": "02",
                "status": "source_only_not_ready",
                "exact_critical_root_gate": True,
                "model_started": False,
                "server_started": False,
                "external_owner_and_completed_selection_required": True,
            }
        )
    )
