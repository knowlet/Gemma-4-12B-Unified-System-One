"""Decision3 runtime integrity checks, independent of browser cases or models.

This module only inspects caller-provided runtime objects and pinned local model
files. It does not import a browser, load a model or run a benchmark.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MODEL_ID = "vllm-sr/d3-edge"


REVISION = "cab71e6ad3461b8086d51ddb572c656babd63b0b"


MANIFEST_SHA256 = "3bae8989a6feb36a66bf4d8c44112dc63199d0dc69f564dc895fe368b1e08e59"


MODEL_SHA256 = "6e00d04abbdbdf6d35c9e12065769d9edaacc3bfe9404583891797718d239dc8"


W0_SHA256 = "d408d94a0ab2915fa85c511dd9b8d62af49903c4e4ca28e2e2d5aff9ef565e0f"


def _need(value, message):
    if not value:
        raise ValueError(message)


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _file_sha(path, *, within=None):
    path = Path(path)
    _need(
        path.is_absolute()
        and path.is_file()
        and (
            (within is not None and path.resolve().is_relative_to(within))
            or (within is None and not path.is_symlink())
        ),
        "ordinary absolute pinned file required",
    )
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_manifest(runtime, cache_root):
    root = Path(runtime.root)
    cache_root = Path(cache_root)
    _need(
        root.is_absolute()
        and root.is_dir()
        and cache_root.is_absolute()
        and cache_root.is_dir()
        and root.resolve().is_relative_to(cache_root.resolve()),
        "pinned model snapshot within selected cache required",
    )
    cache_root = cache_root.resolve()
    manifest_path = root / "MODEL_MANIFEST.json"
    _need(
        _file_sha(manifest_path, within=cache_root) == MANIFEST_SHA256,
        "publisher manifest pin differs",
    )
    manifest = json.loads(manifest_path.read_bytes())
    _need(
        manifest.get("schema") == "d3-package-manifest/1"
        and manifest.get("identity", {}).get("model_sha256") == MODEL_SHA256,
        "publisher checkpoint identity differs",
    )
    files = manifest.get("files_sha256")
    _need(type(files) is dict and files, "manifest file inventory required")
    for name, digest in files.items():
        if (
            name in ("LICENSE", "NOTICE", "requirements.txt", ".gitattributes")
            or name.endswith(".md")
            or name.startswith(("assets/", "LICENSES/"))
        ):
            continue
        relative = Path(name)
        _need(not relative.is_absolute() and ".." not in relative.parts, "unsafe model file path")
        path = root / relative
        _need(
            _file_sha(path, within=cache_root) == digest, "model file differs from pinned manifest"
        )
    _need(runtime.manifest == manifest, "loaded runtime manifest differs")
    return _sha(manifest_path.read_bytes())


def _tensor_sha(tensor):
    return _sha(tensor.detach().to(device="cpu").contiguous().numpy().tobytes())


def _validate_provider_spec(spec):
    _need(
        getattr(spec, "adapter", None) == "decision3"
        and getattr(spec, "model_id", None) == MODEL_ID
        and getattr(spec, "revision", None) == REVISION
        and getattr(spec, "device", None) == "cuda:0"
        and getattr(spec, "precision", None) == "bfloat16"
        and getattr(spec, "context_limit", None) == 16384,
        "pinned native provider spec required",
    )
