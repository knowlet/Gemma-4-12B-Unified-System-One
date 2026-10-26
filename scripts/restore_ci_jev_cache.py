"""Recover the byte-identical historical field-JSON cache for CPU contracts.

Run prepare_ci_jev.py first. This script never downloads or loads a model. The
original pure compiler reproduces the seven payloads, but its manifest includes
elapsed time and peak RSS. Recover those *historical* observations from the
authenticated original report, then require the complete original manifest hash.
This restores old bytes; it does not claim those observations describe this run.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("scripts/jev_field_json_grammar.py")
SOURCE_SHA = "3efe0870aa03fd036a89a83dd9c56fdba32dac673231a81bb0507aeccdedb6de"
NATIVE = Path(
    "artifacts/jev-local-hf/hub/models--knowlet--Gemma-4-12B-Unified-System-One/"
    "snapshots/a66f836b56605039fe040f330180e336d19b3362"
)
CACHE = Path("artifacts/jev-runtime/field-json-grammar-20261018-01/full-compile-01/cache")
PROOF = Path("docs/validation/2026-10-18/jev-field-json-grammar-prototype.json")
INVENTORY = Path(
    "artifacts/jev-runtime/native-v2-browser-execution-20261022-01/"
    "source-population-proposal-01.json"
)
DOCUMENT_PINS = {
    PROOF: ("bd038964b420a623d49ae9d3253657968410766d5ce9d144d9e58180866393ef", 2349),
    INVENTORY: ("1c3f888a68e1f88b6f1ad624b05ab6891a78037645403dae6e6c7bca5d247c13", 237016),
}
INPUT_PINS = {
    "tokenizer.json": (
        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
        32169626,
    ),
    "generation_config.json": (
        "040d0ccb9e10df02df9738424c1ce6d20d5d15b09362da922093ccd97573dbb4",
        255,
    ),
}
MANIFEST_PIN = ("6e5ac130fa1f5b6bb315f71d17cb8d3d1b8f2552eef319a1e695c06ebf5b0dc6", 1906)
PAYLOADS = frozenset(
    {
        "classes.bin",
        "json-dfa.bin",
        "json-states.json",
        "offsets.bin",
        "ordinary-mask.bin",
        "ordinary-next.bin",
        "pieces.bin",
    }
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        f"Ordinary path required: {path}",
    )
    return path


def read(path, expected=None, *, limit=64 * 1024**2):
    path = ordinary(path)
    require(path.is_file() and path.stat().st_size <= limit, f"Bounded file required: {path}")
    with path.open("rb") as stream:
        blob = stream.read(limit + 1)
    require(len(blob) <= limit, f"File exceeds bound: {path}")
    if expected is not None:
        require((digest(blob), len(blob)) == expected, f"Pinned bytes differ: {path}")
    return blob


def authorities(root):
    documents = {path: json.loads(read(root / path, pin)) for path, pin in DOCUMENT_PINS.items()}
    rows = documents[INVENTORY]["grammar_cache"]["files"]
    pins = {}
    for row in rows:
        name = Path(row["path"]).name
        require(
            row["path"] == (CACHE / name).as_posix() and name not in pins,
            "Exact unique historical cache paths required",
        )
        pins[name] = (row["sha256"], row["size_bytes"])
    require(set(pins) == PAYLOADS | {"manifest.json"}, "Exact historical cache inventory required")
    require(pins["manifest.json"] == MANIFEST_PIN, "Original manifest pin differs")
    report = documents[PROOF]["verification"]
    observations = {
        "compile_elapsed_seconds": report["compiler_seconds"],
        "peak_rss_KiB": report["compiler_peak_RSS_KiB"],
        "compiler_address_limit_bytes": report["compiler_address_cap_bytes"],
        "max_cache_bytes": report["cache_cap_bytes"],
    }
    return pins, observations


def compiler(root):
    path = ordinary(root / SOURCE)
    blob = read(path, limit=1024**2)
    require(digest(blob) == SOURCE_SHA, "Original compiler source pin differs")
    # Execute the authenticated bytes, not a second unverified import-path read.
    module = types.ModuleType("_ci_authenticated_field_json_grammar")
    module.__file__ = str(path)
    previous = sys.modules.get(module.__name__)
    sys.modules[module.__name__] = module
    try:
        exec(compile(blob, str(path), "exec"), module.__dict__)
    finally:
        if previous is None:
            sys.modules.pop(module.__name__, None)
        else:
            sys.modules[module.__name__] = previous
    return module


def verify_files(directory, pins):
    directory = ordinary(directory)
    require(directory.is_dir(), "Complete cache directory required")
    require(
        {p.name for p in directory.iterdir()} == set(pins), "Exact cache file inventory required"
    )
    for name, expected in pins.items():
        read(directory / name, expected)


def restore(root=ROOT):
    root = ordinary(root)
    pins, observations = authorities(root)
    for name, expected in INPUT_PINS.items():
        read(root / NATIVE / name, expected)
    grammar = compiler(root)
    destination = ordinary(root / CACHE)
    arguments = {
        "expected_tokenizer_sha256": INPUT_PINS["tokenizer.json"][0],
        "expected_generation_sha256": INPUT_PINS["generation_config.json"][0],
        "expected_source_sha256": SOURCE_SHA,
    }

    def verify(directory):
        verify_files(directory, pins)
        grammar.load_cache(directory, expected_manifest_sha256=MANIFEST_PIN[0], **arguments)

    if destination.exists():
        verify(destination)
        return {"restored": False, "manifest_sha256": MANIFEST_PIN[0]}

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".restore-cache-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "cache"
        grammar.compile_cache(
            root / NATIVE / "tokenizer.json",
            root / NATIVE / "generation_config.json",
            staged,
            **arguments,
        )
        require(
            {p.name for p in staged.iterdir()} == set(pins),
            "Exact compiler output inventory required",
        )
        for name in PAYLOADS:
            read(staged / name, pins[name])
        historical = json.loads(read(staged / "manifest.json", limit=1024**2))
        historical.update(observations)
        recovered = grammar.encode(historical)
        require(
            (digest(recovered), len(recovered)) == MANIFEST_PIN,
            "Recovered historical manifest pin differs",
        )
        (staged / "manifest.json").write_bytes(recovered)
        verify(staged)
        require(not destination.exists(), "Cache appeared during recovery; refusing replacement")
        ordinary(destination)
        staged.rename(destination)
    return {"restored": True, "manifest_sha256": MANIFEST_PIN[0]}


if __name__ == "__main__":
    print(json.dumps(restore()))
