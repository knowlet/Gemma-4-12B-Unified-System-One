#!/usr/bin/env python3
"""Build a prospective x86 runtime JSON from real source/metadata without ML imports.

This is an independent stdlib-only preparation helper, not part of training90
or qualification91. Its output SHA must be reviewed and selected externally.
No model, processor, adapter tensor, base weight or network access is performed.
The historical ARM builder/runtime remain separate and unchanged.
"""

import argparse
import hashlib
import json
import os
import stat
from email.parser import BytesHeaderParser
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
REFERENCE = (
    ROOT
    / "artifacts/jev-runtime/public-native-action-compiler-07-root-20261016-01/actual-compile-01/processor-source-closure.json"
)
REFERENCE_PIN = {
    "sha256": "ff87ceb2563bdee706c4b8cc1064a4682534be2925372c0c02c7ccd2d04e7bbc",
    "size_bytes": 20246,
}
HISTORICAL = ROOT / "artifacts/jev-browser/20261015-bf16-qualified-bundle-01/qualification.json"
HISTORICAL_PIN = {
    "sha256": "1e303a9c243da12917b313c09c8354bf6a85b09f9f3d652c8c969dcd7e035e1b",
    "size_bytes": 1949873,
}
MODEL = "knowlet/Gemma-4-12B-Unified-System-One"
REVISION = "a66f836b56605039fe040f330180e336d19b3362"
GIB, MAX_FILES, MAX_ONE = 1024**3, 10000, 64 * 1024**2
BASE_METADATA = {
    "artifact-manifest.json",
    "config.json",
    "generation_config.json",
    "s1_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
}
EXTRA_DISTRIBUTIONS = {
    "peft": "0.21.0",
    "fastapi": "0.115.12",
    "uvicorn": "0.34.3",
    "httpx": "0.28.1",
}
ALL_DISTRIBUTIONS = {
    "torch",
    "torchvision",
    "numpy",
    "pillow",
    "jinja2",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "safetensors",
    *EXTRA_DISTRIBUTIONS,
}
NATIVE_NAME = "_C.cpython-313-x86_64-linux-gnu.so"
TREE_ROOTS = (
    "peft",
    "transformers/generation",
    "transformers/models/gemma4",
    "transformers/models/gemma4_assistant",
    "transformers/models/gemma4_unified",
    "transformers/models/gemma4_unified_assistant",
)
DEVICE = {
    "device": "cuda",
    "dtype": "bfloat16",
    "quantization": "none",
    "attention": "sdpa",
    "cudnn_sdp": False,
}
MANDATORY = {
    "torch",
    "torch._C",
    "transformers.generation.utils",
    "transformers.generation.configuration_utils",
    "transformers.cache_utils",
    "peft.peft_model",
    "peft.tuners.lora.layer",
    "peft.utils.save_and_load",
    "transformers.models.gemma4_unified.modeling_gemma4_unified",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def strict_json(blob):
    def pairs(items):
        result = {}
        for name, value in items:
            require(name not in result, "Duplicate JSON key")
            result[name] = value
        return result

    def constant(_):
        raise ValueError("Nonfinite JSON constant")

    return json.loads(blob, object_pairs_hook=pairs, parse_constant=constant)


def ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary path required",
    )
    return path


def shape(st):
    return (st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def validate_pin(value):
    require(
        type(value) is dict
        and set(value) == {"sha256", "size_bytes"}
        and type(value["sha256"]) is str
        and len(value["sha256"]) == 64
        and all(c in "0123456789abcdef" for c in value["sha256"])
        and type(value["size_bytes"]) is int
        and 0 <= value["size_bytes"] <= MAX_ONE,
        "Strict bounded pin",
    )


class SourcePins:
    def __init__(self):
        self.values, self.bytes, self.entries = {}, 0, 0

    def pin(self, path, expected=None):
        path = ordinary(path)
        require(
            path.suffix not in {".safetensors", ".bin", ".pt", ".pth", ".ckpt"},
            "Tensor/base payload access is forbidden",
        )
        before = path.lstat()
        require(
            stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= MAX_ONE,
            "Bounded source/metadata file required",
        )
        key = str(path)
        if key not in self.values:
            require(
                len(self.values) + 1 <= MAX_FILES and self.bytes + before.st_size <= GIB,
                "Projected source inventory cap",
            )
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            require(shape(os.fstat(stream.fileno())) == shape(before), "Source changed before read")
            digest = hashlib.sha256()
            count = 0
            while chunk := stream.read(1024**2):
                count += len(chunk)
                require(count <= before.st_size, "Source grew during bounded read")
                digest.update(chunk)
            require(
                count == before.st_size
                and shape(os.fstat(stream.fileno())) == shape(before) == shape(path.lstat()),
                "Source changed during read",
            )
        value = {"sha256": digest.hexdigest(), "size_bytes": count}
        require(key not in self.values or self.values[key] == value, "Repeated source pin changed")
        if expected is not None:
            validate_pin(expected)
            require(value == expected, "Known source/metadata authority differs: " + key)
        if key not in self.values:
            self.values[key] = value
            self.bytes += count
        return value

    def json(self, path, expected):
        self.pin(path, expected)
        require(expected["size_bytes"] <= 4 * 1024**2, "Small JSON authority only")
        with ordinary(path).open("rb") as stream:
            blob = stream.read(expected["size_bytes"] + 1)
        require(
            {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)} == expected,
            "Authority readback differs",
        )
        return strict_json(blob)

    def python_files(self, root):
        pending, result = [ordinary(root)], []
        while pending:
            directory = pending.pop()
            require(directory.is_dir(), "Static runtime source directory missing")
            with os.scandir(directory) as entries:
                for entry in entries:
                    self.entries += 1
                    require(self.entries <= MAX_FILES, "Bounded static source traversal")
                    info = entry.stat(follow_symlinks=False)
                    require(not stat.S_ISLNK(info.st_mode), "Static source tree link rejected")
                    if stat.S_ISDIR(info.st_mode):
                        if entry.name != "__pycache__":
                            pending.append(Path(entry.path))
                    elif entry.name.endswith(".py"):
                        require(stat.S_ISREG(info.st_mode), "Python source must be ordinary")
                        result.append(Path(entry.path))
        require(bool(result), "Empty static source tree")
        return sorted(result)


def distribution(name, version, metadata, pins, expected=None):
    value = pins.pin(metadata, expected)
    with ordinary(metadata).open("rb") as stream:
        blob = stream.read(value["size_bytes"] + 1)
    require(
        {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)} == value,
        "Distribution metadata readback differs",
    )
    headers = BytesHeaderParser().parsebytes(blob)
    require(
        headers.get("Name", "").lower().replace("-", "_") == name.lower().replace("-", "_")
        and headers.get("Version") == version,
        "Exact declared distribution name/version",
    )
    return {"version": version, "metadata": {"path": str(ordinary(metadata)), **value}}


def module_name(path, overlay):
    relative = path.relative_to(overlay).with_suffix("")
    parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
    require(parts and all(part.isidentifier() for part in parts), "Qualified static module name")
    return ".".join(parts)


def native_binding(torch_root, pins):
    """Find the one actual CPython313/x86 binding without importing Torch."""
    torch_root = ordinary(torch_root)
    candidates = []
    with os.scandir(torch_root) as entries:
        for entry in entries:
            pins.entries += 1
            require(pins.entries <= MAX_FILES, "Bounded native source discovery")
            if entry.name.startswith("_C") and entry.name.endswith(".so"):
                info = entry.stat(follow_symlinks=False)
                require(stat.S_ISREG(info.st_mode), "Native binding must be ordinary, not a link")
                candidates.append(ordinary(entry.path))
    require(len(candidates) == 1, "Exactly one unambiguous native Torch binding required")
    native = candidates[0]
    require(native.name == NATIVE_NAME, "CPython313 x86_64 native Torch binding required")
    require(0 < native.stat().st_size <= MAX_ONE, "Bounded nonempty native Torch binding")
    # Validate only the ELF identification and machine fields. The binding is
    # still opaque; this does not attest its dependencies or all Torch/CUDA code.
    expected = pins.pin(native)
    before = native.stat()
    with native.open("rb") as stream:
        require(
            shape(os.fstat(stream.fileno())) == shape(before),
            "Native binding changed before header read",
        )
        header = stream.read(20)
        require(
            shape(os.fstat(stream.fileno())) == shape(before) == shape(native.stat()),
            "Native binding changed during header read",
        )
    require(
        len(header) == 20
        and header[:6] == b"\x7fELF\x02\x01"
        and int.from_bytes(header[18:20], "little") == 62,
        "Actual little-endian ELF64 x86_64 native binding required",
    )
    pins.pin(native, expected)
    return native


def build_profile():
    pins = SourcePins()
    closure = pins.json(REFERENCE, REFERENCE_PIN)
    history = pins.json(HISTORICAL, HISTORICAL_PIN)
    overlay = ordinary(closure["overlay"])
    require(
        overlay == ROOT / "artifacts/jev-runtime/overlay"
        and closure["no_remote_code"] is True
        and closure["interpreter"] == "/home/albert_hsieh/ComfyUI/.venv/bin/python"
        and closure["python_major_minor"] == [3, 13]
        and len(closure["files"]) == 51
        and len(closure["packages"]) == 9,
        "Fixed CPU reference profile",
    )
    require(
        history["kind"] == "jev_action_bf16_qualification_v1", "Historical metadata pin source kind"
    )
    base_metadata = history["observed_diagnostics"]["base_metadata"]
    require(set(base_metadata) == BASE_METADATA, "Exact known eight base metadata pins")
    files, modules, distributions = {}, {}, {}

    def add(path, name=None, expected=None):
        path = ordinary(path)
        files[str(path)] = pins.pin(path, expected)
        if name is not None:
            require(name not in modules or modules[name] == str(path), "Duplicate qualified origin")
            modules[name] = str(path)

    for spec in closure["files"].values():
        add(spec["path"], expected={k: spec[k] for k in ("sha256", "size_bytes")})
    for name, path in closure["module_origins"].items():
        require(str(ordinary(path)) in files, "Known processor module missing from source51")
        modules[name] = str(ordinary(path))
    for name, spec in closure["packages"].items():
        distributions[name] = distribution(
            name,
            closure["versions"][name],
            spec["metadata_path"],
            pins,
            {k: spec[k] for k in ("sha256", "size_bytes")},
        )
    for name, version in EXTRA_DISTRIBUTIONS.items():
        path = overlay / (name + "-" + version + ".dist-info/METADATA")
        distributions[name] = distribution(name, version, path, pins)
        files[str(path)] = {k: distributions[name]["metadata"][k] for k in ("sha256", "size_bytes")}
    for label in TREE_ROOTS:
        for path in pins.python_files(overlay / label):
            add(path, module_name(path, overlay))
    for label in (
        "transformers/cache_utils.py",
        "transformers/modeling_utils.py",
        "transformers/masking_utils.py",
        "transformers/integrations/sdpa_attention.py",
    ):
        path = overlay / label
        add(path, module_name(path, overlay))
    torch_root = ordinary(Path(closure["module_origins"]["torch"]).parent)
    native = native_binding(torch_root, pins)
    add(torch_root / "__init__.py", "torch")
    add(native, "torch._C")
    add(torch_root / "nn/functional.py", "torch.nn.functional")
    add(torch_root / "backends/cuda/__init__.py", "torch.backends.cuda")
    cache = ordinary(
        ROOT
        / "artifacts/jev-local-hf/hub"
        / ("models--" + MODEL.replace("/", "--"))
        / "snapshots"
        / REVISION
    )
    for name, expected in base_metadata.items():
        validate_pin(expected)
        path = (cache / name).resolve(strict=True)
        require(
            path.is_relative_to(cache.parents[1]),
            "Base metadata retains original per-model boundary",
        )
        pins.pin(path, expected)
    require(
        MANDATORY <= set(modules) and all(path in files for path in modules.values()),
        "Complete mandatory runtime origins",
    )
    require(set(distributions) == ALL_DISTRIBUTIONS, "Exact thirteen runtime distributions")
    for path, value in tuple(pins.values.items()):
        pins.pin(path, value)
    profile = {
        "schema_id": "jev_public_action_qualification_runtime_v1",
        "distributions": distributions,
        "files": files,
        "modules": modules,
        "base_metadata": base_metadata,
        "device": DEVICE,
    }
    blob = canonical(profile)
    require(
        len(blob) <= MAX_ONE
        and len(files) + len(distributions) + len(base_metadata) + 3 <= MAX_FILES
        and sum(row["size_bytes"] for row in files.values())
        + sum(row["metadata"]["size_bytes"] for row in distributions.values())
        + sum(row["size_bytes"] for row in base_metadata.values())
        + len(blob)
        <= GIB,
        "Projected runtime capture cap",
    )
    builder = ordinary(__file__)
    builder_pin = pins.pin(builder)
    receipt = {
        "stage": "built_unqualified_external_sha_review_required",
        "runtime": {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)},
        "builder": {"path": str(builder), **builder_pin},
        "reference": {"path": str(REFERENCE), **REFERENCE_PIN},
        "historical_metadata_authority": {"path": str(HISTORICAL), **HISTORICAL_PIN},
        "files": len(files),
        "modules": len(modules),
        "distributions": len(distributions),
        "base_metadata": len(base_metadata),
        "source_walk_entries": pins.entries,
        "all_unique_read_files": len(pins.values),
        "all_unique_read_bytes": pins.bytes,
        "static_tree_roots": list(TREE_ROOTS),
        "opaque_native_binding": str(native),
        "native_binding": {
            "path": str(native),
            **files[str(native)],
            "architecture": "ELF64 little-endian x86_64",
            "python_major_minor": [3, 13],
        },
        "scope": "Static prospective source/metadata superset; no framework import, model, processor, tensor, generation, network, qualification or task score. Torch binding bytes are opaque; no full Torch/CUDA binary attestation.",
    }
    return blob, canonical(receipt)


def create(path, blob):
    with path.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    require(
        args.run, "Explicit --run required; outputs must be reviewed before their SHA is selected"
    )
    output = ordinary(args.output)
    require(not output.exists(), "Fresh output directory required")
    blob, receipt = build_profile()
    output.mkdir(parents=True)
    create(output / "build-receipt.json", receipt)
    pending = output / "runtime.pending"
    create(pending, blob)
    with pending.open("rb") as stream:
        actual = stream.read(len(blob) + 1)
    require(
        actual == blob and canonical(strict_json(actual)) == blob,
        "Actual pending runtime readback differs",
    )
    os.replace(pending, output / "runtime.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
