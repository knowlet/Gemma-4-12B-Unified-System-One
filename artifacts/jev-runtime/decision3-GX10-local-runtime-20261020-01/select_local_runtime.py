#!/usr/bin/env python3
"""Select local Decision3 source/metadata origins without importing ML packages.

This records selected Python source and distribution METADATA bytes. Native
module origins have path/size authority only; no native payload is read here.
It is a runtime selection, not a capacity, kernel or model qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import stat
import sys
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
BOOTSTRAP = (
    ROOT / "artifacts/jev-runtime/decision3-GX10-warm-successor-20261020-01/gx10_warm_bootstrap.py"
)
BOOTSTRAP_SHA = "4f383edd45dc4ce3d5e6be90a83bed0a431f42dcb115f92fcbb20be25e62f91a"
INTERPRETER = "/home/albert_hsieh/ComfyUI/.venv/bin/python"
OVERLAY = ROOT / "artifacts/jev-runtime/overlay"
SITE = Path("/home/albert_hsieh/ComfyUI/.venv/lib/python3.13/site-packages")
DISTRIBUTIONS = (
    "torch",
    "torchvision",
    "transformers",
    "tokenizers",
    "safetensors",
    "accelerate",
    "huggingface-hub",
    "numpy",
    "triton",
    "einops",
    "pydantic",
)
SOURCE_MODULES = {
    "torch": SITE / "torch/__init__.py",
    "torch.cuda": SITE / "torch/cuda/__init__.py",
    "torch.cuda.memory": SITE / "torch/cuda/memory.py",
    "torch.nn.functional": SITE / "torch/nn/functional.py",
    "torchvision": SITE / "torchvision/__init__.py",
    "transformers": OVERLAY / "transformers/__init__.py",
    "transformers.modeling_utils": OVERLAY / "transformers/modeling_utils.py",
    "transformers.modeling_attn_mask_utils": OVERLAY / "transformers/modeling_attn_mask_utils.py",
    "transformers.cache_utils": OVERLAY / "transformers/cache_utils.py",
    "transformers.dynamic_module_utils": OVERLAY / "transformers/dynamic_module_utils.py",
    "transformers.integrations.sdpa_attention": OVERLAY
    / "transformers/integrations/sdpa_attention.py",
    "tokenizers": OVERLAY / "tokenizers/__init__.py",
    "safetensors": OVERLAY / "safetensors/__init__.py",
    "safetensors.torch": OVERLAY / "safetensors/torch.py",
    "accelerate": OVERLAY / "accelerate/__init__.py",
    "huggingface_hub": OVERLAY / "huggingface_hub/__init__.py",
    "numpy": SITE / "numpy/__init__.py",
    "triton": SITE / "triton/__init__.py",
    "einops": SITE / "einops/__init__.py",
    "pydantic": OVERLAY / "pydantic/__init__.py",
}
for _family in ("qwen3_5", "qwen3_vl"):
    for _kind in ("configuration", "modeling"):
        _name = f"transformers.models.{_family}.{_kind}_{_family}"
        SOURCE_MODULES[_name] = OVERLAY / (_name.replace(".", "/") + ".py")
NATIVE_MODULES = {
    "torch._C": SITE / "torch/_C.cpython-313-aarch64-linux-gnu.so",
    "tokenizers.tokenizers": OVERLAY / "tokenizers/tokenizers.abi3.so",
    "safetensors._safetensors_rust": OVERLAY / "safetensors/_safetensors_rust.abi3.so",
    "numpy._core._multiarray_umath": SITE
    / "numpy/_core/_multiarray_umath.cpython-313-aarch64-linux-gnu.so",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ordinary(path):
    require(path.is_absolute(), "Absolute selected source path required")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Linked selected source")
    require(stat.S_ISREG(path.stat().st_mode), "Selected ordinary file required")


def small_pin(path):
    ordinary(path)
    require(0 < path.stat().st_size <= 1024**2, "Small source/METADATA only")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        blob = stream.read(1024**2 + 1)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(
        len(blob) == before.st_size
        and tuple(getattr(before, k) for k in fields)
        == tuple(getattr(after, k) for k in fields)
        == tuple(getattr(path.stat(), k) for k in fields),
        "Selected file changed while reading",
    )
    return {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}


def select():
    require(
        sys.executable == INTERPRETER
        and list(sys.version_info[:2]) == [3, 13]
        and platform.machine() == "aarch64"
        and sys.flags.no_user_site
        and sys.dont_write_bytecode,
        "Use the selected Comfy Python with -B -s",
    )
    require(not any(name in sys.modules for name in SOURCE_MODULES), "ML modules already loaded")
    require(small_pin(BOOTSTRAP)["sha256"] == BOOTSTRAP_SHA, "Bootstrap source changed")
    files, distributions, modules = {}, {}, {}
    for name in DISTRIBUTIONS:
        distribution = importlib.metadata.distribution(name)
        path = Path(distribution._path) / "METADATA"
        files[str(path)] = small_pin(path)
        distributions[name] = {"version": distribution.version, "metadata_path": str(path)}
    for name, path in SOURCE_MODULES.items():
        files[str(path)] = small_pin(path)
        modules[name] = {
            "path": str(path),
            "kind": "source",
            "size_bytes": files[str(path)]["size_bytes"],
        }
    for name, path in NATIVE_MODULES.items():
        ordinary(path)
        modules[name] = {
            "path": str(path),
            "kind": "native_origin",
            "size_bytes": path.stat().st_size,
        }
    value = {
        "schema_id": "decision3-gx10-local-runtime-selection-v1",
        "machine": "aarch64",
        "interpreter": INTERPRETER,
        "python": [3, 13],
        "files": files,
        "distributions": distributions,
        "modules": modules,
        "kernel_policy": {
            "fast": False,
            "permutation_average": False,
            "native_batch_size": 8,
            "temperature": 1.0,
            "backbone": "bfloat16",
            "readout": "float32",
            "context": 8192,
        },
    }
    blob = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    spec = importlib.util.spec_from_file_location("gx10_runtime_selection_bootstrap", BOOTSTRAP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = module.load_worker()
    checked = module.RuntimeSelection(blob, hashlib.sha256(blob).hexdigest(), worker)
    receipt = checked.verify()
    require(not any(name in sys.modules for name in SOURCE_MODULES), "Selection imported ML")
    return blob, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    require(output.parent.is_dir(), "Existing output parent required")
    require(not any(p.is_symlink() for p in (output, *output.parents)), "Linked output path")
    blob, receipt = select()
    with output.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "output": str(output),
                "pin": small_pin(output),
                "verification": receipt,
                "ML_imported": False,
                "native_payloads_read": False,
                "model_loaded": False,
            }
        )
    )


if __name__ == "__main__":
    main()
