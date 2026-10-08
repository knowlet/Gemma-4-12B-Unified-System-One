"""Native Blackwell NVFP4 inference and an explicit packed checkpoint format.

Transformers 5.17 cannot serialize/reload its NVFP4 modules. This adapter stores
those exact four kernel buffers in safetensors and reconstructs their modules;
it never expands packed weights or claims compatibility with ModelOpt exports.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import struct
import threading
from pathlib import Path

FORMAT = "s1-transformers-nvfp4-v1"
MANIFEST = "nvfp4_manifest.json"
KERNEL_REPO = "kernels-community/nvfp4-gemm"
KERNEL_REVISION = "a66348abcb8cc22da69f2b77a3f5d4e01748f495"
DENSE_MODULES = ("lm_head", "model.embed_vision", "model.embed_audio")
_LANGUAGE_PREFIX = "model.language_model."
_KERNEL_LOCK = threading.Lock()
_DTYPE_BYTES = {
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "BOOL": 1,
}
_TORCH_DTYPES = {
    "uint8": "U8",
    "int8": "I8",
    "int16": "I16",
    "int32": "I32",
    "int64": "I64",
    "float16": "F16",
    "bfloat16": "BF16",
    "float32": "F32",
    "float64": "F64",
    "bool": "BOOL",
}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def configure_nvfp4_kernel():
    """Pin the upstream native kernel before its first lazy load."""
    from transformers.integrations import hub_kernels, nvfp4

    wanted = {"repo_id": KERNEL_REPO, "revision": KERNEL_REVISION}
    with _KERNEL_LOCK:
        if hub_kernels._HUB_KERNEL_MAPPING["nvfp4"] != wanted:
            if hub_kernels._KERNEL_MODULE_MAPPING.get("nvfp4") is not None:
                raise RuntimeError("NVFP4 kernel was loaded before the release revision was pinned")
            hub_kernels._HUB_KERNEL_MAPPING["nvfp4"] = wanted
            nvfp4.load_nvfp4_kernel.cache_clear()
        return nvfp4.load_nvfp4_kernel()


def _validate_device(device, precision):
    if not str(device).startswith("cuda") or precision != "bfloat16":
        raise ValueError("NVFP4 requires a single Blackwell CUDA device and bfloat16 precision")
    import torch

    if not torch.cuda.is_available():
        raise ValueError("NVFP4 requires a Blackwell CUDA GPU")
    if torch.cuda.get_device_capability(device)[0] < 10:
        raise ValueError("NVFP4 requires Blackwell (compute capability 10.0 or later)")
    torch_version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if torch_version < (2, 9):
        raise ValueError(
            "the pinned NVFP4 kernel requires PyTorch >= 2.9; use the NVFP4 environment"
        )


def _packed_specs(in_features, out_features):
    return {
        "weight": ("U8", [out_features, in_features // 2]),
        "weight_sf": (
            "I32",
            [math.ceil(out_features / 128) * 128, math.ceil((in_features // 16) / 4)],
        ),
        "weight_sf_rowmajor": ("U8", [out_features, in_features // 16]),
        "weight_global_scale": ("F32", [1]),
    }


def _tensor_spec(tensor):
    dtype = str(tensor.dtype).removeprefix("torch.")
    if dtype not in _TORCH_DTYPES:
        raise ValueError(f"unsupported checkpoint dtype: {dtype}")
    return {"dtype": _TORCH_DTYPES[dtype], "shape": list(tensor.shape)}


def inspect_nvfp4(lm):
    """Require real packed modules and a floating candidate readout."""
    import torch
    from transformers.integrations.nvfp4 import NVFP4Linear

    base = lm.get_base_model() if hasattr(lm, "peft_config") else lm
    modules = []
    packed_bytes = 0
    for name, module in base.named_modules():
        if not isinstance(module, NVFP4Linear):
            continue
        if not name.startswith(_LANGUAGE_PREFIX):
            raise ValueError(f"NVFP4 unexpectedly converted a media module or LM head: {name}")
        modules.append(name)
        for field, (dtype, shape) in _packed_specs(module.in_features, module.out_features).items():
            tensor = getattr(module, field)
            if _tensor_spec(tensor) != {"dtype": dtype, "shape": shape} or tensor.is_meta:
                raise ValueError(f"NVFP4 packed tensor has invalid storage: {name}.{field}")
            if tensor.device.type != "cuda":
                raise ValueError("NVFP4 does not support CPU or disk offload")
            packed_bytes += tensor.numel() * tensor.element_size()
        if (
            not torch.isfinite(module.weight_global_scale).all()
            or not (module.weight_global_scale > 0).all()
        ):
            raise ValueError(f"NVFP4 global scale must be finite and positive: {name}")
    if not modules:
        raise ValueError("requested NVFP4 did not convert any language linear modules")
    head = base.get_output_embeddings()
    if isinstance(head, NVFP4Linear) or not head.weight.is_floating_point():
        raise ValueError("candidate readout requires an unquantized floating-point LM head")
    return {
        "method": "transformers_native_nvfp4",
        "mode": "nvfp4",
        "format": FORMAT,
        "quantized_linear_modules": len(modules),
        "packed_tensor_bytes": packed_bytes,
        "dense_modules": list(DENSE_MODULES),
        "head_dtype": str(head.weight.dtype).removeprefix("torch."),
        "weight_format": "E2M1",
        "block_scale_format": "E4M3",
        "block_size": 16,
        "global_scale_dtype": "float32",
        "activation_quantization": "dynamic_nvfp4_prefill",
        "small_batch_compute": "W4A16_GEMV_for_1_or_2_rows",
        "cpu_offload": False,
        "kernel": {"repo_id": KERNEL_REPO, "revision": KERNEL_REVISION},
    }


def _safe_path(root, name):
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("checkpoint shard names must be plain relative filenames")
    candidate = root / name
    if not candidate.is_file():
        raise ValueError(f"checkpoint file is absent: {name}")
    return candidate


def _safetensor_header(path):
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(100_000_000, size - 8):
            raise ValueError("invalid safetensors header length")
        header = json.loads(handle.read(length))
    if not isinstance(header, dict):
        raise ValueError("invalid safetensors header")
    entries = {}
    intervals = []
    for key, spec in header.items():
        if key == "__metadata__":
            continue
        dtype, shape, offsets = spec.get("dtype"), spec.get("shape"), spec.get("data_offsets")
        if (
            dtype not in _DTYPE_BYTES
            or not isinstance(shape, list)
            or any(type(x) is not int or x < 0 for x in shape)
        ):
            raise ValueError(f"invalid tensor metadata: {key}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(x) is not int for x in offsets)
        ):
            raise ValueError(f"invalid tensor offsets: {key}")
        start, end = offsets
        if start < 0 or end < start or end > size - 8 - length:
            raise ValueError(f"tensor offsets exceed shard: {key}")
        if end - start != math.prod(shape) * _DTYPE_BYTES[dtype]:
            raise ValueError(f"tensor byte count does not match shape: {key}")
        entries[key] = {"dtype": dtype, "shape": shape}
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("safetensors offsets have overlaps or unreferenced gaps")
        cursor = end
    if cursor != size - 8 - length:
        raise ValueError("safetensors shard has trailing unreferenced bytes")
    return entries


def validate_nvfp4_artifact(path):
    """Validate packed storage from headers without importing torch or a GPU."""
    root = Path(path)
    manifest = _json(root / MANIFEST)
    if manifest.get("format") != FORMAT or manifest.get("schema_version") != 1:
        raise ValueError("unsupported packed NVFP4 checkpoint format")
    if manifest.get("kernel") != {"repo_id": KERNEL_REPO, "revision": KERNEL_REVISION}:
        raise ValueError("packed checkpoint requires a different NVFP4 kernel revision")
    config = _json(root / "config.json")
    if config.get("s1_nvfp4_format") != FORMAT or config.get("model_type") != "gemma4_unified":
        raise ValueError("packed checkpoint config must identify Gemma 4 Unified NVFP4")
    modules, tensors, aliases = (manifest.get(key) for key in ("modules", "tensors", "aliases"))
    if not isinstance(modules, dict) or not modules or not isinstance(tensors, dict) or not tensors:
        raise ValueError("packed checkpoint must contain module and tensor inventories")
    if not isinstance(aliases, dict) or set(aliases) & set(tensors):
        raise ValueError("invalid tied tensor aliases")
    if any(source not in tensors for source in aliases.values()):
        raise ValueError("tied tensor alias does not reference a stored tensor")
    if set(manifest.get("shards", {})) != {spec.get("filename") for spec in tensors.values()}:
        raise ValueError("shard inventory does not match tensors")
    actual = {}
    for filename, file_spec in manifest["shards"].items():
        shard = _safe_path(root, filename)
        if not filename.endswith(".safetensors") or shard.stat().st_size != file_spec["size_bytes"]:
            raise ValueError("packed checkpoint shard size mismatch")
        if _sha256(shard) != file_spec["sha256"]:
            raise ValueError(f"packed checkpoint checksum mismatch: {filename}")
        for name, spec in _safetensor_header(shard).items():
            if name in actual:
                raise ValueError(f"duplicate packed checkpoint tensor: {name}")
            actual[name] = {**spec, "filename": filename}
    if actual != tensors:
        raise ValueError("safetensors contents differ from the declared tensor inventory")
    packed_keys = set()
    packed_weight_bytes = 0
    for name, module in modules.items():
        if not name.startswith(_LANGUAGE_PREFIX):
            raise ValueError("only language model modules may be packed in NVFP4")
        inp, out = module.get("in_features"), module.get("out_features")
        if any(type(x) is not int or x < 16 or x % 16 for x in (inp, out)):
            raise ValueError("NVFP4 dimensions must be positive multiples of 16")
        for field, (dtype, shape) in _packed_specs(inp, out).items():
            key = f"{name}.{field}"
            if key not in tensors or any(
                tensors[key].get(k) != v for k, v in {"dtype": dtype, "shape": shape}.items()
            ):
                raise ValueError(f"invalid packed NVFP4 tensor: {key}")
            packed_keys.add(key)
        packed_weight_bytes += out * inp // 2
    for name, spec in tensors.items():
        if (
            name.endswith((".weight_sf", ".weight_sf_rowmajor", ".weight_global_scale"))
            and name not in packed_keys
        ):
            raise ValueError("packed scales exist outside the declared NVFP4 modules")
        if name.endswith(".weight") and spec["dtype"] in ("U8", "I8") and name not in packed_keys:
            raise ValueError("unexpected quantized weight outside the declared NVFP4 modules")
    head_key = aliases.get("lm_head.weight", "lm_head.weight")
    if head_key not in tensors or tensors[head_key]["dtype"] != "BF16":
        raise ValueError("packed checkpoint must preserve the BF16 candidate head")
    return {
        "status": "ok",
        "format": "nvfp4",
        "storage_format": FORMAT,
        "quantized_linear_modules": len(modules),
        "packed_weight_bytes": packed_weight_bytes,
        "tensor_count": len(tensors),
        "tied_tensor_aliases": len(aliases),
    }


def save_nvfp4(lm, path, *, max_shard_size="2GB"):
    """Save packed native kernel tensors, with explicit tied-weight aliases."""
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file
    from transformers.integrations.nvfp4 import NVFP4Linear

    if hasattr(lm, "peft_config"):
        raise ValueError("merge adapters into BF16 before exporting a packed NVFP4 release")
    receipt = inspect_nvfp4(lm)
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.glob("*.safetensors")) or (root / MANIFEST).exists():
        raise ValueError("refusing to overwrite an existing packed checkpoint")
    config = copy.deepcopy(lm.config)
    config.s1_nvfp4_format = FORMAT
    # Preserve an explicit quantization marker so stock loaders fail closed.
    config.quantization_config = {
        "quant_method": "nvfp4",
        "modules_to_not_convert": list(DENSE_MODULES),
    }
    config.save_pretrained(root)
    state = lm.state_dict()
    aliases = {}
    seen = {}
    for name, tensor in list(state.items()):
        signature = (
            str(tensor.device),
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
        )
        if signature in seen and tensor.numel():
            aliases[name] = seen[signature]
            del state[name]
        else:
            seen[signature] = name
    split = split_torch_state_dict_into_shards(state, max_shard_size=max_shard_size)
    tensor_specs, shards = {}, {}
    for filename, keys in split.filename_to_tensors.items():
        shard_state = {name: state[name].detach().cpu().contiguous() for name in keys}
        save_file(shard_state, root / filename, metadata={"format": "pt"})
        for name in keys:
            tensor_specs[name] = {**_tensor_spec(state[name]), "filename": filename}
        shards[filename] = {
            "size_bytes": (root / filename).stat().st_size,
            "sha256": _sha256(root / filename),
        }
        del shard_state
    _write_json(
        root / "model.safetensors.index.json",
        {
            "metadata": split.metadata,
            "weight_map": split.tensor_to_filename,
        },
    )
    manifest = {
        "schema_version": 1,
        "format": FORMAT,
        "kernel": receipt["kernel"],
        "libraries": {
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "kernels")
        },
        "modules": {
            name: {"in_features": module.in_features, "out_features": module.out_features}
            for name, module in lm.named_modules()
            if isinstance(module, NVFP4Linear)
        },
        "tensors": tensor_specs,
        "aliases": aliases,
        "shards": shards,
    }
    _write_json(root / MANIFEST, manifest)
    return validate_nvfp4_artifact(root)


def _load_packed(root, config, *, device, attn_implementation):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import AutoModelForMultimodalLM
    from transformers.integrations.nvfp4 import NVFP4Linear

    validate_nvfp4_artifact(root)
    manifest = _json(root / MANIFEST)
    config = copy.deepcopy(config)
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    kwargs = {"dtype": torch.bfloat16}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    with init_empty_weights():
        lm = AutoModelForMultimodalLM.from_config(config, **kwargs)
    for name, spec in manifest["modules"].items():
        original = lm.get_submodule(name)
        if (
            not isinstance(original, torch.nn.Linear)
            or original.bias is not None
            or (
                original.in_features != spec["in_features"]
                or original.out_features != spec["out_features"]
            )
        ):
            raise ValueError(f"packed module differs from model architecture: {name}")
        with torch.device("meta"):
            replacement = NVFP4Linear(spec["in_features"], spec["out_features"])
            replacement.weight = torch.empty(
                spec["out_features"], spec["in_features"] // 2, dtype=torch.uint8
            )
        parent, _, child = name.rpartition(".")
        setattr(lm.get_submodule(parent), child, replacement)
    expected = lm.state_dict()
    available = set(manifest["tensors"]) | set(manifest["aliases"])
    if set(expected) != available:
        raise ValueError("packed checkpoint tensor keys differ from the model architecture")
    for name, tensor in expected.items():
        source = manifest["aliases"].get(name, name)
        stored = manifest["tensors"][source]
        if _tensor_spec(tensor) != {k: stored[k] for k in ("dtype", "shape")}:
            raise ValueError(f"packed tensor differs from model architecture: {name}")
    for filename in manifest["shards"]:
        weights = load_file(root / filename, device=str(device))
        lm.load_state_dict(weights, strict=False, assign=True)
        del weights
    for alias, source in manifest["aliases"].items():
        parent, _, child = alias.rpartition(".")
        source_parent, _, source_child = source.rpartition(".")
        setattr(
            lm.get_submodule(parent), child, getattr(lm.get_submodule(source_parent), source_child)
        )
    if any(tensor.is_meta for tensor in lm.state_dict().values()):
        raise ValueError("packed checkpoint left unloaded model tensors")
    # Move only initial, nonpersistent buffers (e.g. rotary frequencies), too.
    lm.to(device)
    lm.requires_grad_(False)
    lm.is_quantized = True
    return lm.eval()


def load_nvfp4(
    name, *, revision=None, device="cuda", precision="bfloat16", attn_implementation=None
):
    """Load our packed release, or quantize a BF16 checkpoint during loading."""
    _validate_device(device, precision)
    configure_nvfp4_kernel()
    import torch
    from transformers import AutoConfig, AutoModelForMultimodalLM, NVFP4Config

    config = AutoConfig.from_pretrained(name, revision=revision)
    if config.model_type != "gemma4_unified":
        raise ValueError("NVFP4 decision inference requires Gemma 4 Unified")
    pinned_revision = getattr(config, "_commit_hash", None) or revision
    if getattr(config, "s1_nvfp4_format", None) is not None:
        if config.s1_nvfp4_format != FORMAT:
            raise ValueError("unsupported NVFP4 packed checkpoint format")
        root = Path(name)
        if not root.is_dir():
            from huggingface_hub import snapshot_download

            root = Path(
                snapshot_download(
                    name, revision=pinned_revision, allow_patterns=["*.json", "*.safetensors"]
                )
            )
        lm = _load_packed(root, config, device=device, attn_implementation=attn_implementation)
    else:
        kwargs = {
            "dtype": torch.bfloat16,
            "device_map": {"": str(device)},
            "quantization_config": NVFP4Config(
                modules_to_not_convert=[
                    r"^(?!model\.language_model\.).*",
                    *DENSE_MODULES,
                ]
            ),
        }
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        lm = AutoModelForMultimodalLM.from_pretrained(name, revision=pinned_revision, **kwargs)
    inspect_nvfp4(lm)
    return lm
