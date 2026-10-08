"""Packed release validation must work without GPU/inference dependencies."""

import hashlib
import json
import struct

import pytest

from s1.nvfp4 import (
    FORMAT,
    KERNEL_REPO,
    KERNEL_REVISION,
    MANIFEST,
    _safetensor_header,
    _validate_device,
    validate_nvfp4_artifact,
)


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def packed(tmp_path):
    name = "model.language_model.layers.0.mlp.up_proj"
    specs = {
        name + ".weight": ("U8", [32, 32], 1024),
        name + ".weight_sf": ("I32", [128, 1], 512),
        name + ".weight_sf_rowmajor": ("U8", [32, 4], 128),
        name + ".weight_global_scale": ("F32", [1], 4),
        "model.language_model.embed_tokens.weight": ("BF16", [32, 64], 4096),
    }
    header, inventory, cursor = {}, {}, 0
    for key, (dtype, shape, count) in specs.items():
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + count]}
        inventory[key] = {"dtype": dtype, "shape": shape, "filename": "model.safetensors"}
        cursor += count
    encoded = json.dumps(header).encode()
    blob = struct.pack("<Q", len(encoded)) + encoded + bytes(cursor)
    (tmp_path / "model.safetensors").write_bytes(blob)
    write_json(
        tmp_path / "config.json", {"model_type": "gemma4_unified", "s1_nvfp4_format": FORMAT}
    )
    manifest = {
        "schema_version": 1,
        "format": FORMAT,
        "kernel": {"repo_id": KERNEL_REPO, "revision": KERNEL_REVISION},
        "modules": {name: {"in_features": 64, "out_features": 32}},
        "tensors": inventory,
        "aliases": {"lm_head.weight": "model.language_model.embed_tokens.weight"},
        "shards": {
            "model.safetensors": {
                "size_bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        },
    }
    write_json(tmp_path / MANIFEST, manifest)
    return tmp_path, manifest, name


def test_packed_inventory_and_tied_head(packed):
    path, _, _ = packed
    receipt = validate_nvfp4_artifact(path)
    assert receipt["status"] == "ok"
    assert receipt["quantized_linear_modules"] == 1
    assert receipt["packed_weight_bytes"] == 1024
    assert receipt["tied_tensor_aliases"] == 1


def test_hub_cache_symlinks_can_be_read(packed, tmp_path):
    path, _, _ = packed
    shard = path / "model.safetensors"
    blob = path / "hub-cache-blob"
    shard.rename(blob)
    shard.symlink_to(blob)
    assert validate_nvfp4_artifact(path)["status"] == "ok"


def test_corrupt_payload_rejected_even_with_valid_header(packed):
    path, _, _ = packed
    shard = path / "model.safetensors"
    data = bytearray(shard.read_bytes())
    data[-1] ^= 1
    shard.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        validate_nvfp4_artifact(path)


def test_missing_scale_rejected(packed):
    path, manifest, name = packed
    del manifest["tensors"][name + ".weight_sf_rowmajor"]
    write_json(path / MANIFEST, manifest)
    with pytest.raises(ValueError, match="tensor inventory"):
        validate_nvfp4_artifact(path)


def test_wrong_packed_dimensions_rejected(packed):
    path, manifest, name = packed
    manifest["modules"][name]["in_features"] = 128
    write_json(path / MANIFEST, manifest)
    with pytest.raises(ValueError, match="invalid packed NVFP4 tensor"):
        validate_nvfp4_artifact(path)


def test_dense_shape_cannot_masquerade_as_packed_weight(packed):
    path, manifest, name = packed
    manifest["modules"][name]["in_features"] = 32
    write_json(path / MANIFEST, manifest)
    with pytest.raises(ValueError, match="invalid packed NVFP4 tensor"):
        validate_nvfp4_artifact(path)


def test_media_conversion_rejected(packed):
    path, manifest, name = packed
    manifest["modules"]["model.embed_vision.proj"] = manifest["modules"].pop(name)
    write_json(path / MANIFEST, manifest)
    with pytest.raises(ValueError, match="only language model"):
        validate_nvfp4_artifact(path)


def test_packed_head_rejected(packed):
    path, manifest, name = packed
    manifest["aliases"]["lm_head.weight"] = name + ".weight"
    write_json(path / MANIFEST, manifest)
    with pytest.raises(ValueError, match="BF16 candidate head"):
        validate_nvfp4_artifact(path)


def test_unresolved_alias_rejected(packed):
    path, manifest, _ = packed
    manifest["aliases"]["lm_head.weight"] = "missing"
    write_json(path / MANIFEST, manifest)
    with pytest.raises(ValueError, match="alias"):
        validate_nvfp4_artifact(path)


def test_shard_path_traversal_rejected(packed):
    path, manifest, _ = packed
    for spec in manifest["tensors"].values():
        spec["filename"] = "../model.safetensors"
    manifest["shards"]["../model.safetensors"] = manifest["shards"].pop("model.safetensors")
    write_json(path / MANIFEST, manifest)
    with pytest.raises(ValueError, match="relative filenames"):
        validate_nvfp4_artifact(path)


def test_kernel_revision_must_match(packed):
    path, manifest, _ = packed
    manifest["kernel"]["revision"] = "main"
    write_json(path / MANIFEST, manifest)
    with pytest.raises(ValueError, match="kernel revision"):
        validate_nvfp4_artifact(path)


@pytest.mark.parametrize("device,precision", [("cpu", "bfloat16"), ("cuda", "float32")])
def test_invalid_runtime_rejected_before_importing_torch(device, precision):
    with pytest.raises(ValueError, match="Blackwell CUDA"):
        _validate_device(device, precision)


@pytest.mark.parametrize("offsets", [[0, 2], [1, 2], [0, 100]])
def test_invalid_header_offsets_rejected(tmp_path, offsets):
    header = json.dumps({"weight": {"dtype": "U8", "shape": [1], "data_offsets": offsets}}).encode()
    shard = tmp_path / "invalid.safetensors"
    shard.write_bytes(struct.pack("<Q", len(header)) + header + bytes(2))
    with pytest.raises(ValueError):
        _safetensor_header(shard)
