#!/usr/bin/env python3
"""Verify and record a complete Gemma 4 Unified Q8_0 + F16 projector export."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from prepare_mlx_validation import checkpoint_identity, file_sha256

LLAMA_REVISION = "5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd"
MODEL_FILE = "s1-boolq-Q8_0.gguf"
MMPROJ_FILE = "mmproj-s1-boolq-f16.gguf"
SIDECARS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
)


def git_revision(directory):
    return subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "-C", str(directory), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--source-capture-sha256", required=True)
    parser.add_argument("--model-file", type=str, default=MODEL_FILE)
    parser.add_argument("--mmproj-file", type=str, default=MMPROJ_FILE)
    parser.add_argument(
        "--language-quant",
        type=str,
        default="Q8_0",
        help="Language-model quantization label (Q8_0, Q6_K, Q5_K_M, Q4_K_M, F16, ...). "
        "Recorded verbatim; structural checks beyond tensor count are only enforced for Q8_0.",
    )
    parser.add_argument("--projector-quant", type=str, default="F16")
    parser.add_argument(
        "--direct-conversion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="True for direct HF->target conversion; False for F16-intermediate + llama-quantize "
        "(required for K-quants; prefer F16 source over Q8 requantize).",
    )
    args = parser.parse_args(argv)
    if git_revision(args.llama_cpp) != LLAMA_REVISION:
        raise ValueError("llama.cpp must be at the pinned conversion/runtime revision")
    subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-C",
            str(args.llama_cpp),
            "diff",
            "--quiet",
            "HEAD",
            "--",
        ],
        check=True,
    )
    config = json.loads((args.source / "config.json").read_text())
    if config.get("architectures") != ["Gemma4UnifiedForConditionalGeneration"]:
        raise ValueError("source must be the complete Gemma 4 Unified checkpoint")
    sidecar = json.loads((args.source / "s1_config.json").read_text())
    source_identity = checkpoint_identity(args.source)
    # Hash the original training artifact again; the export must derive from the
    # exact merged weights recorded by both the CUDA and MLX evaluations.
    if file_sha256(args.source_capture) != args.source_capture_sha256:
        raise ValueError("source capture manifest SHA256 differs from the pinned receipt")
    capture = json.loads(args.source_capture.read_text())
    if source_identity["sha256"] != capture["source_checkpoint_sha256"]:
        raise ValueError("source weights/config differ from the evaluated merged checkpoint")
    source_sidecar_hash = file_sha256(args.source / "s1_config.json")
    if source_sidecar_hash != capture["source_s1_config_sha256"]:
        raise ValueError("source calibration sidecar differs from the original input captures")
    for name in SIDECARS:
        shutil.copyfile(args.source / name, args.model / name)
    shutil.copyfile(args.source / "s1_config.json", args.model / "base_s1_config.json")

    sys.path.insert(0, str(args.llama_cpp / "gguf-py"))
    import numpy as np
    from gguf import GGUFReader
    from safetensors import safe_open

    with safe_open(args.source / "model.safetensors", framework="np") as source:
        source_keys = list(source.keys())
    source_counts = Counter(".".join(key.split(".")[:2]) for key in source_keys)
    inventory = {}
    model_file, mmproj_file = args.model_file, args.mmproj_file
    for filename in (model_file, mmproj_file):
        path = args.model / filename
        reader = GGUFReader(path)
        metadata = {
            key: value.contents()
            for key, value in reader.fields.items()
            if not key.startswith("tokenizer.")
        }
        tensors = [
            {
                "name": tensor.name,
                "shape": tensor.shape.tolist(),
                "type": int(tensor.tensor_type),
                "n_bytes": int(tensor.n_bytes),
            }
            for tensor in reader.tensors
        ]
        inventory[filename] = {"metadata": metadata, "tensors": tensors}
        if filename == model_file:
            if (
                metadata["general.architecture"] != "gemma4"
                or metadata["gemma4.block_count"] != config["text_config"]["num_hidden_layers"]
                or len(tensors) != source_counts["model.language_model"] + 1
            ):
                raise ValueError("GGUF language-model architecture/tensor count differs")
            if args.language_quant == "Q8_0" and metadata["general.file_type"] != 7:
                raise ValueError("Q8_0 language model must have GGUF file_type 7")
            if args.language_quant != "Q8_0":
                # K-quants / F16: file_type varies by target; tensor count + RoPE
                # checks above remain enforced, file_type is recorded verbatim.
                pass
            rope = [tensor for tensor in reader.tensors if tensor.name == "rope_freqs.weight"]
            if (
                len(rope) != 1
                or rope[0].data.size != 256
                or not np.all(rope[0].data.ravel()[:64] == 1)
                or not np.all(rope[0].data.ravel()[64:] >= 1e29)
            ):
                raise ValueError("Gemma 4 Unified proportional RoPE factors were not retained")
        elif (
            metadata.get("clip.has_vision_encoder") is not True
            or metadata.get("clip.has_audio_encoder") is not True
            or metadata.get("clip.vision.projector_type") != "gemma4uv"
            or metadata.get("clip.audio.projector_type") != "gemma4ua"
            or len(tensors) != len(source_keys) - source_counts["model.language_model"]
        ):
            raise ValueError("projector must retain both Unified vision and audio tensors")
    (args.model / "tensor-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    files = [
        {
            "path": name,
            "size_bytes": (args.model / name).stat().st_size,
            "sha256": file_sha256(args.model / name),
        }
        for name in sorted(
            (*SIDECARS, model_file, mmproj_file, "base_s1_config.json", "tensor-inventory.json")
        )
    ]
    receipt = {
        "schema_version": 1,
        "format": "gguf",
        "status": "converted_and_structurally_verified",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source_model": sidecar["source_model"],
        "source_revision": sidecar["source_revision"],
        "source_checkpoint_sha256": source_identity["sha256"],
        "source_checkpoint_files": source_identity["files"],
        "source_s1_config_sha256": source_sidecar_hash,
        "training_recipe_sha256": sidecar["training_recipe_sha256"],
        "source_hub_repository": "knowlet/Gemma-4-12B-Unified-System-One",
        "source_hub_weights_revision": "66626de5cbcf8c5fecb8e9b58710805a02a42f67",
        "llama_cpp_revision": LLAMA_REVISION,
        "conversion_record_source_revision": git_revision(Path(__file__).resolve().parents[1]),
        "recording_program_sha256": file_sha256(Path(__file__)),
        "source_capture_manifest_sha256": args.source_capture_sha256,
        "model_file": model_file,
        "mmproj_file": mmproj_file,
        "quantization": {
            "language_model": args.language_quant,
            "multimodal_projector": args.projector_quant,
            "direct_conversion": args.direct_conversion,
        },
        "source_tensor_count": len(source_keys),
        "source_tensor_groups": dict(source_counts),
        "files": files,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "safetensors")
        },
        "commands": (
            [
                ".venv/bin/python",
                str(args.llama_cpp / "convert_hf_to_gguf.py"),
                str(args.source),
                "--outtype",
                "q8_0",
                "--outfile",
                str(args.model / model_file),
                "--model-name",
                "Gemma-4-12B-Unified-System-One",
            ],
            [
                ".venv/bin/python",
                str(args.llama_cpp / "convert_hf_to_gguf.py"),
                str(args.source),
                "--mmproj",
                "--outtype",
                "f16",
                "--outfile",
                str(args.model / mmproj_file),
            ],
        )
        if args.direct_conversion and args.language_quant == "Q8_0"
        else [
            f"convert {args.source} to F16 intermediate, then llama-quantize to "
            f"{args.language_quant} (direct_conversion=False); see quant-sweep plan; "
            f"prefer F16 source over Q8 requantize",
            f"mmproj {args.projector_quant} -> {mmproj_file}",
        ],
        "scope": "Structural conversion receipt; runtime quality requires the separate full-population evaluation.",
    }
    path = args.model / "conversion-manifest.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "manifest_sha256": file_sha256(path),
                "weight_bytes": sum(
                    item["size_bytes"] for item in files if item["path"].endswith(".gguf")
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
