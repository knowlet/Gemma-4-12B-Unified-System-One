#!/usr/bin/env python3
"""Persistent S1 GGUF predictions, or gold-free raw-logit release evaluation.

HF constructs the authoritative prompt and media inputs on CPU. llama.cpp and
mtmd execute the language model and both media projections. No answer tokens are
generated; all question slots belong to the same causal input sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s1.contracts import DecisionRequest, answer_from_probabilities
from s1.unified import UnifiedDecisionModel, candidate_ids


def processed_image_rgb(patches, positions):
    """Invert HF's patch layout, retaining its exact resized uint8 pixels.

    Feeding this already-aligned grid to mtmd disables any native resize. The
    round-trip check rejects altered normalization or non-uint8 source inputs.
    Padding patches are omitted, and their learned positions are never used.
    """
    patches = np.asarray(patches, dtype=np.float32)
    positions = np.asarray(positions)
    if patches.ndim != 2 or positions.shape != (patches.shape[0], 2):
        raise ValueError("invalid processed image patch/position shape")
    valid = (positions >= 0).all(axis=1)
    if not valid.any() or not np.all(valid | (positions == -1).all(axis=1)):
        raise ValueError("invalid image padding positions")
    patches, positions = patches[valid], positions[valid]
    patch = math.isqrt(patches.shape[1] // 3)
    if patch != 48 or patches.shape[1] != patch * patch * 3:
        raise ValueError("native Gemma4Unified requires 48 by 48 RGB model patches")
    if not np.array_equal(positions, positions.astype(np.int64)):
        raise ValueError("image positions must be integers")
    cols, rows = (positions.max(axis=0) + 1).astype(int)
    expected = np.array([(x, y) for y in range(rows) for x in range(cols)])
    if not np.array_equal(positions, expected):
        raise ValueError("native image positions require a complete row-major grid")
    scaled = patches * np.float32(255)
    if not np.isfinite(scaled).all() or scaled.min() < -0.0001 or scaled.max() > 255.0001:
        raise ValueError("processed image values are outside the uint8 rescale contract")
    pixels = np.rint(scaled).clip(0, 255).astype(np.uint8)
    restored = pixels.astype(np.float32) * np.float32(1 / 255)
    if not np.allclose(restored, patches, rtol=0, atol=1e-7):
        raise ValueError("processed image is not an exact uint8 rescale round-trip")
    return (
        pixels.reshape(rows, cols, patch, patch, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(rows * patch, cols * patch, 3)
    )


def make_native_payload(envelope, request, inputs, slots, nopts, letters, directory):
    """Validate the captured identity and retain HF text/control tokens verbatim."""
    ids = inputs["input_ids"][0].tolist()
    slots, nopts = list(slots), list(nopts)
    question_ids = [question.id for question in request.questions]
    actual = {
        "question_ids": question_ids,
        "slots": slots,
        "nopts": nopts,
        "letters": list(letters),
        "input_ids": ids,
        "has_media": bool(request.media),
        "source_input_token_count": len(ids),
    }
    if envelope is not None:
        for key, value in actual.items():
            if key not in envelope or envelope[key] != value:
                raise ValueError(f"captured {key} differs from authoritative HF preparation")
        if "gold" in envelope:
            raise ValueError("evaluation envelope must not contain gold labels")
    mask = inputs.get("mm_token_type_ids")
    types = mask[0].tolist() if mask is not None else [0] * len(ids)
    if len(types) != len(ids) or any(value not in (0, 1, 3) for value in types):
        raise ValueError("unsupported multimodal token layout")
    parts, cursor, media_index, image_index = [], 0, 0, 0
    while cursor < len(ids):
        kind = types[cursor]
        end = cursor + 1
        while end < len(ids) and types[end] == kind:
            end += 1
        if kind == 0:
            parts.append({"type": "text", "tokens": ids[cursor:end]})
        else:
            if media_index >= len(request.media):
                raise ValueError("multimodal token layout exceeds media count")
            media = request.media[media_index]
            if (kind == 1 and media.type != "image") or (kind == 3 and media.type != "audio"):
                raise ValueError("multimodal token order differs from request media order")
            if kind == 1:
                rgb = processed_image_rgb(
                    inputs["pixel_values"][image_index].numpy(),
                    inputs["image_position_ids"][image_index].numpy(),
                )
                path = Path(directory) / f"image-{image_index}.rgb"
                path.write_bytes(rgb.tobytes())
                parts.append(
                    {
                        "type": "image",
                        "rgb_path": str(path.resolve()),
                        "width": rgb.shape[1],
                        "height": rgb.shape[0],
                        "expected_tokens": end - cursor,
                    }
                )
                image_index += 1
            else:
                # Match HF float32 input conversion and native 640-sample frame padding.
                parts.append(
                    {
                        "type": "audio",
                        "samples": np.asarray(media.samples, dtype=np.float32).tolist(),
                        "expected_tokens": end - cursor,
                    }
                )
            media_index += 1
        cursor = end
    if media_index != len(request.media):
        raise ValueError("media item has no authoritative placeholder tokens")
    return {
        "case_id": envelope["case_id"] if envelope is not None else "prediction",
        "question_ids": question_ids,
        "slots": slots,
        "nopts": nopts,
        "letters": list(letters),
        "source_input_token_count": len(ids),
        "parts": parts,
    }


def load_calibration(path, *, model=None, mmproj=None):
    # Predictions require a concrete GGUF calibration sidecar, never a silent T=1.
    config = json.loads(Path(path).read_text())
    temperature = config.get("temperature") if isinstance(config, dict) else None
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("GGUF s1_config.json requires a positive finite temperature")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("GGUF s1_config.json requires a positive finite temperature")
    if config.get("runtime") != "llama.cpp" or config.get("runtime_format") != "gguf":
        raise ValueError(
            "prediction requires GGUF-owned llama.cpp calibration, not source-model calibration"
        )
    package = Path(model).parent if model is not None else Path(path).parent
    manifest_bytes = (package / "conversion-manifest.json").read_bytes()
    if config.get("conversion_manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("GGUF calibration conversion manifest digest mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "converted_and_structurally_verified":
        raise ValueError("GGUF conversion manifest is not structurally verified")
    files = {item["path"]: item for item in manifest.get("files", [])}
    for key, selected in (("model_file", model), ("mmproj_file", mmproj)):
        name = manifest.get(key)
        if not isinstance(name, str) or Path(name).name != name or name not in files:
            raise ValueError(f"GGUF conversion manifest lacks a local {key} identity")
        expected = package / name
        if selected is not None and Path(selected).resolve() != expected.resolve():
            raise ValueError(f"selected {key} differs from calibrated conversion manifest")
        if expected.stat().st_size != files[name].get("size_bytes"):
            raise ValueError(f"calibrated {key} size differs from conversion manifest")
    # Full tensor-file hashes are checked by the release evaluator/publication
    # receipt. Startup binds calibration and selected files without rereading 12 GB.
    return float(temperature)


def validate_processor_identity(processor_path, package):
    """Bind preprocessing and the prompt to the same converted/calibrated package."""
    manifest = json.loads((Path(package) / "conversion-manifest.json").read_text())
    files = {item["path"]: item for item in manifest.get("files", [])}
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "processor_config.json",
    ):
        if name not in files:
            raise ValueError(f"conversion manifest lacks processor identity for {name}")
        content = (Path(processor_path) / name).read_bytes()
        if len(content) != files[name].get("size_bytes") or hashlib.sha256(
            content
        ).hexdigest() != files[name].get("sha256"):
            raise ValueError(
                f"selected processor {name} differs from calibrated conversion manifest"
            )


def prediction_from_logits(request, response, temperature, model):
    if response.get("status") != "ok":
        return response
    rows = response["raw_logits"]
    if len(rows) != len(request.questions):
        raise ValueError("native response omitted questions")
    answers = {}
    for question, row in zip(request.questions, rows):
        logits = np.asarray(row, dtype=np.float64)
        if logits.shape != (len(question.labels()),) or not np.isfinite(logits).all():
            raise ValueError("invalid legal native logits")
        logits = logits / temperature
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        answers[question.id] = answer_from_probabilities(question, probabilities)
    return {
        "status": "ok",
        "answers": answers,
        "model": str(model),
        "temperature": temperature,
        "passes": 1,
        "latency_ms": response["latency_ms"],
        "runtime": response["runtime"],
        "runtime_slots": response["runtime_slots"],
        "native_token_count": response["native_token_count"],
        "preprocessing": response["preprocessing"],
    }


class NativeProcess:
    def __init__(self, argv):
        self.process = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )

    def score(self, payload):
        if self.process.poll() is not None:
            raise RuntimeError(f"native GGUF process exited with code {self.process.returncode}")
        self.process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("native GGUF process closed without a response; inspect stderr")
        response = json.loads(line)
        if response.get("case_id") != payload["case_id"]:
            raise ValueError("native response case identity mismatch")
        if response.get("status") == "ok":
            for key in ("question_ids", "slots"):
                if response.get(key) != payload[key]:
                    raise ValueError(f"native response {key} mismatch")
        return response

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="GGUF package directory or text GGUF file"
    )
    parser.add_argument(
        "--mmproj", type=Path, help="Defaults to the sole mmproj GGUF beside the model"
    )
    parser.add_argument(
        "--processor", type=Path, help="Local saved HF processor; defaults beside the model"
    )
    parser.add_argument("--native-runner", type=Path, required=True)
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="Read capture envelopes; emit raw logits without T",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Default: read DecisionRequest; emit typed calibrated answers",
    )
    parser.add_argument(
        "--s1-config", type=Path, help="GGUF calibration sidecar; defaults beside --model"
    )
    parser.add_argument("--ctx-size", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    if args.evaluation and args.predict:
        parser.error("choose prediction or evaluation mode")
    package = args.model if args.model.is_dir() else args.model.parent
    if args.model.is_dir():
        files = [path for path in args.model.glob("*.gguf") if not path.name.startswith("mmproj")]
        if len(files) != 1:
            parser.error("model directory must contain exactly one text GGUF")
        args.model = files[0]
    if args.mmproj is None:
        files = list(package.glob("mmproj*.gguf"))
        if len(files) != 1:
            parser.error("provide --mmproj when the package has no unique mmproj GGUF")
        args.mmproj = files[0]
    args.processor = args.processor or package
    temperature = (
        None
        if args.evaluation
        else load_calibration(
            args.s1_config or args.model.parent / "s1_config.json",
            model=args.model,
            mmproj=args.mmproj,
        )
    )
    validate_processor_identity(args.processor, package)
    from transformers import AutoProcessor

    config = json.loads((args.processor / "config.json").read_text())
    if config.get("model_type") != "gemma4_unified":
        raise ValueError("processor source must be a Gemma 4 Unified checkpoint")
    if config.get("text_config", {}).get("hidden_size_per_layer_input", 0) != 0:
        raise ValueError("GGUF S1 adapter does not support per-layer input embeddings")
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    context = SimpleNamespace(
        processor=processor, tok=processor.tokenizer, device="cpu", max_context=args.ctx_size
    )
    letters = candidate_ids(processor.tokenizer)
    native = NativeProcess(
        [
            str(args.native_runner.resolve()),
            "--model",
            str(args.model.resolve()),
            "--mmproj",
            str(args.mmproj.resolve()),
            "--ctx-size",
            str(args.ctx_size),
            "--batch-size",
            str(args.batch_size),
            "--gpu-layers",
            str(args.gpu_layers),
            "--threads",
            str(args.threads),
            "--image-max-tokens",
            str(processor.image_processor.max_soft_tokens),
        ]
    )
    try:
        for line in sys.stdin:
            item, request = None, None
            started = time.perf_counter()
            try:
                item = json.loads(line)
                envelope = item if args.evaluation else None
                request = DecisionRequest.model_validate(
                    item["request"] if args.evaluation else item
                )
                inputs, slots, nopts = UnifiedDecisionModel.prepare(context, request)
                with tempfile.TemporaryDirectory(prefix="s1-gguf-") as directory:
                    payload = make_native_payload(
                        envelope,
                        request,
                        inputs,
                        slots.tolist(),
                        nopts.tolist(),
                        letters,
                        directory,
                    )
                    prepared = time.perf_counter()
                    response = native.score(payload)
                if response.get("status") == "ok":
                    response["preprocessing"]["hf_prepare_ms"] = (prepared - started) * 1000
                    response["native_latency_ms"] = response["latency_ms"]
                    response["latency_ms"] = (time.perf_counter() - started) * 1000
                    if not args.evaluation:
                        response = prediction_from_logits(
                            request, response, temperature, args.model
                        )
            except Exception as error:
                response = {
                    "status": "error",
                    "case_id": item.get("case_id") if isinstance(item, dict) else None,
                    "error": f"{type(error).__name__}: {error}",
                }
            print(json.dumps(response, allow_nan=False), flush=True)
    finally:
        native.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
