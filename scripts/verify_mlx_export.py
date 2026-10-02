#!/usr/bin/env python3
"""Experimentally compare a full MLX export with exact HF-prepared S1 inputs.

This is a validation harness, not a production backend. Successful execution
records numerical drift; it does not certify parity, calibration or performance.
MLX is imported only after the manifest/reference checks have passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    return parser.parse_args(argv)


def _positive_temperature(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("temperature must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("temperature must be a positive finite number")
    return float(value)


def _unique_index(items, label):
    result = {}
    for item in items:
        key = item.get("id")
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f"{label} must have unique nonempty ids")
        result[key] = item
    return result


def _distribution(values, size):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError("probabilities have an invalid shape or nonfinite values")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("probabilities must be in [0, 1]")
    if not math.isclose(float(values.sum()), 1.0, abs_tol=1e-5):
        raise ValueError("legal candidate probabilities must sum to one")
    return values


def validate_reference(manifest, reference):
    """Validate alignment before loading large weights or initializing MLX."""
    if reference.get("status") != "ok":
        raise ValueError("reference must be a successful MPS profile")
    if manifest.get("prompt_version") != 1:
        raise ValueError("only S1 prompt_version 1 is supported")
    if not manifest.get("dataset_sha256") or manifest["dataset_sha256"] != reference.get(
        "dataset_sha256"
    ):
        raise ValueError("manifest and reference dataset hashes differ")
    ref_model = reference.get("model", {})
    if not manifest.get("source_model") or manifest["source_model"] != ref_model.get("name"):
        raise ValueError("manifest and reference source models differ")
    revision = manifest.get("source_revision")
    if not revision or revision != ref_model.get("resolved_revision"):
        raise ValueError("manifest and reference source revisions differ")
    if reference.get("pinned_revision") not in (None, revision):
        raise ValueError("reference pinned_revision contradicts resolved_revision")
    temperature = _positive_temperature(manifest.get("temperature"))
    if temperature != _positive_temperature(ref_model.get("temperature")):
        raise ValueError("manifest and reference temperatures differ")
    letters = manifest.get("letters", [])
    if (
        len(letters) != 52
        or len(set(letters)) != 52
        or any(type(token) is not int or token < 0 for token in letters)
    ):
        raise ValueError("manifest must contain 52 distinct candidate token ids")
    cases = _unique_index(manifest.get("cases", []), "manifest cases")
    ref_cases = _unique_index(reference.get("cases", []), "reference cases")
    if not cases or set(cases) != set(ref_cases):
        raise ValueError("manifest and reference case ids differ")
    paired = {}
    for case_id, case in cases.items():
        questions = case.get("questions", [])
        question_map = _unique_index(questions, f"{case_id} questions")
        slots, nopts = case.get("slots", []), case.get("nopts", [])
        if (
            not 1 <= len(questions) <= 64
            or len(slots) != len(questions)
            or len(nopts) != len(questions)
        ):
            raise ValueError(f"{case_id}: slot/option/question counts disagree")
        if any(type(slot) is not int or slot < 0 for slot in slots):
            raise ValueError(f"{case_id}: answer slots must be nonnegative integers")
        if slots != sorted(set(slots)):
            raise ValueError(f"{case_id}: answer slots must be strictly increasing")
        samples = ref_cases[case_id].get("end_to_end_samples", [])
        if not samples:
            raise ValueError(f"{case_id}: reference has no end-to-end samples")
        answers = samples[0].get("answers", {})
        if set(answers) != set(question_map):
            raise ValueError(f"{case_id}: reference question ids differ")
        if ref_cases[case_id].get("question_count") != len(questions):
            raise ValueError(f"{case_id}: reference question count differs")
        distributions = []
        for question, count in zip(questions, nopts):
            labels = question.get("labels", [])
            if type(count) is not int or not 2 <= count <= len(letters):
                raise ValueError(f"{case_id}: invalid option count")
            if (
                len(labels) != count
                or len(set(labels)) != count
                or any(not isinstance(label, str) or not label for label in labels)
            ):
                raise ValueError(f"{case_id}: invalid question labels")
            probabilities = answers[question["id"]].get("probabilities", {})
            if set(probabilities) != set(labels):
                raise ValueError(f"{case_id}: reference candidate labels differ")
            distributions.append(_distribution([probabilities[label] for label in labels], count))
        paired[case_id] = distributions
    return paired


def validate_model_config(config):
    if config.get("model_type") != "gemma4_unified":
        raise ValueError("this experiment requires a complete gemma4_unified export")
    text = config.get("text_config", {})
    if (
        config.get("tie_word_embeddings") is not True
        or text.get("tie_word_embeddings", config.get("tie_word_embeddings")) is not True
    ):
        raise ValueError(
            "untied output heads are unsupported by this selected-embedding experiment"
        )
    if not config.get("vision_config") or not config.get("audio_config"):
        raise ValueError("full multimodal verification requires vision_config and audio_config")
    softcap = text.get("final_logit_softcapping")
    if softcap is not None and (
        isinstance(softcap, bool)
        or not isinstance(softcap, (int, float))
        or not math.isfinite(softcap)
        or softcap <= 0
    ):
        raise ValueError("invalid final logit softcap")
    return softcap


def load_case_arrays(directory, case):
    file_path = (directory / case["file"]).resolve()
    if not file_path.is_relative_to(directory.resolve()):
        raise ValueError("case tensor file must be within the input directory")
    with np.load(file_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    allowed = {
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "token_type_ids",
        "pixel_values",
        "image_position_ids",
        "input_features",
        "input_features_mask",
        "pixel_values_videos",
        "video_position_ids",
    }
    if set(arrays) - allowed:
        raise ValueError(f"unsupported prepared tensors: {sorted(set(arrays) - allowed)}")
    ids = arrays.get("input_ids")
    if ids is None or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
        raise ValueError("only batch size 1 with nonempty prepared input_ids is supported")
    if ids.dtype.kind not in "iu" or (ids < 0).any():
        raise ValueError("input_ids must contain nonnegative integers")
    attention = arrays.get("attention_mask")
    if attention is None or attention.shape != ids.shape or not np.all(attention == 1):
        raise ValueError("this experiment requires an all-ones, unpadded attention_mask")
    if max(case["slots"]) >= ids.shape[1]:
        raise ValueError("an answer slot is outside the prepared input sequence")
    for name, value in arrays.items():
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError(f"{name}: tensor values must be finite numeric values")
    mm_types = arrays.get("mm_token_type_ids", arrays.get("token_type_ids"))
    if mm_types is not None and (mm_types.shape != ids.shape or mm_types.dtype.kind not in "iu"):
        raise ValueError("modality token types must be integer values matching input_ids")
    if any(key in arrays for key in ("pixel_values", "input_features", "pixel_values_videos")):
        if mm_types is None:
            raise ValueError("multimodal inputs require explicit modality token types")
    return arrays, hashlib.sha256(file_path.read_bytes()).hexdigest()


def selected_probabilities(model, arrays, case, letters, temperature, softcap):
    import mlx.core as mx

    inputs = {name: mx.array(value) for name, value in arrays.items() if name != "attention_mask"}
    # The outer model wrapper drops mm_token_type_ids. Pass it directly to the
    # normalized backbone, preserving full/sliding and bidirectional vision masks.
    embedded = model.get_input_embeddings(**inputs)
    hidden = model.language_model.model(
        inputs=inputs["input_ids"],
        inputs_embeds=embedded.inputs_embeds,
        per_layer_inputs=embedded.per_layer_inputs,
        cache=None,
        mask=None,
        mm_token_type_ids=inputs.get("mm_token_type_ids", inputs.get("token_type_ids")),
    )
    hidden = hidden[0, mx.array(case["slots"], dtype=mx.int32)]
    # QuantizedEmbedding.__call__ dequantizes only these rows with their scales
    # and quantization offsets. It does not materialize the vocabulary projection.
    head_rows = model.language_model.model.embed_tokens(mx.array(letters, dtype=mx.int32))
    logits = hidden.astype(head_rows.dtype) @ head_rows.T
    if softcap is not None:
        logits = mx.tanh(logits / softcap) * softcap
    logits = logits.astype(mx.float32)
    legal = mx.arange(len(letters))[None, :] < mx.array(case["nopts"], dtype=mx.int32)[:, None]
    probabilities = mx.softmax(mx.where(legal, logits / temperature, -mx.inf), axis=-1)
    mx.eval(probabilities)
    result = np.asarray(probabilities)
    return [_distribution(row[:count], count) for row, count in zip(result, case["nopts"])]


def _library_versions():
    result = {"python": platform.python_version()}
    for package in ("mlx", "mlx-metal", "mlx-vlm", "transformers", "huggingface-hub", "numpy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def main(argv=None):
    args = parse_args(argv)
    report = {
        "schema_version": 1,
        "experimental": True,
        "status": "error",
        "parity_accepted": False,
        "scope": "numerical measurements only; no production, calibration or performance claim",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model.resolve()),
        "reference_path": str(args.reference.resolve()),
        "library_versions": _library_versions(),
        "reference_strategy": "first end_to_end_samples entry, matched by case/question/label ids",
        "head_projection": "dequantize selected tied embedding rows; native softcap once",
        "measurement": {
            "warmup_per_case": args.warmup,
            "repeat_per_case": args.repeat,
            "scope": "prepared tensors to validated CPU probabilities, synchronized MLX calls",
            "includes": "array transfer, backbone, selected-row projection, softmax and CPU copy",
            "excludes": "HF preprocessing, file I/O and model loading; not end-to-end request latency",
        },
        "cases": [],
    }
    stage = "preflight"
    try:
        if args.warmup < 0 or args.repeat < 1:
            raise ValueError("warmup must be nonnegative and repeat must be positive")
        manifest_path = args.inputs / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        reference = json.loads(args.reference.read_text())
        paired = validate_reference(manifest, reference)
        config_path = args.model / "config.json"
        config = json.loads(config_path.read_text())
        softcap = validate_model_config(config)
        report.update(
            {
                "source_model": manifest["source_model"],
                "source_revision": manifest["source_revision"],
                "dataset_sha256": manifest["dataset_sha256"],
                "temperature": manifest["temperature"],
                "prompt_version": manifest["prompt_version"],
                "quantization": config.get("quantization"),
                "quantization_config": config.get("quantization_config"),
                "softcap": softcap,
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            }
        )
        sidecar = args.model / "s1_config.json"
        report["s1_config"] = json.loads(sidecar.read_text()) if sidecar.exists() else None
        if report["s1_config"] is not None:
            for key in ("source_model", "source_revision", "prompt_version", "temperature"):
                if report["s1_config"].get(key) != manifest[key]:
                    raise ValueError(f"export s1_config.json {key} differs from captured inputs")
        prepared = [(case, *load_case_arrays(args.inputs, case)) for case in manifest["cases"]]
        for case, arrays, _ in prepared:
            ref_case = next(item for item in reference["cases"] if item["id"] == case["id"])
            if arrays["input_ids"].shape[1] != ref_case.get("input_tokens"):
                raise ValueError(f"{case['id']}: captured and reference input token counts differ")
        stage = "load_mlx_model"
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_vlm import load

        model, processor = load(str(args.model), strict=True)
        tokenizer = processor.tokenizer
        boundary = tokenizer.encode("\nAnswer: (", add_special_tokens=False)
        for letter, token_id in zip(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", manifest["letters"]
        ):
            if tokenizer.encode(letter, add_special_tokens=False) != [token_id] or tokenizer.encode(
                "\nAnswer: (" + letter, add_special_tokens=False
            ) != boundary + [token_id]:
                raise ValueError("export tokenizer candidate ids differ at the answer boundary")
        parameter_names = [name for name, _ in tree_flatten(model.parameters())]
        inventory = {
            component: [name for name in parameter_names if name.startswith(component + ".")]
            for component in ("vision_embedder", "embed_vision", "embed_audio")
        }
        report["modality_weight_inventory"] = inventory
        report["loaded_parameter_tensor_count"] = len(parameter_names)
        if any(not names for names in inventory.values()):
            raise ValueError("export is missing a required multimodal component's weights")
        if getattr(model.language_model, "lm_head", None) is not None:
            raise ValueError(
                "unexpected separate output head; refusing to substitute tied embeddings"
            )
        stage = "verify_cases"
        all_drifts, all_agreements = [], []
        for case, arrays, digest in prepared:
            result = {
                "id": case["id"],
                "question_count": len(case["questions"]),
                "input_tokens": int(arrays["input_ids"].shape[1]),
                "tensor_file_sha256": digest,
                "tensor_shapes": {name: list(value.shape) for name, value in arrays.items()},
                "tensor_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
                "status": "error",
            }
            try:
                for _ in range(args.warmup):
                    selected_probabilities(
                        model, arrays, case, manifest["letters"], manifest["temperature"], softcap
                    )
                samples = []
                probabilities = None
                repeat_drift = 0.0
                for _ in range(args.repeat):
                    mx.synchronize()
                    started = time.perf_counter()
                    measured = selected_probabilities(
                        model, arrays, case, manifest["letters"], manifest["temperature"], softcap
                    )
                    mx.synchronize()
                    samples.append((time.perf_counter() - started) * 1000)
                    if probabilities is None:
                        probabilities = measured
                    else:
                        repeat_drift = max(
                            repeat_drift,
                            max(
                                float(np.abs(a - b).max()) for a, b in zip(probabilities, measured)
                            ),
                        )
                answers, comparisons = {}, []
                for question, values, baseline in zip(
                    case["questions"], probabilities, paired[case["id"]]
                ):
                    drift = float(np.abs(values - baseline).max())
                    best, ref_best = int(np.argmax(values)), int(np.argmax(baseline))
                    agree = best == ref_best
                    answers[question["id"]] = {
                        "probabilities": dict(zip(question["labels"], values.tolist())),
                        "argmax_label": question["labels"][best],
                    }
                    comparisons.append(
                        {
                            "id": question["id"],
                            "max_abs_probability_drift": drift,
                            "argmax_agreement": agree,
                            "reference_argmax_label": question["labels"][ref_best],
                        }
                    )
                    all_drifts.append(drift)
                    all_agreements.append(agree)
                result.update(
                    {
                        "status": "ok",
                        "finite_probabilities": True,
                        "answers": answers,
                        "comparisons": comparisons,
                        "prepared_forward_samples_ms": samples,
                        "prepared_forward_p50_ms": float(np.median(samples)),
                        "max_repeat_probability_drift": repeat_drift,
                        "max_abs_probability_drift": max(
                            c["max_abs_probability_drift"] for c in comparisons
                        ),
                        "argmax_agreement_count": sum(c["argmax_agreement"] for c in comparisons),
                    }
                )
            except Exception as exc:
                result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            report["cases"].append(result)
            print(f"{case['id']}: {result['status']}", file=sys.stderr, flush=True)
        success_count = sum(case["status"] == "ok" for case in report["cases"])
        report["summary"] = {
            "case_count": len(prepared),
            "successful_case_count": success_count,
            "compared_question_count": len(all_agreements),
            "max_abs_probability_drift": max(all_drifts) if all_drifts else None,
            "argmax_agreement_count": sum(all_agreements),
            "argmax_agreement_fraction": float(np.mean(all_agreements)) if all_agreements else None,
            "acceptance_threshold": None,
        }
        if success_count == len(prepared):
            report["status"] = "ok"
    except Exception as exc:
        report["failure"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['status']}: {args.output}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
