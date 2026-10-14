#!/usr/bin/env python3
"""Fit text-attention LoRA on audited development-browser action demonstrations.

The native full-vocabulary LM predicts only the action JSON and native turn end.
Teacher-forced losses are fitting diagnostics, never browser task success.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import jev_browser_fit_readout as integrity

MODEL_ID = "knowlet/Gemma-4-12B-Unified-System-One"
REVISION = "a66f836b56605039fe040f330180e336d19b3362"
IDENTITY = "generated-actions-nf4"
LORA = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": r"model\.language_model\.layers\.[0-9]+\.self_attn\.(q_proj|v_proj)",
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
SEQUENCE_KEYS = {
    "input_ids",
    "attention_mask",
    "mm_token_type_ids",
    "token_type_ids",
    "position_ids",
}
TRAINABLE_NAME = re.compile(
    r"base_model\.model\.model\.language_model\.layers\.[0-9]+\.self_attn\.(q_proj|v_proj)\.lora_[AB]\.default\.weight\Z"
)


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _tensor_sha(tensor):
    """Hash exact tensor bytes with bounded CPU copies, including BF16 tensors."""
    import torch

    digest = hashlib.sha256()
    flat = tensor.detach().flatten()
    for offset in range(0, flat.numel(), 1_000_000):
        chunk = flat[offset : offset + 1_000_000].cpu().contiguous().view(torch.uint8)
        digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


def _resources(model):
    import resource

    import torch

    result = {"process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if str(model.device).startswith("cuda"):
        result.update(
            cuda_allocated_bytes=torch.cuda.memory_allocated(model.device),
            cuda_reserved_bytes=torch.cuda.memory_reserved(model.device),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(model.device),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(model.device),
        )
    return result


def compile_actions(model, cases, snapshot, *, identity=IDENTITY):
    """Compile the exact inference prefix; gold never enters that prefix.

    Completed-assistant chat templates can differ from the native generation
    prefix. Append to the actual inference prompt and verify every token/mask
    prefix instead of assuming those templates are interchangeable.
    """
    import torch

    from s1.evaluation.datasets import request_fingerprint
    from s1.jev_generated import INSTRUCTIONS, POLICY, TARGETS, _request

    wires = [
        integrity._json(line)
        for line in snapshot["blobs"]["wire_requests.jsonl"].splitlines()
        if line.strip()
    ]
    stop = model.tok.convert_tokens_to_ids("<turn|>")
    stops = model.lm.generation_config.eos_token_id
    stops = [stops] if isinstance(stops, int) else stops
    if (
        type(stop) is not int
        or stop == getattr(model.tok, "unk_token_id", None)
        or stop not in (stops or [])
    ):
        raise ValueError("Native <turn|> must be an actual generation stop token")
    rows = []
    for case, wire in zip(cases, wires, strict=True):
        if wire["case_id"] != case.id:
            raise ValueError("Action wire order differs from validated cases")
        original = wire["payload"]
        payload = copy.deepcopy(original)
        payload["model"] = identity
        encoded, source, labels = _request(payload, identity)
        action = copy.deepcopy(wire["teacher_choice"])
        operation = case.gold["operation"]
        head = TARGETS.get(operation)
        if (
            set(action) != {"operation", "target"}
            or action["operation"] != operation
            or operation not in labels["operation"]
            or operation not in wire["valid_next_targets"]
            or (
                head
                and (
                    action["target"] not in labels[head]
                    or action["target"] not in wire["valid_next_targets"][operation]
                    or (head in case.gold and case.gold[head] != action["target"])
                    or (head not in case.gold and len(labels[head]) != 1)
                )
            )
            or (not head and action["target"] is not None)
        ):
            raise ValueError("Audited teacher action differs from the active gold head")
        prompt = model.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": INSTRUCTIONS + "\n" + encoded}],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        target = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
        prefix = dict(model.processor(text=prompt, return_tensors="pt", add_special_tokens=False))
        inputs = dict(
            model.processor(
                text=prompt + target + "<turn|>", return_tensors="pt", add_special_tokens=False
            )
        )
        if (
            set(inputs) != set(prefix)
            or not {"input_ids", "attention_mask"} <= set(inputs)
            or set(inputs) - SEQUENCE_KEYS
        ):
            raise ValueError("Action fitting supports only unchanged text sequence tensors")
        p = prefix["input_ids"].shape[-1]
        length = inputs["input_ids"].shape[-1]
        for key, tensor in inputs.items():
            before = prefix[key]
            if (
                not torch.is_tensor(tensor)
                or tensor.ndim != 2
                or tuple(tensor.shape) != (1, length)
                or before.ndim != 2
                or tuple(before.shape) != (1, p)
                or tensor.device.type != "cpu"
                or before.device.type != "cpu"
                or not torch.equal(tensor[:, :p], before)
            ):
                raise ValueError("Native prompt token/mask prefix changed at the action boundary")
            if tensor.dtype not in (torch.int64, torch.int32, torch.bool):
                raise ValueError("Native sequence tensors must contain integer token/mask values")
        ids = inputs["input_ids"][0]
        count = length - p
        if p < 1 or not 1 <= count <= 32 or p + 32 > min(model.max_context, 8192):
            raise ValueError("Action context/target exceeds native budget; nothing truncated")
        if ids[-1].item() != stop or any(token in set(stops) for token in ids[p:-1].tolist()):
            raise ValueError("Action target must end at its first native generation stop")
        mask = inputs["attention_mask"]
        if not torch.all((mask == 0) | (mask == 1)) or not mask[0, p - 1 :].all():
            raise ValueError("Action target and preceding readout position must be unmasked")
        if "mm_token_type_ids" in inputs and inputs["mm_token_type_ids"].any():
            raise ValueError("Action LoRA fitting is text only")
        special_media = [
            getattr(model.lm.config, key, None)
            for key in ("image_token_id", "audio_token_id", "video_token_id")
        ]
        if any(token is not None and (ids == token).any() for token in special_media):
            raise ValueError("Media placeholders are unsupported in action training")
        positions = torch.arange(p - 1, length - 1, dtype=torch.long)
        record = {
            "case_id": case.id,
            "split": case.split,
            "group_id": case.group_id or case.id,
            "request_fingerprint": request_fingerprint(case.request),
            "original_wire_sha256": integrity._sha(_encode(original)),
            "normalized_wire_sha256": integrity._sha(source),
            "original_model_alias": original.get("model"),
            "inference_model_identity": identity,
            "normalization": "only payload.model replaced; all other original fields/order retained",
            "policy": POLICY,
            "prompt": prompt,
            "prompt_tokens": p,
            "target_json": target,
            "target_token_ids": ids[p:].tolist(),
            "native_stop_token_id": stop,
            "input_tokens": length,
            "logical_input_tokens": int(mask.sum()),
            "logits_positions": positions.tolist(),
            "processor_tensors": {key: value.tolist() for key, value in inputs.items()},
        }
        rows.append(
            {"record": record, "inputs": inputs, "positions": positions, "targets": ids[p:].clone()}
        )
    return rows


def action_loss(model, row):
    """Project only target readouts: input position i predicts token i+1."""
    import torch
    import torch.nn.functional as functional

    inputs = {key: value.to(model.device) for key, value in row["inputs"].items()}
    output = model.lm(
        **inputs,
        logits_to_keep=row["positions"].to(model.device),
        use_cache=False,
        return_dict=True,
    )
    logits, targets = output.logits[0], row["targets"].to(model.device)
    if logits.ndim != 2 or logits.shape[0] != len(targets) or not torch.isfinite(logits).all():
        raise ValueError("Native target logits have invalid shape or nonfinite values")
    loss = functional.cross_entropy(logits.float(), targets)
    if not torch.isfinite(loss):
        raise ValueError("Native action loss is nonfinite")
    correct = logits.detach().argmax(-1).eq(targets)
    return loss, {
        "target_tokens": len(targets),
        "correct_target_tokens": int(correct.sum()),
        "teacher_forced_exact_tokens": bool(correct.all()),
    }


def train_actions(model, train, development, output, *, steps=None, learning_rate=1e-4, seed=42):
    """One bounded prompt per native forward; only train rows update adapters."""
    import torch

    if (
        not train
        or not development
        or any(row["record"]["split"] != "train" for row in train)
        or any(row["record"]["split"] != "development" for row in development)
    ):
        raise ValueError("Separate nonempty train and development rows are required")
    steps = len(train) if steps is None else steps
    if (
        type(steps) is not int
        or steps < 1
        or type(seed) is not int
        or type(learning_rate) not in (int, float)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("Positive updates/learning rate and integer seed required")
    parameters = [
        (name, value) for name, value in model.lm.named_parameters() if value.requires_grad
    ]
    if not parameters or any(
        TRAINABLE_NAME.fullmatch(name) is None
        or value.dtype != torch.float32
        or not torch.isfinite(value).all()
        for name, value in parameters
    ):
        raise ValueError("Only finite FP32 q/v LoRA adapters may be trainable")
    frozen = {
        name: (value, value._version)
        for name, value in model.lm.named_parameters()
        if not value.requires_grad
    }
    head_hash = _tensor_sha(model.head.weight)
    if model.head.weight.requires_grad:
        raise ValueError("Original full vocabulary head must stay frozen")
    output = Path(output)
    optimizer = torch.optim.AdamW(
        [value for _, value in parameters], lr=learning_rate, weight_decay=0.0
    )
    rng, order = random.Random(seed), list(range(len(train)))
    torch.manual_seed(seed)
    trace = {
        "backbone_calls": [],
        "decoder_layer_calls": {},
        "scope": "entered native calls; backward checkpoint layer recomputations are counted separately",
    }
    active = {"split": "train", "step": 0, "phase": "setup", "case_id": None}

    def backbone(_module, _args, kwargs):
        ids = kwargs.get("input_ids")
        trace["backbone_calls"].append(
            {**active, "input_shape": list(ids.shape) if ids is not None else None}
        )

    def layer(index):
        def entered(_module, _args):
            key = active["phase"] + ":" + str(index)
            trace["decoder_layer_calls"][key] = trace["decoder_layer_calls"].get(key, 0) + 1

        return entered

    hooks = []
    layers = getattr(getattr(model.backbone, "language_model", model.backbone), "layers", [])
    completed, eval_rows = [], []
    training_started = time.perf_counter()
    try:
        hooks.append(model.backbone.register_forward_pre_hook(backbone, with_kwargs=True))
        for index, module in enumerate(layers):
            hooks.append(module.register_forward_pre_hook(layer(index)))
        model.lm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.lm.train()
        for index in range(steps):
            if index % len(train) == 0:
                rng.shuffle(order)
            row = train[order[index % len(train)]]
            active.update(
                split="train", step=index + 1, phase="forward", case_id=row["record"]["case_id"]
            )
            before, row_started = len(trace["backbone_calls"]), time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = action_loss(model, row)
            active["phase"] = "backward"
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                [value for _, value in parameters], 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            if any(not torch.isfinite(value).all() for _, value in parameters):
                raise ValueError("Action LoRA update produced nonfinite parameters")
            record = {
                "step": index + 1,
                "case_id": row["record"]["case_id"],
                "loss": float(loss.detach()),
                "gradient_norm": float(norm),
                **metrics,
                "input_tokens": row["record"]["input_tokens"],
                "logical_input_tokens": row["record"]["logical_input_tokens"],
                "forward_batch_size": 1,
                "backbone_forward_calls": len(trace["backbone_calls"]) - before,
                "elapsed_ms": (time.perf_counter() - row_started) * 1000,
                "latency_scope": "forward_backward_optimizer_and_synchronized_metrics; excludes evidence writes",
                "resources": _resources(model),
            }
            completed.append(record)
            _append(output / "updates.jsonl", record)
            _write(
                output / "progress.json",
                {
                    "stage": "training",
                    "completed_updates": len(completed),
                    "planned_updates": steps,
                    "last": record,
                },
            )
            if index == 0 or (index + 1) % 16 == 0:
                print(f"action LoRA {index + 1}/{steps}, loss={record['loss']:.4f}", flush=True)
        model.lm.eval()
        model.lm.gradient_checkpointing_disable()
        with torch.no_grad():
            for index, row in enumerate(development):
                active.update(
                    split="development",
                    step=index + 1,
                    phase="evaluation",
                    case_id=row["record"]["case_id"],
                )
                loss, metrics = action_loss(model, row)
                record = {
                    "case_id": row["record"]["case_id"],
                    "loss": float(loss),
                    **metrics,
                    "input_tokens": row["record"]["input_tokens"],
                    "logical_input_tokens": row["record"]["logical_input_tokens"],
                }
                eval_rows.append(record)
                _append(output / "development.jsonl", record)
        unchanged = all(
            value._version == version and not value.requires_grad
            for value, version in frozen.values()
        )
        after_hash = _tensor_sha(model.head.weight)
        if not unchanged or after_hash != head_hash:
            raise ValueError("Original frozen base/head parameters changed")
        result = {
            "scope": "teacher-forced action-token fitting diagnostics; no browser success or generated-action evaluation",
            "completed_updates": len(completed),
            "development_examples": len(eval_rows),
            "development_updates": 0,
            "trainable_parameters": sum(value.numel() for _, value in parameters),
            "trainable_names": [name for name, _ in parameters],
            "original_head_sha256_before": head_hash,
            "original_head_sha256_after": after_hash,
            "frozen_base_versions_unchanged": unchanged,
            "development_loss_mean": sum(row["loss"] for row in eval_rows) / len(eval_rows),
            "trace": trace,
            "execution": {
                "backbone_forward_calls": len(trace["backbone_calls"]),
                "max_forward_batch_size": 1,
                "physical_input_positions": sum(
                    math.prod(call["input_shape"]) for call in trace["backbone_calls"]
                ),
                "logical_input_tokens": sum(
                    row["logical_input_tokens"] for row in completed + eval_rows
                ),
                "target_projection_positions": sum(
                    row["target_tokens"] for row in completed + eval_rows
                ),
                "decoder_layer_calls_including_checkpoint_recomputation": sum(
                    trace["decoder_layer_calls"].values()
                ),
                "scope": "whole training and held-aside evaluation; no padding/candidate pruning",
            },
            "resources": _resources(model),
            "elapsed_ms": (time.perf_counter() - training_started) * 1000,
        }
        _write(output / "training.json", result)
        return result
    except BaseException as exc:
        evidence = {
            "active": active,
            "completed_updates": len(completed),
            "completed_development_examples": len(eval_rows),
            "trace": trace,
            "elapsed_ms": (time.perf_counter() - training_started) * 1000,
        }
        try:
            evidence["resources"] = _resources(model)
        except BaseException as resource_error:
            evidence["resource_error"] = {
                "type": type(resource_error).__name__,
                "message": str(resource_error),
            }
        exc.action_training_evidence = evidence
        try:
            _write(output / "training-failure.json", evidence)
        except BaseException as write_error:
            exc.add_note(
                "Could not preserve partial training evidence: " + type(write_error).__name__
            )
        raise
    finally:
        primary = sys.exception()
        cleanup_errors = []
        for cleanup in [
            *[hook.remove for hook in hooks],
            model.lm.eval,
            model.lm.gradient_checkpointing_disable,
        ]:
            try:
                cleanup()
            except BaseException as cleanup_error:
                cleanup_errors.append(
                    {"type": type(cleanup_error).__name__, "message": str(cleanup_error)}
                )
        if cleanup_errors:
            if primary is not None:
                primary.add_note("Training cleanup failures: " + json.dumps(cleanup_errors))
                if hasattr(primary, "action_training_evidence"):
                    primary.action_training_evidence["cleanup_errors"] = cleanup_errors
            else:
                raise RuntimeError("Training cleanup failed: " + json.dumps(cleanup_errors))


def _save_adapter(model, directory):
    directory = Path(directory)
    model.lm.save_pretrained(directory, safe_serialization=True, save_embedding_layers=False)
    (directory / "README.md").unlink(missing_ok=True)
    expected = {"adapter_config.json", "adapter_model.safetensors"}
    if {path.name for path in directory.iterdir()} - {"manifest.invalid.json"} != expected:
        raise ValueError("Adapter writer emitted unexpected artifacts")
    return {
        name: {
            "sha256": integrity._sha((directory / name).read_bytes()),
            "size_bytes": (directory / name).stat().st_size,
        }
        for name in sorted(expected)
    }


def _finalize(output, receipt, *, primary_error=None, **kwargs):
    """Preserve the original error even if final verification is interrupted."""
    try:
        integrity._finalize_fit(output, receipt, primary_error=primary_error, **kwargs)
    except BaseException as final_error:
        receipt.update(
            stage="failed",
            finalization_error={"type": type(final_error).__name__, "message": str(final_error)},
        )
        try:
            _write(Path(output) / "receipt.json", receipt)
        except BaseException as write_error:
            final_error.add_note(
                "Could not write failed finalization receipt: " + type(write_error).__name__
            )
        if primary_error is None:
            raise
        primary_error.add_note("Secondary finalization failure: " + type(final_error).__name__)


def fit(*, train_cases, development_cases, output_dir, steps=None, learning_rate=1e-4, seed=42):
    if (
        (steps is not None and (type(steps) is not int or steps < 1))
        or type(seed) is not int
        or type(learning_rate) not in (int, float)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("Positive updates/learning rate and integer seed required")
    train, train_receipt, train_snapshot = integrity._validated_snapshot(train_cases, "train")
    development, development_receipt, development_snapshot = integrity._validated_snapshot(
        development_cases, "development"
    )
    audit = integrity._audit_snapshots([train, development])
    if not audit["valid"]:
        raise ValueError("Training and development episode/source/request groups overlap")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "stage": "setup",
        "scope": "development-only native action LoRA fitting; not browser task success",
        "steps": len(train) if steps is None else steps,
        "learning_rate": learning_rate,
        "seed": seed,
        "split_audit": audit,
        "train_collection": train_receipt,
        "development_collection": development_receipt,
        "base_model_id": MODEL_ID,
        "base_revision": REVISION,
        "training_quantization": "none",
        "training_dtype": "bfloat16",
        "deployment_quantization": "nf4",
        "lora": LORA,
    }
    snapshots = {"train": train_snapshot, "development": development_snapshot}
    source_files = {
        "scripts/jev_browser_fit_actions.py": Path(__file__),
        "scripts/jev_browser_fit_readout.py": Path(integrity.__file__),
        "scripts/jev_browser_training_data.py": Path(integrity.training_case.__code__.co_filename),
        "scripts/breakthrough_training.py": Path(integrity.fit_head.__code__.co_filename),
    }
    started, source_manifest, primary, model, adapter_manifest = (
        time.perf_counter(),
        None,
        None,
        None,
        None,
    )
    try:
        import importlib.metadata

        import torch

        from s1.browser_evidence import capture_runtime_sources
        from s1.evaluation.datasets import request_fingerprint
        from s1.jev_generated import INSTRUCTIONS, POLICY
        from s1.unified import UnifiedDecisionModel

        receipt.update(policy=POLICY, instructions_sha256=integrity._sha(INSTRUCTIONS.encode()))
        receipt["versions"] = {
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")
        }
        receipt["attention_policy"] = {"implementation": "sdpa", "cudnn_sdp_enabled": False}
        source_manifest = capture_runtime_sources(output, additional_files=source_files)
        receipt["source_manifest"] = source_manifest
        receipt["input_manifests"] = {}
        for split, snapshot in snapshots.items():
            destination = output / "inputs" / split
            receipt["input_manifests"][split] = {
                **integrity._freeze_collection(snapshot, destination),
                "manifest_sha256": integrity._sha(
                    (destination / "input_manifest.json").read_bytes()
                ),
                "artifact_path": "inputs/" + split,
                "preservation_timing": "validated bytes frozen before model loading or training",
            }
        receipt["stage"] = "model_loading"
        _write(output / "receipt.json", receipt)
        torch.manual_seed(seed)
        torch.backends.cuda.enable_cudnn_sdp(False)
        load_started = time.perf_counter()
        model = UnifiedDecisionModel(
            MODEL_ID,
            revision=REVISION,
            device="cuda",
            precision="bfloat16",
            quantization="none",
            max_context=8192,
            attn_implementation="sdpa",
            lora={**LORA, "revision": REVISION},
        )
        receipt["model_load_ms"] = (time.perf_counter() - load_started) * 1000
        if (
            model.name != MODEL_ID
            or model.revision != REVISION
            or model.quantization != "none"
            or model.head.weight.dtype != torch.bfloat16
        ):
            raise ValueError("Training model identity differs from the pinned BF16 base")
        for config in model.lm.peft_config.values():
            config.base_model_name_or_path, config.revision = MODEL_ID, REVISION
        receipt["stage"] = "preflight"
        _write(output / "receipt.json", receipt)
        rows = {}
        for split, cases in (("train", train), ("development", development)):
            rows[split] = compile_actions(model, cases, snapshots[split])
            for row in rows[split]:
                _append(output / ("compiled-" + split + ".jsonl"), row["record"])
        receipt["compiled"] = {
            split: {
                "examples": len(records),
                "sha256": integrity._sha((output / ("compiled-" + split + ".jsonl")).read_bytes()),
                "prompt_tokens": sum(row["record"]["prompt_tokens"] for row in records),
                "target_tokens": sum(len(row["targets"]) for row in records),
            }
            for split, records in rows.items()
        }
        receipt["stage"] = "training"
        _write(output / "receipt.json", receipt)
        receipt["training"] = train_actions(
            model,
            rows["train"],
            rows["development"],
            output,
            steps=steps,
            learning_rate=learning_rate,
            seed=seed,
        )
        files = _save_adapter(model, output / "adapter")

        def inventory(cases):
            return {
                "ids": sorted({case.id for case in cases}),
                "groups": sorted({case.group_id or case.id for case in cases}),
                "request_fingerprints": sorted(
                    {request_fingerprint(case.request) for case in cases}
                ),
            }

        adapter_manifest = {
            "schema_version": 1,
            "kind": "jev_action_lora_v1",
            "base_model_id": MODEL_ID,
            "base_revision": REVISION,
            "training_quantization": "none",
            "training_dtype": "bfloat16",
            "deployment_quantization": "nf4",
            "policy": POLICY,
            "instructions_sha256": integrity._sha(INSTRUCTIONS.encode()),
            "inference_model_identity": IDENTITY,
            "lora": LORA,
            "files": files,
            "provenance": {
                "source_scope": "development_only",
                "train": inventory(train),
                "development": inventory(development),
            },
        }
        _write(output / "adapter/manifest.invalid.json", adapter_manifest)
        receipt.update(
            stage="complete",
            adapter_files=files,
            adapter_artifact="adapter",
            full_lm_head_unchanged=True,
        )
    except BaseException as exc:
        primary = exc
        receipt.update(stage="failed", error={"type": type(exc).__name__, "message": str(exc)})
        if hasattr(exc, "action_training_evidence"):
            receipt["partial_training"] = exc.action_training_evidence
        if model is not None:
            try:
                receipt["partial_adapter_files"] = _save_adapter(model, output / "adapter")
                _write(
                    output / "adapter/manifest.invalid.json",
                    {"kind": "jev_action_lora_v1", "stage": "failed", "error": receipt["error"]},
                )
            except BaseException as save_error:
                receipt["partial_adapter_error"] = {
                    "type": type(save_error).__name__,
                    "message": str(save_error),
                }
        raise
    finally:
        _finalize(
            output,
            receipt,
            source_manifest=source_manifest,
            source_files=source_files,
            input_snapshots=snapshots,
            started=started,
            primary_error=primary,
            owned_artifacts=[output / "adapter"],
        )
    # Valid manifests are published only after the final immutable receipt passes.
    adapter_manifest["training_receipt"] = {
        "path": "../receipt.json",
        "sha256": integrity._sha((output / "receipt.json").read_bytes()),
    }
    try:
        (output / "adapter/manifest.invalid.json").unlink()
        _write(output / "adapter/manifest.json", adapter_manifest)
    except BaseException as exc:
        receipt.update(stage="failed", error={"type": type(exc).__name__, "message": str(exc)})
        _finalize(
            output,
            receipt,
            source_manifest=source_manifest,
            source_files=source_files,
            input_snapshots=snapshots,
            started=started,
            primary_error=exc,
            owned_artifacts=[output / "adapter"],
        )
        raise
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cases", required=True)
    parser.add_argument("--development-cases", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    receipt = fit(**vars(parser.parse_args(argv)))
    print(
        json.dumps({key: receipt[key] for key in ("stage", "scope", "steps", "adapter_artifact")})
    )


if __name__ == "__main__":
    main()
