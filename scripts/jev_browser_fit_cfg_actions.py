"""Versioned matched A/C native action fitting; no model loading or CLI execution.

The optimizer/sampler/freeze/checkpoint loop derives from pinned historical
train_actions. The objective is an explicit argument, never a global replacement.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

from jev_browser_fit_actions import TRAINABLE_NAME, _append, _resources, _tensor_sha, _write


def objective(mode):
    from jev_public_cfg_action_artifact import objective as descriptor

    return descriptor(mode)


def _bound(row, plan):
    import jev_cfg_action_loss as cfg

    cfg._verified(plan)
    if hashlib.sha256(cfg._encode(cfg._snapshot(row))).hexdigest() != plan.compiled_row_sha256:
        raise ValueError("Compiled row changed after request-bound CFG plan verification")


def _validate_plans(train, development, plans, mode):
    objective(mode)
    if type(plans) is not dict:
        raise ValueError("Exact case-to-plan dictionary required")
    rows = [*train, *development]
    ids = [row["record"]["case_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(plans) != set(ids):
        raise ValueError("Every original train/development row needs one unique CFG plan")
    for row in rows:
        _bound(row, plans[row["record"]["case_id"]])


def action_loss(model, row, *, plan, mode):
    """One complete causal forward; A/C share both full and branch diagnostics."""
    import jev_cfg_action_loss as cfg
    import torch
    import torch.nn.functional as functional

    descriptor = objective(mode)
    _bound(row, plan)
    inputs = {key: value.to(model.device) for key, value in row["inputs"].items()}
    output = model.lm(
        **inputs,
        logits_to_keep=row["positions"].to(model.device),
        use_cache=False,
        return_dict=True,
    )
    if output.logits.ndim != 3 or output.logits.shape[0] != 1:
        raise ValueError("Exactly one unchanged native teacher-forced forward required")
    branch_loss, metrics = cfg.decision_token_loss(output.logits[0], plan, mode="C")
    targets = torch.tensor(plan.target_ids, dtype=torch.long, device=output.logits.device)
    loss = (
        functional.cross_entropy(output.logits[0].float(), targets) if mode == "A" else branch_loss
    )
    if not torch.isfinite(loss):
        raise ValueError("Matched native action objective is nonfinite")
    metrics.update(
        training_objective=descriptor["loss"],
        objective_mode=mode,
        compiled_row_sha256=plan.compiled_row_sha256,
        request_sha256=plan.request_sha256,
    )
    return loss, metrics


def train_actions(
    model, train, development, output, *, plans, mode, steps=120, learning_rate=5e-5, seed=42
):
    """One bounded prompt per native forward; only train rows update adapters."""
    import torch

    _validate_plans(train, development, plans, mode)

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
            loss, metrics = action_loss(model, row, plan=plans[row["record"]["case_id"]], mode=mode)
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
                loss, metrics = action_loss(
                    model, row, plan=plans[row["record"]["case_id"]], mode=mode
                )
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
            "objective": objective(mode),
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
