"""Research-only soft-target training; never reads public benchmark labels."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np


def _training_options(steps, learning_rate):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("positive integer update count required")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("positive finite learning rate required")


def _row(row):
    from s1.contracts import validate_distribution

    truth = np.asarray(validate_distribution(row["target"], len(row["target"])), dtype=np.float64)
    logits = np.asarray(row["logits"], dtype=np.float64)
    if logits.ndim != 1 or len(logits) != len(truth) or not np.isfinite(logits).all():
        raise ValueError("aligned finite candidate logits required")
    return logits, truth


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")


def target(case, question):
    labels = question.labels()
    if not case.soft_gold or question.id not in case.soft_gold:
        raise ValueError("research training requires complete explicit soft targets")
    from s1.contracts import validate_distribution

    return validate_distribution(
        [case.soft_gold[question.id][label] for label in labels], len(labels)
    )


def soft_loss(logits, targets, *, brier_weight=0.1):
    import torch

    if logits.ndim != 1 or logits.numel() != len(targets):
        raise ValueError("aligned finite candidate logits required")
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite candidate logits")
    if not math.isfinite(brier_weight) or brier_weight < 0:
        raise ValueError("finite nonnegative Brier weight required")
    from s1.contracts import validate_distribution

    targets = validate_distribution(targets, len(targets))
    truth = torch.tensor(targets, device=logits.device, dtype=torch.float32)
    logp = logits.float().log_softmax(-1)
    return -(truth * logp).sum() + brier_weight * (logp.exp() - truth).square().sum()


def capture(model, request, prompt_mode, head=None):
    """One real native forward per question, independent of other batch shapes."""
    import torch
    from breakthrough_prompt import wrap_model

    from s1.unified import project_candidates

    proxy = wrap_model(model, prompt_mode)
    result = []
    for question in request.questions:
        part = request.model_copy(update={"questions": [question]})
        inputs, slots, counts = proxy.prepare(part)
        out = proxy.backbone(**inputs, use_cache=False, return_dict=True)
        hidden = out.last_hidden_state[0, slots]
        if head is None:
            logits = project_candidates(hidden, proxy.head, proxy.letters, counts, proxy.softcap)
        else:
            logits = torch.nn.functional.linear(hidden.float(), head["weight"], head["bias"])
            if proxy.softcap is not None:
                logits = torch.tanh(logits / proxy.softcap) * proxy.softcap
        result.append(
            {
                "question": question,
                "logits": logits[0, : len(question.labels())],
                "hidden": hidden[0],
                "sequence_tokens": int(inputs["input_ids"].shape[1]),
            }
        )
    return result


def records(model, cases, prompt_mode, head=None, *, features=False):
    import torch

    model.lm.eval()
    output = []
    with torch.inference_mode():
        for i, case in enumerate(cases):
            rows = capture(model, case.request, prompt_mode, head)
            for row in rows:
                question = row["question"]
                entry = {
                    "case_id": case.id,
                    "group_id": case.group_id or case.id,
                    "split": case.split,
                    "type": question.type,
                    "question_id": question.id,
                    "labels": question.labels(),
                    "score_values": question.score_values() if question.type == "score" else None,
                    "logits": row["logits"].float().cpu().tolist(),
                    "target": target(case, question),
                    "sequence_tokens": row["sequence_tokens"],
                }
                if features:
                    entry["hidden"] = row["hidden"].float().cpu()
                output.append(entry)
            if (i + 1) % 128 == 0:
                print(f"capture {prompt_mode}: {i + 1}/{len(cases)}", flush=True)
    return output


def fit_temperatures(rows):
    if not rows or any(row["split"] != "calibration" for row in rows):
        raise ValueError("only calibration records may fit temperatures")
    result = {}
    for kind in ("choice", "noul", "score"):
        selected = [row for row in rows if row["type"] == kind]
        if not selected:
            raise ValueError("calibration must cover every primitive")

        def loss(t):
            values = []
            for row in selected:
                logits, truth = _row(row)
                logits = logits / t
                shifted = logits - logits.max()
                logp = shifted - np.log(np.exp(shifted).sum())
                values.append(float(-(truth * logp).sum()))
            return float(np.mean(values))

        grid = np.geomspace(0.1, 10.0, 81)
        temperature = float(min(grid, key=loss))
        result[kind] = {
            "temperature": temperature,
            "n": len(selected),
            "soft_ce_before": loss(1.0),
            "soft_ce_after": loss(temperature),
            "boundary": temperature in (float(grid[0]), float(grid[-1])),
        }
    return result


def summarize(rows, temperatures):
    output = {}
    for kind in ("all", "choice", "noul", "score"):
        selected = rows if kind == "all" else [r for r in rows if r["type"] == kind]
        if not selected:
            output[kind] = {"n": 0}
            continue
        values = []
        for row in selected:
            t = temperatures[row["type"]]
            if not math.isfinite(t) or t <= 0:
                raise ValueError("positive finite temperature required")
            x, truth = _row(row)
            x = x / t
            x -= x.max()
            logp = x - np.log(np.exp(x).sum())
            p = np.exp(logp)
            entropy = -sum(v * math.log(v) for v in truth if v > 0)
            value = {
                "soft_ce": float(-(truth * logp).sum()),
                "kl": float(-(truth * logp).sum() - entropy),
                "brier": float(np.square(p - truth).sum()),
                "oracle_argmax_match": float(int(p.argmax()) == int(truth.argmax())),
            }
            if row["type"] == "score":
                levels = np.asarray(row["score_values"])
                if (
                    levels.ndim != 1
                    or len(levels) != len(p)
                    or not np.isfinite(levels).all()
                    or np.any(np.diff(levels) <= 0)
                ):
                    raise ValueError("aligned increasing finite score values required")
                value["expected_score_mae"] = float(abs(np.dot(p - truth, levels)))
            values.append(value)
        output[kind] = {
            "n": len(selected),
            **{
                key: float(np.mean([v[key] for v in values if key in v]))
                for key in {k for v in values for k in v}
            },
        }
    return output


def fit_head(model, rows, *, steps=128, seed=42, learning_rate=1e-4):
    import torch

    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("only training features may fit head")
    _training_options(steps, learning_rate)
    torch.manual_seed(seed)
    initial = model.head.weight.detach().index_select(0, model.letters).float().cpu().clone()
    weight = torch.nn.Parameter(initial.clone())
    original_bias = getattr(model.head, "bias", None)
    bias = torch.nn.Parameter(
        original_bias.detach().index_select(0, model.letters).float().cpu().clone()
        if original_bias is not None
        else torch.zeros(len(model.letters))
    )
    features = torch.stack([row["hidden"].detach().cpu().float() for row in rows])
    if not torch.isfinite(features).all():
        raise ValueError("finite frozen hidden features required")
    optimizer = torch.optim.AdamW([weight, bias], lr=learning_rate, weight_decay=0.0)
    rng, losses = random.Random(seed), []
    for step in range(steps):
        indices = rng.sample(range(len(rows)), min(64, len(rows)))
        logits = torch.nn.functional.linear(features[indices], weight, bias)
        if model.softcap is not None:
            logits = torch.tanh(logits / model.softcap) * model.softcap
        loss = (
            torch.stack(
                [
                    soft_loss(logits[i, : len(rows[index]["target"])], rows[index]["target"])
                    for i, index in enumerate(indices)
                ]
            ).mean()
            + 0.01 * (weight - initial).square().mean()
        )
        if not torch.isfinite(loss):
            raise ValueError("nonfinite head loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([weight, bias], 1.0, error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in (weight, bias)):
            raise ValueError("head update produced nonfinite weights")
        losses.append(float(loss.detach()))
    return {"weight": weight.detach(), "bias": bias.detach()}, {
        "steps": steps,
        "seed": seed,
        "learning_rate": learning_rate,
        "batch_size": min(64, len(rows)),
        "losses": losses,
        "scope": "frozen backbone; initialized selected 52 LM-head rows; FP32 head; soft CE + 0.1 Brier + 0.01 anchor",
    }


def train_lora(
    model,
    cases,
    output,
    *,
    steps=128,
    seed=42,
    learning_rate=1e-5,
    accumulation=1,
    commit=lambda: None,
):
    import torch

    if not cases or any(case.split != "train" for case in cases):
        raise ValueError("only training cases may update LoRA")
    _training_options(steps, learning_rate)
    _training_options(accumulation, learning_rate)
    parameters = [p for p in model.lm.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("LoRA parameters required")
    if any(p.dtype != torch.float32 or not torch.isfinite(p).all() for p in parameters):
        raise ValueError("finite FP32 trainable adapters required")
    torch.manual_seed(seed)
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.train()
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    order, rng, losses = list(cases), random.Random(seed), []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        case_ids, loss_sum = [], 0.0
        for micro in range(accumulation):
            index = step * accumulation + micro
            if index % len(order) == 0:
                rng.shuffle(order)
            case = order[index % len(order)]
            rows = capture(model, case.request, "user_question")
            loss = torch.stack(
                [soft_loss(row["logits"], target(case, row["question"])) for row in rows]
            ).mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite LoRA loss")
            (loss / accumulation).backward()
            case_ids.append(case.id)
            loss_sum += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in parameters):
            raise ValueError("LoRA update produced nonfinite weights")
        record = {
            "step": step + 1,
            "case_id": case.id,
            "case_ids": case_ids,
            "loss": loss_sum / accumulation,
            "gradient_norm": float(norm),
        }
        losses.append(record)
        with (output / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if (step + 1) % 32 == 0 or step == 0:
            print(f"mixed LoRA: {step + 1}/{steps}, loss={record['loss']:.4f}", flush=True)
            model.lm.save_pretrained(output / "adapter")
            write_json(output / "progress.json", {"status": "running", "step": step + 1})
            commit()
    model.lm.eval()
    model.lm.gradient_checkpointing_disable()
    model.lm.save_pretrained(output / "adapter")
    result = {
        "status": "completed",
        "steps": steps,
        "seed": seed,
        "learning_rate": learning_rate,
        "gradient_accumulation": accumulation,
        "n_training_calls": steps * accumulation,
        "n_unique_training_cases": len({case_id for row in losses for case_id in row["case_ids"]}),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "losses": losses,
        "scope": "attention LoRA; soft CE + 0.1 Brier; user-question prompt; no public fitting",
        "adapter_files": {
            p.name: {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
            }
            for p in (output / "adapter").iterdir()
            if p.is_file()
        },
    }
    write_json(output / "training.json", result)
    commit()
    return result
