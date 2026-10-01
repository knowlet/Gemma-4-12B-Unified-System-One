"""Supervised multimodal LoRA and calibration on a separate, explicit split."""

from __future__ import annotations

import json
import math
import random

import numpy as np


def fit_temperature(logits, golds):
    """Deterministic scalar temperature fit; masked -inf candidates are allowed."""
    if not logits or len(logits) != len(golds):
        raise ValueError("nonempty, aligned calibration logits and labels are required")
    rows = [np.asarray(row, dtype=float) for row in logits]
    for row, gold in zip(rows, golds):
        if (
            row.ndim != 1
            or not 0 <= gold < len(row)
            or not np.isfinite(row[gold])
            or np.isnan(row).any()
            or np.isposinf(row).any()
        ):
            raise ValueError("invalid calibration logits or gold")

    def nll(temperature):
        losses = []
        for row, gold in zip(rows, golds):
            scaled = row / temperature
            peak = scaled.max()
            losses.append(peak + np.log(np.exp(scaled - peak).sum()) - scaled[gold])
        return float(np.mean(losses))

    candidates = np.unique(np.r_[np.geomspace(0.05, 20, 241), 1.0])
    temperature = float(min(candidates, key=nll))
    return {
        "temperature": temperature,
        "n": len(rows),
        "nll_before": nll(1),
        "nll_after": nll(temperature),
    }


def permute_request(request, rng):
    """Shuffle Choice positions while keeping semantic label golds unchanged."""
    request = request.model_copy(deep=True)
    for q in request.questions:
        if q.type != "choice":
            continue
        entries = list(zip(q.labels(), q.descriptions()))
        rng.shuffle(entries)
        q.criteria = dict(entries)
    return request


def _request_key(request):
    """Ignore mapping insertion order for overlap checks; retain sequence order."""
    return json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def train_model(
    model, train_cases, calibration_cases, *, steps=100, lr=2e-5, brier_weight=0.1, seed=0
):
    import torch
    from torch.nn import functional as F

    if not train_cases or not calibration_cases:
        raise ValueError("both training and calibration datasets are required")
    if any(c.split != "train" for c in train_cases) or any(
        c.split != "calibration" for c in calibration_cases
    ):
        raise ValueError(
            "training/calibration split labels are required; test data cannot be fitted"
        )
    if {c.id for c in train_cases} & {c.id for c in calibration_cases}:
        raise ValueError("training and calibration ids overlap")
    if {getattr(c, "group_id", None) or c.id for c in train_cases} & {
        getattr(c, "group_id", None) or c.id for c in calibration_cases
    }:
        raise ValueError("training and calibration source groups overlap")
    if {_request_key(c.request) for c in train_cases} & {
        _request_key(c.request) for c in calibration_cases
    }:
        raise ValueError("training and calibration requests overlap")
    if (
        steps < 1
        or not math.isfinite(lr)
        or lr <= 0
        or not math.isfinite(brier_weight)
        or brier_weight < 0
    ):
        raise ValueError("invalid training hyperparameters")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    parameters = [p for p in model.lm.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    model.lm.train()
    losses = []
    order = list(train_cases)
    for step in range(steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        case = order[step % len(order)]
        request = permute_request(case.request, rng)
        logits = model.logits(request)
        golds = torch.tensor(
            [q.labels().index(case.gold[q.id]) for q in request.questions], device=logits.device
        )
        probabilities = logits.softmax(-1)
        onehot = F.one_hot(golds, logits.shape[1]).to(probabilities.dtype)
        loss = (
            F.cross_entropy(logits, golds)
            + brier_weight * (probabilities - onehot).square().sum(-1).mean()
        )
        if not torch.isfinite(loss):
            raise ValueError("training produced a nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.lm.eval()
    logits, golds = [], []
    with torch.inference_mode():
        for case in calibration_cases:
            rows = model.logits(case.request).cpu().tolist()
            for q, row in zip(case.request.questions, rows):
                logits.append(row[: len(q.labels())])
                golds.append(q.labels().index(case.gold[q.id]))
    calibration = fit_temperature(logits, golds)
    model.set_temperature(calibration["temperature"])
    return {
        "steps": steps,
        "seed": seed,
        "learning_rate": lr,
        "brier_weight": brier_weight,
        "loss_initial": losses[0],
        "loss_final": losses[-1],
        "calibration": calibration,
        "train_ids": [c.id for c in train_cases],
        "calibration_ids": [c.id for c in calibration_cases],
    }
