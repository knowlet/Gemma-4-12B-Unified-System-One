"""TRAIN-only CPU experiment for label-addressed operation residuals.

No file access, model execution, browser routing or GPU is performed here.
The caller must independently verify TRAIN-only admission and array hashes.
This is an experimental objective, not a deployed adapter or model score.
"""

from __future__ import annotations

import math

import numpy as np
from jev_decision3_train_operation_diagnostic import _criteria


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(*, features, targets, counts, baseline, row_map, train_only):
    _need(train_only is True, "independent TRAIN-only admission required")
    _need(type(features) is np.ndarray and features.ndim == 2, "feature matrix required")
    n, width = features.shape
    _need(
        n > 0
        and type(targets) is np.ndarray
        and targets.ndim == 2
        and targets.shape[0] == n
        and type(counts) is np.ndarray
        and counts.shape == (n,)
        and np.issubdtype(counts.dtype, np.integer)
        and type(baseline) is np.ndarray
        and baseline.shape == (targets.shape[1], width)
        and type(row_map) is list
        and len(row_map) == n,
        "aligned training arrays required",
    )
    _need(
        all(np.isfinite(a).all() for a in (features, targets, baseline))
        and np.all(counts >= 1)
        and np.all(counts <= targets.shape[1]),
        "finite legal input required",
    )
    menus = [_criteria(row) for row in row_map]
    _need(
        all(len(menu) == int(k) for menu, k in zip(menus, counts, strict=True)),
        "original menu/count mismatch",
    )
    mask = np.arange(targets.shape[1])[None, :] < counts[:, None]
    _need(
        (targets >= 0).all()
        and (targets[~mask] == 0).all()
        and np.allclose(targets.sum(axis=1, dtype=np.float64), 1, atol=1e-6, rtol=0),
        "original soft targets required",
    )
    names = sorted({name for menu in menus for name in menu})
    ids = {name: index for index, name in enumerate(names)}
    indexes = np.zeros(targets.shape, dtype=np.int64)
    for row, menu in enumerate(menus):
        indexes[row, : len(menu)] = [ids[name] for name in menu]
    x = features.astype(np.float32, copy=True)
    y = targets.astype(np.float32, copy=True)
    offset = x @ baseline.astype(np.float32, copy=False).T
    offset64 = features.astype(np.float64) @ baseline.astype(np.float64).T
    return {
        "features": x,
        "targets": y,
        "offset": offset,
        "offset64": offset64,
        "mask": mask,
        "indexes": indexes,
        "names": names,
        "menus": menus,
    }


def loss_gradient(data, delta, rows, *, l2, sample_weights=None):
    """Masked soft CE / row count + half-L2 and exact FP32 gradient."""
    x, y = data["features"][rows], data["targets"][rows]
    mask, indexes, offset = data["mask"][rows], data["indexes"][rows], data["offset"][rows]
    _need(
        len(x) > 0 and delta.shape == (len(data["names"]), x.shape[1]) and np.isfinite(delta).all(),
        "finite semantic residual required",
    )
    _need(
        type(l2) in (int, float) and math.isfinite(l2) and l2 >= 0, "finite nonnegative L2 required"
    )
    if sample_weights is None:
        weights = np.ones(len(x), dtype=np.float32)
    else:
        weights = np.asarray(sample_weights, dtype=np.float32)
        _need(
            weights.shape == (len(x),) and np.isfinite(weights).all() and (weights > 0).all(),
            "positive finite per-row weights required",
        )
    semantic = x @ delta.T
    logits = np.where(mask, offset + np.take_along_axis(semantic, indexes, axis=1), -np.inf)
    maximum = logits.max(axis=1, keepdims=True)
    exponent = np.exp(logits - maximum)
    probabilities = exponent / exponent.sum(axis=1, keepdims=True)
    logsum = maximum[:, 0] + np.log(exponent.sum(axis=1))
    finite_logits = np.where(mask, logits, 0.0)
    ce = logsum - (y * finite_logits).sum(axis=1)
    objective = float((weights * ce).mean() + 0.5 * l2 * np.sum(delta * delta))
    error = ((probabilities - y) * weights[:, None]) * mask
    by_name = np.zeros((len(x), len(data["names"])), dtype=np.float32)
    np.add.at(by_name, (np.arange(len(x))[:, None], indexes), error)
    gradient = by_name.T @ x / len(x) + np.float32(l2) * delta
    _need(
        math.isfinite(objective) and np.isfinite(gradient).all(),
        "finite objective and gradient required",
    )
    return objective, gradient


def fit(data, *, steps, batch_size, learning_rate, l2, seed, deadline, row_weights=None):
    _need(
        type(steps) is int
        and steps > 0
        and type(batch_size) is int
        and 1 <= batch_size <= len(data["features"])
        and type(learning_rate) in (int, float)
        and math.isfinite(learning_rate)
        and learning_rate > 0
        and type(seed) is int
        and callable(deadline),
        "bounded explicit optimizer settings required",
    )
    if row_weights is not None:
        _need(
            type(row_weights) is np.ndarray
            and row_weights.shape == (len(data["features"]),)
            and np.isfinite(row_weights).all()
            and (row_weights > 0).all(),
            "positive complete row weights required",
        )
    delta = np.zeros((len(data["names"]), data["features"].shape[1]), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        deadline()
        rows = rng.choice(len(data["features"]), size=batch_size, replace=False)
        weights = None if row_weights is None else row_weights[rows]
        _, gradient = loss_gradient(data, delta, rows, l2=l2, sample_weights=weights)
        delta -= np.float32(learning_rate) * gradient
    deadline()
    return delta


def predict(data, delta):
    _need(
        delta.shape == (len(data["names"]), data["features"].shape[1]) and np.isfinite(delta).all(),
        "semantic residual shape required",
    )
    x = data["features"].astype(np.float64)
    logits = data["offset64"] + np.take_along_axis(
        x @ delta.astype(np.float64).T, data["indexes"], axis=1
    )
    logits[~data["mask"]] = -np.inf
    return np.argmax(logits, axis=1)
