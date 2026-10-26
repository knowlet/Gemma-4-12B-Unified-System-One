"""Unselected FP32 delta-head math; no data loading, model or artifact API.

A future owned caller must authenticate COMMITTED train-only features first.
This numerical primitive cannot qualify features or publish trained parameters.
It intentionally has no CLI or file IO. Tests use tiny synthetic arrays only.
"""

from __future__ import annotations

import math

import numpy as np


def _positive(value, name, *, zero=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (not zero and value == 0)
    ):
        raise ValueError(name + " must be finite and " + ("nonnegative" if zero else "positive"))
    with np.errstate(over="raise", under="ignore"):
        try:
            result = np.float32(value)
        except FloatingPointError as error:
            raise ValueError(name + " must fit FP32") from error
    if not np.isfinite(result) or (not zero and result == 0):
        raise ValueError(name + " must fit FP32")
    return result


def _inputs(features, baseline, delta, counts, targets=None):
    for name, array in (("features", features), ("baseline", baseline), ("delta", delta)):
        if type(array) is not np.ndarray or array.dtype != np.float32 or array.ndim != 2:
            raise ValueError(name + " requires a finite FP32 matrix")
    n, width = features.shape
    codes, dimension = baseline.shape
    if not (
        1 <= n <= 16384
        and 1 <= width == dimension <= 1024
        and 1 <= codes <= 255
        and delta.shape == baseline.shape
    ):
        raise ValueError("Bounded, matching feature/readout dimensions required")
    if any(not np.isfinite(a).all() for a in (features, baseline, delta)):
        raise ValueError("Finite FP32 feature/readout matrices required")
    if (
        type(counts) is not np.ndarray
        or counts.dtype.kind not in "iu"
        or counts.shape != (n,)
        or np.any(counts < 1)
        or np.any(counts > codes)
    ):
        raise ValueError("One positive legal option count per row required")
    illegal = np.arange(codes)[None, :] >= counts[:, None]
    if targets is not None:
        if (
            type(targets) is not np.ndarray
            or targets.dtype != np.float32
            or targets.shape != (n, codes)
            or not np.isfinite(targets).all()
            or np.any(targets < 0)
            or np.any(targets > 1)
        ):
            raise ValueError("Finite matching FP32 target probabilities required")
        if np.any(targets[illegal] != 0) or not np.allclose(
            targets.sum(axis=1, dtype=np.float64), 1, rtol=0, atol=1e-6
        ):
            raise ValueError("Targets must sum to one with zero illegal-option mass")
    return illegal


def _probabilities(features, baseline, delta, illegal):
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        logits = features @ baseline.T + features @ delta.T
        logits[illegal] = -np.inf
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        denominator = exp.sum(axis=1, keepdims=True, dtype=np.float32)
        probabilities = exp / denominator
        log_probabilities = shifted - np.log(denominator)
    return probabilities, log_probabilities


def probabilities(features, baseline, delta, counts):
    """Separate NumPy profile; fixed T=1 and original count-based legal mask."""
    illegal = _inputs(features, baseline, delta, counts)
    return _probabilities(features, baseline, delta, illegal)[0]


def loss_gradient(features, baseline, delta, counts, targets, *, l2=0.0001):
    """Mean soft-target CE plus half-L2; return only the delta gradient."""
    coefficient = _positive(l2, "l2", zero=True)
    illegal = _inputs(features, baseline, delta, counts, targets)
    probabilities, log_probabilities = _probabilities(features, baseline, delta, illegal)
    positive = targets > 0
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        loss = -(targets[positive] * log_probabilities[positive]).sum(
            dtype=np.float32
        ) / np.float32(len(features))
        loss += np.float32(0.5) * coefficient * (delta * delta).sum(dtype=np.float32)
        target_mass = targets.sum(axis=1, keepdims=True, dtype=np.float32)
        gradient = ((probabilities * target_mass - targets).T @ features) / np.float32(
            len(features)
        ) + coefficient * delta
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise ValueError("Nonfinite loss or delta gradient")
    return float(loss), gradient


def fit_arrays(
    features,
    baseline,
    counts,
    targets,
    *,
    check_deadline,
    steps=32,
    learning_rate=0.001,
    l2=0.0001,
    seed=20261023,
    batch_size=64,
):
    """Bounded deterministic SGD for a future qualified caller, not a trainer CLI.

    The mandatory callback must enforce the owned caller's entire transaction.
    A callback failure returns no result; this function never writes checkpoints.
    Baseline, features, counts and target arrays remain unchanged on all paths.
    """
    if not callable(check_deadline):
        raise ValueError("Owned caller deadline callback required")
    check_deadline()
    rate = _positive(learning_rate, "learning_rate")
    _positive(l2, "l2", zero=True)
    if (
        type(steps) is not int
        or not 1 <= steps <= 10000
        or type(seed) is not int
        or not 0 <= seed < 2**32
        or type(batch_size) is not int
        or not 1 <= batch_size <= 256
    ):
        raise ValueError("Bounded integer steps, seed and minibatch required")
    if (
        type(baseline) is not np.ndarray
        or baseline.dtype != np.float32
        or baseline.ndim != 2
        or not (1 <= baseline.shape[0] <= 255 and 1 <= baseline.shape[1] <= 1024)
    ):
        raise ValueError("Bounded FP32 baseline matrix required before allocation")
    delta = np.zeros_like(baseline, dtype=np.float32)
    _inputs(features, baseline, delta, counts, targets)
    rng, losses = np.random.default_rng(seed), []
    check_deadline()
    for _ in range(steps):
        check_deadline()
        index = rng.choice(len(features), size=min(batch_size, len(features)), replace=False)
        loss, gradient = loss_gradient(
            features[index], baseline, delta, counts[index], targets[index], l2=l2
        )
        with np.errstate(over="raise", invalid="raise"):
            candidate = delta - rate * gradient
        if not np.isfinite(candidate).all():
            raise ValueError("Nonfinite delta update")
        delta = candidate
        losses.append(loss)
        check_deadline()
    check_deadline()
    return {
        "delta": delta,
        "losses": losses,
        "steps": steps,
        "seed": seed,
        "numerical_profile": "numpy-fp32-masked-delta-sgd-v1",
        "data_qualification_supplied": False,
    }
