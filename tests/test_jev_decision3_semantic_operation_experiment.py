"""Synthetic CPU-only semantic readout tests."""

import copy

import numpy as np
import pytest
from jev_decision3_semantic_operation_experiment import fit, loss_gradient, predict, prepare
from test_jev_decision3_train_operation_diagnostic import example


def prepared():
    value = example()
    del value["old_rows"]
    del value["delta"]
    return prepare(**value)


def test_semantic_delta_follows_operation_name_across_permuted_menus():
    data = prepared()
    delta = np.zeros((len(data["names"]), 2), dtype=np.float32)
    delta[data["names"].index("TYPE_TEXT")] = [1, 1]
    selected = predict(data, delta)
    assert selected.tolist()[:2] == [1, 0]
    assert [data["menus"][i][choice] for i, choice in enumerate(selected[:2])] == [
        "TYPE_TEXT",
        "TYPE_TEXT",
    ]
    assert selected[2] < 2  # Mask the deliberately huge unused third native logit.


def test_loss_gradient_matches_independent_finite_difference():
    data = prepared()
    rng = np.random.default_rng(7)
    delta = rng.normal(size=(len(data["names"]), 2)).astype(np.float32) * 0.1
    rows = np.arange(3)
    _, gradient = loss_gradient(
        data, delta, rows, l2=0.01, sample_weights=np.array([0.5, 2, 1], dtype=np.float32)
    )
    for name in range(len(data["names"])):
        for column in range(2):
            plus, minus = delta.copy(), delta.copy()
            plus[name, column] += 0.001
            minus[name, column] -= 0.001
            p, _ = loss_gradient(
                data, plus, rows, l2=0.01, sample_weights=np.array([0.5, 2, 1], dtype=np.float32)
            )
            m, _ = loss_gradient(
                data, minus, rows, l2=0.01, sample_weights=np.array([0.5, 2, 1], dtype=np.float32)
            )
            assert gradient[name, column] == pytest.approx((p - m) / 0.002, abs=2e-4)


def test_weighted_loss_and_gradient_agree_with_independent_fp64_loop():
    data = prepared()
    delta = np.arange(len(data["names"]) * 2, dtype=np.float32).reshape(-1, 2) / 37
    weights = np.array([0.5, 2.0, 1.0], dtype=np.float32)
    actual_loss, actual_gradient = loss_gradient(
        data, delta, np.arange(3), l2=0.01, sample_weights=weights
    )
    gradient = np.zeros(delta.shape, dtype=np.float64)
    total = 0.0
    for row, menu in enumerate(data["menus"]):
        x = data["features"][row].astype(np.float64)
        ids = [data["names"].index(name) for name in menu]
        logits = np.asarray(
            [
                float(data["offset"][row, position]) + x @ delta[index].astype(np.float64)
                for position, index in enumerate(ids)
            ]
        )
        exponent = np.exp(logits - logits.max())
        probabilities = exponent / exponent.sum()
        target = data["targets"][row, : len(menu)].astype(np.float64)
        total -= float(weights[row]) * float(target @ np.log(probabilities))
        for position, index in enumerate(ids):
            gradient[index] += (
                float(weights[row]) * (probabilities[position] - target[position]) * x / 3
            )
    expected_loss = total / 3 + 0.005 * float(np.square(delta.astype(np.float64)).sum())
    gradient += 0.01 * delta.astype(np.float64)
    assert actual_loss == pytest.approx(expected_loss, abs=2e-7)
    np.testing.assert_allclose(actual_gradient, gradient, atol=1e-7, rtol=1e-6)
    unweighted = loss_gradient(data, delta, np.arange(3), l2=0.01)
    unit_weighted = loss_gradient(
        data, delta, np.arange(3), l2=0.01, sample_weights=np.ones(3, dtype=np.float32)
    )
    assert unweighted[0] == unit_weighted[0]
    np.testing.assert_array_equal(unweighted[1], unit_weighted[1])


def test_zero_delta_reproduces_native_ordinal_choice_and_inputs_unchanged():
    source = example()
    saved = copy.deepcopy(source)
    data = prepare(
        **{key: value for key, value in source.items() if key not in ("old_rows", "delta")}
    )
    delta = np.zeros((len(data["names"]), 2), dtype=np.float32)
    assert predict(data, delta).tolist() == [0, 0, 0]
    for key in ("features", "baseline", "delta", "targets", "counts"):
        np.testing.assert_array_equal(source[key], saved[key])
    assert source["row_map"] == saved["row_map"]


def test_fit_is_deterministic_and_propagates_deadline():
    data = prepared()
    args = dict(steps=10, batch_size=2, learning_rate=0.03, l2=0.001, seed=3, deadline=lambda: None)
    np.testing.assert_array_equal(fit(data, **args), fit(data, **args))

    def expired():
        raise TimeoutError("stop")

    with pytest.raises(TimeoutError, match="stop"):
        fit(data, **{**args, "deadline": expired})


def test_invalid_weight_or_unadmitted_data_rejected():
    data = prepared()
    delta = np.zeros((len(data["names"]), 2), dtype=np.float32)
    with pytest.raises(ValueError, match="positive"):
        loss_gradient(data, delta, np.arange(3), l2=0, sample_weights=[1, 0, 1])
    values = example()
    values.pop("old_rows")
    values.pop("delta")
    values["train_only"] = False
    with pytest.raises(ValueError, match="TRAIN-only"):
        prepare(**values)
