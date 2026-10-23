"""Independent closed-form/finite-difference checks, tiny synthetic arrays only."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import head_core as core


class HeadMathTests(unittest.TestCase):
    def arrays(self):
        return (
            np.array([[0.3, -0.4], [0.8, 0.2]], dtype=np.float32),
            np.array([[0.1, -0.2], [0.3, 0.1], [900, 800]], dtype=np.float32),
            np.zeros((3, 2), dtype=np.float32),
            np.array([1, 2], dtype=np.int64),
            np.array([[1, 0, 0], [0.25, 0.75, 0]], dtype=np.float32),
        )

    def test_masked_probabilities_match_independent_two_class_closed_form(self):
        x, w, d, counts, _ = self.arrays()
        actual = core.probabilities(x, w, d, counts)
        a = sum(float(x[1, j]) * float(w[0, j]) for j in range(2))
        b = sum(float(x[1, j]) * float(w[1, j]) for j in range(2))
        expected = 1 / (1 + math.exp(b - a))
        self.assertAlmostEqual(float(actual[1, 0]), expected, places=6)
        np.testing.assert_array_equal(actual[0], [1, 0, 0])
        self.assertEqual(float(actual[1, 2]), 0)
        self.assertEqual(actual.dtype, np.float32)

    def test_soft_target_delta_gradient_matches_independent_finite_differences(self):
        x, w, d, counts, target = self.arrays()
        d[:2] = [[0.1, -0.05], [0.2, 0.4]]
        _, gradient = core.loss_gradient(x, w, d, counts, target, l2=0.2)
        for row in range(3):
            for column in range(2):
                plus, minus = d.copy(), d.copy()
                plus[row, column] += 0.002
                minus[row, column] -= 0.002
                lp, _ = core.loss_gradient(x, w, plus, counts, target, l2=0.2)
                lm, _ = core.loss_gradient(x, w, minus, counts, target, l2=0.2)
                self.assertAlmostEqual(
                    float(gradient[row, column]), (lp - lm) / 0.004, delta=0.00008
                )

    def test_singleton_has_zero_data_gradient_without_regularization(self):
        x = np.array([[2, -3]], dtype=np.float32)
        w = np.zeros((3, 2), dtype=np.float32)
        loss, gradient = core.loss_gradient(
            x, w, w.copy(), np.array([1]), np.array([[1, 0, 0]], dtype=np.float32), l2=0
        )
        self.assertEqual(loss, 0)
        np.testing.assert_array_equal(gradient, np.zeros_like(w))

    def test_wrong_high_confidence_class_matches_closed_form_loss_and_gradient(self):
        x = np.array([[1]], dtype=np.float32)
        w = np.array([[2], [-2]], dtype=np.float32)
        loss, gradient = core.loss_gradient(
            x, w, np.zeros_like(w), np.array([2]), np.array([[0, 1]], dtype=np.float32), l2=0
        )
        self.assertAlmostEqual(loss, math.log1p(math.exp(4)), places=6)
        probability = 1 / (1 + math.exp(-4))
        np.testing.assert_allclose(gradient, [[probability], [-probability]], rtol=1e-6, atol=1e-7)

    def test_soft_target_rounding_mass_has_exact_cross_entropy_derivative(self):
        x = np.array([[2**20]], dtype=np.float32)
        w = np.zeros((2, 1), dtype=np.float32)
        target = np.array([[0.5, 0.5 + 2**-21]], dtype=np.float32)
        _, gradient = core.loss_gradient(x, w, w.copy(), np.array([2]), target, l2=0)
        np.testing.assert_allclose(gradient, [[0.25], [-0.25]], rtol=0, atol=1e-8)

    def test_synthetic_separable_examples_learn_without_mutating_any_input(self):
        x = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float32)
        w = np.zeros((2, 2), dtype=np.float32)
        counts = np.array([2, 2, 2, 2])
        target = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
        before = [a.copy() for a in (x, w, counts, target)]
        for a in (x, w, counts, target):
            a.flags.writeable = False
        result = core.fit_arrays(
            x,
            w,
            counts,
            target,
            check_deadline=lambda: None,
            steps=64,
            learning_rate=0.2,
            batch_size=4,
        )
        self.assertLess(result["losses"][-1], 0.3)
        np.testing.assert_array_equal(
            core.probabilities(x, w, result["delta"], counts).argmax(axis=1), target.argmax(axis=1)
        )
        for actual, original in zip((x, w, counts, target), before):
            np.testing.assert_array_equal(actual, original)
        self.assertIs(result["data_qualification_supplied"], False)

    def test_same_seed_and_data_produce_same_delta(self):
        x, w, _, counts, target = self.arrays()
        a = core.fit_arrays(
            x, w, counts, target, check_deadline=lambda: None, steps=8, batch_size=1
        )
        b = core.fit_arrays(
            x, w, counts, target, check_deadline=lambda: None, steps=8, batch_size=1
        )
        np.testing.assert_array_equal(a["delta"], b["delta"])
        self.assertEqual(a["losses"], b["losses"])

    def test_late_deadline_refuses_result_and_leaves_inputs_unchanged(self):
        x, w, _, counts, target = self.arrays()
        old = w.copy()
        called = 0

        def check():
            nonlocal called
            called += 1
            if called == 5:
                raise TimeoutError("synthetic deadline")

        with self.assertRaisesRegex(TimeoutError, "synthetic deadline"):
            core.fit_arrays(x, w, counts, target, check_deadline=check, steps=1)
        np.testing.assert_array_equal(w, old)

    def test_invalid_targets_cannot_place_mass_on_illegal_or_negative_positions(self):
        for mutation in ["illegal", "negative", "sum", "nan"]:
            with self.subTest(mutation=mutation):
                x, w, d, counts, target = self.arrays()
                if mutation == "illegal":
                    target[0] = [0, 1, 0]
                elif mutation == "negative":
                    target[1] = [-0.1, 1.1, 0]
                elif mutation == "sum":
                    target[1] = [0.1, 0.1, 0]
                else:
                    target[1, 0] = np.nan
                with self.assertRaises(ValueError):
                    core.loss_gradient(x, w, d, counts, target)

    def test_bad_count_types_or_ranges_are_refused(self):
        for bad in [
            np.array([True, True]),
            np.array([1.0, 2.0]),
            np.array([0, 2]),
            np.array([1, 4]),
            np.array([1]),
        ]:
            x, w, d, _, target = self.arrays()
            with self.assertRaises(ValueError):
                core.loss_gradient(x, w, d, bad, target)

    def test_nonfinite_or_wrong_precision_input_is_refused(self):
        for bad in ["nan", "fp64", "shape"]:
            x, w, d, counts, _ = self.arrays()
            if bad == "nan":
                x[0, 0] = np.nan
            elif bad == "fp64":
                x = x.astype(np.float64)
            else:
                d = np.zeros((2, 2), dtype=np.float32)
            with self.assertRaises(ValueError):
                core.probabilities(x, w, d, counts)

    def test_oversized_readout_refused_before_delta_allocation(self):
        x, _, _, counts, target = self.arrays()
        oversized = np.broadcast_to(np.float32(0), (256, 2))
        with self.assertRaisesRegex(ValueError, "before allocation"):
            core.fit_arrays(x, oversized, counts, target, check_deadline=lambda: None)

    def test_unrepresentable_or_invalid_hyperparameters_are_refused(self):
        for rate in [0, -1, True, float("nan"), 1e100, 1e-100]:
            x, w, _, counts, target = self.arrays()
            with self.assertRaises(ValueError):
                core.fit_arrays(
                    x, w, counts, target, check_deadline=lambda: None, learning_rate=rate
                )
        x, w, _, counts, target = self.arrays()
        with self.assertRaises(ValueError):
            core.fit_arrays(x, w, counts, target, check_deadline=None)

    def test_no_model_framework_or_file_api_is_needed(self):
        self.assertNotIn("torch", sys.modules)
        self.assertFalse(hasattr(core, "load_model"))
        self.assertFalse(hasattr(core, "write_checkpoint"))


if __name__ == "__main__":
    unittest.main()
