"""Source-authored stdlib fakes only; deliberately not executed during authoring."""

import copy
import importlib.util
import types
import unittest
from pathlib import Path


def candidate_module():
    path = Path(__file__).with_name("train_metrics_actor_held.py")
    spec = importlib.util.spec_from_file_location("_train_metrics_held_fake", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def binding(module):
    train_ref = {
        "path": str(module.SELECTION),
        "sha256": module.SELECTION_SHA256,
        "size_bytes": 14175,
    }
    selection = {
        "schema": "root-selected-dsl01-committed-train-readout-v1",
        "status": "ROOT_SELECTED_COMMITTED_TRAIN_ONLY",
        "split": "train",
        "model": {"model_id": "vllm-sr/d3-edge", "revision": "fixed-test"},
        "baseline": {"FP32_bytes_sha256": "a" * 64},
    }
    candidate = {
        "schema": "root-accepted-dsl01-train-delta-v1",
        "status": "ROOT_ACCEPTED_TRAIN_ONLY_READOUT",
        "split": "train",
        "training_eligible": True,
        "nontraining": False,
        "profile": "decision3-dsl01-delta-readout-cpu5-4g-candidate-v1",
        "alias": "d3-edge-dsl01-train-delta-cpu5-candidate",
        "owner_profile": "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1",
        "model": copy.deepcopy(selection["model"]),
        "fit_capability": copy.deepcopy(train_ref),
        "baseline_fp32_bytes_sha256": "a" * 64,
        "root_acceptance_ref": {
            "path": str(module.HERE / "root-accepted-trained-readout-01.json"),
            "sha256": module.ACCEPTANCE_SHA256,
        },
        "delta_ref": {
            "sha256": module.DELTA_SHA256,
            "format": "npy_no_pickle",
            "dtype": "float32",
            "shape": [255, 1024],
        },
        "frozen_before_dev_sealed_benchmark_or_browser": True,
        "W0_merge_allowed": False,
    }
    return candidate, selection, train_ref


class MetricsSourceTests(unittest.TestCase):
    def setUp(self):
        self.module = candidate_module()

    def test_hold_precedes_parser_owner_path_and_helper_import(self):
        class Bomb:
            def __fspath__(self):
                raise AssertionError("owner path accessed")

            def __iter__(self):
                raise AssertionError("arguments parsed")

        self.assertTrue(self.module.SOURCE_ONLY)
        with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
            self.module.main(Bomb())
        with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
            self.module.execute(Bomb())
        with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
            self.module.bootstrap_fit()

    def test_exact_two_capability_binding_and_foreign_delta_rejection(self):
        original, selection, train_ref = binding(self.module)
        self.module.validate_candidate(original, selection, train_ref)
        mutations = (
            ("split", "dev"),
            ("nontraining", True),
            ("training_eligible", False),
            ("W0_merge_allowed", True),
            ("model", {"model_id": "foreign"}),
            ("fit_capability", {**train_ref, "sha256": "0" * 64}),
            ("baseline_fp32_bytes_sha256", "b" * 64),
            ("delta_ref", {**original["delta_ref"], "sha256": "c" * 64}),
            ("delta_ref", {**original["delta_ref"], "shape": [254, 1024]}),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                candidate = copy.deepcopy(original)
                candidate[key] = value
                with self.assertRaisesRegex(ValueError, "binding differs"):
                    self.module.validate_candidate(candidate, selection, train_ref)

    def test_accuracy_uses_all_positive_targets_and_first_ordered_tie(self):
        pad = [0.0] * 252
        probabilities = [[0.4, 0.4, 0.2] + pad, [0.1, 0.7, 0.2] + pad, [1.0, 0.0, 0.0] + pad]
        targets = [[0.5, 0.0, 0.5] + pad, [0.5, 0.0, 0.5] + pad, [1.0, 0.0, 0.0] + pad]
        result = self.module.valid_target_accuracy(probabilities, targets, [3, 3, 1])
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["accuracy"], 2 / 3)
        # An illegal column cannot become the argmax, even if a fake gives it
        # larger mass. Genuine core additionally enforces exactly zero mass.
        probabilities[2][2] = 100.0
        self.assertEqual(
            self.module.valid_target_accuracy(probabilities, targets, [3, 3, 1])["correct"], 2
        )

    def test_readonly_retention_fingerprint_rejects_mutation_and_writable_inputs(self):
        class Array:
            def __init__(self):
                self.flags = types.SimpleNamespace(writeable=False)
                self.dtype, self.shape, self.value = "float32", (1, 1), b"before"

            def tobytes(self, order):
                assert order == "C"
                return self.value

        array = Array()
        before = self.module.array_fingerprints({"baseline": array})
        array.value = b"after"
        self.assertNotEqual(self.module.array_fingerprints({"baseline": array}), before)
        array.flags.writeable = True
        with self.assertRaisesRegex(ValueError, "readonly"):
            self.module.array_fingerprints({"baseline": array})


if __name__ == "__main__":
    unittest.main()
