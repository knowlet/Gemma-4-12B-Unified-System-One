"""Authored stdlib-only fakes; not executed during this source-only pass.

No runtime module, NumPy, Torch, fixture data, capability or delta is loaded.
These bounded pure checks cannot establish real tensor or runtime qualification.
"""

import copy
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load():
    path = Path(__file__).with_name("actor_held_v2.py")
    spec = importlib.util.spec_from_file_location("_held_operation_numeric_v2_fake", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SourceAndPureContractTests(unittest.TestCase):
    def setUp(self):
        self.actor = load()

    def test_source_hold_precedes_parser_source_plan_and_model_access(self):
        with (
            patch.object(
                self.actor.argparse, "ArgumentParser", side_effect=AssertionError("parser")
            ),
            patch.object(self.actor, "load_plan", side_effect=AssertionError("plan")),
            patch.object(self.actor, "source", side_effect=AssertionError("dependency")),
        ):
            for operation in (
                lambda: self.actor.main([]),
                lambda: self.actor.execute(None, None, None),
            ):
                with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
                    operation()

    def test_changing_only_hold_cannot_release_pending_source_pins(self):
        with patch.object(self.actor, "SOURCE_ONLY", False):
            with self.assertRaisesRegex(RuntimeError, "pins remain pending"):
                self.actor.release_gate()

    def test_only_four_original_calls_and_fixed_budgets(self):
        self.assertEqual(
            self.actor.CALLS, (("original", 4), ("candidate", 4), ("original", 2), ("candidate", 2))
        )
        self.assertEqual((self.actor.WALL_SECONDS, self.actor.CALL_SECONDS), (120, 20))
        self.assertEqual((self.actor.ABS_TOL, self.actor.REL_TOL), (1e-6, 1e-5))
        self.assertEqual(self.actor.OWNER_CONTRACT["cpu_affinity"], [15, 16, 17, 18, 19])
        self.assertEqual(self.actor.OWNER_CONTRACT["cpu_quota_percent"], 500)
        self.assertEqual(self.actor.OWNER_CONTRACT["memory_max_bytes"], 4294967296)
        self.assertEqual(
            self.actor.HEADS, ("click_target", "operation", "select_target", "type_text_target")
        )
        self.assertEqual(self.actor.COUNTS, (1, 8, 2, 3))
        self.assertIsNone(self.actor.PINS["backend"])
        self.assertNotIn("fixture", self.actor.DEPENDENCIES)

    def test_two_rows_keep_original_state_option_ids_and_singleton_without_dummy(self):
        request = {"state": {"invented": [1, 2]}, "questions": {}}
        for key, count in zip(self.actor.HEADS, self.actor.COUNTS):
            request["questions"][key] = {
                "type": "choice",
                "criteria": {f"offered-{i}": str(i) for i in range(count)},
            }
        before = copy.deepcopy(request)
        selected = self.actor.projection(request, 2)
        self.assertEqual(list(selected["questions"]), ["click_target", "operation"])
        self.assertEqual([len(q["criteria"]) for q in selected["questions"].values()], [1, 8])
        self.assertEqual(selected["state"], request["state"])
        selected["state"]["invented"].append(3)
        self.assertEqual(request, before)
        with self.assertRaises(RuntimeError):
            self.actor.projection(request, 3)

    def test_fixture_is_source_defined_without_gold_and_not_shared_mutable(self):
        first = self.actor.fixture_request()
        second = self.actor.fixture_request()
        self.assertEqual(list(first), ["state", "questions"])
        self.assertEqual(list(first["questions"]), list(self.actor.HEADS))
        self.assertEqual([len(q["criteria"]) for q in first["questions"].values()], [1, 8, 2, 3])
        self.assertTrue(all(q["type"] == "choice" for q in first["questions"].values()))
        first["questions"]["operation"]["criteria"].pop("DONE")
        self.assertIn("DONE", second["questions"]["operation"]["criteria"])

    def test_only_new_closed_operation_training_acceptance_schema_is_accepted(self):
        accepted = {
            "schema": "root-accepted-dsl01-operation-trained-readout-v1",
            "status": "accepted",
        }
        self.actor.validate_training_acceptance(accepted)
        for key, value in (
            ("schema", "root-accepted-dsl01-trained-readout-v1"),
            ("schema", "root-committed-train-operation-fit-v1"),
            ("status", "ROOT_COMMITTED_TRAIN_OPERATION_FIT"),
            ("status", "PENDING"),
        ):
            changed = {**accepted, key: value}
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                self.actor.validate_training_acceptance(changed)

    def test_actual_target_logit_signed_zero_and_probability_changes_are_not_tolerated(self):
        keys, counts = ["click_target", "operation", "select_target"], [1, 2, 2]
        native = [[1.0], [0.5, 0.5], [0.25, 0.75]]
        raw = {
            "answers": {
                key: {"probabilities": dict(enumerate(row))} for key, row in zip(keys, native)
            }
        }
        projection = {
            "ordered_head_ids": keys,
            "base_legal_logits": [[-0.0], [0.0, 0.0], [-1.0, 1.0]],
            "adapted_legal_logits": [[-0.0], [100.0, 0.0], [-1.0, 1.0]],
            "delta_legal_logits": [[0.0], [100.0, 0.0], [0.0, 0.0]],
        }
        checked = self.actor.target_preservation(raw, native, keys, counts, projection)
        self.assertEqual(checked["preserved_head_ids"], ["click_target", "select_target"])
        for kind in ("signed_zero", "target_delta", "probability", "reordered"):
            changed, response = copy.deepcopy(projection), copy.deepcopy(raw)
            if kind == "signed_zero":
                changed["adapted_legal_logits"][0][0] = 0.0
            elif kind == "target_delta":
                changed["delta_legal_logits"][2][0] = 1e-12
            elif kind == "probability":
                response["answers"]["select_target"]["probabilities"][0] += 1e-12
            else:
                changed["ordered_head_ids"] = list(reversed(keys))
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                self.actor.target_preservation(response, native, keys, counts, changed)

    def test_required_final_retention_failure_and_primary_identity_are_preserved(self):
        error = KeyboardInterrupt("original forward interruption")
        secondary = OSError("required retention failed")

        def fail(*_args):
            raise secondary

        helper = SimpleNamespace(finalize_state=fail)
        with self.assertRaises(OSError) as raised:
            self.actor.finalize_state(helper, Path("/not-read"), {}, None)
        self.assertIs(raised.exception, secondary)
        self.actor.finalize_state(helper, Path("/not-read"), {}, error)
        self.assertIn("State durability also failed", error.__notes__[0])

    def test_original_respond_normalization_precedes_exact_target_comparison(self):
        keys, counts = ["click_target", "operation", "select_target"], [1, 2, 2]
        criteria = {key: [f"offered-{i}" for i in range(count)] for key, count in zip(keys, counts)}
        prepared = SimpleNamespace(
            keys=keys, questions={key: SimpleNamespace(keys=criteria[key]) for key in keys}
        )
        lengths = dict(zip(keys, [3, 5, 7]))
        original_replay = [[0.2], [0.2, 0.3], [0.2, 0.3]]
        before = copy.deepcopy(original_replay)
        calls = []

        def respond(actual_prepared, values, tokens):
            self.assertIs(actual_prepared, prepared)
            self.assertEqual(list(values), keys)
            self.assertEqual(values, dict(zip(keys, original_replay)))
            self.assertEqual(tokens, 15)
            calls.append(tokens)
            answers = {}
            for key in keys:
                floats = [float(value) for value in values[key]]
                total = sum(floats)
                answers[key] = {
                    "probabilities": dict(zip(criteria[key], [v / total for v in floats]))
                }
            return {"model": "invented-original-native", "answers": answers}

        runtime = SimpleNamespace(
            respond=respond,
            logits=lambda *_args: (_ for _ in ()).throw(AssertionError("extra forward")),
        )
        control, normalized = self.actor.native_response_control(
            runtime, prepared, keys, lengths, original_replay
        )
        self.assertEqual(calls, [15])
        self.assertEqual(original_replay, before)
        self.assertEqual(normalized, [[1.0], [0.4, 0.6], [0.4, 0.6]])
        raw = copy.deepcopy(control)
        # A candidate may change operation; target output must remain exact.
        raw["answers"]["operation"]["probabilities"] = dict(zip(criteria["operation"], [0.9, 0.1]))
        projection = {
            "ordered_head_ids": keys,
            "base_legal_logits": [[-0.0], [0.0, 0.0], [-1.0, 1.0]],
            "adapted_legal_logits": [[-0.0], [100.0, 0.0], [-1.0, 1.0]],
            "delta_legal_logits": [[0.0], [100.0, 0.0], [0.0, 0.0]],
        }
        self.assertTrue(
            self.actor.target_preservation(raw, normalized, keys, counts, projection)[
                "target_native_probabilities_bitwise_unchanged"
            ]
        )
        with self.assertRaises(RuntimeError):
            self.actor.target_preservation(raw, original_replay, keys, counts, projection)
        changed = copy.deepcopy(raw)
        changed["answers"]["select_target"]["probabilities"][criteria["select_target"][0]] += 1e-15
        with self.assertRaises(RuntimeError):
            self.actor.target_preservation(changed, normalized, keys, counts, projection)

    def test_all_legal_logit_rows_and_fixed_tolerance(self):
        expected = {
            key: [[1.0, -2.0], [0.0]]
            for key in ("base_legal_logits", "delta_legal_logits", "adapted_legal_logits")
        }
        self.assertTrue(
            self.actor.compare_logits(copy.deepcopy(expected), expected, [2, 1])[
                "all_legal_logits_matched"
            ]
        )
        for kind in ("missing", "nonfinite", "out_of_tolerance"):
            changed = copy.deepcopy(expected)
            if kind == "missing":
                changed["base_legal_logits"][1] = []
            elif kind == "nonfinite":
                changed["base_legal_logits"][1][0] = float("nan")
            else:
                changed["adapted_legal_logits"][0][0] = 1.001
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                self.actor.compare_logits(changed, expected, [2, 1])

    def test_singleton_and_uniform_entropy_are_mathematical_without_labels(self):
        questions = {
            "singleton": {"criteria": {"only": "invented"}},
            "uniform": {"criteria": {"second-offered": "A", "first-named": "B"}},
        }
        raw = {
            "answers": {
                "singleton": {"probabilities": {"only": 1.0}, "confidence": 1.0},
                "uniform": {
                    "probabilities": {"second-offered": 0.5, "first-named": 0.5},
                    "confidence": 0.0,
                },
            }
        }
        result = self.actor.entropy_witness(raw, questions)
        self.assertTrue(result["all_entropy_confidences_matched"])
        self.assertEqual([row["singleton"] for row in result["rows"]], [True, False])
        for value in (0.0, True, float("nan")):
            changed = copy.deepcopy(raw)
            changed["answers"]["singleton"]["confidence"] = value
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.actor.entropy_witness(changed, questions)

    def test_dsl_inputs_are_rejected_before_source_or_payload_open(self):
        with self.assertRaisesRegex(RuntimeError, "DSL data"):
            self.actor.ordinary(self.actor.DSL / "corpus/data/train/requests.jsonl")


class IndependentReplaySourceFakes(unittest.TestCase):
    """Small-width arithmetic fakes, not genuine 1024-wide Torch qualification."""

    def test_operation_second_mixed_counts_real_tails_and_absent_operation(self):
        actor = load()
        root = Path(__file__).resolve().parents[3]
        fake_source = (
            root
            / "artifacts/jev-runtime/decision3-operation-head-backend-cpu5-20261024-01/test_operation_backend_held.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_operation_numeric_matrix_source", fake_source
        )
        fake = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fake)

        class Matrix(fake.Matrix):
            def float(self):
                return self

            def clone(self):
                return Matrix(self.values)

            def __getitem__(self, key):
                if isinstance(key, slice):
                    return Matrix(self.values[key])
                return super().__getitem__(key)

        baseline = Matrix([[float(code), 0.0] for code in range(255)])
        delta = Matrix([[20.0 if code == 0 else 0.0, 0.0] for code in range(255)])
        for keys, counts in (
            (["click_target", "operation", "select_target", "type_text_target"], [1, 3, 2, 3]),
            (["click_target", "operation"], [1, 3]),
            (["select_target"], [2]),
        ):
            features = Matrix([[float(i + 1), 0.0] for i in range(len(keys))])
            before = copy.deepcopy((features.values, baseline.values, delta.values))
            fake.Matrix.matmul_calls = []
            masked, values, proof = actor.same_forward_replay(
                fake.Torch, features, baseline, delta, counts, keys
            )
            self.assertEqual(proof["ordered_head_ids"], keys)
            self.assertTrue(proof["target_full_logits_bitwise_unchanged"])
            self.assertEqual([len(row) for row in values], counts)
            self.assertEqual(len(masked.values), len(keys))
            self.assertTrue(
                all(
                    all(v == float("-inf") for v in row[n:])
                    for row, n in zip(masked.values, counts)
                )
            )
            self.assertEqual(len(fake.Matrix.matmul_calls), 1 + keys.count("operation"))
            for index, key in enumerate(keys):
                if key == "operation":
                    self.assertEqual(max(range(counts[index]), key=values[index].__getitem__), 0)
                else:
                    self.assertEqual(
                        proof["base_legal_logits"][index], proof["adapted_legal_logits"][index]
                    )
                    self.assertEqual(proof["delta_legal_logits"][index], [0.0] * counts[index])
            self.assertEqual((features.values, baseline.values, delta.values), before)


if __name__ == "__main__":
    unittest.main()
