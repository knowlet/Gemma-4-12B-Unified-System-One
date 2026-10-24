"""Authored stdlib-only fakes; not executed during this source-only pass.

No runtime module, NumPy, Torch, fixture data, capability or delta is loaded.
These bounded pure checks cannot establish real tensor or runtime qualification.
"""

import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


def load():
    path = Path(__file__).with_name("actor.py")
    spec = importlib.util.spec_from_file_location("_held_delta_numeric_fake", path)
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

    def test_two_rows_keep_original_state_option_ids_and_singleton_without_dummy(self):
        request = {"state": {"invented": [1, 2]}, "questions": {}}
        for key, count in zip(self.actor.HEADS, (8, 1, 2, 3)):
            request["questions"][key] = {
                "type": "choice",
                "criteria": {f"offered-{i}": str(i) for i in range(count)},
            }
        before = copy.deepcopy(request)
        selected = self.actor.projection(request, 2)
        self.assertEqual(list(selected["questions"]), ["operation", "click_target"])
        self.assertEqual([len(q["criteria"]) for q in selected["questions"].values()], [8, 1])
        self.assertEqual(selected["state"], request["state"])
        selected["state"]["invented"].append(3)
        self.assertEqual(request, before)
        with self.assertRaises(RuntimeError):
            self.actor.projection(request, 3)

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


if __name__ == "__main__":
    unittest.main()
