"""Unexecuted stdlib source fakes; no actual/capability/model/array access."""

import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


def load():
    path = Path(__file__).with_name("root-closed-operation-numeric-audit-held.py")
    spec = importlib.util.spec_from_file_location("_operation_numeric_audit_fake", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HeldAuditSourceTests(unittest.TestCase):
    def setUp(self):
        self.audit = load()

    def test_hold_precedes_parser_or_selected_artifact_access(self):
        with (
            patch.object(
                self.audit.argparse, "ArgumentParser", side_effect=AssertionError("parser")
            ),
            patch.object(self.audit, "selected", side_effect=AssertionError("artifact")),
        ):
            for operation in (self.audit.main, self.audit.execute):
                with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
                    operation()
        with patch.object(self.audit, "SOURCE_ONLY", False):
            with self.assertRaisesRegex(ValueError, "remain pending"):
                self.audit.release_gate()

    def test_target_bits_use_native_formatted_control_and_skip_operation(self):
        heads, counts = ["click_target", "operation", "select_target"], [1, 2, 2]
        control = {
            "answers": {
                "click_target": {"probabilities": {"single": 1.0}},
                "operation": {"probabilities": {"op-a": 0.5, "op-b": 0.5}},
                "select_target": {"probabilities": {"first": 0.25, "second": 0.75}},
            }
        }
        raw = copy.deepcopy(control)
        raw["answers"]["operation"]["probabilities"] = {"op-a": 0.9, "op-b": 0.1}
        projection = {
            "ordered_head_ids": heads,
            "base_legal_logits": [[-0.0], [0.0, 0.0], [-1.0, 1.0]],
            "adapted_legal_logits": [[-0.0], [9.0, 0.0], [-1.0, 1.0]],
            "delta_legal_logits": [[0.0], [9.0, 0.0], [0.0, 0.0]],
        }
        self.assertEqual(
            self.audit.target_bits(projection, raw, control, heads, counts),
            ["click_target", "select_target"],
        )
        for kind in ("signed_zero", "target_delta", "probability", "order"):
            changed, response = copy.deepcopy(projection), copy.deepcopy(raw)
            if kind == "signed_zero":
                changed["adapted_legal_logits"][0][0] = 0.0
            elif kind == "target_delta":
                changed["delta_legal_logits"][2][0] = 1e-12
            elif kind == "probability":
                response["answers"]["select_target"]["probabilities"]["first"] += 1e-12
            else:
                changed["ordered_head_ids"] = list(reversed(heads))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.audit.target_bits(changed, response, control, heads, counts)

    def test_fixture_is_literal_source_and_operation_second_not_imported(self):
        actor_path = Path(__file__).with_name("actor_held_v2.py")
        fixture = self.audit.source_fixture(actor_path)
        self.assertEqual(list(fixture["questions"]), list(self.audit.HEADS))
        self.assertEqual([len(q["criteria"]) for q in fixture["questions"].values()], [1, 8, 2, 3])
        self.assertEqual(list(fixture["questions"])[1], "operation")


if __name__ == "__main__":
    unittest.main()
