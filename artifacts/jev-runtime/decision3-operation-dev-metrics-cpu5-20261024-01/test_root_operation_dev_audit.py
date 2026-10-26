"""Invented stdlib witnesses; no actual bundle, model or labels are read."""

import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).with_name("root-closed-operation-dev-metrics-audit-held.py")
spec = importlib.util.spec_from_file_location("_operation_dev_root_witness", PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def invented(heads):
    x = [1.0 if j % 1024 == 0 else 0.0 for j in range(len(heads) * 1024)]
    w0 = [0.0] * (255 * 1024)
    w0[0] = 1.0
    delta = [0.0] * (255 * 1024)
    delta[1024] = 3.0
    targets = [1.0 if j % 255 == 0 else 0.0 for j in range(len(heads) * 255)]
    cap = {"rows": [{"head_id": head, "ordered_criteria_ids": ["a", "b"]} for head in heads]}
    return cap, x, w0, delta, [2] * len(heads), targets


class OperationWitnessTests(unittest.TestCase):
    def test_only_actual_operation_changes_when_it_is_not_first(self):
        args = invented(["select_target", "operation", "click_target"])
        baseline, base_losses = audit.witness(*args, False)
        candidate, losses = audit.witness(*args, True)
        self.assertEqual(baseline, [True, True, True])
        self.assertEqual(candidate, [True, False, True])
        self.assertEqual(losses[0], base_losses[0])
        self.assertEqual(losses[2], base_losses[2])
        self.assertGreater(losses[1], base_losses[1])

    def test_absent_operation_retains_every_original_result(self):
        args = invented(["click_target", "type_text_target", "select_target"])
        self.assertEqual(audit.witness(*args, False), audit.witness(*args, True))

    def test_held_audit_stops_before_arguments_or_actual_reads(self):
        with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
            audit.main([])


if __name__ == "__main__":
    unittest.main()
