"""Unexecuted pure stdlib fakes; no actual/capability/DOM/array access."""

import importlib.util
import math
import unittest
from pathlib import Path
from unittest.mock import patch


def load():
    path = Path(__file__).with_name("root-closed-operation-browser-audit-held.py")
    spec = importlib.util.spec_from_file_location("_operation_browser_audit_source_fake", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report():
    return {
        "initial_judge": {
            "url": "http://127.0.0.1/fictional",
            "active": "none",
            "clicks": [],
            "buttons": [
                {"id": key, "visible": True, "disabled": False} for key in ("copper", "silver")
            ],
        },
        "final_judge": {
            "url": "http://127.0.0.1/fictional",
            "active": "copper",
            "text": "Copper lamp armed",
            "visible": True,
            "last_trusted": True,
            "clicks": [{"id": "copper", "trusted": True}],
        },
        "goal_satisfied": True,
        "success": True,
        "final_status": "done",
        "final_snapshot": {"status": "done"},
        "error": None,
        "secondary_errors": [],
        "source_unchanged": True,
    }


class BrowserAuditSourceTests(unittest.TestCase):
    def setUp(self):
        self.audit = load()

    def test_source_hold_and_pending_refs_precede_any_artifact_or_parser(self):
        with (
            patch.object(
                self.audit.argparse, "ArgumentParser", side_effect=AssertionError("parser")
            ),
            patch.object(Path, "read_bytes", side_effect=AssertionError("artifact")),
        ):
            for operation in (self.audit.main, self.audit.execute):
                with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
                    operation()
        with patch.object(self.audit, "SOURCE_ONLY", False):
            with self.assertRaisesRegex(ValueError, "remain pending"):
                self.audit.release_gate()

    def test_strict_goal_requires_one_trusted_click_and_real_done(self):
        self.assertEqual(self.audit.strict_result(report()), (True, True))
        for kind in ("wrong_before_correct", "untrusted", "blocked", "budget", "premature_done"):
            changed = report()
            if kind == "wrong_before_correct":
                changed["final_judge"]["clicks"].insert(0, {"id": "silver", "trusted": True})
            elif kind == "untrusted":
                changed["final_judge"]["clicks"][0]["trusted"] = False
            elif kind in ("blocked", "budget"):
                changed["final_status"] = "blocked" if kind == "blocked" else "running"
                changed["final_snapshot"]["status"] = changed["final_status"]
            else:
                changed["final_judge"].update(
                    active="none", clicks=[], text="Neither lamp armed", last_trusted=False
                )
            goal = kind in ("blocked", "budget")
            changed.update(goal_satisfied=goal, success=False)
            with self.subTest(kind=kind):
                self.assertEqual(self.audit.strict_result(changed), (goal, False))
                changed["success"] = True
                with self.assertRaisesRegex(ValueError, "strict success"):
                    self.audit.strict_result(changed)

    def test_initial_controls_must_both_be_visible_enabled(self):
        for field, value in (("visible", False), ("disabled", True)):
            changed = report()
            changed["initial_judge"]["buttons"][1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit.strict_result(changed)

    def test_projection_uses_actual_head_id_and_preserves_target_bits(self):
        zero = [0.0, 0.0]
        p = [
            math.exp(1) / (math.exp(1) + math.exp(-1)),
            math.exp(-1) / (math.exp(1) + math.exp(-1)),
        ]
        self.audit.projection_witness("operation", 2, zero, [1.0, -1.0], [1.0, -1.0], p)
        self.audit.projection_witness("click_target", 1, [-0.0], [0.0], [-0.0], [1.0])
        for delta, adapted in (([1e-12], [-0.0]), ([0.0], [0.0])):
            with self.assertRaisesRegex(ValueError, "target"):
                self.audit.projection_witness("click_target", 1, [-0.0], delta, adapted, [1.0])
        with self.assertRaises(ValueError):
            self.audit.projection_witness("select_target", 2, zero, [1.0, -1.0], [1.0, -1.0], p)


if __name__ == "__main__":
    unittest.main()
