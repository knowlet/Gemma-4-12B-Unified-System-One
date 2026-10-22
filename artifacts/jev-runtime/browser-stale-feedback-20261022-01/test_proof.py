import copy
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "stale_feedback_proof", Path(__file__).with_name("proof.py")
)
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


class OriginalAgentMechanismTests(unittest.TestCase):
    def test_rejected_stale_page_repeats_input_without_execution_history(self):
        result = proof.isolated_fixture(ticks=3)
        self.assertEqual(result["act_attempts"], 3)
        self.assertEqual(result["executions"], 0)
        self.assertEqual(len({proof.canonical(row) for row in result["requests"]}), 1)
        self.assertEqual(len(result["snapshots"][-1]["decisions"]), 3)
        for row in result["snapshots"]:
            self.assertEqual(row["history"], result["initial_history"])
            self.assertEqual(row["status"], "ready")
            self.assertIsNone(row["decision"])

    def test_changed_observation_changes_the_request(self):
        result = proof.isolated_fixture("changed_page", ticks=2)
        self.assertEqual(len({proof.canonical(row) for row in result["requests"]}), 2)
        self.assertEqual(result["executions"], 0)

    def test_post_input_stale_observation_preserves_executed_history(self):
        result = proof.isolated_fixture("post_input_stale", ticks=1)
        self.assertEqual(result["executions"], 1)
        self.assertEqual(len(result["snapshots"][0]["history"]), 1)
        self.assertIsNone(result["snapshots"][0]["decision"])

    def test_separate_feedback_preserves_candidates_questions_and_recent_actions(self):
        original = proof.isolated_fixture(ticks=1)["requests"][0]
        result = proof.with_proposed_attempt_context(
            original,
            operation="CLICK",
            target="1",
            recorded_action_delta=0,
            same_page=True,
            consecutive_count=1,
        )
        self.assertEqual(result["questions"], original["questions"])
        for key in ["elements", "recent_actions", "page"]:
            self.assertEqual(result["state"][key], original["state"][key])
        restored = copy.deepcopy(result)
        del restored["state"]["last_attempt"]
        self.assertEqual(restored, original)
        self.assertNotEqual(proof.canonical(result), proof.canonical(original))
        self.assertNotIn("physical_cause", result["state"]["last_attempt"])


if __name__ == "__main__":
    unittest.main()
