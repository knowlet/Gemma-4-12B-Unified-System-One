"""Invented stdlib-only campaign protocol tests; no services or actual data."""

import importlib.util
import json
import os
import stat
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def source():
    path = Path(__file__).with_name("campaign_held.py")
    spec = importlib.util.spec_from_file_location("_fake_matched_campaign", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.module = source()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.module.ROOT = self.root

    def test_hold_precedes_parser_or_plan_io(self):
        with (
            patch.object(self.module.argparse, "ArgumentParser") as parser,
            patch.object(Path, "read_bytes", side_effect=AssertionError("unexpected IO")),
        ):
            with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
                self.module.main(["--run"])
            parser.assert_not_called()
        with patch.object(self.module, "SOURCE_ONLY", False):
            with self.assertRaisesRegex(RuntimeError, "pins remain pending"):
                self.module.main([])

    def test_combines_unchanged_original_false_outcomes(self):
        inputs = [
            {
                "attempts": 1,
                "successes": 0,
                "goals_satisfied": 0,
                "premature_done_count": 1,
                "policy_calls": 2,
                "text_calls": 1,
                "outcomes": [{"task_id": case, "success": False}],
            }
            for case in self.module.KNOWN4
        ]
        result = self.module.combined_summary(inputs)
        self.assertEqual(
            (result["attempts"], result["successes"], result["success_rate"]), (4, 0, 0)
        )
        self.assertEqual((result["policy_calls"], result["text_calls"]), (8, 4))
        self.assertEqual(result["outcomes"], [row["outcomes"][0] for row in inputs])

    def test_handoff_becomes_visible_only_after_complete_file_fsync(self):
        target = self.root / "request.json"
        original_fsync = os.fsync
        phases = []

        def fsync(fd):
            if stat.S_ISREG(os.fstat(fd).st_mode):
                self.assertFalse(target.exists())
                phases.append("durable-before-visible")
            else:
                self.assertEqual(json.loads(target.read_bytes()), {"case": "invented"})
                phases.append("published-directory")
            return original_fsync(fd)

        with patch.object(self.module.os, "fsync", side_effect=fsync):
            self.module.write(target, {"case": "invented"})
        self.assertEqual(phases, ["durable-before-visible", "published-directory"])
        with self.assertRaises(FileExistsError):
            self.module.write(target, {"replacement": "must not win"})
        self.assertEqual(json.loads(target.read_bytes()), {"case": "invented"})
        self.assertFalse(list(self.root.glob("*.pending")))

    def test_late_failure_has_independent_terminal_evidence(self):
        receipt = {"stage": "complete_pending_external_closure", "error": None}
        self.module.write(self.root / "receipt.json", receipt)
        primary = TimeoutError("invented final retention timeout")
        self.module.retain_finalization_failure(self.root, primary)
        self.assertEqual(json.loads((self.root / "receipt.json").read_bytes()), receipt)
        failure = json.loads((self.root / "finalization-failure.json").read_bytes())
        self.assertFalse(failure["complete_matched_score_authorized"])
        self.assertEqual(failure["error"]["message"], str(primary))
        self.assertEqual(
            failure["earlier_receipt_ref"], self.module.reference(self.root / "receipt.json")
        )

    def test_failed_failure_evidence_keeps_original_error(self):
        primary, secondary = (
            TimeoutError("invented first timeout"),
            OSError("invented failed sidecar"),
        )
        with patch.object(self.module, "write", side_effect=secondary):
            self.module.retain_finalization_failure(self.root, primary)
        self.assertTrue(
            any("failure finalization evidence also failed" in note for note in primary.__notes__)
        )

    def protocol(self, *, failed_caller=None, bad_closure=None):
        module = self.module
        output, handoff = self.root / "output", self.root / "handoff"
        output.mkdir()
        for arm in ("baseline", "candidate"):
            (output / arm).mkdir()
        for index in range(1, 11):
            (handoff / f"case-{index:02d}").mkdir(parents=True)
        calls, completed, events = [], [], []
        root_sha = "a" * 64
        plan = {
            "upstream": "upstream",
            "chromium": "invented-chromium",
            "matched": {"handoff_root": str(handoff)},
        }
        original = OSError("invented original caller failure")

        def call_case(case_id, directory, **kwargs):
            arm = directory.parent.name
            index = module.CASE_IDS.index(case_id) + 1
            if arm == "candidate" and index > 1:
                prior = handoff / f"case-{index - 1:02d}/stop.json"
                self.assertTrue(prior.exists())
                self.assertIn(("closed", index - 1), events)
            calls.append((arm, case_id, kwargs))
            directory.mkdir()
            summary = {
                "attempts": 1,
                "successes": 0,
                "goals_satisfied": 0,
                "policy_calls": 2,
                "text_calls": 1,
                "outcomes": [{"task_id": case_id, "success": False}],
            }
            summary_path = directory / "summary.json"
            summary_path.write_text(json.dumps(summary))
            record = {
                "case_id": case_id,
                "group": "known4" if index <= 4 else "public6",
                "task_success": False,
                "infrastructure_complete": True,
                "summary_ref": module.reference(summary_path),
            }
            record["receipt_ref"] = module.write(directory / "caller-receipt.json", record)
            if arm == "candidate" and failed_caller == index:
                original.jev_v7_case_record = record
                raise original
            return record

        def aggregate(records):
            self.assertEqual([row["case_id"] for row in records], list(module.CASE_IDS))
            return {
                "infrastructure_complete": True,
                "successes": 0,
                "denominator": 10,
                "success_rate": 0,
                "completed_cases": 10,
            }

        def wait(path, deadline, *, seconds):
            deadline()
            index = int(path.parent.name.split("-")[-1])
            case_id = module.CASE_IDS[index - 1]
            item = {
                "root_plan_sha256": root_sha,
                "case_index": index,
                "case_id": case_id,
                "plan_sha256": "b" * 64,
                "nonce": "c" * 32,
                "owner_output": str(self.root / f"cpu-{index}"),
            }
            if path.name == "ready.json":
                self.assertTrue((path.parent / "request.json").exists())
                item.update(
                    schema="jev-v7-matched-case-ready-v1",
                    model=module.ALIAS,
                    endpoint="http://127.0.0.1:12345/v1/systemone",
                )
            else:
                stop = json.loads((path.parent / "stop.json").read_bytes())
                self.assertEqual(stop["plan_sha256"], item["plan_sha256"])
                events.append(("closed", index))
                item.update(
                    schema="jev-v7-matched-case-closed-v1",
                    controller_exit_code=0,
                    owner_completed=True,
                    original_scoped_closure_positive=True,
                )
                if bad_closure == index:
                    item["nonce"] = "d" * 32
            return item

        cases = module.MatchedCases(
            owner=object(),
            plan=plan,
            root_sha=root_sha,
            directory=self.root,
            output=output,
            caller=types.SimpleNamespace(call_case=call_case, aggregate_cases=aggregate),
            lease=object(),
            verify_owned=lambda: None,
            deadline=lambda: None,
        )
        return cases, calls, completed, events, wait, original

    def test_false_tasks_complete_both_exact_ten_arms_with_same_helper(self):
        cases, calls, completed, _events, wait, _original = self.protocol()
        with patch.object(self.module, "wait_json", side_effect=wait):
            cases.run(
                "unused",
                self.root / "upstream",
                "http://127.0.0.1:23456",
                "invented-helper-token",
                "invented-chromium",
                completed=completed.append,
            )
        self.assertEqual(
            [(arm, case_id) for arm, case_id, _ in calls],
            [
                (arm, case_id)
                for arm in ("baseline", "candidate")
                for case_id in self.module.CASE_IDS
            ],
        )
        for arm, _case_id, kwargs in calls:
            self.assertEqual(kwargs["text_endpoint"], "http://127.0.0.1:23456")
            self.assertEqual(kwargs["text_api_key"], "invented-helper-token")
            self.assertEqual(kwargs["text_model"], "generated-actions-nf4")
            self.assertEqual(
                kwargs["policy_api_key"], "invented-helper-token" if arm == "baseline" else None
            )
        self.assertEqual(
            [(row["group"], row["summary"]["attempts"]) for row in completed],
            [("known4", 4), ("public6", 6)],
        )
        self.assertEqual([row["successes"] for row in cases.arms()], [0, 0])

    def test_invalid_cpu_closure_stops_before_second_candidate(self):
        cases, calls, completed, _events, wait, _original = self.protocol(bad_closure=1)
        with patch.object(self.module, "wait_json", side_effect=wait):
            with self.assertRaisesRegex(RuntimeError, "nonce/output closure"):
                cases.run(
                    "unused",
                    self.root / "upstream",
                    "http://127.0.0.1:23456",
                    "invented-token",
                    "invented-chromium",
                    completed=completed.append,
                )
        self.assertEqual(len(calls), 11)
        self.assertEqual(len(cases.children), 0)

    def test_caller_error_still_stops_service_and_preserves_primary(self):
        cases, calls, completed, events, wait, original = self.protocol(failed_caller=1)
        with patch.object(self.module, "wait_json", side_effect=wait):
            with self.assertRaises(OSError) as caught:
                cases.run(
                    "unused",
                    self.root / "upstream",
                    "http://127.0.0.1:23456",
                    "invented-token",
                    "invented-chromium",
                    completed=completed.append,
                )
        self.assertIs(caught.exception, original)
        self.assertEqual(len(calls), 11)
        self.assertIn(("closed", 1), events)
        self.assertTrue((self.root / "handoff/case-01/stop.json").exists())


if __name__ == "__main__":
    unittest.main()
