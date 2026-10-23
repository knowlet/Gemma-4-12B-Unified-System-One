"""Stdlib source/data boundary tests; no predictions, browser or ML imports."""

import copy
import pathlib
import tempfile
import unittest

from audit import audit_cases, check_role_selection
from dsl import DSL_VERSION, FAMILIES, OPERATIONS, digest, generate_cases, labels_for, solve
from generate import artifacts, publish
from request_builder import SOURCE_PINS, capture_request


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = generate_cases()

    def test_exact_population_ancestry_and_position_counts(self):
        report = audit_cases(self.cases)
        self.assertEqual(report["counts"], {"train": 96, "dev": 24, "sealed": 48})
        self.assertEqual(report["ancestor_counts"], {"train": 12, "dev": 3, "sealed": 6})
        self.assertEqual(report["family_counts"], {"train": 4, "dev": 1, "sealed": 2})
        self.assertEqual(
            report["operation_position_total_counts"],
            {"train": [12] * 8, "dev": [3] * 8, "sealed": [6] * 8},
        )

    def test_generation_reproducible_without_global_rng_or_environment(self):
        self.assertEqual(self.cases, generate_cases())

    def test_family_variants_never_cross_split(self):
        broken = copy.deepcopy(self.cases)
        broken[0]["split"] = "sealed"
        with self.assertRaisesRegex(ValueError, "Family crossed"):
            audit_cases(broken)

    def test_ancestor_reparenting_refused(self):
        broken = copy.deepcopy(self.cases)
        broken[0]["ancestor_id"] = "equipment_checkout/workflow-1"
        with self.assertRaisesRegex(ValueError, "Ancestor changed"):
            audit_cases(broken)

    def test_unknown_ancestor_refused(self):
        broken = copy.deepcopy(self.cases)
        broken[0]["ancestor_id"] = "appointment_edit/workflow-99"
        with self.assertRaisesRegex(ValueError, "Unknown workflow"):
            audit_cases(broken)

    def test_request_reordering_after_labels_refused(self):
        broken = copy.deepcopy(self.cases)
        question = broken[0]["request"]["questions"]["operation"]
        question["criteria"] = dict(reversed(list(question["criteria"].items())))
        with self.assertRaisesRegex(ValueError, "Actual request changed"):
            audit_cases(broken)

    def test_complete_denied_state_and_label_swap_refused(self):
        broken = copy.deepcopy(self.cases)
        complete = next(
            c
            for c in broken
            if c["ancestor_id"] == "appointment_edit/workflow-1" and c["branch_stage"] == "complete"
        )
        denied = next(
            c
            for c in broken
            if c["ancestor_id"] == complete["ancestor_id"] and c["branch_stage"] == "denied"
        )
        complete["state"], denied["state"] = denied["state"], complete["state"]
        complete["labels"], denied["labels"] = denied["labels"], complete["labels"]
        with self.assertRaisesRegex(ValueError, "Semantic state/page/goal/control binding"):
            audit_cases(broken)

    def test_rehashed_question_head_reordering_refused(self):
        broken = copy.deepcopy(self.cases)
        case = broken[0]
        case["request"]["questions"] = dict(reversed(list(case["request"]["questions"].items())))
        case["request_sha256"] = digest(case["request"])
        case["native_payload_sha256"] = digest(
            {k: case["request"][k] for k in ("state", "questions")}
        )
        with self.assertRaisesRegex(ValueError, "Original question head order changed"):
            audit_cases(broken)

    def test_fixed_position_or_wrong_target_refused(self):
        broken = copy.deepcopy(self.cases)
        label = broken[0]["labels"]["click_target"]
        wrong = next(k for k in label["criteria_ids"] if k not in label["valid_ids"])
        label["valid_ids"] = [wrong]
        label["valid_positions"] = [label["criteria_ids"].index(wrong)]
        label["soft_target"] = [float(k == wrong) for k in label["criteria_ids"]]
        with self.assertRaisesRegex(ValueError, "semantic control"):
            audit_cases(broken)

    def test_unused_target_head_is_not_supervised(self):
        done = next(c for c in self.cases if c["branch_stage"] == "complete")
        self.assertEqual(set(done["labels"]), {"operation"})
        self.assertEqual(len(done["request"]["questions"]), 4)

    def test_real_preconditions_not_first_offered_option(self):
        changed = False
        for c in self.cases:
            label = c["labels"]["operation"]
            changed |= label["valid_positions"] != [0]
            self.assertEqual(
                label["valid_ids"],
                [
                    op
                    for op in c["request"]["questions"]["operation"]["criteria"]
                    if op in {v[0] for v in solve(c["state"])}
                ],
            )
        self.assertTrue(changed)

    def test_parallel_requirements_retain_both_valid_actions(self):
        state = copy.deepcopy(
            next(c for c in self.cases if c["branch_stage"] == "details")["state"]
        )
        state["ordering"] = "parallel_details_and_classification"
        self.assertEqual(solve(state), {("TYPE_TEXT", "details"), ("SELECT", "classification")})

    def test_valid_sets_uniform_over_actual_criteria_positions(self):
        c = copy.deepcopy(next(c for c in self.cases if c["branch_stage"] == "details"))
        _, targets = capture_request(c["page"], c["goal"])
        labels = labels_for(
            c["request"],
            {("TYPE_TEXT", "details"), ("SELECT", "classification")},
            targets,
            c["semantic_control_ids"],
            2,
        )
        self.assertEqual(sorted(labels["operation"]["soft_target"]), [0.0] * 6 + [0.5] * 2)
        self.assertEqual(set(labels), {"operation", "type_text_target", "select_target"})

    def test_checked_control_not_clicked_again(self):
        c = next(
            c
            for c in self.cases
            if c["branch_stage"] == "acknowledge" and c["state"]["acknowledged"]
        )
        self.assertEqual(solve(c["state"]), {("CLICK", "submit")})

    def test_done_requires_all_values_and_saved_receipt(self):
        state = copy.deepcopy(
            next(c for c in self.cases if c["branch_stage"] == "complete")["state"]
        )
        self.assertEqual(solve(state), {("DONE", None)})
        state["saved_evidence"] = False
        self.assertEqual(solve(state), {("CLICK", "submit")})
        state.update(saved_evidence=True, current_text="wrong")
        with self.assertRaisesRegex(ValueError, "contradicts"):
            solve(state)

    def test_transient_wait_is_distinct_from_permanent_blocked(self):
        pending = next(c for c in self.cases if c["branch_stage"] == "pending")
        blocked = next(c for c in self.cases if c["branch_stage"] == "denied")
        self.assertEqual(solve(pending["state"]), {("WAIT", None)})
        self.assertEqual(solve(blocked["state"]), {("BLOCKED", None)})
        state = copy.deepcopy(pending["state"])
        state["pending"] = False
        self.assertEqual(solve(state), {("CLICK", "submit")})

    def test_unknown_semantic_order_refused(self):
        state = copy.deepcopy(self.cases[0]["state"])
        state["ordering"] = "arbitrary-first-target"
        with self.assertRaisesRegex(ValueError, "Unknown semantic"):
            solve(state)

    def test_fill_and_check_use_existing_standard_operations(self):
        self.assertEqual(
            set(OPERATIONS),
            {"CLICK", "TYPE_TEXT", "SELECT", "SCROLL_DOWN", "SCROLL_UP", "WAIT", "DONE", "BLOCKED"},
        )
        self.assertTrue(
            any(
                c["branch_stage"] == "acknowledge" and not c["state"]["acknowledged"]
                for c in self.cases
            )
        )
        self.assertTrue(
            all(
                "FIND_TEXT" not in c["request"]["questions"]["operation"]["criteria"]
                for c in self.cases
            )
        )

    def test_upstream_capture_never_calls_answer_validation_or_network(self):
        self.assertEqual(len(SOURCE_PINS), 2)
        c = self.cases[0]
        request, _ = capture_request(c["page"], c["goal"])
        self.assertEqual(request, c["request"])
        self.assertEqual(
            set(request["questions"]),
            {"operation", "click_target", "type_text_target", "select_target"},
        )

    def test_split_exports_never_put_labels_in_request_files(self):
        files = artifacts(self.cases)
        for split in ("train", "dev", "sealed"):
            request_bytes = files[f"data/{split}/requests.jsonl"]
            self.assertNotIn(b'"soft_target"', request_bytes)
            self.assertNotIn(b'"branch_stage"', request_bytes)
            self.assertNotIn(b'"desired_text"', request_bytes)
        self.assertIn(b'"soft_target"', files["data/sealed/labels.jsonl"])

    def test_publisher_refuses_existing_output(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).resolve().parent) as temporary:
            with self.assertRaisesRegex(ValueError, "overwrite"):
                publish({"x": b"x"}, pathlib.Path(temporary))


class RoleTests(unittest.TestCase):
    def setUp(self):
        self.path = "data/train/requests.jsonl"
        self.taint = {
            "dsl_version": DSL_VERSION,
            "splits": ["train"],
            "family_ids": ["appointment_edit"],
            "ancestor_ids": ["appointment_edit/workflow-1"],
            "sha256": "a" * 64,
        }
        self.families = {f[0]: f[1] for f in FAMILIES}

    def check(self, paths=None, taint=None, role="optimizer"):
        paths = [self.path] if paths is None else paths
        return check_role_selection(
            role, paths, {p: self.taint if taint is None else taint for p in paths}, self.families
        )

    def test_optimizer_train_only_acceptance(self):
        self.assertEqual(self.check()["split"], "train")

    def test_caller_cannot_relabel_sealed_family_as_train(self):
        forged = {**self.families, "equipment_checkout": "train"}
        with self.assertRaisesRegex(ValueError, "Caller family map"):
            check_role_selection("optimizer", [self.path], {self.path: self.taint}, forged)

    def test_optimizer_refuses_sealed_dev_source_and_manifest(self):
        for path in (
            "data/sealed/labels.jsonl",
            "data/dev/labels.jsonl",
            "dsl.py",
            "corpus/manifest.json",
            "authored-cases.jsonl",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.check([path])

    def test_role_spelling_and_traversal_fail_closed(self):
        for path in (
            "../data/train/requests.jsonl",
            "/data/train/requests.jsonl",
            "data/train/./requests.jsonl",
        ):
            with self.assertRaises(ValueError):
                self.check([path])
        with self.assertRaises(ValueError):
            self.check(role="trainer-with-evaluation-access")

    def test_family_or_cross_split_taint_refused(self):
        for patch in (
            {"splits": ["train", "sealed"]},
            {"family_ids": ["equipment_checkout"]},
            {"ancestor_ids": ["equipment_checkout/workflow-1"]},
            {"ancestor_ids": ["appointment_edit/workflow-99"]},
            {"sha256": "unknown"},
        ):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.check(taint={**self.taint, **patch})

    def test_missing_taint_refused(self):
        with self.assertRaises(ValueError):
            check_role_selection("optimizer", [self.path], {}, self.families)

    def test_duplicates_and_empty_selection_refused(self):
        for paths in ([], [self.path, self.path]):
            with self.assertRaises(ValueError):
                self.check(paths)


if __name__ == "__main__":
    unittest.main()
