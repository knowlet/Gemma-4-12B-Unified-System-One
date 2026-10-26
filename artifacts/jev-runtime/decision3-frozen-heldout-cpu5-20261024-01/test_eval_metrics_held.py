"""Source-authored stdlib fakes only; no imports/checks executed at authoring."""

import copy
import importlib.util
import unittest
from pathlib import Path


def module():
    path = Path(__file__).with_name("eval_metrics_held.py")
    spec = importlib.util.spec_from_file_location("_held_dev24_metrics_fake", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def capability(actor):
    bundle = actor.HERE / "private-fake-dev24-bundle"
    refs = {
        key: {"path": str(bundle / (key + ".npy")), "sha256": "a" * 64, "size_bytes": 128}
        for key in ("features", "counts", "soft_targets", "W0")
    }
    cases, rows = [], []
    for i in range(24):
        heads = ["operation", "click_target"] if i == 0 else ["operation"]
        case = {
            "case_id": "fake-dev-case-" + str(i),
            "offered_heads": list(actor.HEADS),
            "supervised_heads": heads,
        }
        cases.append(case)
        for head in heads:
            rows.append(
                {
                    "row_index": len(rows),
                    "case_id": case["case_id"],
                    "head_id": head,
                    "ordered_criteria_ids": [
                        "second-named-first-offered",
                        "first-named-second-offered",
                    ],
                }
            )
    ref = {
        "path": str(actor.HERE / "fake-root-runtime-metadata.json"),
        "sha256": "b" * 64,
        "size_bytes": 12,
    }
    return {
        "schema": actor.SCHEMA,
        "status": "ROOT_SELECTED_FROZEN_DEV24_METRICS_ONLY",
        "split": "dev",
        "nontraining": True,
        "training_eligible": False,
        "frozen_candidate_steps": 32,
        "bundle_root": str(bundle),
        "arrays": refs,
        "runtime": {"root_np5_prerequisite_ref": dict(ref), "numpy_runtime_catalog": dict(ref)},
        "candidate_capability": dict(ref),
        "delta_ref": dict(ref),
        "cases": cases,
        "rows": rows,
    }


class HeldMetricsTests(unittest.TestCase):
    def setUp(self):
        self.actor = module()

    def test_default_hold_precedes_argument_or_owner_path_access(self):
        class Bomb:
            def __iter__(self):
                raise AssertionError("arguments parsed")

            def __fspath__(self):
                raise AssertionError("owner path accessed")

        self.assertTrue(self.actor.SOURCE_ONLY)
        self.assertIsNone(self.actor.CAPABILITY_SHA)
        with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
            self.actor.main(Bomb())
        with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
            self.actor.execute(Bomb())

    def test_complete_mapping_keeps_unused_heads_out_of_denominator(self):
        cap = capability(self.actor)
        self.assertEqual(self.actor.validate_mapping(cap), 71)
        correct = [True] * 25
        correct[1] = False  # First case click fails while its operation is valid.
        result = self.actor.aggregate_rows(cap["rows"], cap["cases"], correct, [0.5] * 25)
        self.assertEqual(result["supervised_rows"], 25)
        self.assertEqual(result["valid_target_argmax_correct"], 24)
        self.assertEqual(result["by_head"]["operation"]["rows"], 24)
        self.assertEqual(result["by_head"]["operation"]["accuracy"], 1.0)
        self.assertEqual(result["by_head"]["click_target"]["accuracy"], 0.0)
        self.assertIsNone(result["by_head"]["select_target"]["accuracy"])
        self.assertEqual(
            result["all_supervised_heads_correct"],
            {"cases": 24, "correct": 23, "accuracy": 23 / 24},
        )

    def test_actual_target_head_insertion_order_is_preserved_not_canonicalized(self):
        cap = capability(self.actor)
        cap["cases"][0]["offered_heads"] = [
            "operation",
            "select_target",
            "type_text_target",
            "click_target",
        ]
        cap["cases"][0]["supervised_heads"] = ["operation", "select_target", "click_target"]
        first = copy.deepcopy(cap["rows"][0])
        added = {**first, "head_id": "select_target"}
        cap["rows"].insert(1, added)
        for i, row in enumerate(cap["rows"]):
            row["row_index"] = i
        self.assertEqual(self.actor.validate_mapping(cap), 70)
        self.assertEqual(
            [row["head_id"] for row in cap["rows"][:3]],
            ["operation", "select_target", "click_target"],
        )
        cap["cases"][0]["supervised_heads"] = ["operation", "click_target", "select_target"]
        with self.assertRaisesRegex(ValueError, "actual complete offered order"):
            self.actor.validate_mapping(cap)

    def test_multiple_valid_targets_and_first_offered_tie_are_not_single_gold(self):
        pad = [0.0] * 252
        targets = [0.5, 0.0, 0.5] + pad
        self.assertTrue(self.actor.valid_target_hit([0.1, 0.2, 0.7] + pad, targets, 3))
        self.assertTrue(self.actor.valid_target_hit([0.7, 0.2, 0.1] + pad, targets, 3))
        self.assertFalse(self.actor.valid_target_hit([0.1, 0.8, 0.1] + pad, targets, 3))
        self.assertTrue(self.actor.valid_target_hit([0.4, 0.4, 0.2] + pad, targets, 3))
        self.assertFalse(
            self.actor.valid_target_hit([0.4, 0.4, 0.2] + pad, [0.0, 0.5, 0.5] + pad, 3)
        )

    def test_release_rejects_nonhex_64_character_pin_before_io(self):
        self.actor.SOURCE_ONLY = False
        self.actor.CAPABILITY_SHA = "x" * 64
        with self.assertRaisesRegex(ValueError, "pin remains pending"):
            self.actor.released()

    def test_partial_duplicate_reordered_or_foreign_mapping_never_scores(self):
        original = capability(self.actor)
        variants = []
        changed = copy.deepcopy(original)
        changed["cases"].pop()
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["rows"].pop()
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["rows"][1]["head_id"] = "operation"
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["rows"][0], changed["rows"][1] = changed["rows"][1], changed["rows"][0]
        for i, row in enumerate(changed["rows"]):
            row["row_index"] = i
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["rows"][0]["ordered_criteria_ids"] = ["duplicate", "duplicate"]
        variants.append(changed)
        for changed in variants:
            with self.assertRaises(ValueError):
                self.actor.validate_mapping(changed)
        with self.assertRaisesRegex(ValueError, "complete aggregate"):
            self.actor.aggregate_rows(
                original["rows"][:-1], original["cases"], [True] * 24, [0.0] * 24
            )

    def test_train_role_other_split_or_candidate_update_is_refused(self):
        for key, value in (
            ("split", "train"),
            ("split", "sealed"),
            ("nontraining", False),
            ("training_eligible", True),
            ("frozen_candidate_steps", 64),
        ):
            changed = capability(self.actor)
            changed[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "dev-only"):
                self.actor.validate_mapping(changed)

    def test_reader_rejects_unselected_raw_or_train_ref_before_io(self):
        cap = capability(self.actor)
        reader = self.actor.RuntimeReader(cap)

        def forbidden(*args, **kwargs):
            raise AssertionError("unselected file opened")

        self.actor.read_fixed = forbidden
        foreign = {
            "path": str(self.actor.HERE / "never-selected-private-raw.json"),
            "sha256": "c" * 64,
            "size_bytes": 7,
        }
        with self.assertRaisesRegex(ValueError, "unselected/non-metric"):
            reader.read_ref(foreign, lambda: None)


if __name__ == "__main__":
    unittest.main()
