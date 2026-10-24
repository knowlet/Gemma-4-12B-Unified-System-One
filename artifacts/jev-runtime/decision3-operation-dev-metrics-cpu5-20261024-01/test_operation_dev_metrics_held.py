"""Source-authored stdlib fakes only; no imports/checks executed at authoring."""

import copy
import importlib.util
import math
import struct
import unittest
from pathlib import Path


def module():
    path = Path(__file__).with_name("eval_metrics_held.py")
    spec = importlib.util.spec_from_file_location("_held_operation_dev24_metrics_fake", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def capability(actor):
    bundle = actor.BUNDLE_SOURCE_DIR / "private-fake-dev24-bundle"
    refs = {
        key: {"path": str(bundle / (key + ".npy")), "sha256": "a" * 64, "size_bytes": 128}
        for key in ("features", "counts", "soft_targets", "W0")
    }
    cases, rows = [], []
    for i in range(24):
        heads = ["operation", actor.HEADS[1 + i % 3]] if i < 9 else ["operation"]
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
        "status": "ROOT_SELECTED_OPERATION_DEV24_METRICS_ONLY",
        "split": "dev",
        "nontraining": True,
        "training_eligible": False,
        "frozen_candidate_steps": 512,
        "only_head": "operation",
        "model_selection_scope": "dev_model_selection_only_no_unseen_claim",
        "dev_used_for_model_selection": True,
        "optimizer_input_dev": False,
        "optimizer_input_sealed": False,
        "candidate_selection_before_dev_or_sealed": False,
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
        self.assertIsNone(self.actor.CANDIDATE_SHA)
        self.assertIsNone(self.actor.DELTA_SHA)
        with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
            self.actor.main(Bomb())
        with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
            self.actor.execute(Bomb())

    def test_complete_mapping_keeps_unused_heads_out_of_denominator(self):
        cap = capability(self.actor)
        self.assertEqual(self.actor.validate_mapping(cap), 63)
        correct = [True] * 33
        correct[1] = False  # First case click fails while its operation is valid.
        result = self.actor.aggregate_rows(cap["rows"], cap["cases"], correct, [0.5] * 33)
        self.assertEqual(result["supervised_rows"], 33)
        self.assertEqual(result["valid_target_argmax_correct"], 32)
        self.assertEqual(result["by_head"]["operation"]["rows"], 24)
        self.assertEqual(result["by_head"]["operation"]["accuracy"], 1.0)
        self.assertEqual(result["by_head"]["click_target"]["accuracy"], 2 / 3)
        self.assertEqual(result["by_head"]["select_target"]["accuracy"], 1.0)
        self.assertEqual(
            result["all_supervised_heads_correct"],
            {"cases": 24, "correct": 23, "accuracy": 23 / 24},
        )

    def test_actual_target_head_insertion_order_is_preserved_not_canonicalized(self):
        cap = capability(self.actor)
        cap["cases"][0]["offered_heads"] = [
            "select_target",
            "operation",
            "type_text_target",
            "click_target",
        ]
        cap["cases"][0]["supervised_heads"] = ["select_target", "operation", "click_target"]
        # Move an existing target supervision slot, keeping exactly33 rows.
        case_two = cap["cases"][2]["case_id"]
        cap["cases"][2]["supervised_heads"] = ["operation"]
        cap["rows"] = [
            row
            for row in cap["rows"]
            if not (row["case_id"] == case_two and row["head_id"] == "select_target")
        ]
        first = copy.deepcopy(cap["rows"][0])
        added = {**first, "head_id": "select_target"}
        cap["rows"].insert(0, added)
        for i, row in enumerate(cap["rows"]):
            row["row_index"] = i
        self.assertEqual(self.actor.validate_mapping(cap), 63)
        self.assertEqual(
            [row["head_id"] for row in cap["rows"][:3]],
            ["select_target", "operation", "click_target"],
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
        with self.assertRaisesRegex(ValueError, "pins remain pending"):
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
            ("frozen_candidate_steps", 32),
            ("only_head", "all_heads"),
            ("dev_used_for_model_selection", False),
            ("optimizer_input_dev", True),
            ("optimizer_input_sealed", True),
            ("candidate_selection_before_dev_or_sealed", True),
            ("model_selection_scope", "new_unseen_test"),
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


class FakeArray:
    """Only indexing/copy/bytes protocol, never a substitute numerical runtime."""

    def __init__(self, values):
        self.values = copy.deepcopy(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, indices):
        return FakeArray([self.values[index] for index in indices])

    def __setitem__(self, indices, source):
        for index, row in zip(indices, source.values):
            self.values[index] = copy.deepcopy(row)

    def copy(self):
        return FakeArray(self.values)

    def tobytes(self, *, order):
        if order != "C":
            raise AssertionError("unexpected array byte order")
        flat = [value for row in self.values for value in row]
        return struct.pack("<" + "f" * len(flat), *flat)


class FakeCore:
    """Invented distributions test routing, not trained/core numerical parity."""

    def __init__(self, zero):
        self.zero, self.events = zero, []
        self.failure = ValueError("invented excluded target poison")

    def _inputs(self, x, _w0, delta, _counts, targets):
        self.events.append(("validate", len(x), "zero" if delta is self.zero else "delta"))
        if any(row[-1] == "poison" for row in targets.values):
            raise self.failure
        return None

    def _probabilities(self, x, _w0, delta, _illegal):
        kind = "zero" if delta is self.zero else "delta"
        self.events.append(("project", len(x), kind))
        rows, logs = [], []
        for feature in x.values:
            first = (0.25 if kind == "zero" else 0.65) + feature[0] / 1000
            rows.append([first, 1 - first] + [0.0] * 253)
            logs.append([math.log(first), math.log(1 - first)] + [-math.inf] * 253)
        return FakeArray(rows), FakeArray(logs)


class OperationRoutingTests(unittest.TestCase):
    def setUp(self):
        self.actor = module()
        self.cap = capability(self.actor)
        # A real target row comes before operation in this invented case.
        self.cap["cases"][0]["offered_heads"] = [
            "click_target",
            "operation",
            "type_text_target",
            "select_target",
        ]
        self.cap["cases"][0]["supervised_heads"] = ["click_target", "operation"]
        self.cap["rows"][0], self.cap["rows"][1] = self.cap["rows"][1], self.cap["rows"][0]
        for index, row in enumerate(self.cap["rows"]):
            row["row_index"] = index
        self.x = FakeArray([[index] for index in range(33)])
        self.counts = FakeArray([2] * 33)
        self.targets = FakeArray([[1.0, 0.0] + [0.0] * 253 for _ in range(33)])
        self.zero, self.delta = object(), object()
        self.core = FakeCore(self.zero)

    def test_only_true_operation_rows_change_and_targets_keep_control_bytes(self):
        self.assertEqual(self.actor.validate_mapping(self.cap), 63)
        projected = self.actor.distributions(
            None,
            self.core,
            self.x,
            object(),
            self.counts,
            self.targets,
            self.delta,
            self.zero,
            self.cap,
            lambda: None,
        )
        control_p, control_logp = projected["zero_delta"]
        candidate_p, candidate_logp = projected["frozen_candidate"]
        selected = self.actor.operation_indices(self.cap["rows"])
        self.assertNotIn(0, selected)
        self.assertIn(1, selected)
        for index, row in enumerate(self.cap["rows"]):
            if row["head_id"] == "operation":
                expected = 0.65 + index / 1000
                self.assertEqual(candidate_p.values[index][0], expected)
                self.assertNotEqual(candidate_p.values[index], control_p.values[index])
            else:
                self.assertEqual(
                    candidate_p[[index]].tobytes(order="C"), control_p[[index]].tobytes(order="C")
                )
                self.assertEqual(
                    candidate_logp[[index]].tobytes(order="C"),
                    control_logp[[index]].tobytes(order="C"),
                )
        self.assertEqual(
            self.core.events,
            [
                ("validate", 33, "zero"),
                ("validate", 33, "delta"),
                ("project", 33, "zero"),
                ("validate", 24, "delta"),
                ("project", 24, "delta"),
            ],
        )

    def test_invalid_excluded_target_cannot_escape_full_validation(self):
        self.targets.values[0][-1] = "poison"
        with self.assertRaises(ValueError) as raised:
            self.actor.distributions(
                None,
                self.core,
                self.x,
                object(),
                self.counts,
                self.targets,
                self.delta,
                self.zero,
                self.cap,
                lambda: None,
            )
        self.assertIs(raised.exception, self.core.failure)
        self.assertFalse(any(event[0] == "project" for event in self.core.events))

    def test_complete_case_coverage_cannot_authorize32_or34_row_score(self):
        rows = copy.deepcopy(self.cap["rows"])
        rows.pop(0)  # Still leaves all24 operation/case rows represented.
        with self.assertRaisesRegex(ValueError, "complete aggregate"):
            self.actor.aggregate_rows(rows, self.cap["cases"], [True] * 32, [0.5] * 32)


if __name__ == "__main__":
    unittest.main()
