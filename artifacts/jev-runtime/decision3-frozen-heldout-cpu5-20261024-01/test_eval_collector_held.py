"""Authored, unexecuted stdlib fakes for the held dev-only collector.

The exports and capability below are invented schema witnesses. These tests
read no split payload, label, authoring module, actual artifact or capability.
Running them later requires an explicitly authorized source-check window.
"""

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DSL_ROOT = ROOT / "artifacts/jev-runtime/browser-decision-dsl-20261023-01"
SPEC = importlib.util.spec_from_file_location(
    "_held_dev_collector_fake_tests", HERE / "eval_collector_held.py"
)
COLLECTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COLLECTOR
SPEC.loader.exec_module(COLLECTOR)


def invented_exports():
    """Twenty-four fake requests; no labels or semantic expected answers."""
    family = "invented_development_domain"
    ancestors = [f"{family}/workflow-{number}" for number in (1, 2, 3)]
    capability = {
        "role": "dev_feature_collector",
        "split": "dev",
        "dsl_version": COLLECTOR.DSL_VERSION,
        "case_count": 24,
        "family_ids": [family],
        "ancestor_ids": ancestors,
    }
    rows = []
    for ancestor in ancestors:
        for branch in range(8):
            operations = {
                key: "invented operation description"
                for key in (
                    "CLICK",
                    "TYPE_TEXT",
                    "SELECT",
                    "SCROLL_DOWN",
                    "SCROLL_UP",
                    "WAIT",
                    "DONE",
                    "BLOCKED",
                )
            }
            questions = {
                "operation": {
                    "type": "choice",
                    "instructions": "invented",
                    "criteria": operations,
                }
            }
            targets = ["click_target", "select_target", "type_text_target"]
            if branch % 2:
                targets.reverse()
            for head in targets:
                questions[head] = {
                    "type": "choice",
                    "instructions": "invented",
                    "criteria": {
                        key: "invented target description" for key in ("7", "2", "9", "1")
                    },
                }
            request = {
                "model": "jev-latest",
                "state": {"invented_branch": branch},
                "questions": questions,
            }
            payload = {key: request[key] for key in ("state", "questions")}
            rows.append(
                {
                    "case_id": f"{len(rows):024x}",
                    "dsl_version": COLLECTOR.DSL_VERSION,
                    "family_id": family,
                    "ancestor_id": ancestor,
                    "request_sha256": COLLECTOR.digest(request),
                    "native_payload_sha256": COLLECTOR.digest(payload),
                    "request": request,
                }
            )
    summary = {
        "dsl_version": COLLECTOR.DSL_VERSION,
        "split": "dev",
        "case_count": 24,
        "family_ids": [family],
        "ancestor_ids": ancestors,
        "is_authored_synthetic_not_NN": True,
    }
    return rows, summary, capability


def encoded(rows, summary):
    return (
        b"\n".join(COLLECTOR.canonical(row) for row in rows) + b"\n",
        COLLECTOR.canonical(summary),
    )


def fake_record(path):
    return {"path": str(path), "sha256": "0" * 64, "size_bytes": 1}


class HeldEntryTests(unittest.TestCase):
    def test_source_hold_precedes_parser_and_any_selected_read(self):
        self.assertIs(COLLECTOR.SOURCE_ONLY, True)
        self.assertIsNone(COLLECTOR.DEV_CAPABILITY_SHA256)
        self.assertIsNone(COLLECTOR.NUMERIC_ACCEPTANCE_SHA256)
        with (
            patch.object(
                COLLECTOR.argparse, "ArgumentParser", side_effect=AssertionError("parser")
            ),
            patch.object(COLLECTOR, "read", side_effect=AssertionError("selected read")),
        ):
            with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
                COLLECTOR.main(["--not-a-real-option"])

    def test_clearing_only_source_hold_does_not_release_pending_pins(self):
        with patch.object(COLLECTOR, "SOURCE_ONLY", False):
            with self.assertRaises(RuntimeError):
                COLLECTOR.release_gate()
        for field in ("DEV_CAPABILITY_SHA256", "NUMERIC_ACCEPTANCE_SHA256"):
            for invalid in (None, "pending", "f" * 63, True):
                with (
                    self.subTest(field=field, invalid=invalid),
                    patch.object(COLLECTOR, "SOURCE_ONLY", False),
                    patch.object(COLLECTOR, "DEV_CAPABILITY_SHA256", "a" * 64),
                    patch.object(COLLECTOR, "NUMERIC_ACCEPTANCE_SHA256", "b" * 64),
                    patch.object(COLLECTOR, field, invalid),
                    self.assertRaises(RuntimeError),
                ):
                    COLLECTOR.release_gate()

    def test_dev_scope_and_original_transaction_budgets_remain_explicit(self):
        self.assertEqual(COLLECTOR.ROLE, "dev_feature_collector")
        self.assertEqual(COLLECTOR.SPLIT, "dev")
        self.assertEqual(COLLECTOR.PENDING, "PENDING_DEV_NONTRAINING")
        self.assertEqual(COLLECTOR.WALL_SECONDS, 120)
        self.assertEqual(COLLECTOR.CALL_SECONDS, 20)


class AllPathRoleTests(unittest.TestCase):
    def test_ordinary_allows_only_dev_requests_and_summary_within_dsl_tree(self):
        with patch.object(Path, "is_symlink", return_value=False):
            for name in ("requests.jsonl", "summary.json"):
                path = DSL_ROOT / "corpus/data/dev" / name
                self.assertEqual(COLLECTOR.ordinary(path), path)

    def test_generic_reference_rejects_labels_train_sealed_and_authoring_before_open(self):
        forbidden = [
            DSL_ROOT / "corpus/data/dev/labels.jsonl",
            DSL_ROOT / "corpus/data/train/requests.jsonl",
            DSL_ROOT / "corpus/data/train/summary.json",
            DSL_ROOT / "corpus/data/train/labels.jsonl",
            DSL_ROOT / "corpus/data/sealed/requests.jsonl",
            DSL_ROOT / "corpus/data/sealed/summary.json",
            DSL_ROOT / "corpus/data/sealed/labels.jsonl",
            DSL_ROOT / "corpus/authored-cases.jsonl",
            DSL_ROOT / "corpus/manifest.json",
            DSL_ROOT / "dsl.py",
            DSL_ROOT / "role-policy.json",
        ]
        with (
            patch.object(COLLECTOR, "read", side_effect=AssertionError("forbidden read")),
            patch.object(Path, "open", side_effect=AssertionError("forbidden open")),
        ):
            for path in forbidden:
                with self.subTest(path=path), self.assertRaises(RuntimeError):
                    # No expected_path argument: the generic-reference path must
                    # enforce the same role restriction as the main input path.
                    COLLECTOR.verify_record(fake_record(path))


class DevProjectionTests(unittest.TestCase):
    def test_exact_capability_inventory_and_eight_cases_per_ancestor(self):
        rows, summary, capability = invented_exports()
        families, ancestors = COLLECTOR.capability_inventory(capability)
        self.assertEqual(list(families), capability["family_ids"])
        self.assertEqual(list(ancestors), capability["ancestor_ids"])
        before = encoded(rows, summary)
        for ancestor in capability["ancestor_ids"]:
            selected = COLLECTOR.select_cases(*before, ancestor, capability)
            expected = [row for row in rows if row["ancestor_id"] == ancestor]
            self.assertEqual(selected, expected)
            self.assertEqual(len(selected), 8)
            self.assertEqual(
                [row["request"]["state"]["invented_branch"] for row in selected], list(range(8))
            )
        self.assertEqual(encoded(rows, summary), before)

    def test_capability_inventory_refuses_other_roles_splits_and_lineages(self):
        _, _, capability = invented_exports()
        changes = (
            {"role": "train_collector"},
            {"role": "sealed_feature_collector"},
            {"split": "train"},
            {"split": "sealed"},
            {"dsl_version": "invented-other-version"},
            {"case_count": 23},
            {"case_count": True},
            {"family_ids": [*capability["family_ids"], "another_invented_family"]},
            {"family_ids": ["appointment_edit"]},
            {"ancestor_ids": capability["ancestor_ids"][:-1]},
            {"ancestor_ids": [capability["ancestor_ids"][0]] * 3},
            {
                "ancestor_ids": [
                    *capability["ancestor_ids"][:-1],
                    "another_invented_family/workflow-3",
                ]
            },
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                COLLECTOR.capability_inventory({**copy.deepcopy(capability), **change})

    def test_wrong_summary_or_selected_ancestor_does_not_relabel_a_split(self):
        rows, summary, capability = invented_exports()
        ancestor = capability["ancestor_ids"][0]
        for change in (
            {"split": "train"},
            {"split": "sealed"},
            {"case_count": 96},
            {"dsl_version": "invented-other-version"},
            {"family_ids": ["another_invented_family"]},
            {"ancestor_ids": capability["ancestor_ids"][:-1]},
        ):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                COLLECTOR.select_cases(*encoded(rows, {**summary, **change}), ancestor, capability)
        with self.assertRaises(RuntimeError):
            COLLECTOR.select_cases(
                *encoded(rows, summary), "unselected_family/workflow-1", capability
            )

    def test_missing_duplicate_or_wrong_ancestor_population_is_not_a_partial_success(self):
        rows, summary, capability = invented_exports()
        moved = copy.deepcopy(rows)
        moved[0]["ancestor_id"] = capability["ancestor_ids"][1]
        for changed in (rows[:-1], [*rows[:-1], rows[0]], moved):
            with self.subTest(count=len(changed)), self.assertRaises(RuntimeError):
                COLLECTOR.select_cases(
                    *encoded(changed, summary), capability["ancestor_ids"][0], capability
                )

    def test_mixed_family_foreign_version_and_extra_label_fields_are_refused(self):
        rows, summary, capability = invented_exports()
        for mutation in (
            {"family_id": "appointment_edit", "ancestor_id": "appointment_edit/workflow-1"},
            {"family_id": "unselected_family", "ancestor_id": "unselected_family/workflow-1"},
            {"dsl_version": "invented-other-version"},
            {"heads": {"operation": {"invented_label": "not permitted"}}},
            {"split": "sealed"},
        ):
            changed = copy.deepcopy(rows)
            changed[0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                COLLECTOR.select_cases(
                    *encoded(changed, summary), capability["ancestor_ids"][0], capability
                )

    def test_full_request_and_native_projection_hashes_both_remain_required(self):
        rows, summary, capability = invented_exports()
        for field, value in (("model", "other-routing-tag"), ("state", {"changed": True})):
            changed = copy.deepcopy(rows)
            changed[0]["request"][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                COLLECTOR.select_cases(
                    *encoded(changed, summary), capability["ancestor_ids"][0], capability
                )
        changed = copy.deepcopy(rows)
        changed[0]["native_payload_sha256"] = "f" * 64
        with self.assertRaises(RuntimeError):
            COLLECTOR.select_cases(
                *encoded(changed, summary), capability["ancestor_ids"][0], capability
            )

    def test_mapping_preserves_actual_head_and_criteria_order_without_labels(self):
        rows, _, _ = invented_exports()
        for item in rows[:2]:
            actual_heads = list(item["request"]["questions"])
            mapped = COLLECTOR.mapping(item, 3)
            self.assertEqual([row["head_id"] for row in mapped], actual_heads)
            self.assertEqual([row["h_index"] for row in mapped], [0, 1, 2, 3])
            self.assertEqual([row["legal_count"] for row in mapped], [8, 4, 4, 4])
            for row, question in zip(mapped, item["request"]["questions"].values()):
                self.assertEqual(row["ordered_criteria_ids"], list(question["criteria"]))
                self.assertEqual(row["case_id"], item["case_id"])
                self.assertEqual(row["native_payload_sha256"], item["native_payload_sha256"])
                self.assertEqual(row["same_call_id"], "call-0003")
                self.assertEqual(row["native_call_index"], 3)
                self.assertNotIn("labels", row)
                self.assertNotIn("targets", row)
            self.assertEqual(mapped[1]["ordered_criteria_ids"], ["7", "2", "9", "1"])
        self.assertNotEqual(
            COLLECTOR.mapping(rows[0], 1)[1]["head_id"],
            COLLECTOR.mapping(rows[1], 2)[1]["head_id"],
        )

    def test_original_four_heads_and_counts_cannot_be_shortened_or_dummy_padded(self):
        rows, summary, capability = invented_exports()
        shortened = copy.deepcopy(rows)
        del shortened[0]["request"]["questions"]["click_target"]
        wrong_count = copy.deepcopy(rows)
        wrong_count[0]["request"]["questions"]["click_target"]["criteria"].pop("7")
        for changed in (shortened, wrong_count):
            # Keep the fake export hashes internally consistent so the shape
            # gate, rather than stale-hash detection alone, must refuse it.
            request = changed[0]["request"]
            changed[0]["request_sha256"] = COLLECTOR.digest(request)
            changed[0]["native_payload_sha256"] = COLLECTOR.digest(
                {key: request[key] for key in ("state", "questions")}
            )
            with self.assertRaises(RuntimeError):
                COLLECTOR.select_cases(
                    *encoded(changed, summary), capability["ancestor_ids"][0], capability
                )

    def test_duplicate_json_and_nonfinite_json_are_refused(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                COLLECTOR.decode(raw)


if __name__ == "__main__":
    unittest.main()
