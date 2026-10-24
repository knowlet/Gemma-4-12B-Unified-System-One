"""Unexecuted stdlib fake source; no actual DSL payload/model/framework reads."""

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "_dsl_train_collector_fake_tests", HERE / "collector.py"
)
COLLECTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COLLECTOR
SPEC.loader.exec_module(COLLECTOR)


def invented_exports():
    """Fake schema cases only; no actual DSL authoring/labels or Copper page."""
    rows, ancestors = [], []
    for family in sorted(COLLECTOR.TRAIN_FAMILIES):
        for number in (1, 2, 3):
            ancestor = f"{family}/workflow-{number}"
            ancestors.append(ancestor)
            for branch in range(8):
                operation = {
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
                target_heads = ["click_target", "select_target", "type_text_target"]
                if branch % 2:
                    target_heads.reverse()
                questions = {
                    "operation": {
                        "type": "choice",
                        "instructions": "invented",
                        "criteria": operation,
                    }
                }
                for head in target_heads:
                    questions[head] = {
                        "type": "choice",
                        "instructions": "invented",
                        "criteria": {key: "invented target" for key in ("7", "2", "9", "1")},
                    }
                request = {
                    "model": "jev-latest",
                    "state": {"invented_branch": branch},
                    "questions": questions,
                }
                projection = {key: request[key] for key in ("state", "questions")}
                rows.append(
                    {
                        "case_id": f"{len(rows):024x}",
                        "dsl_version": COLLECTOR.DSL_VERSION,
                        "family_id": family,
                        "ancestor_id": ancestor,
                        "request_sha256": COLLECTOR.digest(request),
                        "native_payload_sha256": COLLECTOR.digest(projection),
                        "request": request,
                    }
                )
    summary = {
        "dsl_version": COLLECTOR.DSL_VERSION,
        "split": "train",
        "case_count": 96,
        "family_ids": sorted(COLLECTOR.TRAIN_FAMILIES),
        "ancestor_ids": ancestors,
        "is_authored_synthetic_not_NN": True,
    }
    return rows, summary


def encode(rows, summary):
    return b"\n".join(COLLECTOR.canonical(row) for row in rows) + b"\n", COLLECTOR.canonical(
        summary
    )


class TrainProjectionTests(unittest.TestCase):
    def test_source_hold_refuses_before_parser_or_selected_input_read(self):
        with (
            patch.object(COLLECTOR, "SOURCE_ONLY", True),
            patch.object(COLLECTOR, "read", side_effect=AssertionError("must not read")),
        ):
            with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
                COLLECTOR.main(["--not-a-real-option"])

    def test_selects_all_eight_existing_branches_in_original_order(self):
        rows, summary = invented_exports()
        request_bytes, summary_bytes = encode(rows, summary)
        ancestor = summary["ancestor_ids"][4]
        selected = COLLECTOR.select_cases(request_bytes, summary_bytes, ancestor)
        expected = [row for row in rows if row["ancestor_id"] == ancestor]
        self.assertEqual(selected, expected)
        self.assertEqual(len(selected), 8)
        self.assertEqual(
            [row["request"]["state"]["invented_branch"] for row in selected], list(range(8))
        )
        self.assertEqual(encode(rows, summary), (request_bytes, summary_bytes))
        self.assertTrue(all(len(row["request"]["questions"]) == 4 for row in selected))

    def test_mapping_uses_actual_head_and_criterion_order_including_unused_heads(self):
        rows, _ = invented_exports()
        for item in rows[:2]:
            mapped = COLLECTOR.mapping(item, 3)
            self.assertEqual([row["head_id"] for row in mapped], list(item["request"]["questions"]))
            self.assertEqual([row["h_index"] for row in mapped], [0, 1, 2, 3])
            self.assertEqual([row["legal_count"] for row in mapped], [8, 4, 4, 4])
            self.assertEqual(mapped[1]["ordered_criteria_ids"], ["7", "2", "9", "1"])
            self.assertTrue(all(row["same_call_id"] == "call-0003" for row in mapped))
            self.assertTrue(
                all(row["native_payload_sha256"] == item["native_payload_sha256"] for row in mapped)
            )
        self.assertNotEqual(
            COLLECTOR.mapping(rows[0], 1)[1]["head_id"], COLLECTOR.mapping(rows[1], 2)[1]["head_id"]
        )

    def test_non_train_ancestor_or_summary_is_refused(self):
        rows, summary = invented_exports()
        with self.assertRaisesRegex(RuntimeError, "non-train ancestor"):
            COLLECTOR.select_cases(*encode(rows, summary), "invented-sealed/workflow-1")
        summary["split"] = "dev"
        with self.assertRaisesRegex(RuntimeError, "train summary"):
            COLLECTOR.select_cases(*encode(rows, summary), "admin_filters/workflow-1")

    def test_changed_native_projection_and_original_request_tag_are_both_detected(self):
        rows, summary = invented_exports()
        for field in ("state", "model"):
            changed = copy.deepcopy(rows)
            changed[0]["request"][field] = (
                {"changed": True} if field == "state" else "another-model"
            )
            with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                COLLECTOR.select_cases(*encode(changed, summary), summary["ancestor_ids"][0])

    def test_duplicate_or_missing_branch_is_not_a_smaller_eligible_chunk(self):
        rows, summary = invented_exports()
        for changed in (rows[:-1], [*rows[:-1], rows[0]]):
            with self.assertRaisesRegex(RuntimeError, "unique train request"):
                COLLECTOR.select_cases(*encode(changed, summary), summary["ancestor_ids"][0])

    def test_extra_labels_or_authoring_fields_are_rejected(self):
        rows, summary = invented_exports()
        rows[0]["heads"] = {"operation": {"gold": "invented"}}
        with self.assertRaisesRegex(RuntimeError, "request lineage"):
            COLLECTOR.select_cases(*encode(rows, summary), summary["ancestor_ids"][0])

    def test_label_or_sealed_file_ref_is_rejected_before_file_read(self):
        record = {"path": str(COLLECTOR.DSL / "labels.jsonl"), "sha256": "0" * 64, "size_bytes": 1}
        with patch.object(COLLECTOR, "read", side_effect=AssertionError("must not read labels")):
            with self.assertRaisesRegex(RuntimeError, "before open|fixed input projection"):
                COLLECTOR.verify_record(record, COLLECTOR.DSL / "requests.jsonl")

    def test_duplicate_json_and_nonfinite_values_are_refused(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}'):
            with self.assertRaises(RuntimeError):
                COLLECTOR.decode(raw)

    def test_only_exact_closed_np5_runtime_and_mapped_files_are_accepted(self):
        first = {"resolved_path": "/invented/numpy.so", "sha256": "a" * 64, "size_bytes": 3}
        second = {"resolved_path": "/invented/libblas.so", "sha256": "b" * 64, "size_bytes": 4}
        accepted = {
            "numpy_version": "invented2",
            "python": {"version": [3, 13, 14]},
            "numpy_build_configuration": {"blas": "invented"},
            "critical_files_and_mapped_math_candidates": [first, second],
            "mapped_math_candidates": ["/invented/libblas.so"],
            "mapped_gpu_math_candidates_excluded_from_CPU_projection": [],
        }
        COLLECTOR.check_catalog(copy.deepcopy(accepted), accepted, complete=True)
        early = {**accepted, "critical_files_and_mapped_math_candidates": [first]}
        COLLECTOR.check_catalog(early, accepted, complete=False)
        for key, value in (
            ("numpy_version", "other"),
            ("python", {"version": [3, 12, 1]}),
            ("numpy_build_configuration", {}),
            ("mapped_math_candidates", []),
        ):
            changed = {**accepted, key: value}
            with self.assertRaises(RuntimeError):
                COLLECTOR.check_catalog(changed, accepted, complete=True)
        changed = copy.deepcopy(accepted)
        changed["critical_files_and_mapped_math_candidates"][0]["sha256"] = "c" * 64
        with self.assertRaisesRegex(RuntimeError, "file identity"):
            COLLECTOR.check_catalog(changed, accepted, complete=True)


class NumericalAdmissionTests(unittest.TestCase):
    def test_closed_catalog_is_authenticated_before_first_import_inside_call(self):
        events = []
        cpu = {"path": "/invented/cpu", "size_bytes": 4}
        gpu = {"path": "/invented/libcublas.so.13", "size_bytes": 700000000}
        accepted = {
            "critical_files_and_mapped_math_candidates": [cpu],
            "mapped_gpu_math_candidates_excluded_from_CPU_projection": [gpu],
        }
        probe = SimpleNamespace(
            MAX_SELECTED_FILES=24,
            MAX_SELECTED_BYTES=512 * 1024**2,
            file_identity=lambda *a: events.append("CPU hash") or cpu,
            gpu_candidate_metadata=lambda *a: events.append("GPU stat only") or gpu,
            load_numpy=lambda *a: events.append("NumPy import") or "np",
        )
        shared = {}
        with patch.object(
            COLLECTOR, "source", side_effect=lambda *a: events.append("core import") or "core"
        ):
            self.assertEqual(
                COLLECTOR.numerical_context(shared, probe, None, None, {}, {}, accepted, 0),
                ("np", "core"),
            )
            self.assertEqual(events, ["CPU hash", "GPU stat only", "NumPy import", "core import"])
            COLLECTOR.numerical_context(shared, probe, None, None, {}, {}, accepted, 0)
            self.assertEqual(len(events), 4)
        probe.file_identity = lambda *a: {"changed": True}
        with self.assertRaisesRegex(RuntimeError, "before numerical import"):
            COLLECTOR.numerical_context({}, probe, None, None, {}, {}, accepted, 0)

    def test_generic_metadata_path_cannot_read_disallowed_DSL_exports(self):
        for path in (
            COLLECTOR.DSL / "labels.jsonl",
            COLLECTOR.DSL.parent / "dev/requests.jsonl",
            COLLECTOR.DSL.parents[2] / "dsl.py",
        ):
            with patch.object(Path, "open", side_effect=AssertionError("must refuse before open")):
                with self.assertRaisesRegex(RuntimeError, "before open"):
                    COLLECTOR.read(path)

    def test_directory_sync_failure_preserves_primary_and_refuses_success(self):
        primary = RuntimeError("original failure")
        with patch.object(COLLECTOR.os, "open", side_effect=OSError("sync failure")):
            COLLECTOR.sync_directories([Path("/invented")], primary)
            self.assertIn("Directory sync also failed", primary.__notes__[0])
            with self.assertRaisesRegex(OSError, "sync failure"):
                COLLECTOR.sync_directories([Path("/invented")], None)

    def test_changed_gpu_stat_only_catalog_is_refused(self):
        accepted = {
            "numpy_version": "2fake",
            "python": {},
            "numpy_build_configuration": {},
            "critical_files_and_mapped_math_candidates": [],
            "mapped_math_candidates": [],
            "mapped_gpu_math_candidates_excluded_from_CPU_projection": [
                {"path": "/invented/gpu", "hashed": False}
            ],
        }
        changed = {**accepted, "mapped_gpu_math_candidates_excluded_from_CPU_projection": []}
        with self.assertRaisesRegex(RuntimeError, "GPU stat-only"):
            COLLECTOR.check_catalog(changed, accepted, complete=True)


if __name__ == "__main__":
    raise SystemExit("SOURCE ONLY: these fake tests have not been executed or selected")
