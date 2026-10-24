"""Fake-data-only tests.  Authored while NN96310 runs; NOT EXECUTED.

No official package, actual suite, model or host observation is needed.
"""

import contextlib
import copy
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import runner


def fake_report():
    checks = {"fake_relation_" + str(i): "tv" for i in range(50)}
    relations = dict.fromkeys(checks, 0.5)
    dimensions = dict.fromkeys(runner.DIMENSIONS, 0.5)
    groups = dict.fromkeys(runner.GROUPS, 0.5)
    scores = {
        "backend": runner.NAME,
        "tolerance": "fixed 0.05",
        "min_n": 5,
        "bootstrap": 1000,
        "min_coverage": 0.95,
        "overall": 0.5,
        "pillars": dimensions,
        "groups": groups,
        "relations": relations,
        "graded": {"overall": 0.5, "pillars": dimensions, "groups": groups, "relations": relations},
        "ci": {"overall": [0.4, 0.6], "graded_overall": [0.4, 0.6]},
        "checks": {
            name + "/tv": {"n": 24, "planned": 24, "coverage": 1.0, "score": 0.5, "graded": 0.5}
            for name in checks
        },
        "excluded": {"low_n": [], "exploratory": [], "diagnostic": []},
        "incomplete": [],
        "n_cases": 240,
        "n_checks": 50,
        "n_outcomes": 1200,
        "skipped_outcomes": 0,
    }
    result = {
        "backend": runner.NAME,
        "jevbench_version": "0.1.0",
        "seed": 0,
        "trials": 1,
        "repeats": 1,
        "concurrency": 1,
        "truncated_by_budget": False,
        "backend_down": None,
        "suite": {"name": "jevbench-mini"},
        "requests": 1248,
        "cases": ["private_case_" + str(i) for i in range(240)],
        "case_domains": {"private_case_" + str(i): "private_domain" for i in range(240)},
        "relations": list(checks),
        "planned_tests": dict.fromkeys(checks, 24),
        "selection": {
            "domains": None,
            "dimensions": None,
            "groups": None,
            "relations": None,
            "diagnostics": False,
        },
        "cache_hits": 5,
        "errors": [{"error": "PRIVATE_ERROR_SENTINEL"}],
        "bases": {"PRIVATE_BASE_SENTINEL": {}},
        "outcomes": [{"detail": "PRIVATE_EXAMPLE_SENTINEL"}],
    }
    control = runner.UniformControl()
    # A synthetic metadata observation, never a benchmark request.
    control.requests = 1248
    return (
        SimpleNamespace(result=result, scores=scores, settings=dict(runner.SETTINGS)),
        control,
        checks,
    )


class UniformControlTests(unittest.TestCase):
    def test_all_wire_primitives_preserve_ids_order_and_inputs(self):
        class UnreadableState:
            def __str__(self):
                raise AssertionError("state inspected")

            def __repr__(self):
                raise AssertionError("state inspected")

        questions = {
            "n": {"type": "noul", "instructions": "fixture", "criteria": None},
            "c": {"type": "choice", "instructions": "fixture", "criteria": {"z": "Z", "a": "A"}},
            "s": {
                "type": "score",
                "instructions": "fixture",
                "criteria": ["low", "middle", "high"],
            },
        }
        original = copy.deepcopy(questions)
        state = UnreadableState()
        answers = runner.uniform_answers(state, questions)
        self.assertEqual(list(answers), ["n", "c", "s"])
        self.assertEqual(answers["n"], {"type": "noul", "noul": 0.5})
        self.assertEqual(list(answers["c"]["probabilities"]), ["z", "a"])
        self.assertEqual(answers["c"]["probabilities"], {"z": 0.5, "a": 0.5})
        self.assertEqual(
            answers["s"], {"type": "score", "probabilities": dict.fromkeys(["0", "1", "2"], 1 / 3)}
        )
        self.assertEqual(questions, original)
        with self.assertRaises(ValueError):
            runner.uniform_answers(None, {"q": {"type": "unknown"}})

    def test_uniform_can_be_invariant_but_fail_partition_mass(self):
        def answer(ids):
            return runner.uniform_answers(
                None, {"q": {"type": "choice", "criteria": dict.fromkeys(ids, "fixture")}}
            )["q"]["probabilities"]

        original = answer(["a", "b", "c", "d"])
        permuted = answer(["d", "c", "b", "a"])
        self.assertEqual(original, permuted)
        cloned = answer(["a", "a_alt", "b", "c", "d"])
        self.assertNotAlmostEqual(cloned["a"] + cloned["a_alt"], original["a"])

    def test_projection_keeps_nonzero_errors_but_never_raw_content(self):
        report, control, checks = fake_report()
        projected = runner.project_report(report, control, checks)
        encoded = json.dumps(projected)
        self.assertEqual(projected["official_errors"]["count"], 1)
        self.assertEqual(projected["official_metrics"]["overall"], 0.5)
        for marker in ("PRIVATE_", "private_case_", "private_domain"):
            self.assertNotIn(marker, encoded)
        self.assertFalse(projected["uses_model"])
        self.assertFalse(projected["ranking_qualification"])

    def test_invalid_metric_or_incomplete_inventory_refuses_projection(self):
        for invalid in (True, float("nan"), float("inf")):
            report, control, checks = fake_report()
            report.scores["overall"] = invalid
            with self.assertRaises(ValueError):
                runner.project_report(report, control, checks)
        report, control, checks = fake_report()
        report.result["requests"] = 1247
        with self.assertRaises(ValueError):
            runner.project_report(report, control, checks)

    def test_held_execute_stops_before_any_dependency_or_pin_access(self):
        with (
            patch.object(runner, "SOURCE_ONLY", True),
            patch.object(runner, "checked_file", side_effect=AssertionError("pin touched")) as pins,
            patch.object(
                runner.importlib, "import_module", side_effect=AssertionError("dependency imported")
            ) as imports,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(
                runner.main(
                    ["--execute", "--output", "/unused", "--expected-runner-sha256", "0" * 64]
                ),
                2,
            )
            pins.assert_not_called()
            imports.assert_not_called()


if __name__ == "__main__":
    unittest.main()
