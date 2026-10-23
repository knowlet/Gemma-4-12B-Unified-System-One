"""Stdlib-only synthetic campaign boundaries; no frameworks or benchmark data."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("batch4_campaign_tests", HERE / "campaign.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
batch = c.BATCH
bspec = importlib.util.spec_from_file_location("batch4_frozen_helpers_tests", batch / "actor.py")
b = importlib.util.module_from_spec(bspec)
bspec.loader.exec_module(b)
h = b.load_helper()


class Timer:
    def __init__(self, seconds, output, stage):
        self.seconds = seconds

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.helper = NS(
            write_json=h.write_json,
            diagnostic=h.diagnostic,
            ensure_elapsed=h.ensure_elapsed,
            Deadline=Timer,
        )

    def native(self, key="edge"):
        return c.read_json(batch / (key + "-plan.json"))

    def smoke(self, key="edge"):
        native = self.native(key)
        event = {
            "shape": [4, 135, 1024],
            "dtype": "torch.bfloat16",
            "device": "cpu",
            "inputs_or_outputs_modified": False,
        }
        return {
            "profile": native["profile"],
            "model_key": key,
            "plan_sha256": c.NATIVE_PLAN_SHA[key],
            "status": "native_batch4_two_call_synthetic_probe_complete_no_benchmark_claim",
            "model_loads": 1,
            "native_request_calls": 2,
            "benchmark_calls": 0,
            "browser_calls": 0,
            "cuda_initialized_final": False,
            "final_full_model_manifest_runtime_source_identities_unchanged": True,
            "elapsed_seconds": 10,
            "native_provenance": {
                "model_name": native["model"]["name"],
                "batch_size": 4,
                "model_sha256": native["model"]["model_sha256"],
                "files_sha256": {"MODEL_MANIFEST.json": native["model"]["manifest_sha256"]},
            },
            "calls": [
                {
                    "index": i,
                    "status": "validated",
                    "validation": {"contract_valid": True},
                    "actual_first_layer_event": copy.deepcopy(event),
                    "durable_validation_seconds": 1.2,
                    "validation_seconds": 1,
                }
                for i in range(2)
            ],
            "actual_first_layer_events": [copy.deepcopy(event) for _ in range(2)],
        }

    def save(self, name, value):
        p = self.output / name
        h.write_json(p, value)
        return p

    def test_same_model_actual_smoke_rejects_dry_batch1_gpu_late_missing_auth_or_rows(self):
        good = self.smoke()
        for mutate in [
            lambda x: x.update(status="source_ready"),
            lambda x: x.update(model_key="lite"),
            lambda x: x.update(model_loads=2),
            lambda x: x.update(cuda_initialized_final=True),
            lambda x: x.update(elapsed_seconds=120.0001),
            lambda x: x.update(final_full_model_manifest_runtime_source_identities_unchanged=False),
            lambda x: x["native_provenance"].update(batch_size=1),
            lambda x: x["calls"][1].update(durable_validation_seconds=20.00001),
            lambda x: x["actual_first_layer_events"][0].update(shape=[1, 135, 1024]),
        ]:
            invalid = copy.deepcopy(good)
            mutate(invalid)
            p = self.save("synthetic-smoke.json", invalid)
            with self.assertRaises(RuntimeError):
                c.check_smoke(p, c.sha(p), self.native())
        for key in ("edge", "lite"):
            p = self.save("synthetic-smoke.json", self.smoke(key))
            self.assertEqual(c.check_smoke(p, c.sha(p), self.native(key))["model_key"], key)
            with self.assertRaises(RuntimeError):
                c.check_smoke(p, "wrongSHA", self.native(key))

    def audit(self, state):
        native = self.native()
        refs = {}
        for name in ["state", "owner_receipt", "root_execution", "spec", "closure_observation"]:
            p = (
                state
                if name == "state"
                else self.save(name + ".json", {"synthetic_test_only": True})
            )
            refs[name] = {"path": str(p), "sha256": c.sha(p), "size_bytes": p.stat().st_size}
        p = batch / "edge-plan.json"
        refs["plan"] = {
            "path": str(p),
            "sha256": c.NATIVE_PLAN_SHA["edge"],
            "size_bytes": p.stat().st_size,
        }
        return {
            "schema": "independent-native-batch4-synthetic-result-audit-v1",
            "model_key": "edge",
            "review_kind": "closed_real_smoke_result",
            "result": "PASS_WITH_LIMITS",
            "qualification": dict.fromkeys(c.QUALIFICATIONS, True),
            "artifact_refs": refs,
            "model_revision": native["model"]["revision"],
            "model_identity_sha256": native["model"]["model_sha256"],
            "manifest_sha256": native["model"]["manifest_sha256"],
            "owner_unit": native["owner_unit"],
            "memory_max_bytes": native["memory_max_bytes"],
            "batch_size": 4,
        }

    def test_independent_audit_requires_exact_closed_state_plan_identity_and_all_refs(self):
        state = self.save("synthetic-smoke.json", self.smoke())
        audit = self.audit(state)
        p = self.save("synthetic-audit.json", audit)
        self.assertEqual(
            c.check_audit(p, c.sha(p), state, c.sha(state), self.native())["result"],
            "PASS_WITH_LIMITS",
        )
        for mutate in [
            lambda x: x["qualification"].update(external_authenticated_closure=False),
            lambda x: x["qualification"].update(whole_command_120s=False),
            lambda x: x.update(memory_max_bytes=2147483648),
            lambda x: x.update(model_key="lite"),
            lambda x: x["artifact_refs"].pop("root_execution"),
            lambda x: x["artifact_refs"]["state"].update(sha256="wrong"),
        ]:
            invalid = copy.deepcopy(audit)
            mutate(invalid)
            p = self.save("synthetic-audit.json", invalid)
            with self.assertRaises(RuntimeError):
                c.check_audit(p, c.sha(p), state, c.sha(state), self.native())
        p = self.save("synthetic-audit.json", audit)
        self.save("closure_observation.json", {"changed_after_audit": True})
        with self.assertRaises(RuntimeError):
            c.check_audit(p, c.sha(p), state, c.sha(state), self.native())

    def predictor(self, raw=None, failure=None):
        response = raw if raw is not None else {"answers": {"q": {"type": "noul", "noul": 0.4}}}
        seen = []

        class Backend:
            def __call__(self, state, questions):
                seen.append((state, questions))
                self.evidence_sink(
                    "prepared", {"lengths": {"q": 4}, "original_question_order": list(questions)}
                )
                self.evidence_sink("raw_returned_before_validation", response)
                if failure is not None:
                    raise failure
                return response

        p = c.TimedPredictor(
            Backend(), self.helper, self.output / "native", lambda: {"synthetic": True}
        )
        p.stage = c.STAGES[0]
        return p, response, seen

    def test_timed_bridge_retains_raw_before_validation_preserves_inputs_and_returns_same_object(
        self,
    ):
        p, response, seen = self.predictor()
        state, questions = "developer fixture", {"q": {"type": "noul"}}
        self.assertIs(p(state, questions), response)
        self.assertIs(seen[0][1], questions)
        self.assertEqual(p.completed, 1)
        self.assertEqual(p.raw_returned, 1)
        self.assertEqual(
            c.read_json(self.output / "native/00000-raw_returned_before_validation.json")["value"],
            response,
        )
        self.assertFalse(p.summary()[c.STAGES[0]]["startup_included"])
        self.assertFalse(p.summary()[c.STAGES[0]]["browser_20_second_qualification"])

    def test_nonfinite_original_retained_and_primary_validation_error_not_replaced(self):
        primary = ValueError("synthetic original validation failure")
        p, response, _ = self.predictor({"answers": {"q": {"noul": float("nan")}}}, primary)
        with self.assertRaises(ValueError) as caught:
            p("developer", {"q": {"type": "noul"}})
        self.assertIs(caught.exception, primary)
        self.assertTrue(response["answers"]["q"]["noul"] != response["answers"]["q"]["noul"])
        saved = c.read_json(self.output / "native/00000-raw_returned_before_validation.json")
        self.assertFalse(saved["valid_response_claim"])
        self.assertIn("nan", saved["invalid_native_response_diagnostic"]["repr_prefix"])
        self.assertEqual(p.completed, 0)
        self.assertFalse(any(q.name.startswith(".") for q in (self.output / "native").iterdir()))

    def test_failure_evidence_cannot_replace_primary(self):
        primary = ValueError("primary synthetic invalid response")
        p, _, _ = self.predictor(failure=primary)
        actual_write = self.helper.write_json

        def write(path, value):
            if Path(path).name.endswith("-failed.json"):
                raise OSError("synthetic failed evidence")
            actual_write(path, value)

        self.helper.write_json = write
        with self.assertRaises(ValueError) as caught:
            p("developer", {"q": {"type": "noul"}})
        self.assertIs(caught.exception, primary)
        self.assertIn("evidence also failed", primary.__notes__[0])

    def test_late_durable_progress_aborts_outside_harness_exception_catch_and_reverses_counts(self):
        p, _, _ = self.predictor()
        fake_clock = [0.0]
        actual_write = self.helper.write_json

        def write(path, value):
            actual_write(path, value)
            if Path(path).name == "progress.json" and value["completed_calls"] == 1:
                fake_clock[0] = 180.01

        self.helper.write_json = write
        with patch.object(c.time, "monotonic", side_effect=lambda: fake_clock[0]):
            with self.assertRaises(c.CampaignAbort):
                p("developer", {"q": {"type": "noul"}})
        self.assertEqual(p.completed, 0)
        self.assertEqual(p.latencies[c.STAGES[0]], [])

    def test_late_final_results_or_state_never_return_complete(self):
        for late_file in ["results.json", "state.json"]:
            fake_clock = [0.0]
            helper = NS(write_json=None, ensure_elapsed=h.ensure_elapsed)

            def write(path, value):
                if Path(path).name == late_file:
                    fake_clock[0] = 10000.01

            helper.write_json = write
            with patch.object(c.time, "monotonic", side_effect=lambda: fake_clock[0]):
                with self.assertRaises(RuntimeError):
                    c.durable_success(
                        helper, self.output, {"latencies": {}, "model_identity": {}}, {}, 0
                    )

    def test_failure_evidence_is_also_inside_call_watchdog_and_late_checks(self):
        primary = ValueError("primary invalid synthetic response")
        p, _, _ = self.predictor(failure=primary)
        clock, timer_active = [0.0], [False]

        class ObservedTimer(Timer):
            def __enter__(self):
                timer_active[0] = True
                return self

            def __exit__(self, *_):
                timer_active[0] = False
                return False

        self.helper.Deadline = ObservedTimer
        actual_write = self.helper.write_json

        def write(path, value):
            if Path(path).name.endswith("-failed.json"):
                self.assertTrue(timer_active[0])
                clock[0] = 180.01
            actual_write(path, value)

        self.helper.write_json = write
        with patch.object(c.time, "monotonic", side_effect=lambda: clock[0]):
            with self.assertRaises(c.CampaignAbort) as caught:
                p("developer", {"q": {"type": "noul"}})
        self.assertIs(caught.exception.__cause__, primary)
        self.assertFalse(timer_active[0])

    def test_original_harness_interfaces_and_populations_remain_unchanged(self):
        calls = []
        predictor = NS(stage=None, answers=lambda state, questions: questions)
        identity = {"profile": c.profile("edge")}

        def public(p, root, output, model):
            calls.append(("public", p.stage, root, model))
            output.mkdir()
            h.write_json(output / "manifest.json", {"n_attempted": 231})
            return {"synthetic_score_only": 0.5}

        report = NS(
            result={
                "requests": 1248,
                "truncated_by_budget": False,
                "case_domains": list(range(240)),
                "planned_tests": dict.fromkeys(range(50), 24),
            },
            overall=0.4,
            interval=[0.3, 0.5],
            dimensions={},
            errors={},
            scores={},
        )
        report.save = lambda path: calls.append(("save", Path(path).name))

        def evaluate(model, **kwargs):
            calls.append(("coherence", predictor.stage, model, kwargs))
            return report

        jb = NS(from_callable=lambda fn, **kwargs: (fn, kwargs), evaluate=evaluate)
        result = c.run_unchanged_suites(
            predictor,
            identity,
            self.output,
            run_public=public,
            heaven_root="synthetic primary source root",
            jb=jb,
        )
        self.assertEqual(calls[0][1], c.STAGES[0])
        self.assertEqual(calls[1][1], c.STAGES[1])
        self.assertEqual(calls[1][2][1], {"name": c.profile("edge")})
        self.assertEqual(calls[1][3]["concurrency"], 1)
        self.assertEqual(calls[1][3]["suite"], "jevbench-mini")
        self.assertEqual(result["coherence_mini"]["planned_tests"], 1200)
        self.assertEqual(result["profile"], identity["profile"])
        report.result["truncated_by_budget"] = True
        with self.assertRaises(RuntimeError):
            c.run_unchanged_suites(
                predictor,
                identity,
                self.output,
                run_public=lambda *args: {},
                heaven_root="synthetic",
                jb=jb,
            )

    def test_full_call_counts_and_latency_population_refuse_partial_or_cached_campaigns(self):
        p = NS(
            started=1479,
            completed=1479,
            raw_returned=1479,
            latencies={c.STAGES[0]: [1] * 231, c.STAGES[1]: [2] * 1248},
        )
        c.check_counts(p)
        p.raw_returned = 1478
        with self.assertRaises(RuntimeError):
            c.check_counts(p)
        p.raw_returned = 1479
        p.latencies[c.STAGES[1]].pop()
        with self.assertRaises(RuntimeError):
            c.check_counts(p)
        self.assertEqual(c.latency_distribution([9, 1, 3, 2])["median_seconds"], 2.5)
        self.assertEqual(c.latency_distribution([9, 1, 3, 2])["p95_seconds"], 9)

    def test_both_model_plans_are_separate_with_provenance_only_head(self):
        for key in ("edge", "lite"):
            p = HERE / (key + "-plan.json")
            plan = c.load_plan(p, c.sha(p))
            self.assertEqual(plan["profile"], c.profile(key))
            self.assertTrue(plan["head_field_is_provenance_only"])
            invalid = copy.deepcopy(plan)
            invalid["batch_size"] = 1
            q = self.save("bad-plan.json", invalid)
            with self.assertRaises(RuntimeError):
                c.load_plan(q, c.sha(q))

    def test_default_dry_requires_no_plan_smoke_framework_or_benchmark(self):
        run = subprocess.run(
            [sys.executable, "-B", "-s", str(HERE / "campaign.py")],
            capture_output=True,
            text=True,
            check=True,
        )
        info = json.loads(run.stdout)
        self.assertEqual(info["status"], "dry_default_no_model_or_benchmark_execution")
        self.assertEqual(info["fresh_calls"], 1479)
        self.assertEqual(info["whole_campaign_seconds"], 10000)
        self.assertFalse(info["browser20_qualification_or_batch1_scores_inherited"])
        self.assertFalse(
            {"torch", "transformers", "jevbench", "numpy", "safetensors"}.intersection(sys.modules)
        )

    def test_frozen_closed_audit_schema_matches_without_reading_referenced_artifacts(self):
        expected = {
            "edge": "8c95ce973d1f634f73d58682f5ef4547137479ef4d2c7fb9e07cdeb093eb3691",
            "lite": "21f1bc822c86b35a5d8efd8a0f0f27a1f1b09088944dc7457a3c10fa375bc5fd",
        }
        for key, digest in expected.items():
            audit = c.selected_small_json(
                batch / ("independent-" + key + "-actual-review.json"), digest
            )
            self.assertEqual(audit["schema"], "independent-native-batch4-synthetic-result-audit-v1")
            self.assertEqual(audit["model_key"], key)
            self.assertEqual(audit["model_revision"], self.native(key)["model"]["revision"])
            self.assertTrue(all(audit["qualification"][name] is True for name in c.QUALIFICATIONS))
            self.assertTrue(
                {"state", "plan", "owner_receipt", "root_execution", "spec", "closure_observation"}
                <= audit["artifact_refs"].keys()
            )
            self.assertTrue(
                all(
                    {"path", "sha256", "size_bytes"} <= ref.keys()
                    for ref in audit["artifact_refs"].values()
                )
            )


if __name__ == "__main__":
    unittest.main()
