"""Stdlib-only synthetic campaign boundaries; no frameworks or benchmark data."""

import ast
import copy
import hashlib
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


def public_metrics():
    return {
        **dict.fromkeys(
            ("n_planned", "n_attempted", "n_scorable", "n_valid", "n_correct", "calibration_n"), 231
        ),
        "n_renormalized": 0,
        "complete": True,
        **dict.fromkeys(
            (
                "accuracy",
                "macro_accuracy",
                "coverage",
                "schema_validity",
                "schema_validity_strict",
                "operational_success",
                "brier_mean",
                "ece",
                "ordinal_mae",
                "price_per_1000_decisions_usd",
                "ledger_charged_usd",
            ),
            None,
        ),
        "ece": {
            "ece": 0.5,
            "n": 231,
            "bins": [
                {
                    "lo": i / 10,
                    "hi": (i + 1) / 10,
                    "n": 231 if i == 5 else 0,
                    "mean_confidence": 0.5 if i == 5 else None,
                    "accuracy": 1.0 if i == 5 else None,
                }
                for i in range(10)
            ],
        },
    }


def mini_metrics():
    return {
        "overall": 0.25,
        "interval": [0.2, 0.3],
        "requests": 1248,
        "cases": 240,
        "relations": 50,
        "planned_tests": 1200,
    }


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
            "selected_owner_host_observation_policy": native["host_observation_policy"],
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
            lambda x: x.pop("selected_owner_host_observation_policy"),
            lambda x: x.update(selected_owner_host_observation_policy="original_idle_only"),
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
        for key in ("edge",):
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
            "host_observation_policy": native["host_observation_policy"],
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
            lambda x: x.update(memory_max_bytes=3221225472),
            lambda x: x.update(model_key="lite"),
            lambda x: x.pop("host_observation_policy"),
            lambda x: x.update(host_observation_policy="original_idle_only"),
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

    def test_callback_cleanup_expiry_or_raise_reverses_complete_count_and_keeps_failure(self):
        for mode in ("expired", "raised"):
            with self.subTest(mode=mode):
                self.output = Path(self.temp.name) / mode
                self.output.mkdir()
                p, _, _ = self.predictor()
                clock = [0.0]
                cleanup = BaseException("synthetic cleanup refusal")

                class CleanupTimer(Timer):
                    def __exit__(self, *_):
                        if mode == "raised":
                            raise cleanup
                        clock[0] = c.CALL_SECONDS + 0.01
                        return False

                self.helper.Deadline = CleanupTimer
                with patch.object(c.time, "monotonic", side_effect=lambda: clock[0]):
                    with self.assertRaises(c.CampaignAbort) as caught:
                        p("synthetic", {"q": {"type": "noul"}})
                if mode == "raised":
                    self.assertIs(caught.exception.__cause__, cleanup)
                self.assertEqual(p.completed, 0)
                self.assertEqual(p.latencies[c.STAGES[0]], [])
                self.assertIsNone(p.active)
                self.assertEqual(
                    c.read_json(self.output / "native/progress.json")["completed_calls"], 0
                )
                failed = c.read_json(self.output / "native/00000-failed.json")
                self.assertEqual(failed["status"], "failed_no_valid_return")

    def test_callback_cleanup_error_preserves_original_primary(self):
        primary = ValueError("synthetic primary")
        p, _, _ = self.predictor(failure=primary)
        cleanup = BaseException("secondary cleanup")

        class CleanupTimer(Timer):
            def __exit__(self, *_):
                raise cleanup

        self.helper.Deadline = CleanupTimer
        with self.assertRaises(ValueError) as caught:
            p("synthetic", {"q": {"type": "noul"}})
        self.assertIs(caught.exception, primary)
        self.assertIn("Deadline cleanup also failed", caught.exception.__notes__[-1])
        self.assertEqual(p.completed, 0)

    def test_callback_accepted_latency_includes_successful_deadline_cleanup(self):
        p, response, _ = self.predictor()
        clock = [0.0]

        class CleanupTimer(Timer):
            def __exit__(self, *_):
                clock[0] = 2.0
                return False

        self.helper.Deadline = CleanupTimer
        with patch.object(c.time, "monotonic", side_effect=lambda: clock[0]):
            self.assertIs(p("synthetic", {"q": {"type": "noul"}}), response)
        self.assertEqual(p.latencies[c.STAGES[0]], [2.0])

    def execute_with_fake_campaign(
        self,
        timer,
        *,
        primary=None,
        use_real_stages=False,
        later_stage_error=None,
        stage_auth_error=None,
    ):
        """Run actual execute control flow; replace model/scorers with stdlib fakes."""
        native = self.native()
        plan = {
            "model_key": "edge",
            "profile": c.profile("edge"),
            "local_head_at_creation": "synthetic",
            "campaign_sha256": "synthetic-campaign",
        }
        source = [{"path": str(c.ROOT / "scripts/run_jevbench_public.py"), "sha256": "fake"}]
        source += [{"path": f"/FAKE-source-{i}", "sha256": "fake"} for i in range(77)]
        benchmark = {
            "source_files": source,
            "existing_dataset_metadata": [{}, {}],
            "public": {"root": "/FAKE-public", "revision": "fake"},
            "coherence": {"root": "/FAKE-coherence", "revision": "fake"},
        }
        helper = NS(
            Deadline=timer,
            START_ENV={},
            process_identity=lambda: {
                "pid": 1234,
                "starttime_ticks": 456,
                "pgid": 1234,
                "sid": 1234,
                "cgroup": "0::/synthetic/" + native["owner_unit"],
                "affinity": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                "executable": "/FAKE-interpreter",
                "environment": {},
            },
            owned_limits=lambda _: {
                "memory.max": "4294967296",
                "memory.swap.max": "0",
                "cpu.max": "1600000 100000",
            },
            host_gate=lambda: {},
            check_record=lambda *a, **k: None,
            write_json=h.write_json,
            ensure_elapsed=h.ensure_elapsed,
            finalize_state=h.finalize_state,
            diagnostic=h.diagnostic,
        )
        fake_torch = NS(
            cuda=NS(is_initialized=lambda: False),
            get_num_threads=lambda: 16,
            get_num_interop_threads=lambda: 1,
        )

        def load_runtime(helper, native, output, state):
            if primary is not None:
                raise primary
            state["model_loads"] = 1
            return NS(), fake_torch

        fake_jb = NS(__file__="/FAKE-coherence/src/jevbench/__init__.py")

        def public_import(*_):
            sys.modules["jevbench"] = fake_jb
            return NS(run_public=lambda *a: None)

        shared = NS(
            load_helper=lambda: helper,
            load_plan=lambda *a: native,
            validate_owner=lambda *a: None,
            own_counters=lambda *a: {},
            load_runtime=load_runtime,
            validate_runtime=lambda *a: None,
            authenticate=(lambda *a: (_ for _ in ()).throw(stage_auth_error))
            if stage_auth_error
            else (lambda *a: None),
            NativeBatch4Backend=lambda *a, **k: NS(),
            import_verified=public_import,
            HELPER=b.HELPER,
            HELPER_SHA=b.HELPER_SHA,
            SOURCE_AUDIT_SHA=b.SOURCE_AUDIT_SHA,
        )
        plan_path, smoke_path, audit_path = map(Path, ("/FAKE-plan", "/FAKE-smoke", "/FAKE-audit"))
        hashes = {
            str(plan_path): "plan",
            str(smoke_path): "smoke",
            str(audit_path): "audit",
            str(Path(c.__file__)): plan["campaign_sha256"],
            str(batch / "actor.py"): c.BATCH_ACTOR_SHA,
            str(batch / "freeze.json"): c.BATCH_FREEZE_SHA,
            str(batch / "edge-plan.json"): c.NATIVE_PLAN_SHA["edge"],
            str(c.BENCHMARK_PLAN): c.BENCHMARK_PLAN_SHA,
            str(b.HELPER): b.HELPER_SHA,
            str(batch / "source_audit.py"): b.SOURCE_AUDIT_SHA,
        }

        def suites(predictor, *a, **k):
            if use_real_stages:
                predictor.stage = c.STAGES[0]
                predictor.started = predictor.completed = predictor.raw_returned = 231
                predictor.latencies = {c.STAGES[0]: [0.0] * 231}
                k["complete_stage"](c.STAGES[0], public_metrics(), c.time.monotonic())
                if later_stage_error is not None:
                    raise later_stage_error
                predictor.stage = c.STAGES[1]
                predictor.started = predictor.completed = predictor.raw_returned = 1479
                predictor.latencies = {stage: [0.0] * n for stage, n in c.COUNTS.items()}
                k["complete_stage"](c.STAGES[1], mini_metrics(), c.time.monotonic())
            predictor.started = predictor.completed = predictor.raw_returned = 1479
            predictor.latencies = {stage: [0.0] * n for stage, n in c.COUNTS.items()}
            return {"synthetic_no_benchmark": True}

        output = self.output / "campaign"
        original_sha = c.sha
        with (
            patch.object(c.os, "environ", {"JEV_COMMAND_NONCE": "jev-d3-boundary-" + "a" * 32}),
            patch.object(c, "load_plan", return_value=plan),
            patch.object(
                c,
                "exec",
                side_effect=lambda code, namespace: namespace.update(vars(shared)),
                create=True,
            ),
            patch.object(c, "check_smoke"),
            patch.object(c, "check_audit", return_value={}),
            patch.object(c, "selected_small_json", return_value=benchmark),
            patch.object(
                c,
                "sha",
                side_effect=lambda path: (
                    hashes[str(Path(path))] if str(Path(path)) in hashes else original_sha(path)
                ),
            ),
            patch.object(c, "run_unchanged_suites", side_effect=suites),
            patch.dict(sys.modules),
            patch.object(c, "HERE", self.output),
            patch.object(
                c,
                "StageCompletions",
                c.StageCompletions
                if use_real_stages
                else lambda *a: NS(complete=lambda *a: None, verify_all=lambda: None),
            ),
        ):
            return c.execute(plan_path, "plan", smoke_path, "smoke", audit_path, "audit", output)

    def test_real_stage_flow_full_auth_failure_never_qualifies_public(self):
        primary = ValueError("synthetic stage full auth refused")
        with self.assertRaises(ValueError) as caught:
            self.execute_with_fake_campaign(Timer, use_real_stages=True, stage_auth_error=primary)
        self.assertIs(caught.exception, primary)
        failed = c.read_json(self.output / "campaign/state.json")
        self.assertEqual(failed["completed_stage_receipts"], {})
        self.assertIsNone(failed["benchmark_scores"])
        self.assertEqual(
            c.read_json(
                self.output / "campaign/stage-completions" / c.STAGES[0] / "completion.json"
            )["status"],
            "failed_no_stage_qualification",
        )

    def test_real_stage_flow_later_suite_failure_preserves_public_and_overall_false(self):
        primary = c.CampaignAbort("synthetic later suite admission interrupt")
        with self.assertRaises(c.CampaignAbort) as caught:
            self.execute_with_fake_campaign(Timer, use_real_stages=True, later_stage_error=primary)
        self.assertIs(caught.exception, primary)
        failed = c.read_json(self.output / "campaign/state.json")
        self.assertEqual(failed["status"], "failed_or_incomplete_no_complete_score_claim")
        self.assertIsNone(failed["benchmark_scores"])
        self.assertEqual(set(failed["completed_stage_receipts"]), {c.STAGES[0]})
        ref = failed["completed_stage_receipts"][c.STAGES[0]]
        receipt = c.selected_small_json(ref["path"], ref["sha256"])
        self.assertFalse(receipt["overall_campaign_complete"])
        self.assertTrue(receipt["external_current_command_and_owner_closure_required"])
        evidence = c.selected_small_json(receipt["evidence"]["path"], receipt["evidence"]["sha256"])
        self.assertEqual(evidence["counts"]["stage_callbacks"], 231)
        self.assertTrue(evidence["full_authentication"]["unchanged_batch_authenticate"])

    def test_whole_cleanup_expiry_or_raise_never_returns_complete_or_keeps_score_claim(self):
        for mode in ("expired", "raised"):
            with self.subTest(mode=mode):
                self.output = Path(self.temp.name) / mode
                self.output.mkdir()
                clock = [0.0]
                cleanup = BaseException("synthetic whole cleanup refusal")

                class CleanupTimer(Timer):
                    def __exit__(self, *_):
                        if mode == "raised":
                            raise cleanup
                        clock[0] = c.WALL_SECONDS + 0.01
                        return False

                with patch.object(c.time, "monotonic", side_effect=lambda: clock[0]):
                    with self.assertRaises(BaseException) as caught:
                        self.execute_with_fake_campaign(CleanupTimer)
                if mode == "raised":
                    self.assertIs(caught.exception, cleanup)
                else:
                    self.assertIsInstance(caught.exception, RuntimeError)
                    self.assertIn("after finalization and deadline cleanup", str(caught.exception))
                failed = c.read_json(self.output / "campaign/state.json")
                self.assertEqual(failed["status"], "failed_or_incomplete_no_complete_score_claim")
                self.assertIsNone(failed["benchmark_scores"])

    def test_whole_cleanup_failure_keeps_existing_primary(self):
        primary = ValueError("synthetic primary model refusal")

        class CleanupTimer(Timer):
            def __exit__(self, *_):
                raise BaseException("synthetic secondary cleanup")

        with self.assertRaises(ValueError) as caught:
            self.execute_with_fake_campaign(CleanupTimer, primary=primary)
        self.assertIs(caught.exception, primary)
        self.assertIn("Deadline cleanup also failed", primary.__notes__[-1])

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
            complete_stage=lambda stage, aggregate, began: calls.append(("stage", stage)),
        )
        self.assertEqual(calls[0][1], c.STAGES[0])
        self.assertEqual(calls[1], ("stage", c.STAGES[0]))
        self.assertEqual(calls[2][1], c.STAGES[1])
        self.assertEqual(calls[2][2][1], {"name": c.profile("edge")})
        self.assertEqual(calls[2][3]["concurrency"], 1)
        self.assertEqual(calls[2][3]["suite"], "jevbench-mini")
        self.assertEqual(result["coherence_mini"]["planned_tests"], 1200)
        self.assertEqual(result["profile"], identity["profile"])
        self.assertEqual(calls[3], ("save", "coherence-mini.json"))
        self.assertEqual(calls[4], ("stage", c.STAGES[1]))
        report.result["truncated_by_budget"] = True
        with self.assertRaises(RuntimeError):
            c.run_unchanged_suites(
                predictor,
                identity,
                self.output,
                run_public=lambda *args: {},
                heaven_root="synthetic",
                jb=jb,
                complete_stage=lambda *a: None,
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

    def test_edge_cpu16_4g_plan_has_distinct_profile_and_provenance_only_head(self):
        for key in ("edge",):
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

    def test_pending_prerequisites_cannot_qualify_smoke_or_audit(self):
        pending_path = HERE / "pending-prerequisites.json"
        pending = c.read_json(pending_path)
        self.assertIsNone(pending["smoke_receipt"])
        self.assertIsNone(pending["closed_independent_audit"])
        self.assertFalse(pending["actual4GiB_smoke_or_audit_available"])
        with self.assertRaises(RuntimeError):
            c.check_smoke(pending_path, c.sha(pending_path), self.native())
        state = self.save("synthetic-state.json", self.smoke())
        with self.assertRaises(RuntimeError):
            c.check_audit(pending_path, c.sha(pending_path), state, c.sha(state), self.native())

    def test_resource_cap_mixed_owner_and_shortened_budget_refuse(self):
        selected = HERE / "edge-plan.json"
        plan = c.load_plan(selected, c.sha(selected))
        for change in (
            {"memory_max_bytes": 4294967295},
            {"memory_max_bytes": 4294967297},
            {"memory_max_bytes": 3221225472},
            {"startup_available_bytes": 30414140211},
            {"startup_available_bytes": 26119172916},
            {"reserve_bytes": 26119172915},
            {"owner_unit": "jev-d3-llama-cpp-cpu4-3g-long-20261023-05.service"},
            {"owner_profile": "llama-cpp-cpu4-3g-long-command-owned-v1"},
            {"owner_unit": "jev-d3-llama-cpp-cpu4-2g-long-20261023-06.service"},
            {"owner_profile": "llama-cpp-cpu4-2g-long-command-owned-v1"},
            {"host_observation_policy": None},
            {"host_observation_policy": "original_idle_only"},
            {"owner_runtime_seconds": 7200},
            {"whole_campaign_seconds": 9999},
            {"per_call_seconds": 179},
            {"context_limit": 16383},
            {"stage_completion_schema": None},
            {"stage_completion_policy": "publish_without_full_auth"},
        ):
            invalid = copy.deepcopy(plan)
            invalid.update(change)
            path = self.save("invalid-resource-plan.json", invalid)
            with self.assertRaises(RuntimeError):
                c.load_plan(path, c.sha(path))
        self.assertEqual(set(c.NATIVE_PLAN_SHA), {"edge"})

    def test_prior_profiles_or_plans_cannot_qualify_new_coexist_profile(self):
        for change in (
            {"profile": "decision3-edge-publisher-cpu-batch4-synthetic-20261023-v1"},
            {"plan_sha256": "be0ce6e6c7fb3588218d51f884e182ec10b54b8547fd3840533b60b6ebb01ae7"},
            {"profile": "decision3-edge-publisher-cpu-batch4-2g-synthetic-20261023-v1"},
            {"plan_sha256": "115b7850ac80b35aa0aacb5880644e1ee7ea79deb59beb748c7947c86d0492e8"},
        ):
            invalid = self.smoke()
            invalid.update(change)
            path = self.save("synthetic-prior3GiB-smoke.json", invalid)
            with self.assertRaises(RuntimeError):
                c.check_smoke(path, c.sha(path), self.native())


class StageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.clock = [0.0]
        self.clock_patch = patch.object(c.time, "monotonic", side_effect=lambda: self.clock[0])
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.env_patch = patch.object(
            c.os, "environ", {"JEV_COMMAND_NONCE": "jev-d3-boundary-" + "a" * 32}
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.identity = {
            "pid": 123,
            "starttime_ticks": 456,
            "pgid": 123,
            "sid": 123,
            "cgroup": "0::/synthetic-unit",
            "affinity": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "executable": "/FAKE-python",
            "environment": {},
        }
        self.limits = {
            "memory.max": "4294967296",
            "memory.swap.max": "0",
            "cpu.max": "1600000 100000",
        }
        self.helper = NS(
            write_json=h.write_json,
            ensure_elapsed=h.ensure_elapsed,
            Deadline=Timer,
            process_identity=lambda: copy.deepcopy(self.identity),
            owned_limits=lambda _: self.limits,
            START_ENV={},
        )
        self.p = NS(
            stage=c.STAGES[0],
            active=None,
            started=231,
            completed=231,
            raw_returned=231,
            latencies={c.STAGES[0]: [0.0] * 231},
            summary=lambda: {s: c.latency_distribution(v) for s, v in self.p.latencies.items()},
        )
        self.state = {
            "profile": c.profile("edge"),
            "model_key": "edge",
            "plan_sha256": "synthetic",
            "owned_limits": self.limits,
            "completed_stage_receipts": {},
            "benchmark_scores": None,
        }
        self.auth_calls = []
        self.native = {
            "owner_unit": "synthetic-unit",
            "owner_profile": "synthetic-profile",
            "model": {"name": "synthetic-no-NN"},
            "manifest": {},
            "model_files": [],
            "runtime_files": [],
            "owner_sources": [],
        }
        self.stage = c.StageCompletions(
            self.helper,
            NS(validate_owner=lambda *a: None),
            self.native,
            {"source_files": [], "existing_dataset_metadata": []},
            copy.deepcopy(self.identity),
            self.output,
            self.state,
            self.p,
            0.0,
            1000.0,
            lambda: self.auth_calls.append("full_auth"),
            [],
        )

    def complete_public(self):
        return self.stage.complete(c.STAGES[0], public_metrics(), 0.0)

    def test_both_stages_require_exact_order_counts_and_full_auth_before_receipt(self):
        with self.assertRaises(RuntimeError):
            self.stage.complete(c.STAGES[1], mini_metrics(), 0.0)
        self.p.completed = 230
        with self.assertRaises(RuntimeError):
            self.complete_public()
        self.p.completed = 231
        ref = self.complete_public()
        self.assertEqual(self.auth_calls, ["full_auth"])
        self.assertEqual(self.stage.completed, [c.STAGES[0]])
        with self.assertRaises(RuntimeError):
            self.stage.verify_all()
        self.p.stage = c.STAGES[1]
        self.p.started = self.p.completed = self.p.raw_returned = 1479
        self.p.latencies[c.STAGES[1]] = [0.0] * 1248
        self.stage.complete(c.STAGES[1], mini_metrics(), 0.0)
        self.stage.verify_all()
        self.assertEqual(self.auth_calls, ["full_auth", "full_auth"])
        self.assertEqual(c.sha(ref["path"]), ref["sha256"])

    def test_stage_full_auth_consumes_same_remaining_whole_budget(self):
        self.clock[0] = 10.0
        selected_seconds = []

        class ObservedTimer(Timer):
            def __init__(self, seconds, output, stage):
                selected_seconds.append(seconds)
                super().__init__(seconds, output, stage)

        self.helper.Deadline = ObservedTimer
        self.stage.authenticate = lambda: self.clock.__setitem__(0, c.WALL_SECONDS + 0.01)
        with self.assertRaisesRegex(RuntimeError, "stage full authentication"):
            self.complete_public()
        self.assertEqual(selected_seconds, [c.WALL_SECONDS - 10.0])
        self.assertEqual(self.stage.completed, [])
        self.assertEqual(self.state["completed_stage_receipts"], {})

    def test_stage_cleanup_expiry_demotes_current_receipt(self):
        clock = self.clock

        class LateTimer(Timer):
            def __exit__(self, *_):
                clock[0] = c.WALL_SECONDS + 0.01
                return False

        self.helper.Deadline = LateTimer
        with self.assertRaisesRegex(RuntimeError, "after deadline cleanup"):
            self.complete_public()
        self.assertEqual(self.state["completed_stage_receipts"], {})
        self.assertEqual(
            c.read_json(self.stage.output / c.STAGES[0] / "completion.json")["status"],
            "failed_no_stage_qualification",
        )

    def test_late_completion_or_final_handoff_write_never_qualifies(self):
        for target in ("completion.json", "state.json"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                self.stage.output = directory / "stage-completions"
                self.stage.output.mkdir()
                self.clock[0] = 0.0

                def write(path, value):
                    h.write_json(path, value)
                    if (
                        Path(path).name == target
                        and value.get("status") != "failed_no_stage_qualification"
                    ):
                        self.clock[0] = c.WALL_SECONDS + 0.01

                self.helper.write_json = write
                with self.assertRaises(RuntimeError):
                    self.complete_public()
                self.assertEqual(self.state["completed_stage_receipts"], {})
                self.assertEqual(
                    c.read_json(self.stage.output / c.STAGES[0] / "completion.json")["status"],
                    "failed_no_stage_qualification",
                )
                self.assertEqual(self.stage.completed, [])

    def test_primary_auth_error_survives_cleanup_and_failure_demotion_errors(self):
        primary = ValueError("synthetic primary authentication refusal")

        class BadTimer(Timer):
            def __exit__(self, *_):
                raise BaseException("synthetic cleanup failure")

        self.helper.Deadline = BadTimer
        self.stage.authenticate = lambda: (_ for _ in ()).throw(primary)
        self.helper.write_json = lambda *a: (_ for _ in ()).throw(
            OSError("synthetic evidence refusal")
        )
        with self.assertRaises(ValueError) as caught:
            self.complete_public()
        self.assertIs(caught.exception, primary)
        self.assertTrue(any("Deadline cleanup also failed" in v for v in primary.__notes__))
        self.assertTrue(any("Stage failure demotion also failed" in v for v in primary.__notes__))
        self.assertEqual(self.state["completed_stage_receipts"], {})

    def test_owner_nonce_or_process_change_refuses_stage(self):
        for changed in ("nonce", "starttime_ticks", "cgroup"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temp:
                self.stage.output = Path(temp)
                if changed == "nonce":
                    with patch.dict(
                        c.os.environ, {"JEV_COMMAND_NONCE": "jev-d3-boundary-" + "b" * 32}
                    ):
                        with self.assertRaisesRegex(RuntimeError, "binding changed"):
                            self.complete_public()
                else:
                    current = copy.deepcopy(self.identity)
                    current[changed] = 999 if changed == "starttime_ticks" else "0::/other-unit"
                    with patch.object(self.helper, "process_identity", return_value=current):
                        with self.assertRaisesRegex(RuntimeError, "binding changed"):
                            self.complete_public()
                self.assertEqual(self.auth_calls, [])

    def test_metric_projection_rejects_nonfinite_and_drops_item_content(self):
        public = public_metrics()
        public.update(raw_tasks="synthetic forbidden field", response="synthetic forbidden field")
        self.assertNotIn("raw_tasks", c.stage_metrics(c.STAGES[0], public))
        self.assertNotIn("response", c.stage_metrics(c.STAGES[0], public))
        public["accuracy"] = float("nan")
        with self.assertRaises(RuntimeError):
            c.stage_metrics(c.STAGES[0], public)
        mini = mini_metrics()
        mini.update(errors=["synthetic per-item details"], scores={"synthetic-item": 1})
        self.assertNotIn("errors", c.stage_metrics(c.STAGES[1], mini))
        self.assertNotIn("scores", c.stage_metrics(c.STAGES[1], mini))
        mini["interval"] = [0.0, float("inf")]
        with self.assertRaises(RuntimeError):
            c.stage_metrics(c.STAGES[1], mini)

    def test_durable_reread_or_directory_fsync_failure_refuses(self):
        for target in ("reread", "fsync"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp:
                self.stage.output = Path(temp)
                context = (
                    patch.object(c, "read_json", return_value={"changed": True})
                    if target == "reread"
                    else patch.object(c.os, "fsync", side_effect=OSError("synthetic fsync failure"))
                )
                with context, self.assertRaises((RuntimeError, OSError)):
                    self.complete_public()
                self.assertEqual(self.stage.completed, [])
                self.assertEqual(self.state["completed_stage_receipts"], {})


class OfficialECEProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute only this exact pinned pure metric function on invented pairs.
        path = (
            c.ROOT
            / "artifacts/jevbench/20261022-local-score-01/upstream/heaven/jevbench/metrics.py"
        )
        raw = path.read_bytes()
        if (
            hashlib.sha256(raw).hexdigest()
            != "4038c8423ad1cf28601afb6babe777b71f08e379b9ca125b09b52c0ac7d60515"
        ):
            raise RuntimeError("pinned official ECE source changed")
        nodes = [
            n
            for n in ast.parse(raw).body
            if isinstance(n, ast.FunctionDef) and n.name == "ece_top_label"
        ]
        if len(nodes) != 1:
            raise RuntimeError("unique official ECE function required")
        namespace = {}
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec", dont_inherit=True),
            namespace,
        )
        cls.official_ece = staticmethod(namespace["ece_top_label"])

    def aggregate(self, pairs=None):
        pairs = [(0.0, False), (0.35, True), (0.35, False), (1.0, True)] if pairs is None else pairs
        result = public_metrics()
        result["calibration_n"] = len(pairs)
        result["ece"] = self.official_ece(pairs) if pairs else None
        return result

    def test_actual_official_function_invented_pairs_preserved_without_alias(self):
        aggregate = self.aggregate()
        before = copy.deepcopy(aggregate)
        projected = c.stage_metrics(c.STAGES[0], aggregate)
        self.assertEqual(projected["ece"], aggregate["ece"])
        self.assertEqual(aggregate, before)
        self.assertIsNot(projected["ece"], aggregate["ece"])
        self.assertIsNot(projected["ece"]["bins"], aggregate["ece"]["bins"])
        projected["ece"]["bins"][0]["n"] = 999
        self.assertEqual(aggregate, before)

    def test_frozen_full05_refuses_same_legitimate_official_object(self):
        path = (
            c.ROOT
            / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-both-20261024-05/campaign.py"
        )
        raw = path.read_bytes()
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "11d81a7e5e1010e23870667f936a29bb978e82ee0f5c5a264752ce3ae1cfd017",
        )
        predecessor = NS(__file__=str(path), __name__="frozen_full05_projection_regression")
        exec(compile(raw, str(path), "exec", dont_inherit=True), vars(predecessor))
        aggregate = self.aggregate()
        with self.assertRaisesRegex(
            RuntimeError, "official metric must be finite numeric or absent"
        ):
            predecessor.stage_metrics(predecessor.STAGES[0], aggregate)
        self.assertEqual(c.stage_metrics(c.STAGES[0], aggregate)["ece"], aggregate["ece"])

    def test_absent_and_zero_pair_official_ece_semantics(self):
        empty = self.aggregate([])
        self.assertIsNone(c.stage_metrics(c.STAGES[0], empty)["ece"])
        empty["ece"] = self.official_ece([])
        self.assertEqual(c.stage_metrics(c.STAGES[0], empty)["ece"], empty["ece"])
        wrong = self.aggregate()
        wrong["ece"] = None
        with self.assertRaises(RuntimeError):
            c.stage_metrics(c.STAGES[0], wrong)

    def test_scalar_string_and_malformed_ece_refused(self):
        for value in (0.1, "0.1", [], {}, {"ece": 0.1, "n": 4}, True):
            with self.subTest(value=value):
                aggregate = self.aggregate()
                aggregate["ece"] = value
                with self.assertRaises(RuntimeError):
                    c.stage_metrics(c.STAGES[0], aggregate)

    def test_nonfinite_wrong_types_ranges_and_counts_refused(self):
        mutations = [
            lambda e: e.update(ece=float("nan")),
            lambda e: e.update(ece=float("inf")),
            lambda e: e.update(ece=-0.1),
            lambda e: e.update(ece=1.1),
            lambda e: e.update(ece=True),
            lambda e: e.update(n=3),
            lambda e: e.update(n=True),
            lambda e: e["bins"].pop(),
            lambda e: e["bins"][0].update(lo=True),
            lambda e: e["bins"][0].update(hi=0.2),
            lambda e: e["bins"][0].update(n=-1),
            lambda e: e["bins"][0].update(n=True),
            lambda e: e["bins"][0].update(n=2),
            lambda e: e["bins"][0].update(mean_confidence=float("nan")),
            lambda e: e["bins"][0].update(accuracy=1.1),
            lambda e: e["bins"][1].update(mean_confidence=0.0),
            lambda e: e["bins"][0].update(accuracy=None),
        ]
        for index, change in enumerate(mutations):
            with self.subTest(index=index):
                aggregate = self.aggregate()
                change(aggregate["ece"])
                with self.assertRaises(RuntimeError):
                    c.stage_metrics(c.STAGES[0], aggregate)

    def test_unallowlisted_global_ece_or_bin_fields_refused(self):
        for bin_field in (False, True):
            aggregate = self.aggregate()
            target = aggregate["ece"]["bins"][0] if bin_field else aggregate["ece"]
            target["synthetic_forbidden"] = "synthetic"
            with self.assertRaises(RuntimeError):
                c.stage_metrics(c.STAGES[0], aggregate)

    def test_ordinary_scalar_metrics_still_refuse_objects_and_nonfinite(self):
        for name in (
            "accuracy",
            "macro_accuracy",
            "coverage",
            "schema_validity",
            "schema_validity_strict",
            "operational_success",
            "brier_mean",
            "ordinal_mae",
            "price_per_1000_decisions_usd",
            "ledger_charged_usd",
        ):
            for value in ({"value": 0.1}, "0.1", float("nan"), float("inf"), True):
                with self.subTest(name=name, value=repr(value)):
                    aggregate = self.aggregate()
                    aggregate[name] = value
                    with self.assertRaises(RuntimeError):
                        c.stage_metrics(c.STAGES[0], aggregate)


if __name__ == "__main__":
    unittest.main()
