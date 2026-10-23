"""Stdlib-only campaign boundaries with synthetic receipts/callables, no model or dataset reads."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cpu_both_campaign", HERE / "campaign.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class CampaignTests(unittest.TestCase):
    def receipt(self):
        return {"profile": "decision3-publisher-edge-cpu-batch1-smoke-v1", "status": "publisher_cpu_single_request_complete_no_benchmark_claim", "plan_sha256": c.SMOKE_PLAN_SHA, "model_loads": 1, "forward_calls": 1, "benchmark_calls": 0, "cuda_initialized_final": False, "validation": {"contract_valid": True}, "forward_seconds": 1, "elapsed_seconds": 3, "native_provenance": {"model_name": "d3-edge", "batch_size": 1}}

    def test_source_readiness_or_wrong_smoke_cannot_enable_campaign(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic-smoke.json"
            for key, value in (("status", "dry_default_no_model_import_or_payload_read"), ("cuda_initialized_final", True), ("model_loads", 2), ("plan_sha256", "changed"), ("forward_seconds", 20.01), ("elapsed_seconds", 120.01)):
                raw = self.receipt()
                raw[key] = value
                path.write_text(json.dumps(raw))
                with self.assertRaises(RuntimeError):
                    c.check_smoke(path, c.sha(path))
            path.write_text(json.dumps(self.receipt()))
            self.assertTrue(c.check_smoke(path, c.sha(path))["not_a_benchmark_or_browser_score"])
            with self.assertRaises(RuntimeError):
                c.check_smoke(path, "wrong")
        self.assertFalse({"torch", "transformers", "numpy", "safetensors", "jevbench", "s1"}.intersection(sys.modules))

    def test_call_timer_is180_and_suite_samples_exclude_startup(self):
        writes, deadlines, calls = [], [], []
        class Timer:
            def __init__(self, seconds, output, stage):
                deadlines.append((seconds, stage))
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return False
        class Adapter:
            stage = "legacy-public231"
            def __call__(self, state, questions):
                calls.append((state, questions))
                return {"answers": {"q": {"type": "noul", "noul": .4}}}
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        predictor = c.TimedPredictor(Adapter(), NS(Deadline=Timer), temp.name)
        self.assertEqual(predictor.answers("synthetic", {"q": {"type": "noul"}}), {"q": {"type": "noul", "noul": .4}})
        predictor.stage = "jevbench-mini"
        predictor("synthetic2", {"q": {"type": "noul"}})
        self.assertEqual(deadlines, [(180, "call-00000"), (180, "call-00001")])
        self.assertEqual(predictor.completed, 2)
        self.assertEqual(set(predictor.summary()), {"legacy-public231", "jevbench-mini"})
        self.assertTrue(all(not row["startup_included"] and not row["browser_20_second_qualification"] for row in predictor.summary().values()))
        self.assertEqual(len(calls), 2)

    def test_latency_quantiles_use_actual_complete_call_values(self):
        values = [9, 1, 3, 2]
        result = c.latency_distribution(values)
        self.assertEqual(result["median_seconds"], 2.5)
        self.assertEqual(result["p95_seconds"], 9)
        self.assertEqual(result["max_seconds"], 9)
        self.assertEqual(values, [9, 1, 3, 2])
        self.assertIsNone(c.latency_distribution([])["median_seconds"])

    def test_nonfinite_raw_is_retained_unchanged_without_poisoned_temp(self):
        raw = {"model": "synthetic", "answers": {"q": {"noul": float("nan")}}}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "raw"
            runtime = c.PreservedRuntime(NS(system_one=lambda **_: raw), output)
            self.assertIs(runtime.system_one(state="synthetic", questions={}), raw)
            artifact = json.loads((output / "00000.json").read_text())
            self.assertEqual(artifact["encoding"], "invalid_native_response_diagnostic")
            self.assertIn("nan", artifact["raw_native_response_diagnostic"]["repr_prefix"])
            self.assertFalse(artifact["valid_response_claim"])
            self.assertFalse(any(p.name.startswith(".") for p in output.iterdir()))
            with self.assertRaises(ValueError):
                c.write_json(output / "bad.json", raw)
            self.assertFalse((output / "bad.json").exists())
            self.assertFalse(any(p.name.startswith(".") for p in output.iterdir()))

    def test_default_is_dry_and_does_not_need_real_prerequisite(self):
        result = subprocess.run([sys.executable, "-B", "-s", str(HERE / "campaign.py")], capture_output=True, text=True, check=True)
        raw = json.loads(result.stdout)
        self.assertEqual(raw["status"], "dry_default_no_model_or_benchmark_execution")
        self.assertEqual(raw["whole_campaign_seconds"], 6500)
        self.assertEqual(raw["per_call_seconds"], 180)
        self.assertEqual(raw["constructor_count"], 1)
        self.assertFalse(raw["browser_20_second_qualification"])


if __name__ == "__main__":
    unittest.main()
