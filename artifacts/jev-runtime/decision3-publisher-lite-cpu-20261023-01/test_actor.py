"""Stdlib boundary tests. No Torch/model import or model payload access."""

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
spec = importlib.util.spec_from_file_location("publisher_cpu_actor", HERE / "actor.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class ActorBoundaryTests(unittest.TestCase):
    def identity(self):
        return {
            "cgroup": "0::/user.slice/user-1000.slice/user@1000.service/app.slice/" + a.UNIT,
            "affinity": [16, 17, 18, 19],
            "environment": dict(a.START_ENV),
        }

    def limits(self):
        return {"memory.max": str(a.MEMORY_MAX), "memory.swap.max": "0", "cpu.max": "400000 100000"}

    def test_owner_rejects_wrong_unit_affinity_environment_or_budget(self):
        a.validate_owner(self.identity(), self.limits())
        for key, value in (
            ("cgroup", "0::/other.service"),
            ("affinity", [16]),
            ("environment", {**a.START_ENV, "CUDA_VISIBLE_DEVICES": "0"}),
        ):
            identity = self.identity()
            identity[key] = value
            with self.assertRaises(RuntimeError):
                a.validate_owner(identity, self.limits())
        for key, value in (
            ("memory.max", "max"),
            ("memory.max", str(a.MEMORY_MAX + 1)),
            ("memory.swap.max", "1"),
            ("cpu.max", "100000 100000"),
        ):
            limits = self.limits()
            limits[key] = value
            with self.assertRaises(RuntimeError):
                a.validate_owner(self.identity(), limits)

    def test_kernel_closure_detects_bound_gpu_implementation(self):
        def cpu():
            return None

        self.assertEqual(a.binding_identity(cpu), "torch-reference")

        def gpu():
            return None

        gpu.__module__ = "fla.ops.gated_delta_rule"

        def outer(fn):
            def wrapped():
                return fn()

            return wrapped

        with self.assertRaisesRegex(RuntimeError, "GPU-only"):
            a.binding_identity(outer(gpu))
        gpu.__module__ = "kernels.community"
        with self.assertRaisesRegex(RuntimeError, "GPU-only"):
            a.binding_identity(outer(gpu))

    def runtime(self):
        torch = NS(bfloat16="BF16", float32="FP32", cuda=NS(is_initialized=lambda: False))
        parameter = NS(device=NS(type="cpu"), dtype="BF16")
        config = NS(
            hidden_size=1024,
            num_hidden_layers=24,
            layer_types=[
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
            * 6,
        )
        runtime = NS(
            model_name="d3-lite",
            manifest={
                "identity": {
                    "model_sha256": "fda5cfc7bc9063fbb7cd40b50fa9a0963d83a4790a3d6a11a356833fd1171992"
                }
            },
            device="cpu",
            batch_size=1,
            temperature=1.0,
            max_length=None,
            readout_dtype="float32",
            attention_mode="noncausal_full_attention",
            permutation_average=False,
            backbone=NS(parameters=lambda: [parameter], config=NS(text_config=config)),
            readout=NS(device=NS(type="cpu"), dtype="FP32", shape=(255, 1024)),
            fast=None,
            kernels={
                "torch_chunk_gated_delta_rule": "torch-reference",
                "causal_conv1d_fn": "torch-reference",
                "fla": "missing",
            },
        )
        return runtime, torch, parameter

    def test_cpu_policy_rejects_real_device_dtype_or_numerics_drift(self):
        runtime, torch, parameter = self.runtime()
        a.validate_runtime(runtime, torch)
        parameter.dtype = "FP32"
        with self.assertRaisesRegex(RuntimeError, "BF16"):
            a.validate_runtime(runtime, torch)
        parameter.dtype = "BF16"
        parameter.device.type = "cuda"
        with self.assertRaises(RuntimeError):
            a.validate_runtime(runtime, torch)
        for key, value in (
            ("batch_size", 8),
            ("temperature", 0.5),
            ("max_length", 512),
            ("permutation_average", True),
            ("fast", object()),
        ):
            runtime, torch, _ = self.runtime()
            setattr(runtime, key, value)
            with self.assertRaises(RuntimeError):
                a.validate_runtime(runtime, torch)

    def test_response_checks_schema_inventory_entropy_usage_but_not_quality(self):
        request = {"questions": {"q": {"criteria": {"first": "a", "second": "b"}}}}
        raw = {
            "model": "d3-lite",
            "answers": {
                "q": {
                    "type": "choice",
                    "choice": "second",
                    "probabilities": {"first": 0.5, "second": 0.5},
                    "confidence": 0,
                }
            },
            "usage": {"input_tokens": 12, "output_tokens": 0},
        }
        # First legal argmax wins ties, regardless of any semantic target.
        with self.assertRaises(RuntimeError):
            a.validate_answer(raw, request, 12)
        raw["answers"]["q"]["choice"] = "first"
        self.assertFalse(a.validate_answer(raw, request, 12)["quality_or_expected_answer_claim"])
        for change in (
            lambda x: x["answers"]["q"]["probabilities"].update(first=float("nan")),
            lambda x: x["answers"]["q"].update(confidence=1),
            lambda x: x["usage"].update(output_tokens=1),
            lambda x: x.update(model="fake"),
        ):
            bad = copy.deepcopy(raw)
            change(bad)
            with self.assertRaises(RuntimeError):
                a.validate_answer(bad, request, 12)

    def test_deadline_exits_only_synthetic_owned_fixture_and_leaves_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            code = "import importlib.util, pathlib, time; p=pathlib.Path(__import__('sys').argv[1]); s=importlib.util.spec_from_file_location('deadline_test',p); a=importlib.util.module_from_spec(s); s.loader.exec_module(a); d=a.Deadline(.02,pathlib.Path(__import__('sys').argv[2]),'synthetic'); d.__enter__(); time.sleep(5)"
            result = subprocess.run(
                [sys.executable, "-B", "-s", "-c", code, str(HERE / "actor.py"), temp], timeout=2
            )
            self.assertEqual(result.returncode, 124)
            receipt = json.loads((Path(temp) / "synthetic-timeout.json").read_text())
            self.assertEqual(receipt["stage"], "synthetic")
            self.assertEqual(receipt["exit_code"], 124)

    def test_edge_identity_or_architecture_cannot_qualify_as_lite(self):
        runtime, torch, _ = self.runtime()
        runtime.model_name = "d3-edge"
        with self.assertRaisesRegex(RuntimeError, "pinned Lite"):
            a.validate_runtime(runtime, torch)
        runtime, torch, _ = self.runtime()
        runtime.backbone.config.text_config.num_hidden_layers = 12
        with self.assertRaisesRegex(RuntimeError, "Lite architecture"):
            a.validate_runtime(runtime, torch)
        limits = self.limits()
        limits["memory.max"] = "2147483648"
        with self.assertRaisesRegex(RuntimeError, "memory/swap"):
            a.validate_owner(self.identity(), limits)

    def test_nonfinite_raw_retention_and_finalization_preserve_primary(self):
        raw = {"answers": {"q": {"probabilities": {"first": float("nan")}}}}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            self.assertFalse(a.retain_raw_response(output, raw))
            receipt = json.loads((output / "raw-response.json").read_text())
            self.assertFalse(receipt["valid_response_claim"])
            self.assertIn("nan", receipt["raw_response_diagnostic"]["repr_prefix"])
            primary = RuntimeError("actual primary")
            a.finalize_state(output, {"raw": raw}, primary)
            final = json.loads((output / "final-evidence-error.json").read_text())
            self.assertEqual(final["primary_error"]["message"], "actual primary")
            self.assertFalse(any(p.name.startswith(".") for p in output.iterdir()))
            self.assertTrue(primary.__notes__)

    def test_late_response_validation_is_refused_even_if_watchdog_delayed(self):
        with patch.object(a.time, "monotonic", return_value=20.001):
            with self.assertRaisesRegex(RuntimeError, "after its fixed deadline"):
                a.ensure_elapsed(0, 20, "validation")
        with patch.object(a.time, "monotonic", return_value=19.999):
            self.assertEqual(a.ensure_elapsed(0, 20, "validation"), 19.999)

    def test_dry_default_never_imports_model_or_needs_owner(self):
        result = subprocess.run(
            [sys.executable, "-B", "-s", str(HERE / "actor.py")],
            check=True,
            text=True,
            capture_output=True,
        )
        body = json.loads(result.stdout)
        self.assertFalse(body["model_execution"])
        self.assertEqual(body["request_count"], 1)
        self.assertEqual(body["status"], "dry_default_no_model_import_or_payload_read")
        self.assertFalse(a.MODEL_MODULES.intersection(sys.modules))


if __name__ == "__main__":
    unittest.main()
