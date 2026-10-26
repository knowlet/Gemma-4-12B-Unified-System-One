"""Bounded stdlib/fake tests; no Torch, model, job or benchmark input."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("batch4_actor_test", HERE / "actor.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
spec2 = importlib.util.spec_from_file_location("batch4_ast_test", HERE / "source_audit.py")
source = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(source)


def fake_runtime(key="edge"):
    model = a.MODELS[key]
    torch = NS(bfloat16="BF16", float32="FP32", cuda=NS(is_initialized=lambda: False))
    parameter = NS(device=NS(type="cpu"), dtype="BF16", requires_grad=False)
    config = NS(
        hidden_size=1024, num_hidden_layers=len(model["layers"]), layer_types=model["layers"]
    )
    backbone = NS(parameters=lambda: [parameter], training=False, config=NS(text_config=config))
    runtime = NS(
        model_name=model["name"],
        manifest={"identity": {"model_sha256": model["model_sha256"]}},
        device="cpu",
        batch_size=4,
        temperature=1.0,
        max_length=None,
        readout_dtype="float32",
        attention_mode="noncausal_full_attention",
        permutation_average=False,
        backbone=backbone,
        readout=NS(device=NS(type="cpu"), dtype="FP32", shape=(255, 1024)),
        fast=None,
        kernels={
            "torch_chunk_gated_delta_rule": "torch-reference",
            "causal_conv1d_fn": "torch-reference",
            "fla": "missing",
        },
    )
    return runtime, torch, model


class Batch4BoundaryTests(unittest.TestCase):
    def test_both_original_publisher_sources_have_native_one_constructor_batch_math(self):
        for key, model in a.MODELS.items():
            path = (
                a.ROOT
                / f"artifacts/jev-local-hf/hub/models--vllm-sr--d3-{key}/snapshots/{model['revision']}/d3_runtime.py"
            )
            result = source.verify_file(path)
            self.assertTrue(result["one_constructor"])
            self.assertTrue(result["left_padding_and_original_masks"])
            self.assertEqual(result["shipped_default_batch_unchanged"], 8)
            self.assertFalse(result["runtime_executed"])

    def test_AST_guard_rejects_accidental_default_grouping_or_readout_changes(self):
        model = a.MODELS["edge"]
        path = (
            a.ROOT
            / f"artifacts/jev-local-hf/hub/models--vllm-sr--d3-edge/snapshots/{model['revision']}/d3_runtime.py"
        )
        original = path.read_text()
        for changed in [
            original.replace("DEFAULT_BATCH_SIZE = 8", "DEFAULT_BATCH_SIZE = 4"),
            original.replace("range(0, len(keys), self.batch_size)", "range(0, len(keys), 1)"),
            original.replace("hidden.float() @ self.readout.T", "hidden.half() @ self.readout.T"),
        ]:
            with self.assertRaises(RuntimeError):
                source.verify_tree(ast.parse(changed))

    def test_both_frozen_model_plans_match_actual_profile_and_reject_batch1(self):
        helper = a.load_helper()
        for key in a.MODELS:
            path = HERE / (key + "-plan.json")
            with patch.object(a.sys, "executable", str(helper.RUNTIME_PYTHON)):
                plan = a.load_plan(helper, path, a.sha(path))
                self.assertEqual(plan["model_key"], key)
                self.assertEqual(plan["batch_size"], 4)
                with tempfile.TemporaryDirectory() as temp:
                    modified = copy.deepcopy(plan)
                    modified["batch_size"] = 1
                    bad = Path(temp) / "synthetic-plan.json"
                    bad.write_text(json.dumps(modified))
                    with self.assertRaisesRegex(RuntimeError, "batch/count/deadline"):
                        a.load_plan(helper, bad, a.sha(bad))

    def test_owner05_validator_rejects_old_units_or_caps(self):
        env = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"}
        identity = {
            "cgroup": "0::/owned/" + a.UNIT,
            "affinity": [16, 17, 18, 19],
            "environment": dict(env),
        }
        limits = {
            "memory.max": str(a.MEMORY_MAX),
            "memory.swap.max": "0",
            "cpu.max": "400000 100000",
        }
        a.validate_owner(identity, limits, env)
        for key, value in [
            ("cgroup", "0::/owned/jev-d3-llama-cpp-cpu4-3g-20261023-04.service"),
            ("affinity", [16]),
            ("environment", {"CUDA_VISIBLE_DEVICES": "0"}),
        ]:
            bad = copy.deepcopy(identity)
            bad[key] = value
            with self.assertRaises(RuntimeError):
                a.validate_owner(bad, limits, env)
        for key, value in [
            ("memory.max", "2147483648"),
            ("memory.swap.max", "1"),
            ("cpu.max", "100000 100000"),
        ]:
            bad = dict(limits)
            bad[key] = value
            with self.assertRaises(RuntimeError):
                a.validate_owner(identity, bad, env)

    def test_actual_architecture_batch_dtype_and_identity_required(self):
        for key in a.MODELS:
            runtime, torch, model = fake_runtime(key)
            a.validate_runtime(runtime, torch, model)
            for field, value in [
                ("batch_size", 1),
                ("temperature", 2.0),
                ("device", "cuda:0"),
                ("model_name", "wrong"),
                ("max_length", 512),
                ("fast", object()),
            ]:
                bad, torch, model = fake_runtime(key)
                setattr(bad, field, value)
                with self.assertRaises(RuntimeError):
                    a.validate_runtime(bad, torch, model)
            runtime, torch, model = fake_runtime(key)
            runtime.backbone.config.text_config.num_hidden_layers = 99
            with self.assertRaises(RuntimeError):
                a.validate_runtime(runtime, torch, model)
            runtime, torch, model = fake_runtime(key)
            runtime.backbone.parameters = lambda: [
                NS(device=NS(type="cpu"), dtype="FP32", requires_grad=False)
            ]
            with self.assertRaises(RuntimeError):
                a.validate_runtime(runtime, torch, model)

    def test_shape_hook_proves_four_rows_and_never_changes_inputs(self):
        _, torch, _ = fake_runtime()
        witness = a.ShapeWitness(torch, 123)
        hidden = NS(shape=(4, 123, 1024), dtype="BF16", device=NS(type="cpu"))
        before = copy.deepcopy(hidden.__dict__)
        self.assertIsNone(witness(None, (hidden,)))
        self.assertEqual(hidden.__dict__["shape"], before["shape"])
        self.assertEqual(hidden.__dict__["dtype"], before["dtype"])
        self.assertEqual(witness.events[0]["shape"], [4, 123, 1024])
        self.assertFalse(witness.events[0]["inputs_or_outputs_modified"])
        with self.assertRaises(RuntimeError):
            witness(None, (NS(shape=(1, 123, 1024), dtype="BF16", device=NS(type="cpu")),))
        self.assertEqual(len(witness.events), 1)

    def test_reusable_backend_keeps_nine_heads_and255_options_without_pruning(self):
        runtime, torch, model = fake_runtime()
        questions = {
            str(i): {"type": "choice", "criteria": {str(j): str(j) for j in range(255)}}
            for i in range(9)
        }
        groups = []
        evidence = []
        runtime.prepare = lambda **request: NS(
            keys=list(request["questions"]),
            sequences={k: [1, 2, 3] for k in request["questions"]},
            lengths={},
            errors={},
        )

        def native(**request):
            self.assertEqual(request, {"state": "synthetic", "questions": questions})
            keys = list(request["questions"])
            groups.extend(
                keys[i : i + runtime.batch_size] for i in range(0, len(keys), runtime.batch_size)
            )
            return {
                "model": "d3-edge",
                "answers": {
                    k: {
                        "type": "choice",
                        "choice": "0",
                        "probabilities": {str(j): (1.0 if j == 0 else 0.0) for j in range(255)},
                    }
                    for k in keys
                },
                "usage": {"input_tokens": 27, "output_tokens": 0},
            }

        runtime.system_one = native
        backend = a.NativeBatch4Backend(
            runtime, torch, model, evidence_sink=lambda kind, value: evidence.append((kind, value))
        )
        result = backend("synthetic", questions)
        self.assertEqual(list(result["answers"]), list(questions))
        self.assertEqual([len(g) for g in groups], [4, 4, 1])
        self.assertEqual(len(result["answers"]["0"]["probabilities"]), 255)
        self.assertEqual(evidence[-1][0], "raw_returned_before_validation")
        runtime.batch_size = 1
        with self.assertRaises(a.CPUProfileAbort):
            backend("synthetic", questions)
        self.assertFalse(issubclass(a.CPUProfileAbort, Exception))

    def test_context_refuses_whole_request_and_invalid_raw_retained_before_failure(self):
        runtime, torch, model = fake_runtime()
        runtime.prepare = lambda **_: NS(
            keys=["q"], sequences={"q": list(range(16385))}, lengths={}, errors={}
        )
        runtime.system_one = lambda **_: self.fail("over-limit request must never forward")
        backend = a.NativeBatch4Backend(runtime, torch, model)
        with self.assertRaisesRegex(RuntimeError, "nothing truncated"):
            backend("synthetic", {"q": {"type": "noul"}})
        runtime.prepare = lambda **_: NS(keys=["q"], sequences={"q": [1]}, lengths={}, errors={})
        raw = {
            "model": "d3-edge",
            "answers": {"q": {"noul": float("nan")}},
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }
        runtime.system_one = lambda **_: raw
        retained = []
        backend = a.NativeBatch4Backend(
            runtime, torch, model, evidence_sink=lambda kind, value: retained.append((kind, value))
        )
        with self.assertRaises(ValueError):
            backend("synthetic", {"q": {"type": "noul"}})
        self.assertIs(retained[-1][1], raw)
        self.assertTrue(math.isnan(raw["answers"]["q"]["noul"]))

    def test_all_four_native_choice_distributions_order_usage_and_entropy(self):
        model = a.MODELS["lite"]
        request = {
            "questions": {
                key: {"type": "choice", "criteria": {"first": "a", "second": "b"}}
                for key in a.HEADS
            }
        }
        lengths = {key: 103 + i for i, key in enumerate(a.HEADS)}
        raw = {
            "model": "d3-lite",
            "answers": {
                key: {
                    "type": "choice",
                    "choice": "first",
                    "probabilities": {"first": 0.5, "second": 0.5},
                    "confidence": 0.0,
                }
                for key in a.HEADS
            },
            "usage": {"input_tokens": sum(lengths.values()), "output_tokens": 0},
        }
        self.assertFalse(
            a.validate_smoke_answers(raw, request, lengths, model)[
                "semantic_expected_answer_or_quality_claim"
            ]
        )
        for change in [
            lambda value: value["answers"]["color"]["probabilities"].update(first=float("nan")),
            lambda value: value["answers"]["shape"].update(confidence=1.0),
            lambda value: value["usage"].update(input_tokens=0),
            lambda value: value.update(model="d3-edge"),
        ]:
            bad = copy.deepcopy(raw)
            change(bad)
            with self.assertRaises((RuntimeError, ValueError)):
                a.validate_smoke_answers(bad, request, lengths, model)

    def test_dry_default_requires_no_model_or_plan_and_exposes_two_call_scope(self):
        result = subprocess.run(
            [sys.executable, "-B", "-s", str(HERE / "actor.py")],
            check=True,
            capture_output=True,
            text=True,
        )
        value = json.loads(result.stdout)
        self.assertEqual(value["native_calls"], 2)
        self.assertEqual(value["batch_size"], 4)
        self.assertEqual(value["constructor_count"], 1)
        self.assertEqual(value["owner_unit"], a.UNIT)
        self.assertFalse(value["benchmark_or_batch1_score_inherited"])
        self.assertFalse(a.MODEL_MODULES.intersection(sys.modules))


if __name__ == "__main__":
    unittest.main()
