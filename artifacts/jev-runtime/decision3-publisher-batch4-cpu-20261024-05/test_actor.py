"""UNEXECUTED CPU5 stdlib/fake tests, retained from the reviewed native04 tests.

Only small source/plan/freeze metadata is read. No model payload, Torch, live
unit, observer or benchmark request is used. Root must finalize pins and build
the metadata-only edge plan before deliberately running this suite.
"""

from __future__ import annotations

import ast
import copy
import hashlib
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
PARENT = HERE.parent / "decision3-publisher-batch4-cpu-20261024-04"
OWNER = HERE.parent / "llama-cpp-owner-20261024-09"
PARENT_ACTOR_SHA = "7e1028e63d347c906358e5aa7a21d989611a9d43096d07ca3f585d4b6a6c5916"
PARENT_HELPER_SHA = "e34f254a40dd2346cf94bbc140a92d999a1f696a719b19acfe038d2dc79115c1"
PARENT_PLAN_SHA = "9a1a48e46663240dd05f45a4558f48cfdfe0c4955ee2c2d7a74565da5015950d"
AUDIT_SHA = "ae299d87cbf7338bb9ff29055efc335e96a44b73a29370e08428aad2fb2b68d3"
OWNER_PROFILE = "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"
OWNER_UNIT = "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service"
OWNER_CONTROLLER_SHA = "81846efe2ee0357da07283c1a77a606283a920cbddef668c8f62def291057dee"
OWNER_FREEZE_SHA = "52ac647a80ca1ef0c05acc45bf90c990f3b753b66df216c87cfb2bdf495020ef"
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
    def test_both_frozen_model_plans_match_actual_profile_and_reject_batch1(self):
        helper = a.load_helper()
        for key in ["edge"]:
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

    def test_owner09_validator_rejects_old_units_or_caps(self):
        env = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"}
        identity = {
            "cgroup": "0::/owned/" + a.UNIT,
            "affinity": [15, 16, 17, 18, 19],
            "environment": dict(env),
        }
        limits = {
            "memory.max": str(a.MEMORY_MAX),
            "memory.swap.max": "0",
            "cpu.max": "500000 100000",
        }
        a.validate_owner(identity, limits, env)
        for key, value in [
            ("cgroup", "0::/owned/jev-d3-llama-cpp-cpu4-3g-20261023-04.service"),
            ("cgroup", "0::/owned/jev-d3-llama-cpp-cpu4-2g-long-20261023-06.service"),
            ("cgroup", "0::/owned/jev-d3-llama-cpp-cpu4-2g-coexist-20261024-07.service"),
            ("affinity", [16]),
            ("environment", {"CUDA_VISIBLE_DEVICES": "0"}),
        ]:
            bad = copy.deepcopy(identity)
            bad[key] = value
            with self.assertRaises(RuntimeError):
                a.validate_owner(bad, limits, env)
        for key, value in [
            ("memory.max", "2147483648"),
            ("memory.max", "3221225472"),
            ("memory.swap.max", "1"),
            ("cpu.max", "100000 100000"),
        ]:
            bad = dict(limits)
            bad[key] = value
            with self.assertRaises(RuntimeError):
                a.validate_owner(identity, bad, env)

    def test_actual_architecture_batch_dtype_and_identity_required(self):
        for key in ["edge"]:
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
        model = a.MODELS["edge"]
        request = {
            "questions": {
                key: {"type": "choice", "criteria": {"first": "a", "second": "b"}}
                for key in a.HEADS
            }
        }
        lengths = {key: 103 + i for i, key in enumerate(a.HEADS)}
        raw = {
            "model": "d3-edge",
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
            lambda value: value.update(model="d3-lite"),
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


class ResourceClosureTests(unittest.TestCase):
    def assert_only_loader_thread_literals_changed(self, original, current):
        expected = copy.deepcopy(original)
        setters = [
            node
            for node in ast.walk(expected)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "set_num_threads"
        ]
        comparisons = [
            node
            for node in ast.walk(expected)
            if isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Call)
            and isinstance(node.left.func, ast.Attribute)
            and isinstance(node.left.func.value, ast.Name)
            and node.left.func.value.id == "torch"
            and node.left.func.attr == "get_num_threads"
        ]
        self.assertEqual(len(setters), 1)
        self.assertEqual(len(comparisons), 1)
        setter, comparison = setters[0], comparisons[0]
        self.assertEqual(len(setter.args), 1)
        self.assertEqual(setter.keywords, [])
        self.assertEqual(ast.literal_eval(setter.args[0]), 4)
        self.assertEqual(len(comparison.ops), 1)
        self.assertIsInstance(comparison.ops[0], ast.Eq)
        self.assertEqual(len(comparison.comparators), 1)
        self.assertEqual(ast.literal_eval(comparison.comparators[0]), 4)
        setter.args[0] = ast.Constant(value=5)
        comparison.comparators[0] = ast.Constant(value=5)
        self.assertEqual(
            ast.dump(expected, include_attributes=False),
            ast.dump(current, include_attributes=False),
        )

    def test_native02_NN_auth_and_postcleanup_timing_AST_unchanged(self):
        original = (
            a.ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261023-02/actor.py"
        )
        old = {
            n.name: n
            for n in ast.parse(original.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        new = {
            n.name: n
            for n in ast.parse((HERE / "actor.py").read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        self.assertEqual(set(old), set(new))
        allowed_resource_bindings = {"validate_owner", "load_plan", "execute", "main"}
        for name in set(old) - allowed_resource_bindings:
            if name == "load_runtime":
                self.assert_only_loader_thread_literals_changed(old[name], new[name])
                continue
            self.assertEqual(
                ast.dump(old[name], include_attributes=False),
                ast.dump(new[name], include_attributes=False),
                name,
            )
        # Execute differs only by the truthful selected-owner policy metadata.
        current = copy.deepcopy(new["execute"])
        for node in ast.walk(current):
            if isinstance(node, ast.Dict):
                indices = [
                    i
                    for i, key in enumerate(node.keys)
                    if isinstance(key, ast.Constant)
                    and key.value == "selected_owner_host_observation_policy"
                ]
                for i in reversed(indices):
                    del node.keys[i]
                    del node.values[i]
        self.assertEqual(
            ast.dump(old["execute"], include_attributes=False),
            ast.dump(current, include_attributes=False),
        )

    def test_native02_helper_functions_all_byte_equivalent_AST(self):
        original = (
            a.ROOT
            / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261023-02/resource_helper.py"
        )
        old = {
            n.name: n
            for n in ast.parse(original.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        new = {
            n.name: n
            for n in ast.parse((HERE / "resource_helper.py").read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        self.assertEqual(set(old), set(new))
        for name in old:
            expected = copy.deepcopy(old[name])
            if name == "host_gate":
                diagnostics = [
                    node
                    for node in ast.walk(expected)
                    if isinstance(node, ast.Constant)
                    and node.value == "unchanged reserve plus 2GiB startup floor refused"
                ]
                self.assertEqual(len(diagnostics), 1)
                diagnostics[0].value = "unchanged reserve plus 4GiB startup floor refused"
            self.assertEqual(
                ast.dump(expected, include_attributes=False),
                ast.dump(new[name], include_attributes=False),
                name,
            )

    def test_original_numerical_and_budget_constants_unchanged(self):
        original = (
            a.ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261023-02/actor.py"
        )

        def selected(path):
            return {
                target.id: ast.dump(node.value, include_attributes=False)
                for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Name)
                and target.id
                in {
                    "RESERVE",
                    "MEMORY_MAX",
                    "WALL_SECONDS",
                    "CALL_SECONDS",
                    "BATCH_SIZE",
                    "CALLS",
                    "HEADS",
                    "MODEL_MODULES",
                    "MODELS",
                }
            }

        old, new = selected(original), selected(HERE / "actor.py")
        self.assertEqual(old.pop("MEMORY_MAX"), ast.dump(ast.Constant(value=2147483648)))
        self.assertEqual(new.pop("MEMORY_MAX"), ast.dump(ast.Constant(value=4294967296)))
        self.assertEqual(old, new)

    def test_coexist_policy_must_join_new_plan_and_helper(self):
        helper = a.load_helper()
        path = HERE / "edge-plan.json"
        original = json.loads(path.read_text())
        with patch.object(a.sys, "executable", str(helper.RUNTIME_PYTHON)):
            a.load_plan(helper, path, a.sha(path))
            for policy in (None, "comfy-idle-only", "unknown", "comfy-observed-cpu-coexist-v2"):
                changed = copy.deepcopy(original)
                changed["host_observation_policy"] = policy
                with tempfile.TemporaryDirectory() as temp:
                    bad = Path(temp) / "policy.json"
                    bad.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(RuntimeError, "observation policy"):
                        a.load_plan(helper, bad, a.sha(bad))
            with patch.object(helper, "HOST_OBSERVATION_POLICY", "comfy-idle-only"):
                with self.assertRaisesRegex(RuntimeError, "observation policy"):
                    a.load_plan(helper, path, a.sha(path))

    def test_publisher_constructor_backend_and_validation_AST_unchanged(self):
        original = (
            a.ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261023-01/actor.py"
        )
        old = {
            n.name: n
            for n in ast.parse(original.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        new = {
            n.name: n
            for n in ast.parse((HERE / "actor.py").read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        for name in (
            "load_runtime",
            "validate_runtime",
            "NativeBatch4Backend",
            "ShapeWitness",
            "validate_smoke_answers",
            "authenticate",
        ):
            if name == "load_runtime":
                self.assert_only_loader_thread_literals_changed(old[name], new[name])
                continue
            self.assertEqual(
                ast.dump(old[name], include_attributes=False),
                ast.dump(new[name], include_attributes=False),
                name,
            )
        self.assertEqual(a.sha(HERE / "source_audit.py"), a.SOURCE_AUDIT_SHA)

    def test_extracted_helper_preserves_all_used_function_AST(self):
        helper = a.load_helper()
        original = (
            a.ROOT / "artifacts/jev-runtime/decision3-publisher-lite-cpu-20261023-01/actor.py"
        )
        old = {
            n.name: n
            for n in ast.parse(original.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        new = {
            n.name: n
            for n in ast.parse((HERE / "resource_helper.py").read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        self.assertNotIn("load_publisher_cpu", new)
        self.assertNotIn("load_plan", new)
        for name in new:
            if name != "host_gate":
                self.assertEqual(
                    ast.dump(old[name], include_attributes=False),
                    ast.dump(new[name], include_attributes=False),
                    name,
                )
        self.assertEqual(helper.MEMORY_MAX, 4294967296)
        self.assertEqual(helper.RESERVE, 26119172916)

    def test_resource_hostgate_full_four_gib_and_no_credit(self):
        helper = a.load_helper()
        total = 130595864576

        def memory(available):
            return "MemTotal: %d kB\nMemAvailable: %d kB\n" % (total // 1024, available // 1024)

        floor = 30414140212
        with patch.object(helper.Path, "read_text", return_value=memory(floor + 1024)):
            result = helper.host_gate()
        self.assertEqual(result["required_available_bytes"], floor)
        self.assertEqual(result["swap_credit_bytes"], 0)
        self.assertEqual(result["cache_credit_bytes"], 0)
        for available in (floor - 1, helper.RESERVE + 1024**3, helper.RESERVE):
            with patch.object(helper.Path, "read_text", return_value=memory(available)):
                with self.assertRaisesRegex(RuntimeError, "plus 4GiB"):
                    helper.host_gate()
        old = ast.parse(
            (
                a.ROOT / "artifacts/jev-runtime/decision3-publisher-lite-cpu-20261023-01/actor.py"
            ).read_text()
        )
        fn = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "host_gate")
        namespace = {
            "Path": helper.Path,
            "RESERVE": helper.RESERVE,
            "MEMORY_MAX": 3 * 1024**3,
            "require": helper.require,
        }
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), "frozen-old-hostgate-only", "exec"),
            namespace,
        )
        with patch.object(helper.Path, "read_text", return_value=memory(floor + 1024)):
            old_gate = namespace["host_gate"]()
        self.assertEqual(old_gate["required_available_bytes"], helper.RESERVE + 3 * 1024**3)
        self.assertLess(old_gate["required_available_bytes"], floor)

    def test_selected_plan_refuses_mixed_owner_cap_floor_or_shortened_budget(self):
        helper = a.load_helper()
        plan = json.loads((HERE / "edge-plan.json").read_text())
        mutations = [
            lambda p: p.update(memory_max_bytes=3221225472),
            lambda p: p.update(startup_available_bytes=a.RESERVE),
            lambda p: p.update(startup_available_bytes=a.RESERVE + a.MEMORY_MAX - 1),
            lambda p: p.update(swap_max_bytes=1),
            lambda p: p.update(owner_runtime_seconds=7200),
            lambda p: p.update(owner_profile="llama-cpp-cpu4-3g-long-command-owned-v1"),
            lambda p: p.update(owner_unit="jev-d3-llama-cpp-cpu4-3g-long-20261023-05.service"),
            lambda p: p.update(owner_profile="llama-cpp-cpu4-2g-long-command-owned-v1"),
            lambda p: p.update(owner_unit="jev-d3-llama-cpp-cpu4-2g-long-20261023-06.service"),
            lambda p: p.update(owner_profile="llama-cpp-cpu4-2g-coexist-command-owned-v1"),
            lambda p: p.update(owner_unit="jev-d3-llama-cpp-cpu4-2g-coexist-20261024-07.service"),
            lambda p: p["owner_sources"][0].update(sha256="0" * 64),
            lambda p: p.update(wall_seconds=119),
            lambda p: p.update(call_seconds=19),
            lambda p: p.update(model_key="lite", model=a.MODELS["lite"]),
        ]
        for change in mutations:
            with tempfile.TemporaryDirectory() as temp:
                bad = copy.deepcopy(plan)
                change(bad)
                path = Path(temp) / "fake-plan.json"
                path.write_text(json.dumps(bad))
                with patch.object(a.sys, "executable", str(helper.RUNTIME_PYTHON)):
                    with self.assertRaises(RuntimeError):
                        a.load_plan(helper, path, a.sha(path))

    def test_owner_caps_adjacent_and_old_unit_refuse(self):
        env = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"}
        identity = {
            "cgroup": "0::/owned/" + a.UNIT,
            "affinity": [15, 16, 17, 18, 19],
            "environment": env,
        }
        for cap in (a.MEMORY_MAX - 1, a.MEMORY_MAX + 1, 3221225472):
            with self.assertRaises(RuntimeError):
                a.validate_owner(
                    identity,
                    {"memory.max": str(cap), "memory.swap.max": "0", "cpu.max": "500000 100000"},
                    env,
                )
        self.assertFalse(a.MODEL_MODULES.intersection(sys.modules))


class TimingFinalizationTests(unittest.TestCase):
    def fake_execute(
        self,
        module=a,
        *,
        advances=None,
        cleanup=None,
        primary=None,
        preflight=0,
        fail_failure_write=False,
        final_success_write_advance=0,
    ):
        from contextlib import ExitStack

        helper = a.load_helper()
        advances, cleanup = advances or {}, cleanup or {}
        clock = NS(now=0.0)
        clock.read = lambda: clock.now
        timers = []

        class FakeDeadline:
            def __init__(self, seconds, output, stage):
                self.stage = stage
                timers.append((stage, seconds))

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                clock.now += advances.get(self.stage, 0)
                if self.stage in cleanup:
                    raise cleanup[self.stage]

        witness = NS(events=[])
        handle = NS(remove=lambda: None)
        layer = NS(register_forward_pre_hook=lambda _fn: handle)
        runtime = NS(backbone=NS(language_model=NS(layers=[layer])))

        def native(**_request):
            if primary is not None:
                raise primary
            witness.events.append({"shape": [4, 4, 1024]})
            return {"fixture": "timing-only-not-NN"}

        runtime.system_one = native
        torch = NS(cuda=NS(is_initialized=lambda: False))
        request = {
            "state": "fake",
            "questions": {k: {"type": "choice", "criteria": {"x": "x"}} for k in module.HEADS},
        }
        plan = {
            "profile": "fake-timing",
            "model_key": "edge",
            "model": module.MODELS["edge"],
            "request": request,
            "actor_sha256": module.sha(module.__file__),
        }

        def selected(*_args):
            clock.now += preflight
            return plan

        def loaded(_helper, _plan, _output, state):
            state["model_loads"] = 1
            return runtime, torch

        class Backend:
            def prepare(self, **_request):
                lengths = {k: i + 1 for i, k in enumerate(module.HEADS)}
                return NS(sequences={k: list(range(n)) for k, n in lengths.items()}), lengths

        real_write = helper.write_json
        successful_state_writes = 0

        def write(path, state):
            nonlocal successful_state_writes
            if fail_failure_write and state.get("status") == "failed_no_batch4_qualification":
                raise OSError("secondary failure evidence")
            real_write(path, state)
            if (
                state.get("status")
                == "native_batch4_two_call_synthetic_probe_complete_no_benchmark_claim"
            ):
                successful_state_writes += 1
                if successful_state_writes == 2:
                    clock.now += final_success_write_advance

        with (
            tempfile.TemporaryDirectory(dir=HERE, prefix=".timing-fixture-") as temp,
            ExitStack() as stack,
        ):
            output = Path(temp) / "probe"
            for target, name, value in [
                (module, "load_helper", lambda: helper),
                (module, "load_plan", selected),
                (module, "validate_owner", lambda *_args: None),
                (module, "own_counters", lambda *_args: {}),
                (module, "load_runtime", loaded),
                (module, "authenticate", lambda *_args: None),
                (module, "validate_runtime", lambda *_args: None),
                (module, "validate_smoke_answers", lambda *_args: {"fixture_valid": True}),
                (module, "ShapeWitness", lambda *_args: witness),
                (module, "NativeBatch4Backend", lambda *_args: Backend()),
                (helper, "process_identity", lambda: {}),
                (helper, "owned_limits", lambda *_args: {}),
                (helper, "host_gate", lambda: {"fixture": "no host observation"}),
                (helper, "Deadline", FakeDeadline),
                (helper, "write_json", write),
                (module.time, "monotonic", clock.read),
            ]:
                stack.enter_context(patch.object(target, name, value))
            try:
                result = module.execute(Path("unused-fake-plan"), "fake-sha", output)
            except BaseException as exc:
                state = (
                    json.loads((output / "state.json").read_text())
                    if (output / "state.json").exists()
                    else None
                )
                return None, exc, state, timers, clock.now
            return result, None, json.loads((output / "state.json").read_text()), timers, clock.now

    def baseline(self):
        spec = importlib.util.spec_from_file_location(
            "_frozen_native01_timing_only",
            a.ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261023-01/actor.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_late_call_after_context_cleanup_demotes_candidate(self):
        result, error, state, _, _ = self.fake_execute(advances={"call-0": 21})
        self.assertIsNone(result)
        self.assertIsInstance(error, RuntimeError)
        self.assertFalse(state["completed"])
        self.assertFalse(state["calls"][0]["completed"])
        self.assertEqual(state["calls"][0]["status"], "failed_no_call_qualification")
        old, error, _, _, _ = self.fake_execute(self.baseline(), advances={"call-0": 21})
        self.assertIsNone(error)
        self.assertIn("probe_complete", old["status"])

    def test_late_whole_after_context_cleanup_demotes_candidate(self):
        result, error, state, _, _ = self.fake_execute(advances={"actor": 121})
        self.assertIsNone(result)
        self.assertIsInstance(error, RuntimeError)
        self.assertFalse(state["completed"])
        self.assertEqual(state["status"], "failed_no_batch4_qualification")
        old, error, _, _, _ = self.fake_execute(self.baseline(), advances={"actor": 121})
        self.assertIsNone(error)
        self.assertIn("probe_complete", old["status"])

    def test_final_finally_state_write_expiry_demotes_success(self):
        result, error, state, _, _ = self.fake_execute(final_success_write_advance=121)
        self.assertIsNone(result)
        self.assertIsInstance(error, RuntimeError)
        self.assertFalse(state["completed"])
        self.assertEqual(state["status"], "failed_no_batch4_qualification")
        old, error, _, _, _ = self.fake_execute(self.baseline(), final_success_write_advance=121)
        self.assertIsNone(error)
        self.assertIn("probe_complete", old["status"])

    def test_call_and_whole_cleanup_raise_cannot_return_success(self):
        for stage in ("call-0", "actor"):
            failure = OSError("synthetic cleanup")
            result, error, state, _, _ = self.fake_execute(cleanup={stage: failure})
            self.assertIsNone(result)
            self.assertIs(error, failure)
            self.assertFalse(state["completed"])

    def test_original_primary_preserved_across_cleanup_and_failure_writer(self):
        failure = KeyboardInterrupt("synthetic original native primary")
        result, error, _, _, _ = self.fake_execute(
            primary=failure,
            cleanup={"call-0": OSError("call cleanup"), "actor": OSError("whole cleanup")},
            fail_failure_write=True,
        )
        self.assertIsNone(result)
        self.assertIs(error, failure)
        self.assertTrue(any("cleanup" in note for note in failure.__notes__))
        self.assertTrue(any("state write" in note for note in failure.__notes__))
        baseline_primary = KeyboardInterrupt("baseline original")
        _, old_error, _, _, _ = self.fake_execute(
            self.baseline(),
            primary=baseline_primary,
            cleanup={"call-0": OSError("baseline cleanup")},
        )
        self.assertIsNot(old_error, baseline_primary)

    def test_preflight_consumes_original_whole120_without_reset(self):
        result, error, _, timers, _ = self.fake_execute(preflight=119)
        self.assertIsNone(error)
        self.assertTrue(result["completed"])
        self.assertEqual(next(seconds for stage, seconds in timers if stage == "actor"), 1)
        result, error, state, timers, _ = self.fake_execute(preflight=121)
        self.assertIsNone(result)
        self.assertIsInstance(error, RuntimeError)
        self.assertIsNone(state)
        self.assertEqual(timers, [])

    def test_success_durable_call_latency_includes_cleanup(self):
        result, error, _, _, _ = self.fake_execute(advances={"call-0": 0.25, "call-1": 0.5})
        self.assertIsNone(error)
        self.assertTrue(result["completed"])
        self.assertEqual([r["durable_validation_seconds"] for r in result["calls"]], [0.25, 0.5])


def small_bytes(path, expected=None):
    """Only the fixed small source/metadata paths supplied by these tests."""
    path = Path(path)
    if path.suffix not in {".py", ".json"} or not 0 < path.stat().st_size < 100_000:
        raise ValueError("small Python/JSON source or metadata required")
    raw = path.read_bytes()
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("independently fixed source/metadata SHA differs")
    return raw


def source_record(path):
    raw = small_bytes(path)
    info = path.stat()
    return {
        "path": str(path),
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "stat": {key: getattr(info, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns")},
    }


class CPU5PreparationTests(unittest.TestCase):
    def selected_sources(self):
        controller = source_record(OWNER / "controller.py")
        freeze_raw = small_bytes(OWNER / "freeze.json", OWNER_FREEZE_SHA)
        freeze = json.loads(freeze_raw)
        self.assertEqual(controller["sha256"], OWNER_CONTROLLER_SHA)
        self.assertEqual(controller["size"], 41886)
        self.assertEqual(freeze["schema_id"], "cpu5-fast-coexist-owner-source-freeze-v1")
        self.assertEqual(freeze["status"], "SOURCE_CHECKS_PASS_NO_ACTUAL")
        self.assertEqual((freeze["profile"], freeze["unit"]), (OWNER_PROFILE, OWNER_UNIT))
        matches = [
            row
            for row in freeze["files"]
            if row["path"] == "artifacts/jev-runtime/llama-cpp-owner-20261024-09/controller.py"
        ]
        self.assertEqual(len(matches), 1)
        self.assertEqual(
            (matches[0]["sha256"], matches[0]["size_bytes"]),
            (OWNER_CONTROLLER_SHA, controller["size"]),
        )
        helper = source_record(HERE / "resource_helper.py")
        self.assertEqual(a.HELPER_SHA, helper["sha256"], "Finalize the real helper pin first")
        self.assertEqual(
            a.OWNER_CONTROLLER_SHA,
            OWNER_CONTROLLER_SHA,
            "Finalize the independently frozen owner09 pin first",
        )
        return controller, source_record(OWNER / "freeze.json"), helper

    def replace_expression(self, tree, before, after, count=1):
        old = ast.dump(ast.parse(before, mode="eval").body, include_attributes=False)
        replacement = ast.parse(after, mode="eval").body

        class Exact(ast.NodeTransformer):
            def __init__(self):
                self.changed = 0

            def visit(self, node):
                if ast.dump(node, include_attributes=False) == old:
                    self.changed += 1
                    return ast.copy_location(copy.deepcopy(replacement), node)
                return super().visit(node)

        edit = Exact()
        result = edit.visit(tree)
        self.assertEqual(edit.changed, count, before)
        return result

    def test_full_frozen_parent_AST_allows_only_fixed_cpu5_delta_and_real_final_pins(self):
        controller, _, helper = self.selected_sources()
        expected = ast.parse(small_bytes(PARENT / "actor.py", PARENT_ACTOR_SHA).decode())
        assignments = {
            target.id: node
            for node in expected.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        bindings = {
            "HELPER_SHA": (PARENT_HELPER_SHA, helper["sha256"]),
            "OWNER_CONTROLLER_SHA": (
                "ba91e1cbf66f9d1a0f4e16e610a1fe8952c2efde81e2a28f39a111a98ca10c9b",
                controller["sha256"],
            ),
            "UNIT": ("jev-d3-llama-cpp-cpu16-4g-coexist-20261024-08.service", OWNER_UNIT),
            "OWNER_PROFILE": ("llama-cpp-cpu16-4g-coexist-command-owned-v1", OWNER_PROFILE),
        }
        for name, (before, after) in bindings.items():
            self.assertEqual(ast.literal_eval(assignments[name].value), before, name)
            assignments[name].value = ast.Constant(value=after)
        changes = (
            (repr("selected owner08 cgroup required"), repr("selected owner09 cgroup required"), 1),
            ("[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]", "[15,16,17,18,19]", 2),
            ("quota == 16 * period", "quota == 5 * period", 1),
            (
                repr("-publisher-cpu16-batch4-4g-coexist-synthetic-20261024-v1"),
                repr("-publisher-cpu5-fast-same-cluster-batch4-4g-coexist-synthetic-20261024-v1"),
                1,
            ),
            (
                repr("selected owner08 CPU-coexist observation policy differs"),
                repr("selected owner09 CPU-coexist observation policy differs"),
                1,
            ),
            ('plan.get("cpu_quota_percent") == 1600', 'plan.get("cpu_quota_percent") == 500', 1),
            (repr("owner08 profile/budget differs"), repr("owner09 profile/budget differs"), 1),
            (
                repr("artifacts/jev-runtime/llama-cpp-owner-20261024-08/controller.py"),
                repr("artifacts/jev-runtime/llama-cpp-owner-20261024-09/controller.py"),
                1,
            ),
            (
                repr("selected owner08 controller identity differs"),
                repr("selected owner09 controller identity differs"),
                1,
            ),
            ('plan.get("threads") == 16', 'plan.get("threads") == 5', 1),
            ("torch.set_num_threads(16)", "torch.set_num_threads(5)", 1),
            ("torch.get_num_threads() == 16", "torch.get_num_threads() == 5", 1),
        )
        for before, after, count in changes:
            expected = self.replace_expression(expected, before, after, count)
        actual = ast.parse(small_bytes(HERE / "actor.py").decode())
        self.assertEqual(
            ast.dump(actual, include_attributes=False), ast.dump(expected, include_attributes=False)
        )

    def test_helper_whole_AST_only_changes_owner_docstring_and_four_thread_values(self):
        expected = ast.parse(small_bytes(PARENT / "resource_helper.py", PARENT_HELPER_SHA).decode())
        doc = expected.body[0].value
        self.assertEqual(doc.value.count("Distinct owner08 CPU-coexist policy."), 1)
        doc.value = doc.value.replace(
            "Distinct owner08 CPU-coexist policy.", "Distinct owner09 CPU-coexist policy."
        )
        env = next(
            node.value
            for node in expected.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "CPU_ENV" for target in node.targets
            )
        )
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            indices = [i for i, key in enumerate(env.keys) if ast.literal_eval(key) == name]
            self.assertEqual(len(indices), 1)
            index = indices[0]
            self.assertEqual(ast.literal_eval(env.values[index]), "16")
            env.values[index] = ast.Constant(value="5")
        self.assertEqual(
            ast.dump(expected, include_attributes=False),
            ast.dump(
                ast.parse(small_bytes(HERE / "resource_helper.py").decode()),
                include_attributes=False,
            ),
        )
        self.assertEqual(
            small_bytes(HERE / "source_audit.py", AUDIT_SHA),
            small_bytes(PARENT / "source_audit.py", AUDIT_SHA),
        )

    def test_final_plan_pins_authenticate_real_small_sources_and_owner_freeze(self):
        controller, owner_freeze, helper = self.selected_sources()
        plan = json.loads(small_bytes(HERE / "edge-plan.json"))
        expected = [
            controller,
            owner_freeze,
            source_record(HERE / "actor.py"),
            helper,
            source_record(HERE / "source_audit.py"),
        ]
        self.assertEqual(plan["owner_sources"], expected)
        self.assertEqual(plan["actor_sha256"], expected[2]["sha256"])
        self.assertEqual(plan["helper_sha256"], helper["sha256"])
        self.assertEqual(plan["source_audit_sha256"], AUDIT_SHA)
        self.assertEqual(
            (plan["profile"], plan["owner_profile"], plan["owner_unit"]),
            (
                "decision3-edge-publisher-cpu5-fast-same-cluster-batch4-4g-coexist-synthetic-20261024-v1",
                OWNER_PROFILE,
                OWNER_UNIT,
            ),
        )

    def builder(self):
        spec = importlib.util.spec_from_file_location(
            "cpu5_metadata_builder_test", HERE / "build_plan.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_builder_preserves_every_nonported_metadata_field_without_payload_reads_or_stats(self):
        base = json.loads(small_bytes(PARENT / "edge-plan.json", PARENT_PLAN_SHA))
        protected = {
            str(Path(row["path"]).absolute())
            for row in [base["manifest"], *base["model_files"], *base["runtime_files"]]
        }
        protected.add(str(Path(base["snapshot"]).absolute()))
        original_open, original_stat, original_lstat = Path.open, Path.stat, Path.lstat

        def guard(operation):
            def checked(path, *args, **kwargs):
                self.assertNotIn(
                    str(path.absolute()),
                    protected,
                    "metadata builder must not inspect model/runtime reference paths",
                )
                return operation(path, *args, **kwargs)

            return checked

        builder = self.builder()
        with (
            patch.object(Path, "open", new=guard(original_open)),
            patch.object(Path, "stat", new=guard(original_stat)),
            patch.object(Path, "lstat", new=guard(original_lstat)),
        ):
            plan = builder.build()
        ported = {
            "profile",
            "actor_sha256",
            "helper_sha256",
            "source_audit_sha256",
            "owner_unit",
            "owner_profile",
            "host_observation_policy",
            "owner_sources",
            "cpu_affinity",
            "cpu_quota_percent",
            "threads",
            "environment",
            "resource_helper_provenance",
            "preparation",
        }
        self.assertEqual(set(plan), set(base))
        for key in set(base) - ported:
            self.assertEqual(plan[key], base[key], key)
        self.assertEqual(
            (plan["cpu_affinity"], plan["cpu_quota_percent"], plan["threads"]),
            ([15, 16, 17, 18, 19], 500, 5),
        )
        expected_env = {
            **base["environment"],
            **dict.fromkeys(
                (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ),
                "5",
            ),
        }
        self.assertEqual(plan["environment"], expected_env)

    def test_builder_refuses_payload_suffix_before_opening_or_statting_it(self):
        builder = self.builder()
        with (
            patch.object(Path, "open", side_effect=AssertionError("payload opened")),
            patch.object(Path, "stat", side_effect=AssertionError("payload statted")),
        ):
            with self.assertRaisesRegex(RuntimeError, "Python/JSON"):
                builder.read_small(HERE / "never-open-model.safetensors")

    def test_cpu5_accepts_positive_period_ratio_and_rejects_old_cpu16_or_cpu4_selection(self):
        helper = a.load_helper()
        identity = {
            "cgroup": "0::/FAKE/" + OWNER_UNIT,
            "affinity": [15, 16, 17, 18, 19],
            "environment": dict(helper.START_ENV),
        }
        limits = {"memory.max": "4294967296", "memory.swap.max": "0", "cpu.max": "500000 100000"}
        for quota in ("500000 100000", "1000000 200000"):
            a.validate_owner(identity, {**limits, "cpu.max": quota}, helper.START_ENV)
        for cores in (list(range(4, 20)), list(range(5, 10)), [16, 17, 18, 19], [15, 16, 17, 18]):
            with self.subTest(cores=cores), self.assertRaisesRegex(RuntimeError, "affinity"):
                a.validate_owner({**identity, "affinity": cores}, limits, helper.START_ENV)
        for unit in (
            "jev-d3-llama-cpp-cpu16-4g-coexist-20261024-08.service",
            "jev-d3-llama-cpp-cpu4-2g-coexist-20261024-07.service",
        ):
            with self.subTest(unit=unit), self.assertRaisesRegex(RuntimeError, "owner09"):
                a.validate_owner(
                    {**identity, "cgroup": "0::/FAKE/" + unit}, limits, helper.START_ENV
                )
        for quota in ("1600000 100000", "400000 100000", "500000 100001", "0 0"):
            with self.subTest(quota=quota), self.assertRaisesRegex(RuntimeError, "quota"):
                a.validate_owner(identity, {**limits, "cpu.max": quota}, helper.START_ENV)
        self.assertEqual((a.WALL_SECONDS, a.CALL_SECONDS, a.BATCH_SIZE, a.CALLS), (120, 20, 4, 2))
        self.assertEqual(
            [
                helper.CPU_ENV[name]
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            ],
            ["5"] * 4,
        )


if __name__ == "__main__":
    unittest.main()
