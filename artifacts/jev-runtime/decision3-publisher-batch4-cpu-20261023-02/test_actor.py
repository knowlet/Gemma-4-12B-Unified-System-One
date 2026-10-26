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

    def test_owner06_validator_rejects_old_units_or_caps(self):
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
        self.assertEqual(helper.MEMORY_MAX, 2147483648)
        self.assertEqual(helper.RESERVE, 26119172916)

    def test_resource_hostgate_full_two_gib_and_no_credit(self):
        helper = a.load_helper()
        total = 130595864576

        def memory(available):
            return "MemTotal: %d kB\nMemAvailable: %d kB\n" % (total // 1024, available // 1024)

        floor = 28266656564
        with patch.object(helper.Path, "read_text", return_value=memory(floor + 1024)):
            result = helper.host_gate()
        self.assertEqual(result["required_available_bytes"], floor)
        self.assertEqual(result["swap_credit_bytes"], 0)
        self.assertEqual(result["cache_credit_bytes"], 0)
        for available in (floor - 1, helper.RESERVE + 1024**3, helper.RESERVE):
            with patch.object(helper.Path, "read_text", return_value=memory(available)):
                with self.assertRaisesRegex(RuntimeError, "plus 2GiB"):
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
            with self.assertRaisesRegex(RuntimeError, "plus 3GiB"):
                namespace["host_gate"]()

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
            "affinity": [16, 17, 18, 19],
            "environment": env,
        }
        for cap in (a.MEMORY_MAX - 1, a.MEMORY_MAX + 1, 3221225472):
            with self.assertRaises(RuntimeError):
                a.validate_owner(
                    identity,
                    {"memory.max": str(cap), "memory.swap.max": "0", "cpu.max": "400000 100000"},
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


if __name__ == "__main__":
    unittest.main()
