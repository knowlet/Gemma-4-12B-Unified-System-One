"""HELD stdlib fake-test SOURCE; no import, test, model or host check was run.

These cases exercise admission refusal, diagnostic error retention and complete
compiler/hook composition. They cannot establish any28GiB fit or GPU behavior.
"""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_held_native28_memory_probe_test", HERE / "memory_probe.py")
actor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(actor)


class EntryTests(unittest.TestCase):
    def test_default_dry_does_not_load_plan_dependency_data_or_model(self):
        output = io.StringIO()
        with (patch.object(actor, "load_source", side_effect=AssertionError("source import")),
              patch.object(actor, "read_plan", side_effect=AssertionError("plan")),
              contextlib.redirect_stdout(output)):
            self.assertEqual(actor.main([]), 0)
        result = json.loads(output.getvalue())
        self.assertTrue(result["source_only"])
        self.assertEqual(result["counts"], {"train": 142, "development": 37, "total": 179})
        self.assertEqual(result["steps"], 2)
        self.assertFalse(result["normal_adapter_export"])
        self.assertFalse(result["training_browser_or_capacity_qualified"])

    def test_source_hold_refuses_before_reading_even_a_selected_plan(self):
        with (patch.object(actor, "read_plan", side_effect=AssertionError("plan")),
              patch.object(actor, "load_source", side_effect=AssertionError("dependency")),
              self.assertRaisesRegex(ValueError, "HELD SOURCE ONLY")):
            actor.execute(Path("/invented/not-a-real-plan.json"), "0" * 64,
                          Path("/invented/not-a-real-output"))

    def test_raw_constructor_and_partial_training_evidence_are_retained(self):
        error = ValueError("invented unit rejection")
        original = {"memory_current": 30, "events": {"oom": 1}}
        error.native28_resource_evidence = original
        error.action_training_evidence = {"completed_updates": 1,
                                          "active": {"split": "train", "phase": "backward"}}
        result = actor.failure_record(error)
        self.assertEqual(result["message"], "invented unit rejection")
        self.assertIs(result["native28_resource_evidence"], original)
        self.assertEqual(result["action_training_evidence"]["completed_updates"], 1)
        self.assertEqual(original, {"memory_current": 30, "events": {"oom": 1}})

    def test_forward_guard_preserves_hook_input_and_routes_every_native_mode(self):
        captured = {}
        handle = object()

        def register(callback, **kwargs):
            captured.update(callback=callback, kwargs=kwargs)
            return handle

        gate = NS(check=Mock(return_value={"admitted": True}))
        model = NS(lm=NS(register_forward_pre_hook=register))
        self.assertIs(actor.install_forward_guard(model, gate), handle)
        self.assertEqual(captured["kwargs"], {"with_kwargs": True})
        inputs, kwargs = (object(),), {"use_cache": False}
        for training, phase in ((True, "before-native-train-forward"),
                                (False, "before-native-development-forward")):
            self.assertIsNone(captured["callback"](NS(training=training), inputs, kwargs))
            gate.check.assert_called_with(phase, "train")
        gate.check.side_effect = ValueError("invented stop")
        with self.assertRaisesRegex(ValueError, "invented stop"):
            captured["callback"](NS(training=True), inputs, kwargs)


class PlanTests(unittest.TestCase):
    def valid(self):
        return {
            "schema": actor.PLAN_SCHEMA, "profile": actor.PROFILE, "executable": True,
            "actor_sha256": "0" * 64, "native_fitter_sha256": actor.NATIVE_SHA,
            "training_kernel_sha256": actor.KERNEL_SHA, "public_fitter_sha256": actor.PUBLIC_SHA,
            "counts": dict(actor.COUNTS), "steps": 2, "seed": 42, "learning_rate": 1e-4,
            "prompt_format": actor.FORMAT, "max_context": 8192, "whole_seconds": 10800,
            "historical_root": str(actor.ROOT), "normal_adapter_export": False,
            "training_or_browser_qualified": False, "initial_action_adapter": None,
            "resource_gate_sha256": "1" * 64,
            "expected_runtime_source_identity_sha256": "2" * 64,
            "root_guardian_selection": {"status": "invented-metadata-only-not-a-live-proof"},
            "expected_boot_id": "INVENTED-BOOT",
            "expected_unit": actor.EXPECTED_UNIT,
            "runtime_python": str(Path(sys.executable).resolve()),
        }

    def read_invented(self, plan):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invented-plan.json"
            raw = json.dumps(plan).encode()
            path.write_bytes(raw)
            with patch.object(actor, "source_ref", return_value={"sha256": "0" * 64}):
                return actor.read_plan(path, hashlib.sha256(raw).hexdigest())

    def test_fixed_full_population_and_two_update_policy(self):
        value = self.read_invented(self.valid())
        self.assertEqual(value["counts"], actor.COUNTS)
        for change in (
            {"steps": True}, {"steps": 1}, {"steps": 120}, {"seed": 43},
            {"learning_rate": 2e-4}, {"max_context": 4096}, {"max_context": 8192.0},
            {"whole_seconds": 10800.0}, {"prompt_format": "raw-json-v1"},
            {"counts": {"train": 142.0, "development": 37, "total": 179}},
            {"counts": {"train": 141, "development": 37, "total": 178}},
            {"normal_adapter_export": True}, {"training_or_browser_qualified": True},
            {"resource_gate_sha256": None}, {"root_guardian_selection": None},
            {"executable": False}, {"initial_action_adapter": 123},
            {"runtime_python": "/invented/wrong/python"},
            {"expected_unit": "/invented/wrong-unit"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read_invented({**self.valid(), **change})

    def test_duplicate_field_refused_even_when_raw_hash_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invented-duplicate.json"
            raw = b'{"schema":"a","schema":"b"}'
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                actor.read_plan(path, hashlib.sha256(raw).hexdigest())


class CompilerCompositionTests(unittest.TestCase):
    def population(self):
        cases = {split: [NS(id=f"invented-{split}-{index}") for index in range(count)]
                 for split, count in (("train", 142), ("development", 37))}
        wires = [{"case_id": case.id, "payload": {"model": "invented-original-model"}}
                 for split in cases for case in cases[split]]
        return cases, wires

    def test_all179_original_compositions_are_retained_without_payload_mutation(self):
        cases, wires = self.population()
        original = copy.deepcopy(wires)
        calls, phases = [], []

        def compile_actions(model, population, snapshot, **kwargs):
            self.assertEqual(len(population), 1)
            self.assertEqual(kwargs, {"identity": "invented-identity", "prompt_format": actor.FORMAT})
            wire = json.loads(snapshot["blobs"]["wire_requests.jsonl"])
            case = population[0]
            self.assertEqual(case.id, wire["case_id"])
            calls.append(case.id)
            return [{"record": {"case_id": case.id, "prompt_tokens": 7,
                                "native_stop_token_id": 9}, "targets": [1, 2]}]

        def append(path, record):
            with Path(path).open("a") as stream:
                stream.write(json.dumps(record) + "\n")

        def formatter(payload, identity, **kwargs):
            self.assertEqual(payload["model"], identity)
            self.assertEqual(kwargs, {"prompt_format": actor.FORMAT})
            return None, None, ["invented-label"], None

        kernel = NS(IDENTITY="invented-identity", compile_actions=compile_actions, _append=append)
        cfg = NS(compile_action_plan=lambda *a, **k: NS(evidence=lambda: {"invented": True}),
                 LegalActionTokenTrie=lambda *a, **k: NS(canonical_actions=["invented"],
                                                        encoded_actions=[[1]], telemetry=lambda: {}))
        gate = NS(check=lambda phase, scope: phases.append((phase, scope)))
        with tempfile.TemporaryDirectory() as directory:
            rows, metadata, cfg_meta = actor.compile_population(
                NS(tok=object()), cases, wires, Path(directory), gate, kernel, cfg, formatter)
        self.assertEqual({split: len(values) for split, values in rows.items()},
                         {"train": 142, "development": 37})
        self.assertEqual(len(calls), 179)
        self.assertEqual(len(phases), 179)
        self.assertEqual(phases[0], ("before-BF16-processor-compile-train-0", "train"))
        self.assertEqual(phases[141], ("before-BF16-processor-compile-train-141", "train"))
        self.assertEqual(phases[-1], ("before-BF16-processor-compile-development-36", "train"))
        self.assertEqual(metadata["train"]["prompt_tokens"], 142 * 7)
        self.assertEqual(metadata["development"]["target_tokens"], 37 * 2)
        self.assertEqual(cfg_meta["train"]["examples"], 142)
        self.assertEqual(wires, original)

    def test_missing_population_fails_before_any_compiler_or_forward(self):
        cases, wires = self.population()
        cases["train"].pop()
        kernel = NS(compile_actions=Mock(side_effect=AssertionError("compiler")))
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            actor.compile_population(None, cases, wires, Path(directory), None, kernel, None, None)
        kernel.compile_actions.assert_not_called()


if __name__ == "__main__":
    unittest.main()
