"""Authored CPU5 stdlib fakes only; NOT EXECUTED in this source-only port.

No NumPy/Torch/native library or model payload is imported by these tests.
Actual NumPy conversion/BLAS equivalence must be checked inside a future owner.
"""

import ast
import copy
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEGACY = ROOT / "artifacts/jev-runtime/decision3-readout-numpy-qualification-20261024-01"


def load(name, path, expected=None):
    raw = path.read_bytes()
    if expected is not None:
        assert hashlib.sha256(raw).hexdigest() == expected
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


actor = load("_cpu5_qualification_actor_test", HERE / "actor.py")
observer = load("_cpu5_qualification_observer_test", HERE / "observer.py")
probe = load("_cpu5_qualification_numpy_probe_test", HERE / "numpy_probe.py")
native = load(
    "_cpu5_qualification_native05_source_test", actor.NATIVE / "actor.py", actor.NATIVE_SHA
)
fakes = load(
    "_frozen_replay_stdlib_fakes",
    ROOT / "artifacts/jev-runtime/decision3-readout-feature-replay-20261023-01/test_replay.py",
    "f6b937a370843fb563d1643894ceb8da5ab28ba47af5bfb139ab2e5cab3cf79f",
)


class TimingTests(fakes.TimingTests):
    """Run the exact inherited expiry/primary cases against the new transaction."""

    def setUp(self):
        self.enterContext(patch.object(fakes, "actor", actor))
        self.enterContext(patch.object(fakes, "native", native))


class SourceOnlyCPU5Tests(unittest.TestCase):
    def pending(self):
        return json.loads((HERE / "plan.pending.json").read_text())

    def test_direct_and_cli_execute_hold_before_any_dependency_or_plan_read(self):
        with (
            patch.object(actor, "SOURCE_ONLY", True),
            patch.object(actor, "load_source", side_effect=AssertionError("dependency import")),
            patch.object(actor, "load_plan", side_effect=AssertionError("plan read")),
            patch.object(actor.time, "monotonic", return_value=0),
        ):
            with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
                actor.execute(Path("/FAKE/no-plan"), "0" * 64, Path("/FAKE/no-output"))
            with patch.object(
                actor.argparse, "ArgumentParser", side_effect=AssertionError("argument parser")
            ):
                for argv in ([], ["--execute"]):
                    with (
                        self.subTest(argv=argv),
                        self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"),
                    ):
                        actor.main(argv)

    def test_default_cli_after_fake_release_only_describes_no_execution(self):
        with (
            patch.object(actor, "SOURCE_ONLY", False),
            patch.object(actor, "load_source", side_effect=AssertionError("dependency import")),
            patch.object(actor, "sha", side_effect=AssertionError("source read")),
            patch("builtins.print") as printed,
        ):
            actor.main([])
        row = json.loads(printed.call_args.args[0])
        self.assertIs(row["source_only"], False)
        self.assertEqual(row["status"], "dry_default_no_model_or_job_execution")
        self.assertEqual(row["cpu_affinity"], [15, 16, 17, 18, 19])
        self.assertEqual(row["cpu_quota_percent"], 500)
        self.assertEqual(row["threads"], 5)
        self.assertEqual(row["interop_threads"], 1)
        self.assertEqual(row["memory_max_bytes"], 4294967296)
        self.assertEqual(row["whole_seconds_from_execute_entry"], 120)
        self.assertEqual(row["complete_prediction_seconds"], 20)
        self.assertEqual(row["feature_commit_status"], "PENDING_NONTRAINING")
        self.assertIsNone(row["labels"])
        self.assertIs(row["training_eligible"], False)

    def test_new_pins_catalog_actual_review_and_checks_remain_pending(self):
        pending = self.pending()
        self.assertIs(pending["executable"], False)
        self.assertIs(pending["source_only"], True)
        self.assertIs(pending["checks_executed"], False)
        self.assertIs(pending["actual_executed"], False)
        for key in (
            "actor_sha256",
            "observer_sha256",
            "numpy_probe_sha256",
            "fixture_sha256",
            "test_sha256",
            "selected_plan_sha256",
            "source_records",
            "actual_ML313_numpy_version_origins_BLAS_identities",
            "genuine_owned_result",
            "independent_source_review",
            "independent_actual_review",
        ):
            self.assertIsNone(pending[key], key)
        prerequisite = pending["prior_native05_root_review"]
        self.assertEqual(prerequisite["authority"], "root")
        self.assertIs(prerequisite["is_independent_review"], False)
        self.assertEqual(prerequisite["reported_sha256_prefix"], "ae81")
        for key in ("path", "sha256", "schema", "exact_acceptance_binding"):
            self.assertIsNone(prerequisite[key], key)
        self.assertIs(pending["training_eligible"], False)
        self.assertEqual(pending["feature_commit_status"], "PENDING_NONTRAINING")
        self.assertIsNone(pending["labels"])
        self.assertFalse((HERE / "plan.json").exists())

    def test_fixed_native05_and_owner09_pins_join_real_small_source_metadata(self):
        self.assertEqual(actor.NATIVE.name, "decision3-publisher-batch4-cpu-20261024-05")
        self.assertEqual(
            actor.NATIVE_SHA, "9a7e5af918c17f6824a720fb6d112ffa5fdac03493dde3efa176c7d50470a365"
        )
        self.assertEqual(
            actor.BASE_PLAN_SHA, "095db88b78687d90fbed51aafd736c1b06fb1a582312de8dff97bf6109131e96"
        )
        self.assertEqual(
            actor.NATIVE_HELPER_SHA,
            "4df54e06e8a795ebf5b300a3e2074319e781a278a2a870c21c2bd52897d7eb11",
        )
        self.assertEqual(actor.sha(actor.NATIVE / "edge-plan.json"), actor.BASE_PLAN_SHA)
        base = json.loads((actor.NATIVE / "edge-plan.json").read_text())
        self.assertEqual(base["actor_sha256"], actor.NATIVE_SHA)
        self.assertEqual(base["helper_sha256"], actor.NATIVE_HELPER_SHA)
        owner = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
        source_rows = {Path(row["path"]): row for row in base["owner_sources"]}
        for filename, expected in (
            ("controller.py", "81846efe2ee0357da07283c1a77a606283a920cbddef668c8f62def291057dee"),
            ("freeze.json", "52ac647a80ca1ef0c05acc45bf90c990f3b753b66df216c87cfb2bdf495020ef"),
        ):
            self.assertEqual(source_rows[owner / filename]["sha256"], expected)
            self.assertEqual(actor.sha(owner / filename), expected)
        freeze = json.loads((owner / "freeze.json").read_text())
        self.assertEqual(freeze["schema_id"], "cpu5-fast-coexist-owner-source-freeze-v1")
        self.assertEqual(
            base["owner_unit"], "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service"
        )
        self.assertEqual(
            base["owner_profile"], "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"
        )
        self.assertEqual(base["cpu_affinity"], [15, 16, 17, 18, 19])
        self.assertEqual(base["cpu_quota_percent"], 500)
        self.assertEqual(base["memory_max_bytes"], 4294967296)
        self.assertEqual(base["startup_available_bytes"], 30414140212)
        self.assertEqual(base["threads"], 5)
        self.assertEqual(base["call_seconds"], 20)
        self.assertEqual(base["wall_seconds"], 120)
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            self.assertEqual(base["environment"][name], "5")

    def test_observer_fixture_math_and_probe_are_minimal_source_port(self):
        for name in ("observer.py", "fixture.json"):
            self.assertEqual((HERE / name).read_bytes(), (LEGACY / name).read_bytes())
        self.assertEqual(
            (HERE / "numpy_probe.py").read_bytes(),
            (LEGACY / "numpy_probe.py")
            .read_bytes()
            .replace(b"new owner07 policy required", b"new owner09 policy required"),
        )
        old_tree = ast.parse(
            (LEGACY / "actor.py").read_text().replace('"PENDING"', '"PENDING_NONTRAINING"')
        )
        new_tree = ast.parse((HERE / "actor.py").read_text())
        for name in (
            "sha",
            "load_source",
            "authenticate_sources",
            "remove_handles",
            "transaction",
            "perform_call",
        ):
            old = next(
                node
                for node in old_tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            new = next(
                node
                for node in new_tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            self.assertEqual(
                ast.dump(new, include_attributes=False),
                ast.dump(old, include_attributes=False),
                name,
            )
        probe_tree = ast.parse((HERE / "numpy_probe.py").read_bytes())
        qualify = next(
            node
            for node in probe_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "qualify"
        )
        core_calls = [
            node.func.attr
            for node in ast.walk(qualify)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "core"
        ]
        self.assertEqual(core_calls, ["probabilities"])


class SingletonObservationTests(unittest.TestCase):
    def test_declared_observer_byte_delta_is_only_singleton_bound(self):
        original = (
            ROOT / "artifacts/jev-runtime/decision3-readout-feature-replay-20261023-01/observer.py"
        )
        raw = original.read_bytes()
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "d5dbcbc77775d0dcd6c42345b4bf0e633bb3d54f9a9b4fbdfda4bb6109873783",
        )
        expected = raw.replace(b"1 < n <= CODES", b"1 <= n <= CODES")
        self.assertEqual(expected, (HERE / "observer.py").read_bytes())

    def test_four_rows_varying_counts_singleton_and_illegal_mask(self):
        hidden = fakes.full_hidden(rows=4)
        witness = observer.FinalNormObserver(fakes.Torch, 4, 2)
        self.assertIsNone(witness.first_layer(None, (hidden,)))
        self.assertIsNone(witness(None, (), hidden))
        weight = fakes.Tensor(
            [[float(index) if index < 8 else 1000.0] + [0.0] * 1023 for index in range(255)]
        )
        runtime = types.SimpleNamespace(readout=weight, temperature=1.0, device="cpu")
        rows = observer.replay_probabilities(runtime, fakes.Torch, witness, (8, 1, 2, 3))
        self.assertEqual(list(map(len, rows)), [8, 1, 2, 3])
        self.assertEqual(rows[1], [1.0])
        self.assertEqual(len(witness.first_layer_events), 1)
        self.assertEqual(len(witness.norm_events), 1)
        with self.assertRaises(RuntimeError):
            observer.replay_probabilities(runtime, fakes.Torch, witness, (8, 0, 2, 3))


class ContractTests(unittest.TestCase):
    def request(self):
        return json.loads((HERE / "fixture.json").read_text())["request"]

    def raw(self):
        request = self.request()
        values = [[0.125] * 8, [1.0], [0.25, 0.75], [0.3, 0.3, 0.4]]
        result = fakes.raw_for(request["questions"], values)
        for answer in result["answers"].values():
            probabilities = list(answer["probabilities"].values())
            entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
            answer["type"] = "choice"
            answer["confidence"] = (
                1.0
                if len(probabilities) < 2
                else max(0.0, min(1.0, 1 - entropy / math.log(len(probabilities))))
            )
        return result

    def test_published_singleton_rule_without_repair_or_semantic_labels(self):
        request, raw = self.request(), self.raw()
        fake = types.SimpleNamespace(validate_native=Mock())
        result = probe.validate_answers(fake, raw, request, {}, {})
        self.assertTrue(result["contract_valid"])
        self.assertFalse(result["semantic_expected_answer_or_quality_claim"])
        fake.validate_native.assert_called_once()
        for bad in (0.0, True, float("nan")):
            invalid = copy.deepcopy(raw)
            invalid["answers"]["click_target"]["confidence"] = bad
            with self.subTest(confidence=bad), self.assertRaises(RuntimeError):
                probe.validate_answers(fake, invalid, request, {}, {})

    def test_order_probability_choice_and_nonfinite_refuse(self):
        request, raw = self.request(), self.raw()
        fake = types.SimpleNamespace(validate_native=Mock())
        changes = (
            lambda value: value["answers"]["operation"].update(choice="BLOCKED"),
            lambda value: value["answers"]["click_target"]["probabilities"].update(
                folder_control=float("nan")
            ),
            lambda value: value["answers"]["type_text_target"].update(
                probabilities={"drawer_note": 0.75, "desk_note": 0.25}
            ),
        )
        for change in changes:
            invalid = copy.deepcopy(raw)
            change(invalid)
            with self.assertRaises(RuntimeError):
                probe.validate_answers(fake, invalid, request, {}, {})

    def test_numpy_cannot_import_before_genuine_loader_completion(self):
        for state in (
            {},
            {"model_loads": 0},
            {"model_loads": 1, "initial_full_model_manifest_runtime_sources_authenticated": False},
        ):
            with (
                patch.object(probe, "owned", side_effect=AssertionError("owned")),
                patch.object(probe.importlib, "import_module", side_effect=AssertionError("NumPy")),
                self.assertRaises(RuntimeError),
            ):
                probe.load_numpy(None, None, state)

    def test_numpy_origin_and_version_are_observed_not_an_assumed_repo_runtime(self):
        fake = types.SimpleNamespace(
            __file__="/FAKE/venv/lib/python3.13/site-packages/numpy/__init__.py",
            __version__="2.5.3",
        )
        helper = types.SimpleNamespace(RUNTIME_PYTHON=Path("/FAKE/venv/bin/python"))
        state = {
            "model_loads": 1,
            "initial_full_model_manifest_runtime_sources_authenticated": True,
        }
        with (
            patch.object(probe, "owned"),
            patch.object(probe.importlib, "import_module", return_value=fake),
        ):
            self.assertIs(probe.load_numpy(None, helper, state), fake)
            fake.__file__ = "/FAKE/repo3.12/numpy/__init__.py"
            with self.assertRaises(RuntimeError):
                probe.load_numpy(None, helper, state)
            fake.__file__ = "/FAKE/venv/lib/python3.13/site-packages/numpy/__init__.py"
            for version in ("1.26.4", "unknown", True):
                fake.__version__ = version
                with self.subTest(version=version), self.assertRaises(RuntimeError):
                    probe.load_numpy(None, helper, state)

    def test_executed_source_projection_cannot_omit_duplicate_or_escape(self):
        paths = [
            HERE / name for name in ("actor.py", "observer.py", "fixture.json", "numpy_probe.py")
        ]
        paths.append(actor.CORE)
        paths.extend(
            actor.NATIVE / name
            for name in (
                "actor.py",
                "resource_helper.py",
                "source_audit.py",
                "edge-plan.json",
                "freeze.json",
            )
        )
        rows = [{"path": str(path), "size": 1, "sha256": "0" * 64} for path in paths]
        helper = types.SimpleNamespace(check_record=Mock())
        actor.authenticate_sources(helper, {"source_records": rows})
        self.assertEqual(helper.check_record.call_count, 10)
        for invalid in (
            rows[:-1],
            [*rows[:-1], rows[0]],
            [*rows[:-1], {**rows[-1], "path": "/tmp/not-selected.py"}],
        ):
            with self.subTest(rows=invalid), self.assertRaises(RuntimeError):
                actor.authenticate_sources(helper, {"source_records": invalid})

    def test_owned_hash_is_chunked_and_changed_identity_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fake-source-not-a-native-binary.txt"
            path.write_bytes(b"FAKE")
            helper = types.SimpleNamespace(ensure_elapsed=Mock())
            with patch.object(probe, "owned"):
                row = probe.file_identity(path, None, helper, 0)
                self.assertEqual(row["sha256"], hashlib.sha256(b"FAKE").hexdigest())
                self.assertEqual(row["size_bytes"], 4)
                helper.ensure_elapsed.side_effect = lambda *a: path.write_bytes(b"CHANGED")
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    probe.file_identity(path, None, helper, 0)

    def test_wrong_fp32_dtype_shape_and_device_refuse_before_array_bridge(self):
        for dtype, device, shape in (
            ("bfloat16", "cpu", (4, 1024)),
            ("float32", "cuda", (4, 1024)),
            ("float32", "cpu", (3, 1024)),
        ):
            tensor = types.SimpleNamespace(
                dtype=dtype, device=types.SimpleNamespace(type=device), shape=shape
            )
            with (
                self.subTest(dtype=dtype, device=device, shape=shape),
                self.assertRaises(RuntimeError),
            ):
                probe.copy_fp32(None, fakes.Torch, tensor, (4, 1024))

    def test_fixture_nontraining_four_original_head_names_and_counts(self):
        fixture = json.loads((HERE / "fixture.json").read_text())
        self.assertIsNone(fixture["semantic_expected_answers"])
        self.assertTrue(fixture["not_DSL_train_dev_sealed_or_six_diagnostics"])
        self.assertEqual(list(fixture["request"]["questions"]), list(actor.HEADS))
        self.assertEqual(
            [len(q["criteria"]) for q in fixture["request"]["questions"].values()], [8, 1, 2, 3]
        )

    def test_pending_plan_cannot_be_consumed(self):
        path = HERE / "plan.pending.json"
        helper = types.SimpleNamespace(read_json=lambda _: json.loads(path.read_text()))
        with self.assertRaises(RuntimeError):
            actor.load_plan(helper, path, actor.sha(path))

    def test_selected_plan_resource_policy_and_typed_counts_refuse_mismatches(self):
        # Fake metadata only: source checks are mocked and source row digests are
        # intentionally invalid. No runnable selected plan is written to HERE.
        paths = [
            HERE / name for name in ("actor.py", "observer.py", "fixture.json", "numpy_probe.py")
        ]
        paths.append(actor.CORE)
        paths.extend(
            actor.NATIVE / name
            for name in (
                "actor.py",
                "resource_helper.py",
                "source_audit.py",
                "edge-plan.json",
                "freeze.json",
            )
        )
        selected = {
            "schema": "decision3-owned-numpy-zero-delta-nontraining-plan-v1",
            "executable": True,
            "fake_test_metadata_only": True,
            "profile": "decision3-edge-numpy-zero-delta-nontraining-cpu5-fast-4g-coexist-20261024-v1",
            "actor_sha256": actor.sha(HERE / "actor.py"),
            "fixture_sha256": actor.sha(HERE / "fixture.json"),
            "native_actor_sha256": "9a7e5af918c17f6824a720fb6d112ffa5fdac03493dde3efa176c7d50470a365",
            "native_helper_sha256": "4df54e06e8a795ebf5b300a3e2074319e781a278a2a870c21c2bd52897d7eb11",
            "native_plan_sha256": "095db88b78687d90fbed51aafd736c1b06fb1a582312de8dff97bf6109131e96",
            "owner_controller_sha256": "81846efe2ee0357da07283c1a77a606283a920cbddef668c8f62def291057dee",
            "owner_freeze_sha256": "52ac647a80ca1ef0c05acc45bf90c990f3b753b66df216c87cfb2bdf495020ef",
            "head_core_sha256": "6761dd654481424f4d6c908a7c9fd2d43e4762eeb5420af9f22612ffe77657ed",
            "native_plan_path": str(actor.NATIVE / "edge-plan.json"),
            "wall_seconds": 120,
            "call_seconds": 20,
            "requests": 1,
            "head_counts": [8, 1, 2, 3],
            "temperature": 1.0,
            "absolute_tolerance": 1e-6,
            "relative_tolerance": 1e-5,
            "catalog_mode": "owned_actual_discovery_not_preselected",
            "training_eligible": False,
            "owner_unit": "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service",
            "owner_profile": "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1",
            "host_observation_policy": "comfy-observed-cpu-coexist-v1",
            "memory_max_bytes": 4294967296,
            "reserve_bytes": 26119172916,
            "startup_available_bytes": 30414140212,
            "swap_max_bytes": 0,
            "owner_runtime_seconds": 10800,
            "cpu_affinity": [15, 16, 17, 18, 19],
            "cpu_quota_percent": 500,
            "threads": 5,
            "interop_threads": 1,
            "source_records": [
                {"path": str(path), "size": 1, "sha256": "0" * 64} for path in paths
            ],
        }
        helper = types.SimpleNamespace(
            read_json=lambda path: json.loads(Path(path).read_text()),
            check_record=Mock(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "only-small-plan-metadata.json"
            path.write_text(json.dumps(selected))
            plan, request = actor.load_plan(helper, path, actor.sha(path))
            self.assertEqual(plan["head_counts"], [8, 1, 2, 3])
            self.assertEqual(list(request["questions"]), list(actor.HEADS))
            for key, value in (
                ("head_counts", [8, True, 2, 3]),
                ("memory_max_bytes", 3 * 1024**3),
                ("startup_available_bytes", 30414140211),
                ("cpu_affinity", [16, 17, 18, 19]),
                ("cpu_quota_percent", 400),
                ("threads", 4),
                ("threads", True),
                ("interop_threads", 2),
                ("owner_unit", "jev-d3-llama-cpp-cpu4-2g-coexist-20261024-07.service"),
                ("native_helper_sha256", "0" * 64),
                ("owner_controller_sha256", "0" * 64),
                ("owner_freeze_sha256", "0" * 64),
                ("executable", False),
                ("host_observation_policy", "old-idle-only-policy"),
                ("fixture_sha256", "0" * 64),
            ):
                invalid = {**selected, key: value}
                path.write_text(json.dumps(invalid))
                with self.subTest(field=key), self.assertRaises(RuntimeError):
                    actor.load_plan(helper, path, actor.sha(path))

    def test_source_imports_no_numpy_and_original_transaction_ast_unchanged(self):
        for name in ("actor.py", "observer.py", "numpy_probe.py"):
            tree = ast.parse((HERE / name).read_bytes())
            roots = {
                node.module.split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            roots |= {
                alias.name.split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            self.assertFalse({"numpy", "torch", "transformers"} & roots)
        old = ROOT / "artifacts/jev-runtime/decision3-readout-feature-replay-20261023-01/actor.py"

        def definition(path):
            return next(
                node
                for node in ast.parse(path.read_bytes()).body
                if isinstance(node, ast.FunctionDef) and node.name == "transaction"
            )

        self.assertEqual(
            ast.dump(definition(old), include_attributes=False),
            ast.dump(definition(HERE / "actor.py"), include_attributes=False),
        )
        self.assertFalse(actor.MODEL_MODULES.intersection(sys.modules))


if __name__ == "__main__":
    unittest.main()
