"""Stdlib-only boundary and fake API tests; no Torch/CUDA/NVML execution."""

import ast
import contextlib
import importlib.util
import io
import json
import os
import struct
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ENTRY = Path(__file__).with_name("fixture.py").absolute()
spec = importlib.util.spec_from_file_location("gpu_probe_static", ENTRY)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
ENV = {
    "CUDA_VISIBLE_DEVICES": fixture.EXPECTED_GPU,
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "USE_TF": "0",
    "USE_FLAX": "0",
    "JEV_GPU_PROBE_DEADLINE_MONOTONIC": str(time.monotonic() + 25),
}


class StaticTests(unittest.TestCase):
    def manifest(self, root):
        hardware = root / "IMPOSSIBLE_LIVE_GPU_IDENTITY.json"
        hardware.write_text(
            json.dumps(
                {
                    "exit_code": 0,
                    "stdout": fixture.EXPECTED_GPU + ", SYNTHETIC\n",
                    "boot_id": "STATIC-BOOT-NOT-LIVE",
                }
            )
        )
        paths = {"entry": ENTRY, "hardware_identity": hardware}
        sources = {
            key: {
                "path": str(path),
                "sha256": fixture.sha(path.read_bytes()),
                "size_bytes": path.stat().st_size,
            }
            for key, path in paths.items()
        }
        value = {"profile": fixture.PROFILE, "sources": sources, "versions": {}, "binary_stats": []}
        p = root / "selection.json"
        p.write_text(json.dumps(value))
        return value, [
            "--run",
            "--profile",
            fixture.PROFILE,
            "--selection",
            str(p),
            "--selection-sha256",
            fixture.sha(p.read_bytes()),
            "--entry-sha256",
            sources["entry"]["sha256"],
            "--output",
            str(root / "fresh"),
        ]

    def test_dry_no_actor_or_writes(self):
        with tempfile.TemporaryDirectory() as name:
            p = Path(name) / "fresh"
            with (
                patch.object(fixture, "run_torch", side_effect=AssertionError("forbidden")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(fixture.main(["--output", str(p)]), 0)
            self.assertFalse(p.exists())

    def test_no_top_framework_import_or_model_mapping_apis(self):
        tree = ast.parse(ENTRY.read_text())
        top = [
            name.name.split(".")[0]
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for name in node.names
        ]
        self.assertFalse(set(top) & {"torch", "numpy", "safetensors", "transformers"})
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                self.assertNotIn(
                    node.attr, {"from_pretrained", "mmap", "safe_open", "load_state_dict"}
                )
            if isinstance(node, ast.Import):
                self.assertFalse(
                    {name.name for name in node.names} & {"mmap", "safetensors", "transformers"}
                )

    def test_uuid_normalization_only_optional_prefix(self):
        expected = fixture.EXPECTED_GPU.removeprefix("GPU-")
        self.assertEqual(fixture.normalize_uuid(fixture.EXPECTED_GPU), expected)
        self.assertEqual(fixture.normalize_uuid(expected), expected)
        for value in (expected.upper(), "GPU-" + expected + ",other", None, ""):
            with self.assertRaises(ValueError):
                fixture.normalize_uuid(value)

    def test_pre_sync_snapshot_does_not_query_gpu_metrics(self):
        gpu = types.SimpleNamespace(is_initialized=lambda: True)
        value = fixture.cuda_snapshot(types.SimpleNamespace(cuda=gpu), False)
        self.assertTrue(value["initialized"])
        for key in ("free_bytes", "total_bytes", "allocated_bytes", "reserved_bytes"):
            self.assertIsNone(value[key])

    def test_post_sync_snapshot_requires_initialized(self):
        gpu = types.SimpleNamespace(is_initialized=lambda: False)
        with self.assertRaisesRegex(ValueError, "prior explicit init"):
            fixture.cuda_snapshot(types.SimpleNamespace(cuda=gpu), True)

    def test_post_sync_snapshot_records_actual_fake_api_values(self):
        gpu = types.SimpleNamespace(
            is_initialized=lambda: True,
            mem_get_info=lambda i: (100, 200),
            memory_allocated=lambda i: 3,
            memory_reserved=lambda i: 4,
            max_memory_allocated=lambda i: 5,
            max_memory_reserved=lambda i: 6,
        )
        value = fixture.cuda_snapshot(types.SimpleNamespace(cuda=gpu), True)
        self.assertEqual(
            (
                value["free_bytes"],
                value["total_bytes"],
                value["allocated_bytes"],
                value["reserved_bytes"],
            ),
            (100, 200, 3, 4),
        )

    def test_independent_expected_BF16_bit_pattern(self):
        data = struct.pack("=H", 0x4180) * 256
        self.assertEqual(len(data), 512)
        self.assertEqual(struct.unpack("=f", struct.pack("=I", 0x41800000))[0], 16.0)

    def test_cleanup_sync_failure_preserves_primary_and_still_clears_cache(self):
        calls = []

        class Owned(list):
            def clear(self):
                calls.append("clear")
                super().clear()

        def failed_sync(_index):
            calls.append("sync")
            raise RuntimeError("controlled cleanup sync failure")

        gpu = types.SimpleNamespace(
            is_initialized=lambda: True,
            synchronize=failed_sync,
            empty_cache=lambda: calls.append("empty_cache"),
        )
        primary = fixture.DeadlinePartial("controlled original partial")
        owned, timings = Owned([object()]), {}
        returned, errors = fixture.release_owned(
            types.SimpleNamespace(cuda=gpu), owned, timings, primary
        )
        self.assertIs(returned, primary)
        self.assertEqual(calls, ["sync", "clear", "empty_cache"])
        self.assertEqual(owned, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("controlled cleanup sync failure", primary.__notes__[0])
        self.assertIn("cleanup_sync_seconds", timings)
        self.assertIn("own_empty_cache_seconds", timings)

    def test_gpu_metrics_failure_preserves_cpu_stage_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            result = {"started_monotonic": time.monotonic(), "stages": [], "completed_stages": []}
            gpu = types.SimpleNamespace(
                is_initialized=lambda: True,
                mem_get_info=lambda _i: (_ for _ in ()).throw(
                    RuntimeError("controlled GPU query refusal")
                ),
            )
            original_read = fixture.read

            def fake_read(path):
                return (
                    b"MemTotal: 100 kB\nMemAvailable: 50 kB\n"
                    if str(path) == "/proc/meminfo"
                    else original_read(path)
                )

            with (
                patch.object(fixture, "memory_snapshot", return_value={"STATIC_STUB": True}),
                patch.object(fixture, "read", side_effect=fake_read),
            ):
                with self.assertRaisesRegex(RuntimeError, "controlled GPU query refusal"):
                    fixture.stage(
                        root,
                        result,
                        "synthetic-failed-query",
                        types.SimpleNamespace(cuda=gpu),
                        True,
                    )
            saved = json.loads((root / "stage-00.json").read_text())
            self.assertEqual(saved["memory"], {"STATIC_STUB": True})
            self.assertEqual(
                saved["CUDA"]["metrics_error"]["message"], "controlled GPU query refusal"
            )
            self.assertEqual(result["completed_stages"], [])

    def test_environment_refuses_other_gpu_before_actor(self):
        with tempfile.TemporaryDirectory() as name:
            _, args = self.manifest(Path(name))
            with (
                patch.dict(os.environ, {**ENV, "CUDA_VISIBLE_DEVICES": "0"}, clear=True),
                patch.object(fixture, "run_torch", side_effect=AssertionError("forbidden")),
            ):
                with self.assertRaisesRegex(ValueError, "selected GPU"):
                    fixture.main(args)

    def test_missing_deadline_refuses_before_actor(self):
        with tempfile.TemporaryDirectory() as name:
            _, args = self.manifest(Path(name))
            env = {
                key: value
                for key, value in ENV.items()
                if key != "JEV_GPU_PROBE_DEADLINE_MONOTONIC"
            }
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(KeyError):
                    fixture.main(args)

    def test_selection_and_entry_hash_refusals(self):
        with tempfile.TemporaryDirectory() as name:
            _, args = self.manifest(Path(name))
            for selected, own in (("0" * 64, args[8]), (args[6], "0" * 64)):
                with self.assertRaises(ValueError):
                    fixture.selection(Path(args[4]), selected, own)

    def test_symlink_source_refused(self):
        with tempfile.TemporaryDirectory() as name:
            p = Path(name) / "link"
            p.symlink_to(ENTRY)
            with self.assertRaisesRegex(ValueError, "Ordinary"):
                fixture.read(p)

    def stub_main(self, mode, source_refusal=False):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            selected, args = self.manifest(root)
            original_read = fixture.read

            def fake_read(path):
                return (
                    b"STATIC-BOOT-NOT-LIVE\n"
                    if str(path) == "/proc/sys/kernel/random/boot_id"
                    else original_read(path)
                )

            def fake_stage(_out, result, name, *_args, **_kwargs):
                result["completed_stages"].append(name)

            def fake_actor(_out, _selected, result, *_args):
                result["STATIC_STUB_NO_GPU_OR_TORCH_QUALIFICATION"] = True
                if mode == "success":
                    result.update(
                        completed_stages=list(fixture.PLANNED_STAGES),
                        mechanism_checks_completed=True,
                        context_and_tiny_operator_completed=True,
                    )
                elif mode == "partial":
                    result["completed_stages"] = fixture.PLANNED_STAGES[:2]
                    raise fixture.DeadlinePartial("controlled partial")
                else:
                    raise RuntimeError("controlled primary")

            choose = (
                patch.object(
                    fixture,
                    "selection",
                    side_effect=[selected, ValueError("controlled final source refusal")],
                )
                if source_refusal
                else contextlib.nullcontext()
            )
            with (
                patch.dict(os.environ, ENV, clear=True),
                patch.object(fixture, "read", side_effect=fake_read),
                patch.object(fixture, "stage", side_effect=fake_stage),
                patch.object(fixture, "run_torch", side_effect=fake_actor),
                choose,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                rc = fixture.main(args)
            return rc, json.loads((root / "fresh" / "receipt.json").read_text())

    def test_stub_all_stages_complete(self):
        rc, result = self.stub_main("success")
        self.assertEqual(rc, 0)
        self.assertTrue(result["passed"] and result["completed"])
        self.assertFalse(
            result["CPU_ONLY"] or result["model_payload_read"] or result["model_loaded"]
        )

    def test_partial_is_saved_exit2_not_completed(self):
        rc, result = self.stub_main("partial")
        self.assertEqual(rc, 2)
        self.assertEqual(result["status"], "partial_deadline")
        self.assertFalse(
            result["passed"] or result["completed"] or result["context_and_tiny_operator_completed"]
        )
        self.assertEqual(len(result["completed_stages"]), 2)

    def test_final_source_failure_preserves_primary(self):
        rc, result = self.stub_main("failure", True)
        self.assertEqual(rc, 1)
        self.assertEqual(result["error"]["message"], "controlled primary")
        self.assertIn("controlled final source refusal", result["error"]["notes"][0])

    def test_partial_with_source_refusal_is_failure(self):
        rc, result = self.stub_main("partial", True)
        self.assertEqual(rc, 1)
        self.assertEqual(result["status"], "failed")
        self.assertIn("Earlier deadline partial", result["error"]["notes"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
