"""Stdlib synthetic guards only: no runtime import, native or model payload reads."""

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

HERE = Path(__file__).absolute().parent
SPEC = importlib.util.spec_from_file_location("_catalog_tests", HERE / "catalog.py")
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


class Tests(unittest.TestCase):
    def plan(self):
        return json.loads((HERE / "plan.json").read_text())

    def selected(self, plan):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            blob = json.dumps(plan).encode()
            path.write_bytes(blob)
            return c.load_plan(path, c.digest(blob))

    def test_dry_never_imports_runtime_or_opens_native(self):
        before = set(sys.modules)
        with (
            patch.object(c, "execute", side_effect=AssertionError("execute")),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(c.main([]), 0)
        self.assertEqual(
            json.loads(output.getvalue())["status"], "dry_source_only_no_runtime_import"
        )
        self.assertFalse(set(c.MODULES).intersection(set(sys.modules) - before))

    def test_plan_metadata_is_exact_without_statting_or_reading_binary(self):
        with patch.object(c, "file_stat", side_effect=AssertionError("binary stat")):
            plan = self.selected(self.plan())
        self.assertEqual(len(plan["native_files"]), 9)
        self.assertEqual(len(plan["runtime_sources"]), 13)
        for mutate in (
            lambda p: p.update(whole_seconds=601),
            lambda p: p["module_origins"].update(torch="/FAKE-other/torch/__init__.py"),
            lambda p: p["native_files"][0].update(path="/FAKE/model.safetensors"),
            lambda p: p["native_files"][0].update(expected_sha256=None),
            lambda p: p["runtime_sources"].pop(),
        ):
            plan = self.plan()
            mutate(plan)
            with self.assertRaises(ValueError):
                self.selected(plan)

    def test_wrong_plan_sha_precedes_owner_import_and_native_access(self):
        with (
            patch.object(c, "load_guard", side_effect=AssertionError("owner")),
            self.assertRaisesRegex(ValueError, "plan SHA"),
        ):
            c.execute(types.SimpleNamespace(plan=HERE / "plan.json", plan_sha256="0" * 64))

    def row(self, path, expected=None):
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "stat": c.file_stat(path),
            "expected_sha256": expected,
        }

    def test_chunk_hash_counts_exact_bytes_and_checks_historical_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.so"
            blob = b"synthetic-native-fixture" * 20
            path.write_bytes(blob)
            checkpoint = Mock()
            with patch.object(c, "CHUNK", 17):
                row = c.hash_native(
                    self.row(path, hashlib.sha256(blob).hexdigest()), c.time.monotonic(), checkpoint
                )
            self.assertEqual(row["sha256"], hashlib.sha256(blob).hexdigest())
            self.assertEqual(row["size_bytes"], len(blob))
            self.assertGreater(checkpoint.call_count, 1)
            with self.assertRaisesRegex(ValueError, "content SHA differs"):
                c.hash_native(self.row(path, "0" * 64), c.time.monotonic(), lambda: None)

    def test_hash_deadline_owner_refusal_and_midread_stat_change_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.so"
            path.write_bytes(b"synthetic fixture")
            row = self.row(path)
            with (
                patch.object(c.time, "monotonic", return_value=601),
                self.assertRaisesRegex(ValueError, "600s"),
            ):
                c.hash_native(row, 0, lambda: None)
            with self.assertRaisesRegex(ValueError, "fake owner refusal"):
                c.hash_native(
                    row, c.time.monotonic(), Mock(side_effect=ValueError("fake owner refusal"))
                )
            with self.assertRaisesRegex(ValueError, "identity remains stable|stat drift"):
                c.hash_native(
                    row, c.time.monotonic(), lambda: path.write_bytes(b"different fixture")
                )

    def test_owned_admission_refuses_before_native_hash_or_runtime_import(self):
        plan = self.plan()
        fake_python = Path("/FAKE/source-only/python")
        plan["logical_interpreter"] = str(fake_python)
        plan["interpreter"] = {"path": str(fake_python)}
        with (
            patch.object(c, "load_plan", return_value=plan),
            patch.dict(os.environ, c.ENV),
            patch.object(c, "PYTHON", fake_python),
            patch.object(sys, "executable", str(fake_python)),
            patch.object(c, "check_stat", side_effect=lambda _row: None),
            patch.object(c, "load_guard", return_value=types.SimpleNamespace()),
            patch.object(
                c,
                "Admission",
                return_value=types.SimpleNamespace(
                    check=Mock(side_effect=ValueError("fresh initial floor refused"))
                ),
            ),
            patch.object(c, "hash_native", side_effect=AssertionError("native")) as native,
            patch.object(c, "import_runtime", side_effect=AssertionError("runtime")) as runtime,
            self.assertRaisesRegex(ValueError, "fresh initial floor refused"),
        ):
            c.execute(
                types.SimpleNamespace(
                    plan=HERE / "plan.json",
                    plan_sha256="FAKE",
                    expected_entry_sha256=c.digest(c.small(c.__file__)),
                )
            )
        native.assert_not_called()
        runtime.assert_not_called()

    def test_preimport_origin_mismatch_refuses_without_runtime_import(self):
        with (
            patch.object(
                c.importlib.util,
                "find_spec",
                return_value=types.SimpleNamespace(origin="/FAKE-other/__init__.py"),
            ),
            patch.object(
                c.importlib, "import_module", side_effect=AssertionError("runtime")
            ) as runtime,
            self.assertRaisesRegex(ValueError, "Pre-import selected module origin"),
        ):
            c.import_runtime(self.plan())
        runtime.assert_not_called()

    def test_fake_runtime_records_all_five_versions_and_three_native_origins(self):
        plan = self.plan()
        modules = {
            name: types.SimpleNamespace(__file__=path, __version__=plan["versions"][name])
            for name, path in plan["module_origins"].items()
        }
        torch = modules["torch"]
        torch.set_num_threads = Mock()
        torch.set_num_interop_threads = Mock()
        torch.get_num_threads = lambda: 4
        torch.get_num_interop_threads = lambda: 1
        torch.cuda = types.SimpleNamespace(is_initialized=lambda: False)
        extensions = {
            name: types.SimpleNamespace(__file__=path) for name, path in c.EXTENSIONS.items()
        }
        with (
            patch.object(sys, "path", sys.path.copy()),
            patch.object(
                c.importlib.util,
                "find_spec",
                side_effect=lambda name: types.SimpleNamespace(origin=plan["module_origins"][name]),
            ),
            patch.object(
                c.importlib,
                "import_module",
                side_effect=lambda name: {**modules, **extensions}[name],
            ),
        ):
            identities, observed, actual = c.import_runtime(plan)
        self.assertIs(actual, torch)
        self.assertEqual(observed, c.EXTENSIONS)
        self.assertEqual(set(identities), set(c.MODULES))
        self.assertEqual(identities["torch"]["version"], "2.13.0+cu130")
        torch.set_num_threads.assert_called_once_with(4)
        self.assertFalse(set(c.MODULES).intersection(sys.modules))

    def test_maps_observe_only_self_and_keep_stat_identity_without_binary_hash(self):
        own = Path("/proc") / str(os.getpid()) / "maps"
        with (
            patch.object(
                c, "small", return_value=b"1000-2000 r-xp 0 01:02 3 /FAKE/libx.so\n"
            ) as read,
            patch.object(c, "file_stat", return_value={"FAKE": "stat only"}) as observed,
        ):
            row = c.mapped_native()
        read.assert_called_once_with(own)
        observed.assert_called_once_with("/FAKE/libx.so")
        self.assertEqual(row, [{"path": "/FAKE/libx.so", "stat": {"FAKE": "stat only"}}])

    def test_unchanged_owner_source_has_two_gib_cpu4_and_no_stop_call(self):
        guard = c.load_guard(self.plan())
        selected = guard.selection(
            {"nonce": "FAKE", "boot_id": "FAKE", "runtime_seconds": 7200},
            types.SimpleNamespace(Selection=lambda *args: args),
        )
        self.assertEqual(selected, (c.UNIT, "FAKE", "FAKE", 2147483648, 7200, 3))
        self.assertEqual(guard.UNIT, c.UNIT)
        self.assertEqual(guard.CPU_CORES, {16, 17, 18, 19})
        tree = ast.parse((HERE / "catalog.py").read_bytes())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(
                    node.func.attr,
                    {"from_pretrained", "generate", "kill", "terminate", "close_unit"},
                )
        self.assertFalse(set(c.MODULES).intersection(sys.modules))

    def test_plan_rejects_adjacent_cap_incomplete_floor_and_mixed_owner(self):
        for mutate in (
            lambda p: p["owner_policy"].update(memory_max_bytes=3 * 1024**3),
            lambda p: p["owner_policy"].update(initial_available_bytes=26119172916 + 1024**3),
            lambda p: p["owner_policy"].update(initial_available_bytes=26119172916 + 3 * 1024**3),
            lambda p: p["owner_policy"].update(runtime_available_bytes=26119172915),
            lambda p: p["owner_policy"].update(owner_runtime_seconds=10800),
            lambda p: p["owner_policy"].update(profile="llama-cpp-cpu4-3g-long-command-owned-v1"),
            lambda p: p["owner_policy"].update(
                unit="jev-d3-llama-cpp-cpu4-3g-long-20261023-05.service"
            ),
            lambda p: p["owner_source"].update(
                sha256="81a492bb00a430c8728e0fcafe94a4220961312e521e8c0d7863c7f601c8ef2e"
            ),
        ):
            plan = self.plan()
            mutate(plan)
            with self.assertRaises(ValueError):
                self.selected(plan)

    def test_exact_owner03_fake_host_requires_entire_two_gib_initial_headroom(self):
        guard = c.load_guard(self.plan())
        reserve = 26119172916
        row = {
            "boot_id": "FAKE-boot",
            "mem_total_bytes": reserve * 5 - 4,
            "mem_available_bytes": reserve + 2 * 1024**3,
            "comfy_state": "http_idle",
            "comfy_running": 0,
            "comfy_pending": 0,
        }
        observer = types.SimpleNamespace(observe=lambda _boot: row.copy())
        admitted = guard.host(observer, "FAKE-boot", initial=True)
        self.assertEqual(admitted["required_available_bytes"], 28266656564)
        for available in (28266656563, reserve + 1024**3, reserve):
            row["mem_available_bytes"] = available
            with self.assertRaisesRegex(ValueError, "reserve"):
                guard.host(observer, "FAKE-boot", initial=True)
        row["mem_available_bytes"] = reserve
        self.assertEqual(guard.host(observer, "FAKE-boot")["required_available_bytes"], reserve)
        row["mem_available_bytes"] = reserve - 1
        with self.assertRaisesRegex(ValueError, "reserve"):
            guard.host(observer, "FAKE-boot")

    def test_mixed_loaded_owner_identity_refuses_without_host_or_native_access(self):
        fake = b'UNIT="jev-d3-llama-cpp-cpu4-3g-long-20261023-05.service"\nPROFILE="llama-cpp-cpu4-3g-long-command-owned-v1"\nCPU_CORES=frozenset({16,17,18,19})\n'
        with (
            patch.object(c, "check_ref", return_value=fake),
            self.assertRaisesRegex(ValueError, "owner03 resource profile"),
        ):
            c.load_guard(self.plan())


if __name__ == "__main__":
    unittest.main(verbosity=2)
