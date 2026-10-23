"""Stdlib-only boundary tests. These never import Torch or mmap a tensor."""

import ast
import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ENTRY = Path(__file__).with_name("fixture.py")
spec = importlib.util.spec_from_file_location("tiny_cpu_fixture_static", ENTRY)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "ACCELERATE_USE_CPU": "true",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "USE_TF": "0",
    "USE_FLAX": "0",
}


class StaticBoundary(unittest.TestCase):
    def manifest(self, root):
        blob = ENTRY.read_bytes()
        entry_sha = fixture.sha(blob)
        value = {
            "profile": fixture.PROFILE,
            "sources": {
                "entry": {"path": str(ENTRY), "sha256": entry_sha, "size_bytes": len(blob)}
            },
            "versions": {},
            "binary_stats": [],
        }
        path = root / "selection.json"
        path.write_text(json.dumps(value))
        return path, fixture.sha(path.read_bytes()), entry_sha

    def argv(self, root):
        path, digest, entry_sha = self.manifest(root)
        return [
            "--run",
            "--profile",
            fixture.PROFILE,
            "--selection",
            str(path),
            "--selection-sha256",
            digest,
            "--entry-sha256",
            entry_sha,
            "--output",
            str(root / "fresh"),
        ]

    def test_dry_never_calls_actor_or_creates_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "unused"
            with (
                patch.object(fixture, "run_torch", side_effect=AssertionError("actor forbidden")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(fixture.main(["--output", str(target)]), 0)
            self.assertFalse(target.exists())

    def test_heavy_imports_only_below_explicit_actor_function(self):
        tree = ast.parse(ENTRY.read_text())
        top = [
            name.name.split(".")[0]
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for name in node.names
        ]
        self.assertFalse({"torch", "accelerate", "peft", "safetensors"} & set(top))
        actor = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_torch"
        )
        calls = [
            node
            for node in ast.walk(actor)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "cpu_offload"
        ]
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertTrue(
                any(
                    isinstance(node, ast.Try)
                    and call in list(ast.walk(ast.Module(body=node.body, type_ignores=[])))
                    and node.finalbody
                    for node in ast.walk(actor)
                )
            )
            device = next(item.value for item in call.keywords if item.arg == "execution_device")
            self.assertEqual(ast.literal_eval(device), "cpu")

    def test_run_refuses_environment_before_actor(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.argv(Path(temporary))
            with (
                patch.dict(os.environ, {**ENV, "CUDA_VISIBLE_DEVICES": "0"}, clear=True),
                patch.object(fixture, "run_torch", side_effect=AssertionError("actor forbidden")),
            ):
                with self.assertRaisesRegex(ValueError, "empty CUDA"):
                    fixture.main(args)
            self.assertFalse((Path(temporary) / "fresh").exists())

    def test_selection_requires_exact_manifest_and_entry_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, digest, own = self.manifest(Path(temporary))
            fixture.selection(path, digest, own)
            for a, b in (("0" * 64, own), (digest, "0" * 64)):
                with self.assertRaises(ValueError):
                    fixture.selection(path, a, b)

    def test_symlink_and_oversize_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "link").symlink_to(ENTRY)
            with self.assertRaisesRegex(ValueError, "Ordinary"):
                fixture.read(root / "link")
            with patch.object(fixture, "MAX_FILE", 4):
                with self.assertRaisesRegex(ValueError, "bound"):
                    fixture.read(ENTRY)

    def test_existing_output_refused_before_actor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.argv(root)
            (root / "fresh").mkdir()
            with (
                patch.dict(os.environ, ENV, clear=True),
                patch.object(fixture, "run_torch", side_effect=AssertionError("actor forbidden")),
            ):
                with self.assertRaisesRegex(ValueError, "Fresh"):
                    fixture.main(args)

    def test_stub_completion_records_final_source_and_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def stub(_output, _selected, result):
                result["mechanism_checks_completed"] = True
                result["STATIC_STUB_NO_MECHANISM_QUALIFICATION"] = True

            with (
                patch.dict(os.environ, ENV, clear=True),
                patch.object(fixture, "run_torch", side_effect=stub),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(fixture.main(self.argv(root)), 0)
            receipt = json.loads((root / "fresh" / "receipt.json").read_text())
            self.assertTrue(receipt["source_integrity_final"])
            self.assertTrue(receipt["passed"])
            self.assertEqual(receipt["identity"], receipt["identity_after"])
            self.assertFalse(receipt["large_model_runtime_or_quality_qualified"])

    def test_final_source_refusal_preserves_actor_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.argv(root)
            selected = fixture.selection(Path(args[4]), args[6], args[8])
            with (
                patch.dict(os.environ, ENV, clear=True),
                patch.object(
                    fixture,
                    "selection",
                    side_effect=[selected, ValueError("controlled final-source refusal")],
                ),
                patch.object(
                    fixture,
                    "run_torch",
                    side_effect=RuntimeError("controlled original body failure"),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(fixture.main(args), 1)
            receipt = json.loads((root / "fresh" / "receipt.json").read_text())
            self.assertEqual(receipt["error"]["message"], "controlled original body failure")
            self.assertIn("controlled final-source refusal", receipt["error"]["notes"][0])
            self.assertFalse(receipt["source_integrity_final"])

    def test_final_source_refusal_after_stub_success_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.argv(root)
            selected = fixture.selection(Path(args[4]), args[6], args[8])

            def stub(_output, _selected, result):
                result["mechanism_checks_completed"] = True

            with (
                patch.dict(os.environ, ENV, clear=True),
                patch.object(
                    fixture,
                    "selection",
                    side_effect=[selected, ValueError("controlled final-source refusal")],
                ),
                patch.object(fixture, "run_torch", side_effect=stub),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(fixture.main(args), 1)
            self.assertFalse(json.loads((root / "fresh" / "receipt.json").read_text())["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
