"""Pure stdlib temporary fixtures; no Torch, real tensor payload or actor."""

import ast
import contextlib
import importlib.util
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ENTRY = Path(__file__).with_name("fixture.py").absolute()
spec = importlib.util.spec_from_file_location("real_layer_fixture_static", ENTRY)
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
    "JEV_LAYER_DEADLINE_MONOTONIC": str(time.monotonic() + 25),
}


class Boundaries(unittest.TestCase):
    def manifest(self, root):
        header = root / "impossible-live-header.json"
        header.write_text("{}")
        paths = {"entry": ENTRY, "header_selection": header}
        sources = {
            key: {
                "path": str(path),
                "sha256": fixture.sha(path.read_bytes()),
                "size_bytes": path.stat().st_size,
            }
            for key, path in paths.items()
        }
        value = {"profile": fixture.PROFILE, "sources": sources, "versions": {}, "binary_stats": []}
        path = root / "selection.json"
        path.write_text(json.dumps(value))
        return value, [
            "--run",
            "--profile",
            fixture.PROFILE,
            "--selection",
            str(path),
            "--selection-sha256",
            fixture.sha(path.read_bytes()),
            "--entry-sha256",
            sources["entry"]["sha256"],
            "--output",
            str(root / "fresh"),
        ]

    def test_dry_is_no_actor_or_writes(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name) / "fresh"
            with (
                patch.object(fixture, "run_torch", side_effect=AssertionError("forbidden")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(fixture.main(["--output", str(output)]), 0)
            self.assertFalse(output.exists())

    def test_heavy_imports_not_top_level_and_no_full_model(self):
        tree = ast.parse(ENTRY.read_text())
        imports = [
            name.name.split(".")[0]
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for name in node.names
        ]
        self.assertFalse(set(imports) & {"torch", "safetensors", "transformers"})
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                self.assertNotIn(
                    node.attr,
                    {
                        "from_pretrained",
                        "clone",
                        "Gemma4UnifiedTextModel",
                        "Gemma4UnifiedForConditionalGeneration",
                    },
                )

    def test_invalid_environment_refuses_before_actor(self):
        with tempfile.TemporaryDirectory() as name:
            _, args = self.manifest(Path(name))
            with (
                patch.dict(os.environ, {**ENV, "CUDA_VISIBLE_DEVICES": "0"}, clear=True),
                patch.object(fixture, "run_torch", side_effect=AssertionError("forbidden")),
            ):
                with self.assertRaisesRegex(ValueError, "CPU-only"):
                    fixture.main(args)

    def test_missing_deadline_refuses_before_actor(self):
        with tempfile.TemporaryDirectory() as name:
            _, args = self.manifest(Path(name))
            env = {k: v for k, v in ENV.items() if k != "JEV_LAYER_DEADLINE_MONOTONIC"}
            with (
                patch.dict(os.environ, env, clear=True),
                patch.object(fixture, "run_torch", side_effect=AssertionError("forbidden")),
            ):
                with self.assertRaises(KeyError):
                    fixture.main(args)

    def test_manifest_and_entry_sha_refusals(self):
        with tempfile.TemporaryDirectory() as name:
            _, args = self.manifest(Path(name))
            for manifest_sha, own_sha in (("0" * 64, args[8]), (args[6], "0" * 64)):
                with self.assertRaises(ValueError):
                    fixture.selection(Path(args[4]), manifest_sha, own_sha)

    def test_symlink_output_or_source_refused(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "link").symlink_to(ENTRY)
            with self.assertRaisesRegex(ValueError, "Ordinary"):
                fixture.read(root / "link")

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, args = self.manifest(root)
            (root / "fresh").mkdir()
            with (
                patch.dict(os.environ, ENV, clear=True),
                patch.object(
                    fixture, "checked_header", return_value=Path("/tmp/IMPOSSIBLE_NO_ACTOR")
                ),
            ):
                with self.assertRaisesRegex(ValueError, "Fresh"):
                    fixture.main(args)

    def main_stub(self, mode, final_refusal=False):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            selected, args = self.manifest(root)

            def actor(_out, _selected, result, *_args):
                result["STATIC_STUB_NO_TORCH_OR_NUMERICAL_QUALIFICATION"] = True
                if mode == "success":
                    result.update(
                        completed_cases=list(fixture.PLANNED_CASES),
                        mechanism_checks_completed=True,
                        attention_interface_restored=True,
                    )
                elif mode == "partial":
                    result["completed_cases"] = fixture.PLANNED_CASES[:1]
                    raise fixture.DeadlinePartial("controlled partial")
                else:
                    raise RuntimeError("controlled primary")

            choose = (
                patch.object(
                    fixture,
                    "selection",
                    side_effect=[selected, ValueError("controlled final source refusal")],
                )
                if final_refusal
                else contextlib.nullcontext()
            )
            with (
                patch.dict(os.environ, ENV, clear=True),
                patch.object(
                    fixture, "checked_header", return_value=Path("/tmp/IMPOSSIBLE_NO_ACTOR")
                ),
                patch.object(fixture, "run_torch", side_effect=actor),
                choose,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                rc = fixture.main(args)
            return rc, json.loads((root / "fresh" / "receipt.json").read_text())

    def test_all_four_stub_cases_needed_for_success(self):
        rc, receipt = self.main_stub("success")
        self.assertEqual(rc, 0)
        self.assertTrue(receipt["passed"] and receipt["completed"])
        self.assertEqual(len(receipt["completed_cases"]), 4)

    def test_partial_exit2_preserves_case_and_not_completed(self):
        rc, receipt = self.main_stub("partial")
        self.assertEqual(rc, 2)
        self.assertEqual(len(receipt["completed_cases"]), 1)
        self.assertFalse(
            receipt["passed"] or receipt["completed"] or receipt["mechanism_checks_completed"]
        )
        self.assertEqual(receipt["status"], "partial_deadline")

    def test_final_source_failure_preserves_original_body_error(self):
        rc, receipt = self.main_stub("failure", True)
        self.assertEqual(rc, 1)
        self.assertEqual(receipt["error"]["message"], "controlled primary")
        self.assertIn("controlled final source refusal", receipt["error"]["notes"][0])

    def test_partial_then_source_failure_is_failure(self):
        rc, receipt = self.main_stub("partial", True)
        self.assertEqual(rc, 1)
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("Earlier deadline partial", receipt["error"]["notes"][0])

    def synthetic_header(self, root):
        p = root / "SYNTHETIC_HEADER_NOT_A_MODEL.txt"
        meta = {"synthetic.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}
        body = json.dumps(meta).encode()
        prefix = len(body).to_bytes(8, "little")
        p.write_bytes(prefix + body + b"XX")
        descriptor = {
            "base": {"canonical_blob_path": str(p), "strict_stat_before": fixture.strict_stat(p)},
            "header": {
                "json_bytes": len(body),
                "json_sha256": fixture.sha(body),
                "prefix_and_json_sha256": fixture.sha(prefix + body),
            },
            "layers": [
                {
                    "states": [
                        {
                            "checkpoint_key": "synthetic.weight",
                            "dtype": "BF16",
                            "shape": [1],
                            "payload_offsets_half_open": [0, 2],
                        }
                    ]
                }
            ],
        }
        return p, descriptor

    def test_header_reader_requests_no_payload_bytes(self):
        with tempfile.TemporaryDirectory() as name:
            p, descriptor = self.synthetic_header(Path(name))
            calls, original = [], os.pread

            def pread(fd, count, offset):
                calls.append((offset, count))
                return original(fd, count, offset)

            with patch.object(fixture.os, "pread", side_effect=pread):
                self.assertEqual(fixture.checked_header(descriptor), p)
            self.assertEqual(calls, [(0, 8), (8, descriptor["header"]["json_bytes"])])

    def test_header_row_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as name:
            _, descriptor = self.synthetic_header(Path(name))
            descriptor["layers"][0]["states"][0]["shape"] = [2]
            with self.assertRaisesRegex(ValueError, "Selected range"):
                fixture.checked_header(descriptor)

    def test_range_hash_reads_only_requested_synthetic_ranges(self):
        with tempfile.TemporaryDirectory() as name:
            p = Path(name) / "SYNTHETIC_BYTES_NOT_MODEL.txt"
            p.write_bytes(b"abcUNSELECTEDxyz")
            rows = [
                {"checkpoint_key": key, "file_offsets_half_open": limits}
                for key, limits in [("a", [0, 3]), ("z", [13, 16])]
            ]
            calls, original, progress = [], os.pread, {"ranges": []}

            def pread(fd, count, offset):
                calls.append((offset, count))
                return original(fd, count, offset)

            with patch.object(fixture.os, "pread", side_effect=pread):
                fixture.range_hashes(p, rows, time.monotonic() + 10, progress)
            self.assertEqual(calls, [(0, 3), (13, 3)])
            self.assertEqual(progress["explicit_selected_payload_hash_bytes"], 6)

    def test_expired_deadline_prevents_range_read(self):
        with tempfile.TemporaryDirectory() as name:
            p = Path(name) / "SYNTHETIC.txt"
            p.write_text("xx")
            with patch.object(fixture.os, "pread", side_effect=AssertionError("no read")):
                with self.assertRaises(fixture.DeadlinePartial):
                    fixture.range_hashes(
                        p,
                        [{"checkpoint_key": "x", "file_offsets_half_open": [0, 2]}],
                        time.monotonic() - 1,
                        {"ranges": []},
                    )

    def test_memory_snapshot_records_breakdown_with_fake_proc_and_cgroup(self):
        def fake(path):
            name = Path(path).name
            if name == "status":
                return b"VmRSS:\t10 kB\nRssAnon:\t6 kB\nRssFile:\t4 kB\n"
            if name == "cgroup":
                return b"0::/IMPOSSIBLE_STATIC_CGROUP\n"
            if name == "memory.current":
                return b"100\n"
            if name == "memory.peak":
                return b"200\n"
            if name == "memory.stat":
                return b"anon 60\nfile 40\n"
            raise AssertionError(name)

        with patch.object(fixture, "read", side_effect=fake):
            value = fixture.memory_snapshot()
        self.assertEqual(value["process"]["RssAnon_bytes"], 6144)
        self.assertEqual(value["own_cgroup"]["memory_peak_bytes"], 200)

    def test_stage_is_exclusive_bounded_json(self):
        with tempfile.TemporaryDirectory() as name:
            result = {"stages": [], "completed_cases": [], "started_monotonic": time.monotonic()}
            with patch.object(fixture, "memory_snapshot", return_value={"STATIC_STUB": True}):
                fixture.stage(Path(name), result, "static-only", {"payload": False})
            data = json.loads(Path(result["stages"][0]["path"]).read_text())
            self.assertIn("elapsed_seconds", data)
            self.assertEqual(data["memory"], {"STATIC_STUB": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
