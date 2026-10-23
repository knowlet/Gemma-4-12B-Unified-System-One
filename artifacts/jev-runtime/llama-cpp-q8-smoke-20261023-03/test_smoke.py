"""Stdlib fake RPC/schema tests: no native child, Torch, processor or weight reads."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE))
prep = importlib.import_module("prepare_smoke")
run = importlib.import_module("run_smoke")
common = importlib.import_module("smoke_common")


def payload(long=True):
    count = 1665 if long else 65
    return {
        "case_id": "synthetic-cache-long-v1" if long else "synthetic-cache-warm-v1",
        "question_ids": ["synthetic-early", "synthetic-late"] if long else ["synthetic-warm"],
        "slots": [64, 1664] if long else [64],
        "nopts": [2, 2] if long else [2],
        "letters": list(range(100, 152)),
        "source_input_token_count": count,
        "parts": [{"type": "text", "tokens": [88] * count}],
    }


def response(p=None, context=2048, full=True):
    p = p or payload()
    return {
        "status": "ok",
        "case_id": p["case_id"],
        "question_ids": p["question_ids"],
        "slots": p["slots"],
        "runtime_slots": p["slots"],
        "native_token_count": p["source_input_token_count"],
        "raw_logits": [[1.0, -1.0] for _ in p["nopts"]],
        "latency_ms": 123,
        "runtime": {
            "engine": "llama.cpp",
            "llama_cpp_revision": "184f6536024668e8908ab8c86033d56708e62b40",
            "context_size": context,
            "batch_size": 512,
            "microbatch_size": 512,
            "gpu_layers_requested": 0,
            "swa_full": full,
            "cpu_mmap_only": True,
            "decision_passes": 1,
            "autoregressive_tokens": 0,
            "cache_scope": "one_request_cleared_between_requests",
            "logits_semantics": "post_native_softcap_pre_temperature",
        },
    }


class FakePipe:
    def __init__(self, fd):
        self.fd = fd
        self.close = Mock()

    def fileno(self):
        return self.fd


class FakeSelector:
    def __init__(self, events):
        self.events, self.registered = list(events), {}

    def register(self, pipe, mode, data):
        self.registered[pipe] = data

    def unregister(self, pipe):
        return self.registered.pop(pipe)

    def select(self, timeout):
        return [(self.events.pop(0), 1)] if self.events else []


class Tests(unittest.TestCase):
    def test_default_dry_never_prepares_or_starts_native(self):
        for module in (prep, run):
            with (
                patch.object(module, "run", side_effect=AssertionError("run")),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(module.main([]), 0)
                self.assertEqual(json.loads(output.getvalue())["mode"], "DRY_SOURCE_ONLY")
        self.assertNotIn("torch", sys.modules)

    def test_wrong_entry_sha_refuses_before_selection_or_framework(self):
        args = types.SimpleNamespace(expected_entry_sha256="0" * 64)
        for module in (prep, run):
            with (
                patch.object(module, "selection", side_effect=AssertionError("selection")),
                self.assertRaisesRegex(ValueError, "entry SHA"),
            ):
                module.run(args)
        self.assertNotIn("torch", sys.modules)

    def test_synthetic_request_padding_only_in_late_question_and_no_gold(self):
        first, long = (
            prep.request_data(long=True, padding=0),
            prep.request_data(long=True, padding=200),
        )
        self.assertEqual(first["state"], long["state"])
        self.assertEqual(first["questions"][0], long["questions"][0])
        self.assertEqual(len(long["questions"]), 2)
        self.assertEqual(
            [q["id"] for q in long["questions"]], ["synthetic-early", "synthetic-late"]
        )
        self.assertNotIn("gold", json.dumps(long))
        self.assertLess(len(long["questions"][1]["instructions"]), 16000)

    def test_payload_requires_early_late_geometry_and_exact_counts(self):
        self.assertIsNotNone(common.payload_check(payload(), long=True))
        for mutate in (
            lambda p: p["slots"].__setitem__(1, 1536),
            lambda p: p.__setitem__("source_input_token_count", 1664),
            lambda p: p["question_ids"].reverse(),
            lambda p: p["letters"].__setitem__(1, p["letters"][0]),
            lambda p: p.__setitem__("gold", [0, 0]),
        ):
            p = payload()
            mutate(p)
            with self.assertRaises(ValueError):
                common.payload_check(p, long=True)

    def test_response_exact_join_and_actual_context(self):
        for ctx, full in ((2048, True), (2048, False), (8192, False)):
            self.assertEqual(
                common.response_check(response(context=ctx, full=full), payload(), ctx, full),
                [[1.0, -1.0]] * 2,
            )
        for key, value in (
            ("case_id", "wrong"),
            ("slots", [65, 1664]),
            ("native_token_count", 1664),
            ("runtime_slots", [64, 1663]),
        ):
            r = response()
            r[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                common.response_check(r, payload(), 2048, True)
        r = response(context=2048, full=False)
        with self.assertRaisesRegex(ValueError, "runtime contract"):
            common.response_check(r, payload(), 8192, False)

    def test_response_rejects_missing_rows_nonfinite_boolean_and_narrow_rows(self):
        for rows in (
            [],
            [[1.0, 2.0]],
            [[1.0], [1.0, 2.0]],
            [[True, 0.0], [1.0, 2.0]],
            [[float("nan"), 0.0], [1.0, 2.0]],
        ):
            r = response()
            r["raw_logits"] = rows
            with self.assertRaises(ValueError):
                common.response_check(r, payload(), 2048, True)

    def test_math_threshold_does_not_mask_argmax_changes(self):
        self.assertTrue(common.compare([[1.0, 0.0]], [[1.0001, 0.0]])["passed"])
        self.assertFalse(common.compare([[1.0, 0.0]], [[1.002, 0.0]])["passed"])
        result = common.compare([[0.0001, 0.0]], [[0.0, 0.0001]])
        self.assertLess(result["max_abs_logit_difference"], 0.001)
        self.assertFalse(result["passed"])

    def client(self, directory, events):
        c = run.NativeRPC.__new__(run.NativeRPC)
        c.directory = directory
        c.launch_monotonic = run.time.monotonic()
        c.context, c.full = 2048, True
        c.process = types.SimpleNamespace(
            pid=123,
            stdin=FakePipe(10),
            stdout=FakePipe(11),
            stderr=FakePipe(12),
            poll=Mock(return_value=None),
        )
        c.selector = FakeSelector(
            [
                types.SimpleNamespace(fileobj=getattr(c.process, stream), data=stream)
                for stream in events
            ]
        )
        c.selector.register(c.process.stdout, 1, "stdout")
        c.selector.register(c.process.stderr, 1, "stderr")
        c.stdout, c.rows = bytearray(), []
        c.stderr = io.BytesIO()
        c.stderr_hash = hashlib.sha256()
        c.stderr_seen = c.stderr_saved = 0
        return c

    def test_fragmented_response_and_partial_writes_preserve_exact_wire_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = self.client(root, ["stdin", "stdin", "stdout", "stdout"])
            p = payload(False)
            raw = common.encoded(response(p))
            reads = iter((raw[:12], raw[12:]))
            writes = iter((8, len(common.encoded(p)) - 8))
            with (
                patch.object(common, "ROOT", root),
                patch.object(run.os, "read", side_effect=lambda *a: next(reads)),
                patch.object(run.os, "write", side_effect=lambda *a: next(writes)),
                patch.object(run.subprocess, "Popen", side_effect=AssertionError("actual native")),
            ):
                result = c.score(p, 60, "warm")
            self.assertEqual(result, response(p))
            self.assertEqual(c.rows[0]["response_wire_sha256"], common.sha(raw))
            self.assertEqual((root / "warm-response.raw").read_bytes(), raw)
            self.assertEqual(c.rows[0]["stdin_bytes_sent"], len(common.encoded(p)))

    def test_timeout_keeps_partial_response_and_does_not_kill_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = self.client(root, ["stdin", "stdout"])
            times = iter((0, 0, 0.1, 600, 600))
            with (
                patch.object(common, "ROOT", root),
                patch.object(run.time, "monotonic", side_effect=lambda: next(times)),
                patch.object(run.os, "write", side_effect=lambda fd, data: len(data)),
                patch.object(run.os, "read", return_value=b'{"status":'),
                self.assertRaises(TimeoutError) as captured,
            ):
                c.score(payload(), 600, "long")
            self.assertEqual((root / "long-response.partial").read_bytes(), b'{"status":')
            self.assertFalse(c.rows[0]["complete_response"])
            self.assertFalse(c.rows[0]["meets20s_observed"])
            self.assertEqual(captured.exception.evidence["deadline_seconds"], 600)
            self.assertEqual(c.process.poll(), None)

    def test_warm_launch_and_proc_delay_counts_against_twenty_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = self.client(root, ["stdin", "stdout"])
            c.launch_monotonic = 0.0
            # The native launch and initial /proc observation consumed25s before submission.
            times = iter((25.0, 25.0, 25.1, 25.2, 25.2))
            raw = common.encoded(response(payload(False)))
            with (
                patch.object(common, "ROOT", root),
                patch.object(run.time, "monotonic", side_effect=lambda: next(times)),
                patch.object(run.os, "write", side_effect=lambda fd, data: len(data)),
                patch.object(run.os, "read", return_value=raw),
            ):
                c.score(payload(False), 60, "warm")
            row = c.rows[0]
            self.assertEqual(row["deadline_monotonic"], 60.0)
            self.assertEqual(row["timing_started_monotonic"], 0.0)
            self.assertAlmostEqual(row["submission_elapsed_seconds"], 0.2)
            self.assertAlmostEqual(row["elapsed_seconds"], 25.2)
            self.assertTrue(row["response_contract_validated"])
            self.assertFalse(row["meets20s_observed"])

    def test_warm_already_expired_at_submission_cannot_reset_sixty_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = self.client(root, [])
            c.launch_monotonic = 0.0
            times = iter((61.0, 61.0, 61.0))
            with (
                patch.object(common, "ROOT", root),
                patch.object(run.time, "monotonic", side_effect=lambda: next(times)),
                patch.object(run.os, "write", side_effect=AssertionError("expired request write")),
                self.assertRaises(TimeoutError),
            ):
                c.score(payload(False), 60, "warm")
            row = c.rows[0]
            self.assertEqual(row["deadline_monotonic"], 60.0)
            self.assertEqual(row["stdin_bytes_sent"], 0)
            self.assertFalse(row["meets20s_observed"])
            self.assertEqual(row["elapsed_seconds"], 61.0)

    def test_long_rpc_deadline_remains_from_submission_after_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = self.client(root, ["stdin", "stdout"])
            c.launch_monotonic = 0.0
            times = iter((100.0, 100.0, 100.1, 100.2, 100.2))
            with (
                patch.object(common, "ROOT", root),
                patch.object(run.time, "monotonic", side_effect=lambda: next(times)),
                patch.object(run.os, "write", side_effect=lambda fd, data: len(data)),
                patch.object(run.os, "read", return_value=common.encoded(response())),
            ):
                c.score(payload(), 600, "long")
            row = c.rows[0]
            self.assertEqual(row["deadline_monotonic"], 700.0)
            self.assertEqual(row["timing_started_monotonic"], 100.0)
            self.assertAlmostEqual(row["elapsed_seconds"], 0.2)
            self.assertTrue(row["meets20s_observed"])

    def test_valid_reply_crossing_deadline_during_validation_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = self.client(root, ["stdin", "stdout"])
            c.launch_monotonic = 0.0
            # The read becomes ready before60, but parse/validation finish after60.
            times = iter((0.0, 59.8, 59.9, 60.1, 60.1))
            with (
                patch.object(common, "ROOT", root),
                patch.object(run.time, "monotonic", side_effect=lambda: next(times)),
                patch.object(run.os, "write", side_effect=lambda fd, data: len(data)),
                patch.object(run.os, "read", return_value=common.encoded(response(payload(False)))),
                self.assertRaisesRegex(TimeoutError, "validated response exceeded"),
            ):
                c.score(payload(False), 60, "warm")
            row = c.rows[0]
            self.assertTrue(row["complete_response"])
            self.assertFalse(row["response_contract_validated"])
            self.assertFalse(row["meets20s_observed"])
            self.assertEqual(row["elapsed_seconds"], 60.1)
            self.assertTrue((root / "warm-response.raw").exists())

    def test_accepted_reply_uses_validation_completion_not_later_receipt_housekeeping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = self.client(root, ["stdin", "stdout"])
            c.launch_monotonic = 0.0
            times = iter((50.0, 59.7, 59.8, 59.9, 61.0))
            with (
                patch.object(common, "ROOT", root),
                patch.object(run.time, "monotonic", side_effect=lambda: next(times)),
                patch.object(run.os, "write", side_effect=lambda fd, data: len(data)),
                patch.object(run.os, "read", return_value=common.encoded(response(payload(False)))),
            ):
                c.score(payload(False), 60, "warm")
            row = c.rows[0]
            self.assertTrue(row["response_contract_validated"])
            self.assertEqual(row["validated_response_monotonic"], 59.9)
            self.assertEqual(row["rpc_finished_monotonic"], 59.9)
            self.assertEqual(row["elapsed_seconds"], 59.9)
            self.assertAlmostEqual(row["local_finalization_seconds"], 1.1)
            self.assertFalse(row["meets20s_observed"])

    def test_model_VMA_metadata_only_selected_path_and_bounded_retention(self):
        selected = "/FAKE/public-model.gguf"
        rows = [
            f"{4096 * n:x}-{4096 * (n + 1):x} r--p {4096 * n:08x} 01:02 123 {selected}\n"
            for n in range(65)
        ]
        rows += [
            "a0000-b0000 r--p 00000000 01:02 456 /FAKE/other-model.gguf\n",
            "b0000-c0000 r-xp 00000000 01:02 789 /FAKE/libllama.so.0\n",
            "c0000-d0000 rw-p 00000000 00:00 0 [heap]\n",
        ]
        libraries, model = run.map_metadata("".join(rows).encode(), selected)
        self.assertEqual(libraries, {"/FAKE/libllama.so.0"})
        self.assertEqual(model["observed_VMA_count"], 65)
        self.assertEqual(model["observed_virtual_bytes"], 65 * 4096)
        self.assertEqual(len(model["retained_VMAs"]), 64)
        self.assertTrue(model["truncated"])
        self.assertEqual({r["path"] for r in model["retained_VMAs"]}, {selected})
        self.assertIn("not resident memory", model["scope"])

    def test_malformed_response_retains_raw_and_original_parse_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = self.client(root, ["stdin", "stdout"])
            with (
                patch.object(common, "ROOT", root),
                patch.object(run.os, "write", side_effect=lambda fd, data: len(data)),
                patch.object(run.os, "read", return_value=b"NOT JSON\n"),
                self.assertRaises(json.JSONDecodeError),
            ):
                c.score(payload(False), 60, "warm")
            self.assertEqual((root / "warm-response.raw").read_bytes(), b"NOT JSON\n")
            self.assertEqual(c.rows[0]["primary_error"]["type"], "JSONDecodeError")

    def test_fast_invalid_native_reply_is_not_twenty_second_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = self.client(root, ["stdin", "stdout"])
            invalid = response(payload(False))
            invalid["case_id"] = "WRONG-CASE"
            with (
                patch.object(common, "ROOT", root),
                patch.object(run.os, "write", side_effect=lambda fd, data: len(data)),
                patch.object(run.os, "read", return_value=common.encoded(invalid)),
                self.assertRaisesRegex(ValueError, "case_id"),
            ):
                c.score(payload(False), 60, "warm")
            self.assertTrue(c.rows[0]["elapsed_le20s"])
            self.assertFalse(c.rows[0]["response_contract_validated"])
            self.assertFalse(c.rows[0]["meets20s_observed"])

    def test_exchange_save_failure_does_not_replace_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = self.client(root, [])
            times = iter((0, 600, 600))
            original_save = run.save

            def fake_save(path, value):
                if path.name.endswith("exchange.json"):
                    raise OSError("fake disk failure")
                return original_save(path, value)

            with (
                patch.object(common, "ROOT", root),
                patch.object(run, "save", side_effect=fake_save),
                patch.object(run.time, "monotonic", side_effect=lambda: next(times)),
                self.assertRaises(TimeoutError) as captured,
            ):
                c.score(payload(), 600, "long")
            self.assertEqual(
                captured.exception.evidence["secondary_errors"][0]["phase"], "exchange_save"
            )

    def test_stderr_prefix_cap_keeps_draining_and_hashes_all_observed(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.client(Path(directory), [])
            block = b"x" * (run.STDERR_LIMIT + 5)
            event = types.SimpleNamespace(fileobj=c.process.stderr, data="stderr")
            with patch.object(run.os, "read", return_value=block):
                c.consume(event)
            self.assertEqual(len(c.stderr.getvalue()), run.STDERR_LIMIT)
            self.assertEqual(c.stderr_seen, len(block))
            self.assertEqual(c.stderr_hash.hexdigest(), hashlib.sha256(block).hexdigest())

    def test_native_start_refusal_closes_owned_setup_and_preserves_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selector = types.SimpleNamespace(close=Mock())
            opened = []
            original_open = Path.open

            def track_open(path, *args, **kwargs):
                stream = original_open(path, *args, **kwargs)
                opened.append(stream)
                return stream

            primary = OSError("simulated exec refusal")
            with (
                patch.object(common, "ROOT", root),
                patch.object(run.selectors, "DefaultSelector", return_value=selector),
                patch.object(run.subprocess, "Popen", side_effect=primary),
                patch.object(Path, "open", new=track_open),
                self.assertRaises(OSError) as captured,
            ):
                run.NativeRPC(["FAKE-NOT-EXECUTED"], root, 2048, True)
            self.assertIs(captured.exception, primary)
            self.assertTrue(opened and all(stream.closed for stream in opened))
            selector.close.assert_called_once()

    def test_source_has_no_LM_loader_generate_or_manual_child_kill(self):
        for path in (HERE / "prepare_smoke.py", HERE / "run_smoke.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, {"generate", "terminate", "kill", "killpg"})
                    if node.func.attr == "from_pretrained":
                        self.assertIsInstance(node.func.value, ast.Name)
                        self.assertEqual(node.func.value.id, "AutoProcessor")
        tree = ast.parse((HERE / "run_smoke.py").read_text())
        imports = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        ]
        self.assertFalse(
            any(name.startswith(("torch", "transformers", "numpy", "s1")) for name in imports)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
