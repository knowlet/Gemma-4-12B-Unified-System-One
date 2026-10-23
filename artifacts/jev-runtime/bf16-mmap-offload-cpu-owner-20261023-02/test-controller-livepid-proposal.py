"""Pure stdlib tests: no unit, observer HTTP, framework, model or mmap execution."""

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ENTRY = Path(__file__).with_name("controller.py")
SPEC = importlib.util.spec_from_file_location("_cpu_owner_tested", ENTRY)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def process(state="R", start=20):
    return {
        "pid": 123,
        "start_ticks": start,
        "pgid": 123,
        "sid": 123,
        "uid": 1000,
        "cgroup": "/fake-unit",
        "state": state,
    }


class ControllerTests(unittest.TestCase):
    def test_timeout_byte_output_remains_bounded_json_evidence(self):
        error = subprocess.TimeoutExpired(
            ["FAKE-NOT-EXECUTED"], 1, output=b"x" * 40000, stderr=b"bad\xff"
        )
        row = controller.failure(error)
        self.assertEqual(len(row["stdout"]), 32768)
        self.assertEqual(row["stderr"], "bad\ufffd")
        json.dumps(row, allow_nan=False)

    def test_default_dry_imports_no_framework_and_calls_no_authority(self):
        before = set(sys.modules)
        with (
            patch.object(controller, "load_fixed", side_effect=AssertionError("authority")),
            patch.object(controller, "run", side_effect=AssertionError("run")),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(controller.main([]), 0)
        self.assertEqual(json.loads(output.getvalue())["mode"], "DRY_SOURCE_ONLY")
        self.assertFalse(
            any(name == "torch" or name.startswith("torch.") for name in set(sys.modules) - before)
        )

    def test_wrong_entry_sha_refuses_before_authorities_or_actor(self):
        args = types.SimpleNamespace(expected_entry_sha256="0" * 64)
        with (
            patch.object(controller, "load_fixed", side_effect=AssertionError("authority")),
            self.assertRaisesRegex(ValueError, "controller entry SHA"),
        ):
            controller.run(args)

    def test_stable_identity_allows_R_S_but_rejects_changed_start(self):
        left = {"process": process(), "exe": "/fake-python", "argv": ["fake"], "cwd": "/fake"}
        right = {**left, "process": process("S")}
        self.assertTrue(controller.same_identity(left, right))
        self.assertFalse(controller.same_identity(left, {**right, "process": process("S", 21)}))
        self.assertFalse(controller.same_identity(left, {**right, "argv": ["different"]}))

    def test_existing_unit_is_refused_without_command_or_stop(self):
        owner = types.SimpleNamespace(
            status=Mock(return_value={"LoadState": "loaded"}),
            CGROUP_ROOT=Path("/fake-cgroup"),
            command=Mock(side_effect=AssertionError("mutation")),
        )
        with self.assertRaisesRegex(ValueError, "never reuse"):
            controller.ensure_absent(owner, types.SimpleNamespace(cgroup="/fake-unit"))
        owner.command.assert_not_called()

    def test_existing_cgroup_is_refused_even_if_unit_absent(self):
        owner = types.SimpleNamespace(
            status=Mock(return_value={"LoadState": "not-found"}), CGROUP_ROOT=Path("/fake-cgroup")
        )
        with (
            patch.object(Path, "exists", return_value=True),
            self.assertRaisesRegex(ValueError, "never reuse"),
        ):
            controller.ensure_absent(owner, types.SimpleNamespace(cgroup="/fake-unit"))

    def test_original_selection_has_exact_tiny_cap_without_modification(self):
        owner = controller.load_fixed("boundary")
        selected = controller.selection(
            {"nonce": "jev-d3-boundary-" + "a" * 32, "boot_id": "FAKE-BOOT-NOT-LIVE"}, owner
        )
        self.assertEqual(
            (
                selected.unit,
                selected.memory_max_bytes,
                selected.runtime_seconds,
                selected.stop_seconds,
            ),
            (controller.UNIT, 2 * controller.GIB, 30, 3),
        )
        self.assertEqual(selected.properties()["MemorySwapMax"], "0")
        self.assertEqual(selected.properties()["RemainAfterExit"], "yes")

    def test_host_initial_and_runtime_floor_and_busy_evidence(self):
        row = {
            "boot_id": "FAKE",
            "mem_total_bytes": 100 * controller.GIB,
            "mem_available_bytes": 24 * controller.GIB,
            "comfy_state": "http_idle",
            "comfy_running": 0,
            "comfy_pending": 0,
        }
        observer = types.SimpleNamespace(observe=Mock(return_value=row))
        self.assertEqual(controller.host(observer, "FAKE")["reserve_bytes"], 24 * controller.GIB)
        with self.assertRaisesRegex(ValueError, "plus tiny2GiB") as captured:
            controller.host(observer, "FAKE", initial=True)
        self.assertIs(captured.exception.evidence["observation"], row)
        row["mem_available_bytes"] = 30 * controller.GIB
        row["comfy_pending"] = 1
        with self.assertRaisesRegex(ValueError, "idle gate"):
            controller.host(observer, "FAKE")
        self.assertEqual(
            controller.reserve(130 * controller.GIB + 1), (130 * controller.GIB + 5) // 5
        )

    def test_environment_has_only_selected_cpu_overlay(self):
        env = controller.environment({"nonce": "FAKE-NONCE", "output": "/tmp/FAKE-OUTPUT"})
        self.assertEqual(env["PYTHONPATH"], str(controller.ROOT / "artifacts/jev-runtime/overlay"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["ACCELERATE_USE_CPU"], "true")
        self.assertNotIn("LD_PRELOAD", env)
        self.assertNotIn("HF_TOKEN", env)

    def test_finalization_preserves_primary_and_receipt_when_final_checks_fail(self):
        saved = {}
        owner = types.SimpleNamespace(save=lambda out, name, value: saved.update({name: value}))
        cfg = {"sources": [], "selection": "/FAKE", "selection_sha256": "FAKE", "boot_id": "FAKE"}
        primary = {"phase": "actor", "type": "FakeActorFailure", "message": "primary"}
        with (
            patch.object(controller, "sources_match", side_effect=ValueError("source changed")),
            patch.object(controller, "host", side_effect=ValueError("host low")),
        ):
            result = controller.finalize(owner, None, False, cfg, None, Path("/FAKE"), primary, {})
        receipt = saved["receipt.json"]
        self.assertEqual(result, 1)
        self.assertIs(receipt["primary_error"], primary)
        self.assertEqual(
            [row["phase"] for row in receipt["secondary_errors"]], ["final_source", "final_host"]
        )
        self.assertFalse(receipt["completed"])
        self.assertFalse(receipt["whole_tree_exit_confirmed"])

    def test_finalization_closes_before_final_source_or_host_failure(self):
        calls, saved = [], {}
        owner = types.SimpleNamespace(
            command=lambda *a, **k: (
                "Description=FAKE-NONCE\nInvocationID=FAKE-INVOCATION\nMemoryPeak=1234\nMainPID=123\n"
            ),
            save=lambda out, name, value: saved.update({name: value}),
        )
        boundary = types.SimpleNamespace(
            invocation="FAKE-INVOCATION",
            actor=process(),
            close=lambda **k: (
                calls.append("close")
                or {
                    "whole_unit_exit_confirmed": True,
                    "observed_cgroup_cohort_exit_confirmed": True,
                }
            ),
        )
        cfg = {
            "nonce": "FAKE-NONCE",
            "sources": [],
            "selection": "/FAKE",
            "selection_sha256": "FAKE",
            "boot_id": "FAKE",
        }

        def source_failure(rows):
            calls.append("source")
            raise ValueError("source disappeared")

        def host_failure(*a):
            calls.append("host")
            raise ValueError("host changed")

        with (
            patch.object(controller, "sources_match", side_effect=source_failure),
            patch.object(controller, "host", side_effect=host_failure),
        ):
            controller.finalize(
                owner, boundary, True, cfg, None, Path("/FAKE"), None, {"actor_success": True}
            )
        self.assertEqual(calls, ["close", "source", "host"])
        self.assertEqual(saved["receipt.json"]["primary_error"]["phase"], "final_source")
        self.assertEqual(saved["receipt.json"]["secondary_errors"][0]["phase"], "final_host")
        self.assertTrue(saved["receipt.json"]["state"]["preclose_exit_memory_authenticated"])

    def test_expired_postreceipt_hold_never_qualifies_as_completed(self):
        saved = {}
        owner = types.SimpleNamespace(command=lambda *a, **k: "Description=FAKE-NONCE\nInvocationID=FAKE-INVOCATION\nMainPID=0\nExecMainStatus=125\nMemoryPeak=1234\n", save=lambda out, name, value: saved.update({name: value}))
        boundary = types.SimpleNamespace(invocation="FAKE-INVOCATION", actor=process(), close=lambda **k: {"whole_unit_exit_confirmed": True, "observed_cgroup_cohort_exit_confirmed": True})
        cfg = {"nonce": "FAKE-NONCE", "sources": [], "selection": "/FAKE", "selection_sha256": "FAKE", "boot_id": "FAKE"}
        with patch.object(controller, "sources_match", return_value=True), patch.object(controller, "actor_selection", return_value={}), patch.object(controller, "host", return_value={"fake": True}):
            result = controller.finalize(owner, boundary, True, cfg, None, Path("/FAKE"), None, {"actor_success": True})
        self.assertEqual(result, 1)
        self.assertFalse(saved["receipt.json"]["completed"])
        self.assertIn("expiry125", saved["receipt.json"]["primary_error"]["message"])


if __name__ == "__main__":
    unittest.main()
