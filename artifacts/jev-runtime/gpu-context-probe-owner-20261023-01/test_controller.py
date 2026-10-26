"""Pure stdlib tests: no unit, observer HTTP, framework, model or mmap execution."""

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
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

    def test_environment_selects_only_pinned_gpu_with_controlled_overlay(self):
        env = controller.environment(
            {"nonce": "FAKE-NONCE", "output": "/tmp/FAKE-OUTPUT", "actor_deadline_monotonic": 100}
        )
        self.assertEqual(env["PYTHONPATH"], str(controller.ROOT / "artifacts/jev-runtime/overlay"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-" + controller.GPU_UUID)
        self.assertNotIn("ACCELERATE_USE_CPU", env)
        self.assertEqual(env["JEV_GPU_PROBE_DEADLINE_MONOTONIC"], "100")
        self.assertEqual(env["CUDA_CACHE_PATH"], "/tmp/FAKE-OUTPUT/cache/cuda")
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
            controller.finalize(owner, boundary, True, cfg, None, Path("/FAKE"), None, {})
        self.assertEqual(calls, ["close", "source", "host"])
        self.assertEqual(saved["receipt.json"]["primary_error"]["phase"], "final_source")
        self.assertEqual(saved["receipt.json"]["secondary_errors"][0]["phase"], "final_host")
        self.assertTrue(saved["receipt.json"]["state"]["preclose_exit_memory_authenticated"])

    def test_expired_postreceipt_hold_never_qualifies_as_completed(self):
        saved = {}
        owner = types.SimpleNamespace(
            command=lambda *a, **k: (
                "Description=FAKE-NONCE\nInvocationID=FAKE-INVOCATION\nMainPID=0\nExecMainStatus=125\nMemoryPeak=1234\n"
            ),
            save=lambda out, name, value: saved.update({name: value}),
        )
        boundary = types.SimpleNamespace(
            invocation="FAKE-INVOCATION",
            actor=process(),
            close=lambda **k: {
                "whole_unit_exit_confirmed": True,
                "observed_cgroup_cohort_exit_confirmed": True,
            },
        )
        cfg = {
            "nonce": "FAKE-NONCE",
            "sources": [],
            "selection": "/FAKE",
            "selection_sha256": "FAKE",
            "boot_id": "FAKE",
        }
        with (
            patch.object(controller, "sources_match", return_value=True),
            patch.object(controller, "actor_selection", return_value={}),
            patch.object(controller, "host", return_value={"fake": True}),
        ):
            result = controller.finalize(
                owner, boundary, True, cfg, None, Path("/FAKE"), None, {"actor_success": True}
            )
        self.assertEqual(result, 1)
        self.assertFalse(saved["receipt.json"]["completed"])
        self.assertIn("expiry125", saved["receipt.json"]["primary_error"]["message"])

    def test_affinity_dual_read_accepts_cpu0_and_rejects_widening(self):
        owner = types.SimpleNamespace(process=Mock(side_effect=[process("R"), process("S")]))
        with (
            patch.object(controller.os, "sched_getaffinity", side_effect=[{0}, {0}]),
            patch.object(Path, "read_text", return_value="Cpus_allowed_list:\t0\n"),
        ):
            self.assertEqual(controller.main_affinity(owner, 123)["sched_after"], [0])
        owner.process = Mock(return_value=process())
        with (
            patch.object(controller.os, "sched_getaffinity", side_effect=[{0}, {0, 1}]),
            patch.object(Path, "read_text", return_value="Cpus_allowed_list:\t0\n"),
            self.assertRaisesRegex(ValueError, "main-thread affinity"),
        ):
            controller.main_affinity(owner, 123)

    def test_missing_cpuset_is_recorded_with_real_affinity_gate(self):
        limits = "Max core file size 0 0 bytes\nMax file size 8388608 8388608 bytes\nMax open files 512 512 files\n"
        text = {
            "cpu.max": "100000 100000",
            "limits": limits,
            "memory.peak": "1234",
            "status": "Cpus_allowed_list:\t0\n",
        }
        owner = types.SimpleNamespace(
            CGROUP_ROOT=Path("/FAKE-CGROUP"),
            command=lambda *a, **k: (
                "CPUQuotaPerSecUSec=1s\nCPUAffinity=RAW-NOT-PARSED\nLimitCORE=0\nLimitNOFILE=512\nLimitFSIZE=8388608\nWorkingDirectory="
                + str(controller.ROOT)
                + "\nMemoryPeak=1234\n"
            ),
            process=lambda pid: process(),
        )
        with (
            patch.object(Path, "read_text", new=lambda path: text[path.name]),
            patch.object(Path, "exists", return_value=False),
            patch.object(controller.os, "sched_getaffinity", return_value={0}),
        ):
            value = controller.properties(owner, types.SimpleNamespace(cgroup="/FAKE"), 123)
        self.assertIsNone(value["cpuset_cpus_effective_raw"])
        self.assertEqual(value["main_thread_affinity"]["sched_before"], [0])
        self.assertEqual(value["kernel_cpu_max"], ["100000", "100000"])

    def test_hold_success_requires_authenticated_live_close_before_deadline(self):
        for mode, finish, expected in [
            ("authenticated_stop", 1, 0),
            ("authenticated_stop", 3, 1),
            ("already_terminal_verified", 1, 1),
        ]:
            with self.subTest(mode=mode, finish=finish):
                saved = {}
                owner = types.SimpleNamespace(
                    command=lambda *a, **k: (
                        "Description=FAKE-NONCE\nInvocationID=FAKE-INVOCATION\nMainPID=123\nMemoryPeak=1234\n"
                    ),
                    save=lambda out, name, value: saved.update({name: value}),
                )
                boundary = types.SimpleNamespace(
                    invocation="FAKE-INVOCATION",
                    actor=process(),
                    close=lambda **k: {
                        "closure_mode": mode,
                        "before_stop": {"MainPID": "123"},
                        "whole_unit_exit_confirmed": True,
                        "observed_cgroup_cohort_exit_confirmed": True,
                    },
                )
                cfg = {
                    "nonce": "FAKE-NONCE",
                    "sources": [],
                    "selection": "/FAKE",
                    "selection_sha256": "FAKE",
                    "boot_id": "FAKE",
                }
                state = {"actor_success": True, "actor_exit": {"hold_deadline_monotonic": 2}}
                with (
                    patch.object(controller, "sources_match", return_value=True),
                    patch.object(controller, "actor_selection", return_value={}),
                    patch.object(controller, "host", return_value={"fake": True}),
                    patch.object(controller.time, "monotonic", return_value=finish),
                ):
                    result = controller.finalize(
                        owner, boundary, True, cfg, None, Path("/FAKE"), None, state
                    )
                self.assertEqual(result, expected)

    def test_linux_limit_lines_allow_trailing_space_but_reject_wrong_hard_limit(self):
        text = {
            "cpu.max": "100000 100000",
            "limits": "Max core file size        0                    0                    bytes     \nMax file size            8388608              8388608              bytes     \nMax open files           512                  512                  files     \n",
            "memory.peak": "1234",
            "status": "Cpus_allowed_list:\t0\n",
        }
        owner = types.SimpleNamespace(
            CGROUP_ROOT=Path("/FAKE-CGROUP"),
            command=lambda *a, **k: (
                "CPUQuotaPerSecUSec=1s\nCPUAffinity=RAW-NOT-PARSED\nLimitCORE=0\nLimitNOFILE=512\nLimitFSIZE=8388608\nWorkingDirectory="
                + str(controller.ROOT)
                + "\nMemoryPeak=1234\n"
            ),
            process=lambda pid: process(),
        )
        with (
            patch.object(Path, "read_text", new=lambda path: text[path.name]),
            patch.object(Path, "exists", return_value=False),
            patch.object(controller.os, "sched_getaffinity", return_value={0}),
        ):
            value = controller.properties(owner, types.SimpleNamespace(cgroup="/FAKE"), 123)
            self.assertEqual(value["kernel_memory_peak_bytes"], 1234)
            text["limits"] = text["limits"].replace(
                "0                    0                    bytes",
                "0                    unlimited            bytes",
            )
            with self.assertRaisesRegex(ValueError, "Max core file size"):
                controller.properties(owner, types.SimpleNamespace(cgroup="/FAKE"), 123)

    def test_partial_actor_receipt_is_retained_before_exit2_refusal(self):
        partial = {
            "status": "partial",
            "passed": False,
            "mechanism_checks_completed": False,
            "cases": [
                {"case": "FAKE-completed", "completed": True},
                {"case": "FAKE-pending", "completed": False},
            ],
        }
        state = {}
        result = {
            "config_sha256": "FAKE-SHA",
            "actor": process("R"),
            "entry_exit_code": 2,
            "error": {"type": "SystemExit", "message": "2"},
        }
        with tempfile.TemporaryDirectory(prefix="FAKE-real-layer-partial-") as directory:
            out = Path(directory)
            (out / "fixture").mkdir()
            (out / "fixture/receipt.json").write_text(json.dumps(partial))
            with self.assertRaisesRegex(ValueError, "partial exit2"):
                controller.retain_and_validate_result(
                    {"config_sha256": "FAKE-SHA"}, process("S"), result, out, state
                )
        self.assertEqual(state["actor_receipt"], partial)
        self.assertNotIn("actor_success", state)

    def test_native_library_and_pyi_are_metadata_only_and_model_blob_is_refused(self):
        actor_data = b"FAKE actor source, never executed"
        stub = "/FAKE-NOT-A-MODEL/props.pyi"
        stub_data = b"FAKE CUDA property typing, never executed"
        actor_ref = {
            "path": str(controller.ACTOR),
            "sha256": controller.sha(actor_data),
            "size_bytes": len(actor_data),
        }
        weight = "/FAKE-NOT-A-MODEL/libtorch_cuda.so.13"
        stat = {"st_dev": 1, "st_ino": 2, "st_size": 23900000000, "st_mtime_ns": 3}
        value = {
            "profile": controller.ACTOR_PROFILE,
            "sources": {
                "entry": actor_ref,
                "gpu_typing": {
                    "path": stub,
                    "sha256": controller.sha(stub_data),
                    "size_bytes": len(stub_data),
                },
            },
            "versions": {"torch": "FAKE"},
            "binary_stats": [{"path": weight, "stat": stat}],
        }
        selected = controller.HERE / "FAKE-selection-never-used.json"
        data = controller.encoded(value)
        reads = []

        def metadata_read(path):
            reads.append(str(path))
            if Path(path) == selected:
                return data
            if Path(path) == controller.ACTOR:
                return actor_data
            if str(path) == stub:
                return stub_data
            raise AssertionError("Weight payload must never be read by parent")

        with (
            patch.object(controller, "read", side_effect=metadata_read),
            patch.object(controller, "ordinary", side_effect=lambda path, **k: Path(path)),
            patch.object(
                controller,
                "ref",
                return_value={
                    **actor_ref,
                    "path": str(controller.ACTOR.relative_to(controller.ROOT)),
                },
            ),
            patch.object(Path, "stat", return_value=types.SimpleNamespace(**stat)),
        ):
            self.assertEqual(controller.actor_selection(selected, controller.sha(data), []), value)
        self.assertEqual(reads, [str(selected), str(controller.ACTOR), stub])
        value["binary_stats"][0]["path"] = "/FAKE-NOT-A-MODEL/huge.safetensors"
        data = controller.encoded(value)
        with (
            patch.object(controller, "read", side_effect=metadata_read),
            patch.object(controller, "ordinary", side_effect=lambda path, **k: Path(path)),
            patch.object(
                controller,
                "ref",
                return_value={
                    **actor_ref,
                    "path": str(controller.ACTOR.relative_to(controller.ROOT)),
                },
            ),
            patch.object(
                Path, "stat", side_effect=AssertionError("Model blob must not even be statted")
            ),
            self.assertRaisesRegex(ValueError, "no model/blob"),
        ):
            controller.actor_selection(selected, controller.sha(data), [])

    def test_saved_gpu_identity_same_boot_and_uuid_only_is_not_live_probe(self):
        label, _ = controller.FIXED["gpu_identity"]
        saved = json.loads(controller.read(controller.ROOT / label))
        self.assertEqual(controller.gpu_identity_baseline(saved["boot_id"]), saved)
        with self.assertRaisesRegex(ValueError, "same boot/physical UUID"):
            controller.gpu_identity_baseline("FAKE-NOT-THIS-BOOT")
        self.assertEqual(controller.gpu_uuid("GPU-" + controller.GPU_UUID), controller.GPU_UUID)
        with self.assertRaisesRegex(ValueError, "UUID format"):
            controller.gpu_uuid("0")

    def test_gpu_context_success_requires_uuid_init_tiny_operator_and_no_model(self):
        success = {
            "status": "completed_mechanism_observations",
            "passed": True,
            "completed": True,
            "CPU_ONLY": False,
            "mechanism_checks_completed": True,
            "context_and_tiny_operator_completed": True,
            "source_integrity_final": True,
            "GPU_context_measured": True,
            "GPU_model_transfer_measured": False,
            "model_loaded": False,
            "no_model_loaded": True,
            "model_payload_read": False,
            "large_model_runtime_or_quality_qualified": False,
            "CUDA": {
                "initialized_final": True,
                "logical_device_index": 0,
                "device_uuid_raw": "GPU-" + controller.GPU_UUID,
                "device_uuid_normalized": controller.GPU_UUID,
            },
            "tiny_operator": {
                "dtype": "bfloat16",
                "shape": [16, 16],
                "completed": True,
                "all_exact_16": True,
                "readback_bytes": 512,
                "output_sha256": controller.sha(b"\x80\x41" * 256),
                "closed_form_expected_sha256": controller.sha(b"\x80\x41" * 256),
                "optional_diagnostic": "FAKE",
            },
        }
        result = {
            "config_sha256": "FAKE-SHA",
            "actor": process("R"),
            "entry_exit_code": 0,
            "error": None,
        }
        for changed, expected in [
            (None, True),
            ("wrong_uuid", False),
            ("not_initialized", False),
            ("model_loaded", False),
            ("payload_read", False),
            ("operator_incomplete", False),
            ("wrong_readback_size", False),
            ("wrong_readback_hash", False),
            ("untyped_shape", False),
        ]:
            with self.subTest(changed=changed):
                value = json.loads(json.dumps(success))
                if changed == "wrong_uuid":
                    value["CUDA"]["device_uuid_raw"] = "00000000-0000-0000-0000-000000000000"
                if changed == "not_initialized":
                    value["CUDA"]["initialized_final"] = False
                if changed == "model_loaded":
                    value["model_loaded"] = True
                if changed == "payload_read":
                    value["model_payload_read"] = True
                if changed == "operator_incomplete":
                    value["tiny_operator"]["completed"] = False
                if changed == "wrong_readback_size":
                    value["tiny_operator"]["readback_bytes"] = 511
                if changed == "wrong_readback_hash":
                    value["tiny_operator"]["output_sha256"] = "0" * 64
                if changed == "untyped_shape":
                    value["tiny_operator"]["shape"] = [16.0, 16.0]
                state = {}
                with patch.object(controller, "read", return_value=controller.encoded(value)):
                    if expected:
                        controller.retain_and_validate_result(
                            {"config_sha256": "FAKE-SHA"},
                            process("S"),
                            result,
                            Path("/FAKE"),
                            state,
                        )
                    else:
                        with self.assertRaisesRegex(ValueError, "Pinned GPU context"):
                            controller.retain_and_validate_result(
                                {"config_sha256": "FAKE-SHA"},
                                process("S"),
                                result,
                                Path("/FAKE"),
                                state,
                            )
                self.assertEqual(state.get("actor_success", False), expected)
                self.assertEqual(state["actor_receipt"], value)

    def test_all_accepted_gate_aggregate_keeps_lowpoint_outside_retained64(self):
        state = {}

        def row(available):
            return {"observation": {"mem_available_bytes": available}, "reserve_bytes": 1}

        controller.record_runtime_host(state, row(900), "readiness")
        controller.record_runtime_host(state, row(800), "preexec")
        values = [1000] * 200
        values[128] = 5
        for value in values:
            controller.record_runtime_host(state, row(value), "loop")
        retained = values[:63] + values[-1:]
        summary = state["successful_parent_postlaunch_host_gate_summary"]
        self.assertEqual(summary["runtime_host_gate_count"], 202)
        self.assertEqual(summary["phase_counts"], {"readiness": 1, "preexec": 1, "loop": 200})
        self.assertEqual(summary["min_MemAvailable_bytes"], 5)
        self.assertEqual(summary["min_margin_vsR_bytes"], 4)
        self.assertEqual(min(retained), 1000)
        self.assertIn("not a continuous-time minimum", summary["scope"])


if __name__ == "__main__":
    unittest.main()
