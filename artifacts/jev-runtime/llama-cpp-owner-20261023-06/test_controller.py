"""Pure stdlib regressions; fake identities, no unit, observer, command, Torch or payload."""

import contextlib
import hashlib
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
SPEC = importlib.util.spec_from_file_location("_selected_command_owner_test", ENTRY)
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


def process(state="R", start=20):
    return {
        "pid": 123,
        "start_ticks": start,
        "pgid": 123,
        "sid": 123,
        "uid": 1000,
        "cgroup": "/FAKE-NOT-LIVE",
        "state": state,
    }


def spec():
    return {
        "profile": c.SPEC_PROFILE,
        "runtime_seconds": 30,
        "file_size_max_bytes": c.FILE_SIZE_MAX,
        "commands": [{"argv": ["/FAKE-NOT-EXECUTED", "{OUTPUT}/build"], "cwd": "{OUTPUT}"}],
        "sources": [],
        "binary_stats": [],
    }


class Tests(unittest.TestCase):
    def test_default_dry_no_authority_or_framework(self):
        before = set(sys.modules)
        with (
            patch.object(c, "load_fixed", side_effect=AssertionError("authority")),
            patch.object(c, "run", side_effect=AssertionError("run")),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(c.main([]), 0)
        self.assertEqual(json.loads(output.getvalue())["mode"], "DRY_SOURCE_ONLY")
        self.assertFalse(
            any(k == "torch" or k.startswith("torch.") for k in set(sys.modules) - before)
        )

    def test_wrong_sha_refuses_before_authorities_or_commands(self):
        with (
            patch.object(c, "load_fixed", side_effect=AssertionError("authority")),
            patch.object(c.subprocess, "Popen", side_effect=AssertionError("command")),
            self.assertRaisesRegex(ValueError, "controller entry SHA"),
        ):
            c.run(types.SimpleNamespace(expected_entry_sha256="0" * 64))

    def test_original_boundary_accepts_selected_30_7200_10800_unchanged(self):
        owner = c.load_fixed("boundary")
        for runtime in (30, 7200, 10800):
            selected = c.selection(
                {
                    "nonce": "jev-d3-boundary-" + "a" * 32,
                    "boot_id": "FAKE-NOT-LIVE",
                    "runtime_seconds": runtime,
                },
                owner,
            )
            self.assertEqual(selected.memory_max_bytes, 2 * c.GIB)
            self.assertEqual(selected.properties()["MemoryMax"], str(2 * c.GIB))
            self.assertEqual(selected.runtime_seconds, runtime)
            self.assertEqual(selected.stop_seconds, 3)
            self.assertEqual(selected.properties()["MemorySwapMax"], "0")
            self.assertEqual(selected.properties()["RemainAfterExit"], "yes")

    def test_identity_ignores_dynamic_state_but_not_start_or_argv(self):
        one = {"process": process(), "exe": "/fake", "argv": ["fake"], "cwd": "/fake"}
        two = {**one, "process": process("S")}
        self.assertTrue(c.same_identity(one, two))
        self.assertFalse(c.same_identity(one, {**two, "process": process("S", 21)}))
        self.assertFalse(c.same_identity(one, {**two, "argv": ["changed"]}))

    def test_existing_unit_rejected_without_stop(self):
        owner = types.SimpleNamespace(
            status=Mock(return_value={"LoadState": "loaded"}),
            CGROUP_ROOT=Path("/FAKE"),
            command=Mock(),
        )
        with self.assertRaisesRegex(ValueError, "never reuse"):
            c.ensure_absent(owner, types.SimpleNamespace(cgroup="/FAKE"))
        owner.command.assert_not_called()

    def test_existing_cgroup_rejected_when_unit_absent(self):
        owner = types.SimpleNamespace(
            status=Mock(return_value={"LoadState": "not-found"}), CGROUP_ROOT=Path("/FAKE")
        )
        with (
            patch.object(Path, "exists", return_value=True),
            self.assertRaisesRegex(ValueError, "never reuse"),
        ):
            c.ensure_absent(owner, types.SimpleNamespace(cgroup="/FAKE"))

    def test_host_reserve_initial_full2_and_busy_refusal(self):
        row = {
            "boot_id": "FAKE",
            "mem_total_bytes": 100 * c.GIB,
            "mem_available_bytes": 24 * c.GIB,
            "comfy_state": "http_idle",
            "comfy_running": 0,
            "comfy_pending": 0,
        }
        observer = types.SimpleNamespace(observe=Mock(return_value=row))
        self.assertEqual(c.host(observer, "FAKE")["reserve_bytes"], 24 * c.GIB)
        with self.assertRaisesRegex(ValueError, "plus full selected2GiB") as captured:
            c.host(observer, "FAKE", initial=True)
        self.assertEqual(captured.exception.evidence["required_available_bytes"], 26 * c.GIB)
        row["comfy_running"] = 1
        with self.assertRaisesRegex(ValueError, "Comfy idle"):
            c.host(observer, "FAKE")

    def test_timeout_byte_output_bounded_and_json_safe(self):
        value = c.failure(
            subprocess.TimeoutExpired(["FAKE"], 1, output=b"x" * 40000, stderr=b"bad\xff")
        )
        self.assertEqual(len(value["stdout"]), 32768)
        self.assertEqual(value["stderr"], "bad\ufffd")
        json.dumps(value, allow_nan=False)

    def selected(self, value, *, bad_sha=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.json"
            blob = c.encoded(value)
            path.write_bytes(blob)
            with patch.object(c, "ROOT", Path(directory)):
                return c.command_selection(
                    path, "0" * 64 if bad_sha else c.sha(blob), [c.ref(path)]
                )

    def test_exact_spec_accepts_runtime30_7200_10800(self):
        for runtime in (30, 7200, 10800):
            value = spec()
            value["runtime_seconds"] = runtime
            self.assertEqual(self.selected(value), value)

    def test_spec_sha_is_independent_gate(self):
        with self.assertRaisesRegex(ValueError, "spec SHA"):
            self.selected(spec(), bad_sha=True)

    def test_spec_refuses_runtime_or_file_cap_relaxation(self):
        for key, value in (
            ("runtime_seconds", 10801),
            ("runtime_seconds", 10799),
            ("runtime_seconds", True),
            ("file_size_max_bytes", 8 * 1024**2),
            ("file_size_max_bytes", 40 * c.GIB),
        ):
            data = spec()
            data[key] = value
            with (
                self.subTest(key=key, value=value),
                self.assertRaisesRegex(ValueError, "profile/runtime/file-size"),
            ):
                self.selected(data)

    def test_spec_refuses_relative_executable_or_unselected_substitution(self):
        for argv in (["python", "test.py"], ["/fake", "{HOME}"], ["/fake", "nul\0arg"]):
            data = spec()
            data["commands"][0]["argv"] = argv
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                self.selected(data)

    def test_sources_bounded_exact_no_duplicate_or_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "source.py"
            p.write_text("# fake source only\n")
            with patch.object(c, "ROOT", Path(directory)):
                row = c.ref(p)
                self.assertTrue(c.sources_match([row]))
                with self.assertRaisesRegex(ValueError, "Distinct"):
                    c.sources_match([row, row])
                p.write_text("# changed\n")
                with self.assertRaisesRegex(ValueError, "unchanged"):
                    c.sources_match([row])

    def test_binary_stats_never_open_tool_payload(self):
        data = spec()
        data["binary_stats"] = [
            {
                "path": "/FAKE-NOT-A-MODEL",
                "stat": {"st_dev": 1, "st_ino": 2, "st_size": 20_000_000_000, "st_mtime_ns": 4},
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "spec.json"
            blob = c.encoded(data)
            p.write_bytes(blob)
            original_read = c.read

            def small_only(path):
                self.assertEqual(Path(path), p)
                return original_read(path)

            with (
                patch.object(c, "ROOT", Path(directory)),
                patch.object(c, "read", side_effect=small_only),
                patch.object(c, "ordinary", side_effect=lambda path, **kw: Path(path)),
                patch.object(
                    Path,
                    "stat",
                    return_value=types.SimpleNamespace(**data["binary_stats"][0]["stat"]),
                ),
            ):
                self.assertEqual(
                    c.command_selection(
                        p,
                        c.sha(blob),
                        [{"path": "spec.json", "sha256": c.sha(blob), "size_bytes": len(blob)}],
                    ),
                    data,
                )

    def test_stream_is_bounded_but_all_bytes_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fake.stdout"
            sink = c.BoundedOutput(path)
            blob = b"x" * (c.LOG_LIMIT + 32)
            sink.feed(blob)
            sink.feed(b"last")
            row = sink.close()
            self.assertEqual(path.stat().st_size, c.LOG_LIMIT)
            self.assertEqual(row["observed_bytes"], len(blob) + 4)
            self.assertEqual(row["observed_sha256"], hashlib.sha256(blob + b"last").hexdigest())
            self.assertTrue(row["truncated"])

    def test_all_successful_sample_min_survives_discarded_raw_low(self):
        state = {}
        raw = []
        for n in range(200):
            available = 29 * c.GIB if n != 128 else 27 * c.GIB
            row = {"observation": {"mem_available_bytes": available}, "reserve_bytes": 24 * c.GIB}
            c.record_runtime_host(state, row, "loop")
            if len(raw) < 64:
                raw.append(row)
            else:
                raw[-1] = row
        summary = state["successful_parent_postlaunch_host_gate_summary"]
        self.assertEqual(summary["runtime_host_gate_count"], 200)
        self.assertEqual(summary["phase_counts"], {"readiness": 0, "preexec": 0, "loop": 200})
        self.assertEqual(summary["min_MemAvailable_bytes"], 27 * c.GIB)
        self.assertEqual(summary["min_margin_vsR_bytes"], 3 * c.GIB)
        self.assertEqual(min(r["observation"]["mem_available_bytes"] for r in raw), 29 * c.GIB)

    def test_child_failure_retained_before_refusal(self):
        result = {
            "config_sha256": "fake",
            "actor": process("S"),
            "entry_exit_code": 1,
            "error": {"type": "SyntheticChildError", "message": "original fake failure"},
            "command_results": [],
            "planned_command_count": 1,
        }
        state = {}
        with self.assertRaisesRegex(ValueError, "original error retained") as captured:
            c.retain_and_validate_result(
                {"config_sha256": "fake", "command_count": 1}, process(), result, None, state
            )
        self.assertIs(state["actor_exit"], result)
        self.assertEqual(
            c.failure(captured.exception)["evidence"]["error"]["type"], "SyntheticChildError"
        )
        self.assertNotIn("actor_success", state)

    def test_exit0_does_not_allow_missing_command_successes(self):
        result = {
            "config_sha256": "fake",
            "actor": process(),
            "entry_exit_code": 0,
            "error": None,
            "command_results": [],
            "planned_command_count": 1,
        }
        with self.assertRaisesRegex(ValueError, "All explicitly"):
            c.retain_and_validate_result(
                {"config_sha256": "fake", "command_count": 1}, process(), result, None, {}
            )

    def test_final_source_and_host_failures_preserve_primary_and_receipt(self):
        owner = types.SimpleNamespace(save=Mock())
        primary = {"phase": "command", "type": "SyntheticPrimary", "message": "keep me"}
        cfg = {
            "sources": [],
            "base_sources": [],
            "selection": "/FAKE",
            "selection_sha256": "fake",
            "boot_id": "FAKE",
        }
        with (
            patch.object(c, "sources_match", side_effect=ValueError("final source missing")),
            patch.object(c, "host", side_effect=ValueError("final host unavailable")),
        ):
            self.assertEqual(
                c.finalize(owner, None, False, cfg, None, Path("/FAKE"), primary, {}), 1
            )
        receipt = owner.save.call_args.args[2]
        self.assertIs(receipt["primary_error"], primary)
        self.assertEqual(
            [r["phase"] for r in receipt["secondary_errors"]], ["final_source", "final_host"]
        )
        self.assertFalse(receipt["completed"])

    def test_final_success_requires_authenticated_live_hold_close(self):
        for mode in ("authenticated_stop", "already_terminal_verified"):
            actor = process()
            boundary = types.SimpleNamespace(
                actor=actor,
                invocation="a" * 32,
                close=Mock(
                    return_value={
                        "closure_mode": mode,
                        "before_stop": {"MainPID": "123"},
                        "whole_unit_exit_confirmed": True,
                        "observed_cgroup_cohort_exit_confirmed": True,
                    }
                ),
            )
            owner = types.SimpleNamespace(
                save=Mock(),
                command=Mock(
                    return_value="Description=fake-nonce\nInvocationID="
                    + "a" * 32
                    + "\nMainPID=123\nMemoryPeak=1234\n"
                ),
            )
            cfg = {
                "nonce": "fake-nonce",
                "sources": [],
                "base_sources": [],
                "selection": "/FAKE",
                "selection_sha256": "fake",
                "boot_id": "FAKE",
            }
            state = {"actor_success": True, "actor_exit": {"hold_deadline_monotonic": 105}}
            with (
                patch.object(c, "sources_match", return_value=True),
                patch.object(c, "command_selection", return_value={}),
                patch.object(c, "host", return_value={"fake": True}),
                patch.object(c.time, "monotonic", return_value=100),
            ):
                code = c.finalize(owner, boundary, True, cfg, None, Path("/FAKE"), None, state)
            self.assertEqual(code, 0 if mode == "authenticated_stop" else 1)
            boundary.close.assert_called_once()

    def test_explicit_logical_symlink_preserves_argv_and_pins_target(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "fake-python"
            target.write_text("FAKE NOT EXECUTED")
            logical = Path(directory) / "fake-venv-python"
            logical.symlink_to(target)
            stat = target.stat()
            pinned = [
                {
                    "path": str(target),
                    "stat": {
                        key: getattr(stat, key)
                        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
                    },
                }
            ]
            self.assertEqual(c.selected_executable(str(logical), pinned), target)
            with self.assertRaisesRegex(ValueError, "explicit selected tool stat"):
                c.selected_executable(str(logical), [])
            target.write_text("CHANGED FAKE NOT EXECUTED")
            with self.assertRaisesRegex(ValueError, "selected stat identity"):
                c.selected_executable(str(logical), pinned)

    def test_generated_binary_only_inside_owned_fresh_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "actual-FAKE"
            output.mkdir()
            generated = output / "fake-server"
            generated.write_text("FAKE NOT EXECUTED")
            with patch.object(c, "ROOT", Path(directory)):
                self.assertEqual(
                    c.selected_executable(str(generated), [], generated_under=output), generated
                )
            external = Path(directory) / "external-fake"
            external.write_text("FAKE NOT EXECUTED")
            escape = output / "escape-symlink"
            escape.symlink_to(external)
            with (
                patch.object(c, "ROOT", Path(directory)),
                self.assertRaisesRegex(ValueError, "fresh owned build origin"),
            ):
                c.selected_executable(str(escape), [], generated_under=output)

    def test_command_cleanup_attempts_all_steps_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            executable = output / "fake-executable"
            executable.write_text("NOT EXECUTED")
            original = KeyError("original simulated Popen failure")
            selector = types.SimpleNamespace(close=Mock(side_effect=ValueError("selector close")))
            streams = [
                types.SimpleNamespace(close=Mock(side_effect=ValueError("log close1"))),
                types.SimpleNamespace(close=Mock(side_effect=ValueError("log close2"))),
            ]

            def save(out, name, value):
                if name.endswith("-output.json"):
                    raise ValueError("output save")

            owner = types.SimpleNamespace(save=Mock(side_effect=save))
            cfg = {
                "output": str(output),
                "binary_stats": [],
                "nonce": "FAKE",
                "command_deadline_monotonic": 100,
            }
            with (
                patch.object(c, "ROOT", output),
                patch.object(c, "selected_executable", return_value=executable),
                patch.object(c.selectors, "DefaultSelector", return_value=selector),
                patch.object(c, "BoundedOutput", side_effect=streams),
                patch.object(c.subprocess, "Popen", side_effect=original),
                self.assertRaises(KeyError) as captured,
            ):
                c.execute_command(cfg, 1, {"argv": [str(executable)], "cwd": str(output)}, owner)
            self.assertIs(captured.exception, original)
            self.assertEqual(
                [n["phase"] for n in original.command_cleanup_errors],
                ["selector_close", "log_close", "log_close", "command_output_receipt"],
            )
            selector.close.assert_called_once()
            for stream in streams:
                stream.close.assert_called_once()

    def test_command_pipe_close_failure_does_not_hide_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            executable = output / "fake-executable"
            executable.write_text("NOT EXECUTED")
            pipes = [
                types.SimpleNamespace(
                    closed=False, close=Mock(side_effect=ValueError("pipeclose1"))
                ),
                types.SimpleNamespace(
                    closed=False, close=Mock(side_effect=ValueError("pipeclose2"))
                ),
            ]
            child = types.SimpleNamespace(
                pid=123, stdout=pipes[0], stderr=pipes[1], poll=Mock(return_value=None)
            )
            primary = ValueError("original identity refusal")
            cfg = {
                "output": str(output),
                "binary_stats": [],
                "nonce": "FAKE",
                "command_deadline_monotonic": 100,
            }
            owner = types.SimpleNamespace(save=Mock())
            with (
                patch.object(c, "ROOT", output),
                patch.object(c, "selected_executable", return_value=executable),
                patch.object(c.subprocess, "Popen", return_value=child),
                patch.object(c, "process_identity", side_effect=primary),
                self.assertRaises(ValueError) as captured,
            ):
                c.execute_command(cfg, 1, {"argv": [str(executable)], "cwd": str(output)}, owner)
            self.assertIs(captured.exception, primary)
            self.assertEqual(
                [n["phase"] for n in primary.command_cleanup_errors], ["pipe_close", "pipe_close"]
            )
            for pipe in pipes:
                pipe.close.assert_called_once()

    def test_final_source_verification_uses_base_refs_with_nonempty_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "fake-source.py"
            source.write_text("# not executed")
            path = output / "spec.json"
            with patch.object(c, "ROOT", output):
                data = spec()
                data["sources"] = [c.ref(source)]
                blob = c.encoded(data)
                path.write_bytes(blob)
                base = [c.ref(path)]
                cfg = {
                    "sources": [*base, *data["sources"]],
                    "base_sources": base,
                    "selection": str(path),
                    "selection_sha256": c.sha(blob),
                    "boot_id": "FAKE",
                }
                owner = types.SimpleNamespace(save=Mock())
                with patch.object(c, "host", return_value={"FAKE": True}):
                    c.finalize(owner, None, False, cfg, None, output, {"type": "FAKE-NO-RUN"}, {})
                receipt = owner.save.call_args.args[2]
                self.assertTrue(receipt["sources_unchanged"])
                self.assertEqual(receipt["secondary_errors"], [])

    def test_environment_exact_cpu_overlay_no_inherited_secret(self):
        env = c.environment({"output": "/FAKE", "nonce": "fake"})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["ACCELERATE_USE_CPU"], "true")
        self.assertEqual(env["PYTHONPATH"], str(c.ROOT / "artifacts/jev-runtime/overlay"))
        self.assertEqual(
            [env[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")],
            ["1"] * 3,
        )
        self.assertNotIn("HOME", env)


class CPU4Tests(unittest.TestCase):
    def test_exact_main_sched_proc_affinity_accepts_selected_four(self):
        owner = types.SimpleNamespace(process=Mock(side_effect=[process(), process("S")]))
        with (
            patch.object(c.os, "sched_getaffinity", side_effect=[set(c.CPU_CORES)] * 2),
            patch.object(Path, "read_text", return_value="Cpus_allowed_list:\t16-19\n"),
        ):
            row = c.main_affinity(owner, 123)
        self.assertEqual(row["sched_before"], [16, 17, 18, 19])
        self.assertEqual(row["sched_after"], row["sched_before"])
        self.assertEqual(row["proc_Cpus_allowed_list"], "16-19")
        self.assertEqual(row["process"], c.stable_process(process()))

    def test_main_affinity_refuses_widening_clipping_proc_disagreement_and_race(self):
        for first, second, proc, end in (
            (set(c.CPU_CORES), set(range(20)), "16-19", process()),
            ({16, 17, 18}, {16, 17, 18}, "16-18", process()),
            (set(c.CPU_CORES), set(c.CPU_CORES), "0-19", process()),
            (set(c.CPU_CORES), set(c.CPU_CORES), "16-19", process(start=21)),
        ):
            owner = types.SimpleNamespace(process=Mock(side_effect=[process(), end]))
            with (
                self.subTest(first=first, second=second, proc=proc, end=end),
                patch.object(c.os, "sched_getaffinity", side_effect=[first, second]),
                patch.object(Path, "read_text", return_value="Cpus_allowed_list:\t" + proc + "\n"),
                self.assertRaisesRegex(ValueError, "sched/proc dual read is CPU16-19"),
            ):
                c.main_affinity(owner, 123)

    def resource_readback(self, quota="400000 100000", manager="4s"):
        with tempfile.TemporaryDirectory() as directory:
            group = Path(directory) / "fake"
            group.mkdir()
            (group / "cpu.max").write_text(quota + "\n")
            (group / "memory.peak").write_text("1234\n")
            (group / "memory.events").write_text("low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n")
            owner = types.SimpleNamespace(
                CGROUP_ROOT=Path(directory),
                command=Mock(
                    return_value="\n".join(
                        key + "=" + value
                        for key, value in {
                            "CPUQuotaPerSecUSec": manager,
                            "CPUAffinity": "16-19",
                            "LimitCORE": "0",
                            "LimitNOFILE": "512",
                            "LimitFSIZE": str(c.FILE_SIZE_MAX),
                            "WorkingDirectory": str(c.ROOT),
                            "MemoryPeak": "1234",
                        }.items()
                    )
                ),
            )
            original = Path.read_text
            limits = "\n".join(
                (
                    "Max core file size 0 0 bytes",
                    "Max file size " + str(c.FILE_SIZE_MAX) + " " + str(c.FILE_SIZE_MAX) + " bytes",
                    "Max open files 512 512 files",
                )
            )

            def read(path, *args, **kwargs):
                if path == Path("/proc/123/limits"):
                    return limits
                return original(path, *args, **kwargs)

            with (
                patch.object(Path, "read_text", new=read),
                patch.object(c, "main_affinity", return_value={"FAKE": "affinity already tested"}),
            ):
                return c.properties(owner, types.SimpleNamespace(cgroup="/fake"), 123)

    def test_kernel_quota_accepts_four_times_any_positive_period(self):
        for quota in ("400000 100000", "800000 200000"):
            with self.subTest(quota=quota):
                row = self.resource_readback(quota)
                self.assertEqual(row["kernel_cpu_max"], quota.split())
                self.assertEqual(row["unit"]["CPUQuotaPerSecUSec"], "4s")
                self.assertEqual(row["kernel_memory_peak_bytes"], 1234)
                self.assertEqual(row["kernel_memory_events"]["counters"]["oom_kill"], 0)

    def test_quota_refuses_one_core_unlimited_ratio_mismatch_and_zero_period(self):
        for quota in ("100000 100000", "max 100000", "400000 100001", "0 0"):
            with self.subTest(quota=quota), self.assertRaisesRegex(ValueError, "CPU400%"):
                self.resource_readback(quota)
        with self.assertRaisesRegex(ValueError, "extra unit resource properties"):
            self.resource_readback(manager="1s")

    def test_bootstrap_dual_affinity_failure_precedes_identity_publish_or_commands(self):
        owner = types.SimpleNamespace(
            Selection=Mock(return_value=types.SimpleNamespace(cgroup="/FAKE-NOT-LIVE")),
            boot=Mock(return_value="FAKE"),
            save=Mock(),
        )
        cfg = {
            "nonce": "FAKE",
            "boot_id": "FAKE",
            "runtime_seconds": 30,
            "sources": [],
            "selection": "FAKE",
            "selection_sha256": "FAKE",
            "base_sources": [],
        }
        identity = {"process": process(), "exe": "/fake", "argv": [], "cwd": str(c.ROOT)}
        with (
            patch.dict(c.os.environ, {"JEV_COMMAND_NONCE": "FAKE"}),
            patch.object(c.os, "getpid", return_value=123),
            patch.object(c.os, "getuid", return_value=1000),
            patch.object(c, "sources_match", return_value=True),
            patch.object(c, "command_selection", return_value=spec()),
            patch.object(c, "process_identity", return_value=identity),
            patch.object(
                c, "main_affinity", side_effect=ValueError("dual affinity refused")
            ) as affinity,
            patch.object(c, "execute_command", side_effect=AssertionError("command")) as command,
            self.assertRaisesRegex(ValueError, "dual affinity refused"),
        ):
            c.bootstrap(cfg, "FAKE", owner)
        affinity.assert_called_once_with(owner, 123)
        owner.save.assert_not_called()
        command.assert_not_called()

    def test_memory_event_evidence_requires_bounded_numeric_own_oom_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            group = Path(directory)
            path = group / "memory.events"
            path.write_text("oom 2\noom_kill 1\n")
            self.assertEqual(c.memory_events(group)["counters"], {"oom": 2, "oom_kill": 1})
            for content in ("oom 0\n", "oom 0\noom 1\n", "oom -1\noom_kill 0\n", "x" * 8193):
                path.write_text(content)
                with self.subTest(content=content[:30]), self.assertRaises(ValueError):
                    c.memory_events(group)

    def test_cpu4_successor_keeps_original_reserve_and_unique_unit(self):
        self.assertEqual(c.reserve(130595864576), 26119172916)
        self.assertEqual(c.UNIT, "jev-d3-llama-cpp-cpu4-2g-long-20261023-06.service")
        self.assertEqual(c.PROFILE, "llama-cpp-cpu4-2g-long-command-owned-v1")
        self.assertEqual(c.SPEC_PROFILE, "llama-cpp-command-spec-v1")

    def test_mixed_owner05_bootstrap_refuses_before_authority_or_command(self):
        for profile, unit in (
            ("llama-cpp-cpu4-3g-long-command-owned-v1", c.UNIT),
            (c.PROFILE, "jev-d3-llama-cpp-cpu4-3g-long-20261023-05.service"),
        ):
            cfg = {
                "profile": profile,
                "unit": unit,
                "output": str(c.HERE / "synthetic-unused-output"),
            }
            raw = json.dumps(cfg).encode()
            with (
                patch.object(c, "read", return_value=raw),
                patch.object(c, "ordinary", side_effect=lambda p: Path(p)),
                patch.object(c, "load_fixed", side_effect=AssertionError("authority")),
                patch.object(c, "bootstrap", side_effect=AssertionError("command")),
                self.assertRaisesRegex(ValueError, "Fixed bootstrap profile/output"),
            ):
                c.main(
                    [
                        "--internal-stage",
                        "actor",
                        "--config",
                        "/FAKE",
                        "--config-sha256",
                        hashlib.sha256(raw).hexdigest(),
                    ]
                )


class FullCapAdmissionTests(unittest.TestCase):
    def test_two_gib_cap_and_full_startup_extra_share_fixed_profile(self):
        self.assertEqual(c.fixed_memory(), 2147483648)
        self.assertEqual(c.INITIAL_HOST_EXTRA_BYTES, c.MEMORY_MAX_BYTES)
        with (
            patch.object(c, "run", side_effect=AssertionError("run")),
            patch.object(c, "load_fixed", side_effect=AssertionError("authority")),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(c.main([]), 0)
        dry = json.loads(output.getvalue())
        self.assertEqual(dry["memory_max_bytes"], 2147483648)
        self.assertEqual(dry["initial_host_extra_bytes"], 2147483648)

    def test_cap_floor_two_three_mismatch_refused_before_observer_or_unit_selection(self):
        for cap, extra in (
            (3 * c.GIB, 2 * c.GIB),
            (2 * c.GIB, 3 * c.GIB),
            (3 * c.GIB, 3 * c.GIB),
            (2 * c.GIB - 1, 2 * c.GIB),
            (2 * c.GIB, 2 * c.GIB + 1),
        ):
            observer = types.SimpleNamespace(observe=Mock(side_effect=AssertionError("observer")))
            owner = types.SimpleNamespace(
                Selection=Mock(side_effect=AssertionError("unit selection"))
            )
            with (
                self.subTest(cap=cap, extra=extra),
                patch.object(c, "MEMORY_MAX_BYTES", cap),
                patch.object(c, "INITIAL_HOST_EXTRA_BYTES", extra),
            ):
                with self.assertRaisesRegex(ValueError, "fixed2GiB cap and full-cap"):
                    c.host(observer, "FAKE", initial=True)
                with self.assertRaisesRegex(ValueError, "fixed2GiB cap and full-cap"):
                    c.selection(
                        {"nonce": "FAKE", "boot_id": "FAKE", "runtime_seconds": 7200}, owner
                    )
            observer.observe.assert_not_called()
            owner.Selection.assert_not_called()

    def test_initial_floor_is_entire_two_gib_above_same_runtime_reserve(self):
        total = 130595864576
        reserve = c.reserve(total)
        self.assertEqual(reserve, 26119172916)
        floor = reserve + 2 * c.GIB
        self.assertEqual(floor, 28266656564)
        row = {
            "boot_id": "FAKE",
            "mem_total_bytes": total,
            "mem_available_bytes": floor,
            "comfy_state": "http_idle",
            "comfy_running": 0,
            "comfy_pending": 0,
        }
        observer = types.SimpleNamespace(observe=Mock(return_value=row))
        actual = c.host(observer, "FAKE", total, initial=True)
        self.assertEqual(actual["required_available_bytes"], floor)
        for available in (floor - 1, reserve + c.GIB, reserve):
            row["mem_available_bytes"] = available
            with (
                self.subTest(available=available),
                self.assertRaisesRegex(ValueError, "full selected2GiB"),
            ):
                c.host(observer, "FAKE", total, initial=True)
        row["mem_available_bytes"] = reserve
        self.assertEqual(c.host(observer, "FAKE", total)["required_available_bytes"], reserve)
        row["mem_available_bytes"] = reserve - 1
        with self.assertRaisesRegex(ValueError, "Original whole-host reserve"):
            c.host(observer, "FAKE", total)


if __name__ == "__main__":
    unittest.main(verbosity=2)
