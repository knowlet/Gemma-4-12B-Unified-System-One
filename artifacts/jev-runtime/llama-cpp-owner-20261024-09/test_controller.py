"""UNEXECUTED CPU5 source/fake tests. No units, host probes or model calls.

This file has not been imported, syntax-checked, formatted or executed.
"""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "llama-cpp-owner-20261024-08/controller.py"
spec = importlib.util.spec_from_file_location("cpu5_owner_source_fake", HERE / "controller.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def process(start=20):
    return {
        "pid": 123,
        "start_ticks": start,
        "pgid": 123,
        "sid": 123,
        "uid": 1000,
        "cgroup": "/FAKE-NOT-LIVE",
        "state": "R",
    }


class CPU5Tests(unittest.TestCase):
    def test_exact_parent_source_changes_only_named_profile_and_cpu_resource_literals(self):
        expected = PARENT.read_text()
        replacements = (
            (
                'UNIT = "jev-d3-llama-cpp-cpu16-4g-coexist-20261024-08.service"',
                'UNIT = "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service"',
            ),
            (
                'PROFILE = "llama-cpp-cpu16-4g-coexist-command-owned-v1"',
                'PROFILE = "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"',
            ),
            ("CPU_CORES = frozenset(range(4, 20))", "CPU_CORES = frozenset(range(15, 20))"),
            ('CPU_ALLOWED_LIST = "4-19"', 'CPU_ALLOWED_LIST = "15-19"'),
            ('"CPUQuotaPerSecUSec": "16s"', '"CPUQuotaPerSecUSec": "5s"'),
            ("int(quota[0]) == 16 * int(quota[1])", "int(quota[0]) == 5 * int(quota[1])"),
            ("Actual CPU1600% quota kernel readback", "Actual CPU500% quota kernel readback"),
            (r"^Cpus_allowed_list:\s*4-19$", r"^Cpus_allowed_list:\s*15-19$"),
            ("sched/proc dual read is CPU4-19", "sched/proc dual read is CPU15-19"),
            ('"CPUQuota": "1600%"', '"CPUQuota": "500%"'),
            ('"cpu_quota_percent": 1600', '"cpu_quota_percent": 500'),
        )
        for before, after in replacements:
            self.assertEqual(expected.count(before), 1, before)
            expected = expected.replace(before, after)
        self.assertEqual((HERE / "controller.py").read_text(), expected)

    def test_affinity_requires_exact_five_high_capacity_selected_cores_and_stable_process(self):
        owner = NS(process=Mock(side_effect=[process(), process()]))
        with (
            patch.object(c.os, "sched_getaffinity", side_effect=[set(range(15, 20))] * 2),
            patch.object(Path, "read_text", return_value="Cpus_allowed_list:\t15-19\n"),
        ):
            actual = c.main_affinity(owner, 123)
        self.assertEqual(actual["sched_before"], [15, 16, 17, 18, 19])
        self.assertEqual(actual["sched_after"], actual["sched_before"])
        for first, second, declared, final in (
            (set(range(5, 10)), set(range(5, 10)), "5-9", process()),
            (set(range(4, 20)), set(range(4, 20)), "4-19", process()),
            (set(range(16, 20)), set(range(16, 20)), "16-19", process()),
            (set(range(15, 20)), set(range(20)), "15-19", process()),
            (set(range(15, 20)), set(range(15, 20)), "0-19", process()),
            (set(range(15, 20)), set(range(15, 20)), "15-19", process(start=21)),
        ):
            owner = NS(process=Mock(side_effect=[process(), final]))
            with (
                self.subTest(declared=declared, final=final),
                patch.object(c.os, "sched_getaffinity", side_effect=[first, second]),
                patch.object(
                    Path, "read_text", return_value="Cpus_allowed_list:\t" + declared + "\n"
                ),
                self.assertRaises(ValueError),
            ):
                c.main_affinity(owner, 123)

    def resource_readback(self, quota, manager="5s"):
        with tempfile.TemporaryDirectory() as temporary:
            group = Path(temporary) / "fake"
            group.mkdir()
            (group / "cpu.max").write_text(quota)
            (group / "memory.peak").write_text("1234")
            (group / "memory.events").write_text("low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n")
            owner = NS(
                CGROUP_ROOT=Path(temporary),
                command=Mock(
                    return_value="\n".join(
                        key + "=" + value
                        for key, value in {
                            "CPUQuotaPerSecUSec": manager,
                            "CPUAffinity": "15-19",
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

            def read(path, *args, **kwargs):
                if path == Path("/proc/123/limits"):
                    return (
                        "Max core file size 0 0 bytes\nMax file size "
                        + str(c.FILE_SIZE_MAX)
                        + " "
                        + str(c.FILE_SIZE_MAX)
                        + " bytes\nMax open files 512 512 files\n"
                    )
                return original(path, *args, **kwargs)

            with (
                patch.object(Path, "read_text", new=read),
                patch.object(c, "main_affinity", return_value={"FAKE": "separately tested"}),
            ):
                return c.properties(owner, NS(cgroup="/fake"), 123)

    def test_quota_preserves_five_times_any_positive_period_and_rejects_old_profiles(self):
        for quota in ("500000 100000", "1000000 200000"):
            self.assertEqual(self.resource_readback(quota)["kernel_cpu_max"], quota.split())
        for quota in (
            "1600000 100000",
            "400000 100000",
            "100000 100000",
            "max 100000",
            "500000 100001",
            "0 0",
        ):
            with self.subTest(quota=quota), self.assertRaisesRegex(ValueError, "CPU500%"):
                self.resource_readback(quota)
        for manager in ("4s", "16s", "1s"):
            with self.subTest(manager=manager), self.assertRaises(ValueError):
                self.resource_readback("500000 100000", manager)

    def test_four_gib_floor_reserve_authorities_and_unknown_host_policy_not_weakened(self):
        self.assertEqual(c.fixed_memory(), 4294967296)
        self.assertEqual(c.INITIAL_HOST_EXTRA_BYTES, 4294967296)
        self.assertEqual(c.reserve(130595864576), 26119172916)
        row = {
            "boot_id": "FAKE",
            "mem_total_bytes": 130595864576,
            "mem_available_bytes": 30414140212,
            "comfy_running": 0,
            "comfy_pending": 0,
            "comfy_state": "http_idle",
            "comfy_evidence": {
                "transport": "HTTP",
                "http_status": 200,
                "response_sha256": "a" * 64,
                "comfy_running": 0,
                "comfy_pending": 0,
            },
        }
        observer = NS(observe=lambda _boot: copy.deepcopy(row))
        self.assertEqual(
            c.host(observer, "FAKE", initial=True)["required_available_bytes"], 30414140212
        )
        row["mem_available_bytes"] -= 1
        with self.assertRaisesRegex(ValueError, "full selected4GiB"):
            c.host(observer, "FAKE", initial=True)
        row["mem_available_bytes"] += 1
        row["comfy_state"] = "unknown"
        with self.assertRaises(ValueError):
            c.host(observer, "FAKE", initial=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
