"""Pure stdlib fake-test SOURCE ONLY; not imported, executed or validated.

No Torch, Driver, H15 import, host read, model, service or real observer is used
by these fake scenarios. They do not attest actual28GiB capacity.
"""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

HERE = Path(__file__).absolute().parent
SPEC = importlib.util.spec_from_file_location("_native28_resource_candidate_test", HERE / "resource_gate.py")
g = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g)
TOTAL = 130595864576


def host():
    return {
        "boot_id": "FAKE",
        "MemTotal": TOTAL,
        "MemAvailable": TOTAL,
        "MemFree": 0,
        "Cached": 0,
        "SReclaimable": 0,
        "SwapTotal": 0,
        "SwapFree": 0,
        "Hugepagesize": 2 * g.MIB,
        "Hugetlb": 0,
        "HugePages_Total": 0,
        "HugePages_Free": 0,
    }


def unit():
    return {
        "path": g.EXPECTED_UNIT,
        "device": 1,
        "inode": 2,
        "actor": {"pid": 123, "start_ticks": 1, "pgid": 123, "sid": 123},
        "InvocationID": "a" * 32,
        "memory_max": g.CAP,
        "memory_high": "max",
        "memory_current": 1,
        "memory_peak": g.CAP,
        "swap_max": 0,
        "swap_current": 0,
        "events": {"max": 0, "oom": 0, "oom_kill": 0, "oom_group_kill": 0},
        "cgroup_events": {"populated": 1, "frozen": 0},
        "cgroup_stat": {"nr_descendants": 0, "nr_dying_descendants": 0},
        "members": [123],
        "properties": g._expected_properties(g.EXPECTED_UNIT, 123, "a" * 32),
    }


class Tests(unittest.TestCase):
    def test_explicit_28_bounds_and_both_floors_without_original_maximum(self):
        with patch.object(g, "mechanism", side_effect=AssertionError("no frozen40 helper")):
            startup = g.floor(host(), loaded=False)
            loaded = g.floor(host(), loaded=True)
        self.assertEqual(g.CAP, 30064771072)
        self.assertEqual(g.STOP, 29527900160)
        self.assertEqual(g.ORIGINAL_CUDA_FREE_REQUIRED, 41 * g.GIB)
        self.assertEqual(startup["reserve_bytes"], 26119172916)
        self.assertEqual(startup["required_bytes"], 90543682356)
        self.assertEqual(loaded["required_bytes"], 64773878580)
        self.assertEqual(startup["old_train_required_bytes"], 106438961908)
        self.assertEqual(loaded["old_train_required_bytes"], startup["old_train_required_bytes"])
        self.assertLess(startup["required_bytes"], startup["old_train_required_bytes"])
        for invalid in (None, 0, 1, "loaded"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                g.floor(host(), loaded=invalid)

    def fake_torch(self):
        cuda = NS(
            device_count=lambda: 1,
            current_device=lambda: 0,
            get_device_properties=lambda _: NS(
                uuid=g.EXPECTED_UUID, name="NVIDIA GB10", total_memory=TOTAL
            ),
            memory=NS(get_allocator_backend=lambda: "native"),
        )
        cuda.fraction = None
        cuda.set_per_process_memory_fraction = Mock(
            side_effect=lambda fraction, _: setattr(cuda, "fraction", fraction)
        )
        cuda.get_per_process_memory_fraction = lambda _: cuda.fraction
        return NS(cuda=cuda)

    def test_own_allocator_initializes_only_28_with_conservative_readback(self):
        torch = self.fake_torch()
        with patch.object(g, "mechanism", side_effect=AssertionError("no frozen40 initializer")):
            result = g.allocator_cap(torch, g.EXPECTED_UUID)
        torch.cuda.set_per_process_memory_fraction.assert_called_once_with(
            (g.CAP - 4 * g.MIB) / TOTAL, 0
        )
        self.assertLessEqual(result["allowed_bytes_upper"], g.CAP)
        self.assertGreaterEqual(result["allowed_bytes_upper"], g.CAP - 8 * g.MIB)
        torch.cuda.get_per_process_memory_fraction = lambda _: 1.0
        with self.assertRaisesRegex(ValueError, "28GiB native allocator fraction"):
            g.allocator_cap(torch, g.EXPECTED_UUID)

    def test_allocator_wrong_uuid_or_backend_refuses_before_setting_fraction(self):
        for wrong_uuid, wrong_backend in ((True, False), (False, True)):
            torch = self.fake_torch()
            if wrong_backend:
                torch.cuda.memory.get_allocator_backend = lambda: "cudaMallocAsync"
            with self.assertRaises(ValueError):
                g.allocator_cap(torch, "wrong" if wrong_uuid else g.EXPECTED_UUID)
            torch.cuda.set_per_process_memory_fraction.assert_not_called()

    def test_memcg_stop_equality_and_old_cap_refuse_without_resetting_peaks(self):
        value = unit()
        value["memory_current"] = g.STOP - 1
        g.validate_unit(value, g.EXPECTED_UNIT)
        self.assertEqual(value["memory_peak"], g.CAP)
        for current in (g.STOP, g.STOP + 1):
            value["memory_current"] = current
            with self.subTest(current=current), self.assertRaisesRegex(ValueError, "27.5GiB"):
                g.validate_unit(value, g.EXPECTED_UNIT)
        value = unit()
        value["memory_max"] = 40 * g.GIB
        with self.assertRaisesRegex(ValueError, "28GiB"):
            g.validate_unit(value, g.EXPECTED_UNIT)
        value = unit()
        value["events"]["oom_kill"] = 1
        with self.assertRaisesRegex(ValueError, "pressure/OOM"):
            g.validate_unit(value, g.EXPECTED_UNIT)
        self.assertEqual(value["events"]["oom_kill"], 1)

    def test_constructor_refusal_attaches_unmodified_unit_peak_and_oom(self):
        value = unit()
        value["events"]["oom_kill"] = 1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "pressure/OOM") as captured:
                g.Gate(
                    Path(directory) / "journal.jsonl", "FAKE", observer=host,
                    unit_reader=lambda _: copy.deepcopy(value),
                )
        evidence = captured.exception.native28_resource_evidence
        self.assertFalse(evidence["admitted"])
        self.assertEqual(evidence["unit"], value)

    def test_phase_stop_or_oom_retains_actual_failure_readback(self):
        for oom in (False, True):
            with self.subTest(oom=oom), tempfile.TemporaryDirectory() as directory:
                good, failed = unit(), unit()
                failed["memory_current"] = g.STOP if not oom else 1
                failed["events"]["oom_kill"] = int(oom)
                reader = Mock(side_effect=[good, failed])
                path = Path(directory) / "journal.jsonl"
                gate = g.Gate(path, "FAKE", observer=host, unit_reader=reader)
                with patch.object(g, "phase_scope", return_value="cpu"):
                    with self.assertRaises(ValueError):
                        gate.check("fake-resource-only", "cpu")
                row = json.loads(path.read_bytes())
                self.assertFalse(row["admitted"])
                self.assertEqual(row["unit"], failed)
                self.assertEqual(row["unit"]["memory_peak"], g.CAP)
                self.assertIsNotNone(row["error"])

    def test_cuda_reserved_stop_equality_keeps_raw_high_water_values(self):
        fraction = (g.CAP - 4 * g.MIB) / TOTAL
        native = {
            "identity": {
                "context_device": 0, "integrated": 1, "uuid": g.EXPECTED_UUID,
                "name": "NVIDIA GB10", "total_bytes": TOTAL, "boot_id": "FAKE",
                "library_metadata": {}, "symbol_addresses": {},
                "load_mode": "exact-canonical-RTLD_NOLOAD", "pin_scope": "FAKE",
            },
            "allocator": {
                "uuid": g.EXPECTED_UUID, "name": "NVIDIA GB10", "total_bytes": TOTAL,
                "fraction": fraction, "backend": "native",
                "allowed_bytes_upper": int(fraction * TOTAL),
            },
            "cuda": {
                "free": TOTAL, "total": TOTAL, "allocated": 1,
                "reserved": g.STOP, "peak_allocated": g.CAP, "peak_reserved": g.CAP,
            },
        }
        before = copy.deepcopy(native)
        # Identity machinery has separate unchanged-source checks; this fake
        # isolates the actual CUDA bound validation without reading a Driver.
        with patch.object(g, "mechanism", return_value=NS()), patch.object(
            g, "uma_identity", return_value=None
        ):
            with self.assertRaisesRegex(ValueError, "27.5GiB"):
                g.validate_native(native, host(), "FAKE")
        self.assertEqual(native, before)


if __name__ == "__main__":
    unittest.main()
