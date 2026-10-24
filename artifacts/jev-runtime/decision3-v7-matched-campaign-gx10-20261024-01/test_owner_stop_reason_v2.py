"""Stdlib fakes: keep the first trigger when pump cleanup fails.

The existing invented runtime fixture supplies temporary files and mocks every
process, systemd, host, ComfyUI, cgroup and cache operation. No actual runtime
output, model, browser or network is used.
"""

import contextlib
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch


def source(filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location("_stop_reason_fake_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StopReasonV2Tests(unittest.TestCase):
    def setUp(self):
        previous = source("test_owner_held.py")
        self.fixture = previous.OwnerSourceTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.owner = source("owner_held_v2.py")
        # Reuse only invented fixture paths. The old owner and tests are intact.
        self.owner.ROOT = self.fixture.root
        self.owner.HERE = self.fixture.here
        self.owner.CGROUP_ROOT = Path(self.fixture.temporary.name) / "cgroups"
        self.fixture.owner = self.owner
        self.host_row = previous.host_row

    def run_with_failure(self, *, trigger, monitor_error=None):
        cleanup = OSError("invented CPU closure failure")
        fake = self.fixture.runtime(
            busy=trigger != "terminal", monitor_error=monitor_error, pump_close_error=cleanup
        )
        with self.fixture.patched_runtime(fake), contextlib.ExitStack() as stack:
            if trigger == "host_reserve_margin_crossed":
                stack.enter_context(
                    patch.object(
                        self.owner,
                        "observe_host",
                        side_effect=[
                            self.host_row(),
                            {**self.host_row(), "mem_available_bytes": 1},
                            self.host_row(),
                        ],
                    )
                )
            elif trigger == "owner_deadline":
                stack.enter_context(
                    patch.object(self.owner, "observe_host", return_value=self.host_row())
                )
                stack.enter_context(patch.object(self.owner, "OWNER_STOP_SECONDS", -1))
            verify = stack.enter_context(patch.object(self.owner, "verify_closed"))
            expected_error = monitor_error if monitor_error is not None else cleanup
            with self.assertRaises(type(expected_error)) as raised:
                self.owner.execute(fake.plan, fake.digest, fake.guard, {}, plan_blob=fake.blob)
            self.assertIs(raised.exception, expected_error)
            verify.assert_not_called()  # Cleanup failure never authorizes success.
        result = json.loads((fake.directory / "root-external-closure.json").read_bytes())
        self.assertTrue(fake.state["stopped"])
        self.assertLess(fake.events.index("close-candidate-pump"), fake.events.index("stop-outer"))
        self.assertEqual(result["initial_stop_trigger"], trigger)
        self.assertEqual(result["candidate_pump_closure_error"]["type"], "OSError")
        self.assertEqual(result["primary_error"]["type"], type(expected_error).__name__)
        return result, expected_error

    def test_comfy_stop_remains_visible_when_cpu_cleanup_also_fails(self):
        reason = "Comfy_became_busy_stop_own_unit_only"
        result, _ = self.run_with_failure(trigger=reason)
        self.assertEqual(result["reason"], reason)

    def test_resource_stop_remains_visible_when_cpu_cleanup_also_fails(self):
        reason = "host_reserve_margin_crossed"
        result, _ = self.run_with_failure(trigger=reason)
        self.assertEqual(result["reason"], reason)

    def test_deadline_stop_remains_visible_when_cpu_cleanup_also_fails(self):
        reason = "owner_deadline"
        result, _ = self.run_with_failure(trigger=reason)
        self.assertEqual(result["reason"], reason)

    def test_monitor_primary_is_retained_with_both_error_records(self):
        primary = KeyboardInterrupt("invented monitor abort")
        result, error = self.run_with_failure(trigger="owner_error", monitor_error=primary)
        self.assertEqual(result["reason"], "owner_error")
        self.assertIs(error, primary)
        self.assertTrue(any("Candidate pump closure" in note for note in primary.__notes__))

    def test_terminal_trigger_is_distinct_from_fatal_cleanup_reason(self):
        result, _ = self.run_with_failure(trigger="terminal")
        self.assertEqual(result["reason"], "candidate_pump_closure_error")


if __name__ == "__main__":
    unittest.main()
