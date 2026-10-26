"""Authored stdlib fakes; not executed by this source-only author.

No process, selected runtime output, framework, browser or HTTP is used. The
invented owner-completion stub does not stand in for actual closure evidence.
"""

import importlib.util
import io
import signal
import unittest
from pathlib import Path
from unittest import mock

PATH = Path(__file__).with_name("pump_held_v3.py")
SPEC = importlib.util.spec_from_file_location("_held_v3_pump_close_source_fakes", PATH)
pump = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pump)


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        if not 0 < seconds <= 0.05:
            raise AssertionError("Only bounded polling sleeps are permitted")
        self.sleeps.append(seconds)
        self.now += seconds


class FakeController:
    """One invented held Popen with natural exit and optional signal response."""

    def __init__(self, clock, *, natural_exit=None, signal_delay=1.0, code=0):
        self.clock = clock
        self.pid = 20
        self.exit_at = natural_exit
        self.signal_delay = signal_delay
        self.code = code
        self.signal_times = []
        self.send_signal = mock.Mock(side_effect=self._signal)
        self.kill = mock.Mock(side_effect=AssertionError("Never kill the CPU controller"))

    def poll(self):
        return self.code if self.exit_at is not None and self.clock.now >= self.exit_at else None

    def _signal(self, selected):
        if selected != signal.SIGINT:
            raise AssertionError("Only the original SIGINT cleanup fallback is permitted")
        self.signal_times.append(self.clock.now)
        if self.signal_delay is not None:
            self.exit_at = self.clock.now + self.signal_delay


def invented_subject(clock, process):
    subject = object.__new__(pump.Pump)
    subject.closed, subject.failure, subject.records = False, None, []
    identity = {"pid": 20, "start_ticks": 7, "pgid": 20, "sid": 20, "cgroup": "root"}
    active = {"process": process, "identity": identity, "stream": io.BytesIO()}
    subject.active = active
    subject._stop = mock.Mock()

    def authenticated_finish(value, *, complete):
        if value is not active or process.poll() is None or complete is not False:
            raise AssertionError("Cleanup requires a terminal original controller and partial mode")
        # This stands only for a successful call to the unchanged _finish().
        # Production _finish independently authenticates original owner closure.
        subject.records.append(
            {
                "original_scoped_closure_positive": True,
                "owner_completed": process.code == 0,
                "controller_exit_code": process.code,
                "interrupted_for_cleanup": True,
            }
        )
        subject.active = None

    subject._finish = mock.Mock(side_effect=authenticated_finish)
    patches = (
        mock.patch.object(pump, "released"),
        mock.patch.object(pump, "identity", return_value=identity),
        mock.patch.object(pump.time, "monotonic", side_effect=clock.monotonic),
        mock.patch.object(pump.time, "sleep", side_effect=clock.sleep),
    )
    return subject, active, identity, patches


class CooperativeCloseFakes(unittest.TestCase):
    def enter_patches(self, patches):
        return [self.enterContext(patch) for patch in patches]

    def test_source_hold_still_precedes_close_io(self):
        subject = object.__new__(pump.Pump)
        with (
            mock.patch.object(pump, "SOURCE_ONLY", True),
            mock.patch.object(pump.time, "monotonic", side_effect=AssertionError("clock")),
            mock.patch.object(pump, "identity", side_effect=AssertionError("process read")),
        ):
            with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
                subject.close()

    def test_cooperative_exit_after_stop_avoids_sigint_and_stays_partial(self):
        clock = FakeClock()
        process = FakeController(clock, natural_exit=3.0)
        subject, active, _, patches = invented_subject(clock, process)
        mocked = self.enter_patches(patches)
        result = subject.close()
        subject._stop.assert_called_once_with(active, forced=True)
        process.send_signal.assert_not_called()
        process.kill.assert_not_called()
        mocked[1].assert_not_called()
        subject._finish.assert_called_once_with(active, complete=False)
        self.assertGreaterEqual(clock.now, 3.0)
        self.assertLess(clock.now, 3.051)
        self.assertTrue(result["active_controller_absent"])
        self.assertTrue(result["independently_closed"])
        self.assertFalse(result["complete_ten_cases"])
        self.assertTrue(result["children"][0]["interrupted_for_cleanup"])

    def test_controller_exit_during_stop_publication_is_not_signalled(self):
        clock = FakeClock()
        process = FakeController(clock)
        subject, active, _, patches = invented_subject(clock, process)
        mocked = self.enter_patches(patches)

        def stop(_active, *, forced):
            self.assertIs(_active, active)
            self.assertTrue(forced)
            process.exit_at = clock.now

        subject._stop.side_effect = stop
        subject.close()
        self.assertEqual(clock.now, 0.0)
        process.send_signal.assert_not_called()
        mocked[1].assert_not_called()

    def test_grace_expiry_uses_original_identity_checked_sigint(self):
        clock = FakeClock()
        process = FakeController(clock, signal_delay=1.0, code=1)
        subject, active, _, patches = invented_subject(clock, process)
        mocked = self.enter_patches(patches)
        result = subject.close()
        self.assertEqual(process.signal_times, [pump.COOPERATIVE_CLOSE_SECONDS])
        process.send_signal.assert_called_once_with(signal.SIGINT)
        mocked[1].assert_called_once_with(process.pid)
        self.assertLessEqual(clock.now, pump.CLOSE_SECONDS)
        subject._finish.assert_called_once_with(active, complete=False)
        self.assertTrue(result["active_controller_absent"])
        self.assertFalse(result["complete_ten_cases"])
        process.kill.assert_not_called()

    def test_stop_write_failure_skips_grace_and_preserves_primary(self):
        clock = FakeClock()
        process = FakeController(clock, signal_delay=1.0, code=1)
        subject, active, _, patches = invented_subject(clock, process)
        self.enter_patches(patches)
        original = OSError("invented stop durability failure")
        subject._stop.side_effect = original
        with self.assertRaises(OSError) as found:
            subject.close()
        self.assertIs(found.exception, original)
        self.assertEqual(process.signal_times, [0.0])
        process.send_signal.assert_called_once_with(signal.SIGINT)
        process.kill.assert_not_called()
        subject._finish.assert_called_once_with(active, complete=False)
        self.assertTrue(original.jev_v7_pump_closure["active_controller_absent"])
        self.assertFalse(original.jev_v7_pump_closure["complete_ten_cases"])

    def test_hung_controller_keeps_original_total_wait_and_has_no_closure(self):
        clock = FakeClock()
        process = FakeController(clock, signal_delay=None)
        subject, _, _, patches = invented_subject(clock, process)
        self.enter_patches(patches)
        with self.assertRaisesRegex(ValueError, "remains live"):
            subject.close()
        self.assertEqual(clock.now, pump.CLOSE_SECONDS)
        self.assertEqual(process.signal_times, [pump.COOPERATIVE_CLOSE_SECONDS])
        self.assertFalse(subject.closed)
        self.assertIsNotNone(subject.active)
        subject._finish.assert_not_called()
        process.kill.assert_not_called()

    def test_changed_controller_identity_cannot_receive_fallback_signal(self):
        clock = FakeClock()
        process = FakeController(clock, signal_delay=None)
        subject, _, expected, patches = invented_subject(clock, process)
        mocked = self.enter_patches(patches)
        mocked[1].return_value = {**expected, "start_ticks": expected["start_ticks"] + 1}
        with self.assertRaisesRegex(ValueError, "same held controller before SIGINT"):
            subject.close()
        process.send_signal.assert_not_called()
        process.kill.assert_not_called()
        subject._finish.assert_not_called()
        self.assertFalse(subject.closed)

    def test_grace_exit_still_requires_original_owner_closure_verification(self):
        clock = FakeClock()
        process = FakeController(clock, natural_exit=1.0)
        subject, active, _, patches = invented_subject(clock, process)
        self.enter_patches(patches)
        original = ValueError("invented original owner closure rejection")
        subject._finish.side_effect = original
        with self.assertRaises(ValueError) as found:
            subject.close()
        self.assertIs(found.exception, original)
        subject._finish.assert_called_once_with(active, complete=False)
        self.assertFalse(subject.closed)
        self.assertEqual(subject.records, [])
        process.send_signal.assert_not_called()

    def test_stop_publication_time_is_charged_before_grace_fallback(self):
        clock = FakeClock()
        process = FakeController(clock, signal_delay=1.0, code=1)
        subject, _, _, patches = invented_subject(clock, process)
        self.enter_patches(patches)
        subject._stop.side_effect = lambda *_args, **_kwargs: setattr(clock, "now", 30.0)
        subject.close()
        self.assertEqual(process.signal_times, [30.0])
        self.assertGreaterEqual(clock.now, 31.0)
        self.assertLessEqual(clock.now, pump.CLOSE_SECONDS)

    def test_already_exited_controller_has_no_stop_wait_or_signal(self):
        clock = FakeClock()
        process = FakeController(clock, natural_exit=0.0)
        subject, active, _, patches = invented_subject(clock, process)
        self.enter_patches(patches)
        subject.close()
        subject._stop.assert_not_called()
        subject._finish.assert_called_once_with(active, complete=False)
        self.assertEqual(clock.sleeps, [])
        process.send_signal.assert_not_called()

    def test_successful_cleanup_close_is_idempotent(self):
        clock = FakeClock()
        process = FakeController(clock, natural_exit=1.0)
        subject, _, _, patches = invented_subject(clock, process)
        self.enter_patches(patches)
        result = subject.close()
        elapsed = clock.now
        self.assertIs(subject.close(), result)
        self.assertEqual(clock.now, elapsed)
        self.assertEqual(subject._stop.call_count, 1)
        self.assertEqual(subject._finish.call_count, 1)

    def test_expired_poll_deadline_does_not_sleep_or_invent_exit(self):
        clock = FakeClock()
        process = FakeController(clock)
        with (
            mock.patch.object(pump.time, "monotonic", side_effect=clock.monotonic),
            mock.patch.object(pump.time, "sleep", side_effect=clock.sleep),
        ):
            self.assertFalse(pump.wait_controller_exit(process, 0.0))
        self.assertEqual(clock.sleeps, [])
        process.send_signal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
