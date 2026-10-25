"""Invented stdlib controls for the held shared GPU serializer.

These fixtures are not Torch, GPU, model, browser or resource qualification.
No framework is imported. Root may run them after reviewing the held sources.
"""

from __future__ import annotations

import contextlib
import fcntl
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "jev_gpu_forward_lease.py"


def fresh_module(*, released=True):
    spec = importlib.util.spec_from_file_location("_invented_gpu_serial_controls", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if released:
        module.SOURCE_ONLY = False
    return module


class Poison:
    def __getattribute__(self, name):
        raise AssertionError("Held source touched dependency: " + name)

    def __fspath__(self):
        raise AssertionError("Held source touched path")


@contextlib.contextmanager
def invented_torch(on_fence=None):
    """Named registered module fixtures; deliberately no framework provenance."""
    torch, cuda = types.ModuleType("torch"), types.ModuleType("torch.cuda")
    events = []

    def synchronize(device):
        events.append(("cuda-fence", device))
        if on_fence is not None:
            on_fence(device)

    synchronize.__module__, synchronize.__name__ = "torch.cuda", "synchronize"
    cuda.synchronize = synchronize
    cuda.is_available = lambda: True
    cuda.is_initialized = lambda: True
    cuda.device_count = lambda: 1
    cuda.current_device = lambda: 0
    torch.cuda = cuda
    with mock.patch.dict(sys.modules, {"torch": torch, "torch.cuda": cuda}):
        yield torch, events


class Fixture:
    def __init__(self, module):
        self.module = module
        self.temporary = tempfile.TemporaryDirectory(prefix="invented-gpu-lease-")
        self.directory = Path(self.temporary.name)
        self.lock, self.journal = self.directory / "lock", self.directory / "journal"
        self.lock.write_bytes(b"")
        self.journal.write_bytes(b"")
        expected = {
            name: {"device": path.stat().st_dev, "inode": path.stat().st_ino}
            for name, path in (("lock", self.lock), ("journal", self.journal))
        }
        self.lease = module.ForwardLease(self.lock, self.journal, expected, nonce="a" * 32)

    def records(self):
        return [json.loads(line) for line in self.journal.read_text().splitlines()]

    def assert_locked(self, test):
        test.assertIsNotNone(self.lease.fd)
        fd = os.open(self.lock, os.O_RDWR)
        try:
            with test.assertRaises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)

    def close(self):
        # Test-only stand-in for process exit, never a serializer recovery API.
        if self.lease.fd is not None:
            os.close(self.lease.fd)
            self.lease.fd, self.lease.held = None, None
        self.temporary.cleanup()


class Provider:
    name = "invented-complete-operation-composer"
    spec = object()

    def __init__(self, action):
        self.action, self.calls = action, []

    def predict(self, request, **kwargs):
        self.calls.append((request, kwargs))
        return self.action(request, kwargs)


class HookHandle:
    def __init__(self, events, label):
        self.events, self.label = events, label

    def remove(self):
        self.events.append(("remove", self.label))


class CompleteCausalLM:
    def __init__(self, events):
        self.events, self.before, self.after = events, None, None

    def register_forward_pre_hook(self, callback, *, with_kwargs):
        assert with_kwargs is True
        self.before = callback
        self.events.append(("install", "complete-causal-lm-pre"))
        return HookHandle(self.events, "complete-causal-lm-pre")

    def register_forward_hook(self, callback, *, with_kwargs, always_call):
        assert with_kwargs is True and always_call is True
        self.after = callback
        self.events.append(("install", "complete-causal-lm-post"))
        return HookHandle(self.events, "complete-causal-lm-post")


class SerializerControls(unittest.TestCase):
    def setUp(self):
        self.module = fresh_module()
        self.fixture = Fixture(self.module)
        self.addCleanup(self.fixture.close)

    def test_held_gate_precedes_paths_framework_and_hook_dependencies(self):
        held, poison = fresh_module(released=False), Poison()
        with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
            held.ForwardLease(poison, poison, poison, nonce=poison)
        with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
            held.SerialBackend(poison, poison, torch=poison)
        with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
            held.install_nf4_hooks(
                poison,
                poison,
                poison,
                poison,
                resource_gate=poison,
                campaign_deadline=poison,
                register_original=poison,
            )

    def test_missing_torch_and_generic_noop_fences_are_refused(self):
        provider = Provider(lambda _request, _kwargs: None)
        with self.assertRaises(TypeError):
            self.module.SerialBackend(provider, self.fixture.lease)
        with self.assertRaisesRegex(RuntimeError, "Torch module"):
            self.module.SerialBackend(provider, self.fixture.lease, torch=lambda: None)
        with invented_torch() as (torch, _events):
            torch.cuda.synchronize = lambda _device: None
            with self.assertRaisesRegex(RuntimeError, "generic/no-op"):
                self.module.SerialBackend(provider, self.fixture.lease, torch=torch)
        self.assertEqual(provider.calls, [])
        self.assertEqual(self.fixture.records(), [])

    def test_cpu_uninitialized_or_multiple_device_fallback_is_refused(self):
        provider = Provider(lambda _request, _kwargs: None)
        for key, value in (
            ("is_available", False),
            ("is_initialized", False),
            ("device_count", 2),
            ("current_device", 1),
        ):
            with self.subTest(key=key), invented_torch() as (torch, _events):
                setattr(torch.cuda, key, lambda value=value: value)
                with self.assertRaisesRegex(RuntimeError, "single CUDA device0"):
                    self.module.SerialBackend(provider, self.fixture.lease, torch=torch)
        self.assertEqual(provider.calls, [])

    def test_whole_composition_witness_terminal_and_fence_stay_inside_lease(self):
        phases, request, response = [], object(), {"invented": "composed-response"}
        lease = self.fixture.lease

        def deadline():
            phases.append("deadline")

        def evidence_sink(event, evidence):
            self.fixture.assert_locked(self)
            self.assertEqual(lease.held[0], "decision3-gpu")
            phases.append(event)
            self.assertEqual(evidence, {"invented": True})

        def compose(actual_request, kwargs):
            self.assertIs(actual_request, request)
            self.assertIs(kwargs["deadline"], deadline)
            self.assertIs(kwargs["evidence_sink"], evidence_sink)
            self.fixture.assert_locked(self)
            phases.append("baseline-forward")
            phases.append("projection")
            self.fixture.assert_locked(self)
            phases.append("private-witness-sink")
            kwargs["evidence_sink"]("composed-terminal-sink", {"invented": True})
            self.fixture.assert_locked(self)
            phases.append("final-composition-guard")
            return response

        def fence(device):
            self.assertEqual(device, 0)
            self.fixture.assert_locked(self)
            self.assertEqual(phases[-1], "final-composition-guard")
            phases.append("cuda-fence")

        original_emit = lease._emit

        def emit(event, kind):
            self.fixture.assert_locked(self)
            if event == "completed":
                self.assertEqual(phases[-2:], ["cuda-fence", "deadline"])
            phases.append("journal-" + event)
            return original_emit(event, kind)

        lease._emit = emit
        provider = Provider(compose)
        with invented_torch(fence) as (torch, cuda_events):
            serial = self.module.SerialBackend(provider, lease, torch=torch)
            self.assertIs(
                serial.predict(request, deadline=deadline, evidence_sink=evidence_sink), response
            )
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(cuda_events, [("cuda-fence", 0)])
        self.assertEqual(phases[-2:], ["journal-completed", "deadline"])
        self.assertIsNone(lease.fd)
        self.assertIsNone(lease.fatal)
        rows = self.fixture.records()
        self.assertEqual(
            [(r["event"], r["kind"]) for r in rows],
            [("acquired", "decision3-gpu"), ("completed", "decision3-gpu")],
        )
        self.assertTrue(all(r["nonce"] == "a" * 32 for r in rows))

    def test_completed_journal_fsync_reaching_exact20_retains_failed_lease(self):
        lease, now, current_event = self.fixture.lease, [0.0], [None]
        timeout = TimeoutError("invented outer HTTP-entry20 expired")

        def deadline():
            if now[0] >= 20:
                raise timeout

        original_emit, original_fsync = lease._emit, self.module.os.fsync

        def emit(event, kind):
            current_event[0] = event
            return original_emit(event, kind)

        def fsync(fd):
            original_fsync(fd)
            if current_event[0] == "completed":
                now[0] = 20.0

        lease._emit = emit
        provider = Provider(lambda _request, _kwargs: {"invented": True})
        with (
            invented_torch() as (torch, _events),
            mock.patch.object(self.module.os, "fsync", fsync),
        ):
            serial = self.module.SerialBackend(provider, lease, torch=torch)
            with self.assertRaises(TimeoutError) as captured:
                serial.predict(object(), deadline=deadline)
        self.assertIs(captured.exception, timeout)
        self.assertIs(lease.fatal, timeout)
        self.fixture.assert_locked(self)
        self.assertEqual([r["event"] for r in self.fixture.records()], ["acquired", "completed"])
        with self.assertRaisesRegex(RuntimeError, "latched"):
            lease.check()

    def test_cuda_fence_abort_remains_primary_when_retention_also_fails(self):
        lease, primary = self.fixture.lease, KeyboardInterrupt("invented GPU fence abort")
        original_fail = lease.fail

        def retain_then_fail(error):
            original_fail(error)
            raise OSError("invented secondary failure journal")

        def fence(_device):
            self.fixture.assert_locked(self)
            raise primary

        lease.fail = retain_then_fail
        provider = Provider(lambda _request, _kwargs: {"invented": True})
        with invented_torch(fence) as (torch, _events):
            serial = self.module.SerialBackend(provider, lease, torch=torch)
            with self.assertRaises(KeyboardInterrupt) as captured:
                serial.predict(object(), deadline=lambda: None)
        self.assertIs(captured.exception, primary)
        self.assertIs(lease.fatal, primary)
        self.fixture.assert_locked(self)
        self.assertTrue(any("secondary failure journal" in note for note in primary.__notes__))
        self.assertEqual([r["event"] for r in self.fixture.records()], ["acquired"])

    def test_composer_abort_remains_primary_and_does_not_run_release_fence(self):
        lease, primary = self.fixture.lease, KeyboardInterrupt("invented composed callback abort")
        original_fail = lease.fail

        def compose(_request, _kwargs):
            self.fixture.assert_locked(self)
            raise primary

        def retain_then_fail(error):
            original_fail(error)
            raise OSError("invented secondary retention error")

        lease.fail = retain_then_fail
        with invented_torch(lambda _device: self.fail("No release fence after original abort")) as (
            torch,
            cuda_events,
        ):
            serial = self.module.SerialBackend(Provider(compose), lease, torch=torch)
            with self.assertRaises(KeyboardInterrupt) as captured:
                serial.predict(object(), deadline=lambda: None)
        self.assertIs(captured.exception, primary)
        self.assertIs(lease.fatal, primary)
        self.assertEqual(cuda_events, [])
        self.fixture.assert_locked(self)
        self.assertTrue(any("secondary retention error" in note for note in primary.__notes__))

    def test_required_composed_sink_failure_retains_lease(self):
        lease, primary = self.fixture.lease, OSError("invented required composed evidence sink")

        def sink(_event, _evidence):
            self.fixture.assert_locked(self)
            raise primary

        def compose(_request, kwargs):
            kwargs["evidence_sink"]("composed-native-raw-returned", {"invented": True})
            self.fail("Required sink error must propagate")

        with invented_torch() as (torch, cuda_events):
            serial = self.module.SerialBackend(Provider(compose), lease, torch=torch)
            with self.assertRaises(OSError) as captured:
                serial.predict(object(), deadline=lambda: None, evidence_sink=sink)
        self.assertIs(captured.exception, primary)
        self.assertIs(lease.fatal, primary)
        self.assertEqual(cuda_events, [])
        self.fixture.assert_locked(self)

    def test_lock_wait_uses_original_deadline_and_never_enters_composer(self):
        lease, calls, timeout = self.fixture.lease, [0], TimeoutError("invented blocked lease")
        other = os.open(self.fixture.lock, os.O_RDWR)
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.addCleanup(os.close, other)

        def deadline():
            calls[0] += 1
            if calls[0] >= 3:
                raise timeout

        provider = Provider(lambda _request, _kwargs: self.fail("Composer entered while blocked"))
        with (
            invented_torch() as (torch, _events),
            mock.patch.object(self.module.time, "sleep", lambda _seconds: None),
        ):
            serial = self.module.SerialBackend(provider, lease, torch=torch)
            with self.assertRaises(TimeoutError) as captured:
                serial.predict(object(), deadline=deadline)
        self.assertIs(captured.exception, timeout)
        self.assertEqual(provider.calls, [])
        self.assertIs(lease.fatal, timeout)
        self.assertIsNone(lease.fd)
        self.assertEqual(self.fixture.records(), [])

    def test_changed_lock_inode_is_refused_before_composer_entry(self):
        old = os.open(self.fixture.lock, os.O_RDONLY)
        self.addCleanup(os.close, old)
        self.fixture.lock.unlink()
        self.fixture.lock.write_bytes(b"invented replacement")
        provider = Provider(lambda _request, _kwargs: self.fail("Changed lock admitted"))
        with invented_torch() as (torch, _events):
            serial = self.module.SerialBackend(provider, self.fixture.lease, torch=torch)
            with self.assertRaisesRegex(RuntimeError, "original shared lock inode"):
                serial.predict(object(), deadline=lambda: None)
        self.assertEqual(provider.calls, [])
        self.assertIsNone(self.fixture.lease.fd)
        self.assertIsNotNone(self.fixture.lease.fatal)

    def test_changed_completed_journal_inode_fails_with_lock_retained(self):
        old = os.open(self.fixture.journal, os.O_RDONLY)
        self.addCleanup(os.close, old)

        def compose(_request, _kwargs):
            self.fixture.journal.unlink()
            self.fixture.journal.write_bytes(b"")
            return {"invented": True}

        with invented_torch() as (torch, cuda_events):
            serial = self.module.SerialBackend(Provider(compose), self.fixture.lease, torch=torch)
            with self.assertRaisesRegex(RuntimeError, "Original shared journal inode"):
                serial.predict(object(), deadline=lambda: None)
        self.assertEqual(cuda_events, [("cuda-fence", 0)])
        self.assertIsNotNone(self.fixture.lease.fatal)
        self.fixture.assert_locked(self)

    def test_original_journal_abort_survives_secondary_journal_close_error(self):
        lease, primary = self.fixture.lease, KeyboardInterrupt("invented durable journal abort")
        original_close = self.module.os.close

        def write(_fd, _raw):
            raise primary

        def close(fd):
            original_close(fd)
            raise OSError("invented secondary journal-close error")

        with (
            mock.patch.object(self.module.os, "write", write),
            mock.patch.object(self.module.os, "close", close),
        ):
            with self.assertRaises(KeyboardInterrupt) as captured:
                lease.begin("decision3-gpu", lambda: None)
        self.assertIs(captured.exception, primary)
        self.assertIs(lease.fatal, primary)
        self.fixture.assert_locked(self)
        self.assertTrue(any("secondary journal-close error" in note for note in primary.__notes__))

    def test_changed_selected_cuda_entry_is_refused_before_release(self):
        lease = self.fixture.lease
        with invented_torch() as (torch, cuda_events):

            def compose(_request, _kwargs):
                torch.cuda.synchronize = lambda _device: None
                return {"invented": True}

            serial = self.module.SerialBackend(Provider(compose), lease, torch=torch)
            with self.assertRaisesRegex(RuntimeError, "generic/no-op"):
                serial.predict(object(), deadline=lambda: None)
        self.assertEqual(cuda_events, [])
        self.assertIsNotNone(lease.fatal)
        self.fixture.assert_locked(self)

    def test_source_guard_failure_in_original_deadline_retains_lease_after_fence(self):
        lease, completed = self.fixture.lease, [False]
        primary = RuntimeError("invented selected source guard changed")

        def compose(_request, _kwargs):
            completed[0] = True
            return {"invented": True}

        def deadline():
            if completed[0]:
                raise primary

        with invented_torch() as (torch, cuda_events):
            serial = self.module.SerialBackend(Provider(compose), lease, torch=torch)
            with self.assertRaises(RuntimeError) as captured:
                serial.predict(object(), deadline=deadline)
        self.assertIs(captured.exception, primary)
        self.assertIs(lease.fatal, primary)
        self.assertEqual(cuda_events, [("cuda-fence", 0)])
        self.fixture.assert_locked(self)
        self.assertEqual([r["event"] for r in self.fixture.records()], ["acquired"])

    def helper_fixture(self, trace):
        events, complete = [], None
        complete = CompleteCausalLM(events)
        model = types.SimpleNamespace(
            backbone=object(), lm=types.SimpleNamespace(get_base_model=lambda: complete)
        )
        native = types.SimpleNamespace(
            TRACE=types.SimpleNamespace(get=lambda: trace),
            http_deadline=lambda started: events.append(("http-deadline", started)),
        )

        def register_original(actual_model, gate):
            self.assertIs(actual_model, model)
            self.assertEqual(gate, "invented-resource-gate")
            events.append(("install", "original-resource-hook"))
            return HookHandle(events, "original-resource-hook")

        return model, native, complete, events, register_original

    def test_helper_complete_causal_lm_stays_inside_shared_fenced_lease(self):
        model, native, complete, events, register = self.helper_fixture({"started": 123.0})
        lease = self.fixture.lease

        def fence(device):
            self.assertEqual(device, 0)
            self.fixture.assert_locked(self)
            self.assertEqual(lease.held[0], "nf4")

        with invented_torch(fence) as (torch, cuda_events):
            handles = self.module.install_nf4_hooks(
                model,
                torch,
                native,
                lease,
                resource_gate="invented-resource-gate",
                campaign_deadline=lambda: events.append(("campaign-deadline",)),
                register_original=register,
            )
            complete.before(complete, (), {})
            self.fixture.assert_locked(self)
            events.append(("actual-complete-causal-lm-including-head",))
            complete.after(complete, (), {}, object())
            self.assertIsNone(lease.fd)
            handles.remove()
        self.assertEqual(cuda_events, [("cuda-fence", 0)])
        self.assertEqual(
            [(r["event"], r["kind"]) for r in self.fixture.records()],
            [("acquired", "nf4"), ("completed", "nf4")],
        )
        self.assertIn(("http-deadline", 123.0), events)
        self.assertEqual(
            events[-3:],
            [
                ("remove", "complete-causal-lm-post"),
                ("remove", "complete-causal-lm-pre"),
                ("remove", "original-resource-hook"),
            ],
        )

    def test_helper_failed_forward_retains_lease_and_refuses_hook_removal(self):
        model, native, complete, events, register = self.helper_fixture(None)
        lease = self.fixture.lease
        with invented_torch() as (torch, cuda_events):
            handles = self.module.install_nf4_hooks(
                model,
                torch,
                native,
                lease,
                resource_gate="invented-resource-gate",
                campaign_deadline=lambda: None,
                register_original=register,
            )
            complete.before(complete, (), {})
            complete.after(complete, (), {}, None)
            with self.assertRaisesRegex(RuntimeError, "latched"):
                handles.remove()
        self.assertEqual(cuda_events, [])
        self.assertIsNotNone(lease.fatal)
        self.fixture.assert_locked(self)
        self.assertFalse(any(event[0] == "remove" for event in events))

    def test_helper_changed_fence_binding_cannot_release_shared_lease(self):
        model, native, complete, _events, register = self.helper_fixture(None)
        lease = self.fixture.lease
        with invented_torch() as (torch, cuda_events):
            self.module.install_nf4_hooks(
                model,
                torch,
                native,
                lease,
                resource_gate="invented-resource-gate",
                campaign_deadline=lambda: None,
                register_original=register,
            )
            complete.before(complete, (), {})
            torch.cuda.synchronize = lambda _device: None
            with self.assertRaisesRegex(RuntimeError, "generic/no-op"):
                complete.after(complete, (), {}, object())
        self.assertEqual(cuda_events, [])
        self.assertIsNotNone(lease.fatal)
        self.fixture.assert_locked(self)

    def test_helper_partial_hook_install_preserves_primary_and_cleans_prior_hooks(self):
        model, native, complete, events, _register = self.helper_fixture(None)
        primary = KeyboardInterrupt("invented helper last-hook installation abort")

        def last_hook(_callback, *, with_kwargs, always_call):
            self.assertTrue(with_kwargs and always_call)
            raise primary

        class OriginalHandle:
            def remove(inner_self):
                events.append(("remove", "original-resource-hook"))
                raise OSError("invented secondary original-hook cleanup")

        def register(_model, _gate):
            return OriginalHandle()

        complete.register_forward_hook = last_hook
        with invented_torch() as (torch, _cuda_events):
            with self.assertRaises(KeyboardInterrupt) as captured:
                self.module.install_nf4_hooks(
                    model,
                    torch,
                    native,
                    self.fixture.lease,
                    resource_gate="invented-resource-gate",
                    campaign_deadline=lambda: None,
                    register_original=register,
                )
        self.assertIs(captured.exception, primary)
        self.assertEqual(
            events[-2:],
            [("remove", "complete-causal-lm-pre"), ("remove", "original-resource-hook")],
        )
        self.assertTrue(
            any("secondary original-hook cleanup" in note for note in primary.__notes__)
        )
        self.assertIsNone(self.fixture.lease.fd)
        self.assertEqual(self.fixture.records(), [])

    def test_old_cpu_phase_name_and_malformed_campaign_nonce_are_refused(self):
        with self.assertRaisesRegex(RuntimeError, "GPU/helper family"):
            self.fixture.lease.begin("decision3", lambda: None)
        expected = self.fixture.lease.expected
        for nonce in ("g" * 32, "A" * 32, "a" * 31):
            with (
                self.subTest(nonce=nonce),
                self.assertRaisesRegex(RuntimeError, "lowercase hexadecimal"),
            ):
                self.module.ForwardLease(
                    self.fixture.lock, self.fixture.journal, expected, nonce=nonce
                )
        self.assertEqual(self.fixture.records(), [])

    def test_end_preserves_an_already_latched_original_abort(self):
        lease, primary = self.fixture.lease, KeyboardInterrupt("invented original latched abort")
        lease.begin("decision3-gpu", lambda: None)
        lease.fail(primary)
        with self.assertRaises(KeyboardInterrupt) as captured:
            lease.end(
                synchronize=lambda: self.fail("Failed lease cannot reach release fence"),
                deadline=lambda: None,
            )
        self.assertIs(captured.exception, primary)
        self.assertIs(lease.fatal, primary)
        self.fixture.assert_locked(self)
        self.assertEqual([r["event"] for r in self.fixture.records()], ["acquired"])

    def test_close_error_after_successful_unlock_is_latched_and_reported_honestly(self):
        lease, primary = self.fixture.lease, OSError("invented close error after unlock")
        lease.begin("decision3-gpu", lambda: None)
        lock_fd, original_close = lease.fd, self.module.os.close

        def close(fd):
            original_close(fd)
            if fd == lock_fd:
                raise primary

        with mock.patch.object(self.module.os, "close", close):
            with self.assertRaises(OSError) as captured:
                lease.end(synchronize=lambda: None, deadline=lambda: None)
        self.assertIs(captured.exception, primary)
        self.assertIs(lease.fatal, primary)
        self.assertIsNone(lease.fd)
        self.assertIsNone(lease.held)
        other = os.open(self.fixture.lock, os.O_RDWR)
        try:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(other)
        with self.assertRaisesRegex(RuntimeError, "latched"):
            lease.check()

    def test_custom_abort_refusing_notes_does_not_replace_original_fatal(self):
        class RefusingAbort(KeyboardInterrupt):
            def add_note(self, _note):
                raise RuntimeError("invented custom exception refuses notes")

        primary, lease = RefusingAbort("invented original abort"), self.fixture.lease
        original_fail = lease.fail

        def compose(_request, _kwargs):
            raise primary

        def retain_then_fail(error):
            original_fail(error)
            raise OSError("invented secondary retention error")

        lease.fail = retain_then_fail
        with invented_torch() as (torch, cuda_events):
            serial = self.module.SerialBackend(Provider(compose), lease, torch=torch)
            with self.assertRaises(RefusingAbort) as captured:
                serial.predict(object(), deadline=lambda: None)
        self.assertIs(captured.exception, primary)
        self.assertIs(lease.fatal, primary)
        self.assertEqual(cuda_events, [])
        self.fixture.assert_locked(self)


if __name__ == "__main__":
    unittest.main()
