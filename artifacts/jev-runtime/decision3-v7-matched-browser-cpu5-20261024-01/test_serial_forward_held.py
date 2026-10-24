"""Invented stdlib fixtures only; no framework, weights, browser or actual data."""

import importlib.util
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class ForwardLeaseTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name("serial_forward_held.py")
        spec = importlib.util.spec_from_file_location("_fake_serial_forward", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock, self.journal = self.root / "forward.lock", self.root / "events.jsonl"
        for target in (self.lock, self.journal):
            target.touch(mode=0o600)
        self.expected = {
            label: {"device": path.stat().st_dev, "inode": path.stat().st_ino}
            for label, path in (("lock", self.lock), ("journal", self.journal))
        }

    def lease(self):
        # This mutates only the freshly imported fake module, never any source
        # release flag, source pin, real model, admission or actual receipt.
        self.module.SOURCE_ONLY = False
        result = self.module.ForwardLease(self.lock, self.journal, self.expected, nonce="a" * 32)

        def cleanup():
            if result.fd is not None:
                os.close(result.fd)  # Destroy only this test's invented lease.
                result.fd = None

        self.addCleanup(cleanup)
        return result

    @staticmethod
    def deadline():
        return None

    def blocked(self, lease):
        calls = 0

        def deadline():
            nonlocal calls
            calls += 1
            if calls >= 3:
                raise TimeoutError("invented original deadline")

        with self.assertRaisesRegex(TimeoutError, "invented original"):
            lease.begin("decision3", deadline)
        self.assertIsNone(lease.fd)

    def test_default_hold_precedes_missing_file_io(self):
        with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
            self.module.ForwardLease(
                self.root / "missing", self.root / "missing-journal", {}, nonce="a" * 32
            )

    def test_second_process_cannot_enter_before_completion(self):
        first = self.lease()
        first.begin("nf4", self.deadline)
        read_fd, write_fd = os.pipe()
        child = os.fork()
        if child == 0:
            try:
                os.close(read_fd)
                os.close(first.fd)
                second = self.lease()
                self.blocked(second)
                os.write(write_fd, b"blocked")
                os._exit(0)
            except BaseException:
                os._exit(1)
        os.close(write_fd)
        try:
            message = os.read(read_fd, 32)
        finally:
            os.close(read_fd)
        _, status = os.waitpid(child, 0)
        self.assertEqual(status, 0)
        self.assertEqual(message, b"blocked")
        first.end(synchronize=lambda: None, deadline=self.deadline)
        second = self.lease()
        second.begin("decision3", self.deadline)
        second.end(synchronize=lambda: None, deadline=self.deadline)
        rows = [json.loads(line) for line in self.journal.read_bytes().splitlines()]
        self.assertEqual(
            [(row["kind"], row["event"]) for row in rows],
            [
                ("nf4", "acquired"),
                ("nf4", "completed"),
                ("decision3", "acquired"),
                ("decision3", "completed"),
            ],
        )

    def test_device_error_keeps_os_lock_until_process_cleanup(self):
        first = self.lease()
        first.begin("nf4", self.deadline)
        error = RuntimeError("invented device failure")

        def synchronize():
            raise error

        with self.assertRaises(RuntimeError) as caught:
            first.end(synchronize=synchronize, deadline=self.deadline)
        self.assertIs(caught.exception, error)
        self.assertIs(first.fatal, error)
        self.assertIsNotNone(first.fd)
        self.blocked(self.lease())
        self.assertEqual(len(self.journal.read_bytes().splitlines()), 1)

    def test_journal_replacement_refuses_forward_and_keeps_lock(self):
        lease = self.lease()
        self.journal.rename(self.root / "original-journal")
        self.journal.touch(mode=0o600)
        with self.assertRaisesRegex(RuntimeError, "journal inode"):
            lease.begin("nf4", self.deadline)
        self.assertIsNotNone(lease.fd)
        self.blocked(self.lease())

    def test_journal_close_failure_preserves_first_write_error_and_lock(self):
        lease = self.lease()
        primary, secondary = OSError("invented write failure"), OSError("invented close failure")
        real_close = os.close

        def close(fd):
            real_close(fd)
            raise secondary

        with (
            patch.object(self.module.os, "write", side_effect=primary),
            patch.object(self.module.os, "close", side_effect=close),
        ):
            with self.assertRaises(OSError) as caught:
                lease.begin("nf4", self.deadline)
        self.assertIs(caught.exception, primary)
        self.assertIs(lease.fatal, primary)
        self.assertIsNotNone(lease.fd)
        self.assertTrue(any("journal close also failed" in note for note in primary.__notes__))
        self.blocked(self.lease())

    def test_unacquired_close_failure_preserves_original_identity_error(self):
        lease = self.lease()
        lease.expected["lock"]["inode"] += 1
        secondary, real_close = OSError("invented secondary close failure"), os.close

        def close(fd):
            real_close(fd)
            raise secondary

        with patch.object(self.module.os, "close", side_effect=close):
            with self.assertRaisesRegex(RuntimeError, "shared lock inode") as caught:
                lease.begin("nf4", self.deadline)
        self.assertIs(lease.fatal, caught.exception)
        self.assertIsNone(lease.fd)
        self.assertTrue(
            any("lock close also failed" in note for note in caught.exception.__notes__)
        )

    def test_cpu_wrapper_returns_same_object_after_completion(self):
        lease = self.lease()
        sentinel = object()
        calls = []

        def predict(request, **kwargs):
            self.assertIsNotNone(lease.fd)
            calls.append((request, kwargs))
            return sentinel

        backend = types.SimpleNamespace(spec=object(), name="invented", predict=predict)
        wrapper = self.module.SerialBackend(backend, lease)
        request = object()
        self.assertIs(wrapper.predict(request, deadline=self.deadline), sentinel)
        self.assertIs(calls[0][0], request)
        self.assertIsNone(lease.fd)

    def test_nf4_barrier_occurs_before_lock_release(self):
        lease = self.lease()
        handlers, removed, barriers = {}, [], []

        def handle(name):
            return types.SimpleNamespace(remove=lambda: removed.append(name))

        def pre(callback, **kwargs):
            self.assertEqual(kwargs, {"with_kwargs": True})
            handlers["before"] = callback
            return handle("before")

        def post(callback, **kwargs):
            self.assertEqual(kwargs, {"with_kwargs": True, "always_call": True})
            handlers["after"] = callback
            return handle("after")

        def synchronize(device):
            self.assertEqual(device, 0)
            self.assertIsNotNone(lease.fd)
            barriers.append("device-complete")

        complete_model = types.SimpleNamespace(
            register_forward_pre_hook=pre, register_forward_hook=post
        )
        model = types.SimpleNamespace(
            backbone=object(),
            lm=types.SimpleNamespace(get_base_model=lambda: complete_model),
        )
        native = types.SimpleNamespace(TRACE=types.SimpleNamespace(get=lambda: None))
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=synchronize))
        hooks = self.module.install_nf4_hooks(
            model,
            torch,
            native,
            lease,
            resource_gate=object(),
            campaign_deadline=self.deadline,
            register_original=lambda *_args: handle("original"),
        )
        handlers["before"](None, (), {})
        handlers["after"](None, (), {}, object())
        self.assertEqual(barriers, ["device-complete"])
        self.assertIsNone(lease.fd)
        hooks.remove()
        self.assertEqual(removed, ["after", "before", "original"])

    def test_full_causal_lm_failure_retains_lease_after_decoder_return(self):
        lease = self.lease()
        handlers = {}

        def register(name, callback, **_kwargs):
            handlers[name] = callback
            return types.SimpleNamespace(remove=lambda: None)

        complete_model = types.SimpleNamespace(
            register_forward_pre_hook=lambda callback, **kwargs: register(
                "before", callback, **kwargs
            ),
            register_forward_hook=lambda callback, **kwargs: register("after", callback, **kwargs),
        )
        model = types.SimpleNamespace(
            backbone=object(), lm=types.SimpleNamespace(get_base_model=lambda: complete_model)
        )
        self.module.install_nf4_hooks(
            model,
            object(),
            types.SimpleNamespace(TRACE=types.SimpleNamespace(get=lambda: None)),
            lease,
            resource_gate=object(),
            campaign_deadline=self.deadline,
            register_original=lambda *_args: types.SimpleNamespace(remove=lambda: None),
        )
        handlers["before"](None, (), {})
        # Simulate an exception from lm_head after the decoder has returned.
        # Torch's always_call hook receives None for the failed full forward.
        handlers["after"](None, (), {}, None)
        self.assertIsNotNone(lease.fd)
        self.assertIsNotNone(lease.fatal)
        self.blocked(self.lease())


if __name__ == "__main__":
    unittest.main()
