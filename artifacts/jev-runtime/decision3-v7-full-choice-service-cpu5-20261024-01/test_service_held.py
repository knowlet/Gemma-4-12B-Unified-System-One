"""Unexecuted stdlib fakes for the held HTTP-only service; no runtime or browser."""

import copy
import importlib.util
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def module():
    return source(Path(__file__).with_name("worker_held.py"), "_held_full_choice_service_fake")


def selected():
    return {
        "case_index": 2,
        "case_id": "public-arxiv-original-paper-open-v1",
        "root_plan_sha256": "a" * 64,
        "handoff": {"nonce": "b" * 32, "stop_path": "unused-by-fake"},
    }


class ServiceSourceFakes(unittest.TestCase):
    def test_hold_precedes_arguments_bootstrap_and_io(self):
        worker = module()
        with patch.object(worker, "bootstrap", side_effect=AssertionError("bootstrap")):
            with patch.object(
                worker.argparse, "ArgumentParser", side_effect=AssertionError("args")
            ):
                with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
                    worker.main(["--help"])
            with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
                worker.run("unread-plan", "c" * 64, "unread-output")

    def test_release_flag_alone_does_not_release_pending_pins(self):
        worker = module()
        worker.SOURCE_ONLY = False
        with self.assertRaisesRegex(RuntimeError, "pending pin"):
            worker.released()

    def test_stop_exact_case_root_plan_service_plan_nonce_and_schema(self):
        worker, plan, sha = module(), selected(), "c" * 64
        record = {"schema": worker.STOP_SCHEMA, **worker.binding(plan, sha)}
        self.assertIs(worker.stop_record(plan, sha, record), record)
        for key, value in (
            ("schema", "closed"),
            ("case_index", 3),
            ("case_id", "different-case"),
            ("root_plan_sha256", "d" * 64),
            ("plan_sha256", "e" * 64),
            ("nonce", "f" * 32),
            ("task_success", True),
        ):
            changed = {**record, key: value}
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "exact selected stop"):
                worker.stop_record(plan, sha, changed)

    def test_fatal_wait_rethrows_same_object_before_stop_access(self):
        worker = module()
        original = KeyboardInterrupt("fake native failure")

        def abort():
            raise original

        with patch.object(worker, "ordinary", side_effect=AssertionError("stop IO")):
            with self.assertRaises(KeyboardInterrupt) as caught:
                worker.wait_for_stop(
                    None,
                    selected(),
                    "c" * 64,
                    fatal=types.SimpleNamespace(raise_if_set=abort),
                    server=None,
                    thread=None,
                    deadline=lambda: None,
                )
        self.assertIs(caught.exception, original)

    def test_expired_whole_deadline_precedes_stop_and_never_authorizes_exit_success(self):
        worker = module()
        original = TimeoutError("cooperative budget expired")

        def expire():
            raise original

        with patch.object(worker, "read_stop", side_effect=AssertionError("stop read")):
            with self.assertRaises(TimeoutError) as caught:
                worker.wait_for_stop(
                    None,
                    selected(),
                    "c" * 64,
                    fatal=types.SimpleNamespace(raise_if_set=lambda: None),
                    server=types.SimpleNamespace(should_exit=False),
                    thread=types.SimpleNamespace(is_alive=lambda: True),
                    deadline=expire,
                )
        self.assertIs(caught.exception, original)

    def test_serialization_wraps_actual_provider_with_same_arguments_and_result(self):
        worker = module()
        serial = source(worker.MATCHED / "serial_forward_held.py", "_held_serial_service_fake")
        serial.SOURCE_ONLY = False
        events, descriptor = (
            [],
            {
                "lock_path": "lock",
                "journal_path": "journal",
                "expected_stat": {},
                "nonce": "b" * 32,
            },
        )
        request, result = object(), {"unchanged": object()}

        def deadline():
            events.append("deadline")

        def sink(*_args):
            events.append("retained")

        class Lease:
            def __init__(self, **received):
                self.received, self.live = copy.deepcopy(received), False

            def begin(self, kind, received_deadline):
                self.live = True
                events.append(("begin", kind))
                self.assert_deadline = received_deadline
                received_deadline()

            def end(self, *, synchronize, deadline):
                self.assert_deadline = deadline
                synchronize()
                events.append("end")
                self.live = False

            def fail(self, error):
                events.append(("fail", error))

        class Provider:
            name, spec = "unchanged-alias", object()

            def predict(self, actual, **kwargs):
                self.actual, self.kwargs = actual, kwargs
                self.assert_live = lease.live
                events.append("actual-provider-predict")
                kwargs["evidence_sink"]("raw", result)
                return result

        provider = Provider()
        with patch.object(serial, "ForwardLease", Lease):
            wrapped, lease = worker.serialized_provider(serial, provider, descriptor)
        returned = wrapped.predict(request, deadline=deadline, evidence_sink=sink)
        self.assertIs(returned, result)
        self.assertIs(provider.actual, request)
        self.assertIs(provider.kwargs["deadline"], deadline)
        self.assertIs(provider.kwargs["evidence_sink"], sink)
        self.assertTrue(provider.assert_live)
        self.assertEqual(lease.received, descriptor)
        self.assertEqual(
            events,
            [("begin", "decision3"), "deadline", "actual-provider-predict", "retained", "end"],
        )

    def test_serial_failure_retains_original_failure_and_no_end(self):
        worker = module()
        serial = source(worker.MATCHED / "serial_forward_held.py", "_held_serial_failure_fake")
        original, events = KeyboardInterrupt("fake forward failure"), []

        def predict(*_args, **_kwargs):
            raise original

        lease = types.SimpleNamespace(
            begin=lambda *_args: events.append("begin"),
            end=lambda **_kwargs: events.append("end"),
            fail=lambda error: events.append(error),
        )
        wrapped = serial.SerialBackend(
            types.SimpleNamespace(name="unchanged", spec=object(), predict=predict), lease
        )
        with self.assertRaises(KeyboardInterrupt) as caught:
            wrapped.predict(object(), deadline=lambda: None, evidence_sink=lambda *_args: None)
        self.assertIs(caught.exception, original)
        self.assertEqual(events, ["begin", original])

    def test_shutdown_refuses_new_provider_work_without_changing_prior_result(self):
        worker, stopping, seen = module(), threading.Event(), []
        answer = object()
        provider = types.SimpleNamespace(
            name="unaltered",
            spec=object(),
            predict=lambda request, **kwargs: seen.append((request, kwargs)) or answer,
        )
        guarded = worker.ServiceAdmission(provider, stopping)
        request, callback = object(), lambda: None
        self.assertIs(guarded.predict(request, deadline=callback), answer)
        self.assertEqual(seen, [(request, {"deadline": callback})])
        stopping.set()
        with self.assertRaisesRegex(RuntimeError, "no new native admission"):
            guarded.predict(object(), deadline=callback)
        self.assertEqual(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
