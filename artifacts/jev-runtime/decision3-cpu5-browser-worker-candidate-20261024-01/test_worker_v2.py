"""UNEXECUTED stdlib-fake test source; no model/browser/listener/owner launch.

Written during the active NN campaign without importing or running this file.
These proposed tests cannot qualify actual resources, native math or wiring.
"""

import asyncio
import copy
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace as NS


def source(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, (Path(__file__).parent / filename).resolve()
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


WORKER = source("owned_worker_fake_tests", "worker_v2.py")
BRIDGE = source("owned_bridge_fake_tests", "bridge_successor.py")


def proof():
    return {
        "schema": BRIDGE.OWNED_WORKER_BINDING_VERSION,
        "owner_unit": WORKER.UNIT,
        "owner_profile": WORKER.OWNER_PROFILE,
        "device": "cpu",
        "precision": "bfloat16",
        "context_limit": 16384,
        "memory_max_bytes": 4294967296,
        "swap_max_bytes": 0,
        "cpu_affinity": [15, 16, 17, 18, 19],
        "cpu_quota_percent": 500,
        "expected_threads": 5,
        "expected_interop_threads": 1,
        "worker_pid": os.getpid(),
        "worker_start_ticks": 123,
        "nonce": "jev-d3-boundary-" + "a" * 32,
        "invocation_id": "b" * 32,
        "plan_path": "/fake/test-only-plan.json",
        "plan_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "selection_sha256": "e" * 64,
        "native_model": {
            "model_id": "vllm-sr/d3-edge",
            "revision": "cab71e6ad3461b8086d51ddb572c656babd63b0b",
            "model_sha256": "6e00d04abbdbdf6d35c9e12065769d9edaacc3bfe9404583891797718d239dc8",
        },
    }


class WorkerSourceTests(unittest.TestCase):
    def test_phase_environment_tracks_loader_transition_without_mutating_constants(self):
        helper = NS(
            START_ENV={"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
            CPU_ENV={"OMP_NUM_THREADS": "5", "MKL_NUM_THREADS": "5"},
        )
        start, cpu = copy.deepcopy(helper.START_ENV), copy.deepcopy(helper.CPU_ENV)
        state = {"model_loads": 0}
        before = WORKER._phase_environment(helper, state)
        self.assertEqual(before, start)
        self.assertIsNot(before, helper.START_ENV)
        state["model_loads"] = 1
        self.assertEqual(
            WORKER._phase_environment(helper, state),
            {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "5", "MKL_NUM_THREADS": "5"},
        )
        before["OMP_NUM_THREADS"] = "changed fake copy"
        self.assertEqual(helper.START_ENV, start)
        self.assertEqual(helper.CPU_ENV, cpu)

    def test_phase_environment_refuses_unknown_repeated_or_boolean_load_count(self):
        helper = NS(START_ENV={}, CPU_ENV={})
        for count in (None, -1, 2, True, "1", 1.0):
            with (
                self.subTest(count=count),
                self.assertRaisesRegex(RuntimeError, "environment phase"),
            ):
                WORKER._phase_environment(helper, {"model_loads": count})

    def test_source_only_before_arguments_or_poison_plan_read(self):
        class Poison:
            def __fspath__(self):
                raise AssertionError("Held entry touched a path")

        if WORKER.SOURCE_ONLY:
            with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
                WORKER.run(Poison(), None, Poison())
            with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
                WORKER.main(["--not-an-option"])
        else:
            # A released CLI still refuses absent/invalid selection arguments.
            # No plan, owner, dependency or model is opened by these calls.
            for args in ([], ["--not-an-option"]):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    WORKER.main(args)
                self.assertEqual(error.exception.code, 2)

    def test_original_abort_crosses_backend_and_blocks_another_native_call(self):
        class NativeAbort(BaseException):
            pass

        abort, calls, fatal = NativeAbort("actual original abort object"), [], WORKER.FatalLatch()

        class Backend:
            spec, name = NS(device="cpu"), "cpu-test"

            def predict(self, request, **kwargs):
                calls.append((request, kwargs))
                raise abort

        backend = WORKER.FatalBackend(Backend(), fatal)
        request = {"all_original_questions": ["operation", "click_target"]}
        for _ in range(2):
            with self.assertRaises(NativeAbort) as raised:
                backend.predict(request, deadline="original callback")
            self.assertIs(raised.exception, abort)
        self.assertEqual(calls, [(request, {"deadline": "original callback"})])

    def test_asgi_rethrows_same_baseexception_and_does_not_mutate_scope(self):
        class NativeAbort(BaseException):
            pass

        abort, fatal, seen = NativeAbort("abort"), WORKER.FatalLatch(), []

        async def app(scope, receive, send):
            seen.append(scope)
            raise abort

        scope = {"type": "http", "path": "/v1/systemone"}
        original = copy.deepcopy(scope)
        guarded = WORKER.GuardedASGI(app, fatal)
        with self.assertRaises(NativeAbort) as raised:
            asyncio.run(guarded(scope, None, None))
        self.assertIs(raised.exception, abort)
        self.assertIs(fatal.first, abort)
        self.assertEqual(scope, original)
        self.assertEqual(seen, [scope])

    def test_successor_lease_rechecks_original_binding_and_refuses_changes(self):
        current = proof()
        release = BRIDGE.bind_owned_selection(lambda: copy.deepcopy(current))
        self.assertEqual(release.verify(), current)
        current["nonce"] = "jev-d3-boundary-" + "f" * 32
        with self.assertRaisesRegex(ValueError, "binding changed"):
            release.verify()

    def test_cpu5_lease_refuses_old_owner_cap_affinity_quota_and_threads(self):
        for key, value in (
            ("owner_unit", "jev-d3-llama-cpp-cpu16-4g-coexist-20261024-08.service"),
            ("owner_profile", "llama-cpp-cpu4-2g-coexist-command-owned-v1"),
            ("memory_max_bytes", 2147483648),
            ("memory_max_bytes", 4294967295),
            ("swap_max_bytes", 1),
            ("cpu_affinity", [5, 6, 7, 8, 9]),
            ("cpu_affinity", list(range(4, 20))),
            ("cpu_quota_percent", 1600),
            ("cpu_quota_percent", 400),
            ("expected_threads", 4),
            ("expected_threads", 16),
            ("expected_interop_threads", 2),
        ):
            data = proof()
            data[key] = value
            with (
                self.subTest(key=key, value=value),
                self.assertRaisesRegex(ValueError, "CPU5 identity"),
            ):
                BRIDGE.bind_owned_selection(lambda: copy.deepcopy(data))

    def test_successor_cpu_constructor_inherits_predict_and_retains_other_blockers(self):
        release = BRIDGE.bind_owned_selection(proof)

        class Base:
            def predict(self, value):
                return value

        cpu = BRIDGE._cpu_adapter_type(
            Base, lambda _spec: ["decision3_requires_cuda_bfloat16"], owned_release=release
        )
        backend = NS(
            name="d3-edge-cpu-toy",
            spec=NS(adapter="decision3", device="cpu", precision="bfloat16", context_limit=16384),
        )
        adapter = cpu(backend)
        self.assertIs(type(adapter).predict, Base.predict)
        self.assertEqual(backend.spec.device, "cpu")
        backend.spec.precision = "float32"
        with self.assertRaisesRegex(ValueError, "BF16"):
            cpu(backend)
        backend.spec.precision = "bfloat16"
        invalid = BRIDGE._cpu_adapter_type(
            Base,
            lambda _spec: ["decision3_requires_cuda_bfloat16", "decision3_checkpoint_unverified"],
            owned_release=release,
        )
        with self.assertRaisesRegex(ValueError, "checkpoint_unverified"):
            invalid(backend)


class ReceiptProgressRegressionTests(unittest.TestCase):
    def writer(self, output, fail_latest=None):
        boundary = source(
            "receipt_boundary_fake_test",
            "../decision3-local-owner-terminal-20261022-01/owner_boundary.py",
        )
        helper = source(
            "receipt_helper_fake_test",
            "../decision3-publisher-batch4-cpu-20261024-05/resource_helper.py",
        )
        if fail_latest is not None:
            helper.write_json = fail_latest
        return WORKER._ReceiptWriter(boundary, helper, output), helper

    def test_repeated_progress_retains_immutable_snapshots_and_latest(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            output = Path(temp)
            writer, _ = self.writer(output)
            first, second = (
                {"stage": "bound", "model_loads": 0},
                {"stage": "loading", "model_loads": 0},
            )
            writer(first)
            frozen = (output / "worker-receipt-000001.json").read_bytes()
            writer(second)
            self.assertEqual((output / "worker-receipt-000001.json").read_bytes(), frozen)
            self.assertEqual(json.loads(frozen), first)
            self.assertEqual(
                json.loads((output / "worker-receipt-000002.json").read_text()), second
            )
            self.assertEqual(json.loads((output / "worker-receipt.json").read_text()), second)

    def test_latest_failure_preserves_original_and_next_snapshot_does_not_clobber(self):
        failure = OSError("invented latest publish failure")

        def fail(*_args):
            raise failure

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            output = Path(temp)
            writer, helper = self.writer(output, fail)
            with self.assertRaises(OSError) as error:
                writer({"stage": "first"})
            self.assertIs(error.exception, failure)
            frozen = (output / "worker-receipt-000001.json").read_bytes()
            real = source(
                "receipt_helper_publish_test",
                "../decision3-publisher-batch4-cpu-20261024-05/resource_helper.py",
            )
            helper.write_json = real.write_json
            writer({"stage": "failed_final"})
            self.assertEqual((output / "worker-receipt-000001.json").read_bytes(), frozen)
            self.assertEqual(
                json.loads((output / "worker-receipt.json").read_text()), {"stage": "failed_final"}
            )


if __name__ == "__main__":
    raise SystemExit("SOURCE ONLY: no checks have been authorized/executed for this candidate")
