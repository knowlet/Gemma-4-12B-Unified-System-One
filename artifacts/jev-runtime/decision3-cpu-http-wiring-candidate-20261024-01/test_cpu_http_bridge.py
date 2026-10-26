"""UNEXECUTED fake-test source; no model, HTTP listener, or browser is loaded.

The normal SOURCE_ONLY guard is asserted first. Tests of draft internals patch
ONLY that guard and supply stdlib fakes; they do not qualify a native backend.
These tests were written without running/importing/formatting/checking them.
"""

import asyncio
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch


SOURCE = Path(__file__).with_name("cpu_http_bridge.py")
SPEC = importlib.util.spec_from_file_location("unselected_cpu_http_candidate", SOURCE)
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


def payload():
    return {
        "model": "d3-edge-cpu-toy",
        "state": {"page": {"title": "Two lamp controls", "text": "Neither lamp armed"}},
        "questions": {
            "operation": {"type": "choice", "instructions": {"goal": "Arm the copper lamp"},
                          "criteria": {"CLICK": "Click", "WAIT": "Wait", "DONE": "Complete", "BLOCKED": "Blocked"}},
            "click_target": {"type": "choice", "instructions": "Choose observed target",
                             "criteria": {"1": {"element": "Arm copper lamp"}, "2": {"element": "Arm silver lamp"}}},
        },
    }


class SourceGateTests(unittest.TestCase):
    def test_factory_refuses_before_reading_poison_plan_or_dependencies(self):
        class Poison:
            def __fspath__(self):
                raise AssertionError("Plan was read")

        with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
            BRIDGE.create_application(Poison(), runtime=Poison())
        with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
            BRIDGE._draft_dependencies()

    def test_even_source_flag_removal_leaves_unselected_entry_refused(self):
        with patch.object(BRIDGE, "SOURCE_ONLY", False):
            with self.assertRaisesRegex(RuntimeError, "Pending owned guardian"):
                BRIDGE.create_application("must-not-be-opened")

    def test_no_project_dependency_constructor_before_source_gate(self):
        with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
            BRIDGE._BackendShim(spec=None, name=None, actor=None, runtime=None,
                                torch=None, model=None, native_payload=None)
        with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
            BRIDGE._Gateway(output=None, backend=None, adapter_type=None,
                            named_browser_response=None, cpu_identity=None,
                            timeout_type=None, validation_type=None)


class CPUConstructorTests(unittest.TestCase):
    def test_inherits_exact_predict_without_cuda_spec_mutation(self):
        class Base:
            def predict(self, value):
                return value

        observed = []

        def blockers(spec):
            observed.append(spec.device)
            return ["decision3_requires_cuda_bfloat16"]

        cpu = BRIDGE._cpu_adapter_type(Base, blockers)
        spec = NS(adapter="decision3", device="cpu", precision="bfloat16", context_limit=16384)
        with patch.object(BRIDGE, "_release_gate"):
            adapter = cpu(NS(spec=spec, name="d3-edge-cpu-toy"))
        self.assertIs(cpu.predict, Base.predict)
        self.assertEqual(observed, ["cpu"])
        self.assertEqual(spec.device, "cpu")
        self.assertEqual(adapter.identity, "d3-edge-cpu-toy")

    def test_combined_cuda_blocker_cannot_hide_wrong_precision_or_other_blocker(self):
        cpu = BRIDGE._cpu_adapter_type(object, lambda _spec: [
            "decision3_requires_cuda_bfloat16", "decision3_checkpoint_unverified"
        ])
        backend = NS(name="cpu", spec=NS(adapter="decision3", device="cpu", precision="float32", context_limit=16384))
        with patch.object(BRIDGE, "_release_gate"):
            with self.assertRaisesRegex(ValueError, "BF16"):
                cpu(backend)
            backend.spec.precision = "bfloat16"
            with self.assertRaisesRegex(ValueError, "checkpoint_unverified"):
                cpu(backend)
            backend.spec.device = "cuda"
            with self.assertRaisesRegex(ValueError, "CUDA spoofing"):
                cpu(backend)


class ScopeAndRowsTests(unittest.TestCase):
    def test_all_original_operations_and_target_order_unchanged(self):
        request = payload()
        request["questions"]["operation"]["criteria"].update({
            "TYPE_TEXT": "Offered without target", "SELECT": "Offered without target",
            "SCROLL_UP": "Up", "SCROLL_DOWN": "Down",
        })
        original = copy.deepcopy(request)
        BRIDGE._toy_scope(request)
        self.assertEqual(request, original)
        self.assertEqual(list(request["questions"]), ["operation", "click_target"])
        self.assertEqual(list(request["questions"]["click_target"]["criteria"]), ["1", "2"])

    def test_actual_text_target_rejects_whole_request_without_filtering(self):
        request = payload()
        request["questions"]["type_text_target"] = {"type": "choice", "criteria": {"3": "Field"}}
        original = copy.deepcopy(request)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            BRIDGE._toy_scope(request)
        self.assertEqual(request, original)

    def test_duplicate_json_keys_and_nonfinite_values_refused(self):
        for body in (b'{"model":"a","model":"b"}', b'{"x":NaN}'):
            with self.assertRaises(ValueError):
                BRIDGE._decode(body)

    def test_real_tail_shapes_one_to_four_and_five_questions_no_dummy_rows(self):
        torch = NS(bfloat16="bf16", cuda=NS(is_initialized=lambda: False))
        for count in (1, 2, 3, 4, 5):
            ids = [f"q{index}" for index in range(count)]
            lengths = {key: 10 + index for index, key in enumerate(ids)}
            observed = []
            witness = BRIDGE._RowWitness(torch, ids, lengths, lambda event, value: observed.append((event, value)))
            for start in range(0, count, 4):
                chunk = ids[start:start + 4]
                hidden = NS(shape=(len(chunk), max(lengths[key] for key in chunk), 1024), dtype="bf16", device=NS(type="cpu"))
                self.assertIsNone(witness(None, (hidden,)))
            proof = witness.completed()
            self.assertEqual(proof["observed_question_rows"], count)
            self.assertEqual(proof["question_order"], ids)
            self.assertEqual(proof["dummy_questions_added"], 0)
            self.assertTrue(all(value["scope"] == "first_layer_entered_not_forward_completed" for _, value in observed))

    def test_unseen_or_missing_rows_are_not_reported_complete(self):
        torch = NS(bfloat16="bf16", cuda=NS(is_initialized=lambda: False))
        witness = BRIDGE._RowWitness(torch, ["a", "b"], {"a": 7, "b": 9}, lambda *_: None)
        with self.assertRaisesRegex(ValueError, "Missing genuine"):
            witness.completed()
        with self.assertRaisesRegex(ValueError, "shape differs"):
            witness(None, (NS(shape=(4, 9, 1024), dtype="bf16", device=NS(type="cpu")),))


class BackendShimTests(unittest.TestCase):
    def test_ordered_dto_payload_raw_identity_and_actual_tail_witness(self):
        request = payload()
        raw = {"model": "d3-edge", "answers": {"operation": "original raw", "click_target": "original raw"},
               "usage": {"input_tokens": 17, "output_tokens": 0}}
        hook_state, events = {}, []

        def register(callback):
            hook_state["callback"] = callback
            return NS(remove=lambda: hook_state.update(removed=True))

        runtime = NS(backbone=NS(language_model=NS(layers=[NS(register_forward_pre_hook=register)])))
        torch = NS(bfloat16="bf16", cuda=NS(is_initialized=lambda: False))

        class FakeOriginalBackend:
            def __init__(self, actual_runtime, actual_torch, model, context_limit, evidence_sink):
                self.sink = evidence_sink
                self.assertions = (actual_runtime is runtime, actual_torch is torch, context_limit == 16384)
                if not all(self.assertions):
                    raise AssertionError("Original wrapper bindings differ")

            def __call__(self, state, questions):
                if list(questions) != ["operation", "click_target"] or state != request["state"]:
                    raise AssertionError("Ordered complete native payload changed")
                self.sink("prepared", {"lengths": {"operation": 8, "click_target": 9}, "original_question_order": list(questions)})
                hook_state["callback"](None, (NS(shape=(2, 9, 1024), dtype="bf16", device=NS(type="cpu")),))
                self.sink("raw_returned_before_validation", raw)
                return raw

        with patch.object(BRIDGE, "_release_gate"):
            shim = BRIDGE._BackendShim(spec=NS(), name="cpu", actor=NS(NativeBatch4Backend=FakeOriginalBackend),
                                      runtime=runtime, torch=torch, model={},
                                      native_payload=lambda _dto: {"state": copy.deepcopy(request["state"]), "questions": copy.deepcopy(request["questions"])})
        actual = shim.predict(object(), deadline=lambda: None, evidence_sink=lambda event, value: events.append((event, value)))
        self.assertIs(actual, raw)  # No second response normalization in the shim.
        self.assertTrue(hook_state["removed"])
        self.assertEqual(events[-1][0], "native_rows_completed")
        self.assertEqual(events[-1][1]["actual_shapes"], [[2, 9, 1024]])
        self.assertEqual(events[-1][1]["helper_calls"], 0)


class GatewayTests(unittest.TestCase):
    def make_gateway(self, output, clock, *, primary=None, operation="BLOCKED"):
        backend = NS(name="d3-edge-cpu-toy", calls=0)

        class FakeAdapter:
            def __init__(self, actual, evidence_sink):
                self.backend, self.sink = actual, evidence_sink

            def predict(self, request):
                self.backend.calls += 1
                self.sink({"event": "backend_raw_returned", "data": {"untouched": "raw", "request": request}})
                if primary is not None:
                    raise primary
                return {"answers": {"operation": {"choice": operation, "probabilities": {operation: 1.0}}},
                        "model": self.backend.name, "raw": {"untouched": "raw"}}

        identity = {"device": "cpu", "precision": "bfloat16", "context_limit": 16384,
                    "status": "FAKE_TEST_ONLY_NOT_QUALIFIED"}
        with patch.object(BRIDGE, "_release_gate"):
            gateway = BRIDGE._Gateway(output=output, backend=backend, adapter_type=FakeAdapter,
                                      named_browser_response=lambda _request, result: copy.deepcopy(result),
                                      cpu_identity=identity, timeout_type=TimeoutError,
                                      validation_type=ValueError, clock=lambda: clock[0])
        return gateway, backend

    def call(self, gateway, request):
        sent = []

        async def receive():
            return {"type": "http.request", "body": BRIDGE._json_bytes(request), "more_body": False}

        async def send(message):
            sent.append(copy.deepcopy(message))

        asyncio.run(gateway({"type": "http", "method": "POST", "path": "/v1/systemone"}, receive, send))
        return sent

    def new_output(self, parent):
        with patch.object(BRIDGE, "_release_gate"):
            return BRIDGE._new_output(Path(parent) / "owned-output")

    def test_legal_blocked_is_returned_unchanged_not_forced_to_goal_click(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = self.new_output(temporary)
            gateway, backend = self.make_gateway(output, [0.0])
            sent = self.call(gateway, payload())
            result = json.loads(sent[1]["body"])
            self.assertEqual(sent[0]["status"], 200)
            self.assertEqual(result["answers"]["operation"]["choice"], "BLOCKED")
            self.assertEqual(backend.calls, 1)
            self.assertEqual(result["decision3_cpu_http"]["helper_calls"], 0)
            self.assertFalse(gateway.serial.locked())

    def test_final_terminal_fsync_at_exact_twenty_cannot_authorize_200(self):
        for delay in (19.999, 20.0, 21.0):
            with self.subTest(delay=delay), tempfile.TemporaryDirectory() as temporary:
                output = self.new_output(temporary)
                clock = [0.0]
                gateway, _ = self.make_gateway(output, clock)
                original = BRIDGE._Record.write

                def slow_terminal(record, name, data, **kwargs):
                    original(record, name, data, **kwargs)
                    if name == "exchange.json":
                        clock[0] = delay

                with patch.object(BRIDGE._Record, "write", slow_terminal):
                    sent = self.call(gateway, payload())
                expected = 200 if delay < 20 else 504
                self.assertEqual(sent[0]["status"], expected)
                request_dir = next(path for path in output.iterdir() if path.is_dir())
                self.assertTrue((request_dir / "candidate.json").exists())
                self.assertEqual((request_dir / "response.bin").read_bytes(), sent[1]["body"])
                exchange = json.loads((request_dir / "exchange.json").read_bytes())
                self.assertEqual(exchange["status"], expected)
                self.assertEqual(dict(sent[0]["headers"])[b"content-length"], str(len(sent[1]["body"])).encode())
                self.assertFalse(gateway.serial.locked())

    def test_publication_failure_keeps_primary_and_releases_serial_lock(self):
        primary = RuntimeError("actual native failure")
        with tempfile.TemporaryDirectory() as temporary:
            output = self.new_output(temporary)
            gateway, _ = self.make_gateway(output, [0.0], primary=primary)
            original = BRIDGE._Record.write

            def fail_terminal(record, name, data, **kwargs):
                if name == "response.bin":
                    raise OSError("disk failure")
                return original(record, name, data, **kwargs)

            with patch.object(BRIDGE._Record, "write", fail_terminal):
                with self.assertRaises(RuntimeError) as raised:
                    self.call(gateway, payload())
            self.assertIs(raised.exception, primary)
            self.assertTrue(any("disk failure" in note for note in primary.__notes__))
            self.assertFalse(gateway.serial.locked())

    def test_request_parent_fsync_counts_toward_original_twenty(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = self.new_output(temporary)
            clock = [0.0]
            gateway, backend = self.make_gateway(output, clock)
            original = BRIDGE._fsync_directory

            def slow_parent(path):
                original(path)
                if path == output:
                    clock[0] = 20.0

            with patch.object(BRIDGE, "_fsync_directory", slow_parent):
                sent = self.call(gateway, payload())
            self.assertEqual(sent[0]["status"], 504)
            self.assertEqual(backend.calls, 0)
            self.assertFalse(gateway.serial.locked())

    def test_candidate_write_failure_does_not_claim_durable_retention(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = self.new_output(temporary)
            gateway, _ = self.make_gateway(output, [0.0])
            original = BRIDGE._Record.write

            def fail_candidate(record, name, data, **kwargs):
                if name == "candidate.json":
                    raise OSError("candidate not retained")
                return original(record, name, data, **kwargs)

            with patch.object(BRIDGE._Record, "write", fail_candidate):
                sent = self.call(gateway, payload())
            self.assertEqual(sent[0]["status"], 500)
            request_dir = next(path for path in output.iterdir() if path.is_dir())
            exchange = json.loads((request_dir / "exchange.json").read_bytes())
            self.assertTrue(exchange["candidate_constructed"])
            self.assertFalse(exchange["candidate_retained"])
            self.assertFalse((request_dir / "candidate.json").exists())

    def test_observer_abort_object_escapes_after_original_observer_isolation(self):
        class ResourceAbort(BaseException):
            pass

        abort = ResourceAbort("invented owned-resource abort")
        with tempfile.TemporaryDirectory() as temporary:
            output = self.new_output(temporary)
            gateway, backend = self.make_gateway(output, [0.0])
            base = gateway.adapter_type

            class IsolatingAdapter(base):
                def predict(self, request):
                    try:
                        return super().predict(request)
                    except BaseException:
                        # Mirror the original adapter emit's isolation without
                        # pretending that native execution was interrupted.
                        return {"answers": {}, "model": self.backend.name}

            gateway.adapter_type = IsolatingAdapter
            original = BRIDGE._Record.write

            def abort_observer(record, name, data, **kwargs):
                if name.startswith("event-"):
                    raise abort
                return original(record, name, data, **kwargs)

            with patch.object(BRIDGE._Record, "write", abort_observer):
                with self.assertRaises(ResourceAbort) as raised:
                    self.call(gateway, payload())
            self.assertIs(raised.exception, abort)
            self.assertEqual(backend.calls, 1)
            self.assertFalse(gateway.serial.locked())
            request_dir = next(path for path in output.iterdir() if path.is_dir())
            exchange = json.loads((request_dir / "exchange.json").read_bytes())
            self.assertEqual(exchange["status"], 500)
            self.assertEqual(exchange["primary_error"]["type"], "ResourceAbort")
            self.assertFalse(exchange["candidate_retained"])


if __name__ == "__main__":
    raise SystemExit("SOURCE ONLY: tests intentionally not executed; future owner review required")
