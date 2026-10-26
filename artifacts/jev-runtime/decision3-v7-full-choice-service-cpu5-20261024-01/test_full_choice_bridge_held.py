"""UNEXECUTED stdlib fake source; no native/model/browser/service qualification."""

import asyncio
import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_SPEC = importlib.util.spec_from_file_location(
    "full_choice_bridge_source_fakes", Path(__file__).with_name("bridge_full_choice_held.py")
)
BRIDGE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(BRIDGE)


def proof():
    return {
        "schema": BRIDGE.OWNED_WORKER_BINDING_VERSION,
        "owner_unit": "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service",
        "owner_profile": "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1",
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
        "plan_path": "/fake/owned-full-choice-plan.json",
        "plan_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "selection_sha256": "e" * 64,
        "native_model": {
            "model_id": "vllm-sr/d3-edge",
            "revision": "cab71e6ad3461b8086d51ddb572c656babd63b0b",
            "model_sha256": "6e00d04abbdbdf6d35c9e12065769d9edaacc3bfe9404583891797718d239dc8",
        },
    }


def payload():
    # Intentionally neither head order nor labels are lexically sorted.
    def question(criteria):
        return {"type": "choice", "criteria": criteria, "instructions": "Invented source fake"}

    return {
        "model": BRIDGE.BASE_MODEL_IDENTITY,
        "state": {"page": "Invented non-browser fake; not training or evaluation data"},
        "questions": {
            "select_target": question({"19:2": "Option B", "19:0": "Option A"}),
            "operation": question(
                {
                    key: key
                    for key in (
                        "BLOCKED",
                        "SELECT",
                        "TYPE_TEXT",
                        "WAIT",
                        "DONE",
                        "CLICK",
                        "SCROLL_UP",
                        "SCROLL_DOWN",
                    )
                }
            ),
            "type_text_target": question({"31": "A field", "7": "Another field"}),
            "click_target": question({str(key): f"Item {key}" for key in (23, 3, 18, 8, 41)}),
        },
        "media": [],
    }


class ScopeTests(unittest.TestCase):
    def test_held_before_owned_callback_or_dependency_construction(self):
        def poison():
            raise AssertionError("Held bridge called its external verifier")

        with patch.object(BRIDGE, "SOURCE_ONLY", True):
            with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
                BRIDGE.bind_owned_selection(poison)
            with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
                BRIDGE._draft_dependencies(owned_release=object())
            with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
                BRIDGE.create_application(
                    None,
                    owned_release=object(),
                    actor=None,
                    runtime=None,
                    torch=None,
                    model=None,
                    model_identity=None,
                    output=None,
                    cpu_identity=None,
                    backend_wrapper=None,
                )

    def test_full_four_heads_many_targets_and_unusual_order_remain_exact(self):
        value = payload()
        before = BRIDGE._json_bytes(value)
        BRIDGE._full_choice_scope(value)
        self.assertEqual(BRIDGE._json_bytes(value), before)
        self.assertEqual(
            list(value["questions"]),
            [
                "select_target",
                "operation",
                "type_text_target",
                "click_target",
            ],
        )
        self.assertEqual(
            list(value["questions"]["click_target"]["criteria"]),
            [
                "23",
                "3",
                "18",
                "8",
                "41",
            ],
        )
        self.assertEqual(set(value["questions"]["operation"]["criteria"]), BRIDGE.OPERATIONS)

    def test_available_target_heads_are_not_synthesized_or_coupled_to_operations(self):
        value = payload()
        value["questions"] = {"operation": value["questions"]["operation"]}
        before = copy.deepcopy(value)
        BRIDGE._full_choice_scope(value)
        self.assertEqual(value, before)  # TYPE_TEXT/SELECT remain offered without targets.

    def test_native_255_limit_is_retained_and_256_rejects_without_pruning(self):
        value = payload()
        criteria = {str(index): {"element": f"Item {index}"} for index in range(255)}
        value["questions"]["click_target"]["criteria"] = criteria
        BRIDGE._full_choice_scope(value)
        criteria["255"] = {"element": "Last offered item"}
        before = copy.deepcopy(value)
        with self.assertRaisesRegex(ValueError, "1..255"):
            BRIDGE._full_choice_scope(value)
        self.assertEqual(value, before)

    def test_nonstandard_freeform_media_and_missing_terminal_choices_reject_whole(self):
        for kind in ("unknown_head", "freeform", "media", "operation", "missing_done"):
            value = payload()
            if kind == "unknown_head":
                value["questions"]["text_value"] = {"type": "choice", "criteria": {"x": "x"}}
            elif kind == "freeform":
                value["questions"]["type_text_target"]["type"] = "text"
            elif kind == "media":
                value["media"] = [{"type": "image"}]
            elif kind == "operation":
                value["questions"]["operation"]["criteria"]["HOVER"] = "Unsupported"
            else:
                del value["questions"]["operation"]["criteria"]["DONE"]
            before = copy.deepcopy(value)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                BRIDGE._full_choice_scope(value)
            self.assertEqual(value, before)


class GatewayTests(unittest.TestCase):
    def gateway(self, root, clock, run):
        release = BRIDGE.bind_owned_selection(proof)
        output = BRIDGE._new_output(root / "evidence", owned_release=release)

        class Adapter:
            def __init__(self, backend, *, evidence_sink):
                self.backend, self.sink = backend, evidence_sink

            def predict(self, original):
                self.sink({"event": "planned", "data": copy.deepcopy(original)})
                return run(original)

        identity = {"device": "cpu", "precision": "bfloat16", "context_limit": 16384}
        return BRIDGE._Gateway(
            output=output,
            backend=SimpleNamespace(name=BRIDGE.BASE_MODEL_IDENTITY),
            adapter_type=Adapter,
            named_browser_response=lambda _request, result: result,
            cpu_identity=identity,
            timeout_type=TimeoutError,
            validation_type=ValueError,
            owned_release=release,
            clock=lambda: clock[0],
        ), output

    def request(self, gateway, value):
        body, messages = BRIDGE._json_bytes(value), []

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            messages.append(copy.deepcopy(message))

        asyncio.run(
            gateway(
                {"type": "http", "method": "POST", "path": "/v1/systemone"},
                receive,
                send,
            )
        )
        return body, messages

    def files(self, output):
        records = [path for path in output.iterdir() if path.is_dir()]
        self.assertEqual(len(records), 1)
        return records[0]

    def test_whole_invalid_request_is_retained_and_never_calls_adapter(self):
        calls, clock = [], [0.0]
        with (
            tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp,
            patch.object(BRIDGE, "SOURCE_ONLY", False),
        ):
            gateway, output = self.gateway(Path(temp), clock, lambda p: calls.append(p))
            value = payload()
            value["questions"]["foreign"] = copy.deepcopy(value["questions"]["click_target"])
            raw, messages = self.request(gateway, value)
            record = self.files(output)
            self.assertEqual(messages[0]["status"], 422)
            self.assertEqual(calls, [])
            self.assertEqual((record / "request.bin").read_bytes(), raw)
            self.assertEqual(json.loads((record / "request.json").read_bytes())["request"], value)
            self.assertFalse((record / "candidate.json").exists())

    def test_legal_wrong_blocked_choice_and_all_offered_values_are_not_repaired(self):
        calls, clock = [], [0.0]
        original = payload()

        def choose(value):
            calls.append(copy.deepcopy(value))
            return {"answers": [{"question_id": "operation", "choice": "BLOCKED"}]}

        with (
            tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp,
            patch.object(BRIDGE, "SOURCE_ONLY", False),
        ):
            gateway, output = self.gateway(Path(temp), clock, choose)
            _, messages = self.request(gateway, original)
            self.assertEqual(calls, [original])
            self.assertEqual(messages[0]["status"], 200)
            body = messages[1]["body"]
            self.assertEqual(json.loads(body)["answers"][0]["choice"], "BLOCKED")
            self.assertEqual((self.files(output) / "response.bin").read_bytes(), body)

    def test_final_terminal_writes_are_charged_including_boundary_equality(self):
        for name in ("candidate.json", "response.bin", "exchange.json"):
            for elapsed in (19.999, 20.0, 20.001):
                with (
                    self.subTest(name=name, elapsed=elapsed),
                    tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp,
                    patch.object(BRIDGE, "SOURCE_ONLY", False),
                ):
                    clock, advanced = [0.0], []
                    gateway, output = self.gateway(Path(temp), clock, lambda _p: {"answers": []})
                    original_write = BRIDGE._Record.write

                    def slow_write(record, filename, data, **kwargs):
                        original_write(record, filename, data, **kwargs)
                        if filename == name and not advanced:
                            clock[0] = elapsed
                            advanced.append(filename)

                    with patch.object(BRIDGE._Record, "write", slow_write):
                        _, messages = self.request(gateway, payload())
                    expected = 200 if elapsed < 20 else 504
                    self.assertEqual(messages[0]["status"], expected)
                    record, body = self.files(output), messages[1]["body"]
                    self.assertTrue((record / "candidate.json").exists())
                    self.assertEqual((record / "response.bin").read_bytes(), body)
                    self.assertEqual(
                        json.loads((record / "exchange.json").read_bytes())["status"], expected
                    )
                    headers = dict(messages[0]["headers"])
                    self.assertEqual(headers[b"content-length"], str(len(body)).encode("ascii"))
                    if expected == 504:
                        self.assertEqual(json.loads(body)["error"]["type"], "_DeadlineExceeded")

    def test_primary_abort_survives_failed_publication_and_serial_lock_releases(self):
        class OriginalAbort(BaseException):
            pass

        original = OriginalAbort("Original native failure")

        def fail(_payload):
            raise original

        with (
            tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp,
            patch.object(BRIDGE, "SOURCE_ONLY", False),
        ):
            gateway, output = self.gateway(Path(temp), [0.0], fail)
            original_write = BRIDGE._Record.write

            def failed_write(record, name, data, **kwargs):
                if name == "response.bin":
                    raise OSError("Secondary terminal retention failure")
                original_write(record, name, data, **kwargs)

            with (
                patch.object(BRIDGE._Record, "write", failed_write),
                self.assertRaises(OriginalAbort) as raised,
            ):
                self.request(gateway, payload())
            self.assertIs(raised.exception, original)
            self.assertTrue(
                any("Secondary terminal retention" in note for note in original.__notes__)
            )
            self.assertFalse(gateway.serial.locked())
            self.assertTrue((self.files(output) / "request.bin").exists())

    def test_missing_observer_retention_refuses_200_without_replacing_backend_error(self):
        for fail_backend in (False, True):
            with (
                self.subTest(fail_backend=fail_backend),
                tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp,
                patch.object(BRIDGE, "SOURCE_ONLY", False),
            ):

                def run(_payload):
                    if fail_backend:
                        raise RuntimeError("Original backend error after observer failure")
                    return {"answers": []}

                gateway, output = self.gateway(Path(temp), [0.0], run)
                original_write = BRIDGE._Record.write

                def failed_event(record, name, data, **kwargs):
                    if name.startswith("event-"):
                        raise OSError("Required observer retention failed")
                    original_write(record, name, data, **kwargs)

                with patch.object(BRIDGE._Record, "write", failed_event):
                    _, messages = self.request(gateway, payload())
                self.assertEqual(messages[0]["status"], 500)
                response = json.loads(messages[1]["body"])
                expected = (
                    "Original backend error after observer failure"
                    if fail_backend
                    else "Native observer evidence incomplete; successful HTTP refused"
                )
                self.assertEqual(response["error"]["message"], expected)
                record = self.files(output)
                exchange = json.loads((record / "exchange.json").read_bytes())
                self.assertEqual(exchange["observer_errors"][0]["type"], "OSError")
                self.assertFalse(gateway.serial.locked())


if __name__ == "__main__":
    unittest.main()
