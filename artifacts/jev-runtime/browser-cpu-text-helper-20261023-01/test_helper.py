"""Stdlib fake boundaries only; no HTTP network, model, torch, or browser."""

import asyncio
import contextlib
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "cpu_text_helper_under_test", Path(__file__).with_name("helper.py")
)
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
KEY = "independent-fake-worker-secret"


def request():
    # Fresh synthetic protocol data, never benchmark tasks/HTML/gold fields.
    context = {
        "goal": uuid.uuid4().hex,
        "field": {"label": "", "role": "textbox", "value": ""},
        "page": {"title": "", "text": ""},
        "recent_actions": [],
    }
    return {
        "model": h.MODEL_ID,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
        "reasoning": {"enabled": False},
        "messages": [
            {"role": "system", "content": h.TEXT_VALUE},
            {"role": "user", "content": json.dumps(context, separators=(",", ":"))},
        ],
    }


class Clock:
    value = 100.0

    def __call__(self):
        return self.value


class Authority:
    def __init__(self):
        self.events = []

    def check(self, phase, **kwargs):
        self.events.append((phase, kwargs))

    def arm(self, **kwargs):
        self.events.append(("arm", kwargs))


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.raw = json.dumps({"text": uuid.uuid4().hex})
        self.failure = None

    def generate(self, messages, deadline, checkpoint):
        self.calls.append(copy.deepcopy(messages))
        checkpoint("fake-native-forward")
        if self.failure:
            raise self.failure
        return self.raw, {
            "model": h.MODEL_ID,
            "raw_output": self.raw,
            "usage": {"input_tokens": 3, "output_tokens": 3},
        }


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (
            (len(values), len(values[0]))
            if values and isinstance(values[0], list)
            else (len(values),)
        )
        self.device = SimpleNamespace(type="cpu")

    def __getitem__(self, item):
        return Tensor(self.values[item])

    def tolist(self):
        return self.values


class Tokenizer:
    pad_token_id = 0

    def __init__(self, count=2):
        self.ids, self.seen = [10] * count, None
        self.raw = json.dumps({"text": uuid.uuid4().hex})

    def apply_chat_template(self, messages, **kwargs):
        self.seen = (copy.deepcopy(messages), kwargs)
        return "native-template:" + json.dumps(messages)

    def __call__(self, prompt, **kwargs):
        self.tokenization = (prompt, kwargs)
        return {"input_ids": Tensor([self.ids])}

    def decode(self, ids, **kwargs):
        self.decoding = (ids, kwargs)
        return self.raw


class Model:
    def __init__(self, tok, emitted):
        self.tok, self.emitted, self.removed, self.calls = tok, emitted, False, []

    def register_forward_pre_hook(self, hook):
        self.hook = hook
        return SimpleNamespace(remove=lambda: setattr(self, "removed", True))

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        self.hook(self, None)
        kwargs["streamer"].put(Tensor([self.tok.ids]))
        for token in self.emitted:
            kwargs["streamer"].put(Tensor([token]))
        kwargs["streamer"].end()
        return Tensor([self.tok.ids + self.emitted])


def native_stub(count=2, emitted=None):
    tok = Tokenizer(count)
    model = Model(tok, [44, 151645] if emitted is None else emitted)
    torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
    transformers = SimpleNamespace(StoppingCriteria=object, StoppingCriteriaList=list)
    runtime = h._TorchRuntime(
        torch, transformers, model, tok, (151645,), Authority(), "f" * 64, h._VERIFIED
    )
    return runtime, model, tok


class ProtocolTests(unittest.TestCase):
    def completed_metadata(self):
        selection = json.loads(
            Path(__file__).with_name("runtime-selection.pending.json").read_text()
        )
        for name, entry in selection["model_files"].items():
            entry["path"] = "/nonexistent/selected/" + name
            if entry["sha256"] is None:
                entry["sha256"] = "1" * 64
        selection["owner"]["profile"] = "independent-fake-metadata-profile"
        selection["runtime_binary_files"] = [
            {"path": "/nonexistent/torch/_C.so", "size_bytes": 1, "sha256": "1" * 64},
            {"path": "/nonexistent/tokenizers/tokenizers.so", "size_bytes": 1, "sha256": "1" * 64},
        ]
        return selection

    def test_original_messages_are_preserved(self):
        payload = request()
        self.assertEqual(h.validate_request(h.encode(payload)), payload["messages"])

    def test_request_rejects_prompt_or_reasoning_changes(self):
        for change in (
            lambda p: p.update(max_tokens=128),
            lambda p: p.update(reasoning={"enabled": 0}),
            lambda p: p["messages"][0].update(content="changed"),
            lambda p: p.update(extra=True),
        ):
            payload = request()
            change(payload)
            with self.assertRaises(h.Refusal):
                h.validate_request(h.encode(payload))

    def test_json_rejects_duplicates_nonfinite_and_overflow(self):
        for blob in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}'):
            with self.assertRaises(h.Refusal):
                h.strict_json(blob)

    def test_output_requires_whole_exact_json_and_preserves_string(self):
        value = '  "opaque" \\ 台灣  '
        self.assertEqual(h.validate_output(json.dumps({"text": value})), value)
        for raw in (
            '```json\n{"text":"x"}\n```',
            '{"text":"x"} trailing',
            '{"text":null}',
            '{"text":"x","extra":1}',
            '{"text":""}',
            '{"text":"x","text":"y"}',
            json.dumps({"text": "a" * 2001}),
        ):
            with self.assertRaises((ValueError, h.Refusal)):
                h.validate_output(raw)

    def test_pending_selection_refuses_before_payload_or_heavy_import(self):
        path = Path(__file__).with_name("runtime-selection.pending.json")
        before = set(sys.modules)
        with (
            self.assertRaises(h.Refusal),
            patch.object(h, "_pinned_file", side_effect=AssertionError("payload open")),
        ):
            h.load_runtime(path, h.digest(path.read_bytes()))
        self.assertFalse((set(sys.modules) - before) & {"torch", "transformers", "tokenizers"})

    def test_production_factory_refuses_arbitrary_fake_runtime(self):
        with self.assertRaises(h.Refusal):
            h.make_application(FakeRuntime(), lambda *_args: None, KEY)
        runtime, _model, _tok = native_stub()
        with self.assertRaises(h.Refusal):
            h.make_application(runtime, lambda *_args: None, KEY)

    def test_all_runtime_origins_and_critical_sources_must_be_pinned(self):
        selection = self.completed_metadata()
        h.validate_selection(selection)
        selection["runtime_files"] = selection["runtime_files"][1:]
        with self.assertRaises(h.Refusal):
            h.validate_selection(selection)
        selection = self.completed_metadata()
        selection["runtime_files"] = [
            entry
            for entry in selection["runtime_files"]
            if not entry["path"].endswith("transformers/generation/utils.py")
        ]
        with self.assertRaises(h.Refusal):
            h.validate_selection(selection)
        selection = self.completed_metadata()
        selection["versions"]["torch"] = None
        with self.assertRaises(h.Refusal):
            h.validate_selection(selection)

    def test_publisher_native_file_identity_cannot_be_reselected(self):
        selection = self.completed_metadata()
        selection["model_files"]["generation_config.json"]["publisher_git_blob_sha1"] = "0" * 40
        with self.assertRaises(h.Refusal):
            h.validate_selection(selection)

    def test_wrong_import_origin_refuses_before_heavy_import(self):
        selection = {"module_origins": {"torch": "/selected/torch/__init__.py"}}
        with patch.object(
            h.importlib.util,
            "find_spec",
            return_value=SimpleNamespace(origin="/other/torch/__init__.py"),
        ):
            with self.assertRaises(h.Refusal):
                h._check_import_origins(selection)
        self.assertNotIn("torch", sys.modules)

    def test_duplicate_context_key_is_refused(self):
        payload = request()
        payload["messages"][1]["content"] = '{"goal":"x","goal":"y"}'
        with self.assertRaises(h.Refusal):
            h.validate_request(h.encode(payload))

    def test_journal_is_exclusive_and_hash_checks_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "new-journal"
            sink = h.DirectoryJournal(path)
            sink("f" * 32, "original_request", {"body_base64": "AA=="})
            self.assertEqual(len(list(path.iterdir())), 1)
            with self.assertRaises(FileExistsError):
                h.DirectoryJournal(path)
            file = Path(temp) / "selected.py"
            file.write_bytes(b"known")
            entry = {"path": str(file), "size_bytes": 5, "sha256": h.digest(b"known")}
            h._pinned_file(entry)
            file.write_bytes(b"drift")
            with self.assertRaises(h.Refusal):
                h._pinned_file(entry)


class NativeGenerationTests(unittest.TestCase):
    def test_native_contract_and_actual_eos_capture(self):
        runtime, model, tok = native_stub()
        messages = request()["messages"]
        with patch.object(runtime, "check"):
            raw, metadata = runtime.generate(
                messages, h.Deadline(100, Clock()), lambda _phase: None
            )
        self.assertEqual(raw, tok.raw)
        self.assertEqual(tok.seen, (messages, {"tokenize": False, "add_generation_prompt": True}))
        self.assertFalse(tok.tokenization[1]["truncation"])
        self.assertEqual(metadata["raw_token_ids"], [44, 151645])
        self.assertEqual(tok.decoding[0], [44])
        self.assertFalse(tok.decoding[1]["skip_special_tokens"])
        options = model.calls[0]
        self.assertEqual(
            (
                options["max_new_tokens"],
                options["do_sample"],
                options["num_beams"],
                options["logits_to_keep"],
                options["use_cache"],
            ),
            (128, False, 1, 1, True),
        )
        self.assertTrue(model.removed)

    def test_missing_early_or_overlength_eos_refuses(self):
        for emitted in ([44] * 128, [151645, 44, 151645], [44] * 128 + [151645], []):
            runtime, model, _tok = native_stub(emitted=emitted)
            with patch.object(runtime, "check"), self.assertRaises(h.Refusal) as failure:
                runtime.generate(
                    request()["messages"], h.Deadline(100, Clock()), lambda _phase: None
                )
            self.assertTrue(model.removed)
            self.assertEqual(failure.exception.generation["raw_token_ids"], emitted)

    def test_over_context_refuses_without_generation_or_truncation(self):
        runtime, model, tok = native_stub(count=8065)
        with patch.object(runtime, "check"), self.assertRaises(h.Refusal):
            runtime.generate(request()["messages"], h.Deadline(100, Clock()), lambda _phase: None)
        self.assertFalse(tok.tokenization[1]["truncation"])
        self.assertEqual(model.calls, [])

    def test_forward_deadline_refuses_and_removes_hook(self):
        runtime, model, _tok = native_stub()

        def checkpoint(phase):
            if phase == "before-native-forward":
                raise TimeoutError("synthetic deadline")

        with patch.object(runtime, "check"), self.assertRaises(TimeoutError):
            runtime.generate(request()["messages"], h.Deadline(100, Clock()), checkpoint)
        self.assertTrue(model.removed)


class ApplicationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock, self.runtime, self.authority, self.events = (
            Clock(),
            FakeRuntime(),
            Authority(),
            [],
        )
        self.delay_event = None
        self.fail_event = None

        def journal(_rid, event, data):
            self.events.append((event, data))
            if event == self.fail_event:
                raise OSError("synthetic evidence failure")
            if event == self.delay_event:
                self.clock.value += 20.0

        self.app = h._Application(self.runtime, journal, self.authority, KEY, self.clock)

    async def invoke(self, payload=None, auth=True, send_hook=None, receive_message=None):
        payload = request() if payload is None else payload
        sent = []

        async def receive():
            return receive_message or {
                "type": "http.request",
                "body": h.encode(payload),
                "more_body": False,
            }

        async def send(message):
            sent.append(message)
            if send_hook:
                send_hook(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/chat/completions",
            "server": ("127.0.0.1", 12345),
            "client": ("127.0.0.1", 23456),
            "headers": [(b"authorization", ("Bearer " + KEY).encode())] if auth else [],
        }
        await self.app(scope, receive, send)
        return sent, payload

    async def test_success_uses_original_message_and_retains_before_publish(self):
        sent, payload = await self.invoke()
        self.assertEqual(sent[0]["status"], 200)
        result = json.loads(sent[1]["body"])
        self.assertEqual(result["choices"][0]["message"]["content"], self.runtime.raw)
        self.assertEqual(self.runtime.calls, [payload["messages"]])
        self.assertEqual(
            [event for event, _ in self.events],
            ["original_request", "native_generation", "response_candidate"],
        )
        self.assertTrue(any(phase == "arm" for phase, _ in self.authority.events))

    async def test_wrong_auth_disconnected_and_invalid_output_refuse(self):
        sent, _ = await self.invoke(auth=False)
        self.assertEqual(sent[0]["status"], 502)
        self.assertEqual(self.runtime.calls, [])
        sent, _ = await self.invoke(receive_message={"type": "http.disconnect"})
        self.assertEqual(sent[0]["status"], 502)
        self.runtime.raw = '{"text":null}'
        sent, _ = await self.invoke()
        self.assertEqual(sent[0]["status"], 502)

    async def test_retention_time_is_inside_original_deadline(self):
        for event in ("original_request", "native_generation", "response_candidate"):
            self.setUp()
            self.delay_event = event
            sent, _ = await self.invoke()
            self.assertEqual(sent[0]["status"], 504)
            self.assertNotIn(self.runtime.raw.encode(), sent[1]["body"])

    async def test_late_response_start_never_sends_candidate_body(self):
        sent = []

        def hook(message):
            sent.append(message)
            if message["type"] == "http.response.start":
                self.clock.value += 20

        with self.assertRaises(TimeoutError):
            await self.invoke(send_hook=hook)
        self.assertEqual([message["type"] for message in sent], ["http.response.start"])

    async def test_late_body_send_is_unknown_and_cannot_be_retracted_or_replaced(self):
        sent = []

        def hook(message):
            sent.append(message)
            if message["type"] == "http.response.body":
                self.clock.value += 20

        with self.assertRaises(TimeoutError):
            await self.invoke(send_hook=hook)
        self.assertEqual(
            [message["type"] for message in sent], ["http.response.start", "http.response.body"]
        )
        candidates = [data for event, data in self.events if event == "response_candidate"]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["client_delivery"], "unknown")

    async def test_queue_time_is_inside_original_deadline(self):
        await self.app.serial.acquire()
        task = asyncio.create_task(self.invoke())
        await asyncio.sleep(0)
        self.clock.value += 20
        self.app.serial.release()
        sent, _ = await task
        self.assertEqual(sent[0]["status"], 504)
        self.assertEqual(self.runtime.calls, [])

    async def test_evidence_failure_is_latched_across_requests(self):
        self.fail_event = "native_generation"
        sent, _ = await self.invoke()
        self.assertEqual(sent[0]["status"], 502)
        calls = len(self.runtime.calls)
        self.fail_event = None
        sent, _ = await self.invoke()
        self.assertEqual(sent[0]["status"], 502)
        self.assertEqual(len(self.runtime.calls), calls)

    async def test_cancellation_cannot_publish_candidate(self):
        self.runtime.failure = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.invoke()
        self.assertNotIn("response_candidate", [event for event, _ in self.events])


if __name__ == "__main__":
    unittest.main()
