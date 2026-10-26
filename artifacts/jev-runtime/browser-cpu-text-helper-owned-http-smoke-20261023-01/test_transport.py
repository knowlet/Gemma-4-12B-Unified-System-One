"""Pure HTTP orchestration/client fakes. No sockets, server imports or model runtime."""

import asyncio
import builtins
import hashlib
import importlib.util
import json
import os
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_http_transport_test", HERE / "transport.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


class Clock:
    value = 100.0

    def __call__(self):
        return self.value


class Response:
    http_version = "HTTP/1.1"

    def __init__(self, status=200, content=None):
        self.status_code, self.is_error = status, status >= 400
        self.content = (
            content
            or json.dumps(
                {"choices": [{"message": {"content": '{"text":"opaque"}'}}], "usage": {}}
            ).encode()
        )

    def json(self):
        return json.loads(self.content)


class Client:
    def __init__(self, **kwargs):
        if kwargs != {"http2": True, "timeout": 25}:
            raise AssertionError("Original Client constructor changed")
        self.timeout = types.SimpleNamespace(
            as_dict=lambda: {k: 25 for k in ("connect", "read", "write", "pool")}
        )
        self._transport = types.SimpleNamespace(_pool=types.SimpleNamespace(_http2=True))
        self.responses, self.seen, self.closed = [Response()], [], False

    def post(self, url, **kwargs):
        self.seen.append((url, kwargs))
        return self.responses.pop(0)

    def close(self):
        self.closed = True


class Listener:
    def __init__(self, *_args):
        self.closed = False

    def bind(self, address):
        assert address == ("127.0.0.1", 0)

    def listen(self, backlog):
        assert backlog == 1

    def setblocking(self, value):
        assert value is False

    def getsockname(self):
        return "127.0.0.1", 12345

    def close(self):
        self.closed = True


class Config:
    def __init__(self, app, **kwargs):
        self.app, self.kwargs = app, kwargs

    def load(self):
        self.http_protocol_class = types.SimpleNamespace(
            __module__="uvicorn.protocols.http.h11_impl"
        )
        self.ws_protocol_class = None
        self.lifespan_class = types.SimpleNamespace(__module__="uvicorn.lifespan.off")


class Server:
    def __init__(self, config):
        self.config, self.started, self.should_exit = config, False, False

    async def serve(self, sockets):
        assert len(sockets) == 1
        self.started = True
        while not self.should_exit:
            await asyncio.sleep(0)


class Tests(unittest.TestCase):
    def tearDown(self):
        for name in list(sys.modules):
            if name.startswith("_owned_one_shot_http_upstream"):
                del sys.modules[name]

    def upstream(self):
        manifest = json.loads((HERE / "http-stack.json").read_text())

        def checked(row):
            return Path(row["path"]).read_bytes()

        with patch.dict(
            sys.modules, {"httpx": types.SimpleNamespace(Client=Client, HTTPError=OSError)}
        ):
            return t.load_upstream(manifest, checked)

    def test_upstream_original_constructor_payload_reasoning_and_retry_unchanged(self):
        model = self.upstream()
        model.CLIENT.responses = [Response(503), Response(529), Response()]
        context = {"goal": "opaque"}
        variables = {
            "TEXT_MODEL_API_KEY": "key",
            "TEXT_MODEL_BASE_URL": "http://127.0.0.1:12345/v1/",
            "TEXT_MODEL": "Qwen/Qwen2.5-0.5B-Instruct",
            "TEXT_MODEL_REASONING": "none",
        }
        with patch.dict(os.environ, variables), patch.object(model.time, "sleep") as sleep:
            value, _ = model.field_text(context)
        self.assertEqual(value, "opaque")
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.5, 1.0])
        self.assertEqual(len(model.CLIENT.seen), 3)
        for url, kwargs in model.CLIENT.seen:
            self.assertEqual(url, "http://127.0.0.1:12345/v1/chat/completions")
            self.assertEqual(kwargs["json"]["max_tokens"], 1024)
            self.assertEqual(kwargs["json"]["reasoning"], {"enabled": False})
            self.assertEqual(kwargs["json"]["messages"][0]["content"], model.TEXT_VALUE)
            self.assertEqual(kwargs["json"]["messages"][1]["content"], json.dumps(context))
        self.assertNotIn("_owned_one_shot_http_upstream.agent", sys.modules)

    def test_private_package_reads_only_exact_questions_model_and_preserves_code(self):
        manifest = json.loads((HERE / "http-stack.json").read_text())
        opened = []

        def checked(row):
            name = Path(row["path"]).name
            self.assertIn(name, {"questions.py", "model.py"})
            opened.append(name)
            blob = Path(row["path"]).read_bytes()
            self.assertEqual(len(blob), row["size_bytes"])
            self.assertEqual(hashlib.sha256(blob).hexdigest(), row["sha256"])
            return blob

        original_import = builtins.__import__

        def only_selected(name, globals=None, locals=None, fromlist=(), level=0):
            self.assertTrue(
                (level == 1 and name == "questions")
                or (level == 0 and name in {"json", "math", "os", "time", "httpx"}),
                "Unselected package/__init__/Browser import: " + name,
            )
            return original_import(name, globals, locals, fromlist, level)

        with (
            patch.dict(
                sys.modules, {"httpx": types.SimpleNamespace(Client=Client, HTTPError=OSError)}
            ),
            patch.object(builtins, "__import__", side_effect=only_selected),
        ):
            model = t.load_upstream(manifest, checked)
            package = sys.modules["_owned_one_shot_http_upstream"]
            questions = sys.modules["_owned_one_shot_http_upstream.questions"]
            self.assertNotIn("_owned_one_shot_http_upstream.agent", sys.modules)
            self.assertNotIn("_owned_one_shot_http_upstream.browser", sys.modules)
        self.assertEqual(opened, ["questions.py", "model.py"])
        self.assertEqual(package.__path__, [])
        self.assertFalse(hasattr(package, "__file__"))
        self.assertIs(model.TEXT_VALUE, questions.TEXT_VALUE)
        row = manifest["upstream"]["model.py"]
        code = compile(Path(row["path"]).read_bytes(), row["path"], "exec", dont_inherit=True)
        original_functions = {
            item.co_name: item for item in code.co_consts if isinstance(item, types.CodeType)
        }
        for name in ("field_text", "post_json"):
            self.assertEqual(getattr(model, name).__code__, original_functions[name])
        self.assertNotIn("_owned_one_shot_http_upstream.agent", sys.modules)
        self.assertNotIn("_owned_one_shot_http_upstream.browser", sys.modules)

    def test_observer_returns_same_response_forwards_same_objects_and_restores(self):
        model = self.upstream()
        response = model.CLIENT.responses[0]
        original = model.CLIENT.post
        retained = []
        attempts, restore = t.observe_client(
            model, "http://127.0.0.1:12345/v1", "key", lambda r: retained.append(r.copy())
        )
        body, headers = {"unchanged": "synthetic"}, {"Authorization": "Bearer key"}
        observed = model.CLIENT.post(
            "http://127.0.0.1:12345/v1/chat/completions", json=body, headers=headers
        )
        self.assertIs(observed, response)
        self.assertIs(model.CLIENT.seen[0][1]["json"], body)
        self.assertIs(model.CLIENT.seen[0][1]["headers"], headers)
        self.assertEqual(attempts[0]["body_size_bytes"], len(response.content))
        self.assertEqual(len(retained), 1)
        restore()
        self.assertEqual(model.CLIENT.post, original)

    def test_observer_preserves_primary_when_evidence_also_fails(self):
        model = self.upstream()
        primary = OSError("synthetic original failure")
        model.CLIENT.post = Mock(side_effect=primary)
        t.observe_client(
            model, "http://127.0.0.1:12345/v1", "key", Mock(side_effect=RuntimeError("evidence"))
        )
        with self.assertRaises(OSError) as caught:
            model.CLIENT.post(
                "http://127.0.0.1:12345/v1/chat/completions",
                headers={"Authorization": "Bearer key"},
            )
        self.assertIs(caught.exception, primary)
        self.assertTrue(any("evidence" in note for note in primary.__notes__))

    def test_original_502_refusal_does_not_retry(self):
        model = self.upstream()
        model.CLIENT.responses = [Response(502)]
        with (
            self.assertRaisesRegex(RuntimeError, "502"),
            patch.object(model.time, "sleep") as sleep,
        ):
            model.post_json("http://127.0.0.1/v1/chat/completions", "key", {})
        self.assertEqual(len(model.CLIENT.seen), 1)
        sleep.assert_not_called()

    def test_second_request_never_reaches_application_or_another_arm(self):
        calls, sent = [], []

        async def app(*args):
            calls.append(args)

        async def send(message):
            sent.append(message)

        async def exercise():
            wrapped = t.OneRequest(app)
            await wrapped({"type": "http"}, None, send)
            await wrapped({"type": "http"}, None, send)
            self.assertEqual(wrapped.extra, 1)

        asyncio.run(exercise())
        self.assertEqual(len(calls), 1)
        self.assertEqual(sent[0]["status"], 409)

    def run_fake(self, *, advance=0, check=None, close_failure=False, client_failure=None):
        model, clock = self.upstream(), Clock()
        main_thread = threading.get_ident()
        original = model.CLIENT.post

        def post(url, **kwargs):
            self.assertNotEqual(threading.get_ident(), main_thread)
            clock.value += advance
            if client_failure is not None:
                raise client_failure
            return original(url, **kwargs)

        model.CLIENT.post = post
        listener = Listener()
        if close_failure:
            listener.close = Mock(side_effect=RuntimeError("listener cleanup"))

        class InvokingServer(Server):
            async def serve(self, sockets):
                # A synthetic delivery exercises the one-request wrapper; no TCP/HTTP occurs.
                async def no_message(*_args):
                    return None

                await self.config.app({"type": "http"}, no_message, no_message)
                await super().serve(sockets)

        async def app(*_args):
            return None

        result = asyncio.run(
            t.run_one(
                app,
                model,
                types.SimpleNamespace(Config=Config, Server=InvokingServer),
                t.SERVER_SETTINGS,
                {"goal": "opaque"},
                "key",
                lambda _r: {"synthetic": "private evidence"},
                check or (lambda _phase: None),
                clock=clock,
                socket_factory=lambda *_args: listener,
            )
        )
        self.assertTrue(listener.closed)
        self.assertTrue(model.CLIENT.closed)
        self.assertEqual(model.CLIENT.post, post)
        return result

    def test_server_is_not_blocked_by_sync_client_and_cleanup_completes(self):
        result = self.run_fake()
        self.assertEqual(result["value"], "opaque")
        self.assertEqual(result["server_request_count"], 1)
        self.assertNotIn("body_base64", result["attempts"][0])

    def test_client_elapsed20_is_observation_only_not_extra_scoregate(self):
        result = self.run_fake(advance=21)
        self.assertFalse(result["client_elapsed_lt20_observation_only"])
        self.assertEqual(result["value"], "opaque")

    def test_post_cleanup_parent_refusal_cannot_return_success(self):
        def check(phase):
            if phase == "HTTP-after-server-client-cleanup":
                raise TimeoutError("synthetic original parent expired during cleanup")

        with self.assertRaisesRegex(TimeoutError, "parent expired"):
            self.run_fake(check=check)

    def test_success_cleanup_failure_refuses_and_failure_primary_is_preserved(self):
        with self.assertRaisesRegex(RuntimeError, "listener cleanup"):
            self.run_fake(close_failure=True)
        self.tearDown()
        primary = ValueError("primary client failure")
        with self.assertRaises(ValueError) as caught:
            self.run_fake(close_failure=True, client_failure=primary)
        self.assertIs(caught.exception, primary)
        self.assertTrue(any("Listener" in note for note in primary.__notes__))

    def test_auto_protocol_policy_refuses_before_source_import_or_socket(self):
        manifest = json.loads((HERE / "http-stack.json").read_text())
        manifest["server_settings"]["http"] = "auto"
        check = Mock(side_effect=AssertionError("source access"))
        with self.assertRaisesRegex(ValueError, "explicit HTTP"):
            t.load_stack(manifest, check)
        check.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
