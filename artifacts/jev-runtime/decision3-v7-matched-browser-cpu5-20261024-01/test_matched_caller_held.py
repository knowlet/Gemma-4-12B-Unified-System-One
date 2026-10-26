"""Authored stdlib protocol fakes only; no checks or actual/data reads performed."""

import base64
import importlib.util
import json
import os
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


def module():
    path = Path(__file__).with_name("caller_held.py")
    spec = importlib.util.spec_from_file_location("_held_matched_caller_fake", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Headers:
    raw = [(b"X-Repeat", b"first"), (b"x-repeat", b"second")]

    def get(self, key):
        return "invented-request-id" if key == "X-Request-ID" else None


def response():
    request = types.SimpleNamespace(content=b'{ "actual" : [1] }', headers=Headers())
    return types.SimpleNamespace(
        request=request, content=b"\x00\xfforiginal-response", headers=Headers(), status_code=200
    )


class HTTPProtocolTests(unittest.TestCase):
    def setUp(self):
        self.actor = module()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.recorder = self.actor.HTTPRecorder(Path(self.temp.name) / "http")

    def test_once_original_kwargs_response_and_actual_raw_bytes_and_headers(self):
        raw = response()
        nested, calls = {"not": "the already serialized HTTP request"}, []

        def post(url, **kwargs):
            self.assertIs(kwargs["json"], nested)
            calls.append((url, kwargs))
            return raw

        result = self.recorder.wrap(post)("http://invented/policy", json=nested, timeout=25)
        self.assertIs(result, raw)
        self.assertEqual(len(calls), 1)
        trace = json.loads(Path(self.recorder.finish()["path"]).read_bytes())
        row = trace["attempts"][0]
        self.assertEqual(Path(row["request"]["path"]).read_bytes(), raw.request.content)
        self.assertEqual(Path(row["response"]["path"]).read_bytes(), raw.content)
        self.assertEqual(row["request_sha256"], self.actor.sha(raw.request.content))
        self.assertEqual(row["response_sha256"], self.actor.sha(raw.content))
        self.assertEqual(
            [[base64.b64decode(k), base64.b64decode(v)] for k, v in row["request_headers_base64"]],
            [[k, v] for k, v in Headers.raw],
        )
        self.assertEqual(row["server_request_id"], "invented-request-id")
        self.assertTrue(row["capture_complete"])

    def test_transport_error_identity_and_each_original_retry_are_retained(self):
        raw, calls = response(), []
        original = OSError("invented transport failure")
        original.request = raw.request

        def post(url, **kwargs):
            calls.append(url)
            if len(calls) == 1:
                raise original
            return raw

        observed = self.recorder.wrap(post)
        with self.assertRaises(OSError) as caught:
            observed("http://invented/policy", timeout=25)
        self.assertIs(caught.exception, original)
        self.assertEqual(len(calls), 1)  # Observer did not add a retry.
        self.assertIs(observed("http://invented/policy", timeout=25), raw)
        self.assertEqual(len(calls), 2)  # Represents the unchanged upstream retry.
        trace = json.loads(Path(self.recorder.finish()["path"]).read_bytes())
        self.assertEqual(trace["http_attempts"], 2)
        self.assertIsNone(trace["attempts"][0]["response"])
        self.assertTrue(trace["capture_complete"])

    def test_retention_exception_does_not_create_retry_or_qualify_completion(self):
        raw, calls = response(), []
        failure = OSError("invented required retention failure")

        def post(url, **kwargs):
            calls.append(url)
            return raw

        with patch.object(self.actor, "retain", side_effect=failure):
            self.assertIs(self.recorder.wrap(post)("http://invented/policy"), raw)
        self.assertEqual(len(calls), 1)
        self.assertFalse(self.recorder.rows[0]["capture_complete"])
        with self.assertRaises(OSError) as caught:
            self.recorder.finish()
        self.assertIs(caught.exception, failure)

    def test_original_baseexception_survives_secondary_retention_failure(self):
        primary, secondary = KeyboardInterrupt("original abort"), OSError("retention")
        primary.request = response().request

        def post(*args, **kwargs):
            raise primary

        with patch.object(self.actor, "retain", side_effect=secondary):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.recorder.wrap(post)("http://invented/policy")
        self.assertIs(caught.exception, primary)
        self.assertTrue(any("HTTP observation also failed" in note for note in primary.__notes__))

    def test_final_trace_failure_does_not_replace_first_latched_capture_error(self):
        primary = OSError("invented first capture failure")
        secondary = OSError("invented final trace failure")
        self.recorder.failures.append(primary)
        with patch.object(self.actor, "retain_json", side_effect=secondary):
            with self.assertRaises(OSError) as caught:
                self.recorder.finish()
        self.assertIs(caught.exception, primary)
        self.assertTrue(
            any("Final HTTP trace retention also failed" in note for note in primary.__notes__)
        )

    def test_context_restores_original_post_loader_and_preserves_arguments(self):
        raw, arguments = response(), []

        def original_post(url, **kwargs):
            return raw

        model = types.SimpleNamespace(CLIENT=types.SimpleNamespace(post=original_post))

        @contextmanager
        def loader(*args, **kwargs):
            arguments.append((args, kwargs))
            yield object(), model

        runner = types.SimpleNamespace(load_upstream=loader)
        abort = KeyboardInterrupt("invented case abort")
        with self.assertRaises(KeyboardInterrupt):
            with self.actor.observe_http(runner, self.recorder):
                with runner.load_upstream("original-path", observation_runtime="unchanged"):
                    self.assertIs(model.CLIENT.post("http://invented/policy"), raw)
                    raise abort
        self.assertIs(runner.load_upstream, loader)
        self.assertIs(model.CLIENT.post, original_post)
        self.assertEqual(arguments, [(("original-path",), {"observation_runtime": "unchanged"})])


class MatchedContractTests(unittest.TestCase):
    def setUp(self):
        self.actor = module()

    def records(self):
        return [
            {
                "case_id": case,
                "group": "known4" if case in self.actor.KNOWN4 else "public6",
                "profile": self.actor.PROFILE,
                "profile_helper_sha256": self.actor.PROFILE_HELPER_SHA256,
                "seed": 20261014,
                "repeats": 1,
                "max_steps": 16,
                "text_model": "generated-actions-nf4",
                "policy_model": "invented-policy",
                "text_endpoint": "http://invented-nf4-helper",
                "text_model_reasoning": "none",
                "action_profile": "standard",
                "infrastructure_complete": True,
                "task_success": index < 4,
            }
            for index, case in enumerate(self.actor.CASE_IDS)
        ]

    def test_false_tasks_keep_ten_denominator_and_incomplete_infra_has_no_score(self):
        rows = self.records()
        result = self.actor.aggregate_cases(rows)
        self.assertEqual(
            (result["successes"], result["denominator"], result["failed_tasks"]), (4, 10, 6)
        )
        self.assertEqual(result["success_rate"], 0.4)
        rows[-1].update(infrastructure_complete=False, task_success=None)
        result = self.actor.aggregate_cases(rows)
        self.assertEqual(result["denominator"], 10)
        self.assertIsNone(result["success_rate"])
        self.assertFalse(result["infrastructure_complete"])

    def test_missing_duplicate_or_mismatched_profile_never_produces_main_score(self):
        for mutation in (
            "missing",
            "duplicate",
            "steps",
            "helper",
            "policy",
            "order",
            "reasoning",
            "action-profile",
            "helper-endpoint",
        ):
            rows = self.records()
            if mutation == "missing":
                rows.pop()
            elif mutation == "duplicate":
                rows[-1]["case_id"] = rows[0]["case_id"]
            elif mutation == "steps":
                rows[0]["max_steps"] = 8
            elif mutation == "helper":
                rows[0]["text_model"] = "invented-policy"
            elif mutation == "policy":
                rows[0]["policy_model"] = "different-policy"
            elif mutation == "order":
                rows[0], rows[1] = rows[1], rows[0]
            elif mutation == "reasoning":
                rows[0]["text_model_reasoning"] = "low"
            elif mutation == "action-profile":
                rows[0]["action_profile"] = "invented-experimental"
            else:
                rows[0]["text_endpoint"] = "http://different-helper"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.actor.aggregate_cases(rows)

    def test_default_hold_and_pending_pins_precede_arguments_paths_or_callbacks(self):
        class Bomb:
            def __fspath__(self):
                raise AssertionError("path read")

        with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
            self.actor.main(Bomb())
        with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
            self.actor.call_case(
                Bomb(),
                Bomb(),
                policy_endpoint=Bomb(),
                policy_model=Bomb(),
                text_endpoint=Bomb(),
                upstream=Bomb(),
                chromium=Bomb(),
                capability_sha256=Bomb(),
                verify_owned=Bomb(),
            )
        with patch.object(self.actor, "SOURCE_ONLY", False):
            with self.assertRaisesRegex(ValueError, "pins remain pending"):
                self.actor.main(Bomb())

    def test_call_case_forwards_original_settings_and_returns_false_task(self):
        case, captured = self.actor.PUBLIC6[0], {}
        guards = []

        def prepare(path, *, case_module):
            self.assertEqual(case_module, "heldout")
            path.mkdir()
            (path / "bundle-manifest.json").write_bytes(b"{}\n")
            return path, {
                "profile": self.actor.PROFILE,
                "case_module": case_module,
                "profile_helper_sha256": self.actor.PROFILE_HELPER_SHA256,
                "files": {"runner.py": {"sha256": "c" * 64}},
            }

        def benchmark(**kwargs):
            captured.update(kwargs)
            captured["reasoning_in_original_runner"] = os.environ.get("TEXT_MODEL_REASONING")
            result = {
                "attempts": 1,
                "outcomes": [{"case_id": case + "-000", "task_id": case, "success": False}],
            }
            kwargs["output_dir"].mkdir()
            (kwargs["output_dir"] / "summary.json").write_text(json.dumps(result))
            return result

        original_loader = object()  # The fake runner never opens upstream.
        v7 = types.SimpleNamespace(
            PROFILE=self.actor.PROFILE, prepare_bundle=prepare, verify_bundle=lambda *_args: None
        )
        runner = types.SimpleNamespace(load_upstream=original_loader, run_benchmark=benchmark)

        def selected_source(name, path, expected):
            return v7 if "launcher" in name else runner

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(self.actor, "SOURCE_ONLY", False),
            patch.object(self.actor, "CAPABILITY_SHA256", "a" * 64),
            patch.object(self.actor, "V7_LAUNCHER_SHA256", "b" * 64),
            patch.object(self.actor, "import_source", side_effect=selected_source),
            patch.dict(os.environ, {"TEXT_MODEL_REASONING": "invented-previous-low"}),
        ):
            record = self.actor.call_case(
                case,
                Path(temporary) / "case",
                policy_endpoint="http://invented/policy",
                policy_model="invented-policy",
                text_endpoint="http://different-helper",
                upstream="invented-upstream",
                chromium="invented-chromium",
                capability_sha256="a" * 64,
                verify_owned=lambda: guards.append("checked"),
                policy_api_key="invented-policy-key",
                text_api_key="invented-helper-key",
            )
            self.assertTrue(record["infrastructure_complete"])
            self.assertFalse(record["task_success"])
            self.assertEqual(record["text_model_reasoning"], "none")
            self.assertEqual(captured["reasoning_in_original_runner"], "none")
            self.assertEqual(os.environ["TEXT_MODEL_REASONING"], "invented-previous-low")
            self.assertEqual(record["group"], "public6")
            self.assertEqual(record["suite_denominator"], 10)
            self.assertEqual(captured["public_case_ids"], [case])
            self.assertEqual(
                (captured["seed"], captured["repeats"], captured["max_steps"]), (20261014, 1, 16)
            )
            self.assertEqual(captured["observation_profile"], self.actor.PROFILE)
            self.assertEqual(
                captured["observation_profile_helper_sha256"], self.actor.PROFILE_HELPER_SHA256
            )
            self.assertEqual(captured["text_model"], "generated-actions-nf4")
            self.assertNotEqual(captured["model"], captured["text_model"])
            self.assertNotEqual(captured["endpoint"], captured["text_endpoint"])
            self.assertEqual(captured["api_key"], "invented-policy-key")
            self.assertEqual(captured["text_api_key"], "invented-helper-key")
            self.assertIs(runner.load_upstream, original_loader)
            self.assertEqual(len(guards), 2)

    def test_no_reasoning_scope_restores_missing_environment_after_abort(self):
        previous = os.environ.pop("TEXT_MODEL_REASONING", None)
        abort = KeyboardInterrupt("invented runner abort")
        try:
            with self.assertRaises(KeyboardInterrupt) as caught:
                with self.actor.helper_reasoning_disabled():
                    self.assertEqual(os.environ["TEXT_MODEL_REASONING"], "none")
                    raise abort
            self.assertIs(caught.exception, abort)
            self.assertNotIn("TEXT_MODEL_REASONING", os.environ)
        finally:
            if previous is not None:
                os.environ["TEXT_MODEL_REASONING"] = previous


if __name__ == "__main__":
    unittest.main()
