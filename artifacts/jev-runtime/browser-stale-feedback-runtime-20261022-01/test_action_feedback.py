"""Exercise verified original Agent/choose sources with the existing fake I/O fixture."""

import copy
import hashlib
import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).parent


def load(name, path, expected=None):
    data = path.read_bytes()
    if expected is not None:
        assert hashlib.sha256(data).hexdigest() == expected
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(data, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


feedback = load("action_feedback", HERE / "action_feedback.py")
proof = load(
    "stale_proof",
    HERE.parent / "browser-stale-feedback-20261022-01/proof.py",
    "0146d0300ae2dbbb7a7b369713bebc4e4ac223ca40bb69652689441aeb116465",
)
benchmark = load(
    "selected_original_benchmark",
    HERE.parents[2] / "scripts/jev_browser_benchmark.py",
    "fe1410675e4decb06a8491a5a83d2ec965629c99f8912b797446c4cb2c155f6c",
)


def make_fixture(*, predict_stale=False, real_adapter=False):
    source = inspect.getsource(proof.isolated_fixture)
    before = '        agent = agent_module.Agent("https://fixture.invalid/", "Exercise the fixture control")\n'
    assert source.count(before) == 1
    guard = 'feedback.install(agent_module, sys.modules[prefix + ".model"])'
    if real_adapter:
        guard = (
            "feedback.wrap_model_adapter(benchmark.model_adapter)("
            'agent_module, sys.modules[prefix + ".model"], '
            'endpoint="http://127.0.0.1:9/v1/systemone", model="fixture-constant-choice", '
            'api_key="fixture-no-credential")'
        )
    source = source.replace(
        before,
        "        guard = " + guard + "\n        adapter_result = guard.__enter__()\n" + before,
    )
    source = source.replace(
        "            agent.close()\n",
        "            agent.close()\n            guard.__exit__(None, None, None)\n",
    )
    if predict_stale:
        source = source.replace(
            "            def fresh(self, _page, _action=None):\n                return True\n",
            "            def fresh(self, _page, _action=None):\n"
            "                if not getattr(self, 'staled_in_predict', False):\n"
            "                    self.staled_in_predict = True\n"
            "                    raise browser_definition.StalePage('fixture predict stale')\n"
            "                return True\n",
        )
    if real_adapter:
        source = source.replace(
            '                "requests": bodies,\n',
            '                "adapter_records": adapter_result[0],\n'
            '                "requests": bodies,\n',
        )
    namespace = {**proof.__dict__, "feedback": feedback, "benchmark": benchmark}
    exec(compile(source, "<selected fake-I/O fixture with feedback>", "exec"), namespace)
    return namespace["isolated_fixture"]


fixture = make_fixture()


def without_feedback(body):
    value = copy.deepcopy(body)
    value["state"].pop("action_feedback", None)
    return value


class FeedbackTests(unittest.TestCase):
    def test_real_benchmark_adapter_records_and_sends_identical_feedback(self):
        result = make_fixture(real_adapter=True)(ticks=3)
        self.assertEqual(len(result["adapter_records"]), 3)
        self.assertEqual(len({proof.canonical(v) for v in result["requests"]}), 3)
        for sent, recorded in zip(result["requests"], result["adapter_records"], strict=True):
            self.assertEqual(sent, recorded["request"])
            self.assertEqual(sent, recorded["transport_attempts"][0]["request"])
            self.assertEqual(recorded["endpoint"], "http://127.0.0.1:9/v1/systemone")
        self.assertIn("action_feedback", result["adapter_records"][1]["request"]["state"])

    def test_pre_input_refusal_changes_next_request_without_changing_candidates(self):
        original = proof.isolated_fixture(ticks=3)
        revised = fixture(ticks=3)
        self.assertEqual(original["requests"][0], revised["requests"][0])
        self.assertEqual(revised["executions"], 0)
        self.assertEqual(len({proof.canonical(v) for v in revised["requests"]}), 3)
        for old, new in zip(original["requests"], revised["requests"], strict=True):
            self.assertEqual(old, without_feedback(new))
        row = revised["requests"][1]["state"]["action_feedback"][0]
        self.assertEqual(row["outcome"], "StalePage")
        self.assertEqual(row["operation"], "CLICK")
        self.assertEqual(row["recorded_action_delta"], 0)
        self.assertEqual(row["physical_input_outcome"], "unknown")
        self.assertEqual(revised["snapshots"][-1]["history"], [])

    def test_post_input_stale_keeps_actual_execution_history(self):
        original = proof.isolated_fixture("post_input_stale", ticks=2)
        revised = fixture("post_input_stale", ticks=2)
        self.assertEqual(revised["executions"], 2)
        self.assertEqual(len(revised["snapshots"][-1]["history"]), 2)
        self.assertEqual(original["requests"][1], without_feedback(revised["requests"][1]))
        row = revised["requests"][1]["state"]["action_feedback"][0]
        self.assertEqual(row["recorded_action_delta"], 1)
        self.assertEqual(row["physical_input_outcome"], "unknown")

    def test_predict_stale_does_not_invent_selected_action(self):
        result = make_fixture(predict_stale=True)(ticks=2)
        row = result["requests"][0]["state"]["action_feedback"][0]
        self.assertEqual(row["phase"], "predict")
        self.assertIsNone(row["operation"])
        self.assertIsNone(row["target"])
        self.assertIsNone(row["choice"])
        self.assertEqual(row["recorded_action_delta"], 0)

    def test_feedback_is_bounded_and_agents_are_independent(self):
        result = fixture(ticks=7)
        recent = result["snapshots"][-1]["action_feedback"]
        self.assertEqual(len(recent), feedback.MAX_RECENT)
        self.assertEqual([v["attempt"] for v in recent], [4, 5, 6, 7])
        second = fixture(ticks=1)
        self.assertNotIn("action_feedback", second["requests"][0]["state"])
        self.assertEqual(second["snapshots"][0]["action_feedback_count"], 1)

    def test_original_runtime_is_restored_after_wrapper(self):
        fixture(ticks=2)
        result = proof.isolated_fixture(ticks=2)
        self.assertEqual(len({proof.canonical(v) for v in result["requests"]}), 1)
        self.assertNotIn("action_feedback", result["snapshots"][-1])


if __name__ == "__main__":
    unittest.main()
