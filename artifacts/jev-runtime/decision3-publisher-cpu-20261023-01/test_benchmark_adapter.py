"""Contract tests with synthetic requests only; no benchmark input or model reads."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cpu_benchmark_adapter", HERE / "benchmark_adapter.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class AdapterTests(unittest.TestCase):
    def test_multi_question_and_all255_options_pass_through_unchanged(self):
        calls = []
        questions = {"wide": {"type": "choice", "instructions": "synthetic", "criteria": {str(i): str(i) for i in range(255)}}, "binary": {"type": "noul"}}
        state = {"synthetic": True}
        raw = {"model": "d3-edge", "answers": {"wide": {"type": "choice", "choice": "0", "probabilities": {str(i): (1.0 if i == 0 else 0.0) for i in range(255)}}, "binary": {"type": "noul", "noul": .4}}, "usage": {"input_tokens": 7, "output_tokens": 0}}
        def prepare(**request):
            self.assertEqual(request, {"state": state, "questions": questions})
            return NS(keys=list(questions), sequences={"wide": [1, 2, 3], "binary": [1, 2, 3, 4]}, lengths={}, errors={})
        def system_one(**request):
            calls.append(request)
            return raw
        runtime = NS(prepare=prepare, system_one=system_one)
        with tempfile.TemporaryDirectory() as temp:
            adapter = b.PublisherCPUAdapter(runtime, None, Path(temp) / "native", check_runtime=lambda *_: None)
            self.assertIs(adapter(state, questions), raw)
            self.assertEqual(adapter.answers(state, questions), raw["answers"])
            self.assertEqual(calls, [{"state": state, "questions": questions}] * 2)
            self.assertEqual(adapter.calls, 2)

    def test_rejected_full_request_never_forwards_and_resource_abort_escapes(self):
        def forbidden_forward(**_):
            self.fail("rejected request must not forward")
        runtime = NS(prepare=lambda **_: NS(keys=["q"], sequences={"q": list(range(16385))}, lengths={}, errors={}), system_one=forbidden_forward)
        with tempfile.TemporaryDirectory() as temp:
            adapter = b.PublisherCPUAdapter(runtime, None, Path(temp) / "native", check_runtime=lambda *_: None)
            with self.assertRaisesRegex(ValueError, "nothing truncated"):
                adapter("synthetic", {"q": {"type": "noul"}})
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(b.CPUProfileAbort):
                b.PublisherCPUAdapter(runtime, None, Path(temp) / "native", check_runtime=lambda *_: (_ for _ in ()).throw(RuntimeError("CUDA")))
        self.assertFalse(issubclass(b.CPUProfileAbort, Exception))

    def test_both_entry_points_keep_full_populations_and_separate_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            seen = []
            adapter = NS(stage="", answers=lambda *_: None)
            def run_public(predictor, root, destination, identity):
                self.assertIs(predictor, adapter)
                destination.mkdir()
                (destination / "manifest.json").write_text(json.dumps({"n_attempted": 231}))
                seen.append((root, identity))
                return {"accuracy": .5}
            report = NS(result={"requests": 1248, "case_domains": {str(i): "test" for i in range(240)}, "planned_tests": {str(i): 24 for i in range(50)}}, overall=12, interval=[10,14], dimensions={}, errors=[], scores=[], save=lambda path: path.write_text("{}"))
            def evaluate(backend, **kwargs):
                self.assertEqual(kwargs, {"suite": "jevbench-mini", "cache": str(output / "coherence-cache"), "concurrency": 1, "progress": True})
                return report
            jb = NS(from_callable=lambda fn, name: (fn, name), evaluate=evaluate)
            result = b.run_unchanged_suites(adapter, {"model_id": "synthetic"}, output, run_public=run_public, heaven_root="original", jb=jb)
            self.assertEqual(result["legacy_public231"], {"accuracy": .5})
            self.assertEqual(result["coherence_mini"]["overall"], 12)
            self.assertEqual(seen, [("original", {"model_id": "synthetic"})])
            (output / "coherence-cache").mkdir()
            with self.assertRaises(b.CPUProfileAbort):
                b.run_unchanged_suites(adapter, {}, output, run_public=run_public, heaven_root="original", jb=jb)


if __name__ == "__main__":
    unittest.main()
