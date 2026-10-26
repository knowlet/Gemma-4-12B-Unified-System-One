import ast
import contextlib
import hashlib
import importlib.util
import inspect
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name, path, sha=None):
    blob = path.read_bytes()
    if sha is not None:
        assert hashlib.sha256(blob).hexdigest() == sha
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(blob, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


B = load("tested_feedback_bundle", HERE / "feedback_bundle.py")
A = load("tested_feedback_assembly", HERE / "assemble_sources.py")
proof = load(
    "tested_original_stale_proof",
    HERE.parent / "browser-stale-feedback-20261022-01/proof.py",
    "0146d0300ae2dbbb7a7b369713bebc4e4ac223ca40bb69652689441aeb116465",
)


@contextlib.contextmanager
def runner_module(bundle):
    previous = sys.modules.pop("action_feedback", None)
    sys.path.insert(0, str(bundle))
    try:
        yield load("tested_generated_feedback_runner", bundle / "runner.py")
    finally:
        sys.path.remove(str(bundle))
        sys.modules.pop("action_feedback", None)
        if previous is not None:
            sys.modules["action_feedback"] = previous


def exercise_runner(runner):
    source = inspect.getsource(proof.isolated_fixture)
    anchor = '        agent = agent_module.Agent("https://fixture.invalid/", "Exercise the fixture control")\n'
    assert source.count(anchor) == 1
    source = source.replace(
        anchor,
        '        guard = runner.model_adapter(agent_module, sys.modules[prefix + ".model"], endpoint="http://127.0.0.1:9/v1/systemone", model="fixture-constant-choice", api_key="fixture-no-credential")\n        adapter_result = guard.__enter__()\n'
        + anchor,
        1,
    )
    source = source.replace(
        "            agent.close()\n",
        "            agent.close()\n            guard.__exit__(None, None, None)\n",
        1,
    )
    source = source.replace(
        '                "requests": bodies,\n',
        '                "adapter_records": adapter_result[0],\n                "requests": bodies,\n',
        1,
    )
    namespace = {**proof.__dict__, "runner": runner}
    exec(
        compile(source, "<original verified fake-I/O fixture, generated adapter>", "exec"),
        namespace,
    )
    return namespace["isolated_fixture"](ticks=3)


class FeedbackBundleTests(unittest.TestCase):
    def test_both_groups_preserve_complete_original_v7_bytes_and_AST(self):
        for group in ["original", "heldout"]:
            with self.subTest(group=group), tempfile.TemporaryDirectory(dir=HERE) as directory:
                bundle, manifest = B.prepare_bundle(Path(directory) / "bundle", case_module=group)
                B.verify_bundle(bundle, manifest)
                self.assertEqual(manifest["agent_protocol"], "typed-stale-action-feedback-v1")
                self.assertEqual(manifest["profile"], "composed-visible-text-settled-v7")
                self.assertTrue(manifest["source_delta"]["inverse_complete_bytes_equal"])
                self.assertTrue(manifest["source_delta"]["inverse_complete_AST_equal"])
                inverse = (bundle / "runner.py").read_bytes().replace(B.INSTALL, b"", 1)
                original, _ = B.expected_bundle(group)
                self.assertEqual(
                    ast.dump(ast.parse(inverse)),
                    ast.dump(ast.parse(original["runner.py"].replace(B.INSTALL, b"", 1))),
                )
                self.assertEqual(
                    (bundle / "action_feedback.py").read_bytes(),
                    B.checked(B.UTILITY, B.UTILITY_SHA),
                )

    def test_actual_generated_adapter_sends_and_records_identical_feedback(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            bundle, _ = B.prepare_bundle(Path(directory) / "bundle")
            with runner_module(bundle) as runner:
                result = exercise_runner(runner)
            self.assertEqual(result["executions"], 0)
            self.assertEqual(len({proof.canonical(v) for v in result["requests"]}), 3)
            self.assertEqual(len(result["adapter_records"]), 3)
            for sent, recorded in zip(result["requests"], result["adapter_records"], strict=True):
                self.assertEqual(sent, recorded["request"])
                self.assertEqual(sent, recorded["transport_attempts"][0]["request"])
            self.assertNotIn("action_feedback", result["requests"][0]["state"])
            self.assertIn("action_feedback", result["requests"][1]["state"])

    def test_added_feedback_reaches_complete_original_V2_model_input(self):
        from s1.jev_generated import format_action_request
        from s1.jev_prompt import format_payload, unpack_payload

        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            bundle, _ = B.prepare_bundle(Path(directory) / "bundle")
            with runner_module(bundle) as runner:
                body = exercise_runner(runner)["requests"][1]
            text, source, _labels, evidence = format_action_request(
                body, "fixture-constant-choice", prompt_format="shared-elements-v2"
            )
            self.assertIn(b'"action_feedback"', source)
            self.assertIn('"action_feedback"', text)
            self.assertTrue(evidence["ordered_roundtrip_verified"])
            packed, _ = format_payload(body, prompt_format="shared-elements-v2")
            self.assertEqual(unpack_payload(packed), body)

    def test_tamper_missing_extra_and_symlink_refuse(self):
        for kind in ["tamper", "missing", "extra", "symlink"]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir=HERE) as directory:
                bundle, manifest = B.prepare_bundle(Path(directory) / "bundle")
                if kind == "tamper":
                    with (bundle / "runner.py").open("ab") as stream:
                        stream.write(b"\n# unselected change\n")
                elif kind == "missing":
                    (bundle / "action_feedback.py").unlink()
                elif kind == "extra":
                    (bundle / "unexpected.txt").write_text("unselected")
                else:
                    (bundle / "action_feedback.py").unlink()
                    (bundle / "action_feedback.py").symlink_to(B.UTILITY)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    B.verify_bundle(bundle, manifest)

    def test_source_successors_have_exact_inverse_and_selected_body(self):
        for name, (blob, _patch, row) in A.projections().items():
            self.assertEqual((HERE / name).read_bytes(), blob)
            self.assertTrue(row["inverse_complete_bytes_equal"])
            self.assertTrue(row["inverse_complete_AST_equal"])
        native = load("tested_feedback_native_driver", HERE / "feedback_native_driver.py")
        owner = load("tested_feedback_native_owner", HERE / "launch_browser.py")
        self.assertEqual(native.ROOT, ROOT)
        self.assertEqual(owner.ROOT, ROOT)
        self.assertEqual(owner.FORMATS, {"v2-feedback": "shared-elements-v2"})
        self.assertEqual(owner.CAP, 40 * 1024**3)
        self.assertEqual(native.V7, "composed-visible-text-settled-v7")
        self.assertIn("feedback_bundle.py", " ".join(native.composition_paths()))


if __name__ == "__main__":
    unittest.main()
