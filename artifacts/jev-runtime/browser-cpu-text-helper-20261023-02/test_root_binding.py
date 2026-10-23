"""Stdlib source-gate fakes only; no runtime payload, native import or network."""

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "cpu_text_helper_successor02", Path(__file__).with_name("helper.py")
)
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)


class FakeLoadedRuntime:
    pass


def completed_metadata():
    # Placeholder metadata is never opened or passed to a real runtime loader.
    selection = json.loads((h.BASE_DIRECTORY / "runtime-selection.pending.json").read_text())
    for name, entry in selection["model_files"].items():
        entry["path"] = "/nonexistent/selected/" + name
        if entry["sha256"] is None:
            entry["sha256"] = "1" * 64
    selection["runtime_binary_files"] = [
        {"path": "/nonexistent/torch/_C.so", "size_bytes": 1, "sha256": "1" * 64},
        {"path": "/nonexistent/tokenizers/tokenizers.so", "size_bytes": 1, "sha256": "1" * 64},
    ]
    selection["owner"]["profile"] = "independent-source-gate-fake"
    selection["owner"]["ticket"] = {
        "path": "/nonexistent/owner-ticket.json",
        "size_bytes": 1,
        "sha256": "1" * 64,
    }
    selection["owner"]["verifier"] = {
        "path": "/nonexistent/owner-verifier.py",
        "size_bytes": 1,
        "sha256": "1" * 64,
    }
    return selection


def mixed_root(selection):
    selection = copy.deepcopy(selection)
    root = Path(selection["module_origins"]["transformers"]).resolve().parent
    critical = {str(root / relative) for relative in h.CRITICAL_RELATIVE_PATHS}
    for entry in selection["runtime_files"]:
        if entry["path"] in critical:
            entry["path"] = str(
                Path("/another/tree/transformers") / Path(entry["path"]).relative_to(root)
            )
    return selection


class RootBindingTests(unittest.TestCase):
    def test_same_selected_tree_complete_critical_pins_pass_metadata(self):
        selection = completed_metadata()
        self.assertIs(h.validate_selection(selection), selection)

    def test_other_tree_same_suffix_does_not_satisfy_selected_root(self):
        selection = mixed_root(completed_metadata())
        self.assertIs(
            h._BASE.validate_selection(selection), selection
        )  # Distinguishes the old gap.
        with self.assertRaisesRegex(h.Refusal, "selected package root"):
            h.validate_selection(selection)

    def test_one_wrong_tree_pin_cannot_be_hidden_by_other_correct_pins(self):
        selection = completed_metadata()
        suffix = "transformers/generation/utils.py"
        entry = next(item for item in selection["runtime_files"] if item["path"].endswith(suffix))
        entry["path"] = "/another/tree/" + suffix
        with self.assertRaisesRegex(h.Refusal, "generation/utils.py"):
            h.validate_selection(selection)

    def test_ancestor_symlink_cannot_escape_selected_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            selected = Path(temp) / "selected/transformers"
            other = Path(temp) / "other/transformers"
            selected.mkdir(parents=True)
            other.mkdir(parents=True)
            (selected / "generation").symlink_to(other / "generation", target_is_directory=True)
            selection = completed_metadata()
            original = Path(selection["module_origins"]["transformers"]).parent
            selection["module_origins"]["transformers"] = str(selected / "__init__.py")
            for entry in selection["runtime_files"]:
                path = Path(entry["path"])
                if path.is_relative_to(original):
                    entry["path"] = str(selected / path.relative_to(original))
            with self.assertRaisesRegex(h.Refusal, "selected package root"):
                h.validate_selection(selection)

    def test_bad_selection_is_refused_before_delegate_or_heavy_import(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "selection.json"
            blob = json.dumps(mixed_root(completed_metadata())).encode()
            path.write_bytes(blob)
            with patch.object(
                h._BASE, "load_runtime", side_effect=AssertionError("must not delegate")
            ):
                with self.assertRaises(h.Refusal):
                    h.load_runtime(path, h._BASE.digest(blob))
        self.assertFalse(set(sys.modules) & {"torch", "transformers", "tokenizers"})

    def test_pending_payload_dso_owner_and_watchdog_still_refuse(self):
        pending = json.loads((h.BASE_DIRECTORY / "runtime-selection.pending.json").read_text())
        with self.assertRaises(h.Refusal):
            h.validate_selection(pending)
        for alter in (
            lambda selection: selection["model_files"]["tokenizer.json"].update(sha256=None),
            lambda selection: selection["runtime_binary_files"][0].update(sha256=None),
            lambda selection: selection["owner"]["verifier"].update(sha256=None),
            lambda selection: selection["owner"]["ticket"].update(path=None),
            lambda selection: selection["owner"].update(watchdog_external=False),
        ):
            selection = completed_metadata()
            alter(selection)
            with self.assertRaises(h.Refusal):
                h.validate_selection(selection)

    def test_only_successor_loader_registers_public_factory_runtime(self):
        runtime = FakeLoadedRuntime()
        with patch.object(
            h._BASE, "make_application", side_effect=AssertionError("must not delegate")
        ):
            with self.assertRaises(h.Refusal):
                h.make_application(runtime, None, "unused")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "selection.json"
            blob = json.dumps(completed_metadata()).encode()
            path.write_bytes(blob)
            with patch.object(h._BASE, "load_runtime", return_value=runtime) as loader:
                self.assertIs(h.load_runtime(path, h._BASE.digest(blob)), runtime)
                loader.assert_called_once_with(str(path), h._BASE.digest(blob))
            with patch.object(h._BASE, "make_application", return_value="fake-app"):
                self.assertEqual(h.make_application(runtime, None, "unused"), "fake-app")

    def test_reused_native_and_request_source_are_exactly_hash_bound(self):
        self.assertEqual(
            h._BASE.digest((h.BASE_DIRECTORY / "helper.py").read_bytes()), h.BASE_SOURCE_SHA256
        )
        self.assertEqual(h.TEXT_VALUE, h._BASE.TEXT_VALUE)
        self.assertEqual((h._BASE.TOKENS, h._BASE.CONTEXT, h._BASE.SECONDS), (128, 8192, 20.0))


if __name__ == "__main__":
    unittest.main()
