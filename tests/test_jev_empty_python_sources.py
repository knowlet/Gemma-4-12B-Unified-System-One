"""Invented empty-source controls for the unchanged stdlib SourcePins reader.

Only temporary files are read. No source fixture is imported, and the public
builder's historical preparation entrypoint is never called.
"""

import hashlib
import importlib.util
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/jev_build_public_bf16_runtime_profile.py"
SPEC = importlib.util.spec_from_file_location("empty_python_source_builder_controls", SCRIPT)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
EMPTY_PIN = {"sha256": hashlib.sha256(b"").hexdigest(), "size_bytes": 0}


class EmptyPythonSourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="invented-empty-python-")
        self.addCleanup(temporary.cleanup)
        # Canonicalize only the allocated root; child symlinks remain observable.
        self.root = Path(temporary.name).resolve(strict=True)
        self.source = self.root / "__init__.py"
        self.source.write_bytes(b"")

    def test_empty_init_is_exactly_pinned_and_repeat_checks_retain_one_entry(self):
        pins = BUILDER.SourcePins()
        self.assertEqual(pins.pin(self.source, dict(EMPTY_PIN)), EMPTY_PIN)
        self.assertEqual(pins.pin(self.source, dict(EMPTY_PIN)), EMPTY_PIN)
        self.assertEqual(pins.values, {str(self.source): EMPTY_PIN})
        self.assertEqual(pins.bytes, 0)
        positive = self.root / "invented.py"
        raw = b"# invented source; never imported\n"
        positive.write_bytes(raw)
        expected = {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
        self.assertEqual(pins.pin(positive, expected), expected)
        self.assertEqual((len(pins.values), pins.bytes), (2, len(raw)))

    def test_empty_source_with_wrong_expected_hash_is_not_admitted(self):
        pins = BUILDER.SourcePins()
        with self.assertRaisesRegex(ValueError, "Known source/metadata authority differs"):
            pins.pin(self.source, {"sha256": "0" * 64, "size_bytes": 0})
        self.assertEqual(pins.values, {})
        self.assertEqual(pins.bytes, 0)

    def test_nonempty_source_cannot_match_an_empty_pin(self):
        self.source.write_bytes(b"# invented nonempty source\n")
        pins = BUILDER.SourcePins()
        with self.assertRaisesRegex(ValueError, "Known source/metadata authority differs"):
            pins.pin(self.source, dict(EMPTY_PIN))
        self.assertEqual(pins.values, {})

    def test_leaf_and_parent_symlinks_remain_refused(self):
        leaf = self.root / "alias.py"
        leaf.symlink_to(self.source)
        actual = self.root / "actual-package"
        actual.mkdir()
        (actual / "__init__.py").write_bytes(b"")
        parent = self.root / "alias-package"
        parent.symlink_to(actual, target_is_directory=True)
        for source in (leaf, parent / "__init__.py"):
            with self.subTest(source=source):
                pins = BUILDER.SourcePins()
                with self.assertRaisesRegex(ValueError, "Ordinary path required"):
                    pins.pin(source, dict(EMPTY_PIN))
                self.assertEqual(pins.values, {})

    def test_same_empty_bytes_at_replaced_path_fail_the_identity_postcheck(self):
        original_fdopen = BUILDER.os.fdopen
        replacement = self.root / "replacement.py"

        @contextmanager
        def replace_after_read(descriptor, mode):
            with original_fdopen(descriptor, mode) as stream:

                class ReplacingReader:
                    def fileno(self):
                        return stream.fileno()

                    def read(self, size):
                        raw = stream.read(size)
                        replacement.write_bytes(b"")
                        replacement.replace(self_source)
                        return raw

                yield ReplacingReader()

        self_source = self.source
        pins = BUILDER.SourcePins()
        with mock.patch.object(BUILDER.os, "fdopen", replace_after_read):
            with self.assertRaisesRegex(ValueError, "Source changed during read"):
                pins.pin(self.source, dict(EMPTY_PIN))
        self.assertEqual(self.source.read_bytes(), b"")
        self.assertEqual(pins.values, {})

    def test_expected_pin_rejects_bool_size_negative_size_and_extra_fields(self):
        invalid = (
            {**EMPTY_PIN, "size_bytes": False},
            {**EMPTY_PIN, "size_bytes": -1},
            {**EMPTY_PIN, "extra": True},
        )
        for expected in invalid:
            with self.subTest(expected=expected):
                pins = BUILDER.SourcePins()
                with self.assertRaisesRegex(ValueError, "Strict bounded pin"):
                    pins.pin(self.source, expected)
                self.assertEqual(pins.values, {})

    def test_empty_tensor_named_payloads_still_do_not_enter_source_pins(self):
        for suffix in (".safetensors", ".bin", ".pt", ".pth", ".ckpt"):
            with self.subTest(suffix=suffix):
                payload = self.root / ("invented-model" + suffix)
                payload.write_bytes(b"")
                pins = BUILDER.SourcePins()
                with self.assertRaisesRegex(ValueError, "Tensor/base payload access is forbidden"):
                    pins.pin(payload, dict(EMPTY_PIN))
                self.assertEqual(pins.values, {})


if __name__ == "__main__":
    unittest.main()
