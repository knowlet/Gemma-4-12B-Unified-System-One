"""Stdlib-only complete source inverse checks; never run a native executable."""

import difflib
import hashlib
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = HERE.with_name("llama-cpp-native-s1-20261023-02")
V1 = HERE.with_name("llama-cpp-native-s1-20261023-01")
SOURCE_SHA = "d9ff808d54cee375a09120b49e0c0e0840198d916a0185ecbd7d9e389493f36b"
DELTA_SHA = "8667ffa72040d2715d3590fc4e9259f5ec879401540ff96747f3fa743e76080a"
PARENT_SHA = "b9205973d97145adff2ba5368d935597ab904245b701551d3148f94999cff19a"
PARENT_DELTA_SHA = "264f0ec86d8c992372e677bfaa1ef7991ff3ace151fe67e3d6b120ecad8ba1e2"
V1_SHA = "8dd7f74a3ba38fda24d305c2bcbb7793d23eb38011ccb58f4f0230f677caa70e"
CMAKE_SHA = "ac0ba97b62097685e449bafbb096e9835b2ba21204c46c2c7b2d640f1a2ece4b"


def read_source(path, expected):
    if path.stat().st_size > 64 * 1024:
        raise AssertionError(f"Source exceeded small-file limit: {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise AssertionError(f"Source identity mismatch: {path}")
    return raw.decode()


class SourceDeltaTests(unittest.TestCase):
    def invert(self, source, changes):
        for row in reversed(changes):
            self.assertEqual(source.count(row["after"]), 1)
            source = source.replace(row["after"], row["before"], 1)
        return source

    def test_exact_v3_inverse_and_saved_patch_preserve_v2_and_v1(self):
        """Protect every byte outside the selected loader/context options."""
        source = read_source(HERE / "s1_gguf.cpp", SOURCE_SHA)
        parent = read_source(PARENT / "s1_gguf.cpp", PARENT_SHA)
        delta = json.loads(read_source(HERE / "source-delta.json", DELTA_SHA))
        self.assertEqual(len(delta["changes"]), 7)
        self.assertEqual(self.invert(source, delta["changes"]), parent)
        parent_delta = json.loads(read_source(PARENT / "source-delta.json", PARENT_DELTA_SHA))
        self.assertEqual(len(parent_delta["changes"]), 5)
        self.assertEqual(
            self.invert(parent, parent_delta["changes"]),
            read_source(V1 / "s1_gguf.cpp", V1_SHA),
        )
        expected_patch = "".join(
            difflib.unified_diff(
                parent.splitlines(keepends=True),
                source.splitlines(keepends=True),
                fromfile=str((PARENT / "s1_gguf.cpp").relative_to(ROOT)),
                tofile=str((HERE / "s1_gguf.cpp").relative_to(ROOT)),
            )
        )
        self.assertEqual((HERE / "source.patch").read_text(), expected_patch)

    def test_cmake_is_exact_v2_and_v1_copy(self):
        source = read_source(HERE / "CMakeLists.txt", CMAKE_SHA)
        self.assertEqual(source, read_source(PARENT / "CMakeLists.txt", CMAKE_SHA))
        self.assertEqual(source, read_source(V1 / "CMakeLists.txt", CMAKE_SHA))


if __name__ == "__main__":
    unittest.main()
