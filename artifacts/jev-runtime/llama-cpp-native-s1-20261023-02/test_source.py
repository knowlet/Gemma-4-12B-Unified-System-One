"""Stdlib-only source delta checks. Never compile or execute the native runner."""

import difflib
import hashlib
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.with_name("llama-cpp-native-s1-20261023-01")
PARENT_SHA = "8dd7f74a3ba38fda24d305c2bcbb7793d23eb38011ccb58f4f0230f677caa70e"
SOURCE_SHA = "b9205973d97145adff2ba5368d935597ab904245b701551d3148f94999cff19a"
CMAKE_SHA = "ac0ba97b62097685e449bafbb096e9835b2ba21204c46c2c7b2d640f1a2ece4b"
CHANGES = (
    (
        "    int ctx = 16384, batch = 8192, gpu_layers = 99, threads = 8, image_max_tokens = 280;\n",
        "    int ctx = 16384, batch = 8192, gpu_layers = 99, threads = 8, image_max_tokens = 280;\n"
        "    bool swa_full = true;\n",
    ),
    (
        '                         "--gpu-layers 99 --threads 8 --image-max-tokens 280]\\n";\n',
        '                         "--gpu-layers 99 --threads 8 --image-max-tokens 280 --swa-full 0|1]\\n"\n'
        '                         "  --swa-full: 1 = full-size SWA cache (default); 0 = compact SWA cache.\\n";\n',
    ),
    (
        '        else if (key == "--image-max-tokens") out.image_max_tokens = std::stoi(value);\n',
        '        else if (key == "--image-max-tokens") out.image_max_tokens = std::stoi(value);\n'
        '        else if (key == "--swa-full") {\n'
        '            require(value == "0" || value == "1", "--swa-full must be exactly 0 or 1");\n'
        '            out.swa_full = value == "1";\n'
        "        }\n",
    ),
    ("        cp.swa_full = true;\n", "        cp.swa_full = opts.swa_full;\n"),
    (
        '                    {"max_output_slots", MAX_ANSWER_SLOTS},\n',
        '                    {"max_output_slots", MAX_ANSWER_SLOTS}, {"swa_full", opts.swa_full},\n',
    ),
)


def read_source(path, expected):
    raw = path.read_bytes()
    if len(raw) > 64 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
        raise AssertionError(f"Unexpected bounded source identity: {path}")
    return raw.decode()


class SourceDeltaTests(unittest.TestCase):
    def test_entire_source_is_exact_five_change_successor_and_patch(self):
        """Protect the complete unchanged decision path, not a Python parser imitation."""
        parent = read_source(PARENT / "s1_gguf.cpp", PARENT_SHA)
        child = read_source(HERE / "s1_gguf.cpp", SOURCE_SHA)
        expected = parent
        for before, after in CHANGES:
            self.assertEqual(expected.count(before), 1)
            expected = expected.replace(before, after, 1)
        self.assertEqual(child, expected)
        restored = child
        for before, after in reversed(CHANGES):
            self.assertEqual(restored.count(after), 1)
            restored = restored.replace(after, before, 1)
        self.assertEqual(restored, parent)
        relative = "artifacts/jev-runtime/"
        patch = "".join(
            difflib.unified_diff(
                parent.splitlines(keepends=True),
                child.splitlines(keepends=True),
                fromfile=relative + PARENT.name + "/s1_gguf.cpp",
                tofile=relative + HERE.name + "/s1_gguf.cpp",
            )
        )
        self.assertEqual((HERE / "source.patch").read_text(), patch)

    def test_cmake_is_byte_identical_to_fixed_parent(self):
        self.assertEqual(
            read_source(HERE / "CMakeLists.txt", CMAKE_SHA),
            read_source(PARENT / "CMakeLists.txt", CMAKE_SHA),
        )


if __name__ == "__main__":
    unittest.main()
