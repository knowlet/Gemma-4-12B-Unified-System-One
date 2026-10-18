"""Pure temporary source/metadata fixtures; no historical helper or ML execution."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/jev_build_public_bf16_runtime_profile.py"
SPEC = importlib.util.spec_from_file_location("prospective_x86_runtime_builder", SCRIPT)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)

VERSIONS = {
    "transformers": "5.17.0",
    "tokenizers": "0.23.3",
    "huggingface_hub": "1.33.0",
    "safetensors": "0.8.0",
    "torch": "2.13.0+cu130",
    "torchvision": "0.28.0+cu130",
    "numpy": "2.5.2",
    "pillow": "12.3.0",
    "jinja2": "3.1.6",
}


def pin(path):
    blob = path.read_bytes()
    return {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}


def write(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path


def elf(machine=62, padding=128):
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    header[18:20] = machine.to_bytes(2, "little")
    return bytes(header) + b"opaque synthetic binding" * padding


def fixture(root):
    overlay = root / "artifacts/jev-runtime/overlay"
    base_site = root / "synthetic-venv/lib/python3.13/site-packages"
    torch_root = base_site / "torch"
    torch_init = write(torch_root / "__init__.py", b"# synthetic source; never imported\n")
    files = {"torch-init": {"path": str(torch_init), **pin(torch_init)}}
    packages, metadata = {}, {}
    for name, version in {**VERSIONS, **BUILDER.EXTRA_DISTRIBUTIONS}.items():
        site = (
            overlay
            if name
            in {
                "transformers",
                "tokenizers",
                "huggingface_hub",
                "safetensors",
                *BUILDER.EXTRA_DISTRIBUTIONS,
            }
            else base_site
        )
        path = write(
            site / f"{name}-{version}.dist-info/METADATA",
            f"Name: {name}\nVersion: {version}\n\nfixture\n".encode(),
        )
        metadata[name] = path
        if name in VERSIONS:
            packages[name] = {"metadata_path": str(path), **pin(path)}
            files[name + "-metadata"] = {"path": str(path), **pin(path)}
    for index in range(41):
        path = write(overlay / f"processor_fixture_{index}.py", f"# source {index}\n".encode())
        files[f"processor-{index}"] = {"path": str(path), **pin(path)}
    closure = {
        "overlay": str(overlay),
        "interpreter": "/home/albert_hsieh/ComfyUI/.venv/bin/python",
        "python_major_minor": [3, 13],
        "no_remote_code": True,
        "files": files,
        "packages": packages,
        "versions": VERSIONS,
        "module_origins": {"torch": str(torch_init)},
    }
    reference = write(root / "reference.json", BUILDER.canonical(closure))
    for label in BUILDER.TREE_ROOTS:
        write(overlay / label / "__init__.py", b"# unimported package\n")
    for label in (
        "peft/peft_model.py",
        "peft/tuners/lora/layer.py",
        "peft/utils/save_and_load.py",
        "transformers/generation/utils.py",
        "transformers/generation/configuration_utils.py",
        "transformers/models/gemma4_unified/modeling_gemma4_unified.py",
        "transformers/cache_utils.py",
        "transformers/modeling_utils.py",
        "transformers/masking_utils.py",
        "transformers/integrations/sdpa_attention.py",
    ):
        write(overlay / label, b"# unimported runtime module\n")
    write(torch_root / "nn/functional.py", b"# unimported torch source\n")
    write(torch_root / "backends/cuda/__init__.py", b"# unimported torch source\n")
    native = write(torch_root / BUILDER.NATIVE_NAME, elf())
    cache = (
        root
        / "artifacts/jev-local-hf/hub"
        / ("models--" + BUILDER.MODEL.replace("/", "--"))
        / "snapshots"
        / BUILDER.REVISION
    )
    base_metadata = {}
    for name in BUILDER.BASE_METADATA:
        path = write(cache / name, b"synthetic fixed metadata\n")
        base_metadata[name] = pin(path)
    history = write(
        root / "historical.json",
        BUILDER.canonical(
            {
                "kind": "jev_action_bf16_qualification_v1",
                "observed_diagnostics": {"base_metadata": base_metadata},
            }
        ),
    )
    config = {
        "ROOT": root,
        "REFERENCE": reference,
        "REFERENCE_PIN": pin(reference),
        "HISTORICAL": history,
        "HISTORICAL_PIN": pin(history),
    }
    return config, native, metadata


class RuntimeBuilderTests(unittest.TestCase):
    def test_profile_uses_actual_native_bytes_exact_thirteen_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, native, _ = fixture(Path(temporary))
            # Different real file sizes must never inherit the historical ARM size.
            for padding in (1, 257):
                native.write_bytes(elf(padding=padding))
                with patch.multiple(BUILDER, **config):
                    blob, receipt_blob = BUILDER.build_profile()
                profile, receipt = json.loads(blob), json.loads(receipt_blob)
                self.assertEqual(
                    set(profile),
                    {"schema_id", "distributions", "files", "modules", "base_metadata", "device"},
                )
                self.assertEqual(set(profile["distributions"]), BUILDER.ALL_DISTRIBUTIONS)
                self.assertEqual(len(profile["distributions"]), 13)
                self.assertEqual(profile["modules"]["torch._C"], str(native))
                self.assertEqual(profile["files"][str(native)], pin(native))
                self.assertEqual(receipt["native_binding"]["size_bytes"], native.stat().st_size)
                self.assertEqual(receipt["native_binding"]["sha256"], pin(native)["sha256"])
                self.assertEqual(receipt["builder"], {"path": str(SCRIPT), **pin(SCRIPT)})
                self.assertEqual(receipt["stage"], "built_unqualified_external_sha_review_required")
                self.assertEqual(
                    receipt["runtime"],
                    {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)},
                )

    def test_native_discovery_refuses_missing_ambiguous_links_wrong_abi_or_architecture(self):
        for mode in (
            "missing",
            "ambiguous",
            "link",
            "arm_name",
            "wrong_python",
            "arm_elf",
            "empty",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                config, native, _ = fixture(Path(temporary))
                if mode == "missing":
                    native.unlink()
                elif mode == "ambiguous":
                    write(native.with_name("_C.abi3.so"), elf())
                elif mode == "link":
                    target = write(native.with_name("external-binding"), elf())
                    native.unlink()
                    native.symlink_to(target)
                elif mode == "arm_name":
                    native.rename(native.with_name("_C.cpython-313-aarch64-linux-gnu.so"))
                elif mode == "wrong_python":
                    native.rename(native.with_name("_C.cpython-312-x86_64-linux-gnu.so"))
                elif mode == "arm_elf":
                    native.write_bytes(elf(machine=183))
                else:
                    native.write_bytes(b"")
                output = Path(temporary) / "must-not-exist"
                with patch.multiple(BUILDER, **config), self.assertRaises(ValueError):
                    BUILDER.main(["--run", "--output", str(output)])
                self.assertFalse(output.exists())

    def test_each_missing_extra_package_metadata_refuses_before_output(self):
        for name in BUILDER.EXTRA_DISTRIBUTIONS:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                config, _, metadata = fixture(Path(temporary))
                metadata[name].unlink()
                output = Path(temporary) / "missing-package"
                with patch.multiple(BUILDER, **config), self.assertRaises(FileNotFoundError):
                    BUILDER.main(["--run", "--output", str(output)])
                self.assertFalse(output.exists())

    def test_package_metadata_name_version_and_symlink_are_not_substituted(self):
        for mode in ("name", "version", "link"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                config, _, metadata = fixture(Path(temporary))
                path = metadata["fastapi"]
                if mode == "link":
                    target = write(path.with_name("actual"), path.read_bytes())
                    path.unlink()
                    path.symlink_to(target)
                else:
                    path.write_bytes(
                        b"Name: other\nVersion: 0.115.12\n"
                        if mode == "name"
                        else b"Name: fastapi\nVersion: 0.0.0\n"
                    )
                with patch.multiple(BUILDER, **config), self.assertRaises(ValueError):
                    BUILDER.build_profile()

    def test_explicit_run_and_fresh_output_precede_any_package_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new"
            with patch.object(
                BUILDER, "build_profile", side_effect=AssertionError("unexpected package read")
            ):
                with self.assertRaises(ValueError):
                    BUILDER.main(["--output", str(output)])
                output.mkdir()
                with self.assertRaises(ValueError):
                    BUILDER.main(["--run", "--output", str(output)])
            self.assertEqual(list(output.iterdir()), [])

    def test_successful_publication_is_canonical_and_preserves_external_review_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, _, _ = fixture(Path(temporary))
            output = Path(temporary) / "built"
            with patch.multiple(BUILDER, **config):
                self.assertEqual(BUILDER.main(["--run", "--output", str(output)]), 0)
            self.assertEqual(
                {p.name for p in output.iterdir()}, {"runtime.json", "build-receipt.json"}
            )
            blob = (output / "runtime.json").read_bytes()
            self.assertEqual(BUILDER.canonical(json.loads(blob)), blob)
            receipt = json.loads((output / "build-receipt.json").read_bytes())
            self.assertEqual(receipt["runtime"], pin(output / "runtime.json"))
            self.assertEqual(receipt["stage"], "built_unqualified_external_sha_review_required")


if __name__ == "__main__":
    unittest.main()
