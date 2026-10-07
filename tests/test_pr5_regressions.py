"""Targeted PR #5 regressions; no model downloads or service calls."""

import hashlib
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    return runpy.run_path(str(ROOT / "scripts" / name))


def test_resume_without_checkpoint_archives_abandoned_ledger(tmp_path):
    helper = runpy.run_path(str(ROOT / "tests/test_release_training.py"))
    release = helper["release"]
    cases = [helper["_case"](i) for i in range(6)]
    recipe = helper["_recipe"]()
    reference = helper["_model"]()
    release.train_epoch(reference, cases, tmp_path / "reference", recipe, identity="pinned")
    interrupted = helper["_model"]()
    original_logits = interrupted.logits
    calls = 0

    def fail_before_first_checkpoint(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("before first checkpoint")
        return original_logits(request)

    interrupted.logits = fail_before_first_checkpoint
    output = tmp_path / "retry"
    with pytest.raises(RuntimeError, match="before first checkpoint"):
        release.train_epoch(interrupted, cases, output, recipe, identity="pinned")
    assert not (output / "latest.json").exists()
    ledger = output / "updates.jsonl"
    abandoned = ledger.read_bytes() + b'{"step":'  # A crash can also leave a partial tail.
    ledger.write_bytes(abandoned)
    restored = helper["_model"]()
    release.train_epoch(restored, cases, output, recipe, identity="pinned", resume=True)
    assert [json.loads(row)["step"] for row in ledger.read_text().splitlines()] == list(range(1, 7))
    assert [path.read_bytes() for path in output.glob("attempt-*.jsonl")] == [abandoned]
    import torch

    for actual, expected in zip(restored.lm.parameters(), reference.lm.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def model_config():
    return {
        "model_type": "gemma4_unified",
        "tie_word_embeddings": True,
        "text_config": {"final_logit_softcapping": 30.0},
        "vision_config": {"model_type": "fixture-vision"},
        "audio_config": {"model_type": "fixture-audio"},
        "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
    }


@pytest.mark.parametrize("field,value", [("bits", 4), ("group_size", 32), ("mode", "mxfp4")])
def test_mlx_declared_quantization_recipe_must_match_actual(tmp_path, field, value):
    module = script("evaluate_mlx_release.py")
    _, preparation = module["shared_scripts"]()
    config = model_config()
    config["quantization"][field] = value
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    actual = preparation["checkpoint_identity"](tmp_path)
    receipt = {
        "source_checkpoint_sha256": "c" * 64,
        "files": actual["files"],
        "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
    }
    with pytest.raises(ValueError, match="quantization"):
        module["validate_conversion"](tmp_path, receipt, {"source_checkpoint_sha256": "c" * 64})


@pytest.mark.parametrize("filename", ["model.gguf", "mmproj.gguf"])
def test_calibration_rejects_same_size_weight_substitution(tmp_path, filename):
    module = script("gguf_adapter.py")
    files = []
    for name, content in [("model.gguf", b"model"), ("mmproj.gguf", b"media")]:
        (tmp_path / name).write_bytes(content)
        files.append(
            {
                "path": name,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    manifest = tmp_path / "conversion-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "converted_and_structurally_verified",
                "model_file": "model.gguf",
                "mmproj_file": "mmproj.gguf",
                "files": files,
            }
        )
    )
    sidecar = tmp_path / "s1_config.json"
    sidecar.write_text(
        json.dumps(
            {
                "runtime": "llama.cpp",
                "runtime_format": "gguf",
                "temperature": 1.25,
                "conversion_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
        )
    )
    assert module["load_calibration"](sidecar) == 1.25
    (tmp_path / filename).write_bytes(b"other")
    with pytest.raises(ValueError, match="SHA256|digest|hash"):
        module["load_calibration"](sidecar)


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink", "manifest"])
def test_evaluation_output_cannot_clobber_inputs_even_on_preflight_failure(tmp_path, alias):
    module = script("evaluate_gguf_release.py")
    package = tmp_path / "package"
    package.mkdir()
    protected = package / "s1_config.json"
    protected.write_text('{"temperature":7.0}')
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    manifest = inputs / "manifest.json"
    manifest.write_text("{}")
    if alias == "direct":
        output = protected
    elif alias == "manifest":
        output = manifest
        protected = manifest
    else:
        output = tmp_path / "aliased-report.json"
        if alias == "symlink":
            output.symlink_to(protected)
        else:
            output.hardlink_to(protected)
    before = protected.read_bytes()
    marker = tmp_path / "launched"
    argv = [
        "--model",
        str(package),
        "--output",
        str(output),
        "--runtime-source-revision",
        "a" * 40,
    ]
    for role in ("calibration", "test", "media"):
        argv += [f"--{role}-inputs", str(inputs), f"--{role}-manifest-sha256", "0" * 64]
    for role in ("dataset", "conversion"):
        argv += [f"--{role}-manifest", str(manifest), f"--{role}-manifest-sha256", "0" * 64]
    argv += [
        "--adapter-command",
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).touch()",
    ]
    with pytest.raises(ValueError, match="output.*input|output.*package"):
        module["main"](argv)
    assert protected.read_bytes() == before
    assert not marker.exists()


def test_cmake_reconfigure_uses_new_checkout_libraries(tmp_path):
    if shutil.which("cmake") is None:
        pytest.skip("cmake not installed")
    actual = (ROOT / "tools/gguf/CMakeLists.txt").read_text()
    begin = actual.index("foreach(lib llama mtmd ggml ggml-base)")
    end = actual.index("endforeach()", begin) + len("endforeach()")
    project = tmp_path / "project"
    project.mkdir()
    (project / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\nproject(cache_regression NONE)\n" + actual[begin:end]
    )
    for checkout in ("A", "B"):
        library_dir = tmp_path / checkout / "build/bin"
        library_dir.mkdir(parents=True)
        for name in ("llama", "mtmd", "ggml", "ggml-base"):
            (library_dir / f"lib{name}.so").touch()
        subprocess.run(
            [
                "cmake",
                "-S",
                str(project),
                "-B",
                str(tmp_path / "build"),
                f"-DLLAMA_CPP_DIR={tmp_path / checkout}",
            ],
            check=True,
            capture_output=True,
        )
    cache = (tmp_path / "build/CMakeCache.txt").read_text()
    paths = [line.split("=", 1)[1] for line in cache.splitlines() if "_LIBRARY:FILEPATH=" in line]
    assert len(paths) == 4
    assert all(str(tmp_path / "B/build/bin") in path for path in paths)
