"""Audit dependency bytes with a tiny real adapter environment; no model inference."""

import runpy
import subprocess
import sys
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    return runpy.run_path(str(ROOT / "scripts" / name))


@pytest.fixture
def tiny_runtime(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    env = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(env)
    python = env / "bin/python"
    site = next((env / "lib").glob("python*/site-packages"))
    package = site / "proofdist"
    package.mkdir()
    payload = package / "__init__.py"
    payload.write_text("VALUE = 1\n")
    info = site / "proofdist-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: proofdist\nVersion: 1.0\n")
    (info / "RECORD").write_text(
        "proofdist/__init__.py,,\nproofdist-1.0.dist-info/METADATA,,\nproofdist-1.0.dist-info/RECORD,,\n"
    )
    (root / "scripts").mkdir()
    (root / "src/s1").mkdir(parents=True)
    for name in (
        "evaluate_mlx_release.py",
        "summarize_release.py",
        "prepare_mlx_validation.py",
        "verify_mlx_export.py",
    ):
        (root / "scripts" / name).write_text("# fixture\n")
    (root / "uv.lock").write_text("version = 1\n")
    adapter = root / "scripts/adapter.py"
    adapter.write_text(
        "raise RuntimeError('adapter must not launch in provenance preflight tests')\n"
    )

    def git(*args):
        return subprocess.check_output(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "commit.gpgsign=false",
                "-C",
                str(root),
                *args,
            ],
            stderr=subprocess.DEVNULL,
        )

    git("init", "-q")
    git("add", ".")
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return root, python, payload, adapter, git("rev-parse", "HEAD").decode().strip()


def test_dependencies_same_version_changed_bytes_change_identity(tiny_runtime):
    root, python, payload, adapter, head = tiny_runtime
    guard_class = script("evaluate_gguf_release.py")["RuntimeGuard"]
    command = [str(python), str(adapter)]
    first = guard_class(head, command, root=root, python=str(python))
    first.check()
    assert first.receipt()["python_environment"]["prefix"] == str(python.parent.parent)
    payload.write_text("VALUE = 2\n")  # Same length, version, and untouched RECORD.
    with pytest.raises(ValueError, match="Python environment changed"):
        first.check()
    second = guard_class(head, command, root=root, python=str(python))
    assert first.receipt()["source_revision"] == second.receipt()["source_revision"]
    assert first.receipt()["implementation_sha256"] != second.receipt()["implementation_sha256"]
    assert first.receipt()["files_sha256"] != second.receipt()["files_sha256"]


def test_added_unrecorded_dependency_module_is_detected(tiny_runtime):
    root, python, payload, adapter, head = tiny_runtime
    guard = script("evaluate_gguf_release.py")["RuntimeGuard"](
        head, [str(python), str(adapter)], root=root, python=str(python)
    )
    payload.with_name("added.py").write_text("VALUE = 1\n")
    with pytest.raises(ValueError, match="Python environment changed"):
        guard.check()


def test_changed_lockfile_is_detected(tiny_runtime):
    root, python, _, adapter, head = tiny_runtime
    guard = script("evaluate_gguf_release.py")["RuntimeGuard"](
        head, [str(python), str(adapter)], root=root, python=str(python)
    )
    (root / "uv.lock").write_text("version = 2\n")
    with pytest.raises(ValueError, match="runtime file changed"):
        guard.check()


def test_cannot_attribute_adapter_to_another_python(tiny_runtime):
    root, python, _, adapter, head = tiny_runtime
    with pytest.raises(ValueError, match="directly invoked"):
        script("evaluate_gguf_release.py")["RuntimeGuard"](
            head, [str(python), str(adapter)], root=root, python=sys.executable
        )


def test_missing_distribution_file_fails_closed(tiny_runtime):
    root, python, payload, _, _ = tiny_runtime
    payload.unlink()
    with pytest.raises(ValueError, match="missing"):
        script("evaluate_gguf_release.py")["capture_python_environment"](str(python), root)
