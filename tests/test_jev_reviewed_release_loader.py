"""Synthetic sources only: reviewed release loading without cached bytecode."""

import hashlib
import importlib.util
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import jev_decision3_candidate_http_owner as owner
import pytest


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def sources(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, "__file__", str(tmp_path / "owner.py"))
    held, release = tmp_path / "held.py", tmp_path / "release.py"

    def create(body):
        source = b"SOURCE_ONLY = True\n" + body
        actual = source.replace(b"SOURCE_ONLY = True", b"SOURCE_ONLY = False", 1)
        held.write_bytes(source)
        release.write_bytes(actual)
        return {str(held): sha(source), str(release): sha(actual)}

    yield held, release, create
    sys.modules.pop("_synthetic_verified_release", None)


def load(release, pins):
    return owner._load_reviewed_release(
        str(release), "held.py", pins, "_synthetic_verified_release"
    )


def test_verified_module_preserves_import_metadata(sources):
    _, release, create = sources
    pins = create(b"VALUE = 42\n")
    module = load(release, pins)
    assert module.VALUE == 42 and module.SOURCE_ONLY is False
    assert module.__file__ == str(release)
    assert module.__spec__ is not None and module.__spec__.origin == str(release)
    assert sys.modules["_synthetic_verified_release"] is module


def test_wrong_pin_gate_and_existing_module_are_refused(sources):
    held, release, create = sources
    pins = create(b"VALUE = 42\n")
    with pytest.raises(ValueError, match="bytes changed"):
        load(release, {**pins, str(held): "0" * 64})
    release.write_bytes(release.read_bytes() + b"# extra\n")
    pins[str(release)] = sha(release.read_bytes())
    with pytest.raises(ValueError, match="only by the explicit source gate"):
        load(release, pins)
    pins = create(b"VALUE = 42\n")
    sys.modules["_synthetic_verified_release"] = object()
    with pytest.raises(ValueError, match="fresh release module name"):
        load(release, pins)
    assert sys.modules["_synthetic_verified_release"] is not None


def test_stale_pyc_is_not_executed(sources):
    _, release, create = sources
    pins = create(b"VALUE = 'safe'\n")
    original = release.read_bytes()
    poisoned = original.replace(b"'safe'", b"'evil'")
    assert len(poisoned) == len(original)
    release.write_bytes(poisoned)
    modified = release.stat()
    py_compile.compile(str(release), doraise=True)
    release.write_bytes(original)
    os.utime(release, ns=(modified.st_atime_ns, modified.st_mtime_ns))
    assert load(release, pins).VALUE == "safe"


def test_changed_file_during_load_executes_only_verified_bytes_then_refuses(sources, monkeypatch):
    _, release, create = sources
    pins = create(b"VALUE = 'safe'\n")
    make_module = importlib.util.module_from_spec
    observed = []
    original_compile = compile

    def capture_compile(source, filename, mode, *args, **kwargs):
        if filename == str(release):
            observed.append(source)
        return original_compile(source, filename, mode, *args, **kwargs)

    def change_after_capture(spec):
        module = make_module(spec)
        release.write_bytes(release.read_bytes().replace(b"'safe'", b"'evil'"))
        return module

    monkeypatch.setattr(importlib.util, "module_from_spec", change_after_capture)
    monkeypatch.setattr("builtins.compile", capture_compile)
    with pytest.raises(ValueError, match="bytes changed"):
        load(release, pins)
    assert len(observed) == 1 and b"'safe'" in observed[0] and b"'evil'" not in observed[0]
    assert "_synthetic_verified_release" not in sys.modules


@pytest.mark.parametrize(
    "body, error",
    [
        (b"raise RuntimeError('synthetic')\n", RuntimeError),
        (b"SOURCE_ONLY=True\n", ValueError),
        (b"import sys\ndel sys.modules[__name__]\n", ValueError),
    ],
)
def test_failed_import_or_identity_check_cleans_module(sources, body, error):
    _, release, create = sources
    pins = create(body)
    with pytest.raises(error):
        load(release, pins)
    assert "_synthetic_verified_release" not in sys.modules


def test_oversized_or_symlinked_source_is_refused(sources, tmp_path):
    held, release, create = sources
    pins = create(b"VALUE = 42\n")
    data = held.read_bytes()
    held.unlink()
    target = tmp_path / "target.py"
    target.write_bytes(data)
    held.symlink_to(target)
    with pytest.raises(ValueError, match="ordinary pinned source"):
        load(release, pins)
    held.unlink()
    with held.open("wb") as stream:
        stream.truncate(8 * 1024**2 + 1)
    with pytest.raises(ValueError, match="bounded ordinary pinned source"):
        load(release, pins)


@pytest.mark.parametrize("throws", [False, True])
def test_replaced_module_is_preserved_and_load_refused(sources, monkeypatch, throws):
    _, release, create = sources
    foreign = object()
    monkeypatch.setattr("builtins._synthetic_foreign_module", foreign, raising=False)
    body = b"import sys, builtins\nsys.modules[__name__] = builtins._synthetic_foreign_module\n"
    if throws:
        body += b"raise RuntimeError('synthetic failure')\n"
    pins = create(body)
    error = RuntimeError if throws else ValueError
    with pytest.raises(error):
        load(release, pins)
    assert sys.modules["_synthetic_verified_release"] is foreign
    with pytest.raises(ValueError, match="fresh release module name"):
        load(release, pins)
    assert sys.modules["_synthetic_verified_release"] is foreign


def test_source_does_not_inherit_loader_future_annotations(sources):
    _, release, create = sources
    pins = create(b"T = int\ndef example(value: T):\n    return value\n")
    module = load(release, pins)
    assert module.example.__annotations__ == {"value": int}


def test_fifo_reviewed_source_refused_without_blocking(tmp_path):
    fifo = tmp_path / "synthetic-held.py"
    os.mkfifo(fifo, 0o600)
    script = """\
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import jev_decision3_candidate_http_owner as owner
try:
    owner._read_reviewed_source(Path(sys.argv[1]), '0' * 64)
except ValueError as error:
    assert 'bounded ordinary pinned source' in str(error)
else:
    raise AssertionError('FIFO unexpectedly accepted')
"""
    subprocess.run(
        [sys.executable, "-B", "-s", "-c", script, str(fifo)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        timeout=3,
    )
