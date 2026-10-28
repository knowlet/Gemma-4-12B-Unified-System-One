"""Synthetic files and GPU responses only; no real source or model execution."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import jev_decision3_cuda_synthetic_bootstrap as bootstrap
import jev_decision3_cuda_synthetic_prepare as prepare
import pytest


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    base = root / "artifacts/jev-runtime/decision3-cuda-synthetic"
    base.parent.mkdir(parents=True)
    snapshot = root / "publisher"
    snapshot.mkdir()
    real_python = root / "runtime/python3.13"
    real_python.parent.mkdir()
    real_python.write_bytes(b"synthetic interpreter")
    python = root / "python"
    python.symlink_to(real_python)
    for name, value in (
        ("ROOT", root),
        ("SCRIPTS", scripts),
        ("BASE", base),
        ("SNAPSHOT", snapshot),
        ("PYTHON", python),
        ("PYTHON_REAL", real_python),
    ):
        monkeypatch.setattr(prepare, name, value)
    monkeypatch.setattr(prepare, "SOURCE_ONLY", False)
    monkeypatch.setattr(prepare.secrets, "token_hex", lambda _: "a" * 32)
    monkeypatch.setattr(prepare, "_gpu_uuid", lambda: "GPU-12345678-1234-1234-1234-123456789abc")
    held = {}
    pins = {}
    for index, name in enumerate(prepare.HELD_SHA):
        path = scripts / name
        raw = ("SOURCE_ONLY = True\nsynthetic = " + str(index) + "\n").encode()
        path.write_bytes(raw)
        held[name] = sha(raw)
        pins[path] = sha(raw)
    monkeypatch.setattr(prepare, "HELD_SHA", held)
    for index in range(17):
        path = root / "support" / (str(index) + ".py")
        path.parent.mkdir(exist_ok=True)
        raw = ("synthetic support " + str(index)).encode()
        path.write_bytes(raw)
        pins[path] = sha(raw)
    fast = snapshot / "d3_fast.py"
    fast.write_bytes(b"synthetic publisher")
    (snapshot / "fast-hardlink").hardlink_to(fast)
    manifest = snapshot / "MODEL_MANIFEST.json"
    manifest.write_bytes(b"synthetic manifest")
    (snapshot / "manifest-hardlink").hardlink_to(manifest)
    monkeypatch.setattr(prepare, "MANIFEST_SHA", sha(b"synthetic manifest"))
    pins[fast] = sha(b"synthetic publisher")
    pins[real_python] = sha(b"synthetic interpreter")
    assert len(pins) == 23
    monkeypatch.setattr(prepare, "_pinned", lambda: dict(pins))
    return root, pins


def test_exact_fixed_inventory_matches_both_reviewed_consumers():
    fixed = {str(path) for path in prepare._pinned()}
    released = {str(prepare._release(name)) for name in prepare.HELD_SHA}
    assert len(fixed) == 23 and len(released) == 4
    assert fixed | released == bootstrap._source_set()
    assert all(
        "heldout" not in path and "root-done-feedback" not in path for path in fixed | released
    )


def test_source_gate_refuses_before_any_read(monkeypatch):
    monkeypatch.setattr(prepare, "SOURCE_ONLY", True)
    monkeypatch.setattr(prepare, "_pinned", lambda: pytest.fail("must not read"))
    with pytest.raises(ValueError, match="preparer release"):
        prepare.prepare()


def test_prepares_exclusive_gate_only_releases_and_private_plan(synthetic):
    _root, expected = synthetic
    result = prepare.prepare()
    path = Path(result["plan"])
    raw = path.read_bytes()
    plan = json.loads(raw)
    assert result["sha256"] == sha(raw) and result["pin_count"] == len(plan["source_pins"]) == 27
    assert set(plan["source_pins"]) == {str(path) for path in expected} | {
        str(prepare._release(name)) for name in prepare.HELD_SHA
    }
    assert plan["wall_seconds"] == 120 and plan["memory_floor_bytes"] == 34709107508
    assert plan["gpu_uuid"] == "GPU-12345678-1234-1234-1234-123456789abc"
    assert plan["unit"] == "jev-cuda-synthetic-" + "a" * 32 + ".service"
    assert path.stat().st_mode & 0o077 == 0 and path.parent.stat().st_mode & 0o077 == 0
    assert Path(plan["output_dir"]).is_dir() and not any(Path(plan["output_dir"]).iterdir())
    assert not Path(result["report"]).exists()
    for name in prepare.HELD_SHA:
        original = (prepare.SCRIPTS / name).read_bytes()
        release = prepare._release(name).read_bytes()
        assert release == original.replace(prepare.MARKER, b"SOURCE_ONLY = False", 1)
        assert plan["source_pins"][str(prepare._release(name))] == sha(release)


def test_existing_release_is_never_overwritten_or_removed(synthetic):
    name = next(iter(prepare.HELD_SHA))
    existing = prepare._release(name)
    existing.write_bytes(b"existing user content")
    with pytest.raises(FileExistsError, match="already exists"):
        prepare.prepare()
    assert existing.read_bytes() == b"existing user content"
    assert not prepare.BASE.exists()


def test_changed_reviewed_source_stops_before_creating_any_files(synthetic):
    path = prepare.SCRIPTS / next(iter(prepare.HELD_SHA))
    path.write_bytes(b"SOURCE_ONLY = True\nchanged = 1\n")
    with pytest.raises(ValueError, match="digest changed"):
        prepare.prepare()
    assert all(not os.path.lexists(prepare._release(name)) for name in prepare.HELD_SHA)
    assert not prepare.BASE.exists()


def test_changed_manifest_stops_before_creating_files(synthetic):
    (prepare.SNAPSHOT / "MODEL_MANIFEST.json").write_bytes(b"changed manifest")
    with pytest.raises(ValueError, match="digest changed"):
        prepare.prepare()
    assert all(not os.path.lexists(prepare._release(name)) for name in prepare.HELD_SHA)
    assert not prepare.BASE.exists()


def test_refuses_unselected_source_symlink_or_extra_hardlink(synthetic):
    path = prepare.SCRIPTS / next(iter(prepare.HELD_SHA))
    other = path.parent / "synthetic-unselected"
    other.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(other)
    with pytest.raises(ValueError, match="canonical"):
        prepare.prepare()
    assert not prepare.BASE.exists()


def test_existing_base_must_be_private(synthetic):
    prepare.BASE.mkdir(mode=0o755)
    prepare.BASE.chmod(0o755)
    with pytest.raises(ValueError, match="owned private"):
        prepare.prepare()
    assert all(not os.path.lexists(prepare._release(name)) for name in prepare.HELD_SHA)


def test_failed_publish_rolls_back_only_owned_files(synthetic, monkeypatch):
    original = prepare.os.link
    names = list(prepare.HELD_SHA)
    first, second = prepare._release(names[0]), prepare._release(names[1])

    def contested(src, dst, **kwargs):
        if Path(dst) == second:
            first.unlink()
            first.write_bytes(b"replacement user content")
            second.write_bytes(b"concurrent user content")
            raise FileExistsError("synthetic competing file")
        return original(src, dst, **kwargs)

    monkeypatch.setattr(prepare.os, "link", contested)
    with pytest.raises(FileExistsError, match="competing file"):
        prepare.prepare()
    assert first.read_bytes() == b"replacement user content"
    assert second.read_bytes() == b"concurrent user content"
    assert not prepare.BASE.exists()
    assert not any(path.name.endswith(".tmp") for path in prepare.SCRIPTS.iterdir())


@pytest.mark.parametrize(
    "output",
    [
        "",
        "GPU-123, NVIDIA GB10",
        "GPU-12345678-1234-1234-1234-123456789abc, another device",
        "GPU-12345678-1234-1234-1234-123456789abc, NVIDIA GB10\nGPU-12345678-1234-1234-1234-123456789abc, NVIDIA GB10",
    ],
)
def test_gpu_observation_rejects_other_or_multiple_devices(monkeypatch, output):
    monkeypatch.setattr(
        prepare.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout=output)
    )
    with pytest.raises(ValueError, match="one NVIDIA GB10"):
        prepare._gpu_uuid()


def test_gpu_observation_accepts_exactly_one_gb10(monkeypatch):
    value = "GPU-12345678-1234-1234-1234-123456789abc"
    monkeypatch.setattr(
        prepare.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=value + ", NVIDIA GB10\n"),
    )
    assert prepare._gpu_uuid() == value
