"""Only generated /tmp files; no official task content, systemd or GPU."""

import json
import os

import jev_decision3_jevbench_coherence_prepare as prep
import jev_decision3_jevbench_coherence_snapshot as snapshot
import pytest


@pytest.fixture
def fake(tmp_path, monkeypatch, request):
    source = tmp_path / "official"
    destination = tmp_path / "controlled"
    source.mkdir()
    pins = {}
    for name in prep.MINI_SDK | prep.MINI_ASSETS:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = ("synthetic " + name).encode()
        path.write_bytes(raw)
        pins[name] = snapshot.digest(raw)
    unselected = source / "src/jevbench/examples/unselected.jsonl"
    unselected.parent.mkdir(parents=True)
    unselected.write_bytes(b"synthetic example outside selection")
    monkeypatch.setattr(snapshot, "DESTINATION", destination)
    monkeypatch.setattr(snapshot, "SOURCE_ONLY", False)
    monkeypatch.setattr(snapshot, "_catalog", lambda: dict(sorted(pins.items())))

    def restore_permissions():
        if destination.is_dir():
            destination.chmod(0o700)
            for path in sorted(destination.rglob("*"), key=lambda p: len(p.parts)):
                if path.is_dir() and not path.is_symlink():
                    path.chmod(0o700)

    request.addfinalizer(restore_permissions)
    return source, destination, pins, unselected


def test_gate_refuses_before_catalog_or_filesystem(monkeypatch):
    monkeypatch.setattr(snapshot, "SOURCE_ONLY", True)
    monkeypatch.setattr(snapshot, "_catalog", lambda: pytest.fail("unexpected read"))
    with pytest.raises(ValueError, match="source-only"):
        snapshot.create("unused")


def test_exact_snapshot_read_only_and_unselected_never_opened(fake, monkeypatch):
    source, destination, pins, unselected = fake
    seen = []
    original = snapshot._read

    def read(path, expected, **kwargs):
        assert path != unselected, "unselected example must not be read"
        seen.append(str(path.relative_to(source)))
        return original(path, expected, **kwargs)

    monkeypatch.setattr(snapshot, "_read", read)
    report = snapshot.create(str(source))
    assert set(seen) == set(pins) and len(seen) == 48
    assert report["sdk_files"] == 42 and report["asset_files"] == 6
    assert report["snapshot_atomic_or_immutable"] is False
    assert destination.stat().st_mode & 0o777 == 0o555
    assert not (destination / "src/jevbench/examples").exists()
    for name, expected in pins.items():
        path = destination / name
        assert path.stat().st_mode & 0o777 == 0o444
        assert path.stat().st_nlink == 1 and snapshot.digest(path.read_bytes()) == expected


def test_synthetic_snapshot_passes_preparer_controlled_preflight(fake, monkeypatch):
    source, destination, pins, _unselected = fake
    snapshot.create(str(source))
    monkeypatch.setattr(prep, "OFFICIAL", destination)
    config = {"source_revision": "e18733694623aa93058e279c3534f8b8e2edefa3", "files": pins}
    prep._official(json.dumps(config).encode())


def test_catalog_uses_pinned_synthetic_config_and_selects_only_48(tmp_path, monkeypatch):
    source = tmp_path / "preparer.py"
    body = (
        "SOURCE_ONLY = True\nMINI_SDK = frozenset("
        + repr(sorted(prep.MINI_SDK))
        + ")\nMINI_ASSETS = frozenset("
        + repr(sorted(prep.MINI_ASSETS))
        + ")\n"
    ).encode()
    source.write_bytes(body)
    config = tmp_path / "config.json"
    pins = {name: "a" * 64 for name in prep.MINI_SDK | prep.MINI_ASSETS}
    pins["src/jevbench/examples/unselected.jsonl"] = "b" * 64
    raw = json.dumps(
        {"source_revision": "e18733694623aa93058e279c3534f8b8e2edefa3", "files": pins}
    ).encode()
    config.write_bytes(raw)
    monkeypatch.setattr(snapshot, "PREPARER", source)
    monkeypatch.setattr(snapshot, "PREPARER_SHA", snapshot.digest(body))
    monkeypatch.setattr(snapshot, "CONFIG", config)
    monkeypatch.setattr(snapshot, "CONFIG_SHA", snapshot.digest(raw))
    assert set(snapshot._catalog()) == prep.MINI_SDK | prep.MINI_ASSETS
    config.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source changed"):
        snapshot._catalog()


def test_existing_destination_never_overwritten_or_removed(fake):
    source, destination, _pins, _unselected = fake
    destination.mkdir()
    sentinel = destination / "user-file"
    sentinel.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="already exists"):
        snapshot.create(str(source))
    assert sentinel.read_bytes() == b"existing"


@pytest.mark.parametrize("fault", ["relative", "symlink", "heldout"])
def test_unapproved_source_root_refused_before_destination(fake, tmp_path, fault):
    source, destination, _pins, _unselected = fake
    if fault == "relative":
        name = "relative"
    elif fault == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        name = str(alias)
    else:
        bad = tmp_path / "heldout-private"
        bad.mkdir()
        name = str(bad)
    with pytest.raises(ValueError, match="checkout"):
        snapshot.create(name)
    assert not os.path.lexists(destination)


def test_changed_selected_source_rolls_back_only_created_files(fake):
    source, destination, pins, _unselected = fake
    (source / next(iter(pins))).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        snapshot.create(str(source))
    assert not os.path.lexists(destination)


def test_mid_write_failure_removes_owned_partial_snapshot(fake, monkeypatch):
    source, destination, _pins, _unselected = fake
    original = snapshot._write_new

    def write(path, raw, files):
        original(path, raw, files)
        raise OSError("synthetic interrupted publication")

    monkeypatch.setattr(snapshot, "_write_new", write)
    with pytest.raises(OSError, match="interrupted"):
        snapshot.create(str(source))
    assert not os.path.lexists(destination)


def test_partial_short_write_is_removed_on_failure(fake, monkeypatch):
    source, destination, _pins, _unselected = fake
    original = snapshot.os.write
    calls = 0

    def write(fd, raw):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(fd, raw[:3])
        raise OSError("synthetic partial write")

    monkeypatch.setattr(snapshot.os, "write", write)
    with pytest.raises(OSError, match="partial write"):
        snapshot.create(str(source))
    assert calls == 2 and not os.path.lexists(destination)


def test_replaced_file_preserved_and_cleanup_reported(fake, monkeypatch):
    source, destination, _pins, _unselected = fake
    original = snapshot._write_new
    replaced = []

    def write(path, raw, files):
        original(path, raw, files)
        path.unlink()
        path.write_bytes(b"someone else's data")
        replaced.append(path)
        raise OSError("synthetic race")

    monkeypatch.setattr(snapshot, "_write_new", write)
    with pytest.raises(OSError, match="race") as failure:
        snapshot.create(str(source))
    assert replaced[0].read_bytes() == b"someone else's data"
    assert any("cleanup unconfirmed" in note for note in failure.value.__notes__)
    assert destination.exists()


def test_partial_read_only_transition_is_rolled_back(fake, monkeypatch):
    source, destination, _pins, _unselected = fake
    original = snapshot._dir_mode
    calls = 0

    def mode(path, identity, value):
        nonlocal calls
        if value == 0o555:
            calls += 1
            if calls == 3:
                raise OSError("synthetic chmod failure")
        return original(path, identity, value)

    monkeypatch.setattr(snapshot, "_dir_mode", mode)
    with pytest.raises(OSError, match="chmod failure"):
        snapshot.create(str(source))
    assert not destination.exists()


def test_selected_symlink_fails_without_opening_target(fake, tmp_path):
    source, destination, pins, _unselected = fake
    selected = source / next(iter(pins))
    other = tmp_path / "other"
    other.write_bytes(selected.read_bytes())
    selected.unlink()
    selected.symlink_to(other)
    with pytest.raises(ValueError, match="ordinary source"):
        snapshot.create(str(source))
    assert not destination.exists()
