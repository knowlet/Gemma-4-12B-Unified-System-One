"""Held creator for the fixed official coherence MINI controlled snapshot.

Only the 42 SDK sources and six data assets selected by the reviewed preparer
are read. The destination is fixed, private until complete, and never replaced.
Read-only mode bits deter accidental writes; an owner or privileged process can
change them. Identity checks/unlink are not atomic and hard termination can
leave partial output. This script does not create a benchmark plan or run GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import types
from pathlib import Path

SOURCE_ONLY = True
ROOT = Path(__file__).absolute().parent.parent
PREPARER = ROOT / "scripts/jev_decision3_jevbench_coherence_prepare.py"
PREPARER_SHA = "d7976a8ae7f4da85e865a01ce4a4a81f20272449fa94962cf9fd86c0526cc3a1"
CONFIG = ROOT / "configs/benchmarks/jevbench-coherence-public.json"
CONFIG_SHA = "ebacca34e3d4193419e43629f73fa5fed6617834415ff87ff6d12a09b6d3904a"
DESTINATION = Path("/tmp/jev-coherence-mini-controlled-snapshot")


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _decode(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        raw, object_pairs_hook=pairs, parse_constant=lambda _: need(False, "nonfinite JSON")
    )


def _read(path, expected, limit=16 * 1024**2):
    need(path.is_absolute() and path.resolve() == path, "ordinary source path required")
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        first = os.fstat(fd)
        need(
            stat.S_ISREG(first.st_mode) and first.st_size <= limit,
            "bounded regular source required",
        )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        last, current = os.fstat(fd), os.stat(path, follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
        need(
            tuple(getattr(first, x) for x in fields)
            == tuple(getattr(last, x) for x in fields)
            == tuple(getattr(current, x) for x in fields)
            and len(raw) == first.st_size
            and digest(raw) == expected,
            "selected source changed",
        )
        return raw
    finally:
        os.close(fd)


def _catalog():
    raw = _read(PREPARER, PREPARER_SHA, limit=8 * 1024**2)
    module = types.ModuleType("_reviewed_coherence_preparer")
    module.__file__ = str(PREPARER)
    exec(compile(raw, str(PREPARER), "exec", dont_inherit=True), module.__dict__)
    need(
        module.SOURCE_ONLY is True and len(module.MINI_SDK) == 42 and len(module.MINI_ASSETS) == 6,
        "reviewed MINI inventory required",
    )
    selected = module.MINI_SDK | module.MINI_ASSETS
    need(
        len(selected) == 48
        and all(
            "/examples/" not in name
            and "heldout" not in name.lower()
            and "root-done-feedback" not in name.lower()
            for name in selected
        ),
        "exact MINI inventory required",
    )
    config = _decode(_read(CONFIG, CONFIG_SHA, limit=8 * 1024**2))
    files = config.get("files") if type(config) is dict else None
    need(
        type(files) is dict
        and selected.issubset(files)
        and config.get("source_revision") == "e18733694623aa93058e279c3534f8b8e2edefa3"
        and all(
            type(files[name]) is str and re.fullmatch(r"[0-9a-f]{64}", files[name])
            for name in selected
        ),
        "official MINI config required",
    )
    return {name: files[name] for name in sorted(selected)}


def _identity(path):
    info = path.lstat()
    return info.st_dev, info.st_ino


def _file_identity(info):
    return tuple(
        getattr(info, name)
        for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
    )


def _write_new(path, raw, files):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        files.append((path, _file_identity(info)))
        offset = 0
        while offset < len(raw):
            count = os.write(fd, raw[offset:])
            need(count > 0, "snapshot write made no progress")
            offset += count
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        if files and files[-1][0] == path:
            files[-1] = (path, _file_identity(os.fstat(fd)))
        os.close(fd)


def _dir_mode(path, identity, mode):
    need(path.resolve() == path, "owned directory path changed")
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        need(
            (info.st_dev, info.st_ino) == identity
            and _identity(path) == identity
            and info.st_uid == os.getuid(),
            "owned directory identity changed",
        )
        os.fchmod(fd, mode)
        os.fsync(fd)
    finally:
        os.close(fd)


def _cleanup(files, directories):
    """Touch only identities created by this invocation; report any failure."""
    errors = []
    for path, identity in directories:
        try:
            _dir_mode(path, identity, 0o700)
        except (OSError, ValueError) as error:
            errors.append(type(error).__name__)
    directory_ids = dict(directories)
    for path, identity in reversed(files):
        try:
            need(
                path.resolve() == path and _identity(path.parent) == directory_ids[path.parent],
                "file parent changed",
            )
            if _file_identity(path.lstat()) == identity:
                path.unlink()
            else:
                errors.append("file_identity_changed")
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError) as error:
            errors.append(type(error).__name__)
    for path, identity in reversed(directories):
        try:
            need(
                path.resolve() == path and _identity(path) == identity, "directory identity changed"
            )
            path.rmdir()
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as error:
            errors.append(type(error).__name__)
    return errors


def create(source_root):
    need(SOURCE_ONLY is False, "reviewed source-only snapshot creator required")
    selected = _catalog()
    need(type(source_root) is str, "ordinary source checkout required")
    source = Path(source_root)
    need(
        source.is_absolute()
        and str(source) == source_root
        and source.resolve() == source
        and source.is_dir()
        and not any(term in source_root.lower() for term in ("heldout", "root-done-feedback")),
        "ordinary selected public checkout required",
    )
    dest = DESTINATION
    need(
        dest.is_absolute()
        and dest.parent.resolve() == dest.parent
        and dest.parent.is_dir()
        and not source.is_relative_to(dest)
        and not dest.is_relative_to(source),
        "fixed separate destination required",
    )
    if os.path.lexists(dest):
        raise FileExistsError("snapshot destination already exists")
    directories, files = [], []
    try:
        dest.mkdir(mode=0o700)
        directories.append((dest, _identity(dest)))
        selected_dirs = set()
        for name in selected:
            path = Path(name)
            need(
                not path.is_absolute()
                and ".." not in path.parts
                and str(path) == name
                and path.parts[:2] == ("src", "jevbench"),
                "fixed MINI relative path required",
            )
            selected_dirs.update(path.parents)
        selected_dirs.discard(Path("."))
        for relative in sorted(selected_dirs, key=lambda x: (len(x.parts), str(x))):
            target = dest / relative
            target.mkdir(mode=0o700)
            directories.append((target, _identity(target)))
        for name, expected in selected.items():
            raw = _read(source / name, expected)
            _write_new(dest / name, raw, files)
        for path, identity in reversed(directories):
            _dir_mode(path, identity, 0o555)
        inventory = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
        return {
            "snapshot": str(dest),
            "sdk_files": 42,
            "asset_files": 6,
            "inventory_sha256": digest(inventory),
            "config_sha256": CONFIG_SHA,
            "snapshot_atomic_or_immutable": False,
        }
    except BaseException as error:
        errors = _cleanup(files, directories)
        if errors:
            error.add_note("owned snapshot cleanup unconfirmed: " + ",".join(sorted(set(errors))))
        raise


def main():
    need(SOURCE_ONLY is False, "reviewed source-only snapshot creator required")
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    print(json.dumps(create(args.source_root), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
