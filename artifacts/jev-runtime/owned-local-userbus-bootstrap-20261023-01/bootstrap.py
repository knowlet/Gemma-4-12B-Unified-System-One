#!/usr/bin/env python3
"""Dry-default, same-Python bootstrap for the verified local user bus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTRY_ROOT = ROOT / "artifacts/jev-runtime"
INTERPRETERS = {
    "/home/albert_hsieh/ComfyUI/.venv/bin/python": {
        "canonical": "/home/albert_hsieh/.local/share/uv/python/cpython-3.13.14-linux-aarch64-gnu/bin/python3.13",
        "stat": {
            "st_dev": 66306,
            "st_ino": 17956884,
            "st_size": 22257464,
            "st_mtime_ns": 1784109750677901111,
        },
    },
    "/usr/bin/python3.12": {
        "canonical": "/usr/bin/python3.12",
        "stat": {
            "st_dev": 66306,
            "st_ino": 47972942,
            "st_size": 7845048,
            "st_mtime_ns": 1788171506000000000,
        },
    },
}
BUS_MODES = frozenset((0o600, 0o660, 0o666))
BUS_KEYS = ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")
MODEL_MODULES = frozenset(("torch", "transformers", "safetensors", "numpy", "s1"))
MAX_SOURCE = 200_000


def need(value, message):
    if not value:
        raise RuntimeError(message)


def signature(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_size, info.st_mtime_ns)


def checked_entry(value, expected):
    path = Path(value)
    need(
        path.is_absolute()
        and str(path) == os.path.normpath(str(path))
        and path.suffix == ".py"
        and path.is_relative_to(ENTRY_ROOT)
        and path.resolve(strict=True) == path,
        "Explicit ordinary Python source inside the runtime source tree",
    )
    need(
        type(expected) is str and re.fullmatch(r"[a-f0-9]{64}", expected), "Explicit source SHA256"
    )
    before = path.lstat()
    need(
        stat.S_ISREG(before.st_mode)
        and before.st_uid == os.geteuid()
        and before.st_nlink == 1
        and 0 < before.st_size < MAX_SOURCE
        and not (stat.S_IMODE(before.st_mode) & 0o7022),
        "Small owned regular source without writable group/other, links or special bits",
    )
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(descriptor)
        need(signature(before) == signature(opened), "Source identity changed while opening")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(MAX_SOURCE)
        need(
            len(raw) == before.st_size
            and hashlib.sha256(raw).hexdigest() == expected
            and signature(os.fstat(descriptor)) == signature(before)
            and signature(path.lstat()) == signature(before),
            "Selected source bytes/identity changed",
        )
    finally:
        os.close(descriptor)
    return path


def local_bus():
    uid = os.geteuid()
    need(
        uid > 0 and uid == os.getuid() and os.getegid() == os.getgid(),
        "Ordinary same-credential user",
    )
    directory = Path("/run/user") / str(uid)
    bus = directory / "bus"
    need(directory.resolve(strict=True) == directory, "Canonical local runtime directory")
    parent = directory.lstat()
    endpoint = bus.lstat()
    need(
        stat.S_ISDIR(parent.st_mode)
        and parent.st_uid == uid
        and stat.S_IMODE(parent.st_mode) == 0o700,
        "Private same-user runtime directory mode0700",
    )
    need(
        stat.S_ISSOCK(endpoint.st_mode)
        and endpoint.st_uid == uid
        and stat.S_IMODE(endpoint.st_mode) in BUS_MODES,
        "Same-user local socket with an explicit accepted mode",
    )
    return {
        "environment": {
            "XDG_RUNTIME_DIR": str(directory),
            "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + str(bus),
        },
        "directory_identity": (parent.st_dev, parent.st_ino, parent.st_uid, parent.st_mode),
        "socket_identity": (endpoint.st_dev, endpoint.st_ino, endpoint.st_uid, endpoint.st_mode),
    }


def runtime_flags():
    need(
        sys.flags.no_user_site == 1 and sys.dont_write_bytecode,
        "Original -B -s Python flags required",
    )
    selected = INTERPRETERS.get(sys.executable)
    need(selected is not None, "Exact selected interpreter allowlist")
    canonical = Path(selected["canonical"])
    need(
        Path(sys.executable).resolve(strict=True) == canonical,
        "Selected logical Python and canonical executable without exec handoff",
    )
    actual = canonical.stat()
    need(
        {key: getattr(actual, key) for key in selected["stat"]} == selected["stat"],
        "Selected canonical executable stat identity",
    )
    need(not MODEL_MODULES.intersection(sys.modules), "No preloaded model/runtime modules")


def execute(entry, expected_entry, expected_wrapper, arguments):
    runtime_flags()
    wrapper = checked_entry(__file__, expected_wrapper)
    target = checked_entry(entry, expected_entry)
    need(target != wrapper, "Wrapper cannot dispatch itself")
    need(
        type(arguments) is list and all(type(value) is str for value in arguments),
        "Literal entry arguments",
    )
    selected_bus = local_bus()
    os.environ.update(selected_bus["environment"])
    # This mutates no other environment values and never copies a parent/session environment.
    previous_argv = sys.argv
    sys.argv = [str(target), *arguments]
    primary = None
    try:
        try:
            return runpy.run_path(str(target), run_name="__main__")
        except SystemExit as result:
            if result.code is not None and result.code != 0:
                raise
            return None
    except BaseException as error:
        primary = error
        raise
    finally:
        sys.argv = previous_argv
        try:
            checked_entry(wrapper, expected_wrapper)
            checked_entry(target, expected_entry)
            need(local_bus() == selected_bus, "Selected local bus identity changed during entry")
            need(
                all(
                    os.environ.get(key) == value
                    for key, value in selected_bus["environment"].items()
                ),
                "Entry changed the selected local bus environment",
            )
        except BaseException as secondary:
            if primary is None:
                raise
            primary.add_note(
                "Local user-bus/source final authentication also failed: " + repr(secondary)
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--expected-wrapper-sha256")
    parser.add_argument("--entry", type=Path)
    parser.add_argument("--entry-sha256")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "dry_default_no_bus_query_entry_or_runtime_execution",
                    "environment_keys": list(BUS_KEYS),
                    "dispatch": "same Python runpy.run_path",
                    "model_or_service_or_qualification_action": False,
                }
            )
        )
        return 0
    need(
        args.entry is not None and args.arguments[:1] == ["--"],
        "Explicit selected entry and -- argument separator",
    )
    execute(args.entry, args.entry_sha256, args.expected_wrapper_sha256, args.arguments[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
