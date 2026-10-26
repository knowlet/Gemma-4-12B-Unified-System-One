"""Model-free controls for selected same-user unit readback.

All UID, path-stat and subprocess observations below are invented.  These
controls never contact a user manager, socket, model, browser or network.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import stat
import subprocess
import sys
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "scripts/jev_user_unit_readback.py"
UNIT = "jev-invented-unit.service"
UID = 1234
DIRECTORY = f"/run/user/{UID}"
BUS = DIRECTORY + "/bus"
PROPERTY_NAMES = ("Description", "InvocationID", "MainPID", "RuntimeMaxUSec", "TimeoutStopUSec")
PROPERTIES = {
    "Description": "invented=nonce",
    "InvocationID": "a" * 32,
    "MainPID": "42",
    "RuntimeMaxUSec": "2h",
    "TimeoutStopUSec": "3s",
}


def load_helper():
    name = "_jev_user_unit_readback_invented_controls"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return name, module


def selection():
    return {
        "uid": UID,
        "xdg_runtime_dir": DIRECTORY,
        "dbus_session_bus_address": "unix:path=" + BUS,
    }


def invented_stat(inode, mode, uid=0):
    return types.SimpleNamespace(
        st_dev=7, st_ino=inode, st_uid=uid, st_mode=mode, st_mtime_ns=100, st_ctime_ns=200
    )


class BusFixture:
    def __init__(self, module):
        self.module = module
        self.tree = {
            "/": invented_stat(1, stat.S_IFDIR | 0o755),
            "/run": invented_stat(2, stat.S_IFDIR | 0o755),
            "/run/user": invented_stat(3, stat.S_IFDIR | 0o755),
            DIRECTORY: invented_stat(4, stat.S_IFDIR | 0o700, UID),
            BUS: invented_stat(5, stat.S_IFSOCK | 0o666, UID),
        }
        self.after = None
        self.events, self.calls = [], []
        self.queried = False
        self.stdout = "".join(f"{name}={PROPERTIES[name]}\n" for name in PROPERTY_NAMES)
        self.query_error = self.stat_after_error = self.deadline_after_error = None

    def lstat(self, path, *_args, **_kwargs):
        path = str(path)
        self.events.append("stat:" + path)
        if self.queried and self.stat_after_error is not None and path == BUS:
            raise self.stat_after_error
        tree = self.after if self.queried and self.after is not None else self.tree
        if path not in tree:
            raise AssertionError("unexpected path observation: " + path)
        return copy.copy(tree[path])

    def run(self, argv, **kwargs):
        self.events.append("query")
        self.calls.append((list(argv), copy.deepcopy(kwargs)))
        self.queried = True
        if self.query_error is not None:
            raise self.query_error
        return types.SimpleNamespace(stdout=self.stdout, stderr="", returncode=0)

    def deadline(self):
        self.events.append("deadline")
        if self.queried and self.deadline_after_error is not None:
            raise self.deadline_after_error

    def patches(self):
        stack = ExitStack()
        for target, attribute, value in (
            (self.module.os, "getuid", lambda: UID),
            (self.module.os, "geteuid", lambda: UID),
            (self.module.os, "lstat", self.lstat),
            (self.module.subprocess, "run", self.run),
        ):
            stack.enter_context(mock.patch.object(target, attribute, value))
        return stack

    def read(self, *, unit=UNIT, selected=None, deadline=None):
        with self.patches():
            return self.module.read_user_unit_properties(
                unit,
                selection() if selected is None else selected,
                deadline=self.deadline if deadline is None else deadline,
            )


class UserUnitReadbackControls(unittest.TestCase):
    def setUp(self):
        name, self.module = load_helper()
        self.addCleanup(sys.modules.pop, name, None)

    def test_selection_returns_exact_child_allowlist_without_changing_global_environment(self):
        fixture = BusFixture(self.module)
        invented = {
            "PATH": "/invented/untrusted",
            "LANG": "invented",
            "XDG_RUNTIME_DIR": "/invented/wrong-runtime",
            "DBUS_SESSION_BUS_ADDRESS": "tcp:host=invented,port=1",
            "LD_PRELOAD": "/invented/not-loaded",
            "HF_TOKEN": "invented-not-a-secret",
        }
        with mock.patch.dict(os.environ, invented, clear=True), fixture.patches():
            before = dict(os.environ)
            env = self.module.validate_selection(selection())
            self.assertEqual(
                env,
                {
                    "PATH": os.defpath,
                    "LANG": "C.UTF-8",
                    "XDG_RUNTIME_DIR": DIRECTORY,
                    "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + BUS,
                },
            )
            self.assertEqual(dict(os.environ), before)
            self.module.read_user_unit_properties(UNIT, selection(), deadline=fixture.deadline)
            self.assertEqual(dict(os.environ), before)
        self.assertEqual(len(fixture.calls), 1)
        self.assertEqual(fixture.calls[0][1]["env"], env)

    def test_fixed_argv_properties_timeout_and_deadline_order(self):
        fixture = BusFixture(self.module)
        self.assertEqual(fixture.read(), PROPERTIES)
        self.assertEqual(len(fixture.calls), 1)
        argv, kwargs = fixture.calls[0]
        self.assertEqual(
            argv,
            [
                "/usr/bin/systemctl",
                "--user",
                "show",
                UNIT,
                *("--property=" + name for name in PROPERTY_NAMES),
            ],
        )
        self.assertIs(kwargs["check"], True)
        self.assertIs(kwargs["capture_output"], True)
        self.assertIs(kwargs["text"], True)
        self.assertEqual(kwargs["timeout"], 1)
        self.assertFalse(kwargs.get("shell", False))
        self.assertEqual(fixture.events[0], "deadline")
        query = fixture.events.index("query")
        self.assertEqual(fixture.events[query - 1], "deadline")
        self.assertTrue(any(event.startswith("stat:") for event in fixture.events[:query]))
        self.assertTrue(any(event.startswith("stat:") for event in fixture.events[query + 1 :]))
        self.assertEqual(fixture.events[-1], "deadline")

    def test_selection_rejects_extra_missing_wrong_uid_and_noncanonical_addresses(self):
        invalid = [
            {**selection(), "extra": True},
            {key: value for key, value in selection().items() if key != "uid"},
            {**selection(), "uid": True},
            {**selection(), "uid": UID + 1},
            {**selection(), "xdg_runtime_dir": DIRECTORY + "/../1234"},
            {**selection(), "xdg_runtime_dir": "/invented/runtime"},
            {**selection(), "dbus_session_bus_address": "unix:abstract=invented"},
            {**selection(), "dbus_session_bus_address": "unix:path=/tmp/invented"},
            {**selection(), "dbus_session_bus_address": "unix:path=" + BUS + "\n"},
        ]
        for selected in invalid:
            with self.subTest(selected=selected):
                fixture = BusFixture(self.module)
                with self.assertRaises((ValueError, RuntimeError)):
                    fixture.read(selected=selected)
                self.assertEqual(fixture.calls, [])
        fixture = BusFixture(self.module)
        with fixture.patches(), mock.patch.object(self.module.os, "geteuid", return_value=UID + 1):
            with self.assertRaises((ValueError, RuntimeError)):
                self.module.validate_selection(selection())

    def test_unit_options_paths_nuls_and_unsupported_suffixes_never_launch(self):
        for unit in (
            "--user.service",
            "-n.service",
            "/tmp/invented.service",
            "a/b.service",
            "invented\0.service",
            "invented.service --user",
            "invented.socket",
            "x" * 256 + ".service",
            "",
            7,
        ):
            with self.subTest(unit=unit):
                fixture = BusFixture(self.module)
                with self.assertRaises((ValueError, RuntimeError, TypeError)):
                    fixture.read(unit=unit)
                self.assertEqual(fixture.calls, [])
        fixture = BusFixture(self.module)
        self.assertEqual(fixture.read(unit="invented.target"), PROPERTIES)

    def test_canonical_tree_refuses_symlinks_wrong_owner_mode_or_file_type(self):
        cases = (
            ("/run/user", "st_mode", stat.S_IFLNK | 0o777),
            (DIRECTORY, "st_mode", stat.S_IFREG | 0o700),
            (DIRECTORY, "st_mode", stat.S_IFDIR | 0o755),
            (DIRECTORY, "st_uid", UID + 1),
            (BUS, "st_mode", stat.S_IFREG | 0o666),
            (BUS, "st_mode", stat.S_IFLNK | 0o777),
            (BUS, "st_uid", UID + 1),
        )
        for path, field, value in cases:
            with self.subTest(path=path, field=field, value=value):
                fixture = BusFixture(self.module)
                setattr(fixture.tree[path], field, value)
                with self.assertRaises((ValueError, RuntimeError)):
                    fixture.read()
                self.assertEqual(fixture.calls, [])
        # The private0700 parent, UID and socket type are the socket authority;
        # a particular socket permission mask is not an extra requirement.
        fixture = BusFixture(self.module)
        fixture.tree[BUS].st_mode = stat.S_IFSOCK | 0o777
        self.assertEqual(fixture.read(), PROPERTIES)

    def test_postquery_identity_retarget_is_refused_but_mtime_and_ctime_are_ignored(self):
        for path, field in (
            (DIRECTORY, "st_ino"),
            (BUS, "st_ino"),
            (BUS, "st_dev"),
            (BUS, "st_uid"),
            (BUS, "st_mode"),
        ):
            with self.subTest(path=path, field=field):
                fixture = BusFixture(self.module)
                fixture.after = copy.deepcopy(fixture.tree)
                if field == "st_mode":
                    fixture.after[path].st_mode = stat.S_IFSOCK | 0o600
                else:
                    setattr(fixture.after[path], field, getattr(fixture.after[path], field) + 1)
                with self.assertRaises((ValueError, RuntimeError)):
                    fixture.read()
                self.assertEqual(len(fixture.calls), 1)
        fixture = BusFixture(self.module)
        fixture.after = copy.deepcopy(fixture.tree)
        for row in fixture.after.values():
            row.st_mtime_ns += 1_000_000
            row.st_ctime_ns += 1_000_000
        self.assertEqual(fixture.read(), PROPERTIES)

    def test_only_requested_property_subsets_are_returned_without_inventing_missing_fields(self):
        normal = "".join(f"{name}={PROPERTIES[name]}\n" for name in PROPERTY_NAMES)
        fixture = BusFixture(self.module)
        fixture.stdout = "MainPID=0\n"
        self.assertEqual(fixture.read(), {"MainPID": "0"})
        invalid = [
            normal + "Extra=invented\n",
            normal + "Description=duplicate\n",
            normal + "malformed-line\n",
            "x" * 65_537,
            b"MainPID=0\n",
        ]
        for stdout in invalid:
            with self.subTest(stdout=stdout):
                fixture = BusFixture(self.module)
                fixture.stdout = stdout
                with self.assertRaises((ValueError, RuntimeError)):
                    fixture.read()
                self.assertEqual(len(fixture.calls), 1)

    def test_deadline_refusal_before_launch_and_after_return_never_produces_success(self):
        fixture = BusFixture(self.module)
        primary = TimeoutError("invented initial deadline")

        def expired():
            raise primary

        with self.assertRaises(TimeoutError) as caught:
            fixture.read(deadline=expired)
        self.assertIs(caught.exception, primary)
        self.assertEqual(fixture.calls, [])
        fixture = BusFixture(self.module)
        before_query = TimeoutError("invented deadline after path checks")
        checks = []

        def expires_before_query():
            checks.append(True)
            if len(checks) == 2:
                raise before_query

        with self.assertRaises(TimeoutError) as caught:
            fixture.read(deadline=expires_before_query)
        self.assertIs(caught.exception, before_query)
        self.assertTrue(any(event.startswith("stat:") for event in fixture.events))
        self.assertEqual(fixture.calls, [])
        fixture = BusFixture(self.module)
        late = TimeoutError("invented postquery deadline")
        fixture.deadline_after_error = late
        with self.assertRaises(TimeoutError) as caught:
            fixture.read()
        self.assertIs(caught.exception, late)
        self.assertEqual(len(fixture.calls), 1)

    def test_timeout_or_baseexception_remains_primary_when_postchecks_and_notes_fail(self):
        class RefusesNotes(BaseException):
            def add_note(self, _note):
                raise RuntimeError("invented add_note refusal")

        failures = (
            subprocess.TimeoutExpired(["invented-systemctl"], 1),
            RefusesNotes("invented primary"),
        )
        for primary in failures:
            with self.subTest(primary=type(primary).__name__):
                fixture = BusFixture(self.module)
                fixture.query_error = primary
                fixture.stat_after_error = OSError("invented postquery lstat failure")
                fixture.deadline_after_error = TimeoutError("invented final deadline failure")
                with self.assertRaises(type(primary)) as caught:
                    fixture.read()
                self.assertIs(caught.exception, primary)
                self.assertEqual(len(fixture.calls), 1)
                self.assertEqual(fixture.events[-1], "deadline")
                if isinstance(primary, subprocess.TimeoutExpired):
                    self.assertTrue(getattr(primary, "__notes__", []))

    def test_nonzero_systemctl_exit_is_original_failure_with_no_retry_or_fallback(self):
        fixture = BusFixture(self.module)
        primary = subprocess.CalledProcessError(
            1, ["invented-systemctl"], stderr="invented unavailable bus"
        )
        fixture.query_error = primary
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            fixture.read()
        self.assertIs(caught.exception, primary)
        self.assertEqual(len(fixture.calls), 1)
        self.assertEqual(fixture.calls[0][0][0], "/usr/bin/systemctl")
        self.assertEqual(fixture.events[-1], "deadline")


if __name__ == "__main__":
    unittest.main()
