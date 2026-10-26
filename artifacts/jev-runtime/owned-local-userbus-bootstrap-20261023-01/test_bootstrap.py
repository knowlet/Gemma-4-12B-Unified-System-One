"""Stdlib fakes and tiny owned source fixtures; no real bus, ML, service or HTTP."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import stat
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("local_bus_bootstrap_test", HERE / "bootstrap.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def info(kind, mode, uid=1001, inode=7):
    return NS(st_dev=3, st_ino=inode, st_uid=uid, st_mode=kind | mode)


class SourceFixtureTests(unittest.TestCase):
    def setUp(self):
        self.previous_umask = os.umask(0o022)

    def tearDown(self):
        os.umask(self.previous_umask)


class LocalBusTests(SourceFixtureTests):
    def fixture(self, directory=None, endpoint=None, resolved=None):
        stack = ExitStack()
        entries = {
            "/run/user/1001": directory or info(stat.S_IFDIR, 0o700),
            "/run/user/1001/bus": endpoint or info(stat.S_IFSOCK, 0o666, inode=8),
        }
        stack.enter_context(patch.object(b.os, "geteuid", return_value=1001))
        stack.enter_context(patch.object(b.os, "getuid", return_value=1001))
        stack.enter_context(patch.object(b.os, "getegid", return_value=1001))
        stack.enter_context(patch.object(b.os, "getgid", return_value=1001))
        stack.enter_context(
            patch.object(
                b.Path, "resolve", autospec=True, side_effect=lambda p, **_kw: resolved or p
            )
        )
        stack.enter_context(
            patch.object(b.Path, "lstat", autospec=True, side_effect=lambda p: entries[str(p)])
        )
        return stack

    def test_verified_private_directory_and_socket0666_generate_only_local_values(self):
        with self.fixture():
            value = b.local_bus()
        self.assertEqual(
            value["environment"],
            {
                "XDG_RUNTIME_DIR": "/run/user/1001",
                "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1001/bus",
            },
        )

    def test_socket0600_and0660_also_require_same_private_parent(self):
        for mode in (0o600, 0o660):
            with self.fixture(endpoint=info(stat.S_IFSOCK, mode)):
                self.assertEqual(len(b.local_bus()["environment"]), 2)

    def test_wrong_directory_owner_mode_type_or_symlink_refuses(self):
        for directory in (
            info(stat.S_IFDIR, 0o700, uid=1002),
            info(stat.S_IFDIR, 0o770),
            info(stat.S_IFREG, 0o700),
        ):
            with self.fixture(directory=directory):
                with self.assertRaisesRegex(RuntimeError, "Private"):
                    b.local_bus()
        with self.fixture(resolved=Path("/wrong/runtime")):
            with self.assertRaisesRegex(RuntimeError, "Canonical"):
                b.local_bus()

    def test_wrong_bus_owner_mode_or_type_refuses(self):
        for endpoint in (
            info(stat.S_IFSOCK, 0o666, uid=1002),
            info(stat.S_IFSOCK, 0o777),
            info(stat.S_IFREG, 0o666),
            info(stat.S_IFLNK, 0o777),
        ):
            with self.fixture(endpoint=endpoint):
                with self.assertRaisesRegex(RuntimeError, "Same-user local socket"):
                    b.local_bus()

    def test_different_effective_credentials_refuse(self):
        with self.fixture(), patch.object(b.os, "getuid", return_value=1002):
            with self.assertRaisesRegex(RuntimeError, "same-credential"):
                b.local_bus()


class EntryTests(SourceFixtureTests):
    def test_valid_small_owned_source_and_wrong_sha(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            path = Path(temp) / "fixture.py"
            path.write_text("fixture = 1\n")
            self.assertEqual(b.checked_entry(path, sha(path)), path)
            with self.assertRaisesRegex(RuntimeError, "Selected source"):
                b.checked_entry(path, "0" * 64)
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                b.checked_entry(path, "bad")

    def test_symlink_outside_non_python_and_writable_sources_refuse(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            path = Path(temp) / "fixture.py"
            path.write_text("fixture=1\n")
            link = Path(temp) / "link.py"
            link.symlink_to(path)
            with self.assertRaises(RuntimeError):
                b.checked_entry(link, sha(path))
            with self.assertRaises(RuntimeError):
                b.checked_entry(Path("/tmp/outside.py"), "0" * 64)
            other = Path(temp) / "fixture.bin"
            other.write_text("fixture=1\n")
            with self.assertRaises(RuntimeError):
                b.checked_entry(other, sha(other))
            path.chmod(0o666)
            with self.assertRaisesRegex(RuntimeError, "Small owned regular"):
                b.checked_entry(path, sha(path))

    def test_original_flags_and_selected_python_are_required(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            python = Path(temp) / "fake-python"
            python.write_text("not an executable\n")
            actual = python.stat()
            selected = {
                key: getattr(actual, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
            }
            with (
                patch.object(
                    b, "INTERPRETERS", {str(python): {"canonical": str(python), "stat": selected}}
                ),
                patch.object(b.sys, "executable", str(python)),
            ):
                b.runtime_flags()
                with patch.object(b.sys, "dont_write_bytecode", False):
                    with self.assertRaisesRegex(RuntimeError, "-B -s"):
                        b.runtime_flags()
                with patch.object(b.sys, "flags", NS(no_user_site=0)):
                    with self.assertRaisesRegex(RuntimeError, "-B -s"):
                        b.runtime_flags()
                with patch.dict(sys.modules, {"torch": NS()}):
                    with self.assertRaisesRegex(RuntimeError, "No preloaded"):
                        b.runtime_flags()
            with patch.object(b.sys, "executable", "/unselected/python"):
                with self.assertRaisesRegex(RuntimeError, "interpreter allowlist"):
                    b.runtime_flags()

    def test_exact_two_logical_interpreters_and_symlink_canonical_stats(self):
        policies = {
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
        self.assertEqual(b.INTERPRETERS, policies)
        for logical, policy in policies.items():
            canonical = Path(policy["canonical"])
            selected = policy["stat"]
            with (
                patch.object(b.sys, "executable", logical),
                patch.object(b.Path, "resolve", autospec=True, return_value=canonical),
                patch.object(b.Path, "stat", autospec=True, return_value=NS(**selected)),
                patch.object(
                    b,
                    "checked_entry",
                    side_effect=AssertionError("source rules must not apply to the interpreter"),
                ),
            ):
                b.runtime_flags()
                with patch.object(b.Path, "resolve", return_value=Path("/wrong/python")):
                    with self.assertRaisesRegex(RuntimeError, "canonical executable"):
                        b.runtime_flags()
                for key in selected:
                    changed = {**selected, key: selected[key] + 1}
                    with patch.object(b.Path, "stat", return_value=NS(**changed)):
                        with self.assertRaisesRegex(RuntimeError, "stat identity"):
                            b.runtime_flags()
        for logical in [
            policies["/home/albert_hsieh/ComfyUI/.venv/bin/python"]["canonical"],
            "/usr/bin/python3",
            str(b.ROOT / ".venv/bin/python"),
        ]:
            with patch.object(b.sys, "executable", logical):
                with self.assertRaisesRegex(RuntimeError, "interpreter allowlist"):
                    b.runtime_flags()


class DispatchTests(SourceFixtureTests):
    def bus(self):
        return {
            "environment": {
                "XDG_RUNTIME_DIR": "/run/user/1001",
                "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1001/bus",
            },
            "directory_identity": (3, 7, 1001, stat.S_IFDIR | 0o700),
            "socket_identity": (3, 8, 1001, stat.S_IFSOCK | 0o666),
        }

    def dispatch(self, path, *, run=None, bus=None, wrapper_sha=None):
        with (
            patch.object(b, "runtime_flags"),
            patch.object(b, "local_bus", side_effect=bus or [self.bus(), self.bus()]),
        ):
            if run is None:
                return b.execute(
                    path,
                    sha(path),
                    wrapper_sha or sha(HERE / "bootstrap.py"),
                    ["--literal", "value"],
                )
            with patch.object(b.runpy, "run_path", side_effect=run):
                return b.execute(
                    path,
                    sha(path),
                    wrapper_sha or sha(HERE / "bootstrap.py"),
                    ["--literal", "value"],
                )

    def test_real_runpy_tiny_source_keeps_pid_python_file_and_original_arguments(self):
        with (
            tempfile.TemporaryDirectory(dir=HERE) as temp,
            patch.dict(
                os.environ,
                {
                    "JEV_COMMAND_NONCE": "keep",
                    "DBUS_SESSION_BUS_ADDRESS": "tcp:host=untrusted",
                    "XDG_RUNTIME_DIR": "/untrusted",
                },
                clear=True,
            ),
        ):
            path = Path(temp) / "fixture.py"
            path.write_text(
                "import os,sys\nobserved=(os.getpid(),sys.executable,__file__,list(sys.argv),dict(os.environ))\n"
            )
            before_pid, before_python = os.getpid(), sys.executable
            result = self.dispatch(path)
            pid, python, file, argv, env = result["observed"]
            self.assertEqual((pid, python, file), (before_pid, before_python, str(path)))
            self.assertEqual(argv, [str(path), "--literal", "value"])
            self.assertEqual(env, {"JEV_COMMAND_NONCE": "keep", **self.bus()["environment"]})
            self.assertEqual(
                os.environ["DBUS_SESSION_BUS_ADDRESS"],
                self.bus()["environment"]["DBUS_SESSION_BUS_ADDRESS"],
            )

    def test_success_systemexit_zero_does_not_hide_postauth_failure(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp, patch.dict(os.environ, {}, clear=True):
            path = Path(temp) / "fixture.py"
            path.write_text("raise SystemExit(0)\n")
            changed = self.bus()
            changed["socket_identity"] = (3, 99, 1001, stat.S_IFSOCK | 0o666)
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                self.dispatch(path, bus=[self.bus(), changed])

    def test_zero_exit_success_passes_and_nonzero_primary_survives_secondary(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp, patch.dict(os.environ, {}, clear=True):
            path = Path(temp) / "fixture.py"
            path.write_text("raise SystemExit(0)\n")
            self.assertIsNone(self.dispatch(path))
            original = KeyboardInterrupt("original synthetic primary")

            def fail(*_args, **_kw):
                raise original

            with self.assertRaises(KeyboardInterrupt) as caught:
                self.dispatch(path, run=fail, bus=[self.bus(), OSError("secondary fake bus check")])
            self.assertIs(caught.exception, original)
            self.assertTrue(any("final authentication" in note for note in original.__notes__))
            original_exit = SystemExit(7)

            def exit7(*_args, **_kw):
                raise original_exit

            with self.assertRaises(SystemExit) as caught:
                self.dispatch(path, run=exit7)
            self.assertIs(caught.exception, original_exit)

    def test_changed_entry_or_bus_environment_demotes_normal_return(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp, patch.dict(os.environ, {}, clear=True):
            path = Path(temp) / "fixture.py"
            path.write_text("fixture=1\n")

            def mutate(*_args, **_kw):
                path.write_text("fixture=2\n")

            with self.assertRaisesRegex(RuntimeError, "Selected source"):
                self.dispatch(path, run=mutate)

            def env_mutate(*_args, **_kw):
                os.environ["DBUS_SESSION_BUS_ADDRESS"] = "tcp:host=untrusted"

            with self.assertRaisesRegex(RuntimeError, "environment"):
                self.dispatch(path, run=env_mutate)

    def test_bad_wrapper_hash_and_self_dispatch_refuse_before_runpy(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp, patch.dict(os.environ, {}, clear=True):
            path = Path(temp) / "fixture.py"
            path.write_text("fixture=1\n")
            with patch.object(b.runpy, "run_path") as run:
                with self.assertRaisesRegex(RuntimeError, "Selected source"):
                    self.dispatch(path, wrapper_sha="0" * 64)
                with self.assertRaisesRegex(RuntimeError, "itself"):
                    self.dispatch(HERE / "bootstrap.py")
                run.assert_not_called()

    def test_argv_restored_and_explicit_separator_required(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp, patch.dict(os.environ, {}, clear=True):
            path = Path(temp) / "fixture.py"
            path.write_text("fixture=1\n")
            previous = sys.argv
            self.dispatch(path)
            self.assertIs(sys.argv, previous)
            with self.assertRaisesRegex(RuntimeError, "separator"):
                b.main(
                    [
                        "--run",
                        "--entry",
                        str(path),
                        "--entry-sha256",
                        sha(path),
                        "--expected-wrapper-sha256",
                        sha(HERE / "bootstrap.py"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
