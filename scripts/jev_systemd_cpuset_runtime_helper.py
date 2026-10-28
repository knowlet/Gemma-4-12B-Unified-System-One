"""Held, narrowly scoped helper for a proposed temporary systemd drop-in.

The real target and both systemctl commands are fixed. The source-only gate
prevents execution pending separate review and explicit host-change approval.
A daemon-reload reruns system-wide generators; removing this drop-in and
reloading cannot guarantee reversal of already-applied cgroup changes. This
helper never restarts services, writes AllowedCPUs or starts a model.
Inode checks and unlink are not atomic; hard termination can leave files behind.
Protect the target directory against concurrent privileged changes and retain
the receipt externally; no crash-safe receipt storage is provided here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

SOURCE_ONLY = True
UNIT = "user@1001.service"
BASE = Path("/run/systemd/system")
DIRNAME = "user@1001.service.d"
FILENAME = "90-jev-cpuset.conf"
TARGET = str(BASE / DIRNAME / FILENAME)
FRAGMENT = "/usr/lib/systemd/system/user@.service"
BASELINE_DROPINS = frozenset(
    {
        "/usr/lib/systemd/system/user@.service.d/10-login-barrier.conf",
        "/usr/lib/systemd/system/user@.service.d/timeout.conf",
    }
)
OLD_CONTROLLERS = frozenset({"cpu", "memory", "pids"})
NEW_CONTROLLERS = OLD_CONTROLLERS | {"cpuset"}
CONTENT = b"[Service]\nDelegate=\nDelegate=cpu memory pids cpuset\n"
CONTENT_SHA = hashlib.sha256(CONTENT).hexdigest()
PROPERTIES = (
    "LoadState",
    "ActiveState",
    "InvocationID",
    "MainPID",
    "ControlGroup",
    "Delegate",
    "DelegateControllers",
    "FragmentPath",
    "DropInPaths",
    "NeedDaemonReload",
)


class OperationFailed(RuntimeError):
    """A refusal or failure with bounded, non-secret operation evidence."""

    def __init__(self, record):
        super().__init__("cpuset drop-in operation did not complete")
        self.record = record


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _pair(info):
    return {"device": info.st_dev, "inode": info.st_ino}


def _safe_directory(info):
    return stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and info.st_mode & 0o022 == 0


def _checked_base():
    _need(
        BASE.is_absolute() and BASE.resolve() == BASE, "ordinary fixed systemd directory required"
    )
    fd = os.open(BASE, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _need(_safe_directory(os.fstat(fd)), "owned non-writable systemd directory required")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _drop_directory(base_fd, *, create):
    created = False
    if create:
        try:
            os.mkdir(DIRNAME, mode=0o755, dir_fd=base_fd)
            created = True
        except FileExistsError:
            pass
    fd = os.open(DIRNAME, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=base_fd)
    try:
        _need(_safe_directory(os.fstat(fd)), "owned non-writable instance directory required")
        return fd, created, _pair(os.fstat(fd))
    except BaseException:
        os.close(fd)
        raise


def _parse_state(state, invocation, pid, *, dropins, controllers, reload_states=frozenset({"no"})):
    _need(type(state) is dict and set(state) == set(PROPERTIES), "exact unit properties required")
    _need(
        type(invocation) is str
        and re.fullmatch(r"[0-9a-f]{32}", invocation)
        and type(pid) is int
        and pid > 0,
        "explicit baseline identity required",
    )
    _need(
        state["LoadState"] == "loaded"
        and state["ActiveState"] == "active"
        and state["InvocationID"] == invocation
        and state["MainPID"] == str(pid)
        and state["ControlGroup"] == "/user.slice/user-1001.slice/user@1001.service"
        and state["Delegate"] == "yes"
        and state["FragmentPath"] == FRAGMENT
        and state["NeedDaemonReload"] in reload_states,
        "unit baseline identity changed",
    )
    paths = state["DropInPaths"].split()
    _need(
        len(paths) == len(set(paths)) and set(paths) == set(dropins),
        "unit drop-in baseline changed",
    )
    values = state["DelegateControllers"].split()
    _need(
        len(values) == len(set(values)) and frozenset(values) in controllers,
        "unit delegated controllers changed",
    )
    return frozenset(values)


class SystemCtl:
    """Fixed commands only; caller must supply an already-authorized execution context."""

    def _call(self, arguments):
        return subprocess.run(
            ["/usr/bin/systemctl", *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            env={
                "PATH": "/usr/bin:/bin",
                "LANG": "C",
                "SYSTEMD_PAGER": "cat",
                "SYSTEMD_COLORS": "0",
            },
        ).stdout

    def show(self):
        output = self._call(["show", UNIT, *["--property=" + name for name in PROPERTIES]])
        state = {}
        for line in output.splitlines():
            name, separator, value = line.partition("=")
            _need(separator and name in PROPERTIES and name not in state, "invalid unit properties")
            state[name] = value
        return state

    def reload(self):
        self._call(["daemon-reload"])


def _open_owned_file(directory, expected, *, allow_partial=False):
    fd = os.open(FILENAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    try:
        before = os.fstat(fd)
        _need(
            stat.S_ISREG(before.st_mode)
            and before.st_uid == os.geteuid()
            and before.st_nlink == 1
            and before.st_mode & 0o022 == 0
            and _pair(before) == expected
            and (
                0 <= before.st_size <= len(CONTENT)
                if allow_partial
                else before.st_size == len(CONTENT)
            ),
            "owned unchanged drop-in required",
        )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(len(CONTENT) + 1)
        after = os.fstat(fd)
        current = os.stat(FILENAME, dir_fd=directory, follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
        _need(
            all(
                tuple(getattr(item, key) for key in fields)
                == tuple(getattr(before, key) for key in fields)
                for item in (after, current)
            )
            and (CONTENT.startswith(data) if allow_partial else data == CONTENT),
            "owned unchanged drop-in required",
        )
    finally:
        os.close(fd)


def _remove_owned(base_fd, directory, file_identity, directory_identity, *, allow_partial=False):
    _need(_pair(os.fstat(directory)) == directory_identity, "instance directory changed")
    current_dir = os.stat(DIRNAME, dir_fd=base_fd, follow_symlinks=False)
    _need(
        _pair(current_dir) == directory_identity and _safe_directory(current_dir),
        "instance directory changed",
    )
    _open_owned_file(directory, file_identity, allow_partial=allow_partial)
    os.unlink(FILENAME, dir_fd=directory)


def _remove_created_directory(base_fd, directory_identity):
    current_dir = os.stat(DIRNAME, dir_fd=base_fd, follow_symlinks=False)
    _need(
        _pair(current_dir) == directory_identity and _safe_directory(current_dir),
        "instance directory changed",
    )
    os.rmdir(DIRNAME, dir_fd=base_fd)  # only the original, still-empty directory
    os.fsync(base_fd)


def _receipt(invocation, pid, file_identity, directory_identity, created_directory):
    return {
        "schema": "jev-cpuset-helper-receipt-v1",
        "unit": UNIT,
        "target": TARGET,
        "content_sha256": CONTENT_SHA,
        "invocation": invocation,
        "pid": pid,
        "file_identity": file_identity,
        "directory_identity": directory_identity,
        "created_directory": created_directory,
    }


def install(invocation, pid, *, backend):
    """Install only the proposed drop-in; caller persists the returned rollback receipt."""
    _need(SOURCE_ONLY is False, "reviewed cpuset helper release required")
    record = {
        "schema": "jev-cpuset-helper-operation-v1",
        "operation": "install",
        "status": "failed",
        "daemon_reload_scope": "system-wide generators and dependency tree",
        "rollback_limit": "removal and reload cannot guarantee reversal of cgroup effects",
        "events": [],
    }
    base_fd = directory = None
    created_directory = False
    dir_identity = file_identity = None
    reload_attempted = False
    cleanup_reload_attempted = False
    write_complete = False
    directory_attempted = False
    file_opened = False
    try:
        before = backend.show()
        _parse_state(
            before, invocation, pid, dropins=BASELINE_DROPINS, controllers={OLD_CONTROLLERS}
        )
        record["events"].append("baseline_verified")
        base_fd = _checked_base()
        directory_attempted = True
        directory, created_directory, dir_identity = _drop_directory(base_fd, create=True)
        try:
            fd = os.open(
                FILENAME,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o644,
                dir_fd=directory,
            )
        except FileExistsError:
            raise FileExistsError("target drop-in already exists") from None
        file_opened = True
        try:
            info = os.fstat(fd)
            _need(
                stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and info.st_nlink == 1,
                "new regular drop-in required",
            )
            file_identity = _pair(info)
            offset = 0
            while offset < len(CONTENT):
                written = os.write(fd, CONTENT[offset:])
                _need(written > 0, "drop-in write made no progress")
                offset += written
            os.fsync(fd)
            write_complete = True
        finally:
            os.close(fd)
        os.fsync(directory)
        _open_owned_file(directory, file_identity)
        # Catch identity/controller/drop-in changes occurring while file was created.
        _parse_state(
            backend.show(),
            invocation,
            pid,
            dropins=BASELINE_DROPINS,
            controllers={OLD_CONTROLLERS},
            reload_states={"no", "yes"},
        )
        record["events"].append("dropin_created")
        reload_attempted = True
        backend.reload()
        record["events"].append("daemon_reload_returned")
        after = backend.show()
        controllers = _parse_state(
            after,
            invocation,
            pid,
            dropins=BASELINE_DROPINS | {TARGET},
            controllers={OLD_CONTROLLERS, NEW_CONTROLLERS},
        )
        _open_owned_file(directory, file_identity)
        record.update(
            status="configured_effect_unverified",
            observed_controllers=sorted(controllers),
            receipt=_receipt(invocation, pid, file_identity, dir_identity, created_directory),
        )
        return record
    except BaseException as error:
        record["error_type"] = type(error).__name__
        if file_identity is not None and directory is not None and base_fd is not None:
            try:
                _remove_owned(
                    base_fd,
                    directory,
                    file_identity,
                    dir_identity,
                    allow_partial=not write_complete,
                )
                record["events"].append("owned_dropin_removed")
                if reload_attempted:
                    cleanup_reload_attempted = True
                    backend.reload()
                    record["events"].append("cleanup_reload_returned")
                os.fsync(directory)
                if created_directory:
                    _remove_created_directory(base_fd, dir_identity)
                record["cleanup"] = "dropin_removed_cgroup_effects_unverified"
            except BaseException as cleanup_error:
                record["cleanup"] = "unconfirmed"
                record["cleanup_error_type"] = type(cleanup_error).__name__
        elif created_directory and base_fd is not None and dir_identity is not None:
            try:
                _remove_created_directory(base_fd, dir_identity)
                record["cleanup"] = "empty_created_directory_removed"
            except BaseException as cleanup_error:
                record["cleanup"] = "unconfirmed"
                record["cleanup_error_type"] = type(cleanup_error).__name__
        else:
            record["cleanup"] = (
                "unconfirmed"
                if file_opened or (directory_attempted and directory is None)
                else "no_owned_dropin"
            )
        record["reload_attempted"] = reload_attempted
        record["cleanup_reload_attempted"] = cleanup_reload_attempted
        if reload_attempted:
            record["recovery_receipt"] = _receipt(
                invocation, pid, file_identity, dir_identity, created_directory
            )
        raise OperationFailed(record) from error
    finally:
        if directory is not None:
            os.close(directory)
        if base_fd is not None:
            os.close(base_fd)


def rollback(receipt, *, backend):
    """Remove only an intact receipt-bound file; reload does not prove full reversal."""
    _need(SOURCE_ONLY is False, "reviewed cpuset helper release required")
    record = {
        "schema": "jev-cpuset-helper-operation-v1",
        "operation": "rollback",
        "status": "failed",
        "daemon_reload_scope": "system-wide generators and dependency tree",
        "rollback_limit": "removal and reload cannot guarantee reversal of cgroup effects",
        "events": [],
    }
    base_fd = directory = None
    try:
        _need(
            type(receipt) is dict
            and set(receipt)
            == {
                "schema",
                "unit",
                "target",
                "content_sha256",
                "invocation",
                "pid",
                "file_identity",
                "directory_identity",
                "created_directory",
            }
            and receipt["schema"] == "jev-cpuset-helper-receipt-v1"
            and receipt["unit"] == UNIT
            and receipt["target"] == TARGET
            and receipt["content_sha256"] == CONTENT_SHA
            and type(receipt["created_directory"]) is bool
            and all(
                type(receipt[key]) is dict
                and set(receipt[key]) == {"device", "inode"}
                and all(type(value) is int and value >= 0 for value in receipt[key].values())
                for key in ("file_identity", "directory_identity")
            ),
            "exact rollback receipt required",
        )
        base_fd = _checked_base()
        directory, _created, actual = _drop_directory(base_fd, create=False)
        _need(actual == receipt["directory_identity"], "instance directory differs from receipt")
        current_dir = os.stat(DIRNAME, dir_fd=base_fd, follow_symlinks=False)
        _need(_pair(current_dir) == actual and _safe_directory(current_dir), "directory changed")
        try:
            os.stat(FILENAME, dir_fd=directory, follow_symlinks=False)
            file_exists = True
        except FileNotFoundError:
            file_exists = False
        state = backend.show()
        paths = set(state.get("DropInPaths", "").split())
        permitted = [BASELINE_DROPINS | {TARGET}]
        if not file_exists:
            permitted.append(BASELINE_DROPINS)
        _need(paths in permitted, "unit drop-in baseline changed")
        _parse_state(
            state,
            receipt["invocation"],
            receipt["pid"],
            dropins=paths,
            controllers={OLD_CONTROLLERS, NEW_CONTROLLERS},
            reload_states={"no"} if file_exists else {"no", "yes"},
        )
        if file_exists:
            _remove_owned(base_fd, directory, receipt["file_identity"], actual)
            record["events"].append("owned_dropin_removed")
        else:
            record["events"].append("dropin_absent_reload_retry")
        backend.reload()
        record["events"].append("daemon_reload_returned")
        os.fsync(directory)
        restored = backend.show()
        _parse_state(
            restored,
            receipt["invocation"],
            receipt["pid"],
            dropins=BASELINE_DROPINS,
            controllers={OLD_CONTROLLERS, NEW_CONTROLLERS},
        )
        if receipt["created_directory"]:
            _remove_created_directory(base_fd, actual)
        record.update(
            status="dropin_removed_cgroup_effects_unverified",
            observed_controllers=sorted(restored["DelegateControllers"].split()),
        )
        return record
    except BaseException as error:
        record.update(error_type=type(error).__name__, cleanup="unconfirmed")
        raise OperationFailed(record) from error
    finally:
        if directory is not None:
            os.close(directory)
        if base_fd is not None:
            os.close(base_fd)


def main():
    _need(SOURCE_ONLY is False, "reviewed cpuset helper release required")
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("operation", choices=("install", "rollback"))
    parser.add_argument("--invocation")
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    try:
        if args.operation == "install":
            result = install(args.invocation, args.pid, backend=SystemCtl())
        else:
            _need(args.invocation is None and args.pid is None, "rollback uses receipt only")
            raw = sys.stdin.buffer.read(65537)
            _need(len(raw) <= 65536, "bounded rollback receipt required")
            result = rollback(json.loads(raw), backend=SystemCtl())
    except OperationFailed as error:
        print(json.dumps(error.record, separators=(",", ":")))
        raise SystemExit(1) from error
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
