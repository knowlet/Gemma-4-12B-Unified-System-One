"""Read selected systemd user-unit properties without changing the caller env.

This stdlib library has no CLI or import effects. An owned caller supplies a
deadline and separately verifies the returned unit/invocation/PID properties.
Only the explicitly selected current user's canonical local bus is accepted.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

PROPERTIES = (
    "Description",
    "InvocationID",
    "MainPID",
    "RuntimeMaxUSec",
    "TimeoutStopUSec",
)


def _need(value, message):
    if not value:
        raise RuntimeError(message)


def validate_selection(selection):
    """Return a new, four-key child env; never inherit a bus or ML setting."""
    _need(
        type(selection) is dict
        and set(selection)
        == {
            "uid",
            "xdg_runtime_dir",
            "dbus_session_bus_address",
        },
        "exact selected same-user bus metadata required",
    )
    uid = selection["uid"]
    _need(
        type(uid) is int and uid >= 0 and uid == os.getuid() == os.geteuid(),
        "selected bus UID must equal real and effective user",
    )
    directory = f"/run/user/{uid}"
    address = f"unix:path={directory}/bus"
    _need(
        type(selection["xdg_runtime_dir"]) is str
        and selection["xdg_runtime_dir"] == directory
        and type(selection["dbus_session_bus_address"]) is str
        and selection["dbus_session_bus_address"] == address,
        "only the selected canonical local user bus is allowed",
    )
    return {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "XDG_RUNTIME_DIR": directory,
        "DBUS_SESSION_BUS_ADDRESS": address,
    }


def _snapshot(selection):
    uid = selection["uid"]
    directory = Path(selection["xdg_runtime_dir"])
    bus = directory / "bus"
    records = {}
    for path in (Path("/"), Path("/run"), Path("/run/user"), directory, bus):
        observed = os.lstat(path)
        _need(not stat.S_ISLNK(observed.st_mode), "linked user-bus path refused")
        _need(
            stat.S_ISSOCK(observed.st_mode) if path == bus else stat.S_ISDIR(observed.st_mode),
            "ordinary user-bus directory/socket geometry required",
        )
        records[str(path)] = observed
    own_directory, own_bus = records[str(directory)], records[str(bus)]
    _need(
        own_directory.st_uid == own_bus.st_uid == uid
        and stat.S_IMODE(own_directory.st_mode) == 0o700,
        "same-user private0700 directory and owned Unix socket required",
    )
    # Directory times change when other legitimate runtime entries change.
    # Identity, ownership and permissions must remain stable during the query.
    fields = ("st_dev", "st_ino", "st_uid", "st_mode")
    return {name: {key: getattr(record, key) for key in fields} for name, record in records.items()}


def _note(primary, phase, secondary):
    try:
        if primary is not secondary:
            primary.add_note(phase + " also failed: " + repr(secondary))
    except BaseException:
        pass


def read_user_unit_properties(unit, selection, *, deadline):
    """Query only the five fixed properties through a selected local user bus.

    The subprocess's one-second timeout is additional to the owned caller's
    cooperative deadline. Transport/identity/deadline failure propagates;
    there is no retry, fallback, global environment update or unit mutation.
    """
    _need(callable(deadline), "owned caller deadline required")
    primary, before, properties = None, None, None
    try:
        deadline()
        _need(
            type(unit) is str
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,239}\.(?:service|target)", unit),
            "literal bounded service/target unit required",
        )
        environment = validate_selection(selection)
        before = _snapshot(selection)
        deadline()
        completed = subprocess.run(
            [
                "/usr/bin/systemctl",
                "--user",
                "show",
                unit,
                *("--property=" + name for name in PROPERTIES),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=1,
            env=environment,
        )
        _need(
            type(completed.stdout) is str and len(completed.stdout) <= 65_536,
            "bounded textual user-unit properties required",
        )
        properties = {}
        for line in completed.stdout.splitlines():
            key, separator, value = line.partition("=")
            _need(
                separator == "=" and key in PROPERTIES and key not in properties,
                "only distinct requested user-unit properties allowed",
            )
            properties[key] = value
    except BaseException as error:
        primary = error
    finally:
        if before is not None:
            try:
                validate_selection(selection)
                _need(
                    _snapshot(selection) == before,
                    "user-bus path identity/owner/mode changed during query",
                )
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    _note(primary, "user-bus identity postcheck", error)
        try:
            deadline()
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                _note(primary, "user-unit caller deadline", error)
    if primary is not None:
        raise primary
    return properties
