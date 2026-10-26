"""Externally observe and close one selected D3 systemd unit; no model imports.

Project MIT owner pattern. Cgroup inheritance covers audited fork/setsid/exec;
same-UID migration is explicitly outside an enforced no-escape guarantee.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
CGROUP_ROOT, PROC = Path("/sys/fs/cgroup"), Path("/proc")
PROFILE = "decision3-external-unit-boundary-v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary nontraversing path required",
    )
    return path


def pin(path):
    path = ordinary(path)
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
            size += len(block)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": digest.hexdigest(),
        "size_bytes": size,
    }


def sources_match(refs):
    require(type(refs) is list and 1 <= len(refs) <= 64, "Bounded explicit source population")
    seen = set()
    for ref in refs:
        require(
            type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"},
            "Exact ordinary source schema",
        )
        path = ref["path"]
        require(
            type(path) is str
            and path
            and not Path(path).is_absolute()
            and ".." not in Path(path).parts
            and path not in seen,
            "Distinct repository-relative source",
        )
        require(
            type(ref["sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"])
            and type(ref["size_bytes"]) is int
            and ref["size_bytes"] >= 0,
            "Complete typed source identity",
        )
        seen.add(path)
    return all(pin(ROOT / ref["path"]) == ref for ref in refs)


def save(directory, name, value):
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]*\.json", name), "Bounded evidence filename")
    blob = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode() + b"\n"
    require(len(blob) <= 8 * 1024**2, "Bounded boundary evidence")
    target = ordinary(directory / name)
    pending = ordinary(directory / (".pending-" + name))
    fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    # Atomic visibility and exclusive final creation; readers never consume a
    # partially written admission or identity receipt. Interrupted files remain.
    os.link(pending, target)
    pending.unlink()


def boot():
    return (PROC / "sys/kernel/random/boot_id").read_text().strip()


def command(*argv, timeout=5):
    return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=timeout).stdout


def process(pid):
    path = PROC / str(pid)
    raw = (path / "stat").read_text()
    fields = raw[raw.rfind(")") + 2 :].split()
    require(len(fields) >= 20, "Complete process identity")
    groups = (path / "cgroup").read_text().splitlines()
    require(len(groups) == 1 and groups[0].startswith("0::/"), "Unified process cgroup")
    row = {
        "pid": pid,
        "start_ticks": int(fields[19]),
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "uid": path.stat().st_uid,
        "state": fields[0],
        "cgroup": groups[0][3:],
    }
    final = (path / "stat").read_text()
    after = final[final.rfind(")") + 2 :].split()
    require(
        len(after) >= 20
        and int(after[19]) == row["start_ticks"]
        and int(after[2]) == row["pgid"]
        and int(after[3]) == row["sid"],
        "Stable process identity across cgroup read",
    )
    return row


@dataclass(frozen=True)
class Selection:
    unit: str
    nonce: str
    boot_id: str
    memory_max_bytes: int
    runtime_seconds: int
    stop_seconds: int

    def __post_init__(self):
        require(
            type(self.unit) is str
            and re.fullmatch(r"jev-d3-[a-z0-9][a-z0-9-]{0,90}\.service", self.unit),
            "Explicit D3-only unit namespace",
        )
        require(
            type(self.nonce) is str and re.fullmatch(r"jev-d3-boundary-[0-9a-f]{32}", self.nonce),
            "Selected nonce",
        )
        require(type(self.boot_id) is str and bool(self.boot_id), "Selected boot")
        require(
            type(self.memory_max_bytes) is int and 1024**2 <= self.memory_max_bytes <= 40 * 1024**3,
            "Selected positive bounded memory cap",
        )
        require(
            self.runtime_seconds in (30, 7200, 10800) and type(self.runtime_seconds) is int,
            "Selected independent runtime",
        )
        require(
            type(self.stop_seconds) is int and 1 <= self.stop_seconds <= 20,
            "Bounded selected stop timeout",
        )

    @property
    def cgroup(self):
        uid = os.getuid()
        return f"/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{self.unit}"

    def properties(self):
        return {
            "Type": "exec",
            "Description": self.nonce,
            "MemoryAccounting": "yes",
            "MemoryMax": str(self.memory_max_bytes),
            "MemoryHigh": "infinity",
            "MemorySwapMax": "0",
            "OOMPolicy": "kill",
            "KillMode": "control-group",
            "RuntimeMaxUSec": {30: "30s", 7200: "2h", 10800: "3h"}[self.runtime_seconds],
            "TimeoutStopUSec": str(self.stop_seconds) + "s",
            "KillSignal": "15",
            "FinalKillSignal": "9",
            "SendSIGKILL": "yes",
            "RemainAfterExit": "yes",
            "Restart": "no",
            "Delegate": "no",
        }


def status(selection):
    keys = (
        *selection.properties(),
        "LoadState",
        "ActiveState",
        "SubState",
        "InvocationID",
        "ControlGroup",
        "MainPID",
        "ExecMainPID",
        "ExecMainCode",
        "ExecMainStatus",
        "Result",
    )
    text = command(
        "systemctl", "--user", "show", selection.unit, *["--property=" + k for k in keys]
    )
    return dict(line.split("=", 1) for line in text.splitlines())


def captured(callback):
    try:
        return callback()
    except BaseException as error:
        return {"observation_error": {"type": type(error).__name__, "message": str(error)}}


def migration_capabilities(selection):
    group = CGROUP_ROOT / selection.cgroup.lstrip("/")
    rows = []
    for path in (
        group / "cgroup.procs",
        group.parent / "cgroup.procs",
        group.parent.parent / "cgroup.procs",
    ):
        stat = path.stat()
        rows.append(
            {
                "path": str(path),
                "uid": stat.st_uid,
                "mode": oct(stat.st_mode & 0o777),
                "effective_writable": os.access(path, os.W_OK, effective_ids=True),
            }
        )
    return {
        "paths": rows,
        "migration_attempted": False,
        "same_uid_migration_may_be_possible": any(r["effective_writable"] for r in rows[1:]),
        "no_escape_enforced": False,
        "scope": "Permission observations only; delegated same-UID subtree and user manager remain accessible. No adversarial containment claim.",
    }


class UnitBoundary:
    """Kernel-derived unit/cohort facts; never accepts caller closure booleans."""

    def __init__(self, selection, output, source_refs):
        self.selection, self.output, self.sources = selection, ordinary(output), source_refs
        require(self.output.is_dir(), "Existing fresh owned evidence directory")
        require(
            not any(
                (self.output / name).exists()
                for name in ("unit-readback.json", "external-boundary-closure.json")
            ),
            "Never reuse existing boundary evidence",
        )
        require(sources_match(self.sources), "Exact selected boundary source pins")
        self.invocation, self.actor, self.readback = None, None, None
        self.tracked, self.closed = {}, None

    def owned_status(self):
        require(boot() == self.selection.boot_id, "Same selected boot")
        row = status(self.selection)
        require(row.get("Description") == self.selection.nonce, "Foreign unit nonce; no stop")
        invocation = row.get("InvocationID")
        require(
            type(invocation) is str and re.fullmatch(r"[0-9a-f]{32}", invocation),
            "Known owned invocation; no stop",
        )
        require(
            self.invocation is None or self.invocation == invocation,
            "Changed unit invocation; no stop",
        )
        require(row.get("ControlGroup") == self.selection.cgroup, "Exact selected cgroup")
        self.invocation = invocation
        return row

    def members(self):
        require(boot() == self.selection.boot_id, "Same boot for cgroup observation")
        group = CGROUP_ROOT / self.selection.cgroup.lstrip("/")
        require(group.is_dir(), "Realized selected cgroup")
        paths = [group / "cgroup.procs", *group.glob("**/cgroup.procs")]
        paths = sorted(set(paths))
        require(len(paths) <= 64, "Bounded owned cgroup hierarchy")
        members, vanished = {}, []
        for path in paths:
            require(not path.is_symlink(), "Kernel ordinary cgroup inventory")
            blob = path.read_bytes()
            require(len(blob) <= 65536, "Bounded cgroup process inventory")
            for value in blob.splitlines():
                pid = int(value)
                require(len(members) < 4096, "Bounded cgroup process count")
                try:
                    row = process(pid)
                except (FileNotFoundError, ProcessLookupError):
                    vanished.append(pid)
                    continue
                require(
                    row["uid"] == os.getuid()
                    and (
                        row["cgroup"] == self.selection.cgroup
                        or row["cgroup"].startswith(self.selection.cgroup + "/")
                    ),
                    "Fresh member identity/cgroup differs",
                )
                members[pid] = row
                self.tracked[(pid, row["start_ticks"])] = row
        return {"members": list(members.values()), "vanished": vanished}

    def verify(self):
        row = self.owned_status()
        require(
            all(row.get(k) == v for k, v in self.selection.properties().items()),
            "Actual unit properties differ",
        )
        group = CGROUP_ROOT / self.selection.cgroup.lstrip("/")
        limits = {
            k: (group / k).read_text().strip()
            for k in ("memory.max", "memory.swap.max", "cgroup.type")
        }
        require(
            limits
            == {
                "memory.max": str(self.selection.memory_max_bytes),
                "memory.swap.max": "0",
                "cgroup.type": "domain",
            },
            "Actual kernel memory/swap/domain readback differs",
        )
        actor = process(int(row["MainPID"]))
        require(
            actor["uid"] == os.getuid()
            and actor["cgroup"] == self.selection.cgroup
            and actor["pid"] == actor["pgid"] == actor["sid"],
            "Actual owned main session",
        )
        self.actor = actor
        cohort = self.members()
        require(
            any(
                r["pid"] == actor["pid"] and r["start_ticks"] == actor["start_ticks"]
                for r in cohort["members"]
            ),
            "Main actor in actual cgroup inventory",
        )
        readback = {
            "unit": row,
            "kernel": limits,
            "actor": actor,
            "cohort": cohort,
            "migration": migration_capabilities(self.selection),
        }
        save(self.output, "unit-readback.json", readback)
        self.readback = readback
        return self.readback

    def scan_absence(self):
        require(boot() == self.selection.boot_id, "Same boot for final process scan")
        matches, unknown = [], []
        for path in PROC.iterdir():
            if not path.name.isdecimal():
                continue
            try:
                row = process(int(path.name))
                exact = (row["pid"], row["start_ticks"]) in self.tracked
                grouped = self.actor and (
                    row["pgid"] == self.actor["pgid"] or row["sid"] == self.actor["sid"]
                )
                in_group = row["cgroup"] == self.selection.cgroup or row["cgroup"].startswith(
                    self.selection.cgroup + "/"
                )
                if exact or grouped or in_group:
                    matches.append(row)
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (PermissionError, OSError, ValueError, IndexError) as error:
                unknown.append({"pid": path.name, "type": type(error).__name__})
        return {"matches": matches, "unknown": unknown}

    def close(self, *, primary_error=None):
        if self.closed is not None:
            return self.closed
        before, stopped, stop = None, None, None
        errors = []
        try:
            before = self.owned_status()
            # Capture the last actual cgroup cohort before the authenticated stop.
            try:
                self.members()
            except BaseException as error:
                errors.append(
                    {"stage": "final_cohort", "type": type(error).__name__, "message": str(error)}
                )
            # Even a failed cohort read must not skip cleanup of the authenticated
            # unit. Re-authenticate immediately before the selected stop command.
            self.owned_status()
            stop = command(
                "systemctl",
                "--user",
                "stop",
                self.selection.unit,
                timeout=self.selection.stop_seconds + 5,
            )
            stopped = status(self.selection)
        except BaseException as error:
            errors.append(
                {"stage": "authenticated_stop", "type": type(error).__name__, "message": str(error)}
            )
        scans = []
        for _ in range(2):
            scans.append(captured(self.scan_absence))
            time.sleep(0.05)
        absent = not (CGROUP_ROOT / self.selection.cgroup.lstrip("/")).exists()
        source_check = captured(lambda: sources_match(self.sources))
        terminal = (
            type(stopped) is dict
            and stopped.get("MainPID") == "0"
            and (
                (
                    stopped.get("Description") == self.selection.nonce
                    and stopped.get("InvocationID") in (self.invocation, "")
                    and stopped.get("ActiveState") in ("inactive", "failed")
                )
                or (
                    stopped.get("LoadState") == "not-found"
                    and stopped.get("InvocationID") == ""
                    and stopped.get("ControlGroup") == ""
                )
            )
        )
        complete = bool(
            self.readback
            and not errors
            and type(stop) is str
            and terminal
            and absent
            and source_check is True
            and all(
                type(s) is dict and s.get("matches") == [] and s.get("unknown") == [] for s in scans
            )
        )
        result = {
            "profile": PROFILE,
            "selection": asdict(self.selection),
            "invocation_id": self.invocation,
            "actor": self.actor,
            "tracked_cohort": list(self.tracked.values()),
            "before_stop": before,
            "stop_result": stop,
            "after_stop": stopped,
            "primary_error": primary_error,
            "closure_errors": errors,
            "post_stop_terminal_verified": bool(terminal),
            "cgroup_absent": absent,
            "two_process_scans": scans,
            "sources_unchanged": source_check,
            "whole_unit_exit_confirmed": complete,
            "observed_cgroup_cohort_exit_confirmed": complete,
            "no_escape_enforced": False,
            "whole_tree_exit_confirmed": False,
            "continuation_authorized": False,
            "model_qualified": False,
            "scope": "Authenticated selected unit and observed cgroup cohort only. The audited fork/setsid/empty-env case is covered; explicit delegated cgroup migration or user-manager requests are not prevented.",
        }
        save(self.output, "external-boundary-closure.json", result)
        self.closed = result
        return self.closed
