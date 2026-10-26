"""Stdlib worker-side authority; production checks require its live sibling parent.

No owner, process, socket or payload operation occurs at import time.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import time
from pathlib import Path

PROFILE = "llama-cpp-cpu4-3g-long-command-owned-v1"
UNIT = "jev-d3-llama-cpp-cpu4-3g-long-20261023-05.service"
CAP = 3 * 1024**3
SECONDS = 20.0
MAX_MESSAGE = 65536


def need(value, message):
    if not value:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def read_small(path, maximum=1024 * 1024):
    path = Path(path)
    need(path.is_absolute() and path.is_file() and not path.is_symlink(), "Ordinary metadata")
    need(path.stat().st_size <= maximum, "Bounded metadata")
    with path.open("rb") as stream:
        blob = stream.read(maximum + 1)
    need(len(blob) <= maximum, "Bounded actual metadata read")
    return blob


def identity(pid):
    need(type(pid) is int and pid > 0, "Actual PID")
    base = Path("/proc") / str(pid)
    first = read_small(base / "stat", 16384).decode()
    fields = first[first.rfind(")") + 2 :].split()
    groups = read_small(base / "cgroup", 16384).decode().splitlines()
    need(
        len(fields) >= 20 and len(groups) == 1 and groups[0].startswith("0::/"), "Unified identity"
    )
    result = {
        "pid": pid,
        "start_ticks": int(fields[19]),
        "ppid": int(fields[1]),
        "uid": base.stat().st_uid,
        "cgroup": groups[0][3:],
        "state": fields[0],
    }
    final = read_small(base / "stat", 16384).decode()
    after = final[final.rfind(")") + 2 :].split()
    need(int(after[19]) == result["start_ticks"] and int(after[1]) == result["ppid"], "Stable PID")
    return result


def same_identity(expected, observed):
    return all(observed.get(k) == expected.get(k) for k in ("pid", "start_ticks", "uid", "cgroup"))


def check_ticket(ticket, worker_pid, *, inspect=identity, boot=None, parent_pid=None):
    need(
        ticket.get("schema") == "owned-helper-smoke-ticket-v1"
        and ticket.get("profile") == PROFILE
        and ticket.get("unit") == UNIT
        and ticket.get("seconds") == SECONDS,
        "Selected owner ticket",
    )
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip() if boot is None else boot
    parent_pid = os.getppid() if parent_pid is None else parent_pid
    need(
        ticket["boot_id"] == boot
        and ticket["worker"]["pid"] == worker_pid
        and ticket["parent"]["pid"] == parent_pid,
        "Actual parent/worker/boot binding",
    )
    for key in ("worker", "parent", "unit_main"):
        actual = inspect(ticket[key]["pid"])
        need(
            same_identity(ticket[key], actual) and actual["state"] not in {"Z", "X"},
            "Live bound " + key,
        )
    need(
        ticket["worker"]["cgroup"] == ticket["parent"]["cgroup"] == ticket["unit_main"]["cgroup"]
        and ticket["worker"]["cgroup"].endswith("/" + UNIT)
        and ticket["worker"]["uid"] == ticket["parent"]["uid"] == ticket["unit_main"]["uid"],
        "Same original owner cgroup/UID",
    )
    need(
        type(ticket.get("control_fd")) is int
        and ticket["control_fd"] >= 3
        and type(ticket.get("secret")) is str
        and len(ticket["secret"]) == 64,
        "Inherited parent control channel",
    )


class Channel:
    def __init__(self, sock):
        self.sock, self.pending = sock, bytearray()

    def send(self, value):
        blob = encoded(value)
        need(len(blob) <= MAX_MESSAGE, "Bounded control frame")
        self.sock.sendall(blob)

    def receive(self, timeout):
        need(
            type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0,
            "Control timeout",
        )
        self.sock.settimeout(timeout)
        while b"\n" not in self.pending:
            block = self.sock.recv(4096)
            if not block:
                raise EOFError("External parent channel closed")
            self.pending.extend(block)
            need(len(self.pending) <= MAX_MESSAGE, "Bounded control input")
        line, _, rest = self.pending.partition(b"\n")
        self.pending = bytearray(rest)
        return json.loads(line)


class Authority:
    def __init__(self, ticket, ticket_sha, channel, clock=time.monotonic):
        self.ticket, self.ticket_sha, self.channel, self.clock = ticket, ticket_sha, channel, clock
        self.sequence, self.deadline = 0, None

    def _call(self, operation, **fields):
        check_ticket(self.ticket, os.getpid())
        timeout = 2.0 if self.deadline is None else min(2.0, self.deadline - self.clock())
        need(timeout > 0, "Request deadline reached before parent authorization")
        self.sequence += 1
        self.channel.send(
            {
                "operation": operation,
                "sequence": self.sequence,
                "secret": self.ticket["secret"],
                "ticket_sha256": self.ticket_sha,
                **fields,
            }
        )
        reply = self.channel.receive(timeout)
        need(
            reply == {"ok": True, "sequence": self.sequence},
            "Live external parent authorization refused",
        )
        if self.deadline is not None:
            need(self.clock() < self.deadline, "Parent reply completed late")

    def check(self, phase, **fields):
        self._call("check", phase=phase, **fields)

    def arm(self, *, request_id, started, deadline):
        need(
            self.deadline is None
            and type(started) in (int, float)
            and math.isfinite(started)
            and deadline == started + SECONDS
            and started <= self.clock() < deadline,
            "One original whole-request deadline",
        )
        self.deadline = deadline
        self._call("arm", request_id=request_id, started=started, deadline=deadline)


def verify_owner(*, ticket, expected_ticket_sha256, expected_profile, worker_pid):
    need(expected_profile == PROFILE and worker_pid == os.getpid(), "Selected actual worker")
    blob = read_small(ticket)
    need(hashlib.sha256(blob).hexdigest() == expected_ticket_sha256, "Immutable ticket SHA")
    value = json.loads(blob)
    check_ticket(value, worker_pid)
    channel = Channel(socket.socket(fileno=os.dup(value["control_fd"])))
    authority = Authority(value, expected_ticket_sha256, channel)
    authority.check("external-verifier-attached")
    return authority
