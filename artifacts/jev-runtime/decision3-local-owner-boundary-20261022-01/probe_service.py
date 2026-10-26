"""Tiny, timed cgroup-inheritance fixture. No network, model, or other-unit calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import time
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary selected path",
    )
    return path


def publish(directory, name, row):
    pending, target = ordinary(directory / (".pending-" + name)), ordinary(directory / name)
    fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(json.dumps(row, allow_nan=False, indent=2).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, target)
    pending.unlink()


def identity():
    pid = os.getpid()
    raw = Path(f"/proc/{pid}/stat").read_text()
    fields = raw[raw.rfind(")") + 2 :].split()
    groups = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
    require(
        len(fields) >= 20 and len(groups) == 1 and groups[0].startswith("0::/"),
        "Actual unified process identity",
    )
    return {
        "pid": pid,
        "ppid": os.getppid(),
        "start_ticks": int(fields[19]),
        "pgid": os.getpgrp(),
        "sid": os.getsid(0),
        "cgroup": groups[0][3:],
        "environment_key_count": len(os.environ),
        "initial_exec_environment_bytes": len(Path(f"/proc/{pid}/environ").read_bytes()),
    }


def load_plan(path, digest):
    require(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest), "Selected digest")
    blob = ordinary(path).read_bytes()
    require(
        len(blob) <= 65536 and hashlib.sha256(blob).hexdigest() == digest,
        "Exact bounded selected plan",
    )
    plan = json.loads(blob)
    require(
        plan["unit"] == "jev-d3-boundary-probe-20261022-01.service"
        and plan["memory_max_bytes"] == 64 * 1024**2
        and plan["runtime_seconds"] == 30
        and plan["stop_seconds"] == 2,
        "Only authorized tiny fixture",
    )
    require(
        Path("/proc/sys/kernel/random/boot_id").read_text().strip() == plan["boot_id"],
        "Same selected boot",
    )
    require(ordinary(plan["output"]).is_dir(), "Owned output exists")
    require(re.fullmatch(r"jev-d3-boundary-[0-9a-f]{32}", plan["nonce"]), "Selected nonce")
    return plan


def run(plan, digest, plan_path, *, grandchild=False):
    # A Python-level timeout complements the independent systemd RuntimeMaxSec.
    signal.alarm(25)
    if os.getsid(0) != os.getpid():
        os.setsid()
    row = identity()
    require(row["pid"] == row["pgid"] == row["sid"], "Own fixture session")
    require(row["cgroup"] == plan["cgroup"], "Fixture in selected actual cgroup")
    envelope = {
        "plan_sha256": digest,
        "unit": plan["unit"],
        "nonce": plan["nonce"],
        "boot_id": plan["boot_id"],
        "actor": row,
    }
    output = Path(plan["output"])
    if grandchild:
        # CPython may add LC_CTYPE while initializing its locale; /proc exposes
        # the original exec environment, which is the actual cleared-marker test.
        require(row["initial_exec_environment_bytes"] == 0, "Exec environment really cleared")
        publish(output, "grandchild-intent.json", envelope)
    else:
        require(
            os.environ.get("JEV_D3_BOUNDARY_NONCE") == plan["nonce"],
            "Explicit owner launch environment",
        )
        publish(output, "actor-intent.json", envelope)
        # Do not create the detached cohort until the external controller has
        # read back the independent kernel/unit limits and authenticated stop.
        deadline = time.monotonic() + 6
        admission_path = output / "admission.json"
        while not admission_path.is_file():
            require(time.monotonic() < deadline, "Timed external readback admission")
            time.sleep(0.02)
        admission = admission_path.read_bytes()
        require(
            len(admission) <= 4096
            and json.loads(admission)
            == {"plan_sha256": digest, "nonce": plan["nonce"], "readback_verified": True},
            "External selected unit readback admission",
        )
        child = os.fork()
        if child == 0:
            os.setsid()
            grandchild_pid = os.fork()
            if grandchild_pid:
                os._exit(0)
            os.execve(
                plan["interpreter"],
                [
                    plan["interpreter"],
                    "-I",
                    "-B",
                    str(Path(__file__).absolute()),
                    "--run",
                    "--grandchild",
                    "--plan",
                    str(plan_path),
                    "--plan-sha256",
                    digest,
                ],
                {},
            )
        waited, status = os.waitpid(child, 0)
        require(
            waited == child and os.waitstatus_to_exitcode(status) == 0,
            "Intermediate double-fork child reaped",
        )
        publish(output, "intermediate-reaped.json", {"pid": child, "exit_code": 0})
    stopped = False

    def stop(_signal, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    while not stopped:
        time.sleep(0.02)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--grandchild", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args(argv)
    if not args.run:
        print("DRY_NO_ACTOR; explicit sealed owner launch required")
        return 0
    require(args.plan is not None, "Explicit sealed plan")
    plan = load_plan(args.plan, args.plan_sha256)
    return run(plan, args.plan_sha256, args.plan, grandchild=args.grandchild)


if __name__ == "__main__":
    raise SystemExit(main())
