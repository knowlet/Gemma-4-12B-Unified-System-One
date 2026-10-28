"""Source-only design for one system-manager synthetic CUDA unit.

A separately reviewed release must bind the nonce, plan SHA and bootstrap SHA
below to one concrete plan. This module never imports workspace code as root.
Only fixed start/show/stop operations are exposed; there is no generic command.
Read-then-stop and source verification are not atomic. A failed or ambiguous
start retains a root-owned intent record, does not claim ownership, and never
stops an unbound unit. Receipts are retained for separate explicit cleanup.
The caller, interpreter, release bytes and privilege authorization are external
trust boundaries. This source has no approved privileged deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path

SOURCE_ONLY = True
# Bind all three in a separately reviewed one-run release; not runtime arguments.
RUN_NONCE = None
PLAN_SHA256 = None
BOOTSTRAP_SHA256 = None
ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
RUN_ROOT = ROOT / "artifacts/jev-runtime/decision3-cuda-synthetic-system"
RECEIPT_ROOT = Path("/run")
BOOTSTRAP = ROOT / "scripts/jev_decision3_cuda_synthetic_system_bootstrap_release.py"
PYTHON = "/home/albert_hsieh/ComfyUI/.venv/bin/python"
SYSTEMCTL = "/usr/bin/systemctl"
SYSTEMD_RUN = "/usr/bin/systemd-run"
ENV = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
PROPERTIES = (
    "InvocationID",
    "LoadState",
    "ControlGroup",
    "MainPID",
    "ActiveState",
    "Type",
    "User",
    "MemoryAccounting",
    "MemoryMax",
    "MemorySwapMax",
    "OOMPolicy",
    "KillMode",
    "RuntimeMaxUSec",
    "TimeoutStopUSec",
    "ExecMainPID",
    "ExecMainCode",
    "ExecMainStatus",
    "Result",
    "SubState",
    "MemoryHigh",
    "MemoryPeak",
    "MemorySwapPeak",
)


def need(value, message):
    if not value:
        raise ValueError(message)


def _identity():
    need(SOURCE_ONLY is False, "reviewed broker release required")
    need(os.geteuid() == 0, "privileged broker required")
    need(
        type(RUN_NONCE) is str and re.fullmatch(r"[0-9a-f]{32}", RUN_NONCE), "fixed nonce required"
    )
    for digest in (PLAN_SHA256, BOOTSTRAP_SHA256):
        need(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest), "fixed digest required")
    return (
        "jev-cuda-synthetic-system-" + RUN_NONCE + ".service",
        RUN_ROOT / ("run-" + RUN_NONCE),
        RECEIPT_ROOT / ("jev-cuda-synthetic-system-" + RUN_NONCE),
    )


def _read(path, limit, expected=None):
    need(path.is_absolute() and path.resolve() == path, "ordinary fixed path required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        first = os.fstat(fd)
        need(
            stat.S_ISREG(first.st_mode) and first.st_nlink == 1 and first.st_size <= limit,
            "bounded ordinary file required",
        )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        last, current = os.fstat(fd), os.stat(path, follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
        need(
            tuple(getattr(first, k) for k in fields)
            == tuple(getattr(last, k) for k in fields)
            == tuple(getattr(current, k) for k in fields)
            and len(raw) == first.st_size,
            "file changed",
        )
        if expected is not None:
            need(hashlib.sha256(raw).hexdigest() == expected, "fixed digest mismatch")
        return raw
    finally:
        os.close(fd)


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        raw, object_pairs_hook=pairs, parse_constant=lambda _: need(False, "nonfinite JSON")
    )


def _plan(unit, run):
    plan_path = run / "plan.json"
    plan = _json(_read(plan_path, 1024 * 1024, PLAN_SHA256))
    need(
        type(plan) is dict
        and set(plan)
        == {
            "schema",
            "unit",
            "output_dir",
            "python",
            "source_pins",
            "gpu_uuid",
            "wall_seconds",
            "memory_floor_bytes",
        },
        "fixed plan fields required",
    )
    need(
        plan["schema"] == "decision3-cuda-synthetic-system-plan-v1"
        and plan["unit"] == unit
        and plan["output_dir"] == str(run / "worker")
        and plan["python"] == PYTHON
        and type(plan["wall_seconds"]) is int
        and plan["wall_seconds"] == 120
        and type(plan["memory_floor_bytes"]) is int
        and plan["memory_floor_bytes"] >= 34709107508
        and type(plan["gpu_uuid"]) is str
        and re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", plan["gpu_uuid"]
        ),
        "fixed plan contract required",
    )
    need(
        type(plan["source_pins"]) is dict
        and plan["source_pins"].get(str(BOOTSTRAP)) == BOOTSTRAP_SHA256,
        "fixed bootstrap pin required",
    )
    _read(BOOTSTRAP, 8 * 1024 * 1024, BOOTSTRAP_SHA256)
    output = run / "worker"
    need(
        output.resolve() == output
        and output.is_dir()
        and output.stat().st_uid == 1001
        and output.stat().st_mode & 0o077 == 0
        and not any(output.iterdir()),
        "fresh private output required",
    )
    return plan_path


def _command(unit, plan_path):
    properties = {
        "Type": "exec",
        "User": "1001",
        "MemoryAccounting": "yes",
        "MemoryMax": str(8 * 1024**3),
        "MemorySwapMax": "0",
        "CPUQuota": "500%",
        "AllowedCPUs": "15-19",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxSec": "120",
        "TimeoutStopSec": "3s",
        "RemainAfterExit": "yes",
        "StandardOutput": "null",
        "StandardError": "null",
    }
    return [
        SYSTEMD_RUN,
        "--system",
        "--slice=system.slice",
        "--unit=" + unit,
        *["--property=" + k + "=" + v for k, v in properties.items()],
        "/usr/bin/env",
        "-u",
        "PYTHONPATH",
        "-u",
        "PYTHONHOME",
        "-u",
        "PYTHONSTARTUP",
        "/usr/bin/taskset",
        "-c",
        "15-19",
        PYTHON,
        "-I",
        "-B",
        "-s",
        str(BOOTSTRAP),
        "--mode",
        "launcher",
        "--plan",
        str(plan_path),
        "--plan-sha256",
        PLAN_SHA256,
    ]


def _call(args, *, timeout):
    result = subprocess.run(
        args, capture_output=True, check=True, timeout=timeout, env=ENV, cwd="/"
    )
    need(
        len(result.stdout) <= 1024 * 1024 and len(result.stderr) <= 1024 * 1024,
        "bounded system output required",
    )
    return result.stdout


def _show(unit):
    raw = _call(
        [SYSTEMCTL, "--system", "show", unit, *["--property=" + k for k in PROPERTIES]], timeout=2
    )
    rows = raw.decode("utf-8").splitlines()
    need(all("=" in row for row in rows), "invalid system metadata")
    pairs = [row.split("=", 1) for row in rows]
    data = dict(pairs)
    need(
        len(data) == len(pairs) and set(data) == set(PROPERTIES),
        "complete unique system metadata required",
    )
    return data


def _boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _process(pid):
    # Parse proc stat after its final ')' because comm can contain spaces.
    raw = (Path("/proc") / str(pid) / "stat").read_text()
    fields = raw[raw.rfind(")") + 2 :].split()
    need(len(fields) >= 20, "process start time unavailable")
    status = (Path("/proc") / str(pid) / "status").read_text().splitlines()
    uids = [line.split()[1:] for line in status if line.startswith("Uid:")]
    need(uids == [["1001"] * 4], "worker UID differs")
    cg = (Path("/proc") / str(pid) / "cgroup").read_text().splitlines()
    groups = [line[3:] for line in cg if line.startswith("0::")]
    need(len(groups) == 1, "unified process cgroup required")
    return {"startticks": int(fields[19]), "cgroup": groups[0]}


def _binding(unit, props):
    invocation, pid = props["InvocationID"], props["MainPID"]
    need(
        props["LoadState"] == "loaded"
        and props["ActiveState"] == "active"
        and props["Type"] == "exec"
        and props["User"] == "1001"
        and re.fullmatch(r"[0-9a-f]{32}", invocation)
        and re.fullmatch(r"[1-9][0-9]*", pid),
        "running system unit identity required",
    )
    group = "/system.slice/" + unit
    process = _process(int(pid))
    need(
        props["ControlGroup"] == group and process["cgroup"] == group,
        "system cgroup identity differs",
    )
    return {
        "unit": unit,
        "invocation": invocation,
        "pid": int(pid),
        "startticks": process["startticks"],
        "boot_id": _boot_id(),
        "cgroup": group,
        "plan_sha256": PLAN_SHA256,
    }


def _write_new(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)


def start():
    unit, run, receipt = _identity()
    plan_path = _plan(unit, run)
    old = _show(unit)
    need(
        old["LoadState"] == "not-found"
        and old["ActiveState"] == "inactive"
        and old["MainPID"] == "0",
        "fresh system unit required",
    )
    need(
        RECEIPT_ROOT.resolve() == RECEIPT_ROOT and RECEIPT_ROOT.is_dir(),
        "ordinary receipt root required",
    )
    receipt.mkdir(mode=0o700, exist_ok=False)
    _write_new(
        receipt / "intent.json",
        {
            "schema": "jev-system-broker-intent-v1",
            "unit": unit,
            "plan_sha256": PLAN_SHA256,
            "boot_id": _boot_id(),
        },
    )
    # On timeout or any other uncertainty, retain intent and do not claim unit ownership.
    _call(_command(unit, plan_path), timeout=10)
    binding = _binding(unit, _show(unit))
    _write_new(receipt / "binding.json", binding)
    return {"status": "started", **binding}


def show():
    unit, _run, _receipt = _identity()
    return _show(unit)


def _root_owned(info):
    return info.st_uid == 0 and info.st_mode & 0o077 == 0


def stop():
    unit, run, receipt = _identity()
    need(receipt.resolve() == receipt and receipt.is_dir(), "ordinary receipt required")
    info = receipt.stat()
    need(_root_owned(info), "root-owned private receipt required")
    binding_path = receipt / "binding.json"
    info = binding_path.lstat()
    need(
        stat.S_ISREG(info.st_mode) and _root_owned(info),
        "root-owned binding required",
    )
    binding = _json(_read(binding_path, 4096))
    current = _binding(unit, _show(unit))
    need(binding == current, "bound invocation changed")
    hello = _json(_read(run / "worker/hello.json", 4096))
    need(
        type(hello) is dict
        and hello.get("schema") == "decision3-cuda-synthetic-hello-v1"
        and all(
            hello.get(k) == binding[k]
            for k in ("unit", "invocation", "pid", "startticks", "boot_id", "cgroup", "plan_sha256")
        )
        and hello.get("pgid") == binding["pid"],
        "matching worker hello required",
    )
    # This recheck and stop are deliberately not represented as atomic.
    need(_binding(unit, _show(unit)) == binding, "bound invocation changed before stop")
    _call([SYSTEMCTL, "--system", "stop", unit], timeout=20)
    return {
        "status": "stop_returned_closure_unconfirmed",
        "unit": unit,
        "invocation": binding["invocation"],
    }


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("action", choices=("start", "show", "stop"))
    args = parser.parse_args()
    print(
        json.dumps(
            {"start": start, "show": show, "stop": stop}[args.action](), separators=(",", ":")
        )
    )


if __name__ == "__main__":
    main()
