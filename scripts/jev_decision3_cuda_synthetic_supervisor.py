"""Held independent supervisor for one synthetic CUDA Decision3 request.

The supervised unit has a 120-second systemd hard wall. The worker's separate
20-second request deadline is cooperative and cannot interrupt a stuck forward.
Host observations and scoped cleanup do not prove machine-wide GPU exclusivity.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import os
import re
import subprocess
import time
from pathlib import Path

SOURCE_ONLY = True
# Injected only after bootstrap authenticates and executes selected source bytes.
child = common = cpu = launcher = None


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _seconds(value):
    units = {"h": 3600, "min": 60, "s": 1, "ms": 0.001, "us": 0.000001}
    _need(type(value) is str and value, "unit runtime required")
    parts = re.findall(r"([0-9]+(?:\.[0-9]+)?)(h|min|ms|us|s)", value)
    _need(bool(parts) and " ".join(a + b for a, b in parts) == value, "invalid unit runtime")
    return sum(float(number) * units[unit] for number, unit in parts)


def _unit_args(plan, path, digest):
    properties = {
        "Type": "exec",
        "MemoryAccounting": "yes",
        "MemoryMax": str(common.MEMORY_MAX),
        "MemorySwapMax": "0",
        "CPUQuota": "500%",
        "AllowedCPUs": "15-19",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxSec": str(launcher.WALL_SECONDS),
        "TimeoutStopSec": "3s",
        "RemainAfterExit": "yes",
        "StandardOutput": "null",
        "StandardError": "null",
    }
    return [
        "systemd-run",
        "--user",
        "--slice=app.slice",
        "--unit=" + plan["unit"],
        *["--property=" + key + "=" + value for key, value in properties.items()],
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
        plan["python"],
        "-I",
        "-B",
        "-s",
        str(launcher._release("bootstrap")),
        "--mode",
        "launcher",
        "--plan",
        str(path),
        "--plan-sha256",
        digest,
    ]


def _unit_identity(properties, unit):
    invocation, group = properties["InvocationID"], properties["ControlGroup"]
    _need(
        properties["LoadState"] == "loaded"
        and type(invocation) is str
        and re.fullmatch(r"[0-9a-f]{32}", invocation)
        and type(group) is str
        and group.startswith("/")
        and ".." not in Path(group).parts
        and group.endswith("/app.slice/" + unit),
        "fresh owned unit identity required",
    )
    return invocation, group


def _bind(properties, hello, plan, plan_sha):
    invocation, group = _unit_identity(properties, plan["unit"])
    _need(
        type(hello) is dict
        and set(hello)
        == {
            "schema",
            "pid",
            "startticks",
            "boot_id",
            "unit",
            "invocation",
            "plan_sha256",
            "pgid",
            "cgroup",
        }
        and hello["schema"] == "decision3-cuda-synthetic-hello-v1"
        and hello["unit"] == plan["unit"]
        and hello["plan_sha256"] == plan_sha
        and hello["invocation"] == invocation
        and hello["cgroup"] == group
        and type(hello["pid"]) is int
        and hello["pid"] > 0
        and hello["pgid"] == hello["pid"]
        and str(hello["pid"]) == properties["MainPID"]
        and properties["ActiveState"] == "active"
        and hello["boot_id"] == common.systemd.boot_id()
        and hello["startticks"] == common.systemd.process_identity(hello["pid"])["startticks"]
        and common.systemd.cgroup_membership(hello["pid"]) == group
        and os.getpgid(hello["pid"]) == hello["pgid"],
        "exact worker identity required",
    )
    _need(
        properties["Type"] == "exec"
        and properties["MemoryAccounting"] == "yes"
        and properties["MemoryMax"] == str(common.MEMORY_MAX)
        and properties["MemorySwapMax"] == "0"
        and properties["OOMPolicy"] == "kill"
        and properties["KillMode"] == "control-group"
        and _seconds(properties["RuntimeMaxUSec"]) == launcher.WALL_SECONDS
        and _seconds(properties["TimeoutStopUSec"]) == 3,
        "exact unit limits required",
    )
    cpu._cpu_limits(group, hello["pid"], launcher.PROFILE)
    return invocation, group


def _terminal(output, invocation, plan_sha):
    value = common._decode(child._read_bounded(output / "terminal.json", 4096))
    _need(
        type(value) is dict
        and set(value) == {"schema", "status", "invocation", "plan_sha256", "result_sha256"}
        and value["schema"] == "decision3-cuda-synthetic-terminal-v1"
        and value["status"] == "completed_pending_external_audit"
        and value["invocation"] == invocation
        and value["plan_sha256"] == plan_sha
        and type(value["result_sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", value["result_sha256"]),
        "bound terminal required",
    )
    raw = child._read_bounded(output / "result.json", 4 * 1024 * 1024)
    _need(hashlib.sha256(raw).hexdigest() == value["result_sha256"], "result digest differs")
    result = common._decode(raw)
    _need(
        type(result) is dict
        and result.get("schema") == "decision3-cuda-synthetic-worker-v1"
        and result.get("status") == "completed_pending_external_audit"
        and result.get("device") == "cuda:0"
        and result.get("calls") == 1
        and result.get("score") is None
        and result.get("final_allocated_bytes") == 0,
        "complete synthetic worker result required",
    )
    return value


def _closed_group(pgid):
    _need(type(pgid) is int and pgid > 0, "positive process group required")
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return "absent"
    except OSError as error:
        if error.errno == errno.ESRCH:
            return "absent"
        raise
    raise ValueError("owned process group still exists or identity reused")


def _recover(output, plan, plan_sha):
    until = time.monotonic() + 8
    while time.monotonic() < until:
        try:
            properties = common.systemd.service_show(plan["unit"], timeout=2)
            hello = common._decode(child._read_bounded(output / "hello.json", 4096))
            invocation, group = _bind(properties, hello, plan, plan_sha)
            return invocation, group, hello
        except Exception:
            time.sleep(0.2)
    raise ValueError("owned unit identity unavailable for scoped stop") from None


def supervise(plan_path, plan_sha, report_dir):
    _need(SOURCE_ONLY is False, "reviewed synthetic CUDA supervisor release required")
    plan = launcher._plan(plan_path, plan_sha)
    output, report = Path(plan["output_dir"]), Path(report_dir)
    _need(
        report == Path(plan_path).parent / "supervisor"
        and not report.exists()
        and report.parent.is_dir()
        and report.parent.resolve() == report.parent,
        "fresh separate report required",
    )
    _need(
        output.is_dir()
        and output.resolve() == output
        and output.stat().st_mode & 0o077 == 0
        and not any(output.iterdir()),
        "fresh private output required",
    )
    report.mkdir(mode=0o700, exist_ok=False)
    events = report / "events.jsonl"
    common._write_new(events, b"")
    invocation = group = hello = None
    launched = stopped = bound = False
    failure = None
    started = time.monotonic()
    try:
        common._validate_interpreter(plan)
        launcher._source_guard(plan)
        common._append_event(
            events, {"event": "prelaunch", **common._observe(plan["memory_floor_bytes"])}
        )
        old = common.systemd.service_show(plan["unit"], timeout=2)
        _need(
            old["LoadState"] == "not-found"
            and old["ActiveState"] == "inactive"
            and old["MainPID"] == "0",
            "fresh absent unit required",
        )
        launched = True
        subprocess.run(
            _unit_args(plan, plan_path, plan_sha), check=True, capture_output=True, timeout=10
        )
        while True:
            _need(
                time.monotonic() - started < launcher.WALL_SECONDS, "supervisor hard wall exceeded"
            )
            common._append_event(
                events, {"event": "host", **common._observe(plan["memory_floor_bytes"])}
            )
            properties = common.systemd.service_show(plan["unit"], timeout=2)
            if invocation is None:
                invocation, group = _unit_identity(properties, plan["unit"])
            _need(
                properties["InvocationID"] == invocation and properties["LoadState"] == "loaded",
                "unit invocation changed",
            )
            if hello is None and (output / "hello.json").is_file():
                hello = common._decode(child._read_bounded(output / "hello.json", 4096))
                _bind(properties, hello, plan, plan_sha)
                bound = True
                common._append_event(
                    events,
                    {
                        "event": "binding_verified",
                        "invocation": invocation,
                        "pid": hello["pid"],
                        "pgid": hello["pgid"],
                        "cgroup": group,
                    },
                )
                child._publish_control(
                    output / "go.json",
                    child._encode(
                        {
                            "schema": "decision3-cuda-synthetic-go-v1",
                            "invocation": invocation,
                            "plan_sha256": plan_sha,
                        }
                    ),
                )
            elif hello is not None and properties["MainPID"] != "0":
                _bind(properties, hello, plan, plan_sha)
            terminal = common._terminal_metadata(properties, invocation)
            if terminal is not None:
                break
            time.sleep(0.5)
        _need(
            hello is not None and common._successful_exit(terminal),
            "worker did not exit successfully",
        )
        _terminal(output, invocation, plan_sha)
        launcher._source_guard(plan)
        launcher._plan(plan_path, plan_sha)
    except BaseException as error:
        failure = error
        if launched:
            try:
                if not bound:
                    invocation, group, hello = _recover(output, plan, plan_sha)
                    bound = True
                common._stop_same(plan["unit"], invocation)
                stopped = True
            except BaseException:
                error.add_note("owned unit identity or stop unconfirmed")
    finally:
        closure = {
            "schema": "decision3-cuda-synthetic-supervisor-closure-v1",
            "status": "failed" if failure else "completed_pending_external_audit",
            "invocation": invocation,
            "plan_sha256": plan_sha,
            "cgroup": "unconfirmed",
            "process_group": "unconfirmed",
            "machine_exclusive": False,
        }
        try:
            if not launched:
                closure["cgroup"] = closure["process_group"] = "not_started"
            if bound and invocation is not None and group is not None:
                final = common.systemd.service_show(plan["unit"], timeout=2)
                closure["unit"] = common.systemd.closed_unit_metadata(
                    final, invocation, stopped=stopped
                )
                closure["cgroup"] = common._closed_cgroup(group, plan["unit"])
            if bound and hello is not None:
                closure["process_group"] = _closed_group(hello["pgid"])
            _need(
                (not launched and closure["cgroup"] == closure["process_group"] == "not_started")
                or (
                    closure["cgroup"] in {"absent", "confirmed_unpopulated"}
                    and closure["process_group"] == "absent"
                ),
                "complete process closure required",
            )
        except BaseException as error:
            closure["status"] = "cleanup_unconfirmed"
            if failure is None:
                failure = error
            else:
                failure.add_note("process closure unconfirmed")
        common._write_new(report / "closure.json", child._encode(closure))
    if failure is not None:
        raise failure
    return closure


def main():
    _need(SOURCE_ONLY is False, "reviewed synthetic CUDA supervisor release required")
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    supervise(args.plan, args.plan_sha256, args.report_dir)


if __name__ == "__main__":
    raise SystemExit("invoke the independently reviewed bootstrap")
