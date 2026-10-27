"""Held external owner for a bounded synthetic Decision3 CPU experiment.

This supervises one fresh systemd cgroup; it cannot establish machine-wide
exclusive CPU or memory ownership, complete provenance or benchmark validity.
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

import jev_decision3_candidate_browser_child as child
import jev_decision3_candidate_http_owner as common
import jev_decision3_cpu_experiment_worker as worker
import jev_decision3_jevbench_heaven_supervisor as shared

SOURCE_ONLY = True


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _unit_args(plan, plan_path, plan_sha):
    properties = {
        "Type": "exec",
        "MemoryAccounting": "yes",
        "MemoryMax": str(common.MEMORY_MAX),
        "MemorySwapMax": "0",
        "CPUQuota": "500%",
        "AllowedCPUs": "15-19",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxSec": str(plan["wall_seconds"]),
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
        *["--property=" + k + "=" + v for k, v in properties.items()],
        "/usr/bin/taskset",
        "-c",
        "15-19",
        plan["python"],
        "-B",
        "-s",
        str(worker.SCRIPTS / "jev_decision3_cpu_experiment_worker_release.py"),
        "--plan",
        str(plan_path),
        "--plan-sha256",
        plan_sha,
    ]


def _verify_release(held, released, pins):
    source = common._read_reviewed_source(held, pins[str(held)])
    actual = common._read_reviewed_source(released, pins[str(released)])
    marker = b"SOURCE_ONLY" + b" = True"
    _need(
        source.count(marker) == 1 and actual == source.replace(marker, b"SOURCE_ONLY = False", 1),
        "gate-only release required",
    )


def _bind(properties, hello, plan, plan_sha):
    invocation, group = shared._unit_identity(properties, plan["unit"])
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
        and hello["schema"] == "decision3-cpu-hello-v1"
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
        and common.systemd.cgroup_membership(hello["pid"]) == group,
        "exact worker process and group required",
    )
    _need(
        properties["Type"] == "exec"
        and properties["MemoryAccounting"] == "yes"
        and properties["MemoryMax"] == str(common.MEMORY_MAX)
        and properties["MemorySwapMax"] == "0"
        and properties["OOMPolicy"] == "kill"
        and properties["KillMode"] == "control-group"
        and shared._seconds(properties["RuntimeMaxUSec"]) == plan["wall_seconds"]
        and shared._seconds(properties["TimeoutStopUSec"]) == 3,
        "exact unit limits required",
    )
    _need(os.getpgid(hello["pid"]) == hello["pgid"], "worker process group changed")
    shared._live_limits(group, hello["pid"])
    return invocation, group


def _recover_identity(output, plan, plan_sha):
    """Bind the worker's nonce-bearing handshake to systemd before any stop."""
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        try:
            properties = common.systemd.service_show(plan["unit"], timeout=2)
            hello = common._decode(child._read_bounded(output / "hello.json", 4096))
            invocation, group = _bind(properties, hello, plan, plan_sha)
            return invocation, group, hello
        except Exception:
            time.sleep(0.2)
    raise ValueError("cannot prove launch identity for safe stop") from None


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
    raise ValueError("owned process group still exists or identity was reused")


def _terminal(output, invocation, plan_sha):
    raw = child._read_bounded(output / "terminal.json", 4096)
    value = common._decode(raw)
    _need(
        type(value) is dict
        and set(value) == {"schema", "status", "invocation", "plan_sha256", "result_sha256"}
        and value["schema"] == "decision3-cpu-terminal-v1"
        and value["status"] == "completed_pending_external_audit"
        and value["invocation"] == invocation
        and value["plan_sha256"] == plan_sha
        and type(value["result_sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", value["result_sha256"]),
        "matching worker terminal required",
    )
    result = child._read_bounded(output / "result.json", 4 * 1024 * 1024)
    _need(
        hashlib.sha256(result).hexdigest() == value["result_sha256"], "worker result hash differs"
    )
    return value


def supervise(plan_path, plan_sha, report_dir):
    _need(SOURCE_ONLY is False, "reviewed CPU watcher release required")
    plan = worker._plan(plan_path, plan_sha)
    output = Path(plan["output_dir"])
    report = Path(report_dir)
    _need(
        report == Path(plan_path).parent / "watcher"
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
    invocation = group = hello = terminal = None
    launched = stopped = bound = False
    failure = None
    started = time.monotonic()
    try:
        common._validate_interpreter(plan)
        common._verify_source_pins(plan["source_pins"])
        for name in ("worker", "watcher", "candidate"):
            held = worker.SCRIPTS / worker._HELD[name]
            release = worker.SCRIPTS / worker._HELD[name].replace(".py", "_release.py")
            _verify_release(held, release, plan["source_pins"])
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
                time.monotonic() - started < plan["wall_seconds"], "watcher wall deadline exceeded"
            )
            common._append_event(
                events, {"event": "host", **common._observe(plan["memory_floor_bytes"])}
            )
            properties = common.systemd.service_show(plan["unit"], timeout=2)
            if invocation is None:
                invocation, group = shared._unit_identity(properties, plan["unit"])
            _need(
                properties["InvocationID"] == invocation and properties["LoadState"] == "loaded",
                "worker invocation changed",
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
                            "schema": "decision3-cpu-go-v1",
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
        common._verify_source_pins(plan["source_pins"])
        worker._plan(plan_path, plan_sha)
    except BaseException as error:
        failure = error
        if launched:
            try:
                if not bound:
                    invocation, group, hello = _recover_identity(output, plan, plan_sha)
                    bound = True
                common._stop_same(plan["unit"], invocation)
                stopped = True
            except BaseException:
                error.add_note("owned unit identity or stop unconfirmed")
    finally:
        closure = {
            "schema": "decision3-cpu-watcher-closure-v1",
            "status": "failed" if failure else "completed_pending_external_audit",
            "invocation": invocation,
            "plan_sha256": plan_sha,
            "cgroup": "unconfirmed",
            "process_group": "unconfirmed",
            "machine_exclusive": False,
        }
        try:
            if not launched:
                closure["cgroup"] = "not_started"
                closure["process_group"] = "not_started"
            if bound and invocation is not None and group is not None:
                final = common.systemd.service_show(plan["unit"], timeout=2)
                closure["unit"] = common.systemd.closed_unit_metadata(
                    final, invocation, stopped=stopped
                )
                closure["cgroup"] = common._closed_cgroup(group, plan["unit"])
            if bound and hello is not None:
                closure["process_group"] = _closed_group(hello["pgid"])
            _need(
                (
                    not launched
                    and closure["cgroup"] == "not_started"
                    and closure["process_group"] == "not_started"
                )
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
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    supervise(args.plan, args.plan_sha256, args.report_dir)


if __name__ == "__main__":
    main()
