"""Held external supervisor for one coherence worker; no score is accepted here.

The worker is launched in a fresh transient unit with a systemd-enforced wall
limit. This process monitors host memory and the ComfyUI queue independently of
the worker. Results remain quarantined pending independent scoring audit.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

import jev_decision3_candidate_browser_child as child
import jev_decision3_candidate_http_owner as common
import jev_decision3_jevbench_coherence_owner as worker

SOURCE_ONLY = True
HELLO = "decision3-jevbench-coherence-hello-v1"
GO = "decision3-jevbench-coherence-go-v1"
TERMINAL = "decision3-jevbench-coherence-terminal-v1"


def _need(value, message):
    if not value:
        raise ValueError(message)


def _gate():
    _need(SOURCE_ONLY is False, "held supervisor requires independent review")


def _seconds(value):
    """Parse the bounded textual timespan emitted by systemctl show."""
    _need(type(value) is str and value, "unit runtime limit missing")
    units = {"h": 3600, "min": 60, "s": 1, "ms": 0.001, "us": 0.000001}
    pieces = re.findall(r"([0-9]+(?:\.[0-9]+)?)(h|min|ms|us|s)", value)
    _need(bool(pieces) and " ".join(a + b for a, b in pieces) == value, "invalid unit runtime")
    return sum(float(number) * units[unit] for number, unit in pieces)


def _verified_release(path, pins):
    held = Path(worker.__file__).absolute()
    released = Path(path)
    _need(child._ordinary_absolute(str(released)), "absolute worker release required")
    _need(str(held) in pins and str(released) in pins, "both worker sources must be pinned")
    raw = child._read_bounded(held, 8 * 1024**2)
    actual = child._read_bounded(released, 8 * 1024**2)
    marker = b"SOURCE_ONLY = True"
    _need(
        raw.count(marker) == 1
        and actual == raw.replace(marker, b"SOURCE_ONLY = False", 1)
        and common._sha(raw) == pins[str(held)]
        and common._sha(actual) == pins[str(released)],
        "worker release must be exactly the pinned source gate replacement",
    )


def _unit_args(plan, plan_path, plan_sha, release):
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
        *["--property=" + key + "=" + value for key, value in properties.items()],
        "/usr/bin/taskset",
        "-c",
        "15-19",
        plan["python"],
        "-B",
        "-s",
        str(release),
        "--plan",
        str(plan_path),
        "--plan-sha256",
        plan_sha,
    ]


def _unit_identity(properties, unit):
    invocation, group = properties["InvocationID"], properties["ControlGroup"]
    _need(
        properties["LoadState"] == "loaded"
        and re.fullmatch(r"[0-9a-f]{32}", invocation) is not None
        and type(group) is str
        and group.startswith("/")
        and ".." not in Path(group).parts
        and group.endswith("/app.slice/" + unit),
        "fresh owned invocation and cgroup required",
    )
    return invocation, group


def _hello_identity(properties, hello, plan, plan_sha):
    invocation, group = _unit_identity(properties, plan["unit"])
    _need(
        type(hello) is dict
        and set(hello)
        == {"schema", "pid", "startticks", "boot_id", "unit", "invocation", "plan_sha256"}
        and hello["schema"] == HELLO
        and hello["unit"] == plan["unit"]
        and hello["invocation"] == invocation
        and hello["plan_sha256"] == plan_sha
        and type(hello["pid"]) is int
        and hello["pid"] > 0
        and str(hello["pid"]) == properties["MainPID"]
        and properties["ActiveState"] == "active"
        and properties["Type"] == "exec"
        and hello["boot_id"] == common.systemd.boot_id()
        and hello["startticks"] == common.systemd.process_identity(hello["pid"])["startticks"]
        and common.systemd.cgroup_membership(hello["pid"]) == group,
        "exact worker hello, process and invocation required",
    )
    return invocation, group


def _binding(properties, hello, plan, plan_sha):
    invocation, group = _hello_identity(properties, hello, plan, plan_sha)
    _need(
        properties["MemoryAccounting"] == "yes"
        and properties["MemoryMax"] == str(common.MEMORY_MAX)
        and properties["MemorySwapMax"] == "0"
        and properties["OOMPolicy"] == "kill"
        and properties["KillMode"] == "control-group"
        and _seconds(properties["RuntimeMaxUSec"]) == plan["wall_seconds"]
        and _seconds(properties["TimeoutStopUSec"]) == 3,
        "exact unit resource and hard deadline limits required",
    )
    _live_limits(group, hello["pid"])
    return invocation, group


def _live_limits(group, pid):
    path = Path("/sys/fs/cgroup") / group.lstrip("/")
    quota, period = map(int, (path / "cpu.max").read_text().split())
    _need(
        os.sched_getaffinity(pid) == {15, 16, 17, 18, 19}
        and (path / "memory.max").read_text().strip() == str(common.MEMORY_MAX)
        and (path / "memory.swap.max").read_text().strip() == "0"
        and period > 0
        and quota == period * 5,
        "actual worker CPU, memory and swap limits required",
    )


def _publish_go(output, invocation, plan_sha, deadline):
    temporary, destination = output / ".go.tmp", output / "go.json"
    common._write_new(
        temporary, child._encode({"schema": GO, "invocation": invocation, "plan_sha256": plan_sha})
    )
    try:
        deadline()
        os.link(temporary, destination, follow_symlinks=False)
    finally:
        temporary.unlink()
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _terminal(output, invocation, plan_sha):
    status = common._decode(child._read_bounded(output / "terminal.json", 1024**2))
    _need(
        type(status) is dict
        and status.get("schema") == TERMINAL
        and status.get("invocation") == invocation
        and status.get("plan_sha256") == plan_sha
        and status.get("status") == "completed_pending_external_audit"
        and status.get("host_supervision") == "external"
        and status.get("model_score") is None
        and status.get("final_cuda_allocated") == 0
        and type(status.get("evaluation")) is dict,
        "complete worker result with CUDA cleanup required",
    )
    return status


def supervise(plan_path, plan_sha, release, report_dir, memory_floor):
    _gate()
    plan = worker._load_plan(plan_path, plan_sha)
    output = Path(plan["output_dir"])
    report = Path(report_dir)
    _need(
        child._ordinary_absolute(str(report))
        and not report.exists()
        and report.parent.is_dir()
        and report.parent.resolve() == report.parent
        and report != output
        and not report.is_relative_to(output),
        "fresh separate private supervisor report required",
    )
    _need(
        type(memory_floor) is int and memory_floor >= common.MEMORY_FLOOR, "memory floor required"
    )
    _need(
        output.is_dir()
        and output.resolve() == output
        and output.stat().st_mode & 0o077 == 0
        and not any(output.iterdir()),
        "fresh empty private worker output required",
    )
    report.mkdir(mode=0o700, exist_ok=False)
    journal = report / "events.jsonl"
    common._write_new(journal, b"")
    invocation = group = terminal = None
    launched = launch_succeeded = stopped = False
    failure = None
    start = time.monotonic()
    try:
        common._validate_interpreter(plan)
        common._verify_source_pins(plan["source_pins"])
        for path in (
            __file__,
            worker.__file__,
            common.__file__,
            child.__file__,
            common.systemd.__file__,
        ):
            _need(
                str(Path(path).absolute()) in plan["source_pins"], "supervisor source pins required"
            )
        _verified_release(release, plan["source_pins"])
        common._append_event(journal, {"event": "prelaunch", **common._observe(memory_floor)})
        initial = common.systemd.service_show(plan["unit"], timeout=2)
        _need(
            initial["LoadState"] == "not-found"
            and initial["ActiveState"] == "inactive"
            and initial["MainPID"] == "0",
            "fresh absent unit required",
        )
        launched = True
        subprocess.run(
            _unit_args(plan, plan_path, plan_sha, release),
            check=True,
            capture_output=True,
            timeout=10,
        )
        launch_succeeded = True
        while True:
            _need(
                time.monotonic() - start < plan["wall_seconds"], "supervisor wall deadline exceeded"
            )
            common._append_event(journal, {"event": "host", **common._observe(memory_floor)})
            properties = common.systemd.service_show(plan["unit"], timeout=2)
            if invocation is None and (output / "hello.json").is_file():
                hello = common._decode(child._read_bounded(output / "hello.json", 4096))
                invocation, group = _binding(properties, hello, plan, plan_sha)
                common._append_event(
                    journal,
                    {
                        "event": "binding_verified",
                        "invocation": invocation,
                        "pid": hello["pid"],
                        "startticks": hello["startticks"],
                        "boot_id": hello["boot_id"],
                        "cgroup": group,
                    },
                )

                def before_go():
                    _need(time.monotonic() - start < plan["wall_seconds"], "deadline before GO")

                _publish_go(output, invocation, plan_sha, before_go)
            elif invocation is not None:
                _need(
                    properties["InvocationID"] == invocation
                    and properties["LoadState"] == "loaded",
                    "worker invocation changed",
                )
                if properties["MainPID"] != "0":
                    _need(
                        properties["MainPID"] == str(hello["pid"])
                        and common.systemd.process_identity(hello["pid"])["startticks"]
                        == hello["startticks"]
                        and common.systemd.cgroup_membership(hello["pid"]) == group,
                        "running worker process identity changed",
                    )
                    _live_limits(group, hello["pid"])
            terminal = common._terminal_metadata(properties, invocation)
            if terminal is not None:
                break
            time.sleep(0.5)
        _need(common._successful_exit(terminal), "worker did not exit successfully")
        _terminal(output, invocation, plan_sha)
        common._verify_source_pins(plan["source_pins"])
        worker._load_plan(plan_path, plan_sha)
        _verified_release(release, plan["source_pins"])
    except BaseException as error:
        failure = error
        if launch_succeeded and invocation is None and (output / "hello.json").is_file():
            try:
                hello = common._decode(child._read_bounded(output / "hello.json", 4096))
                invocation, group = _hello_identity(
                    common.systemd.service_show(plan["unit"], timeout=2), hello, plan, plan_sha
                )
            except BaseException:
                pass
        if invocation is not None:
            try:
                common._stop_same(plan["unit"], invocation)
                stopped = True
            except BaseException:
                error.add_note("owned unit stop was not confirmed")
        raise
    finally:
        closure = {
            "schema": "decision3-jevbench-coherence-supervisor-closure-v1",
            "unit": plan["unit"],
            "invocation": invocation,
            "launched": launched,
            "launch_succeeded": launch_succeeded,
            "stop_returned": stopped,
            "status": "failed_no_score" if failure else "completed_pending_external_audit",
            "model_score": None,
            "official_rank": None,
            "closure": "unknown",
        }
        try:
            _need(
                invocation is not None and group is not None, "owned invocation closure unavailable"
            )
            final = common.systemd.service_show(plan["unit"], timeout=2)
            closure["closure"] = common.systemd.closed_unit_metadata(
                final, invocation, stopped=stopped
            )
            closure["cgroup_closure"] = common._closed_cgroup(group, plan["unit"])
        except BaseException as close_error:
            closure["closure"] = "unknown"
            closure["status"] = "failed_no_score"
            closure["closure_error_type"] = type(close_error).__name__
            if invocation is not None and group is not None:
                try:
                    common._stop_same(plan["unit"], invocation)
                    stopped = True
                    closure["stop_returned"] = True
                    final = common.systemd.service_show(plan["unit"], timeout=2)
                    closure["closure"] = common.systemd.closed_unit_metadata(
                        final, invocation, stopped=True
                    )
                    closure["cgroup_closure"] = common._closed_cgroup(group, plan["unit"])
                except BaseException:
                    closure["closure"] = "unknown"
            if failure is not None:
                failure.add_note("owned unit or cgroup closure unconfirmed")
        try:
            child._publish_control(report / "closure.json", child._encode(closure))
        except BaseException:
            if failure is None:
                raise
            failure.add_note("supervisor closure record could not be written")
        if failure is None and closure["status"] != "completed_pending_external_audit":
            raise RuntimeError("owned unit or cgroup closure required recovery")
    return closure


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--worker-release", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--memory-floor-bytes", required=True, type=int)
    args = parser.parse_args(argv)
    supervise(
        args.plan, args.plan_sha256, args.worker_release, args.report_dir, args.memory_floor_bytes
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
