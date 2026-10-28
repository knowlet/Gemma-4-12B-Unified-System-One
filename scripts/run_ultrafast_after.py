"""Queue the frozen external S1 browser comparison after identified MPS jobs.

No model is loaded here. Process absence uses the same conservative ps/kill(0)
guard as the prefix experiment; a reused PID is never followed or signalled.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import benchmark_mps_prefix as guard

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_FILES = (
    "scripts/run_ultrafast_after.py",
    "scripts/run_ultrafast.py",
    "scripts/ultrafast_browser.py",
    "scripts/benchmark_mps_prefix.py",
    "scripts/run_jevbench_public.py",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def stable_identity(identity):
    """Ignore changing ps state while retaining the original start time and command."""
    remainder = identity[24:].split(maxsplit=1)
    if len(identity[:24]) != 24 or len(remainder) != 2:
        raise RuntimeError("predecessor process identity is incomplete")
    return identity[:24], remainder[1]


def process_state(identity):
    return identity[24:].split(maxsplit=1)[0]


def capture_predecessors(pids):
    handles = []
    for pid in pids:
        identity = guard.process_identity(pid)
        if identity is None:
            raise ValueError(f"predecessor {pid} is not live at startup")
        stable_identity(identity)
        handles.append(
            {"pid": pid, "identity": identity, "captured_at_utc": now(), "status": "captured"}
        )
    return handles


def wait_original(handle, save):
    """Wait on the stored handle, even if it ended before its turn in the queue."""
    original = stable_identity(handle["identity"])
    while True:
        current = guard.process_identity(handle["pid"])
        handle["last_observation"] = current
        handle["observed_at_utc"] = now()
        if current is None:
            handle["status"] = "original_exited"
        elif stable_identity(current) != original:
            handle["status"] = "original_ended_pid_reused"
        elif process_state(current).startswith("Z"):
            handle["status"] = "original_terminal_zombie"
        else:
            handle["status"] = "waiting_same_process"
        save()
        if handle["status"] != "waiting_same_process":
            return
        time.sleep(15)


def freeze_sources(manifest, output):
    paths = sorted(
        {
            *(ROOT / name for name in SCRIPT_FILES),
            *ROOT.glob("src/s1/**/*.py"),
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
            Path(manifest).resolve(),
        }
    )
    expected = {}
    for path in paths:
        relative = path.resolve().relative_to(ROOT)
        target = output / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        expected[str(relative)] = guard.sha256(target)
    guard.validate_frozen_sources(paths, expected)
    return paths, expected


def child_argv(args):
    argv = [
        sys.executable,
        str(ROOT / "scripts/run_ultrafast.py"),
        "--upstream",
        str(args.upstream.resolve()),
        "--output",
        str(args.run_output.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--model",
        str(args.model.resolve()),
        "--device",
        "mps",
        "--profiles",
        "causal_multislot,independent",
        "--repeats",
        str(args.repeats),
        "--max-decisions",
        "12",
        "--task-timeout",
        str(args.task_timeout),
        "--native-text-helper",
    ]
    if args.mps_call_maintenance:
        argv.append("--mps-call-maintenance")
    if args.oversized_head_policy != "reject":
        argv.extend(["--oversized-head-policy", args.oversized_head_policy])
    return argv


def supervise(args):
    if args.run_output.exists():
        raise FileExistsError("child output must be fresh")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt_path = args.output / "receipt.json"
    receipt = {
        "status": "initializing",
        "created_at_utc": now(),
        "supervisor_pid": os.getpid(),
        "runtime": "MPS only",
        "profiles": ["causal_multislot", "independent"],
        "candidate_prefix_requested": False,
        "mps_call_maintenance_requested": args.mps_call_maintenance,
        "oversized_head_policy": args.oversized_head_policy,
        "predecessors": [],
        "child": None,
        "reported_cost_usd": None,
    }

    def save():
        guard.write_json(receipt_path, receipt)

    save()
    try:
        # Capture ALL handles before waiting on any one predecessor. The second
        # original may finish while the first is still active.
        receipt["predecessors"] = capture_predecessors(args.predecessor_pid)
        paths, receipt["source_sha256"] = freeze_sources(args.manifest, args.output)
        receipt["status"] = "waiting"
        save()
        for handle in receipt["predecessors"]:
            wait_original(handle, save)
        guard.validate_frozen_sources(paths, receipt["source_sha256"])
        if args.run_output.exists():
            raise FileExistsError("child output appeared while queued; refusing overwrite")
        argv = child_argv(args)
        child_record = {
            "argv": argv,
            "command": shlex.join(argv),
            "cwd": str(ROOT),
            "started_at_utc": now(),
            "pid": None,
            "identity": None,
            "exit_code": None,
            "log": str(args.output / "child.log"),
            "run_output": str(args.run_output),
            "source_sha256": receipt["source_sha256"],
        }
        receipt["child"] = child_record
        receipt["status"] = "launching"
        save()
        with (args.output / "child.log").open("xb") as log:
            child = subprocess.Popen(
                argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, shell=False
            )
            child_record["pid"] = child.pid
            receipt["status"] = "running"
            save()
            try:
                child_record["identity"] = guard.process_identity(child.pid)
            except Exception as exc:
                # Popen remains the authoritative live handle even when a
                # secondary ps metadata read fails. Never orphan this child.
                child_record["identity_observation_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            save()
            child_record["exit_code"] = child.wait()
        child_record["finished_at_utc"] = now()
        receipt["status"] = "completed" if child_record["exit_code"] == 0 else "failed"
        save()
        return child_record["exit_code"]
    except Exception as exc:
        receipt.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
        save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-pid", type=int, action="append", required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh supervisor receipt directory"
    )
    parser.add_argument(
        "--run-output", type=Path, required=True, help="Fresh child result directory"
    )
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/release/checkpoint")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "configs/benchmarks/ultrafast-external.json"
    )
    parser.add_argument(
        "--upstream", type=Path, default=ROOT / "artifacts/benchmark-tools/jev-ultrafast"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--task-timeout", type=float, default=180)
    parser.add_argument("--mps-call-maintenance", action="store_true")
    parser.add_argument(
        "--oversized-head-policy", choices=("reject", "hierarchical"), default="reject"
    )
    args = parser.parse_args(argv)
    if (
        any(pid <= 0 for pid in args.predecessor_pid)
        or len(set(args.predecessor_pid)) != len(args.predecessor_pid)
        or args.repeats < 1
        or not 0 < args.task_timeout < float("inf")
    ):
        parser.error(
            "unique positive predecessor PIDs, positive repeats and finite timeout required"
        )
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
