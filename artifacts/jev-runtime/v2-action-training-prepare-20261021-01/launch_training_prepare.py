"""External owner of the sealed CPU-only preparation; default dry."""

import argparse
import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).absolute().parent
PLAN_SHA = "d83672b6511225d97c45cbf540423aa4f488ddc4491323dc67a3aafd9c42ac43"
PRIOR_SHA = "1c2909722a94cb340ccafed4adad79014bc2b290f56185957086f2a5fc0e4601"


def module(name):
    path = HERE / (name + ".py")
    spec = importlib.util.spec_from_file_location("_root_" + name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def scan(actors):
    pids = {a["pid"] for a in actors.values()}
    groups = {a["pgid"] for a in actors.values()}
    sessions = {a["sid"] for a in actors.values()}
    matches, unreadable = [], []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            text = (path / "stat").read_text()
            fields = text[text.rfind(")") + 2 :].split()
            pid, pgid, sid = int(path.name), int(fields[2]), int(fields[3])
            if pid in pids or pgid in groups or sid in sessions:
                matches.append(
                    {"pid": pid, "pgid": pgid, "sid": sid, "start_ticks": int(fields[19])}
                )
        except FileNotFoundError:
            continue
        except (PermissionError, OSError, ValueError, IndexError) as error:
            unreadable.append({"pid": path.name, "error": type(error).__name__})
    return {"matches": matches, "unreadable": unreadable}


def save(name, value):
    with (HERE / name).open("x") as stream:
        json.dump(value, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print("DRY_NO_PROCESSOR_OR_ACTOR; explicit --run required")
        return
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    entry, guard = module("training_prepare"), module("run_compiler_tests")
    plan_path = HERE / "prepare-plan-01.json"
    r, c, plan, cpu, old, files, _ = entry.load_plan(plan_path, PLAN_SHA)
    assert not (c.ROOT / plan["output"]).exists()
    before = guard.host()
    assert before["mem_available_bytes"] >= guard.RESERVE + guard.CAP + 64 * 1024**2
    prior_path = (
        c.ROOT
        / "artifacts/jev-runtime/shared-elements-capacity-20261021-01/root-external-closure-01.json"
    )
    prior_blob = prior_path.read_bytes()
    assert hashlib.sha256(prior_blob).hexdigest() == PRIOR_SHA
    prior = json.loads(prior_blob)
    initial = guard.status(c.UNIT)
    assert initial["InvocationID"] == prior["properties"]["InvocationID"]
    assert initial["MainPID"] == "0" and initial["SubState"] == "exited"
    scans = [scan(prior["actors"])]
    time.sleep(0.2)
    scans.append(scan(prior["actors"]))
    assert all(not row["matches"] and not row["unreadable"] for row in scans)
    guard.command("systemctl", "--user", "stop", c.UNIT)
    stopped = guard.status(c.UNIT)
    assert stopped["MainPID"] == "0" and stopped["LoadState"] == "not-found"
    save(
        "root-prior-unit-stop-01.json",
        {"prior_sha256": PRIOR_SHA, "before": initial, "scans": scans, "after": stopped},
    )
    argv = [
        "systemd-run",
        "--user",
        "--unit=" + c.UNIT,
        "--working-directory=" + str(c.ROOT),
        "--property=Type=exec",
        "--property=MemoryAccounting=yes",
        "--property=MemoryMax=3221225472",
        "--property=MemoryHigh=infinity",
        "--property=MemorySwapMax=0",
        "--property=OOMPolicy=kill",
        "--property=KillMode=control-group",
        "--property=RuntimeMaxSec=660",
        "--property=TimeoutStopSec=10",
        "--property=RemainAfterExit=yes",
        "--property=StandardOutput=append:" + str(HERE / "unit-service-01.log"),
        "--property=StandardError=append:" + str(HERE / "unit-service-01.log"),
        *[
            "--setenv=" + k + "=" + v
            for k, v in {
                **plan["environment"],
                "JEV_CPU3G_BOOT": plan["unit"]["boot_id"],
                "JEV_CPU3G_CGROUP": c.CGROUP,
            }.items()
        ],
        c.PYTHON,
        "-B",
        "-s",
        str(HERE / "training_prepare.py"),
        "--run",
        "--plan",
        str(plan_path),
        "--plan-sha256",
        PLAN_SHA,
    ]
    save(
        "root-launch-01.json",
        {
            "plan_sha256": PLAN_SHA,
            "host_before": before,
            "argv": argv,
            "sources": {
                name: c.pin(HERE / name)
                for name in ("launch_training_prepare.py", "run_compiler_tests.py")
            },
            "queue_policy": "Observe only; unchanged CPU memory reserve, 3GiB hard cap, no Comfy mutation.",
        },
    )
    print(guard.command(*argv), flush=True)
    first = guard.status(c.UNIT)
    invocation = first["InvocationID"]
    assert len(invocation) == 32 and invocation != prior["properties"]["InvocationID"]
    closure = guard.watch(
        invocation,
        first,
        unit=c.UNIT,
        get_status=lambda: guard.status(c.UNIT),
        deadline=650,
        save=lambda _, value: save("root-unit-exit-01.json", value),
    )
    c.authenticate(files)
    parent = json.loads(c.read(c.ROOT / plan["output"] / "parent-terminal.json"))
    worker = json.loads(c.read(c.ROOT / plan["output"] / "worker/terminal.json"))
    actors = {"parent": parent["parent"], "worker": worker["child"]}
    scans = [scan(actors)]
    time.sleep(0.2)
    scans.append(scan(actors))
    summary = {
        "unit_exit": closure,
        "actors": actors,
        "two_proc_scans": scans,
        "sources_count_unchanged": len(files),
        "parent_review": parent["review"],
        "parent_primary_error": parent["primary_error"],
        "final_postexit_cgroup_counters": None,
        "counter_scope": "Cgroup removed; no final zero-OOM claim.",
        "task_quality_attested": False,
    }
    save("root-external-closure-01.json", summary)
    assert closure["reason"] == "terminal" and closure["after_stop"]["MainPID"] == "0"
    assert closure["cgroup_absent_after_stop"]
    terminal = closure["terminal_before_stop"]
    assert all(
        terminal[k] == v
        for k, v in {
            "Result": "success",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "MainPID": "0",
        }.items()
    )
    assert all(not row["matches"] and not row["unreadable"] for row in scans)
    assert parent["InvocationID"] == invocation and parent["primary_error"] is None
    assert parent["complete_pending_external_unit_exit"] is True
    print(
        json.dumps(
            {"InvocationID": invocation, "review": parent["review"], "whole_owned_closed": True}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
