"""External owner of the new single-worker CPU179 preparation; default dry."""

import argparse
import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
PREVIOUS = ROOT / "artifacts/jev-runtime/v2-action-training-prepare-20261021-01"
PRIOR_SHA = "16dca51868c10334b231f106cfd79273f2a2d9c932ae2e2e857bcb5614d0258b"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
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
        json.dump(value, stream, indent=2, allow_nan=False)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    entry = module(HERE / "prepare_cpu.py", "_current_native179_prepare")
    w, r, c, cpu, old, plan, technical = entry.load_plan(args.plan.absolute(), args.plan_sha256)
    index = json.loads((PREVIOUS / "selected-evidence-index-01.json").read_bytes())
    ref = next(p for p in index["sources"] if p["path"].endswith("/run_compiler_tests.py"))
    blob = (ROOT / ref["path"]).read_bytes()
    assert len(blob) == ref["size_bytes"] and hashlib.sha256(blob).hexdigest() == ref["sha256"]
    guard = module(ROOT / ref["path"], "_native179_external_guardian")
    assert not (ROOT / plan["output"]).exists()
    if not args.run:
        print("DRY_AUTHENTICATED_NEW_SINGLE_WORKER_PLAN_NO_PROCESSOR_OR_ACTOR")
        return 0
    before = guard.host()
    assert before["mem_available_bytes"] >= guard.RESERVE + guard.CAP + 64 * 1024**2
    prior_blob = (PREVIOUS / "root-external-closure-01.json").read_bytes()
    assert hashlib.sha256(prior_blob).hexdigest() == PRIOR_SHA
    prior = json.loads(prior_blob)
    initial = guard.status(c.UNIT)
    assert initial["InvocationID"] == prior["unit_exit"]["invocation_id"]
    assert initial["MainPID"] == "0" and not initial["ControlGroup"]
    assert initial["SubState"] in ("failed", "exited", "dead")
    scans = [scan(prior["actors"])]
    time.sleep(0.2)
    scans.append(scan(prior["actors"]))
    assert all(not row["matches"] and not row["unreadable"] for row in scans)
    guard.command("systemctl", "--user", "stop", c.UNIT)
    stopped = guard.status(c.UNIT)
    if stopped["LoadState"] != "not-found":
        assert stopped["MainPID"] == "0" and not stopped["ControlGroup"]
        assert stopped["InvocationID"] == initial["InvocationID"]
        guard.command("systemctl", "--user", "reset-failed", c.UNIT)
        stopped = guard.status(c.UNIT)
    assert stopped["MainPID"] == "0" and stopped["LoadState"] == "not-found"
    save(
        "prepare-root-prior-stop-01.json",
        {"prior_sha256": PRIOR_SHA, "before": initial, "scans": scans, "after": stopped},
    )
    argv = [
        "systemd-run",
        "--user",
        "--unit=" + c.UNIT,
        "--working-directory=" + str(ROOT),
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
        "--property=Description=" + plan["nonce"],
        "--property=StandardOutput=append:" + str(HERE / "prepare-unit-service-01.log"),
        "--property=StandardError=append:" + str(HERE / "prepare-unit-service-01.log"),
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
        str(HERE / "prepare_cpu.py"),
        "--run",
        "--plan",
        str(args.plan.absolute()),
        "--plan-sha256",
        args.plan_sha256,
    ]
    save(
        "prepare-root-launch-01.json",
        {
            "plan_sha256": args.plan_sha256,
            "argv": argv,
            "host_before": before,
            "guardian": ref,
            "source": c.pin(Path(__file__)),
            "scope": "Single owned processor-only CPU worker; no model, browser, or old qualification.",
        },
    )
    launched = False
    try:
        launched = True
        print(guard.command(*argv), flush=True)
        first = guard.status(c.UNIT)
        invocation = first["InvocationID"]
        assert len(invocation) == 32 and invocation != initial["InvocationID"]
        assert (
            guard.command(
                "systemctl", "--user", "show", c.UNIT, "--value", "--property=Description"
            ).strip()
            == plan["nonce"]
        )
        closure = guard.watch(
            invocation,
            first,
            unit=c.UNIT,
            get_status=lambda: guard.status(c.UNIT),
            deadline=650,
            save=lambda _, value: save("prepare-root-unit-exit-01.json", value),
        )
    except BaseException as error:
        if launched and not (HERE / "prepare-root-unit-exit-01.json").exists():
            # Authentication can itself fail. Preserve UNKNOWN and the fixed
            # 660-second unit bound; never stop an unrecognised invocation.
            emergency = {"primary_error": {"type": type(error).__name__, "message": str(error)}}
            try:
                state = guard.status(c.UNIT)
                description = guard.command(
                    "systemctl", "--user", "show", c.UNIT, "--value", "--property=Description"
                ).strip()
                emergency["before"] = state
                if (
                    description == plan["nonce"]
                    and len(state["InvocationID"]) == 32
                    and state["InvocationID"] != initial["InvocationID"]
                ):
                    emergency["stop_result"] = guard.command("systemctl", "--user", "stop", c.UNIT)
                    emergency["after"] = guard.status(c.UNIT)
                else:
                    emergency["closure"] = "UNKNOWN_NO_AUTHENTICATED_OWNED_INVOCATION"
            except BaseException as secondary:
                emergency["closure"] = "UNKNOWN"
                emergency["observation_error"] = {
                    "type": type(secondary).__name__,
                    "message": str(secondary),
                }
            save("prepare-root-startup-failure-01.json", emergency)
        raise
    c.authenticate(plan["sources"])
    output = ROOT / plan["output"]
    terminal = json.loads(c.read(output / "terminal.json"))
    actors = {"worker": terminal["actor"]}
    scans = [scan(actors)]
    time.sleep(0.2)
    scans.append(scan(actors))
    review, review_error = None, None
    try:
        review = entry.validate_saved(args.plan.absolute(), args.plan_sha256)
    except BaseException as error:
        review_error = {"type": type(error).__name__, "message": str(error)}
    summary = {
        "unit_exit": closure,
        "actors": actors,
        "two_proc_scans": scans,
        "source_files_unchanged": len(plan["sources"]),
        "review": review,
        "review_error": review_error,
        "worker_primary_error": terminal["primary_error"],
        "final_postexit_cgroup_counters": None,
        "counter_scope": "Cgroup removed; no final zero-OOM claim.",
        "task_quality_attested": False,
    }
    save("prepare-root-external-closure-01.json", summary)
    assert closure["reason"] == "terminal" and closure["cgroup_absent_after_stop"]
    assert closure["after_stop"]["MainPID"] == "0"
    assert all(not row["matches"] and not row["unreadable"] for row in scans)
    assert terminal["InvocationID"] == invocation
    assert str(terminal["actor"]["pid"]) == closure["terminal_before_stop"]["ExecMainPID"]
    assert terminal["primary_error"] is None and review_error is None
    assert terminal["complete_pending_external_unit_exit"] is True
    assert all(
        closure["terminal_before_stop"][k] == v
        for k, v in {
            "Result": "success",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "MainPID": "0",
        }.items()
    )
    print(json.dumps({"InvocationID": invocation, "review": review, "closed": True}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
