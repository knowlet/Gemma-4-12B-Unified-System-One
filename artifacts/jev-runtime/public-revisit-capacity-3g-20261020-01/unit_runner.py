"""Run the unchanged full75 CPU workload inside one new 3GiB/swap0 service.

Default dry. This supervisor, controller, and worker share the hard unit cap.
An external operator must separately attest unit exit and empty membership.
This result never substitutes for the original10GiB capacity authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
KIND = "jev_revisit_full75_cpu_3g_unit_supervisor_v1"
UNIT = "jev-cpu3g-20261020-01.service"
CGROUP = "/user.slice/user-1001.slice/user@1001.service/app.slice/" + UNIT
MIB, GIB = 1024**2, 1024**3


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path, maximum=2 * MIB):
    path = Path(path).absolute()
    require(path.is_relative_to(ROOT) and ".." not in path.parts, "Repository confined path")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Ordinary input path")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, "Bounded regular input")
        blob = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())

    def signature(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    require(signature(before) == signature(after) == signature(path.stat()), "Stable input")
    require(len(blob) == before.st_size, "Complete input bytes")
    return blob


def pin(path):
    blob = read(path)
    return {
        "path": Path(path).relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
    }


def checked(ref):
    require(type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"}, "Typed source ref")
    require(type(ref["path"]) is str and not Path(ref["path"]).is_absolute(), "Relative source ref")
    require(pin(ROOT / ref["path"]) == ref, "Selected source bytes differ")
    return ROOT / ref["path"]


def load_plan(path, expected):
    blob = read(path)
    require(hashlib.sha256(blob).hexdigest() == expected, "Externally selected unit plan")
    plan = json.loads(blob)
    require(
        set(plan)
        == {"schema_id", "self_source", "controller_source", "controller_plan", "unit", "output"},
        "Strict unit plan keys",
    )
    require(plan["schema_id"] == KIND, "Separate3GiB unit kind")
    require(checked(plan["self_source"]) == Path(__file__).absolute(), "Exact supervisor origin")
    controller_path = checked(plan["controller_source"])
    require(controller_path == HERE / "revisit_capacity_controller.py", "Exact sibling controller")
    checked(plan["controller_plan"])
    require(
        type(plan["unit"]) is dict
        and set(plan["unit"]) == {"boot_id", "cgroup_path"}
        and plan["unit"]["cgroup_path"] == CGROUP
        and type(plan["unit"]["boot_id"]) is str
        and len(plan["unit"]["boot_id"]) == 36,
        "Exact fresh unit identity",
    )
    require(ROOT / plan["output"] == HERE / "unit-actual-01", "Fresh fixed supervisor output")
    spec = importlib.util.spec_from_file_location("_cpu3g_selected_controller", controller_path)
    controller = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = controller
    exec(compile(read(controller_path), str(controller_path), "exec"), controller.__dict__)
    selected = controller.load_plan(
        ROOT / plan["controller_plan"]["path"], plan["controller_plan"]["sha256"]
    )
    return plan, controller, selected


def unit_properties():
    keys = (
        "InvocationID",
        "LoadState",
        "ControlGroup",
        "MainPID",
        "Type",
        "MemoryAccounting",
        "MemoryMax",
        "MemoryHigh",
        "MemorySwapMax",
        "OOMPolicy",
        "KillMode",
        "RuntimeMaxUSec",
        "TimeoutStopUSec",
    )
    result = subprocess.run(
        ["systemctl", "--user", "show", UNIT, *["--property=" + k for k in keys]],
        capture_output=True,
        timeout=5,
        check=True,
    )
    require(len(result.stdout) < MIB and not result.stderr, "Bounded unit property readback")
    values = dict(line.split("=", 1) for line in result.stdout.decode().splitlines())
    validate_properties(values, os.getpid())
    return values


def validate_properties(values, pid):
    require(
        values["LoadState"] == "loaded"
        and values["ControlGroup"] == CGROUP
        and values["Type"] == "exec"
        and values["MainPID"] == str(pid)
        and values["MemoryAccounting"] == "yes"
        and values["MemoryMax"] == str(3 * GIB)
        and values["MemoryHigh"] == "infinity"
        and values["MemorySwapMax"] == "0"
        and values["OOMPolicy"] == "kill"
        and values["KillMode"] == "control-group"
        and values["RuntimeMaxUSec"] == "11min"
        and values["TimeoutStopUSec"] == "10s"
        and len(values["InvocationID"]) == 32
        and all(c in "0123456789abcdef" for c in values["InvocationID"]),
        "Exact own whole-tree3GiB/swap0 service and660s deadline",
    )


def supervise(path, expected, *, run=False):
    plan, controller, selected = load_plan(path, expected)
    capacity_plan, cpu, worker, cohort, files, capacity_output = selected
    if not run:
        return {
            "schema_id": KIND,
            "status": "DRY_NO_UNIT_PROCESSOR_OR_OUTPUT",
            "rows": 75,
            "unit_memory_max": 3 * GIB,
            "source_union_files": len(files),
        }
    require(
        not any(
            n.split(".")[0] in {"torch", "transformers", "tokenizers", "s1"} for n in sys.modules
        ),
        "Fresh framework-free supervisor",
    )
    require(
        all(os.environ.get(k) == v for k, v in capacity_plan["environment"].items()),
        "Original exact CPU environment",
    )
    unit_before = cpu.assert_current_unit(plan["unit"])
    properties = unit_properties()
    cpu.configure_limits("parent")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * MIB, 64 * MIB))
    observation = cpu.memory_observation()
    admission = controller.prospective(cpu, plan["unit"]["boot_id"], observation["MemTotal"])
    output = HERE / "unit-actual-01"
    require(not output.exists() and not capacity_output.exists(), "No reuse of actual outputs")
    output.mkdir(mode=0o700)
    common = controller.component(controller.COMMON)
    parent = common.process_table()[os.getpid()]
    identity = {k: parent[k] for k in ("pid", "start_ticks", "pgid", "sid")}
    argv = [
        capacity_plan["python"],
        "-B",
        "-s",
        str(checked(plan["controller_source"])),
        "--run",
        "--plan",
        str(checked(plan["controller_plan"])),
        "--plan-sha256",
        plan["controller_plan"]["sha256"],
    ]
    env = {k: os.environ[k] for k in cpu.SECRET_FREE_ENV if k in os.environ}
    env.update(capacity_plan["environment"])
    cpu.save(
        output,
        "unit-intent.json",
        cpu.encode(
            {
                "schema_id": KIND,
                "plan_sha256": expected,
                "parent": identity,
                "argv": argv,
                "unit": unit_before,
                "properties": properties,
                "admission": admission,
            }
        ),
    )
    execution, monitor = controller.run_owned(
        cpu,
        common,
        output,
        argv,
        env,
        plan["unit"]["boot_id"],
        {**identity, "mem_total": observation["MemTotal"]},
        wall=630,
    )
    primary, review = execution["primary_error"], None
    if primary is None and not execution["secondary_errors"] and execution["returncode"] == 0:
        try:
            cleanup = execution["cleanup"]
            require(
                cleanup["confirmed"] is True and cleanup["remaining"] == [],
                "Controller whole owned closure",
            )
            terminal = cpu.parse(cpu.read(capacity_output / "controller-terminal.json"))
            require(
                terminal["schema_id"] == controller.KIND
                and terminal["plan_sha256"] == plan["controller_plan"]["sha256"]
                and terminal["status"] == "CLOSED_CPU_COHORT_PENDING_EXTERNAL_PARENT_EXIT"
                and terminal["primary_error"] is None
                and terminal["execution"]["returncode"] == 0
                and terminal["execution"]["secondary_errors"] == [],
                "Genuine complete controller",
            )
            owned_parent = terminal["parent"]
            require(
                cleanup["tracked_PID_start_ticks"].get(str(owned_parent["pid"]))
                == owned_parent["start_ticks"],
                "Receipt joins exact closed controller",
            )
            intent = cpu.parse(cpu.read(capacity_output / "controller-intent.json"))
            require(
                intent["parent"] == owned_parent and intent["nonce"] == terminal["nonce"],
                "Same controller intent and nonce",
            )
            config_ref = pin(capacity_output / "worker-config.json")
            reviewed = controller.validate_worker(
                cpu,
                worker,
                cohort,
                capacity_output,
                config_ref["sha256"],
                terminal["nonce"],
                terminal["execution"]["cleanup"],
            )
            require(
                reviewed == terminal["selected"], "Independent full75 saved-evidence validation"
            )
            child = cpu.parse(cpu.read(capacity_output / "worker/terminal.json"))["child"]
            gate_rows = [
                cpu.parse(line)
                for line in cpu.read(capacity_output / "worker/resource-gates.jsonl").splitlines()
            ]
            require(
                all(
                    row["unit"]["caller"] == child
                    and cpu.unit_identity(row["unit"]) == cpu.unit_identity(unit_before)
                    for row in gate_rows
                ),
                "Every worker phase joins the same exact unit inode and closed worker identity",
            )
            for monitor_path in (
                capacity_output / "owned-memory-monitor.jsonl",
                output / "owned-memory-monitor.jsonl",
            ):
                require(
                    all(
                        cpu.unit_identity(cpu.parse(line)["unit"]) == cpu.unit_identity(unit_before)
                        for line in cpu.read(monitor_path).splitlines()
                    ),
                    "Both tracked-tree monitors join the same initial unit inode",
                )
            inner_samples = cpu.validate_monitor_evidence(
                cpu.read(capacity_output / "owned-memory-monitor.jsonl"),
                plan["unit"]["boot_id"],
                parent=owned_parent,
                cleanup=terminal["execution"]["cleanup"],
                unit=plan["unit"],
            )
            monitor.check({})
            outer_samples = cpu.validate_monitor_evidence(
                cpu.read(output / "owned-memory-monitor.jsonl"),
                plan["unit"]["boot_id"],
                parent=identity,
                cleanup=cleanup,
                unit=plan["unit"],
            )
            worker.authenticate(files)
            require(pin(path)["sha256"] == expected, "Unit plan unchanged")
            for key in ("self_source", "controller_source", "controller_plan"):
                checked(plan[key])
            unit_after = cpu.assert_current_unit(plan["unit"])
            require(
                unit_properties()["InvocationID"] == properties["InvocationID"],
                "Same service invocation after full75",
            )
            controller.prospective(cpu, plan["unit"]["boot_id"], observation["MemTotal"])
            review = {
                "counts": reviewed["counts"],
                "rows": 75,
                "all68_policy_rows_accepted": all(
                    r["outcome"] == "ACCEPTED" for r in reviewed["rows"][:68]
                ),
                "controller_monitor_samples": inner_samples,
                "supervisor_monitor_samples": outer_samples,
                "unit_after": unit_after,
            }
        except BaseException as error:
            primary = cpu.error_value(error, "independent-3g-unit-closure-validation")
    result = {
        "schema_id": KIND,
        "plan_sha256": expected,
        "parent": identity,
        "unit": plan["unit"],
        "InvocationID": properties["InvocationID"],
        "status": "FULL75_VALIDATED_PENDING_EXTERNAL_UNIT_EXIT" if review else "FAILED_OR_UNKNOWN",
        "primary_error": primary,
        "execution": execution,
        "review": review,
        "task_quality_attested": False,
        "original10GiB_qualification_attested": False,
        "external_unit_exit0_empty_cgroup_and_same_invocation_required": True,
        "disk_scope": "two separate sampled1GiB trees; aggregate at most2GiB;64MiB per file",
    }
    try:
        cpu.save(output, "unit-terminal.json", cpu.encode(result), terminal=True)
    except BaseException as error:
        if primary is not None:
            raise RuntimeError(f"{primary}; unit terminal persistence also failed") from error
        raise
    return 0 if review else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    result = supervise(args.plan, args.plan_sha256, run=args.run)
    if isinstance(result, dict):
        print(json.dumps(result))
        return 0
    return result


if __name__ == "__main__":
    raise SystemExit(main())
