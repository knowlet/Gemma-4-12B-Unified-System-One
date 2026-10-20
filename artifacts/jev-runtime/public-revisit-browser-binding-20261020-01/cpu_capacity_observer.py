"""Owned external observer for the frozen CPU75 controller, default dry.

No model or browser runs here. The original CPU75 parent/worker remain unchanged.
The reused owned lifecycle monitors this observer plus both child levels under
the original10GiB aggregate RSS/reserve and600s outer deadline. A successful
measurement can contain explicit request refusals; benchmark readiness is a
separate result from the pinned new selector.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
KIND = "jev_revisit_cpu_capacity_external_observer_v1"
MAX_OUTPUT, MAX_FILE, RESERVE = 1024**3, 64 * 1024**2, 1024**2
CONTROLLER = {
    "path": "artifacts/jev-runtime/public-revisit-capacity-execution-20261020-01/revisit_capacity_controller.py",
    "sha256": "ff31c2765c18ebf1aae92e226396ed91c15f744071b1dc5a8ce8e630b1f351b4",
    "size_bytes": 21290,
}
BINDING = {
    "path": "artifacts/jev-runtime/public-revisit-browser-binding-20261020-01/revisit_capacity_binding.py",
    "sha256": "0ef8275bd2f4f57f50710ec70bde5ce3c809f743cf783405e8db2cbbbdeb29f7",
    "size_bytes": 15758,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def controller_module():
    path = ROOT / CONTROLLER["path"]
    require(
        not any(p.is_symlink() for p in (path, *path.parents)), "Ordinary frozen CPU controller"
    )
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and before.st_size == CONTROLLER["size_bytes"],
            "Bounded controller source",
        )
        blob = stream.read(CONTROLLER["size_bytes"] + 1)
        after = os.fstat(stream.fileno())

    def identity(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    require(
        identity(before) == identity(after) == identity(path.stat())
        and len(blob) == CONTROLLER["size_bytes"]
        and sha(blob) == CONTROLLER["sha256"],
        "Frozen controller bytes differ",
    )
    spec = importlib.util.spec_from_file_location("_revisit_external_observer_controller", path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    exec(compile(blob, str(path), "exec"), result.__dict__)
    return result


def sources_and_plan():
    controller = controller_module()
    binding = controller.pinned_module(BINDING, "_revisit_external_observer_binding")
    expected = binding.expected_sources()
    for name in binding.PINS:
        binding.checked(expected[name], ROOT)
    require(expected["controller_source"] == CONTROLLER, "Same frozen controller authority")
    plan, cpu, worker, cohort, files, actual = controller.load_plan(
        ROOT / expected["plan"]["path"], expected["plan"]["sha256"]
    )
    return controller, binding, expected, plan, cpu, worker, cohort, files, actual


def parent_argv(expected, python):
    return [
        python,
        "-B",
        "-s",
        str(ROOT / expected["controller_source"]["path"]),
        "--run",
        "--plan",
        str(ROOT / expected["plan"]["path"]),
        "--plan-sha256",
        expected["plan"]["sha256"],
    ]


def combined_caps(observer_output, actual, pending=0, *, terminal=False):
    """Observe both owned trees, not an atomic cross-process disk quota."""
    roots = [Path(p).absolute() for p in (observer_output, actual)]
    require(
        not roots[0].is_relative_to(roots[1]) and not roots[1].is_relative_to(roots[0]),
        "Disjoint owned output roots",
    )
    files = []
    for root in roots:
        require(
            not any(p.is_symlink() for p in (root, *root.parents)), "Ordinary owned output roots"
        )
        for path in root.rglob("*"):
            require(not path.is_symlink(), "Ordinary owned output population")
            if path.is_file():
                files.append(path)
    sizes = [p.stat().st_size for p in files]
    require(
        len(files) < 10000 and all(s <= MAX_FILE for s in sizes), "Combined file count/per-file cap"
    )
    require(
        sum(sizes) + pending <= MAX_OUTPUT - (0 if terminal else RESERVE),
        "Combined observed1GiB output/reserve cap",
    )


def combined_save(cpu, directory, label, blob, *, observer_output, actual, terminal=False):
    combined_caps(observer_output, actual, len(blob), terminal=terminal)
    saved = cpu.save(directory, label, blob, terminal=terminal)
    combined_caps(observer_output, actual, terminal=terminal)
    return saved


class CombinedMonitor:
    def __init__(self, cpu, output, boot, parent, actual):
        self.delegate = cpu.OwnedMemoryMonitor(output, boot, parent)
        self.output, self.actual = output, actual

    def check(self, rows):
        value = self.delegate.check(rows)
        combined_caps(self.output, self.actual)
        return value


def absence(table, parent, cleanup):
    rows = list(table.values())
    return {
        "parent_pid_absent": parent["pid"] not in table,
        "matching_parent_start_absent": not any(
            r["pid"] == parent["pid"] and r["start_ticks"] == parent["start_ticks"] for r in rows
        ),
        "PGID_members": [r for r in rows if r["pgid"] == parent["pgid"]],
        "SID_members": [r for r in rows if r["sid"] == parent["sid"]],
        "tracked_owned_PID_start_remaining": [
            r
            for r in rows
            if cleanup["tracked_PID_start_ticks"].get(str(r["pid"])) == r["start_ticks"]
            or r["pgid"] in cleanup["tracked_groups"]
        ],
    }


def external_record(
    binding,
    expected,
    *,
    config,
    outer,
    terminal_ref,
    identity,
    argv,
    sources,
    missing,
    started,
    finished,
    observed,
):
    value = {
        "schema_id": binding.EXTERNAL,
        "plan_sha256": expected["plan"]["sha256"],
        "nonce": config["nonce"],
        "boot_id": config["boot_id"],
        "parent": identity,
        "argv": argv,
        "returncode": 0,
        "source_before": sources,
        "source_after": {k: expected[k] for k in ("controller_source", "worker_source")},
        "controller_terminal": terminal_ref,
        "absence": missing,
        "started_unix": started,
        "finished_unix": finished,
        "observed_unix": observed,
    }
    binding.external_exit(
        value,
        expected=expected,
        selection={"plan_sha256": expected["plan"]["sha256"], "controller_terminal": terminal_ref},
        config=config,
        outer=outer,
        root=ROOT,
    )
    return value


def observe(output, expected_source=None, *, run=False):
    controller, binding, expected, plan, cpu, worker, cohort, files, actual = sources_and_plan()
    output = controller.ordinary(output)
    require(
        output.parent == HERE and output.name.startswith("cpu-observer-") and not output.exists(),
        "Fresh separate external observer output",
    )
    if not run:
        return {
            "schema_id": KIND,
            "status": "DRY_SOURCE_ONLY_NO_CHILD_PROCESSOR_OR_OUTPUT",
            "actual_output_exists": actual.exists(),
            "output_created": False,
            "benchmark_ready": False,
            "pending_selection": binding.pending_selection(),
        }
    self_ref = {
        "path": str(Path(__file__).absolute().relative_to(ROOT)),
        "sha256": expected_source,
        "size_bytes": Path(__file__).stat().st_size,
    }
    cpu.checked(self_ref)
    require(
        not any(
            n.split(".")[0] in {"torch", "transformers", "tokenizers", "peft", "s1"}
            for n in sys.modules
        ),
        "Fresh framework-free external observer",
    )
    require(
        all(os.environ.get(k) == v for k, v in plan["environment"].items()),
        "Exact original CPU environment for observer",
    )
    obs = cpu.memory_observation()
    formula = cpu.resource_formula(obs, obs["boot_id"], prospective=True)
    if obs["MemAvailable"] < formula["required_bytes"]:
        return {
            "schema_id": KIND,
            "status": "NOT_STARTED_ORIGINAL_RESOURCE_FLOOR_REFUSED",
            "observation": obs,
            "formula": formula,
            "process_started": False,
            "output_created": False,
            "benchmark_ready": False,
        }
    # Same component parent limits: its monitor includes this observer and every
    # observed parent/worker descendant, so the aggregate budget stays10GiB.
    cpu.configure_limits("parent")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    cpu.cuda_state()
    output.mkdir(mode=0o700)
    common = controller.component(controller.COMMON)
    identity = common.process_table()[os.getpid()]
    monitor_parent = {k: identity[k] for k in ("pid", "start_ticks", "pgid", "sid")}
    sources = {k: expected[k] for k in ("controller_source", "worker_source")}
    argv = parent_argv(expected, plan["python"])
    combined_save(
        cpu,
        output,
        "intent.json",
        cpu.encode(
            {
                "schema_id": KIND,
                "observer_source": self_ref,
                "binding_source": BINDING,
                "sources": sources,
                "plan": expected["plan"],
                "argv": argv,
                "observation": obs,
                "formula": formula,
                "outer_limits": controller.LIMITS,
                "reused_owned_component": CONTROLLER,
                "monitor_component_profile": cpu.PROFILE,
                "limitations": cpu.LIMITATIONS,
            }
        ),
        observer_output=output,
        actual=actual,
    )
    env = {k: os.environ[k] for k in cpu.SECRET_FREE_ENV if k in os.environ}
    env.update(plan["environment"])
    started = time.time()
    execution, monitor = controller.run_owned(
        cpu,
        common,
        output,
        argv,
        env,
        obs["boot_id"],
        {**monitor_parent, "mem_total": obs["MemTotal"]},
        monitor_factory=lambda out, boot, parent: CombinedMonitor(cpu, out, boot, parent, actual),
    )
    finished = time.time()
    primary, selection, selected = execution["primary_error"], binding.pending_selection(), None
    if primary is None and not execution["secondary_errors"] and execution["returncode"] == 0:
        try:
            require(
                execution["cleanup"] and execution["cleanup"]["confirmed"],
                "External whole owned cleanup",
            )
            parent = cpu.parse(cpu.read(output / "worker-identity.json"))
            parent = {k: parent[k] for k in ("pid", "start_ticks", "pgid", "sid")}
            for k in binding.PINS:
                binding.checked(expected[k], ROOT)
            cpu.checked(self_ref)
            worker.authenticate(files)
            config = cpu.parse(cpu.read(actual / "worker-config.json"))
            outer_blob = cpu.read(actual / "controller-terminal.json")
            outer = cpu.parse(outer_blob)
            terminal_ref = {
                "path": str((actual / "controller-terminal.json").relative_to(ROOT)),
                "sha256": sha(outer_blob),
                "size_bytes": len(outer_blob),
            }
            monitor.check({})
            cpu.validate_monitor_evidence(
                cpu.read(output / "owned-memory-monitor.jsonl"),
                obs["boot_id"],
                parent=monitor_parent,
                cleanup=execution["cleanup"],
            )
            controller.prospective(cpu, obs["boot_id"], obs["MemTotal"])
            missing = absence(common.process_table(), parent, execution["cleanup"])
            value = external_record(
                binding,
                expected,
                config=config,
                outer=outer,
                terminal_ref=terminal_ref,
                identity=parent,
                argv=argv,
                sources=sources,
                missing=missing,
                started=started,
                finished=finished,
                observed=time.time(),
            )
            saved = combined_save(
                cpu,
                actual,
                "external-parent-exit.json",
                cpu.encode(value),
                observer_output=output,
                actual=actual,
                terminal=True,
            )
            selection = {
                **selection,
                "controller_terminal": terminal_ref,
                "external_parent_exit": {
                    **saved,
                    "path": str((actual / saved["path"]).relative_to(ROOT)),
                },
            }
            selected = binding.validate_selection(selection, expected, root=ROOT)
            require(selected["closed_evidence"] is True, "Closed CPU evidence required")
            cpu.caps(actual)
            combined_save(
                cpu,
                output,
                "capacity-selection.json",
                cpu.encode(selection),
                observer_output=output,
                actual=actual,
            )
        except BaseException as error:
            primary, selected = cpu.error_value(error, "external-closed-evidence"), None
    result = {
        "schema_id": KIND,
        "status": "CLOSED_CPU_MEASUREMENT" if selected else "FAILED_OR_UNKNOWN",
        "primary_error": primary,
        "execution": execution,
        "selection": selection,
        "selected": selected,
        "benchmark_ready": bool(selected and selected["benchmark_ready"]),
        "task_quality_attested": False,
        "scope": "CPU measurement only. Successful exit/closed evidence can contain explicit policy/schema/context refusals and cannot start a browser without benchmark_ready plus independent live service admission.",
    }
    try:
        combined_save(
            cpu,
            output,
            "terminal.json",
            cpu.encode(result),
            observer_output=output,
            actual=actual,
            terminal=True,
        )
    except BaseException as error:
        if primary is not None:
            raise RuntimeError(
                f"{primary}; external observer terminal persistence also failed"
            ) from error
        raise
    return 0 if selected else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha256")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    require(
        not args.run or args.source_sha256 is not None,
        "Explicit reviewed observer source SHA for execution",
    )
    result = observe(args.output, args.source_sha256, run=args.run)
    if isinstance(result, dict):
        print(json.dumps(result))
        return 2 if args.run else 0
    return result


if __name__ == "__main__":
    raise SystemExit(main())
