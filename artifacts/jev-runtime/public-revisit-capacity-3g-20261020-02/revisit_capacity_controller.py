"""Default-dry3GiB whole-unit controller for the unchanged68+7 CPU cohort.

Reuses pinned3GiB CPU containment and matrix process ownership as components.
Success requires an external observer to confirm this controller's exit/closure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import resource
import signal
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
KIND = "jev_revisit_cpu_processor_capacity_controller_3g_v2"
PROFILE = "composed-visible-hover-entries-settled-revisit-v1"
PYTHON = "/home/albert_hsieh/ComfyUI/.venv/bin/python"
CPU = "artifacts/jev-runtime/public-revisit-capacity-3g-20261020-02/cpu_prepare_form_3g.py"
COMMON = "artifacts/jev-runtime/public-affordance-matrix-harness-20261019-01/matrix_common.py"
PINS = {
    CPU: ("85fa2f6894ee92cc7947e2da65bf2557794d6d6c935020051abb5dc34c3b70a8", 70678),
    COMMON: ("71251036c63e8df852ae22814d7234db0daaee90436c265e8a5549de468c6291", 7229),
}
MIB = 1024**2
LIMITS = {
    "parent_AS_soft": 2 * 1024**3,
    "parent_AS_hard": 8 * 1024**3,
    "worker_AS_soft_and_hard": 8 * 1024**3,
    "tracked_parent_and_owned_RSS": 3 * 1024**3,
    "resource_profile": "CPU_REFERENCE_ONLY_3G_V2",
    "whole_unit_memory_max": 3 * 1024**3,
    "whole_unit_swap_max": 0,
    "wall_seconds": 600,
    "monitor_interval_seconds": 0.05,
    "output_bytes": 1024**3,
    "file_bytes": 64 * MIB,
    "sampled_log_bytes_each": MIB,
    "terminal_reserve_bytes": MIB,
    "swap_or_resident_credit_bytes": 0,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def ordinary(path):
    path = Path(path).absolute()
    require(path.is_relative_to(ROOT) and ".." not in path.parts, "Repository confined path")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Ordinary path")
    return path


def pinned_module(ref, name):
    require(
        type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"}, "Typed module reference"
    )
    require(
        type(ref["path"]) is str
        and not Path(ref["path"]).is_absolute()
        and type(ref["size_bytes"]) is int
        and 0 < ref["size_bytes"] <= MIB
        and type(ref["sha256"]) is str
        and re.fullmatch("[0-9a-f]{64}", ref["sha256"]),
        "Bounded module reference fields",
    )
    path = ordinary(ROOT / ref["path"])
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and before.st_size == ref["size_bytes"],
            "Bounded regular module source",
        )
        blob = stream.read(ref["size_bytes"] + 1)
        after = os.fstat(stream.fileno())

    def signature(s):
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    require(
        signature(before) == signature(after) == signature(path.stat()),
        "Stable module source identity",
    )
    require(len(blob) == ref["size_bytes"] and sha(blob) == ref["sha256"], "Source pin differs")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(blob, str(path), "exec"), module.__dict__)
    return module


def component(label):
    digest, size = PINS[label]
    return pinned_module(
        {"path": label, "sha256": digest, "size_bytes": size}, "_revisit_" + Path(label).stem
    )


def load_plan(path, expected):
    cpu = component(CPU)
    blob = cpu.read(ordinary(path), 2 * MIB)
    require(sha(blob) == expected, "Externally selected controller plan SHA")
    plan = cpu.parse(blob)
    require(
        set(plan)
        == {
            "schema_id",
            "profile",
            "python",
            "environment",
            "parent_source",
            "worker_source",
            "cohort",
            "technical_plan",
            "processor_plan",
            "output",
            "limits",
        },
        "Strict controller plan keys",
    )
    require(
        plan["schema_id"] == KIND and plan["profile"] == PROFILE, "New controller/profile identity"
    )
    require(
        plan["python"] == PYTHON
        and plan["environment"] == cpu.EXPECTED_ENVIRONMENT
        and plan["limits"] == LIMITS,
        "Original CPU environment/containment",
    )
    require(
        ordinary(ROOT / plan["parent_source"]["path"]) == Path(__file__).absolute(),
        "Exact controller origin",
    )
    cpu.checked(plan["parent_source"])
    require(
        ordinary(ROOT / plan["worker_source"]["path"]) == HERE / "revisit_capacity_worker.py",
        "Exact new worker origin",
    )
    worker = pinned_module(plan["worker_source"], "_revisit_selected_worker")
    for key, label in (
        ("cohort", worker.COHORT),
        ("technical_plan", worker.TECHNICAL),
        ("processor_plan", worker.PROCESSOR),
    ):
        require(plan[key] == worker.fixed(label), "Original immutable input selection")
    cohort, technical = (cpu.parse(cpu.checked(plan[k])) for k in ("cohort", "technical_plan"))
    cpu.checked(plan["processor_plan"])
    worker.slots(cohort)
    files = worker.source_union(plan, cohort, technical)
    worker.authenticate(files)
    component(COMMON)
    output = ordinary(ROOT / plan["output"])
    require(
        output.parent == HERE and output.name.startswith("actual-"),
        "Separate confined actual namespace",
    )
    require(not output.exists(), "Fresh output namespace")
    return plan, cpu, worker, cohort, files, output


def prospective(cpu, boot, total):
    unit = cpu.assert_current_unit()
    obs = cpu.memory_observation()
    formula = cpu.resource_formula(obs, boot, prospective=True)
    require(
        obs["MemTotal"] == total and obs["MemAvailable"] >= formula["required_bytes"],
        "Prospective3GiB whole-unit cap plus unchanged host reserve refused",
    )
    return {"observation": obs, "unit": unit, **formula}


def run_owned(cpu, common, output, argv, env, boot, parent, *, wall=600, monitor_factory=None):
    """Always close the exact child tree; tiny-child tests inject only the monitor."""
    process, primary, cleanup = None, None, None
    known, groups, secondary, handlers = {}, set(), [], {}
    monitor_factory = cpu.OwnedMemoryMonitor if monitor_factory is None else monitor_factory
    monitor = None

    def interrupt(number, _frame):
        raise InterruptedError(f"Owned controller signal {number}")

    try:
        monitor = monitor_factory(output, boot, parent)
        monitor.check({})
        handlers = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
        with (
            (output / "child-stdout.txt").open("xb") as stdout,
            (output / "child-stderr.txt").open("xb") as stderr,
        ):
            os.chmod(stdout.name, 0o600)
            os.chmod(stderr.name, 0o600)
            # A fresh check immediately precedes the only child creation.
            prospective(cpu, boot, parent["mem_total"])
            process = subprocess.Popen(
                argv,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            identity = common.process_table().get(process.pid)
            require(
                identity is not None and identity["pgid"] == identity["sid"] == process.pid,
                "Fresh owned worker PID/PG/SID",
            )
            known[process.pid], groups = identity["start_ticks"], {process.pid}
            require(cpu.child_argv(process.pid) == argv, "Exact owned worker argv")
            cpu.save(output, "worker-identity.json", cpu.encode(identity))
            deadline, recorded = time.monotonic() + wall, set()
            while True:
                rows, _ = common.extend_owned(process.pid, known, groups)
                for pid, row in rows.items():
                    if (pid, row["start_ticks"]) not in recorded:
                        cpu.save(
                            output, f"processes/{pid}-{row['start_ticks']}.json", cpu.encode(row)
                        )
                        recorded.add((pid, row["start_ticks"]))
                if process.poll() is not None:
                    break
                require(time.monotonic() < deadline, "Owned worker600s hard deadline")
                monitor.check(rows)
                cpu.caps(output)
                require(
                    stdout.tell() < MIB and stderr.tell() < MIB, "Sampled stdout/stderr1MiB limit"
                )
                time.sleep(0.05)
            require(stdout.tell() < MIB and stderr.tell() < MIB, "Final stdout/stderr1MiB limit")
    except BaseException as error:
        primary = cpu.error_value(error, "owned-execution")
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if process is not None:
            try:
                if not known:
                    identity = common.process_table().get(process.pid)
                    if identity and identity["pgid"] == identity["sid"] == process.pid:
                        known[process.pid], groups = identity["start_ticks"], {process.pid}
                cleanup = common.stop_owned(process, known, groups)
                cpu.save(output, "owned-cleanup.json", cpu.encode(cleanup), terminal=True)
            except BaseException as error:
                secondary.append(cpu.error_value(error, "owned-cleanup"))
    return {
        "started": process is not None,
        "returncode": process.returncode if process else None,
        "primary_error": primary,
        "secondary_errors": secondary,
        "cleanup": cleanup,
    }, monitor


def validate_worker(cpu, worker, cohort, output, config_sha, nonce, cleanup):
    require(
        cleanup and cleanup["confirmed"] is True and cleanup["remaining"] == [],
        "Whole owned closure required",
    )
    root = output / "worker"
    config, selected_cohort, technical, processor = worker.load_config(
        output / "worker-config.json", config_sha
    )
    require(
        config["nonce"] == nonce and config["output"] == str(output) and selected_cohort == cohort,
        "Exact retained worker configuration",
    )
    gate_evidence = worker.validate_gates(cpu.read(root / "resource-gates.jsonl"), config)
    union = worker.source_union(config, cohort, technical)
    union_receipt = cpu.parse(cpu.read(root / "source-union.json"))
    require(
        union_receipt
        == {
            "schema_id": worker.KIND + "_source_union",
            "files": union,
            "count": len(union),
            "old95_modified": False,
        },
        "Complete new mechanical source union",
    )
    for label, pin in processor["reference_items"].items():
        worker.checked({"path": label, **pin}, root / "reference-environment")
    runtime = cpu.parse(cpu.read(root / "runtime-component.json"))
    require(
        runtime
        == {
            "runtime": technical["technical"]["runtime"],
            "runtime_files": 348,
            "original_execution": technical["technical"]["execution"],
            "original_execution_files": 95,
            "processor_plan": config["processor_plan"],
            "processor_reference_items": 58,
            "loaded_mandatory_generation_imports_requested": False,
        },
        "Original runtime/reference component authorities",
    )
    before = cpu.read(root / "processor-evidence-before.json")
    after = cpu.read(root / "processor-evidence-after.json")
    evidence = cpu.parse(before)
    require(
        before == after
        and evidence["tokenizer_is_processor_tokenizer"] is True
        and evidence["max_context"] == 8192
        and evidence["generation_EOS"] == [1, 106, 50]
        and evidence["CPU_threads"] == evidence["CPU_interop_threads"] == 2,
        "Stable original processor and two-thread evidence",
    )
    blob = cpu.read(root / "terminal.json")
    terminal = cpu.parse(blob)
    require(
        terminal["schema_id"] == worker.KIND
        and terminal["profile"] == PROFILE
        and terminal["config_sha256"] == config_sha
        and terminal["nonce"] == nonce,
        "Exact worker receipt binding",
    )
    child = terminal["child"]
    require(
        cleanup["tracked_PID_start_ticks"].get(str(child["pid"])) == child["start_ticks"]
        and child["pgid"] == child["sid"] == child["pid"]
        and child["pgid"] in cleanup["tracked_groups"],
        "Receipt joins closed worker identity",
    )
    require(
        terminal["complete"] is True
        and terminal["status"] == "CPU_PREFIXES_COMPLETE_AWAIT_WHOLE_OWNED_CLOSURE"
        and terminal["primary_error"] is None
        and terminal["secondary_errors"] == [],
        "Complete worker with no errors",
    )
    require(
        terminal["processor_capacity_only"] is True
        and terminal["task_quality_attested"] is False
        and terminal["universal_profile_capacity_attested"] is False
        and terminal["native_model_constructor_forward_HTTP_browser_calls"] == 0,
        "CPU-only scoped evidence",
    )
    rows = terminal["rows"]
    expected = worker.slots(cohort)
    require(len(rows) == 75, "All75 rows retained")
    validator = worker.module(worker.VALIDATOR)
    for row, slot in zip(rows, expected, strict=True):
        require(
            all(row[k] == slot[k] for k in ("ordinal", "id", "kind", "native_forward_calls")),
            "Exact ordered row identity",
        )
        require(
            row["expectation_met"] is True and row["outcome"] != "UNKNOWN",
            "Complete explicit row outcome",
        )
        require(
            cpu.parse(cpu.read(root / f"rows/{row['ordinal']:02d}.json")) == row,
            "Terminal joins saved row bytes",
        )
        payload_bytes = worker.checked(row["payload"], root)
        if row["ordinal"] < 68:
            selection = cohort["rows"][row["ordinal"]]
            require(
                row["provenance"] == {"scope": selection["scope"], "cohort_selection": selection},
                "Exact scoped cohort provenance",
            )
            require(
                payload_bytes
                == worker.checked(selection["payload_ref"], (ROOT / worker.COHORT).parent),
                "Full ordered cohort payload byte equality",
            )
        if row["prefix"] is not None:
            prompt = worker.checked(row["prompt"], root).decode()
            validator.validate_prefix(row["prefix"], prompt)
            require(
                row["outcome"] == validator.outcome(row["prefix"], row["kind"])
                and row["error"] is None,
                "Full IDs/mask and context outcome",
            )
        else:
            require(
                row["outcome"] == "REFUSED_SCHEMA" and row["error"] is not None,
                "Explicit schema refusal",
            )
        acceptable = (
            {"ACCEPTED", "REFUSED_SCHEMA", "REFUSED_CONTEXT"}
            if row["ordinal"] < 68
            else validator.expected_outcomes(row["id"])
        )
        require(row["outcome"] in acceptable, "Technical boundary expectation")
    counts = {
        name: sum(r["outcome"] == name for r in rows)
        for name in ("ACCEPTED", "REFUSED_SCHEMA", "REFUSED_CONTEXT", "UNKNOWN")
    }
    require(counts == terminal["counts"], "Mechanically recounted75 outcomes")
    return {
        "terminal_sha256": sha(blob),
        "worker_resource_gates": gate_evidence,
        "processor_evidence_sha256": sha(before),
        "counts": counts,
        "rows": rows,
    }


def controller(path, expected, *, run=False):
    plan, cpu, worker, cohort, files, output = load_plan(path, expected)
    if not run:
        return {
            "schema_id": KIND,
            "status": "DRY_SOURCE_ONLY_NO_PROCESSOR_CHILD_OR_OUTPUT",
            "rows": 75,
            "new_source_union_files": len(files),
        }
    require(
        not any(
            n.split(".")[0] in {"torch", "transformers", "tokenizers", "peft", "s1"}
            for n in sys.modules
        ),
        "Fresh framework-free parent",
    )
    require(
        all(os.environ.get(k) == v for k, v in plan["environment"].items()),
        "Exact parent CPU environment",
    )
    cpu.configure_limits("parent")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * MIB, 64 * MIB))
    cpu.cuda_state()
    obs = cpu.memory_observation()
    admission = prospective(cpu, obs["boot_id"], obs["MemTotal"])
    output.mkdir(mode=0o700)
    common = component(COMMON)
    parent = common.process_table()[os.getpid()]
    parent_fields = {k: parent[k] for k in ("pid", "start_ticks", "pgid", "sid")}
    require(parent["pgid"] == parent["sid"] == parent["pid"], "Controller owns fresh PG/SID")
    nonce = uuid.uuid4().hex
    config = {
        "schema_id": worker.KIND,
        **{
            k: plan[k]
            for k in (
                "cohort",
                "technical_plan",
                "processor_plan",
                "worker_source",
                "parent_source",
            )
        },
        "owner": parent_fields,
        "nonce": nonce,
        "boot_id": obs["boot_id"],
        "mem_total": obs["MemTotal"],
        "unit": {k: admission["unit"][k] for k in ("boot_id", "cgroup_path")},
        "output": str(output),
    }
    config_ref = cpu.save(output, "worker-config.json", cpu.encode(config))
    argv = [
        PYTHON,
        "-B",
        "-s",
        str(HERE / "revisit_capacity_worker.py"),
        "--config",
        str(output / "worker-config.json"),
        "--config-sha256",
        config_ref["sha256"],
        "--run",
    ]
    cpu.save(
        output,
        "controller-intent.json",
        cpu.encode(
            {
                "schema_id": KIND,
                "plan_sha256": expected,
                "nonce": nonce,
                "parent": parent_fields,
                "admission": admission,
                "argv": argv,
                "limits": LIMITS,
                "monitor_reused_component_profile": cpu.PROFILE,
                "limitations": cpu.LIMITATIONS,
            }
        ),
    )
    env = {k: os.environ[k] for k in cpu.SECRET_FREE_ENV if k in os.environ}
    env.update(plan["environment"])
    execution, monitor = run_owned(
        cpu,
        common,
        output,
        argv,
        env,
        obs["boot_id"],
        {**parent_fields, "mem_total": obs["MemTotal"]},
    )
    selected, primary = None, execution["primary_error"]
    if primary is None and not execution["secondary_errors"] and execution["returncode"] == 0:
        try:
            selected = validate_worker(
                cpu, worker, cohort, output, config_ref["sha256"], nonce, execution["cleanup"]
            )
            require(
                sha(cpu.read(ordinary(path), 2 * MIB)) == expected,
                "Controller plan unchanged after child",
            )
            for key in (
                "cohort",
                "technical_plan",
                "processor_plan",
                "parent_source",
                "worker_source",
            ):
                cpu.checked(plan[key])
            worker.authenticate(files)
            monitor.check({})
            prospective(cpu, obs["boot_id"], obs["MemTotal"])
            cpu.caps(output)
            cpu.validate_monitor_evidence(
                cpu.read(output / "owned-memory-monitor.jsonl"),
                obs["boot_id"],
                parent=parent_fields,
                cleanup=execution["cleanup"],
                unit=config["unit"],
            )
        except BaseException as error:
            primary, selected = cpu.error_value(error, "closed-evidence-validation"), None
    result = {
        "schema_id": KIND,
        "profile": PROFILE,
        "plan_sha256": expected,
        "nonce": nonce,
        "parent": parent_fields,
        "status": "CLOSED_CPU_COHORT_PENDING_EXTERNAL_PARENT_EXIT"
        if selected
        else "FAILED_OR_UNKNOWN",
        "primary_error": primary,
        "execution": execution,
        "selected": selected,
        "task_quality_attested": False,
        "universal_profile_capacity_attested": False,
        "external_parent_exit0_and_PID_start_PG_SID_absence_required": True,
    }
    try:
        cpu.save(output, "controller-terminal.json", cpu.encode(result), terminal=True)
    except BaseException as error:
        if primary is not None:
            raise RuntimeError(f"{primary}; terminal persistence also failed") from error
        raise
    return 0 if selected else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args(argv)
    result = controller(args.plan, args.plan_sha256, run=args.run)
    if isinstance(result, dict):
        print(json.dumps(result))
        return 0
    return result


if __name__ == "__main__":
    raise SystemExit(main())
