"""Pure selector for the new closed75 CPU evidence; no service or browser actor.

Pending references never admit a benchmark. Closed evidence and acceptance of
all68 policy bodies are separate results. Existing technical authorities remain
unchanged and do not substitute for this new prompt measurement.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
KIND = "jev_revisit_closed_cpu_capacity_binding_v1"
PROFILE = "composed-visible-hover-entries-settled-revisit-v1"
SELECTION = "jev_revisit_closed_cpu_capacity_selection_v1"
EXTERNAL = "jev_revisit_cpu_capacity_external_parent_exit_v1"
CPU_BASE = "artifacts/jev-runtime/public-revisit-capacity-execution-20261020-01"
PINS = {
    "controller_source": (
        CPU_BASE + "/revisit_capacity_controller.py",
        "ff31c2765c18ebf1aae92e226396ed91c15f744071b1dc5a8ce8e630b1f351b4",
        21290,
    ),
    "worker_source": (
        CPU_BASE + "/revisit_capacity_worker.py",
        "5ee6f5d15a492bb95b3b21d932a12ca919b3a3f83fa0fdc23264b982b9ece288",
        27546,
    ),
    "plan": (
        CPU_BASE + "/plan-01.json",
        "c0289c5ac7c11aa3b80c5c9c2b685eb09a907d01e5d2c57c4525286a151bc15b",
        2501,
    ),
    "cohort": (
        "artifacts/jev-runtime/public-revisit-capacity-payloads-20261020-01/prepared-01/payload-cohort.json",
        "c39984b1af50d49686a79492f78f618fa37f590aabc423934ee4d8da9d793091",
        146149,
    ),
    "profile_manifest": (
        "artifacts/jev-runtime/public-revisit-profile-20261020-01/profile-01/profile-manifest.json",
        "f3e65535836518581deab8487f6c20c837bcecfa03173db85f953cc02975ce88",
        12379,
    ),
    "profile_helper": (
        "scripts/jev_observation_profile_revisit.py",
        "873abc6aa2a02272fd9b54d299d683598bdb3d5c0a848cc43a356daa5a36a0ca",
        13642,
    ),
    "launcher_source": (
        "scripts/jev_browser_revisit_benchmark.py",
        "f7f0c4f708203e107563fa70656bbeb3e04afcba463a0a9cf9eb156fbd8e51b3",
        13193,
    ),
}
MAX_FILE = 64 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def parse(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(blob, object_pairs_hook=pairs, parse_constant=lambda n: require(False, n))


def ordinary(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    require(path.is_relative_to(root) and ".." not in path.parts, "Repository confined path")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Ordinary source/evidence path")
    return path


def read(path, root, maximum=MAX_FILE):
    path = ordinary(path, root)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(
            stat.S_ISREG(before.st_mode) and before.st_size <= maximum, "Bounded regular evidence"
        )
        chunks, remaining = [], before.st_size
        while remaining:
            part = os.read(fd, min(1024**2, remaining))
            require(bool(part), "Complete evidence read")
            chunks.append(part)
            remaining -= len(part)

        def identity(s):
            return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

        require(
            identity(before) == identity(os.fstat(fd)) == identity(path.stat()),
            "Stable evidence descriptor",
        )
        return b"".join(chunks)
    finally:
        os.close(fd)


def checked(ref, root):
    require(
        type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"},
        "Typed evidence reference",
    )
    require(
        type(ref["path"]) is str
        and not Path(ref["path"]).is_absolute()
        and type(ref["size_bytes"]) is int
        and 0 <= ref["size_bytes"] <= MAX_FILE
        and type(ref["sha256"]) is str
        and len(ref["sha256"]) == 64
        and all(c in "0123456789abcdef" for c in ref["sha256"]),
        "Bounded reference fields",
    )
    blob = read(Path(root) / ref["path"], root)
    require(
        len(blob) == ref["size_bytes"] and sha(blob) == ref["sha256"],
        "Exact selected evidence SHA/size",
    )
    return blob


def expected_sources():
    return {
        "schema_id": KIND + "_expected",
        "profile": PROFILE,
        **{
            name: {"path": path, "sha256": digest, "size_bytes": size}
            for name, (path, digest, size) in PINS.items()
        },
    }


def pending_selection():
    return {
        "schema_id": SELECTION,
        "kind": KIND,
        "profile": PROFILE,
        "plan_sha256": PINS["plan"][1],
        "controller_terminal": None,
        "external_parent_exit": None,
    }


def module(ref, root, name):
    blob = checked(ref, root)
    path = Path(root) / ref["path"]
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    exec(compile(blob, str(path), "exec"), result.__dict__)
    return result


def components(expected, root):
    controller = module(
        expected["controller_source"], root, "_revisit_capacity_selected_controller"
    )
    cpu = controller.component(controller.CPU)
    worker = controller.pinned_module(
        expected["worker_source"], "_revisit_capacity_selected_worker"
    )
    return controller, cpu, worker


def external_exit(value, *, expected, selection, config, outer, root):
    require(
        type(value) is dict
        and set(value)
        == {
            "schema_id",
            "plan_sha256",
            "nonce",
            "boot_id",
            "parent",
            "argv",
            "returncode",
            "source_before",
            "source_after",
            "controller_terminal",
            "absence",
            "started_unix",
            "finished_unix",
            "observed_unix",
        },
        "Strict external parent exit schema",
    )
    require(
        value["schema_id"] == EXTERNAL
        and value["plan_sha256"] == selection["plan_sha256"]
        and value["nonce"] == config["nonce"] == outer["nonce"]
        and value["boot_id"] == config["boot_id"],
        "Exact external plan/nonce/boot binding",
    )
    parent = value["parent"]
    require(
        type(parent) is dict
        and set(parent) == {"pid", "start_ticks", "pgid", "sid"}
        and all(type(v) is int and v > 0 for v in parent.values())
        and parent["pid"] == parent["pgid"] == parent["sid"]
        and parent == config["owner"] == outer["parent"],
        "Exact externally closed parent PID/start/PG/SID",
    )
    require(
        type(value["returncode"]) is int and value["returncode"] == 0,
        "External actual parent exit0 required",
    )
    require(
        value["argv"]
        == [
            "/home/albert_hsieh/ComfyUI/.venv/bin/python",
            "-B",
            "-s",
            str(Path(root) / expected["controller_source"]["path"]),
            "--run",
            "--plan",
            str(Path(root) / expected["plan"]["path"]),
            "--plan-sha256",
            expected["plan"]["sha256"],
        ],
        "Exact externally observed parent argv",
    )
    sources = {key: expected[key] for key in ("controller_source", "worker_source")}
    require(
        value["source_before"] == value["source_after"] == sources
        and value["controller_terminal"] == selection["controller_terminal"],
        "External source before/after and exact terminal byte pin",
    )
    require(
        value["absence"]
        == {
            "parent_pid_absent": True,
            "matching_parent_start_absent": True,
            "PGID_members": [],
            "SID_members": [],
            "tracked_owned_PID_start_remaining": [],
        }
        and value["absence"]["parent_pid_absent"] is True
        and value["absence"]["matching_parent_start_absent"] is True,
        "External whole parent and owned process absence required",
    )
    times = [value[k] for k in ("started_unix", "finished_unix", "observed_unix")]
    require(
        all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in times)
        and times == sorted(times),
        "Finite ordered external observation times",
    )


def validate_selection(selection, expected, *, root):
    require(Path(root).absolute() == ROOT, "Exact reviewed repository root")
    require(expected == expected_sources(), "Fixed new CPU/profile/launcher authorities")
    require(
        type(selection) is dict
        and set(selection) == set(pending_selection())
        and all(
            selection[key] == pending_selection()[key]
            for key in ("schema_id", "kind", "profile", "plan_sha256")
        ),
        "New typed selection; old23 kind never accepted",
    )
    for name in PINS:
        checked(expected[name], root)
    if selection["controller_terminal"] is None or selection["external_parent_exit"] is None:
        return {
            "kind": KIND,
            "profile": PROFILE,
            "status": "NOT_READY_MISSING_CLOSED_CPU_EVIDENCE",
            "closed_evidence": False,
            "benchmark_ready": False,
            "selection": copy.deepcopy(selection),
            "task_quality_attested": False,
            "universal_profile_capacity_attested": False,
        }
    plan = parse(checked(expected["plan"], root))
    cohort = parse(checked(expected["cohort"], root))
    output = ordinary(Path(root) / plan["output"], root)
    require(
        output.parent == Path(root) / CPU_BASE and output.name == "actual-01",
        "Actual tree only from fixed CPU plan",
    )
    for key, name in (
        ("controller_terminal", "controller-terminal.json"),
        ("external_parent_exit", "external-parent-exit.json"),
    ):
        require(
            ordinary(Path(root) / selection[key]["path"], root) == output / name,
            "One exact closed CPU output tree",
        )
    outer = parse(checked(selection["controller_terminal"], root))
    external = parse(checked(selection["external_parent_exit"], root))
    require(
        outer["schema_id"] == "jev_revisit_cpu_processor_capacity_controller_v1"
        and outer["profile"] == PROFILE
        and outer["plan_sha256"] == expected["plan"]["sha256"]
        and outer["status"] == "CLOSED_CPU_COHORT_PENDING_EXTERNAL_PARENT_EXIT"
        and outer["primary_error"] is None
        and outer["selected"] is not None
        and outer["task_quality_attested"] is False
        and outer["universal_profile_capacity_attested"] is False
        and outer["external_parent_exit0_and_PID_start_PG_SID_absence_required"] is True,
        "Successful provisional controller receipt",
    )
    execution = outer["execution"]
    require(
        execution["started"] is True
        and type(execution["returncode"]) is int
        and execution["returncode"] == 0
        and execution["primary_error"] is None
        and execution["secondary_errors"] == [],
        "Exact successful owned worker execution",
    )
    controller, cpu, worker = components(expected, root)
    config_path = output / "worker-config.json"
    config_blob = read(config_path, root, 2 * 1024**2)
    config = parse(config_blob)
    external_exit(
        external, expected=expected, selection=selection, config=config, outer=outer, root=root
    )
    require(
        config["cohort"] == expected["cohort"] and config["output"] == str(output),
        "New cohort and retained config output join",
    )
    selected = controller.validate_worker(
        cpu, worker, cohort, output, sha(config_blob), config["nonce"], execution["cleanup"]
    )
    require(
        canonical(selected) == canonical(outer["selected"]),
        "Independently recomputed whole75 evidence equals parent selection",
    )
    monitor_rows = cpu.validate_monitor_evidence(
        read(output / "owned-memory-monitor.jsonl", root),
        config["boot_id"],
        parent=outer["parent"],
        cleanup=execution["cleanup"],
    )
    worker.authenticate(
        worker.source_union(config, cohort, parse(checked(config["technical_plan"], root)))
    )
    helper = module(expected["profile_helper"], root, "_revisit_capacity_selected_profile")
    helper.verify_materialized_profile(
        Path(root) / expected["profile_manifest"]["path"].rsplit("/", 1)[0],
        expected_manifest_sha256=expected["profile_manifest"]["sha256"],
        expected_source_sha256=expected["profile_helper"]["sha256"],
    )
    require(
        cohort["profile"] == PROFILE
        and cohort["profile_manifest"]
        == {
            **expected["profile_manifest"],
            "path": str(Path(root) / expected["profile_manifest"]["path"]),
        },
        "Exact new cohort/runtime profile identity",
    )
    rows = selected["rows"]
    refusals = [
        {"ordinal": r["ordinal"], "id": r["id"], "outcome": r["outcome"], "error": r["error"]}
        for r in rows[:68]
        if r["outcome"] != "ACCEPTED"
    ]
    ready = not refusals
    return {
        "kind": KIND,
        "profile": PROFILE,
        "status": "CLOSED_COHORT_POLICY68_ACCEPTED"
        if ready
        else "CLOSED_COHORT_POLICY_REFUSALS_BENCHMARK_NOT_READY",
        "closed_evidence": True,
        "benchmark_ready": ready,
        "selection": copy.deepcopy(selection),
        "counts": selected["counts"],
        "required_policy_ids": [r["id"] for r in rows[:68]],
        "required_policy_denominator": 68,
        "policy_refusals": refusals,
        "cohort_scopes": cohort["counts"],
        "technical_edges_denominator": 7,
        "monitor_rows": monitor_rows,
        "worker_resource_gates": selected["worker_resource_gates"],
        "profile_manifest": copy.deepcopy(expected["profile_manifest"]),
        "task_quality_attested": False,
        "universal_profile_capacity_attested": False,
        "scope": "Only these68 complete policy bodies plus7 separately labeled technical edges; counterfactual sequences and stress repetition are not new live browser observations.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if not args.validate:
        print(
            json.dumps(
                {
                    "kind": KIND,
                    "status": "DRY_SOURCE_ONLY_NOT_ADMITTED",
                    "benchmark_ready": False,
                    "pending_selection": pending_selection(),
                    "processor_model_service_browser_calls": 0,
                }
            )
        )
        return 0
    require(
        args.selection is not None and args.selection_sha256 is not None,
        "Explicit selected pure evidence bytes required",
    )
    blob = read(args.selection, ROOT, 1024**2)
    require(sha(blob) == args.selection_sha256, "External selection SHA")
    result = validate_selection(parse(blob), expected_sources(), root=ROOT)
    print(json.dumps(result))
    return 0 if result["benchmark_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
