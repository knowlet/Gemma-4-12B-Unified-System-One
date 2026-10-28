"""Held launcher for one fixed synthetic CUDA request under an external owner.

This file loads no collection, case, scorer or browser input. Its parent must
provide the independently monitored systemd unit, hard wall, and cleanup audit.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from pathlib import Path

SOURCE_ONLY = True
SCHEMA = "decision3-cuda-synthetic-plan-v1"
ROOT = Path(__file__).absolute().parent.parent
BASE = ROOT / "artifacts/jev-runtime/decision3-cuda-synthetic"
SCRIPTS = ROOT / "scripts"
# Injected only after bootstrap authenticates and executes selected source bytes.
child = common = cpu = None
PROFILE = {"cpus": frozenset(range(15, 20)), "cpu_list": "15-19", "threads": 5}
WALL_SECONDS = 120
FILES = {
    "bootstrap": "jev_decision3_cuda_synthetic_bootstrap.py",
    "launcher": "jev_decision3_cuda_synthetic_launcher.py",
    "supervisor": "jev_decision3_cuda_synthetic_supervisor.py",
    "worker": "jev_decision3_cuda_synthetic_worker.py",
}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _release(name):
    _need(name in FILES, "known source required")
    return SCRIPTS / (FILES[name].removesuffix(".py") + "_release.py")


def _source_set():
    names = [SCRIPTS / name for name in FILES.values()]
    names += [_release(name) for name in FILES]
    names += [
        SCRIPTS / "jev_decision3_candidate_browser_child.py",
        SCRIPTS / "jev_decision3_candidate_http_owner.py",
        SCRIPTS / "jev_decision3_cpu_experiment_worker.py",
        SCRIPTS / "jev_stage_memory_profile_v2.py",
        cpu.STAGED / "d3_runtime.py",
        cpu.STAGED / "d3_format.py",
        cpu.STAGED / "modeling_d3.py",
        cpu.SNAPSHOT / "d3_fast.py",
        cpu.OVERLAY / "transformers/__init__.py",
        cpu.OVERLAY / "transformers/models/qwen3_5/modeling_qwen3_5.py",
        cpu.TORCH,
        Path(common.GPU_PYTHON_REAL),
    ]
    names += [
        cpu.ROOT / "src/s1" / name
        for name in (
            "__init__.py",
            "backends.py",
            "contracts.py",
            "errors.py",
            "evaluation/__init__.py",
            "evaluation/contracts.py",
            "evaluation/decision3.py",
        )
    ]
    return {str(path) for path in names}


def _plan(path, digest):
    _need(
        type(path) is str and type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest),
        "explicit plan identity required",
    )
    selected = Path(path)
    _need(
        selected.is_absolute()
        and selected.name == "plan.json"
        and selected.parent.parent == BASE
        and re.fullmatch(r"run-[0-9a-f]{32}", selected.parent.name)
        and selected.resolve() == selected,
        "dedicated ordinary plan path required",
    )
    raw = common._read_reviewed_source(selected, digest)
    _need(len(raw) <= 1024 * 1024, "bounded plan required")
    plan = common._decode(raw)
    _need(
        type(plan) is dict
        and set(plan)
        == {
            "schema",
            "unit",
            "output_dir",
            "python",
            "source_pins",
            "gpu_uuid",
            "wall_seconds",
            "memory_floor_bytes",
        }
        and plan["schema"] == SCHEMA
        and plan["unit"]
        == "jev-cuda-synthetic-" + selected.parent.name.removeprefix("run-") + ".service"
        and plan["output_dir"] == str(selected.parent / "worker")
        and plan["python"] == common.GPU_PYTHON
        and type(plan["wall_seconds"]) is int
        and plan["wall_seconds"] == WALL_SECONDS
        and type(plan["memory_floor_bytes"]) is int
        and plan["memory_floor_bytes"] >= common.MEMORY_FLOOR
        and type(plan["gpu_uuid"]) is str
        and re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", plan["gpu_uuid"]
        ),
        "exact fixed synthetic plan required",
    )
    pins = plan["source_pins"]
    _need(
        type(pins) is dict
        and set(pins) == _source_set()
        and all(
            type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) for value in pins.values()
        ),
        "exact public source allowlist required",
    )
    return plan


def _verify_release(name, pins):
    held = SCRIPTS / FILES[name]
    released = _release(name)
    source = common._read_reviewed_source(held, pins[str(held)])
    actual = common._read_reviewed_source(released, pins[str(released)])
    marker = b"SOURCE_ONLY" + b" = True"
    _need(
        source.count(marker) == 1 and actual == source.replace(marker, b"SOURCE_ONLY = False", 1),
        "gate-only release required",
    )


def _source_guard(plan):
    common._verify_source_pins(plan["source_pins"])
    cpu._verify_model_manifest(plan["source_pins"])
    for name in FILES:
        _verify_release(name, plan["source_pins"])
    for name, module in tuple(sys.modules.items()):
        if (
            name == "s1"
            or name.startswith("s1.")
            or (
                name.startswith("transformers_modules.")
                and name.rsplit(".", 1)[-1] in {"modeling_d3", "d3_runtime", "d3_format", "d3_fast"}
            )
        ):
            path = getattr(module, "__file__", None)
            _need(
                type(path) is str and str(Path(path).absolute()) in plan["source_pins"],
                "actual project and publisher module source pin required",
            )


def _group(plan):
    return cpu._own_profile_cgroup(plan["unit"], PROFILE)


def run(plan_path, plan_sha):
    _need(SOURCE_ONLY is False, "reviewed synthetic CUDA launcher release required")
    plan = _plan(plan_path, plan_sha)
    output = Path(plan["output_dir"])
    _need(
        output.is_dir()
        and output.resolve() == output
        and output.stat().st_mode & 0o077 == 0
        and not any(output.iterdir()),
        "fresh private output required",
    )
    if os.getpgrp() != os.getpid():
        os.setsid()
    _need(os.getpgrp() == os.getpid(), "isolated process group required")
    group = _group(plan)
    invocation = os.environ.get("INVOCATION_ID", "")
    _need(re.fullmatch(r"[0-9a-f]{32}", invocation), "unit invocation required")
    _source_guard(plan)
    common._validate_interpreter(plan)
    hello = {
        "schema": "decision3-cuda-synthetic-hello-v1",
        "pid": os.getpid(),
        "startticks": common.systemd.process_identity(os.getpid())["startticks"],
        "boot_id": common.systemd.boot_id(),
        "unit": plan["unit"],
        "invocation": invocation,
        "plan_sha256": plan_sha,
        "pgid": os.getpgrp(),
        "cgroup": group,
    }
    common._write_new(output / "hello.json", child._encode(hello))
    started = time.monotonic()
    while not (output / "go.json").exists():
        _need(time.monotonic() - started < 20, "supervisor handshake timed out")
        time.sleep(0.1)
    child._control(
        output / "go.json",
        schema="decision3-cuda-synthetic-go-v1",
        invocation=invocation,
        plan_sha=plan_sha,
    )
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_CACHE": str(cpu.CACHE),
            "HF_MODULES_CACHE": str(cpu.STAGED.parents[2]),
            "CUDA_VISIBLE_DEVICES": plan["gpu_uuid"],
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "5",
            "MKL_NUM_THREADS": "5",
            "OPENBLAS_NUM_THREADS": "5",
        }
    )
    _source_guard(plan)
    common._bind_project_source(cpu.ROOT)
    torch = common._bind_overlay(cpu.ROOT, plan["source_pins"])
    torch.set_num_threads(5)
    torch.set_num_interop_threads(1)
    worker = common._load_reviewed_release(
        str(_release("worker")),
        FILES["worker"],
        plan["source_pins"],
        "_cuda_synthetic_worker_release",
    )

    def ownership_guard():
        _need(
            os.environ.get("INVOCATION_ID") == invocation and _group(plan) == group,
            "worker ownership changed",
        )

    try:
        result = worker.run(
            torch=torch,
            gpu_uuid=plan["gpu_uuid"],
            source_guard=lambda: _source_guard(plan),
            ownership_guard=ownership_guard,
        )
        _source_guard(plan)
        ownership_guard()
        raw = child._encode(result)
        _need(len(raw) <= 4 * 1024 * 1024, "bounded synthetic result required")
        common._write_new(output / "result.json", raw)
        common._write_new(
            output / "terminal.json",
            child._encode(
                {
                    "schema": "decision3-cuda-synthetic-terminal-v1",
                    "status": "completed_pending_external_audit",
                    "invocation": invocation,
                    "plan_sha256": plan_sha,
                    "result_sha256": hashlib.sha256(raw).hexdigest(),
                }
            ),
        )
    except BaseException as error:
        evidence = getattr(error, "cuda_synthetic_evidence", None)
        failure = {
            "schema": "decision3-cuda-synthetic-failure-v1",
            "invocation": invocation,
            "plan_sha256": plan_sha,
            "error_type": type(error).__name__,
            "cleanup": evidence.get("cleanup", "unknown") if type(evidence) is dict else "unknown",
        }
        try:
            common._write_new(output / "failure.json", child._encode(failure))
        except BaseException:
            error.add_note("private failure record unavailable")
        raise


def main():
    _need(SOURCE_ONLY is False, "reviewed synthetic CUDA launcher release required")
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    run(args.plan, args.plan_sha256)


if __name__ == "__main__":
    raise SystemExit("invoke the independently reviewed bootstrap")
