#!/usr/bin/env python3
"""Owned GX10 Decision3 text48 diagnostic; no helper, requests or browser score.

The unchanged policy memory floors apply at every gate. This separate diagnostic
cannot admit a co-resident helper or qualify a matched browser comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import signal
import stat
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
SELF = Path(__file__).resolve()
BASE = (
    ROOT / "artifacts/jev-runtime/decision3-GX10-warm-successor-20261020-01/gx10_warm_bootstrap.py"
)
BASE_SHA = "4f383edd45dc4ce3d5e6be90a83bed0a431f42dcb115f92fcbb20be25e62f91a"
INTERPRETER = "/home/albert_hsieh/ComfyUI/.venv/bin/python"
PROFILE = "decision3-gx10-publisher-text48-standalone-diagnostic-v1"
WALL_SECONDS = 1800
LOG_BYTES = 4 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def load_base():
    require(digest(BASE.read_bytes()) == BASE_SHA, "Frozen GX10 bootstrap differs")
    spec = importlib.util.spec_from_file_location("gx10_warm_diagnostic_base", BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_plan(path, expected):
    base = load_base()
    blob = base.checked_blob(path, expected)
    plan = json.loads(blob)
    base.fields(
        plan,
        "profile model nonce boot_id memtotal gpu_uuid runtime runtime_sha output sources",
        "Diagnostic plan",
    )
    require(
        plan["profile"] == PROFILE and plan["model"] in base.CACHE_CANDIDATES,
        "Staged diagnostic model",
    )
    worker = base.load_worker()
    original = base.source_inventory(worker)
    sources = {row["path"]: row["pin"] for row in original["rows"]}
    sources[str(SELF)] = worker.pin(SELF)
    require(plan["sources"] == sources, "Mechanical diagnostic source union differs")
    require(worker.SAFE_ID.fullmatch(plan["nonce"]), "Safe diagnostic nonce")
    require(type(plan["memtotal"]) is int and plan["memtotal"] > 0, "Positive MemTotal")
    for key in ("boot_id", "gpu_uuid"):
        require(type(plan[key]) is str and bool(plan[key]), "Expected host/GPU identity")
    output = Path(plan["output"])
    require(
        output.is_absolute() and output.is_relative_to(base.REPLAY_ROOT),
        "Bounded diagnostic output root",
    )
    require(
        not any(p.is_symlink() for p in (output, *output.parents)), "Ordinary diagnostic output"
    )
    selection = base.RuntimeSelection(
        base.checked_blob(plan["runtime"], plan["runtime_sha"]), plan["runtime_sha"], worker
    )
    selection.verify()
    identities = worker.authenticate_sources(original["original_projection"])
    return base, worker, selection, identities, plan


def worker_main(path, expected):
    base, worker, selection, identities, plan = read_plan(path, expected)
    output = Path(plan["output"]) / "worker"
    require(output.is_dir() and not list(output.iterdir()), "Fresh empty worker evidence")
    core, name = worker.CORE, plan["model"]
    gates_base = core.Gates(output, identities[name], plan["boot_id"], plan["memtotal"])
    gates = base.UUIDGates(gates_base, worker, plan["gpu_uuid"])
    state = {
        "profile": PROFILE,
        "model": name,
        "helper_used": False,
        "requests": 0,
        "browser_tasks": 0,
        "matched_quality_qualified": False,
        "warmup_complete": False,
        "whole_worker_exit_confirmed": False,
        "error": None,
        "secondary_errors": [],
    }
    backend = torch = None
    primary = None
    before = None
    try:
        gates.check("before-input-and-framework")
        before = core.authenticate_cache(name, identities[name], output, gates, "initial")
        state["network_attempts_blocked"] = base.offline_environment(
            {"nonce": plan["nonce"], "model": name}, core
        )
        started = time.monotonic()
        deps = base.dependencies(selection)
        torch = deps.torch
        require(
            deps.checkpoints == identities
            and torch.cuda.is_available()
            and torch.cuda.device_count() == 1,
            "Original six identities and exactly one CUDA device",
        )
        gates_base.torch = torch
        gates.check("before-model-constructor")
        state["constructor_entered"] = True
        backend = deps.backend(
            worker.matched_context_spec(deps.spec(name, device="cuda:0")),
            expected_boot_id=plan["boot_id"],
            expected_memtotal=plan["memtotal"],
        )
        torch.cuda.synchronize(0)
        state["framework_and_constructor_seconds"] = time.monotonic() - started
        initial = base.geometry(backend, identities[name], torch, name)
        worker.publisher_warmup(
            backend.runtime,
            torch,
            gates,
            output,
            state,
            plan["gpu_uuid"],
            before["files"]["d3_runtime.py"],
        )
        worker.authenticate_sources(base.source_inventory(worker)["original_projection"])
        require(
            core.authenticate_cache(name, identities[name], output, gates, "post-warmup") == before,
            "Whole cache changed after warmup",
        )
        require(
            base.geometry(backend, identities[name], torch, name) == initial,
            "Postwarm geometry differs",
        )
        state["runtime_after"] = selection.verify(loaded=True)
        state["actual_runtime_info"] = worker.evidence_value(backend.runtime.runtime_info())
        state["actual_kernels"] = worker.evidence_value(backend.runtime.kernels)
        state["geometry"] = initial
        gates.check("post-warmup-final")
        state["warmup_complete"] = True
    except BaseException as exc:
        primary = exc
    finally:
        for stage, operation in (
            ("backend_close", backend.close if backend else None),
            ("cuda_fence", (lambda: torch.cuda.synchronize(0)) if torch else None),
        ):
            if operation is not None:
                try:
                    operation()
                except BaseException as exc:
                    state["secondary_errors"].append({"stage": stage, **worker.error_info(exc)})
                    if primary is None:
                        primary = exc
                    else:
                        primary.add_note(stage + ": " + repr(exc))
        if primary is None:
            try:
                worker.authenticate_sources(base.source_inventory(worker)["original_projection"])
                require(
                    core.authenticate_cache(name, identities[name], output, gates, "after-close")
                    == before,
                    "Whole cache changed after backend close",
                )
                selection.verify(loaded=True)
                require(worker.pin(SELF) == plan["sources"][str(SELF)], "Diagnostic source changed")
                gates.check("final-after-close")
            except BaseException as exc:
                primary = exc
        if state.get("constructor_entered") and backend is None:
            state["constructor_partial_effects"] = "UNKNOWN_UNTIL_WHOLE_WORKER_EXIT"
        if primary is not None:
            state["error"] = worker.error_info(primary)
        state["status"] = (
            "WARM_DIAGNOSTIC_COMPLETE_PENDING_PROCESS_EXIT"
            if primary is None
            else "DIAGNOSTIC_FAILED"
        )
        try:
            worker.persist(output / "terminal.json", state, terminal=True)
        except BaseException as exc:
            if primary is not None:
                primary.add_note("Terminal persistence also failed: " + repr(exc))
            else:
                primary = exc
    if primary is not None:
        raise primary


def proc_identity(pid):
    try:
        value = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return {
            "pid": pid,
            "state": value[0],
            "pgid": int(value[2]),
            "sid": int(value[3]),
            "start_ticks": int(value[19]),
        }
    except FileNotFoundError:
        return None


def group_members(owner):
    leader = proc_identity(owner["pid"])
    require(
        leader is None or all(leader[key] == owner[key] for key in ("start_ticks", "pgid", "sid")),
        "Owned leader PID identity changed; no signal permitted",
    )
    result = []
    for path in Path("/proc").iterdir():
        if path.name.isdecimal():
            identity = proc_identity(int(path.name))
            if identity is not None and identity["pgid"] == owner["pgid"]:
                require(
                    identity["sid"] == owner["sid"]
                    and identity["start_ticks"] >= owner["start_ticks"],
                    "Owned process group identity changed",
                )
                result.append(identity)
    return result


def close_group(process, owner, *, grace=2):
    """Only the owned fresh session/group; never signal a user service."""
    signals = []
    for number in (signal.SIGTERM, signal.SIGKILL):
        members = group_members(owner)
        if not members:
            break
        try:
            os.killpg(owner["pgid"], number)
        except ProcessLookupError:
            pass
        signals.append(number)
        until = time.monotonic() + grace
        while time.monotonic() < until:
            process.poll()  # Reap the direct child when it exits.
            if not group_members(owner):
                break
            time.sleep(0.05)
    process.wait(timeout=grace)
    remaining = group_members(owner)
    require(
        not remaining and proc_identity(owner["pid"]) is None,
        "Owned worker/group not wholly absent",
    )
    return {
        "signals": signals,
        "remaining": remaining,
        "worker_absent": True,
        "returncode": process.returncode,
    }


def child_limits():
    resource.setrlimit(resource.RLIMIT_FSIZE, (LOG_BYTES, LOG_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def supervise(argv, directory, observe, *, wall_seconds=WALL_SECONDS):
    """Bounded parent with fresh resource checks and complete owned PG cleanup."""
    process = owner = None
    primary = None
    state = {
        "argv": argv,
        "wall_seconds": wall_seconds,
        "log_file_bytes": LOG_BYTES,
        "error": None,
        "whole_worker_exit_confirmed": False,
        "samples": 0,
    }
    started = time.monotonic()
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupted(_number, _frame):
        raise RuntimeError("Owned diagnostic supervisor received SIGTERM")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (
            (directory / "stdout.log").open("xb") as stdout,
            (directory / "stderr.log").open("xb") as stderr,
        ):
            observe()
            process = subprocess.Popen(
                argv,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                preexec_fn=child_limits,
                close_fds=True,
            )
            owner = proc_identity(process.pid)
            require(
                owner is not None and owner["pgid"] == owner["sid"] == process.pid,
                "Fresh independently owned worker session",
            )
            state["owner"] = owner
            while process.poll() is None:
                require(time.monotonic() - started <= wall_seconds, "Warm diagnostic wall limit")
                observe()
                state["samples"] += 1
                time.sleep(0.25)
    except BaseException as exc:
        primary = exc
        state["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if process is not None:
            if owner is None:
                # Popen created a fresh session. No unrelated PID/group is used.
                owner = {
                    "pid": process.pid,
                    "pgid": process.pid,
                    "sid": process.pid,
                    "start_ticks": 0,
                }
            try:
                state["closure"] = close_group(process, owner)
                state["whole_worker_exit_confirmed"] = True
            except BaseException as exc:
                state["closure_error"] = {"type": type(exc).__name__, "message": str(exc)}
                if primary is None:
                    primary = exc
        state["elapsed_seconds"] = time.monotonic() - started
        signal.signal(signal.SIGTERM, previous_term)
    return state, primary


def persist_closed(worker, path, state, primary):
    try:
        worker.persist(path, state, terminal=True)
    except BaseException as secondary:
        if primary is None:
            primary = secondary
        else:
            primary.add_note("Closed evidence also failed: " + repr(secondary))
    if primary is not None:
        raise primary


def parent_main(args):
    base, worker, selection, identities, plan = read_plan(args.plan, args.plan_sha256)
    output = Path(plan["output"])
    require(not output.exists(), "Fresh diagnostic output required")
    core = worker.CORE

    def observe():
        snapshot = core.resource_snapshot()
        require(
            snapshot["boot_id"] == plan["boot_id"]
            and snapshot["proc_meminfo"]["MemTotal"] == plan["memtotal"],
            "Same host boot/MemTotal",
        )
        decision = core.memory_decision(snapshot, identities[plan["model"]]["resident_file_bytes"])
        require(
            decision["admitted"], "Unchanged full policy floor refused: " + json.dumps(decision)
        )
        count = total = 0
        if output.exists():
            for folder, directories, files in os.walk(output, followlinks=False):
                require(
                    not any((Path(folder) / name).is_symlink() for name in directories),
                    "Linked diagnostic output directory",
                )
                for name in files:
                    path = Path(folder) / name
                    try:
                        observed = path.lstat()
                    except FileNotFoundError:
                        continue  # Atomic evidence/cache writer renamed a temporary file.
                    require(stat.S_ISREG(observed.st_mode), "Ordinary diagnostic output file")
                    count += 1
                    total += observed.st_size
            require(count <= core.MAX_FILES and total <= core.MAX_BYTES, "Diagnostic output budget")
        return snapshot

    initial = observe()  # No output/module cache/child before this admission.
    selection.verify()
    output.mkdir(parents=True)
    (output / "worker").mkdir()
    worker.persist(
        output / "intent.json",
        {
            "profile": PROFILE,
            "plan_path": str(args.plan),
            "plan_sha256": args.plan_sha256,
            "initial": initial,
        },
    )
    module_cache = ROOT / "artifacts/jev-d3-browser-module-cache" / plan["nonce"] / plan["model"]
    module_cache.mkdir(parents=True, exist_ok=False)
    required = {
        "HF_HUB_CACHE": str(core.HUB),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_TOKEN_PATH": "/nonexistent-d3-public-token",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_ENABLE_PARALLEL_LOADING": "false",
        "HF_PARALLEL_LOADING_WORKERS": "1",
        "HF_MODULES_CACHE": str(module_cache),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "TRITON_CACHE_DIR": str(output / "triton-cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(output / "inductor-cache"),
        "TORCH_EXTENSIONS_DIR": str(output / "torch-extensions"),
    }
    os.environ.update(required)
    argv = [
        INTERPRETER,
        "-B",
        "-s",
        str(SELF),
        "--worker",
        "--plan",
        str(args.plan),
        "--plan-sha256",
        args.plan_sha256,
    ]
    state, primary = supervise(argv, output, observe)
    state.update(
        profile=PROFILE,
        initial=initial,
        matched_quality_qualified=False,
        helper_used=False,
        browser_tasks=0,
    )
    try:
        state["final_resource_snapshot"] = observe()
        read_plan(args.plan, args.plan_sha256)  # Recheck selected source/runtime after whole exit.
    except BaseException as exc:
        state["final_check_error"] = worker.error_info(exc)
        if primary is None:
            primary = exc
        else:
            primary.add_note("Final resource/source check also failed: " + repr(exc))
    terminal_path = output / "worker/terminal.json"
    terminal = None
    try:
        terminal = core.read(terminal_path) if terminal_path.is_file() else None
    except BaseException as exc:
        state["terminal_read_error"] = worker.error_info(exc)
        if primary is None:
            primary = exc
    state["worker_terminal"] = terminal
    state["status"] = (
        "WARM_DIAGNOSTIC_COMPLETE"
        if (
            primary is None
            and state.get("closure", {}).get("returncode") == 0
            and terminal is not None
            and terminal.get("error") is None
            and terminal.get("warmup_complete") is True
            and state["whole_worker_exit_confirmed"]
        )
        else "DIAGNOSTIC_FAILED"
    )
    persist_closed(worker, output / "closed.json", state, primary)
    print(
        json.dumps(
            {
                "profile": PROFILE,
                "status": state["status"],
                "output": str(output),
                "whole_worker_exit_confirmed": state["whole_worker_exit_confirmed"],
            }
        )
    )
    return 0 if state["status"] == "WARM_DIAGNOSTIC_COMPLETE" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    if not args.run and not args.worker:
        print(
            json.dumps(
                {
                    "profile": PROFILE,
                    "mode": "DRY",
                    "actual_started": False,
                    "warmup_calls": 48,
                    "helper_used": False,
                    "browser_tasks": 0,
                    "wall_seconds": WALL_SECONDS,
                    "matched_quality_qualified": False,
                }
            )
        )
        return 0
    require(
        args.plan is not None and args.plan_sha256 is not None,
        "Explicit pinned diagnostic plan required",
    )
    require(not (args.run and args.worker), "Independent parent/worker mode")
    if args.worker:
        worker_main(args.plan, args.plan_sha256)
        return 0
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
