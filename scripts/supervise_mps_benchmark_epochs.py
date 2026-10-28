"""Run fresh, sequential MPS benchmark processes with durable per-epoch caches.

No model is imported by this supervisor. Each child must exit/reap before the
next starts. Checkpoint exit 75 is resource control, never a scored model error.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_resume():
    path = ROOT / "scripts/resume_mps_benchmarks.py"
    spec = importlib.util.spec_from_file_location("_mps_epoch_resume", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache_inventory(args, resume):
    result = {}
    for profile in ("causal_multislot", "independent"):
        for name, details in resume.inventory(args.parent / profile / "coherence-cache").items():
            result[f"coherence/{profile}/{name}"] = details
    for name, details in resume.inventory(args.public_cache_root).items():
        if name.endswith(".json"):
            result[f"public/{name}"] = details
    return result


def child_argv(args, index):
    argv = [
        sys.executable,
        "-u",
        str(args.output / "sources/scripts/resume_mps_benchmarks.py"),
        "--parent",
        str(args.parent),
        "--output",
        str(args.output / "epochs" / f"{index:04d}"),
        "--coherence-root",
        str(args.coherence_root),
        "--heaven-root",
        str(args.heaven_root),
        "--predecessor-receipt",
        str(args.predecessor_receipt),
        "--wait-for-pid",
        str(args.wait_for_pid),
        "--original-pid",
        str(args.original_pid),
        "--poll-seconds",
        str(args.poll_seconds),
        "--max-new-calls",
        str(args.max_new_calls),
        "--max-driver-fraction",
        str(args.max_driver_fraction),
        "--public-cache-root",
        str(args.public_cache_root),
    ]
    for pid in args.additional_predecessor_pid:
        argv.extend(["--additional-predecessor-pid", str(pid)])
    return argv


def verify_final_receipt(child):
    required = {
        ("causal_multislot", "coherence-mini"),
        ("independent", "public231"),
        ("independent", "coherence-mini"),
    }
    stages = {(row["profile"], row["kind"]): row for row in child.get("stages", [])}
    if not required <= set(stages):
        raise ValueError("completed child lacks required benchmark stages")
    for key in required:
        row = stages[key]
        if row.get("status") not in {"completed", "skipped_completed_audited"}:
            raise ValueError("completed child contains incomplete stage")
        if (
            key[1] == "coherence-mini"
            and row.get("audit", {}).get("raw_replay", {}).get("raw_cache_files") != 1248
        ):
            raise ValueError("completed coherence lacks full raw cache replay")
        if key[1] == "public231" and row.get("audit", {}).get("n_attempted") != 231:
            raise ValueError("completed public benchmark lacks full population audit")


def dispatched_call_evidence(root, resume):
    rows, hashes = [], {}
    for path in sorted(Path(root).glob("*/*-model-calls.jsonl")):
        hashes[str(path.relative_to(root))] = resume.digest(path)
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    if any(row.get("origin") != "new_model_inference" for row in rows):
        raise ValueError("model-call trace contains a non-inference cache event")
    return {
        "dispatched_callbacks": len(rows),
        "failed_callbacks": sum(row.get("error") is not None for row in rows),
        "maintenance_failures": sum(
            row.get("allocator_maintenance", {}).get("error") is not None for row in rows
        ),
        "trace_sha256": hashes,
    }


def supervise(args):
    resume = load_resume()
    additional = resume.predecessor_pids(args)
    resume.EpochBudget(args.max_new_calls, args.max_driver_fraction)
    if type(args.max_epochs) is not int or args.max_epochs < 1:
        raise ValueError("positive max epochs required")
    args.output.mkdir(parents=True, exist_ok=False)
    args.public_cache_root = args.public_cache_root or args.output / "public-response-cache"
    receipt = {
        "status": "initializing",
        "pid": os.getpid(),
        "created_at_utc": resume.now(),
        "epochs": [],
        "active_child": None,
        "max_epochs": args.max_epochs,
        "epoch_policy": {
            "max_new_calls": args.max_new_calls,
            "max_driver_fraction": args.max_driver_fraction,
        },
        "public_cache_root": str(args.public_cache_root),
        "reported_cost_usd": None,
        "official_rank": None,
        "latency_scope": "Cached reconstruction/epoch setup is not steady-state model serving latency. Original dispatched observations remain in immutable cache entries and per-epoch traces.",
    }

    def save():
        resume.write_json(args.output / "receipt.json", receipt)

    save()
    active = None
    try:
        receipt["original_baseline"] = resume.capture_handle(args.original_pid)
        receipt["predecessor"] = resume.capture_handle(args.wait_for_pid)
        receipt["additional_predecessors"] = [resume.capture_handle(pid) for pid in additional]
        resume.freeze_sources(args, receipt)
        receipt["child_source_inventory"] = copy.deepcopy(receipt["source_inventory"])
        frozen = args.output / "sources"
        shutil.copyfile(Path(__file__), frozen / "scripts/supervise_mps_benchmark_epochs.py")
        for name, source in (
            ("parent-identity.json", args.parent / "identity.json"),
            (
                "checkpoint-manifest.json",
                Path(receipt["checkpoint"]["path"]) / "artifact-manifest.json",
            ),
        ):
            target = frozen / "inputs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        receipt["source_inventory"] = resume.inventory(frozen)
        receipt["status"] = "waiting"
        save()
        for handle in [
            receipt["predecessor"],
            receipt["original_baseline"],
            *receipt["additional_predecessors"],
        ]:
            resume.wait_handle(handle, save, args.poll_seconds)
        resume.verify_external_child(
            args.predecessor_receipt,
            save,
            receipt,
            args.poll_seconds,
            expected_supervisor_pid=args.wait_for_pid,
        )
        for index in range(1, args.max_epochs + 1):
            resume.verify_frozen_sources(args.output, receipt["source_inventory"])
            if resume.digest(args.parent / "identity.json") != resume.digest(
                frozen / "inputs/parent-identity.json"
            ):
                raise ValueError("original parent identity changed between epochs")
            if (
                resume.digest(Path(receipt["checkpoint"]["path"]) / "artifact-manifest.json")
                != receipt["checkpoint"]["manifest_sha256"]
            ):
                raise ValueError("checkpoint manifest changed between epochs")
            before = cache_inventory(args, resume)
            meta = args.output / "epoch-metadata" / f"{index:04d}"
            resume.write_json(meta / "cache-before.json", before)
            argv = child_argv(args, index)
            epoch = {
                "index": index,
                "argv": argv,
                "output": str(args.output / "epochs" / f"{index:04d}"),
                "status": "launching",
                "started_at_utc": resume.now(),
                "pid": None,
                "identity": None,
                "exit_code": None,
                "cache_before": {
                    "path": str(meta / "cache-before.json"),
                    "sha256": resume.digest(meta / "cache-before.json"),
                },
            }
            receipt["epochs"].append(epoch)
            receipt["status"] = "running"
            save()
            log = args.output / "logs" / f"epoch-{index:04d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("xb") as stream:
                active = subprocess.Popen(
                    argv, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT, shell=False
                )
                epoch["pid"] = active.pid
                receipt["active_child"] = {"pid": active.pid, "output": epoch["output"]}
                save()
                try:
                    epoch["identity"] = resume.process_identity(active.pid)
                    receipt["active_child"]["identity"] = epoch["identity"]
                except Exception as exc:
                    epoch["identity_observation_error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                save()
                epoch["exit_code"] = active.wait()
            # Popen.wait is authoritative termination evidence; do not start a
            # replacement on a timeout, stale status, or failed ps observation.
            active = None
            receipt["active_child"] = None
            epoch["finished_at_utc"] = resume.now()
            child_path = Path(epoch["output"]) / "receipt.json"
            child = json.loads(child_path.read_text())
            epoch.update(
                child_status=child.get("status"),
                child_error=child.get("error"),
                child_receipt_sha256=resume.digest(child_path),
            )
            # Error exits can still leave completed official cache writes.
            # Audit immutable prior entries before interpreting any exit code.
            after = cache_inventory(args, resume)
            resume.write_json(meta / "cache-after.json", after)
            epoch["cache_after"] = {
                "path": str(meta / "cache-after.json"),
                "sha256": resume.digest(meta / "cache-after.json"),
            }
            violations = [key for key, value in before.items() if after.get(key) != value]
            if violations:
                epoch["immutable_cache_violations"] = violations
                save()
                raise ValueError("existing shared cache entry was modified or deleted")
            save()
            if epoch["exit_code"] not in (0, 75) or child.get("status") not in {
                "completed",
                "checkpointed",
            }:
                raise RuntimeError(
                    "epoch terminated without a valid resource checkpoint or complete result"
                )
            if (
                child.get("source_inventory") != receipt["child_source_inventory"]
                or child.get("checkpoint") != receipt["checkpoint"]
            ):
                raise ValueError(
                    "child execution source/checkpoint differs from frozen epoch protocol"
                )
            calls = dispatched_call_evidence(Path(epoch["output"]), resume)
            reported_calls = child.get("epoch_budget", {}).get("new_calls")
            if (
                type(reported_calls) is not int
                or not 0 <= reported_calls <= args.max_new_calls
                or calls["dispatched_callbacks"] != reported_calls
            ):
                raise ValueError("epoch dispatched-call evidence differs from declared budget")
            epoch["call_evidence"] = calls
            changed = [key for key, value in after.items() if before.get(key) != value]
            epoch.update(
                status=child.get("status"),
                new_model_calls=child.get("epoch_budget", {}).get("new_calls"),
                changed_cache_entries=changed,
                child_receipt_sha256=resume.digest(child_path),
            )
            receipt["total_new_model_calls"] = sum(
                row.get("new_model_calls") or 0 for row in receipt["epochs"]
            )
            receipt["total_failed_callbacks"] = sum(
                row.get("call_evidence", {}).get("failed_callbacks", 0) for row in receipt["epochs"]
            )
            receipt["total_maintenance_failures"] = sum(
                row.get("call_evidence", {}).get("maintenance_failures", 0)
                for row in receipt["epochs"]
            )
            save()
            stage_failures = any(
                row.get("error")
                or row.get("allocator_maintenance_before", {}).get("error")
                or row.get("allocator_maintenance_after", {}).get("error")
                for row in child.get("stages", [])
            )
            if calls["failed_callbacks"] or calls["maintenance_failures"] or stage_failures:
                raise RuntimeError(
                    "real callback/maintenance failure cannot be a clean checkpoint or completion"
                )
            if epoch["exit_code"] == 0 and child.get("status") == "completed":
                verify_final_receipt(child)
                receipt.update(
                    status="completed", final_epoch=epoch["output"], finished_at_utc=resume.now()
                )
                save()
                return 0
            if epoch["exit_code"] != 75 or child.get("status") != "checkpointed":
                raise RuntimeError(
                    "epoch terminated without a valid resource checkpoint or complete result"
                )
            checkpoint = child.get("epoch_checkpoint", {})
            if checkpoint.get("new_calls") != reported_calls or checkpoint.get("reason") not in {
                "new_model_call_budget",
                "driver_memory_fraction",
            }:
                raise ValueError("epoch checkpoint lacks matching latched resource evidence")
            if not changed:
                raise RuntimeError(
                    "checkpoint epoch made no successful cache progress; refusing restart loop"
                )
        receipt.update(status="epoch_limit_reached", finished_at_utc=resume.now())
        save()
        return 2
    except BaseException as exc:
        receipt.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
        if active is not None and active.poll() is None:
            receipt["status"] = "attention_waiting_owned_child"
            save()
            # Never orphan or replace a known live child. No signals are sent.
            try:
                receipt["active_child"]["exit_code"] = active.wait()
                receipt["active_child"]["terminal"] = True
            finally:
                save()
        save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "output", "coherence-root", "heaven-root", "predecessor-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--original-pid", type=int, required=True)
    parser.add_argument("--wait-for-pid", type=int, required=True)
    parser.add_argument("--additional-predecessor-pid", type=int, action="append", default=[])
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--max-new-calls", type=int, default=20)
    parser.add_argument("--max-driver-fraction", type=float, default=0.8)
    parser.add_argument("--max-epochs", type=int, default=256)
    parser.add_argument("--public-cache-root", type=Path)
    args = parser.parse_args(argv)
    if not 0 < args.poll_seconds <= 60:
        parser.error("poll interval in (0,60] required")
    for name in ("parent", "output", "coherence_root", "heaven_root", "predecessor_receipt"):
        setattr(args, name, getattr(args, name).resolve())
    if args.public_cache_root is not None:
        args.public_cache_root = args.public_cache_root.resolve()
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
