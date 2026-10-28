"""Queue missing MPS benchmark stages using the original frozen S1 implementation.

Source snapshots precede the queue. No S1/Torch import or model load occurs
until the captured baseline, external supervisor, and its recorded child have
terminated. Old results and cache entries retain their original execution epoch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
COHERENCE_PIN = "e18733694623aa93058e279c3534f8b8e2edefa3"
HEAVEN_PIN = "bb05a335bc809e61b20c0f745d25499a82b326fc"
COHERENCE_SETTINGS = {
    "suite": "jevbench-mini",
    "concurrency": 1,
    "progress": True,
    "seed": 0,
    "repeats": 1,
    "trials": 1,
    "tolerance": 0.05,
    "bootstrap": 1000,
}
ALLOCATOR_POLICY = {
    "name": "mps-unoccupied-cache-trim-v1",
    "scope": "After every dispatched callback and at each stage boundary, both profiles.",
    "operations": ["synchronize", "snapshot", "empty_cache", "synchronize", "snapshot"],
    "limits_changed": False,
    "interpretation": "Hypothesis: release unoccupied allocator buffers; no performance or OOM remediation claim before measurement.",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root):
    root = Path(root)
    return (
        {
            str(path.relative_to(root)): {"sha256": digest(path), "size_bytes": path.stat().st_size}
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        if root.exists()
        else {}
    )


def maintain_mps_allocator(torch):
    """Measure then release only unoccupied MPS buffers; never change memory limits."""
    started = time.perf_counter()
    record = {
        "policy": copy.deepcopy(ALLOCATOR_POLICY),
        "before": None,
        "after": None,
        "error": None,
    }

    def snapshot():
        return {
            "current_allocated_bytes": torch.mps.current_allocated_memory(),
            "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
            "recommended_max_bytes": torch.mps.recommended_max_memory(),
        }

    try:
        torch.mps.synchronize()
        record["before"] = snapshot()
        torch.mps.empty_cache()
        torch.mps.synchronize()
        record["after"] = snapshot()
    except Exception as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    record["wall_ms"] = (time.perf_counter() - started) * 1000
    return record


def process_identity(pid):
    observed = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart=,stat=,command="], capture_output=True, text=True
    )
    if observed.stderr.strip() or observed.returncode not in (0, 1):
        raise RuntimeError("process observation failed; refusing overlapping MPS load")
    if observed.returncode == 0 and observed.stdout.strip():
        value = observed.stdout.strip()
        stable_identity(value)
        return value
    if observed.stdout.strip():
        raise RuntimeError("inconsistent process observation")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError as exc:
        raise RuntimeError("cannot verify process absence") from exc
    raise RuntimeError("process is live but its identity could not be observed")


def stable_identity(identity):
    if not isinstance(identity, str) or len(identity) < 27:
        raise ValueError("incomplete captured process identity")
    tail = identity[24:].split(maxsplit=1)
    if len(tail) != 2:
        raise ValueError("incomplete captured process identity")
    return identity[:24], tail[1]


def capture_handle(pid):
    identity = process_identity(pid)
    return {"pid": pid, "identity": identity, "captured_at_utc": now()}


def wait_handle(handle, save, poll_seconds):
    original = stable_identity(handle["identity"]) if handle["identity"] else None
    while True:
        current = process_identity(handle["pid"])
        handle.update(last_observation=current, observed_at_utc=now())
        if current is None:
            handle["status"] = "original_exited"
        elif original is None:
            raise RuntimeError("PID appeared after an absent capture; cannot identify predecessor")
        elif stable_identity(current) != original:
            handle["status"] = "original_ended_pid_reused"
        elif current[24:].split(maxsplit=1)[0].startswith("Z"):
            handle["status"] = "original_terminal_zombie"
        else:
            handle["status"] = "waiting_same_process"
        save()
        if handle["status"] != "waiting_same_process":
            return
        time.sleep(poll_seconds)


def verify_external_child(path, save, receipt, poll_seconds, *, expected_supervisor_pid):
    source = json.loads(Path(path).read_text())
    if source.get("supervisor_pid") != expected_supervisor_pid:
        raise RuntimeError("external child receipt belongs to a different supervisor")
    child = source.get("child")
    if source.get("status") not in {"completed", "failed"} or not isinstance(child, dict):
        raise RuntimeError(
            "external supervisor has no terminal child receipt; GPU freedom unproven"
        )
    if (
        type(child.get("pid")) is not int
        or child["pid"] <= 0
        or type(child.get("exit_code")) is not int
    ):
        raise RuntimeError("external child terminal identity/exit code is incomplete")
    stable_identity(child.get("identity"))
    handle = {"pid": child["pid"], "identity": child["identity"], "captured_at_utc": now()}
    receipt["external_child"] = handle
    receipt["external_supervisor_receipt"] = {"path": str(path), "sha256": digest(path)}
    wait_handle(handle, save, poll_seconds)


def archive_repository(source, target, pin):
    source = Path(source).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if revision != pin or dirty:
        raise ValueError("benchmark source must match its clean immutable pin")
    names = (
        subprocess.check_output(["git", "-C", str(source), "ls-files", "-z"]).decode().split("\0")
    )
    for name in filter(None, names):
        origin = source / name
        if origin.is_symlink() or not origin.is_file():
            raise ValueError("benchmark source contains unsupported nonregular tracked member")
        raw = origin.read_bytes()
        if raw != subprocess.check_output(["git", "-C", str(source), "show", f"HEAD:{name}"]):
            raise ValueError("benchmark tracked bytes differ from pin")
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    return {
        "source": str(source),
        "revision": revision,
        "archived_files": len(list(filter(None, names))),
    }


def freeze_sources(args, receipt):
    destination = args.output / "sources"
    destination.mkdir()
    identity = json.loads((args.parent / "identity.json").read_text())
    source_root = args.parent / "sources"
    s1_hashes = {
        name: value
        for name, value in identity["source_sha256"].items()
        if name.startswith("src/s1/")
    }
    if "src/s1/unified.py" not in s1_hashes or "src/s1/backends.py" not in s1_hashes:
        raise ValueError("parent lacks frozen S1 execution sources")
    for name, expected in s1_hashes.items():
        source = source_root / name
        if source.is_symlink() or digest(source) != expected:
            raise ValueError("parent frozen S1 source differs from execution receipt")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in (
        "scripts/resume_mps_benchmarks.py",
        "scripts/run_jevbench_public.py",
        "scripts/summarize_jevbench_campaign.py",
        "configs/benchmarks/jevbench-public.json",
    ):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    receipt["coherence_source"] = archive_repository(
        args.coherence_root, destination / "coherence", COHERENCE_PIN
    )
    receipt["heaven_source"] = archive_repository(
        args.heaven_root, destination / "heaven", HEAVEN_PIN
    )
    receipt["source_inventory"] = inventory(destination)
    receipt["parent_identity"] = {
        "path": str(args.parent / "identity.json"),
        "sha256_at_queue": digest(args.parent / "identity.json"),
    }
    receipt["checkpoint"] = copy.deepcopy(identity["checkpoint"])
    receipt["runtime_expected"] = copy.deepcopy(identity["runtime"])
    receipt["source_attestation"] = {
        "old_epoch": "Parent archived S1 bytes; original public runner/config were not archived before launch, so full old execution-source attestation remains partial.",
        "resume_epoch": "Frozen S1 plus current public runner/config and complete clean pinned benchmark repositories archived before queue.",
    }


def verify_frozen_sources(output, expected):
    if inventory(output / "sources") != expected:
        raise ValueError("resume source snapshot changed while queued")


def install_frozen_paths(output):
    frozen = (output / "sources").resolve()
    for name, module in tuple(sys.modules.items()):
        if name == "s1" or name.startswith("s1."):
            source = getattr(module, "__file__", None)
            if not source or not Path(source).resolve().is_relative_to(frozen / "src"):
                raise RuntimeError(
                    "S1 was imported before the original frozen source was installed"
                )
        if name == "jevbench" or name.startswith("jevbench."):
            source = getattr(module, "__file__", None)
            if not source or not Path(source).resolve().is_relative_to(frozen / "coherence/src"):
                raise RuntimeError("coherence was imported before its frozen source was installed")
    sys.path[:0] = [str(frozen / "src"), str(frozen / "coherence/src"), str(frozen / "scripts")]


def load_runtime(output):
    install_frozen_paths(output)
    backends = importlib.import_module("s1.backends")
    contracts = importlib.import_module("s1.contracts")
    torch = importlib.import_module("torch")
    jb = importlib.import_module("jevbench")
    public_path = output / "sources/scripts/run_jevbench_public.py"
    spec = importlib.util.spec_from_file_location("_s1_resume_public_runner", public_path)
    public = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(public)
    audit_path = output / "sources/scripts/summarize_jevbench_campaign.py"
    audit_spec = importlib.util.spec_from_file_location("_s1_resume_coherence_auditor", audit_path)
    auditor = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(auditor)
    return SimpleNamespace(
        backends=backends, contracts=contracts, torch=torch, jb=jb, public=public, auditor=auditor
    )


def verify_checkpoint(checkpoint):
    root = Path(checkpoint["path"]).resolve()
    if digest(root / "artifact-manifest.json") != checkpoint["manifest_sha256"]:
        raise ValueError("checkpoint manifest differs from parent")
    manifest = json.loads((root / "artifact-manifest.json").read_text())
    if manifest.get("files") != checkpoint["verified_files"]:
        raise ValueError("checkpoint member identity differs from parent receipt")
    if not {"model.safetensors", "config.json", "s1_config.json"} <= {
        row["path"] for row in checkpoint["verified_files"]
    }:
        raise ValueError("checkpoint parent receipt lacks weights/config/calibration members")
    for row in checkpoint["verified_files"]:
        relative = Path(row["path"])
        target = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(root):
            raise ValueError("unsafe checkpoint member")
        if target.stat().st_size != row["size_bytes"] or digest(target) != row["sha256"]:
            raise ValueError("checkpoint member differs from verified parent")


def audit_public(root, runtime, heaven_root, *, profile, revision):
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError("public result is incomplete")
    identity = manifest.get("identity", {})
    if (
        identity.get("revision") != revision
        or identity.get("runtime", {}).get("question_mode") != profile
    ):
        raise ValueError("completed public model/profile identity differs from parent")
    tasks, official, config = runtime.public.load_public(heaven_root)
    records = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
    if manifest["dataset"] != config or [row["task_id"] for row in records] != [
        task.id for task in tasks
    ]:
        raise ValueError("completed public result task identity differs")
    for relative, expected in manifest["artifact_sha256"].items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()) or digest(target) != expected:
            raise ValueError("completed public result artifact differs")
    for task, row in zip(tasks, records, strict=True):
        raw = json.loads((root / row["raw_path"]).read_text())
        expected_request = {
            "state": task.state,
            "questions": {"decision": official.base.build_question(task)},
        }
        if (
            raw["request"] != expected_request
            or digest(root / row["raw_path"]) != row["raw_sha256"]
        ):
            raise ValueError("completed public raw request differs")
        outcome = official.scoring.score_task(
            runtime.public._native_probs(raw["response"], task), task
        )
        if outcome != row["official_task_outcome"] or not row["ok"] or not outcome["strict_valid"]:
            raise ValueError("completed public score/validity differs")
    recomputed = official.summarize.summarize(tasks, records)
    for key in ("n_attempted", "n_correct", "n_valid", "accuracy", "ece", "brier_mean"):
        if recomputed[key] != summary[key]:
            raise ValueError("completed public summary differs from official replay")
    return {
        "path": str(root),
        "summary_sha256": digest(root / "summary.json"),
        "n_attempted": len(records),
    }


def validate_coherence_identity(report, *, expected_profile, expected_suite):
    result, scores = report["result"], report["scores"]
    if (
        report.get("settings")
        != {"tolerance": 0.05, "min_n": 5, "bootstrap": 1000, "min_coverage": 0.95}
        or report.get("name") != expected_profile
        or result.get("backend") != expected_profile
        or scores.get("backend") != expected_profile
        or result.get("suite") != expected_suite
        or result.get("seed") != 0
        or result.get("repeats") != 1
        or result.get("trials") != 1
        or result.get("selection")
        != {
            "domains": None,
            "dimensions": None,
            "groups": None,
            "relations": None,
            "diagnostics": False,
        }
        or len(result.get("planned_tests", {})) != 50
        or any(value != 24 for value in result["planned_tests"].values())
    ):
        raise ValueError("coherence execution identity/suite/settings differ from frozen stage")


def inspect_original_coherence(path, *, expected_profile, expected_suite):
    """Distinguish a valid interrupted execution from corrupted complete evidence."""
    report = json.loads(path.read_text())
    validate_coherence_identity(
        report, expected_profile=expected_profile, expected_suite=expected_suite
    )
    result, scores = report["result"], report["scores"]
    partial = bool(
        result.get("errors")
        or result.get("backend_down")
        or result.get("truncated_by_budget")
        or scores.get("incomplete")
        or any(row.get("coverage", 1) < 1 for row in scores.get("checks", {}).values())
    )
    if partial:
        if scores.get("n_outcomes") != len(result.get("outcomes", [])):
            raise ValueError("partial coherence raw outcome counters are inconsistent")
        return {
            "status": "preserved_partial_execution",
            "path": str(path),
            "sha256": digest(path),
            "errors": result.get("errors"),
            "backend_down": result.get("backend_down"),
            "truncated_by_budget": result.get("truncated_by_budget"),
            "n_checks": scores.get("n_checks"),
            "n_outcomes": scores.get("n_outcomes"),
            "incomplete": scores.get("incomplete"),
            "check_coverage": {
                name: row.get("coverage") for name, row in scores.get("checks", {}).items()
            },
            "reported_overall": scores.get("overall"),
        }
    # A nominally complete artifact must pass all checks. Do not classify
    # corruption, wrong identity, or fabricated full counters as an interruption.
    audit_coherence(path, expected_profile=expected_profile, expected_suite=expected_suite)
    return {"status": "complete_requires_raw_replay", "path": str(path), "sha256": digest(path)}


def audit_coherence(
    path, *, expected_profile, expected_suite, replay=None, cache_root=None, source_root=None
):
    report = json.loads(path.read_text())
    validate_coherence_identity(
        report, expected_profile=expected_profile, expected_suite=expected_suite
    )
    result, scores = report["result"], report["scores"]
    expected_settings = {"tolerance": 0.05, "min_n": 5, "bootstrap": 1000, "min_coverage": 0.95}
    planned = result.get("planned_tests", {})
    checks = scores.get("checks", {})
    outcomes = result.get("outcomes", [])
    if (
        report.get("settings") != expected_settings
        or report.get("name") != expected_profile
        or result.get("backend") != expected_profile
        or scores.get("backend") != expected_profile
        or result.get("suite") != expected_suite
        or result.get("seed") != 0
        or result.get("repeats") != 1
        or result.get("trials") != 1
        or result.get("selection")
        != {
            "domains": None,
            "dimensions": None,
            "groups": None,
            "relations": None,
            "diagnostics": False,
        }
        or result.get("truncated_by_budget")
        or result.get("backend_down")
        or result.get("errors")
        or scores.get("n_checks") != 50
        or scores.get("n_cases") != 240
        or len(outcomes) != 1468
        or scores.get("n_outcomes") != len(outcomes)
        or scores.get("skipped_outcomes") != 0
        or scores.get("incomplete")
        or scores.get("min_coverage") != 0.95
        or len(planned) != 50
        or any(value != 24 for value in planned.values())
        or sum(planned.values()) != 1200
        or len(checks) != 50
        or {name.split("/")[0] for name in checks} != set(planned)
        or any(
            row.get("n") != 24
            or row.get("planned") != 24
            or row.get("coverage") != 1
            or row.get("excluded")
            for row in checks.values()
        )
    ):
        raise ValueError("coherence report is incomplete or contains errors")
    audit = {
        "path": str(path),
        "sha256": digest(path),
        "n_checks": 50,
        "planned_tests": 1200,
        "judged_tests": sum(row["n"] for row in checks.values()),
        "n_outcomes": scores["n_outcomes"],
        "achieved_check_coverage": 1.0,
        "configured_min_coverage": 0.95,
    }
    if replay is not None:
        audit["raw_replay"] = replay(report, cache_root, source_root)
    return audit


def run_stages(args, receipt, save):
    runtime = load_runtime(args.output)
    heaven = args.output / "sources/heaven"
    # Already completed browser tasks/public231 stay in the parent untouched.
    receipt["preserved_causal_public231"] = audit_public(
        args.parent / "causal_multislot/public231",
        runtime,
        heaven,
        profile="causal_multislot",
        revision=receipt["checkpoint"]["manifest_sha256"],
    )
    expected = receipt["runtime_expected"]
    requested_policy = {
        "device": "mps",
        "dtype": "bfloat16",
        "precision": "bfloat16",
        "attn_implementation": "sdpa",
        "quantization": "none",
        "max_context": 16384,
        "compile_mode": None,
        "adapter_path": None,
    }
    if any(expected.get(field) != value for field, value in requested_policy.items()):
        raise ValueError("parent runtime differs from requested resume policy before model load")
    if (
        receipt["packages"].get("torch") != expected["torch_version"]
        or receipt["packages"].get("transformers") != expected["transformers_version"]
    ):
        raise ValueError("resume package versions differ from parent before model load")
    verify_checkpoint(receipt["checkpoint"])
    shared = runtime.backends.GemmaBackend(
        receipt["checkpoint"]["path"],
        device="mps",
        dtype="bfloat16",
        attn_implementation="sdpa",
        question_mode="causal_multislot",
    )
    fields = (
        "device",
        "dtype",
        "precision",
        "attn_implementation",
        "quantization",
        "max_context",
        "torch_version",
        "transformers_version",
        "temperature",
        "compile_mode",
        "max_forward_batch_size",
        "max_prepared_prompts",
        "adapter_path",
    )
    if any(
        field not in expected or shared.metadata.get(field) != expected[field] for field in fields
    ):
        raise ValueError("resume runtime differs from original MPS/temperature policy")
    receipt["runtime_loaded"] = shared.metadata
    receipt["allocator_maintenance_policy"] = copy.deepcopy(ALLOCATOR_POLICY)
    suite = runtime.jb.load_suite("jevbench-mini")
    expected_suite = {**suite.record(), "plan": "tests.jsonl.gz"}
    receipt["suite_identity"] = expected_suite
    save()
    for profile, kind in (
        ("causal_multislot", "coherence-mini"),
        ("independent", "public231"),
        ("independent", "coherence-mini"),
    ):
        original = (
            args.parent / profile / ("public231" if kind == "public231" else "coherence-mini.json")
        )
        stage = {
            "profile": profile,
            "kind": kind,
            "status": "running",
            "started_at_utc": now(),
            "model_calls": 0,
            "reported_cost_usd": None,
            "original": str(original),
            "cache_before": None,
            "cache_after": None,
        }
        receipt["stages"].append(stage)
        save()
        cache = args.parent / profile / "coherence-cache"
        folder = args.output / profile
        folder.mkdir(exist_ok=True)
        try:
            stage["allocator_maintenance_before"] = maintain_mps_allocator(runtime.torch)
            save()
            if stage["allocator_maintenance_before"]["error"]:
                raise RuntimeError("MPS stage-boundary allocator maintenance failed")
            completed = False
            if original.exists() and kind == "public231":
                try:
                    completed = (
                        json.loads((original / "manifest.json").read_text()).get("status")
                        == "completed"
                    )
                    if not completed:
                        manifest_path = original / "manifest.json"
                        stage["preserved_partial_original"] = {
                            "path": str(original),
                            "manifest_sha256": digest(manifest_path),
                            "manifest": json.loads(manifest_path.read_text()),
                        }
                        summary_path = original / "summary.json"
                        if summary_path.exists():
                            stage["preserved_partial_original"].update(
                                summary_sha256=digest(summary_path),
                                summary=json.loads(summary_path.read_text()),
                            )
                except (FileNotFoundError, json.JSONDecodeError):
                    stage["preserved_partial_original"] = (
                        "Missing/unreadable manifest; original bytes retained, fresh output required."
                    )
            elif original.exists():
                prior = inspect_original_coherence(
                    original, expected_profile=profile, expected_suite=expected_suite
                )
                completed = prior["status"] == "complete_requires_raw_replay"
                if not completed:
                    stage["preserved_partial_original"] = prior
            if completed:
                stage["audit"] = (
                    audit_public(
                        original,
                        runtime,
                        heaven,
                        profile=profile,
                        revision=receipt["checkpoint"]["manifest_sha256"],
                    )
                    if kind == "public231"
                    else audit_coherence(
                        original,
                        expected_profile=profile,
                        expected_suite=expected_suite,
                        replay=runtime.auditor.replay_coherence,
                        cache_root=cache,
                        source_root=args.coherence_root,
                    )
                )
                stage["status"] = "skipped_completed_audited"
                continue
            backend = copy.copy(shared)
            backend.question_mode = profile
            backend.metadata = {**shared.metadata, "question_mode": profile}
            trace = folder / f"{kind}-model-calls.jsonl"

            def predict(state, questions):
                started = time.perf_counter()
                row = {
                    "request": {
                        "state": copy.deepcopy(state),
                        "questions": copy.deepcopy(questions),
                    },
                    "response": None,
                    "error": None,
                    "reported_cost_usd": None,
                }
                stage["model_calls"] += 1
                try:
                    runtime.torch.mps.synchronize()
                    response = backend.predict(
                        runtime.contracts.DecisionRequest.model_validate(row["request"])
                    )
                    runtime.torch.mps.synchronize()
                    try:
                        json.dumps(response, allow_nan=False)
                        row["response"] = response
                        row["response_encoding"] = "json"
                    except (TypeError, ValueError, OverflowError):
                        row["response"] = repr(response)
                        row["response_encoding"] = "python_repr_not_json"
                    return response
                except Exception as exc:
                    row["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    raise
                finally:
                    row["allocator_maintenance"] = maintain_mps_allocator(runtime.torch)
                    row["latency_ms"] = (time.perf_counter() - started) * 1000
                    with trace.open("a") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    if row["allocator_maintenance"]["error"] and row["error"] is None:
                        raise RuntimeError("MPS callback allocator maintenance failed")

            if kind == "public231":
                identity = {
                    "model_id": backend.name,
                    "revision": receipt["checkpoint"]["manifest_sha256"],
                    "runtime": backend.metadata,
                    "profile": profile,
                    "resume_source_inventory": receipt["source_inventory"],
                    "official_rank": None,
                }
                runtime.public.run_public(predict, heaven, folder / "public231", identity)
                stage["audit"] = audit_public(
                    folder / "public231",
                    runtime,
                    heaven,
                    profile=profile,
                    revision=receipt["checkpoint"]["manifest_sha256"],
                )
            else:
                stage["evaluation_settings"] = copy.deepcopy(COHERENCE_SETTINGS)
                stage["cache_path"] = str(cache)
                stage["cache_before"] = inventory(cache)
                stage["cache_epoch"] = (
                    "Existing entries retain original parent execution; callback JSONL contains only newly dispatched model calls."
                )
                save()
                report = runtime.jb.evaluate(
                    runtime.jb.from_callable(
                        lambda state, questions: predict(state, questions)["answers"], name=profile
                    ),
                    cache=str(cache),
                    **COHERENCE_SETTINGS,
                )
                stage["report_path"] = str(folder / "coherence-mini.json")
                report.save(folder / "coherence-mini.json")
                stage["audit"] = audit_coherence(
                    folder / "coherence-mini.json",
                    expected_profile=profile,
                    expected_suite=expected_suite,
                    replay=runtime.auditor.replay_coherence,
                    cache_root=cache,
                    source_root=args.coherence_root,
                )
            stage["status"] = "completed"
        except Exception as exc:
            stage.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            if kind == "coherence-mini" and stage["cache_before"] is not None:
                stage["cache_after"] = inventory(cache)
            stage["allocator_maintenance_after"] = maintain_mps_allocator(runtime.torch)
            stage["finished_at_utc"] = now()
            save()
            if stage["allocator_maintenance_after"]["error"] and stage["status"] != "error":
                stage.update(
                    status="error",
                    error={
                        "type": "RuntimeError",
                        "message": "MPS stage-boundary allocator maintenance failed",
                    },
                )
                save()
                raise RuntimeError("MPS stage-boundary allocator maintenance failed")


def supervise(args):
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "initializing",
        "pid": os.getpid(),
        "created_at_utc": now(),
        "stages": [],
        "reported_cost_usd": None,
        "official_rank": None,
        "allocator_maintenance_policy": copy.deepcopy(ALLOCATOR_POLICY),
        "parent": str(args.parent),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {name: package_version(name) for name in ("torch", "transformers", "numpy")},
    }

    def save():
        write_json(args.output / "receipt.json", receipt)

    save()
    try:
        receipt["original_baseline"] = capture_handle(args.original_pid)
        receipt["predecessor"] = capture_handle(args.wait_for_pid)
        freeze_sources(args, receipt)
        receipt["status"] = "waiting"
        save()
        wait_handle(receipt["predecessor"], save, args.poll_seconds)
        wait_handle(receipt["original_baseline"], save, args.poll_seconds)
        verify_external_child(
            args.predecessor_receipt,
            save,
            receipt,
            args.poll_seconds,
            expected_supervisor_pid=args.wait_for_pid,
        )
        verify_frozen_sources(args.output, receipt["source_inventory"])
        # Re-observe the captured original once more immediately before load.
        wait_handle(receipt["original_baseline"], save, args.poll_seconds)
        receipt.update(status="running", started_at_utc=now())
        save()
        run_stages(args, receipt, save)
        receipt.update(status="completed", finished_at_utc=now())
        save()
        return 0
    except Exception as exc:
        receipt.update(
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
            finished_at_utc=now(),
        )
        save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coherence-root", type=Path, required=True)
    parser.add_argument("--heaven-root", type=Path, required=True)
    parser.add_argument("--predecessor-receipt", type=Path, required=True)
    parser.add_argument("--wait-for-pid", type=int, required=True)
    parser.add_argument("--original-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args(argv)
    if min(args.wait_for_pid, args.original_pid) <= 0 or not 0 < args.poll_seconds <= 60:
        parser.error("positive PIDs and poll interval in (0,60] required")
    for name in ("parent", "output", "coherence_root", "heaven_root", "predecessor_receipt"):
        setattr(args, name, getattr(args, name).resolve())
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
