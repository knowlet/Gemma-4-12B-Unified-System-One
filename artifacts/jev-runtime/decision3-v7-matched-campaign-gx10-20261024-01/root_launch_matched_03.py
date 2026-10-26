"""Fresh root-selected matched-03; preserve the failed matched-02 review."""

import datetime
import hashlib
import json
import math
import os
import stat
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = Path(__file__).absolute().parent
PLAN_SHA256 = '7c0338fa66f600734b02510401d50ad126d9a6e7b3b756ffde0c94782e04b8e3'
OWNER_SHA256 = '8c14ce981f3d2b8e67ef6b57903e705c231bee4725dddddb58413d04d467dadf'
TEMPLATE_SHA256 = 'c6e2b128284208efe791409ba240a8c82fca4a2ff3f5f3570ee88f5ad83c7691'
PRIOR_INTERRUPTED_REVIEW = {'path': 'artifacts/jev-runtime/decision3-v7-matched-campaign-gx10-20261024-01/root-interrupted-matched02-review-01.json', 'sha256': '53345a10bc255b8f2579ac0292568aa3b796890fde9ad179ad6ddc91b134d96f', 'size_bytes': 3029}


def require(value, message):
    if not value:
        raise ValueError(message)


def ordinary(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_relative_to(ROOT) and ".." not in path.parts
            and not any(item.is_symlink() for item in (path, *path.parents)),
            "Fixed ordinary repository evidence path required")
    return path


def raw(value):
    path = ordinary(value)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        blob = stream.read(16 * 1024**2 + 1)
        after = os.fstat(stream.fileno())
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(stat.S_ISREG(before.st_mode) and 0 < len(blob) == before.st_size <= 16 * 1024**2
            and all(getattr(before, key) == getattr(after, key) == getattr(path.stat(), key)
                    for key in keys), "Bounded unchanged retained metadata/source required")
    return blob


def reference(value):
    path = ordinary(value)
    blob = raw(path)
    return {"path": str(path), "sha256": hashlib.sha256(blob).hexdigest(),
            "size_bytes": len(blob)}


def authenticate(record):
    require(type(record) is dict and set(record) == {"path", "sha256", "size_bytes"},
            "Exact retained interrupted-run reference required")
    path = ordinary(record["path"])
    blob = raw(path)
    require(type(record["size_bytes"]) is int and len(blob) == record["size_bytes"]
            and hashlib.sha256(blob).hexdigest() == record["sha256"],
            "Retained interrupted-run evidence changed")
    return json.loads(blob)


def save(path, value):
    with path.open("x") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def previous_failed_review():
    review = authenticate(PRIOR_INTERRUPTED_REVIEW)
    require(review["schema"] == "root-matched02-interrupted-runtime-review-v1"
            and review["status"] == "INCOMPLETE_COMFY_BECAME_BUSY",
            "The interrupted original run remains incomplete")
    require(review["original_CPU_positive_closure"] is False
            and review["later_process_absence_retroactively_qualifies_original_run"] is False
            and review["outer_cgroup_absent"] is True
            and review["outer_sources_unchanged"] is True
            and review["complete_matched_score"] is None
            and type(review["candidate_completed_attempts"]) is int
            and review["candidate_completed_attempts"] == 0,
            "No retrospective CPU closure, candidate result, or complete-score qualification")
    proofs = {}
    for key in ("root_terminal", "root_outer_closure", "original_CPU_closure",
                "baseline_only_independent_audit"):
        authenticate(review[key])
        proofs[key] = review[key]
    scans = review["later_root_scans"]
    require(type(scans) is list and len(scans) == 2, "Two retained later root scans required")
    previous_time, pid_sets = -1.0, []
    for scan in scans:
        require(set(scan) == {"at_monotonic", "identities"}
                and type(scan["at_monotonic"]) is float
                and math.isfinite(scan["at_monotonic"])
                and scan["at_monotonic"] > previous_time,
                "Typed ordered later root scan times required")
        previous_time = scan["at_monotonic"]
        rows = scan["identities"]
        require(type(rows) is list and len(rows) == 4, "Four retained process observations required")
        require(all(type(row) is dict
                    and set(row) == {"pid", "recorded_identity_still_present"}
                    and type(row["pid"]) is int and row["pid"] > 0
                    and row["recorded_identity_still_present"] is False for row in rows),
                "Retained scans establish later absence only")
        pids = {row["pid"] for row in rows}
        require(len(pids) == 4, "Four distinct retained PIDs required")
        pid_sets.append(pids)
    require(pid_sets[0] == pid_sets[1], "The same four original PIDs must have been checked")
    fresh = []
    for pid in sorted(pid_sets[0]):
        # Refuse even a reused PID. Do not signal, reset a unit, or requalify the
        # original false closure from a later observation.
        require(not (Path("/proc") / str(pid)).exists(), "An original recorded PID is currently present")
        fresh.append({"pid": pid, "currently_absent": True})
    return proofs, fresh


def main():
    plan_path = HERE / "root-matched-plan-04.json"
    owner = HERE / "owner_released_v3.py"
    require(reference(plan_path)["sha256"] == PLAN_SHA256, "Selected new plan changed")
    require(reference(owner)["sha256"] == OWNER_SHA256, "Selected new owner changed")
    template_path = HERE / "root-cpu-template-03.json"
    require(reference(template_path)["sha256"] == TEMPLATE_SHA256, "Selected new CPU template changed")
    selected = json.loads(raw(plan_path))
    require(selected["schema_id"] == "jev-nf4-d3-matched-v7-owned-v1"
            and selected["unit"] == "jev-nf4-d3-matched-v7-20261024-02.service"
            and selected["attempt"] == "03", "Only the fresh matched-03 selection")
    require(not ordinary(selected["run_directory"]).exists(), "Fresh outer output required")
    cases = json.loads(raw(template_path))["cases"]
    require(len(cases) == 10 and all(not ordinary(row["owner_output"]).exists() for row in cases),
            "Fresh original CPU actual32 through actual41 outputs required")
    proofs, fresh = previous_failed_review()
    intent_path, terminal_path = HERE / "root-intent-matched-03.json", HERE / "root-execution-matched-03.json"
    require(not intent_path.exists() and not terminal_path.exists(), "Fresh root execution records required")
    argv = [str(ROOT / ".venv/bin/python"), "-B", "-s", str(owner), "--run",
            "--plan", str(plan_path), "--plan-sha256", PLAN_SHA256]
    intent = {
        "schema": "root-selected-integrated-matched-v7-execution-v1", "argv": argv,
        "selected_plan": reference(plan_path), "selected_owner": reference(owner),
        "root_launcher_source": reference(Path(__file__)),
        "previous_interrupted_run_review": PRIOR_INTERRUPTED_REVIEW,
        "previous_interrupted_run_proof_refs": proofs,
        "previous_original_CPU_positive_closure": False,
        "later_absence_retroactively_qualifies_original_run": False,
        "fresh_previous_recorded_pid_absence_checks": fresh,
        "workload": "one integrated campaign; NF4 and per-case CPU services serialize actual forwards",
        "new_original_outer_and_each_CPU_admission_and_closure_required": True,
        "inherited_quality_or_full_suite_success": False,
        "base_git_commit_at_selection": selected["base_git_commit"],
        "executed_git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                             check=True, capture_output=True, text=True).stdout.strip(),
    }
    save(intent_path, intent)
    started, began = datetime.datetime.now(datetime.timezone.utc).isoformat(), time.monotonic()
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=False)
    intent.update(started_at_utc=started,
                  finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  wall_seconds=time.monotonic() - began, exit_code=result.returncode,
                  stdout=result.stdout.decode("utf-8", errors="replace"),
                  stderr=result.stderr.decode("utf-8", errors="replace"))
    save(terminal_path, intent)
    print(json.dumps({"exit_code": result.returncode, "wall_seconds": intent["wall_seconds"],
                      "root_terminal": str(terminal_path)}), flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
