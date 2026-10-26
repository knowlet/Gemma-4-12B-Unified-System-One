"""Root-selected one integrated serial NF4/D3 campaign, using the original owners."""

import datetime
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = Path(__file__).absolute().parent
PLAN_SHA256 = "9785df82188af79610974a6f4821e5623936eeee09d094b3bd1fa1dd7d321087"
OWNER_SHA256 = "635b24c059361eb780bb9c89cd5cf0b0975ff72662b5f270c92834a32e24bdc5"


def reference(path):
    blob = path.read_bytes()
    assert len(blob) <= 1024**2
    return {"path": str(path), "sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}


def save(path, value):
    with path.open("x") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    plan_path = HERE / "root-matched-plan-02.json"
    owner = HERE / "owner_released_v2.py"
    assert reference(plan_path)["sha256"] == PLAN_SHA256
    assert reference(owner)["sha256"] == OWNER_SHA256
    selected = json.loads(plan_path.read_bytes())
    assert selected["schema_id"] == "jev-nf4-d3-matched-v7-owned-v1"
    assert not (ROOT / selected["run_directory"]).exists()
    assert not any(
        Path(row["owner_output"]).exists()
        for row in json.loads((HERE / "root-cpu-template-02.json").read_bytes())["cases"]
    )
    previous = (
        ROOT
        / "artifacts/jev-runtime/llama-cpp-owner-20261024-09/actual-30/external-boundary-closure.json"
    )
    closure = json.loads(previous.read_bytes())
    assert closure["closure_mode"] == "authenticated_stop" and not closure["closure_errors"]
    assert all(
        closure[key] is True
        for key in (
            "whole_unit_exit_confirmed",
            "observed_cgroup_cohort_exit_confirmed",
            "post_stop_terminal_verified",
            "cgroup_absent",
            "sources_unchanged",
        )
    )
    assert len(closure["two_process_scans"]) == 2
    assert all(not row["matches"] and not row["unknown"] for row in closure["two_process_scans"])
    intent_path, terminal_path = (
        HERE / "root-intent-matched-01.json",
        HERE / "root-execution-matched-01.json",
    )
    assert not intent_path.exists() and not terminal_path.exists()
    argv = [
        str(ROOT / ".venv/bin/python"),
        "-B",
        "-s",
        str(owner),
        "--run",
        "--plan",
        str(plan_path),
        "--plan-sha256",
        PLAN_SHA256,
    ]
    intent = {
        "schema": "root-selected-integrated-matched-v7-execution-v1",
        "argv": argv,
        "selected_plan": reference(plan_path),
        "selected_owner": reference(owner),
        "root_launcher_source": reference(Path(__file__)),
        "previous_original_cpu_scoped_closure": reference(previous),
        "workload": "one integrated campaign; NF4 and per-case CPU services serialize actual forwards",
        "new_original_outer_and_each_CPU_admission_and_closure_required": True,
        "inherited_quality_or_full_suite_success": False,
        "base_git_commit_at_selection": selected["base_git_commit"],
        "executed_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
    }
    save(intent_path, intent)
    started, began = datetime.datetime.now(datetime.timezone.utc).isoformat(), time.monotonic()
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=False)
    intent.update(
        started_at_utc=started,
        finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        wall_seconds=time.monotonic() - began,
        exit_code=result.returncode,
        stdout=result.stdout.decode("utf-8", errors="replace"),
        stderr=result.stderr.decode("utf-8", errors="replace"),
    )
    save(terminal_path, intent)
    print(
        json.dumps(
            {
                "exit_code": result.returncode,
                "wall_seconds": intent["wall_seconds"],
                "root_terminal": str(terminal_path),
            }
        ),
        flush=True,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
