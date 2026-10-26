"""Root preselected serial train collection; original owner09 owns every run."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = Path(__file__).resolve().parent
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
CONTROLLER_SHA = "81846efe2ee0357da07283c1a77a606283a920cbddef668c8f62def291057dee"
JOBS_SHA = "13cd17ff57ab08bee89ec93d69dc1dc446a64a67a220856857e70ae2924b8f5d"
UNIT = "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service"
PROFILE = "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"


def read(path):
    path = Path(path)
    assert path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024**2
    return json.loads(path.read_text())


def ref(path):
    path = Path(path)
    blob = path.read_bytes()
    assert len(blob) <= 1024**2
    return {"path": str(path), "sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}


def check_ref(record):
    assert ref(record["path"]) == record


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def positive(closure_path, terminal_path):
    closure, terminal = read(closure_path), read(terminal_path)
    out = Path(closure_path).parent
    receipt = read(out / "receipt.json")
    assert terminal["exit_code"] == 0 and terminal["argv"][-1] == str(out)
    assert receipt["profile"] == PROFILE and receipt["completed"] is True
    assert receipt["primary_error"] is None and receipt["secondary_errors"] == []
    assert receipt["sources_unchanged"] is True and receipt["closure"] == closure
    config = receipt["selected_config"]
    assert config["profile"] == PROFILE and config["unit"] == UNIT and config["output"] == str(out)
    assert closure["selection"]["unit"] == UNIT and closure["selection"]["nonce"] == config["nonce"]
    assert closure["closure_mode"] == "authenticated_stop" and closure["closure_errors"] == []
    assert all(
        closure[k] is True
        for k in (
            "whole_unit_exit_confirmed",
            "observed_cgroup_cohort_exit_confirmed",
            "post_stop_terminal_verified",
            "cgroup_absent",
            "sources_unchanged",
        )
    )
    assert all(
        closure[k] is False
        for k in (
            "no_escape_enforced",
            "whole_tree_exit_confirmed",
            "continuation_authorized",
            "model_qualified",
        )
    )
    assert len(closure["two_process_scans"]) == 2
    assert all(not row["matches"] and not row["unknown"] for row in closure["two_process_scans"])
    return {
        "terminal": ref(terminal_path),
        "receipt": ref(out / "receipt.json"),
        "current_closure": ref(closure_path),
    }


def main():
    manifest_path = HERE / "root-remaining-collector-jobs-01.json"
    assert ref(manifest_path)["sha256"] == JOBS_SHA
    assert ref(OWNER / "controller.py")["sha256"] == CONTROLLER_SHA
    manifest = read(manifest_path)
    assert manifest["serial_only"] is True and len(manifest["jobs"]) == 11
    assert not (HERE / "root-remaining-collector-batch-terminal-01.json").exists()
    check_ref(manifest["initial_previous_closure"])
    check_ref(manifest["initial_previous_terminal"])
    prior = positive(
        manifest["initial_previous_closure"]["path"], manifest["initial_previous_terminal"]["path"]
    )
    results = []
    for job in manifest["jobs"]:
        check_ref(job["plan"])
        check_ref(job["spec"])
        output = Path(job["actual_output"])
        terminal_path = Path(job["root_terminal"])
        intent = HERE / f"root-intent-train-collector-{output.name[-2:]}.json"
        assert output.parent == OWNER and not any(
            p.exists() or p.is_symlink() for p in (output, terminal_path, intent)
        )
        spec = read(job["spec"]["path"])
        assert len(spec["sources"]) == 56
        for row in spec["sources"]:
            assert ref(ROOT / row["path"]) == {**row, "path": str(ROOT / row["path"])}
        for row in spec["binary_stats"]:
            info = Path(row["path"]).stat()
            assert all(getattr(info, key) == value for key, value in row["stat"].items())
        argv = [
            str(ROOT / ".venv/bin/python"),
            "-B",
            "-s",
            str(OWNER / "controller.py"),
            "--run",
            "--expected-entry-sha256",
            CONTROLLER_SHA,
            "--selection",
            job["spec"]["path"],
            "--selection-sha256",
            job["spec"]["sha256"],
            "--output",
            str(output),
        ]
        selected = {
            "profile": "root-selected-train-collector-cpu5-terminal-v1",
            "argv": argv,
            "selection_sha256": job["spec"]["sha256"],
            "launcher_source": ref(Path(__file__)),
            "jobs_manifest": ref(manifest_path),
            "previous_original_scoped_closure": prior["current_closure"],
            "new_original_owner_admission_and_closure_required": True,
            "no_previous_native_fit_or_score_inheritance": True,
            "ancestor_id": job["ancestor_id"],
        }
        save(intent, selected)
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        began = time.monotonic()
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=False)
        selected.update(
            started_at_utc=started,
            finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            wall_seconds=time.monotonic() - began,
            exit_code=result.returncode,
            stdout=result.stdout.decode("utf-8", errors="replace"),
            stderr=result.stderr.decode("utf-8", errors="replace"),
        )
        save(terminal_path, selected)
        row = {
            "index": job["index"],
            "ancestor_id": job["ancestor_id"],
            "exit_code": result.returncode,
            "wall_seconds": selected["wall_seconds"],
            "root_terminal": ref(terminal_path),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        if result.returncode != 0:
            save(
                HERE / "root-remaining-collector-batch-terminal-01.json",
                {
                    "completed": False,
                    "results": results,
                    "stop_on_first_failure": True,
                    "training_eligible": False,
                },
            )
            return result.returncode
        prior = positive(output / "external-boundary-closure.json", terminal_path)
    save(
        HERE / "root-remaining-collector-batch-terminal-01.json",
        {
            "completed": True,
            "results": results,
            "last_original_proofs": prior,
            "training_eligible": False,
            "all_chunks_require_separate_root_actual_audit_and_COMMITTED_acceptance": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
