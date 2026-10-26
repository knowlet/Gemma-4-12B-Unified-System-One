"""Root-selected CPU5 NumPy zero-delta qualification; original owner09 owns admission and cleanup."""

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
SPEC_SHA = "d09aa897b2ae8f0c4931e80c527d561b820d6cd4cd35e535af1ef7c50c8a1002"


def ref(path):
    raw = path.read_bytes()
    assert len(raw) <= 1048576
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    spec_path = HERE / "root-fit-command-spec-01.json"
    terminal = OWNER / "root-execution-train-fit-18.json"
    intent = HERE / "root-intent-train-fit-18.json"
    output = OWNER / "actual-18"
    assert ref(OWNER / "controller.py")["sha256"] == CONTROLLER_SHA
    assert ref(spec_path)["sha256"] == SPEC_SHA
    assert not any(p.exists() or p.is_symlink() for p in (terminal, intent, output))
    spec = json.loads(spec_path.read_text())
    for row in spec["sources"]:
        assert ref(ROOT / row["path"]) == row
    for row in spec["binary_stats"]:
        info = Path(row["path"]).stat()
        assert all(getattr(info, key) == value for key, value in row["stat"].items())
    prior = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
    closure_path = prior / "actual-17/external-boundary-closure.json"
    assert (
        ref(closure_path)["sha256"]
        == "4d06c9c0ee2fc7a12cd7b353e1c2c00270f43a4a56bc08042873d377766e9e61"
    )
    closure = json.loads(closure_path.read_text())
    assert closure["closure_mode"] == "authenticated_stop" and not closure["closure_errors"]
    for key in (
        "whole_unit_exit_confirmed",
        "observed_cgroup_cohort_exit_confirmed",
        "post_stop_terminal_verified",
        "cgroup_absent",
        "sources_unchanged",
    ):
        assert closure[key] is True
    assert len(closure["two_process_scans"]) == 2
    assert all(not row["matches"] and not row["unknown"] for row in closure["two_process_scans"])
    capability = json.loads((HERE / "root-committed-train-fit-capability-01.json").read_text())
    assert (
        capability["status"] == "ROOT_SELECTED_COMMITTED_TRAIN_ONLY"
        and capability["train_states"] == 96
    )
    for entry in capability["chunks"]:
        accepted = json.loads(Path(entry["acceptance"]["path"]).read_text())
        assert accepted["status"] == "COMMITTED" and accepted["training_eligible"] is True
    argv = [
        str(ROOT / ".venv/bin/python"),
        "-B",
        "-s",
        str(OWNER / "controller.py"),
        "--run",
        "--expected-entry-sha256",
        CONTROLLER_SHA,
        "--selection",
        str(spec_path),
        "--selection-sha256",
        SPEC_SHA,
        "--output",
        str(output),
    ]
    selected = {
        "profile": "root-selected-train-fit-cpu5-terminal-v1",
        "argv": argv,
        "selection_sha256": SPEC_SHA,
        "launcher_source": ref(Path(__file__)),
        "previous_original_scoped_closure": ref(closure_path),
        "new_original_owner_admission_and_closure_required": True,
        "no_previous_native_fit_or_score_inheritance": True,
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
    save(terminal, selected)
    print(
        json.dumps(
            {
                "exit_code": result.returncode,
                "wall_seconds": selected["wall_seconds"],
                "root_terminal": str(terminal),
            }
        )
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
