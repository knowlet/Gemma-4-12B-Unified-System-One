"""Root-selected CPU5 frozen delta real-browser diagnostic; original owner09 owns admission and cleanup."""

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
SPEC_SHA = "0ecf0256e3f9dd0c790df68326a9daf8fc41457305b3ebca3ec618ad72ee8381"


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
    spec_path = HERE / "root-command-spec-02.json"
    terminal = OWNER / "root-execution-delta-browser-22.json"
    intent = HERE / "root-intent-delta-browser-22.json"
    output = OWNER / "actual-22"
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
    closure_path = prior / "actual-21/external-boundary-closure.json"
    assert (
        ref(closure_path)["sha256"]
        == "3efc3a2f72de5ebcfd72a36668428ed4f4f825258c938c19978c4c8da330863a"
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
    capability_path = ROOT / "artifacts/jev-runtime/decision3-train-readout-cpu5-20261024-01/root-frozen-delta-candidate-capability-01.json"
    assert ref(capability_path)["sha256"] == "239165d486bad16963a6e5c94c1f9503fc145ea5b85b114e9fbf94d84eb1fd76"
    capability = json.loads(capability_path.read_text())
    assert capability["status"] == "ROOT_ACCEPTED_TRAIN_ONLY_READOUT"
    assert capability["quality_or_win_claim"] is False
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
        "profile": "root-selected-delta-browser-cpu5-terminal-v1",
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
