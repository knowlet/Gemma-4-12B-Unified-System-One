"""Root-selected CPU4 coherence continuation in the unchanged original owner07."""

import datetime
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = Path(__file__).absolute().parent
SPEC_SHA = "1f3b73a74a84f09a826c1133d83635df3eb1b1c79aa173eebdd1131c9bb5cbad"
CONTROLLER_SHA = "685a9f809f429f4e9658083dc8530b563740625f9ba7816908fdc58002de256d"


def ref(path):
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
    selected = json.loads((HERE / "root-execution-selection-02.json").read_bytes())
    spec = HERE / "root-command-spec-01.json"
    controller = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-07/controller.py"
    output = Path(selected["actual_output"])
    assert ref(spec)["sha256"] == SPEC_SHA and ref(controller)["sha256"] == CONTROLLER_SHA
    assert not output.exists()
    for key in ("actor", "accepted_private_inputs"):
        row = selected[key]
        actual = ref(ROOT / row["path"])
        assert actual["sha256"] == row["sha256"] and actual["size_bytes"] == row["size_bytes"]
    closure_ref = selected["original_scope_closure"]
    path = ROOT / closure_ref["path"]
    assert ref(path)["sha256"] == closure_ref["sha256"]
    closure = json.loads(path.read_bytes())
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
    assert len(closure["two_process_scans"]) == 2 and all(
        not x["matches"] and not x["unknown"] for x in closure["two_process_scans"]
    )
    argv = [
        str(ROOT / ".venv/bin/python"),
        "-B",
        "-s",
        str(controller),
        "--run",
        "--expected-entry-sha256",
        CONTROLLER_SHA,
        "--selection",
        str(spec),
        "--selection-sha256",
        SPEC_SHA,
        "--output",
        str(output),
    ]
    intent = {
        "schema": "root-original-cpu4-coherence-resume-execution-v1",
        "argv": argv,
        "selected_spec": ref(spec),
        "controller": ref(controller),
        "launcher": ref(Path(__file__)),
        "selection": ref(HERE / "root-execution-selection-02.json"),
        "original_failed_campaign_success": False,
        "accepted_prior_count": 1234,
        "missing_callbacks": 14,
        "executed_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    assert not (HERE / "root-execution-02.json").exists()
    save(HERE / "root-intent-02.json", intent)
    began = time.monotonic()
    intent["started_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result = subprocess.run(argv, cwd=ROOT, check=False, capture_output=True)
    intent.update(
        exit_code=result.returncode,
        wall_seconds=time.monotonic() - began,
        finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        stdout=result.stdout.decode("utf-8", errors="replace"),
        stderr=result.stderr.decode("utf-8", errors="replace"),
    )
    save(HERE / "root-execution-02.json", intent)
    print(json.dumps({key: intent[key] for key in ("exit_code", "wall_seconds")}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
