"""Owned harmless CPU children; natural exit only, no signals or services."""

import hashlib
import importlib.util
import json
import os
import resource
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
resource.setrlimit(resource.RLIMIT_CPU, (45, 45))
signal.alarm(60)
HERE = Path(__file__).absolute().parent
OLD = HERE.parent / "comfy-observer-vanished-context-20261022-01/comfy_observer.py"
NEW = HERE / "comfy_observer.py"
OLD_SHA = "5220799afc459a3fb702aff5e0d30a7362d14030293b79872a83def6dd1efae2"
NEW_SHA = "103804b05e3047011aa5daf099c04ee71a60b3e1435c0690dc7b0377bd486df7"


def load(path, digest, name):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load(OLD, OLD_SHA, "_old_resolution_control")
new = load(NEW, NEW_SHA, "_new_resolution_control")
boot = new._boot()
assert new.observe(boot)["comfy_state"] == "absent_confirmed"
CHILD = """import ctypes,os,select,sys,time
assert ctypes.CDLL(None).prctl(4,0,0,0,0)==0
os.write(1,b'ready\\n')
ready,_,_=select.select([sys.stdin],[],[],1.5)
if ready: sys.stdin.buffer.read(1)
time.sleep(0.08)
"""


def owned_child(fake_comfy=False):
    child = subprocess.Popen(
        ["comfyui" if fake_comfy else sys.executable, "-B", "-c", CHILD],
        executable=sys.executable,
        cwd=Path.cwd(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    assert select.select([child.stdout], [], [], 2)[0]
    assert child.stdout.readline() == b"ready\n"
    reaper = threading.Thread(target=child.wait, daemon=True)
    reaper.start()
    return child, reaper


def finish(child, reaper):
    child.stdin.close()
    reaper.join(timeout=3)
    assert not reaper.is_alive() and child.returncode == 0
    child.stdout.close()


def refusal(observer):
    started = time.monotonic()
    try:
        result = observer.observe(boot)
        return {"accepted": True, "elapsed_seconds": time.monotonic() - started, "result": result}
    except observer.ObservationError as error:
        return {
            "accepted": False,
            "elapsed_seconds": time.monotonic() - started,
            "failure_type": type(error).__name__,
            "evidence": error.evidence,
        }


def samples(value):
    evidence = value["evidence"]
    return evidence if isinstance(evidence, list) else evidence.get("samples", [])


results = {}
child, reaper = owned_child()
try:
    before = refusal(old)
    assert before["accepted"] is False
    assert any(
        row["pid"] == child.pid and row["classification"] == "uncertain_python_cwd"
        for sample in samples(before)
        for row in sample["processes"]["uncertain"]
    )
    wait = new._wait_disappeared
    triggered = []

    def natural_exit(rows):
        if any(row["pid"] == child.pid for row in rows):
            child.stdin.write(b"x")
            child.stdin.flush()
            triggered.append(time.monotonic())
        return wait(rows)

    new._wait_disappeared = natural_exit
    try:
        after = new.observe(boot)
    finally:
        new._wait_disappeared = wait
    assert len(triggered) == 1
    proof = after["comfy_evidence"]["natural_exit_resolution"]
    assert child.pid in proof["disappeared"] and len(after["comfy_evidence"]["samples"]) == 2
    assert proof["finished_monotonic"] <= after["comfy_evidence"]["samples"][0]["started_monotonic"]
    assert proof["finished_monotonic"] - proof["started_monotonic"] <= 0.5
    results["owned_undumpable_natural_exit"] = {"pid": child.pid, "old": before, "new": after}
finally:
    finish(child, reaper)

child, reaper = owned_child()
try:
    result = refusal(new)
    assert result["accepted"] is False and child.poll() is None
    assert result["elapsed_seconds"] < 2
    results["owned_undumpable_still_alive"] = {"pid": child.pid, **result}
finally:
    finish(child, reaper)

child, reaper = owned_child(fake_comfy=True)
try:
    result = refusal(new)
    assert result["accepted"] is False and child.poll() is None
    assert any(
        row["pid"] == child.pid
        for sample in samples(result)
        for row in sample["processes"]["known_comfy"]
    )
    assert result["elapsed_seconds"] < 0.5
    results["owned_fake_comfy_argv0_sticky"] = {"pid": child.pid, **result}
finally:
    finish(child, reaper)

assert hashlib.sha256(OLD.read_bytes()).hexdigest() == OLD_SHA
assert hashlib.sha256(NEW.read_bytes()).hexdigest() == NEW_SHA
payload = {
    "source_sha256": NEW_SHA,
    "original_sha256": OLD_SHA,
    "boot_id": boot,
    "scope": "Owned harmless PR_SET_DUMPABLE=0 Python fixtures, natural exits, no signals",
    "all_3_owned_children_reaped_exit_zero": True,
    "results": results,
}
(HERE / "controlled-resolution-01.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(
    json.dumps(
        {
            "source_sha256": NEW_SHA,
            "natural_exit_resolved": True,
            "still_alive_refused": True,
            "fake_comfy_refused": True,
            "all_3_reaped": True,
            "resolution_seconds": results["owned_undumpable_natural_exit"]["new"]["comfy_evidence"][
                "natural_exit_resolution"
            ]["finished_monotonic"]
            - results["owned_undumpable_natural_exit"]["new"]["comfy_evidence"][
                "natural_exit_resolution"
            ]["started_monotonic"],
        },
        sort_keys=True,
    )
)
