"""Real owned pidfd/zombie controls; never signal or inspect a foreign process."""

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


new = load("_zombie_new", HERE / "comfy_observer.py")
old = load(
    "_zombie_old", HERE.parent / "comfy-observer-resolved-context-20261022-01/comfy_observer.py"
)


def child(seconds):
    script = (
        "import ctypes,os,time; "
        "assert ctypes.CDLL(None).prctl(4,0,0,0,0)==0; "
        "os.write(1,b'R'); "
        f"time.sleep({seconds!r})"
    )
    p = subprocess.Popen([sys.executable, "-B", "-c", script], stdout=subprocess.PIPE)
    assert p.stdout.read(1) == b"R"
    observed, _ = new._process_attempt(Path("/proc") / str(p.pid))
    assert observed["classification"] == "uncertain_python_cwd"
    row = {
        **observed,
        "pid": p.pid,
        "proc_owner_uid": os.getuid(),
        "disappearance_resolution_eligible": True,
    }
    return p, row


def reap(p):
    # This subprocess was created above and belongs to this test only.
    if p.poll() is None:
        p.terminate()
    p.wait(timeout=3)
    p.stdout.close()


def wait_zombie(p):
    until = time.monotonic() + 2
    while new._stat(Path(f"/proc/{p.pid}/stat"))[1] != "Z":
        assert time.monotonic() < until
        time.sleep(0.002)
    # Deliberately do not poll()/wait(): leave the exited child unreaped.


def test_actual_unreaped_zombie_baseline_refuses_successor_resolves():
    p, row = child(0.08)
    try:
        wait_zombie(p)
        with pytest.raises(old.ObservationError, match="remains after natural-exit budget"):
            old._wait_disappeared([row])
        result = new._wait_disappeared([row])
        assert result["disappeared"] == []
        assert result["exited_zombies"] == [{"pid": p.pid, "start_ticks": row["start_ticks"]}]
        assert new._stat(Path(f"/proc/{p.pid}/stat"))[1] == "Z"
    finally:
        reap(p)


def test_actual_live_unknown_still_refuses_and_records_reason():
    p, row = child(2)
    try:
        with pytest.raises(new.ObservationError, match="remains after natural-exit budget") as got:
            new._wait_disappeared([row])
        assert got.value.evidence["failure_reason"] == str(got.value)
        assert got.value.evidence["finished_monotonic"] >= got.value.evidence["started_monotonic"]
        assert got.value.evidence["exited_zombies"] == []
        assert p.poll() is None
    finally:
        reap(p)


def test_wrong_start_identity_is_never_accepted_for_actual_zombie():
    p, row = child(0.08)
    try:
        wait_zombie(p)
        with pytest.raises(new.ObservationError, match="identity changed"):
            new._wait_disappeared([{**row, "start_ticks": row["start_ticks"] + 1}])
    finally:
        reap(p)
