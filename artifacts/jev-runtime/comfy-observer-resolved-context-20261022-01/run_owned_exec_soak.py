"""Bounded read-only observer soak with explicitly owned harmless exec churn."""

import hashlib
import importlib.util
import json
import os
import resource
import signal
import subprocess
import threading
import time
from pathlib import Path

resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
resource.setrlimit(resource.RLIMIT_CPU, (45, 45))
signal.alarm(60)
HERE = Path(__file__).absolute().parent
SOURCE = HERE / "comfy_observer.py"
EXPECTED = "103804b05e3047011aa5daf099c04ee71a60b3e1435c0690dc7b0377bd486df7"
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
spec = importlib.util.spec_from_file_location("_stable_observer_soak", SOURCE)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)
boot = observer._boot()
stopped = threading.Event()
churn = {"started": 0, "reaped": 0, "nonzero": 0, "own_timeout": 0}


def owned_exec_churn():
    while not stopped.is_set():
        child = subprocess.Popen(
            ["/bin/sh", "-c", 'exec "$1" -B -c pass', "owned-observer-soak", "/usr/bin/python3"],
            cwd=Path.cwd(),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        churn["started"] += 1
        try:
            result = child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            churn["own_timeout"] += 1
            child.terminate()
            result = child.wait(timeout=2)
        churn["reaped"] += 1
        churn["nonzero"] += result != 0


started = time.monotonic()
worker = threading.Thread(target=owned_exec_churn, daemon=True)
worker.start()
rows, states, refusals, retries = [], {}, 0, 0
resolutions, longest_call = 0, 0.0
try:
    for index in range(150):
        try:
            call_started = time.monotonic()
            result = observer.observe(boot)
            longest_call = max(longest_call, time.monotonic() - call_started)
            resolutions += result["comfy_evidence"].get("natural_exit_resolution") is not None
            state = result["comfy_state"]
            states[state] = states.get(state, 0) + 1
            for sample in result["comfy_evidence"].get("samples", []):
                retries += sample["processes"]["identity_retries"]
            rows.append({"index": index, "result": result})
            if state == "http_busy":
                break
        except observer.ObservationError as error:
            refusals += 1
            rows.append({"index": index, "error": str(error), "evidence": error.evidence})
finally:
    stopped.set()
    worker.join(timeout=5)
    assert not worker.is_alive(), "Owned harmless child churn failed to finish"
    assert churn["started"] == churn["reaped"]

assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
summary = {
    "source_sha256": EXPECTED,
    "boot_id": boot,
    "calls": len(rows),
    "states": states,
    "refusals": refusals,
    "successful_scan_identity_retries": retries,
    "natural_exit_resolutions": resolutions,
    "maximum_successful_call_seconds": longest_call,
    "elapsed_seconds": time.monotonic() - started,
    "owned_exec_churn": churn,
    "bounds": {"address_space_bytes": 256 * 1024**2, "cpu_seconds": 45, "wall_seconds": 60},
    "scope": "Read-only observer and owned /bin/sh exec into /usr/bin/python3 -B -c pass only",
}
(HERE / "new-observer-owned-exec-soak-01.json").write_text(
    json.dumps({"summary": summary, "observations": rows}, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, sort_keys=True), flush=True)
