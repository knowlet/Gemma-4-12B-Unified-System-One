"""Own a bounded, CPU-only test unit without disturbing other workloads."""

import hashlib
import json
import resource
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
UNIT = "jev-training-compiler-tests-20261021-01.service"
RESERVE = 26_119_172_096
CAP = 3_221_225_472
PROPS = (
    "LoadState",
    "ActiveState",
    "SubState",
    "InvocationID",
    "MainPID",
    "ExecMainPID",
    "ExecMainCode",
    "ExecMainStatus",
    "ControlGroup",
    "MemoryCurrent",
    "MemoryPeak",
    "MemorySwapPeak",
    "Result",
)


def command(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def status():
    output = command("systemctl", "--user", "show", UNIT, *["--property=" + p for p in PROPS])
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def host():
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    with urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=5) as reply:
        queue = json.load(reply)
    return {
        "mem_available_bytes": int(mem["MemAvailable"].split()[0]) * 1024,
        "comfy_running": len(queue["queue_running"]),
        "comfy_pending": len(queue["queue_pending"]),
    }


def write(name, data):
    with (HERE / name).open("x") as stream:
        json.dump(data, stream, indent=2)


def main():
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    before = host()
    assert before["mem_available_bytes"] >= RESERVE + CAP + 64 * 1024**2, before
    assert before["comfy_running"] == before["comfy_pending"] == 0, before
    assert status()["LoadState"] == "not-found"
    env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "HF_HUB_CACHE": str(ROOT / "artifacts/jev-local-hf/hub"),
        "PYTHONPATH": ":".join(
            str(ROOT / p) for p in ("artifacts/jev-runtime/overlay", "src", "scripts")
        ),
        "CUDA_VISIBLE_DEVICES": "",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "JEV_COMPILER_TESTS_RUN": "cpu-regression-only-v1",
    }
    wrapper = HERE / "compiler_cpu_tests.py"
    argv = [
        "systemd-run",
        "--user",
        "--unit=" + UNIT,
        "--property=Type=exec",
        "--property=RemainAfterExit=yes",
        "--property=MemoryMax=" + str(CAP),
        "--property=MemorySwapMax=0",
        "--property=OOMPolicy=kill",
        "--property=KillMode=control-group",
        "--property=RuntimeMaxSec=600",
        "--property=TimeoutStopSec=10",
        "--property=WorkingDirectory=" + str(ROOT),
        "--property=StandardOutput=append:" + str(HERE / "compiler-tests-service.log"),
        "--property=StandardError=append:" + str(HERE / "compiler-tests-service.log"),
        *["--setenv=" + k + "=" + v for k, v in env.items()],
        "/home/albert_hsieh/ComfyUI/.venv/bin/python",
        "-B",
        "-s",
        str(wrapper),
    ]
    write(
        "compiler-tests-launch.json",
        {
            "scope": "CPU regression tests, tiny fixture models; no 12B or browser qualification",
            "host_before": before,
            "reserve_bytes": RESERVE,
            "memory_max_bytes": CAP,
            "argv": argv,
            "wrapper_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        },
    )
    print(command(*argv), flush=True)
    first = status()
    invocation = first["InvocationID"]
    assert invocation
    observations = []
    started = time.monotonic()
    reason = "terminal"
    while True:
        state, snapshot = status(), host()
        assert state["InvocationID"] == invocation
        observations.append(
            {"elapsed_seconds": time.monotonic() - started, "unit": state, "host": snapshot}
        )
        print(json.dumps(observations[-1]), flush=True)
        if state["SubState"] in ("exited", "dead", "failed"):
            break
        if snapshot["comfy_running"] or snapshot["comfy_pending"]:
            reason = "comfy_queue_became_busy"
            break
        if snapshot["mem_available_bytes"] < RESERVE:
            reason = "host_reserve_crossed"
            break
        if time.monotonic() - started > 580:
            reason = "guardian_deadline"
            break
        time.sleep(10)
    terminal = status()
    assert terminal["InvocationID"] == invocation
    cg = terminal["ControlGroup"] or first["ControlGroup"]
    command("systemctl", "--user", "stop", UNIT)
    closed = status()
    cgroup_absent = not (Path("/sys/fs/cgroup") / cg.lstrip("/")).exists() if cg else None
    write(
        "compiler-tests-root-closure.json",
        {
            "invocation_id": invocation,
            "reason": reason,
            "observations": observations,
            "terminal_before_stop": terminal,
            "after_stop": closed,
            "cgroup_absent_after_stop": cgroup_absent,
            "host_after": host(),
        },
    )
    assert reason == "terminal" and closed["MainPID"] == "0" and cgroup_absent
    result = json.loads((HERE / "compiler-tests-terminal.json").read_text())
    assert result["pytest_returncode"] == 0 and not result["CUDA_initialized"]
    print("CPU compiler regression tests passed and dedicated unit closed.", flush=True)


if __name__ == "__main__":
    main()
