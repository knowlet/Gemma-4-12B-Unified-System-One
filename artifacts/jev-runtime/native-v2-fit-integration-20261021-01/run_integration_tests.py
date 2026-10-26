"""Own the bounded CPU test unit; reuse the pinned root-only memory guardian."""

import argparse
import hashlib
import importlib.util
import json
import re
import resource
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
PREVIOUS = ROOT / "artifacts/jev-runtime/v2-action-training-prepare-20261021-01"


def save(output, name, data):
    with (output / name).open("x") as stream:
        json.dump(data, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--attempt", default="tests-03")
    attempt = parser.parse_args().attempt
    assert re.fullmatch(r"tests-[0-9]{2}", attempt)
    output = HERE / attempt
    unit = "jev-native-v2-fit-" + attempt + "-20261021.service"
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    index = json.loads((PREVIOUS / "selected-evidence-index-01.json").read_bytes())
    ref = next(p for p in index["sources"] if p["path"].endswith("/run_compiler_tests.py"))
    path = ROOT / ref["path"]
    blob = path.read_bytes()
    assert len(blob) == ref["size_bytes"] and hashlib.sha256(blob).hexdigest() == ref["sha256"]
    spec = importlib.util.spec_from_file_location("_native_fit_root_guardian", path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    before = guard.host()
    assert before["mem_available_bytes"] >= guard.RESERVE + guard.CAP + 64 * 1024**2, before
    assert guard.status(unit)["LoadState"] == "not-found"
    output.mkdir(mode=0o700)
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
        "JEV_NATIVE_V2_CPU_TESTS": "tiny-regression-only-v1",
        "JEV_NATIVE_V2_TEST_ATTEMPT": attempt,
    }
    argv = [
        "systemd-run",
        "--user",
        "--unit=" + unit,
        "--working-directory=" + str(ROOT),
        "--property=Type=exec",
        "--property=RemainAfterExit=yes",
        "--property=MemoryMax=3221225472",
        "--property=MemorySwapMax=0",
        "--property=OOMPolicy=kill",
        "--property=KillMode=control-group",
        "--property=RuntimeMaxSec=600",
        "--property=TimeoutStopSec=10",
        "--property=StandardOutput=append:" + str(output / "integration-tests-service.log"),
        "--property=StandardError=append:" + str(output / "integration-tests-service.log"),
        *["--setenv=" + k + "=" + v for k, v in env.items()],
        "/home/albert_hsieh/ComfyUI/.venv/bin/python",
        "-B",
        "-s",
        str(HERE / "integration_tests.py"),
    ]
    save(
        output,
        "integration-tests-launch.json",
        {
            "host_before": before,
            "argv": argv,
            "guardian": ref,
            "wrapper_sha256": hashlib.sha256(
                (HERE / "integration_tests.py").read_bytes()
            ).hexdigest(),
            "scope": "Full selected regression suite inCPU-only3GiB unit; no12B training or deployment qualification",
        },
    )
    print(guard.command(*argv), flush=True)
    first = guard.status(unit)
    invocation = first["InvocationID"]
    assert len(invocation) == 32
    closure = guard.watch(
        invocation,
        first,
        unit=unit,
        get_status=lambda: guard.status(unit),
        save=lambda _, data: save(output, "integration-tests-root-closure.json", data),
    )
    terminal = closure["terminal_before_stop"]
    assert closure["reason"] == "terminal" and closure["cgroup_absent_after_stop"]
    assert closure["after_stop"]["MainPID"] == "0"
    assert all(
        terminal[k] == v
        for k, v in {
            "Result": "success",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "MainPID": "0",
        }.items()
    )
    test = json.loads((output / "integration-tests-terminal.json").read_bytes())
    assert (
        test["pytest_returncode"] == 0
        and test["sources_unchanged"]
        and not test["CUDA_initialized"]
    )
    print("Selected CPU regressions passed; dedicated unit closed.", flush=True)


if __name__ == "__main__":
    main()
