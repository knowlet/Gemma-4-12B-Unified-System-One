"""Bounded CPU regression tests; tiny test models are not 12B qualification."""

import hashlib
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
OUTPUT = HERE / "compiler-tests-02"
FILES = (
    "scripts/jev_browser_fit_actions.py",
    "scripts/jev_cfg_action_loss.py",
    "tests/test_jev_action_prompt_training.py",
    "tests/test_jev_cfg_action_loss.py",
    "tests/test_jev_browser_fit_actions.py",
)


def sources():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in FILES}


def main():
    if os.environ.get("JEV_COMPILER_TESTS_RUN") != "cpu-regression-only-v1":
        print("Explicit CPU-regression environment required; no tests or models loaded.")
        return 0
    if os.getsid(0) != os.getpid():
        os.setsid()
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    signal.alarm(540)
    before, started = sources(), time.monotonic()
    intent = {
        "scope": "CPU regression tests with tiny fixtures, not full12B or browser qualification",
        "pid": os.getpid(),
        "sources_before": before,
        "RLIMIT_AS": resource.getrlimit(resource.RLIMIT_AS),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    with (OUTPUT / "compiler-tests-intent.json").open("x") as stream:
        json.dump(intent, stream, indent=2)
    import pytest
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized()
    result = int(
        pytest.main(
            [
                "-q",
                "-ra",
                "--maxfail=1",
                "--junitxml=" + str(OUTPUT / "compiler-tests.xml"),
                "-o",
                "cache_dir=" + str(OUTPUT / "pytest-cache"),
                *[str(ROOT / path) for path in FILES if path.startswith("tests/")],
            ]
        )
    )
    after = sources()
    assert after == before and not torch.cuda.is_initialized()
    terminal = {
        **intent,
        "pytest_returncode": result,
        "sources_after": after,
        "elapsed_seconds": time.monotonic() - started,
        "CUDA_initialized": torch.cuda.is_initialized(),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    with (OUTPUT / "compiler-tests-terminal.json").open("x") as stream:
        json.dump(terminal, stream, indent=2)
    return result


if __name__ == "__main__":
    sys.exit(main())
