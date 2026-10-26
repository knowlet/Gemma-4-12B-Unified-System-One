"""CPU-only regressions, including genuine tiny LoRA; no12B qualification."""

import hashlib
import json
import os
import re
import resource
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
ATTEMPT = os.environ.get("JEV_NATIVE_V2_TEST_ATTEMPT", "tests-03")
OUTPUT = HERE / ATTEMPT
TESTS = (
    "test_jev_browser_fit_native_actions.py",
    "test_jev_native_action_adapter.py",
    "test_jev_native_action_data.py",
    "test_jev_local_model.py",
    "test_jev_generated.py",
    "test_jev_browser_fit_actions.py",
    "test_jev_cfg_action_loss.py",
    "test_jev_action_prompt_training.py",
    "test_browser_evidence.py",
    "test_browser_action_adapter.py",
)


def sources():
    paths = [
        *sorted((ROOT / "src/s1").rglob("*.py")),
        *[ROOT / "tests" / name for name in TESTS],
        *[
            ROOT / "scripts" / name
            for name in (
                "jev_browser_fit_native_actions.py",
                "jev_browser_fit_actions.py",
                "jev_cfg_action_loss.py",
                "jev_local_model.py",
                "jev_browser_fit_public_actions.py",
                "jev_public_action_artifact.py",
            )
        ],
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def save(name, data):
    with (OUTPUT / name).open("x") as stream:
        json.dump(data, stream, indent=2)


def main():
    if os.environ.get("JEV_NATIVE_V2_CPU_TESTS") != "tiny-regression-only-v1":
        print("DRY_NO_TEST_OR_MODEL; explicit test environment required")
        return 0
    assert re.fullmatch(r"tests-[0-9]{2}", ATTEMPT)
    if os.getsid(0) != os.getpid():
        os.setsid()
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    signal.alarm(540)
    before, started = sources(), time.monotonic()
    save(
        "integration-tests-intent.json",
        {
            "pid": os.getpid(),
            "sources": before,
            "scope": "Regression tests; mocked12B gates/base and synthetic tokenizer with genuine tinyCPU updates",
        },
    )
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
                "--junitxml=" + str(OUTPUT / "integration-tests.xml"),
                "--basetemp=" + str(OUTPUT / "test-scratch"),
                "-o",
                "cache_dir=" + str(OUTPUT / "pytest-cache"),
                *[str(ROOT / "tests" / name) for name in TESTS],
            ]
        )
    )
    after = sources()
    save(
        "integration-tests-terminal.json",
        {
            "pytest_returncode": result,
            "sources_before": before,
            "sources_after": after,
            "sources_unchanged": before == after,
            "CUDA_initialized": torch.cuda.is_initialized(),
            "elapsed_seconds": time.monotonic() - started,
            "process_peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
    )
    assert before == after and not torch.cuda.is_initialized()
    return result


if __name__ == "__main__":
    sys.exit(main())
