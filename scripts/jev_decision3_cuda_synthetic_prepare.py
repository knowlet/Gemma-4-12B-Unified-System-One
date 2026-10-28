"""Held preparer for one reviewed, private synthetic CUDA smoke plan.

This creates no model process. All source paths are fixed; no discovery, case,
collection or benchmark data is read. The result and the four gate-only
releases require independent review and a separate supervised execution.
The pinned hashes cover selected files, not all interpreter/DSO dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
from pathlib import Path

SOURCE_ONLY = True
ROOT = Path(__file__).absolute().parent.parent
SCRIPTS = ROOT / "scripts"
BASE = ROOT / "artifacts/jev-runtime/decision3-cuda-synthetic"
CACHE = ROOT / "artifacts/jev-local-hf/hub"
SNAPSHOT = CACHE / "models--vllm-sr--d3-edge/snapshots/cab71e6ad3461b8086d51ddb572c656babd63b0b"
STAGED = (
    ROOT
    / "artifacts/jev-runtime/decision3-link-navigation-diagnostic-20261027-01/staged-modules/transformers_modules/cab71e6ad3461b8086d51ddb572c656babd63b0b/5e85afd45ec2b404"
)
OVERLAY = ROOT / "artifacts/jev-runtime/overlay"
PYTHON = Path("/home/albert_hsieh/ComfyUI/.venv/bin/python")
PYTHON_REAL = Path(
    "/home/albert_hsieh/.local/share/uv/python/cpython-3.13.14-linux-aarch64-gnu/bin/python3.13"
)
TORCH = Path("/home/albert_hsieh/ComfyUI/.venv/lib/python3.13/site-packages/torch/__init__.py")
MANIFEST_SHA = "3bae8989a6feb36a66bf4d8c44112dc63199d0dc69f564dc895fe368b1e08e59"
FAST_SHA = "68a1fb9a2c3bde9ac2763bf504decae592ca12722698fbde1b39a31262955535"
HELD_SHA = {
    "jev_decision3_cuda_synthetic_bootstrap.py": "449c25ee32ea943c05d851f337112e376533d13e591f0bcd43fdd76743970667",
    "jev_decision3_cuda_synthetic_launcher.py": "93de5bb8e59a8854602d18222bb15f81a9cbb94ded248afda55b2d0827ab6bf2",
    "jev_decision3_cuda_synthetic_supervisor.py": "95a5474b9de4ecabd6fe11280c92afc31d71799b090d6de3b2d901c0f874e098",
    "jev_decision3_cuda_synthetic_worker.py": "ed3ea76817c8400363d3e5f20695205a24177702a85fae0c31a8117c09c21e48",
}
SHARED_SHA = {
    "jev_decision3_candidate_browser_child.py": "5b71c8a2a927f50edc7f3333a3a0771854553978de5ad27aea3421cdcbd822c8",
    "jev_decision3_candidate_http_owner.py": "d7a9183ff5bd7fef3d4b48564dce5131109eeff3d17ec6358d99e35c35c1923b",
    "jev_decision3_cpu_experiment_worker.py": "45eee52de01128a2dc1a98a029f350d2d4ab3830767843beaab503bcc827e47b",
    "jev_stage_memory_profile_v2.py": "c1ae9e12de593e5ba1f13b57b15edf033116423b2218ca766ea84a0558e76749",
}
PROJECT_SHA = {
    "__init__.py": "c9c7ba0ea4cc81609fcf1f0ff2a853243a173a994257e0b9dd19be1a2e6dd52e",
    "backends.py": "90c084fa5523dea71b49cf140ecfda1fd2165a9468368624bde6161e2dbf67bf",
    "contracts.py": "24c24c3ded9c35bfd263a4488bf88916a8960fa0b79283ccedbac232983f8a0a",
    "errors.py": "e5ad1d9b7e700f07a704d74d3e4ade7156c7a9eaad304d3fe4296bad0dc8c997",
    "evaluation/__init__.py": "127cf68d534aa63995896f51d47ff5015c1d8fcb3563914752aa130f5ff25f51",
    "evaluation/contracts.py": "ead69fb44ba1a0221b9c73fc0c811774ed15b530cdf1ec56f99e7b98bff254d3",
    "evaluation/decision3.py": "aaa21ab6f2ecf5ae0e00694b43755541dc75fc0310ba10d945ade2a4511add52",
}
STAGED_SHA = {
    "d3_runtime.py": "adf7f7aa085674b5682e492d8b2e3cc4f11c67b0d153515c742e907e9b550917",
    "d3_format.py": "3badfbc70561bb3f5b8ec16a11d2c455b96ddb32caace0a8a54ca102eae1f3fa",
    "modeling_d3.py": "f8f8db23ff58f1a990ffa24977d744a50e1b4446d02a64c22070967705b6dbef",
}
MARKER = b"SOURCE_ONLY = True"
FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")


def _need(value, message):
    if not value:
        raise ValueError(message)


def _pinned():
    pins = {SCRIPTS / name: value for name, value in {**HELD_SHA, **SHARED_SHA}.items()}
    pins.update({ROOT / "src/s1" / name: value for name, value in PROJECT_SHA.items()})
    pins.update({STAGED / name: value for name, value in STAGED_SHA.items()})
    pins.update(
        {
            SNAPSHOT / "d3_fast.py": FAST_SHA,
            OVERLAY
            / "transformers/__init__.py": "2725cadd5afaa29b2a12faec1960cb7ac1df77dc76d655310efb33d4607b35f5",
            OVERLAY
            / "transformers/models/qwen3_5/modeling_qwen3_5.py": "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b",
            TORCH: "cf40c075c95864036e835795756d69b8cccfafa76f3bcde5eba9d06065ccd3d1",
            PYTHON_REAL: "92ed50fd9dde3654d421d165214a95361e1889210b3d2063001d6c2e75eef2ab",
        }
    )
    return pins


def _release(name):
    _need(name in HELD_SHA, "only reviewed source release names allowed")
    return SCRIPTS / (name.removesuffix(".py") + "_release.py")


def _checked(path, expected, *, keep=False, links=1):
    _need(
        path.is_absolute() and path.resolve() == path and not path.is_symlink(),
        "canonical fixed source required",
    )
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        _need(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == links
            and 0 <= before.st_size <= 256 * 1024**2,
            "bounded regular source and fixed link count required",
        )
        _need(not keep or 0 < before.st_size <= 8 * 1024**2, "bounded reviewed source required")
        hasher, chunks = hashlib.sha256(), []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            hasher.update(block)
            if keep:
                chunks.append(block)
        after, current = os.fstat(fd), os.stat(path, follow_symlinks=False)
        state = tuple(getattr(before, key) for key in FIELDS)
        _need(
            all(tuple(getattr(value, key) for key in FIELDS) == state for value in (after, current))
            and hasher.hexdigest() == expected,
            "selected source identity or digest changed",
        )
        return b"".join(chunks) if keep else None
    finally:
        os.close(fd)


def _gpu_uuid():
    value = subprocess.run(
        ["/usr/bin/nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        capture_output=True,
        check=True,
        timeout=5,
        text=True,
    ).stdout.strip()
    match = re.fullmatch(
        r"(GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}), NVIDIA GB10", value
    )
    _need(match is not None, "exactly one NVIDIA GB10 UUID required")
    return match.group(1)


def _identity(path):
    info = path.lstat()
    return info.st_dev, info.st_ino


def _remove_owned(files, dirs):
    errors = []
    for path, identity in reversed(files):
        try:
            if _identity(path) == identity:
                path.unlink()
        except FileNotFoundError:
            pass
        except OSError as error:
            errors.append(type(error).__name__)
    for path, identity in reversed(dirs):
        try:
            if _identity(path) == identity:
                path.rmdir()
        except FileNotFoundError:
            pass
        except OSError as error:
            errors.append(type(error).__name__)
    return errors


def _exclusive(path, raw, created):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        created.append((path, (info.st_dev, info.st_ino)))
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def prepare():
    _need(SOURCE_ONLY is False, "reviewed CUDA preparer release required")
    pins, held = {}, {}
    expected = _pinned()
    _need(len(expected) == 23 and len(HELD_SHA) == 4, "exact reviewed source inventory required")
    for path, expected_sha in expected.items():
        is_held = path.parent == SCRIPTS and path.name in HELD_SHA
        data = _checked(
            path, expected_sha, keep=is_held, links=2 if path == SNAPSHOT / "d3_fast.py" else 1
        )
        pins[str(path)] = expected_sha
        if is_held:
            _need(data.count(MARKER) == 1, "one reviewed source gate required")
            held[path.name] = data
    _checked(SNAPSHOT / "MODEL_MANIFEST.json", MANIFEST_SHA, links=2)
    _need(PYTHON.resolve() == PYTHON_REAL, "fixed interpreter path required")
    _need(SCRIPTS.resolve() == SCRIPTS and SCRIPTS.is_dir(), "ordinary scripts directory required")
    for name in HELD_SHA:
        if os.path.lexists(_release(name)):
            raise FileExistsError("reviewed release already exists: " + _release(name).name)
    _need(
        BASE.parent.resolve() == BASE.parent and BASE.parent.is_dir(),
        "ordinary experiment parent required",
    )
    if os.path.lexists(BASE):
        _need(
            BASE.resolve() == BASE and not BASE.is_symlink() and BASE.is_dir(),
            "ordinary experiment base required",
        )
        info = BASE.stat()
        _need(
            info.st_uid == os.getuid() and info.st_mode & 0o077 == 0,
            "owned private experiment base required",
        )
    gpu_uuid = _gpu_uuid()
    nonce = secrets.token_hex(16)
    run = BASE / ("run-" + nonce)
    if os.path.lexists(run):
        raise FileExistsError("experiment run already exists")
    temps, dirs, published = [], [], []
    try:
        for index, (name, data) in enumerate(held.items()):
            release = _release(name)
            staged = SCRIPTS / (".jev-cuda-" + nonce + "-" + str(index) + ".tmp")
            raw = data.replace(MARKER, b"SOURCE_ONLY = False", 1)
            _exclusive(staged, raw, temps)
            os.link(staged, release, follow_symlinks=False)
            published.append((release, temps[-1][1]))
            pins[str(release)] = hashlib.sha256(raw).hexdigest()
        cleanup_errors = _remove_owned(temps, [])
        _need(not cleanup_errors, "temporary release cleanup failed")
        temps.clear()
        if not BASE.exists():
            BASE.mkdir(mode=0o700)
            dirs.append((BASE, _identity(BASE)))
        info = BASE.lstat()
        _need(
            stat.S_ISDIR(info.st_mode)
            and BASE.resolve() == BASE
            and info.st_uid == os.getuid()
            and info.st_mode & 0o077 == 0,
            "owned private experiment base required",
        )
        run.mkdir(mode=0o700)
        dirs.append((run, _identity(run)))
        output = run / "worker"
        output.mkdir(mode=0o700)
        dirs.append((output, _identity(output)))
        plan = {
            "schema": "decision3-cuda-synthetic-plan-v1",
            "unit": "jev-cuda-synthetic-" + nonce + ".service",
            "output_dir": str(output),
            "python": str(PYTHON),
            "source_pins": pins,
            "gpu_uuid": gpu_uuid,
            "wall_seconds": 120,
            "memory_floor_bytes": 34709107508,
        }
        _need(len(pins) == 27, "complete fixed source inventory required")
        raw = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        plan_path = run / "plan.json"
        plan_tmp = run / (".plan-" + nonce + ".tmp")
        _exclusive(plan_tmp, raw, temps)
        os.link(plan_tmp, plan_path, follow_symlinks=False)
        published.append((plan_path, temps[-1][1]))
        cleanup_errors = _remove_owned(temps, [])
        _need(not cleanup_errors, "temporary plan cleanup failed")
        temps.clear()
        return {
            "plan": str(plan_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "report": str(run / "supervisor"),
            "pin_count": len(pins),
        }
    except BaseException as error:
        if _remove_owned(temps + published, dirs):
            error.add_note("owned preparation cleanup incomplete")
        raise


def main():
    _need(SOURCE_ONLY is False, "reviewed CUDA preparer release required")
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.parse_args()
    print(json.dumps(prepare(), separators=(",", ":")))


if __name__ == "__main__":
    main()
