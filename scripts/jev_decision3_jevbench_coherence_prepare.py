"""Held creator for one pinned public coherence MINI CUDA plan and v2 releases.

No model runs here. Exactly selected paths are checked; no data directories are
searched. Releases and private plan use no-overwrite publication. HF and other
external dependencies are not an atomically verified transitive supply chain.
Inode checks and unlink are not atomic; a hard kill may leave owned files.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import secrets
import stat
import struct
import subprocess
import types
from pathlib import Path

SOURCE_ONLY = True
ROOT = Path(__file__).absolute().parent.parent
SCRIPTS = ROOT / "scripts"
BASE = ROOT / "artifacts/jev-runtime/decision3-coherence-v2"
OFFICIAL = Path("/tmp/jev-coherence-mini-controlled-snapshot")
BOOT_NAME = "jev_decision3_jevbench_coherence_bootstrap.py"
BOOT_SHA = "e0014c8998e619f49517a7b03ad531f02eefc155bb722754083754611d6226ad"
MARKER = b"SOURCE_ONLY" + b" = True"
MINI_ASSETS = frozenset(
    "src/jevbench/suites/" + name
    for name in (
        "jevbench-mini/manifest.json",
        "jevbench-mini/tests.jsonl.gz",
        "jevbench-240/manifest.json",
        "jevbench-240/cases.jsonl",
        "jevbench-240/rewrites.jsonl",
        "jevbench-240/tests.jsonl.gz",
    )
)
MINI_SDK = frozenset(
    "src/jevbench/" + name
    for name in (
        "__init__.py",
        "__main__.py",
        "api.py",
        "backends.py",
        "cli.py",
        "datagen/__init__.py",
        "datagen/domains/__init__.py",
        "datagen/domains/_rules.py",
        "datagen/domains/agent_logs.py",
        "datagen/domains/content_moderation.py",
        "datagen/domains/customer_support.py",
        "datagen/domains/ecommerce_returns.py",
        "datagen/domains/education_grading.py",
        "datagen/domains/finance_ops.py",
        "datagen/domains/healthcare_admin.py",
        "datagen/domains/hr_recruiting.py",
        "datagen/domains/it_incidents.py",
        "datagen/domains/legal_contracts.py",
        "datagen/domains/logistics.py",
        "datagen/domains/security_ops.py",
        "datagen/freeze.py",
        "datagen/sample.py",
        "datagen/spec.py",
        "datagen/text.py",
        "engine.py",
        "plan.py",
        "relations/__init__.py",
        "relations/_shared.py",
        "relations/base.py",
        "relations/bat.py",
        "relations/cho.py",
        "relations/diagnostics.py",
        "relations/log.py",
        "relations/mea.py",
        "relations/rep.py",
        "render.py",
        "results.py",
        "rewrite.py",
        "schema.py",
        "score.py",
        "suites/__init__.py",
        "taxonomy.py",
    )
)


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _checked(path, expected, *, limit=256 * 1024**2):
    need(path.is_absolute() and path.resolve() == path, "ordinary selected path required")
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        need(
            stat.S_ISREG(before.st_mode) and before.st_size <= limit,
            "bounded regular file required",
        )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        after, current = os.fstat(fd), os.stat(path, follow_symlinks=False)
        keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
        need(
            tuple(getattr(before, x) for x in keys)
            == tuple(getattr(after, x) for x in keys)
            == tuple(getattr(current, x) for x in keys)
            and len(raw) == before.st_size
            and sha(raw) == expected,
            "selected source changed",
        )
        return raw
    finally:
        os.close(fd)


def _bootstrap():
    path = SCRIPTS / BOOT_NAME
    raw = _checked(path, BOOT_SHA, limit=8 * 1024**2)
    module = types.ModuleType("_reviewed_coherence_bootstrap")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    return module, raw


def _controlled_path(path, *, directory):
    need(path.is_absolute() and path.resolve() == path, "ordinary snapshot path required")
    info = path.lstat()
    need(
        (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
        and info.st_uid in {0, os.getuid()}
        and info.st_mode & 0o222 == 0
        and (directory or info.st_nlink == 1),
        "read-only controlled snapshot path required",
    )


def _official(config_raw):
    need(
        OFFICIAL.is_absolute() and OFFICIAL.resolve() == OFFICIAL,
        "ordinary official snapshot required",
    )
    config = json.loads(config_raw)
    files = config.get("files") if type(config) is dict else None
    selected = MINI_SDK | MINI_ASSETS
    need(
        len(MINI_SDK) == 42
        and len(MINI_ASSETS) == 6
        and len(selected) == 48
        and type(files) is dict
        and selected.issubset(files)
        and config.get("source_revision") == "e18733694623aa93058e279c3534f8b8e2edefa3"
        and all(
            type(files[name]) is str and re.fullmatch(r"[0-9a-f]{64}", files[name])
            for name in selected
        ),
        "fixed MINI inventory required",
    )
    _controlled_path(OFFICIAL, directory=True)
    directories = {OFFICIAL}
    for name in sorted(selected):
        path = OFFICIAL / name
        for parent in path.parents:
            if parent == OFFICIAL:
                break
            need(parent.is_relative_to(OFFICIAL), "snapshot path escaped")
            directories.add(parent)
    for directory in sorted(directories):
        _controlled_path(directory, directory=True)
    # The official loader examines these two directories before its hash checks.
    # Inspect names only, never opening unselected files or other directories.
    for suite, expected in (
        ("jevbench-mini", {"tests.jsonl.gz"}),
        ("jevbench-240", {"cases.jsonl", "rewrites.jsonl", "tests.jsonl.gz"}),
    ):
        directory = OFFICIAL / "src/jevbench/suites" / suite
        found = set()
        for entry in directory.iterdir():
            if entry.name.endswith((".jsonl", ".jsonl.gz")):
                need(not entry.is_symlink() and entry.is_file(), "ordinary suite data required")
                found.add(entry.name)
        need(found == expected, "official suite data inventory differs")
    for name in sorted(selected):
        path = OFFICIAL / name
        _controlled_path(path, directory=False)
    for name in sorted(selected):
        _checked(OFFICIAL / name, files[name], limit=16 * 1024**2)
    for directory in sorted(directories):
        _controlled_path(directory, directory=True)
    for name in sorted(selected):
        _controlled_path(OFFICIAL / name, directory=False)


def _delta(raw, expected):
    need(raw[:6] == b"\x93NUMPY" and raw[6:8] in (b"\x01\x00", b"\x02\x00"), "NPY version required")
    width = 2 if raw[6] == 1 else 4
    need(len(raw) >= 8 + width, "NPY header required")
    length = int.from_bytes(raw[8 : 8 + width], "little")
    need(0 < length <= 16384 and len(raw) >= 8 + width + length, "bounded NPY header required")
    header = ast.literal_eval(raw[8 + width : 8 + width + length].decode("latin1"))
    need(
        type(header) is dict
        and header == {"descr": "<f4", "fortran_order": False, "shape": (255, 1024)},
        "fixed FP32 matrix required",
    )
    body = raw[8 + width + length :]
    need(
        len(body) == 255 * 1024 * 4
        and sha(body) == expected
        and all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", body)),
        "finite FP32 delta required",
    )


def _gpu_uuid():
    result = subprocess.run(
        ["/usr/bin/nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout.strip()
    match = re.fullmatch(
        r"(GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}), NVIDIA GB10", result
    )
    need(match is not None, "one NVIDIA GB10 required")
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
    need(SOURCE_ONLY is False, "reviewed source-only preparer required")
    boot, bootstrap_raw = _bootstrap()
    need(
        boot.ROOT == ROOT
        and boot.SCRIPTS == SCRIPTS
        and boot.BASE == BASE
        and len(boot.source_set()) == 32
        and boot.SOURCE_ONLY is True,
        "fixed bootstrap inventory required",
    )
    need(
        OFFICIAL == Path("/tmp/jev-coherence-mini-controlled-snapshot"),
        "fixed official checkout required",
    )
    pins = {str(ROOT / name): value for name, value in boot.PUBLIC.items()}
    pins.update(boot.runtime_pins())
    pins[str(SCRIPTS / BOOT_NAME)] = BOOT_SHA
    raws = {str(SCRIPTS / BOOT_NAME): bootstrap_raw}
    for name, expected in pins.items():
        if name != str(SCRIPTS / BOOT_NAME):
            raws[name] = _checked(Path(name), expected)
    for held, active in (
        ("jev_decision3_jevbench_generic.py", "jev_decision3_jevbench_generic_release_20261027.py"),
        ("jev_gpu_forward_lease.py", "jev_gpu_forward_lease_release_20261027.py"),
    ):
        source, release = raws[str(SCRIPTS / held)], raws[str(SCRIPTS / active)]
        need(
            source.count(MARKER) == 1
            and release == source.replace(MARKER, b"SOURCE_ONLY = False", 1),
            "existing release differs",
        )
    _official(raws[str(ROOT / "configs/benchmarks/jevbench-coherence-public.json")])
    snapshot = (
        ROOT / "artifacts/jev-local-hf/hub/models--vllm-sr--d3-edge/snapshots" / boot.REVISION
    )
    _checked(snapshot / "MODEL_MANIFEST.json", boot.MANIFEST_SHA, limit=8 * 1024**2)
    for ref in boot.DELTAS.values():
        raw = _checked(Path(ref["path"]), ref["file_sha256"], limit=2 * 1024**2)
        _delta(raw, ref["fp32_sha256"])
    pairs = (
        (boot.BOOTSTRAP, boot.BOOT_RELEASE),
        (boot.OWNER, boot.OWNER_RELEASE),
        (boot.SUPERVISOR, boot.SUPERVISOR_RELEASE),
    )
    need(SCRIPTS.resolve() == SCRIPTS and SCRIPTS.is_dir(), "ordinary scripts directory required")
    for _held, release in pairs:
        if os.path.lexists(SCRIPTS / release):
            raise FileExistsError("reviewed release already exists: " + release)
    need(
        BASE.parent.resolve() == BASE.parent and BASE.parent.is_dir(),
        "ordinary experiment parent required",
    )
    if os.path.lexists(BASE):
        info = BASE.lstat()
        need(
            BASE.resolve() == BASE
            and stat.S_ISDIR(info.st_mode)
            and info.st_uid == os.getuid()
            and info.st_mode & 0o077 == 0,
            "owned private base required",
        )
    gpu_uuid = _gpu_uuid()
    nonce, lease_nonce = secrets.token_hex(16), secrets.token_hex(16)
    run = BASE / ("run-" + nonce)
    if os.path.lexists(run):
        raise FileExistsError("fresh run required")
    temps, published, dirs = [], [], []
    try:
        for index, (held, release) in enumerate(pairs):
            source = raws[str(SCRIPTS / held)]
            need(source.count(MARKER) == 1, "one source gate required")
            released = source.replace(MARKER, b"SOURCE_ONLY = False", 1)
            temp, target = (
                SCRIPTS / (".jev-coherence-" + nonce + "-" + str(index) + ".tmp"),
                SCRIPTS / release,
            )
            _exclusive(temp, released, temps)
            os.link(temp, target, follow_symlinks=False)
            published.append((target, temps[-1][1]))
            pins[str(target)] = sha(released)
        need(not _remove_owned(temps, []), "temporary release cleanup failed")
        temps.clear()
        if not BASE.exists():
            BASE.mkdir(mode=0o700)
            dirs.append((BASE, _identity(BASE)))
        info = BASE.lstat()
        need(
            BASE.resolve() == BASE
            and stat.S_ISDIR(info.st_mode)
            and info.st_uid == os.getuid()
            and info.st_mode & 0o077 == 0,
            "owned private base required",
        )
        run.mkdir(mode=0o700)
        dirs.append((run, _identity(run)))
        worker = run / "worker"
        worker.mkdir(mode=0o700)
        dirs.append((worker, _identity(worker)))
        for name in ("lock", "journal"):
            _exclusive(run / name, b"", published)
        transport = {
            "lease_release": str(SCRIPTS / "jev_gpu_forward_lease_release_20261027.py"),
            "lock": str(run / "lock"),
            "journal": str(run / "journal"),
            "nonce": lease_nonce,
            "stat": {
                name: dict(zip(("device", "inode"), _identity(run / name), strict=True))
                for name in ("lock", "journal")
            },
        }
        plan = {
            "schema": "decision3-jevbench-coherence-owner-plan-v1",
            "unit": "jev-coherence-" + nonce + ".service",
            "output_dir": str(worker),
            "python": "/home/albert_hsieh/ComfyUI/.venv/bin/python",
            "cache_root": str(ROOT / "artifacts/jev-local-hf/hub"),
            "modules_cache": str(
                ROOT
                / "artifacts/jev-runtime/decision3-link-navigation-diagnostic-20261027-01/staged-modules"
            ),
            "overlay": str(ROOT / "artifacts/jev-runtime/overlay"),
            "source_pins": pins,
            "gpu_uuid": gpu_uuid,
            "deltas": boot.DELTAS,
            "transport": transport,
            "generic_release": str(SCRIPTS / "jev_decision3_jevbench_generic_release_20261027.py"),
            "tasks_root": str(OFFICIAL),
            "wall_seconds": 1800,
            "request_seconds": 180,
        }
        need(
            set(pins) == boot.source_set() and len(pins) == 32,
            "complete reviewed source inventory required",
        )
        raw = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        temp, target = run / (".plan-" + nonce + ".tmp"), run / "plan.json"
        _exclusive(temp, raw, temps)
        os.link(temp, target, follow_symlinks=False)
        published.append((target, temps[-1][1]))
        need(not _remove_owned(temps, []), "temporary plan cleanup failed")
        temps.clear()
        return {
            "plan": str(target),
            "sha256": sha(raw),
            "report": str(run / "supervisor"),
            "pin_count": len(pins),
        }
    except BaseException as error:
        if _remove_owned(temps + published, dirs):
            error.add_note("owned preparation cleanup incomplete")
        raise


if __name__ == "__main__":
    print(json.dumps(prepare(), separators=(",", ":")))
