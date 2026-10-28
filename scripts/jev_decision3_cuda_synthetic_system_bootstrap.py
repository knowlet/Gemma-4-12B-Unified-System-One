"""Held stdlib-first loader for one reviewed synthetic CUDA owner.

The caller must independently authenticate this bootstrap release, plan digest,
interpreter and standard-library environment before running it in isolated mode.
Only exact plan-pinned workspace module bytes are compiled. External packages,
model internals and native libraries are not a closed transitive provenance set.
The system broker has an independently supplied SHA and must be installed in a
root-controlled fixed path through a separately authorized process. Its
pre-execution authentication and execution are not atomic. Nothing here grants
privileges, deploys the broker, or authorizes a system transient service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
import types
from pathlib import Path

SOURCE_ONLY = True
ROOT = Path(__file__).absolute().parent.parent
SCRIPTS = ROOT / "scripts"
BASE = ROOT / "artifacts/jev-runtime/decision3-cuda-synthetic-system"
CACHE = ROOT / "artifacts/jev-local-hf/hub"
SNAPSHOT = CACHE / "models--vllm-sr--d3-edge/snapshots/cab71e6ad3461b8086d51ddb572c656babd63b0b"
STAGED = (
    ROOT
    / "artifacts/jev-runtime/decision3-link-navigation-diagnostic-20261027-01/staged-modules/transformers_modules/cab71e6ad3461b8086d51ddb572c656babd63b0b/5e85afd45ec2b404"
)
OVERLAY = ROOT / "artifacts/jev-runtime/overlay"
TORCH = Path("/home/albert_hsieh/ComfyUI/.venv/lib/python3.13/site-packages/torch/__init__.py")
PYTHON_REAL = Path(
    "/home/albert_hsieh/.local/share/uv/python/cpython-3.13.14-linux-aarch64-gnu/bin/python3.13"
)
FILES = {
    "bootstrap": "jev_decision3_cuda_synthetic_system_bootstrap.py",
    "launcher": "jev_decision3_cuda_synthetic_system_launcher.py",
    "supervisor": "jev_decision3_cuda_synthetic_system_supervisor.py",
    "worker": "jev_decision3_cuda_synthetic_worker.py",
}
PROJECT = (
    ("s1", "__init__.py", True),
    ("s1.contracts", "contracts.py", False),
    ("s1.errors", "errors.py", False),
    ("s1.backends", "backends.py", False),
    ("s1.evaluation", "evaluation/__init__.py", True),
    ("s1.evaluation.contracts", "evaluation/contracts.py", False),
    ("s1.evaluation.decision3", "evaluation/decision3.py", False),
)
SUPPORT = (
    ("jev_stage_memory_profile_v2", "jev_stage_memory_profile_v2.py"),
    ("jev_decision3_candidate_http_owner", "jev_decision3_candidate_http_owner.py"),
    ("jev_decision3_candidate_browser_child", "jev_decision3_candidate_browser_child.py"),
    ("jev_decision3_cpu_experiment_worker", "jev_decision3_cpu_experiment_worker.py"),
)


def _need(value, message):
    if not value:
        raise ValueError(message)


def _release(name):
    return SCRIPTS / (FILES[name].removesuffix(".py") + "_release.py")


def _source_set():
    paths = [SCRIPTS / name for name in FILES.values()] + [_release(name) for name in FILES]
    paths += [SCRIPTS / name for _, name in SUPPORT]
    paths += [ROOT / "src/s1" / relative for _, relative, _ in PROJECT]
    paths += [
        STAGED / "d3_runtime.py",
        STAGED / "d3_format.py",
        STAGED / "modeling_d3.py",
        SNAPSHOT / "d3_fast.py",
        OVERLAY / "transformers/__init__.py",
        OVERLAY / "transformers/models/qwen3_5/modeling_qwen3_5.py",
        TORCH,
        PYTHON_REAL,
    ]
    return {str(path) for path in paths}


def _read(path, expected, *, limit=8 * 1024**2, allow_empty=False):
    _need(
        path.is_absolute() and path.resolve() == path and not path.is_symlink(),
        "ordinary fixed source required",
    )
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _need(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and (allow_empty or before.st_size > 0)
            and before.st_size <= limit,
            "bounded regular source required",
        )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        after, current = os.fstat(fd), os.stat(path, follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        _need(
            tuple(getattr(before, key) for key in fields)
            == tuple(getattr(after, key) for key in fields)
            == tuple(getattr(current, key) for key in fields)
            and len(data) == before.st_size
            and hashlib.sha256(data).hexdigest() == expected,
            "selected source bytes changed",
        )
        return data
    finally:
        os.close(fd)


def _decode(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            _need(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        raw, object_pairs_hook=pairs, parse_constant=lambda _: _need(False, "nonfinite JSON")
    )


def _plan(path, expected):
    _need(
        type(path) is str and type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected),
        "explicit plan identity required",
    )
    selected = Path(path)
    _need(
        selected.is_absolute()
        and selected.name == "plan.json"
        and selected.parent.parent == BASE
        and re.fullmatch(r"run-[0-9a-f]{32}", selected.parent.name),
        "dedicated plan required",
    )
    plan = _decode(_read(selected, expected, limit=1024 * 1024))
    _need(
        type(plan) is dict
        and plan.get("schema") == "decision3-cuda-synthetic-system-plan-v1"
        and type(plan.get("source_pins")) is dict
        and set(plan["source_pins"]) == _source_set()
        and all(
            type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in plan["source_pins"].values()
        ),
        "exact reviewed source inventory required",
    )
    return plan


def _module(name, path, data, *, package=False):
    _need(name not in sys.modules, "fresh module namespace required")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name if package else name.rpartition(".")[0]
    if package:
        module.__path__ = [str(path.parent)]
    sys.modules[name] = module
    try:
        exec(compile(data, str(path), "exec", dont_inherit=True), module.__dict__)
        _need(sys.modules.get(name) is module, "module identity changed")
        return module
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise


BROKER_ROOT = Path("/run")


def _root_controlled(info):
    return info.st_uid == 0 and info.st_mode & 0o022 == 0


class SystemControl:
    """Adapter to one externally authorized, root-controlled broker release."""

    def __init__(self, plan_path, plan_sha, broker_sha):
        selected = Path(plan_path)
        _need(
            type(plan_path) is str
            and selected.is_absolute()
            and selected.name == "plan.json"
            and selected.parent.parent == BASE
            and re.fullmatch(r"run-[0-9a-f]{32}", selected.parent.name)
            and type(plan_sha) is str
            and re.fullmatch(r"[0-9a-f]{64}", plan_sha)
            and type(broker_sha) is str
            and re.fullmatch(r"[0-9a-f]{64}", broker_sha),
            "fixed plan and externally authenticated broker SHA required",
        )
        nonce = selected.parent.name[4:]
        self._unit = "jev-cuda-synthetic-system-" + nonce + ".service"
        self._plan = plan_path
        self._plan_sha = plan_sha
        self._broker_sha = broker_sha
        self._broker = BROKER_ROOT / ("jev-cuda-synthetic-system-broker-" + nonce + ".py")

    def _call(self, action, *, timeout=None):
        _need(action in {"start", "show", "stop"}, "fixed broker operation required")
        _need(
            (action == "show" and type(timeout) in (int, float) and 0 < timeout <= 2)
            or (action != "show" and timeout is None),
            "bounded broker timeout required",
        )
        deadline = time.monotonic() + timeout if action == "show" else None
        _need(os.getuid() == 1001 and os.geteuid() == 1001, "unprivileged supervisor UID required")
        _need(
            BROKER_ROOT.is_absolute()
            and BROKER_ROOT.resolve() == BROKER_ROOT
            and BROKER_ROOT.is_dir()
            and _root_controlled(BROKER_ROOT.stat()),
            "root-controlled broker directory required",
        )
        info = self._broker.lstat()
        _need(
            stat.S_ISREG(info.st_mode) and _root_controlled(info),
            "root-controlled broker file required",
        )
        _read(self._broker, self._broker_sha)
        # Include source checks in the caller's budget. Python cannot guarantee
        # interruption of a blocked filesystem operation or process creation.
        remaining = deadline - time.monotonic() if deadline is not None else None
        if remaining is not None and remaining <= 0:
            raise TimeoutError("broker query deadline exceeded before process start")
        # Noninteractive sudo requires separately authorized privilege; this code
        # does not install sudoers or choose a different privilege mechanism.
        proc = subprocess.run(
            [
                "/usr/bin/sudo",
                "-n",
                "--",
                "/usr/bin/python3",
                "-I",
                "-B",
                str(self._broker),
                action,
            ],
            capture_output=True,
            check=True,
            timeout=remaining if action == "show" else {"start": 12, "stop": 22}[action],
            cwd="/",
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
        _need(
            len(proc.stdout) <= 1024 * 1024 and len(proc.stderr) <= 1024 * 1024,
            "bounded broker output required",
        )
        result = _decode(proc.stdout)
        _need(type(result) is dict, "broker result object required")
        return result

    def start(self, plan_path, plan_sha):
        _need(plan_path == self._plan and plan_sha == self._plan_sha, "fixed broker plan required")
        result = self._call("start")
        _need(
            result.get("status") == "started"
            and result.get("unit") == self._unit
            and result.get("plan_sha256") == self._plan_sha,
            "bound broker start required",
        )
        return result

    def show(self, unit, *, timeout=2):
        _need(
            unit == self._unit and type(timeout) in (int, float) and 0 < timeout <= 2,
            "fixed broker unit query required",
        )
        return self._call("show", timeout=timeout)

    def stop(self, unit, invocation, hello):
        _need(
            unit == self._unit
            and type(invocation) is str
            and re.fullmatch(r"[0-9a-f]{32}", invocation)
            and type(hello) is dict
            and hello.get("unit") == self._unit
            and hello.get("invocation") == invocation
            and hello.get("plan_sha256") == self._plan_sha,
            "bound broker stop required",
        )
        result = self._call("stop")
        _need(
            result.get("status") == "stop_returned_closure_unconfirmed"
            and result.get("unit") == self._unit
            and result.get("invocation") == invocation,
            "bound broker stop result required",
        )
        return result


def prepare(plan_path, plan_sha, *, mode, broker_sha=None):
    _need(SOURCE_ONLY is False, "reviewed CUDA bootstrap release required")
    _need(mode in {"launcher", "supervisor"}, "explicit bootstrap mode required")
    _need(mode == "supervisor" or broker_sha is None, "launcher cannot select a broker")
    control = SystemControl(plan_path, plan_sha, broker_sha) if mode == "supervisor" else None
    plan = _plan(plan_path, plan_sha)
    pins = plan["source_pins"]
    # Authenticate every fixed bootstrap dependency before executing ANY of them.
    verified = {}
    for name in FILES:
        held, released = SCRIPTS / FILES[name], _release(name)
        source = _read(held, pins[str(held)])
        replacement = _read(released, pins[str(released)])
        marker = b"SOURCE_ONLY" + b" = True"
        _need(
            source.count(marker) == 1
            and replacement == source.replace(marker, b"SOURCE_ONLY = False", 1),
            "gate-only source release required",
        )
        verified[str(held)], verified[str(released)] = source, replacement
    for _, relative in SUPPORT:
        path = SCRIPTS / relative
        verified[str(path)] = _read(path, pins[str(path)])
    for _, relative, _ in PROJECT:
        path = ROOT / "src/s1" / relative
        verified[str(path)] = _read(path, pins[str(path)], allow_empty=True)
    required_names = [name for name, _, _ in PROJECT] + [name for name, _ in SUPPORT]
    required_names += ["jev_decision3_cuda_synthetic_launcher", "_cuda_synthetic_entry"]
    _need(
        not any(name in sys.modules for name in required_names)
        and not any(name.startswith("s1.") for name in sys.modules),
        "fresh selected module names required",
    )
    loaded = {}
    try:
        for name, relative, package in PROJECT:
            path = ROOT / "src/s1" / relative
            loaded[name] = _module(name, path, verified[str(path)], package=package)
        for name, relative in SUPPORT:
            path = SCRIPTS / relative
            loaded[name] = _module(name, path, verified[str(path)])
        held_launcher = SCRIPTS / FILES["launcher"]
        loaded["jev_decision3_cuda_synthetic_launcher"] = _module(
            "jev_decision3_cuda_synthetic_launcher", held_launcher, verified[str(held_launcher)]
        )
        bindings = {
            "common": loaded["jev_decision3_candidate_http_owner"],
            "child": loaded["jev_decision3_candidate_browser_child"],
            "cpu": loaded["jev_decision3_cpu_experiment_worker"],
        }
        trusted_launcher = loaded["jev_decision3_cuda_synthetic_launcher"]
        for key, value in bindings.items():
            setattr(trusted_launcher, key, value)
        _need(
            trusted_launcher._source_set() == _source_set(),
            "bootstrap and launcher allowlists differ",
        )
        trusted_launcher._plan(plan_path, plan_sha)
        path = _release(mode)
        entry = _module("_cuda_synthetic_entry", path, verified[str(path)])
        loaded["_cuda_synthetic_entry"] = entry
        for key, value in bindings.items():
            setattr(entry, key, value)
        if mode == "supervisor":
            entry.launcher = trusted_launcher
            entry.system_control = control
        _need(entry.SOURCE_ONLY is False, "reviewed entry gate required")
        return entry
    except BaseException:
        for name, module in reversed(tuple(loaded.items())):
            if sys.modules.get(name) is module:
                del sys.modules[name]
        raise


def main():
    _need(SOURCE_ONLY is False, "reviewed CUDA bootstrap release required")
    _need(
        sys.flags.isolated == 1 and Path(__file__).absolute() == _release("bootstrap"),
        "isolated reviewed bootstrap entry required",
    )
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", choices=("supervisor", "launcher"), required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--report-dir")
    parser.add_argument("--broker-sha256")
    args = parser.parse_args()
    _need(
        (args.mode == "supervisor" and args.broker_sha256 is not None)
        or (args.mode == "launcher" and args.broker_sha256 is None),
        "supervisor requires external broker SHA; launcher must not receive one",
    )
    entry = prepare(args.plan, args.plan_sha256, mode=args.mode, broker_sha=args.broker_sha256)
    if args.mode == "launcher":
        _need(args.report_dir is None, "launcher does not accept a report directory")
        entry.run(args.plan, args.plan_sha256)
    else:
        _need(args.report_dir is not None, "supervisor report directory required")
        entry.supervise(args.plan, args.plan_sha256, args.report_dir)


if __name__ == "__main__":
    main()
