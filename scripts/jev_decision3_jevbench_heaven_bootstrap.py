"""Held, stdlib-first bootstrap for the two supervised Heaven entry points.

Only exact pinned project source bytes are executed after verification. The
bootstrap itself, Python/stdlib, native extensions and dynamic HF imports have
separate trust boundaries; this is not a closed or atomic supply-chain proof.
The caller must authenticate this source, interpreter and plan SHA externally.
Both entries launch the fixed bootstrap release by path using isolated Python;
stdin execution is deliberately unsupported. Python loads that file before
this module can verify any pins, leaving an external-authentication race.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import types
from pathlib import Path

SOURCE_ONLY = True
ROOT = Path(__file__).absolute().parent.parent
SCRIPTS = ROOT / "scripts"
BASE = ROOT / "artifacts/jev-runtime/decision3-heaven-v2"
REVISION = "cab71e6ad3461b8086d51ddb572c656babd63b0b"
MANIFEST_SHA = "3bae8989a6feb36a66bf4d8c44112dc63199d0dc69f564dc895fe368b1e08e59"
BOOTSTRAP = "jev_decision3_jevbench_heaven_bootstrap.py"
BOOT_RELEASE = "jev_decision3_jevbench_heaven_bootstrap_release_v2_20261028.py"
OWNER = "jev_decision3_jevbench_heaven_owner.py"
OWNER_RELEASE = "jev_decision3_jevbench_heaven_owner_release_v2_20261028.py"
SUPERVISOR = "jev_decision3_jevbench_heaven_supervisor.py"
SUPERVISOR_RELEASE = "jev_decision3_jevbench_heaven_supervisor_release_v2_20261028.py"
DELTAS = {
    "operation": {
        "path": str(
            ROOT
            / "artifacts/jev-runtime/decision3-link-navigation-diagnostic-20261027-01/candidate-fit-01/operation-delta.npy"
        ),
        "file_sha256": "eb60dd85af2dcb4eae50ac7983b2936cabf5fe24cc6d0e731392435c7fb1f1dc",
        "fp32_sha256": "c60619bac35d4acb668abc78f8cbe74bf43f6d037701fdb40a1e96a3f4cba331",
    },
    "click_target": {
        "path": str(
            ROOT
            / "artifacts/jev-runtime/decision3-link-navigation-diagnostic-20261027-01/candidate-fit-01/click_target-delta.npy"
        ),
        "file_sha256": "28f3d67fec754f0551d4e1243aca21d738a91a9c91fa36f2ca740052f98d1534",
        "fp32_sha256": "84dfef3fae8e1d4af6cf5c78c708373092f9866eb57660976db0289ad7e22870",
    },
}

PROJECT = (
    ("s1", "src/s1/__init__.py", True),
    ("s1.contracts", "src/s1/contracts.py", False),
    ("s1.errors", "src/s1/errors.py", False),
    ("s1.backends", "src/s1/backends.py", False),
    ("s1.evaluation", "src/s1/evaluation/__init__.py", True),
    ("s1.evaluation.contracts", "src/s1/evaluation/contracts.py", False),
    ("s1.evaluation.decision3", "src/s1/evaluation/decision3.py", False),
)
SUPPORT = (
    ("jev_stage_memory_profile_v2", "scripts/jev_stage_memory_profile_v2.py"),
    ("jev_decision3_candidate_http_owner", "scripts/jev_decision3_candidate_http_owner.py"),
    ("jev_decision3_candidate_browser_child", "scripts/jev_decision3_candidate_browser_child.py"),
    ("jev_decision3_runtime_integrity", "scripts/jev_decision3_runtime_integrity.py"),
    ("run_jevbench_public", "scripts/run_jevbench_public.py"),
)
PUBLIC = {
    "scripts/jev_decision3_jevbench_heaven_owner.py": "5e74c53ca5795652e74f95726b61284ac9e63456f2a852804de108b6d3498dda",
    "scripts/jev_decision3_jevbench_heaven_supervisor.py": "58a7f0696709da52cec3db7bdff52f1a8f2db4f878fb4718d9665be164619f99",
    "scripts/jev_decision3_candidate_browser_child.py": "5b71c8a2a927f50edc7f3333a3a0771854553978de5ad27aea3421cdcbd822c8",
    "scripts/jev_decision3_candidate_http_owner.py": "d7a9183ff5bd7fef3d4b48564dce5131109eeff3d17ec6358d99e35c35c1923b",
    "scripts/jev_decision3_runtime_integrity.py": "0482b6da62c53efa90900ef15a6c86714c8acf81b0357de86e53739c611f582c",
    "scripts/jev_stage_memory_profile_v2.py": "c1ae9e12de593e5ba1f13b57b15edf033116423b2218ca766ea84a0558e76749",
    "scripts/jev_decision3_jevbench_generic.py": "e77c43c669eef25043f7f13d0ddb83242b6e3c0def744fc6768cbbd8a683d21f",
    "scripts/jev_decision3_jevbench_generic_release_20261027.py": "04749d676a26453ed71f4b63435e5bbe6a9c3831a5e12864e44a5effd0dd5b5f",
    "scripts/jev_gpu_forward_lease.py": "256aaefbda3148a506043dc81b2211a57ea8adb79e61dd7a9d34c29153363325",
    "scripts/jev_gpu_forward_lease_release_20261027.py": "2b8774e31dee2db9ddb209df784bad6375afc12f2f015dd2a8e5c39c5140b3eb",
    "scripts/run_jevbench_public.py": "cb476d16a6e89ee5223b928e3a550af18de8534934ae5431563503db9f909bd1",
    "configs/benchmarks/jevbench-public.json": "206ca6745290cc4bc5c577d64addad7814381cdc6d336b5b7c09cb8992af8754",
    "src/s1/__init__.py": "c9c7ba0ea4cc81609fcf1f0ff2a853243a173a994257e0b9dd19be1a2e6dd52e",
    "src/s1/backends.py": "90c084fa5523dea71b49cf140ecfda1fd2165a9468368624bde6161e2dbf67bf",
    "src/s1/contracts.py": "24c24c3ded9c35bfd263a4488bf88916a8960fa0b79283ccedbac232983f8a0a",
    "src/s1/errors.py": "e5ad1d9b7e700f07a704d74d3e4ade7156c7a9eaad304d3fe4296bad0dc8c997",
    "src/s1/evaluation/__init__.py": "127cf68d534aa63995896f51d47ff5015c1d8fcb3563914752aa130f5ff25f51",
    "src/s1/evaluation/contracts.py": "ead69fb44ba1a0221b9c73fc0c811774ed15b530cdf1ec56f99e7b98bff254d3",
    "src/s1/evaluation/decision3.py": "aaa21ab6f2ecf5ae0e00694b43755541dc75fc0310ba10d945ade2a4511add52",
}


def need(value, message):
    if not value:
        raise ValueError(message)


def runtime_pins():
    staged = (
        ROOT
        / "artifacts/jev-runtime/decision3-link-navigation-diagnostic-20261027-01/staged-modules/transformers_modules/cab71e6ad3461b8086d51ddb572c656babd63b0b/5e85afd45ec2b404"
    )
    snap = ROOT / "artifacts/jev-local-hf/hub/models--vllm-sr--d3-edge/snapshots" / REVISION
    overlay = ROOT / "artifacts/jev-runtime/overlay"
    return {
        str(
            staged / "d3_runtime.py"
        ): "adf7f7aa085674b5682e492d8b2e3cc4f11c67b0d153515c742e907e9b550917",
        str(
            staged / "d3_format.py"
        ): "3badfbc70561bb3f5b8ec16a11d2c455b96ddb32caace0a8a54ca102eae1f3fa",
        str(
            staged / "modeling_d3.py"
        ): "f8f8db23ff58f1a990ffa24977d744a50e1b4446d02a64c22070967705b6dbef",
        str(
            snap / "d3_fast.py"
        ): "68a1fb9a2c3bde9ac2763bf504decae592ca12722698fbde1b39a31262955535",
        str(
            overlay / "transformers/__init__.py"
        ): "2725cadd5afaa29b2a12faec1960cb7ac1df77dc76d655310efb33d4607b35f5",
        str(
            overlay / "transformers/models/qwen3_5/modeling_qwen3_5.py"
        ): "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b",
        "/home/albert_hsieh/ComfyUI/.venv/lib/python3.13/site-packages/torch/__init__.py": "cf40c075c95864036e835795756d69b8cccfafa76f3bcde5eba9d06065ccd3d1",
        "/home/albert_hsieh/.local/share/uv/python/cpython-3.13.14-linux-aarch64-gnu/bin/python3.13": "92ed50fd9dde3654d421d165214a95361e1889210b3d2063001d6c2e75eef2ab",
    }


def _read(path, expected, limit=256 * 1024**2):
    need(path.is_absolute() and path.resolve() == path, "ordinary source path required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        first = os.fstat(fd)
        need(stat.S_ISREG(first.st_mode) and first.st_size <= limit, "bounded source required")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        last, current = os.fstat(fd), os.stat(path, follow_symlinks=False)
        keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        need(
            tuple(getattr(first, x) for x in keys)
            == tuple(getattr(last, x) for x in keys)
            == tuple(getattr(current, x) for x in keys)
            and len(data) == first.st_size
            and hashlib.sha256(data).hexdigest() == expected,
            "source identity or digest changed",
        )
        return data
    finally:
        os.close(fd)


def _decode(data):
    def pairs(values):
        result = {}
        for key, value in values:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        data, object_pairs_hook=pairs, parse_constant=lambda _: need(False, "nonfinite JSON")
    )


def _module(name, path, data, *, package=False):
    need(name not in sys.modules, "fresh module namespace required")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name if package else name.rpartition(".")[0]
    if package:
        module.__path__ = [str(path.parent)]
    sys.modules[name] = module
    try:
        exec(compile(data, str(path), "exec", dont_inherit=True), module.__dict__)
        need(sys.modules.get(name) is module, "module identity changed")
        return module
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise


def source_set():
    return (
        {str(ROOT / path) for path in PUBLIC}
        | set(runtime_pins())
        | {
            str(SCRIPTS / name)
            for name in (BOOTSTRAP, BOOT_RELEASE, OWNER_RELEASE, SUPERVISOR_RELEASE)
        }
    )


def _plan(path, expected):
    selected = Path(path)
    need(
        type(expected) is str
        and re.fullmatch(r"[0-9a-f]{64}", expected)
        and selected.is_absolute()
        and selected.name == "plan.json"
        and selected.parent.parent == BASE
        and re.fullmatch(r"run-[0-9a-f]{32}", selected.parent.name),
        "fixed plan identity required",
    )
    plan = _decode(_read(selected, expected, 1024**2))
    need(
        type(plan) is dict
        and plan.get("schema") == "decision3-jevbench-heaven-owner-plan-v1"
        and plan.get("unit") == "jev-heaven-" + selected.parent.name[4:] + ".service"
        and plan.get("output_dir") == str(selected.parent / "worker")
        and plan.get("python") == "/home/albert_hsieh/ComfyUI/.venv/bin/python"
        and plan.get("cache_root") == str(ROOT / "artifacts/jev-local-hf/hub")
        and plan.get("modules_cache")
        == str(
            ROOT
            / "artifacts/jev-runtime/decision3-link-navigation-diagnostic-20261027-01/staged-modules"
        )
        and plan.get("overlay") == str(ROOT / "artifacts/jev-runtime/overlay")
        and plan.get("tasks_root") == "/tmp/jev-heaven-public-inventory-20261027"
        and plan.get("generic_release")
        == str(SCRIPTS / "jev_decision3_jevbench_generic_release_20261027.py")
        and plan.get("deltas") == DELTAS,
        "fixed Heaven inputs required",
    )
    transport = plan.get("transport")
    need(
        type(transport) is dict
        and transport.get("lease_release")
        == str(SCRIPTS / "jev_gpu_forward_lease_release_20261027.py")
        and transport.get("lock") == str(selected.parent / "lock")
        and transport.get("journal") == str(selected.parent / "journal"),
        "fixed private lease required",
    )
    pins = plan.get("source_pins") if type(plan) is dict else None
    need(
        type(pins) is dict
        and set(pins) == source_set()
        and all(type(x) is str and re.fullmatch(r"[0-9a-f]{64}", x) for x in pins.values()),
        "exact source allowlist required",
    )
    need(
        all(pins[str(ROOT / name)] == sha for name, sha in PUBLIC.items())
        and all(pins[path] == sha for path, sha in runtime_pins().items()),
        "fixed source digests required",
    )
    return plan


def _verified_bytes(plan):
    pins = plan["source_pins"]
    files = {name: _read(Path(name), pins[name]) for name in sorted(source_set())}
    for held, release in (
        (BOOTSTRAP, BOOT_RELEASE),
        (OWNER, OWNER_RELEASE),
        (SUPERVISOR, SUPERVISOR_RELEASE),
        ("jev_decision3_jevbench_generic.py", "jev_decision3_jevbench_generic_release_20261027.py"),
        ("jev_gpu_forward_lease.py", "jev_gpu_forward_lease_release_20261027.py"),
    ):
        source, target = files[str(SCRIPTS / held)], files[str(SCRIPTS / release)]
        marker = b"\nSOURCE_ONLY" + b" = True\n"
        need(
            source.count(marker) == 1
            and target == source.replace(marker, b"\nSOURCE_ONLY = False\n", 1),
            "gate-only release required",
        )
    manifest = (
        ROOT
        / "artifacts/jev-local-hf/hub/models--vllm-sr--d3-edge/snapshots"
        / REVISION
        / "MODEL_MANIFEST.json"
    )
    _read(manifest, MANIFEST_SHA, 8 * 1024**2)
    return files


def _unit_args_wrapper(original, plan, path, sha, release):
    args = original(plan, path, sha, release)
    old = [plan["python"], "-B", "-s", str(release), "--plan", str(path), "--plan-sha256", sha]
    need(args[-len(old) :] == old, "unexpected worker launch contract")
    return args[: -len(old)] + [
        plan["python"],
        "-I",
        "-B",
        "-s",
        str(SCRIPTS / BOOT_RELEASE),
        "--mode",
        "owner",
        "--plan",
        str(path),
        "--plan-sha256",
        sha,
    ]


def prepare(path, sha, *, mode):
    need(SOURCE_ONLY is False, "reviewed bootstrap release required")
    need(mode in {"owner", "supervisor"}, "explicit mode required")
    plan = _plan(path, sha)
    files = _verified_bytes(plan)
    names = (
        [name for name, _, _ in PROJECT]
        + [name for name, _ in SUPPORT]
        + ["jev_decision3_jevbench_heaven_owner", "_heaven_entry"]
    )
    need(
        not any(name in sys.modules for name in names)
        and not any(name.startswith("s1.") for name in sys.modules),
        "fresh module namespace required",
    )
    loaded = {}
    try:
        for name, relative, package in PROJECT:
            target = ROOT / relative
            loaded[name] = _module(name, target, files[str(target)], package=package)
        for name, relative in SUPPORT:
            target = ROOT / relative
            loaded[name] = _module(name, target, files[str(target)])
        held = SCRIPTS / OWNER
        owner = _module("jev_decision3_jevbench_heaven_owner", held, files[str(held)])
        loaded["jev_decision3_jevbench_heaven_owner"] = owner
        owner._validate_plan(plan)
        target = SCRIPTS / (OWNER_RELEASE if mode == "owner" else SUPERVISOR_RELEASE)
        entry = _module("_heaven_entry", target, files[str(target)])
        loaded["_heaven_entry"] = entry
        need(entry.SOURCE_ONLY is False, "reviewed entry gate required")
        if mode == "supervisor":
            original = entry._unit_args
            entry._unit_args = lambda p, pp, ps, release: _unit_args_wrapper(
                original, p, pp, ps, release
            )
        return entry
    except BaseException:
        for name, module in reversed(tuple(loaded.items())):
            if sys.modules.get(name) is module:
                del sys.modules[name]
        raise


def main():
    need(
        SOURCE_ONLY is False
        and sys.flags.isolated == 1
        and Path(__file__).absolute() == SCRIPTS / BOOT_RELEASE,
        "isolated reviewed entry required",
    )
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", choices=("supervisor", "owner"), required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--report-dir")
    parser.add_argument("--memory-floor-bytes", type=int)
    args = parser.parse_args()
    entry = prepare(args.plan, args.plan_sha256, mode=args.mode)
    if args.mode == "owner":
        need(args.report_dir is None and args.memory_floor_bytes is None, "owner options invalid")
        return entry.run(args.plan, args.plan_sha256)
    need(
        args.report_dir is not None and args.memory_floor_bytes is not None,
        "supervisor options required",
    )
    return entry.supervise(
        args.plan,
        args.plan_sha256,
        str(SCRIPTS / OWNER_RELEASE),
        args.report_dir,
        args.memory_floor_bytes,
    )


if __name__ == "__main__":
    main()
