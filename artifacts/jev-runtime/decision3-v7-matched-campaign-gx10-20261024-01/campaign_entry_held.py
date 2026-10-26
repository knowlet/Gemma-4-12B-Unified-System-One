"""HELD entry for one explicitly selected integrated NF4/D3 campaign session.

The owner source pin is one-way: this entry pins owner; owner selects this entry
through the plan's source union instead of embedding a reciprocal entry hash.
"""

import argparse
import hashlib
import os
import re
import runpy
import signal
import stat
import sys
import types
from pathlib import Path

HERE = Path(__file__).absolute().parent
SOURCE_ONLY = True
OWNER_SOURCE = HERE / "owner_held.py"
OWNER_SOURCE_SHA256 = None


def require(value, message):
    if not value:
        raise ValueError(message)


def released():
    require(SOURCE_ONLY is False, "SOURCE_ONLY: matched campaign entry is held before arguments/IO")
    require(
        type(OWNER_SOURCE_SHA256) is str and re.fullmatch(r"[0-9a-f]{64}", OWNER_SOURCE_SHA256),
        "new outer owner source pin remains pending",
    )


def load_owner():
    released()
    path = OWNER_SOURCE
    require(
        path.parent == HERE and not any(item.is_symlink() for item in (path, *path.parents)),
        "Ordinary fixed selected outer owner source required",
    )
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(1024**2 + 1)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(
        stat.S_ISREG(before.st_mode)
        and 0 < len(raw) == before.st_size <= 1024**2
        and all(
            getattr(before, key) == getattr(after, key) == getattr(path.stat(), key)
            for key in fields
        )
        and hashlib.sha256(raw).hexdigest() == OWNER_SOURCE_SHA256,
        "Selected outer owner source bytes/identity changed",
    )
    name = "_selected_matched_nf4_d3_outer_owner"
    require(name not in sys.modules, "Fresh selected outer owner import required")
    owner = types.ModuleType(name)
    owner.__file__ = str(path)
    sys.modules[name] = owner
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), vars(owner))
        owner.released()
        require(
            owner.PROFILE == "jev-nf4-d3-matched-v7-owned-v1"
            and owner.UNIT == "jev-nf4-d3-matched-v7-20261024-01.service"
            and (owner.OUTER_SECONDS, owner.INNER_SECONDS) == (10800, 10740),
            "Exact new integrated owner profile/deadlines required",
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return owner


def main(argv=None):
    released()  # Before parser, owner source read, host/model or launch work.
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not args.run:
        print("DRY_NO_MODEL_OR_BROWSER; selected matched campaign only")
        return 0
    owner = load_owner()
    owner.require(
        os.environ.get("JEV_NATIVE_BROWSER_EXECUTE") == owner.PROFILE,
        "Owned matched campaign execution environment required",
    )
    owner.require(args.plan and args.plan_sha256, "Explicit sealed integrated plan required")
    plan = owner.load_plan(args.plan, args.plan_sha256)
    owner.require(os.environ.get("JEV_NATIVE_BROWSER_UNIT") == plan["unit"], "Owned unit binding")
    directory, output = owner.run_paths(plan)
    owner.require(
        directory.is_dir()
        and not output.exists()
        and not (directory / "actor-intent.json").exists(),
        "Fresh owned campaign actor output",
    )
    if os.getsid(0) != os.getpid() or os.getpgrp() != os.getpid():
        os.setsid()
    raw = (Path("/proc") / str(os.getpid()) / "stat").read_text()
    fields = raw[raw.rfind(")") + 2 :].split()
    actor = {
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "sid": os.getsid(0),
        "start_ticks": int(fields[19]),
    }
    owner.require(actor["pid"] == actor["pgid"] == actor["sid"], "Owned session leader")
    owner.save(
        directory,
        "actor-intent.json",
        {
            "actor": actor,
            "argv": sys.argv,
            "unit": plan["unit"],
            "nonce": plan["nonce"],
            "boot_id": plan["boot_id"],
            "plan_sha256": args.plan_sha256,
        },
    )

    def expired(_signum, _frame):
        raise TimeoutError("New integrated matched campaign cooperative10740-second deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(owner.INNER_SECONDS)
    try:
        sys.argv = [
            str(owner.ROOT / owner.DRIVER),
            *owner.campaign_arguments(plan, args.plan_sha256),
        ]
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        signal.alarm(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
