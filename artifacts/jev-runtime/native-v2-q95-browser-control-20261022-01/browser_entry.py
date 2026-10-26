"""Own one native Gemma/Q95 actor before importing any ML components."""

import argparse
import os
import runpy
import signal
import sys
from pathlib import Path

import launch_browser as owner


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not args.run:
        print("DRY_NO_MODEL_OR_BROWSER")
        return 0
    owner.require(
        os.environ.get("JEV_NATIVE_BROWSER_EXECUTE") == owner.PROFILE,
        "Owned browser execution environment required",
    )
    owner.require(args.plan and args.plan_sha256, "Explicit sealed plan required")
    plan = owner.load_plan(args.plan, args.plan_sha256)
    owner.require(os.environ.get("JEV_NATIVE_BROWSER_UNIT") == plan["unit"], "Owned unit binding")
    directory, output = owner.run_paths(plan)
    owner.require(
        directory.is_dir()
        and not output.exists()
        and not (directory / "actor-intent.json").exists(),
        "Fresh owned actor output",
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
        raise TimeoutError("Owned current-NF4 browser cooperative7140-second deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(7140)
    sys.argv = [
        str(owner.ROOT / owner.DRIVER),
        *owner.browser_arguments(plan),
        "--q95-control-plan",
        str(args.plan),
        "--q95-control-plan-sha256",
        args.plan_sha256,
    ]
    runpy.run_path(sys.argv[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
