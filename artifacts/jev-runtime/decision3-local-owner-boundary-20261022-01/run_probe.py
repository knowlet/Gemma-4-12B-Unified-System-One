"""Own only the explicitly authorized 64 MiB / 30 s D3 cgroup fixture."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
UNIT = "jev-d3-boundary-probe-20261022-01.service"


def require(value, message):
    if not value:
        raise ValueError(message)


def ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary selected path",
    )
    return path


def load_plan(path, digest):
    require(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest), "Selected digest")
    blob = ordinary(path).read_bytes()
    require(
        len(blob) <= 65536 and hashlib.sha256(blob).hexdigest() == digest,
        "Exact bounded selected plan",
    )
    plan = json.loads(blob)
    require(
        type(plan) is dict
        and plan["unit"] == UNIT
        and plan["profile"] == "decision3-external-unit-boundary-v1",
        "Only authorized tiny boundary profile",
    )
    require(
        type(plan["memory_max_bytes"]) is int
        and plan["memory_max_bytes"] == 64 * 1024**2
        and type(plan["runtime_seconds"]) is int
        and plan["runtime_seconds"] == 30
        and type(plan["stop_seconds"]) is int
        and plan["stop_seconds"] == 2,
        "Only authorized 64MiB / 30s / 2s fixture",
    )
    output = ordinary(plan["output"])
    require(
        output.parent == HERE and re.fullmatch(r"probe-[0-9]{2}", output.name),
        "Fresh output scoped to owned fixture directory",
    )
    require(not output.exists(), "Never overwrite interrupted output")
    require(
        ordinary(plan["interpreter"]) == Path(sys.executable).resolve(),
        "Exact current resolved interpreter",
    )
    refs = plan["sources"]
    require(
        type(refs) is list
        and {r["path"] for r in refs}
        == {
            (HERE / name).relative_to(ROOT).as_posix()
            for name in ("owner_boundary.py", "probe_service.py", "run_probe.py")
        },
        "Complete exact three-file execution population",
    )
    for ref in refs:
        require(
            type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"},
            "Exact source ref schema",
        )
        source = ordinary(ROOT / ref["path"])
        data = source.read_bytes()
        require(
            hashlib.sha256(data).hexdigest() == ref["sha256"] and len(data) == ref["size_bytes"],
            "Exact source before local owner import",
        )
    return plan, blob


def load_owner():
    spec = importlib.util.spec_from_file_location("_owned_d3_boundary", HERE / "owner_boundary.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def error_record(error):
    return {"type": type(error).__name__, "message": str(error)[:8192]}


def execute(plan, digest, blob, owner):
    selection = owner.Selection(
        **{
            k: plan[k]
            for k in (
                "unit",
                "nonce",
                "boot_id",
                "memory_max_bytes",
                "runtime_seconds",
                "stop_seconds",
            )
        }
    )
    require(selection.cgroup == plan["cgroup"], "Exact selected actual cgroup path")
    require(owner.boot() == selection.boot_id, "Same selected boot before launch")
    require(owner.sources_match(plan["sources"]), "All selected sources before launch")
    require(owner.status(selection).get("LoadState") == "not-found", "Fresh exact unit not found")
    require(not (owner.CGROUP_ROOT / selection.cgroup.lstrip("/")).exists(), "Fresh cgroup absent")
    output = ordinary(plan["output"])
    require(not output.exists(), "Fresh output before mutation")
    output.mkdir(mode=0o700)
    owner.save(output, "selected-plan.json", plan)
    # Preserve original bytes: save() above intentionally canonicalizes, so the
    # actor receives a separate exclusively-created exact-byte sealed input.
    sealed = output / "sealed-plan.json"
    with sealed.open("xb") as stream:
        os.chmod(stream.name, 0o600)
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    boundary = owner.UnitBoundary(selection, output, plan["sources"])
    properties = selection.properties()
    launch_properties = {
        **properties,
        "RuntimeMaxSec": str(selection.runtime_seconds),
        "TimeoutStopSec": str(selection.stop_seconds),
        "WorkingDirectory": str(ROOT),
        "LimitCORE": "0",
        "StandardOutput": "append:" + str(output / "service.log"),
        "StandardError": "append:" + str(output / "service.log"),
    }
    # The USec names are readback keys; the Sec aliases are launch assignments.
    del launch_properties["RuntimeMaxUSec"], launch_properties["TimeoutStopUSec"]
    argv = [
        "systemd-run",
        "--user",
        "--unit=" + selection.unit,
        *["--property=" + k + "=" + v for k, v in launch_properties.items()],
        "--setenv=JEV_D3_BOUNDARY_NONCE=" + selection.nonce,
        plan["interpreter"],
        "-I",
        "-B",
        str(HERE / "probe_service.py"),
        "--run",
        "--plan",
        str(sealed),
        "--plan-sha256",
        digest,
    ]
    owner.save(
        output,
        "launch-intent.json",
        {
            "argv": argv,
            "plan_sha256": digest,
            "independent_timeout_seconds": 30,
            "authenticated_stop_timeout_seconds": 7,
        },
    )
    primary, closure, proof = None, None, None
    try:
        owner.command(*argv, timeout=8)
        deadline = time.monotonic() + 6
        names = ("actor-intent.json", "grandchild-intent.json", "intermediate-reaped.json")
        while not (output / names[0]).is_file():
            require(time.monotonic() < deadline, "Timed tiny main fixture readiness")
            time.sleep(0.02)
        readback = boundary.verify()
        owner.save(
            output,
            "admission.json",
            {"plan_sha256": digest, "nonce": selection.nonce, "readback_verified": True},
        )
        deadline = time.monotonic() + 6
        while not all((output / name).is_file() for name in names):
            require(time.monotonic() < deadline, "Timed tiny fixture readiness")
            time.sleep(0.02)
        detached_cohort = boundary.members()
        intents = {name: json.loads((output / name).read_bytes()) for name in names}
        for name in names[:2]:
            row = intents[name]
            require(
                all(
                    row.get(k) == v
                    for k, v in {
                        "plan_sha256": digest,
                        "unit": selection.unit,
                        "nonce": selection.nonce,
                        "boot_id": selection.boot_id,
                    }.items()
                ),
                "Exact actor plan/nonce/boot binding",
            )
        main = intents[names[0]]["actor"]
        child = intents[names[1]]["actor"]
        require(
            all(
                main[k] == readback["actor"][k]
                for k in ("pid", "start_ticks", "pgid", "sid", "cgroup")
            ),
            "Published main matches fresh external actor",
        )
        fresh_child = owner.process(child["pid"])
        require(
            all(
                child[k] == fresh_child[k] for k in ("pid", "start_ticks", "pgid", "sid", "cgroup")
            ),
            "Published detached child matches fresh external identity",
        )
        require(
            child["pid"] == child["pgid"] == child["sid"]
            and child["sid"] != main["sid"]
            and child["initial_exec_environment_bytes"] == 0
            and child["ppid"] != main["pid"]
            and child["cgroup"] == selection.cgroup
            and any(
                r["pid"] == child["pid"] and r["start_ticks"] == child["start_ticks"]
                for r in detached_cohort["members"]
            ),
            "Actual double-fork / setsid / empty-env child stayed in kernel cgroup",
        )
        proof = {
            "main": main,
            "detached_child": child,
            "fresh_child": fresh_child,
            "intermediate": intents[names[2]],
            "detached_cohort": detached_cohort,
            "membership_verified": True,
            "no_escape_enforced": False,
        }
        owner.save(output, "inheritance-proof.json", proof)
    except BaseException as error:
        primary = error
    finally:
        try:
            closure = boundary.close(
                primary_error=None if primary is None else error_record(primary)
            )
        except BaseException as secondary:
            if primary is None:
                primary = secondary
            else:
                primary.add_note("Boundary finalization also failed: " + repr(secondary))
            owner.save(
                output,
                "finalization-error.json",
                {
                    "primary_error": error_record(primary),
                    "finalization_error": error_record(secondary),
                    "closure": "UNKNOWN",
                },
            )
    result = {
        "plan_sha256": digest,
        "primary_error": None if primary is None else error_record(primary),
        "inheritance_verified": proof is not None,
        "whole_unit_exit_confirmed": closure is not None and closure["whole_unit_exit_confirmed"],
        "whole_tree_exit_confirmed": False,
        "continuation_authorized": False,
        "model_qualified": False,
    }
    owner.save(output, "probe-result.json", result)
    if primary is not None:
        raise primary
    require(
        result["inheritance_verified"] and result["whole_unit_exit_confirmed"],
        "Complete tiny fixture inheritance and authenticated unit closure required",
    )
    print(json.dumps(result), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args(argv)
    if not args.run:
        print("DRY_NO_UNIT_OR_ACTOR; explicit --run and sealed plan required")
        return 0
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    require(args.plan is not None, "Explicit sealed plan")
    plan, blob = load_plan(args.plan, args.plan_sha256)
    return execute(plan, args.plan_sha256, blob, load_owner())


if __name__ == "__main__":
    raise SystemExit(main())
