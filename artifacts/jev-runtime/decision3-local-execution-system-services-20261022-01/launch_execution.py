"""Default dry. Actual local execution requires an independently selected plan SHA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]


def selected_module(path, digest):
    # Authenticate our selected source bytes before importing any project actor.
    blob = Path(path).read_bytes()
    if len(blob) > 2 * 1024**2 or hashlib.sha256(blob).hexdigest() != digest:
        raise ValueError("Independently selected plan SHA differs")
    value = json.loads(blob)
    for key, expected in [
        ("execution", HERE / "execution.py"),
        ("entry", Path(__file__).absolute()),
    ]:
        ref = value["components"][key]
        source = ROOT / ref["path"]
        if source != expected or any(p.is_symlink() for p in (source, *source.parents)):
            raise ValueError("Exact ordinary integration source")
        data = source.read_bytes()
        if len(data) != ref["size_bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("Selected integration source differs")
        if key == "execution":
            selected = data
    name = "_selected_d3_local_execution"
    spec = importlib.util.spec_from_file_location(name, HERE / "execution.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(selected, str(HERE / "execution.py"), "exec", dont_inherit=True), module.__dict__)
    return module, blob


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--role", choices=["controller", "actor", "worker"], default="controller")
    parser.add_argument("--plan")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--group", choices=["known4", "public6"])
    parser.add_argument("--cpu-fixture", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "DRY_SOURCE_ONLY",
                    "actual_started": False,
                    "model_qualified": False,
                    "groups": {"known4": 64, "public6": 96},
                    "models": ["d3", "d3-flash", "d3-mini", "d3-nano", "d3-lite", "d3-edge"],
                    "owner_cap_bytes": 80 * 1024**3,
                    "swap_bytes": 0,
                    "runtime_seconds": 7200,
                    "requires": [
                        "independent selected plan SHA",
                        "actual external owner readback",
                        "fresh selected Q95 helper",
                    ],
                }
            )
        )
        return 0
    if not args.plan or not args.plan_sha256 or (args.role != "controller" and args.group is None):
        parser.error(
            "Actual execution requires selected plan SHA and an explicit group for actor/worker"
        )
    if args.cpu_fixture and args.role != "worker":
        parser.error("CPU fixture can only start an owned tiny worker")
    module, blob = selected_module(args.plan, args.plan_sha256)
    plan, bootstrap, _inputs, _identities = module.selected_plan(
        args.plan, args.plan_sha256, fixture=args.cpu_fixture
    )
    if args.role == "controller":
        return module.controller_main(plan, args.plan_sha256, blob)
    if args.role == "actor":
        return module.actor_main(plan, args.group, args.plan_sha256, Path(args.plan).absolute())
    factory = None
    if args.cpu_fixture:
        ref = next(
            row
            for row in plan["sources"]
            if row["path"] == str((HERE / "fake_components.py").relative_to(ROOT))
        )
        factory = module.load_module(ref, "_d3_execution_fixture").components
    return module.worker_main(
        plan, args.group, args.plan_sha256, bootstrap, fixture_factory=factory
    )


if __name__ == "__main__":
    raise SystemExit(main())
