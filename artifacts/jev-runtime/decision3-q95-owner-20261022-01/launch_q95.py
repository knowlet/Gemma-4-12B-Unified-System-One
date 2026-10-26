"""Default dry; exact SHA-selected --run or cooperative --request-close only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).absolute().parent


def selected(path, digest):
    blob = Path(path).read_bytes()
    if len(blob) > 1024**2 or hashlib.sha256(blob).hexdigest() != digest:
        raise ValueError("Selected owner plan SHA")
    value = json.loads(blob)
    selected_source = None
    for key, name in (("owner", "q95_owner.py"), ("entry", "launch_q95.py")):
        ref = value["components"][key]
        p = Path(ref["path"])
        if p != HERE / name or any(q.is_symlink() for q in (p, *p.parents)):
            raise ValueError("Exact ordinary owner source")
        data = p.read_bytes()
        if len(data) != ref["size_bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("Selected owner source drift")
        if key == "owner":
            selected_source = data
    spec = importlib.util.spec_from_file_location("_selected_q95_owner", HERE / "q95_owner.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(
        compile(selected_source, str(HERE / "q95_owner.py"), "exec", dont_inherit=True),
        module.__dict__,
    )
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--request-close", action="store_true")
    parser.add_argument("--actor", action="store_true")
    parser.add_argument("--plan")
    parser.add_argument("--plan-sha256")
    args = parser.parse_args(argv)
    if not args.run and not args.request_close:
        print(
            json.dumps(
                {
                    "mode": "DRY_SOURCE_ONLY",
                    "unit_started": False,
                    "GPU": False,
                    "memory_max_bytes": 40 * 1024**3,
                    "swap_bytes": 0,
                    "runtime_seconds": 10800,
                    "model_qualified": False,
                }
            )
        )
        return 0
    if not args.plan or not args.plan_sha256 or (args.actor and not args.run):
        parser.error("Actual operation requires exact selected plan SHA")
    module = selected(args.plan, args.plan_sha256)
    plan, serve, refs, blob = module.validate(args.plan, args.plan_sha256)
    owner = module.load(plan["components"]["boundary"], "_q95_boundary")
    if args.request_close:
        if owner.boot() != plan["boot_id"]:
            raise ValueError("Selected close boot")
        out = Path(plan["output"])
        admission = json.loads(module.read(out / "admission.json"))
        if (
            admission.get("plan_sha256") != args.plan_sha256
            or admission.get("nonce") != plan["nonce"]
        ):
            raise ValueError("Selected close admission")
        owner.save(
            out, "close-request.json", {"plan_sha256": args.plan_sha256, "nonce": plan["nonce"]}
        )
        return 0
    if args.actor:
        if os.environ.get("JEV_D3_BOUNDARY_NONCE") != plan["nonce"]:
            raise ValueError("Selected actor nonce")
        return module.actor_main(plan, serve, refs, args.plan_sha256, owner)
    return module.controller_main(plan, serve, refs, blob, args.plan_sha256, owner)


if __name__ == "__main__":
    raise SystemExit(main())
