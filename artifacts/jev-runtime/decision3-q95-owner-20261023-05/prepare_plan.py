"""Default dry; seal fresh metadata only after original Q95 host admission.

This command never starts the selected unit or loads a model. The separately
selected launch_q95 entry must still validate the complete plan and repeat all
original admission, source, ownership, identity and cleanup checks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).absolute().parent
OWNER_SHA = "62ac775cedd7d16864564955a40f86aee5a4e6240a8f8b715271749ecd80582b"
ENTRY_SHA = "db259d080227dca3c666d999fd9cf62bd40849c4974735285ecc38fb8b4e998d"


def need(value, message):
    if not value:
        raise ValueError(message)


def checked(path, expected):
    need(not any(p.is_symlink() for p in (path, *path.parents)), "Ordinary selected source")
    with path.open("rb") as stream:
        blob = stream.read(1024**2 + 1)
    need(len(blob) <= 1024**2 and hashlib.sha256(blob).hexdigest() == expected, "Source SHA")
    return blob


def load_owner():
    blob = checked(HERE / "q95_owner.py", OWNER_SHA)
    checked(HERE / "launch_q95.py", ENTRY_SHA)
    spec = importlib.util.spec_from_file_location(
        "_fresh_q95_owner05_metadata", HERE / "q95_owner.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(blob, str(HERE / "q95_owner.py"), "exec", dont_inherit=True), vars(module))
    return module


def seal(owner, target):
    target = Path(target).absolute()
    need(
        target.parent == HERE
        and re.fullmatch(r"launch-plan-[0-9]{2}\.json", target.name)
        and not target.exists()
        and not any(p.is_symlink() for p in (target, *target.parents)),
        "Fresh ordinary selected plan path",
    )
    need(
        not (HERE / "owner-05").exists() and not owner.SERVICE.exists(),
        "Fresh owner and service slots; historical attempts cannot be reused",
    )
    plan = owner.prepare("owner-05")
    for name, filename, sha in (
        ("owner", "q95_owner.py", OWNER_SHA),
        ("entry", "launch_q95.py", ENTRY_SHA),
    ):
        ref = plan["components"][name]
        need(ref["path"] == str(HERE / filename) and ref["sha256"] == sha, "Fixed owner authority")
        checked(HERE / filename, sha)
    # Verify fixed source authorities before executing even the read-only observer.
    for name, (label, sha) in owner.FIXED.items():
        ref = plan["components"][name]
        need(
            ref["path"] == str(owner.ROOT / "artifacts/jev-runtime" / label)
            and ref["sha256"] == sha,
            "Fixed original component authority",
        )
        owner.check_ref(ref)
    observer = owner.load(plan["components"]["observer"], "_fresh_q95_owner05_preflight")
    host = owner.host(plan, observer, initial=True)
    # No plan, live seed, owner output or unit is created when host admission refuses.
    checked(HERE / "q95_owner.py", OWNER_SHA)
    checked(HERE / "launch_q95.py", ENTRY_SHA)
    blob = owner.encode(plan)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    digest = hashlib.sha256(blob).hexdigest()
    owner.validate(target, digest)
    return {
        "status": "fresh_metadata_sealed_not_launched",
        "plan_path": str(target),
        "plan_sha256": digest,
        "unit": plan["unit"],
        "host": {
            key: host[key]
            for key in (
                "boot_id",
                "mem_total_bytes",
                "mem_available_bytes",
                "comfy_state",
                "comfy_running",
                "comfy_pending",
            )
        },
        "unit_started": False,
        "model_loaded": False,
        "quality_score": None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--plan-name", default="launch-plan-01.json")
    args = parser.parse_args(argv)
    if not args.seal:
        print(
            json.dumps({"status": "DRY_SOURCE_ONLY", "unit_started": False, "model_loaded": False})
        )
        return 0
    owner = load_owner()
    try:
        result = seal(owner, HERE / args.plan_name)
    except Exception as error:
        result = {
            "status": "preparation_refused_no_unit_started",
            "error": owner.error_record(error),
            "unit_started": False,
            "model_loaded": False,
            "quality_score": None,
        }
        print(json.dumps(result, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
