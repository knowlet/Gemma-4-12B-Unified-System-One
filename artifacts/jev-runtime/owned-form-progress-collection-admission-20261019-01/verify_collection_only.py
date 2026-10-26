"""Verify the closed form collection only; no processor/compile/model/browser."""

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
PLAN_SHA = "01518a2859b4483631d19c2592498a3dac29ba4221dc6f908ea6f9c19611cc34"
SOURCE_SHA = "481af4730db830e2921f2aca9f4fcc266d0cff5bf21a2b6cd17213b8b2139baf"
ORDER_SHA = "327e3d2bf6980bceca2d60cfe81ac8b0343bc45149bc1969cb35bc897259b620"
PHASE = "before-any-project-framework-or-parent-payload"
AS_LIMIT = 256 * 1024**2


def checked(path, expected):
    assert path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
    assert path.stat().st_size < 2 * 1024**2
    blob = path.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == expected, str(path)
    return blob


def save(output, name, value):
    blob = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    assert len(blob) < 128 * 1024
    path = output / name
    with path.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())


def load(name, path, sources):
    pin = sources[str(path.relative_to(ROOT))]
    blob = checked(path, pin["sha256"])
    assert len(blob) == pin["size_bytes"]
    assert name not in sys.modules
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    if sys.flags.optimize:
        raise RuntimeError("Collection verifier requires assertions enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--source-sha256")
    args = parser.parse_args()
    if not args.run:
        print("DRY_COLLECTION_VERIFICATION_NO_PAYLOAD_OR_PROCESSOR")
        return 0
    checked(Path(__file__).resolve(), args.source_sha256)
    plan_bytes = checked(BASE / "admission-plan-02.json", PLAN_SHA)
    handoff = json.loads(checked(BASE / "source-handoff-104-02.json", SOURCE_SHA))
    order = json.loads(checked(BASE / "CPU-v3-launch-plan-01.json", ORDER_SHA))["ordered_cases"]
    sources = handoff["files"]
    assert len(sources) == 104 and len(order) == 23
    output = BASE / "actual-collection-verification-01"
    assert not output.exists()
    output.mkdir(mode=0o700)
    tail = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    result = {
        "schema": "closed_form_collection_only_verification_v1",
        "status": "UNKNOWN",
        "source_sha256": args.source_sha256,
        "plan_sha256": PLAN_SHA,
        "source104_sha256": SOURCE_SHA,
        "owner": {"pid": os.getpid(), "start_ticks": int(tail[19]), "pgid": os.getpgrp()},
        "all23_rows": [{**item, "status": "NOT_VERIFIED"} for item in order],
        "scope": "Closed evidence admission only; not processor/CFG compilation, training, model qualification or task quality.",
    }
    save(output, "intent.json", result)
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_AS, (AS_LIMIT, AS_LIMIT))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024**2, 2 * 1024**2))
        assert resource.getrlimit(resource.RLIMIT_AS) == (AS_LIMIT, AS_LIMIT)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        common = load(
            "jev_public_action_artifact", ROOT / "scripts/jev_public_action_artifact.py", sources
        )
        producer = load(
            "_collection_only_original_gate",
            ROOT / "scripts/jev_browser_fit_public_actions.py",
            sources,
        )
        assert producer.artifact is common
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        gate = producer.ResourceGate(output / "resource-gates.jsonl", boot)
        gate.check(PHASE, "cpu")
        forms = load(
            "jev_owned_form_progress", ROOT / "scripts/jev_owned_form_progress.py", sources
        )
        collection = forms.validate_form_progress_collection(
            plan_bytes,
            expected_plan_sha256=PLAN_SHA,
            reviewed_source_pins=sources,
            gate=gate,
        )
        assert type(collection) is forms.AdmittedFormProgressCollection
        collection.check()
        cases = collection.train_cases + collection.development_cases
        assert [
            {"id": c.id, "split": c.split, "family": c.annotation()["family"]} for c in cases
        ] == order
        assert len(collection.train_cases) == 13 and len(collection.development_cases) == 10
        rows = []
        for case, item in zip(cases, order, strict=True):
            assert hashlib.sha256(case.wire_bytes).hexdigest() == case.wire_sha256
            rows.append({**item, "status": "VERIFIED", "native_wire_sha256": case.wire_sha256})
        gate.check(PHASE, "cpu")
        forbidden = {"torch", "transformers", "tokenizers", "peft", "s1"}
        assert not forbidden.intersection(name.split(".")[0] for name in sys.modules)
        result.update(
            status="ADMITTED_COLLECTION_NOT_COMPILED_NOT_FIT_NOT_QUALIFIED",
            all23_rows=rows,
            provenance=collection.provenance,
            original_CPU_gate_records=gate.sequence,
            AS_limit_bytes=AS_LIMIT,
            lifetime_maxRSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            forbidden_framework_modules_loaded=False,
            error=None,
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    save(output, "terminal.json", result)
    print(json.dumps({"status": result["status"], "error": result.get("error")}))
    return 0 if result["status"].startswith("ADMITTED_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
