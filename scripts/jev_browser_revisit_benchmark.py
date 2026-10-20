"""Thin independent runner entry for the experimental revisit runtime profile.

Preparation is the default and imports no browser/model. Explicit --run requires
reviewed launcher/profile source pins. Fresh capacity/helper admission remains a
separate prerequisite; old V8/Q95 qualifications do not cover this new prompt.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
PROFILE = "composed-visible-hover-entries-settled-revisit-v1"
PROFILE_SHA = "873abc6aa2a02272fd9b54d299d683598bdb3d5c0a848cc43a356daa5a36a0ca"
PARENT_LAUNCHER_SHA = "af90373aa31a2b485bf28dba98477e49a00c5f2e2681a8478d82bbf80445594b"
BASES = {
    "original": (
        "runner-known4-01",
        "85dc8cd96bc5ae4f6fa16d99ec12795b675f9c59db999839377c1363a0842fb9",
    ),
    "heldout": (
        "runner-public6-01",
        "d3cd283bf692f93342f0f0b09d5adb085ee905cbfa615f8c22e77e4966bb4765",
    ),
}


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _load(path, expected):
    if (
        any(p.is_symlink() for p in (path, *path.parents))
        or not path.is_file()
        or not 0 < path.stat().st_size <= 1024**2
    ):
        raise ValueError("Exact bounded launcher dependency required")
    blob = path.read_bytes()
    if (
        len(blob) > 1024**2
        or _sha(blob) != expected
        or any(p.is_symlink() for p in (path, *path.parents))
    ):
        raise ValueError("Exact bounded launcher dependency required")
    name = "_jev_revisit_entry_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if path.read_bytes() != blob:
        raise ValueError("Launcher dependency changed during import")
    return module


PARENT = _load(ROOT / "scripts/jev_browser_profile_v8_benchmark.py", PARENT_LAUNCHER_SHA)
PROFILE_MODULE = _load(ROOT / "scripts/jev_observation_profile_revisit.py", PROFILE_SHA)


def _derive(blob, pairs):
    derivative = blob
    for old, new, count in pairs:
        if derivative.count(old) != count:
            raise ValueError("Exact new runner replacement count")
        derivative = derivative.replace(old, new)
    inverse = derivative
    for old, new, count in reversed(pairs):
        if inverse.count(new) != count:
            raise ValueError("Exact new runner inverse count")
        inverse = inverse.replace(new, old)
    if inverse != blob or ast.dump(ast.parse(inverse), include_attributes=False) != ast.dump(
        ast.parse(blob), include_attributes=False
    ):
        raise ValueError("New runner contains unrelated changes")
    return derivative


def prepare_bundle(bundle_dir, *, case_module="original"):
    if case_module not in BASES:
        raise ValueError("Original or heldout surface required")
    name, expected = BASES[case_module]
    base = ROOT / "artifacts/jev-runtime/public-native-hover-settled-v8-20261019-01" / name
    manifest_blob = (base / "bundle-manifest.json").read_bytes()
    if _sha(manifest_blob) != expected:
        raise ValueError("Sealed parent runner manifest differs")
    parent_manifest = json.loads(manifest_blob)
    PARENT.verify_bundle(base, parent_manifest)
    files = {
        name: (base / name).read_bytes()
        for name in parent_manifest["files"]
        if name not in {"launcher.py", "runner-profile.patch"}
    }
    old_runner = files["runner.py"]
    files["runner.py"] = _derive(
        old_runner,
        (
            (
                b"import jev_observation_profile_v8 as profile",
                b"import jev_observation_profile_revisit as profile",
                3,
            ),
            (
                b'PRUNED_OBSERVATION_PROFILE = "composed-visible-hover-entries-settled-v8"',
                ('PRUNED_OBSERVATION_PROFILE = "' + PROFILE + '"').encode(),
                1,
            ),
            (
                b"from jev_observation_events_v8 import ObservationProfileTracker",
                b"from jev_observation_events_revisit import ObservationProfileTracker",
                1,
            ),
        ),
    )
    files["jev_observation_events_revisit.py"] = _derive(
        files["jev_observation_events_v8.py"],
        (
            (
                b'PRUNED_PROFILE = "composed-visible-hover-entries-settled-v8"',
                ('PRUNED_PROFILE = "' + PROFILE + '"').encode(),
                1,
            ),
        ),
    )
    files.update(PROFILE_MODULE._sources(PROFILE_SHA))
    files["parent-bundle-manifest.json"] = manifest_blob
    files["launcher.py"] = Path(__file__).read_bytes()
    files["runner-profile.patch"] = "".join(
        difflib.unified_diff(
            old_runner.decode().splitlines(keepends=True),
            files["runner.py"].decode().splitlines(keepends=True),
            fromfile="sealed-v8-runner.py",
            tofile="runner.py",
        )
    ).encode()
    manifest = {
        "schema_id": "jev-revisit-isolated-runner-bundle-v1",
        "status": "SOURCE_PREPARED_NOT_CAPACITY_OR_BROWSER_QUALIFIED",
        "profile": PROFILE,
        "case_module": case_module,
        "profile_helper_sha256": PROFILE_SHA,
        "launcher_source_sha256": _sha(files["launcher.py"]),
        "parent_manifest_sha256": expected,
        "parent_launcher_source_sha256": PARENT_LAUNCHER_SHA,
        "source_delta": {
            "profile_import_occurrences": 3,
            "runner_profile_literal": 1,
            "tracker_import": 1,
            "tracker_identity_literal": 1,
            "inverse_bytes_equal": True,
            "inverse_AST_equal": True,
            "candidate_action_head_decoder_math_changes": 0,
        },
        "files": {
            name: {"sha256": _sha(blob), "size_bytes": len(blob)}
            for name, blob in sorted(files.items())
        },
        "artifact_files_including_manifest": len(files) + 1,
        "fresh_capacity_helper_and_browser_evaluation_required": True,
        "old_V8_or_Q95_qualification_covers_new_prompt": False,
        "model_or_browser_executed": False,
    }
    output = PROFILE_MODULE.V8._ordinary_path(bundle_dir)
    if output.exists() or output.is_relative_to(base) or base.is_relative_to(output):
        raise ValueError("Fresh disjoint runner output required")
    output.mkdir(parents=True, exist_ok=False)
    for name, blob in sorted(files.items()):
        PROFILE_MODULE.V8._write_exclusive(output / name, blob)
    PROFILE_MODULE.V8._write_exclusive(
        output / "bundle-manifest.json", PROFILE_MODULE.V8._encode(manifest) + b"\n"
    )
    PROFILE_MODULE._fsync_directories(output)
    verify_bundle(output, manifest)
    PARENT.verify_bundle(base, parent_manifest)
    return output, manifest


def verify_bundle(bundle, manifest):
    bundle = PROFILE_MODULE.V8._ordinary_path(bundle)
    labels = {*manifest["files"], "bundle-manifest.json"}
    if (bundle / "launch-receipt.json").exists():
        labels.add("launch-receipt.json")
    blobs, first_files, first_dirs = PROFILE_MODULE.TREE.capture_tree(
        bundle,
        labels=labels,
        read_contents=True,
        helpers=PROFILE_MODULE.V8,
        MAX_SOURCE_BYTES=PROFILE_MODULE.MAX_SOURCE_BYTES,
        MAX_FILES=PROFILE_MODULE.MAX_FILES,
        MAX_TOTAL_BYTES=PROFILE_MODULE.MAX_TOTAL_BYTES,
    )
    if blobs["bundle-manifest.json"] != PROFILE_MODULE.V8._encode(manifest) + b"\n":
        raise ValueError("New runner manifest differs")
    for name, pin in manifest["files"].items():
        if len(blobs[name]) != pin["size_bytes"] or _sha(blobs[name]) != pin["sha256"]:
            raise ValueError("New runner source differs: " + name)
    _, last_files, last_dirs = PROFILE_MODULE.TREE.capture_tree(
        bundle,
        labels=labels,
        read_contents=False,
        helpers=PROFILE_MODULE.V8,
        MAX_SOURCE_BYTES=PROFILE_MODULE.MAX_SOURCE_BYTES,
        MAX_FILES=PROFILE_MODULE.MAX_FILES,
        MAX_TOTAL_BYTES=PROFILE_MODULE.MAX_TOTAL_BYTES,
    )
    if first_files != last_files or first_dirs != last_dirs:
        raise ValueError("New runner population changed during verification")


def _arguments(arguments, case_module):
    # Same fixed heldout interface as V8, with an independently named profile.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--observation-profile", choices=[PROFILE], default=PROFILE)
    parser.add_argument("--observation-profile-helper-sha256")
    values, _ = parser.parse_known_args(arguments)
    if values.observation_profile_helper_sha256 not in {None, PROFILE_SHA}:
        parser.error("Explicit new profile helper SHA differs")
    result = list(arguments)
    if not any(v.split("=", 1)[0] == "--observation-profile" for v in result):
        result += ["--observation-profile", PROFILE]
    if values.observation_profile_helper_sha256 is None:
        result += ["--observation-profile-helper-sha256", PROFILE_SHA]
    if case_module == "heldout":
        cases = _load(
            ROOT / "scripts/jev_heldout_public_cases.py", PARENT.HELDOUT_SHA256
        ).PUBLIC_WEB_CASES
        check = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        for flag in ("upstream-path", "endpoint", "model", "output-dir"):
            check.add_argument("--" + flag, required=True)
        for flag in (
            "api-key-env",
            "text-endpoint",
            "text-model",
            "text-api-key-env",
            "chromium-executable",
            "observation-profile-helper-sha256",
        ):
            check.add_argument("--" + flag)
        check.add_argument("--repeats", type=int, choices=[1], default=1)
        check.add_argument("--max-steps", type=int, choices=[16], default=16)
        check.add_argument("--seed", type=int, choices=[20261014], default=20261014)
        check.add_argument("--suite-split", choices=["development"], default="development")
        check.add_argument("--observation-profile", choices=[PROFILE])
        check.add_argument("--public-cases", nargs="+", choices=tuple(cases))
        selected = check.parse_args(result)
        if selected.public_cases is None:
            result += ["--public-cases", *cases]
        elif len(selected.public_cases) != len(set(selected.public_cases)):
            check.error("Public cases must be distinct")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--case-module", choices=list(BASES), default="original")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--profile-source-sha256")
    parser.add_argument("--launcher-source-sha256")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    bundle, manifest = prepare_bundle(args.bundle_dir, case_module=args.case_module)
    if not args.run:
        print(
            json.dumps(
                {
                    "bundle": str(bundle),
                    "manifest_sha256": _sha((bundle / "bundle-manifest.json").read_bytes()),
                    "model_or_browser_executed": False,
                }
            )
        )
        return 0
    if (
        args.profile_source_sha256 != PROFILE_SHA
        or args.launcher_source_sha256 != manifest["launcher_source_sha256"]
    ):
        parser.error(
            "Explicit reviewed new launcher and profile source SHA required before execution"
        )
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    forwarded = _arguments(arguments, args.case_module)
    receipt = {
        "status": "child_starting",
        "profile": PROFILE,
        "arguments_sha256": _sha(json.dumps(forwarded, ensure_ascii=False).encode()),
        "child_exit_is_task_success": False,
        "source_pins_are_not_capacity_admission": True,
    }
    PARENT._write_json(bundle / "launch-receipt.json", receipt)
    code, primary = None, None
    try:
        verify_bundle(bundle, manifest)
        code = subprocess.run(
            [sys.executable, "-B", "-s", str(bundle / "runner.py"), *forwarded], check=False
        ).returncode
    except BaseException as exc:
        primary = exc
    finally:
        try:
            verify_bundle(bundle, manifest)
            receipt.update(
                status="finished" if code == 0 and primary is None else "failed",
                returncode=code,
                source_integrity_verified=True,
            )
            PARENT._write_json(bundle / "launch-receipt.json", receipt)
        except BaseException as secondary:
            if primary is None:
                primary = secondary
            else:
                primary.add_note("Runner closure evidence also failed: " + repr(secondary))
    if primary is not None:
        raise primary
    return code


if __name__ == "__main__":
    raise SystemExit(main())
