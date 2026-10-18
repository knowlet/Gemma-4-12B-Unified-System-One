"""Prepare an isolated v6 Jev runner without changing the pinned benchmark.

The runner edits explicit profile/tracker imports, one profile-choice constant,
and optionally the existing heldout-case import. Inverse bytes and complete AST
are verified. Preparation alone never starts a browser or model.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROFILE = "composed-visible-controls-settled-v6"
BENCHMARK_SHA256 = "fe1410675e4decb06a8491a5a83d2ec965629c99f8912b797446c4cb2c155f6c"
HELPERS_SHA256 = "0ec9826d334cb2fbed24ade8780fa785291017d904b18233419ced760750bca3"
HELDOUT_SHA256 = "339cddf596a25ab433ca66e23924631b6261e62efc06f351878d4ef5c3f6f8f9"
GRANDPARENT_SHA256 = "8d199ac1564af05892e9b21e225d1fdf500f1953840855c902160ed8d56df89d"
V4_PARENT_SHA256 = "f96fb0553c4c8ef1a163633577bfb3ddbb03ddbc1ff6644c36d8fbe2c06d4b43"
EVENTS_SHA256 = "ee6d3ea04d948fd8947d12934b63564494aab6a67cb8d8063ea7e12fb84a0664"
OLD_IMPORT = b"from s1 import jev_observation_profile as profile"
NEW_IMPORT = b"import jev_observation_profile_v6 as profile"
OLD_CHOICE = b'PRUNED_OBSERVATION_PROFILE = "composed-visible-text-v3"'
NEW_CHOICE = b'PRUNED_OBSERVATION_PROFILE = "composed-visible-controls-settled-v6"'
OLD_CASE = b"        from jev_public_web_cases import (\n"
NEW_CASE = b"        from jev_heldout_public_cases import (\n"
OLD_EVENTS_IMPORT = b"from s1.jev_observation_events import ObservationProfileTracker"
NEW_EVENTS_IMPORT = b"from jev_observation_events_v6 import ObservationProfileTracker"
OLD_EVENTS_CHOICE = b'PRUNED_PROFILE = "composed-visible-text-v3"'
NEW_EVENTS_CHOICE = b'PRUNED_PROFILE = "composed-visible-controls-settled-v6"'


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _write_json(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def prepare_bundle(bundle_dir, *, case_module="original", source_dir=None):
    if case_module not in {"original", "heldout"}:
        raise ValueError("Explicit original or heldout case module required")
    source = Path(source_dir) if source_dir is not None else Path(__file__).resolve().parent
    original = (source / "jev_browser_benchmark.py").read_bytes()
    helpers = (source / "jev_public_web_cases.py").read_bytes()
    parent = (source.parent / "src/s1/jev_observation_profile.py").read_bytes()
    events = (source.parent / "src/s1/jev_observation_events.py").read_bytes()
    v4_parent = (source / "jev_observation_profile_v4.py").read_bytes()
    if _sha(v4_parent) != V4_PARENT_SHA256:
        raise ValueError("Pinned v4 parent changed")
    if (_sha(original), _sha(helpers), _sha(parent)) != (
        BENCHMARK_SHA256,
        HELPERS_SHA256,
        GRANDPARENT_SHA256,
    ):
        raise ValueError("Original benchmark, common helpers or v3 parent changed")
    if _sha(events) != EVENTS_SHA256 or events.count(OLD_EVENTS_CHOICE) != 1:
        raise ValueError("Pinned observation tracker changed")
    if original.count(OLD_IMPORT) != 3 or original.count(OLD_CHOICE) != 1:
        raise ValueError("Pinned profile import/choice anchors changed")
    derivative = original.replace(OLD_IMPORT, NEW_IMPORT).replace(OLD_CHOICE, NEW_CHOICE)
    if derivative.count(OLD_EVENTS_IMPORT) != 1:
        raise ValueError("Pinned tracker import anchor changed")
    derivative = derivative.replace(OLD_EVENTS_IMPORT, NEW_EVENTS_IMPORT)
    if case_module == "heldout":
        if derivative.count(OLD_CASE) != 1:
            raise ValueError("Pinned case import anchor changed")
        derivative = derivative.replace(OLD_CASE, NEW_CASE, 1)
    inverse = derivative.replace(NEW_IMPORT, OLD_IMPORT).replace(NEW_CHOICE, OLD_CHOICE)
    inverse = inverse.replace(NEW_EVENTS_IMPORT, OLD_EVENTS_IMPORT)
    if case_module == "heldout":
        inverse = inverse.replace(NEW_CASE, OLD_CASE, 1)
    if inverse != original or ast.dump(ast.parse(inverse), include_attributes=False) != ast.dump(
        ast.parse(original), include_attributes=False
    ):
        raise ValueError("Runner contains an unrelated source/AST edit")
    files = {
        "original_jev_browser_benchmark.py": original,
        "runner.py": derivative,
        "jev_public_web_cases.py": helpers,
        "jev_observation_profile_v6.py": (source / "jev_observation_profile_v6.py").read_bytes(),
        "jev_observation_profile_v4.py": v4_parent,
        "jev_observation_profile_v3_parent.py": parent,
        "jev_observation_events_v3_parent.py": events,
        "jev_observation_events_v6.py": events.replace(OLD_EVENTS_CHOICE, NEW_EVENTS_CHOICE),
        "launcher.py": Path(__file__).read_bytes(),
    }
    if case_module == "heldout":
        heldout = (source / "jev_heldout_public_cases.py").read_bytes()
        if _sha(heldout) != HELDOUT_SHA256:
            raise ValueError("Heldout cases changed")
        files["jev_heldout_public_cases.py"] = heldout
    files["runner-profile.patch"] = "".join(
        difflib.unified_diff(
            original.decode().splitlines(keepends=True),
            derivative.decode().splitlines(keepends=True),
            fromfile="original_jev_browser_benchmark.py",
            tofile="runner.py",
        )
    ).encode()
    bundle = Path(bundle_dir).absolute()
    if any(p.is_symlink() for p in (bundle, *bundle.parents)):
        raise ValueError("Ordinary new bundle path required")
    bundle.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (bundle / name).write_bytes(data)
    manifest = {
        "schema_id": "jev-v6-isolated-runner-bundle-v1",
        "status": "source_prepared_not_site_or_model_qualified",
        "profile": PROFILE,
        "case_module": case_module,
        "profile_helper_sha256": _sha(files["jev_observation_profile_v6.py"]),
        "v4_parent_helper_sha256": V4_PARENT_SHA256,
        "source_delta": {
            "profile_import_substitutions": 3,
            "profile_choice_literal_substitutions": 1,
            "tracker_import_substitutions": 1,
            "tracker_profile_literal_substitutions": 1,
            "tracker_inverse_bytes_equal": files["jev_observation_events_v6.py"].replace(
                NEW_EVENTS_CHOICE, OLD_EVENTS_CHOICE
            )
            == events,
            "heldout_case_import_substitutions": int(case_module == "heldout"),
            "inverse_bytes_equal": True,
            "inverse_AST_equal": True,
            "original_sha256": BENCHMARK_SHA256,
            "derivative_sha256": _sha(derivative),
        },
        "files": {
            name: {"sha256": _sha(blob), "size_bytes": len(blob)} for name, blob in files.items()
        },
        "case005_interface_exposed": True,
        "model_executed": False,
        "runtime_qualification_attested": False,
    }
    _write_json(bundle / "bundle-manifest.json", manifest)
    verify_bundle(bundle, manifest)
    return bundle, manifest


def verify_bundle(bundle, manifest):
    bundle = Path(bundle)
    allowed = {*manifest["files"], "bundle-manifest.json", "launch-receipt.json"}
    if any(p.is_symlink() or not p.is_file() or p.name not in allowed for p in bundle.iterdir()):
        raise ValueError("Unexpected or nonordinary bundle population")
    for name, record in manifest["files"].items():
        blob = (bundle / name).read_bytes()
        if len(blob) != record["size_bytes"] or _sha(blob) != record["sha256"]:
            raise ValueError("Prepared runner source changed: " + name)
    if json.loads((bundle / "bundle-manifest.json").read_bytes()) != manifest:
        raise ValueError("Prepared runner manifest changed")


def _arguments(arguments, case_module, expected_helper):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--observation-profile", choices=[PROFILE], default=PROFILE)
    parser.add_argument("--observation-profile-helper-sha256")
    values, _ = parser.parse_known_args(arguments)
    if values.observation_profile_helper_sha256 not in {None, expected_helper}:
        parser.error("Explicit v6 helper SHA differs from reviewed source")
    result = list(arguments)
    if not any(v.split("=", 1)[0] == "--observation-profile" for v in arguments):
        result += ["--observation-profile", PROFILE]
    if values.observation_profile_helper_sha256 is None:
        result += ["--observation-profile-helper-sha256", expected_helper]
    if case_module == "heldout":
        # Apply the existing isolated heldout surface with only the new profile value.
        from jev_heldout_public_cases import PUBLIC_WEB_CASES

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
        check.add_argument("--public-cases", nargs="+", choices=tuple(PUBLIC_WEB_CASES))
        selected = check.parse_args(result)
        if selected.public_cases is None:
            result += ["--public-cases", *PUBLIC_WEB_CASES]
        elif len(selected.public_cases) != len(set(selected.public_cases)):
            check.error("Public cases must be distinct")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--case-module", choices=["original", "heldout"], default="original")
    parser.add_argument("--profile-source-sha256")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    bundle, manifest = prepare_bundle(args.bundle_dir, case_module=args.case_module)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "bundle": str(bundle),
                    "manifest_sha256": _sha((bundle / "bundle-manifest.json").read_bytes()),
                    "model_executed": False,
                }
            )
        )
        return 0
    if args.profile_source_sha256 != manifest["profile_helper_sha256"]:
        parser.error("A separately reviewed --profile-source-sha256 is required before execution")
    forwarded = _arguments(arguments, args.case_module, args.profile_source_sha256)
    receipt = {
        "status": "child_starting",
        "profile": PROFILE,
        "profile_helper_sha256": args.profile_source_sha256,
        "case_module": args.case_module,
        "case005_interface_exposed": True,
        "child_exit_is_task_success": False,
        "arguments_sha256": _sha(json.dumps(forwarded, ensure_ascii=False).encode()),
    }
    _write_json(bundle / "launch-receipt.json", receipt)
    code = None
    try:
        verify_bundle(bundle, manifest)
        code = subprocess.run(
            [sys.executable, "-B", "-s", str(bundle / "runner.py"), *forwarded], check=False
        ).returncode
    finally:
        verify_bundle(bundle, manifest)
        receipt.update(
            status="finished" if code == 0 else "failed",
            returncode=code,
            source_integrity_verified=True,
        )
        _write_json(bundle / "launch-receipt.json", receipt)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
