#!/usr/bin/env python3
"""Prepare an isolated, pinned public-case evaluator, then delegate its original CLI.

Use --prepare-only to create source evidence without starting Chromium or calling
a model. Otherwise put the original benchmark arguments after ``--``. The bundle
must be new. This launcher does not change serving/training sources or policy.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from jev_heldout_public_cases import PUBLIC_WEB_CASES, SELECTION_SHA256, case_manifest

BENCHMARK_SHA256 = "fe1410675e4decb06a8491a5a83d2ec965629c99f8912b797446c4cb2c155f6c"
HELPERS_SHA256 = "0ec9826d334cb2fbed24ade8780fa785291017d904b18233419ced760750bca3"
PROFILE_SHA256 = "8d199ac1564af05892e9b21e225d1fdf500f1953840855c902160ed8d56df89d"
ORIGINAL_IMPORT = b"        from jev_public_web_cases import (\n"
ISOLATED_IMPORT = b"        from jev_heldout_public_cases import (\n"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def prepare_bundle(bundle_dir, *, source_dir=None):
    """Verify source pins before creating a fresh directory; no external execution."""
    source = Path(source_dir) if source_dir is not None else Path(__file__).resolve().parent
    original = (source / "jev_browser_benchmark.py").read_bytes()
    helpers = (source / "jev_public_web_cases.py").read_bytes()
    if _sha(original) != BENCHMARK_SHA256 or _sha(helpers) != HELPERS_SHA256:
        raise ValueError("Original benchmark or common judge helpers differ from their source pins")
    if original.count(ORIGINAL_IMPORT) != 1 or ISOLATED_IMPORT in original:
        raise ValueError("The isolated public-case import must replace exactly one original line")
    derivative = original.replace(ORIGINAL_IMPORT, ISOLATED_IMPORT, 1)
    files = {
        "original_jev_browser_benchmark.py": original,
        "runner.py": derivative,
        "jev_public_web_cases.py": helpers,
        "jev_heldout_public_cases.py": (source / "jev_heldout_public_cases.py").read_bytes(),
        "launcher.py": Path(__file__).read_bytes(),
    }
    diff = "".join(
        difflib.unified_diff(
            original.decode().splitlines(keepends=True),
            derivative.decode().splitlines(keepends=True),
            fromfile="original_jev_browser_benchmark.py",
            tofile="runner.py",
        )
    ).encode()
    files["runner-import.patch"] = diff
    bundle = Path(bundle_dir).resolve()
    bundle.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (bundle / name).write_bytes(content)
    manifest = {
        "schema_id": "jev-heldout-evaluator-bundle-v1",
        "status": "source_prepared_not_site_qualified_or_evaluated",
        "selection_sha256": SELECTION_SHA256,
        "cases": case_manifest(),
        "source_change": {
            "kind": "one_explicit_public_case_import_substitution",
            "before": ORIGINAL_IMPORT.decode().strip(),
            "after": ISOLATED_IMPORT.decode().strip(),
            "original_sha256": BENCHMARK_SHA256,
            "derivative_sha256": _sha(derivative),
        },
        "files": {
            name: {"sha256": _sha(content), "size_bytes": len(content)}
            for name, content in files.items()
        },
        "readiness": {
            "model_executed": False,
            "native_site_preflight_verified": False,
            "required_before_model_evaluation": [
                "Independent no-model fresh start observations: all initial judges false; "
                "PubMed continuation really starts on asthma with both applied filters.",
                "Native selector/viewport and GET-only reachability preflight; record actual "
                "candidate/context capacity and infrastructure failures without task replacement.",
            ],
        },
    }
    _write_json(bundle / "bundle-manifest.json", manifest)
    return bundle, manifest


def verify_bundle(bundle, manifest):
    """The copied source must remain unchanged, including across child execution."""
    for name, record in manifest["files"].items():
        content = (bundle / name).read_bytes()
        if len(content) != record["size_bytes"] or _sha(content) != record["sha256"]:
            raise ValueError(f"Evaluator bundle source changed: {name}")
    actual = json.loads((bundle / "bundle-manifest.json").read_text())
    if actual != manifest:
        raise ValueError("Evaluator bundle manifest changed")


def _arguments(arguments):
    """Validate only the existing CLI surface; reject suite mixing and abbreviations."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for flag in ("upstream-path", "endpoint", "model", "output-dir"):
        parser.add_argument("--" + flag, required=True)
    for flag in (
        "api-key-env",
        "text-endpoint",
        "text-model",
        "text-api-key-env",
        "chromium-executable",
        "observation-profile-helper-sha256",
    ):
        parser.add_argument("--" + flag)
    parser.add_argument("--repeats", type=int, choices=[1], default=1)
    parser.add_argument("--max-steps", type=int, choices=[16], default=16)
    parser.add_argument("--seed", type=int, choices=[20261014], default=20261014)
    parser.add_argument("--suite-split", choices=["development"], default="development")
    parser.add_argument(
        "--observation-profile",
        choices=["composed-visible-text-v3"],
        default="composed-visible-text-v3",
    )
    parser.add_argument("--public-cases", nargs="+", choices=tuple(PUBLIC_WEB_CASES))
    values = parser.parse_args(arguments)
    if values.public_cases is not None and len(set(values.public_cases)) != len(
        values.public_cases
    ):
        parser.error("Public cases must be distinct")
    if values.observation_profile_helper_sha256 not in (None, PROFILE_SHA256):
        parser.error("The declared v3 observation helper SHA is required")
    forwarded = list(arguments)
    # Explicit defaults are added only where the original CLI would otherwise
    # select native observations or local fixtures. No original argument changes.
    if values.public_cases is None:
        forwarded += ["--public-cases", *PUBLIC_WEB_CASES]
    if not any(arg.split("=", 1)[0] == "--observation-profile" for arg in arguments):
        forwarded += ["--observation-profile", values.observation_profile]
    if values.observation_profile_helper_sha256 is None:
        forwarded += ["--observation-profile-helper-sha256", PROFILE_SHA256]
    return forwarded, values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    forwarded, values = _arguments(arguments) if arguments or not args.prepare_only else ([], None)
    if values is not None and Path(values.output_dir).exists():
        parser.error("The original benchmark output directory must be new")
    bundle, manifest = prepare_bundle(args.bundle_dir)
    manifest_sha = _sha((bundle / "bundle-manifest.json").read_bytes())
    if args.prepare_only:
        print(
            json.dumps(
                {"bundle": str(bundle), "manifest_sha256": manifest_sha, "model_executed": False}
            )
        )
        return 0
    verify_bundle(bundle, manifest)
    receipt = {
        "schema_id": "jev-heldout-evaluator-launch-v1",
        "status": "child_starting",
        "bundle_manifest_sha256": manifest_sha,
        "argument_sha256": _sha(json.dumps(forwarded, ensure_ascii=False).encode()),
        "public_case_ids": values.public_cases or list(PUBLIC_WEB_CASES),
        "selection_case_count": len(PUBLIC_WEB_CASES),
        "requested_case_count": len(values.public_cases or PUBLIC_WEB_CASES),
        "evaluation_scope": (
            "subset_diagnostic"
            if values.public_cases is not None and len(values.public_cases) != len(PUBLIC_WEB_CASES)
            else "predeclared_six_cases"
        ),
        "benchmark_output_dir": str(Path(values.output_dir).resolve()),
        "site_readiness": "separate_no_model_preflight_required_not_certified_by_launcher",
        "child_exit_is_task_success": False,
    }
    _write_json(bundle / "launch-receipt.json", receipt)
    started = time.perf_counter()
    error, integrity_error, returncode = None, None, None
    try:
        child = subprocess.run(
            [sys.executable, "-B", "-s", str(bundle / "runner.py"), *forwarded], check=False
        )
        returncode = child.returncode
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            verify_bundle(bundle, manifest)
        except Exception as exc:
            integrity_error = f"{type(exc).__name__}: {exc}"
        receipt.update(
            status="finished" if returncode == 0 and not integrity_error else "failed",
            returncode=returncode,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            error=error,
            source_integrity_error=integrity_error,
            source_integrity_verified=integrity_error is None,
        )
        try:
            _write_json(bundle / "launch-receipt.json", receipt)
        except Exception:
            if error is None:
                raise
    return returncode if returncode else (1 if integrity_error else 0)


if __name__ == "__main__":
    raise SystemExit(main())
