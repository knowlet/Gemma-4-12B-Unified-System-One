"""Source-only V7 bundle successor with explicit typed stale action feedback."""

import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROFILE = "composed-visible-text-settled-v7"
AGENT_PROTOCOL = "typed-stale-action-feedback-v1"
V7 = ROOT / "scripts/jev_browser_profile_v7_benchmark.py"
V7_SHA = "31450f9002ad216c32b77751997eb27748c491d7bc757c4410294d3704fdfb9e"
UTILITY = HERE / "action_feedback.py"
UTILITY_SHA = "79ccf2ccc2d88f9613c1320ee883ce9742b0a88c0535dbfae4de7c97ee428246"
ANCHOR = b'\nif __name__ == "__main__":\n    main()\n'
INSTALL = b"\nimport action_feedback as feedback\nmodel_adapter = feedback.wrap_model_adapter(model_adapter)\n"


def need(value, message):
    if not value:
        raise ValueError(message)


def ordinary(path):
    path = Path(path).absolute()
    need(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "ordinary bundle/source path required",
    )
    return path


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def checked(path, expected):
    path = ordinary(path)
    blob = path.read_bytes()
    need(len(blob) <= 2 * 1024**2 and digest(blob) == expected, "fixed selected source changed")
    return blob


def load_v7():
    blob = checked(V7, V7_SHA)
    spec = importlib.util.spec_from_file_location("_feedback_selected_v7_preparer", V7)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(blob, str(V7), "exec", dont_inherit=True), module.__dict__)
    return module


def decorate(parent, original_files):
    original = original_files["runner.py"]
    need(
        original.count(ANCHOR) == 1 and INSTALL not in original,
        "exact original adapter/main anchor",
    )
    derivative = original.replace(ANCHOR, INSTALL + ANCHOR, 1)
    inverse = derivative.replace(INSTALL, b"", 1)
    need(
        inverse == original and ast.dump(ast.parse(inverse)) == ast.dump(ast.parse(original)),
        "complete inverse V7 bytes/AST",
    )
    tree = ast.parse(derivative)
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "model_adapter"
    ]
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "model_adapter" for target in node.targets
        )
    ]
    need(
        len(definitions) == len(assignments) == 1
        and definitions[0].lineno < assignments[0].lineno < tree.body[-1].lineno,
        "wrap existing adapter after definition before main",
    )
    files = dict(original_files)
    files["runner.py"] = derivative
    files["action_feedback.py"] = checked(UTILITY, UTILITY_SHA)
    files["runner-feedback.patch"] = "".join(
        difflib.unified_diff(
            original.decode().splitlines(keepends=True),
            derivative.decode().splitlines(keepends=True),
            fromfile="v7_runner.py",
            tofile="runner.py",
        )
    ).encode()
    manifest = {
        "schema_id": "jev-v7-typed-stale-feedback-runner-bundle-v1",
        "status": "source_prepared_not_site_or_model_qualified",
        "profile": PROFILE,
        "agent_protocol": AGENT_PROTOCOL,
        "case_module": parent["case_module"],
        "profile_helper_sha256": parent["profile_helper_sha256"],
        "parent_v7_manifest": parent,
        "source_delta": {
            "model_adapter_wrap_insertions": 1,
            "utility_sha256": UTILITY_SHA,
            "inverse_complete_bytes_equal": True,
            "inverse_complete_AST_equal": True,
            "original_v7_runner_sha256": digest(original),
            "derivative_runner_sha256": digest(derivative),
            "observation_runtime_tasks_judges_guards_unchanged": True,
        },
        "files": {
            name: {"sha256": digest(blob), "size_bytes": len(blob)} for name, blob in files.items()
        },
        "model_executed": False,
        "runtime_qualification_attested": False,
        "measured_browser_improvement": None,
    }
    return files, manifest


def expected_bundle(case_module):
    v7 = load_v7()
    with tempfile.TemporaryDirectory(prefix=".v7-feedback-verify-", dir=HERE) as directory:
        bundle, parent = v7.prepare_bundle(Path(directory) / "v7", case_module=case_module)
        original = {name: (bundle / name).read_bytes() for name in parent["files"]}
    return decorate(parent, original)


def prepare_bundle(bundle_dir, *, case_module="original", source_dir=None):
    need(source_dir is None, "fixed selected V7 source directory only")
    v7 = load_v7()
    bundle, parent = v7.prepare_bundle(bundle_dir, case_module=case_module)
    original = {name: (bundle / name).read_bytes() for name in parent["files"]}
    files, manifest = decorate(parent, original)
    for name in ("runner.py", "action_feedback.py", "runner-feedback.patch"):
        (bundle / name).write_bytes(files[name])
    (bundle / "bundle-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    verify_bundle(bundle, manifest)
    return bundle, manifest


def verify_bundle(bundle, manifest):
    bundle = ordinary(bundle)
    need(manifest["case_module"] in ("original", "heldout"), "original selected case module")
    expected, expected_manifest = expected_bundle(manifest["case_module"])
    need(manifest == expected_manifest, "exact regenerated selected manifest")
    allowed = {*expected, "bundle-manifest.json", "launch-receipt.json"}
    paths = list(bundle.iterdir())
    need(
        all(p.is_file() and not p.is_symlink() and p.name in allowed for p in paths),
        "exact ordinary bundle population",
    )
    need({p.name for p in paths} >= {*expected, "bundle-manifest.json"}, "complete bundle files")
    for name, blob in expected.items():
        need((bundle / name).read_bytes() == blob, "exact selected bundle bytes: " + name)
    need(
        json.loads((bundle / "bundle-manifest.json").read_bytes()) == manifest,
        "saved manifest exact identity",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--case-module", choices=["original", "heldout"], default="original")
    args = parser.parse_args(argv)
    if args.bundle_dir is None:
        print(
            json.dumps(
                {
                    "mode": "DRY_SOURCE_ONLY",
                    "agent_protocol": AGENT_PROTOCOL,
                    "profile": PROFILE,
                    "model_executed": False,
                }
            )
        )
        return 0
    bundle, manifest = prepare_bundle(args.bundle_dir, case_module=args.case_module)
    print(
        json.dumps(
            {
                "bundle": str(bundle),
                "manifest_sha256": digest((bundle / "bundle-manifest.json").read_bytes()),
                "agent_protocol": manifest["agent_protocol"],
                "model_executed": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
