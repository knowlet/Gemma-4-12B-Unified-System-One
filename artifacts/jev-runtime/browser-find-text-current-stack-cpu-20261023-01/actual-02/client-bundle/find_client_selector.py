"""Source-only FIND_TEXT client with a private sixteen-action namespace.

Preparation creates a new isolated bundle; it starts no browser, model, service
or production task. A future independently selected owner must bind this bundle
and the compatible private server, runtime, live helper and final closure.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import sys
import types
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROFILE = "jev-v7-find-text-client-budget-source-v2"
V7 = "composed-visible-text-settled-v7"
V7_SHA = "d9f4236d1d46c9472c3d6c1551342639ecb870e6c0dd419246927645410ea868"
BUILDER = "scripts/jev_browser_profile_v7_benchmark.py"
BUILDER_SHA = "31450f9002ad216c32b77751997eb27748c491d7bc757c4410294d3704fdfb9e"
CLIENT = "artifacts/jev-runtime/browser-find-text-profile-20261022-01/find_profile.py"
CLIENT_SHA = "98c7fa29265e2c0c25bdcae0825522cebcd35dc86c16ed53aa7609bd52e6a08d"
KNOWN4 = (
    "public-arxiv-title-filter-results-v1",
    "public-arxiv-original-paper-open-v1",
    "public-python-counter-most-common-v1",
    "public-nasa-mars-facts-moons-v1",
)
PUBLIC6 = tuple(f"heldout-public-{i:03d}" for i in range(6))
CONSTRUCTOR = (
    b"                            agent = agent_module.Agent(url, goal, screenshots=False)\n"
)
SELECTED_CONSTRUCTOR = b"                            agent = _selected_find_agent_class(agent_module, model_module)(url, goal, screenshots=False)\n"
MAIN = b'\nif __name__ == "__main__":\n'
MAX_SOURCE = 1024**2


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def encode(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode()
        + b"\n"
    )


def ordinary(path):
    path = Path(path).absolute()
    need(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary selected path",
    )
    return path


def read(path):
    path = ordinary(path)
    need(path.is_file(), "Ordinary selected file")
    with path.open("rb") as stream:
        blob = stream.read(MAX_SOURCE + 1)
    need(len(blob) <= MAX_SOURCE, "Bounded selected source")
    return blob


def reference(root, path):
    path = ordinary(path)
    blob = read(path)
    return {"path": str(path.relative_to(root)), "sha256": sha(blob), "size_bytes": len(blob)}


def checked(root, ref):
    need(
        type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"}, "Exact source reference"
    )
    rel = Path(ref["path"])
    need(
        not rel.is_absolute() and ".." not in rel.parts and rel.as_posix() == ref["path"],
        "Repo-relative source",
    )
    blob = read(root / rel)
    need(sha(blob) == ref["sha256"] and len(blob) == ref["size_bytes"], "Selected source changed")
    return blob


def source_module(path, blob, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(blob, str(path), "exec", dont_inherit=True), vars(module))
    return module


def selection(path, expected_sha):
    blob = read(path)
    need(
        type(expected_sha) is str and sha(blob) == expected_sha, "Independently selected client SHA"
    )
    value = json.loads(blob)
    need(
        type(value) is dict
        and set(value) == {"schema_id", "profile", "repository_root", "sources"},
        "Exact client selection",
    )
    need(
        value["schema_id"] == "jev-v7-find-client-selection-v1" and value["profile"] == PROFILE,
        "Explicit client profile",
    )
    root = ordinary(value["repository_root"])
    need(root.is_dir() and root.is_absolute(), "Selected repository root")
    sources = value["sources"]
    need(
        type(sources) is dict and set(sources) == {"selector", "client", "agent", "model"},
        "Exact client source set",
    )
    need(
        sources["client"]["path"] == CLIENT and sources["client"]["sha256"] == CLIENT_SHA,
        "Selected original FIND component",
    )
    need(
        sources["agent"]["path"] == "artifacts/jev-runtime/upstream/jev_ultrafast/agent.py"
        and sources["model"]["path"] == "artifacts/jev-runtime/upstream/jev_ultrafast/model.py",
        "Original canonical Agent/model source paths",
    )
    for ref in sources.values():
        checked(root, ref)
    # A copied bundle selector executes its exact selected bytes without assuming
    # the bundle's directory depth equals the original repository directory.
    need(sha(read(__file__)) == sources["selector"]["sha256"], "Executed selector copy differs")
    return value, root


def agent_class(agent_module, model_module, *, selection_path, selection_sha):
    value, root = selection(selection_path, selection_sha)
    need(
        type(agent_module.MAX_STEPS) is int and agent_module.MAX_STEPS in {16, 60},
        "Known original sixty-action or selected sixteen-action namespace",
    )
    package = agent_module.__name__.rsplit(".", 1)[0]
    browser = sys.modules.get(package + ".browser")
    need(
        browser is not None and browser.Browser is agent_module.Browser,
        "Actual selected Browser module",
    )
    need(type(browser.MARKER) is str and bool(browser.MARKER), "Actual selected Browser marker")
    component = source_module(
        root / CLIENT, checked(root, value["sources"]["client"]), "_selected_v7_find_client"
    )
    # The original runner limits the outer loop to16, while canonical Agent
    # imports60. Supply16 only to the already-private command/factory namespace;
    # the canonical module, Agent class, choose wrappers and global60 stay intact.
    private_agent = types.ModuleType(agent_module.__name__)
    vars(private_agent).update(vars(agent_module))
    private_agent.MAX_STEPS = 16
    return component.make_agent_class(private_agent, model_module, marker_expression=browser.MARKER)


def require_http_model_adapter(original):
    @wraps(original)
    @contextmanager
    def selected_adapter(agent_module, model_module, **kwargs):
        need(
            type(kwargs.get("text_endpoint")) is str
            and bool(kwargs["text_endpoint"])
            and type(kwargs.get("text_model")) is str
            and bool(kwargs["text_model"])
            and kwargs.get("text_helper") is None,
            "FIND profile requires an explicit HTTP text endpoint/model; callback-only helpers are unsupported",
        )
        # Original endpoint redirection, real transport retry records, redaction,
        # environment restoration and callback wrappers remain the same code.
        with original(agent_module, model_module, **kwargs) as result:
            yield result

    return selected_adapter


def validate_arguments(arguments, case_module):
    need(case_module in {"original", "heldout"}, "Explicit fixed case group")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for flag in (
        "upstream-path",
        "endpoint",
        "model",
        "output-dir",
        "text-endpoint",
        "text-model",
        "chromium-executable",
    ):
        parser.add_argument("--" + flag, required=True)
    for flag in ("api-key-env", "text-api-key-env"):
        parser.add_argument("--" + flag)
    parser.add_argument("--repeats", type=int, choices=[1], default=1)
    parser.add_argument("--max-steps", type=int, choices=[16], default=16)
    parser.add_argument("--seed", type=int, choices=[20261014], default=20261014)
    parser.add_argument("--suite-split", choices=["development"], default="development")
    parser.add_argument("--observation-profile", choices=[V7], required=True)
    parser.add_argument("--observation-profile-helper-sha256", choices=[V7_SHA], required=True)
    parser.add_argument("--public-cases", nargs="+", required=True)
    args = parser.parse_args(list(arguments))
    need(
        tuple(args.public_cases) == (KNOWN4 if case_module == "original" else PUBLIC6),
        "Complete fixed group and original task order required",
    )
    return args


def install(selection_sha, case_module):
    return (
        "\nimport find_client_selector as _find_client_selector\n"
        "def _selected_find_agent_class(agent_module, model_module):\n"
        "    return _find_client_selector.agent_class(agent_module, model_module,\n"
        "        selection_path=Path(__file__).with_name('find-client-selection.json'),\n"
        f"        selection_sha={selection_sha!r})\n"
        "model_adapter = _find_client_selector.require_http_model_adapter(model_adapter)\n"
        "_original_v7_main = main\n"
        "def main():\n"
        f"    _find_client_selector.validate_arguments(sys.argv[1:], {case_module!r})\n"
        "    return _original_v7_main()\n"
    ).encode()


def derive_runner(parent, selection_sha, case_module):
    need(
        parent.count(CONSTRUCTOR) == 1 and parent.count(MAIN) == 1,
        "Unique original constructor/main anchors",
    )
    insertion = install(selection_sha, case_module)
    changed = parent.replace(CONSTRUCTOR, SELECTED_CONSTRUCTOR).replace(MAIN, insertion + MAIN)
    inverse = changed.replace(insertion + MAIN, MAIN).replace(SELECTED_CONSTRUCTOR, CONSTRUCTOR)
    need(
        inverse == parent
        and ast.dump(ast.parse(inverse), include_attributes=False)
        == ast.dump(ast.parse(parent), include_attributes=False),
        "Full inverse bytes/AST",
    )
    # The declared function body is preserved; only its private global binding is wrapped.
    original_adapter = next(
        n
        for n in ast.parse(parent).body
        if isinstance(n, ast.FunctionDef) and n.name == "model_adapter"
    )
    derived_adapter = next(
        n
        for n in ast.parse(changed).body
        if isinstance(n, ast.FunctionDef) and n.name == "model_adapter"
    )
    need(
        ast.dump(original_adapter, include_attributes=False)
        == ast.dump(derived_adapter, include_attributes=False),
        "Original instrumentation body changed",
    )
    return changed


def prepare_bundle(bundle_dir, *, case_module="original", profile=None):
    need(profile == PROFILE, "Explicit opt-in V7 FIND profile")
    builder_blob = read(ROOT / BUILDER)
    need(sha(builder_blob) == BUILDER_SHA, "Selected original V7 builder changed")
    builder = source_module(ROOT / BUILDER, builder_blob, "_selected_original_v7_builder")
    bundle, parent = builder.prepare_bundle(bundle_dir, case_module=case_module)
    need(
        parent["profile"] == V7 and parent["profile_helper_sha256"] == V7_SHA,
        "Exact V7 observation helper",
    )
    own = Path(__file__).resolve()
    client_selection = {
        "schema_id": "jev-v7-find-client-selection-v1",
        "profile": PROFILE,
        "repository_root": str(ROOT),
        "sources": {
            "selector": reference(ROOT, own),
            "client": reference(ROOT, ROOT / CLIENT),
            "agent": reference(
                ROOT, ROOT / "artifacts/jev-runtime/upstream/jev_ultrafast/agent.py"
            ),
            "model": reference(
                ROOT, ROOT / "artifacts/jev-runtime/upstream/jev_ultrafast/model.py"
            ),
        },
    }
    need(
        client_selection["sources"]["client"]["sha256"] == CLIENT_SHA,
        "Selected FIND client changed",
    )
    selected_blob = encode(client_selection)
    selected_sha = sha(selected_blob)
    old_runner = read(bundle / "runner.py")
    new_runner = derive_runner(old_runner, selected_sha, case_module)
    new_files = {
        "runner.py": new_runner,
        "find_client_selector.py": read(own),
        "find-client-selection.json": selected_blob,
        "v7-parent-manifest.json": encode(parent),
        "runner-find-profile.patch": "".join(
            difflib.unified_diff(
                old_runner.decode().splitlines(keepends=True),
                new_runner.decode().splitlines(keepends=True),
                fromfile="v7-parent-runner.py",
                tofile="runner.py",
            )
        ).encode(),
    }
    for name, blob in new_files.items():
        (bundle / name).write_bytes(blob)
    manifest = {
        **parent,
        "schema_id": "jev-v7-find-isolated-runner-bundle-v1",
        "agent_profile": PROFILE,
        "client_selection_sha256": selected_sha,
        "v7_parent_manifest_sha256": sha(encode(parent)),
        "fixed_case_ids": list(KNOWN4 if case_module == "original" else PUBLIC6),
        "seed": 20261014,
        "max_steps": 16,
        "helper_action_seconds": 20,
        "source_component_only": True,
        "quality_score": None,
        "files": {
            **parent["files"],
            **{
                name: {"sha256": sha(blob), "size_bytes": len(blob)}
                for name, blob in new_files.items()
            },
        },
    }
    (bundle / "bundle-manifest.json").write_bytes(encode(manifest))
    verify_bundle(bundle, manifest)
    return bundle, manifest


def verify_bundle(bundle, manifest):
    bundle = ordinary(bundle)
    need(
        manifest["schema_id"] == "jev-v7-find-isolated-runner-bundle-v1"
        and manifest["agent_profile"] == PROFILE,
        "Selected FIND bundle identity",
    )
    allowed = {*manifest["files"], "bundle-manifest.json", "launch-receipt.json"}
    need(
        all(p.is_file() and not p.is_symlink() and p.name in allowed for p in bundle.iterdir()),
        "Exact ordinary bundle population",
    )
    for name, ref in manifest["files"].items():
        need(Path(name).name == name, "Flat bundle member")
        blob = read(bundle / name)
        need(
            len(blob) == ref["size_bytes"] and sha(blob) == ref["sha256"],
            "Prepared FIND bundle changed",
        )
    need(
        json.loads(read(bundle / "bundle-manifest.json")) == manifest,
        "Prepared FIND manifest changed",
    )
    parent_blob = read(bundle / "v7-parent-manifest.json")
    need(sha(parent_blob) == manifest["v7_parent_manifest_sha256"], "Selected parent manifest")
    parent = json.loads(parent_blob)
    need(
        parent["profile"] == V7 and parent["profile_helper_sha256"] == V7_SHA,
        "Unchanged V7 observation profile",
    )
    for name, ref in parent["files"].items():
        if name != "runner.py":
            need(manifest["files"][name] == ref, "Unrelated parent file changed")
    runner = read(bundle / "runner.py")
    insertion = install(manifest["client_selection_sha256"], manifest["case_module"])
    need(
        runner.count(insertion + MAIN) == 1 and runner.count(SELECTED_CONSTRUCTOR) == 1,
        "Declared runner selection seams",
    )
    inverse = runner.replace(insertion + MAIN, MAIN).replace(SELECTED_CONSTRUCTOR, CONSTRUCTOR)
    need(
        sha(inverse) == parent["files"]["runner.py"]["sha256"]
        and len(inverse) == parent["files"]["runner.py"]["size_bytes"],
        "Full V7 runner inverse",
    )
    selection(bundle / "find-client-selection.json", manifest["client_selection_sha256"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--prepare-bundle")
    parser.add_argument("--case-module", choices=["original", "heldout"], default="original")
    parser.add_argument("--profile")
    args = parser.parse_args(argv)
    if args.prepare_bundle:
        bundle, manifest = prepare_bundle(
            args.prepare_bundle, case_module=args.case_module, profile=args.profile
        )
        print(
            json.dumps(
                {
                    "bundle": str(bundle),
                    "manifest_sha256": sha(read(bundle / "bundle-manifest.json")),
                    "source_component_only": True,
                    "actual_model_or_browser_started": False,
                }
            )
        )
    else:
        print(
            json.dumps(
                {
                    "profile": PROFILE,
                    "source_component_only": True,
                    "actual_model_or_browser_started": False,
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
