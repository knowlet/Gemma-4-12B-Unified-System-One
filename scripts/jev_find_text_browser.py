"""Prepare an explicit FIND_TEXT client over an unchanged isolated v7 bundle.

This opt-in builder starts no browser, service, model or HTTP request. The
canonical benchmark and original v7 builder remain byte-pinned predecessors.
The additional runner edits select the checked experimental Agent, require its
existing sixteen-action budget and require the original external HTTP helper.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import types
from pathlib import Path

ACTION_PROFILE = "jev-find-text-experimental-v1"
PROFILE = "composed-visible-text-settled-v7"
BASE_BUILDER = "scripts/jev_browser_profile_v7_benchmark.py"
BASE_BUILDER_SHA256 = "31450f9002ad216c32b77751997eb27748c491d7bc757c4410294d3704fdfb9e"
ACTION_SELECTOR = "src/s1/jev_action_profiles.py"
ACTION_SELECTOR_SHA256 = "fdf8873c5ebe68d7947a60b8c320f0fd118ae3918d0b87a870006e8678e7c747"
COMPONENTS = {
    "artifacts/jev-runtime/browser-find-text-profile-20261022-01/private_protocol.py": "7f4dab379ac145818fad73babcb4ba0633520124fd862c365c8429489a352f1f",
    "artifacts/jev-runtime/browser-find-text-profile-20261022-01/find_profile.py": "98c7fa29265e2c0c25bdcae0825522cebcd35dc86c16ed53aa7609bd52e6a08d",
    "artifacts/jev-runtime/browser-find-text-cpu-browser-20261022-01/experimental_wire.py": "b742487c3218355cc61e7cf846b61e3356874f28e84e325a7f914b76c5895e70",
}
MAX_SOURCE_BYTES = 200_000
BASE_DIRECTORY = "base-v7"
GUARD = (
    b"    if repeats < 1 or max_steps < 1:\n"
    b'        raise ValueError("repeats and max_steps must be positive")\n'
)
CONFIGURATION = b"    configuration = {\n"
TASK_SPECS = b"                    task_specs = [\n"
CONSTRUCTOR = (
    b"                            agent = agent_module.Agent(url, goal, screenshots=False)\n"
)
SELECTED_CONSTRUCTOR = (
    b"                            agent = _selected_action_agent(url, goal, screenshots=False)\n"
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _ordinary(path):
    path = Path(path).absolute()
    _require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary selected source/bundle path required",
    )
    return path


def _read(path):
    path = _ordinary(path)
    _require(path.is_file(), "Ordinary selected source file required")
    with path.open("rb") as stream:
        blob = stream.read(MAX_SOURCE_BYTES + 1)
    _require(0 < len(blob) <= MAX_SOURCE_BYTES, "Bounded selected source required")
    return blob


def _record(blob):
    return {"sha256": _sha(blob), "size_bytes": len(blob)}


def _encode(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def _source_references(root):
    result = {}
    for label, expected in {ACTION_SELECTOR: ACTION_SELECTOR_SHA256, **COMPONENTS}.items():
        blob = _read(root / label)
        _require(_sha(blob) == expected, "Pinned FIND_TEXT source changed: " + label)
        result[label] = _record(blob)
    return result


def _base_builder(root):
    path = root / BASE_BUILDER
    blob = _read(path)
    _require(_sha(blob) == BASE_BUILDER_SHA256, "Original v7 builder changed")
    module = types.ModuleType("_find_text_original_v7_builder")
    module.__file__ = str(path)
    exec(compile(blob, str(path), "exec", dont_inherit=True), vars(module))
    return module


def _edits(root):
    selector = str(root / ACTION_SELECTOR)
    selected_factory = (
        "                    _action_selector_path = Path(" + repr(selector) + ")\n"
        "                    if hashlib.sha256(_action_selector_path.read_bytes()).hexdigest() != "
        + repr(ACTION_SELECTOR_SHA256)
        + ":\n"
        "                        raise ValueError('Selected FIND_TEXT action selector changed')\n"
        "                    from s1 import jev_action_profiles as _action_profiles\n"
        "                    if Path(_action_profiles.__file__).resolve() != _action_selector_path:\n"
        "                        raise ValueError('Imported FIND_TEXT action selector origin differs')\n"
        "                    _selected_action_agent = _action_profiles.make_agent_class(\n"
        "                        " + repr(ACTION_PROFILE) + ", agent_module, model_module,\n"
        "                        marker_expression=sys.modules['jev_ultrafast.browser'].MARKER,\n"
        "                    )\n"
    ).encode()
    guard = (
        b"    if (type(max_steps) is not int or max_steps != 16\n"
        b"            or type(text_endpoint) is not str or not text_endpoint.strip()\n"
        b"            or type(text_model) is not str or not text_model.strip()\n"
        b"            or text_helper is not None):\n"
        b"        raise ValueError('FIND_TEXT requires16 decisions and an explicit external HTTP text endpoint/model; callbacks are unsupported')\n"
    )
    configuration = (
        b'        "action_profile": "jev-find-text-experimental-v1",\n'
        b'        "action_profile_scope": "Experimental ninth FIND_TEXT operation; no browser or model success qualification",\n'
        b'        "action_profile_selector_sha256": '
        + repr(ACTION_SELECTOR_SHA256).encode()
        + b",\n"
    )
    return (
        (GUARD, GUARD + guard),
        (CONFIGURATION, CONFIGURATION + configuration),
        (TASK_SPECS, selected_factory + TASK_SPECS),
        (CONSTRUCTOR, SELECTED_CONSTRUCTOR),
    )


def _derive(parent, root):
    changed = parent
    edits = _edits(root)
    for before, after in edits:
        _require(
            changed.count(before) == 1 and changed.count(after) == 0,
            "Unique FIND_TEXT runner seam required",
        )
        changed = changed.replace(before, after, 1)
    inverse = changed
    for before, after in reversed(edits):
        _require(inverse.count(after) == 1, "Unique inverse FIND_TEXT runner seam required")
        inverse = inverse.replace(after, before, 1)
    _require(
        inverse == parent
        and ast.dump(ast.parse(inverse), include_attributes=False)
        == ast.dump(ast.parse(parent), include_attributes=False),
        "Undeclared FIND_TEXT runner source/AST edit",
    )
    return changed


def _patch(parent, derivative):
    return "".join(
        difflib.unified_diff(
            parent.decode().splitlines(keepends=True),
            derivative.decode().splitlines(keepends=True),
            fromfile="base-v7/runner.py",
            tofile="runner.py",
        )
    ).encode()


def prepare_bundle(bundle_dir, *, case_module="original", source_dir=None):
    """Prepare a new opt-in client; return the original builder's public shape."""
    _require(
        case_module in {"original", "heldout"}, "Explicit original or heldout case module required"
    )
    source = _ordinary(source_dir if source_dir is not None else Path(__file__).resolve().parent)
    root = source.parent
    references = _source_references(root)
    builder = _base_builder(root)
    bundle = _ordinary(bundle_dir)
    bundle.mkdir(parents=True, exist_ok=False)
    base, base_manifest = builder.prepare_bundle(
        bundle / BASE_DIRECTORY, case_module=case_module, source_dir=source
    )
    builder.verify_bundle(base, base_manifest)
    files = {name: _read(base / name) for name in base_manifest["files"]}
    parent = files["runner.py"]
    files["runner.py"] = _derive(parent, root)
    files["runner-profile.patch"] = _patch(parent, files["runner.py"])
    files["find-text-builder.py"] = _read(__file__)
    for name, blob in files.items():
        (bundle / name).write_bytes(blob)
    manifest = {
        "schema_id": "jev-v7-find-text-runner-bundle-v1",
        "status": "source_prepared_not_browser_or_model_qualified",
        "profile": PROFILE,
        "action_profile": ACTION_PROFILE,
        "case_module": case_module,
        "repository_root": str(root),
        "profile_helper_sha256": base_manifest["profile_helper_sha256"],
        "base_bundle": {"directory": BASE_DIRECTORY, "manifest": base_manifest},
        "base_builder": {"path": BASE_BUILDER, "sha256": BASE_BUILDER_SHA256},
        "action_profile_sources": references,
        "source_delta": {
            "additional_runner_seams": 4,
            "parent_runner_sha256": _sha(parent),
            "derivative_sha256": _sha(files["runner.py"]),
            "inverse_bytes_equal": True,
            "inverse_AST_equal": True,
            "original_case_population_and_judges_preserved": True,
            "original_model_adapter_transport_and_retries_preserved": True,
            "max_steps": 16,
            "external_http_text_helper_required": True,
            "original_agent_and_protocol_modules_mutated": False,
        },
        "files": {name: _record(blob) for name, blob in files.items()},
        "source_authentication_scope": "Unchanged v7 predecessor plus4 exact FIND_TEXT source pins; factory separately authenticates7 source dependencies; not full transitive Python/native closure",
        "model_executed": False,
        "runtime_qualification_attested": False,
    }
    (bundle / "bundle-manifest.json").write_bytes(_encode(manifest))
    verify_bundle(bundle, manifest)
    return bundle, manifest


def verify_bundle(bundle, manifest):
    """Check exact new population, original v7 base and declared inverse bytes."""
    bundle = _ordinary(bundle)
    _require(
        type(manifest) is dict
        and manifest.get("schema_id") == "jev-v7-find-text-runner-bundle-v1"
        and manifest.get("action_profile") == ACTION_PROFILE
        and manifest.get("profile") == PROFILE,
        "Explicit experimental FIND_TEXT bundle manifest required",
    )
    root = _ordinary(manifest["repository_root"])
    _require(
        manifest["action_profile_sources"] == _source_references(root),
        "Selected FIND_TEXT source references differ",
    )
    _require(
        manifest["base_builder"] == {"path": BASE_BUILDER, "sha256": BASE_BUILDER_SHA256},
        "Original v7 builder selection differs",
    )
    selected_base = manifest["base_bundle"]
    _require(
        type(selected_base) is dict
        and set(selected_base) == {"directory", "manifest"}
        and selected_base["directory"] == BASE_DIRECTORY,
        "Exact original v7 base selection required",
    )
    base = bundle / BASE_DIRECTORY
    base_manifest = selected_base["manifest"]
    _require(
        manifest.get("case_module") in {"original", "heldout"}
        and base_manifest.get("case_module") == manifest["case_module"]
        and manifest["profile_helper_sha256"] == base_manifest["profile_helper_sha256"],
        "Original v7 case/profile selection differs",
    )
    _base_builder(root).verify_bundle(base, base_manifest)
    records = manifest["files"]
    expected = {*base_manifest["files"], "find-text-builder.py"}
    _require(
        all(type(name) is str and Path(name).name == name for name in expected),
        "Flat FIND_TEXT bundle source names required",
    )
    _require(
        type(records) is dict and set(records) == expected,
        "Exact FIND_TEXT root source population required",
    )
    allowed = {*expected, "bundle-manifest.json", "launch-receipt.json", BASE_DIRECTORY}
    _require(
        {entry.name for entry in bundle.iterdir()}.issuperset(
            expected | {BASE_DIRECTORY, "bundle-manifest.json"}
        ),
        "Complete FIND_TEXT bundle population required",
    )
    for entry in bundle.iterdir():
        _require(
            not entry.is_symlink()
            and entry.name in allowed
            and (entry.is_dir() if entry.name == BASE_DIRECTORY else entry.is_file()),
            "Unexpected or nonordinary FIND_TEXT bundle entry",
        )
    for name, record in records.items():
        _require(
            record == _record(_read(bundle / name)), "Prepared FIND_TEXT source changed: " + name
        )
    _require(
        _read(bundle / "find-text-builder.py") == _read(__file__),
        "Executed FIND_TEXT builder copy differs",
    )
    parent = _read(base / "runner.py")
    derivative = _derive(parent, root)
    _require(
        _read(bundle / "runner.py") == derivative
        and _read(bundle / "runner-profile.patch") == _patch(parent, derivative),
        "FIND_TEXT runner source/inverse proof differs",
    )
    for name in base_manifest["files"]:
        if name not in {"runner.py", "runner-profile.patch"}:
            _require(
                _read(bundle / name) == _read(base / name),
                "Unrelated original v7 bundle source changed: " + name,
            )
    _require(
        json.loads(_read(bundle / "bundle-manifest.json")) == manifest,
        "Prepared FIND_TEXT manifest changed",
    )
