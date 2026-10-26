"""Opt-in source seam for the existing c513 V2 generated-action service.

No model, actor, HTTP listener, helper or benchmark is started by this module.
The original constructor/adapter/HTTP/budget/helper implementation is retained;
one local API import selects the private FIND_TEXT validators and grammar.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PROFILE = "jev-native-v2-find-text-source-integration-v1"
INTEGRATION = (
    ROOT / "artifacts/jev-runtime/browser-find-text-integration-20261022-01/integration.py"
)
INTEGRATION_SHA = "6a8fba79c121ddb24858dce455b7b25fdd74df9de08a804aa452738704edcbef"
DRIVER = ROOT / "scripts/jev_native_v2_browser.py"
DRIVER_SHA = "c513dc5bcb091de4fd0d4bd7a0776078e82be16106328b6de9d8424e32f2ee11"
EXTRA_PROJECT = (
    "src/s1/browser_policy.py",
    "src/s1/unified.py",
    "src/s1/checkpoint.py",
    "src/s1/runtime.py",
    "src/s1/schema.py",
    "src/s1/browser_action_prompt.py",
)
SCRIPT_PATHS = {
    "jev_field_json_grammar": "scripts/jev_field_json_grammar.py",
    "jev_field_syntax_helper": "scripts/jev_field_syntax_helper.py",
}


def need(value, message):
    if not value:
        raise ValueError(message)


def checked(ref):
    need(type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"}, "Exact source ref")
    p = Path(ref["path"])
    need(not p.is_absolute() and ".." not in p.parts, "Repo-relative source")
    path = ROOT / p
    need(not any(x.is_symlink() for x in (path, *path.parents)), "Ordinary selected source")
    with path.open("rb") as stream:
        blob = stream.read(1024**2 + 1)
    need(
        len(blob) <= 1024**2
        and len(blob) == ref["size_bytes"]
        and hashlib.sha256(blob).hexdigest() == ref["sha256"],
        "Selected source bytes differ",
    )
    return blob


def load(name, path, blob):
    need(name not in sys.modules, "Fresh selected script namespace")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(blob, str(path), "exec", dont_inherit=True), vars(module))
    return module


@contextmanager
def selected(blob, expected_sha, *, profile=None):
    need(profile == PROFILE, "Explicit nativeV2 FIND profile")
    need(type(blob) is bytes and hashlib.sha256(blob).hexdigest() == expected_sha, "Selected SHA")
    value = json.loads(blob)
    need(
        type(value) is dict
        and set(value) == {"profile", "source", "integration", "driver", "scripts", "project"}
        and value["profile"] == PROFILE,
        "Exact source-only selection",
    )
    need(value["source"]["path"] == str(Path(__file__).relative_to(ROOT)), "Fixed selector")
    need(value["integration"]["path"] == str(INTEGRATION.relative_to(ROOT)), "Fixed integration")
    need(value["integration"]["sha256"] == INTEGRATION_SHA, "Reviewed integration SHA")
    need(value["driver"]["path"] == str(DRIVER.relative_to(ROOT)), "Fixed V2 driver")
    need(value["driver"]["sha256"] == DRIVER_SHA, "Existing V2 driver SHA")
    checked(value["source"])
    integration_blob, driver_blob = checked(value["integration"]), checked(value["driver"])
    need(type(value["scripts"]) is dict and set(value["scripts"]) == set(SCRIPT_PATHS), "Scripts")
    for name, path in SCRIPT_PATHS.items():
        need(value["scripts"][name]["path"] == path, "Fixed script source")
        checked(value["scripts"][name])
    project = value["project"]
    need(type(project) is dict and set(project) == {"descriptor", "sha256"}, "Project selection")
    descriptor = project["descriptor"]
    need(
        hashlib.sha256(json.dumps(descriptor).encode()).hexdigest() == project["sha256"],
        "Project SHA",
    )
    paths = {x["path"] for x in descriptor["project_sources"]}
    need(set(EXTRA_PROJECT) <= paths, "Original V2 backend and prompt sources required")
    # Each descriptor was independently sealed. Existing importer verifies every source byte.
    before = set(sys.modules)
    module = None
    try:
        integration = load("_selected_s1_find_integration", INTEGRATION, integration_blob)
        with integration.bootstrap_selected(
            json.dumps(descriptor).encode(), project["sha256"], profile=integration.PROFILE
        ) as bundle:
            for name, path in SCRIPT_PATHS.items():
                load(name, ROOT / path, checked(value["scripts"][name]))
            old = "    from s1.api import create_app\n"
            new = "    from s1._experimental_find_api import create_app\n"
            text = driver_blob.decode()
            need(text.count(old) == 1 and new not in text, "Unique original local API seam")
            derived = text.replace(old, new)
            need(derived.replace(new, old).encode() == driver_blob, "Exact inverse source")
            module = load("_selected_s1_find_native_v2", DRIVER, derived.encode())
            proof = {
                "original_sha256": DRIVER_SHA,
                "derived_sha256": hashlib.sha256(derived.encode()).hexdigest(),
                "declared_edits": 1,
                "inverse_exact": True,
                "constructor_LoRA_HTTP_helper_resource_math_unchanged": True,
            }
            primary = None
            try:
                yield bundle, module, proof
            except BaseException as error:
                primary = error
                raise
            finally:
                # Authenticate final sources on success and exceptional context exit.
                try:
                    checked(value["source"])
                    checked(value["driver"])
                    for ref in value["scripts"].values():
                        checked(ref)
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note("Final nativeV2 FIND source validation failed: " + str(error))
    finally:
        for name in ("_selected_s1_find_integration", "_selected_s1_find_native_v2", *SCRIPT_PATHS):
            if name not in before:
                sys.modules.pop(name, None)
