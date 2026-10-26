"""Explicit source-only bundle factory; no model/backend/service constructor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import types
from contextlib import ExitStack, contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PROFILE = "jev-find-text-experimental-v1"
SCHEMA = "jev-find-text-source-integration-selection-v1"
API_EDITS = (
    (
        "from .jev_decision3_native import JevDecision3NativeAdapter",
        "from ._experimental_find_jev_decision3_native import JevDecision3NativeAdapter",
    ),
    (
        "from .jev_generated import JevGeneratedActionAdapter",
        "from ._experimental_find_jev_generated import JevGeneratedActionAdapter",
    ),
)
COMPONENT_PATHS = {
    "integration": "artifacts/jev-runtime/browser-find-text-integration-20261022-01/integration.py",
    "client": "artifacts/jev-runtime/browser-find-text-profile-20261022-01/find_profile.py",
    "protocol": "artifacts/jev-runtime/browser-find-text-profile-20261022-01/private_protocol.py",
    "wire": "artifacts/jev-runtime/browser-find-text-cpu-browser-20261022-01/experimental_wire.py",
    "source_importer": "artifacts/jev-runtime/decision3-current-source-bootstrap-20261022-01/current_bootstrap.py",
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def ordinary(label):
    relative = Path(label)
    need(
        not relative.is_absolute() and ".." not in relative.parts and relative.as_posix() == label,
        "Ordinary repo-relative selected path",
    )
    path = ROOT / relative
    need(not any(p.is_symlink() for p in (path, *path.parents)), "Selected ordinary source only")
    need(path.is_file() and path.stat().st_size <= 1024**2, "Small selected source only")
    return path


def checked(ref):
    need(
        type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"},
        "Exact selected source reference",
    )
    blob = ordinary(ref["path"]).read_bytes()
    need(
        hashlib.sha256(blob).hexdigest() == ref["sha256"] and len(blob) == ref["size_bytes"],
        "Selected source bytes changed",
    )
    return blob


def selected(blob, expected_sha, *, profile):
    need(profile == PROFILE, "An explicit experimental profile is required")
    need(
        type(blob) is bytes
        and type(expected_sha) is str
        and hashlib.sha256(blob).hexdigest() == expected_sha,
        "Externally selected descriptor SHA required",
    )
    value = json.loads(blob)
    need(
        type(value) is dict
        and set(value) == {"schema_id", "profile", "components", "project_sources"},
        "Exact integration descriptor",
    )
    need(
        value["schema_id"] == SCHEMA and value["profile"] == PROFILE,
        "Explicit experimental descriptor identity",
    )
    need(
        type(value["components"]) is dict and set(value["components"]) == set(COMPONENT_PATHS),
        "Complete explicit component selection",
    )
    for name, label in COMPONENT_PATHS.items():
        need(value["components"][name]["path"] == label, "Fixed experimental source component path")
        checked(value["components"][name])
    refs = value["project_sources"]
    need(
        type(refs) is list and refs and len({r["path"] for r in refs}) == len(refs),
        "Unique selected project sources",
    )
    for ref in refs:
        need(
            ref["path"].startswith("src/s1/") and ref["path"].endswith(".py"),
            "Selected project Python source",
        )
        checked(ref)
    labels = {r["path"] for r in refs}
    need(
        {
            "src/s1/api.py",
            "src/s1/browser_server_evidence.py",
            "src/s1/jev_generated.py",
            "src/s1/jev_action_grammar.py",
            "src/s1/jev_decision3_native.py",
            "src/s1/evaluation/decision3.py",
        }
        <= labels,
        "Explicit parser/API/native/registry selection",
    )
    return value


def module(ref, name, *, package=None, source=None):
    blob = checked(ref) if source is None else source
    obj = types.ModuleType(name)
    obj.__file__ = str(ROOT / ref["path"])
    if package:
        obj.__package__ = package
    exec(compile(blob, obj.__file__, "exec", dont_inherit=True), vars(obj))
    return obj


def private_api(ref):
    original = checked(ref)
    text = original.decode()
    for before, after in API_EDITS:
        need(text.count(before) == 1, "Unique private API import seam")
        text = text.replace(before, after)
    inverse = text
    for before, after in reversed(API_EDITS):
        inverse = inverse.replace(after, before)
    need(inverse.encode() == original, "No undeclared API source change")
    name = "s1._experimental_find_api"
    need(name not in sys.modules, "Fresh private API namespace")
    obj = types.ModuleType(name)
    obj.__file__, obj.__package__ = str(ROOT / ref["path"]), "s1"
    sys.modules[name] = obj
    try:
        exec(
            compile(text, obj.__file__ + "::find-text-experimental", "exec", dont_inherit=True),
            vars(obj),
        )
    except BaseException:
        del sys.modules[name]
        raise
    return obj, {
        "original_sha256": ref["sha256"],
        "derived_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "declared_edits": 2,
        "inverse_exact": True,
    }


class Bundle:
    """Caller supplies a backend; this component never constructs one."""

    def __init__(self, selection, selection_sha, client, modules, wire, api, proofs):
        self._selection, self.selection_sha = copy.deepcopy(selection), selection_sha
        self.client, self.protocol, self.wire, self.api, self.proofs = (
            client,
            modules,
            wire,
            api,
            proofs,
        )
        self.active = True

    @property
    def selection(self):
        return copy.deepcopy(self._selection)

    def verify(self):
        need(self.active, "Experimental bundle is closed")
        for ref in [*self._selection["components"].values(), *self._selection["project_sources"]]:
            checked(ref)

    def create_app(
        self,
        backend,
        *,
        provider,
        model_identity,
        api_key,
        evidence_sink,
        prompt_format="raw-json-v1",
    ):
        self.verify()
        need(
            provider in {"decision3-native", "s1-generated"},
            "Explicit experimental backend provider",
        )
        need(callable(evidence_sink), "Original server evidence is required")
        kwargs = dict(
            api_key=api_key,
            model_identity=model_identity,
            request_evidence_sink=evidence_sink,
            jev_compatible=True,
        )
        if provider == "decision3-native":
            need(prompt_format == "raw-json-v1", "Native payload remains original raw ordered JSON")
            kwargs["jev_decision3_native"] = True
        else:
            kwargs.update(
                jev_generated_actions=True,
                jev_generated_action_grammar=True,
                jev_generated_prompt_format=prompt_format,
            )
        return self.api.create_app(backend, **kwargs)

    def named_response(self, request, response):
        self.verify()
        return self.wire.named_browser_response(request, response)

    def agent_class(self, agent_module, model_module, *, marker_expression):
        self.verify()
        return self.client.make_agent_class(
            agent_module, model_module, marker_expression=marker_expression
        )


@contextmanager
def bootstrap_selected(blob, expected_sha, *, profile=None):
    """Authenticate sources before any backend is supplied; no model/service run.

    Fresh project namespaces are required. The reviewed importer executes selected
    project source bytes directly rather than accepting timestamp-valid bytecode.
    It is removed on exit and only modules imported by this fresh bundle are removed.
    """
    value = selected(blob, expected_sha, profile=profile)
    initial_project = {name for name in sys.modules if name == "s1" or name.startswith("s1.")}
    need(
        not initial_project, "Fresh selected project modules required; no original globals patched"
    )
    components = value["components"]
    importer = module(components["source_importer"], "_find_selected_source_importer")
    project = {ref["path"]: ref["sha256"] for ref in value["project_sources"]}
    bundle = None
    try:
        with ExitStack() as stack:
            stack.enter_context(importer.ProjectSources(project))
            protocol = module(components["protocol"], "_find_selected_protocol")
            modules, protocol_proof = stack.enter_context(protocol.load_private_protocol())
            client = module(components["client"], "_find_selected_client")
            wire, wire_proof = module(components["wire"], "_find_selected_wire").load_wire()
            api_ref = next(r for r in value["project_sources"] if r["path"] == "src/s1/api.py")
            api, api_proof = private_api(api_ref)
            bundle = Bundle(
                value,
                expected_sha,
                client,
                modules,
                wire,
                api,
                {"protocol": protocol_proof, "wire": wire_proof, "api": api_proof},
            )
            bundle.verify()
            primary = None
            try:
                yield bundle
            except BaseException as error:
                primary = error
                raise
            finally:
                try:
                    bundle.verify()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note(
                        "Final experimental source authentication failed: " + str(error)
                    )
    finally:
        if bundle:
            bundle.active = False
        for name in tuple(sys.modules):
            if (name == "s1" or name.startswith("s1.")) and name not in initial_project:
                del sys.modules[name]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--selection", type=Path, default=HERE / "selection.json")
    parser.add_argument("--selection-sha")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        selected(args.selection.read_bytes(), args.selection_sha, profile=args.profile)
    print(
        json.dumps(
            {
                "profile": PROFILE,
                "source_selection_checked": args.check,
                "source_only": True,
                "backend_constructed": False,
                "server_started": False,
                "model_qualified": False,
                "server_runtime_qualified": False,
                "quality_score": None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
