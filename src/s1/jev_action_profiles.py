"""Explicit browser action profiles; the default keeps the original eight operations.

The experimental profile reuses reviewed, hash-pinned source components. Private
imports use a disjoint permanent namespace so deferred grammar imports remain
available to concurrent app instances without changing any original module.
No model/framework constructor, HTTP endpoint or browser starts on import.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path

ACTION_PROFILES = ("standard", "jev-find-text-experimental-v1")
DEFAULT_ACTION_PROFILE = "standard"
_ROOT = Path(__file__).resolve().parents[2]
_COMPONENTS = {
    "protocol": (
        "artifacts/jev-runtime/browser-find-text-profile-20261022-01/private_protocol.py",
        "7f4dab379ac145818fad73babcb4ba0633520124fd862c365c8429489a352f1f",
    ),
    "agent": (
        "artifacts/jev-runtime/browser-find-text-profile-20261022-01/find_profile.py",
        "98c7fa29265e2c0c25bdcae0825522cebcd35dc86c16ed53aa7609bd52e6a08d",
    ),
    "wire": (
        "artifacts/jev-runtime/browser-find-text-cpu-browser-20261022-01/experimental_wire.py",
        "b742487c3218355cc61e7cf846b61e3356874f28e84e325a7f914b76c5895e70",
    ),
}
_COMPONENT_SIZES = {"protocol": 4245, "agent": 13344, "wire": 1596}
_ORIGINALS = {
    "jev_generated": "c356e0eb3c046cb0787d7be0c5960803006231e8bdb414de644ae2167e72d849",
    "jev_action_grammar": "399e06e28172c96004b774ba97c04100687340e6b1326d317dbdf10557da75d2",
    "jev_decision3_native": "67616bb70a8acdbf96b1d2e4c1407cf8cc2a24db51b5ca383e33b4d9a0488de9",
}
_DERIVED = {
    "jev_generated": "b238a566d161682be579e7fcfda8a1dc910d4aaa34b49df5982f65fa351cf217",
    "jev_action_grammar": "a26f2b968183dd246728daa11ed153d80f8f6ec2a9f5f841615df0fa7e017442",
    "jev_decision3_native": "07a12011f25246666551cb27892fed672a27068da477f2768048816c1478571f",
}
_PRIVATE_NAMES = {name: "s1._action_profiles_find_" + name for name in _ORIGINALS}
_WIRE_SOURCE = _ROOT / "scripts/jev_decision3_browser_wire.py"
_WIRE_SHA = "c048d7779e60fce5d1a56b38e26b9ddf5ed9e1ff9c6adc478786d0267cca3027"
_LOCK = threading.RLock()
_PROTOCOL = None
_WIRES = {}


@dataclass(frozen=True)
class ActionProtocol:
    native_adapter_class: type
    generated_adapter_class: type
    provenance: dict


def _profile(value):
    if type(value) is not str or value not in ACTION_PROFILES:
        raise ValueError("Unknown explicit browser action profile")
    return value


def _checked_source(path, expected):
    path = Path(path)
    try:
        if path.resolve(strict=True) != path or not path.is_file():
            raise ValueError("Ordinary pinned action-profile source required")
        with path.open("rb") as stream:
            raw = stream.read(200_001)
    except OSError as error:
        raise ValueError("Pinned action-profile source unavailable") from error
    if not raw or len(raw) > 200_000 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Pinned action-profile source changed")
    return raw


def _component(name):
    label, expected = _COMPONENTS[name]
    path = _ROOT / label
    module = types.ModuleType("_action_profile_component_" + name)
    module.__file__ = str(path)
    exec(
        compile(_checked_source(path, expected), str(path), "exec", dont_inherit=True), vars(module)
    )
    return module


def _original_sources():
    result = {}
    for name, expected in _ORIGINALS.items():
        _checked_source(_ROOT / "src/s1" / (name + ".py"), expected)
        result[name] = {"original_sha256": expected, "derived_sha256": expected}
    return result


def _dependency_pins(experimental):
    rows = [
        {"path": "src/s1/" + name + ".py", "sha256": expected}
        for name, expected in _ORIGINALS.items()
    ]
    rows.append({"path": "scripts/jev_decision3_browser_wire.py", "sha256": _WIRE_SHA})
    if experimental:
        rows.extend({"path": label, "sha256": expected} for label, expected in _COMPONENTS.values())
    for row in rows:
        _checked_source(_ROOT / row["path"], row["sha256"])
    return rows


def _private_protocol():
    global _PROTOCOL
    component = _component("protocol")
    # Check source identities even on cache reuse; never silently adopt changed sources.
    dependencies = _dependency_pins(True)
    if component.PINS != _ORIGINALS:
        raise ValueError("Private action-profile original source selection differs")
    if _PROTOCOL is not None:
        classes, provenance, modules = _PROTOCOL
        if any(sys.modules.get(name) is not module for name, module in modules.items()):
            raise ValueError("Private action-profile module registry changed")
        return classes, provenance
    if any(name in sys.modules for name in _PRIVATE_NAMES.values()):
        raise ValueError("Private action-profile namespace is already occupied")
    loaded, sources = {}, {}
    try:
        for name in ("jev_action_grammar", "jev_generated", "jev_decision3_native"):
            derived, evidence = component.derive(name)
            if hashlib.sha256(derived.encode()).hexdigest() != _DERIVED[name]:
                raise ValueError("Reviewed experimental derivative differs")
            public = derived
            bindings = []
            for dependency in ("jev_action_grammar", "jev_generated"):
                before = "from ._experimental_find_" + dependency + " import "
                after = "from ._action_profiles_find_" + dependency + " import "
                expected_count = int(
                    (name, dependency)
                    in {
                        ("jev_generated", "jev_action_grammar"),
                        ("jev_decision3_native", "jev_generated"),
                    }
                )
                if public.count(before) != expected_count or after in public:
                    raise ValueError("Private action-profile import seam differs")
                if expected_count:
                    public = public.replace(before, after, 1)
                    bindings.append((before, after))
            inverse = public
            for before, after in reversed(bindings):
                inverse = inverse.replace(after, before, 1)
            if inverse != derived:
                raise ValueError("Undeclared private action-profile change")
            fullname = _PRIVATE_NAMES[name]
            module = types.ModuleType(fullname)
            module.__package__ = "s1"
            module.__file__ = str(_ROOT / "src/s1" / (name + ".py")) + "::action-profile-find"
            sys.modules[fullname] = module
            loaded[fullname] = module
            exec(compile(public, module.__file__, "exec", dont_inherit=True), vars(module))
            sources[name] = {
                **evidence,
                "public_binding_sha256": hashlib.sha256(public.encode()).hexdigest(),
                "private_module": fullname,
                "import_binding_edits": len(bindings),
                "import_binding_inverse_exact": True,
            }
        classes = (
            loaded[_PRIVATE_NAMES["jev_decision3_native"]].JevDecision3NativeAdapter,
            loaded[_PRIVATE_NAMES["jev_generated"]].JevGeneratedActionAdapter,
        )
        provenance = {
            "action_profile": ACTION_PROFILES[1],
            "operation_count": 9,
            "implementation": "checked_private_derivatives",
            "native_profile": "jev_decision3_native_find_text_experimental_v1",
            "generated_policy": "greedy_legal_action_find_text_experimental_v1",
            "wire_profile": "jev_decision3_browser_find_text_experimental_v1",
            "source_dependencies": dependencies,
            "sources": sources,
            "components": {
                name: {"path": label, "sha256": expected, "size_bytes": _COMPONENT_SIZES[name]}
                for name, (label, expected) in _COMPONENTS.items()
            },
            "original_modules_mutated": False,
            "private_module_lifetime": "process; deferred imports retained",
            "no_model_quality_or_browser_success_qualification": True,
        }
        _PROTOCOL = classes, provenance, dict(loaded)
        return classes, provenance
    except BaseException:
        for fullname, module in loaded.items():
            if sys.modules.get(fullname) is module:
                del sys.modules[fullname]
        raise


def resolve_protocol(profile):
    """Return original classes or the checked nine-operation private classes.

    Each call receives its own provenance dictionary. Class/module identities are
    shared under a lock, never installed over the original s1 module names.
    """
    profile = _profile(profile)
    with _LOCK:
        if profile == DEFAULT_ACTION_PROFILE:
            native = importlib.import_module("s1.jev_decision3_native")
            generated = importlib.import_module("s1.jev_generated")
            provenance = {
                "action_profile": profile,
                "operation_count": 8,
                "implementation": "original",
                "native_profile": "jev_decision3_native_v1",
                "generated_policy": "greedy_legal_action_v2",
                "wire_profile": "jev_decision3_browser_named_answers_v1",
                "source_dependencies": _dependency_pins(False),
                "sources": _original_sources(),
                "original_modules_mutated": False,
            }
            return ActionProtocol(
                native.JevDecision3NativeAdapter, generated.JevGeneratedActionAdapter, provenance
            )
        classes, provenance = _private_protocol()
        return ActionProtocol(*classes, copy.deepcopy(provenance))


def make_agent_class(profile, agent_module, model_module, *, marker_expression):
    """Select after original benchmark request instrumentation has been installed."""
    if _profile(profile) == DEFAULT_ACTION_PROFILE:
        return agent_module.Agent
    if type(marker_expression) is not str or not marker_expression.strip():
        raise ValueError("Experimental action profile needs its actual browser marker expression")
    steps = getattr(agent_module, "MAX_STEPS", None)
    if type(steps) is not int or steps not in (16, 60):
        raise ValueError("Experimental client requires canonical60 or private16 action budget")
    # The experimental runner independently requires16. Only compiled private
    # command/choose globals get that budget; original Agent/module globals stay60.
    private_agent = types.ModuleType(getattr(agent_module, "__name__", "_action_profile_agent"))
    vars(private_agent).update(vars(agent_module))
    private_agent.MAX_STEPS = 16
    return _component("agent").make_agent_class(
        private_agent, model_module, marker_expression=marker_expression
    )


def named_browser_response(profile, request, response):
    """Change only the checked ordered answer container for the selected profile."""
    profile = _profile(profile)
    with _LOCK:
        _checked_source(_WIRE_SOURCE, _WIRE_SHA)
        if profile != DEFAULT_ACTION_PROFILE:
            label, expected = _COMPONENTS["wire"]
            _checked_source(_ROOT / label, expected)
        if profile not in _WIRES:
            if profile == DEFAULT_ACTION_PROFILE:
                wire = types.ModuleType("_action_profile_standard_wire")
                wire.__file__ = str(_WIRE_SOURCE)
                exec(
                    compile(
                        _checked_source(_WIRE_SOURCE, _WIRE_SHA),
                        str(_WIRE_SOURCE),
                        "exec",
                        dont_inherit=True,
                    ),
                    vars(wire),
                )
            else:
                wire, _ = _component("wire").load_wire()
            _WIRES[profile] = wire
        wire = _WIRES[profile]
    return wire.named_browser_response(request, response)
