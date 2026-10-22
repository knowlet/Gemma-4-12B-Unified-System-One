"""Checked private derivatives; original eight-operation modules remain unchanged."""

from __future__ import annotations

import hashlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PINS = {
    "jev_generated": "c356e0eb3c046cb0787d7be0c5960803006231e8bdb414de644ae2167e72d849",
    "jev_action_grammar": "399e06e28172c96004b774ba97c04100687340e6b1326d317dbdf10557da75d2",
    "jev_decision3_native": "67616bb70a8acdbf96b1d2e4c1407cf8cc2a24db51b5ca383e33b4d9a0488de9",
}
NAMES = {name: "s1._experimental_find_" + name for name in PINS}
EDITS = {
    "jev_generated": [
        (
            'POLICY = "greedy_legal_action_v2"',
            'POLICY = "greedy_legal_action_find_text_experimental_v1"',
        ),
        (
            'OPERATIONS = {*TARGETS, "SCROLL_DOWN", "SCROLL_UP", "WAIT", "DONE", "BLOCKED"}',
            'OPERATIONS = {*TARGETS, "SCROLL_DOWN", "SCROLL_UP", "WAIT", "DONE", "BLOCKED", "FIND_TEXT"}',
        ),
        (
            "from .jev_action_grammar import LegalActionTokenTrie",
            "from ._experimental_find_jev_action_grammar import LegalActionTokenTrie",
        ),
    ],
    "jev_action_grammar": [
        (
            'CONSTRAINT = "legal_action_token_trie_v1"',
            'CONSTRAINT = "legal_action_find_text_experimental_v1"',
        ),
        (
            'UNTARGETED_OPERATIONS = {"SCROLL_DOWN", "SCROLL_UP", "WAIT", "DONE", "BLOCKED"}',
            'UNTARGETED_OPERATIONS = {"SCROLL_DOWN", "SCROLL_UP", "WAIT", "DONE", "BLOCKED", "FIND_TEXT"}',
        ),
        ('(8 if head == "operation" else 400)', '(9 if head == "operation" else 400)'),
    ],
    "jev_decision3_native": [
        (
            "from .jev_generated import _request",
            "from ._experimental_find_jev_generated import _request",
        ),
        (
            'PROFILE = "jev_decision3_native_v1"',
            'PROFILE = "jev_decision3_native_find_text_experimental_v1"',
        ),
    ],
}


def derive(name):
    path = ROOT / "src/s1" / (name + ".py")
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != PINS[name]:
        raise ValueError("Selected original protocol source changed")
    text = original.decode()
    for before, after in EDITS[name]:
        if text.count(before) != 1:
            raise ValueError("Experimental protocol seam is not unique")
        text = text.replace(before, after)
    inverse = text
    for before, after in reversed(EDITS[name]):
        if inverse.count(after) != 1:
            raise ValueError("Experimental protocol inverse seam is not unique")
        inverse = inverse.replace(after, before)
    if inverse.encode() != original:
        raise ValueError("Experimental protocol changed an undeclared byte")
    return text, {
        "original_sha256": PINS[name],
        "derived_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "declared_edits": len(EDITS[name]),
        "inverse_exact": True,
    }


@contextmanager
def load_private_protocol():
    """Explicit isolated source component, not a qualified endpoint/bootstrap.

    Imports project DTO/support dependencies, no framework/model constructor.
    Private module names cannot collide or replace originals. Source is compiled
    directly; cached bytecode is not consulted for these three derivatives.
    """
    if any(name in sys.modules for name in NAMES.values()):
        raise ValueError("Experimental protocol is already active")
    loaded, evidence = {}, {}
    try:
        for name in ("jev_action_grammar", "jev_generated", "jev_decision3_native"):
            source, evidence[name] = derive(name)
            module = types.ModuleType(NAMES[name])
            module.__package__ = "s1"
            module.__file__ = str(ROOT / "src/s1" / (name + ".py")) + "::find-text-experimental"
            sys.modules[NAMES[name]] = module
            loaded[name] = module
            exec(compile(source, module.__file__, "exec", dont_inherit=True), vars(module))
        yield loaded, evidence
    finally:
        for name, module in loaded.items():
            if sys.modules.get(NAMES[name]) is module:
                del sys.modules[NAMES[name]]
