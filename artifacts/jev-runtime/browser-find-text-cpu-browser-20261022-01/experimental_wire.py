"""Exact checked wire-container logic with explicit experimental profile IDs."""

import hashlib
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scripts/jev_decision3_browser_wire.py"
SOURCE_SHA = "c048d7779e60fce5d1a56b38e26b9ddf5ed9e1ff9c6adc478786d0267cca3027"
EDITS = (
    (
        'PROFILE = "jev_decision3_browser_named_answers_v1"',
        'PROFILE = "jev_decision3_browser_find_text_experimental_v1"',
    ),
    (
        'LEGACY_PROFILE = "jev_decision3_native_v1"',
        'LEGACY_PROFILE = "jev_decision3_native_find_text_experimental_v1"',
    ),
)


def load_wire():
    original = SOURCE.read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA:
        raise ValueError("Selected original wire changed")
    text = original.decode()
    for before, after in EDITS:
        if text.count(before) != 1:
            raise ValueError("Experimental wire seam is not unique")
        text = text.replace(before, after)
    inverse = text
    for before, after in reversed(EDITS):
        inverse = inverse.replace(after, before)
    if inverse.encode() != original:
        raise ValueError("Undeclared experimental wire change")
    module = types.ModuleType("_find_text_experimental_wire")
    exec(
        compile(text, str(SOURCE) + "::find-text-experimental", "exec", dont_inherit=True),
        vars(module),
    )
    return module, {
        "original_sha256": SOURCE_SHA,
        "derived_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "inverse_exact": True,
        "declared_edits": 2,
    }
