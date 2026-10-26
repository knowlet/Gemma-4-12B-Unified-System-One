"""Explicit action-adapter prompt contracts; no framework or model loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

NATIVE_KIND = "jev_native_action_lora_v2"
NATIVE_FORMAT = "shared-elements-v2"
RAW_FORMAT = "raw-json-v1"
NATIVE_PREFIX = "<bos><|turn>user\n"
NATIVE_SUFFIX = "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"


def native_prompt(user_text):
    """The pinned Gemma native, generation-prefix-only nonthinking envelope."""
    if type(user_text) is not str:
        raise ValueError("Native user text must be a string")
    return NATIVE_PREFIX + user_text + NATIVE_SUFFIX


def _encode(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise ValueError("Prompt contract must be finite JSON") from exc


def prompt_contract():
    """Bind the requested renderer, its exact instructions and imported source bytes."""
    from . import jev_generated, jev_prompt

    sources = {}
    for label, module in (("s1/jev_generated.py", jev_generated), ("s1/jev_prompt.py", jev_prompt)):
        path = Path(module.__file__)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Prompt contract requires ordinary imported Python sources")
        blob = path.read_bytes()
        sources[label] = {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}
    return {
        "schema_id": "jev-native-action-prompt-v2",
        "requested_format": NATIVE_FORMAT,
        "effective_formats": [NATIVE_FORMAT, RAW_FORMAT],
        "selection_basis": "complete_user_text_utf8_bytes",
        "shared_selection": "strictly_smaller_than_raw; ties_use_raw",
        "roundtrip": "exact_complete_ordered_original_request_before_selection",
        "native_prefix": NATIVE_PREFIX,
        "native_suffix": NATIVE_SUFFIX,
        "instructions_sha256": hashlib.sha256(jev_generated.INSTRUCTIONS.encode()).hexdigest(),
        "shared_instructions_sha256": hashlib.sha256(
            jev_prompt.SHARED_ELEMENT_V2_INSTRUCTIONS.encode()
        ).hexdigest(),
        "sources": sources,
    }


def expected_format(metadata, requested_format):
    """Check renderer compatibility, not artifact authenticity or qualification.

    Callers pass strictly verified artifact metadata before weights, and the
    already attached guard metadata again before every generated request.
    Historical public diagnostic wrappers are explicitly raw-only.
    """
    if type(metadata) is not dict or type(requested_format) is not str:
        raise ValueError("Explicit verified action adapter prompt metadata/format required")
    kind = metadata.get("kind")
    if kind == NATIVE_KIND:
        if _encode(metadata.get("prompt_contract")) != _encode(prompt_contract()):
            raise ValueError("Native action adapter prompt contract differs from current renderer")
        expected = NATIVE_FORMAT
    elif kind in {"jev_action_lora_v1", "qualified_bf16_exposed_diagnostic_v1"}:
        expected = RAW_FORMAT
        if "prompt_contract" in metadata:
            raise ValueError("Historical raw action adapter must not declare a V2 contract")
    elif (
        set(metadata) == {"artifact", "qualification"}
        and type(metadata["artifact"]) is dict
        and metadata["artifact"].get("kind")
        in {"jev_public_native_action_lora_v1", "jev_public_native_action_cfg_lora_v1"}
        and type(metadata["qualification"]) is dict
    ):
        expected = RAW_FORMAT
        if (
            metadata["artifact"].get("prompt_format", RAW_FORMAT) != RAW_FORMAT
            or "prompt_contract" in metadata["artifact"]
        ):
            raise ValueError("Historical public action adapter requires its raw prompt")
    else:
        raise ValueError("Unknown attached action adapter prompt profile")
    if metadata.get("prompt_format", expected) != expected or requested_format != expected:
        raise ValueError("Requested generated-action prompt format differs from trained adapter")
    return expected
