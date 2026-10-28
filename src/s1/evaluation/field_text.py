"""Normalize a complete field-value JSON transport without repairing its content."""

from __future__ import annotations

import json
import re

_FENCED_JSON = re.compile(r"```(?P<language>json)?[ \t]*\r?\n(?P<body>.*)\r?\n[ \t]*```", re.DOTALL)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Field text JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Field text JSON contains a nonfinite constant: {value}")


def parse_field_text(raw_text: str) -> tuple[str, dict[str, str | bool | None]]:
    """Accept direct JSON or one whole optional-``json`` fenced JSON block.

    Only JSON whitespace may surround the complete transport. Fences need
    separate opening/closing lines; other info strings and outside prose are
    rejected. The decoded field value is returned without trimming or rewriting.
    """
    if not isinstance(raw_text, str):
        raise ValueError("Field text response must be a string")
    payload = raw_text
    normalization = {"format": "direct", "fence_removed": False, "fence_language": None}
    fence = _FENCED_JSON.fullmatch(raw_text.strip(" \t\r\n"))
    if fence is not None:
        payload = fence["body"]
        normalization = {
            "format": "fenced",
            "fence_removed": True,
            "fence_language": fence["language"],
        }
    parsed = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(parsed, dict) or set(parsed) != {"text"}:
        raise ValueError("Field text JSON must be exactly one object with only the text key")
    value = parsed["text"]
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValueError(
            "Field text JSON must contain a nonempty string of at most 2000 characters"
        )
    return value, normalization
