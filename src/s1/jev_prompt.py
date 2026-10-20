"""Explicit, reversible model-input representations for validated Jev requests.

This module does not validate or repair the Jev schema. Callers must first use
the ordinary request validator and retain its complete candidate inventory.
"""

from __future__ import annotations

import copy
import json
import math

PROMPT_FORMATS = ("raw-json-v1", "shared-elements-v1")
SHARED_FORMAT = PROMPT_FORMATS[1]
TARGET_HEADS = ("click_target", "type_text_target", "select_target")
CRITERIA_ALIASES = ("criteria", "options", "levels")
DESCRIPTION_FIELDS = frozenset(
    ("element", "current_value", "role", "checked", "selected", "expanded")
)
SHARED_ELEMENT_INSTRUCTIONS = (
    "The input uses shared-elements-v1. Read payload as the original request. "
    "Its null elements/criteria slots are supplied by the envelope sections. "
    "Element blocks give fields once and every literal row in order; constants "
    "supply fixed columns, so rows contain the other fields in order. "
    "A criteria entry [key,row,shape] references the zero-based element row and "
    "ordered description-field shape; [key,literal] keeps a literal description. "
    "For references, element is '['+key+'] '+label, current_value is value or '' "
    "when value is absent, and other fields copy the element field exactly. "
    "Every original candidate key is retained; return the exact offered key."
)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _json_value(value, depth=0):
    _require(depth <= 64, "JSON nesting exceeds 64 levels")
    if isinstance(value, dict):
        _require(all(isinstance(k, str) for k in value), "JSON object keys must be strings")
        for item in value.values():
            _json_value(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _json_value(item, depth + 1)
    else:
        _require(value is None or type(value) in (str, int, float, bool), "JSON values required")
        if type(value) is float:
            _require(math.isfinite(value), "Finite JSON numbers required")


def _wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _parse(encoded):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON object key")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("Finite JSON numbers required")

    _require(type(encoded) is str, "Encoded payload must be text")
    return json.loads(encoded, object_pairs_hook=pairs, parse_constant=constant)


def _tables(elements):
    blocks = []
    for element in elements:
        fields = list(element)
        if not blocks or blocks[-1]["fields"] != fields:
            blocks.append({"fields": fields, "rows": []})
        blocks[-1]["rows"].append([element[field] for field in fields])
    for index, block in enumerate(blocks):
        constants = {
            field: block["rows"][0][column]
            for column, field in enumerate(block["fields"])
            if all(
                _wire(row[column]) == _wire(block["rows"][0][column]) for row in block["rows"][1:]
            )
        }
        candidate = {
            "fields": block["fields"],
            "constants": constants,
            "rows": [
                [
                    value
                    for field, value in zip(block["fields"], row, strict=True)
                    if field not in constants
                ]
                for row in block["rows"]
            ],
        }
        if constants and len(_wire(candidate).encode()) < len(_wire(block).encode()):
            blocks[index] = candidate
    return blocks


def _elements(blocks):
    _require(type(blocks) is list, "Element blocks must be an ordered list")
    result = []
    for block in blocks:
        _require(
            type(block) is dict
            and set(block) in ({"fields", "rows"}, {"fields", "rows", "constants"}),
            "Exact element block",
        )
        fields, rows = block["fields"], block["rows"]
        _require(
            type(fields) is list
            and all(type(field) is str for field in fields)
            and len(set(fields)) == len(fields)
            and type(rows) is list
            and rows,
            "Ordered unique fields and nonempty rows required",
        )
        constants = block.get("constants", {})
        _require(
            type(constants) is dict and set(constants) <= set(fields), "Known constant columns"
        )
        variable = [field for field in fields if field not in constants]
        for row in rows:
            _require(type(row) is list and len(row) == len(variable), "Complete element row")
            values = dict(zip(variable, row, strict=True))
            result.append(
                {
                    field: constants[field] if field in constants else values[field]
                    for field in fields
                }
            )
    return result


def _description(fields, key, element):
    _require(
        element.get("index") == key and type(element.get("index")) is str,
        "Criterion reference must retain its exact element index",
    )
    result = {}
    for field in fields:
        if field == "element":
            _require(type(element.get("label")) is str, "Literal element label required")
            result[field] = "[" + key + "] " + element["label"]
        elif field == "current_value":
            result[field] = element.get("value", "")
        else:
            _require(field in element, "Referenced element field is absent")
            result[field] = element[field]
    return result


def _section(head, alias, criteria, elements, indices):
    shapes, entries, references = [], [], 0
    for key, literal in criteria.items():
        entry = [key, literal]
        ordinal = indices.get(key)
        if ordinal is not None and isinstance(literal, dict):
            fields = list(literal)
            if set(fields) <= DESCRIPTION_FIELDS:
                try:
                    rebuilt = _description(fields, key, elements[ordinal])
                except ValueError:
                    rebuilt = None
                if rebuilt is not None and _wire(rebuilt) == _wire(literal):
                    shape = shapes.index(fields) if fields in shapes else len(shapes)
                    reference = [key, ordinal, shape]
                    if len(_wire(reference).encode()) < len(_wire(entry).encode()):
                        if shape == len(shapes):
                            shapes.append(fields)
                        entry, references = reference, references + 1
        entries.append(entry)
    return {"head": head, "alias": alias, "shapes": shapes, "entries": entries}, references


def format_payload(payload, *, prompt_format):
    """Return encoded text and counts; the input is never mutated.

    RAW uses the existing JSON bytes exactly. SHARED is always an explicit
    envelope; unpack_payload decodes only that representation, never raw data.
    """
    _require(
        type(prompt_format) is str and prompt_format in PROMPT_FORMATS, "Unknown prompt format"
    )
    _json_value(payload)
    original = _wire(payload)
    original_bytes = len(original.encode("utf-8"))
    metadata = {
        "format": prompt_format,
        "original_bytes": original_bytes,
        "encoded_bytes": original_bytes,
        "element_rows": 0,
        "criterion_refs": 0,
        "literal_criteria": 0,
    }
    if prompt_format == PROMPT_FORMATS[0]:
        return original, metadata
    packed = copy.deepcopy(payload)
    state = packed.get("state") if isinstance(packed, dict) else None
    elements = state.get("elements") if isinstance(state, dict) else None
    blocks, sections = None, []
    if type(elements) is list and elements and all(isinstance(el, dict) for el in elements):
        blocks, indices = _tables(elements), {}
        for ordinal, element in enumerate(elements):
            key = element.get("index")
            if type(key) is str:
                indices[key] = ordinal if key not in indices else None
        state["elements"] = None
        metadata["element_rows"] = len(elements)
        questions = packed.get("questions")
        if isinstance(questions, dict):
            for head, question in questions.items():
                if head not in TARGET_HEADS or not isinstance(question, dict):
                    continue
                aliases = [key for key in CRITERIA_ALIASES if key in question]
                if len(aliases) != 1 or not isinstance(question[aliases[0]], dict):
                    continue
                alias, criteria = aliases[0], question[aliases[0]]
                section, references = _section(head, alias, criteria, elements, indices)
                metadata["criterion_refs"] += references
                metadata["literal_criteria"] += len(criteria) - references
                if references:
                    question[alias] = None
                    sections.append(section)
    envelope = {
        "format": SHARED_FORMAT,
        "payload": packed,
        "elements": blocks,
        "criteria": sections,
    }
    encoded = _wire(envelope)
    metadata["encoded_bytes"] = len(encoded.encode("utf-8"))
    _require(
        _wire(unpack_payload(encoded)) == original, "Prompt representation must roundtrip exactly"
    )
    return encoded, metadata


def unpack_payload(encoded):
    """Decode SHARED only; literal marker-like user objects are never interpreted."""
    envelope = _parse(encoded)
    _require(
        type(envelope) is dict
        and set(envelope) == {"format", "payload", "elements", "criteria"}
        and envelope["format"] == SHARED_FORMAT,
        "Exact shared-elements-v1 envelope required",
    )
    result, blocks, sections = envelope["payload"], envelope["elements"], envelope["criteria"]
    _require(type(sections) is list, "Ordered criterion sections required")
    elements = None
    if blocks is not None:
        state = result.get("state") if isinstance(result, dict) else None
        _require(
            isinstance(state, dict) and "elements" in state and state["elements"] is None,
            "Explicit null state.elements slot required",
        )
        elements = _elements(blocks)
        state["elements"] = elements
    seen = set()
    for section in sections:
        _require(
            type(section) is dict and set(section) == {"head", "alias", "shapes", "entries"},
            "Exact criterion section",
        )
        head, alias, shapes, entries = (section[k] for k in ("head", "alias", "shapes", "entries"))
        _require(
            type(head) is str
            and head in TARGET_HEADS
            and head not in seen
            and type(alias) is str
            and alias in CRITERIA_ALIASES
            and elements is not None
            and type(shapes) is list
            and type(entries) is list,
            "Known unique head, alias and element table required",
        )
        seen.add(head)
        for fields in shapes:
            _require(
                type(fields) is list
                and all(type(field) is str for field in fields)
                and len(fields) == len(set(fields))
                and set(fields) <= DESCRIPTION_FIELDS,
                "Exact ordered description-field shape",
            )
        questions = result.get("questions") if isinstance(result, dict) else None
        question = questions.get(head) if isinstance(questions, dict) else None
        _require(
            isinstance(question, dict) and alias in question and question[alias] is None,
            "Explicit null target criteria slot required",
        )
        criteria = {}
        for entry in entries:
            _require(type(entry) is list and len(entry) in (2, 3), "Complete criterion entry")
            key = entry[0]
            _require(
                type(key) is str and key not in criteria, "Unique original criterion key required"
            )
            if len(entry) == 2:
                criteria[key] = entry[1]
            else:
                ordinal, shape = entry[1:]
                _require(
                    type(ordinal) is int
                    and 0 <= ordinal < len(elements)
                    and type(shape) is int
                    and 0 <= shape < len(shapes),
                    "Bounded element and shape references required",
                )
                criteria[key] = _description(shapes[shape], key, elements[ordinal])
        question[alias] = criteria
    _json_value(result)
    return result
