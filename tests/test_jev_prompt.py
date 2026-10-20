"""Exact reversible representation tests; no model or processor is imported."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from s1.jev_prompt import (
    PROMPT_FORMATS,
    SHARED_ELEMENT_INSTRUCTIONS,
    format_payload,
    unpack_payload,
)


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def description(element, key, **extra):
    return {
        "element": "[" + key + "] " + element["label"],
        "current_value": element.get("value", ""),
        **{k: element[k] for k in ("role", "checked", "selected", "expanded") if k in element},
        **extra,
    }


@pytest.fixture
def payload():
    elements = [
        {
            "role": "checkbox",
            "value": "on",
            "checked": False,
            "index": "7",
            "label": "繁體中文",
            "operations": ["CLICK"],
            "unknown": {"zero": -0.0},
        },
        {
            "index": "3",
            "label": "Query",
            "value": "ready",
            "role": "textbox",
            "operations": ["TYPE_TEXT", "CLICK"],
        },
        {
            "index": "2",
            "label": "Department",
            "value": "Design",
            "role": "combobox",
            "options": [{"index": "2:1", "label": "Department → Design", "value": "design"}],
        },
    ]
    return {
        "before_state": {
            "format": "shared-elements-v1",
            "payload": None,
            "elements": None,
            "criteria": [],
        },
        "state": {
            "unknown": [None, False, "false", 1, True, 1.0, -0.0],
            "elements": elements,
            "page": {"text": "literal [key,row,shape] and null"},
            "recent_actions": [],
        },
        "questions": {
            "operation": {
                "criteria": {"CLICK": "Click", "TYPE_TEXT": "Fill", "SELECT": "Select"},
                "instructions": {"goal": "Literal goal"},
            },
            "click_target": {
                "criteria": {
                    "7": description(elements[0], "7"),
                    "3": {"element": "[3] Open Query", "current_value": "ready"},
                    "unknown": {"element": "different", "future": {"entries": ["not a ref"]}},
                },
                "unknown": True,
            },
            "type_text_target": {"options": {"3": description(elements[1], "3")}},
            "select_target": {
                "levels": {
                    "2:1": {
                        "element": "[2:1] Department → Design",
                        "current_value": "Design",
                        "selected": None,
                    }
                }
            },
        },
        "model": "fixture",
    }


@pytest.mark.parametrize("mode", PROMPT_FORMATS)
def test_exact_order_types_literals_unknown_fields_and_all_heads(payload, mode):
    before = wire(payload)
    encoded, metadata = format_payload(payload, prompt_format=mode)
    decoded = json.loads(encoded) if mode == PROMPT_FORMATS[0] else unpack_payload(encoded)
    assert wire(decoded) == before == wire(payload)
    assert metadata["format"] == mode
    assert metadata["original_bytes"] == len(before.encode())
    assert metadata["encoded_bytes"] == len(encoded.encode())
    if mode == PROMPT_FORMATS[1]:
        assert metadata["criterion_refs"] == 2
        assert metadata["literal_criteria"] == 3
        envelope = json.loads(encoded)
        assert (
            envelope["payload"]["questions"]["select_target"]
            == payload["questions"]["select_target"]
        )
        assert [section["head"] for section in envelope["criteria"]] == [
            "click_target",
            "type_text_target",
        ]


def test_raw_is_existing_json_bytes_and_shared_decoder_is_explicit(payload):
    encoded, _ = format_payload(payload, prompt_format="raw-json-v1")
    assert encoded == wire(payload)
    with pytest.raises(ValueError, match="envelope"):
        unpack_payload(encoded)


def test_literal_envelope_like_userdata_is_not_recursively_decoded():
    marker = {
        "format": "shared-elements-v1",
        "payload": {"state": {"elements": None}},
        "elements": [],
        "criteria": [],
    }
    payload = {"state": {"elements": [marker]}, "unknown": marker}
    encoded, _ = format_payload(payload, prompt_format="shared-elements-v1")
    assert wire(unpack_payload(encoded)) == wire(payload)
    assert unpack_payload(encoded)["unknown"] == marker


@pytest.mark.parametrize(
    "elements", [[], None, [None, "literal", {"index": "1"}], {"not": "a table"}]
)
def test_non_table_elements_remain_literal(elements):
    payload = {
        "state": {"elements": elements},
        "questions": {"click_target": {"criteria": {"1": "literal"}}},
    }
    encoded, metadata = format_payload(payload, prompt_format="shared-elements-v1")
    assert wire(unpack_payload(encoded)) == wire(payload)
    assert metadata["criterion_refs"] == 0


def test_duplicate_element_indices_do_not_choose_an_arbitrary_source():
    elements = [{"index": "1", "label": "same"}, {"index": "1", "label": "same"}]
    payload = {
        "state": {"elements": elements},
        "questions": {"click_target": {"criteria": {"1": description(elements[0], "1")}}},
    }
    encoded, metadata = format_payload(payload, prompt_format="shared-elements-v1")
    assert metadata["criterion_refs"] == 0 and wire(unpack_payload(encoded)) == wire(payload)


def test_constant_columns_preserve_numeric_types_missing_and_optional_fields():
    rows = [
        {
            "index": str(i),
            "label": "label" + str(i),
            "role": "link",
            "value": "",
            "operations": ["CLICK"],
            "unknown": {"v": False},
        }
        for i in range(40)
    ]
    rows += [{"index": "40", "label": "null", "role": "link", "value": None, "unknown": {"v": 0}}]
    payload = {"state": {"elements": rows}}
    encoded, _ = format_payload(payload, prompt_format="shared-elements-v1")
    blocks = json.loads(encoded)["elements"]
    assert any(block.get("constants", {}).get("role") == "link" for block in blocks)
    assert wire(unpack_payload(encoded)) == wire(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra-envelope",
        "non-null-slot",
        "duplicate-head",
        "bad-ordinal",
        "bool-shape",
        "duplicate-key",
        "unknown-shape",
        "bad-row",
        "unknown-constant",
    ],
)
def test_decoder_refuses_ambiguous_or_corrupt_explicit_substitutions(payload, mutation):
    encoded, _ = format_payload(payload, prompt_format="shared-elements-v1")
    envelope = json.loads(encoded)
    if mutation == "extra-envelope":
        envelope["unreviewed"] = True
    elif mutation == "non-null-slot":
        envelope["payload"]["state"]["elements"] = []
    elif mutation == "duplicate-head":
        envelope["criteria"].append(copy.deepcopy(envelope["criteria"][0]))
    elif mutation == "bad-ordinal":
        envelope["criteria"][0]["entries"][0][1] = -1
    elif mutation == "bool-shape":
        envelope["criteria"][0]["entries"][0][2] = False
    elif mutation == "duplicate-key":
        envelope["criteria"][0]["entries"].append(
            copy.deepcopy(envelope["criteria"][0]["entries"][0])
        )
    elif mutation == "unknown-shape":
        envelope["criteria"][0]["shapes"][0].append("unknown")
    elif mutation == "bad-row":
        envelope["elements"][0]["rows"][0].append("extra")
    else:
        envelope["elements"][0]["constants"] = {"unknown-column": True}
    with pytest.raises(ValueError):
        unpack_payload(wire(envelope))


def test_duplicate_keys_and_nonfinite_values_are_not_silently_normalized():
    for value in (float("nan"), float("inf"), ("tuple",), {1: "numeric key"}):
        with pytest.raises(ValueError):
            format_payload({"state": value}, prompt_format="shared-elements-v1")
    for encoded in (
        '{"format":"shared-elements-v1","format":"shared-elements-v1"}',
        '{"value":NaN}',
    ):
        with pytest.raises(ValueError):
            unpack_payload(encoded)


def test_v1_bytes_and_instructions_remain_exactly_the_pre_v2_fixture(payload):
    encoded, metadata = format_payload(payload, prompt_format="shared-elements-v1")
    assert metadata["format"] == "shared-elements-v1" and len(encoded.encode()) == 1465
    assert (
        hashlib.sha256(encoded.encode()).hexdigest()
        == "2aed5d1358fb97eedc3f6b6d933e4fe4940ab59fa2aa355c365a18ff691351b8"
    )
    assert (
        hashlib.sha256(SHARED_ELEMENT_INSTRUCTIONS.encode()).hexdigest()
        == "7bd25ee2b6cc9bd118e4e83a773b84f8f2fee1564ffaa0adafcac47d00bc5377"
    )


def shape_payload(checked_rows):
    elements = [
        {
            "index": str(i),
            "label": "literal label" + str(i),
            "value": "",
            "role": "checkbox",
            **({"checked": False} if i in checked_rows else {}),
        }
        for i in range(4)
    ]
    return {
        "state": {"elements": elements},
        "questions": {
            "click_target": {
                "criteria": {el["index"]: description(el, el["index"]) for el in elements}
            }
        },
    }


@pytest.mark.parametrize("checked_rows,default", [({0}, 1), ({0, 3}, 0)])
def test_v2_uses_most_frequent_shape_and_lowest_index_tie(checked_rows, default):
    payload = shape_payload(checked_rows)
    encoded, metadata = format_payload(payload, prompt_format="shared-elements-v2")
    section = json.loads(encoded)["criteria"][0]
    assert section["default_shape"] == default
    assert all(type(entry) is int or len(entry) == 2 for entry in section["entries"])
    if checked_rows == {0}:
        assert section["entries"] == [[0, 0], 1, 2, 3]
    assert metadata["format"] == "shared-elements-v2" and metadata["criterion_refs"] == 4
    assert wire(unpack_payload(encoded)) == wire(payload)


def test_v2_preserves_literal_overrides_and_marker_like_user_values(payload):
    encoded, metadata = format_payload(payload, prompt_format="shared-elements-v2")
    section = json.loads(encoded)["criteria"][0]
    assert section["entries"][0] == 0
    assert section["entries"][1] == ["3", payload["questions"]["click_target"]["criteria"]["3"]]
    assert section["entries"][2] == [
        "unknown",
        payload["questions"]["click_target"]["criteria"]["unknown"],
    ]
    assert wire(unpack_payload(encoded)) == wire(payload)
    assert metadata["criterion_refs"] == 2 and metadata["literal_criteria"] == 3


@pytest.mark.parametrize(
    "mutation",
    [
        "bool-ordinal",
        "negative",
        "large",
        "bool-shape",
        "nonstring-index",
        "missing-index",
        "duplicate-key",
        "bad-default",
        "explicit-default",
        "v1-triple",
        "extra-section",
    ],
)
def test_v2_rejects_malformed_or_ambiguous_references(payload, mutation):
    encoded, _ = format_payload(payload, prompt_format="shared-elements-v2")
    envelope = json.loads(encoded)
    section = envelope["criteria"][0]
    if mutation == "bool-ordinal":
        section["entries"][0] = True
    elif mutation == "negative":
        section["entries"][0] = -1
    elif mutation == "large":
        section["entries"][0] = 10000
    elif mutation == "bool-shape":
        section["entries"][0] = [0, False]
    elif mutation == "nonstring-index":
        block = envelope["elements"][0]
        block["rows"][0][block["fields"].index("index")] = 7
    elif mutation == "missing-index":
        block = envelope["elements"][0]
        block["fields"][block["fields"].index("index")] = "unknown-index"
    elif mutation == "duplicate-key":
        section["entries"].append(0)
    elif mutation == "bad-default":
        section["default_shape"] = False
    elif mutation == "explicit-default":
        section["entries"][0] = [0, section["default_shape"]]
    elif mutation == "v1-triple":
        section["entries"][0] = ["7", 0, 0]
    else:
        section["unknown"] = True
    with pytest.raises(ValueError):
        unpack_payload(wire(envelope))


def test_v2_rejects_a_nonmodal_default_even_when_references_can_be_decoded():
    encoded, _ = format_payload(shape_payload({0}), prompt_format="shared-elements-v2")
    envelope = json.loads(encoded)
    section = envelope["criteria"][0]
    section["default_shape"] = 0
    section["entries"] = [0, [1, 1], [2, 1], [3, 1]]
    with pytest.raises(ValueError, match="most frequent"):
        unpack_payload(wire(envelope))


def test_local_full_curl_regression_if_saved_evidence_is_available():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/jev-runtime/public-revisit-capacity-3g-20261020-01/actual-01/worker/payloads/51.json"
    )
    if not path.exists():
        pytest.skip("Optional local saved curl capacity evidence is not distributed")
    before = path.read_bytes()
    payload = json.loads(before)
    assert len(before) == 49334 and len(payload["questions"]["click_target"]["criteria"]) == 250
    encoded, metadata = format_payload(payload, prompt_format="shared-elements-v1")
    assert wire(unpack_payload(encoded)).encode() == before
    assert metadata["criterion_refs"] == 250 and metadata["element_rows"] == 250
    assert len(encoded.encode()) == 20741
    assert (
        hashlib.sha256(encoded.encode()).hexdigest()
        == "0eace5e6e78de6f08cd763db19df96f45f4f829711f855ffce3d826c75c3dc29"
    )
