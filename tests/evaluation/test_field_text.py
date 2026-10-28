"""Field transport normalization never extracts or invents a field value."""

import json

import pytest

from s1.evaluation.field_text import parse_field_text


@pytest.mark.parametrize(
    "raw, language",
    [
        ('```json\n{"text": "Graph theory"}\n```', "json"),
        ('```\n{"text": "Graph theory"}\n```', None),
        (' \t\r\n```json \t\r\n{"text": "Graph theory"}\r\n \t``` \r\n', "json"),
        ('```\n\n {"text": "Graph theory"} \n\n```', None),
    ],
)
def test_whole_optional_json_fence_is_logged_without_changing_value(raw, language):
    value, normalization = parse_field_text(raw)
    assert value == "Graph theory"
    assert normalization == {
        "format": "fenced",
        "fence_removed": True,
        "fence_language": language,
    }


@pytest.mark.parametrize(
    "value",
    ["Graph theory", "  Graph theory\n", "台灣 🦊", "x" * 2000, '```json {"text": "data"} ```'],
)
@pytest.mark.parametrize("wrapper", ["direct", "fenced"])
def test_value_is_returned_unchanged_at_accepted_boundaries(value, wrapper):
    payload = json.dumps({"text": value}, ensure_ascii=False)
    raw = " \t\r\n" + payload + "\r\n " if wrapper == "direct" else f"```json\n{payload}\n```"
    actual, metadata = parse_field_text(raw)
    assert actual == value
    assert metadata["format"] == wrapper
    assert metadata["fence_removed"] is (wrapper == "fenced")


@pytest.mark.parametrize(
    "raw",
    [
        "",
        " \t\n",
        "_" * 128,
        "```json\n" + "_" * 127,
        'Here is the value: {"text":"value"}',
        '{"text":"value"} trailing prose',
        'Before\n```json\n{"text":"value"}\n```',
        '```json\n{"text":"value"}\n```\nAfter',
        '```json\n{"text":"value"}\n```\n```json\n{"text":"other"}\n```',
        '```json\n{"text":"value"}\n```\n{"text":"other"}',
        '{"text":"value"} {"text":"other"}',
        '```text\n{"text":"value"}\n```',
        '```JSON\n{"text":"value"}\n```',
        '```json extra\n{"text":"value"}\n```',
        '``` json\n{"text":"value"}\n```',
        '````json\n{"text":"value"}\n````',
        '```json {"text":"value"}```',
        '```json\n{"text":"value"}```',
        '```json\n{"text":"value"}\n```json',
        '```json\n{"text":"value"}\n```\n```',
        '~~~json\n{"text":"value"}\n~~~',
        '\u00a0{"text":"value"}',
        '\ufeff{"text":"value"}',
    ],
)
def test_wrapper_and_underscore_outputs_fail_without_brace_extraction(raw):
    with pytest.raises(ValueError):
        parse_field_text(raw)


@pytest.mark.parametrize(
    "payload",
    [
        "null",
        "[]",
        '[{"text":"value"}]',
        '"value"',
        "true",
        "1",
        "{}",
        '{"Text":"value"}',
        '{"text":"value","extra":1}',
        '{"text":"first","text":"second"}',
        '{"text":"first","\\u0074ext":"second"}',
        '{"text":NaN}',
        '{"text":Infinity}',
        '{"text":-Infinity}',
        '{"text":"value","extra":{"bad":NaN}}',
        '{"text":null}',
        '{"text":false}',
        '{"text":0}',
        '{"text":[]}',
        '{"text":{}}',
        '{"text":""}',
        '{"text":" \\t\\r\\n"}',
        '{"text":"\\u2003"}',
        json.dumps({"text": "x" * 2001}),
        '{"text":"value",}',
        "{'text':'value'}",
    ],
)
@pytest.mark.parametrize("fenced", [False, True])
def test_json_schema_and_nonfinite_or_duplicate_values_are_rejected(payload, fenced):
    raw = f"```json\n{payload}\n```" if fenced else payload
    with pytest.raises(ValueError):
        parse_field_text(raw)


@pytest.mark.parametrize("value", [None, False, 1, {}, [], b'{"text":"value"}'])
def test_nonstring_transport_is_rejected(value):
    with pytest.raises(ValueError, match="must be a string"):
        parse_field_text(value)
