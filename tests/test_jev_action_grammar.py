"""CPU-only checks that legal-action constraints admit exactly the offered actions."""

import hashlib
import json
import re

import pytest

from s1.jev_action_grammar import CONSTRAINT, LegalActionTokenTrie


class Tokenizer:
    """Small deterministic tokenizer with shared target-digit prefixes."""

    all_special_ids = [1, 2, 106]
    bos_token_id, pad_token_id, eos_token_id = all_special_ids

    def __init__(self):
        self.ids = {}
        self.pieces = {}

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        result = []
        for piece in re.findall(
            r'"operation"|"target"|TYPE_TEXT|SCROLL_DOWN|SCROLL_UP|CLICK|SELECT|WAIT|DONE|BLOCKED|.',
            text,
            flags=re.DOTALL,
        ):
            if piece not in self.ids:
                token = 1000 + len(self.ids)
                self.ids[piece] = token
                self.pieces[token] = piece
            result.append(self.ids[piece])
        return result

    def decode(self, tokens):
        return "".join(self.pieces[token] for token in tokens)


class TensorLike:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


def build(labels=None, tokenizer=None, **kwargs):
    return LegalActionTokenTrie(
        {"operation": ["CLICK"], "click_target": ["2", "20"]} if labels is None else labels,
        tokenizer or Tokenizer(),
        prompt_length=3,
        **kwargs,
    )


def enumerate_paths(grammar):
    """Traverse every allowed continuation; no model, expectations or logits involved."""
    pending, completed = [()], set()
    while pending:
        prefix = pending.pop()
        for token in grammar(0, [1, 999, 2, *prefix]):
            path = (*prefix, token)
            if token == grammar.eos_token_id:
                grammar.assert_complete(path)
                completed.add(path)
            else:
                assert len(path) < grammar.max_new_tokens
                pending.append(path)
    return completed


def test_prefix_collisions_retain_targets_2_and_20_and_input_order():
    tokenizer = Tokenizer()
    grammar = build({"operation": ["CLICK"], "click_target": ["20", "2"]}, tokenizer)
    prefix = tokenizer.encode('{"operation":"CLICK","target":"2', add_special_tokens=False)
    assert grammar(0, TensorLike([1, 999, 2, *prefix])) == [tokenizer.ids["0"], tokenizer.ids['"']]
    assert [json.loads(text)["target"] for text in grammar.canonical_actions] == ["20", "2"]
    assert enumerate_paths(grammar) == set(grammar.encoded_actions)


def test_all_eight_operations_exact_targets_null_and_canonical_json():
    labels = {
        "operation": [
            "SELECT",
            "CLICK",
            "TYPE_TEXT",
            "SCROLL_DOWN",
            "SCROLL_UP",
            "WAIT",
            "DONE",
            "BLOCKED",
        ],
        "click_target": ["2", "20"],
        "type_text_target": ["search"],
        "select_target": ["1:2"],
    }
    tokenizer = Tokenizer()
    grammar = build(labels, tokenizer)
    expected = [("SELECT", "1:2"), ("CLICK", "2"), ("CLICK", "20"), ("TYPE_TEXT", "search")]
    expected += [(operation, None) for operation in labels["operation"][3:]]
    assert [tuple(json.loads(text).values()) for text in grammar.canonical_actions] == expected
    assert len(enumerate_paths(grammar)) == 9
    for text, tokens in zip(grammar.canonical_actions, grammar.encoded_actions):
        assert text == json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":"))
        assert tokenizer.decode(tokens[:-1]) == text
        assert grammar.assert_complete(TensorLike(list(tokens))) == text
        assert grammar(0, [1, 999, 2, *tokens[:-1]]) == [106]


def test_escaping_and_unicode_are_json_values_not_action_structure():
    targets = ['a"b', "back\\slash", "line\nnext", "雪", '}:null,"operation":"DONE']
    grammar = build({"operation": ["CLICK"], "click_target": targets}, max_new_tokens=64)
    assert [json.loads(text)["target"] for text in grammar.canonical_actions] == targets
    assert "雪" in grammar.canonical_actions[3]
    assert enumerate_paths(grammar) == set(grammar.encoded_actions)


@pytest.mark.parametrize("batch", [1, -1, True, "0"])
def test_wrong_batch_is_rejected(batch):
    with pytest.raises(ValueError, match="batch ID 0"):
        build()(batch, [1, 999, 2])


def test_unknown_incomplete_and_ended_paths_fail_closed():
    grammar = build()
    complete = grammar.encoded_actions[0]
    for ids in ([1, 999], [1, 999, 2, 106], [1, 999, 2, 999999], [1, 999, 2, *complete]):
        with pytest.raises(ValueError):
            grammar(0, ids)
    for ids in (complete[:-1], complete[:-2] + (106,), complete + (106,), [106], []):
        with pytest.raises(ValueError, match="complete legal canonical action"):
            grammar.assert_complete(ids)
    with pytest.raises(ValueError):
        grammar(0, [[1, 999, 2]])
    allowed = grammar(0, [1, 999, 2])
    allowed.append(999999)
    assert 999999 not in grammar(0, [1, 999, 2])


@pytest.mark.parametrize("special", [1, 2, 106, 117])
def test_special_bos_pad_eos_and_other_special_ids_cannot_appear_in_payload(special):
    tokenizer = Tokenizer()
    tokenizer.all_special_ids = [special]
    tokenizer.encode = lambda text, **kwargs: [1000, special, 1001]
    with pytest.raises(ValueError, match="special token"):
        build(tokenizer=tokenizer)


def test_native_eos_is_parameterized_and_only_allowed_after_complete_payload():
    grammar = build(eos_token_id=777)
    tokens = grammar.encoded_actions[0]
    assert tokens[-1] == 777
    assert grammar(0, [1, 999, 2, *tokens[:-1]]) == [777]
    assert 777 not in grammar(0, [1, 999, 2])


def test_32_token_limit_includes_eos_and_never_prunes_a_longer_action():
    tokenizer = Tokenizer()
    tokenizer.encode = lambda text, **kwargs: list(range(1000, 1031))
    grammar = build({"operation": ["DONE"]}, tokenizer)
    assert len(grammar.encoded_actions[0]) == 32
    tokenizer.encode = lambda text, **kwargs: list(range(1000, 1032))
    with pytest.raises(ValueError, match="bounded"):
        build({"operation": ["DONE"]}, tokenizer)
    tokenizer.encode = lambda text, **kwargs: list(range(1000, 1032 if '"20"' in text else 1031))
    with pytest.raises(ValueError, match="bounded"):
        build(tokenizer=tokenizer)


def test_complete_maximum_inventory_and_no_silent_candidate_pruning():
    labels = {
        "operation": [
            "CLICK",
            "TYPE_TEXT",
            "SELECT",
            "SCROLL_DOWN",
            "SCROLL_UP",
            "WAIT",
            "DONE",
            "BLOCKED",
        ],
        "click_target": [str(index) for index in range(400)],
        "type_text_target": [str(index) for index in range(400)],
        "select_target": [f"{index}:2" for index in range(400)],
    }
    grammar = build(labels)
    assert len(grammar.canonical_actions) == len(enumerate_paths(grammar)) == 1205
    assert json.loads(grammar.canonical_actions[1199]) == {"operation": "SELECT", "target": "399:2"}
    with pytest.raises(ValueError, match="max_legal_actions; none pruned"):
        build(labels, max_legal_actions=1204)
    total = grammar.telemetry()["encoded_tokens_total"]
    with pytest.raises(ValueError, match="max_encoded_tokens; none pruned"):
        build(labels, max_encoded_tokens=total - 1)
    assert len(build(labels, max_encoded_tokens=total).canonical_actions) == 1205


def test_telemetry_binds_ordered_canonical_inventory_and_every_encoded_length():
    grammar = build()
    telemetry = grammar.telemetry()
    expected_hash = hashlib.sha256(
        json.dumps(
            list(grammar.canonical_actions), ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    assert telemetry["constraint"] == CONSTRAINT
    assert telemetry["canonical_inventory_sha256"] == expected_hash
    assert telemetry["legal_action_count"] == 2
    assert telemetry["encoded_token_lengths"] == [len(tokens) for tokens in grammar.encoded_actions]
    assert telemetry["token_lengths_include_eos"] is True
    assert (
        build({"operation": ["CLICK"], "click_target": ["20", "2"]}).telemetry()[
            "canonical_inventory_sha256"
        ]
        != expected_hash
    )


@pytest.mark.parametrize(
    "labels",
    [
        {},
        {"operation": []},
        {"operation": ["UNKNOWN"]},
        {"operation": ["CLICK"]},
        {"operation": ["DONE"], "click_target": ["1"]},
        {"operation": ["DONE", "DONE"]},
        {"operation": ["CLICK"], "click_target": ["1", "1"]},
        {"operation": ["CLICK"], "click_target": [" "]},
        {"operation": ["CLICK"], "click_target": ["x" * 129]},
        {"operation": ["CLICK"], "click_target": [str(index) for index in range(401)]},
    ],
)
def test_invalid_label_inventory_is_rejected_as_a_whole(labels):
    with pytest.raises(ValueError):
        build(labels)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt_length": 0},
        {"eos_token_id": True},
        {"eos_token_id": -1},
        {"max_new_tokens": 0},
        {"max_legal_actions": False},
        {"max_encoded_tokens": 0},
    ],
)
def test_invalid_numeric_bounds_are_rejected(kwargs):
    values = {"prompt_length": 3, **kwargs}
    with pytest.raises(ValueError):
        LegalActionTokenTrie({"operation": ["DONE"]}, Tokenizer(), **values)


def test_ambiguous_or_conflicting_tokenizer_paths_are_rejected_without_pruning():
    tokenizer = Tokenizer()
    tokenizer.encode = lambda text, **kwargs: [1000]
    with pytest.raises(ValueError, match="identical token encodings"):
        build(tokenizer=tokenizer)
    for lengths in ((1, 2), (2, 1)):
        paths = iter([list(range(1000, 1000 + count)) for count in lengths])
        tokenizer.encode = lambda text, **kwargs: next(paths)
        with pytest.raises(ValueError, match="conflicting complete-action prefixes"):
            build(tokenizer=tokenizer)
