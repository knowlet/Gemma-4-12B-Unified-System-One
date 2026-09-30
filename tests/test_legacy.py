import random

import pytest

from s1.metrics import ece
from s1.schema import Example, Q, build, truncate_tokens


class Tokenizer:
    bos_token_id = 1

    def encode(self, text, **kwargs):
        return list(text.encode())


@pytest.mark.parametrize("budget", [0, 1, 8, 19, 20, 32, 100])
def test_truncation_never_exceeds_budget(budget):
    assert len(truncate_tokens(Tokenizer(), "a" * 1000, budget)) <= budget


def test_evaluation_never_uses_gold_to_select_options():
    example = Example("state", [Q("choose", [str(i) for i in range(60)], 59)])
    with pytest.raises(ValueError, match="gold-based"):
        build(example, Tokenizer(), train=False)
    item = build(example, Tokenizer(), rng=random.Random(1), train=True)
    assert 59 in item["perms"][0]
    assert len(set(item["perms"][0])) == 52
    example.qs[0].gold = -1
    assert len(set(build(example, Tokenizer(), rng=random.Random(1))["perms"][0])) == 52


def test_ece_includes_zero_confidence():
    assert ece([0.0, 1.0], [1, 1]) == 0.5
