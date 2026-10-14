"""Finite legal-action token grammar, with no model or tensor-library dependency.

The callable implements Transformers' ``prefix_allowed_tokens_fn(batch_id, ids)``
contract for one prompt (batch ID 0). It constrains decoding, not parsing: every
offered operation/target pair has one compact canonical JSON token path followed
by the caller's native EOS. No candidate is ranked, repaired, retried or pruned.

Default inventory limits match the validated Jev request: three target heads of
400 choices and five untargeted operations, at most 1205 actions. The 32-token
generation limit includes EOS; the 38560-token aggregate limit includes every
candidate's EOS. Exceeding any limit rejects the entire grammar.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

CONSTRAINT = "legal_action_token_trie_v1"
TARGET_HEADS = {"CLICK": "click_target", "TYPE_TEXT": "type_text_target", "SELECT": "select_target"}
UNTARGETED_OPERATIONS = {"SCROLL_DOWN", "SCROLL_UP", "WAIT", "DONE", "BLOCKED"}


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _tokens(value, name, *, maximum):
    if callable(getattr(value, "tolist", None)):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) > maximum:
        raise ValueError(f"{name} must be a bounded one-dimensional token sequence")
    return tuple(_integer(token, name) for token in value)


def _labels(labels):
    if not isinstance(labels, Mapping) or "operation" not in labels:
        raise ValueError("validated operation labels are required")
    copied = {}
    for head, choices in labels.items():
        if (
            head not in {"operation", *TARGET_HEADS.values()}
            or not isinstance(choices, Sequence)
            or isinstance(choices, (str, bytes))
            or not 1 <= len(choices) <= (8 if head == "operation" else 400)
            or any(
                not isinstance(choice, str) or not choice.strip() or len(choice) > 128
                for choice in choices
            )
            or len(set(choices)) != len(choices)
        ):
            raise ValueError("invalid, duplicate or over-limit legal-action labels")
        copied[head] = tuple(choices)
    if set(copied["operation"]) - (TARGET_HEADS.keys() | UNTARGETED_OPERATIONS):
        raise ValueError("unsupported legal operation")
    if any(
        (operation in copied["operation"]) != (head in copied)
        for operation, head in TARGET_HEADS.items()
    ):
        raise ValueError("offered operations and target heads must agree")
    return copied


class LegalActionTokenTrie:
    """A request-local, batch-size-one legal-action generation callback.

    ``tokenizer.encode(text, add_special_tokens=False)`` must return native token
    IDs. Payloads containing any advertised special ID (including BOS, PAD or any
    EOS) are rejected. ``prompt_length`` is the original input length, including
    any prompt special tokens. Input order is retained throughout the inventory.

    Pass this object as ``prefix_allowed_tokens_fn`` and then independently call
    ``assert_complete`` on the returned generated suffix, including its EOS.
    Unknown prefixes, batch IDs and already-ended sequences fail closed.
    """

    def __init__(
        self,
        labels,
        tokenizer,
        *,
        prompt_length,
        eos_token_id=106,
        max_new_tokens=32,
        max_legal_actions=1205,
        max_encoded_tokens=38560,
    ):
        self.prompt_length = _integer(prompt_length, "prompt_length", minimum=1)
        self.eos_token_id = _integer(eos_token_id, "eos_token_id")
        self.max_new_tokens = _integer(max_new_tokens, "max_new_tokens", minimum=1)
        self.max_legal_actions = _integer(max_legal_actions, "max_legal_actions", minimum=1)
        self.max_encoded_tokens = _integer(max_encoded_tokens, "max_encoded_tokens", minimum=1)
        if not callable(getattr(tokenizer, "encode", None)):
            raise ValueError("tokenizer.encode must be callable")
        labels = _labels(labels)
        pairs = [
            (operation, target)
            for operation in labels["operation"]
            for target in (
                labels[TARGET_HEADS[operation]] if operation in TARGET_HEADS else (None,)
            )
        ]
        if len(pairs) > self.max_legal_actions:
            raise ValueError("legal action count exceeds max_legal_actions; none pruned")
        special = {self.eos_token_id}
        for token in getattr(tokenizer, "all_special_ids", ()):
            special.add(_integer(token, "special_token_id"))
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            token = getattr(tokenizer, name, None)
            if token is not None:
                special.add(_integer(token, name))
        self._trie, self._complete = {}, {}
        canonical, encoded, total = [], [], 0
        for operation, target in pairs:
            text = json.dumps(
                {"operation": operation, "target": target},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            try:
                text.encode("utf-8")
            except UnicodeError as exc:
                raise ValueError("canonical legal action must be valid UTF-8") from exc
            payload = _tokens(
                tokenizer.encode(text, add_special_tokens=False),
                "encoded action",
                maximum=self.max_new_tokens - 1,
            )
            if not payload or special.intersection(payload):
                raise ValueError("canonical action encoding is empty or contains a special token")
            tokens = (*payload, self.eos_token_id)
            total += len(tokens)
            if total > self.max_encoded_tokens:
                raise ValueError("inventory exceeds max_encoded_tokens; none pruned")
            if tokens in self._complete:
                raise ValueError("distinct legal actions have identical token encodings")
            node = self._trie
            for token in payload:
                if self.eos_token_id in node:
                    raise ValueError("tokenizer produced conflicting complete-action prefixes")
                node = node.setdefault(token, {})
            if node:
                raise ValueError("tokenizer produced conflicting complete-action prefixes")
            node[self.eos_token_id] = {}
            self._complete[tokens] = text
            canonical.append(text)
            encoded.append(tokens)
        self.canonical_actions = tuple(canonical)
        self.encoded_actions = tuple(encoded)
        self._inventory_sha256 = hashlib.sha256(
            json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def __call__(self, batch_id, input_ids):
        if type(batch_id) is not int or batch_id != 0:
            raise ValueError("legal-action grammar supports only batch ID 0")
        tokens = _tokens(input_ids, "input_ids", maximum=self.prompt_length + self.max_new_tokens)
        if len(tokens) < self.prompt_length:
            raise ValueError("input_ids is shorter than the original prompt")
        node = self._trie
        for token in tokens[self.prompt_length :]:
            if token not in node:
                raise ValueError("generated token prefix is not a legal canonical action")
            node = node[token]
        if not node:
            raise ValueError("legal canonical action has already ended")
        return list(node)

    def assert_complete(self, generated_token_ids):
        """Return canonical JSON only for an exact legal token path plus native EOS."""
        tokens = _tokens(generated_token_ids, "generated_token_ids", maximum=self.max_new_tokens)
        if tokens not in self._complete:
            raise ValueError(
                "generated tokens are not one complete legal canonical action plus EOS"
            )
        return self._complete[tokens]

    def telemetry(self):
        lengths = [len(tokens) for tokens in self.encoded_actions]
        return {
            "constraint": CONSTRAINT,
            "legal_action_count": len(self.canonical_actions),
            "canonical_inventory_sha256": self._inventory_sha256,
            "encoded_token_lengths": lengths,
            "encoded_tokens_total": sum(lengths),
            "max_new_tokens": self.max_new_tokens,
            "max_legal_actions": self.max_legal_actions,
            "max_encoded_tokens": self.max_encoded_tokens,
            "eos_token_id": self.eos_token_id,
            "prompt_length": self.prompt_length,
            "token_lengths_include_eos": True,
        }
