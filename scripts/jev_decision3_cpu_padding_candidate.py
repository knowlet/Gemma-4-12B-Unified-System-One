"""Held text-only Decision3 CPU batching experiment over a prepared request.

The caller owns and authenticates the model, source, exclusive runtime, and
preflight. No model, files, questions, or scorer are loaded here. Modes change
native batch composition and may change floating-point answers. No result is a
claim of equivalence, latency, provenance, or real-world task success.
"""

from __future__ import annotations

import dataclasses
import json
import math
import threading

SOURCE_ONLY = True
_LOCK = threading.Lock()
_MAX_QUESTIONS = 64
_MAX_TOKENS = 16384
_MAX_OPTIONS = 255
_NATIVE_BATCH = 8
_PAIR_BATCH = 2
_MODES = {"whole", "ordered_pairs", "length_pairs"}
_MAX_METADATA_NODES = 65536
_MAX_METADATA_BYTES = 4 * 1024**2
_MAX_METADATA_DEPTH = 64
_MAX_INTEGER_BITS = 16384


def _need(value, message):
    if not value:
        raise ValueError(message)


def _freeze_metadata(value, budget, active, depth=0):
    """Preserve types and mapping order while bounding work before copying.

    This intentionally accepts only built-in JSON-like values plus tuples.
    It distinguishes booleans, integers, floats (including signed zero),
    lists and tuples. Shared acyclic values are fine; cycles are rejected.
    """
    _need(depth <= _MAX_METADATA_DEPTH, "bounded metadata nesting required")
    budget[0] += 1
    _need(budget[0] <= _MAX_METADATA_NODES, "bounded metadata nodes required")

    def charge(amount):
        budget[1] += amount
        _need(budget[1] <= _MAX_METADATA_BYTES, "bounded metadata bytes required")

    kind = type(value)
    if value is None:
        charge(1)
        return ("none",)
    if kind is bool:
        charge(1)
        return ("bool", value)
    if kind is int:
        bits = value.bit_length()
        _need(bits <= _MAX_INTEGER_BITS, "bounded metadata integer required")
        charge(1 + (bits + 7) // 8)
        return ("int", value)
    if kind is float:
        _need(math.isfinite(value), "finite metadata float required")
        charge(8)
        return ("float", value.hex())
    if kind is str:
        _need(len(value) <= _MAX_METADATA_BYTES - budget[1], "bounded metadata bytes required")
        if value.isascii():
            charge(len(value))
            return ("str", value)
        for char in value:
            code = ord(char)
            _need(not 0xD800 <= code <= 0xDFFF, "valid Unicode metadata required")
            charge(1 if code < 0x80 else 2 if code < 0x800 else 3 if code < 0x10000 else 4)
        return ("str", value)
    _need(kind in (list, tuple, dict), "supported metadata type required")
    _need(id(value) not in active, "acyclic metadata required")
    # Each child consumes at least one node; check before creating a copy.
    children = len(value) * (2 if kind is dict else 1)
    _need(children <= _MAX_METADATA_NODES - budget[0], "bounded metadata nodes required")
    charge(1)
    active.add(id(value))
    try:
        if kind is dict:
            items = tuple(
                (
                    _freeze_metadata(key, budget, active, depth + 1),
                    _freeze_metadata(item, budget, active, depth + 1),
                )
                for key, item in value.items()
            )
            return ("dict", items)
        items = tuple(_freeze_metadata(item, budget, active, depth + 1) for item in value)
        return ("list" if kind is list else "tuple", items)
    finally:
        active.remove(id(value))


def _snapshot(runtime, prepared):
    _need(
        dataclasses.is_dataclass(prepared)
        and not isinstance(prepared, type)
        and str(getattr(runtime, "device", "")) == "cpu"
        and type(getattr(runtime, "batch_size", None)) is int
        and runtime.batch_size == _NATIVE_BATCH
        and getattr(runtime, "permutation_average", None) is False
        and getattr(runtime, "fast", None) is None
        and type(getattr(runtime, "model_name", None)) is str
        and 0 < len(runtime.model_name) <= 4096
        and bool(runtime.model_name.strip())
        and callable(getattr(runtime, "run", None))
        and callable(getattr(runtime, "respond", None)),
        "exclusive CPU reference runtime with batch size 8 required",
    )
    _need(
        type(prepared.keys) is list
        and 1 <= len(prepared.keys) <= _MAX_QUESTIONS
        and all(type(key) is str and key for key in prepared.keys)
        and type(prepared.sequences) is dict
        and type(prepared.questions) is dict
        and len(prepared.sequences) == len(prepared.questions) == len(prepared.keys)
        and all(type(key) is str for key in prepared.sequences)
        and all(type(key) is str for key in prepared.questions)
        and not prepared.errors
        and not prepared.images
        and not prepared.videos
        and not prepared.media
        and not prepared.video_texts
        and not prepared.texts
        and not prepared.lengths
        and not prepared.reversed_sequences
        and not prepared.reversed_texts
        and not prepared.reversed_lengths,
        "complete text-only prepared request without reversed inputs required",
    )
    rows = []
    metadata_budget = [0, 0]
    # Bound identifier contents before hashing them in the uniqueness checks.
    _freeze_metadata(prepared.keys, metadata_budget, set())
    _need(
        len(set(prepared.keys)) == len(prepared.keys)
        and set(prepared.sequences) == set(prepared.questions) == set(prepared.keys),
        "complete unique prepared question identifiers required",
    )
    for key in prepared.keys:
        sequence = prepared.sequences[key]
        labels = getattr(prepared.questions[key], "keys", None)
        _need(
            type(sequence) is list
            and 1 <= len(sequence) <= _MAX_TOKENS
            and all(type(token) is int and 0 <= token < 2**63 for token in sequence)
            and type(labels) is list
            and 1 <= len(labels) <= _MAX_OPTIONS
            and all(type(label) is str for label in labels),
            "complete bounded token and option lists required",
        )
        question = prepared.questions[key]
        _need(
            type(getattr(question, "kind", None)) is str
            and question.kind in {"choice", "noul", "score"}
            and type(getattr(question, "original", None)) is dict
            and type(getattr(question, "rendered", None)) is dict
            and type(getattr(question, "descriptions", None)) is list,
            "native question response metadata required",
        )
        metadata = _freeze_metadata(
            [question.kind, question.original, question.rendered, labels, question.descriptions],
            metadata_budget,
            set(),
        )
        _need(len(set(labels)) == len(labels), "unique option labels required")
        rows.append((key, tuple(sequence), tuple(labels), metadata))
    return tuple(rows)


def _groups(rows, mode):
    indexed = list(enumerate(rows))
    if mode == "length_pairs":
        indexed.sort(key=lambda item: (-len(item[1][1]), item[0]))
    width = _NATIVE_BATCH if mode == "whole" else _PAIR_BATCH
    return [
        [row[0] for _, row in indexed[start : start + width]]
        for start in range(0, len(rows), width)
    ]


def _valid_number(value, lower, upper):
    return type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper


def _validate_answer(question, answer, probabilities):
    """Validate the fixed publisher's product shape against native run rows."""
    labels = question.keys
    _need(
        type(answer) is dict and answer.get("type") == question.kind, "valid native answer required"
    )
    total = sum(probabilities)
    shares = [value / total for value in probabilities]
    if question.kind == "noul":
        _need(
            set(answer) == {"type", "noul"}
            and labels == ["false", "true"]
            and _valid_number(answer["noul"], 0, 1)
            and math.isclose(answer["noul"], shares[1], rel_tol=1e-12, abs_tol=1e-12),
            "complete consistent native noul answer required",
        )
        return
    fields = {"type", "probabilities", "confidence"}
    fields |= {"choice"} if question.kind == "choice" else {"score", "legend"}
    _need(
        set(answer) == fields
        and type(answer["probabilities"]) is dict
        and list(answer["probabilities"]) == labels
        and _valid_number(answer["confidence"], 0, 1),
        "complete ordered native answer required",
    )
    for label, expected in zip(labels, shares, strict=True):
        observed = answer["probabilities"][label]
        _need(
            _valid_number(observed, 0, 1)
            and math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12),
            "native answer probability differs from model row",
        )
    confidence = (
        1.0
        if len(shares) < 2
        else max(
            0.0,
            min(1.0, 1 + sum(p * math.log(p) for p in shares if p > 0) / math.log(len(shares))),
        )
    )
    _need(
        math.isclose(answer["confidence"], confidence, rel_tol=1e-12, abs_tol=1e-12),
        "native answer confidence differs from model row",
    )
    if question.kind == "choice":
        _need(
            answer["choice"] == labels[max(range(len(shares)), key=shares.__getitem__)],
            "native choice differs from model row",
        )
        return
    _need(
        labels == [str(index) for index in range(len(labels))]
        and 2 <= len(labels) <= 10
        and len(question.descriptions) == len(labels)
        and type(answer["legend"]) is dict
        and list(answer["legend"]) == labels
        and _valid_number(answer["score"], 0, len(labels) - 1)
        and math.isclose(
            answer["score"],
            sum(index * value for index, value in enumerate(shares)),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "complete consistent native score answer required",
    )
    for label, description in zip(labels, question.descriptions, strict=True):
        expected = (
            description
            if isinstance(description, str)
            else json.dumps(description, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        _need(answer["legend"][label] == expected, "native score legend differs")


def evaluate(runtime, prepared, *, mode, enabled=False):
    """Evaluate B/C/D controls; response is native, statistics are estimates.

    ``whole`` runs one native call (internally batched at eight),
    ``ordered_pairs`` runs consecutive pairs, and ``length_pairs`` runs
    length-sorted pairs. All inputs and option labels remain intact and the
    original prepared request determines response order. Caller must guarantee
    exclusive runtime use and independently authenticate sources and inputs.
    """
    _need(SOURCE_ONLY is False and enabled is True, "reviewed explicit candidate opt-in required")
    _need(type(mode) is str and mode in _MODES, "known candidate mode required")
    _need(_LOCK.acquire(blocking=False), "exclusive candidate execution required")
    try:
        rows = _snapshot(runtime, prepared)
        model_name = runtime.model_name
        order = [row[0] for row in rows]
        lengths = {key: len(tokens) for key, tokens, _, _ in rows}
        options = {key: len(labels) for key, _, labels, _ in rows}
        groups = _groups(rows, mode)
        batch_groups = [order] if mode == "whole" else groups
        probabilities = {}
        tokens = 0
        for group in batch_groups:
            part = dataclasses.replace(prepared, keys=list(group))
            result = runtime.run(part)
            _need(
                type(result) is tuple
                and len(result) == 2
                and type(result[0]) is dict
                and set(result[0]) == set(group)
                and type(result[1]) is int
                and result[1] == sum(lengths[key] for key in group),
                "complete per-group native probabilities and token count required",
            )
            for key in group:
                values = result[0][key]
                _need(
                    type(values) is list
                    and len(values) == options[key]
                    and all(_valid_number(value, 0, 1) for value in values)
                    and math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-5),
                    "valid complete probabilities for every original option required",
                )
                probabilities[key] = list(values)
            tokens += result[1]
            _need(
                _snapshot(runtime, prepared) == rows and runtime.model_name == model_name,
                "original prepared input or runtime changed",
            )
        _need(tokens == sum(lengths.values()), "complete input token count required")
        original_probabilities = {key: tuple(probabilities[key]) for key in order}
        response = runtime.respond(prepared, probabilities, tokens)
        _need(
            _snapshot(runtime, prepared) == rows
            and runtime.model_name == model_name
            and type(response) is dict
            and type(response.get("model")) is str
            and response["model"] == model_name
            and type(response.get("answers")) is dict
            and list(response["answers"]) == order
            and type(response.get("usage")) is dict
            and type(response["usage"].get("input_tokens")) is int
            and response["usage"]["input_tokens"] == tokens
            and type(response["usage"].get("output_tokens")) is int
            and response["usage"]["output_tokens"] == 0,
            "original response order and token usage required",
        )
        for key in order:
            _need(
                type(probabilities.get(key)) is list
                and tuple(probabilities[key]) == original_probabilities[key],
                "native probability row changed during response",
            )
            _validate_answer(prepared.questions[key], response["answers"][key], probabilities[key])
        return {
            "response": response,
            "native_probabilities": {key: list(original_probabilities[key]) for key in order},
            "mode": mode,
            "native_run_calls": len(batch_groups),
            "forward_groups": len(groups),
            "unpadded_tokens": tokens,
            "padded_slots": sum(
                len(group) * max(lengths[key] for key in group) for group in groups
            ),
            "scope": "candidate estimates only; numerical equivalence, timing and provenance unverified",
        }
    finally:
        _LOCK.release()
