"""Independent Unicode field-JSON grammar/cache; never imports a model or helper.

This is a qualification prototype, not an integration with browser_text. Byte
fallback runs are tentative: a later invalid byte replaces the *entire* run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import struct
import sys
import threading
import time
from array import array
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

SCHEMA = "field-json-decoder-run-cache-v1"
MAX_CACHE_BYTES = 512 * 1024**2
MAX_COMPILER_BYTES = 3 * 1024**3
MAX_GENERATED_TOKENS = 128
EOS_IDS = (1, 106, 50)
DECODER_IDENTITY = {
    "version": "0.23.3",
    "binary_sha256": "eb4d4fab7a667086711cfc6b7f0fba6c0275c2059da0f07d52c917bfa854a20d",
}
DECODER_SPEC = {
    "type": "Sequence",
    "decoders": [
        {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
    ],
}
ORDINARY, BYTE, SPECIAL, EOS = range(4)
WS = frozenset(b" \t\n\r")
REPLACEMENT = "\ufffd".encode()
CACHE_FILES = frozenset(
    {
        "pieces.bin",
        "offsets.bin",
        "classes.bin",
        "json-dfa.bin",
        "json-states.json",
        "ordinary-next.bin",
        "ordinary-mask.bin",
    }
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _json(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate cache JSON key")
            result[key] = value
        return result

    return json.loads(
        blob,
        object_pairs_hook=pairs,
        parse_constant=lambda value: require(False, "Nonfinite JSON: " + value),
    )


def _hex(byte):
    if 48 <= byte <= 57:
        return byte - 48
    if 65 <= byte <= 70:
        return byte - 55
    if 97 <= byte <= 102:
        return byte - 87
    return None


def _utf8_next(state, byte):
    remaining, low, high = state
    if remaining:
        return (remaining - 1, 128, 191) if low <= byte <= high else None
    if byte < 128:
        return (0, 128, 191)
    if 194 <= byte <= 223:
        return (1, 128, 191)
    if byte == 224:
        return (2, 160, 191)
    if 225 <= byte <= 236 or 238 <= byte <= 239:
        return (2, 128, 191)
    if byte == 237:
        return (2, 128, 159)
    if byte == 240:
        return (3, 144, 191)
    if 241 <= byte <= 243:
        return (3, 128, 191)
    if byte == 244:
        return (3, 128, 143)
    return None


def _json_next(state, byte):
    kind, *args = state
    if kind in {"start", "key-start", "colon", "value-start", "object-close", "done"}:
        if byte in WS:
            return state
        return {
            "start": ("key-start",) if byte == 123 else None,
            "key-start": ("key", 0) if byte == 34 else None,
            "colon": ("value-start",) if byte == 58 else None,
            "value-start": ("string",) if byte == 34 else (("null", 1) if byte == 110 else None),
            "object-close": ("done",) if byte == 125 else None,
            "done": None,
        }[kind]
    if kind == "key":
        index = args[0]
        if index == 4:
            return ("colon",) if byte == 34 else None
        if byte == b"text"[index]:
            return ("key", index + 1)
        return ("key-escape", index) if byte == 92 else None
    if kind == "key-escape":
        return ("key-unicode", args[0], 0) if byte == 117 else None
    if kind == "key-unicode":
        index, position = args
        expected = f"{b'text'[index]:04x}"
        if _hex(byte) != int(expected[position], 16):
            return None
        return ("key", index + 1) if position == 3 else (kind, index, position + 1)
    if kind == "null":
        index = args[0]
        if byte != b"null"[index]:
            return None
        return ("object-close",) if index == 3 else (kind, index + 1)
    if kind == "string":
        if byte == 34:
            return ("object-close",)
        if byte == 92:
            return ("escape",)
        if byte < 32:
            return None
        if byte < 128:
            return state
        next_state = _utf8_next((0, 128, 191), byte)
        return ("utf8", *next_state) if next_state is not None else None
    if kind == "utf8":
        next_state = _utf8_next(tuple(args), byte)
        if next_state is None:
            return None
        return ("string",) if next_state[0] == 0 else (kind, *next_state)
    if kind == "escape":
        if byte in b'"\\/bfnrt':
            return ("string",)
        return ("unicode-first",) if byte == 117 else None
    if kind == "unicode-first":
        digit = _hex(byte)
        if digit is None:
            return None
        return ("unicode-d",) if digit == 13 else ("unicode-rest", 3, False)
    if kind == "unicode-d":
        digit = _hex(byte)
        if digit is None or digit >= 12:
            return None  # A lone low surrogate is not a Unicode scalar.
        return ("unicode-rest", 2, digit >= 8)
    if kind == "unicode-rest":
        remaining, high = args
        if _hex(byte) is None:
            return None
        if remaining == 1:
            return ("low-slash",) if high else ("string",)
        return (kind, remaining - 1, high)
    if kind == "low-slash":
        return ("low-u",) if byte == 92 else None
    if kind == "low-u":
        return ("low-d",) if byte == 117 else None
    if kind == "low-d":
        return ("low-range",) if _hex(byte) == 13 else None
    if kind == "low-range":
        digit = _hex(byte)
        return ("unicode-rest", 2, False) if digit is not None and digit >= 12 else None
    raise AssertionError("Unknown DFA state")


def _machine(start, transition):
    states, numbers, table, pending = [start], {start: 0}, array("h"), deque([start])
    while pending:
        state = pending.popleft()
        for byte in range(256):
            target = transition(state, byte)
            if target is None:
                table.append(-1)
                continue
            if target not in numbers:
                require(len(states) < 32767, "DFA state representation exhausted")
                numbers[target] = len(states)
                states.append(target)
                pending.append(target)
            table.append(numbers[target])
    return tuple(states), table


JSON_STATES, JSON_TABLE = _machine(("start",), _json_next)
UTF8_STATES, UTF8_TABLE = _machine((0, 128, 191), _utf8_next)
ACCEPT = JSON_STATES.index(("done",))


def json_feed(state, data):
    if state is None:
        return None
    require(type(state) is int and 0 <= state < len(JSON_STATES), "Unknown JSON state")
    require(type(data) is bytes, "Exact bytes required")
    for byte in data:
        state = JSON_TABLE[state * 256 + byte]
        if state == -1:
            return None
    return state


def accepts_json(data):
    return json_feed(0, data) == ACCEPT


@dataclass(frozen=True, slots=True)
class State:
    committed: int = 0
    run: bool = False
    valid: int | None = None
    replacement: int | None = None
    utf8: int = 0
    emitted: int = 0
    terminated: bool = False

    def finalized(self):
        if not self.run:
            return self.committed
        # A valid complete run never chooses the alternative replacement route.
        return self.valid if self.utf8 == 0 else self.replacement

    @property
    def complete(self):
        return self.finalized() == ACCEPT


class Grammar:
    """Full native-ID grammar, with optional source-authenticated ordinary rows."""

    def __init__(self, pieces, classes, *, next_rows=None, mask_rows=None):
        require(len(pieces) == len(classes) and len(pieces) > max(EOS_IDS), "Native ID inventory")
        require(all(type(piece) is bytes for piece in pieces), "Byte pieces required")
        require(
            type(classes) is bytes and all(kind in range(4) for kind in classes), "Token classes"
        )
        require(
            {i for i, kind in enumerate(classes) if kind == EOS} == set(EOS_IDS), "Exact EOS IDs"
        )
        self.pieces, self.classes = tuple(pieces), classes
        self.vocab_size, self.mask_size = len(pieces), (len(pieces) + 7) // 8
        self.byte_ids = tuple(i for i, kind in enumerate(classes) if kind == BYTE)
        require(
            len(self.byte_ids) == 256
            and {self.pieces[i] for i in self.byte_ids} == {bytes([b]) for b in range(256)},
            "Every native fallback byte required",
        )
        require(
            (next_rows is None and mask_rows is None)
            or (
                type(next_rows) is bytes
                and type(mask_rows) is bytes
                and len(next_rows) == len(JSON_STATES) * self.vocab_size * 2
                and len(mask_rows) == len(JSON_STATES) * self.mask_size
            ),
            "Complete ordinary transition/mask rows required",
        )
        self.next_rows, self.mask_rows = next_rows, mask_rows

    def step(self, state, token_id):
        require(type(state) is State, "Owned immutable grammar state required")
        require(
            type(token_id) is int and 0 <= token_id < self.vocab_size, "Native token ID required"
        )
        if state.terminated or state.emitted >= MAX_GENERATED_TOKENS:
            return None
        kind, piece = self.classes[token_id], self.pieces[token_id]
        if kind == SPECIAL:
            return None
        if kind == BYTE:
            utf8 = (
                UTF8_TABLE[state.utf8 * 256 + piece[0]]
                if state.run and state.utf8 >= 0
                else (-1 if state.run else UTF8_TABLE[piece[0]])
            )
            valid = json_feed(state.valid if state.run else state.committed, piece)
            if utf8 == -1:
                valid = None
            replacement = json_feed(
                state.replacement if state.run else state.committed, REPLACEMENT
            )
            if valid is None and replacement is None:
                return None
            return State(
                state.committed,
                True,
                valid,
                replacement,
                utf8,
                state.emitted + 1,
                False,
            )
        final = state.finalized()
        if final is None:
            return None
        if kind == EOS:
            return (
                replace(state, emitted=state.emitted + 1, terminated=True)
                if final == ACCEPT
                else None
            )
        if self.next_rows is None:
            target = json_feed(final, piece)
        else:
            target = struct.unpack_from(
                "<h", self.next_rows, (final * self.vocab_size + token_id) * 2
            )[0]
            target = None if target == -1 else target
        return None if target is None else State(committed=target, emitted=state.emitted + 1)

    def consume(self, ids):
        state = State()
        for index, token_id in enumerate(ids):
            state = self.step(state, token_id)
            require(state is not None, "Grammar refused generated token at index " + str(index))
        return state

    def allowed_mask(self, state):
        require(type(state) is State, "Owned immutable grammar state required")
        result = bytearray(self.mask_size)
        if state.terminated or state.emitted >= MAX_GENERATED_TOKENS:
            return bytes(result)
        final = state.finalized()
        if final is not None and self.mask_rows is not None:
            result[:] = self.mask_rows[final * self.mask_size : (final + 1) * self.mask_size]
        elif final is not None:
            for token_id, kind in enumerate(self.classes):
                if kind == ORDINARY and self.step(state, token_id) is not None:
                    result[token_id // 8] |= 1 << (token_id % 8)
        for token_id in (*self.byte_ids, *EOS_IDS):
            if self.step(state, token_id) is not None:
                result[token_id // 8] |= 1 << (token_id % 8)
        return bytes(result)


def _ordinary(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Ordinary cache path required")
    return path


def _read_pinned(path, expected, limit):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= limit, "Bounded pinned input required")
    blob = path.read_bytes()
    require(len(blob) <= limit and sha(blob) == expected, "Pinned input SHA differs")
    return blob


def tokenizer_mapping(blob, generation_blob, *, expected_vocab_size=262144):
    tokenizer, generation = _json(blob), _json(generation_blob)
    require(
        tokenizer.get("decoder") == DECODER_SPEC, "Exact Replace/ByteFallback/Fuse decoder required"
    )
    model = tokenizer.get("model", {})
    vocab, added = model.get("vocab"), tokenizer.get("added_tokens")
    require(
        model.get("type") == "BPE" and model.get("byte_fallback") is True,
        "BPE byte fallback required",
    )
    require(
        type(vocab) is dict
        and len(vocab) == expected_vocab_size
        and all(type(i) is int for i in vocab.values())
        and sorted(vocab.values()) == list(range(expected_vocab_size)),
        "Complete contiguous native vocabulary required",
    )
    require(
        type(added) is list and len({r.get("id") for r in added}) == len(added),
        "Unique added tokens",
    )
    require(
        all(
            type(r.get("id")) is int
            and vocab.get(r.get("content")) == r["id"]
            and r.get("special") is True
            for r in added
        ),
        "Only exact explicit special added tokens supported",
    )
    special = {r["id"] for r in added}
    require(
        generation.get("eos_token_id") == list(EOS_IDS)
        and all(type(i) is int for i in generation["eos_token_id"])
        and set(EOS_IDS) <= special,
        "Exact three configured EOS IDs required",
    )
    pieces, classes, digest = [None] * len(vocab), bytearray(len(vocab)), hashlib.sha256()
    for text, token_id in vocab.items():
        require(type(text) is str, "Unicode vocabulary piece required")
        if token_id in special:
            piece, kind = b"", EOS if token_id in EOS_IDS else SPECIAL
        elif re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", text):
            piece, kind = bytes([int(text[3:5], 16)]), BYTE
        else:
            piece, kind = text.replace("\u2581", " ").encode("utf-8", errors="strict"), ORDINARY
        pieces[token_id], classes[token_id] = piece, kind
    for token_id, piece in enumerate(pieces):
        name = (
            "special"
            if classes[token_id] in (SPECIAL, EOS)
            else ("byte_fallback" if classes[token_id] == BYTE else "unicode_piece")
        )
        digest.update(encode([token_id, name, piece.hex()]) + b"\n")
    grammar = Grammar(pieces, bytes(classes))
    return grammar, digest.hexdigest()


def _little(values):
    result = array(values.typecode, values)
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


@contextmanager
def _compiler_limit():
    require(threading.active_count() == 1, "Single-thread compiler process required")
    require(
        not any(n.split(".")[0] in {"torch", "transformers", "peft", "modal"} for n in sys.modules),
        "Pure compiler process required",
    )
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = min(v for v in (soft, hard, MAX_COMPILER_BYTES) if v != resource.RLIM_INFINITY)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    try:
        yield limit
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def _trie(grammar):
    parents, edges = array("I", [0]), bytearray([0])
    terminals = array("I", [0]) * grammar.vocab_size
    stack, previous = [0], b""
    ordered = sorted(
        (piece, i) for i, piece in enumerate(grammar.pieces) if grammar.classes[i] == ORDINARY
    )
    for piece, token_id in ordered:
        common = 0
        while common < min(len(piece), len(previous)) and piece[common] == previous[common]:
            common += 1
        del stack[common + 1 :]
        for byte in piece[common:]:
            parents.append(stack[-1])
            edges.append(byte)
            stack.append(len(parents) - 1)
        terminals[token_id], previous = stack[-1], piece
    return parents, edges, terminals


def compile_cache(
    tokenizer_path,
    generation_path,
    output,
    *,
    expected_tokenizer_sha256,
    expected_generation_sha256,
    expected_source_sha256,
    expected_vocab_size=262144,
    max_cache_bytes=MAX_CACHE_BYTES,
):
    """Compile every native ID and every JSON-state ordinary row; no inference."""
    require(
        sha(Path(__file__).read_bytes()) == expected_source_sha256, "Compiler source SHA differs"
    )
    require(
        type(max_cache_bytes) is int and 0 < max_cache_bytes <= MAX_CACHE_BYTES,
        "Cache cap cannot increase",
    )
    output = _ordinary(output)
    require(not output.exists(), "Fresh cache output required")
    started = time.monotonic()
    with _compiler_limit() as actual_limit:
        tokenizer_blob = _read_pinned(tokenizer_path, expected_tokenizer_sha256, 64 * 1024**2)
        generation_blob = _read_pinned(generation_path, expected_generation_sha256, 1024**2)
        grammar, mapping_sha = tokenizer_mapping(
            tokenizer_blob, generation_blob, expected_vocab_size=expected_vocab_size
        )
        pool, offsets = bytearray(), array("I", [0])
        for piece in grammar.pieces:
            pool.extend(piece)
            offsets.append(len(pool))
        projected = (
            len(pool)
            + len(offsets) * 4
            + grammar.vocab_size
            + len(JSON_TABLE) * 2
            + len(JSON_STATES) * (grammar.vocab_size * 2 + grammar.mask_size)
            + 1024**2
        )
        require(
            projected <= max_cache_bytes,
            "Full native cache exceeds unchanged cache cap; no pruning",
        )
        parents, edges, terminals = _trie(grammar)
        output.mkdir(parents=True)
        files = {
            "pieces.bin": bytes(pool),
            "offsets.bin": _little(offsets),
            "classes.bin": grammar.classes,
            "json-dfa.bin": _little(JSON_TABLE),
            "json-states.json": encode(JSON_STATES),
        }
        for name, blob in files.items():
            with (output / name).open("xb") as stream:
                stream.write(blob)
        ordinary_ids = [i for i, kind in enumerate(grammar.classes) if kind == ORDINARY]
        with (
            (output / "ordinary-next.bin").open("xb") as next_file,
            (output / "ordinary-mask.bin").open("xb") as mask_file,
        ):
            for state in range(len(JSON_STATES)):
                states = array("h", [state]) * len(parents)
                for node in range(1, len(parents)):
                    parent = states[parents[node]]
                    states[node] = -1 if parent == -1 else JSON_TABLE[parent * 256 + edges[node]]
                row, mask = array("h", [-1]) * grammar.vocab_size, bytearray(grammar.mask_size)
                for token_id in ordinary_ids:
                    target = states[terminals[token_id]]
                    row[token_id] = target
                    if target != -1:
                        mask[token_id // 8] |= 1 << (token_id % 8)
                next_file.write(_little(row))
                mask_file.write(mask)
        pins = {}
        for path in sorted(output.iterdir()):
            blob = path.read_bytes()
            pins[path.name] = {"sha256": sha(blob), "size_bytes": len(blob)}
        manifest = {
            "schema": SCHEMA,
            "source_sha256": expected_source_sha256,
            "tokenizer_sha256": expected_tokenizer_sha256,
            "generation_config_sha256": expected_generation_sha256,
            "decoder_spec": DECODER_SPEC,
            "decoder_identity": DECODER_IDENTITY,
            "mapping_rows_sha256": mapping_sha,
            "native_vocab_size": grammar.vocab_size,
            "all_native_IDs_retained": True,
            "class_counts": dict(Counter(grammar.classes)),
            "EOS_ids": list(EOS_IDS),
            "JSON_states": len(JSON_STATES),
            "trie_nodes": len(parents),
            "format": "little-endian uint32 offsets; int16 next states (-1 dead); LSB-first ID masks",
            "files": pins,
            "max_cache_bytes": max_cache_bytes,
            "compiler_address_limit_bytes": actual_limit,
            "compile_elapsed_seconds": time.monotonic() - started,
            "peak_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "qualified": False,
            "model_calls": 0,
        }
        manifest_blob = encode(manifest)
        require(
            sum(p["size_bytes"] for p in pins.values()) + len(manifest_blob) <= max_cache_bytes,
            "Final cache cap exceeded",
        )
        with (output / "manifest.json").open("xb") as stream:
            stream.write(manifest_blob)
        return {"manifest_sha256": sha(manifest_blob), **manifest}


def load_cache(
    output,
    *,
    expected_manifest_sha256,
    expected_tokenizer_sha256,
    expected_generation_sha256,
    expected_source_sha256,
):
    output = _ordinary(output)
    require(output.is_dir(), "Ordinary complete cache directory required")
    manifest_blob = _read_pinned(
        _ordinary(output / "manifest.json"), expected_manifest_sha256, 1024**2
    )
    manifest = _json(manifest_blob)
    require(
        manifest.get("schema") == SCHEMA
        and sha(Path(__file__).read_bytes())
        == manifest.get("source_sha256")
        == expected_source_sha256
        and manifest.get("tokenizer_sha256") == expected_tokenizer_sha256
        and manifest.get("generation_config_sha256") == expected_generation_sha256
        and manifest.get("decoder_identity") == DECODER_IDENTITY
        and manifest.get("decoder_spec") == DECODER_SPEC
        and manifest.get("EOS_ids") == list(EOS_IDS)
        and manifest.get("JSON_states") == len(JSON_STATES)
        and manifest.get("all_native_IDs_retained") is True,
        "Closed cache source/tokenizer/decoder/grammar identity differs",
    )
    files = manifest.get("files")
    require(
        type(files) is dict and set(files) == CACHE_FILES, "Exact cache file inventory required"
    )
    require(
        {p.name for p in output.iterdir()} == CACHE_FILES | {"manifest.json"},
        "Orphan/incomplete cache",
    )
    require(
        type(manifest.get("max_cache_bytes")) is int
        and 0 < manifest["max_cache_bytes"] <= MAX_CACHE_BYTES
        and all(
            type(p) is dict
            and set(p) == {"sha256", "size_bytes"}
            and type(p["size_bytes"]) is int
            and p["size_bytes"] >= 0
            for p in files.values()
        )
        and sum(p["size_bytes"] for p in files.values()) + len(manifest_blob)
        <= manifest["max_cache_bytes"],
        "Bounded exact cache inventory required",
    )
    blobs = {}
    for name, pin in files.items():
        path = _ordinary(output / name)
        require(
            path.is_file() and path.stat().st_size == pin["size_bytes"], "Cache file size differs"
        )
        blobs[name] = _read_pinned(path, pin["sha256"], pin["size_bytes"])
    require(
        blobs["json-dfa.bin"] == _little(JSON_TABLE)
        and blobs["json-states.json"] == encode(JSON_STATES),
        "Compiled JSON DFA differs",
    )
    size = manifest.get("native_vocab_size")
    require(
        type(size) is int
        and size > max(EOS_IDS)
        and len(blobs["classes.bin"]) == size
        and len(blobs["offsets.bin"]) == (size + 1) * 4,
        "Native cache dimensions differ",
    )
    offsets = struct.unpack("<" + "I" * (size + 1), blobs["offsets.bin"])
    require(
        offsets[0] == 0
        and offsets[-1] == len(blobs["pieces.bin"])
        and all(a <= b for a, b in zip(offsets, offsets[1:])),
        "Native offsets differ",
    )
    pieces = tuple(blobs["pieces.bin"][a:b] for a, b in zip(offsets, offsets[1:]))
    return Grammar(
        pieces,
        blobs["classes.bin"],
        next_rows=blobs["ordinary-next.bin"],
        mask_rows=blobs["ordinary-mask.bin"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tokenizer-sha256", required=True)
    parser.add_argument("--expected-generation-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args(argv)
    result = compile_cache(
        args.tokenizer,
        args.generation_config,
        args.output,
        expected_tokenizer_sha256=args.expected_tokenizer_sha256,
        expected_generation_sha256=args.expected_generation_sha256,
        expected_source_sha256=args.expected_source_sha256,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "manifest_sha256",
                    "native_vocab_size",
                    "JSON_states",
                    "trie_nodes",
                    "compile_elapsed_seconds",
                    "peak_rss_KiB",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
