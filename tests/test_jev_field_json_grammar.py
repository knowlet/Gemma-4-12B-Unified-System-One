"""Pure grammar/cache tests, including the installed Rust decoder as oracle.

No helper integration, model/framework, HTTP, GPU, browser or fitting occurs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import shutil
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/jev_field_json_grammar.py"
SPEC = importlib.util.spec_from_file_location("field_json_grammar_pure", SOURCE)
G = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = G
SPEC.loader.exec_module(G)
CACHE = (
    ROOT
    / "artifacts/jev-local-hf/hub/models--knowlet--Gemma-4-12B-Unified-System-One/snapshots/a66f836b56605039fe040f330180e336d19b3362"
)
TOKENIZER_SHA = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def reference(text):
    """Independent semantic parser: retain duplicate keys, require scalar text."""

    class Pairs(list):
        pass

    try:
        pairs = json.loads(text, object_pairs_hook=Pairs)
    except ValueError:
        return False
    if type(pairs) is not Pairs or len(pairs) != 1 or pairs[0][0] != "text":
        return False
    value = pairs[0][1]
    return value is None or (
        type(value) is str and all(not 0xD800 <= ord(c) <= 0xDFFF for c in value)
    )


@pytest.mark.parametrize(
    "text",
    [
        '{"text":null}',
        '{"text":""}',
        ' \r\n{\t"text" : "plain" }\n',
        '{"\\u0074e\\u0078t":"key escapes"}',
        '{"text":"\\b\\f\\n\\r\\t\\/\\\\\\"\\u0000"}',
        '{"text":"漢字 café \u2581 😀 \ufffd \ufffe \U0010ffff"}',
        '{"text":"\\uD800\\uDC00\\udbff\\udfff"}',
        '{"text":"\\uD000\\ud7ff\\uE000"}',
    ],
)
def test_unicode_scalar_object_language_accepts_values_without_answer_trie(text):
    blob = text.encode()
    assert reference(text) and G.accepts_json(blob)
    state = 0
    for byte in blob:
        state = G.json_feed(state, bytes([byte]))
        assert state is not None
    assert state == G.ACCEPT


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        "null",
        "[]",
        '[["text","x"]]',
        '{"text":1}',
        '{"text":true}',
        '{"text":[]}',
        '{"text":"x","text":"y"}',
        '{"text":"x","other":null}',
        '{"Text":"x"}',
        '{"te\\u0073t":"x"}',
        '{"text":"x"}{}',
        'thought {"text":"x"}',
        '```json\n{"text":"x"}\n```',
        '{"text":"\\uD800"}',
        '{"text":"\\uDC00"}',
        '{"text":"\\uD800x"}',
        '{"text":"\\uD800\\uD800"}',
        '{"text":"\\uDC00\\uD800"}',
        '{"text":"\\x20"}',
        '{"text":"raw\nnewline"}',
        '\u00a0{"text":null}',
    ],
)
def test_rejects_unpaired_surrogates_extra_keys_and_unmodified_thoughts(text):
    assert not reference(text)
    assert not G.accepts_json(text.encode())


@pytest.mark.parametrize(
    "raw", [b"\xc0\xaf", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"\x80", b"\xe4\xbd"]
)
def test_raw_string_bytes_must_be_complete_scalar_utf8(raw):
    assert not G.accepts_json(b'{"text":"' + raw + b'"}')


def test_every_escape_split_and_scalar_roundtrip_against_semantic_reference():
    rng = random.Random(20261018)
    values = ["", "\x00\x1f", "\U00010000\U0010ffff", "\u2581\ufffd", '"\\/']
    for _ in range(80):
        value = "".join(
            chr(rng.choice([rng.randrange(0xD800), rng.randrange(0xE000, 0x110000)]))
            for _ in range(8)
        )
        values.append(value)
    for value in values:
        for ascii_only in (False, True):
            blob = json.dumps({"text": value}, ensure_ascii=ascii_only).encode()
            assert reference(blob.decode()) and G.accepts_json(blob)
            for cut in range(len(blob) + 1):
                assert G.json_feed(G.json_feed(0, blob[:cut]), blob[cut:]) == G.ACCEPT


@pytest.fixture(scope="module")
def native():
    """Existing metadata/binary only; never downloads or imports Transformers."""
    overlay = str(ROOT / "artifacts/jev-runtime/overlay")
    sys.path.insert(0, overlay)
    try:
        import tokenizers
        from tokenizers import Tokenizer, decoders
    finally:
        sys.path.remove(overlay)
    path = CACHE / "tokenizer.json"
    assert digest(path.read_bytes()) == TOKENIZER_SHA
    binary = next((ROOT / "artifacts/jev-runtime/overlay/tokenizers").glob("*.so"))
    assert tokenizers.__version__ == G.DECODER_IDENTITY["version"]
    assert digest(binary.read_bytes()) == G.DECODER_IDENTITY["binary_sha256"]
    tokenizer = Tokenizer.from_file(str(path))
    blob, generation = path.read_bytes(), (CACHE / "generation_config.json").read_bytes()
    grammar, mapping_sha = G.tokenizer_mapping(blob, generation)
    decoder = decoders.Sequence(
        [decoders.Replace("\u2581", " "), decoders.ByteFallback(), decoders.Fuse()]
    )
    return tokenizer, grammar, decoder, json.loads(blob)["model"]["vocab"], mapping_sha


def ids(native, pieces):
    result = [native[0].token_to_id(piece) for piece in pieces]
    assert None not in result
    return result


PREFIX = ["{", '"', "text", '"', ":", '"']
SUFFIX = ['"', "}"]


def test_all262144_native_ids_retained_and_all_ordinary_pieces_match_rust(native):
    _, grammar, decoder, vocab, mapping_sha = native
    assert grammar.vocab_size == len(vocab) == 262144
    assert len(grammar.byte_ids) == 256
    assert sum(k in (G.SPECIAL, G.EOS) for k in grammar.classes) == 24
    assert sum(k == G.ORDINARY for k in grammar.classes) == 261864
    assert mapping_sha == "eb334cd54294f30a74c64d0130efebf3818546a292c1425242dada2fbca1f217"
    for piece, token_id in vocab.items():
        if grammar.classes[token_id] == G.ORDINARY:
            assert grammar.pieces[token_id].decode() == decoder.decode([piece])


def test_valid_projection_can_die_then_invalid_byte_salvages_whole_run(native):
    tokenizer, grammar, _, _, _ = native
    state = grammar.consume(ids(native, PREFIX))
    for piece in ["<0x22>", "<0x61>"]:
        state = grammar.step(state, ids(native, [piece])[0])
        assert state is not None
    assert state.valid is None and state.replacement is not None
    # The *current* valid complete run really is invalid JSON; an ordinary
    # delimiter cannot select the speculative replacement route.
    assert state.finalized() is None
    assert grammar.step(state, ids(native, ['"'])[0]) is None
    state = grammar.step(state, ids(native, ["<0xFF>"])[0])
    assert state is not None and state.utf8 == -1
    for token_id in ids(native, SUFFIX):
        state = grammar.step(state, token_id)
    assert state.complete
    sequence = ids(native, PREFIX + ["<0x22>", "<0x61>", "<0xFF>"] + SUFFIX)
    raw = tokenizer.decode(sequence, skip_special_tokens=True)
    assert json.loads(raw) == {"text": "\ufffd\ufffd\ufffd"}


@pytest.mark.parametrize(
    "run,expected",
    [
        (["<0xE4>", "<0xBD>", "<0xA0>"], "你"),
        (["<0xE4>", "<0xBD>", "<0xA0>", "<0xFF>"], "\ufffd" * 4),
        (["<0x61>", "<0xE4>"], "\ufffd" * 2),
        (["<0xE2>", "<0x96>", "<0x81>"], "\u2581"),
        (["<0xED>", "<0xA0>", "<0x80>"], "\ufffd" * 3),
        (["<0xC0>", "<0xAF>"], "\ufffd" * 2),
        (["\u2581"], " "),
    ],
)
def test_fallback_flush_matches_actual_rust_and_replace_order(native, run, expected):
    sequence = ids(native, PREFIX + run + SUFFIX)
    raw = native[0].decode(sequence, skip_special_tokens=True)
    assert json.loads(raw) == {"text": expected}
    assert native[1].consume(sequence).complete


def test_late_invalid_byte_revokes_tentative_close_and_eos(native):
    grammar = native[1]
    base = ids(native, PREFIX)
    end = ids(native, ["<0x22>", "<0x7D>"])
    state = grammar.consume(base + end)
    assert state.complete and grammar.step(state, 106).terminated
    state = grammar.step(state, ids(native, ["<0xFF>"])[0])
    assert state is not None and not state.complete
    assert grammar.step(state, 106) is None
    for token_id in ids(native, SUFFIX):
        state = grammar.step(state, token_id)
    assert state.complete


@pytest.mark.parametrize(
    "document",
    [
        '{"text":null}',
        '{"\\u0074ext":"\\ud83d\\ude00"}',
        '{"text":"你\u2581\\n\\u0000"}',
    ],
)
def test_entire_object_can_span_byte_tokens_including_key_and_surrogate_escapes(native, document):
    sequence = ids(native, [f"<0x{byte:02X}>" for byte in document.encode()])
    assert len(sequence) + 1 <= 128
    assert native[0].decode(sequence, skip_special_tokens=True) == document
    assert native[1].consume(sequence + [106]).complete


def test_ordinary_piece_flushes_partial_utf8_run_instead_of_joining_later_bytes(native):
    pieces = PREFIX + ["<0xE4>", "x", "<0xBD>", "<0xA0>"] + SUFFIX
    sequence = ids(native, pieces)
    assert json.loads(native[0].decode(sequence, skip_special_tokens=True)) == {
        "text": "\ufffdx\ufffd\ufffd"
    }
    assert native[1].consume(sequence).complete


def test_random_cross_token_runs_match_rust_full_decode_and_independent_json(native):
    rng = random.Random(70618)
    choices = [
        f"<0x{b:02X}>"
        for b in (0, 9, 32, 34, 58, 92, 97, 123, 125, 0xE4, 0xBD, 0xA0, 0xFF, 0xED, 0x80)
    ]
    choices += ["x", '"', "}", "\\", "u", "\u2581"]
    accepted = 0
    for _ in range(800):
        sequence = ids(
            native, PREFIX + [rng.choice(choices) for _ in range(rng.randrange(14))] + SUFFIX
        )
        raw = native[0].decode(sequence, skip_special_tokens=True)
        expected = reference(raw)
        try:
            state = native[1].consume(sequence)
            actual = state.complete
        except ValueError:
            actual = False
        assert actual is expected, (sequence, raw)
        assert G.accepts_json(raw.encode()) is expected
        accepted += expected
    assert accepted > 80  # Includes genuine malformed-byte replacement outputs.


@pytest.mark.parametrize("eos", G.EOS_IDS)
def test_each_actual_eos_is_only_allowed_after_complete_and_counts_as_emitted(native, eos):
    grammar = native[1]
    sequence = ids(native, ["{", '"', "text", '"', ":", "null", "}"])
    assert grammar.step(G.State(), eos) is None
    state = grammar.consume(sequence + [eos])
    assert state.terminated and state.complete and state.emitted == len(sequence) + 1
    assert native[0].decode(sequence + [eos], skip_special_tokens=True) == '{"text":null}'
    assert grammar.step(state, sequence[0]) is None
    for token_id, kind in enumerate(grammar.classes):
        if kind == G.SPECIAL:
            assert grammar.step(grammar.consume(sequence), token_id) is None


def test_real128_token_boundary_does_not_append_or_spend_synthetic_eos(native):
    grammar = native[1]
    sequence = ids(native, ["{", '"', "text", '"', ":", "null", "}"])
    space = ids(native, ["\u2581"])[0]
    before = sequence + [space] * (127 - len(sequence))
    state = grammar.consume(before + [106])
    assert state.terminated and state.emitted == 128
    capped = grammar.consume(before + [space])
    assert capped.complete and not capped.terminated and capped.emitted == 128
    assert grammar.step(capped, 106) is None
    assert not any(grammar.allowed_mask(capped))
    incomplete = grammar.consume(
        ids(native, PREFIX) + [ids(native, ["x"])[0]] * (128 - len(PREFIX))
    )
    assert incomplete.emitted == 128 and not incomplete.complete


@pytest.fixture(scope="module")
def toy_cache(tmp_path_factory):
    folder = tmp_path_factory.mktemp("field-json-compiler")
    special = set(range(21)) | {50, 106, 107}
    by_id = {i: f"piece{i}" for i in range(384)}
    for i in special:
        by_id[i] = f"<special{i}>"
    for byte in range(256):
        by_id[128 + byte] = f"<0x{byte:02X}>"
    wanted = list(
        dict.fromkeys(
            [
                "{",
                '"',
                "text",
                ":",
                "null",
                "}",
                " ",
                "x",
                "\\",
                "u",
                "\u2581",
                *"0123456789ABCDEFabcdef",
            ]
        )
    )
    available = [i for i in range(128) if i not in special]
    for i, piece in zip(available, wanted, strict=False):
        by_id[i] = piece
    vocab = {piece: i for i, piece in by_id.items()}
    tokenizer = {
        "decoder": G.DECODER_SPEC,
        "model": {"type": "BPE", "byte_fallback": True, "vocab": vocab},
        "added_tokens": [{"id": i, "content": by_id[i], "special": True} for i in sorted(special)],
    }
    generation = {"eos_token_id": list(G.EOS_IDS)}
    paths = folder / "tokenizer.json", folder / "generation_config.json"
    paths[0].write_bytes(G.encode(tokenizer))
    paths[1].write_bytes(G.encode(generation))
    kwargs = {
        "expected_tokenizer_sha256": digest(paths[0].read_bytes()),
        "expected_generation_sha256": digest(paths[1].read_bytes()),
        "expected_source_sha256": digest(SOURCE.read_bytes()),
    }
    output = folder / "cache"
    result = G.compile_cache(*paths, output, expected_vocab_size=384, **kwargs)
    load_kwargs = {**kwargs, "expected_manifest_sha256": result["manifest_sha256"]}
    return paths, output, kwargs, load_kwargs, result, vocab


def test_full_toy_compiler_rows_masks_and_ids_survive_cache_roundtrip(toy_cache):
    paths, output, _, load_kwargs, result, vocab = toy_cache
    cached = G.load_cache(output, **load_kwargs)
    plain, _ = G.tokenizer_mapping(
        paths[0].read_bytes(), paths[1].read_bytes(), expected_vocab_size=384
    )
    assert cached.vocab_size == len(vocab) == 384
    assert result["compiler_address_limit_bytes"] <= 3 * 1024**3
    assert sum(p["size_bytes"] for p in result["files"].values()) < 512 * 1024**2
    # The compiler's prefix-sharing optimization is checked against direct DFA
    # execution for every ordinary native ID in every reachable JSON state.
    for json_state in range(len(G.JSON_STATES)):
        state = G.State(committed=json_state)
        mask = cached.allowed_mask(state)
        for token_id, kind in enumerate(cached.classes):
            assert cached.step(state, token_id) == plain.step(state, token_id)
            assert bool(mask[token_id // 8] & (1 << (token_id % 8))) is (
                plain.step(state, token_id) is not None
            )
            if kind == G.ORDINARY:
                target = struct.unpack_from(
                    "<h", cached.next_rows, (json_state * 384 + token_id) * 2
                )[0]
                expected = G.json_feed(json_state, plain.pieces[token_id])
                assert target == (-1 if expected is None else expected)
    assert vocab[" "] != vocab["\u2581"]
    assert cached.pieces[vocab[" "]] == cached.pieces[vocab["\u2581"]] == b" "


def test_cached_masks_preserve_dynamic_salvage_and_ordinary_flush(toy_cache):
    _, output, _, load_kwargs, _, vocab = toy_cache
    grammar = G.load_cache(output, **load_kwargs)
    sequence = [vocab[x] for x in PREFIX + ["<0x22>", "<0x61>"]]
    state = grammar.consume(sequence)
    assert state.finalized() is None
    for state in [state, grammar.step(state, vocab["<0xFF>"])]:
        mask = grammar.allowed_mask(state)
        for token_id in range(grammar.vocab_size):
            assert bool(mask[token_id // 8] & (1 << (token_id % 8))) is (
                grammar.step(state, token_id) is not None
            )


@pytest.mark.parametrize(
    "mutation", ["missing", "orphan", "piece", "next", "mask", "symlink", "manifest"]
)
def test_closed_cache_corruption_or_substitution_fails_closed(tmp_path, toy_cache, mutation):
    _, original, _, load_kwargs, _, _ = toy_cache
    output = tmp_path / "tampered"
    shutil.copytree(original, output)
    if mutation == "missing":
        (output / "ordinary-mask.bin").unlink()
    elif mutation == "orphan":
        (output / "extra").write_bytes(b"not admitted")
    elif mutation == "symlink":
        (output / "pieces.bin").unlink()
        (output / "pieces.bin").symlink_to(original / "pieces.bin")
    else:
        name = {
            "piece": "pieces.bin",
            "next": "ordinary-next.bin",
            "mask": "ordinary-mask.bin",
            "manifest": "manifest.json",
        }[mutation]
        blob = bytearray((output / name).read_bytes())
        blob[len(blob) // 2] ^= 1
        (output / name).write_bytes(blob)
    with pytest.raises((ValueError, UnicodeError)):
        G.load_cache(output, **load_kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "expected_source_sha256",
        "expected_tokenizer_sha256",
        "expected_generation_sha256",
        "expected_manifest_sha256",
    ],
)
def test_cache_external_identity_is_required(toy_cache, field):
    _, output, _, load_kwargs, _, _ = toy_cache
    with pytest.raises(ValueError):
        G.load_cache(output, **{**load_kwargs, field: "0" * 64})


def test_compiler_resource_and_native_population_refusals_do_not_prune(tmp_path, toy_cache):
    paths, _, kwargs, _, _, _ = toy_cache
    with pytest.raises(ValueError, match="no pruning"):
        G.compile_cache(
            *paths, tmp_path / "too-small", expected_vocab_size=384, max_cache_bytes=1024, **kwargs
        )
    assert not (tmp_path / "too-small").exists()
    with pytest.raises(ValueError, match="Complete contiguous"):
        G.compile_cache(*paths, tmp_path / "wrong-width", **kwargs)
    assert not (tmp_path / "wrong-width").exists()
    with pytest.raises(ValueError, match="cannot increase"):
        G.compile_cache(
            *paths, tmp_path / "increased-cap", max_cache_bytes=G.MAX_CACHE_BYTES + 1, **kwargs
        )


def test_tokenizer_decoder_and_eos_changes_are_not_silently_reinterpreted(toy_cache):
    paths = toy_cache[0]
    tokenizer, generation = json.loads(paths[0].read_bytes()), json.loads(paths[1].read_bytes())
    tokenizer["decoder"]["decoders"].reverse()
    with pytest.raises(ValueError, match="Exact Replace"):
        G.tokenizer_mapping(G.encode(tokenizer), G.encode(generation), expected_vocab_size=384)
    generation["eos_token_id"] = [106]
    with pytest.raises(ValueError, match="Exact three"):
        G.tokenizer_mapping(paths[0].read_bytes(), G.encode(generation), expected_vocab_size=384)


def test_qualified_helper_remains_unchanged_and_module_has_no_model_import():
    assert (
        digest((ROOT / "src/s1/browser_text.py").read_bytes())
        == "381299d3961afba4ba2a296b1ce9ee5e4b137f5494b14cb4291b70340462b577"
    )
    assert not any(
        name.split(".")[0] in {"torch", "transformers", "modal", "peft"} for name in sys.modules
    )
    assert not (ROOT / "scripts/jev_field_json_grammar.py").samefile(
        ROOT / "src/s1/browser_text.py"
    )
