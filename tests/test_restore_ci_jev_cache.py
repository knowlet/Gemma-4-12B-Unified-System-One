"""Explicit tiny recovery fixtures, never substitute production trust anchors."""

import hashlib
import json
from pathlib import Path

import pytest
import restore_ci_jev_cache as recovery

FAKE_COMPILER = '''"""Synthetic unit-test producer only; no native cache claim."""
import hashlib
import json
from pathlib import Path

MODE = "normal"
PAYLOADS = %r

def encode(value):
    return json.dumps(value, separators=(",", ":")).encode()

def compile_cache(tokenizer, generation, output, **arguments):
    output = Path(output)
    assert not output.exists()
    output.mkdir()
    for name in PAYLOADS:
        (output / name).write_bytes(("synthetic " + name).encode())
    if MODE == "partial":
        raise RuntimeError("explicit compiler failure")
    if MODE == "payload":
        (output / PAYLOADS[0]).write_bytes(b"changed")
    if MODE == "extra":
        (output / "orphan").write_bytes(b"extra")
    manifest = {
        "schema": "synthetic-only" if MODE != "manifest" else "wrong-schema",
        "compile_elapsed_seconds": 999.0,
        "peak_rss_KiB": 999,
        "compiler_address_limit_bytes": 999,
        "max_cache_bytes": 999,
    }
    (output / "manifest.json").write_bytes(encode(manifest))
    if MODE == "race":
        (output.parent.parent / "cache").mkdir()

def load_cache(output, *, expected_manifest_sha256, **arguments):
    if MODE == "load":
        raise ValueError("explicit closed-cache refusal")
    assert hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest() == expected_manifest_sha256
'''


def pin(blob):
    return hashlib.sha256(blob).hexdigest(), len(blob)


def write(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)


@pytest.fixture
def miniature(tmp_path, monkeypatch):
    # Every monkeypatched anchor belongs only to this expressly synthetic test.
    root = tmp_path.resolve()
    payload_names = sorted(recovery.PAYLOADS)
    source = (FAKE_COMPILER % payload_names).encode()
    write(root / recovery.SOURCE, source)
    monkeypatch.setattr(recovery, "SOURCE_SHA", pin(source)[0])
    input_pins = {}
    for name in recovery.INPUT_PINS:
        blob = ("synthetic input " + name).encode()
        write(root / recovery.NATIVE / name, blob)
        input_pins[name] = pin(blob)
    monkeypatch.setattr(recovery, "INPUT_PINS", input_pins)
    historical = {
        "schema": "synthetic-only",
        "compile_elapsed_seconds": 10.25,
        "peak_rss_KiB": 12,
        "compiler_address_limit_bytes": 13,
        "max_cache_bytes": 14,
    }
    manifest_blob = json.dumps(historical, separators=(",", ":")).encode()
    monkeypatch.setattr(recovery, "MANIFEST_PIN", pin(manifest_blob))
    rows = []
    for name in sorted(payload_names + ["manifest.json"]):
        blob = manifest_blob if name == "manifest.json" else ("synthetic " + name).encode()
        sha, size = pin(blob)
        rows.append({"path": (recovery.CACHE / name).as_posix(), "sha256": sha, "size_bytes": size})
    documents = {
        recovery.PROOF: {
            "verification": {
                "compiler_seconds": historical["compile_elapsed_seconds"],
                "compiler_peak_RSS_KiB": historical["peak_rss_KiB"],
                "compiler_address_cap_bytes": historical["compiler_address_limit_bytes"],
                "cache_cap_bytes": historical["max_cache_bytes"],
            }
        },
        recovery.INVENTORY: {"grammar_cache": {"files": rows}},
    }
    document_pins = {}
    for path, value in documents.items():
        blob = json.dumps(value).encode()
        write(root / path, blob)
        document_pins[path] = pin(blob)
    monkeypatch.setattr(recovery, "DOCUMENT_PINS", document_pins)
    return root, manifest_blob, documents


def producer_mode(miniature, monkeypatch, mode):
    root, _, _ = miniature
    path = root / recovery.SOURCE
    blob = path.read_bytes().replace(b'MODE = "normal"', f'MODE = "{mode}"'.encode())
    path.write_bytes(blob)
    monkeypatch.setattr(recovery, "SOURCE_SHA", pin(blob)[0])


def test_recovers_exact_historical_bytes_and_reuses_without_recompilation(miniature, monkeypatch):
    root, expected, _ = miniature
    result = recovery.restore(root)
    assert result == {"restored": True, "manifest_sha256": pin(expected)[0]}
    destination = root / recovery.CACHE
    assert (destination / "manifest.json").read_bytes() == expected
    assert {p.name for p in destination.iterdir()} == recovery.PAYLOADS | {"manifest.json"}
    producer_mode(miniature, monkeypatch, "partial")
    assert recovery.restore(root) == {"restored": False, "manifest_sha256": pin(expected)[0]}
    assert not list(destination.parent.glob(".restore-cache-*"))


@pytest.mark.parametrize("name", ["tokenizer.json", "generation_config.json"])
def test_rejects_changed_inputs_before_compilation(miniature, name):
    root, _, _ = miniature
    (root / recovery.NATIVE / name).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Pinned bytes differ"):
        recovery.restore(root)
    assert not (root / recovery.CACHE).exists()


@pytest.mark.parametrize("document", [recovery.PROOF, recovery.INVENTORY])
def test_rejects_changed_historical_documents(miniature, document):
    root, _, _ = miniature
    (root / document).write_bytes(b"{}")
    with pytest.raises(ValueError, match="Pinned bytes differ"):
        recovery.restore(root)
    assert not (root / recovery.CACHE).exists()


def test_rejects_compiler_before_executing_changed_source(miniature):
    root, _, _ = miniature
    (root / recovery.SOURCE).write_text('raise AssertionError("must never execute")\n')
    with pytest.raises(ValueError, match="compiler source pin differs"):
        recovery.restore(root)


@pytest.mark.parametrize("mode", ["partial", "payload", "extra", "manifest", "load"])
def test_failed_recovery_never_publishes_and_removes_partial_staging(miniature, monkeypatch, mode):
    producer_mode(miniature, monkeypatch, mode)
    root, _, _ = miniature
    with pytest.raises((ValueError, RuntimeError)):
        recovery.restore(root)
    assert not (root / recovery.CACHE).exists()
    assert not list((root / recovery.CACHE).parent.glob(".restore-cache-*"))


@pytest.mark.parametrize("fault", ["payload", "missing", "extra", "manifest", "file"])
def test_existing_invalid_cache_fails_closed_without_replacement(miniature, fault):
    root, _, _ = miniature
    destination = root / recovery.CACHE
    if fault == "file":
        write(destination, b"not a directory")
    else:
        recovery.restore(root)
        if fault == "missing":
            (destination / "pieces.bin").unlink()
        elif fault == "extra":
            (destination / "orphan").write_bytes(b"extra")
        else:
            (destination / ("pieces.bin" if fault == "payload" else "manifest.json")).write_bytes(
                b"corrupt"
            )
    before = (
        destination.read_bytes()
        if fault == "file"
        else {p.name: p.read_bytes() for p in destination.iterdir()}
    )
    with pytest.raises(ValueError):
        recovery.restore(root)
    after = (
        destination.read_bytes()
        if fault == "file"
        else {p.name: p.read_bytes() for p in destination.iterdir()}
    )
    assert after == before
    assert not list(destination.parent.glob(".restore-cache-*"))


def test_refuses_a_cache_created_while_compiler_was_running(miniature, monkeypatch):
    producer_mode(miniature, monkeypatch, "race")
    root, _, _ = miniature
    with pytest.raises(ValueError, match="appeared during recovery"):
        recovery.restore(root)
    assert list((root / recovery.CACHE).iterdir()) == []


@pytest.mark.parametrize("target", ["input", "source", "cache", "cache-parent", "payload"])
def test_refuses_symlinks_in_inputs_outputs_and_ancestry(miniature, target):
    root, _, _ = miniature
    if target == "input":
        path = root / recovery.NATIVE / "tokenizer.json"
    elif target == "source":
        path = root / recovery.SOURCE
    else:
        recovery.restore(root)
        path = root / recovery.CACHE
        if target == "cache-parent":
            path = path.parent
        elif target == "payload":
            path = path / "pieces.bin"
    moved = (
        root / "original-payload"
        if target == "payload"
        else path.with_name(path.name + "-original")
    )
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(ValueError, match="Ordinary path required"):
        recovery.restore(root)


def test_bounded_reads_do_not_accept_large_files(tmp_path):
    path = tmp_path / "large"
    path.write_bytes(b"12345")
    with pytest.raises(ValueError, match="Bounded file required"):
        recovery.read(path, limit=4)


def test_real_trust_anchors_match_tracked_original_records():
    # Read-only check of production evidence, not a synthetic cache acceptance.
    root = Path(__file__).resolve().parents[1]
    pins, observations = recovery.authorities(root)
    assert pins["manifest.json"] == recovery.MANIFEST_PIN
    assert sum(size for _, size in pins.values()) == 33461006
    assert observations == {
        "compile_elapsed_seconds": 9.685079502000008,
        "peak_rss_KiB": 324388,
        "compiler_address_limit_bytes": 3221225472,
        "max_cache_bytes": 536870912,
    }
    assert pin((root / recovery.SOURCE).read_bytes())[0] == recovery.SOURCE_SHA
