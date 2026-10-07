"""Hash-bound annotations preserve historical evidence rather than rewriting it."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / "docs/validation/2026-10-04/gguf"


def load(path):
    return json.loads(path.read_text())


def annotation(identifier):
    entries = load(ROOT / "docs/validation/pr5-review-errata.json")["errata"]
    return next(entry["annotations"] for entry in entries if entry["id"] == identifier)


def test_all_errata_are_bound_to_unchanged_source_bytes():
    document = load(ROOT / "docs/validation/pr5-review-errata.json")
    assert document["schema_version"] == 1
    assert document["historical_artifacts_modified"] is False
    for entry in document["errata"]:
        assert entry["source_artifacts"]
        for artifact in entry["source_artifacts"]:
            path = ROOT / artifact["path"]
            assert path.resolve().is_relative_to(ROOT)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_duplicate_receipt_is_one_canonical_observation():
    directory = ROOT / "docs/validation/2026-10-02/release"
    assert (directory / "evaluate/receipt.json").read_bytes() == (
        directory / "evaluate-receipt.json"
    ).read_bytes()
    assert annotation("E07")["canonical_receipt"].endswith("/evaluate/receipt.json")


def test_gguf_file_type_and_generated_tensor_are_not_misreported():
    inventory = load(GGUF / "tensor-inventory.json")
    model = inventory["s1-boolq-Q8_0.gguf"]
    projector = inventory["mmproj-s1-boolq-f16.gguf"]
    assert model["metadata"]["general.file_type"] == 7
    # File-type 7 is Q8_0; tensor-type 8 is independently Q8_0.
    assert next(t for t in model["tensors"] if t["name"] == "token_embd.weight")["type"] == 8
    assert sum(t["name"] == "rope_freqs.weight" for t in model["tensors"]) == 1
    assert len(model["tensors"]) == 667
    assert len(projector["tensors"]) == 11
    conversion = load(GGUF / "conversion-manifest.json")
    assert conversion["source_tensor_count"] == 677
    assert len(model["tensors"]) + len(projector["tensors"]) == 677 + 1
    assert annotation("E03")["generated_tensor"] == "rope_freqs.weight"


def test_execution_revision_is_not_retroactively_rewritten():
    receipt = load(GGUF / "boundary-smoke-receipt.json")
    note = annotation("E02")
    assert receipt["source_git_revision"] == note["recorded_execution_head"]
    assert note["recorded_execution_head"] != note["verified_source_files_revision"]
    assert note["actual_execution_dirty_state"] == "unknown"


def test_compile_reproduction_does_not_substitute_the_evaluated_binary():
    build = load(GGUF / "build-reproduction.json")
    smoke = load(GGUF / "boundary-smoke-receipt.json")
    rebuilt = next(item for item in build["outputs"] if item["path"].endswith("/s1-gguf"))
    evaluated = next(item for item in smoke["source_files"] if item["path"].endswith("/s1-gguf"))
    note = annotation("E04")
    assert note["rebuilt_binary_sha256"] == rebuilt["sha256"]
    assert note["evaluated_binary_sha256"] == evaluated["sha256"]
    assert rebuilt["sha256"] != evaluated["sha256"]
    assert note["hashes_match"] is False
    assert note["byte_reproduction_claimed"] is False


def test_cold_start_remains_the_recorded_wall_measurement():
    row = json.loads((GGUF / "boundary-smoke-output.jsonl").read_text().splitlines()[0])
    note = annotation("E06")
    assert row["latency_ms"] > row["native_latency_ms"] + row["preprocessing"]["hf_prepare_ms"]
    assert note["first_request_wall_time_includes_startup"] is True
    assert note["steady_state_latency_claimed"] is False


@pytest.mark.parametrize(
    "old_name", ["gguf-boundary-smoke.log", "gguf-full-pytest.log", "gguf-code-build.log"]
)
def test_published_log_alias_resolves_to_the_receipt_bytes(old_name):
    aliases = annotation("E01")["path_aliases"]
    path = GGUF / aliases[old_name]
    if not path.exists() and (ROOT / "PKG-INFO").exists():
        pytest.skip("source distribution omits .log evidence; verify in full repository checkout")
    assert path.is_file(), f"published evidence missing: {path}"
    if old_name == "gguf-boundary-smoke.log":
        expected = load(GGUF / "boundary-smoke-receipt.json")["log_sha256"]
    else:
        entry = next(
            item
            for item in load(GGUF / "code-validation.json")["evidence"]
            if item["path"] == old_name
        )
        assert path.stat().st_size == entry["size_bytes"]
        expected = entry["sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
