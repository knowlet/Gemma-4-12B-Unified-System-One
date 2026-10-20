"""Pure source-prefix and honest closed-snapshot replay tests; no ML imports."""

import builtins
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
BASE = Path(__file__).parent
PROFILE = ROOT / "artifacts/jev-runtime/public-revisit-profile-20261020-01/profile-01"
COHORT = (
    ROOT
    / "artifacts/jev-runtime/public-native-hover-settled-v8-20261019-01/pending-capacity-cohort-01.json"
)
REPLAY = (
    ROOT
    / "artifacts/jev-runtime/generic-state-revisit-progress-20261020-01/closed-state-replay-01.json"
)
SOURCE = BASE / "revisit_payload_builder.py"
SPEC = importlib.util.spec_from_file_location("revisit_payload_tested", SOURCE)
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


@pytest.fixture(autouse=True)
def forbidden_runtime(monkeypatch):
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        if name.split(".", 1)[0] in {
            "torch",
            "transformers",
            "httpx",
            "modal",
            "ctypes",
            "playwright",
            "selenium",
        }:
            raise AssertionError("Forbidden model/processor/browser dependency: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    output = tmp_path_factory.mktemp("pure-revisit-payloads") / "prepared"
    forbidden = {"torch", "transformers", "httpx", "modal", "ctypes", "playwright", "selenium"}
    assert not any(name.split(".", 1)[0] in forbidden for name in sys.modules)
    result = B.prepare(
        ROOT,
        PROFILE,
        COHORT,
        REPLAY,
        output,
        expected_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    )
    assert not any(name.split(".", 1)[0] in forbidden for name in sys.modules)
    return output, result


def load_payload(output, row, key="payload_ref"):
    ref = row[key]
    blob = (output / ref["path"]).read_bytes()
    assert len(blob) == ref["size_bytes"] and B.sha(blob) == ref["sha256"]
    return json.loads(blob)


def test_complete_explicit_scopes_and_mechanical_source_union(prepared):
    output, result = prepared
    assert result["schema_id"] == B.KIND and len(result["rows"]) == 68
    assert len({r["case_id"] for r in result["rows"]}) == 68
    assert result["counts"] == {
        "genuine_V8_cell_phase_replays": 16,
        "counterfactual_V7_saved_snapshots": 51,
        "technical_repetition": 1,
    }
    assert result["source_union_files"] == len(result["source_union"]) == 19
    assert result["genuine_new_profile_browser_observation_claimed"] is False
    assert result["processor_tokenizer_model_GPU_browser_calls"] == 0
    assert result["old_technical95_or_other_authority_modified"] is False
    assert (output / "payload-cohort.json").exists()
    for path, pin in result["source_union"].items():
        assert Path(path).stat().st_size == pin["size_bytes"]
        assert B.sha(Path(path).read_bytes()) == pin["sha256"]


def test_all68_bodies_preserve_order_full_heads_and_original_schema_except_declared_delta(prepared):
    output, result = prepared
    no_notes = 0
    for row in result["rows"]:
        payload, original = load_payload(output, row), load_payload(output, row, "baseline_ref")
        assert list(payload) == ["model", "state", "questions"]
        assert payload["model"] == "generated-actions-none"
        restored = copy.deepcopy(payload)
        restored["model"] = "fixture-no-LLM"
        hint = restored["state"].pop("observation_revisit", None)
        assert B.wire(restored) == B.wire(original)
        for name in original["questions"]:
            assert list(payload["questions"][name]["criteria"]) == list(
                original["questions"][name]["criteria"]
            )
        assert (
            row["capacity_admission_measured"] is False
            and row["same_historical_HTTP_wire_claimed"] is False
        )
        if not row["metadata"]["observation_notes"]:
            no_notes += 1
            assert hint is None
            payload["model"] = "fixture-no-LLM"
            assert B.wire(payload) == B.wire(original)
    assert no_notes > 0


def test_real_arxiv_NASA_notes_and_curl_zero_are_replayed_without_adding_observations(prepared):
    _output, result = prepared
    assert result["trace_counts"] == [
        {"case": "arxiv", "rows": 17, "exact_revisits": 0, "projection_revisits": 1},
        {"case": "nasa", "rows": 17, "exact_revisits": 12, "projection_revisits": 12},
        {"case": "curl", "rows": 17, "exact_revisits": 0, "projection_revisits": 0},
    ]
    for row in result["rows"][16:67]:
        assert "COUNTERFACTUAL_V7" in row["scope"]
        assert row["provenance"]["missing_intermediate_predict_refreshes_possible"] is True
        assert row["raw_observation_profile"]["id"] == "composed-visible-text-settled-v7"
        assert not any(
            term in " ".join(row["metadata"]["observation_notes"]).lower()
            for term in ("no effect", "no progress", "goal")
        )


def test_matrix_cells_do_not_share_tracker_or_retag_original_profile(prepared):
    _output, result = prepared
    for row in result["rows"][:16]:
        assert row["raw_observation_profile"]["id"] == "composed-visible-hover-entries-settled-v8"
        assert row["raw_observation_profile"]["id"] != B.PROFILE
        if row["provenance"]["phase"] == "before":
            assert row["metadata"]["observation_sequence"] == 0
            assert row["metadata"]["observation_notes"] == []


def test_technical_max32_is_separate_repetition_with_full_hint(prepared):
    output, result = prepared
    row = result["rows"][-1]
    assert row["scope"].startswith("TECHNICAL_33_DUPLICATE")
    assert row["provenance"]["repetitions"] == 33
    assert row["metadata"]["observation_sequence"] == 2**53 - 1
    assert len(row["metadata"]["exact"]["previous_observation_sequences"]) == 32
    payload = load_payload(output, row)
    hint = payload["state"]["observation_revisit"]
    assert len(hint["exact"]["previous_observation_sequences"]) == 32
    assert len(hint["projection"]["previous_observation_sequences"]) == 32
    assert "not complete state or action identity" in hint["projection"]["scope"]
    assert "sha256" not in B.wire(hint).decode()


@pytest.mark.parametrize("changed", ["model", "questions", "core"])
def test_unpinned_AST_or_tracker_never_executes(changed):
    model = (PROFILE / "runtime/jev_ultrafast/model.py").read_bytes()
    questions = (PROFILE / "runtime/jev_ultrafast/questions.py").read_bytes()
    core = (PROFILE / "runtime/jev_ultrafast/observation_revisit.py").read_bytes()
    with pytest.raises(ValueError, match="Exact"):
        if changed == "core":
            B.core_module(core + b"\n")
        else:
            B.question_builder(
                model + (b"\n" if changed == "model" else b""),
                questions + (b"\n" if changed == "questions" else b""),
            )


def test_exact_reader_refuses_wrong_pin_and_symlink(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"small source")
    assert B.checked(source, B.sha(b"small source")) == b"small source"
    with pytest.raises(ValueError):
        B.checked(source, "0" * 64)
    alias = tmp_path / "alias"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="Ordinary"):
        B.checked(alias, B.sha(b"small source"))
