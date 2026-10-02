"""Offline export preflight checks; no MLX import, model weights or GPU work."""

import builtins
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def verifier():
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_mlx_export.py"
    return runpy.run_path(str(script))


@pytest.fixture
def aligned():
    manifest = {
        "prompt_version": 1,
        "dataset_sha256": "dataset-fingerprint",
        "source_model": "fixture/gemma4",
        "source_revision": "a" * 40,
        "temperature": 1.0,
        "letters": list(range(52)),
        "cases": [
            {
                "id": "case-1",
                "file": "case-1.npz",
                "slots": [2],
                "nopts": [2],
                "questions": [{"id": "route", "labels": ["left", "right"]}],
            }
        ],
    }
    reference = {
        "status": "ok",
        "dataset_sha256": manifest["dataset_sha256"],
        "pinned_revision": manifest["source_revision"],
        "model": {
            "name": manifest["source_model"],
            "resolved_revision": manifest["source_revision"],
            "temperature": manifest["temperature"],
        },
        "cases": [
            {
                "id": "case-1",
                "question_count": 1,
                "input_tokens": 3,
                "end_to_end_samples": [
                    {"answers": {"route": {"probabilities": {"right": 0.75, "left": 0.25}}}}
                ],
            }
        ],
    }
    return manifest, reference


@pytest.fixture
def model_config():
    return {
        "model_type": "gemma4_unified",
        "tie_word_embeddings": True,
        "text_config": {"final_logit_softcapping": 30.0},
        "vision_config": {"model_type": "fixture-vision"},
        "audio_config": {"model_type": "fixture-audio"},
    }


def test_matching_reference_follows_manifest_label_order(verifier, aligned):
    manifest, reference = aligned
    paired = verifier["validate_reference"](manifest, reference)
    assert list(paired) == ["case-1"]
    np.testing.assert_array_equal(paired["case-1"][0], [0.25, 0.75])


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("fingerprint", "dataset hashes differ"),
        ("revision", "source revisions differ"),
        ("temperature", "temperatures differ"),
        ("labels", "candidate labels differ"),
    ],
)
def test_reference_mismatch_rejected(verifier, aligned, mismatch, message):
    manifest, reference = aligned
    if mismatch == "fingerprint":
        reference["dataset_sha256"] = "another-dataset"
    elif mismatch == "revision":
        reference["model"]["resolved_revision"] = "b" * 40
    elif mismatch == "temperature":
        reference["model"]["temperature"] = 2.0
    else:
        reference["cases"][0]["end_to_end_samples"][0]["answers"]["route"]["probabilities"] = {
            "left": 0.25,
            "unknown": 0.75,
        }
    with pytest.raises(ValueError, match=message):
        verifier["validate_reference"](manifest, reference)


def test_complete_tied_model_config_accepted(verifier, model_config):
    assert verifier["validate_model_config"](model_config) == 30.0


@pytest.mark.parametrize("location", ["outer", "text"])
def test_untied_head_rejected(verifier, model_config, location):
    target = model_config if location == "outer" else model_config["text_config"]
    target["tie_word_embeddings"] = False
    with pytest.raises(ValueError, match="untied output heads are unsupported"):
        verifier["validate_model_config"](model_config)


@pytest.mark.parametrize("padded", [False, True])
def test_prepared_attention_contract(verifier, aligned, tmp_path, padded):
    case = aligned[0]["cases"][0]
    path = tmp_path / case["file"]
    ids = np.array([[10, 11, 12]], dtype=np.int64)
    attention = np.array([[1, 1, 0 if padded else 1]], dtype=np.int64)
    np.savez(path, input_ids=ids, attention_mask=attention)
    if padded:
        with pytest.raises(ValueError, match="all-ones, unpadded attention_mask"):
            verifier["load_case_arrays"](tmp_path, case)
    else:
        arrays, digest = verifier["load_case_arrays"](tmp_path, case)
        np.testing.assert_array_equal(arrays["input_ids"], ids)
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_failed_preflight_writes_report_before_mlx_import(verifier, aligned, tmp_path, monkeypatch):
    manifest, reference = aligned
    reference["dataset_sha256"] = "another-dataset"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    ref_path = tmp_path / "reference.json"
    ref_path.write_text(json.dumps(reference))
    output = tmp_path / "report.json"
    original_import = builtins.__import__

    def forbid_mlx(name, *args, **kwargs):
        if name == "mlx" or name.startswith("mlx.") or name == "mlx_vlm":
            pytest.fail("failed preflight must not initialize MLX")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_mlx)
    result = verifier["main"](
        [
            "--inputs",
            str(tmp_path),
            "--model",
            str(tmp_path / "unloaded-model"),
            "--reference",
            str(ref_path),
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())
    assert result == 1
    assert report["status"] == "error"
    assert report["parity_accepted"] is False
    assert report["failure"]["stage"] == "preflight"
    assert "dataset hashes differ" in report["failure"]["message"]
