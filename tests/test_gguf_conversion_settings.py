"""Reject mislabeled research exports before recording a verified receipt."""

import runpy
from pathlib import Path

import pytest


@pytest.fixture
def recorder(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "record_gguf_conversion.py"))


@pytest.mark.parametrize(
    "quant,file_type",
    [
        ("F16", 1),
        ("Q8_0", 7),
        ("Q4_K_S", 14),
        ("Q4_K_M", 15),
        ("Q5_K_S", 16),
        ("Q5_K_M", 17),
        ("Q6_K", 18),
    ],
)
def test_quant_label_must_match_actual_file_type(recorder, quant, file_type):
    validate = recorder["validate_file_type"]
    validate({"general.file_type": file_type}, quant)
    with pytest.raises(ValueError, match="file_type"):
        validate({"general.file_type": file_type + 1}, quant)


@pytest.mark.parametrize("quant", ["Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K"])
def test_k_quant_cannot_claim_direct_conversion(recorder, quant):
    validate = recorder["validate_quantization_settings"]
    validate(quant, "F16", False, "language.gguf", "mmproj.gguf")
    with pytest.raises(ValueError, match="no-direct-conversion"):
        validate(quant, "F16", True, "language.gguf", "mmproj.gguf")


@pytest.mark.parametrize(
    "model,projector",
    [
        ("../bad.gguf", "mm.gguf"),
        ("a.gguf", "../mm.gguf"),
        ("same.gguf", "same.gguf"),
        ("bad.txt", "mm.gguf"),
    ],
)
def test_conversion_names_cannot_escape_or_overwrite_files(recorder, model, projector):
    with pytest.raises(ValueError):
        recorder["validate_quantization_settings"]("Q8_0", "F16", True, model, projector)


@pytest.mark.parametrize("language,projector", [("not-a-quant", "F16"), ("Q8_0", "Q8_0")])
def test_unverified_format_labels_rejected(recorder, language, projector):
    with pytest.raises(ValueError, match="unsupported"):
        recorder["validate_quantization_settings"](
            language, projector, False, "text.gguf", "mm.gguf"
        )
