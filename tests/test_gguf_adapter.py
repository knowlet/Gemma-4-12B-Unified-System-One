"""Offline bridge regressions: exact media pixels, capture identity and typed S1 outputs."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from s1.contracts import DecisionRequest


@pytest.fixture(scope="module")
def adapter():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/gguf_adapter.py"))


def patches_for(rgb):
    rows, cols = rgb.shape[0] // 48, rgb.shape[1] // 48
    patches = rgb.reshape(rows, 48, cols, 48, 3).transpose(0, 2, 1, 3, 4).reshape(
        rows * cols, 48 * 48 * 3
    ).astype(np.float32) * np.float32(1 / 255)
    positions = np.array([(x, y) for y in range(rows) for x in range(cols)])
    return patches, positions


def test_rgb_bridge_retains_pixels_positions_and_omits_padding(adapter):
    rgb = np.random.default_rng(42).integers(0, 256, size=(96, 144, 3), dtype=np.uint8)
    patches, positions = patches_for(rgb)
    patches = np.pad(patches, ((0, 2), (0, 0)))
    positions = np.pad(positions, ((0, 2), (0, 0)), constant_values=-1)
    np.testing.assert_array_equal(adapter["processed_image_rgb"](patches, positions), rgb)


@pytest.mark.parametrize("invalid", ["non_uint8", "position_order", "position_hole", "bad_padding"])
def test_rgb_bridge_rejects_changes_that_native_processing_would_hide(adapter, invalid):
    patches, positions = patches_for(np.zeros((48, 96, 3), dtype=np.uint8))
    if invalid == "non_uint8":
        patches[0, 0] = 0.123456
    elif invalid == "position_order":
        positions = positions[::-1]
    elif invalid == "position_hole":
        positions[1, 0] = 2
    else:
        positions[1] = [-1, 0]
    with pytest.raises(ValueError):
        adapter["processed_image_rgb"](patches, positions)


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def tolist(self):
        return self.values.tolist()

    def numpy(self):
        return self.values


def fixture_payload():
    request = DecisionRequest.model_validate(
        {
            "state": "state",
            "questions": [{"id": "check", "type": "noul", "instructions": "Check"}],
            "media": [{"type": "image", "data": "aW1hZ2U="}, {"type": "audio", "samples": [0.25]}],
        }
    )
    patches, positions = patches_for(np.full((48, 48, 3), 123, dtype=np.uint8))
    inputs = {
        "input_ids": Tensor([[2, 100, 1000, 101, 200, 2000, 201, 3, 4, 5]]),
        "mm_token_type_ids": Tensor([[0, 0, 1, 0, 0, 3, 0, 0, 0, 0]]),
        "pixel_values": Tensor([patches]),
        "image_position_ids": Tensor([positions]),
    }
    envelope = {
        "case_id": "case",
        "question_ids": ["check"],
        "slots": [9],
        "nopts": [2],
        "letters": list(range(52)),
        "input_ids": inputs["input_ids"][0].tolist(),
        "has_media": True,
        "source_input_token_count": 10,
    }
    return request, inputs, envelope


def test_mixed_media_bridge_keeps_hf_control_tokens_and_media_order(adapter, tmp_path):
    request, inputs, envelope = fixture_payload()
    payload = adapter["make_native_payload"](
        envelope, request, inputs, [9], [2], list(range(52)), tmp_path
    )
    assert [part["type"] for part in payload["parts"]] == ["text", "image", "text", "audio", "text"]
    assert payload["parts"][0]["tokens"] == [2, 100]
    assert payload["parts"][2]["tokens"] == [101, 200]
    assert payload["parts"][4]["tokens"] == [201, 3, 4, 5]
    assert payload["parts"][3]["samples"] == [0.25]
    assert Path(payload["parts"][1]["rgb_path"]).read_bytes() == bytes([123]) * (48 * 48 * 3)
    assert payload["slots"] == [9]


@pytest.mark.parametrize(
    "key", ["slots", "input_ids", "letters", "question_ids", "has_media", "gold"]
)
def test_capture_identity_and_gold_leak_are_rejected(adapter, tmp_path, key):
    request, inputs, envelope = fixture_payload()
    envelope[key] = False if key == "has_media" else ["incorrect"]
    with pytest.raises(ValueError):
        adapter["make_native_payload"](
            envelope, request, inputs, [9], [2], list(range(52)), tmp_path
        )


def test_typed_answers_apply_temperature_once_to_already_softcapped_logits(adapter):
    request = DecisionRequest.model_validate(
        {
            "state": "test",
            "questions": [
                {"id": "c", "instructions": "Choose", "criteria": ["left", "right"]},
                {"id": "n", "type": "noul", "instructions": "Is it true?"},
                {
                    "id": "s",
                    "type": "score",
                    "instructions": "Score",
                    "criteria": {"0": "low", "10": "high"},
                },
            ],
        }
    )
    response = {
        "status": "ok",
        "raw_logits": [[0, 2], [0, 2], [0, 2]],
        "latency_ms": 12,
        "runtime": {},
        "runtime_slots": [10, 20, 30],
        "native_token_count": 31,
        "preprocessing": {},
    }
    result = adapter["prediction_from_logits"](request, response, 2.0, "model.gguf")
    probability = 1 / (1 + np.exp(-1))
    assert result["answers"]["c"]["choice"] == "right"
    assert result["answers"]["n"]["noul"] == pytest.approx(probability)
    assert result["answers"]["s"]["score"] == pytest.approx(10 * probability)
    assert result["passes"] == 1


@pytest.mark.parametrize("temperature", [None, True, 0, -1, float("inf"), "1.0"])
def test_predictions_require_explicit_valid_calibration(adapter, tmp_path, temperature):
    path = tmp_path / "s1_config.json"
    path.write_text(json.dumps({"temperature": temperature}))
    with pytest.raises(ValueError):
        adapter["load_calibration"](path)


def test_missing_calibration_does_not_fallback_to_one(adapter, tmp_path):
    with pytest.raises(FileNotFoundError):
        adapter["load_calibration"](tmp_path / "s1_config.json")


def calibration_fixture(directory):
    (directory / "model.gguf").write_bytes(b"model")
    (directory / "mmproj.gguf").write_bytes(b"media")
    receipt = {
        "status": "converted_and_structurally_verified",
        "model_file": "model.gguf",
        "mmproj_file": "mmproj.gguf",
        "files": [
            {
                "path": "model.gguf",
                "size_bytes": 5,
                "sha256": hashlib.sha256(b"model").hexdigest(),
            },
            {
                "path": "mmproj.gguf",
                "size_bytes": 5,
                "sha256": hashlib.sha256(b"media").hexdigest(),
            },
        ],
    }
    manifest = directory / "conversion-manifest.json"
    manifest.write_text(json.dumps(receipt))
    return {
        "temperature": 1.25,
        "runtime": "llama.cpp",
        "runtime_format": "gguf",
        "conversion_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }


def test_gguf_owned_calibration_binds_selected_conversion(adapter, tmp_path):
    path = tmp_path / "s1_config.json"
    path.write_text(json.dumps(calibration_fixture(tmp_path)))
    assert (
        adapter["load_calibration"](
            path, model=tmp_path / "model.gguf", mmproj=tmp_path / "mmproj.gguf"
        )
        == 1.25
    )


@pytest.mark.parametrize(
    "invalid",
    ["source_config", "different_format", "different_digest", "changed_file", "different_file"],
)
def test_foreign_or_mismatched_calibration_rejected(adapter, tmp_path, invalid):
    config = calibration_fixture(tmp_path)
    model = tmp_path / "model.gguf"
    if invalid == "source_config":
        config = {"temperature": 1.25, "source_model": "BF16 source"}
    elif invalid == "different_format":
        config["runtime_format"] = "mlx"
    elif invalid == "different_digest":
        config["conversion_manifest_sha256"] = "0" * 64
    elif invalid == "changed_file":
        model.write_bytes(b"truncated")
    else:
        model = tmp_path / "another.gguf"
        model.write_bytes(b"model")
    path = tmp_path / "s1_config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        adapter["load_calibration"](path, model=model, mmproj=tmp_path / "mmproj.gguf")


@pytest.mark.parametrize(
    "modified", [None, "chat_template.jinja", "tokenizer.json", "processor_config.json"]
)
def test_processor_identity_accepts_equivalent_directory_and_rejects_input_changes(
    adapter, tmp_path, modified
):
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    files = []
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "processor_config.json",
    ):
        content = f"original {name}".encode()
        (alternate / name).write_bytes(content)
        files.append(
            {
                "path": name,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    (tmp_path / "conversion-manifest.json").write_text(json.dumps({"files": files}))
    if modified:
        (alternate / modified).write_text("replacement")
        with pytest.raises(ValueError, match="selected processor"):
            adapter["validate_processor_identity"](alternate, tmp_path)
    else:
        adapter["validate_processor_identity"](alternate, tmp_path)
