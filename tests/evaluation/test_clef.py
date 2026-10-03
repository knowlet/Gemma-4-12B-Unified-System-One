import base64
import io
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from s1.contracts import DecisionRequest
from s1.errors import BackendResponseError, RequestValidationError
from s1.evaluation.adapters import execution_blockers, load_adapter
from s1.evaluation.clef import (
    CONTEXT_LIMIT,
    MODEL_ID,
    REVISION,
    ClefBackend,
    load_publisher_module,
    normalize_logits,
    request_record,
)
from s1.evaluation.integrity import validate_runtime_telemetry
from s1.evaluation.planning import case_eligibility
from s1.evaluation.registry import Registry


@pytest.fixture
def spec():
    return Registry.load(
        "configs/benchmarks/models.toml",
        "configs/benchmarks/suites.toml",
        "configs/benchmarks/profiles.toml",
    ).models["clef-local"]


class Logits:
    """Model boundary test double with explicit post-softmax values."""

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def float(self):
        return self

    def softmax(self, axis):
        assert axis == -1
        return self

    def tolist(self):
        return self.probabilities


def test_registry_native_factory_and_identity(spec, monkeypatch):
    import s1.evaluation.clef as clef
    from s1.evaluation.contracts import ProfileSpec

    assert (spec.model_id, spec.revision) == (MODEL_ID, REVISION)
    assert spec.capabilities.modalities == ("text", "image")
    assert spec.capabilities.independent_questions is False
    assert execution_blockers(spec, ProfileSpec(id="p", suite="s", models=(spec.id,))) == []
    captured = []
    monkeypatch.setattr(clef, "ClefBackend", lambda value: captured.append(value) or object())
    load_adapter(spec, {}).close()
    assert captured == [spec]


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"revision": "a" * 40}, "clef_checkpoint_unverified"),
        ({"context_limit": CONTEXT_LIMIT + 1}, "clef_context_limit_exceeded"),
        ({"device": "cpu"}, "clef_requires_cuda_bfloat16"),
        ({"precision": "float32"}, "clef_requires_cuda_bfloat16"),
        ({"calibration": "checkpoint"}, "clef_policy_mismatch"),
    ],
)
def test_invalid_configuration_rejected_before_loading(spec, changes, reason):
    from s1.evaluation.contracts import ProfileSpec

    changed = spec.model_copy(update=changes)
    assert reason in execution_blockers(changed, ProfileSpec(id="p", suite="s", models=(spec.id,)))
    with pytest.raises(ValueError, match=reason):
        ClefBackend(changed)


def test_joint_question_claim_rejected(spec):
    from s1.evaluation.clef import configuration_blockers

    changed = spec.model_copy(
        update={
            "capabilities": spec.capabilities.model_copy(update={"independent_questions": True})
        }
    )
    assert "clef_questions_are_joint" in configuration_blockers(changed)


def test_changed_publisher_code_rejected_before_import(tmp_path):
    (tmp_path / "joint_schema_model.py").write_text("raise AssertionError('must not import')\n")
    with pytest.raises(ValueError, match="reviewed immutable revision"):
        load_publisher_module(tmp_path)


def test_native_boolean_choice_and_numeric_score_order():
    request = DecisionRequest.model_validate(
        {
            "state": {"message": "Preserve JSON"},
            "questions": [
                {"id": "truth", "type": "noul", "instructions": "True?"},
                {
                    "id": "route",
                    "type": "choice",
                    "instructions": "Route?",
                    "criteria": {"zebra": "Last alphabetically", "alpha": "First alphabetically"},
                },
                {
                    "id": "priority",
                    "type": "score",
                    "instructions": "Priority?",
                    "criteria": {"10": "Low", "20": "High"},
                },
            ],
        }
    )
    record = request_record(request)
    assert record["state"] == request.state
    assert record["questions"]["priority"]["criteria"] == ["Low", "High"]
    encoded = SimpleNamespace(
        questions=[
            SimpleNamespace(question_id="truth", option_ids=("true", "false")),
            SimpleNamespace(question_id="route", option_ids=("alpha", "zebra")),
            SimpleNamespace(question_id="priority", option_ids=("0", "1")),
        ]
    )
    result = normalize_logits(
        request,
        encoded,
        [
            Logits([0.87654321, 0.12345679]),
            Logits([0.8, 0.2]),
            Logits([0.25, 0.75]),
        ],
    )["answers"]
    assert result["truth"]["noul"] == 0.87654321  # No four-decimal wire rounding.
    assert result["truth"]["probabilities"] == {"false": 0.12345679, "true": 0.87654321}
    assert result["route"]["probabilities"] == {"zebra": 0.2, "alpha": 0.8}
    assert result["route"]["choice"] == "alpha"
    assert result["priority"]["probabilities"] == {"10": 0.25, "20": 0.75}
    assert result["priority"]["score"] == 17.5


def test_audio_is_ineligible_and_rejected_before_model_access(spec):
    request = DecisionRequest.model_validate(
        {
            "state": "Audio",
            "questions": [{"id": "q", "type": "noul", "instructions": "True?"}],
            "media": [{"type": "audio", "samples": [0.0], "sampling_rate": 16000}],
        }
    )
    result = case_eligibility(spec, SimpleNamespace(id="audio", request=request))
    assert result["eligibility"] == "unsupported"
    assert result["reasons"] == ["unsupported_modality:audio"]
    with pytest.raises(RequestValidationError, match="not audio"):
        ClefBackend.__new__(ClefBackend).predict(request)


@pytest.mark.parametrize("timestamps", [(None, None), (1.25, 2.5)])
def test_native_images_preserve_pixels_and_frame_timestamps(spec, timestamps):
    image_module = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image_module.new("RGB", (2, 3), (10, 20, 30)).save(buffer, format="PNG")
    request = DecisionRequest.model_validate(
        {
            "state": "Inspect these pixels.",
            "questions": [{"id": "q", "type": "noul", "instructions": "True?"}],
            "media": [
                {
                    "type": "image",
                    "data": base64.b64encode(buffer.getvalue()).decode(),
                    "timestamp_seconds": timestamp,
                }
                for timestamp in timestamps
            ],
        }
    )
    record = request_record(request)
    assert len(record["images"]) == 2
    assert all(image.size == (2, 3) for image in record["images"])
    assert all(image.getpixel((0, 0)) == (10, 20, 30) for image in record["images"])
    expected_state = (
        request.state
        if timestamps[0] is None
        else {"state": request.state, "image_timestamps_seconds": list(timestamps)}
    )
    assert record["state"] == expected_state
    assert (
        case_eligibility(spec, SimpleNamespace(id="image", request=request))["eligibility"]
        == "eligible"
    )


@pytest.mark.parametrize("timestamps", [(1.25, None), (None, 1.25)])
def test_mixed_image_timestamps_rejected_before_decode_or_model(monkeypatch, timestamps):
    request = DecisionRequest.model_validate(
        {
            "state": "Mixed image timestamps.",
            "questions": [{"id": "q", "type": "noul", "instructions": "True?"}],
            "media": [
                {"type": "image", "data": "not-decoded", "timestamp_seconds": timestamp}
                for timestamp in timestamps
            ],
        }
    )
    monkeypatch.setattr(base64, "b64decode", lambda *args, **kwargs: pytest.fail("decoded image"))
    with pytest.raises(RequestValidationError, match="all have timestamps or all omit"):
        ClefBackend.__new__(ClefBackend).predict(request)


@pytest.mark.parametrize("length", [CONTEXT_LIMIT - 1, CONTEXT_LIMIT, CONTEXT_LIMIT + 1])
def test_full_encoding_context_guard_includes_expanded_media(spec, monkeypatch, length):
    request = DecisionRequest.model_validate(
        {
            "state": "Complete input",
            "questions": [{"id": "q", "type": "noul", "instructions": "True?"}],
        }
    )
    encoded = SimpleNamespace(
        input_ids=range(length),
        media={"pixel_values": object()},
        questions=[
            SimpleNamespace(question_id="q", option_ids=("true", "false")),
        ],
    )
    calls = []

    def encode(tokenizer, record, **kwargs):
        assert record["state"] == "Complete input"
        assert kwargs["max_length"] == 2**31 - 1
        assert kwargs["max_state_tokens"] is None
        return encoded

    def collate(records, pad, device):
        assert records[0] is encoded  # Preserve native expanded media without re-encoding.
        assert pad == 0 and device == "cuda"
        calls.append("collate")
        return {"records": records, "media": encoded.media}

    def model(batch):
        assert batch["media"] is encoded.media
        calls.append("model")
        return [[Logits([0.75, 0.25])]]

    backend = ClefBackend.__new__(ClefBackend)
    backend.spec = spec
    backend.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    backend.publisher = SimpleNamespace(encode_record=encode, collate_records=collate)
    backend.model = model
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(inference_mode=nullcontext, device=lambda value: value),
    )
    if length > CONTEXT_LIMIT:
        with pytest.raises(RequestValidationError, match="nothing was truncated"):
            backend.predict(request)
        assert calls == []
    else:
        response = backend.predict(request)
        assert response["answers"]["q"]["noul"] == 0.75
        assert response["usage"]["input_tokens"] == length
        assert calls == ["collate", "model"]


def test_invalid_logits_are_not_reported_as_valid_predictions():
    request = DecisionRequest.model_validate(
        {
            "state": "A fact",
            "questions": [{"id": "q", "type": "noul", "instructions": "True?"}],
        }
    )
    encoded = SimpleNamespace(
        questions=[SimpleNamespace(question_id="q", option_ids=("true", "false"))]
    )
    with pytest.raises(BackendResponseError, match="invalid probabilities"):
        normalize_logits(request, encoded, [Logits([float("nan"), 0.5])])
    encoded.questions[0].option_ids = ("true", "true")
    with pytest.raises(BackendResponseError, match="option IDs"):
        normalize_logits(request, encoded, [Logits([0.5, 0.5])])


def test_clef_runtime_revision_must_match_plan(spec):
    backend = ClefBackend.__new__(ClefBackend)
    backend.spec, backend.runtime_precision = spec, "bfloat16"
    telemetry = validate_runtime_telemetry(backend, spec)
    assert telemetry["context_policy"] == "reject"
    assert telemetry["runtime_options"]["joint_questions"] is True
    wrong = spec.model_copy(update={"revision": "a" * 40})
    with pytest.raises(ValueError, match="revision differs"):
        validate_runtime_telemetry(backend, wrong)
