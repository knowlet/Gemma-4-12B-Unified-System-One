"""Jev-Omni native candidate, media and immutable-runtime contracts."""

import base64
import io
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from s1.contracts import DecisionRequest
from s1.errors import BackendResponseError, RequestValidationError
from s1.evaluation.adapters import execution_blockers, load_adapter
from s1.evaluation.contracts import ProfileSpec
from s1.evaluation.integrity import validate_runtime_telemetry
from s1.evaluation.jev_omni import (
    CONTEXT_LIMIT,
    MODEL_ID,
    REVISION,
    SOURCE_SHA256,
    JevOmniBackend,
    load_publisher_module,
    normalize_logits,
    request_media,
)
from s1.evaluation.planning import case_eligibility
from s1.evaluation.registry import Registry


@pytest.fixture
def spec():
    return Registry.load(
        "configs/benchmarks/models.toml",
        "configs/benchmarks/suites.toml",
        "configs/benchmarks/profiles.toml",
    ).models["jev-omni-local"]


class Logits:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.float_called = False

    def float(self):
        self.float_called = True
        return self

    def softmax(self, axis):
        assert axis == -1 and self.float_called
        return self

    def tolist(self):
        return self.probabilities


def test_registry_and_native_factory(spec, monkeypatch):
    import s1.evaluation.jev_omni as jev

    assert (spec.model_id, spec.revision) == (MODEL_ID, REVISION)
    assert spec.capabilities.modalities == ("text", "image", "audio")
    assert spec.capabilities.independent_questions is True
    assert execution_blockers(spec, ProfileSpec(id="p", suite="s", models=(spec.id,))) == []
    captured = []
    monkeypatch.setattr(jev, "JevOmniBackend", lambda value: captured.append(value) or object())
    load_adapter(spec, {}).close()
    assert captured == [spec]


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"revision": "a" * 40}, "jev_omni_checkpoint_unverified"),
        ({"context_limit": CONTEXT_LIMIT + 1}, "jev_omni_context_limit_exceeded"),
        ({"device": "cpu"}, "jev_omni_requires_cuda_bfloat16"),
        ({"precision": "float32"}, "jev_omni_requires_cuda_bfloat16"),
        ({"calibration": "none"}, "jev_omni_policy_mismatch"),
    ],
)
def test_invalid_configuration_fails_before_download(spec, changes, reason):
    changed = spec.model_copy(update=changes)
    assert reason in execution_blockers(changed, ProfileSpec(id="p", suite="s", models=(spec.id,)))
    with pytest.raises(ValueError, match=reason):
        JevOmniBackend(changed)


def test_changed_publisher_code_fails_before_import(tmp_path):
    (tmp_path / "jev_omni.py").write_text("raise AssertionError('must not import')\n")
    with pytest.raises(ValueError, match="reviewed immutable revision"):
        load_publisher_module(tmp_path)


def test_boolean_choice_and_numeric_score_keep_original_order(decision_request):
    questions = {question.id: question for question in decision_request.questions}
    assert (
        normalize_logits(questions["refund"], Logits([0.12345679, 0.87654321]))["noul"]
        == 0.87654321
    )
    choice = normalize_logits(questions["route"], Logits([0.2, 0.8]))
    assert choice["choice"] == "technical"
    assert choice["probabilities"] == {"billing": 0.2, "technical": 0.8}
    score = normalize_logits(questions["urgency"], Logits([0.25, 0.75]))
    assert score["probabilities"] == {"10": 0.25, "20": 0.75}
    assert score["score"] == 17.5


def test_duplicate_descriptions_remain_distinct_slots():
    request = DecisionRequest.model_validate(
        {
            "state": "Two labels sharing a description",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Choose",
                    "criteria": {
                        "a": "same",
                        "b": "same",
                    },
                }
            },
        }
    )
    result = normalize_logits(request.questions[0], Logits([0.7, 0.3]))
    assert result["probabilities"] == {"a": 0.7, "b": 0.3}


@pytest.mark.parametrize("values", [[1.0], [0.7, 0.7], [float("nan"), 0.5], [-0.1, 1.1]])
def test_invalid_probabilities_never_become_scores(decision_request, values):
    with pytest.raises(BackendResponseError, match="invalid candidate probabilities"):
        normalize_logits(decision_request.questions[0], Logits(values))


def test_audio_preserves_samples_and_rate(spec, decision_request):
    request = DecisionRequest.model_validate(
        {
            **decision_request.model_dump(),
            "media": [
                {"type": "audio", "samples": [0.0, 0.125, -0.875, 1.0], "sampling_rate": 16000},
            ],
        }
    )
    content, kwargs = request_media(request)
    assert content == [{"type": "audio"}]
    assert kwargs["sampling_rate"] == 16000
    assert kwargs["audio"][0].dtype == np.float32
    assert kwargs["audio"][0].tolist() == request.media[0].samples
    assert (
        case_eligibility(spec, SimpleNamespace(id="audio", request=request))["eligibility"]
        == "eligible"
    )


def test_image_preserves_native_pixels(spec, decision_request):
    image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image.new("RGB", (3, 2), (10, 20, 30)).save(buffer, format="PNG")
    request = DecisionRequest.model_validate(
        {
            **decision_request.model_dump(),
            "media": [
                {"type": "image", "data": base64.b64encode(buffer.getvalue()).decode()},
            ],
        }
    )
    content, kwargs = request_media(request)
    assert content == [{"type": "image"}]
    assert kwargs["images"][0].size == (3, 2)
    assert kwargs["images"][0].getpixel((0, 0)) == (10, 20, 30)
    assert (
        case_eligibility(spec, SimpleNamespace(id="image", request=request))["eligibility"]
        == "eligible"
    )


@pytest.mark.parametrize(
    "media,reason",
    [
        (
            [{"type": "audio", "samples": [0.0]}, {"type": "image", "data": "not-decoded"}],
            "multiple_media",
        ),
        (
            [{"type": "image", "data": "not-decoded", "timestamp_seconds": 1.0}],
            "timestamped_frames",
        ),
    ],
)
def test_unsupported_media_is_ineligible_before_decoding(spec, decision_request, media, reason):
    request = DecisionRequest.model_validate({**decision_request.model_dump(), "media": media})
    eligibility = case_eligibility(spec, SimpleNamespace(id="unsupported", request=request))
    assert eligibility["eligibility"] == "unsupported"
    assert any(reason in item for item in eligibility["reasons"])
    with pytest.raises(RequestValidationError, match=reason):
        JevOmniBackend.__new__(JevOmniBackend).predict(request)


@pytest.mark.parametrize("count", [20, 21])
def test_publisher_quality_option_limit_is_explicit(spec, count):
    request = DecisionRequest.model_validate(
        {
            "state": "Options",
            "questions": {
                "q": {
                    "instructions": "Choose",
                    "criteria": [f"value-{index}" for index in range(count)],
                }
            },
        }
    )
    eligibility = case_eligibility(spec, SimpleNamespace(id="options", request=request))
    assert eligibility["eligibility"] == ("eligible" if count == 20 else "unsupported")
    if count > 20:
        assert eligibility["reasons"] == ["max_options_exceeded"]
        with pytest.raises(RequestValidationError, match="max_options"):
            request_media(request)


@pytest.mark.parametrize("length", [CONTEXT_LIMIT, CONTEXT_LIMIT + 1])
def test_native_forward_preserves_question_prompts_and_expanded_context(
    spec, monkeypatch, decision_request, length
):
    calls, prompts = [], []
    backend = JevOmniBackend.__new__(JevOmniBackend)
    backend.spec = spec
    backend.engine = SimpleNamespace(_capture={}, _extra={"logits_to_keep": 1})
    backend.publisher = SimpleNamespace(
        _prompt=lambda state, question, options: json.dumps([state, question, options])
    )

    class Input:
        shape = (1, length)

        def to(self, device, **kwargs):
            calls.append("transfer")
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            prompts.append(json.loads(messages[0]["content"][-1]["text"]))
            return "formatted"

        def __call__(self, **kwargs):
            assert kwargs == {
                "text": "formatted",
                "return_tensors": "pt",
                "add_special_tokens": False,
            }
            return {"input_ids": Input()}

    def model(**kwargs):
        assert kwargs["use_cache"] is False and kwargs["logits_to_keep"] == 1
        backend.engine._capture["hidden"] = "last hidden state"
        calls.append("forward")

    class HeadResult:
        def __getitem__(self, indices):
            assert indices == (0, slice(None, 2))
            return Logits([0.25, 0.75])

    def head(hidden, counts):
        assert hidden == "last hidden state" and counts == [2]
        return HeadResult()

    backend.processor, backend.model, backend.head = Processor(), model, head
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            inference_mode=nullcontext,
            autocast=lambda *args, **kwargs: nullcontext(),
            bfloat16="bfloat16",
            is_floating_point=lambda tensor: False,
            tensor=lambda values, **kwargs: values,
        ),
    )
    if length > CONTEXT_LIMIT:
        with pytest.raises(RequestValidationError, match="nothing was truncated"):
            backend.predict(decision_request)
        assert calls == []  # Reject before CUDA copies or any forward.
    else:
        response = backend.predict(decision_request)
        assert calls.count("forward") == len(decision_request.questions)
        assert response["answers"]["refund"]["noul"] == 0.75
        assert response["answers"]["urgency"]["score"] == 17.5
        assert response["usage"] == {"input_tokens": length * 3, "output_tokens": 0}
        assert prompts == [
            [decision_request.state, question.instructions, question.descriptions()]
            for question in decision_request.questions
        ]


def test_runtime_identity_records_mixed_head_precision_and_native_policy(spec):
    backend = JevOmniBackend.__new__(JevOmniBackend)
    backend.spec, backend.runtime_precision = spec, "bfloat16"
    result = validate_runtime_telemetry(backend, spec)
    assert result["source_revision"] == REVISION
    assert result["context_policy"] == "reject"
    assert result["runtime_options"]["publisher_source_sha256"] == SOURCE_SHA256
    assert result["runtime_options"]["head_storage_precision"] == "float32"
    assert result["runtime_options"]["one_forward_per_question"] is True
