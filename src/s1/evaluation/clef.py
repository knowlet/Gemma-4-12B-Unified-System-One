"""Native Clef inference using the reviewed, immutable publisher implementation."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import sys
from importlib import metadata
from pathlib import Path

from s1.contracts import AudioInput, answer_from_probabilities
from s1.errors import BackendResponseError, RequestValidationError

MODEL_ID = "Cloudflare/clef"
REVISION = "2f3de3dd85f379784083b0814d997ab627200f0c"
SOURCE_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
CONTEXT_LIMIT = 16384


def configuration_blockers(spec):
    reasons = []
    if (spec.model_id, spec.revision) != (MODEL_ID, REVISION):
        reasons.append("clef_checkpoint_unverified")
    if spec.execution_mode != "sequential" or spec.calibration != "none":
        reasons.append("clef_policy_mismatch")
    if not (spec.device or "").startswith("cuda") or spec.precision != "bfloat16":
        reasons.append("clef_requires_cuda_bfloat16")
    if spec.context_limit > CONTEXT_LIMIT:
        reasons.append("clef_context_limit_exceeded")
    if spec.capabilities.probabilities != "complete":
        reasons.append("clef_requires_complete_probabilities")
    if spec.capabilities.independent_questions:
        reasons.append("clef_questions_are_joint")
    if spec.subfolder or (spec.processor_revision and spec.processor_revision != spec.revision):
        reasons.append("clef_processor_override_unavailable")
    return reasons


def load_publisher_module(directory):
    """Import only the reviewed source bytes from the pinned checkpoint snapshot."""
    path = Path(directory) / "joint_schema_model.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Clef publisher source differs from the reviewed immutable revision")
    name = f"_s1_clef_{REVISION}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Clef publisher module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def request_record(request):
    if any(isinstance(media, AudioInput) for media in request.media):
        raise RequestValidationError("Clef supports text and images, not audio")
    questions = request.named_questions_payload()
    for question in request.questions:
        if question.type == "score":
            questions[question.id]["criteria"] = question.descriptions()
        elif question.type == "choice":
            questions[question.id]["criteria"] = dict(
                zip(question.labels(), question.descriptions())
            )
    record = {"state": request.state, "questions": questions}
    if request.media:
        from PIL import Image, UnidentifiedImageError

        images = []
        for media in request.media:
            try:
                raw = base64.b64decode(media.data, validate=True)
                with Image.open(io.BytesIO(raw)) as image:
                    if image.width * image.height > 16_000_000:
                        raise RequestValidationError("image exceeds 16 megapixels")
                    images.append(image.convert("RGB"))
            except (
                ValueError,
                UnidentifiedImageError,
                OSError,
                Image.DecompressionBombError,
            ) as exc:
                raise RequestValidationError("invalid or oversized image") from exc
        record["images"] = images
        if any(media.timestamp_seconds is not None for media in request.media):
            # The shared contract represents video as ordered timestamped images.
            record["state"] = {
                "state": request.state,
                "image_timestamps_seconds": [media.timestamp_seconds for media in request.media],
            }
    return record


def normalize_logits(request, encoded, logits):
    """Map publisher ordering back to original labels without rounded wire output."""
    if len(encoded.questions) != len(request.questions) or len(logits) != len(request.questions):
        raise BackendResponseError("Clef returned an unexpected question count")
    requested = {question.id: question for question in request.questions}
    answers = {}
    for encoded_question, question_logits in zip(encoded.questions, logits):
        question = requested.get(encoded_question.question_id)
        if question is None or question.id in answers:
            raise BackendResponseError("Clef returned unexpected or duplicate question IDs")
        option_ids = list(encoded_question.option_ids)
        expected = (
            [str(index) for index in range(len(question.labels()))]
            if question.type == "score"
            else question.labels()
        )
        if len(option_ids) != len(expected) or set(option_ids) != set(expected):
            raise BackendResponseError("Clef returned unexpected option IDs")
        probabilities = question_logits.float().softmax(-1).tolist()
        if len(probabilities) != len(option_ids):
            raise BackendResponseError("Clef returned an unexpected option count")
        distribution = dict(zip(option_ids, probabilities))
        try:
            answers[question.id] = answer_from_probabilities(
                question, [distribution[label] for label in expected]
            )
        except (ValueError, TypeError) as exc:
            raise BackendResponseError("Clef returned invalid probabilities") from exc
    return {"answers": answers}


class ClefBackend:
    def __init__(self, spec):
        reasons = configuration_blockers(spec)
        if reasons:
            raise ValueError("invalid Clef configuration: " + ", ".join(reasons))
        import torch
        from huggingface_hub import snapshot_download

        self.spec = spec
        directory = snapshot_download(MODEL_ID, revision=REVISION)
        self.publisher = load_publisher_module(directory)
        self.model, self.processor = self.publisher.load_release_model(
            directory, device=spec.device, dtype=torch.bfloat16
        )
        self.runtime_precision = str(next(self.model.parameters()).dtype).removeprefix("torch.")
        if self.runtime_precision != spec.precision:
            raise ValueError("Clef loaded precision differs from the requested precision")

    def predict(self, request):
        record = request_record(request)
        # Upstream slices state to max_length. Preserve the complete sequence,
        # including expanded image tokens, and reject before GPU batch allocation.
        encoded = self.publisher.encode_record(
            self.processor.tokenizer,
            record,
            max_length=2**31 - 1,
            max_state_tokens=None,
            processor=self.processor,
        )
        if len(encoded.input_ids) > self.spec.context_limit:
            raise RequestValidationError("Clef context limit exceeded; nothing was truncated")
        import torch

        with torch.inference_mode():
            batch = self.publisher.collate_records(
                [encoded], self.processor.tokenizer.pad_token_id, torch.device(self.spec.device)
            )
            logits = self.model(batch)
        if len(logits) != 1:
            raise BackendResponseError("Clef returned an unexpected batch size")
        response = normalize_logits(request, encoded, logits[0])
        response.update(
            model=MODEL_ID,
            revision=REVISION,
            usage={"input_tokens": len(encoded.input_ids), "output_tokens": 0},
        )
        return response

    def telemetry(self):
        optional = {}
        for name in ("causal-conv1d", "flash-linear-attention"):
            try:
                optional[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                optional[name] = None
        return {
            "revision": self.spec.revision,
            "device": self.spec.device,
            "precision": self.runtime_precision,
            "compute_precision": self.runtime_precision,
            "temperature": 1.0,
            "context_limit": self.spec.context_limit,
            "context_policy": "reject",
            "source_revision": REVISION,
            "runtime_package": "Cloudflare/clef/joint_schema_model.py",
            "checkpoint_task": "multimodal_joint_typed_decisions",
            "compute_backend": "torch_reference",
            "runtime_options": {
                "publisher_source_sha256": SOURCE_SHA256,
                "joint_questions": True,
                "probability_postprocessing": "float32_softmax_without_wire_rounding",
                "cuda_graphs": False,
                "torch_compile": False,
                "persistent_prefix_cache": False,
                "optional_kernel_packages": optional,
            },
        }

    def resources(self):
        from s1.resources import memory_snapshot

        return memory_snapshot(self.model, device=self.spec.device)

    def close(self):
        self.model = self.processor = self.publisher = None
