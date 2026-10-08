"""Pinned Jev-Omni native prompt, multimodal backbone and decision-head readout."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import sys
from importlib import metadata
from pathlib import Path

from s1.contracts import AudioInput, answer_from_probabilities
from s1.errors import BackendResponseError, RequestValidationError

MODEL_ID = "akhilaaa3/Jev-Omni"
REVISION = "5addda86ddee081a68fb067477ea100c221b8917"
SOURCE_SHA256 = "11d761b0b6cefc8aac29757f43af4b2b02af8f9b6c6dad19834c0b14538180f0"
CONTEXT_LIMIT = 16384
MAX_OPTIONS = 20  # Publisher reports quality support only through 20 candidates.


def configuration_blockers(spec):
    reasons = []
    if (spec.model_id, spec.revision) != (MODEL_ID, REVISION):
        reasons.append("jev_omni_checkpoint_unverified")
    if spec.execution_mode != "sequential" or spec.calibration != "checkpoint":
        reasons.append("jev_omni_policy_mismatch")
    if not (spec.device or "").startswith("cuda") or spec.precision != "bfloat16":
        reasons.append("jev_omni_requires_cuda_bfloat16")
    if spec.context_limit > CONTEXT_LIMIT:
        reasons.append("jev_omni_context_limit_exceeded")
    if spec.capabilities.probabilities != "complete":
        reasons.append("jev_omni_requires_complete_probabilities")
    if spec.capabilities.max_options is None or spec.capabilities.max_options > MAX_OPTIONS:
        reasons.append("jev_omni_option_limit_unverified")
    if spec.subfolder or (spec.processor_revision and spec.processor_revision != spec.revision):
        reasons.append("jev_omni_processor_override_unavailable")
    return reasons


def request_blockers(request):
    reasons = []
    if len(request.media) > 1:
        reasons.append("jev_omni_multiple_media_unsupported")
    if any(getattr(media, "timestamp_seconds", None) is not None for media in request.media):
        reasons.append("jev_omni_timestamped_frames_unsupported")
    if any(len(question.labels()) > MAX_OPTIONS for question in request.questions):
        reasons.append("max_options_exceeded")
    return reasons


def load_publisher_module(directory):
    path = Path(directory) / "jev_omni.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Jev-Omni publisher source differs from the reviewed immutable revision")
    name = f"_s1_jev_omni_{REVISION}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Jev-Omni publisher module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def request_media(request):
    """Decode native media once; never truncate, transcribe or caption inputs."""
    reasons = request_blockers(request)
    if reasons:
        raise RequestValidationError("unsupported Jev-Omni request: " + ", ".join(reasons))
    if not request.media:
        return [], {}
    media = request.media[0]
    if isinstance(media, AudioInput):
        import numpy as np

        return [{"type": "audio"}], {
            "audio": [np.asarray(media.samples, dtype=np.float32)],
            "sampling_rate": media.sampling_rate,
        }
    from PIL import Image, UnidentifiedImageError

    try:
        raw = base64.b64decode(media.data, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 16_000_000:
                raise RequestValidationError("image exceeds 16 megapixels")
            decoded = image.convert("RGB")
    except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise RequestValidationError("invalid or oversized image") from exc
    return [{"type": "image"}], {"images": [decoded]}


def normalize_logits(question, logits):
    # The native head is positional, so labels and duplicate descriptions cannot
    # be lost through the publisher's dict(zip(options, probabilities)) output.
    values = logits.float().softmax(-1).tolist()
    try:
        return answer_from_probabilities(question, values)
    except (ValueError, TypeError) as exc:
        raise BackendResponseError("Jev-Omni returned invalid candidate probabilities") from exc


class JevOmniBackend:
    def __init__(self, spec):
        reasons = configuration_blockers(spec)
        if reasons:
            raise ValueError("invalid Jev-Omni configuration: " + ", ".join(reasons))
        import torch
        import transformers
        from huggingface_hub import snapshot_download

        self.spec = spec
        directory = snapshot_download(
            MODEL_ID,
            revision=REVISION,
            allow_patterns=[
                "jev_omni.py",
                "config.json",
                "generation_config.json",
                "model*.safetensors*",
                "processor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "decision_config.json",
                "head.pt",
            ],
        )
        self.publisher = load_publisher_module(directory)
        config = transformers.AutoConfig.from_pretrained(directory, local_files_only=True)
        self.model = (
            getattr(transformers, config.architectures[0])
            .from_pretrained(
                directory, dtype=torch.bfloat16, device_map=spec.device, local_files_only=True
            )
            .eval()
        )
        decision = json.loads((Path(directory) / "decision_config.json").read_text())
        self.head = self.publisher._Head256(decision["hidden_size"]).to(spec.device).eval()
        self.head.load_state_dict(
            torch.load(Path(directory) / "head.pt", map_location=spec.device, weights_only=True),
            strict=True,
        )
        self.processor = transformers.AutoProcessor.from_pretrained(
            directory, local_files_only=True
        )
        _, decoder = self.publisher._find_backbone(self.model)
        self.engine = self.publisher.JevOmni(
            self.model, self.head, self.processor, decoder, spec.device
        )
        self.runtime_precision = str(next(self.model.parameters()).dtype).removeprefix("torch.")
        if self.runtime_precision != spec.precision:
            raise ValueError("Jev-Omni loaded precision differs from the requested precision")

    def predict(self, request):
        content, media_kwargs = request_media(request)
        state = (
            request.state
            if isinstance(request.state, str)
            else json.dumps(request.state, ensure_ascii=False)
        )
        answers, input_tokens = {}, 0
        import torch

        for question in request.questions:
            options = question.descriptions()
            prompt = self.publisher._prompt(state, question.instructions, options)
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": [*content, {"type": "text", "text": prompt}]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = dict(
                self.processor(
                    text=text, return_tensors="pt", add_special_tokens=False, **media_kwargs
                )
            )
            length = inputs["input_ids"].shape[1]
            if length > self.spec.context_limit:
                raise RequestValidationError(
                    "Jev-Omni context limit exceeded; nothing was truncated"
                )
            input_tokens += length
            inputs = {
                key: value.to(self.spec.device, dtype=torch.bfloat16)
                if torch.is_floating_point(value)
                else value.to(self.spec.device)
                for key, value in inputs.items()
            }
            self.engine._capture.clear()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.model(**inputs, use_cache=False, **self.engine._extra)
                logits = self.head(
                    self.engine._capture["hidden"],
                    torch.tensor([len(options)], device=self.spec.device),
                )[0, : len(options)]
            answers[question.id] = normalize_logits(question, logits)
        return {
            "answers": answers,
            "model": MODEL_ID,
            "revision": REVISION,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }

    def telemetry(self):
        packages = {}
        for package in ("torch", "torchvision", "transformers"):
            try:
                packages[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                packages[package] = None
        return {
            "revision": REVISION,
            "device": self.spec.device,
            "precision": self.runtime_precision,
            "compute_precision": "bfloat16_autocast",
            "temperature": 1.0,
            "context_limit": self.spec.context_limit,
            "context_policy": "reject",
            "source_revision": REVISION,
            "runtime_package": "akhilaaa3/Jev-Omni/jev_omni.py",
            "checkpoint_task": "multimodal_typed_decisions",
            "compute_backend": "torch_reference",
            "runtime_options": {
                "publisher_source_sha256": SOURCE_SHA256,
                "packages": packages,
                "head_storage_precision": "float32",
                "one_forward_per_question": True,
                "probability_postprocessing": "float32_softmax_without_wire_rounding",
                "audio_preprocessing": "native_16000hz_float32_no_ffmpeg_roundtrip",
                "max_options": MAX_OPTIONS,
                "max_media": 1,
                "cuda_graphs": False,
                "torch_compile": False,
                "persistent_prefix_cache": False,
            },
        }

    def resources(self):
        import torch

        from s1.resources import memory_snapshot

        return memory_snapshot(
            torch.nn.ModuleList([self.model, self.head]), device=self.spec.device
        )

    def close(self):
        self.engine = self.model = self.head = self.processor = self.publisher = None
