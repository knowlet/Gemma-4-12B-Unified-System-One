"""Research-only USER-question ablation; the SDK's default prompt is unchanged.

``wrap_model(model, mode)`` retains native prepare/logits and gradient behavior.
``predict(model, request, mode)`` uses the existing independent G4 batching and
candidate readout. The user_question variant changes only question placement:
state, native media and the same question/options are inside USER, followed by
an ASSISTANT ``Answer: (`` slot. Shallow instance copies share immutable model
weights and the processor without rebinding methods on the supplied model.
"""

from __future__ import annotations

import base64
import copy
import io
import json
from types import MethodType

from s1.contracts import AudioInput
from s1.errors import RequestValidationError
from s1.evaluation.gemma import predict_batch
from s1.schema import LETTERS

MODES = ("current", "user_question")


def prepare_user_question(model, request):
    """Prepare one independent question, preserving native processor tensors."""
    import numpy as np
    import torch
    from PIL import Image, UnidentifiedImageError

    if len(request.questions) != 1:
        raise RequestValidationError("user_question research preparation requires one question")
    question = request.questions[0]
    state = (
        request.state
        if isinstance(request.state, str)
        else json.dumps(request.state, ensure_ascii=False)
    )
    content = [{"type": "text", "text": state}]
    images, audio = [], []
    for media in request.media:
        if isinstance(media, AudioInput):
            content.append({"type": "audio"})
            audio.append(np.asarray(media.samples, dtype=np.float32))
        else:
            if media.timestamp_seconds is not None:
                content.append({"type": "text", "text": f"Frame at {media.timestamp_seconds:g}s:"})
            try:
                raw = base64.b64decode(media.data, validate=True)
            except ValueError as exc:
                raise RequestValidationError("invalid base64 image") from exc
            try:
                with Image.open(io.BytesIO(raw)) as image:
                    if image.width * image.height > 16_000_000:
                        raise RequestValidationError("image exceeds 16 megapixels")
                    images.append(image.convert("RGB"))
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise RequestValidationError("invalid or oversized image") from exc
            content.append({"type": "image"})
    lines = [f"\nQuestion ({question.type}): {question.instructions}\n"]
    for letter, label, description in zip(LETTERS, question.labels(), question.descriptions()):
        lines.append(f"({letter}) {label}: {description}\n")
    content.append({"type": "text", "text": "".join(lines)})
    text = model.processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    kwargs = {"text": text, "return_tensors": "pt", "add_special_tokens": False}
    if images:
        kwargs["images"] = images
    if audio:
        kwargs.update(audio=audio, sampling_rate=16000)
    inputs = dict(model.processor(**kwargs))
    # Measure after the complete processor expands all native media placeholders.
    prefix_length = inputs["input_ids"].shape[1]
    suffix = model.tok.encode("\nAnswer: (", add_special_tokens=False)
    length = prefix_length + len(suffix)
    if length > model.max_context:
        raise RequestValidationError(
            f"request exceeds {model.max_context} context tokens; nothing was truncated"
        )
    inputs["input_ids"] = torch.cat(
        [inputs["input_ids"], torch.tensor([suffix], dtype=inputs["input_ids"].dtype)], dim=1
    )
    for key, fill in (("attention_mask", 1), ("mm_token_type_ids", 0), ("token_type_ids", 0)):
        if key in inputs:
            inputs[key] = torch.cat(
                [inputs[key], inputs[key].new_full((1, len(suffix)), fill)], dim=1
            )
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    return (
        inputs,
        torch.tensor([length - 1], device=model.device),
        torch.tensor([len(question.labels())], device=model.device),
    )


def wrap_model(model, mode="current"):
    """Return a separate instance view; backbone, head and processor are shared.

    The user_question prepare/logits path accepts exactly one question. Inference
    can use ``predict`` to split and batch a complete independent request. Training
    may call this view's ordinary ``logits`` without disabling gradients.
    """
    if mode not in MODES:
        raise ValueError(f"research prompt mode must be one of {MODES}")
    experiment = copy.copy(model)
    if mode == "user_question":
        experiment.prepare = MethodType(prepare_user_question, experiment)
    return experiment


def predict(model, request, mode="current"):
    """Compare independent prompts; preserve head rows, softcap and temperature."""
    experiment = wrap_model(model, mode)
    response = predict_batch(experiment, [request], independent=True, readout="candidate")[0]
    response["execution"]["research_prompt_mode"] = mode
    return response
