"""Processor -> multimodal backbone -> answer slots -> selected LM-head rows.

The complete multimodal model (not its language_model child) receives all
processor tensors. No generate(), full-vocabulary projection, or KV cache.
"""

from __future__ import annotations

import base64
import io
import json
import math
import time
from pathlib import Path

from .checkpoint import load_temperature
from .contracts import AudioInput, DecisionRequest, answer_from_probabilities
from .errors import RequestValidationError
from .schema import LETTERS

DEFAULT_MODEL = "google/gemma-4-12B-it"


def candidate_ids(tokenizer) -> list[int]:
    """Validate the actual answer boundary, not just standalone tokenization."""
    prefix = "\nAnswer: ("
    before = tokenizer.encode(prefix, add_special_tokens=False)
    ids = []
    for letter in LETTERS:
        standalone = tokenizer.encode(letter, add_special_tokens=False)
        after = tokenizer.encode(prefix + letter, add_special_tokens=False)
        if len(standalone) != 1 or after != before + standalone:
            raise ValueError(f"candidate {letter!r} is not one stable token at the answer boundary")
        ids.append(standalone[0])
    if len(set(ids)) != len(ids) or tokenizer.unk_token_id in ids:
        raise ValueError("candidate token IDs must be distinct and not unknown")
    return ids


def project_candidates(hidden, head, ids, nopts, softcap=None):
    import torch
    from torch.nn import functional as F

    weight = head.weight[ids]
    bias = head.bias[ids] if getattr(head, "bias", None) is not None else None
    logits = F.linear(hidden.to(weight.dtype), weight, bias)
    if softcap is not None:
        if not math.isfinite(softcap) or softcap <= 0:
            raise ValueError("logit softcap must be positive and finite")
        logits = torch.tanh(logits / softcap) * softcap
    # Match the native LM head's softcap rounding before calibrating in float32.
    logits = logits.float()
    mask = torch.arange(len(ids), device=logits.device)[None, :] >= nopts[:, None]
    return logits.masked_fill(mask, float("-inf"))


class UnifiedDecisionModel:
    def __init__(
        self,
        name=DEFAULT_MODEL,
        *,
        revision=None,
        device=None,
        temperature=None,
        max_context=16384,
        lora=None,
        adapter_path=None,
    ):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        processor = AutoProcessor.from_pretrained(name, revision=revision)
        lm = AutoModelForMultimodalLM.from_pretrained(name, revision=revision, dtype=dtype)
        if lm.config.model_type != "gemma4_unified":
            raise ValueError("UnifiedDecisionModel requires a Gemma 4 Unified checkpoint")
        resolved_revision = getattr(lm.config, "_commit_hash", None) or revision
        if lora and adapter_path:
            raise ValueError("choose a new LoRA or an existing adapter, not both")
        if temperature is None:
            temperature = load_temperature(adapter_path or name, revision=resolved_revision)
        if adapter_path:
            from peft import PeftModel

            lm = PeftModel.from_pretrained(lm, adapter_path, is_trainable=False)
        if lora:
            from peft import LoraConfig, get_peft_model

            lm = get_peft_model(lm, LoraConfig(**lora))
        self._initialize(lm, processor, str(device), temperature, max_context)
        self.name = name
        self.revision = resolved_revision

    @classmethod
    def from_components(cls, lm, processor, *, device="cpu", temperature=1.0, max_context=16384):
        """Dependency injection for offline tests and already-loaded checkpoints."""
        self = cls.__new__(cls)
        self._initialize(lm, processor, device, temperature, max_context)
        self.name = getattr(lm.config, "_name_or_path", "in-memory") or "in-memory"
        self.revision = getattr(lm.config, "_commit_hash", None)
        return self

    def _initialize(self, lm, processor, device, temperature, max_context):
        import torch

        self.set_temperature(temperature)
        if max_context < 1:
            raise ValueError("max_context must be positive")
        self.lm, self.processor, self.device = lm.to(device).eval(), processor, device
        self.tok = processor.tokenizer
        base = lm.get_base_model() if hasattr(lm, "peft_config") else lm
        self.backbone = base.model
        self.head = base.get_output_embeddings()
        config = getattr(base.config, "text_config", base.config)
        self.softcap = getattr(config, "final_logit_softcapping", None)
        self.max_context = min(max_context, getattr(config, "max_position_embeddings", max_context))
        self.letters = torch.tensor(candidate_ids(self.tok), device=device)

    def set_temperature(self, temperature):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        self.temperature = float(temperature)

    def prepare(self, request: DecisionRequest):
        import numpy as np
        import torch
        from PIL import Image, UnidentifiedImageError

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
                    content.append(
                        {"type": "text", "text": f"Frame at {media.timestamp_seconds:g}s:"}
                    )
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
        text = self.processor.apply_chat_template(
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
        inputs = dict(self.processor(**kwargs))
        # Slots are measured AFTER the processor expands image/audio placeholders.
        prefix_length = inputs["input_ids"].shape[1]
        suffix, slots, nopts = [], [], []
        for q in request.questions:
            lines = [f"\nQuestion ({q.type}): {q.instructions}\n"]
            for letter, label, desc in zip(LETTERS, q.labels(), q.descriptions()):
                lines.append(f"({letter}) {label}: {desc}\n")
            lines.append("Answer: (")
            suffix.extend(self.tok.encode("".join(lines), add_special_tokens=False))
            slots.append(prefix_length + len(suffix) - 1)
            nopts.append(len(q.labels()))
        if prefix_length + len(suffix) > self.max_context:
            raise RequestValidationError(
                f"request exceeds {self.max_context} context tokens; nothing was truncated"
            )
        inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([suffix])], dim=1)
        for key, fill in (("attention_mask", 1), ("mm_token_type_ids", 0), ("token_type_ids", 0)):
            if key in inputs:
                inputs[key] = torch.cat(
                    [inputs[key], inputs[key].new_full((1, len(suffix)), fill)], dim=1
                )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        return (
            inputs,
            torch.tensor(slots, device=self.device),
            torch.tensor(nopts, device=self.device),
        )

    def logits(self, request: DecisionRequest):
        inputs, slots, nopts = self.prepare(request)
        out = self.backbone(**inputs, use_cache=False, return_dict=True)
        hidden = out.last_hidden_state[0, slots]
        return project_candidates(hidden, self.head, self.letters, nopts, self.softcap)

    def predict(self, request: DecisionRequest) -> dict:
        import torch

        started = time.perf_counter()
        self.lm.eval()
        with torch.inference_mode():
            probabilities = (self.logits(request) / self.temperature).softmax(-1).cpu().tolist()
        answers = {
            q.id: answer_from_probabilities(q, p[: len(q.labels())])
            for q, p in zip(request.questions, probabilities)
        }
        return {
            "answers": answers,
            "model": self.name,
            "revision": self.revision,
            "temperature": self.temperature,
            "passes": 1,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        lm = self.lm.merge_and_unload() if hasattr(self.lm, "merge_and_unload") else self.lm
        lm.save_pretrained(path, safe_serialization=True)
        self.processor.save_pretrained(path)
        (path / "s1_config.json").write_text(
            json.dumps(
                {
                    "temperature": self.temperature,
                    "source_model": self.name,
                    "source_revision": self.revision,
                    "prompt_version": 1,
                },
                indent=2,
            )
            + "\n"
        )

    def save_adapter(self, path):
        if not hasattr(self.lm, "peft_config"):
            raise ValueError("only a PEFT model can save a separate decision adapter")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        self.lm.save_pretrained(path, safe_serialization=True)
        (path / "s1_config.json").write_text(
            json.dumps(
                {
                    "temperature": self.temperature,
                    "source_model": self.name,
                    "source_revision": self.revision,
                    "prompt_version": 1,
                },
                indent=2,
            )
            + "\n"
        )
