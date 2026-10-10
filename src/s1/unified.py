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
from .runtime import resolve_device, resolve_dtype
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
    weight = head.weight[ids]
    bias = head.bias[ids] if getattr(head, "bias", None) is not None else None
    return _project_candidate_rows(hidden, weight, bias, nopts, softcap)


def _project_candidate_rows(hidden, weight, bias, nopts, softcap):
    import torch
    from torch.nn import functional as F

    logits = F.linear(hidden.to(weight.dtype), weight, bias)
    if softcap is not None:
        if not math.isfinite(softcap) or softcap <= 0:
            raise ValueError("logit softcap must be positive and finite")
        logits = torch.tanh(logits / softcap) * softcap
    # Match the native LM head's softcap rounding before calibrating in float32.
    logits = logits.float()
    mask = torch.arange(len(weight), device=logits.device)[None, :] >= nopts[:, None]
    return logits.masked_fill(mask, float("-inf"))


class UnifiedDecisionModel:
    def __init__(
        self,
        name=DEFAULT_MODEL,
        *,
        revision=None,
        device=None,
        precision=None,
        dtype=None,
        attn_implementation=None,
        temperature=None,
        max_context=16384,
        lora=None,
        adapter_path=None,
        quantization="none",
        compile_mode=None,
        enable_candidate_cache=False,
    ):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        device = resolve_device(device)
        if precision not in (None, "float32", "bfloat16"):
            raise ValueError("precision must be float32 or bfloat16")
        if precision == "bfloat16" and not device.startswith("cuda"):
            raise ValueError("bfloat16 precision requires a CUDA device; use dtype for MPS")
        selected_dtype = resolve_dtype(dtype, device)
        if precision is not None:
            precision_dtype = resolve_dtype(precision, device)
            if dtype not in (None, "auto") and selected_dtype != precision_dtype:
                raise ValueError("precision and dtype must agree when both are specified")
            selected_dtype = precision_dtype
        if attn_implementation not in (None, "eager", "sdpa"):
            raise ValueError("attn_implementation must be eager, sdpa, or None")
        from .quantization import inspect_quantization, loading_kwargs

        quantized_loading = loading_kwargs(
            quantization,
            device=device,
            precision=str(selected_dtype).removeprefix("torch."),
        )
        if lora and adapter_path:
            raise ValueError("choose a new LoRA or an existing adapter, not both")
        if lora and quantization != "none":
            raise ValueError("quantized inference supports existing adapters; train LoRA in BF16")
        if quantization == "nvfp4" and adapter_path:
            raise ValueError("NVFP4 requires a merged checkpoint; merge adapters in BF16 first")
        processor = AutoProcessor.from_pretrained(name, revision=revision)
        kwargs = {"revision": revision, "dtype": selected_dtype, **quantized_loading}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        if quantization == "nvfp4":
            from .nvfp4 import load_nvfp4

            lm = load_nvfp4(
                name,
                revision=revision,
                device=device,
                precision=str(selected_dtype).removeprefix("torch."),
                attn_implementation=attn_implementation,
            )
        else:
            lm = AutoModelForMultimodalLM.from_pretrained(name, **kwargs)
        if lm.config.model_type != "gemma4_unified":
            raise ValueError("UnifiedDecisionModel requires a Gemma 4 Unified checkpoint")
        resolved_revision = getattr(lm.config, "_commit_hash", None) or revision
        if temperature is None:
            temperature = load_temperature(adapter_path or name, revision=resolved_revision)
        if adapter_path:
            from peft import PeftModel

            lm = PeftModel.from_pretrained(lm, adapter_path, is_trainable=False)
        if lora:
            from peft import LoraConfig, get_peft_model

            lm = get_peft_model(lm, LoraConfig(**lora))
        self.quantization = quantization
        self.quantization_details = inspect_quantization(lm, quantization)
        self._initialize(lm, processor, str(device), temperature, max_context)
        self.enable_candidate_cache(enable_candidate_cache)
        self.compile_mode = None
        if compile_mode is not None:
            self.enable_compile(compile_mode)
        self.name = name
        self.revision = resolved_revision

    def enable_compile(self, mode="decoder-max-autotune-no-cudagraphs"):
        """Opt-in torch.compile for the text decoder only (research, off by default).

        Rationale (see artifacts/optimization/compile-int8-analysis.md): full-model
        reduce-overhead hits dynamo recompile_limit on 60-716 tok + 4 modality
        branches. This compiles only the language_model submodule with
        max-autotune-no-cudagraphs + dynamic=True + fullgraph=False, keeping
        vision/audio encoders eager. Quantized (bnb) + compile is untested;
        prefer quantization="none" + torchao weight-only for compile experiments.
        """
        import torch

        allowed = ("decoder-max-autotune-no-cudagraphs", "decoder-default-dynamic")
        if mode not in allowed:
            raise ValueError(f"compile_mode must be one of {allowed}")
        if self.quantization != "none":
            raise ValueError(
                "compile with bitsandbytes quantization is unsupported; use none + torchao"
            )
        target = getattr(getattr(self, "backbone", None), "language_model", None) or getattr(
            self, "backbone", None
        )
        if target is None:
            raise ValueError("compile target unavailable")
        torch_mode = "max-autotune-no-cudagraphs" if "no-cudagraphs" in mode else "default"
        compiled = torch.compile(target, mode=torch_mode, dynamic=True, fullgraph=False)
        # Rebind: backbone.language_model if present, else whole backbone.
        if getattr(getattr(self, "backbone", None), "language_model", None) is not None:
            self.backbone.language_model = compiled
        else:
            self.backbone = compiled
        self.compile_mode = mode
        return mode

    @classmethod
    def from_components(
        cls,
        lm,
        processor,
        *,
        device="cpu",
        temperature=1.0,
        max_context=16384,
        enable_candidate_cache=False,
    ):
        """Dependency injection for offline tests and already-loaded checkpoints."""
        self = cls.__new__(cls)
        self.quantization = "none"
        self.quantization_details = {"method": "none", "quantized_linear_modules": 0}
        self._initialize(lm, processor, device, temperature, max_context)
        self.enable_candidate_cache(enable_candidate_cache)
        self.name = getattr(lm.config, "_name_or_path", "in-memory") or "in-memory"
        self.revision = getattr(lm.config, "_commit_hash", None)
        return self

    def _initialize(self, lm, processor, device, temperature, max_context):
        import torch

        self.set_temperature(temperature)
        if max_context < 1:
            raise ValueError("max_context must be positive")
        # bitsandbytes is already placed by its explicit device map. A second
        # .to() can reject INT8 modules or reallocate their quantization state.
        if self.quantization == "none":
            lm = lm.to(device)
        self.lm, self.processor, self.device = lm.eval(), processor, str(device)
        self.tok = processor.tokenizer
        base = lm.get_base_model() if hasattr(lm, "peft_config") else lm
        self.backbone = base.model
        self.head = base.get_output_embeddings()
        config = getattr(base.config, "text_config", base.config)
        self.dtype = str(self.head.weight.dtype).removeprefix("torch.")
        self.attn_implementation = getattr(config, "_attn_implementation", None)
        self.softcap = getattr(config, "final_logit_softcapping", None)
        self.max_context = min(max_context, getattr(config, "max_position_embeddings", max_context))
        self.letters = torch.tensor(candidate_ids(self.tok), device=device)
        self.candidate_cache_enabled = False
        self._candidate_cache = None

    def enable_candidate_cache(self, enabled=True):
        """Opt in to reusing selected head rows during eval without gradients.

        The cache is transient and normal PyTorch in-place updates invalidate it.
        Call this method again after updates that bypass tensor version counters
        (for example, writes through ``.data``). PEFT heads remain uncached.
        """
        if not isinstance(enabled, bool):
            raise ValueError("enable_candidate_cache must be boolean")
        self.candidate_cache_enabled = enabled
        self._candidate_cache = None
        return enabled

    def _inference_candidate_rows(self):
        import torch

        head = self.head
        if (
            not self.candidate_cache_enabled
            or torch.is_grad_enabled()
            or self.lm.training
            or head.training
            or hasattr(self.lm, "peft_config")
        ):
            self._candidate_cache = None
            return None
        weight, bias, ids = head.weight, getattr(head, "bias", None), self.letters
        try:
            signature = (id(head),) + tuple(
                None
                if tensor is None
                else (
                    id(tensor),
                    tensor._version,
                    tensor.data_ptr(),
                    tensor.device,
                    tensor.dtype,
                    tuple(tensor.shape),
                    tensor.stride(),
                )
                for tensor in (weight, bias, ids)
            )
        except RuntimeError:
            # Inference tensors have no version counter, so mutation cannot be
            # detected safely. Keep the original uncached projection for them.
            self._candidate_cache = None
            return None
        if self._candidate_cache is None or self._candidate_cache[0] != signature:
            rows = (weight[ids].detach(), None if bias is None else bias[ids].detach())
            # Retain the source objects as well as their IDs to prevent identity
            # reuse after a head or tensor is replaced and garbage collected.
            self._candidate_cache = (signature, rows, (head, weight, bias, ids))
        return self._candidate_cache[1]

    def reset_memory_peak(self):
        from .resources import reset_memory_peak

        reset_memory_peak(self.device)

    def memory_snapshot(self):
        from .resources import memory_snapshot

        return memory_snapshot(self.lm, self.device)

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
        rows = self._inference_candidate_rows()
        if rows is not None:
            return _project_candidate_rows(hidden, *rows, nopts, self.softcap)
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
        if self.quantization != "none":
            raise ValueError(
                "save the original BF16 checkpoint and adapter separately before quantizing"
            )
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
