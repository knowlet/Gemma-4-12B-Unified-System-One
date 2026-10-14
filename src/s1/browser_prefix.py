"""Opt-in identical-prefix reuse for independent, prepared browser questions.

The caller supplies original gold-free requests and exact CPU processor tensors.
Each suffix chunk receives its own fork of a native DynamicCache. Every offered
candidate is scored; the utility never constructs a browser action or executes
one. Unsupported inputs fail explicitly before any backbone forward.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid

from .contracts import MAX_OPTIONS, DecisionRequest, answer_from_probabilities
from .errors import BackendResponseError, RequestValidationError
from .unified import project_candidates

SEQUENCE_KEYS = frozenset(
    {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids", "position_ids"}
)
USER_MODES = frozenset({"user", "grounded", "goal-first"})


class BrowserPrefixUnsupported(RequestValidationError):
    """The exact prepared inputs cannot use this cache policy safely."""


def _tensor_identity(inputs):
    content = {
        key: {"dtype": str(value.dtype), "values": value.tolist()}
        for key, value in sorted(inputs.items())
    }
    return hashlib.sha256(json.dumps(content, separators=(",", ":")).encode()).hexdigest()


def _validate(model, rows, prompt_mode, trace):
    import torch
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedModel

    if prompt_mode not in USER_MODES:
        raise BrowserPrefixUnsupported("prefix reuse supports only text-only user browser modes")
    if not rows:
        raise BrowserPrefixUnsupported("prefix reuse requires a nonempty prepared row list")
    if not isinstance(model.backbone, Gemma4UnifiedModel):
        raise BrowserPrefixUnsupported("prefix reuse requires the native Gemma4Unified backbone")
    config = model.backbone.config.get_text_config()
    if getattr(config, "use_bidirectional_attention", None) == "all":
        raise BrowserPrefixUnsupported("bidirectional text cannot reuse a causal prefix")
    rope = getattr(config, "rope_parameters", None)
    if not isinstance(rope, dict) or any(
        not isinstance(parameters, dict)
        or parameters.get("rope_type", "default") not in {"default", "proportional"}
        for parameters in rope.values()
    ):
        raise BrowserPrefixUnsupported(
            "prefix reuse requires fixed default/proportional rotary frequencies"
        )
    if len(model.letters) != MAX_OPTIONS:
        raise BrowserPrefixUnsupported("prefix readout requires the complete native candidate rows")
    records = []
    key_set = None
    for index, row in enumerate(rows):
        trace["validation_row_index"] = index
        request = row.get("request")
        if not isinstance(request, DecisionRequest) or len(request.questions) != 1 or request.media:
            raise BrowserPrefixUnsupported(
                "each prefix row needs one gold-free, text-only question"
            )
        prepared = row.get("prepared")
        if not isinstance(prepared, (tuple, list)) or len(prepared) != 3:
            raise BrowserPrefixUnsupported(
                "prefix rows require exact prepared inputs, slots and counts"
            )
        inputs, slots, counts = prepared
        if not isinstance(inputs, dict) or not {"input_ids", "attention_mask"} <= inputs.keys():
            raise BrowserPrefixUnsupported("prefix inputs require explicit IDs and attention masks")
        if set(inputs) - SEQUENCE_KEYS:
            raise BrowserPrefixUnsupported(
                "non-sequence or media processor tensors cannot use prefix reuse"
            )
        if key_set is None:
            key_set = set(inputs)
        if set(inputs) != key_set:
            raise BrowserPrefixUnsupported(
                "prefix rows must preserve the same processor tensor keys"
            )
        ids = inputs["input_ids"]
        if (
            not torch.is_tensor(ids)
            or ids.ndim != 2
            or ids.shape[0] != 1
            or ids.dtype != torch.long
        ):
            raise BrowserPrefixUnsupported("prefix input IDs must be one int64 sequence")
        length = ids.shape[1]
        if not 1 <= length <= model.max_context:
            raise RequestValidationError(
                "prefix row exceeds the original context budget; nothing was truncated"
            )
        for key, value in inputs.items():
            if (
                not torch.is_tensor(value)
                or value.device.type != "cpu"
                or tuple(value.shape) != (1, length)
            ):
                raise BrowserPrefixUnsupported(
                    "all prepared sequence tensors must be aligned on CPU"
                )
            if value.dtype != rows[0]["prepared"][0][key].dtype:
                raise BrowserPrefixUnsupported("prefix processor tensor dtypes differ across rows")
        if (ids < 0).any() or (ids >= config.vocab_size).any():
            raise BrowserPrefixUnsupported(
                "prefix input IDs exceed the native embedding vocabulary"
            )
        mask = inputs["attention_mask"][0]
        if not ((mask == 0) | (mask == 1)).all() or not (mask[1:] <= mask[:-1]).all():
            raise BrowserPrefixUnsupported(
                "prefix reuse supports explicit causal right-padding masks only"
            )
        if (
            not torch.is_tensor(slots)
            or not torch.is_tensor(counts)
            or slots.device.type != "cpu"
            or counts.device.type != "cpu"
            or slots.dtype != torch.long
            or counts.dtype != torch.long
            or tuple(slots.shape) != (1,)
            or tuple(counts.shape) != (1,)
        ):
            raise BrowserPrefixUnsupported("prefix slots and counts must be one CPU int64 value")
        slot, count = slots.item(), counts.item()
        if slot != int(mask.sum().item()) - 1 or slot < 1:
            raise BrowserPrefixUnsupported("prefix answer slot must be the last unmasked token")
        if count != len(request.questions[0].labels()) or not 2 <= count <= MAX_OPTIONS:
            raise BrowserPrefixUnsupported("prefix option count differs from the original question")
        if "position_ids" in inputs and not torch.equal(
            inputs["position_ids"], torch.arange(length).unsqueeze(0)
        ):
            raise BrowserPrefixUnsupported(
                "prefix positions must preserve absolute zero-based positions"
            )
        if "mm_token_type_ids" in inputs and (inputs["mm_token_type_ids"] != 0).any():
            raise BrowserPrefixUnsupported("non-text modality IDs cannot use prefix reuse")
        for token_id in (
            model.backbone.config.image_token_id,
            model.backbone.config.audio_token_id,
            model.backbone.config.video_token_id,
        ):
            if token_id is not None and (ids == token_id).any():
                raise BrowserPrefixUnsupported("media placeholder tokens cannot use prefix reuse")
        records.append(
            {"request": request, "inputs": inputs, "slot": slot, "count": count, "length": length}
        )
    trace.pop("validation_row_index", None)
    prefix = min(record["slot"] for record in records)
    for key in key_set:
        first = records[0]["inputs"][key][0, :prefix]
        for record in records[1:]:
            difference = (first != record["inputs"][key][0, :prefix]).nonzero()
            if difference.numel():
                prefix = min(prefix, int(difference[0, 0]))
                first = first[:prefix]
    if prefix < 1:
        raise BrowserPrefixUnsupported("prepared rows have no identical causal tensor prefix")
    return records, prefix


def _suffix_inputs(records, prefix, pad_token_id):
    """Right-pad exact CPU suffixes while retaining their full attention masks."""
    import torch
    from torch.nn.functional import pad

    width = max(record["length"] - prefix for record in records)
    result = {}
    for key in records[0]["inputs"]:
        if key == "position_ids":
            continue
        parts = []
        for record in records:
            value = record["inputs"][key]
            if key != "attention_mask":
                value = value[:, prefix:]
            extra = width - (record["length"] - prefix)
            parts.append(pad(value, (0, extra), value=pad_token_id if key == "input_ids" else 0))
        result[key] = torch.cat(parts, dim=0)
    result["position_ids"] = (
        torch.arange(prefix, prefix + width).unsqueeze(0).repeat(len(records), 1)
    )
    return result


def predict_prefix(model, rows, *, prompt_mode, readout=None, max_suffix_batch_size=1):
    """Return ordered responses with one shared prefix ledger on the first item.

    ``rows`` contain a one-question ``request`` and its exact CPU ``prepared``
    tuple of ``(inputs, slots, counts)``. ``readout(hidden, counts, softcap)`` may
    supply an explicit selected-row sidecar; the default is the original native
    candidate projection. It must return all 52 masked logits per row. The
    optional suffix batch size is strictly 1 or 2, with 1 preserving the
    original execution path. New padding is masked and its physical work is
    reported separately from the original full contexts.
    """
    import torch
    from transformers.cache_utils import DynamicCache

    started = time.perf_counter()
    batch_id = uuid.uuid4().hex
    trace = {
        "version": "browser_prefix_v1",
        "batch_id": batch_id,
        "prompt_mode": prompt_mode,
        "status": "validation",
        "latency_scope": "whole_prefix_policy_call",
        "cache_policy": "native_configured_DynamicCache_deepcopy_each_suffix",
        "max_suffix_batch_size": max_suffix_batch_size
        if type(max_suffix_batch_size) is int
        else None,
        "max_live_cache_forks": 1,
        "rows": [],
        "forwards": [],
        "branches": [],
        "completed_response_indices": [],
        "completed_responses": [],
    }
    responses = []

    def forward(inputs, phase, row_index=None, cache=None, row_indices=None, logical_tokens=None):
        physical_tokens = inputs["input_ids"].numel()
        record = {
            "phase": phase,
            "row_index": row_index,
            "row_indices": [] if row_indices is None else list(row_indices),
            "chunk_id": f"{batch_id}:prefix"
            if phase == "prefill"
            else f"{batch_id}:suffix:{row_index}",
            "forward_index": 0,
            "batch_size": inputs["input_ids"].shape[0],
            "input_tokens": physical_tokens,
            "sequence_width": inputs["input_ids"].shape[1],
            "logical_input_tokens": physical_tokens if logical_tokens is None else logical_tokens,
            "padding_input_tokens": 0
            if logical_tokens is None
            else physical_tokens - logical_tokens,
            "attention_tokens": inputs["attention_mask"].numel(),
            "attention_width": inputs["attention_mask"].shape[1],
            "position_start": 0 if phase == "prefill" else trace["prefix_tokens"],
            "status": "started",
            "latency_scope": "host_backbone_call_without_device_synchronize",
        }
        trace["forwards"].append(record)
        began = time.perf_counter()
        try:
            result = model.backbone(
                **inputs, past_key_values=cache, use_cache=True, return_dict=True
            )
            record["status"] = "returned"
            return result
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["elapsed_ms"] = (time.perf_counter() - began) * 1000

    try:
        if type(max_suffix_batch_size) is not int or max_suffix_batch_size not in (1, 2):
            raise BrowserPrefixUnsupported("max_suffix_batch_size must be the integer 1 or 2")
        trace["requests"] = [row["request"].model_dump(exclude_none=True) for row in rows]
        records, prefix = _validate(model, rows, prompt_mode, trace)
        pad_token_id = getattr(model.backbone.config.get_text_config(), "pad_token_id", None)
        if max_suffix_batch_size == 2 and (
            type(pad_token_id) is not int
            or not 0 <= pad_token_id < model.backbone.config.get_text_config().vocab_size
        ):
            raise BrowserPrefixUnsupported("suffix batching requires a valid native padding token")
        # Prepare every chunk on CPU before prefill. Only the current chunk is
        # moved to the model device, keeping suffix forwards bounded at two rows.
        chunks = [
            (
                list(range(start, min(start + max_suffix_batch_size, len(records)))),
                _suffix_inputs(
                    records[start : start + max_suffix_batch_size], prefix, pad_token_id or 0
                ),
            )
            for start in range(0, len(records), max_suffix_batch_size)
        ]
        if max_suffix_batch_size == 2:
            trace["cache_policy"] = (
                "native_configured_DynamicCache_deepcopy_batch_repeat_each_suffix_chunk"
            )
        trace["prefix_tokens"] = prefix
        trace["context_tokens"] = [record["length"] for record in records]
        trace["suffix_tokens"] = [record["length"] - prefix for record in records]
        trace["rows"] = [
            {
                "request": record["request"].model_dump(exclude_none=True),
                "prepared_sha256": _tensor_identity(record["inputs"]),
                "context_tokens": record["length"],
                "answer_slot": record["slot"],
                "option_count": record["count"],
            }
            for record in records
        ]
        trace["prefix_sha256"] = _tensor_identity(
            {key: value[:, :prefix] for key, value in records[0]["inputs"].items()}
        )
        trace["status"] = "prefill"
        model.lm.eval()
        with torch.inference_mode():
            inputs = {
                key: value[:, :prefix].to(model.device)
                for key, value in records[0]["inputs"].items()
            }
            inputs["position_ids"] = torch.arange(prefix, device=model.device).unsqueeze(0)
            output = forward(inputs, "prefill")
            cache = output.past_key_values
            if not isinstance(cache, DynamicCache) or cache.get_seq_length() != prefix:
                raise BackendResponseError(
                    "native prefix forward did not return its complete DynamicCache"
                )
            trace["cache_layers"] = [
                {
                    "is_sliding": layer.is_sliding,
                    "logical_tokens": layer.get_seq_length(),
                    "key_shape": list(layer.keys.shape),
                    "value_shape": list(layer.values.shape),
                    "key_tensor_bytes": layer.keys.numel() * layer.keys.element_size(),
                    "value_tensor_bytes": layer.values.numel() * layer.values.element_size(),
                    "key_storage_bytes": layer.keys.untyped_storage().nbytes(),
                    "value_storage_bytes": layer.values.untyped_storage().nbytes(),
                    "dtype": str(layer.keys.dtype),
                }
                for layer in cache.layers
            ]
            del output, inputs
            trace["status"] = "suffixes"
            for indices, cpu_inputs in chunks:
                first = indices[0]
                chunk_records = [records[index] for index in indices]
                width = cpu_inputs["input_ids"].shape[1]
                branch = {"row_index": first, "row_indices": indices, "status": "forking"}
                trace["branches"].append(branch)
                began = time.perf_counter()
                fork = copy.deepcopy(cache)
                if len(indices) > 1:
                    fork.batch_repeat_interleave(len(indices))
                branch["fork_elapsed_ms"] = (time.perf_counter() - began) * 1000
                branch["fork_latency_scope"] = "host_cache_deepcopy_without_device_synchronize"
                inputs = {key: value.to(model.device) for key, value in cpu_inputs.items()}
                branch["cache_batch_size"] = len(indices)
                branch["status"] = "forwarding"
                output = forward(
                    inputs,
                    "suffix",
                    first,
                    fork,
                    indices,
                    sum(record["length"] - prefix for record in chunk_records),
                )
                branch["cache_logical_tokens_after_forward"] = fork.get_seq_length()
                branch["cache_layers_after_forward"] = [
                    {
                        "logical_tokens": layer.get_seq_length(),
                        "key_shape": list(layer.keys.shape),
                        "value_shape": list(layer.values.shape),
                    }
                    for layer in fork.layers
                ]
                branch["status"] = "readout"
                slots = torch.tensor(
                    [record["slot"] - prefix for record in chunk_records], device=model.device
                )
                hidden = output.last_hidden_state[
                    torch.arange(len(indices), device=model.device), slots
                ]
                counts = torch.tensor(
                    [record["count"] for record in chunk_records], device=model.device
                )
                logits = (
                    project_candidates(hidden, model.head, model.letters, counts, model.softcap)
                    if readout is None
                    else readout(hidden, counts, model.softcap)
                )
                if (
                    not torch.is_tensor(logits)
                    or not logits.is_floating_point()
                    or tuple(logits.shape) != (len(indices), MAX_OPTIONS)
                    or not torch.where(
                        torch.arange(MAX_OPTIONS, device=logits.device)[None, :] < counts[:, None],
                        torch.isfinite(logits),
                        torch.isneginf(logits),
                    ).all()
                ):
                    raise BackendResponseError(
                        "prefix readout must return every native masked candidate logit"
                    )
                probabilities = (logits / model.temperature).softmax(-1).cpu().tolist()
                branch["raw_probabilities"] = [
                    {"row_index": index, "probabilities": probs[: record["count"]]}
                    for index, record, probs in zip(
                        indices, chunk_records, probabilities, strict=True
                    )
                ]
                if cache.get_seq_length() != prefix:
                    raise BackendResponseError("a suffix mutated the shared prefix cache")
                branch["answers"] = {}
                branch["row_answers"] = []
                for index, record, probs in zip(indices, chunk_records, probabilities, strict=True):
                    question = record["request"].questions[0]
                    answer = answer_from_probabilities(question, probs[: record["count"]])
                    branch["answers"][question.id] = answer
                    branch["row_answers"].append(
                        {"row_index": index, "answers": {question.id: answer}}
                    )
                    own_tokens = width + (prefix if index == 0 else 0)
                    own_forwards = int(index == first) + int(index == 0)
                    refs = [
                        {"chunk_id": f"{batch_id}:prefix", "forward_index": 0},
                        {"chunk_id": f"{batch_id}:suffix:{first}", "forward_index": 0},
                    ]
                    responses.append(
                        {
                            "model": model.name,
                            "revision": getattr(model, "revision", None),
                            "temperature": model.temperature,
                            "prompt_mode": prompt_mode,
                            "answers": {question.id: answer},
                            "passes": own_forwards,
                            "usage": {"input_tokens": own_tokens, "output_tokens": 0},
                            "execution": {
                                "batch_id": batch_id,
                                "forward_calls": own_forwards,
                                "forward_calls_scope": "allocated_once_per_shared_browser_batch",
                                "forward_refs": refs,
                                "batch_chunk_ids": [ref["chunk_id"] for ref in refs],
                                "independent_questions": True,
                                "research_prompt_mode": "user_question",
                                "policy_variant": "shared_prefix_dynamic_cache_v1",
                                "input_tokens": own_tokens,
                                "logical_computed_input_tokens": record["length"]
                                - prefix
                                + (prefix if index == 0 else 0),
                                "padding_input_tokens": width - (record["length"] - prefix),
                                "max_forward_batch_size": max_suffix_batch_size,
                                "suffix_batch_size": len(indices),
                                "context_input_tokens": record["length"],
                                "question_input_tokens": {question.id: record["length"]},
                                "reused_prefix_tokens": prefix if index else 0,
                            },
                        }
                    )
                branch["status"] = "completed"
                # Release this entire fork before allocating the next one.
                del output, fork, inputs, hidden, counts, logits, slots
        elapsed = (time.perf_counter() - started) * 1000
        trace.update(
            status="completed",
            elapsed_ms=elapsed,
            forward_calls=len(trace["forwards"]),
            computed_input_tokens=sum(record["input_tokens"] for record in trace["forwards"]),
            logical_computed_input_tokens=prefix + sum(trace["suffix_tokens"]),
            padding_input_tokens=sum(
                record["padding_input_tokens"] for record in trace["forwards"]
            ),
            max_forward_batch_size=max(record["batch_size"] for record in trace["forwards"]),
            full_context_input_tokens=sum(trace["context_tokens"]),
            completed_response_indices=list(range(len(responses))),
        )
        for response in responses:
            response.update(
                latency_ms=elapsed,
                batch_elapsed_ms=elapsed,
                latency_scope="whole_browser_batch",
            )
            response["execution"].update(
                batch_forward_calls=len(trace["forwards"]),
                batch_forward_calls_scope="whole_browser_batch",
                batch_elapsed_ms=elapsed,
                latency_scope="whole_browser_batch",
                batch_computed_input_tokens=trace["computed_input_tokens"],
                batch_logical_computed_input_tokens=trace["logical_computed_input_tokens"],
                batch_padding_input_tokens=trace["padding_input_tokens"],
                batch_full_context_input_tokens=trace["full_context_input_tokens"],
            )
        trace["completed_responses"] = copy.deepcopy(responses)
        responses[0]["prefix"] = trace
        return responses
    except Exception as exc:
        if trace["branches"] and trace["branches"][-1]["status"] != "completed":
            branch = trace["branches"][-1]
            branch.update(
                failure_stage=branch["status"],
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        trace.update(
            failure_stage=trace["status"],
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_ms=(time.perf_counter() - started) * 1000,
            completed_response_indices=list(range(len(responses))),
            completed_responses=copy.deepcopy(responses),
            forward_calls=len(trace["forwards"]),
            forwarded_input_tokens=sum(record["input_tokens"] for record in trace["forwards"]),
            forwarded_input_tokens_scope="issued_backbone_calls_including_failed_unknown_completion",
            completed_computed_input_tokens=sum(
                record["input_tokens"]
                for record in trace["forwards"]
                if record["status"] == "returned"
            ),
            completed_computed_input_tokens_scope="returned_backbone_calls_without_device_synchronize",
            max_forward_batch_size=max(
                (record["batch_size"] for record in trace["forwards"]), default=0
            ),
        )
        exc.prefix_evidence = trace
        exc.batch_evidence = trace
        raise
