"""Explicit request-local prefix reuse for native independent Gemma Unified heads.

This experimental callable changes execution shapes, so mathematical equivalence
does not establish BF16 probability parity or an end-to-end speed improvement.
It never changes the default backend or shares a cache between requests.
"""

from __future__ import annotations

import copy

from s1.backends import GEMMA_MAX_FORWARD_BATCH_SIZE
from s1.contracts import answer_from_probabilities
from s1.unified import project_candidates


def predict_independent_prefix(model, requests):
    """Prefill each complete native state/media prefix, then branch independent heads.

    Cache deep copies retain sliding-layer cumulative positions as well as K/V
    tensors. They are never reconstructed from truncated tensors or cropped.
    At most eight suffixes are prepared and evaluated together. A one-question
    request uses the unchanged reference path because prefix reuse cannot help.
    All failures propagate; there is no alternate-model or text-only fallback.
    """
    import torch
    from torch.nn.functional import pad

    from .gemma import predict_batch

    if not requests:
        raise ValueError("empty batch")
    if getattr(model.backbone.config, "model_type", None) != "gemma4_unified":
        raise ValueError("prefix experiment requires the complete Gemma 4 Unified backbone")
    config = model.backbone.config.get_text_config()
    if getattr(config, "use_bidirectional_attention", None) == "all":
        raise ValueError("bidirectional text attention cannot reuse a suffix-independent prefix")
    model.lm.eval()
    results, batch_sizes, sequence_tokens, cached_prefix_tokens = [], [], [], []
    prefix_calls, suffix_calls, reference_calls, processed_tokens = 0, 0, 0, 0
    optimized_requests, reference_requests = 0, 0
    with torch.inference_mode():
        for request in requests:
            if len(request.questions) == 1:
                reference = predict_batch(model, [request], independent=True, readout="candidate")[
                    0
                ]
                execution = reference["execution"]
                batch_sizes.extend(execution["batch_sizes"])
                sequence_tokens.extend(execution["sequence_tokens"])
                processed_tokens += sum(execution["sequence_tokens"])
                reference_calls += execution["forward_calls"]
                reference_requests += 1
                results.append({"answers": reference["answers"]})
                continue

            prefix = model.prepare_prefix(request)
            if prefix["input_ids"].ndim != 2 or prefix["input_ids"].shape[0] != 1:
                raise ValueError("prefix experiment requires one native prefix row per request")
            prefix_length = prefix["input_ids"].shape[1]
            # Validate every full context before transferring media or building
            # a device cache. Discard each CPU suffix immediately, so an invalid
            # later head cannot trigger any backbone work for this request.
            for question in request.questions:
                model.prepare_suffix([question], prefix_length)
            prefix = {
                key: value.to(model.device) if hasattr(value, "to") else value
                for key, value in prefix.items()
            }
            output = model.backbone(**prefix, use_cache=True, return_dict=True)
            prefix_cache = output.past_key_values
            if prefix_cache is None or prefix_cache.get_seq_length() != prefix_length:
                raise ValueError("native prefix cache did not retain its absolute sequence length")
            del output
            prefix_calls += 1
            optimized_requests += 1
            batch_sizes.append(1)
            cached_prefix_tokens.append(prefix_length)
            processed_tokens += prefix_length
            answers = {}
            for start in range(0, len(request.questions), GEMMA_MAX_FORWARD_BATCH_SIZE):
                questions = request.questions[start : start + GEMMA_MAX_FORWARD_BATCH_SIZE]
                prepared = [
                    model.prepare_suffix([question], prefix_length) for question in questions
                ]
                lengths = [tokens.shape[1] for tokens, _, _ in prepared]
                sequence_tokens.extend(prefix_length + length for length in lengths)
                length = max(lengths)
                ids = torch.cat(
                    [
                        pad(
                            tokens,
                            (0, length - tokens.shape[1]),
                            value=getattr(model.tok, "pad_token_id", None) or 0,
                        )
                        for tokens, _, _ in prepared
                    ]
                ).to(model.device)
                full_masks = []
                for tokens, _, _ in prepared:
                    prefix_mask = prefix.get("attention_mask")
                    if prefix_mask is None:
                        prefix_mask = torch.ones(
                            (1, prefix_length), dtype=torch.long, device=model.device
                        )
                    suffix_mask = prefix_mask.new_ones((1, tokens.shape[1]))
                    full_masks.append(
                        torch.cat(
                            [prefix_mask, pad(suffix_mask, (0, length - tokens.shape[1]))], dim=1
                        )
                    )
                # Updating a branch mutates its cache. Preserve both tensors and
                # cumulative_length metadata; no other suffix sees that mutation.
                branch_cache = copy.deepcopy(prefix_cache)
                branch_cache.batch_repeat_interleave(len(questions))
                suffix_inputs = {
                    "input_ids": ids,
                    "attention_mask": torch.cat(full_masks),
                    "position_ids": torch.arange(
                        prefix_length, prefix_length + length, device=model.device
                    )[None].expand(len(questions), -1),
                    "past_key_values": branch_cache,
                    "mm_token_type_ids": torch.zeros_like(ids),
                }
                if "token_type_ids" in prefix:
                    suffix_inputs["token_type_ids"] = torch.zeros_like(ids)
                # The same complete Unified backbone is used for both passes.
                # All media were consumed in the complete prefix; no media is
                # supplied twice and no model path bypasses its native embedding.
                output = model.backbone(**suffix_inputs, use_cache=True, return_dict=True)
                suffix_calls += 1
                batch_sizes.append(len(questions))
                processed_tokens += len(questions) * length
                hidden = torch.cat(
                    [
                        output.last_hidden_state[index, (slots - prefix_length).to(model.device)]
                        for index, (_, slots, _) in enumerate(prepared)
                    ]
                )
                counts = torch.cat([counts for _, _, counts in prepared]).to(model.device)
                logits = project_candidates(
                    hidden, model.head, model.letters, counts, model.softcap
                )
                probabilities = (logits / model.temperature).softmax(-1).cpu().tolist()
                for question, values in zip(questions, probabilities, strict=True):
                    answers[question.id] = answer_from_probabilities(
                        question, values[: len(question.labels())]
                    )
                del prepared, branch_cache, output, suffix_inputs
            results.append({"answers": answers})
            del prefix_cache, prefix
    execution = {
        "forward_calls": prefix_calls + suffix_calls + reference_calls,
        "prefix_prefill_calls": prefix_calls,
        "suffix_forward_calls": suffix_calls,
        "single_question_reference_calls": reference_calls,
        "optimized_requests": optimized_requests,
        "single_question_reference_requests": reference_requests,
        "max_forward_batch_size": GEMMA_MAX_FORWARD_BATCH_SIZE,
        "max_prepared_prompts": GEMMA_MAX_FORWARD_BATCH_SIZE,
        "batch_sizes": batch_sizes,
        "sequence_tokens": sequence_tokens,
        "processed_tokens": processed_tokens,
        "cached_prefix_tokens": cached_prefix_tokens,
        "scope": "whole_adapter_batch",
        "readout": "candidate",
        "independent_questions": True,
        "prefix_cache": True,
    }
    return [{**result, "execution": copy.deepcopy(execution)} for result in results]
