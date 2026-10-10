"""Controlled Gemma readouts; native batches preserve all processor tensors."""

from __future__ import annotations

from collections import defaultdict

from s1.backends import GEMMA_MAX_FORWARD_BATCH_SIZE
from s1.contracts import answer_from_probabilities
from s1.unified import project_candidates

SEQUENCE_KEYS = {
    "input_ids",
    "attention_mask",
    "mm_token_type_ids",
    "token_type_ids",
    "position_ids",
}


def _bucket_key(inputs):
    # Media tensors are concatenated in sequence order. Different media shapes go
    # to separate real forwards; padding image/audio features with invented data
    # would change the processor contract.
    return tuple(
        (key, tuple(value.shape[1:]) if key not in SEQUENCE_KEYS else (), str(value.dtype))
        for key, value in sorted(inputs.items())
    )


def _collate(prepared, pad_id):
    import torch
    from torch.nn.functional import pad

    length = max(inputs["input_ids"].shape[1] for inputs, _, _ in prepared)
    result = {}
    for key in prepared[0][0]:
        values = [inputs[key] for inputs, _, _ in prepared]
        if key in SEQUENCE_KEYS:
            if any(value.ndim != 2 or value.shape[0] != 1 for value in values):
                raise ValueError("unsupported processor sequence tensor shape")
            fill = pad_id if key == "input_ids" else 0
            values = [pad(value, (0, length - value.shape[1]), value=fill) for value in values]
        result[key] = torch.cat(values, dim=0)
    return result


def predict_batch(model, requests, *, independent=True, readout="candidate"):
    """Native prompts with bounded pending tensors and compatible forward groups.

    Independent mode still uses one prompt per question. Causal mode retains
    each complete ordered multislot request as one prompt.
    """
    import torch

    if readout not in ("candidate", "full"):
        raise ValueError("native batching requires candidate or full readout")
    if not requests:
        raise ValueError("empty batch")
    buckets, answers = defaultdict(list), [{} for _ in requests]
    batch_sizes, token_counts = [], []
    model.lm.eval()
    with torch.inference_mode():

        def forward_group(items):
            prepared = [item[2] for item in items]
            inputs = _collate(prepared, getattr(model.tok, "pad_token_id", None) or 0)
            batch_sizes.append(len(items))
            out = model.backbone(**inputs, use_cache=False, return_dict=True)
            hidden = torch.cat(
                [out.last_hidden_state[i, slots] for i, (_, slots, _) in enumerate(prepared)]
            )
            counts = torch.cat([p[2] for p in prepared])
            if readout == "candidate":
                logits = project_candidates(
                    hidden, model.head, model.letters, counts, model.softcap
                )
            else:
                logits = model.head(hidden.to(model.head.weight.dtype))
                if model.softcap is not None:
                    logits = torch.tanh(logits / model.softcap) * model.softcap
                logits = logits[:, model.letters].float()
                mask = (
                    torch.arange(len(model.letters), device=logits.device)[None] >= counts[:, None]
                )
                logits = logits.masked_fill(mask, float("-inf"))
            probabilities = (logits / model.temperature).softmax(-1).cpu().tolist()
            offset = 0
            for index, questions, _ in items:
                for question in questions:
                    answers[index][question.id] = answer_from_probabilities(
                        question, probabilities[offset][: len(question.labels())]
                    )
                    offset += 1

        pending = 0
        for index, request in enumerate(requests):
            groups = [[q] for q in request.questions] if independent else [request.questions]
            for questions in groups:
                part = request.model_copy(update={"questions": questions})
                prepared = model.prepare(part)
                token_counts.append(prepared[0]["input_ids"].shape[1])
                buckets[_bucket_key(prepared[0])].append((index, questions, prepared))
                pending += 1
                # Clear this extra reference before flushing; removed groups must
                # not keep native media/sequence tensors alive until the next prepare.
                prepared = None
                if pending == GEMMA_MAX_FORWARD_BATCH_SIZE:
                    key = max(buckets, key=lambda key: len(buckets[key]))
                    pending -= len(buckets[key])
                    forward_group(buckets.pop(key))
        while buckets:
            forward_group(buckets.pop(next(iter(buckets))))
    return [
        {
            "answers": {question.id: result[question.id] for question in request.questions},
            "execution": {
                "forward_calls": len(batch_sizes),
                "max_forward_batch_size": GEMMA_MAX_FORWARD_BATCH_SIZE,
                "max_prepared_prompts": GEMMA_MAX_FORWARD_BATCH_SIZE,
                "batch_sizes": batch_sizes,
                "sequence_tokens": token_counts,
                "scope": "whole_adapter_batch",
                "readout": readout,
                "independent_questions": independent,
            },
        }
        for request, result in zip(requests, answers, strict=True)
    ]


def predict_sequential(model, request, *, readout="candidate"):
    answers = {}
    for question in request.questions:
        part = request.model_copy(update={"questions": [question]})
        result = predict_batch(model, [part], readout=readout)[0]
        answers.update(result["answers"])
    return {"answers": answers}


def predict_generated(model, request):
    """G0: exactly one constrained answer token per question, thinking disabled.

    Generation scores are deliberately not presented as complete probabilities.
    """
    import torch

    answers = {}
    model.lm.eval()
    with torch.inference_mode():
        for question in request.questions:
            inputs, _, _ = model.prepare(request.model_copy(update={"questions": [question]}))
            allowed = model.letters[: len(question.labels())].tolist()
            output = model.lm.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                use_cache=False,
                prefix_allowed_tokens_fn=lambda batch_id, ids: allowed,
                pad_token_id=getattr(model.tok, "pad_token_id", None) or 0,
            )
            token = output[0, -1].item()
            if token not in allowed:
                raise ValueError("generated token is outside the candidate set")
            answers[question.id] = {"label": question.labels()[allowed.index(token)]}
    return {"answers": answers}
