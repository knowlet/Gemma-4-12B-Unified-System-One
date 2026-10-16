"""Diagnose the existing tiny Gemma prefix regression; never qualify a model.

Run with the repository's locked inference extra. All models are initialized
locally from the existing test fixture; no weights or historical wire are read.
The original winner assertion is not changed by this diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import runpy
import subprocess
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def difference(left, right):
    import torch

    if left.shape != right.shape:
        return {"left_shape": list(left.shape), "right_shape": list(right.shape)}
    equal = torch.equal(left, right)
    left, right = left.double(), right.double()
    return {
        "equal": equal,
        "max_abs": float((left - right).abs().max()) if left.numel() else 0.0,
        "different_elements": int((left != right).sum()),
        "elements": left.numel(),
    }


def run_case(fixture, dtype, implementation):
    import torch
    from torch.nn import functional as functional

    from s1.browser_prefix import predict_prefix
    from s1.unified import project_candidates

    model = fixture["model"].__wrapped__()
    model.lm.to(dtype=dtype)
    model.backbone.config.text_config._attn_implementation = implementation
    rows = fixture["prepared_rows"](model, "goal-first")
    active = {"phase": "full", "index": -1, "layer": ""}
    captures, kernels, layers = {}, {}, {}
    original_sdpa = functional.scaled_dot_product_attention
    handles = []

    def before_backbone(_module, _args, kwargs):
        active["index"] += 1
        active["tag"] = f'{active["phase"]}-{active["index"]}'
        captures[active["tag"]] = {
            "positions": kwargs.get("position_ids").detach().cpu().clone(),
            "use_cache": kwargs.get("use_cache"),
        }

    def after_backbone(_module, _args, _kwargs, output):
        captures[active["tag"]]["hidden"] = output.last_hidden_state[:, -1].detach().clone()

    def before_attention(module, _args, _kwargs):
        active["layer"] = str(module.layer_idx)

    def after_layer(module, _args, _kwargs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        layers[(active["tag"], str(module.layer_idx))] = hidden.detach().cpu().clone()

    def inspect_sdpa(query, key, value, *args, **kwargs):
        mask = kwargs.get("attn_mask", args[0] if args else None)
        qlength, klength = query.shape[-2], key.shape[-2]
        allowed = torch.ones((qlength, klength), dtype=torch.bool, device=query.device)
        if kwargs.get("is_causal", False):
            allowed = allowed.tril()
        if mask is not None:
            allowed = allowed & (mask if mask.dtype == torch.bool else mask == 0)
        kernels[(active["tag"], active["layer"])] = {
            "query": query.detach().cpu().clone(),
            "key": key.detach().cpu().clone(),
            "value": value.detach().cpu().clone(),
            "allowed": allowed.detach().cpu().clone(),
        }
        return original_sdpa(query, key, value, *args, **kwargs)

    handles.append(model.backbone.register_forward_pre_hook(before_backbone, with_kwargs=True))
    handles.append(model.backbone.register_forward_hook(after_backbone, with_kwargs=True))
    for module in model.backbone.modules():
        if module.__class__.__name__ == "Gemma4UnifiedTextAttention":
            handles.append(module.register_forward_pre_hook(before_attention, with_kwargs=True))
        if module.__class__.__name__ == "Gemma4UnifiedTextDecoderLayer":
            handles.append(module.register_forward_hook(after_layer, with_kwargs=True))
    try:
        with torch.inference_mode(), patch.object(
            functional, "scaled_dot_product_attention", inspect_sdpa
        ):
            expected = fixture["full_answers"](model, rows)
            active.update(phase="prefix", index=-1)
            actual = predict_prefix(model, rows, prompt_mode="goal-first")
    finally:
        for handle in handles:
            handle.remove()

    trace = actual[0]["prefix"]
    prefix = trace["prefix_tokens"]
    comparisons = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            full_tag, suffix_tag = f"full-{index}", f"prefix-{index + 1}"
            count = int(row["prepared"][2][0])
            full_hidden = captures[full_tag]["hidden"]
            suffix_hidden = captures[suffix_tag]["hidden"]
            question_id = row["request"].questions[0].id
            answer = actual[index]["answers"][question_id]
            logits = {}
            for label, hidden in (("full", full_hidden), ("suffix", suffix_hidden)):
                counts = torch.tensor([count], device=hidden.device)
                selected = project_candidates(
                    hidden, model.head, model.letters, counts, model.softcap
                )
                native_full = model.head(hidden)
                if model.softcap is not None:
                    native_full = (native_full / model.softcap).tanh() * model.softcap
                native_full = native_full.float()[:, model.letters][:, :count]
                weights = model.head.weight[model.letters]
                bias = getattr(model.head, "bias", None)
                if bias is not None:
                    bias = bias[model.letters]
                raw = functional.linear(hidden.to(weights.dtype), weights, bias)
                fp32 = functional.linear(
                    hidden.float(), weights.float(), None if bias is None else bias.float()
                )
                logits[label] = {
                    "raw": raw[0, :count].float().tolist(),
                    "softcapped": selected[0, :count].tolist(),
                    "fp32_projection_raw": fp32[0, :count].tolist(),
                    "selected_vs_full_head_same_hidden": difference(
                        selected[:, :count], native_full
                    ),
                }
            layer_comparisons = []
            for layer in range(model.backbone.config.text_config.num_hidden_layers):
                name = str(layer)
                full = kernels.get((full_tag, name))
                suffix = kernels.get((suffix_tag, name))
                item = {
                    "layer": layer,
                    "type": model.backbone.config.text_config.layer_types[layer],
                    "prefix_hidden": difference(
                        layers[(full_tag, name)][:, :prefix], layers[("prefix-0", name)]
                    ),
                    "suffix_hidden": difference(
                        layers[(full_tag, name)][:, prefix:], layers[(suffix_tag, name)]
                    ),
                }
                if full is not None and suffix is not None:
                    offset = full["key"].shape[-2] - suffix["key"].shape[-2]
                    item.update(
                        key_absolute_offset=offset,
                        query=difference(full["query"][..., prefix:, :], suffix["query"]),
                        key=difference(full["key"][..., offset:, :], suffix["key"]),
                        value=difference(full["value"][..., offset:, :], suffix["value"]),
                        mask=difference(
                            full["allowed"][..., prefix:, offset:], suffix["allowed"]
                        ),
                    )
                layer_comparisons.append(item)
            comparisons.append(
                {
                    "row": index,
                    "options": count,
                    "full_choice": expected[index]["choice"],
                    "prefix_choice": answer["choice"],
                    "winner_equal": expected[index]["choice"] == answer["choice"],
                    "final_hidden": difference(full_hidden, suffix_hidden),
                    "position_ids_equal": torch.equal(
                        captures[full_tag]["positions"][:, prefix:],
                        captures[suffix_tag]["positions"],
                    ),
                    "logits": logits,
                    "layers": layer_comparisons,
                }
            )
    return {
        "dtype": str(dtype),
        "attention": implementation,
        "prefix_tokens": prefix,
        "context_tokens": trace["context_tokens"],
        "cache_layers": trace["cache_layers"],
        "rows": comparisons,
    }


def main():
    import torch
    import transformers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture_path = ROOT / "tests/test_browser_prefix.py"
    fixture = runpy.run_path(str(fixture_path))
    report = {
        "schema": "browser-prefix-numerics-v1",
        "scope": "Existing seeded tiny CPU Gemma fixture, not 12B model qualification",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "threads": torch.get_num_threads(),
        "cases": [],
    }
    for dtype, attention in (
        (torch.bfloat16, "sdpa"),
        (torch.bfloat16, "eager"),
        (torch.float32, "sdpa"),
    ):
        report["cases"].append(run_case(fixture, dtype, attention))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
