"""Diagnostic ablation only: preserve absolute key-axis alignment on CPU BF16.

The original production implementation and winner tests remain unchanged.
This process-local patch tests a numerical workaround, not model qualification.
"""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path
from unittest.mock import patch


def main():
    import pytest
    import torch
    from torch.nn import functional as functional
    from transformers.cache_utils import DynamicSlidingWindowLayer

    from diagnose_browser_prefix import run_case

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fixture = runpy.run_path(str(root / "tests/test_browser_prefix.py"))
    original_update = DynamicSlidingWindowLayer.update
    original_mask_sizes = DynamicSlidingWindowLayer.get_mask_sizes

    def needs_alignment(layer):
        return (
            layer.is_initialized
            and layer.keys.dtype == torch.bfloat16
            and layer.keys.device.type == "cpu"
        )

    def aligned_update(layer, key_states, value_states, *positional, **kwargs):
        keys, values = original_update(layer, key_states, value_states, *positional, **kwargs)
        missing = layer.get_seq_length() - keys.shape[-2]
        if needs_alignment(layer) and missing > 0:
            keys = functional.pad(keys, (0, 0, missing, 0))
            values = functional.pad(values, (0, 0, missing, 0))
        return keys, values

    def aligned_mask_sizes(layer, query_length):
        if needs_alignment(layer):
            return layer.get_seq_length() + query_length, 0
        return original_mask_sizes(layer, query_length)

    with (
        patch.object(DynamicSlidingWindowLayer, "update", aligned_update),
        patch.object(DynamicSlidingWindowLayer, "get_mask_sizes", aligned_mask_sizes),
    ):
        report = run_case(fixture, torch.bfloat16, "sdpa")
        # Padded keys are intentionally zero only where the original sliding
        # window disallows attention; they are not restored historical KV.
        report["ablation"] = "cpu-bf16-absolute-key-axis-zero-padding"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False))
        code = pytest.main(["tests/test_browser_prefix.py", "-q"])
    raise SystemExit(code)


if __name__ == "__main__":
    main()
