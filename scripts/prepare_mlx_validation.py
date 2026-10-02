#!/usr/bin/env python3
"""Capture exact S1 processor tensors for an experimental cross-runtime check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s1.benchmark import fingerprint, load_cases
from s1.checkpoint import load_temperature
from s1.unified import UnifiedDecisionModel, candidate_ids


def main():
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dataset", default="examples/benchmarks/mps.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    if any(case.split != "test" for case in cases):
        raise ValueError("validation requires test cases")
    revision = HfApi().model_info(args.model, revision=args.revision).sha
    source = snapshot_download(
        args.model, revision=revision, allow_patterns=["*.json", "*.jinja", "*.model"]
    )
    processor = AutoProcessor.from_pretrained(source)
    # prepare() uses only these fields. Reuse the authoritative implementation
    # while avoiding loading 12B weights just to capture CPU input tensors.
    context = SimpleNamespace(
        processor=processor, tok=processor.tokenizer, device="cpu", max_context=16384
    )
    manifest = {
        "source_model": args.model,
        "source_revision": revision,
        "dataset_sha256": fingerprint(cases),
        "prompt_version": 1,
        "temperature": load_temperature(source),
        "letters": candidate_ids(processor.tokenizer),
        "cases": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases):
        inputs, slots, nopts = UnifiedDecisionModel.prepare(context, case.request)
        filename = f"case-{index}.npz"
        np.savez(args.output / filename, **{key: value.numpy() for key, value in inputs.items()})
        manifest["cases"].append(
            {
                "id": case.id,
                "file": filename,
                "slots": slots.tolist(),
                "nopts": nopts.tolist(),
                "questions": [{"id": q.id, "labels": q.labels()} for q in case.request.questions],
            }
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(cases)} cases from {revision} to {args.output}")


if __name__ == "__main__":
    main()
