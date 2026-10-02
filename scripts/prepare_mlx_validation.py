#!/usr/bin/env python3
"""Capture exact S1 processor tensors for an experimental cross-runtime check."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path):
    """Hash only weight/config files, streaming shards; excludes mutable sidecars."""
    path = Path(path)
    names = {"config.json"}
    indexes = sorted(path.glob("*.safetensors.index.json"))
    if any(index.name != "model.safetensors.index.json" for index in indexes):
        raise ValueError("only the canonical model.safetensors.index.json layout is supported")
    if indexes:
        for index in indexes:
            names.add(index.name)
            mapping = json.loads(index.read_text()).get("weight_map", {})
            if not mapping:
                raise ValueError("checkpoint index must contain a nonempty weight_map")
            for name in mapping.values():
                if (
                    not isinstance(name, str)
                    or Path(name).name != name
                    or not name.endswith(".safetensors")
                ):
                    raise ValueError("checkpoint weight paths must be local safetensors filenames")
                names.add(name)
    else:
        names.add("model.safetensors")
    weights = {item.name for item in path.glob("*.safetensors")}
    if weights != {name for name in names if name.endswith(".safetensors")}:
        raise ValueError("checkpoint contains missing or unindexed safetensors weights")
    files = [
        {
            "path": name,
            "size_bytes": (path / name).stat().st_size,
            "sha256": file_sha256(path / name),
        }
        for name in sorted(names)
    ]
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"sha256": digest, "files": files}


def main(argv=None):
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoProcessor

    from s1.checkpoint import load_temperature
    from s1.evaluation.datasets import fingerprint, load_cases, request_fingerprint
    from s1.unified import UnifiedDecisionModel, candidate_ids

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument(
        "--revision", help="Required for Hub models; local checkpoints use metadata"
    )
    parser.add_argument("--dataset", default="examples/benchmarks/mps.jsonl")
    parser.add_argument("--split", choices=("calibration", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cases = load_cases(args.dataset)
    if any(case.split != args.split for case in cases):
        raise ValueError(f"validation requires only {args.split} cases")
    identity = None
    if Path(args.model).is_dir():
        source = Path(args.model)
        sidecar_path = source / "s1_config.json"
        sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
        revision = sidecar.get("source_revision") or args.revision
        source_model = sidecar.get("source_model") or str(source.resolve())
        if args.revision and sidecar.get("source_revision") not in (None, args.revision):
            raise ValueError("requested revision differs from local checkpoint metadata")
        identity = checkpoint_identity(source)
    else:
        if not args.revision:
            raise ValueError("Hub input capture requires an explicit --revision")
        revision = HfApi().model_info(args.model, revision=args.revision).sha
        source_model = args.model
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
        "schema_version": 2,
        "source_model": source_model,
        "source_revision": revision,
        "source_checkpoint_sha256": identity["sha256"] if identity else None,
        "source_checkpoint_files": identity["files"] if identity else None,
        "source_s1_config_sha256": (
            file_sha256(Path(source) / "s1_config.json")
            if (Path(source) / "s1_config.json").exists()
            else None
        ),
        "dataset_sha256": fingerprint(cases),
        "dataset_file_sha256": file_sha256(args.dataset),
        "split": args.split,
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
                "tensor_file_sha256": file_sha256(args.output / filename),
                "split": case.split,
                "gold": case.gold,
                "request_sha256": request_fingerprint(case.request),
                "group_id": case.group_id or case.id,
                "task_id": case.task_id,
                "language": case.language,
                "schema_id": case.schema_id,
                "modalities": sorted({media.type for media in case.request.media}) or ["text"],
                "slots": slots.tolist(),
                "nopts": nopts.tolist(),
                "questions": [
                    {"id": q.id, "type": q.type, "labels": q.labels()}
                    for q in case.request.questions
                ],
            }
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(cases)} {args.split} cases from {revision} to {args.output}")


if __name__ == "__main__":
    main()
