#!/usr/bin/env python3
"""Assemble a measured NVFP4 package and an explicit external upload whitelist.

The checkpoint, conversion/evaluation receipts and fresh wheel verification
must already exist. No GPU operation, model conversion or Hub upload is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tarfile
from pathlib import Path

from publish_nvfp4 import (
    DEFAULT_REPOSITORY,
    exact_path,
    file_sha256,
    inventory_entries,
    local_file,
    validate_evaluation,
    validate_populations,
    validate_runtime_verification,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "a66f836b56605039fe040f330180e336d19b3362"
JEV_DOCUMENTS = (
    "jev-comparison.md",
    "jev-omni.md",
    "jev-omni-evidence.json",
    "jev-omni-campaign.json",
    "comparison-summary.json",
)
DATASET_FILES = {
    "train": "train.jsonl",
    "calibration": "calibration.jsonl",
    "test": "test.jsonl",
    "media": "media-test.jsonl",
    "regression_text": "regression-text.jsonl",
}


def identity(path, name=None):
    path = Path(path)
    return {
        "path": name or path.name,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def read_json(path):
    return json.loads(Path(path).read_text())


def verified_dataset_archive(path, evaluation):
    """Read and hash the frozen archive without extracting arbitrary paths."""
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [exact_path(member.name) for member in members]
        if len(names) != len(set(names)) or any(not member.isfile() for member in members):
            raise ValueError("frozen dataset archive must contain unique regular files")
        blobs = {}
        for member in members:
            if member.size > 32 * 1024 * 1024:
                raise ValueError("unexpectedly large frozen dataset member")
            with archive.extractfile(member) as stream:
                blobs[member.name] = stream.read()
    manifest = json.loads(blobs["manifest.json"])
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in blobs.items()}
    if hashes["manifest.json"] != evaluation["datasets"]["preparation_manifest_sha256"]:
        raise ValueError("dataset archive manifest differs from the evaluated populations")
    if set(blobs) != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("frozen archive contents differ from its exact manifest")
    for name, expected in manifest["files"].items():
        if hashes[name] != expected:
            raise ValueError(f"frozen dataset archive checksum differs: {name}")
    for split, name in DATASET_FILES.items():
        expected = evaluation["datasets"][split]
        recorded = manifest["datasets"][name]
        if (
            any(
                recorded.get(field) != expected.get(field)
                for field in ("cases", "dataset_sha256", "file_sha256")
            )
            or hashes[name] != expected["file_sha256"]
        ):
            raise ValueError(f"frozen dataset identity differs: {split}")
    return {
        row["id"]: sorted({media["type"] for media in row["request"].get("media", [])})
        for row in (json.loads(line) for line in blobs["media-test.jsonl"].splitlines() if line)
    }


def jev_assets(directory):
    """Select only the named Jev documents and receipt-bound Jev evidence."""
    directory = Path(directory)
    evidence = read_json(directory / "jev-omni-evidence.json")
    names = set(JEV_DOCUMENTS)
    for entry in evidence["files"]:
        name = exact_path(entry["path"])
        if not name.startswith("jev-omni-local/"):
            raise ValueError("Jev source index points outside the Jev evidence directory")
        observed = identity(local_file(directory, name), name)
        if observed["sha256"] != entry["sha256"] or observed["size_bytes"] != entry["bytes"]:
            raise ValueError(f"Jev evidence checksum differs: {name}")
        names.add(name)
    campaign = directory / evidence["archived_campaign"]
    if file_sha256(campaign) != evidence["source_campaign_sha256"]:
        raise ValueError("Jev campaign checksum differs from the archived source index")
    return {f"benchmarks/{name}": local_file(directory, name) for name in sorted(names)}


def checked_requirements(path, evaluation):
    lines = Path(path).read_text().splitlines()
    pins = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ValueError("runtime requirements must use exact package versions")
        name, version = line.split("==")
        if not name or not version or name in pins:
            raise ValueError("runtime requirements contain an invalid or duplicate pin")
        pins[name.lower().replace("_", "-")] = version
    for name, version in evaluation["runtime"]["versions"].items():
        key = name.lower().replace("_", "-")
        if pins.get(key) != version:
            raise ValueError(f"runtime requirement differs from the evaluated package: {name}")
    return pins


def number(value, digits=2):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("the model card requires finite measured metrics")
    return f"{value:.{digits}f}"


def media_accuracy(report, modalities, kind):
    selected = {case_id for case_id, kinds in modalities.items() if kinds == [kind]}
    records = [row for row in report.get("records", []) if row["case_id"] in selected]
    if not selected or {row["case_id"] for row in records} != selected:
        return None
    if len(records) != len(selected) or any(row["status"] != "ok" for row in records):
        raise ValueError("native media card requires complete one-decision-per-case records")
    correct = sum(
        max(row["probabilities"], key=row["probabilities"].get) == row["gold"] for row in records
    )
    return f"{correct}/{len(records)} = {100 * correct / len(records):.2f}%"


def model_card(evaluation, conversion, verification, modalities, wheel, calibration=None):
    reports, baseline = evaluation["reports"], evaluation["bf16_reports"]
    validate_populations(baseline, "BF16 reference")
    runtime = evaluation["runtime"]
    if "B200" not in runtime["gpu"]:
        raise ValueError("this release card requires the measured B200 reference environment")
    rows = []
    for split, label in (
        ("test", "Fresh BoolQ"),
        ("media", "Historical native media"),
        ("regression_text", "Historical BoolQ regression"),
    ):
        for name, report in (("BF16", baseline[split]), ("NVFP4", reports[split])):
            metrics, latency = report["metrics"], report["latency_ms"]
            rows.append(
                f"| {label} | {name} | {report['counts']['ok']}/{metrics['n']} | "
                f"{number(100 * metrics['acc'])}% | {number(metrics['nll'], 6)} | "
                f"{number(metrics['brier'], 6)} | {number(metrics['ece'], 6)} | "
                f"{number(latency['p50'])} / {number(latency['p95'])} |"
            )
    memory_rows = []
    for name, group in (("BF16", baseline), ("NVFP4", reports)):
        memories = [group[split]["memory"] for split in DATASET_FILES if split in group]
        memory_rows.append(
            f"| {name} | {number(memories[0]['model_footprint_bytes'] / 1e9, 3)} | "
            f"{number(max(m['peak_allocated_bytes'] for m in memories) / 1e9, 3)} | "
            f"{number(max(m['peak_reserved_bytes'] for m in memories) / 1e9, 3)} |"
        )
    media_rows = []
    for kind, label in (("image", "MNIST native images"), ("audio", "FSDD native audio")):
        left, right = (
            media_accuracy(group["media"], modalities, kind) for group in (baseline, reports)
        )
        if left is not None and right is not None:
            media_rows.append(f"| {label} | {left} | {right} |")
    media_section = ""
    if media_rows:
        media_section = (
            "\n| Media subset | BF16 accuracy | NVFP4 accuracy |\n"
            "| --- | ---: | ---: |\n" + "\n".join(media_rows) + "\n"
        )
    calibration_details = ""
    if calibration:
        if (
            calibration.get("n") != 256
            or calibration.get("temperature") != evaluation["temperature"]
        ):
            raise ValueError("calibration evidence differs from the evaluated temperature")
        calibration_details = (
            f"Calibration NLL before/after fitting: **{number(calibration['nll_before'], 6)} / "
            f"{number(calibration['nll_after'], 6)}**. See [calibration.json](calibration.json).\n"
        )
    tensor_bytes = sum(
        entry["size_bytes"]
        for name, entry in inventory_entries(conversion["files"]).items()
        if name.endswith(".safetensors")
    )
    version_text = ", ".join(f"{name} {value}" for name, value in runtime["versions"].items())
    parity = evaluation.get("reload_parity", [])
    if len(parity) != 3 or any(row.get("exact_logits_equal") is not True for row in parity):
        raise ValueError("model card requires three exact packed save/reload parity checks")
    tolerance = verification["probability_absolute_tolerance"]
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("runtime verification must record a finite positive tolerance")
    replay_difference = max(
        item["max_absolute_probability_difference"] for item in verification["reports"].values()
    )
    if not math.isfinite(replay_difference) or not 0 <= replay_difference <= tolerance:
        raise ValueError("runtime verification exceeds its declared probability tolerance")
    return f'''---
license: apache-2.0
language:
  - en
base_model: knowlet/Gemma-4-12B-Unified-System-One
base_model_relation: quantized
datasets:
  - google/boolq
tags:
  - gemma4
  - nvfp4
  - system-one
  - multimodal
  - calibrated-classification
---

# Gemma 4 12B Unified System One — NVFP4

Packed NVFP4 language-model linear weights derived from the trained, merged
[BF16 System One checkpoint](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One/tree/{SOURCE_REVISION}).
The model produces calibrated probabilities over declared candidates in one
multimodal backbone forward. The input embedding, candidate head and native
image/audio modules remain BF16. The tensor shards occupy **{tensor_bytes:,} bytes**.

This package uses the **`s1-transformers-nvfp4-v1`** format and its bundled S1
runtime. Stock Hugging Face `AutoModel.from_pretrained` cannot restore this
custom packed format. Compatibility with vLLM, TensorRT-LLM and ModelOpt loaders
has not been established. It is distinct from bitsandbytes NF4.

## Hardware and packed format

Use a **Blackwell CUDA GPU**. This release was evaluated on **{runtime["gpu"]}**;
other Blackwell devices require compatible kernel builds and separate validation.
A100/H100, CPU and MPS are unsupported. The runtime uses an explicit single GPU
and rejects CPU/disk offload or silent precision fallback.

Weights use E2M1 values, E4M3 block scales for groups of 16, and a float32 global
scale. Both swizzled and row-major scales are stored. Dynamic NVFP4 activation
quantization is used for prefill; the kernel's one/two-row path uses W4A16 GEMV.
The kernel is pinned in [nvfp4_manifest.json](nvfp4_manifest.json).
Stored scale buffers and BF16 modules are included in the artifact size.

## Install and run the bundled runtime

Use a fresh Python 3.12 environment. Resolve the repository revision once, then
pin every download to that immutable commit. This example records the commit
locally and verifies the wheel against the fresh-runtime verification receipt.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install huggingface-hub=={runtime["versions"]["huggingface-hub"]}
python - <<'DOWNLOAD'
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

repo = "{DEFAULT_REPOSITORY}"
revision = HfApi().model_info(repo).sha
snapshot_download(repo_id=repo, revision=revision, local_dir="nvfp4-model")
root = Path("nvfp4-model")
(root / "HUB_REVISION").write_text(revision + "\\n")
receipt = json.loads((root / "runtime-verification.json").read_text())
wheel = receipt["runtime_wheel"]
assert hashlib.sha256((root / wheel["filename"]).read_bytes()).hexdigest() == wheel["sha256"]
print("Pinned model revision:", revision)
DOWNLOAD
python -m pip install -r nvfp4-model/requirements-nvfp4.txt nvfp4-model/{wheel}
```

The wheel is installed directly; its historical `inference` extra pins a
different PyTorch version and should not be selected for this NVFP4 environment.

```python
from s1.contracts import DecisionRequest
from s1.unified import UnifiedDecisionModel

model = UnifiedDecisionModel(
    "nvfp4-model",
    device="cuda",
    precision="bfloat16",
    quantization="nvfp4",
    attn_implementation="sdpa",
    max_context=4096,
)
request = DecisionRequest.model_validate({{
    "state": {{"passage": "Paris is the capital of France.",
              "question": "Is Paris the capital of France?"}},
    "questions": {{"answer": {{
        "type": "noul", "instructions": "Answer the question using the supplied passage."
    }}}},
}})
print(model.predict(request)["answers"]["answer"])
```

The saved temperature is loaded automatically. The measured context limit is
4,096 expanded tokens. Generic text generation does not reproduce the candidate
readout or the probabilities measured below.

## Measured BF16 and NVFP4 results on the same B200

All rows below were executed with the same GPU and runtime stack:
**{version_text}**. Both formats use the same frozen case identities and labels.
BF16 retains its previously fitted temperature; NVFP4 is calibrated independently.

| Population | Format | Scored decisions | Accuracy | NLL | Brier | ECE | p50 / p95 request latency (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

| Format | Registered tensors (GB) | Peak CUDA allocated (GB) | Peak CUDA reserved (GB) |
| --- | ---: | ---: | ---: |
{chr(10).join(memory_rows)}

GB is decimal. CUDA peaks include resident weights and the evaluation workload;
they are the maximum across the three reports after the post-load reset, including
warmup. Reserved allocator memory can retain blocks from loading or calibration.
Process RSS has a separate lifetime scope. Peak allocation alone does not
establish the minimum GPU capacity. Request latency includes preprocessing and
one decision forward, excludes model loading and warmup, and is not a throughput
or concurrent-serving benchmark.
{media_section}
The media population comprises native images and audio, without transcripts,
captions or oracle substitutions. These historical digit-recognition cases are
regression evidence, not broad multimodal evaluation. The historical 128 BoolQ
cases are also distinct from the fresh 256-case held-out population.

## Calibration and artifact validation

Only **256 disjoint BoolQ calibration cases** fit the NVFP4 scalar temperature:
**{number(evaluation["temperature"], 12)}**. BF16 temperature:
**{number(evaluation["bf16_temperature"], 12)}**. The native weight quantizer
uses no calibration dataset or label fitting. Media and test labels
are excluded from temperature fitting. This text-only calibration does not
establish calibration on audio, images or deployment traffic.
{calibration_details}
The saved checkpoint was freed and reloaded before calibration and evaluation.
The recorded text/image/audio checks require exactly equal candidate logits
across save/reload. A separate process imported the exact bundled wheel and
replayed all **436 evaluation cases**, requiring matching decisions and candidate
probabilities within **{tolerance:g}**; the observed maximum difference was
**{replay_difference:g}**. See [evaluation.json](evaluation.json),
[runtime-verification.json](runtime-verification.json), and
[conversion-manifest.json](conversion-manifest.json).

The frozen [release-datasets.tar.gz](release-datasets.tar.gz) contains the
train/calibration/test and historical regression data, exact case identities,
source hashes, selection/exclusion policy and separate dataset attributions.

## Jev-Omni comparison: separate historical population

[Jev-Omni's matched benchmark](benchmarks/jev-comparison.md) compares the original
**128 BoolQ cases**, **52 native media cases** and HTTP load measurements with
the archived reference models. Its measured A100 results retain their own
runtime/hardware scope. They must not be ranked against the fresh 256-case
accuracy or treated as a same-hardware NVFP4 timing comparison. The complete
Jev receipts and raw evidence are included under [benchmarks](benchmarks).

## Source, attribution and limits

This release derives from `knowlet/Gemma-4-12B-Unified-System-One` at
`{SOURCE_REVISION}`. The original Google model is
[`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it/tree/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7).
The merged source applies one epoch of LoRA updates on 2,048 English BoolQ examples;
NVFP4 conversion adds no training. Source Apache-2.0 terms and notices are retained
in [LICENSE](LICENSE) and [NOTICE](NOTICE). The bundled S1 implementation is MIT
licensed and builds on [system-one-open](https://github.com/mithalouni/system-one-open).

BoolQ is by Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski,
Michael Collins and Kristina Toutanova (2019), Google Research. BoolQ and the
transformed data are CC-BY-SA-3.0; the historical MNIST and FSDD media retain MIT
and CC-BY-SA-4.0 respectively. Their provenance and license links are preserved
inside the dataset archive and remain distinct from the model license.

This is a small, single-seed English BoolQ specialist evaluation. It does not
establish general assistant quality, multilingual quality, broad multimodal
ability, multi-question independence or production readiness. Foundation-model
pretraining overlap with these public datasets is unknown.
'''


def assemble(args):
    package = args.package.resolve()
    allowlist = (args.allowlist or package.parent / "upload-files.json").resolve()
    if allowlist.is_relative_to(package):
        raise ValueError("the upload whitelist must be outside the package")
    conversion = read_json(package / "conversion-manifest.json")
    evaluation = read_json(package / "evaluation.json")
    verification = read_json(package / "runtime-verification.json")
    if (
        conversion.get("source_model") != "knowlet/Gemma-4-12B-Unified-System-One"
        or conversion.get("source_revision") != SOURCE_REVISION
    ):
        raise ValueError("conversion must derive from the pinned trained BF16 release")
    observed = {}
    for name, expected in inventory_entries(conversion["files"]).items():
        observed[name] = identity(local_file(package, name), name)
        if observed[name] != expected:
            raise ValueError(f"converted checkpoint changed before packaging: {name}")
    for name in ("conversion-manifest.json", "evaluation.json", "runtime-verification.json"):
        observed[name] = identity(local_file(package, name), name)
    validate_evaluation(evaluation, observed["conversion-manifest.json"]["sha256"])
    wheel_name = exact_path(verification["runtime_wheel"]["filename"])
    wheel_source = args.wheel or ROOT / "artifacts/nvfp4/runtime" / wheel_name
    observed[wheel_name] = identity(wheel_source, wheel_name)
    validate_runtime_verification(verification, observed)
    checked_requirements(args.requirements, evaluation)
    modalities = verified_dataset_archive(args.datasets_archive, evaluation)
    assets = jev_assets(args.jev_evidence)
    assets.update(
        {
            wheel_name: wheel_source,
            "requirements-nvfp4.txt": args.requirements,
            "release-datasets.tar.gz": args.datasets_archive,
            "LICENSE": args.publication_assets / "LICENSE",
        }
    )
    evidence = args.evidence or package.parent / "evidence"
    calibration = None
    if (evidence / "calibration.json").is_file():
        calibration = read_json(evidence / "calibration.json")
        assets["calibration.json"] = evidence / "calibration.json"
    if (evidence / "reload-parity.json").is_file():
        if read_json(evidence / "reload-parity.json") != evaluation["reload_parity"]:
            raise ValueError("reload parity evidence differs from evaluation")
        assets["reload-parity.json"] = evidence / "reload-parity.json"
    card = model_card(evaluation, conversion, verification, modalities, wheel_name, calibration)
    notice = (
        (args.publication_assets / "NOTICE").read_text().rstrip()
        + """

NVFP4 derivative (October 7, 2026): the trained merged BF16 source was converted
into packed native NVFP4 language-model linear weights using the pinned
Transformers kernel. Input embeddings, candidate readout, and native image/audio
modules remain BF16. The s1-transformers-nvfp4-v1 artifact preserves both scale
layouts and tied-weight aliases. A new scalar temperature uses only the separate
256-case calibration population. The bundled MIT-licensed S1 runtime is required
to restore this custom packed format. Source and evaluation identities are
recorded in conversion-manifest.json and evaluation.json.
"""
    )
    # Validate sources before making any package changes. Copy only explicitly
    # chosen documents and indexed Jev evidence; never recurse through this date.
    for name, source in assets.items():
        exact_path(name)
        if not Path(source).is_file() or Path(source).is_symlink():
            raise ValueError(f"publication asset is missing or a symlink: {source}")
        if name in observed and name != wheel_name:
            raise ValueError(f"publication asset would replace a checkpoint file: {name}")
    for name, source in assets.items():
        destination = package / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if Path(source).resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        observed[name] = identity(destination, name)
    for name, content in (("README.md", card), ("NOTICE", notice)):
        if name in observed:
            raise ValueError(f"generated document would replace a checkpoint file: {name}")
        (package / name).write_text(content)
        observed[name] = identity(package / name, name)
    artifact_manifest = {
        "schema_version": 1,
        "format": "nvfp4",
        "repository": DEFAULT_REPOSITORY,
        "source_model": conversion["source_model"],
        "source_revision": conversion["source_revision"],
        "conversion_manifest_sha256": observed["conversion-manifest.json"]["sha256"],
        "evaluation_sha256": observed["evaluation.json"]["sha256"],
        "runtime_verification_sha256": observed["runtime-verification.json"]["sha256"],
        "files": [observed[name] for name in sorted(observed)],
    }
    write_json(package / "artifact-manifest.json", artifact_manifest)
    names = sorted([*observed, "artifact-manifest.json"])
    write_json(allowlist, names)
    return {
        "status": "packaged",
        "package": str(package),
        "allowlist": str(allowlist),
        "files": len(names),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--requirements", type=Path, default=ROOT / "requirements-nvfp4.txt")
    parser.add_argument(
        "--datasets-archive",
        type=Path,
        default=ROOT / "docs/validation/2026-10-02/release-datasets.tar.gz",
    )
    parser.add_argument(
        "--publication-assets", type=Path, default=ROOT / "artifacts/nvfp4/publication-assets"
    )
    parser.add_argument("--jev-evidence", type=Path, default=ROOT / "docs/validation/2026-10-07")
    parser.add_argument("--evidence", type=Path)
    result = assemble(parser.parse_args(argv))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
