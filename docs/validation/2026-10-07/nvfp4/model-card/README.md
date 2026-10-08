---
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
[BF16 System One checkpoint](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One/tree/a66f836b56605039fe040f330180e336d19b3362).
The model produces calibrated probabilities over declared candidates in one
multimodal backbone forward. The input embedding, candidate head and native
image/audio modules remain BF16. The tensor shards occupy **8,932,247,864 bytes**.

This package uses the **`s1-transformers-nvfp4-v1`** format and its bundled S1
runtime. Stock Hugging Face `AutoModel.from_pretrained` cannot restore this
custom packed format. Compatibility with vLLM, TensorRT-LLM and ModelOpt loaders
has not been established. It is distinct from bitsandbytes NF4.

## Hardware and packed format

Use a **Blackwell CUDA GPU**. This release was evaluated on **NVIDIA B200**;
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
python -m pip install huggingface-hub==1.33.0
python - <<'DOWNLOAD'
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

repo = "knowlet/Gemma-4-12B-Unified-System-One-NVFP4"
revision = HfApi().model_info(repo).sha
snapshot_download(repo_id=repo, revision=revision, local_dir="nvfp4-model")
root = Path("nvfp4-model")
(root / "HUB_REVISION").write_text(revision + "\n")
receipt = json.loads((root / "runtime-verification.json").read_text())
wheel = receipt["runtime_wheel"]
assert hashlib.sha256((root / wheel["filename"]).read_bytes()).hexdigest() == wheel["sha256"]
print("Pinned model revision:", revision)
DOWNLOAD
python -m pip install -r nvfp4-model/requirements-nvfp4.txt nvfp4-model/gemma_system_one-0.1.0-py3-none-any.whl
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
request = DecisionRequest.model_validate({
    "state": {"passage": "Paris is the capital of France.",
              "question": "Is Paris the capital of France?"},
    "questions": {"answer": {
        "type": "noul", "instructions": "Answer the question using the supplied passage."
    }},
})
print(model.predict(request)["answers"]["answer"])
```

The saved temperature is loaded automatically. The measured context limit is
4,096 expanded tokens. Generic text generation does not reproduce the candidate
readout or the probabilities measured below.

## Measured BF16 and NVFP4 results on the same B200

All rows below were executed with the same GPU and runtime stack:
**torch 2.10.0, transformers 5.17.0, kernels 0.16.0, huggingface-hub 1.33.0, safetensors 0.8.0**. Both formats use the same frozen case identities and labels.
BF16 retains its previously fitted temperature; NVFP4 is calibrated independently.

| Population | Format | Scored decisions | Accuracy | NLL | Brier | ECE | p50 / p95 request latency (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh BoolQ | BF16 | 256/256 | 89.84% | 0.273379 | 0.158446 | 0.054573 | 38.68 / 46.92 |
| Fresh BoolQ | NVFP4 | 256/256 | 88.67% | 0.300083 | 0.178014 | 0.045837 | 45.44 / 49.64 |
| Historical native media | BF16 | 52/52 | 61.54% | 1.083374 | 0.426739 | 0.203648 | 46.95 / 48.63 |
| Historical native media | NVFP4 | 52/52 | 59.62% | 1.327612 | 0.518533 | 0.152375 | 56.52 / 57.26 |
| Historical BoolQ regression | BF16 | 128/128 | 92.19% | 0.248382 | 0.136476 | 0.063173 | 30.53 / 42.93 |
| Historical BoolQ regression | NVFP4 | 128/128 | 89.06% | 0.276518 | 0.161293 | 0.070453 | 45.72 / 47.96 |

| Format | Registered tensors (GB) | Peak CUDA allocated (GB) | Peak CUDA reserved (GB) |
| --- | ---: | ---: | ---: |
| BF16 | 23.919 | 24.188 | 24.623 |
| NVFP4 | 8.932 | 9.143 | 9.771 |

GB is decimal. CUDA peaks include resident weights and the evaluation workload;
they are the maximum across the three reports after the post-load reset, including
warmup. Reserved allocator memory can retain blocks from loading or calibration.
Process RSS has a separate lifetime scope. Peak allocation alone does not
establish the minimum GPU capacity. Request latency includes preprocessing and
one decision forward, excludes model loading and warmup, and is not a throughput
or concurrent-serving benchmark.

| Media subset | BF16 accuracy | NVFP4 accuracy |
| --- | ---: | ---: |
| MNIST native images | 28/32 = 87.50% | 28/32 = 87.50% |
| FSDD native audio | 4/20 = 20.00% | 3/20 = 15.00% |

The media population comprises native images and audio, without transcripts,
captions or oracle substitutions. These historical digit-recognition cases are
regression evidence, not broad multimodal evaluation. The historical 128 BoolQ
cases are also distinct from the fresh 256-case held-out population.

## Calibration and artifact validation

Only **256 disjoint BoolQ calibration cases** fit the NVFP4 scalar temperature:
**2.518551960907**. BF16 temperature:
**2.783034447038**. The native weight quantizer
uses no calibration dataset or label fitting. Media and test labels
are excluded from temperature fitting. This text-only calibration does not
establish calibration on audio, images or deployment traffic.
Calibration NLL before/after fitting: **0.518934 / 0.311825**. See [calibration.json](calibration.json).

The saved checkpoint was freed and reloaded before calibration and evaluation.
The recorded text/image/audio checks require exactly equal candidate logits
across save/reload. A separate process imported the exact bundled wheel and
replayed all **436 evaluation cases**, requiring matching decisions and candidate
probabilities within **1e-06**; the observed maximum difference was
**0**. See [evaluation.json](evaluation.json),
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
`a66f836b56605039fe040f330180e336d19b3362`. The original Google model is
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
