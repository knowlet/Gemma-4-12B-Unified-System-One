---
license: apache-2.0
language:
  - en
base_model: knowlet/Gemma-4-12B-Unified-System-One
base_model_relation: quantized
library_name: mlx-vlm
datasets:
  - google/boolq
tags:
  - gemma4
  - system-one
  - boolq
  - mlx
  - mlx-vlm
  - 8-bit
  - multimodal
---

# Gemma 4 12B Unified System One — MLX 8-bit

The complete multimodal MLX 8-bit conversion of the merged
[BF16 checkpoint](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One).
The `base_model` metadata identifies that source release; its exact weight
snapshot is pinned below. The reproduction workflow retains both downloaded
checkpoints locally so that input capture uses the original BF16 processor and
calibration sidecar.
This export retains the language backbone, native vision/audio modules,
tokenizer and processor. The source model is a **BoolQ specialist**, trained with one epoch of
LoRA on 2,048 English examples and merged into the full 12B checkpoint before
conversion. It is not full-parameter optimization.

The S1 interface for this artifact is an **experimental batch-size-one,
selected-head MLX-VLM path**. A generic MLX chat or generation runner does not
implement that decision contract or return its calibrated candidate probabilities.

**Release date:** 2026-10-04 (Asia/Taipei)  
**Public MLX repository:** [knowlet/Gemma-4-12B-Unified-System-One-MLX-8bit](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-MLX-8bit)  
**Immutable HF weights revision:** `e304ce0b87148933ce43d5ef9bed779f048accc8`  
**Source BF16 weights revision:** `66626de5cbcf8c5fecb8e9b58710805a02a42f67`  
**Export checkpoint SHA-256:** `7e81323f4bfcf98528d65e5abccb8a60fe7f83ecb0370ca88b32426cc66da000`  
**Weight files:** 3 files, 12754909484 bytes

## Runtime and scope

The conversion uses **MLX-VLM 0.7.4**, affine **8-bit** weight quantization with
**group size 64**. Not every tensor is stored in 8 bits: the file inventory
accounts for all **677 source tensors**, with 330 quantized weight tensors and
347 source tensors retained in BF16. Quantized weights use packed U32 storage
with separate scales and offsets. The inventory includes `vision_embedder`,
`embed_vision` and `embed_audio`; native vision/audio tensor retention was checked.
The exact converted files, conversion versions, and source checkpoint hashes
are recorded in [`conversion-manifest.json`](conversion-manifest.json), also
retained locally at `artifacts/release/mlx-conversion-manifest.json`.

The experimental S1 path consumes the exact input tensors captured by the
Transformers processor and the repository's `UnifiedDecisionModel.prepare()`.
It preserves answer-slot locations and multimodal token types, runs the complete
multimodal backbone with no generation or KV cache, and dequantizes only the
selected tied embedding rows for the candidate projection. It applies the native
logit softcap once, masks illegal candidates, and uses the separately fitted MLX
temperature. It does not substitute a full-vocabulary generation call.

Only English passage-grounded BoolQ yes/no decisions were trained. Native image
and audio modules are retained, but their quality is described solely by the
regression results below. Choice, Score, multilingual, multi-question, broad
multimodal, and production suitability are not established by this release.

## Download and reproduce the S1 evaluation

Use Apple Silicon with space for the approximately 24 GB BF16 source and 13 GB
MLX export together. Clone the repository into a fresh checkout and install the
pinned Transformers environment:

```bash
git clone https://github.com/knowlet/Gemma-4-12B-Unified-System-One.git
cd Gemma-4-12B-Unified-System-One
git checkout d4ea1ef79af2e22f666374e529fa56c1de9d3f53
uv sync --locked --extra inference --extra train
```

Run the following with `uv run --no-sync python` from that repository root. It
downloads both immutable snapshots to the paths expected by the reproduction
scripts and copies the published conversion receipt without changing it. Use
empty destinations for this fresh reproduction.

```python
from pathlib import Path
from huggingface_hub import snapshot_download

source_directory = Path(snapshot_download(
    repo_id="knowlet/Gemma-4-12B-Unified-System-One",
    revision="66626de5cbcf8c5fecb8e9b58710805a02a42f67",
    local_dir="artifacts/release/checkpoint",
))
model_directory = Path(snapshot_download(
    repo_id="knowlet/Gemma-4-12B-Unified-System-One-MLX-8bit",
    revision="e304ce0b87148933ce43d5ef9bed779f048accc8",
    local_dir="artifacts/exports/s1-boolq-mlx-8bit",
))
conversion_receipt = Path("artifacts/release/mlx-conversion-manifest.json")
conversion_receipt.parent.mkdir(parents=True, exist_ok=True)
conversion_receipt.write_bytes(
    (model_directory / "conversion-manifest.json").read_bytes()
)
assert (model_directory / "base_s1_config.json").read_bytes() == (
    source_directory / "s1_config.json"
).read_bytes()
```

Create the separate MLX environment; this keeps its Transformers 5.18.0
requirement separate from the locked Transformers 5.17.0 input-capture runtime:

```bash
uv venv --python 3.12 artifacts/mlx-env
uv pip install --python artifacts/mlx-env/bin/python -e . \
  mlx-vlm==0.7.4 mlx==0.32.3 mlx-metal==0.32.3 \
  transformers==5.18.0 huggingface-hub==1.33.0 numpy==2.5.3
```

The [release instructions](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/434909da45fbfbcdba7d55cc9b431a6236134335/docs/release.md)
and [reproduction guide](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/434909da45fbfbcdba7d55cc9b431a6236134335/docs/release-reproduce.md)
provide the frozen dataset archive, capture commands, SHA-256 pins and complete
evaluator command. For the downloaded snapshots above:

```bash
git archive --format=tar 434909da45fbfbcdba7d55cc9b431a6236134335 \
  docs/validation/2026-10-02/release-datasets.tar.gz | tar -xf -
mkdir -p artifacts/release/datasets
tar -xzf docs/validation/2026-10-02/release-datasets.tar.gz \
  -C artifacts/release/datasets
```

These commands obtain the frozen dataset archive without changing the pinned
runtime sources. The guide records the original local workflow, including a
Modal download and new MLX conversion; the Hub snapshots above replace those
steps.

1. In the locked Transformers environment, run the guide's three
   [`prepare_mlx_validation.py`](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/d4ea1ef79af2e22f666374e529fa56c1de9d3f53/scripts/prepare_mlx_validation.py)
   commands against `artifacts/release/checkpoint`. Capture 256 calibration cases
   with `--split calibration`, and the 256 fresh test plus 52 media cases with
   `--split test`. Input capture uses the original BF16 `s1_config.json`.
2. Keep the downloaded MLX weights, both sidecars and conversion receipt. Skip
   the guide's conversion, BF16-sidecar copy and conversion-receipt regeneration
   commands when evaluating this published snapshot. Follow its **Pin and
   evaluate** section using `artifacts/mlx-env/bin/python` and
   [`evaluate_mlx_release.py`](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/d4ea1ef79af2e22f666374e529fa56c1de9d3f53/scripts/evaluate_mlx_release.py).
   That command hashes this reproduction's captured-input manifests, rather
   than assuming regenerated NPZ files match the historical byte hashes.

The published export is already calibrated. Its `s1_config.json` contains the
MLX temperature, while `base_s1_config.json` retains the exact original merged
BF16 sidecar. At the pinned runtime, the evaluator reads `base_s1_config.json`
when present and verifies its hash against the captured BF16 source; a
calibrated `s1_config.json` alone cannot satisfy that original-sidecar check.
No sidecar reset is needed. A successful rerun fits only on the calibration
cases and writes the reproduced MLX temperature to the downloaded local
`s1_config.json`, so use the separate local checkout above for reproduction.

Before initializing MLX, the evaluator verifies source/export hashes, gold
labels, candidate ordering and split separation. It requires full coverage of
**256 calibration + 256 fresh test + 52 media** cases. The workflow is the
experimental S1 selected-head evaluation path; a generic MLX chat or generation
command does not reproduce this decision contract or its candidate probabilities.

## Training, conversion and calibration identity

| Item | Value |
| --- | --- |
| Original base | `google/gemma-4-12B-it` |
| Original base revision | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Source merged checkpoint SHA-256 | `fa96debc68209c193338d8edee4ec4258a9f603e4f2156d4688c16364703d15d` |
| Training | 2,048 examples, one epoch; 2,048 / 2,048 completed updates |
| LoRA recipe | Rank 8, alpha 16; q/k/v/o projections; cross-entropy; LR `2e-5`; seed 42 |
| Training source | [`b2a46b1`](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/tree/b2a46b1) |
| Inference/evaluation source | [`d4ea1ef`](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/tree/d4ea1ef79af2e22f666374e529fa56c1de9d3f53) |
| Training recipe identity | `2a708e5f4acfea4b688e88ba338aa62991608db875fef039ee6b4f33c3a74d73` |
| Conversion manifest SHA-256 | `e42b05bc2e0d26fe5ce869cd58f0ba025677278e82a2437653c8995e46e6156f` |
| Original merged sidecar SHA-256 | `322ffa1c6bfa7c4069da43265be1e79071eee6fbcff07cd4a74f9ed04774b962` |
| Independently fitted MLX temperature | `2.7144176165949068` |
| Calibrated MLX sidecar SHA-256 | `7aa9d1502a72c97504c9bbb8ab3ee4f8f800d73f02203d98913dd381e2578a39` |
| MLX / MLX-Metal versions | `0.32.3` / `0.32.3` |
| MLX environment | MLX-VLM 0.7.4; Transformers 5.18.0; huggingface-hub 1.33.0; NumPy 2.5.3 |
| Conversion host | Apple M1 Max; 64 GiB unified memory; macOS 27.0.1 arm64 |

The MLX temperature is fitted to this quantized artifact's raw selected logits,
after the native softcap and before temperature scaling. The deterministic
NumPy fitter searches 241 log-spaced temperatures in `[0.05, 20]`, plus `1`.
The BF16 checkpoint's temperature is not reused as the fitted MLX value. On the
256 calibration cases, NLL changed from 0.519925 to 0.284724; this is an
in-sample fitting result.

## Evaluation

The dataset manifest SHA-256 is
`e0ab28e3435a910c19abf3a9a7e58df8b5cff7a806b6c7f4f18021db1d3804dd`.
Fresh calibration/test passages exclude every official BoolQ training passage
and all previously sampled local passages. IDs, passage groups and canonical
requests are checked for overlap with training and historical sets.

| Dataset | Scored coverage | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh BoolQ test | 256 / 256 | 89.843750% | 0.270476 | 0.156405 | 0.045502 |
| Historical native media | 52 / 52 | 65.384615% | 1.081998 | 0.425769 | 0.130970 |

Calibration, fresh test and media evaluation completed with full coverage:
256 + 256 + 52 cases. The model answered 230 / 256 fresh BoolQ cases correctly.
All 256 observed fresh-test predictions matched the trained BF16 checkpoint.
This agreement on the measured cases does not establish global BF16 parity.
The [cross-runtime comparison](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/434909da45fbfbcdba7d55cc9b431a6236134335/docs/validation/2026-10-02/release/cross-runtime-parity.json)
records a maximum candidate-probability difference of **27.48 percentage points**
on those fresh cases. Media predictions agree on 49 / 52 cases: three FSDD
predictions changed while aggregate accuracy stayed the same, and the maximum
candidate-probability difference is **31.29 percentage points**. Equal aggregate
accuracy does not establish confidence parity. Each runtime uses its own fitted
temperature, so this compares calibrated outputs and does not isolate the effect
of quantization on raw logits.

Image accuracy over 32 MNIST cases: 87.5% (28 / 32). Audio accuracy over
20 FSDD cases: 30.0% (6 / 20). These media cases have informed earlier
experiments; they are regression evidence. No media labels are used in training
or calibration. **The BoolQ-fitted temperature does not establish media calibration.**

Report: [MLX release evaluation](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/434909da45fbfbcdba7d55cc9b431a6236134335/docs/validation/2026-10-02/release/mlx-release-evaluation.json)  
Report SHA-256: `e783bc1bf4b9361ff5fd6d9397362c0515b8f6ec2455f2688385d0727cc21f00`  
Recorded evaluation hardware: Apple M1 Max, 64 GiB unified memory; macOS 27.0.1 arm64  
Fresh BoolQ prepared-forward latency p50 / p95: 1506.87 / 2854.65 ms

Latency covers synchronized prepared tensors through the backbone, selected
projection and CPU raw logits. It excludes model loading, file I/O,
Transformers preprocessing, NumPy softmax and calibration; it is not end-to-end
request or service latency. Each split warmed up its first case once, then
recorded one measured forward per case. Compilation for other shapes may still
affect these timings.
NLL uses a `1e-12` probability floor, Brier sums squared errors over legal
candidates, and ECE uses 15 equal-width confidence bins.

This is a descriptive, single-seed result on small public datasets. Complete
execution does not certify BF16 parity, production readiness, or calibration
outside the measured BoolQ task. Foundation-model pretraining overlap is unknown.

## Attribution and licenses

Google supplies the original model. The retained BF16 source package includes
the preserved **Apache-2.0** `LICENSE` and a `NOTICE` identifying the source and
LoRA/merge modification. This conversion additionally applies MLX affine
quantization.

BoolQ is by Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski,
Michael Collins, and Kristina Toutanova (2019), Google Research. The
[pinned source](https://huggingface.co/datasets/google/boolq/tree/35b264d03638db9f4ce671b711558bf7ff0f80d5)
and transformed BoolQ dataset files are **CC-BY-SA-3.0**. The frozen archive
preserves attribution, transformations, source hashes and license links. The
media regression data retains the separate **MNIST MIT** and **FSDD CC-BY-SA-4.0**
licenses; no media data is used to fine-tune this model.

The S1 implementation builds on
[system-one-open](https://github.com/mithalouni/system-one-open). See the
[pinned selected-head implementation](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/d4ea1ef79af2e22f666374e529fa56c1de9d3f53/scripts/verify_mlx_export.py)
and [release protocol](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/434909da45fbfbcdba7d55cc9b431a6236134335/docs/release.md).
