---
license: apache-2.0
language:
  - en
base_model: google/gemma-4-12B-it
base_model_relation: finetune
library_name: transformers
datasets:
  - google/boolq
tags:
  - gemma4
  - system-one
  - boolq
  - lora
  - multimodal
  - calibrated-classification
---

# Gemma 4 12B Unified System One

A **BoolQ specialist** using the System One decision interface: one multimodal
backbone forward produces probabilities over declared answer candidates. The
BF16 artifact contains the complete model with its LoRA adaptation
merged, plus the source processor and native image/audio modules.

The training recipe is one epoch of LoRA updates on 2,048 English BoolQ examples.
It does **not** update every base parameter. The checkpoint is a full
model rather than an adapter that requires a separate base download.

**Release date:** 2026-10-04 (Asia/Taipei)  
**Public BF16 repository:** [knowlet/Gemma-4-12B-Unified-System-One](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One)  
**Companion MLX 8-bit repository:** [knowlet/Gemma-4-12B-Unified-System-One-MLX-8bit](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-MLX-8bit)  
**Immutable HF weights revision:** `66626de5cbcf8c5fecb8e9b58710805a02a42f67`  
**Checkpoint identity SHA-256:** `fa96debc68209c193338d8edee4ec4258a9f603e4f2156d4688c16364703d15d`  
**Weight files:** 1 file, 23919549408 bytes  
**`model.safetensors` SHA-256:** `11144621122123135eb79c9cfd0466a2c51f230109124aefe401cdbec8c71a53`

## Intended use and limits

The measured task is passage-grounded, English yes/no classification using the
`Noul` primitive. The implementation also exposes Choice and Score schemas, but
this training run does not establish their quality, multilingual quality,
multi-question independence, or general assistant performance.

A scalar temperature is fitted on the 256-case **BoolQ calibration split only**.
That temperature does not establish calibration for images, audio, other tasks,
or deployment traffic. Native image/audio weights remain present; their results
are reported separately below. This is a small, single-seed public-data study,
not a production-readiness claim. Foundation-model pretraining overlap with
BoolQ is unknown.

## Use the decision interface

Clone the source repository and install the pinned S1 runtime. The model weights
are downloaded separately from Hugging Face in the Python example.

```bash
git clone https://github.com/knowlet/Gemma-4-12B-Unified-System-One.git
cd Gemma-4-12B-Unified-System-One
git checkout d4ea1ef79af2e22f666374e529fa56c1de9d3f53
uv sync --locked --extra inference
```

Run the following with `uv run --no-sync python` from that repository root.
`snapshot_download` pins the weights, processor and `s1_config.json` to the same
immutable Hub commit, then `UnifiedDecisionModel` loads that local snapshot.

```python
from huggingface_hub import snapshot_download

from s1.contracts import DecisionRequest
from s1.unified import UnifiedDecisionModel

model_directory = snapshot_download(
    repo_id="knowlet/Gemma-4-12B-Unified-System-One",
    revision="66626de5cbcf8c5fecb8e9b58710805a02a42f67",
)
model = UnifiedDecisionModel(
    model_directory,
    device="auto",
)
request = DecisionRequest.model_validate({
    "state": {
        "passage": "Paris is the capital of France.",
        "question": "Is Paris the capital of France?",
    },
    "questions": {
        "answer": {
            "type": "noul",
            "instructions": "Answer the question using the supplied passage.",
        }
    },
})
result = model.predict(request)
print(result["answers"]["answer"])
```

`device="auto"` selects CUDA, then Apple MPS, then CPU. The runtime chooses its
default weight dtype for that device and OS. The saved `s1_config.json`
temperature is loaded automatically. Calling a generic chat or text-generation
runner does not reproduce this prompt construction, selected-head readout, or
calibrated candidate probabilities.

## Training and artifact provenance

| Item | Value |
| --- | --- |
| Base checkpoint | `google/gemma-4-12B-it` |
| Base revision | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Updates | One pass over 2,048 examples; 2,048 / 2,048 completed updates |
| Adaptation | LoRA rank 8, alpha 16; `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Objective / learning rate | Cross-entropy / `2e-5` |
| Training seed | 42 |
| Weight format | Complete merged multimodal BF16 safetensors |
| Training source revision | [`b2a46b1`](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/tree/b2a46b1) |
| Runtime source revision | [`d4ea1ef`](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/tree/d4ea1ef79af2e22f666374e529fa56c1de9d3f53) |
| Training recipe identity | `2a708e5f4acfea4b688e88ba338aa62991608db875fef039ee6b4f33c3a74d73` |
| Training hardware | NVIDIA A100-SXM4-80GB |
| Training environment | PyTorch 2.8.0; Transformers 5.17.0; PEFT 0.21.0 |
| Training / complete stage time | 1,349.35 / 1,541.93 seconds |
| Peak allocated CUDA memory during training | 24,698,445,824 bytes |
| Calibration temperature | `2.7830344470383452` |
| Calibration NLL, before / after temperature fitting | 0.541142 / 0.291505 on 256 cases |
| `s1_config.json` SHA-256 | `322ffa1c6bfa7c4069da43265be1e79071eee6fbcff07cd4a74f9ed04774b962` |

The adapter is preserved before merging. Calibration is performed after the
merged checkpoint is reloaded. The training and evaluation protocol, archived
data, exclusions, and artifact receipts are described in the
[release documentation](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/434909da45fbfbcdba7d55cc9b431a6236134335/docs/release.md).

## Evaluation

Calibration and fresh test passages are disjoint from the selected training
examples, all official BoolQ training passages, and previously sampled local
cases. The dataset manifest SHA-256 is
`e0ab28e3435a910c19abf3a9a7e58df8b5cff7a806b6c7f4f18021db1d3804dd`.

| Dataset | Scored coverage | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh BoolQ test, pinned base | 256 / 256 | 80.078125% | 0.440918 | 0.287375 | 0.067065 |
| Fresh BoolQ test, trained | 256 / 256 | 89.843750% | 0.272659 | 0.158027 | 0.052127 |
| Historical native media, trained | 52 / 52 | 65.384615% | 1.072631 | 0.423565 | 0.218368 |

The 52 media cases contain 32 MNIST images and 20 FSDD recordings. Image accuracy:
87.5% (28 / 32); audio accuracy: 30.0% (6 / 20). These cases have
informed previous experiments and are regression evidence, not fresh held-out
evidence. No media labels are used for training or temperature fitting. A
separate historical 128-case BoolQ regression result is
92.1875% (118 / 128), with all 128 cases scored. The pinned base has the same
media accuracy, but lower media NLL (1.049982), Brier (0.416844), and ECE
(0.157409); text-only fine-tuning and calibration did not improve these media
probability metrics.

Metrics and hardware/latency scope come from the saved evaluation report:
[evaluation receipt](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/434909da45fbfbcdba7d55cc9b431a6236134335/docs/validation/2026-10-02/release/evaluate-receipt.json) (evaluation receipt SHA-256
`07da695983a341ed1b488dedb7edd610b426fc86681a9fb45fbf9fe29f6fc280`).
Evaluation ran on an NVIDIA A100-SXM4-80GB. The trained fresh-test request
latency was 88.96 ms p50 / 92.60 ms p95; the 52-case media latency was 111.59 ms
p50 / 113.65 ms p95. These are the saved single-request evaluation measurements,
not throughput or Apple MPS measurements.
NLL uses a probability floor of `1e-12`; Brier is the sum of squared errors over
the legal candidates; ECE uses 15 equal-width confidence bins. No result from
the earlier base-model MPS profiling workload is substituted for this evaluation.

## Attribution and licenses

The source model is by Google. Its **Apache-2.0** license and applicable notices
are retained in `LICENSE` and `NOTICE`; this derivative identifies the LoRA
fine-tuning and merged checkpoint changes.

BoolQ is by Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski,
Michael Collins, and Kristina Toutanova (2019), Google Research. The
[pinned BoolQ source](https://huggingface.co/datasets/google/boolq/tree/35b264d03638db9f4ce671b711558bf7ff0f80d5)
and transformed BoolQ data are **CC-BY-SA-3.0**. The frozen data archive preserves
the attribution, source hashes, transformations, and license links. The media
regression data retains the separate **MNIST MIT** and **FSDD CC-BY-SA-4.0**
licenses. These dataset licenses are distinct from the model's source license.

The decision implementation builds on
[system-one-open](https://github.com/mithalouni/system-one-open); the pinned
[System One implementation](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/d4ea1ef79af2e22f666374e529fa56c1de9d3f53/src/s1/unified.py)
defines the inference contract used here.
