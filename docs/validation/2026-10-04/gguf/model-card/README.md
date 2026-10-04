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
  - system-one
  - boolq
  - gguf
  - llama-cpp
  - q8_0
  - multimodal
---

# Gemma 4 12B Unified System One — GGUF Q8_0

A complete GGUF conversion of the trained, merged
[BF16 checkpoint](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One/tree/66626de5cbcf8c5fecb8e9b58710805a02a42f67),
with a native S1 decision runtime and independently fitted GGUF temperature.
The language model uses **Q8_0** storage; a separate **F16 `mmproj`** contains
the native image and audio projections. Download both files and their saved
processor, tokenizer, chat template, conversion receipt and calibration sidecar.

This is a **BoolQ specialist**: one epoch of rank-8 LoRA on 2,048 English
examples, merged into the complete multimodal model before conversion. It is
not a full-parameter training run. Choice, Noul and Score are supported interface
schemas; this training establishes evidence only for passage-grounded English
yes/no decisions.

**Repository:** [knowlet/Gemma-4-12B-Unified-System-One-GGUF](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-GGUF)  
**Immutable GGUF core-files revision:** `5940bdbe292b33ac86450093a528d39b13c3a8f4`  
**S1 adapter/evaluator source:** [`bcf829acc111fd8054987b1519b8181835f1dfed`](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/tree/bcf829acc111fd8054987b1519b8181835f1dfed)  
**llama.cpp source:** [`5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd`](https://github.com/ggml-org/llama.cpp/tree/5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd)

The core-files revision pins both weights, inference assets and calibration.
The final `README.md` and publication `artifact-manifest.json` are added
separately and are outside that revision.

## Files and provenance

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `s1-boolq-Q8_0.gguf` | 12,669,647,264 | `9d3bc61851e5b7ed8e595b84327b85271830778d0e62fa9c1349c61843f369ed` |
| `mmproj-s1-boolq-f16.gguf` | 122,031,744 | `cffbff05efd290fef28d084ebcb653de7594f5e26bae0eabbedf30d6c782c1d1` |

The two GGUF files total 12,791,679,008 bytes. Q8_0 describes the language-model
quantization format; not every tensor in the package is an 8-bit integer.
[`conversion-manifest.json`](conversion-manifest.json) records source hashes,
conversion commands, tool versions and the immutable conversion-file inventory.
`base_s1_config.json` retains the BF16 sidecar unchanged; `s1_config.json`
contains the separately fitted GGUF calibration and its conversion-receipt
binding. Prediction rejects a copied BF16/MLX calibration or changed processor
assets.

| Source identity | Value |
| --- | --- |
| Original base | `google/gemma-4-12B-it` at `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Trained BF16 weights revision | `66626de5cbcf8c5fecb8e9b58710805a02a42f67` |
| Source checkpoint inventory SHA-256 | `fa96debc68209c193338d8edee4ec4258a9f603e4f2156d4688c16364703d15d` |
| Training | 2,048 English BoolQ cases; one epoch; 2,048 completed updates |
| LoRA | Rank 8, alpha 16; q/k/v/o; cross-entropy; LR `2e-5`; seed 42 |
| Training recipe SHA-256 | `2a708e5f4acfea4b688e88ba338aa62991608db875fef039ee6b4f33c3a74d73` |

## Native S1 execution

The implemented [Python adapter](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/bcf829acc111fd8054987b1519b8181835f1dfed/scripts/gguf_adapter.py)
and [native helper](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/bcf829acc111fd8054987b1519b8181835f1dfed/tools/gguf/s1_gguf.cpp)
score all unfilled answer slots in their original question order. They never
sample or insert earlier answers. `passes: 1` means one decision evaluation;
text batches and media chunks can require several native prefill/decode calls,
reported separately as `preprocessing.decode_calls`. The KV cache is cleared
between requests.

**The native graph computes the full-vocabulary head**, then selects the legal
candidate logits. The GGUF temperature is applied to those logits before
normalizing only the legal candidates. The model's native final-logit softcap
is consumed once. This runtime does not use the PyTorch/MLX selected-head compute
optimization, and a generic chat-generation command does not implement this
decision interface.

HF prepares the exact text/control token IDs, candidate IDs, media ordering and
expanded answer-slot positions. Images are reconstructed from the HF-processed
RGB patches with a checked uint8 round-trip; native `mtmd` then executes their
projection. Audio uses normalized 16 kHz float32 samples and native 640-sample
framing/projection. Image blocks remain intact within a microbatch to preserve
Gemma 4's bidirectional vision attention in sliding-window layers. The runtime
checks native media token counts and positions; it does not replay HF media
embedding tensors. Matching tokens and positions alone does not imply numerical
parity with BF16 or MLX.

**Twelve native execution checks passed on Apple M1 Max, 64 GiB**, covering text,
image, audio, mixed media, 16 questions, a 1,299-token request, and an 8,481-token
request whose answer slots cross the 8,192-token batch boundary. The first slot
in that cross-batch request exactly matched the logits from its identical
49-token prefix. The calibrated public `DecisionRequest` interface also passed
**8 requests with 29 typed answers**: 22 Choice, 6 Noul and 1 Score. These are
functional checks, separate from the accuracy and calibration measurements below.

## Download, build and run

These instructions target Apple Silicon/macOS with Git, CMake, a C++17
compiler and `uv` installed. The measured host had 64 GiB unified memory;
weight-file size alone is not the runtime memory requirement. Use a fresh
checkout and download the core inference package at one immutable Hub revision:

```bash
git clone https://github.com/knowlet/Gemma-4-12B-Unified-System-One.git
cd Gemma-4-12B-Unified-System-One
git checkout bcf829acc111fd8054987b1519b8181835f1dfed
uv sync --locked --extra inference

uv run --no-sync python - <<'PY'
from pathlib import Path
from hashlib import file_digest
from huggingface_hub import snapshot_download

package = Path(snapshot_download(
    repo_id="knowlet/Gemma-4-12B-Unified-System-One-GGUF",
    revision="5940bdbe292b33ac86450093a528d39b13c3a8f4",
    local_dir="artifacts/exports/s1-boolq-gguf",
))
expected = {
    "s1-boolq-Q8_0.gguf": "9d3bc61851e5b7ed8e595b84327b85271830778d0e62fa9c1349c61843f369ed",
    "mmproj-s1-boolq-f16.gguf": "cffbff05efd290fef28d084ebcb653de7594f5e26bae0eabbedf30d6c782c1d1",
}
for name, digest in expected.items():
    with (package / name).open("rb") as stream:
        assert file_digest(stream, "sha256").hexdigest() == digest, name
for name in ("s1_config.json", "conversion-manifest.json", "processor_config.json",
             "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json"):
    assert (package / name).is_file(), name
print(package)
PY
```

The following library-only build recipe was reproduced successfully in a
separate pristine checkout on Apple Silicon. The recorded evaluation build
also enabled llama.cpp's command-line tools.
Keep the llama.cpp tracked source pristine; the adapter build checks its
revision and rejects modifications.

```bash
mkdir -p artifacts/export-tools
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git \
  artifacts/export-tools/llama.cpp
git -C artifacts/export-tools/llama.cpp checkout 5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd
cmake -S artifacts/export-tools/llama.cpp -B artifacts/export-tools/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_METAL=ON \
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_MTMD=ON
cmake --build artifacts/export-tools/llama.cpp/build --target llama mtmd -j 4
cmake -S tools/gguf -B artifacts/gguf-runtime \
  -DLLAMA_CPP_DIR="$PWD/artifacts/export-tools/llama.cpp"
cmake --build artifacts/gguf-runtime -j 4
```

Send one [DecisionRequest](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/bcf829acc111fd8054987b1519b8181835f1dfed/src/s1/contracts.py)
JSON object per line. The default mode loads the GGUF calibration sidecar and
returns typed answers; no separate BF16 weights are loaded.

```bash
cat > requests.jsonl <<'JSON'
{"state":{"passage":"Paris is the capital of France.","question":"Is Paris the capital of France?"},"questions":{"answer":{"type":"noul","instructions":"Answer the question using the supplied passage."} } }
JSON
uv run --no-sync python scripts/gguf_adapter.py \
  --model artifacts/exports/s1-boolq-gguf \
  --native-runner artifacts/gguf-runtime/bin/s1-gguf < requests.jsonl
```

`answers.answer.noul` is the calibrated probability of `true`. Model logs go to
stderr; stdout contains one JSON response per input line. Defaults are a
16,384-token context and an 8,192-token batch/microbatch with GPU offload
requested. `--gpu-layers 0` selects CPU; CPU performance was not measured here.
The [runtime guide](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/bcf829acc111fd8054987b1519b8181835f1dfed/tools/gguf/README.md)
describes media inputs, explicit errors, and the raw-logit `--evaluation` mode.

## Calibration and evaluation

The [frozen evaluator](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/blob/bcf829acc111fd8054987b1519b8181835f1dfed/scripts/evaluate_gguf_release.py)
uses the declared **256 calibration + 256 fresh BoolQ test + 52 historical media**
populations. It fits temperature only on the disjoint BoolQ calibration split,
using native post-softcap logits. Test and media labels do not select the
temperature or conversion settings. Errors, unsupported requests and unrun
cases remain visible in full-population coverage. All 564 declared cases
completed successfully.

The independent fit searched 241 log-spaced temperatures in `[0.05, 20]`, plus
`1`, using the GGUF model's own 256 calibration-logit rows. It happened to select
the same numeric grid value as BF16; the BF16 temperature was not copied.

| Measurement | GGUF result |
| --- | --- |
| Calibration coverage | `256 / 256` |
| Independently fitted temperature | `2.7830344470383452` |
| Calibration NLL, before / after | `0.538595 / 0.290412` |
| Fresh BoolQ coverage | `256 / 256` |
| Fresh BoolQ accuracy | `231 / 256` (`90.23%`) |
| Fresh BoolQ NLL / Brier / ECE | `0.272152 / 0.156864 / 0.046608` |
| Historical media coverage | `52 / 52` |
| Historical media accuracy | `33 / 52` (`63.46%`) |
| Image / audio accuracy | `28 / 32` MNIST; `5 / 20` FSDD |
| Historical media NLL / Brier / ECE | `1.111841 / 0.436341 / 0.192256` |

GGUF per-case results, runtime/file identities and latency scope are in the
[evaluation report](evaluation.json). ECE uses 15 equal-width confidence bins.
**Report SHA-256:** `c812265d0f4b62cbf69a774c728c0c59d1912f259993ca78c38a04f51bfa5fbb`.

The separate [cross-runtime comparison](cross-runtime-comparison.json) compares
GGUF with the trained merged BF16 source and MLX 8-bit outputs. On fresh BoolQ,
GGUF/source-BF16 argmax agreement is **255/256 (99.61%)**; the maximum absolute
candidate-probability difference is **32.89 percentage points**. Historical
media argmax agreement is **47/52**. BF16 here means the trained source
checkpoint, not the original Google base. Agreement compares model outputs;
the accuracy figures above use dataset labels.

GGUF answered one additional fresh BoolQ case correctly compared with the
trained BF16 source (231 versus 230), while media accuracy was 33/52 versus
34/52. These point estimates do not establish overall model superiority or
probability parity.

Calibration NLL is an in-sample fitting result. The 52 media cases have informed
earlier experiments and are a historical regression set, not fresh held-out
evidence. BoolQ-only temperature fitting does not establish image/audio
calibration. Broad multimodal, multilingual, Choice/Score task quality and
production suitability remain unestablished; public-data pretraining overlap
is unknown.

## Attribution and licenses

Google supplies the original model. Preserve the package's Apache-2.0
`LICENSE` and `NOTICE`, including the BoolQ LoRA/merge and GGUF conversion
modifications. The S1 repository's MIT code license does not replace model or
dataset licenses.

BoolQ is by Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski,
Michael Collins and Kristina Toutanova (2019), Google Research. The
[pinned BoolQ source](https://huggingface.co/datasets/google/boolq/tree/35b264d03638db9f4ce671b711558bf7ff0f80d5)
and transformed files are CC-BY-SA-3.0. The frozen archive retains attribution,
transformations and source hashes. Media regression examples retain MNIST MIT
and FSDD CC-BY-SA-4.0 licenses; they were not used for this LoRA training or
BoolQ calibration. The S1 implementation builds on
[system-one-open](https://github.com/mithalouni/system-one-open).
