# GGUF, MLX and Hugging Face release feasibility

Source inspection on **2026-10-02** confirms that Gemma 4 Unified weights have
conversion paths to both GGUF and MLX. Shipping the weights is separate from
shipping this project's decision behavior: a normal chat runner does not provide
the multiple unfilled answer slots, candidate-only probabilities or saved
temperature used by `UnifiedDecisionModel`.

## What is available

| Artifact | Current evidence | Remaining work |
| --- | --- | --- |
| Hugging Face Transformers checkpoint | `UnifiedDecisionModel.save()` merges LoRA, saves safetensors and processor files, and writes `s1_config.json`. | Obtain the chosen trained weights, reload them and validate the release. |
| GGUF | llama.cpp registers `Gemma4UnifiedForConditionalGeneration` for both text and vision/audio conversion. | Convert the language model and its separate `mmproj`, then implement and validate an S1 runtime adapter. |
| Full multimodal MLX | Local 4-bit and 8-bit exports loaded and ran all eight S1 workloads, including image/audio. | Prefer the 8-bit candidate; validate a production S1 adapter, quality and calibration before release. |
| Text-only MLX | MLX-LM maps `gemma4_unified` to its Gemma 4 text wrapper and deliberately drops vision/audio tensors. | This is only suitable for an explicitly text-only distribution. |

The local training artifacts contain adapter configuration, provenance and
calibration metadata under
`artifacts/live-validation/20261001-v2-live-04/training-full/checks/training-curve/`.
They do **not** contain the corresponding adapter safetensors or merged model
weights. Those metadata files alone cannot reconstruct a trained checkpoint.
Recover the selected experiment's actual weights before publishing a fine-tuned
S1 model. A conversion of Google's original weights must be described as a base
model conversion, without claiming S1 training or calibrated S1 accuracy.

`apps/modal/hf_upload.py` handles the historical E2B/270m runs only. Its training
claims, usage example and historical results are not a Unified 12B release card.

## Revisions checked

The source table establishes converter support. The live MLX experiments below
provide separate conversion/inference evidence; GGUF has not been executed in
this session. Keep the runtime revision in each conversion manifest.

| Source | Pinned revision | Evidence |
| --- | --- | --- |
| Google base checkpoint | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` | [`config.json`](https://huggingface.co/google/gemma-4-12B-it/blob/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7/config.json): `gemma4_unified`, `Gemma4UnifiedForConditionalGeneration`, tied embeddings, final logit softcap 30. |
| llama.cpp | `5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd` | [`conversion/gemma.py`](https://github.com/ggml-org/llama.cpp/blob/5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd/conversion/gemma.py#L912): text registration; the same file registers `Gemma4UnifiedVisionAudioModel`, `GEMMA4UV` and `GEMMA4UA`. |
| MLX-VLM | `66e68ce36816cb5134acba3b0d72c6271f35ba24` | [`gemma4_unified.py`](https://github.com/Blaizzy/mlx-vlm/blob/66e68ce36816cb5134acba3b0d72c6271f35ba24/mlx_vlm/models/gemma4_unified/gemma4_unified.py): multimodal modules and weight sanitization; [`convert.py`](https://github.com/Blaizzy/mlx-vlm/blob/66e68ce36816cb5134acba3b0d72c6271f35ba24/mlx_vlm/convert.py): converter arguments. |
| MLX-LM | `53b9af378278d6dd5dacac447edeb0f523b16253` | [`utils.py`](https://github.com/ml-explore/mlx-lm/blob/53b9af378278d6dd5dacac447edeb0f523b16253/mlx_lm/utils.py#L54) and [`gemma4.py`](https://github.com/ml-explore/mlx-lm/blob/53b9af378278d6dd5dacac447edeb0f523b16253/mlx_lm/models/gemma4.py): explicit text-only remapping and modality tensor removal. |

PyPI reported MLX-VLM **0.7.4**, MLX-LM **0.32.0** and MLX **0.32.3** on the same
date. The isolated `artifacts/mlx-env` was installed with MLX-VLM 0.7.4,
MLX 0.32.3, Transformers 5.18.0 and Hugging Face Hub 1.33.0. Its `convert.py`,
`gemma4_unified.py` and Unified `config.py` are byte-for-byte identical to the
pinned MLX-VLM source above. This verifies the installed conversion path without
claiming that conversion or GPU inference has run.

## Prepare the source checkpoint

For a trained release, load the exact base revision and its matching trained
adapter, retain the matching `s1_config.json`, and use the existing
`UnifiedDecisionModel.save()` path to produce a **merged** checkpoint. Do not give
an adapter-only directory to a full-model converter. Confirm that reloading the
merged checkpoint reproduces the unmerged adapter's candidate logits first.

The following examples use `artifacts/checkpoint` as that complete local source.
For an original Google model conversion, replace it with the local snapshot of
the pinned base revision and use names containing `base`, not `finetuned`.

## GGUF conversion

Use an isolated tool environment; do not change the project's inference lockfile.
The commands below are checked against the pinned upstream converter's argument
definitions. They are a reproducible recipe, not a recorded successful run.

```bash
mkdir -p artifacts/export-tools artifacts/exports/s1-gguf
git clone https://github.com/ggml-org/llama.cpp artifacts/export-tools/llama.cpp
git -C artifacts/export-tools/llama.cpp checkout 5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd
uv venv --python 3.12 artifacts/gguf-env
uv pip install --python artifacts/gguf-env/bin/python \
  -r artifacts/export-tools/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

artifacts/gguf-env/bin/python artifacts/export-tools/llama.cpp/convert_hf_to_gguf.py \
  artifacts/checkpoint --outtype bf16 \
  --outfile artifacts/exports/s1-gguf/s1-bf16.gguf
artifacts/gguf-env/bin/python artifacts/export-tools/llama.cpp/convert_hf_to_gguf.py \
  artifacts/checkpoint --mmproj --outtype f16 \
  --outfile artifacts/exports/s1-gguf/mmproj-s1-f16.gguf

cmake -S artifacts/export-tools/llama.cpp -B artifacts/export-tools/llama.cpp/build \
  -DGGML_METAL=ON -DLLAMA_CURL=OFF
cmake --build artifacts/export-tools/llama.cpp/build --config Release \
  --target llama-quantize -j 4
artifacts/export-tools/llama.cpp/build/bin/llama-quantize \
  artifacts/exports/s1-gguf/s1-bf16.gguf \
  artifacts/exports/s1-gguf/s1-Q4_K_M.gguf Q4_K_M
cp artifacts/checkpoint/s1_config.json artifacts/exports/s1-gguf/
```

Distribute the `mmproj` alongside the language-model GGUF for image/audio use.
Retain the processor/tokenizer files used for parity testing and document their
source revision. A text GGUF by itself does not contain the full multimodal
preprocessing and projector path. Budget disk space for source weights, the
intermediate BF16 GGUF and the quantized output before starting.

The llama.cpp C API supports requesting outputs at several positions with
`llama_batch.logits` and reading them with `llama_get_logits_ith()`.
This enables an adapter to select the candidate logits at each S1 answer slot.
The standard output is still an `n_vocab` row: this API does not by itself
reproduce the selected-head compute saving in `src/s1/unified.py`.
An optimized candidate-only projection needs a custom graph/runtime path.
See the pinned [`llama.h`](https://github.com/ggml-org/llama.cpp/blob/5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd/include/llama.h).

## Full multimodal MLX conversion

Use **MLX-VLM** for the complete model. `mlx_lm.convert` removes the vision/audio
weights even though `gemma4_unified` may remain in the configuration.

```bash
uv venv --python 3.12 artifacts/mlx-env
uv pip install --python artifacts/mlx-env/bin/python \
  mlx-vlm==0.7.4 mlx==0.32.3 transformers==5.18.0 huggingface-hub==1.33.0
artifacts/mlx-env/bin/python -m mlx_vlm convert \
  --hf-path artifacts/checkpoint \
  --mlx-path artifacts/exports/s1-mlx-4bit \
  -q --q-bits 4 --q-group-size 64
cp artifacts/checkpoint/s1_config.json artifacts/exports/s1-mlx-4bit/
```

The pinned converter also copies JSON sidecars, but explicitly check
`s1_config.json` after conversion and after download from the Hub. Check the tensor
inventory for `vision_embedder`, `embed_vision` and `embed_audio`; a successful
text-generation call cannot establish that these components survived conversion.

An MLX S1 adapter can use the multimodal embedding preparation and language
backbone, gather hidden states at the recorded answer slots and project only the
selected output rows. Quantized embedding rows need the matching scales and
biases, or a correctly dequantized copy of those rows. Calling the normal
generation API does not implement this contract.

### Live conversion and S1 comparison

On 2026-10-02, the pinned Google base was converted locally with MLX-VLM 0.7.4,
MLX 0.32.3 and affine quantization, group size 64. The 4-bit weights occupy
**6,800,390,133 bytes**; the 8-bit weights occupy **12,754,909,484 bytes**.
Strict reload verified that `vision_embedder`, `embed_vision` and `embed_audio`
weights remain present. Tokenizer candidate IDs were checked at the actual
answer boundary. Output folders are under `artifacts/exports/`, named
`base-gemma4-12b-mlx-4bit` and `base-gemma4-12b-mlx-8bit`.

The converter initially failed when it copied a read-only Hugging Face cache
tokenizer and then tried to rewrite it. A retry succeeded using a staging folder:
metadata was copied with `shutil.copyfile` (writable destination files), while
the large source safetensors were symlinked. The immutable Hub cache was not
modified. If converting directly from a cache snapshot, use that staging step;
a normal writable merged checkpoint does not need it. Treat a converter failure
as an incomplete artifact even if it already wrote weight shards: quantization
configuration is saved near the end.

The experimental verifier uses the exact tensors, candidate IDs and answer slots
captured from this project's HF processor path. It invokes the normalized MLX
backbone and projects selected tied embedding rows with native softcap and the
same temperature. It preserves explicit modality token types and lets the
backbone build separate full/sliding/vision masks. The stock outer wrapper in
this MLX-VLM version does not forward modality token types; passing a 2-D padding
mask directly to the backbone would also bypass native masks. Neither path is
used in the verifier. It only supports unpadded batch size one and tied output
embeddings; it is **not a production SDK backend**.

```bash
uv run --no-sync python scripts/prepare_mlx_validation.py \
  --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
  --dataset examples/benchmarks/mps.jsonl --output artifacts/mps/mlx-inputs
artifacts/mlx-env/bin/python scripts/verify_mlx_export.py \
  --inputs artifacts/mps/mlx-inputs \
  --model artifacts/exports/base-gemma4-12b-mlx-8bit \
  --reference artifacts/mps/bf16-sdpa.json \
  --output artifacts/mps/mlx-8bit-validation.json
```

Use the MPS profiling command in [mps.md](mps.md) to create the reference first.
The verifier refuses mismatched model/revision, dataset, temperature and labels.
It measures prepared-tensor inference only; HF preprocessing, I/O and model
loading are excluded, so these times are not end-to-end request latency.

| MLX path | Completed workloads | Argmax agreement with PyTorch BF16 | Maximum probability difference |
| --- | ---: | ---: | ---: |
| Original BF16 weights, no quantization | 8/8 | 29/29 | 0.03410 |
| 4-bit affine export | 8/8 | 25/29 | 0.88326 |
| 8-bit affine export | 8/8 | 29/29 | 0.09512 |

The 8-bit export is the better candidate for further work. It cuts weight storage
by about 47% relative to the source, preserving all 29 observed argmax decisions,
but confidence changes by up to 9.51 percentage points. This is **not** evidence
of calibrated confidence or general accuracy. The 4-bit candidate should not be
released as an equivalent S1 model. The unquantized control shows that the much
larger 4-bit drift is not intrinsic to the multi-slot adapter alone; runtime
arithmetic and quantization both contribute to differences. No universal speedup
is claimed for MLX: timings vary by workload and exclude preprocessing.

All results use eight synthetic performance cases. Model cards and provenance
are prepared locally; neither artifact has been uploaded to Hugging Face.
Raw results are in [`validation/2026-10-02`](validation/2026-10-02/).

## Preserve the decision contract

The authoritative implementation is [`src/s1/unified.py`](../src/s1/unified.py),
with calibration loading in [`src/s1/checkpoint.py`](../src/s1/checkpoint.py).
For each new runtime and quantization, compare against that implementation:

1. Use identical chat-template output with thinking disabled, tokenizer IDs,
   media preprocessing, expanded modality tokens, attention masks and question
   order. Measure answer positions after media expansion. Later slots can see
   earlier question text; do not split the questions or generate previous answers
   when testing parity with the one-forward implementation.
2. Validate that `A-Z/a-z` are distinct single tokens at the actual `Answer: (`
   boundary. Preserve runtime label order, option count and typed answer mapping.
3. Preserve the actual LM head and any bias. The pinned base ties the head to
   input embeddings. Upstream conversion paths can discard `lm_head.weight` in
   favor of those tied embeddings; do not assume that is valid for an independently
   trained or untied output head. Recheck tying after adapter merge.
4. Apply the native final-logit softcap once, in the projection dtype, then cast
   to float32 before temperature scaling and candidate softmax. If a runtime
   returns already-softcapped logits, do not softcap them again. Quantized kernels
   can introduce numerical differences; compare logits and distributions rather
   than requiring bitwise equality.
5. Preserve positive finite temperature, source model/revision and prompt version
   in `s1_config.json`. Load it from the same immutable revision as the weights.
   Probabilities must sum to one over the supplied legal candidates; they are not
   full-vocabulary probabilities. Do not copy a trained adapter's calibration onto
   the untrained base or assume calibration survives quantization unchanged.
6. Evaluate text, image, audio, mixed media, 1/8/32 slots, varied option counts and
   question-order sensitivity. Record agreement, probability drift, held-out
   accuracy, NLL/Brier/ECE, latency and peak memory. Refit temperature only on the
   calibration split if needed, then evaluate once on untouched test data.

Do not label a converted artifact a drop-in S1 implementation until its adapter
passes these checks. Generation tokens/second and S1 request/decision latency
measure different workloads; report the one actually tested.

## Hugging Face publication

The pinned Google model card declares **Apache-2.0** and the Hub API reported
`gated: false` on 2026-10-02. Gemma 4 12B is also described as Apache-2.0 in
[Google's announcement](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12B/).
The model card and license terms permit the distribution path; preserve applicable
license/attribution notices and identify modifications. Review the training-data
licenses separately for a trained derivative. The repository's MIT code license
does not replace the model or dataset licenses.

A release should include model weights, required processor/projector artifacts,
`s1_config.json`, the runtime adapter or its pinned installation instructions,
conversion provenance and an accurate model card. Record the source weight
revision/hash, trained adapter identity if any, conversion tool revision,
quantization settings and measured validation results. For GGUF and MLX, make the
required runtime explicit instead of implying that generic Hub inference widgets
implement the S1 decision API.

This recipe does not create a Hub repository or upload anything. Once a particular
artifact and destination are selected for publication, a normal Hugging Face model
repository can hold either format. Uploading alone is not evidence of runtime
compatibility or calibrated model quality.
