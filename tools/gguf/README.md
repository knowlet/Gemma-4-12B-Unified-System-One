# Native GGUF S1 adapter

This adapter runs the language model and the image/audio projections with
llama.cpp and libmtmd. It scores every answer slot in one causal input sequence;
it never samples or appends answers. The native graph projects the full
vocabulary before selecting the legal candidate rows. `passes: 1` describes one
decision evaluation; `preprocessing.decode_calls` reports physical decode calls
used for text/media chunks. The KV cache is cleared between requests.

Build against the exact checkout recorded in `CMakeLists.txt`, after building its
shared `llama`, `mtmd`, `ggml`, and `ggml-base` libraries:

```sh
cmake -S tools/gguf -B artifacts/gguf-runtime \
  -DLLAMA_CPP_DIR="$PWD/artifacts/export-tools/llama.cpp"
cmake --build artifacts/gguf-runtime -j 4
```

Install this repository's locked inference dependencies. The Python adapter uses
the saved HF processor on CPU without loading HF model weights. The GGUF package
must contain a text GGUF, `mmproj*.gguf`, tokenizer, chat template, processor and
model configuration, plus its `conversion-manifest.json`.

After the release evaluator writes the GGUF model's own calibrated
`s1_config.json`, send one public `DecisionRequest` JSON object per line:

```sh
.venv/bin/python scripts/gguf_adapter.py \
  --model artifacts/exports/s1-boolq-gguf \
  --native-runner artifacts/gguf-runtime/bin/s1-gguf < requests.jsonl
```

The default mode returns typed Choice/Noul/Score answers. Calibration must name
`runtime: "llama.cpp"`, `runtime_format: "gguf"`, and the matching
`conversion_manifest_sha256`. A copied BF16 or MLX temperature is rejected.
`--evaluation` instead accepts the evaluator's gold-free capture envelopes and
returns legal `raw_logits` after the model's native softcap, before temperature.

HF text/control tokens, media order, and slot positions are preserved. For
images, the adapter reconstructs the exact processed RGB grid from HF patches
and validates the uint8 round-trip before native mmproj inference. Audio uses
the request's normalized 16 kHz float32 samples and native 640-sample framing.
Complete image blocks fit within both the batch and microbatch, retaining the
Gemma 4 sliding-layer bidirectional vision mask. Context overflow, unsupported
layouts, missing modalities, and token-count mismatches return explicit errors.

The default requests Metal/GPU offload with a 16,384-token context and an
8,192-token batch/microbatch. `--gpu-layers 0` selects CPU. GPU availability and
actual offloading are visible in native stderr logs. The release evaluation,
not structural conversion or these implementation checks, establishes quality.
