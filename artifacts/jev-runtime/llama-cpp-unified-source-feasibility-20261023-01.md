# Mature engine candidate: source evidence, not runtime qualification

The continuing goal is real browser success under the pinned Jev/task/judge/context/budget contract, plus matched comparison of all six Decision3 models. A custom streaming loader is not the only route worth investigating. Source review found direct Gemma4 Unified support in upstream llama.cpp; this changes the next implementation choice after the bounded GPU context measurement.

Independent reviewer `/root/q95_owner05_review` read only three upstream source files, small repository metadata and the license, with no build, conversion, model/payload access, Torch import or runtime probing. Pinned upstream commit is [`184f6536024668e8908ab8c86033d56708e62b40`](https://github.com/ggml-org/llama.cpp/commit/184f6536024668e8908ab8c86033d56708e62b40), timestamp2026-10-22T22:38:03Z, with [MIT license](https://github.com/ggml-org/llama.cpp/blob/184f6536024668e8908ab8c86033d56708e62b40/LICENSE).

The [converter](https://github.com/ggml-org/llama.cpp/blob/184f6536024668e8908ab8c86033d56708e62b40/conversion/gemma.py#L603) registers `Gemma4UnifiedForConditionalGeneration` directly. Its per-layer configuration normalizes serialized indices such as05 and can represent the observed original sliding/full pattern, sliding head dimension256/KV heads8, full512/KV heads1, zero KV sharing and zero per-layer embedding. The [runtime graph](https://github.com/ggml-org/llama.cpp/blob/184f6536024668e8908ab8c86033d56708e62b40/src/models/gemma4.cpp#L210) has native K=V when no V projection exists, normalization, RoPE and layer-scalar multiplication. The [tensor-name mapping](https://github.com/ggml-org/llama.cpp/blob/184f6536024668e8908ab8c86033d56708e62b40/gguf-py/gguf/tensor_mapping.py#L777) includes `layer_scalar`.

These inspected paths establish a plausible architecture route, not acceptance of the actual checkpoint. Specific unresolved items:

- The48-layer case receives anUNKNOWN type label in the reviewed classifier. The inspected generic loop does not reject it, but an actual conversion/load is still required.
- Full-attention proportional RoPE uses a very large factor to approximate unrotated dimensions. Do not claim bitwise equivalence to the native framework.
- All677 checkpoint keys, `model.language_model` prefixes and required/ignored tensors must be accounted for before conversion is accepted.
- Tokenizer, chat template, suppression, tool/choice format and any selected LoRA must be traced to the actual source-selected browser configuration.
- Quantization, CPU mmap/hybrid placement,8192 context, transfer/workspace/host memory and20-second task behavior remain unmeasured. No speed or score is predicted.

Next prioritize this pinned mature-engine path's exact conversion and serving suitability over rebuilding a full inference engine. CPU mmap can be evaluated as an independent backend, and GPU/hybrid use still requires actual context and resource evidence. Any new backend is a new profile: it cannot inherit N3/Q95 identity/quality qualification or the earlier two JevBench scores. It must return to the unchanged real-browser contract and fresh separate JevBench evaluations when runnable. Raw model/converted tensor files remain local and out of Git.
