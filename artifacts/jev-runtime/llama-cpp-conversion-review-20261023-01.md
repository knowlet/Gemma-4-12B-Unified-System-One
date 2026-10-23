# Pinned llama.cpp conversion memory review

2026-10-23; reviewer `/root/q95_owner05_review`; source-only, STOPPED after this note. No build, dependency install, Torch/NumPy import, converter execution, checkpoint/header/payload read, mmap, GPU or model probe was performed. The only new file written is this note. Repository HEAD was reported as99a9a93; initial worktree check was clean. This extends the existing Unified feasibility note, not model qualification.

Upstream is [ggml-org/llama.cpp184f6536024668e8908ab8c86033d56708e62b40](https://github.com/ggml-org/llama.cpp/commit/184f6536024668e8908ab8c86033d56708e62b40), [MIT](https://github.com/ggml-org/llama.cpp/blob/184f6536024668e8908ab8c86033d56708e62b40/LICENSE). Small HTTPS source reads were matched to the parent-provided local checkout; no source was changed.

## Memory flow established by source

[ModelBase](https://github.com/ggml-org/llama.cpp/blob/184f6536024668e8908ab8c86033d56708e62b40/conversion/base.py#L249) indexes safetensors into fresh lazy tensor generators. It uses local file ranges rather than loading a model state_dict. `prepare_tensors`1230–1231 nevertheless converts BF16 to F32 before architecture transforms, even for `--outtype bf16`. `LazyTorchTensor.numpy`3055–3061 wraps this deferred result. Lazy does not mean chunked or free of a full-tensor cast.

[SafetensorsLocal/LocalTensor](https://github.com/ggml-org/llama.cpp/blob/184f6536024668e8908ab8c86033d56708e62b40/gguf-py/gguf/utility.py#L280) first reads the header; materialization later maps the individual range with copy-on-write NumPy memmap. This is a mapping, not zero resident memory. [LazyBase](https://github.com/ggml-org/llama.cpp/blob/184f6536024668e8908ab8c86033d56708e62b40/gguf-py/gguf/lazy.py#L165) memoizes evaluated data and eager arguments while the expression is live. `LazyChunkedTensor`232–293 provides a separate row-chunk writer; ordinary lazy tensors do not use it automatically.

[BF16 quantization](https://github.com/ggml-org/llama.cpp/blob/184f6536024668e8908ab8c86033d56708e62b40/gguf-py/gguf/quants.py#L29) allocates a complete output plus a list of row-group results before concatenate, despite each arithmetic group being small. BF16 rounding also quiets NaNs. This is another reason not to describe ordinary BF16 conversion as a raw byte copy.

[GGUFWriter](https://github.com/ggml-org/llama.cpp/blob/184f6536024668e8908ab8c86033d56708e62b40/gguf-py/gguf/gguf_writer.py#L380) keeps lazy tensor descriptions, writes each tensor with `tofile`, checks its byte count and releases `ti.tensor` after writing (448–485). The default therefore avoids materializing every transformed tensor together. `--no-lazy` defeats this advantage. `--use-temp-file` materializes each tensor into a spool during preparation; it does not chunk a large tensor or remove its cast, and adds temporary storage plus a second copy to final output. CLI rejects combining temp files with split output. Splitting separates tensors, not the largest tensor itself. Do not assume `--dry-run` makes a temp/eager path payload-free.

## Capacity arithmetic from admitted metadata only

The existing `llama-cpp-checkpoint-metadata-20261023-01/checkpoint-header-metadata.json` is185662B, SHA`a70a68255ea1648c31403ed829acca95e1a8cf64e6e6caf32c31894b51aff1b5`. It records677 BF16 tensors,23919460448 payload bytes, and a23919549408B source file. No payload values were inspected here. Largest embedding is262144×3840 =1006632960 elements: BF16=2013265920B (1.875GiB), F32=4026531840B (3.75GiB). Source mapping plus cast can coexist at5.625GiB, before framework and file-cache accounting. During ordinary BF16 requantization, F32 input plus complete BF16 output and grouped-output list can require roughly7.5GiB of arrays, before other live data; this is source-derived allocation arithmetic, not a measured peak or hard upper bound. Next-largest tensor is117964800B (112.5MiB).

Thus unmodified full conversion is not admitted by the earlier2GiB/30s tiny probe. Its saved metadata-time host surplus over original reserve was3243646156B, less than the F32 embedding alone; that old sample is not fresh admission. Neither temp nor output splitting cures this. Output writeback/file cache and source pages can also affect memcg and host pressure. Full conversion needs its own reviewed disk/time/memory budget and fresh unchanged host/Comfy predicates; no old Q95 gate is relaxed.

## Smallest practical converter-only path

There is no inspected CLI flag that bypasses BF16→F32. Prefer a narrowly reviewed optional BF16-preserving path in the existing converter/writer, not a new inference engine. Restrict it to the exact Unified class, explicit BF16 output, CPU contiguous source BF16 matrices with no numeric transform, no quantization_config, no QKV/expert fusion or other fusion/quant options. Keep original filtering, mapping, vocabulary handling and generated tensors. Gemma4 has zero norm shift; its inherited embedding transform only truncates to the tokenizer-derived row count. Norms, scalars and generated RoPE stay on the original F32 route. Preserve default lazy/no-temp behavior.

After the original permitted transforms, a lazy `view(torch.uint8).numpy()` can express BF16 bytes to the existing writer with explicit raw_dtypeBF16. The writer already converts the doubled byte last dimension back to the logical BF16 shape (338–365). Require actual eager CPU/dtype/contiguity, logical shape and byte-count checks; do not silently make an embedding-sized contiguous copy. Reject an unaccounted transform or class/flag combination. The official [Qwen row-chunk example](https://github.com/ggml-org/llama.cpp/blob/184f6536024668e8908ab8c86033d56708e62b40/conversion/qwen4exp.py#L174) is a fallback building block, but its special chunk object lacks the `.to`/slicing interface required by the current Gemma preparation path; inserting it unchanged is not a working flag.

Raw BF16 is not automatically identical to upstream for signaling NaNs: require bounded finite-value checks, or explicitly preserve upstream quiet-NaN behavior, plus byte tests at the actual assignment/write seam. For finite unchanged BF16 matrices, F32 roundtrip is exact. Prove output type/shape/ranges and source-to-output byte equality in bounded chunks over all accepted matrix tensors, while norms/scalars match the original F32 cast. No full-size output/reference array should be introduced by validation. This proves conversion values, not cross-engine model mathematics.

The next useful work is one full checkpoint conversion once metadata/key/tokenizer/LoRA acceptance and the converter patch review pass. Write a fresh local partial GGUF, account for every677 input key as retained/transformed/explicitly ignored, verify generated metadata and byte ranges, and promote only after complete write/readback. Provision final output space plus headroom; temp mode would need another output-sized area. Preserve failure/partial evidence. Do not substitute repeated tiny tensor probes for this complete conversion acceptance. The byte path removes large anonymous casts, but source/output page-cache charging means success under any chosen budget still requires measurement.

## Dependencies and CPU mmap route

Pinned converter requirements specify CPU-index Torch2.11.0 and include NumPy~=2.2.6, Transformers4.57.6, sentencepiece, gguf and protobuf constraints. The installed Torch2.13.0+cu130 probe environment is different; do not silently install into Comfy or claim it matches upstream pins. Use an isolated pinned converter environment or explicitly qualify selected versions. The C++ engine build is separate and does not require Torch at inference.

The [official CPU build](https://github.com/ggml-org/llama.cpp/blob/184f6536024668e8908ab8c86033d56708e62b40/docs/build.md#L33) uses CMake. Plan an isolated Release build with bounded jobs, CPU backend and GPU backends disabled; do not let build defaults or dynamic backend discovery select a GPU. Runtime `--device none` is stronger than `-ngl 0` (build note902). [Arguments](https://github.com/ggml-org/llama.cpp/blob/184f6536024668e8908ab8c86033d56708e62b40/common/arg.cpp#L2694) offer explicit `--load-mode mmap`; avoid mlock. [Loader](https://github.com/ggml-org/llama.cpp/blob/184f6536024668e8908ab8c86033d56708e62b40/src/llama-model-loader.cpp#L1418) and [POSIX mapping](https://github.com/ggml-org/llama.cpp/blob/184f6536024668e8908ab8c86033d56708e62b40/src/llama-mmap.cpp#L478) show mapping-backed tensors plus optional prefetch/MAP_POPULATE. Mmap is not an inference memory cap; full forwards still touch model weights and need KV/workspace. Native proportional-RoPE approximation, tokenizer/chat/tool semantics, LoRA handling,48-layer acceptance, context8192 and actual browser/quality/timing remain separate acceptance items. No speed, score or20-second/full12B qualification follows from this review.

## Exact small source receipts

Paths below are relative to the pinned upstream checkout. Hashing these small source files did not execute them.

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| conversion/base.py | 163672 | `727d7889676e2d7ea0e7f86991f5cfb6e00530fe386bc84ebf34248f464dea4f` |
| conversion/gemma.py | 58042 | `e0704e560cdbedea9a89a7d5b49a570e7482971301dc80389ce39ac9ac3c9aac` |
| gguf-py/gguf/utility.py | 13534 | `da920e2c62166ec9f30e407e9fb35ec29fda671310ffbf0f8ce856b62f1c70d7` |
| gguf-py/gguf/lazy.py | 12015 | `dbc98e3ee9ef8606df34e9d91f98ab29c2697e6824995f1dd8952938c135ad85` |
| gguf-py/gguf/quants.py | 66774 | `5c30d83fda855283a3239e479a26f35ea002ddd7650fcc4d255e6210fc9e72f1` |
| gguf-py/gguf/gguf_writer.py | 78312 | `5f7818929091dbfbf78fed8865536f5d92ee8186b2782393d29d6d69667f5001` |
| conversion/qwen4exp.py | 10101 | `a955f632a01068fc0d311325152a37807c290e283798ba6ab0a57e8347bed2fa` |
| convert_hf_to_gguf.py | 13012 | `e9a1da876330bbce9687541ab31736542a01b4ac43c6686126514a50f122fb7f` |
| requirements/requirements-convert_hf_to_gguf.txt | 200 | `3c1d689a0fb8b17057f012848819e405ca3949482e0554d8ae4586391b5bbbd4` |
| requirements/requirements-convert_legacy_llama.txt | 101 | `8e462f800429e87972ad4ecbf06a30c5e38bc381cda5597d0a36fb5110ca3cc8` |
| docs/build.md | 46354 | `6f01261dae39ab75f4d3535a9d26fb93e624df66aec150699570ce995fd7c828` |
| common/arg.cpp | 224650 | `bf4c58c189a774f2308acfe88e4ec5cebb0c43f60cf05cef4c5b2b9affd5822d` |
| src/llama-model-loader.cpp | 81142 | `702c36b758178981be9f18cd6f3111c40ca1bb11c1dc688a7afd4c58fe5bb248` |
| src/llama-mmap.cpp | 30403 | `130601ff8c2b6e16668b70a1dca07cc09ae84328d91bbd5bd259aea5658103e7` |
| LICENSE | 1078 | `94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d` |
