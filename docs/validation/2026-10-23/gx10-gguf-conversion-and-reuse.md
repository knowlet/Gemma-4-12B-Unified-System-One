# GX10 native GGUF route: conversion regression and existing release reuse

The requested endpoint remains unchanged: genuine success on the pinned Jev browser tasks, a matched comparison with all six Decision3 models, and separate measurements for both JevBench suites. No new model score, full-model load,8192-context or20-second qualification follows from this milestone.

The previous goal turn made progress with an actual CUDA context/tiny operation and commit99a9a93. This continuation adds a complete checkpoint-name inventory, an actual low-memory conversion regression, and a concrete reuse path for the project's already published GGUF. A custom full inference engine is unnecessary for the next experiment.

## Exact source accounting

A fresh read through the original read-only Comfy observer admitted a bounded8+88,952-byte safetensors header read. Strict before/after inode/device/size/timestamps and the previously selected header hash matched. No model tensor payload was read for the inventory and the historical23.9GB whole-file hash was not recomputed.

All677 BF16 source states are accounted for:666 text states and11 multimodal states explicitly skipped by the text converter. There are329 text matrices and337 one-dimensional/scalar states. The text path adds one256-element F32 RoPE factor tensor, yielding667 outputs with default unfused conversion. Every text name maps uniquely in pinned upstream `184f6536024668e8908ab8c86033d56708e62b40`. The48 original layers,40 sliding/8 full pattern, full-layer missingV/K=V convention, and zero-prefixed per-layer indices are retained.

The32,169,626-byte tokenizer was separately read under a1GiB address-space bound and original host gate. Its SHA is `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`. The262,144 unique dense base IDs and24 added-special entries overlap at identical IDs, so they add no external vocabulary rows. BPE/byte fallback and the Sequence decoder meet the reviewed vocabulary loader's source predicates; actual model/tokenizer equivalence is still a separate runtime measurement.

Detailed evidence is in `artifacts/jev-runtime/llama-cpp-checkpoint-metadata-20261023-01/README.md`, `converter-coverage.json`, `tokenizer-metadata.json`, and `validation.json`.

## BF16 conversion fix and actual regression

The largest source matrix, `embed_tokens.weight`, is262144x3840:2,013,265,920 BF16 bytes. Upstream lazy conversion still casts this entire matrix to F32 before BF16 requantization. That cast alone is4,026,531,840 bytes; lazy/temp/split options do not remove the per-tensor allocation. These are source-derived allocation sizes, not a measured full-converter memory peak.

The local opt-in `--jev-preserve-bf16` path is restricted to unquantized local Gemma4 Unified BF16 conversion without fusion, eager/temp/remote/big-endian modes. It retains the original tensor transforms and GGUF writer. Multidimensional matrices use a contiguous CPU byte view; bounded masks reject NaN/infinity before writing. Norms, scalars and generated RoPE retain upstream F32 conversion. The default path remains unchanged. This avoids the full-matrix F32 allocation for the selected matrices; it does not demonstrate that a complete conversion fits its cgroup.

An actual Torch2.13 CPU process ran five regressions under the original observer/boundary,2GiB memory limit, zero swap, one CPU, and30-second unit runtime:

- All65,280 finite BF16 bit patterns matched the original F32-to-BF16 reference and their input bits.
- Actual `Gemma4UnifiedModel.prepare_tensors` and GGUF writing matched the default path for synthetic matrices, embedding truncation, norms, scalar and generated RoPE metadata/bytes.
- The selected matrix path made zero F32 casts; the default case exercised them.
- Lazy real-Torch input worked; nonfinite, wrong-dtype and noncontiguous inputs were refused.

All five passed in5.031 seconds; command wall time was12.370692594 seconds and root wall time14.030674426 seconds. Authenticated cgroup peak was524,214,272 bytes, not process RSS. The accepted host-gate aggregate covers17 discrete samples with13 raw loop samples; it is not continuous monitoring. The selected invocation closed through an authenticated live stop before its five-second hold expired, followed by two empty process scans and an absent cgroup. This proves the synthetic converter regression, not full checkpoint conversion, native math parity or model quality.

Sources, patch, reconstruction script,136-file manifest and MIT notice are in `artifacts/jev-runtime/llama-cpp-bf16-converter-20261023-01/`. The generated runtime copy remains local and can be reproduced by obtaining the pinned upstream source under `artifacts/jev-runtime/llama-cpp-source-184f653` and running `prepare_converter.py` in a fresh converter runtime directory. Saved source/result reviews are `independent-source-review.json` there and `llama-cpp-owner-20261023-01/independent-tiny-result-audit.json`.

## Prefer the existing published Q8 artifact

Repository inspection found the existing `tools/gguf/s1_gguf.cpp`, `scripts/gguf_adapter.py` and [published GGUF recipe](../../gguf-release.md). Its recorded trained source file SHA is the same `11144621122123135eb79c9cfd0466a2c51f230109124aefe401cdbec8c71a53` as the selected BF16 checkpoint. This is a provenance join through the published conversion record, not a new numerical equivalence claim.

A fresh read of immutable Hub revision `c418d37a17689fae554f7fc7d78db05ff7d52cfb` verified the3,723-byte artifact manifest SHA `1646a81f062fa3fe5727c835de569b669f288bd646749b2c3b96426bff4bf8f0` and4,090-byte conversion manifest SHA `464874bc51e8ee21ab56f34a4bf7d22b1fb83299a277cbcf966b42c5fc7802b7`.

The selected next native files are the12,669,647,264-byte Q8 language model (`9d3bc61851e5b7ed8e595b84327b85271830778d0e62fa9c1349c61843f369ed`) and122,031,744-byte F16 projector (`cffbff05efd290fef28d084ebcb653de7594f5e26bae0eabbedf30d6c782c1d1`). Full files had not yet been downloaded at this milestone. A reviewed streaming downloader selects only these fixed public files, enforces byte limits, hashes the downloaded stream, and renames only after size/SHA agreement. It retains failed partials and excludes transport URLs from failure records.

The S1 runner scores all ordered answer slots in the same causal sequence and selects legal candidate logits. It does not generate a JSON action or `TYPE_TEXT`. The current native-v2 grammar/PEFT constructor cannot be replaced by this decision-only runner without a genuine helper/backend integration. Its old blocking RPC also needs a real deadline before browser qualification. Saved native-v2 LoRA and calibration are absent from plain GGUF and cannot be silently inherited.

For the next build, an unchanged copy of the existing S1 C++ source uses the new pinned upstream libraries. Only the experimental CMake pin and explicit library build directory differ. The source/header review found the used C APIs present; compile/load evidence remains necessary. The native library/helper build is running under the reviewed7200-second owner at this milestone; it is not recorded here as passed. The earlier generic server build spec remains unused.

Two source-frozen command owners preserve the original observer and terminal authority. Both have2GiB memory, zero swap, one CPU, and explicit30/7200-second runtimes. Owner01 permits256MiB per file for build/tiny outputs; owner02 permits32GiB per file for separately selected downloads. Each passed26 stdlib regressions and Ruff. This changes file-output allowance, not the memory budget. Build and large download are selected serially to preserve host headroom. Only the authenticated selected unit/observed cohort is covered; universal process-tree or GPU restoration claims are not made.

## Remaining end-to-end work and scores

The six Decision3 publisher configs/manifests match upstream `VllmD3Model` source predicates, including the255-symbol readout, but none gains new load/latency/browser qualification from this metadata check. Generated dense readout matrices also need separate memory accounting.

The next sequence is actual native build, verified public artifact download, bounded full-model load/scoring and8192-context resource qualification, then a real text helper/deadline/LoRA integration and the unchanged browser campaign. Full-size sliding-window cache allocation must be budgeted before launch; the existing `swa_full=true` runner cannot be assumed to fit a2GiB unit at8192 context. Any compact-cache option requires its own equivalence and runtime measurement.

The currently measured scores remain Benchmark Heaven195/231=84.4155844155844%, coherence MINI41.26307189542484/100, and best genuine browser4/10. They belong to their recorded previous profiles. No cross-model victory or completed all-six comparison is claimed.
