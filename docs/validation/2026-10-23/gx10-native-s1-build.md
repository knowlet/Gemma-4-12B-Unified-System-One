# Actual GX10 native S1 build

The native S1 runner and its llama.cpp libraries built successfully on GX10. This is a real CPU build and parser/help execution, not a model load or quality measurement. It supersedes the earlier in-progress snapshot in work-state132 and the conversion/reuse report.

All seven selected commands exited0: pristine-source check, upstream configuration, native library build, S1 configuration, S1 build, actual `--help`, and final pristine-source check. Root elapsed time was759.379083290 seconds, from2026-10-23T01:51:28.288334Z to02:04:07.667396Z. The library build took738.303 seconds; the S1 helper build16.063 seconds. The existing S1 C++ source was copied unchanged. The experimental CMake file changes only the upstream pin to184f6536024668e8908ab8c86033d56708e62b40 and selects the explicit library build directory.

The actual helper is a250,496-byte ELF64/AArch64 executable, SHA `22ee835d223f0384b0617cf15f82a24ddece247c78a7c320fa5d3925ff71f89a`. Five shared libraries were separately streamed/hashed for saved result review. All executable/library bytes stay local. The six artifact identities,14 complete command-log hashes, source/tool references, compiled pin and RPATH were joined in the independent saved-result audit. Calling help exits before backend initialization; it proves neither checkpoint loading nor numerical correctness.

The authenticated cgroup peak was669,175,808 bytes (638.17578125MiB), not process RSS. The selected unit had a2GiB hard memory cap, zero swap, one CPU,7200-second runtime and256MiB per-file cap. The unchanged original observer recorded745 accepted post-launch host samples:3 readiness,1 preexec and741 loop. Its aggregate minimum available host memory was29,041,745,920 bytes; the retained64 raw loop rows have a different minimum and cannot reproduce the full aggregate. These are sampled observations, not a continuous-time guarantee.

The unit stopped through its original authenticated boundary4.30305 seconds before fallback-hold expiry, with two empty final scans and an absent cgroup. Scope is the selected unit and observed cgroup cohort, not a universal process-tree/no-escape guarantee. Comfy was observed through its original read-only gate and was not modified.

Several ARM feature probes said `Failed`, but configuration and compilation succeeded. They are capability-test results, not build failures. `FETCHCONTENT_FULLY_DISCONNECTED` was reported unused; the selected flags disable known optional downloads, but the job was not protected by a network namespace or a general hard network denial.

Evidence: `artifacts/jev-runtime/llama-cpp-owner-20261023-01/actual-02/`, `root-execution-build-01.json`, and `llama-cpp-native-s1-20261023-01/independent-build-result-audit.json` (SHA `e26fb02de003277fdd20dd8bd01372c6300b36b14673148dece39ed3b7292fc9`).

After this build's authenticated closure, the independently reviewed download owner began the immutable public Q8/model-projector download. It remains a separate live operation until its exact stream sizes/hashes match and final receipts/closure exist. A growing `.partial` file is progress, not a verified model artifact. Poll its existing handle rather than restart from saved intent.

Before full-model startup, the next candidate must explicitly budget both the KV cache and CPU buffer policy. The original full-size8192-context sliding cache alone exceeds this2GiB unit. Source review also found default CPU extra-buffer/repacking paths for Q8 that may allocate large anonymous copies. A source-selected compact sliding-cache option and explicit CPU mmap/no-repack policy therefore need their own build, model load and numerical/runtime tests. No new defaults, old calibration/LoRA, N3/Q95 qualification,20-second compliance or browser score are inherited.

The goal remains active. The current measured results remain195/231 Benchmark Heaven,41.26307189542484/100 coherence MINI, and best genuine browser4/10, each for its previously recorded profile. A completed matched comparison of all six Decision3 models and a verified cross-model victory remain outstanding.
