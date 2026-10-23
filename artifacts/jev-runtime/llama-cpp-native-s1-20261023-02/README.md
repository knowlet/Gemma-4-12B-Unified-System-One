# Explicit SWA cache flag for the native S1 runner

This fresh source copy adds `--swa-full 0|1` to the existing persistent, non-generative S1 runner. Omitting it retains `true`; `1` selects full-size SWA storage and `0` selects the pinned llama.cpp compact-cache path. The parser accepts only those exact two strings and refuses `2`, negative values, words, `01`, trailing junk and a missing value. Help describes both modes and the unchanged default. Each successful response records `runtime.swa_full` as a JSON boolean from the same `opts.swa_full` value passed to `cp.swa_full`; no separate C API readback of the opaque context is claimed.

The C++ source has exactly five controlled changes, saved in `source.patch` and `source-delta.json`. Its complete inverse reproduces v1 SHA-256 `8dd7f74a3ba38fda24d305c2bcbb7793d23eb38011ccb58f4f0230f677caa70e`. `CMakeLists.txt` is byte-identical to v1 and still requires pristine llama.cpp `184f6536024668e8908ab8c86033d56708e62b40` and the explicit selected library build directory. The v1 files and current build inputs remain untouched. This directory does not copy v1's root build receipt or claim that receipt for the new source.

All original model/context/batch/thread/GPU defaults, 48-layer architecture, token preparation, slot scoring, attention math, native media handling, softcap, calibration, process protocol and cleanup remain in the inverse-protected portion. No LoRA or RPC feature was added. This flag does not create or inherit an N3, NF4, Q95, browser quality or resource qualification.

The source-only cache estimates for F16 K and V, one sequence, window 1024 and microbatch 512 are:

| Context | Full SWA | Compact SWA | Full-attention cache retained |
| --- | ---: | ---: | ---: |
| 2048 | 704,643,072 total bytes (672 MiB) | 536,870,912 total bytes (512 MiB) | 33,554,432 bytes |
| 8192 | 2,818,572,288 total bytes (2688 MiB) | 637,534,208 total bytes (608 MiB) | 134,217,728 bytes |

Totals include the 40 SWA layers and eight full-attention layers. Compact storage has 1536 SWA cells in both rows, using the actual pinned `PAD(min(context, window+n_ubatch),256)` formula. These are arithmetic tensor payloads, excluding graph/output/workspace, model pages, bookkeeping, alignment, GPU context/driver and UMA/memcg charging. The original 2 GiB gate and host reserve remain unchanged; no memory-fit result is claimed.

Two stdlib-only source regressions passed: exact five-change successor plus saved-patch/full-inverse equality, and exact parent CMake identity. Ruff passed for that final test source. Evidence is `cpu-tests-02.txt` and `ruff-02.txt`. The first lint attempt found only import ordering; its saved output is retained, and the final test source was rerun after correction. Neither test compiles/runs C++, simulates the parser, imports Torch/native libraries, or reads any model/header/payload. Actual CLI behavior is a deferred compiled check; this source review must not be called its runtime test.

Future root-only validation must check omitted/`1`/`0` and rejected-value CLI cases, then compare actual full versus compact runs at context 2048 with identical inputs crossing position 1024. Compare complete legal logits and answers as well as actual cache/whole-process memory. A separate actual context-8192 compact run is then required. The smaller comparison does not establish cache eviction/reuse equivalence, full-context memory fit, 20-second performance or any task score.

`research-decision.md` records the upstream source mechanism, license and decision before implementation. `kv-arithmetic.json` records the source pins and exact assumptions. Root owns review, any future build/run, and conventional commit/push; this preparation launches nothing.
