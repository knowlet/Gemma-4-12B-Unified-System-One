# Explicit CPU mmap selection for the native S1 runner

This fresh v3 copies frozen v2 and adds optional `--cpu-mmap-only 0|1`, default `0`. Exact `1` requires `--gpu-layers 0`; the combination is validated after all CLI options and before `ggml_backend_load_all`, backend initialization or model construction. Invalid values such as `2`, `-1`, `true`, `01`, trailing junk and a missing value are refused by the exact-string source guard. A later root-only compiled CLI check must verify that behavior; no native parser was run here.

When enabled, the runner selects existing pinned llama.cpp controls:

```cpp
mp.load_mode = LLAMA_LOAD_MODE_MMAP;
mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
mp.use_extra_bufts = false;
cp.offload_kqv = false;
cp.op_offload = false;
```

The first three fields explicitly request mmap, turn off the architecture's optional lazy tensor loading, and avoid CPU extra-buffer weight repacking. The context fields disable K/Q/V and host-operation offload. The disabled/default branch leaves the native model/context defaults unchanged. Successful inference responses add `runtime.cpu_mmap_only` as the same boolean used to select those settings; opaque model/context flags are not independently read back.

This option is intended only with root's exact selected CPU-only `184f6536024668e8908ab8c86033d56708e62b40` libraries and build identity. CLI flags do not universally prevent backend discovery, driver initialization, accelerator/host buffer selection or all anonymous allocation. CMake retains the exact revision/clean-source and explicit build-directory checks, but does not itself prove the chosen libraries are CPU-only. Actual maps, native diagnostics, allocator/cache/workspace memory, file backing and owned closure still need evidence. No memory-fit or GPU-absence result follows from the flag name.

The known risk addressed is that upstream AUTO loading plus `use_extra_bufts=true` can select the `CPU_REPACK` extra buffer for Q8_0 on supported CPUs and allocate repacked weight storage. `research-decision.md` records the actual pinned source before implementation. Disabling that route is not a custom loader and does not establish universal mmap residency or UMA/memcg behavior.

Seven exact source changes are saved in `source.patch` and `source-delta.json`. The inverse reproduces complete v2 source `b9205973d97145adff2ba5368d935597ab904245b701551d3148f94999cff19a`, whose original five-change inverse reproduces v1 `8dd7f74a3ba38fda24d305c2bcbb7793d23eb38011ccb58f4f0230f677caa70e`. CMake is byte-identical across all three. The v2 `--swa-full` flag/default/response boolean, complete 48-layer S1 decision/slot/logit path, calibration, LoRA, RPC, helper, ownership and resource gates are preserved. No original source/build inputs were edited.

Two pure stdlib source regressions passed on the final test bytes: exact v3/v2/v1 inverse and patch equality, plus complete CMake identity. Final evidence is `cpu-tests-02.txt` and `ruff-02.txt`; the first lint finding was only import-block spacing and is retained. Tests used CPU 0, a 1 GiB address-space limit, 30-second CPU limit, core dump limit 0 and 45-second wall timeout. Ruff used one CPU/thread. No compiler, native executable/library, Torch, model/header/payload or hardware operation was run.

The cache arithmetic remains the frozen v2 `kv-arithmetic.json` hypothesis: at context 8192/microbatch 512, full K/V payload is 2,818,572,288 bytes versus compact 637,534,208 bytes, excluding all other memory. It is not a new measured result. Root's compiled invalid-value/combination checks, real full/compact context-2048 comparison crossing position 1024, separate context-8192 compact run and actual mmap/no-repack evidence remain required. The 2 GiB gate and host reserve are unchanged; no N3/Q95, numerical equivalence, 20-second or quality qualification is inherited.
