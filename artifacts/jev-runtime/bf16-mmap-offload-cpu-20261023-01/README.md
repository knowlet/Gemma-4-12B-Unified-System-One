# Tiny BF16 mmap / CPU offload mechanism fixture

This is a separate, dry-by-default CPU profile prepared for a parent-owned 2 GiB / 30 second / one-CPU experiment. Preparation and the nine stdlib tests do **not** import Torch, map tensors, execute the actor, or qualify any model. The parent controller owns admission, launch, process verification and closure. Original Q95/NF4/N3 sources and resource gates are unchanged.

`fixture.py` creates its own sub-1 MiB deterministic BF16 safetensors file. One tiny embedding → projection → tied output-head model observes the mmap lifetime and address mapping, same-dtype CPU pointer, supplied external state dictionary, real Accelerate meta/CPU hooks and explicit hook removal. It then attaches nonzero PEFT LoRA and compares enabled → disabled → enabled values exactly. Original forward bodies record materialized parameters; ordinary outer Torch pre-hooks would see meta parameters at the wrong seam.

Parameter object identity and non-meta storage alias are separate observations. Accelerate `cpu_offload` supplies neither a tied-parameter map nor a retie operation. The LoRA phase uses the already offloaded-and-detached model; its starting tie state is recorded and no silent retie is performed. A controlled projection-body error measures the residual devices before explicit recursive detach. Accelerate's wrapped forward has no `finally`, so a report of unsafe automatic restoration is an expected observation, not a reason to redefine it as safe. Partial hook installation and subsequent failures remain inside cleanup scopes, preserving the primary error and any cleanup notes.

The selected Python sources are bounded to 1 MiB per file and hashed before and after. Five native libraries have **stat identity only**; this is not a complete runtime closure. The fixture records package versions/origins, PID/start ticks, process group/session, elapsed time, peak RSS, generated-file hash/size and final integrity. Raw tensor files remain local and must not be staged; publish the generation source and JSON evidence instead.

CUDA is hidden before import. The guard is installed **after Torch import**; that import interval and unwrapped native calls are not monitored. Original read-only `is_available` / `device_count` queries are allowed, recorded and required to return False / 0. Wrapped initialization, stream, seed, cache and transfer work APIs refuse calls. Own tensors and offload execution use explicit CPU devices. A false `_initialized` flag and no blocked work attempts do not establish that every native CUDA call was absent. The documented `cpu_offload` execution device should be a GPU; this CPU experiment only observes mechanisms on the same device.

Default invocation prints dry status and performs no writes. The owner must explicitly supply:

```text
fixture.py --run --profile bf16-mmap-offload-cpu-fixture-v1 \
  --selection /absolute/path/selection.json --selection-sha256 EXACT_SHA \
  --entry-sha256 EXACT_ENTRY_SHA --output /absolute/fresh/output
```

The output's parent must exist; the output itself must not exist or have symlink ancestors. Required environment: empty `CUDA_VISIBLE_DEVICES`, `ACCELERATE_USE_CPU=true`, `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, `USE_TF=USE_FLAX=0`; the parent supplies the controlled runtime overlay and disables bytecode/user-site loading. The entry also works through `runpy.run_path` and requires no sibling imports.

Outputs are `intent.json`, the generated `tiny-weights.safetensors`, and `receipt.json`. Exit 0 means all mechanism checks completed, final source and own-file checks passed and `passed=true`; it does **not** mean tied objects survived, automatic exception restoration was safe, or any GPU/model qualification was earned. Exit 1 preserves a bounded primary error in the receipt when the selected output can be written. Pre-admission/environment/source/output refusal occurs before actor execution and may have no output receipt.

The CPU fixture cannot establish GPU transfer, UMA accounting, large-weight residency, 12B execution, 8192-token cache peak, 20-second latency, N3/Q95 compatibility, browser usefulness or benchmark scores. Those remain separate decisions after the parent reviews the actual observations.

Preparation evidence: nine stdlib-only boundary tests passed (unittest 0.084 seconds; subprocess wall 0.278 seconds), static Ruff passed, and the 23 small source / five stat-only selection check passed. The actor was **not run during source preparation**. See `validation.json`, `static-tests-02.txt` and `research-decision.md`; any later execution evidence belongs to the separate owner output.
