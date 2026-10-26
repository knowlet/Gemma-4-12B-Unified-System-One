# Actual GPU context and tiny BF16 operator on GX10

Successor02 completed all10 planned stages once on the selected NVIDIA GB10. Two16x16 BF16 all-ones inputs produced256 exact values of16; the512-byte readback hash independently matches `1cb8e75fc462d0af48a69d1c2ecc3b67abdb9358db5a4d38a1c5b1dbbe6fa003`. Parent exit0, actor completed, no cleanup/source/owner error. This is context/operator evidence, not real-layer or model qualification.

Root execution:2026-10-23 01:08:50–01:08:55 UTC; wall5.0418s, actor3.7871s including Torch import2.5736s. Explicit CUDA init0.0132s and initial synchronization0.1329s were timed separately. First matmul enqueue-plus-sync in this new process took0.8177s and includes backend startup. These are instrumented single-call observations, not cold-boot timing, kernel duration, model throughput or a20-second model result. CUDA metric calls can synchronize and perturb timings.

Native properties identify UUID `e3429caa-b057-cbe0-917d-d3b85b313be7`, NVIDIA GB10, logical0/one visible device, `is_integrated=1`, total130,595,864,576B and allocator backend `native`. The actual UUID binding is `_CUuuid`, despite the installed type hint declaring a string. V2 converts the native value to its canonical string before validation/storage and retains the native type name. Actual physical UUID matching passed inV2. Full HMM/ATS/pageable-memory capabilities were not queried.

## Memory findings

The own PyTorch caching allocator cap was64MiB with native set/get fraction roundtrip. Its observed maximum allocated bytes were33,555,968B and maximum reserved35,651,584B. Clearing the three tiny tensors, synchronizing and calling own `empty_cache()` left **32MiB allocated and reserved**, with CUDA still initialized. The retained allocation is observed; its precise backend ownership was not profiled. Empty cache does not establish context destruction or complete GPU-memory release.

| Stage | Process RSS MiB | Unit current MiB | CUDA allocated MiB | CUDA reserved MiB | Global CUDA free MiB | Host MemAvailable GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| torch-imported-preinit | 498.508 | 282.328 | unmeasured | unmeasured | unmeasured | 27.162 |
| cuda-init-synchronized | 591.289 | 341.695 | 0.000 | 0.000 | 911.113 | 26.939 |
| tiny-fill-synchronized | 604.348 | 352.695 | 0.001 | 2.000 | 875.277 | 26.904 |
| tiny-matmul-synchronized | 806.703 | 555.922 | 32.001 | 34.000 | 948.379 | 26.689 |
| own-tensors-released-empty-cache | 810.957 | 558.172 | 32.000 | 32.000 | 942.254 | 26.683 |

Preinit GPU values are unmeasured: querying them through the runtime would alter the baseline. Global CUDA free even increased between fill and matmul while own allocation increased; it is not an exclusive measure of this process. Host MemAvailable and CUDA free must not be added. Process/cgroup/host/device samples are sequential and have different scopes;2GiB memcg and64MiB allocator limits do not prove a hard cap on all driver/context/shared physical memory.

Actor high-water RSS was850,948,096B (811.527MiB); authenticated manager/kernel unit peak586,072,064B (558.922MiB). Original unit2GiB/swap0/30s, CPUquota100%, observed mainCPU0, core0/FD512/file8MiB, parent25s cooperative deadline and original host/Comfy gates were unchanged. No full-model capacity estimate follows from these peaks.

## Preserved failure and minimal fix

Original01 returned parent exit1 in4.0148s. It saved preinitFalse→initTrue and a failure-cleanup GPU snapshot, but no cap/operator/oracle completion or final actor receipt. The actor wrapper captured `TypeError: Object of type _CUuuid is not JSON serializable`; the owner then reported FileNotFoundError for the missing receipt. Pinned source explains that the strict UUID-string check would fail first, and storing the native object prevented final JSON error evidence from being written. That earlier UUID rejection is source-backed inference; the lost native UUID join is not reconstructed as success.

New actor02 changes only profile and UUID string/type storage; a fake native UUID regression executes the actual assignment seam. New owner02 preserves child type/message/evidence when the receipt is missing or invalid, still rejects successful exit without a receipt and still retains partial receipts before refusing exit2. All other owner functions remain AST-identical. Original01 source/results and its independent failure audit are preserved. Actor02 passed18 stdlib tests and owner02 passed22, plus Ruff; these are fake/metadata tests, separate from the actual CUDA run.

## Host and closure

Initial available29,485,846,528B exceeded original reserve26,119,172,916B plus2GiB. The recorded aggregate includes all70 accepted parent postlaunch samples:15 readiness,2 preexec and53 loop. Its sampled minimum was28,649,074,688B,2,529,901,772B above reserve. Only53 loop rows are retained; the17 setup rows are summarized, so their minimum cannot independently be recomputed from raw records. This is not a continuous minimum. Final available29,499,117,568B. Saved Comfy observations are HTTP idle0/0; no Comfy process/cache/queue was modified.

The original boundary authenticated the live own main328609/start6680815 and invocation7d2ac46c112f42a98ca372dbc80b516b, stopping it4.3955s before fallback-hold expiry. The own unit is not found, cgroup absent and both final scans have no matches/unknowns. A later read-only NVML query also excludes this PID. These prove the selected lifetime/observed cohort scope, not universal escape prevention, total driver-memory restoration or restoration of global GPUfree.

## Goal and next action

Formal scores remain separately **Benchmark Heaven public231:195/231 (84.4155844155844%)** and **JevBench coherence MINI:41.26307189542484/100**. Genuine browser best remains **4/10**. All-six Decision3 matched comparison is incomplete; no cross-model superiority is established. Original Q95 memory/context/identity/finalization gates are unchanged.

In parallel, primary-source review found direct Gemma4 Unified support in pinned [llama.cpp184f6536](https://github.com/ggml-org/llama.cpp/commit/184f6536024668e8908ab8c86033d56708e62b40), with MIT license. This is a mature engine candidate, not a verified replacement. Prioritize exact checkpoint-key/tokenizer/chat/LoRA/conversion acceptance and a bounded real text/browser path. Inspect largest-tensor casting/writer memory behavior before conversion; neither this tiny GPU run nor selected CPU layers prove full conversion fits. The source-only feasibility note lists48-layer classification, proportional-RoPE approximation and quantization gaps. A separately justified real-layer GPU numerical/transfer experiment remains useful if needed; its tolerances and capacity gates must be fixed before running.

Neither route inherits the prior N3/Q95 qualifications or JevBench scores. The requested end state remains actual browser success and matched competition. No model/header/payload/mmap, real layer,8192 context, full12B,20-second model behavior, training, model throughput or quality result was produced by this context probe.

Evidence: `gpu-context-probe-20261023-01/02` and `gpu-context-probe-owner-20261023-01/02` contain frozen sources, selections, tests, source reviews, preserved failure audit, saved result/owner audits, root execution/derived data,10 immediate stages and postclosure NVML metadata. Generated CUDA cache files remain local and are excluded from Git without reading their bytes. No tensor/benchmark task/answer/DOM/credential payload is committed.
