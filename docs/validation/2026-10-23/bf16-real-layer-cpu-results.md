# Native BF16 decoder layers measured on GX10 CPU

The mapped-only experiment completed **all four planned cases and eight native forwards** using real checkpoint layers0 and5. Meta construction and strict assignment preserved every selected BF16 parameter/buffer pointer and file offset; selected-range hashes and state metadata remained unchanged. This establishes a small real-layer execution path under the existing resource boundary. It does not qualify full-model inference or improve a browser/benchmark score.

Root executed the frozen source once at **2026-10-23 00:24:27–00:24:46 UTC**. Parent exit0, actor status `completed_mechanism_observations`, no primary/secondary error. Source preparation passed17 actor and18 owner stdlib tests plus Ruff; these tests used synthetic/stub fixtures. The independent source review was `CLEAR_WITH_LIMITS` before the run. Actual actor/result and owner/resource audits read saved evidence only.

## Exact experiment

Base `knowlet/Gemma-4-12B-Unified-System-One`, revision `a66f836b56605039fe040f330180e336d19b3362`; installed Torch2.13.0+cu130, Transformers5.17.0 and safetensors0.8.0. Preserve the original48-layer config. Instantiate only native `Gemma4UnifiedTextDecoderLayer`0/sliding and5/full under meta, then `load_state_dict(strict=True,assign=True)` for14 and13 states respectively, including each BF16 `layer_scalar` buffer. Only one layer object is live at a time; no clone arm, embedding, final norm, lm_head or complete model is built.

Real CPU RoPE, original causal/sliding mask builders and the original Transformers SDPA adapter are used. A temporary instance-local wrapper records genuine Q/K/V arguments and delegates unchanged to the saved adapter; the registry is restored. All eight traces show CPU BF16 Q/K/V, dropout0 and scaling1. Sliding has KV heads8/head dimension256; full has KV heads1/dimension512 and native K=V construction. A profiler did not establish a native SDPA kernel kind.

Each case is a synthetic batch1 **prefill** with no past KV/cache, repeated twice with identical input. One token here is not autoregressive decoding. Both lengths are below the1024 sliding-window boundary. All output shapes/dtypes are correct and finite; exact repeat hashes agree. This is determinism, not an independent mathematical oracle or CPU/GPU/clone equivalence.

| Layer | Tokens | First forward | Second forward |
| --- | ---: | ---: | ---: |
| 0/sliding | 1 | 0.3880s | 0.3563s |
| 0/sliding | 32 | 0.3633s | 0.3620s |
| 5/full | 1 | 0.4358s | 0.4285s |
| 5/full | 32 | 0.4116s | 0.4065s |

These are two observations per case on CPU0 with one thread, not a latency percentile, throughput ranking or full-model timing estimate. Payload prehash warms file pages before each layer's forward. Hash/import/assignment costs are outside the table. Cold latency, disk IO bytes and kernel internal packing are not measured.

## Memory and integrity findings

Total selected state is **931,986,436B** (448,297,986B sliding;483,688,450B full). The actor explicitly hashes exactly those27 tensor ranges before/after, for1,863,972,872B of explicit payload reads. It never explicitly reads their broad20GB bounding envelope. `safe_open` maps a full virtual file but obtains only selected tensors; OS page-level readahead may touch adjacent bytes. The historical whole23.9GB blob hash is not recomputed; strict stat/header pins plus observed selected payload hashes define this run's narrower integrity scope.

Parent wall time is **18.7587s**, actor elapsed including imports17.7422s; import/native-config readiness alone is10.4818s. Actor high-water RSS is **1,491,501,056B (1.389GiB)**. The authenticated unit/kernel memory peak is **1,593,921,536B (1.484GiB)** under `MemoryMax=2GiB`, swap0. These are different accounting scopes and are not interchangeable physical-memory totals.

| Stage | RSS | Anonymous RSS | File RSS | Unit current | Unit charged file |
| --- | ---: | ---: | ---: | ---: | ---: |
| Imports/config/rotary | 854,102,016 | 511,868,928 | 342,233,088 | 516,464,640 | 630,784 |
| Sliding assigned | 860,647,424 | 516,272,128 | 344,375,296 | 969,449,472 | 449,290,240 |
| Sliding integrity check | 1,449,291,776 | 640,716,800 | 808,574,976 | 1,103,319,040 | 453,160,960 |
| Sliding released + GC | 1,000,980,480 | 640,716,800 | 360,263,680 | 1,102,520,320 | 453,181,440 |
| Full integrity check | 1,486,327,808 | 642,031,616 | 844,296,192 | 1,588,748,288 | 937,197,568 |
| Full released + GC | 1,002,622,976 | 642,031,616 | 360,591,360 | 1,587,945,472 | 937,222,144 |

Deleting each layer/weight dictionary/closed handle and collecting garbage removes roughly that layer's bytes from file RSS. **Charged file cache remains**, as does much of the anonymous allocation. Releasing the object therefore does not make the unit return to its initial footprint. Strict assign avoids an observed dense weight copy at assignment, but these snapshots do not exclude transient kernel workspaces/internal copies. A48-layer resident-memory or speed estimate cannot be obtained by multiplying one observed stage.

## Owner and host result

The original observer, service authority and terminal boundary are unchanged. Own unit `jev-d3-real-layer-cpu-20261023-01.service`, invocation `a4779510347a49708111bffb29c8a2a6`, main PID315824/start6414518. Runtime30s, swap0, CPU quota100000/100000, observed main affinityCPU0, core0, FD512 and file8MiB limits retain their original checks. The parent seals a separate25s cooperative actor deadline after initial host admission. Completion of this tiny experiment within18.8s is not full-model20s qualification.

Host total130,595,864,576B and original reserve26,119,172,916B are unchanged. Initial available28,726,435,840B exceeds reserve+2GiB; final28,591,939,584B. All248 runtime gates were accepted. Only the first63 plus the latest observation,64 runtime observations in total, are retained; their minimum28,027,523,072B is **not the minimum over all248 gates or a continuous minimum**. Saved ComfyUI observations are HTTP idle0/0. ComfyUI was not modified or stopped.

The parent authenticated and stopped the still-live own main after its receipt, finishing4.5759s before the fallback-hold deadline. Two final scans have no matches/unknowns, the own cgroup is absent and the unit is not found. This establishes closure of the selected unit and observed cohort; it does not claim universal prevention of escape or closure of an unobserved whole tree.

CUDA is hidden, no forbidden guarded calls occur, and PyTorch CUDA initialization is false after import and at completion. This actor made no wrapped availability queries. Guards begin after Torch import and do not instrument every native call; no GPU/UMA result is inferred.

## Goal status and next boundary

Formal scores remain separately **Benchmark Heaven public231:195/231 (84.4155844155844%)** and **JevBench coherence MINI:41.26307189542484/100**. Best genuine browser result remains **4/10**; all-six Decision3 matched evaluation is incomplete and no cross-model victory is established. Original Q95 memory/context/identity/finalization gates remain unchanged.

The next useful measurement is a separately reviewed, bounded GPU profile: measure context/allocator overhead first, then one genuine layer with CPU/GPU numerical comparison and live host-reserve checks. GPU allocation/accounting on shared GB10 memory, transfer/workspace peaks and cleanup must be measured before considering a full-model streaming loader. CUDA free memory and host MemAvailable must not be added together; context initialization is expected in that new profile and is released by closing its own process. CPU/GPU BF16 error metrics and tolerances must be fixed before comparison rather than requiring cross-device hash equality. This successful CPU result supplies no GPU memory budget,8192-context result,N3/Q95 equivalence,training result or browser-quality qualification.

Evidence is under `artifacts/jev-runtime/bf16-real-layer-cpu-20261023-01` and `bf16-real-layer-cpu-owner-20261023-01`: frozen source/selection/header metadata, static validation, independent source/result/owner audits, root execution/derived result and actual-01 receipts with21 immediate stage records. No model tensor, hidden-state/output tensor, benchmark input/answer/DOM or credential payload is included in the commit.
