# Frozen Decision3 CPU5 delta: real 4/2-row numerical qualification

The frozen 32-step train-only candidate now passes genuine CPU inference composition checks. This is a numerical prerequisite for HTTP/browser and heldout evaluation; it is not an accuracy, task-completion or model-win result.

One original `vllm-sr/d3-edge` load used the unchanged BF16 CPU backbone, FP32 original readout, temperature 1, noncausal full attention and rejection-only 16,384-token context. Owner09 retained CPU 15–19, five threads, quota 500%, full 4 GiB and swap 0. W0 was not merged or mutated. The candidate keeps its distinct `d3-edge-dsl01-train-delta-cpu5-candidate` alias.

| Call | Real rows | Complete call seconds |
|---|---:|---:|
| original | 4 | 3.123117 |
| candidate | 4 | 3.144741 |
| original | 2 | 2.620912 |
| candidate | 2 | 2.642997 |

Each call had exactly one encoder/final-norm event, `[rows,188,1024]`. Four-row calls retained legal counts `[8,1,2,3]`; two-row calls used only the original operation/click heads with `[8,1]`. The singleton confidence stayed 1. There were no dummy rows or extra encoder calls.

The original controls returned unmodified native responses; only the projection primitive received zero delta. The candidate calls loaded the actual accepted nonzero delta and used the released composition backend. Root checked the retained actual FP32 base/delta/adapted legal logits against same-forward reconstructions, every legal probability, ordered argmax and entropy. A separate stdlib FP64 accumulation witness checked every legal probability/argmax from the retained FP32 h/W0/delta bytes; its largest probability difference was 1.05883320445e-06, within the fixed mixed tolerance `1e-6 + 1e-5*abs(reference)`. This extra witness is not an independent FP32 implementation.

Complete calls include construction, required response/logit/array/catalog retention, full source/model/runtime reauthentication, hook/watchdog cleanup and the final 20-second refusal gate. The original root execution took 26.374785 seconds, below whole120. The sampled prefinal owned peak was 1,025,757,184 bytes; memory events were zero. This short fixture does not establish performance on long browser pages.

Root joined the exact 60-source selection, original terminal0/successful owner receipt/current authenticated scoped closure, all20 model files and28 runtime identities (22 hashes,6 stat-only), unchanged original readout bytes and the frozen trained artifact. NumPy's22 critical CPU files were authenticated; two mapped CUDA BLAS candidates remained stat-only with execution/symbol binding unproven. No full/transitive DSO claim is made. Source peers read source only; root performed the actual artifact audit, so no independent actual-reader claim is made.

The numerical fixture has no semantic expected answers and is independent of the DSL corpus, official benchmarks and exposed browser diagnostic. All new h/W0/raw outputs stay private and permanently nontraining. No weights, features or raw predictions are in this commit.

The existing separate benchmark results remain unchanged: original D3 Edge JevBench public accuracy158/231 (68.40%); original coherence1234/1248 incomplete, score null. Candidate HTTP/browser success, development/sealed quality and both candidate JevBench scores remain unmeasured. The previous native Copper diagnostic failed completion; this numerical pass does not change that result.

See the adjacent JSON and the root acceptance for exact refs and aggregate evidence.
