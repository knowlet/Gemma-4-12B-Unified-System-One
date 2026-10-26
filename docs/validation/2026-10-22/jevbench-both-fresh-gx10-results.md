# Fresh GX10 results for both JevBench suites

2026-10-22. Fresh inference on NVIDIA GB10 completed both suites. The external controller confirmed the selected process and cgroup had closed.

| Suite | Result | Completion |
| --- | ---: | --- |
| Benchmark Heaven public231 accuracy | **84.42%** (195/231) | 231 valid answers, 100% coverage |
| JevBench coherence mini | **41.26/100** | 1,248 requests, 240 cases, 50 relations, zero errors |

The coherence 95% case-bootstrap interval is **38.43–43.80**. Public prediction latency was **0.184 s** at the median and **1.798 s** at the 95th percentile. These latencies exclude model loading and HTTP/browser roundtrips. Local cost is unmeasured. No official composite, sealed score or leaderboard rank is claimed.

The released model is `knowlet/Gemma-4-12B-Unified-System-One` at revision `a66f836b56605039fe040f330180e336d19b3362`, using BF16, no quantization/LoRA/cache/compile, causal-multislot context 16,384 and shipped temperature 2.7830344470383452. All 1,479 model calls used one constructor. Initial and final checkpoint, metadata and execution source identities matched. External closure confirmed the selected invocation, an empty cgroup and two empty process scans.

| Coherence dimension | Score /100 |
| --- | ---: |
| BAT | 40.83 |
| CHO | 22.22 |
| LOG | 67.92 |
| MEA | 16.18 |
| REP | 59.17 |

Batch order and probability complement relations scored 0; option permutation scored 4.17/100. These measurements show weaknesses in invariance and coherence. They do not establish the cause of a browser failure; a complete matched browser comparison is still required.

Historical A100 answers scored 197/231 and 40.50/100 when rescored. The separate independent development profile scored 71.96 historically and has not been freshly rerun. The new public231 result reproduced 195/231 in both GX10 attempts; only the second attempt completed coherence. The historical browser result of 5/10 remains separate.

An independent audit recomputed the official public summary, all coherence scores and 1,000 bootstrap draws exactly. It checked all 1,479 returned calls, 203 source pins and external closure. The aggregate JSON includes evidence hashes and all relation pass rates; raw questions, answers and caches are not published.
