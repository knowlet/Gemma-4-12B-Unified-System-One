# MPS local browser baseline

The six frozen public local tasks were each attempted three times per profile
with the actual 12B specialist checkpoint. Every attempt used a fresh isolated
Chromium context, the pinned Jev Ultrafast policy, a 12-cycle cap and a 90-second
policy budget. The independent final DOM verifier, rather than agent DONE,
determined whether the task succeeded.

| Profile | Verified tasks | Task latency p50 / p95 | Decision calls | False DONE |
| --- | ---: | ---: | ---: | ---: |
| causal_multislot | 15/18 | 40.793 / 52.151 s | 54 | 3 |
| independent | 18/18 | 51.198 / 76.538 s | 48 | 0 |

Both profiles passed all nine development attempts. Causal passed 6/9 reserved
transfer attempts, and independent passed 9/9. All three causal failures occurred
on the same reserved task, so these are not three independent failure domains.
The reserved outcomes did not guide the prefix-cache implementation, which was
derived from duplicated computation in development snapshots.

Independent mode had higher observed success and higher latency on this small
population. No claim of overall optimization or superiority to market models is
made. Text generation was never invoked. These are actual local DOM interactions,
not external website acceptance or official/sealed JevBench scores. Cost remains
unknown. First-use kernel costs were retained, and the host also performed browser
diagnostics; timings are exploratory. CUDA tests belong to gx10 and are not mixed
into these MPS results.

Environment: Apple M1 Max, 64 GiB unified memory; PyTorch 2.8.0, Transformers
5.17.0, MPS BF16/SDPA; Chromium 156.0.8078.4. The checkpoint's temperature stayed
at 2.7830344470383452. Per-task wall time includes initialization, final
verification, screenshot and cleanup. The policy budget excludes the final
verifier/cleanup and rejects inference overruns.

[report.json](report.json) contains every attempt, the runtime/checkpoint/source
identities, audit results and archive member hashes.
[browser-evidence.zip](browser-evidence.zip) retains all 36 full decision and DOM
traces, protocol logs, Playwright traces, final screenshots, and executed source
bytes. The archive is 4,968,387 bytes, SHA256
`708907f750c37dee5f988e03be65c998c708b9416416eb974c30d31c2dd838c6`.
Its CRC and all executed-source hashes were checked, final-state predicates and
backend call counts were recomputed, and raw probability schemas were revalidated.

The executed working tree was archived before its commits. Its recorded
`source_commit=eef847f...` identifies the base revision; it does not claim that
the newly added browser integration existed in that commit. Use the archive's
source bytes and hashes to reproduce this exact run.

Reproduce with the [Ultrafast instructions](../../../ultrafast-evaluation.md)
and a fresh run directory. The parent run continues its separate public231 and
coherence stages on MPS; their scores are not filled by this completed browser
stage. The external Wikipedia suite is frozen but has not yet run with the model.
