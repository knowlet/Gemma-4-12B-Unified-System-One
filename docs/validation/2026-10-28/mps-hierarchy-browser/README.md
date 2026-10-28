# Preserve-all hierarchy browser diagnostic

Both profiles reach the exact original Graph theory article, and every frozen
URL/title/heading/body predicate passes. **Neither attempt passes the original
180-second policy-time budget: verified success remains 0/2.** No goal,
predicate, budget or scoring rule was relaxed.

| Profile | Goal observed | Policy time | Total wall time | Verified |
| --- | --- | ---: | ---: | ---: |
| causal_multislot | true | 201.721 s | 202.283 s | 0/1 |
| independent | true | 234.876 s | 236.077 s | 0/1 |

This is a consumed-task diagnostic after the
[default-reject candidate](../mps-torch214-browser/README.md), not fresh
transfer evidence or a matched performance comparison. It uses one attempt per
profile and preserves the twelve-cycle cap. Causal ends with accepted DONE after
the deadline; independent reaches the destination before the outer loop reports
`task_timeout`. The supervisor's terminal child exit is 1.

## What was exercised

`--oversized-head-policy hierarchical` is opt-in. The observed 57-candidate
target becomes two balanced contiguous groups of 29/28 in original order.
One joint core request contains the unchanged six-option operation head, the
two-option parent head, and **both** conditional heads. Original singleton
handling needs no model call. Every candidate ID and description is retained;
there is no “none” option, shortlist or unscored-candidate zero filling. The core
52-candidate contract and native context limit remain unchanged.

The returned product `P(group) × P(candidate | group)` is an **uncalibrated
hierarchical approximation**, not a claim of equivalence to flat or separate-call
scores. The trace preserves the deterministic group map, original instructions,
joint raw requests/responses and all internal head scores. The normal flat path
and default rejection policy are unchanged.

The hierarchy call takes 68.585 s for causal and 140.292 s for independent,
recording one backend invocation and respectively one and two completed
backbone forwards. Recorded input shapes are causal `[1, 7520]`, independent
`[1, 3136]` and `[3, 4650]`. Whole-call latency includes preparation and
maintenance; the traces do not separate token preparation, compilation and
kernel time. They cannot assign the delay to SDPA alone.

Causal first observes the destination URL/title at 139.957 s. Its first DONE at
169.166 s is discarded by the original freshness guard; the later accepted DONE
arrives at 200.006 s. The two DONE calls consume 59.902 s. Independent begins its
140.292-second hierarchy call after the 92.762-second tick and observes the
article at 233.812 s. These are diagnostic observations, not a finding that the
freshness guard is defective. All original budget failures remain frozen.

Direct cleanup wall totals are only 0.208/0.329 s, and recorded CDP totals are
2.931/2.502 s. These clocks overlap with callback/loop timings and must not be
added as independent phases. Small direct cleanup time cannot exclude indirect
next-call compilation cost from cache clearing; no such phase was measured.

## Provenance and archive

The declared source commit is
`db3cb963d0f7a56a05db914df7b818309762e430`; all 68 declared local source hashes
match the archived bytes. This differs from the earlier candidate's source
commit. Torch 2.14.1, Transformers 5.17.0, MPS BF16/SDPA, checkpoint manifest
`f38d7f50060eabcbed6f73d16404ad42eb6ba1bddc79b68052afc13cab8598eb`, and the
M1 Max hardware identity are preserved. No runtime/policy default is adopted.
The optional maintenance resource gate passes with zero recorded faults; it
does not override the failed task-time gate. Cost and official rank remain null.

[report.json](report.json) contains both raw-summary audits and hierarchy-call
details. [hierarchy-browser-evidence.zip](hierarchy-browser-evidence.zip) is
1,016,898 bytes with 106 payload members plus its embedded member manifest.
SHA-256:

```text
556ed4c3e4d7a55ecf5926e6f935634fe443186dcd27242fc45d856cb6227327
```

[archive-members.json](archive-members.json) binds each payload's size, CRC32
and SHA-256. ZIP CRC, exact member set and every digest passed verification.
The archive includes both raw browser traces, screenshots, final states,
executed sources, original manifest, maintenance trace and terminal supervisor
receipt/log. Weights, symlinks, tokenizer assets and large Playwright recordings
are excluded; omitted recordings retain their local hashes/sizes in the archive.
No model, browser or GPU was started while assembling this evidence.
