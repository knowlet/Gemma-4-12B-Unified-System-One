# Checked-control candidate: GPU numerics and first browser pair

2026-10-26. The 119-row candidate and matched Gemma NF4 baseline each scored
**0/1** on the original first browser task. The added checked-control training
has not improved this diagnostic's task success. The ten-case score and both
candidate JevBench scores remain null. This exposed development case establishes
no competitor win or general browser qualification.

| Arm | Original task judge | Policy calls | NF4 text calls | Actions | Terminal behavior |
| --- | ---: | ---: | ---: | ---: | --- |
| Decision3 Edge + 119-row operation delta | 0/1 | 4 | 0 | 4 | BLOCKED after no page progress |
| Matched Gemma NF4 baseline | 0/1 | 6 | 1 | 4 | DONE before satisfying the original goal |

Both arms completed their infrastructure flow without a task execution error.
The candidate ran first; its service closed before the baseline used the same
resident NF4 helper. The whole original launch, including startup and cleanup,
took 226.04601590003585 seconds. This is not a per-request latency measurement.

The pair retains the original first task, upstream revision
`1231850a0bf1a0c0341fe408ef1668dbbfdfac46`, V7 observation helper, seed 20261014,
one repeat, 16-step limit, standard actions, and the original DOM verification
plus DONE success rule. Native requests retain their 20-second limit and HTTP
callers their 25-second limit. The 119-row candidate's alias and pinned artifacts
replace the [previous 109-row candidate](../2026-10-25/decision3-firstcase-gpu-paired-results.md).
The action executor, available choices, NF4 driver and stagnation rule are unchanged.

The independent browser audit reauthenticated 867 selected files and all 189
service sources, including six empty Python package files. All four candidate
HTTP calls matched their native witnesses byte for byte. Across 16 heads and
144 legal probability positions, independent FP64 replay passed the original
tolerance; maximum absolute error was `1.6805374791761807e-7`. All twelve
non-operation raw answers were unchanged. Both original process scopes passed
their closure checks, and the candidate worker's final allocator count was zero.

Independent page-fingerprint replay found progress flags
`[true, false, false, false]`. The first click checked a radio control; the final
three clicks selected that checked control again. Text-entry and select actions
were available throughout. The original three-transition stagnation rule caused
the stop. On these same captured features, the readout changed W0's BLOCKED to
CLICK in every call. This local comparison supplies no separate Edge-only
browser trajectory.

Before the browser test, independent GPU replay accepted 13 genuine native calls:
one zero control, ten complete training wires and two invented shape controls.
There were 48 heads and 673 legal probability positions per branch, totaling
1,346 probability comparisons. Independent FP64 replay matched every ordered
argmax at the original absolute `1e-6` plus relative `1e-5` tolerance. Maximum
absolute error was `2.3731090803558175e-7`. All 36 non-operation raw answers and
the missing-operation response were preserved. W0 and trained delta bytes matched
their pinned artifacts. The original GPU owner took 30.009860827995 seconds.
This confirms numerical composition, not action accuracy.

A separate posthoc diagnosis joined the accepted cached GPU responses to the
ten new **training** labels. No additional model calls were made and the original
numeric probe remained label-free. The candidate answered 6/10 correctly versus
W0's 2/10. Candidate results by training operation were CLICK 2/4, TYPE_TEXT 0/2,
SELECT 2/2 and DONE 2/2. These are in-sample results, not held-out accuracy or a
browser score. In particular, the added examples have not yet taught reliable
text-entry selection, even on their own training states.

The next experiment isolates optimizer budget on the same 119 frozen training
rows. Keep the original features, W0, labels, learning rate, regularization,
minibatch and seed; compare a longer zero-start fit using full-training-set loss
and per-operation accuracy. The earlier first/last fit losses came from different
minibatches and do not establish convergence. Any improvement still requires a
fresh native qualification and browser test.

| Candidate evaluation | Result |
| --- | --- |
| Original first browser task | 0/1 |
| Full ten-case browser suite | Not completed; null |
| Benchmark Heaven | Not measured; null |
| JevBench coherence | Not measured; null |

The separate original-model [Benchmark Heaven](../2026-10-25/decision3-native-heaven-gpu-results.md)
and [JevBench coherence](../2026-10-25/decision3-native-coherence-gpu-results.md)
scores do not transfer to this trained candidate. Raw tasks, requests, browser
states, labels and model artifacts remain private. Aggregate evidence hashes and
the final independent browser audit are recorded in the
[machine-readable report](decision3-checked-control-gpu-browser-results.json).
