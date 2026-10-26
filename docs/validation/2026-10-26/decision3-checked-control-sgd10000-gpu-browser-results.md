# SGD10000 native GPU and browser results

2026-10-26. The 10,000-step operation readout selects all ten correct operations
on new genuine GPU calls for the added **training** examples. The unchanged
first browser case still fails: candidate 0/1, matched NF4 baseline 0/1. The
training improvement does not establish browser usability or a competitor win.

| Candidate | Genuine GPU TRAIN operation correctness | Correct TRAIN TYPE_TEXT examples | First browser case | Matched NF4 baseline |
| --- | ---: | ---: | ---: | ---: |
| 119 rows, 512 SGD steps | 6/10 | 0/2 | 0/1 | 0/1 |
| Same 119 rows, 10,000 SGD steps | 10/10 | 2/2 | 0/1 | 0/1 |

The new qualification loads the frozen d3-edge encoder once on GX10 and makes
13 native calls: one direct zero-composition control, ten complete training
wires, and two invented structural controls. Independent stdlib FP64 replay
checks all 255 logits per head, 48 heads per branch, 1,346 legal probability
comparisons and 96 ordered argmax checks. Maximum absolute probability error
is 2.3731090803558175e-7 under the unchanged absolute 1e-6 / relative 1e-5
tolerances. All 36 non-operation answer rows and the missing-operation response
remain exactly preserved. W0 and the new trained delta match their selected
FP32 artifacts.

The original owned launch finishes in 30.010252251988277 seconds. The ten real
calls take 0.936782569042407–0.9767610710114241 seconds each. Peak CUDA tensor
allocation is 2,602,280,960 bytes and peak reservation is 3,825,205,248 bytes.
Independent checks accept all 55 owner sources, 130 leaf sources, 21 publisher
load files, original process identity observations and scoped closure. The
selected 8 GiB limit, five CPUs, zero swap and 120-second budget remain fixed.

That numerical probe remains label-free. A separate posthoc audit joins its ten
new native responses to the unchanged TRAIN wire identities and ordered labels:
CLICK 4/4, TYPE_TEXT 2/2, SELECT 2/2 and DONE 2/2. W0 alone selects 2/10 correct
operations on those same calls. This confirms the earlier cached-feature result
through fresh native inference. It provides no held-out accuracy estimate. The
complete 119-row accuracy, CE, margins and gradients still await independent
full-array replay.

The new browser pair retains the pinned upstream revision, V7 observation
helper, standard actions, seed 20261014, one repeat, 16-step limit and original
three-unchanged-action stop. The candidate runs before its matched baseline
using the same resident NF4 text model and helper. Its original launch finishes
in 223.12007703998825 seconds.

| Arm | Successes | Policy calls | Text calls | Actions | Terminal | Premature DONE |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| SGD10000 candidate | 0/1 | 4 | 0 | 4 | blocked | false |
| NF4 baseline | 0/1 | 6 | 1 | 4 | done | true |

The candidate repeats the same failure as the 512-step artifact. Its first
radio click changes the page from unchecked to checked. Three further clicks
on that same checked radio leave the original page fingerprint unchanged,
triggering the original stagnation stop. TYPE_TEXT and SELECT remain available
at every step. The model selects CLICK; the blocked terminal comes from the
browser's stop rule. The baseline reaches DONE without satisfying the task.

The independent pair audit authenticates all 870 outer sources, 190 service
sources including six empty Python packages, and all eight service dependencies.
All four policy calls match the HTTP and native witnesses, including complete
state, instructions, criterion values and order. Independent replay checks 16
heads and 144 legal probability positions with maximum absolute error
1.4973786233163366e-7. All 12 non-operation answer rows remain byte-identical.
Both original scoped closures pass and the owned allocator ends at zero bytes.
These checks confirm that the selected new delta was used correctly during the
failed task; no OS containment claim is made.

The first case is an exposed development diagnostic. No evaluation goal or
label enters optimization, and this result is not an unseen evaluation. The
next experiment is to test independently generated TRAIN scene coverage for
goal-relevant operations and irrelevant checked controls, retaining the offered
choices, native events, stop rule and judge.

The new candidate's full ten-case browser score, Benchmark Heaven score and
JevBench coherence score remain null. The two JevBench measures remain separate;
this experiment supplies neither score. See the
[machine-readable aggregates](decision3-checked-control-sgd10000-gpu-browser-results.json),
[optimizer-budget experiment](decision3-checked-control-optimizer-budget-results.md)
and [512-step browser result](decision3-checked-control-gpu-browser-results.md).
Raw goals, requests, labels, features and weights remain private.

The [goal-routed collection and native TRAIN diagnostic](decision3-goal-routed-native-collection-results.md)
now adds 34 independently admitted rows and 34 genuine base-model GPU calls.
The base model selects only 6/14 correct teacher-active CLICK targets in that
training probe. The next candidate therefore needs separate operation and
CLICK-target supervision; increasing operation-only accuracy cannot correct
those target errors. New candidate browser and JevBench scores remain pending.
