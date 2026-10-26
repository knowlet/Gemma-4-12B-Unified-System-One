# Navigation-distractor two-head GPU verification

The new 225-operation/47-CLICK-target candidate passes genuine GPU composition
and independent FP64 checks for **all 72 complete TRAIN wires**. Two fixed groups
of 36 cover the original ordered population exactly once, with no omissions,
overlap, prediction-based selection or target filtering. This verifies the
composition and its original execution evidence. It provides no browser success
or benchmark accuracy result.

The [fit report](decision3-navigation-distractor-two-head-fit-results.md) records
202/225 operation and 47/47 CLICK-target correctness on cached TRAIN features,
including the old-prefix operation regression from 147/153 to 137/153. The GPU
check uses those full new deltas. That regression remains unresolved.

Each group loads one genuine d3-edge BF16 encoder with its frozen FP32 readout.
For each request, the composition uses the original same-call feature row for
that head. Separate operation and CLICK deltas are applied without merging into
W0; other heads' raw answers remain exact. Every original offered target remains
in its original order. Model input contains no added supervision field.

| Original owned group | Fixed real inputs | Native calls including controls | Whole owner | Selected command |
| --- | --- | --- | --- | --- |
| 1 | 0 through 35 | 39 | 79.0200173179619 s | 77.91924193297746 s |
| 2 | 36 through 71 | 39 | 84.02196988899959 s | 81.99673275800887 s |

Each group retains one explicit both-zero structural control and two invented
boundary controls. Across both groups there are 78 genuine native calls and two
model loads. Controls are reported separately from the 72 real TRAIN inputs.
The two independent scopes each retain the original five-core CPU quota, 8 GiB
memory limit, zero swap, 120-second outer limit, 113-second command limit and
20-second native call limit. Their 163.0419872069615-second summed duration is
not represented as a single 120-second execution.

Independent stdlib FP64 replay computes all 255 logits for each actual head row
and its matching delta, then checks ordered legal softmax, argmax and entropy
confidence. The real inputs contribute 288 heads and 2,075 legal probabilities
per branch; including controls, totals are 304 heads and 3,164 probabilities per
branch. This produces **6,328 probability comparisons and 608 ordered argmax
checks**, all passing. Maximum absolute probability error is
3.4793564546165e-7 for the frozen baseline and 3.1530579946981163e-7 for the
composition, within absolute 1e-6 plus relative 1e-5. Illegal tails remain zero.

Operation choices change on 50/72 inputs and CLICK-target choices on 53/72.
These are changes relative to the frozen baseline, not correctness counts.
The numerical check does not evaluate teacher accuracy or rerun optimization.

The unchanged parent source limit remains 64; each original owner authenticates
44 sources, while the qualifier checks a complete 165-entry leaf inventory and
all 21 publisher load files. Each shard's independent audit checks both original
live identity samples, scoped unit/cohort closure and current unit/cgroup absence.
A final aggregate independently replays both accepted audits, proves the exact
ordered 72-wire union and reauthenticates 789 distinct read identities after both
replays. It makes no universal process-tree containment claim.

Peak CUDA tensor allocation is 2,604,217,344 bytes and peak reservation is
5,425,332,224 bytes across the two scopes. The selected process's CUDA allocator
starts and ends at zero in each scope; these figures are not whole-machine UMA
usage. Complete real native calls, including their watchdog cleanup, take
1.4779887209879234 to 1.7714193349820562 seconds in this probe. These timings do
not include the full browser, HTTP service or text helper.

The [machine-readable report](decision3-navigation-distractor-two-head-gpu-results.json)
contains aggregate evidence and immutable hashes. The next step is a short
matched browser pair with unchanged actions, helper, judge and budgets. This
candidate's full browser score, JevBench coherence score and Benchmark Heaven
score remain unmeasured. The previous candidate's
[0/1 versus 0/1 pair](decision3-goal-routed-two-head-gpu-browser-results.md)
remains historical evidence; no ranking or win is claimed.
