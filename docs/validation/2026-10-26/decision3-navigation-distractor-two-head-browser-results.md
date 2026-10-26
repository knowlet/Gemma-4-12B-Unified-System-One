# Navigation-distractor two-head browser result: still 0/1

The new 225-operation/47-CLICK-target candidate and its matched NF4 baseline both
score **0/1** on the same known first-case diagnostic. The candidate performs
four CLICK actions and ends `blocked`; NF4 performs four actions and ends `done`
with the goal unsatisfied. The original judge therefore records NF4's ending as
premature DONE. Neither arm solves the task, and no competitor win is claimed.

This follows the [TRAIN fit](decision3-navigation-distractor-two-head-fit-results.md)
and [complete 72-wire GPU verification](decision3-navigation-distractor-two-head-gpu-results.md).
The 47/47 CLICK-target TRAIN result counts only teacher-active CLICK rows;
it does not measure whether CLICK is the right operation. Passing GPU numerics
also provide no browser success result. The earlier 147/153 to 137/153
operation-prefix regression remains part of this candidate's record.

| Same-case arm | Success | Policy calls | Text calls | Actions | Terminal |
| --- | --- | --- | --- | --- | --- |
| New d3-edge two-head candidate | 0/1 | 4 | 0 | 4 | blocked |
| Generated-actions NF4 baseline | 0/1 | 6 | 1 | 4 | premature DONE |

The pair preserves the original actions, native menus, V7 observations, text
helper, judge, seed 20261014, one repeat and 16-step budget. Candidate runs first,
then NF4. The original owned execution finishes in 229.05252778396243 seconds.
Both arms have complete infrastructure receipts and no execution exception.
There is no helper refusal in either arm; the candidate never requests text.

## Verified failure: repeatedly selecting an already-checked radio

All four original candidate policy responses choose CLICK on the **same radio
target**. Its first request shows `checked=false`; all three subsequent requests
show `checked=true`. The full native input retains recent-action histories of
length 0, 1, 2 and 3. Thus the selected state and accumulating history are
present when the model repeats the choice.

The page URL and visible text stay unchanged across all four requests. The last
three element snapshots are identical. The last request's history records the
first three clicks with `page_changed` values `[true, false, false]`. This
supports a state/history failure in the decision policy. It does not identify
an encoder defect, prove an optimizer cause, or replay an alternative full
trajectory. The original blocked outcome remains unchanged.

On those same actual feature rows, independent local readout comparison finds
that the operation delta changes frozen W0's BLOCKED choice to CLICK on all four
requests. The CLICK-target delta changes none of the four target choices. This
is a same-feature counterfactual, not the matched NF4 result or a complete W0
browser run. In this trajectory, the new target fit leaves the repeated target
choice unresolved.

Each real request offers seven operation choices, 13 CLICK targets, one text
target and 15 SELECT choices. The 72 newly added TRAIN wires offered six
operation choices, CLICK menus of 14/18/24 and SELECT menus of 3/4. The complete
225-row operation fit already includes five older K7 examples, so K7 is not
entirely absent from training. Existing TRAIN scenes already include a
prechecked requested radio followed by TYPE_TEXT and genuine action history.
These are concrete coverage differences; no single gap is proven to cause the
failure. The next independent TRAIN work should test the joint combinations of
selected state, history, larger SELECT menus and naturally available scroll
operations, including recovery after actual no-effect actions. Evaluation goals
and choices are not copied into TRAIN.

## Native numerics and original evidence

Independent FP64 replay checks all 255 baseline logits per actual head and each
present adapted head's 255 delta logits. Across four genuine native calls it
verifies 16 heads, 144 legal probabilities per branch, 288 probability
comparisons and 32 ordered argmax choices. Maximum absolute error is
2.558862728374578e-7, within absolute 1e-6 plus relative 1e-5. Both original API
normalization passes are replayed for every head; their maximum probability
difference is zero in this pair. Raw unadapted answers remain exact.

The audit reauthenticates 883 outer sources and the 207-entry service inventory,
including six unchanged empty package sources. It checks the two prerequisite
GPU shard reports, full original wire coverage, original execution receipts,
scoped closures and current unit/cgroup absence. The selected process's CUDA
allocator ends at zero. These checks make no universal process-tree containment
claim. Fourteen invented caller/launcher controls and the genuine metadata
preparation also pass before execution.

The [machine-readable report](decision3-navigation-distractor-two-head-browser-results.json)
contains aggregate evidence and immutable hashes. This is a known, previously
observed diagnostic used during development, not a new unseen evaluation. The
full ten-case browser score, JevBench coherence score and Benchmark Heaven score
remain unmeasured for this candidate. The two JevBench scores remain separate;
no full-suite score is inferred from this pair.
