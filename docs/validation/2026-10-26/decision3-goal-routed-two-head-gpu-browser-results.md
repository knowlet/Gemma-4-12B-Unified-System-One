# Goal-routed two-head GPU and matched browser results

2026-10-26. The two-head candidate and its matched NF4 baseline both score
**0/1** on the unchanged first browser case. The new readouts pass native GPU
numerical verification, but the candidate navigates through an unrelated
submission link to a login page. Its subsequent request for login-field text
receives a valid JSON `null`, which the original helper correctly refuses to
authorize. The original baseline instead ends before satisfying the goal.

These are single-case development diagnostics. The full ten-case browser score,
JevBench coherenceMINI score and Benchmark Heaven score remain null. No
competitor win or generalization is established.

The [independently verified TRAIN fit](decision3-goal-routed-two-head-fit-results.md)
uses 153 operation rows and 14 teacher-active CLICK-target rows. Its high TRAIN
correctness does not predict success here. The first real CLICK menu has 13
choices, while the CLICK-target training menus have four or five. The later
menus contain 11 and 13 choices. This is an observed coverage gap; the evidence
does not establish menu size alone as the cause of the navigation error.

## Native GPU verification

One genuine d3-edge load serves 37 calls: one separate zero-control wrapper,
all 34 new full TRAIN wires, and two invented boundary cases. Independent FP64
replay checks 144 heads and 998 legal probabilities per branch, for 1,996
probability comparisons and 288 ordered argmax checks. The maximum probability
error is 2.997878586485925e-7, within absolute 1e-6 plus relative 1e-5.

Each adapted head uses its own actual feature row and separately trained delta.
The full 255 output codes, legal options and their order remain intact. W0 is
unchanged; unadapted raw answers and the entire zero-control response are
preserved. The owned GPU run takes 70.01582123799017 seconds wall time. These
checks establish numerical composition, not task accuracy.

## Browser outcome

The pair keeps the same pinned upstream revision, V7 observation helper,
full choices, text helper, seed, 16-step limit, stopping rules and DOM judge.
Candidate execution precedes the baseline, with candidate closure between
the arms. The whole pair takes 229.05789215298137 seconds.

| Matched arm | Success | Policy calls | Text calls | Actions | Observed failure |
| --- | ---: | ---: | ---: | ---: | --- |
| Two-head candidate | 0/1 | 3 | 1 | 1 | Wrong navigation, then refused null login-field value |
| Original NF4 baseline | 0/1 | 6 | 1 | 4 | Premature DONE |

All three native candidate requests return HTTP 200. The final text-helper
request returns HTTP 502 after producing `{"text": null}`. Generation reaches
actual EOS and a valid grammar terminal in 3.6580500510171987 seconds. Adapter
restoration succeeds, and neither a deadline nor an evidence failure occurs.
The original execution error and failed task result remain recorded.

Independent browser replay checks every actual native head and its selected
feature row, both delta identities, original transport bytes and the original
task outcomes: 10 heads, 74 legal probabilities per branch and 148 probability
comparisons, with maximum FP64 comparison error 3.019622020206114e-7. It also
replays both existing API normalization passes exactly. One head changes by
1.1102230246251565e-16 between raw and API probabilities; its choice is unchanged.
Both original scoped closures and current unit/cgroup absence are checked.
This is evidence about the authenticated owners and observed cohorts, not a
universal containment claim.

An earlier launch stops before starting the candidate GPU controller because
new source references mix absolute and relative paths. A fresh attempt fixes
path spelling and removes three duplicate references while preserving every
original hash and size. The controller, resource limits and scoring rules are
unchanged. That failed launch supplies no task score.

The next training change adds 24 independent local TRAIN scenes with navigation
distractors and 14, 18 or 24 authored clickable elements. Actual publisher menu
counts await genuine collection. The 120 related source and event controls
pass; no new browser examples have been accepted yet. The helper's null refusal remains;
evaluation goals and predictions do not become optimizer examples.

The [machine-readable report](decision3-goal-routed-two-head-gpu-browser-results.json)
records aggregate evidence and hashes. Raw goals, wires, labels, predictions,
generated tokens and weights remain private.
