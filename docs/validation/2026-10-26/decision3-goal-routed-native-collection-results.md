# Goal-routed collection, GPU features and TRAIN diagnosis

2026-10-26. A subsequent
[two-head fit and independent full-TRAIN replay](decision3-goal-routed-two-head-fit-results.md)
now improves operation correctness on these cached rows from 10/34 to 31/34,
and teacher-active CLICK-target correctness from 6/14 to 14/14. The new
candidate's GPU composition, browser and both JevBench results remain pending.
The collection and base-model diagnosis below retain their original scope.

Fresh base-model GPU calls select the correct operation on 10/34 new
TRAIN wires and the correct target on 14/22 target-bearing wires. Joint operation
and target correctness is 10/34. Eight incorrect CLICK targets expose a second
gap that operation-only training cannot repair. Independent checks now accept
both the genuine collection and its GPU features. That collection and diagnosis
do not fit weights.

The collection itself covers twelve newly authored TRAIN scenes and 34 rows
with **zero model calls**. A subsequent feature capture makes 34 genuine
base-model calls. The diagnosis reuses those responses with no additional model
calls. These TRAIN results provide no held-out accuracy or browser benchmark
score.

The initial required operations are CLICK, TYPE_TEXT and SELECT, four scenes
each. The teacher follows the actual goal and current controls, preserving
irrelevant controls and skipping fields that already satisfy the goal. Already
checked radios remain offered and clickable. The collection retains the pinned
upstream publisher, settled V7 observation profile, standard actions, original
stop rule and goal checks.

| Native operation | First step of each scene | All admitted steps |
| --- | ---: | ---: |
| CLICK | 4 | 14 |
| TYPE_TEXT | 4 | 4 |
| SELECT | 4 | 4 |
| DONE | 0 | 12 |
| Total | 12 | 34 |

Every saved wire retains the natural K6 operation menu and all four question
heads in their original order. Across the 34 wires, there are 136 question heads
and 467 legal criterion positions. The collection counts describe the original
requests; the later GPU audit independently verifies their model distributions.
No option is pruned, and the native TYPE_TEXT operation identifier is preserved.

The original owned collection finishes in 9.002997181960382 seconds. All twelve
episodes complete, and the browser, context and original scoped owner closure
pass their checks. The independent audit authenticates 194 frozen snapshot
files, 93 captured source files, all 19 owner sources and 356 file identities in
total. It joins the original retained terminal, exact command identity, two live
OS observations and authenticated scoped closure, then replays the frozen
collector's complete admission and projection APIs. It makes no OS containment
claim.

An earlier attempt produced zero rows because its Python environment selected
an incompatible dependency ABI. That attempt remains quarantined. A fresh
collection used isolated Python startup with `-I`; no package installation was
needed. The accepted collection retains its original evidence and closure.

The admitted projection preserves each complete original payload, instructions,
question order and criterion order. Teacher choices, accepted targets and loss
masks remain outside the payload. A separate teacher file retains those
annotations; the feature-only requests contain exactly the case identity, wire
hash and payload. Their summary and accepted input view authorize feature
capture only. The original rows retain operation-only positive loss masks;
target and text annotations remain execution evidence in that format.

The feature capture loads the unchanged base encoder and W0 once and performs
34 native calls. Independent stdlib FP64 replay computes all 255 logits for
each of the 136 heads, checks all 467 legal probabilities, and matches every
ordered argmax and entropy confidence. Maximum absolute probability error is
2.497324314010285e-7 under absolute 1e-6 / relative 1e-5 tolerances. This numerical
agreement verifies the recorded inference; it does not mean every prediction is
correct.

The original feature owner finishes in 30.00828582298709 seconds; its command
takes 28.222825987031683 seconds, including a 3.332943815039471-second model load.
Complete calls take 0.33219152997480705–1.816344574035611 seconds. Peak CUDA tensor
allocation is 1,803,854,336 bytes, peak reservation is 2,680,160,256 bytes, and
the final allocation is zero. The audit accepts 32 owner sources, 130 leaf
sources, all 21 publisher load files and 286 rechecked file identities, including
the original scoped closure.

The separate posthoc TRAIN diagnosis joins the unchanged native answers to the
external teacher annotations:

| Teacher operation | Correct operation | Correct target | Correct operation and target |
| --- | ---: | ---: | ---: |
| CLICK | 0/14 | 6/14 | 0/14 |
| TYPE_TEXT | 3/4 | 4/4 | 3/4 |
| SELECT | 4/4 | 4/4 | 4/4 |
| DONE | 3/12 | N/A | 3/12 |
| Total | 10/34 | 14/22 | 10/34 |

Joint correctness requires the correct operation and, when the teacher operation
has a target, its correct target. The base model predicts DONE seven times:
three are correct and four are premature; it misses nine of the twelve teacher
DONE states. No new model call is made for this diagnosis.

All eight incorrect targets occur in teacher-active CLICK rows. With these
target answers fixed, even perfect operation selection would cap joint
correctness at **26/34 on these same TRAIN wires**. This is an in-sample
counterfactual; a new browser rollout can visit different states. It motivates a
separate 14-row CLICK-target training component alongside the 153
operation rows. The original row masks and target IDs remain unchanged.

This milestone does not change the earlier trained-candidate result. The 119-row,
10,000-step readout selected 10/10 correct operations on the ten added TRAIN
examples, while the first browser case still failed for both candidate and
matched baseline, 0/1 each. See the
[previous GPU and browser results](decision3-checked-control-sgd10000-gpu-browser-results.md).
The new data and TRAIN diagnosis do not establish that the repeated-radio
failure is fixed.

The [subsequent two-head fit](decision3-goal-routed-two-head-fit-results.md)
combines the unchanged 119 operation rows with these
34 new operation rows and adds only the 14 teacher-active CLICK-target labels in
a separate component. Independent dataset acceptance and a new fit capability
bind that component; the existing raw masks and target identities stay
unchanged. Its final artifacts and cached TRAIN metrics are independently
accepted. New GPU composition and candidate browser evaluation remain pending.
The full ten-case browser score, Benchmark Heaven score and JevBench coherence
score remain null. The two JevBench scores are kept separate, and no competitor
win or new model qualification is claimed.

The accepted independent collection audit has SHA-256
`8fd363a31ce7dadd02a4fefb544b44c1d2870136c9b504cb60ab340c0e245b88`.
The [machine-readable aggregate report](decision3-goal-routed-native-collection-results.json)
records the collection, GPU feature and posthoc TRAIN audit hashes, along with
the input-acceptance, projection, receipt, plan and source hashes.
Raw goals, wires, teacher labels and execution journals remain private.
