# Goal-routed two-head fit and independent TRAIN replay

2026-10-26. Separate operation and CLICK-target readouts now have independently
verified full-TRAIN results. Operation correctness improves from 62/153 to
147/153, including 10/34 to 31/34 on the new goal-routed rows. CLICK-target
correctness improves from 6/14 to 14/14. These measurements reuse the same
cached TRAIN features. At this fit milestone, the new candidate had no GPU
composition, browser, held-out or JevBench result. The subsequent
[GPU verification and matched browser result](decision3-goal-routed-two-head-gpu-browser-results.md)
is reported separately; both browser arms fail the first case, 0/1.

The [collection and base-model diagnosis](decision3-goal-routed-native-collection-results.md)
identified eight incorrect teacher-active CLICK targets that operation-only
training could not repair. This fit therefore uses two distinct, explicitly
authorized datasets: the unchanged 119 operation rows plus 34 new operation
rows, and all 14 teacher-active CLICK-target rows. The latter includes the six
targets the base model already selected correctly; it does not select only
mistakes. No inactive target head receives a label.

The original 119-row operation prefix, the new wires' original operation-only
masks and actual head and criterion ordinals remain unchanged. A separate
supervision authority selects the 14 CLICK-target losses without rewriting
those masks. Each readout uses its
own actual question row. Operation counts are K4: 4, K6: 44, K7: 5 and K8: 100;
CLICK-target counts are K4: 7 and K5: 7. CPU and GPU feature bytes are not claimed
to be identical.

Both independent FP32 deltas start at zero and run 10,000 SGD steps through the
unchanged numerical core: learning rate 0.003, L2 0.0001, seed 20261023 and batch
64. The effective batches are 64 operation rows and 14 CLICK-target rows. No
previous delta is used as a warm start. The d3-edge base remains at revision
`cab71e6ad3461b8086d51ddb572c656babd63b0b`, with 594,582,720 parameters, an
unchanged BF16 encoder and unchanged FP32 255×1024 W0 at temperature 1. The new
GPU features were captured previously; this fit and its separate audit make
zero model or browser calls.

Development diagnostics informed the strategy. Development and sealed data do
not enter the optimizer inputs.

The separate owned audit reconstructs both datasets and independently computes
FP64 zero-delta and final-delta TRAIN metrics:

| Readout and TRAIN group | W0 / zero delta | Fitted delta |
| --- | ---: | ---: |
| Operation, all rows | 62/153 | 147/153 |
| Operation, unchanged prior rows | 52/119 | 116/119 |
| Operation, new goal-routed rows | 10/34 | 31/34 |
| Operation, new teacher CLICK states | 0/14 | 11/14 |
| Operation, new teacher TYPE_TEXT states | 3/4 | 4/4 |
| Operation, new teacher SELECT states | 4/4 | 4/4 |
| Operation, new teacher DONE states | 3/12 | 12/12 |
| CLICK target, all teacher-active CLICK states | 6/14 | 14/14 |

Full-dataset cross entropy falls from 1.7985640715045823 to
0.3637370030299442 for operation, and from 1.1710166604855519 to
0.008150834711335871 for CLICK target. Final delta L2 norms are
4.6318031400625514 and 2.2440160578883614 respectively.

The audit checks all 255 output codes and exact zero tails: operation delta
rows 8–254 and CLICK-target delta rows 5–254 remain zero. Discrete correctness
and criterion-key histograms match exactly. Maximum absolute scalar comparison
errors are 7.719736519362641e-6 for operation and 5.01831018517862e-7 for CLICK
target, under absolute 1e-4 plus relative 1e-4 tolerances. W0 retains raw FP32
SHA-256 `d408d94a0ab2915fa85c511dd9b8d62af49903c4e4ca28e2e2d5aff9ef565e0f`.
The audit verifies the final artifacts and metrics; it does not replay the
10,000 optimizer updates.

Both runs use five CPUs, 8 GiB and zero swap, with the original 120-second outer
limit and stricter 113-second parent command deadline. The fit takes
34.00891153997509 seconds wall time and 32.3656509059947 seconds in its child.
The independent audit takes 4.00342045398429 seconds wall time and
2.5175326610333286 seconds in its child. Independent checks reauthenticate 757
file identities, including 53 owner and 48 leaf references for the fit, and 38
owner and 74 leaf references for the audit. Both original scoped closures and
current unit/cgroup absence are verified. These checks cover the authenticated
owners and observed cohorts; they make no universal containment claim.

Two earlier attempts remain preserved. The first fit stops before optimization
because a guard incorrectly assumes native instruction and criterion values
must be strings. Its replacement accepts the original finite JSON values
without rewriting serialization, reordering or pruning choices. Seven wire
controls pass, and all 34 actual schemas retain their 136 head mappings. The
first posthoc audit stops before NumPy because an authenticated empty Root log
is registered after inventory validation. The replacement fixes that ordering;
a fresh process validates all 74 inventory entries before retry. Seven audit
controls and eleven launcher controls pass. The successful fit artifacts are
unchanged by the audit correction.

At this fit milestone, the two-head GPU backend passes 25 invented controls,
while the planned 37-call native GPU qualifier remains unexecuted. Those
controls provide no CUDA numerical proof. Subsequent native composition and
browser measurements are in the separate follow-up report linked above.

The historical 119-row operation-only candidate and its matched NF4 baseline
each failed the first browser case, 0/1. That is a
[different previous profile](decision3-checked-control-sgd10000-gpu-browser-results.md),
not a result for this two-head candidate. At this fit milestone, the new
candidate's browser score, full ten-case score, JevBench coherenceMINI score
and Benchmark Heaven score remain null. The two JevBench suites remain separate. No held-out accuracy,
generalization, model qualification or competitor win is established.

The accepted fit-integrity audit has SHA-256
`34cff645e60a73505ce1a4dd6bca7fcf1d7dbc0719982e11cc16ac7feaa9f084`;
the independent dataset acceptance has SHA-256
`67dbaf49a56e32410db77304f8185cb0b4b1257eab511f3f7bb08c66858fe747`.
The [machine-readable report](decision3-goal-routed-two-head-fit-results.json)
also records the selected capability and fit-summary hashes. Only aggregates
and hashes are public; raw inputs, labels, predictions and weights remain
private.
