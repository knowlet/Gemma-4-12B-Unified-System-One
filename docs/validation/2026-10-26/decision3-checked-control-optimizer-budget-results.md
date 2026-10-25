# Checked-control optimizer budget diagnosis

Follow-up: [fresh native GPU calls confirm TRAIN 10/10, while the SGD10000 browser
pair remains 0/1](decision3-checked-control-sgd10000-gpu-browser-results.md).
The measurements below retain their original cached-feature scope.

2026-10-26. Increasing the zero-start SGD budget from 512 to 10,000 steps teaches
all ten added checked-control training examples in independent FP64 readout
replay. This is a training-set result. The 10,000-step candidate still requires
fresh native GPU qualification and browser evaluation.

| SGD steps | CPU captures + FP64 readout | GPU captures + FP64 readout | Correct TYPE_TEXT examples, CPU / GPU |
| --- | ---: | ---: | ---: |
| 512 | 6/10 | 6/10 | 0/2 / 0/2 |
| 2,048 | 8/10 | 7/10 | 0/2 / 0/2 |
| 10,000 | 10/10 | 10/10 | 2/2 / 2/2 |

Each column reuses the same ten previously captured, complete **training** wires
and their independently accepted labels. The new readouts are recomputed in
stdlib FP64 using the captured features and W0 with each trained delta. These
are cached-feature counterfactuals, not fresh GPU model calls or browser
trajectories. No additional encoder calls were made for this table.

The CPU/GPU capture pairs are not bitwise identical. At 2,048 steps, one CLICK
decision changes between them; at 10,000 steps, both sets select all ten correct
operations. Independent cross entropy on the ten CPU captures falls from
1.338620842176933 to 1.0682365176041226 to 0.566605332046927. The corresponding
GPU-capture values are 1.3441959411752968, 1.0851793799057146 and
0.5996668506915753. No exactly identical CPU feature vectors carried conflicting
labels in these ten examples.

Both new fits use the original 119 frozen training rows, W0, local criterion
orders, legal masks and unchanged numerical core. Learning rate 0.003, L2 0.0001,
seed 20261023 and minibatch 64 remain fixed. Each fit starts from zero and opens
no prior trained delta. The only optimizer change is the number of steps. No
evaluation-case goal or label enters fitting. The data's original 512-step
capability remains unchanged; separate capabilities authorize the new budgets.

| New owned CPU fit | Whole original launch | Before final durable summary write | Independently checked delta norm |
| --- | ---: | ---: | ---: |
| 2,048 steps | 7.01041297899792 s | 5.0531295200344175 s | 1.8959907750087122 |
| 10,000 steps | 21.012695601966698 s | 18.997982117987704 s | 4.646902266142284 |

These timings cover cached-feature fitting, not encoder inference. Both runs
retained CPUs 15–19, a 500% CPU quota, 4 GiB memory, zero swap and the original
120-second whole-work limit. Independent stdlib audits accepted the original
terminal and scoped closure, all 51 selected sources, complete ordered 119-row
mapping, W0/delta bytes, tensor norms, and exact zero unused delta rows. Neither
run changed ComfyUI service state.

The complete 119-row CE, accuracy, margins and gradients emitted by the fitter
remain self-reported observations awaiting independent full-array replay. The
independent accuracy and CE in the first table cover only the ten added training
examples. Neither check establishes held-out generalization, browser usability,
native GPU qualification, OS containment or a competitor win.

The [512-step browser pair](decision3-checked-control-gpu-browser-results.md)
scored 0/1 for both candidate and matched NF4 baseline. That result belongs to
the 512-step artifact. Browser scores for the two new artifacts, their full
ten-case scores, Benchmark Heaven scores and JevBench coherence scores are all
null. Next, qualify the 10,000-step delta through genuine native GPU calls and
repeat the unchanged first browser pair.

See the [machine-readable aggregates](decision3-checked-control-optimizer-budget-results.json).
Raw features, labels, requests and model artifacts remain private.
