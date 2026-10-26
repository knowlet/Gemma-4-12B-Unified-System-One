# Decision3: accepted two-arm operation273 / CLICK65 training

Both genuine GX10 CPU5 fits and their separate owned FP64 replays are accepted. The equal and balanced arms use identical readonly TRAIN input bytes, the preserved operation225 / CLICK47 prefix and new48 / 18 rows, common frozen encoder and W0, zero-start FP32 deltas, 10,000 steps per head, learning rate 0.003, L2 0.0001 and seed 20261023. Effective minibatches are 64 for both heads. Each fit took 43.01 seconds; each independent numerical replay took 10.01 seconds.

The weighting change improves SELECT while reducing CLICK and total operation correctness. **It establishes a TRAIN tradeoff, with browser performance still unmeasured for these two arms.**

| Operation TRAIN population | Equal correct / rows | Balanced correct / rows |
| --- | ---: | ---: |
| All | 233 / 273 | 224 / 273 |
| Original119 | 105 / 119 | 105 / 119 |
| Goal-routed34 | 29 / 34 | 29 / 34 |
| Navigation72 | 65 / 72 | 64 / 72 |
| Scroll-state48 | 34 / 48 | 26 / 48 |
| New120 CLICK teachers | 50 / 51 | 33 / 51 |
| New120 TYPE_TEXT teachers | 12 / 18 | 13 / 18 |
| New120 SELECT teachers | 4 / 15 | 11 / 15 |
| New120 DONE teachers | 33 / 36 | 33 / 36 |

Both CLICK-target heads achieve 65/65 on these TRAIN rows and produce identical parameter file bytes. Operation classification remains the visible bottleneck in this comparison. Unweighted full operation cross entropy is 0.5884180806 for equal and 0.6229853443 for balanced; CLICK-target cross entropy is 0.02742843654 for both.

Literal state subgroups expose the remaining difficulty even within TRAIN:

| New scroll-state48 subgroup | Equal correct / rows | Balanced correct / rows |
| --- | ---: | ---: |
| CLICK, no checked radio, no history | 5 / 6 | 3 / 6 |
| CLICK, checked radio and history | 12 / 12 | 2 / 12 |
| TYPE_TEXT, checked radio, no history | 3 / 3 | 3 / 3 |
| TYPE_TEXT, checked radio and history | 1 / 6 | 1 / 6 |
| SELECT, checked radio, no history | 1 / 3 | 3 / 3 |
| SELECT, checked radio and history | 1 / 6 | 4 / 6 |
| DONE, checked radio and history | 11 / 12 | 10 / 12 |

Checked means any offered radio has a literal checked flag. It does not establish goal completion. Original119 literal checked/history coverage remains unknown. The persistent TYPE_TEXT-with-history errors motivate checking feature conditioning and exact feature conflicts before attributing the failure to a single cause.

The balanced arm preserves original153 operation weights at one. New120 teacher counts CLICK51 / TYPE_TEXT18 / SELECT15 / DONE36 receive 10/17 / 5/3 / 2 / 5/6 respectively. CLICK65 weights remain one. The objective averages weighted soft-target cross entropy by batch row count and adds unscaled half-L2. The balanced FP32 weights sum to 272.99999982118607 when evaluated in FP64.

Independent replay rebuilds every original225/47 input byte and mapping, all192 newly captured head mappings, all255 logits for each head, legal masks, zero/final losses, weighted and unweighted gradients, correctness and margins. Discrete results match exactly. The largest FP32-to-FP64 scalar discrepancy is 7.24e-6 under unchanged absolute/relative tolerances of 1e-4. Each acceptance reauthenticates 1,205 file identities. It verifies final-weight numerics; the 10,000 SGD updates are bound by source and actual receipts rather than independently rerun.

The [initial source-selection failure](decision3-scroll-state-two-arm-fit-progress.md) is resolved by a fresh campaign. The exact executing original119 loader is directly selected. Both new fits verify all13 actually imported artifact Python sources before publishing parameters; their parameter files exactly reproduce the initial runs. Complete leaf inventories and original scoped process closure remain verified. Fit parent sources total52 and audit parent sources total47 under the unchanged limit64. Fresh local controls passed: 5 loader-entry, 22 fit-launch/import and 16 audit-launch/import cases. Prior attempts remain preserved.

GPU policy composition, fresh browser trajectories, full10, JevBench coherence and Benchmark Heaven scores remain pending for both new arms. DEV has previously informed model selection; DEV and sealed evaluation labels remain outside the optimizers. No competitor win or generalization claim follows from these TRAIN results.

[Aggregate JSON and immutable fingerprints](decision3-scroll-state-two-arm-fit-results.json) · [Collection coverage](decision3-scroll-state-native-collection-results.md) · [Earlier TRAIN diagnosis](decision3-train-subgroup-results.md)
