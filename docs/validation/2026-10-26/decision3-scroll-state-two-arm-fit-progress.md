# Decision3 scroll-state two-arm training progress

Update: the source-selection correction, fresh fits and independent FP64 acceptance are complete. See the [accepted two-arm results](decision3-scroll-state-two-arm-fit-results.md). The initial observations and failure record below remain preserved.

Two genuine GX10 CPU5 training runs completed, each fitting separate operation273 and teacher-active CLICK65 delta heads from zero. Equal weighting took 44.01 seconds; balanced weighting took 43.01 seconds. Both used the same immutable 225/47 prefix, new 48/18 rows, frozen encoder and W0, 10,000 steps per head, learning rate 0.003, L2 0.0001 and seed 20261023. Both heads now use effective minibatches of 64; the previous CLICK47 fit used 47.

**Numerical acceptance is pending.** The independent FP64 replay stopped at its final source-selection guard: one authenticated transitive original119 loader actually executes and also needs explicit parent selection. The guard remains enforced. The next attempt will select that exact source, preserve these outputs and rerun both training arms with fresh capabilities and owner outputs. These figures are the fit programs' TRAIN observations, with no accepted independent numerical output yet.

| Operation TRAIN population | Equal correct / rows | Balanced correct / rows |
| --- | ---: | ---: |
| All | 233 / 273 | 224 / 273 |
| Original119 | 105 / 119 | 105 / 119 |
| Goal-routed34 | 29 / 34 | 29 / 34 |
| Navigation72 | 65 / 72 | 64 / 72 |
| New scroll-state48 | 34 / 48 | 26 / 48 |
| New120 CLICK teachers | 50 / 51 | 33 / 51 |
| New120 TYPE_TEXT teachers | 12 / 18 | 13 / 18 |
| New120 SELECT teachers | 4 / 15 | 11 / 15 |
| New120 DONE teachers | 33 / 36 | 33 / 36 |

The observed weighting tradeoff improves SELECT while reducing CLICK correctness and aggregate correctness. It establishes no winner. Both CLICK-target fits report 65/65 and produce identical delta file bytes, as expected from identical data, weights and seeds. Operation recognition and choosing a target are separate tasks.

The balanced arm keeps original153 weights at one. Its new120 teacher counts are CLICK51, TYPE_TEXT18, SELECT15 and DONE36, with weights 10/17, 5/3, 2 and 5/6 respectively. CLICK65 weights remain one in both arms. Weighted cross entropy divides by batch row count; the L2 term stays unscaled. No weight-sum normalization or encoder training is added. FP32 representation makes the full balanced weight sum, evaluated in FP64, 272.99999982118607.

The source graph retains all needed evidence. The initial metadata projection exceeded the parent bound; the reviewed explicit direct/needed graph allowed the actual fit configuration to retain 135 leaf references with 51 parent references under the unchanged limit of 64. The final imported-source guard then identified the additional executing loader described above. Root ran 64 invented source-contract and launcher controls; these controls are separate from the pending genuine numerical acceptance.

There is no new browser success rate, full10 score, JevBench coherence score or Benchmark Heaven score for either arm. All remain null. DEV has previously been used for model selection; neither DEV nor sealed evaluation labels entered these optimizers. Original119 literal checked/history coverage remains unknown.

Aggregate data and immutable source fingerprints: [JSON report](decision3-scroll-state-two-arm-fit-progress.json). Previous data coverage and diagnosis: [scroll-state collection](decision3-scroll-state-native-collection-results.md), [TRAIN subgroups](decision3-train-subgroup-results.md).
