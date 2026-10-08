JevBench public screening — 20261008-public-primary-six

Released public tasks and labels; potentially exposed to model training or prior tuning. Research screening, not a sealed evaluation.

Official rank: not submitted. Sealed evaluation: not run.

| Model | Source run | Audit | Valid / planned | Correct | Accuracy | Coherence | Coherence audit | Setup seconds |
|---|---|---|---:|---:|---:|---:|---|---:|
| s1-bf16 | 20261008-public-01 | verified | 231/231 | 197 | 85.28% | 40.50% | verified_raw_replay | 264.9239 |
| onejev-4b | 20261008-public-02 | verified | 231/231 | 171 | 74.03% | 72.53% | verified_raw_replay | 51.7704 |
| decider-2b | 20261008-public-02 | verified | 231/231 | 175 | 75.76% | 72.89% | verified_raw_replay | 25.6571 |
| s1-independent | 20261008-public-02 | verified | 231/231 | 197 | 85.28% | 71.96% | verified_raw_replay | 54.0193 |
| s1-compiled | 20261008-public-02 | verified | 231/231 | 197 | 85.28% | unknown | expected_not_run | 88.8155 |
| jev-omni | 20261008-public-03 | verified | 231/231 | 203 | 87.88% | 74.55% | verified_raw_replay | 55.9897 |

Accuracy intervals use paired source groups; failures remain incorrect decisions in the complete population. Coherence measures stability. Its audit status distinguishes raw-answer replay, saved-only metrics and expected omissions. The pinned scorer supplies the overall interval, but no dimension intervals.

- onejev-4b minus s1-bf16: -11.26 percentage points; descriptive 95% CI [-16.23, -6.58].
- decider-2b minus s1-bf16: -9.52 percentage points; descriptive 95% CI [-15.18, -4.10].
- s1-independent minus s1-bf16: +0.00 percentage points; descriptive 95% CI [+0.00, +0.00].
  Recorded hardware: NVIDIA A100-SXM4-80GB versus NVIDIA A100 80GB PCIe; observed quality only, with hardware/configuration confounding.
- s1-compiled minus s1-bf16: +0.00 percentage points; descriptive 95% CI [-1.74, +1.73].
- jev-omni minus s1-bf16: +2.60 percentage points; descriptive 95% CI [-1.29, +6.72].
  Recorded hardware: NVIDIA A100-SXM4-80GB versus NVIDIA A100 80GB PCIe; observed quality only, with hardware/configuration confounding.
- decider-2b minus onejev-4b: +1.73 percentage points; descriptive 95% CI [-3.38, +6.84].
- s1-independent minus onejev-4b: +11.26 percentage points; descriptive 95% CI [+6.58, +16.23].
  Recorded hardware: NVIDIA A100-SXM4-80GB versus NVIDIA A100 80GB PCIe; observed quality only, with hardware/configuration confounding.
- s1-compiled minus onejev-4b: +11.26 percentage points; descriptive 95% CI [+6.84, +16.09].
- jev-omni minus onejev-4b: +13.85 percentage points; descriptive 95% CI [+8.58, +19.55].
  Recorded hardware: NVIDIA A100-SXM4-80GB versus NVIDIA A100 80GB PCIe; observed quality only, with hardware/configuration confounding.
- s1-independent minus decider-2b: +9.52 percentage points; descriptive 95% CI [+4.10, +15.18].
  Recorded hardware: NVIDIA A100-SXM4-80GB versus NVIDIA A100 80GB PCIe; observed quality only, with hardware/configuration confounding.
- s1-compiled minus decider-2b: +9.52 percentage points; descriptive 95% CI [+4.24, +14.88].
- jev-omni minus decider-2b: +12.12 percentage points; descriptive 95% CI [+6.19, +18.22].
  Recorded hardware: NVIDIA A100-SXM4-80GB versus NVIDIA A100 80GB PCIe; observed quality only, with hardware/configuration confounding.
- s1-compiled minus s1-independent: +0.00 percentage points; descriptive 95% CI [-1.74, +1.73].
  Recorded hardware: NVIDIA A100 80GB PCIe versus NVIDIA A100-SXM4-80GB; observed quality only, with hardware/configuration confounding.
- jev-omni minus s1-independent: +2.60 percentage points; descriptive 95% CI [-1.29, +6.72].
- jev-omni minus s1-compiled: +2.60 percentage points; descriptive 95% CI [-1.26, +6.55].
  Recorded hardware: NVIDIA A100-SXM4-80GB versus NVIDIA A100 80GB PCIe; observed quality only, with hardware/configuration confounding.

s1-bf16 memory scope: unknown; recorded cell peaks are not a controlled per-stage comparison. Per-decision cost: unknown.

Candidate-cache equivalence: passed.
Warmed ABBA median cached/uncached ratio: 1.0052; 95% CI [0.998159484778894, 1.0985996000239695]. Time reduction supported on the measured sample: False.
eight predeclared tasks; warmed interleaved ABBA samples, three repeats per block; one model/container; descriptive task-group bootstrap, not an independent deployment replication.

onejev-4b memory scope: Entire loaded-model cell: public231, cache experiment when applicable, and coherence-mini; not a controlled per-stage memory comparison.. Per-decision cost: unknown.

decider-2b memory scope: Entire loaded-model cell: public231, cache experiment when applicable, and coherence-mini; not a controlled per-stage memory comparison.. Per-decision cost: unknown.

s1-independent memory scope: Entire loaded-model cell: public231, cache experiment when applicable, and coherence-mini; not a controlled per-stage memory comparison.. Per-decision cost: unknown.

s1-compiled memory scope: Entire loaded-model cell: public231, cache experiment when applicable, and coherence-mini; not a controlled per-stage memory comparison.. Per-decision cost: unknown.

jev-omni memory scope: Entire loaded-model cell: public231, cache experiment/native regressions when applicable, and requested coherence-mini; not a controlled per-stage memory comparison.. Per-decision cost: unknown.
