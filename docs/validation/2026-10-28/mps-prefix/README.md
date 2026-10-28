# MPS request-local prefix experiment

The full 12B BF16/SDPA experiment completed on Apple M1 Max. The candidate is
**rejected for adoption**: two of eleven cases exceed the predeclared maximum
absolute probability difference of 0.01. All argmax decisions agree, but that
does not override the probability gate. Model defaults remain unchanged.

| Input | Baseline / prefix median (s) | Maximum probability difference | Gate |
| --- | ---: | ---: | --- |
| Browser development snapshot 1 | 24.658 / 16.134 | 0.008141 | pass |
| Browser development snapshot 2 | 24.580 / 16.472 | 0.050089 | fail |
| Browser development snapshot 3 | 25.424 / 16.663 | 0.104847 | fail |
| Billing text | 0.916 / 1.120 | 0.001743 | pass |
| Technical text | 1.053 / 1.223 | 0.001476 | pass |
| Single sentiment question | 0.531 / 0.527 | 0 | pass |
| Synthetic image | 4.464 / 3.099 | 0.001972 | pass |
| Synthetic audio | 1.094 / 1.219 | <0.000001 | pass |
| Synthetic mixed input | 4.604 / 3.163 | 0.001893 | pass |
| Long state | 10.002 / 5.994 | 0.002054 | pass |
| Sixteen questions | 6.549 / 4.700 | 0.002412 | pass |

Each case has separate warmups followed by two alternating ABBA/BAAB blocks:
four measured samples per profile. Every candidate answer is compared to every
baseline sample. Baseline probabilities repeat exactly on all eleven cases.
Measurements synchronize MPS around inference and use the same loaded weights,
temperature, complete Unified backbone and native modality inputs. Only saved
development browser snapshots guide the experiment; reserved outcomes do not.

Speed depends on the workload. Long state and image inputs improve by about
30–40%, while short text and audio can become slower. The larger browser drift
has not yet been attributed to cache semantics or BF16 execution shape effects.
These timings are a bounded runtime experiment, not task-success, calibration,
two-suite regression or superiority evidence. Cost remains unknown. No CUDA
execution was performed.

Post-call driver-memory snapshots range from 26.365 to 33.279 GB (decimal), while
active tensor allocation stays at 23.919 GB. These are snapshots, not peaks, and
do not establish that a long-running process avoids allocator growth. All MPS
environment overrides were unset.

[report.json](report.json) contains all eleven summaries and identities.
[prefix-evidence.zip](prefix-evidence.zip) contains every raw measured/warmup
response, exact input request, before-execution source snapshot, checkpoint
member hashes and queue receipts. It excludes checkpoint weight bytes. Archive
CRC and every member digest were checked; the regression test replays all
summaries from raw responses without inference.

Reproduce with a fresh directory, after other local MPS processes have ended:

```bash
uv run --no-sync python scripts/benchmark_mps_prefix.py \
  --trace-root artifacts/ultrafast/20261028-native-01 \
  --model artifacts/release/checkpoint \
  --output artifacts/mps-prefix/your-unique-run \
  --repeat 2 --max-probability-delta 0.01
```

The source browser trace is also preserved in the earlier
[browser archive](../mps-browser/browser-evidence.zip). Its logical requests are
embedded directly in this experiment's raw report.
