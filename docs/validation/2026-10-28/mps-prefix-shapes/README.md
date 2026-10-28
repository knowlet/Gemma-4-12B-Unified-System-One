# Torch 2.14.1 prefix shape control

This nine-call diagnostic uses only the previously consumed development task
`dev-design-free`, observation 3. A uses complete padded independent prompts;
B runs each complete prompt as a singleton; D prefills the native request prefix
and evaluates cached independent suffixes. It uses the same MPS/BF16/SDPA model,
checkpoint and temperature. No task golds or final outcomes enter the comparison.

| Comparison | Maximum probability difference | Same argmax for all heads | Original 0.01 gate |
| --- | ---: | --- | --- |
| A vs B | 0.014570 | yes | fail |
| A vs D | 0.063042 | yes | fail |
| B vs D | 0.048472 | yes | fail |

Each variant repeats exactly (maximum within-variant probability difference 0).
All three heads choose the same answer across A, B and D. That agreement does
not override the frozen probability gate. Prefix reuse remains experimental and
has not been connected to the default backend.

The synchronized, instrumented measured medians are A 20.981 s, B 19.658 s and
D 13.892 s. Selected-slot GPU clones are included; CPU vector exports are
excluded. The schedule retains three separately identified warmup calls, then
A, B, D, D, B, A, with every attempt preserved. These two measurements per
variant support this numerical diagnostic only; they do not establish browser
latency or general inference performance. Memory readings are snapshots.

A versus B shows that batching/padding complete prompts already changes
probabilities. D differs further from both A and B, so these observations cannot
attribute the result mainly to batching or prove a cache implementation error.
The complete native processor, all original question content, position/mask
construction, checkpoint and raw selected hidden vectors remain auditable.
[Saved-vector replay](vector-audit.json) verifies all 65 source hashes and finite
vectors. A and B have identical vectors on the longest unpadded head while the
two padded heads differ. D also changes that longest head relative to B. This
keeps padding effects distinct from the unresolved split/cache/suffix effects.
The vector audit reports raw probabilities; the main gate first normalizes the
responses using the existing contract.

[report.json](report.json) preserves the separated numerical/timing evidence.
[shape-control-evidence.zip](shape-control-evidence.zip) contains all original
samples, requests, selected vectors, queue/launch identity and frozen source
bytes. Archive CRC and all member digests passed; the archive hash is recorded in
the report. Cost is unknown and no CUDA job ran.

```bash
artifacts/mps-next-env/bin/python scripts/diagnose_mps_prefix_shapes.py \
  --trace-root artifacts/ultrafast/20261028-native-01 \
  --development-task dev-design-free --decision-index 3 \
  --output artifacts/mps-prefix/your-unique-shape-control \
  --wait-for-pid CAPTURED_PREDECESSOR_PID
```

The isolated runtime setup is recorded in the
[Torch 2.14.1 control](../mps-torch214/README.md). Run sequentially after earlier
accelerator processes finish. This is a consumed development case, not fresh
held-out acceptance or an official benchmark submission.
