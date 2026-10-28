# Torch 2.14.1 browser candidate and benchmark checkpoint

The candidate completed all twelve frozen external browser attempts: **6/12
verified**, with 3/6 for each profile. All six follow-link attempts pass. All six
search attempts type the model-generated `Graph theory` value successfully,
then explicitly reject a 57-candidate autocomplete target head. No candidate is
truncated or filtered. This run uses the default rejection policy.

| Profile | Search | Follow link | Total |
| --- | ---: | ---: | ---: |
| causal_multislot | 0/3 | 3/3 | 3/6 |
| independent | 0/3 | 3/3 | 3/6 |

The original 180-second policy-time budget and twelve-cycle cap remain in
force. These are two public tasks with three descriptive repeats, not a sealed
population or evidence of general browser superiority. The subsequent
[hierarchy diagnostic](../mps-hierarchy-browser/README.md) uses this consumed
search task and is reported separately.

The completed causal public231 run has **231 valid / 197 correct** decisions
(85.2814%), Brier 0.2194523193 and ECE 0.0478199723. Its 233 declared raw/summary
files pass their manifest hashes and an independent offline official-scorer
audit. The corresponding original Torch 2.8 causal run also has 197/231 correct
on the same 231 logical request hashes; one prediction changes between two
incorrect answers. Equal accuracy does not establish probability parity or
complete regression acceptance. The separate
[runtime control](../mps-torch214/README.md) still fails its unchanged 0.01
probability-drift gate. Public exposure, no calibration fitting, unknown local
cost and no official rank remain explicit.

## Process and provenance

The declared source commit is
`c783a7809f834ba25c1fa7045e5a28f83462d8b9`; all 68 declared local source-file
hashes match the archived execution bytes. These hashes identify this run even
when the current checkout has later changes. The original `identity.json`
retains `status: running` because the process did not finish its normal final
write; the separate terminal receipt is authoritative.

After the audited causal public run, the owned process was deliberately
checkpointed with SIGTERM and exited 143. Its absence was verified. This was
not an observed model OOM or a paused goal. At that point coherence had **94
cache entries and no accepted coherence score**; independent public231 and the
remaining suites had not run. The stop-request and termination inventories are
identical. Their frozen hashes/sizes are archived without reading the now-shared
mutable cache as a historical snapshot.

The isolated environment reports Torch 2.14.1, TorchVision 0.29.1,
Transformers 5.17.0, MPS BF16/SDPA, the same verified checkpoint manifest
`f38d7f50060eabcbed6f73d16404ad42eb6ba1bddc79b68052afc13cab8598eb`, and an
Apple M1 Max with 64 GiB memory. The project dependency pins remain unchanged.
The five recorded MPS environment overrides are unset. Post-call cache
maintenance is explicit and included in callback latency: 361 saved calls
record no maintenance fault. Post-trim driver snapshots span 24.576–26.042 GB;
these are snapshots, not peaks or proof of unlimited process lifetime.

The bounded benchmark supervisor at
`artifacts/ultrafast/20261028-torch214-benchmark-epochs-01` was **running** when
this report was captured, with at most 2,000 new calls per epoch and a 0.8
driver/recommended-memory checkpoint fraction. Its changing cache and outputs
are excluded here. No completed coherence or independent benchmark score is
claimed. See the [epoch reproduction protocol](../../../ultrafast-evaluation.md#bounded-benchmark-epochs).

## Archive verification

[report.json](report.json) separates browser, audited public accuracy,
checkpoint, partial coherence and pending resume states.
[torch214-browser-evidence.zip](torch214-browser-evidence.zip) is 3,671,836
bytes with 440 payload members plus its embedded member manifest. SHA-256:

```text
d51df680b58d14dd3f3398b5862cc0e8344772bc22ee22d901c2e53b726db21c
```

[archive-members.json](archive-members.json) binds every payload's size, CRC32
and SHA-256; the report separately binds the manifest. ZIP CRC, exact member set
and every member digest were verified. The archive includes all twelve raw
browser transport/CDP/snapshot/final-state traces and screenshots, executed
sources, original checkpoint/termination receipts, all causal public231 raw
responses, and the small pinned public dataset/scorer subset needed for offline
audit. The complete official-source inventory is retained.

Weights, symlinks, large tokenizer assets, redundant full benchmark repositories
and the mutable coherence cache are excluded. Large Playwright `trace.zip`
recordings remain local; their hashes and sizes are retained in
`omitted-large-artifacts.json` inside the archive. No CUDA or additional model
execution was performed while assembling this evidence.
