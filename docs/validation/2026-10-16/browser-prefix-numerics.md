# PR #9: CPU BF16 prefix numerical divergence and missing historical wire

## Reproduced baseline

The original failure belongs to head `7e89f72eddcc8a1ca850325a87b908aa657103c8`.
The original PR CI used GitHub's merge checkout; the dedicated push diagnostic
checks out `github.sha` explicitly. No model weights, Modal job, or paid model
API is involved: this is the existing seed-94 tiny genuine Gemma fixture.

Exact diagnostic commit: `9729d0711aee7d6cb0dfb57758348795fcb9aec1`.
Run: https://github.com/knowlet/Gemma-4-12B-Unified-System-One/actions/runs/38858163214
Artifact: `12028675048`, `prefix-numerics-9729d0711aee7d6cb0dfb57758348795fcb9aec1`.
Downloaded ZIP SHA-256: `08e8a09e6a00741f32d244a24cd07546c8d6e8ade5aebee9688168ae3a47e701`.
Fixture SHA-256: `624aa6eb220e9a81574ca855874e137a21e962ed1bcf654ccb747bc9efaa4b6c`.
Environment: macOS 14.8.9 arm64, Python 3.12.10, torch 2.8.0,
transformers 5.17.0, three CPU threads.

For the 52-candidate goal-first row, prefix length is 1371, full length 3046,
and suffix length 1675. Sliding window 64 retains 63 prefix keys, discarding
1308 already-masked key columns. The cached suffix's key axis is therefore
1738 columns instead of the full forward's 3046 columns.

The absolute positions, effective masks, and first-layer Q/K/V on corresponding
retained columns agree exactly in BF16. Prefix hidden states also agree.
The first decoder-layer suffix output differs by up to 0.0234375, and the
last-token final hidden differs by 0.0078125 in seven of sixteen elements.
Candidate-only projection and the full vocabulary head are exactly equal when
applied to the same hidden; moving projection or softcap to FP32 is not the fix.

| Native BF16 readout | item-37 | item-43 | Winner |
|---|---:|---:|---|
| Full context | 0.154296875 | 0.154296875 | item-37 |
| Cropped-key prefix | 0.154296875 | 0.1552734375 | item-43 |

The full forward has a tie. The cropped-key execution moves item-43 up one
BF16 representable step. Passing a probability tolerance cannot validate the
changed action. Winner equality remains a separate, mandatory assertion.

## Isolating the cause

BF16 eager attention makes all four decoder layers and the final hidden
identical between full and prefix forwards on these rows. More importantly,
an SDPA ablation keeps the original attention implementation, projection,
softcap, cache contents, and winner rule, changing only the absolute key-axis
layout of the suffix's temporary attention tensors. It zero-pads only evicted
columns that are masked for every suffix query, and uses matching zero-offset
mask geometry. All decoder suffix outputs and final hidden states then agree
exactly, including the original item-37 winner. The original prefix test module
passes under that process-local ablation, then fails its winner test again when
the patch is removed in a separate process.

Ablation commit: `32c1cdc81551359f30d101c6d70bf6c068c5ff8f`.
Run: https://github.com/knowlet/Gemma-4-12B-Unified-System-One/actions/runs/38858791103
Artifact: `12028586966`, ZIP SHA-256
`5e039cc01bb91593a052b6094a1a41f9b39b8d399ccff5b3fca1557af74bf09e`.
It contains both the baseline and aligned reports and the diagnostic source.
These GitHub artifacts have 30-day retention; retain their original downloaded
ZIPs if long-term reproduction of this particular observation is required.

The cache eviction is mathematically valid, but low-precision SDPA is not
numerically invariant to removing masked key columns. This is not evidence of
an off-by-one position, shared-KV aliasing, or incorrect candidate selection.
The exact low-level kernel instruction or tile responsible is not established
by this experiment; the key-axis ablation establishes the triggering geometry.

## Correction and scope

`browser_prefix_cache.py` applies the alignment only to independent,
non-offloaded native cache forks used with CPU BF16 SDPA. It uses native layers
and their normal updates, but returns transient keys/values on the absolute
key axis and supplies matching mask sizes. Stored sliding KV remains a window
tail; Gemma's shared-KV consumers receive the same aligned temporary tensors.
There are no global monkeypatches, attention-backend changes, rerun-based winner
substitution, or changes to `project_candidates` and its native softcap.

The branch ledger records key layout, masked columns, and temporary returned
KV bytes. There are still one prefill plus the original suffix forwards.
Padding increases temporary attention work/memory: input-token savings alone
are not a latency or FLOP speedup claim. FP32, eager, CUDA, MPS, custom caches,
and offloaded caches are not changed by this workaround.

The unchanged original acceptance is supplemented with mask/window boundary,
cache independence/storage, actual BF16 SDPA output, and 52-candidate first-chunk
regressions for suffix batch sizes one and two. Current implementation results
must be read from the exact-commit CI, not inferred from the successful ablation.
This is not qualification of trained 12B weights or arbitrary devices/prompts.

## Historical 58-target input: still blocked

The original test remains unchanged and failing when this file is absent:

`artifacts/jev-browser/20261015-public-native-training-start-readiness-01/postgres18-limit-offset/native-original-wire.json`

Recovery checks in this session did not recover its original bytes:

- The exact file path has no commits on the original head's ancestry.
- The readiness report was added in `cdf097d58087e89fa8fee30f1a23b2d2021432a4`;
  its CI run `38747188482` has no retained artifacts.
- Original-head run `38840550086` has contract distribution artifacts only.
  The downloaded 3.11 ZIP matches GitHub's SHA-256
  `cbb8fafe18443b0243877939d1d695bb23e8782fdd251b259a5bb719269d5ec4`.
  Its source distribution and three embedded older evidence archives do not
  contain the named wire or readiness capture. The relevant source blob matches
  the actual PR-head blob, rather than being assumed equal from the merge build.
- Library searches for the exact wire, capture directory, and audit digest
  returned no matching raw capture. This is not an exhaustive proof of absence.
- No usable SSH/development-environment connection was available in this session.
  A reported `/home/albert_hsieh/Gemma-4-12B-Unified-System-One` path is not proof
  that its files were accessed. Repository-wide artifact enumeration was not
  supported by the connected endpoint; only the specified runs were inspected.

The tracked report pins receipt SHA-256
`ed1fc890233016791cae0c29df11fb457c10673dea07f287b8eb9495fd1ac379`
and independent-audit SHA-256
`8b7003b15eb81c1d6514f193f81a16073449a3a25d6edd6565b7fb029f1cfd93`,
but not the individual wire digest. Recovering the exact receipt/audit bytes
might provide the original per-file integrity chain; those hashes alone do not
prove a candidate wire's identity. A new capture or generated 58-target wire
cannot satisfy this historical regression. Keep the PR draft and this evidence
gate failed until original bytes and independently reviewable integrity evidence
are supplied. No skip, xfail, synthesized wire, or historical receipt rewrite is
part of this correction.
