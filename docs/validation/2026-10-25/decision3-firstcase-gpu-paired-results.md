# Decision3 GX10 first-case GPU paired diagnostic

2026-10-25. The candidate and matched NF4 baseline both completed the fixed
first browser case and both failed its original task judge: **0/1 each**.
This is one exposed diagnostic pair. The full ten-case score remains null;
it establishes no browser qualification or competitor win.

| Arm | Task result | Policy calls | NF4 text-helper calls | Actions | Terminal behavior |
| --- | ---: | ---: | ---: | ---: | --- |
| Decision3 Edge + 109-row operation delta | 0/1 | 4 | 0 | 4 | Blocked after three unchanged-page transitions; no error or premature DONE |
| Matched NF4 baseline | 0/1 | 6 | 1 | 4 | DONE before the original goal was satisfied |

Both arms were infrastructure-complete. The candidate ran first, followed by
the baseline, using the same resident NF4 helper. There was one NF4 load and
one Decision3 load; Decision3 used a BF16 encoder and FP32 readout. Seed 20261014, one
repeat, the 16-step limit, native 20-second and HTTP 25-second limits, and the
original V7 observation helper were unchanged. The V7 helper SHA256 was
`d9f4236d1d46c9472c3d6c1551342639ecb870e6c0dd419246927645410ea868`.

The original launch exited 0 after 243.1005604180391 seconds, including startup,
both arms and cleanup. This is not a per-call latency or SLO measurement.
The independent audit reauthenticated 860 selected files and the 189-entry
service source inventory: 183 nonempty sources and six empty Python sources.
The empty-source branch reuses the unchanged public
[`SourcePins.pin`](../../../scripts/jev_build_public_bf16_runtime_profile.py),
covered by [seven invented regression controls](../../../tests/test_jev_empty_python_sources.py).

All four candidate HTTP requests returned 200. Their original client/server
request and response bytes matched the four distinct native witnesses. Across
144 legal probability positions, independent FP64 dot products and softmax
replay passed the original tolerance, absolute `1e-6` plus relative `1e-5`;
the maximum absolute error was `1.4973786233163366e-7`. The W0 and trained delta
raw byte hashes matched their selected artifacts. Every non-operation raw answer
had identical JSON serialization in the W0 and composed branches.

For the same captured features, the operation argmax changed from W0's BLOCKED
to the composed candidate's CLICK in all four calls. This local readout replay
does not provide a separate Edge-only browser trajectory. The observed candidate
trajectory made four clicks. Independent recomputation of five observation
fingerprints gave page-change flags `[true, false, false, false]`: the first
click changed a radio control from unchecked to checked; the final three clicks
selected that already checked control again. TYPE_TEXT and SELECT were available
at all four steps. The original upstream rule stopped after three transitions
without page progress.

The actual worker identity and both original owned process scopes passed the
closure joins. The Decision3 worker's final allocator count was zero; this
does not establish whole-machine GPU zero or OS-level containment.
Accepted main audit SHA256:
`285439ce3bf070d407008ba0fe509c3cfccd367af582456a869f16e3c242229d`.
Supplemental state-progress audit SHA256:
`a2b60b2f6a1275a4ff9fd5159442b0eb2488f7d07c9ef3c29e7a1e986b326249`.

The 109-row candidate is exposed, with `dev_used_for_model_selection=true`.
The separate [Benchmark Heaven](decision3-native-heaven-gpu-results.md) and
[JevBench coherence](decision3-native-coherence-gpu-results.md) reports measure
the original model; their scores do not transfer to this candidate. Both
candidate benchmark scores remain null.

Next work is to collect separate, generic checked-control and state-progress
examples. This evaluation case and its goal must not supply training labels.
Keep the API action choices and original judge unchanged, and evaluate the
resulting candidate separately. See the [machine-readable aggregates](decision3-firstcase-gpu-paired-results.json).
