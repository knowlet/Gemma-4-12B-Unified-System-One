# Jev-Omni matched benchmark — October 7, 2026

`akhilaaa3/Jev-Omni` revision `5addda86ddee081a68fb067477ea100c221b8917`
completed the original 128 BoolQ cases, all 52 native media cases and all six
128-request HTTP load cells on an NVIDIA A100-SXM4-80GB.

| Observation | Measured result |
| --- | ---: |
| BoolQ | 112/128 = 87.50% |
| MNIST | 28/32 = 87.50% |
| FSDD | 2/20 = 10.00% |
| Combined native media | 30/52 = 57.6923% |
| BoolQ serial mean / p99 | 81.295 / 111.939 ms |
| Media serial mean / p99 | 100.700 / 114.712 ms |
| Closed-loop C1 / C16 HTTP p99 | 133.790 / 1,504.210 ms |
| Registered tensor footprint | 23.919463522 GB |
| Full-workload CUDA peak allocated / reserved | 24.384246272 / 24.582815744 GB |
| Observed device-wide usage | 25.466568704 GB |
| Process-lifetime peak RSS | 29.843550208 GB |
| Successful HTTP requests | 768/768 |
| HTTP requests missing the 1,000 ms SLO | 237/768 |

GB is decimal. CUDA peaks include resident weights and the measured workload;
peak allocation alone is not a minimum GPU-capacity claim. Device-wide usage
is an instantaneous observation, not a per-process peak. Serial quality timings
exclude setup and warmup; load timings include queueing over loopback HTTP in
one serialized model process, without WAN latency, batching or result caching.
The [full comparison](../../comparison.md) retains each load shape separately.

Against the historical Gemma-base BF16 result of 111/128, Jev-Omni's BoolQ
difference is **+0.78125 percentage points**, with a source-group bootstrap
95% interval of **[−6.25, +7.8125] pp** (128 groups, 2,000 resamples,
seed 20260930). The interval includes zero and is unadjusted for multiple
comparisons, so it does not establish an accuracy advantage. These 128 BoolQ
examples and digit-recognition media subsets do not measure broad multimodal
quality. The low FSDD result remains visible alongside image accuracy.

## Recorded native runtime

The merged checkpoint's backbone is stored and loaded as BF16; its 256-slot
native decision head is stored in FP32 and runs under BF16 autocast. The
adapter uses the publisher's pinned prompt, hidden-state capture and normalized
head, with one native forward per question. It does not generate text or fit
anything on the benchmark data.

The executed stack was **PyTorch 2.10.0, torchvision 0.25.0 and Transformers
5.17.0**. Earlier comparison rows retain their own recorded stacks, including
PyTorch 2.8.0. Source, weights and processor share the same immutable checkpoint
revision; `jev_omni.py` SHA-256 is
`11d761b0b6cefc8aac29757f43af4b2b02af8f9b6c6dad19834c0b14538180f0`.

Two adapter boundary choices are recorded explicitly in telemetry:

- Softmax runs in float32 over the native candidate logits, avoiding the
  publisher helper's BF16 rounding and retaining a complete normalized
  probability distribution. Original labels are mapped by slot position.
- The already-normalized 16 kHz float32 audio is sent directly to the native
  processor. There is no temporary WAV/ffmpeg roundtrip, transcription or caption.

The adapter rejects overlong fully expanded inputs above 16,384 tokens, more
than 20 options, multiple media inputs and timestamped video frames. It accepts
one native image or up to 30 seconds of native audio. All 180 archived quality
requests were eligible. There are no CUDA graphs, compilation or persistent
prefix cache. These measured timings include preprocessing and differ from the
publisher's optimized H200 timing protocol.

Checkpoint acquisition and model/processor setup took **125.282 seconds**,
excluding Modal scheduling and container startup. Runner manifests time a
factory returning the already-loaded shared adapter; their `setup_ms` values
are not checkpoint loading times.

## Published evidence and replay

The [combined summary](comparison-summary.json) contains **19 selected model
configurations, 31 attempts and 15 paired comparisons**. All 12 source campaign
hashes from the [October 2 summary](../2026-10-02/comparison-summary.json) were
verified unchanged before collection. Historical receipts and scores were not
rewritten or reselected by accuracy.

The [source/hash index](jev-omni-evidence.json) records **18 files / 1,240,010 bytes**
of original Jev-Omni evidence. Every hash recorded by the receipt was verified
before copying; JSON and JSONL bytes were retained unchanged:

- [Campaign](jev-omni-campaign.json) and [model receipt](jev-omni-local/receipt.json).
- [BoolQ predictions](jev-omni-local/boolq/predictions.jsonl),
  [manifest](jev-omni-local/boolq/run_manifest.json),
  [request timings](jev-omni-local/boolq/requests.jsonl) and summary/error records.
- [Media predictions](jev-omni-local/media/predictions.jsonl),
  [manifest](jev-omni-local/media/run_manifest.json),
  [request timings](jev-omni-local/media/requests.jsonl) and summary/error records.
- [Load summary](jev-omni-local/load-summary.json), plus all six raw load files:
  [C1](jev-omni-local/closed_loop-c1-n128.jsonl),
  [C4](jev-omni-local/closed_loop-c4-n128.jsonl),
  [C16](jev-omni-local/closed_loop-c16-n128.jsonl),
  [C64](jev-omni-local/closed_loop-c64-n128.jsonl),
  [fixed](jev-omni-local/fixed-c16-n128.jsonl) and
  [Poisson](jev-omni-local/poisson-c16-n128.jsonl).

The existing [Gemma-base reference](../2026-10-02/clef-reference-base-bf16/receipt.json)
is reused unchanged. Recompute the paired interval without GPU or Modal access:

```bash
uv run --no-sync s1 eval compare \
  docs/validation/2026-10-02/clef-reference-base-bf16/boolq \
  docs/validation/2026-10-07/jev-omni-local/boolq \
  --seed 20260930 --resamples 2000 \
  --output artifacts/jev-omni-replayed-comparison.json
```

This replay was executed and matched the combined summary's estimate,
confidence interval and 128 request identities exactly.

The completed Modal app is `ap-7EG9BCbbhr51vM75NzoIXN`, call
`fc-01M4AKSWAB51RMV255SBSDRZDZ`; its original volume path is
`gemma-unified-system-one:/comparisons/20261007-jev-omni-01/jev-omni-local`.
The [dataset restoration instructions](../../modal.md#reproduce-the-october-1-matched-comparison)
apply before a fresh GPU run:

```bash
uv run --no-sync modal run apps/modal/compare.py \
  --run your-unique-jev-run --models jev-omni-local
```

Local verification: 174 focused adapter/planning/runner/collection tests passed,
including native image/audio preservation, and 70 report-rendering tests passed.
Static lint checks passed. These checks supplement the completed real GPU run.
