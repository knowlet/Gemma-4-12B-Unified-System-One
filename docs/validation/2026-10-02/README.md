# Clef matched benchmark — October 2, 2026

`Cloudflare/clef` revision `2f3de3dd85f379784083b0814d997ab627200f0c` completed
the original 128 BoolQ cases, 52-case media eligibility evaluation, and all six
128-request HTTP load shapes on an NVIDIA A100-SXM4-80GB. The runtime used
PyTorch 2.8.0 and Transformers 5.17.0, eager reference kernels, BF16 computation,
and the pinned publisher's joint schema head. It used no CUDA graphs, compilation,
persistent prefix cache, or optional FLA/causal-conv1d acceleration.

| Observation | Measured result |
| --- | ---: |
| BoolQ | 115/128 = 89.84375% |
| MNIST | 27/32 = 84.375% |
| FSDD | 20 unsupported audio cases; no accuracy value |
| BoolQ serial mean / p99 | 185.018 / 245.796 ms |
| Closed-loop C1 / C16 HTTP p99 | 252.906 / 3,259.146 ms |
| Registered tensor footprint | 54.969570168 GB |
| Full-workload CUDA peak allocated / reserved | 55.661130752 / 55.859740672 GB |
| Observed device-wide usage | 56.747687936 GB |
| Process-lifetime peak RSS | 60.777943040 GB |
| Successful HTTP requests | 768/768 |
| HTTP requests missing the 1,000 ms SLO | 317/768 |

GB is decimal. Peak allocation is not a minimum GPU-capacity claim, and observed
device usage is not a per-process peak. Modal's requested 48 GiB host-memory
reservation was not a hard limit; measured process RSS exceeded it. Load timings
include queueing behind one serial model process over loopback HTTP, without WAN
latency or batching. The [full comparison](../../comparison.md) records all six
load shapes separately. Clef's media latency covers supported images only, unlike
Gemma's combined image/audio timing.

Against the original Gemma-base BF16 run, Clef's BoolQ difference is **+3.125
percentage points**, with a source-group bootstrap 95% CI of **[−3.90625,
+10.15625] pp** (128 groups, 2,000 resamples, seed 20260930). This interval includes
zero and is unadjusted for multiple comparisons. It does not establish a
population-level accuracy advantage.

## Published evidence

The [combined summary](comparison-summary.json) retains **18 selected model
configurations, 30 attempts, and 14 paired comparisons**. The
[October 1 summary](../2026-10-01/comparison-summary.json) and its original sources
remain unchanged. The new collection verified all 11 historical source campaign
hashes before adding Clef.

The [source/hash index](clef-evidence.json) records **423,630 bytes** of original,
unmodified evidence copied into this directory:

- [Clef receipt](clef-local/receipt.json), [BoolQ predictions](clef-local/boolq/predictions.jsonl),
  [BoolQ manifest](clef-local/boolq/run_manifest.json),
  [media predictions](clef-local/media/predictions.jsonl), and
  [media manifest](clef-local/media/run_manifest.json).
- [Gemma-base reference receipt](clef-reference-base-bf16/receipt.json),
  [BoolQ predictions](clef-reference-base-bf16/boolq/predictions.jsonl), and
  [BoolQ manifest](clef-reference-base-bf16/boolq/run_manifest.json).

Prediction and manifest digests match their source receipt's `artifact_sha256`.
Manifests preserve request identities; the published predictions contain no raw
image/audio payloads. The reference receipt also lists historical artifacts that
are not copied here. Only the files listed in the index are included.

Recompute the paired BoolQ interval from these repository files without GPU or
Modal access:

```bash
uv run --no-sync s1 eval compare \
  docs/validation/2026-10-02/clef-reference-base-bf16/boolq \
  docs/validation/2026-10-02/clef-local/boolq \
  --seed 20260930 --resamples 2000 \
  --output artifacts/clef-replayed-comparison.json
```

This replay was checked against the combined summary and gives the identical
estimate and interval. Local implementation checks are recorded in
[clef-local-checks.json](clef-local-checks.json).

## Full archive and fresh execution

The complete Clef archive includes request/error records and all six load-cell
manifests and queueing observations. It remains in the
`gemma-unified-system-one` Modal volume at
`/comparisons/20261002-clef-01/clef-local`; artifact hashes are retained in the
published receipt. With access to that volume, download it to a fresh directory:

```bash
uv run --no-sync modal volume get gemma-unified-system-one \
  /comparisons/20261002-clef-01/clef-local artifacts/clef-full-archive
```

This retrieves saved files without launching a GPU. A fresh measured run uses the
[pinned Clef reproduction procedure](../../modal.md#reproduce-the-clef-comparison).
The completed Modal app was `ap-dIEOA1B4rkxHIYU6JN2ueB`; its call was
`fc-01M3XJCNTFREKQNA5D3EYYRN9H`.
