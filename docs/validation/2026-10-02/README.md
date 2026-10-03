# Validation — October 2, 2026

## Trained BoolQ specialist release

The [release report](../../release.md) covers the completed 2,048-update LoRA
run, full BF16 merge, MPS execution and MLX 8-bit validation completed October 3.
Its [raw results](release/summary/README.md) use a fresh 256-case BoolQ test;
the base-model profiling and historical Clef comparison below use different
workloads and must retain their own populations and timing scopes.

## Base-model MPS and MLX validation

Measured on Apple M1 Max with 64 GiB unified memory using the actual
`google/gemma-4-12B-it` checkpoint at
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.

- [MPS protocol and interpretation](../../mps.md)
- [Model conversion and release assessment](../../model-export.md)
- [Machine-readable summary](summary.json)
- PyTorch reports: [BF16 SDPA](bf16-sdpa.json), [BF16 eager](bf16-eager.json),
  [BF16 prefer Metal](bf16-sdpa-metal.json), [FP16 failure](fp16-sdpa.json).
- MLX comparisons: [unquantized BF16](mlx-bf16-validation.json),
  [4-bit](mlx-4bit-validation.json), [8-bit](mlx-8bit-validation.json).
- Export provenance and SHA-256 hashes: [4-bit](mlx-4bit-manifest.json),
  [8-bit](mlx-8bit-manifest.json).

The first SDPA report predates the profiler's addition of explicit warmup samples
and planned-coverage fields. All eight cases completed with three measured
requests and three separate phase measurements each. The FP16 report stopped at
the fourth case (image warmup); remaining cases were unattempted, not successes.
The summary records its denominator explicitly.

The MLX input capture and numerical verifier are experimental. All reports use
synthetic smoke cases with temperature 1.0; no accuracy/calibration acceptance is
claimed. MLX timings exclude HF preprocessing, I/O and model loading, unlike
PyTorch end-to-end timings. Do not compare them as interchangeable latency figures.

Local gates: **389 tests passed**, Ruff lint and formatting passed, and both source
distribution and wheel built successfully. One existing Starlette/httpx
deprecation warning remains. No Hub upload or cloud deployment was performed.

These measurements concern original-base conversions. Their reports and
manifests are retained; locally generated base weight shards were removed after
validation to make room for the trained release. The source conversions remain
reproducible from their pinned checkpoint and commands.

## Clef matched benchmark

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

### Published evidence

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

The archived manifest's `setup_ms` measures the runner's adapter-factory call and
runtime telemetry validation. This campaign supplies an already-loaded shared
adapter, so Clef's BoolQ value of **0.481779 ms** is not checkpoint-loading time.
The separate [Clef receipt](clef-local/receipt.json) records `load_seconds` as
**244.880485042 seconds** for checkpoint acquisition and model/processor setup
before either quality run. It excludes Modal scheduling/container startup.
The raw manifests retain their original bytes and hashes; future manifests also
declare this distinction in `setup_timing_scope`.

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
estimate and interval. Initial implementation checks are recorded in
[clef-local-checks.json](clef-local-checks.json).

### Full archive and fresh execution

The complete Clef archive includes request/error records and all six load-cell
manifests and queueing observations. It remains in the
`gemma-unified-system-one` Modal volume at
`/comparisons/20261002-clef-01/clef-local`; artifact hashes are retained in the
published receipt. With access to that volume, download it to a fresh directory:

```bash
mkdir -p artifacts/clef-full-archive
uv run --no-sync modal volume get gemma-unified-system-one \
  /comparisons/20261002-clef-01/clef-local artifacts/clef-full-archive
```

This retrieves saved files into `artifacts/clef-full-archive/clef-local` without
launching a GPU. Precreate the destination directory: the observed Modal CLI
treats a nonexistent destination as one file when downloading folder contents.
A fresh measured run uses the
[pinned Clef reproduction procedure](../../modal.md#reproduce-the-clef-comparison).
The completed Modal app was `ap-dIEOA1B4rkxHIYU6JN2ueB`; its call was
`fc-01M3XJCNTFREKQNA5D3EYYRN9H`.
