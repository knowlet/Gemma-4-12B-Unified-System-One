# MPS and MLX validation — 2026-10-02

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

The generated model folders are under `artifacts/exports/` and remain ignored by
Git. They contain original-base conversions, not recovered S1 fine-tuned weights.
