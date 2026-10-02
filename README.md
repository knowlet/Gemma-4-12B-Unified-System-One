# Gemma 4 Unified System One

Native text, image and audio decisions from `google/gemma-4-12B-it`:
one multimodal backbone forward, multiple answer slots, selected LM-head rows,
and typed Choice / Noul / Score probabilities. No text generation or JSON parsing.

This project extends [system-one-open](https://github.com/mithalouni/system-one-open).
The original E2B experiments and demos remain available; their historical results
are not measurements of this Unified model. See [upstream notes](docs/upstream.md).

## Latest measured results — October 1, 2026

The matched Modal campaign completed **17 model/format/seed configurations** on
the same **128 BoolQ questions, 52 native media cases, and six HTTP load shapes**.
The [full comparison](docs/comparison.md) includes per-seed accuracy, media results,
memory, latency percentiles, paired confidence intervals, and historical A100 runs.

| Model / format | BoolQ accuracy | Peak CUDA allocated | Mean time per BoolQ decision |
| --- | ---: | ---: | ---: |
| Gemma CE-128 BF16 | 91.15% | 24.50 GB | 127–156 ms |
| Gemma CE-128 INT8 | **91.41%** | 13.90 GB | 440–595 ms |
| Gemma CE-128 NF4 | 87.24% | 8.34 GB | 160–226 ms |
| Gemma NF4 + 100 additional training updates | **90.36%** | **8.34 GB** | 173–192 ms |
| Gemma base BF16 | 86.72% | 24.45 GB | 88 ms |
| Decider-2B | 88.28% | 4.25 GB | 76 ms |
| Laya | 82.03% | 2.77 GB | 33 ms |
| AgentJev-0.6B | 72.66% | 3.70 GB | 41 ms |
| Kev-0.8B | 71.88% | 2.00 GB | 120 ms |

Gemma CE/quantized rows report all three fixed seeds: mean accuracy, worst peak
allocation, and the range of per-seed mean latency. Other rows use one run. These
are point estimates on a small shared test set. Modal supplied A100 80GB PCIe and
SXM variants; the full report records each GPU and runtime. Latencies exclude model
loading and differ from HTTP service p99. GB is decimal; peak allocation is not the
minimum GPU capacity needed for deployment.

- **INT8 preserves BoolQ accuracy**, but exceeds the 12 GB allocation target and
  runs slower than BF16 in this implementation.
- **NF4 recovery reaches 90.36% mean accuracy**, up from 87.24% without additional
  training. The fixed 100-update continuation is exploratory: individual seeds
  range from 88.28% to 92.19%, and need confirmation on new held-out data. Its
  tensor footprint is 7.61 GB, full-workload peak allocation is 8.34 GB, and observed
  device usage is 9.44 GB; deployment on an 8 GB GPU has not been established.
- **Media remains a weakness.** Gemma base scores 28/32 on MNIST and 6/20 on FSDD;
  every original and recovered NF4 seed scores only 1/20 on FSDD. The four external
  text models are marked unsupported for native images/audio.
- **Smaller models offer useful tradeoffs.** Decider scores 88.28% with 4.25 GB
  peak allocation; Laya has the lowest measured mean latency at 33 ms and 82.03%
  accuracy. Decider/Kev use eager reference kernels; AgentJev uses its current
  coding checkpoint. Their optimized or older published runs are different tests.

See the [saved result summary](docs/validation/2026-10-01/comparison-summary.json)
for 29 retained attempts and 13 paired comparisons, the
[quantization guide](docs/quantization.md) for inference commands, and
[NF4 recovery instructions](docs/modal.md#reproduce-the-fixed-nf4-lora-recovery)
for the fixed training recipe and all three saved adapters.

## Development

```bash
uv sync --extra api
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv build
```

Inference and model tests: `uv sync --extra inference --extra api`.
Cloud tooling: add `--extra modal`; Laya comparisons: add `--extra laya`.
`uv.lock` pins all optional dependencies. No model download occurs during the default tests.

See [architecture](docs/architecture.md), [benchmarking](docs/benchmarking.md),
and [Modal deployment](docs/modal.md) for runnable examples and limitations.

## Benchmarks and reproducibility

[Decision Benchmark v2](docs/benchmark-v2.md) includes model comparison adapters,
Gemma G0–G4 readouts, native batches, load tests, calibration/statistics, matched
training, multimodal pipelines, resettable workflows and standalone reports.

Accuracy gates default to the full decision population and require complete shared
support; a common eligible subset must be declared explicitly. Workflows check every
state for training/calibration overlap, validate the loaded model identity before
inference, and include OCR/ASR bills in reported costs. Unknown stage bills keep the
total unknown. See the v2 guide for gate population and workflow manifest fields.

The [2026-10-01 live validation](docs/validation/2026-10-01/live-validation.md)
records the earlier 12B GPU, training, OCR/ASR and HTTP load results. Its
10,000-request service p99 of 105 ms belongs to that historical workload; it does
not describe the new quantized models or guarantee that latency under concurrency.
The [current comparison](docs/comparison.md) retains both sets of measurements
with their hardware, precision, and workload identities.

Run the matched campaign with a fresh run ID after installing the Modal, inference,
training, API, Laya, and quantization extras and following the
[original-data preparation and restoration steps](docs/modal.md#reproduce-the-october-1-matched-comparison):

```bash
uv sync --extra inference --extra train --extra api --extra modal --extra laya --extra quantization
uv run --no-sync modal run apps/modal/compare.py --run your-unique-run
```

CUDA defaults to BF16; use explicit `precision="float32"` (and matching registry
precision) for probability-parity checks. The pinned 12B BF16 backbone shows
material batch-dependent numerical drift, even with independent question inputs.

```bash
uv run --no-sync s1 eval run --profile text-smoke --enable-model uniform \
  --output artifacts/evaluation/text-smoke-run
uv run --no-sync python scripts/benchmark_v2_acceptance.py \
  --output artifacts/evaluation/acceptance
```

On Apple Silicon, Unified inference now selects MPS automatically and uses BF16
on macOS 14 or newer (FP32 on older supported macOS). Use `--device cpu` or
`--dtype float32` for an explicit reference configuration. See the
[MPS measurements](docs/mps.md) and [GGUF / MLX release paths](docs/model-export.md).

## Layout

| Path | Purpose |
| --- | --- |
| `src/s1/` | Installable decision SDK, validation, multimodal inference, API and evaluation |
| `apps/modal/` | Cloud entrypoints; retained upstream training and demos |
| `tests/` | Offline contracts, numerical regression tests and tiny-model integration |
| `examples/` | Requests and a small benchmark fixture, not an accuracy leaderboard |
| `scripts/` | Dataset preparation, training checks, MPS profiling, MLX validation and comparison reports |
| `docs/` | Design, evaluation protocol and deployment instructions |
| `results/`, `media/`, `report.html` | Historical upstream artifacts, retained unchanged |

The new API exposes `/v1/systemone` and `/decide`; local CLI usage is `uv run s1 --help`.
