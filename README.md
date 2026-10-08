# Gemma 4 Unified System One

Native text, image and audio decisions from `google/gemma-4-12B-it`:
multiple answer slots and typed Choice / Noul / Score probabilities.
The PyTorch and MLX paths use one multimodal backbone forward and selected
LM-head rows. The experimental GGUF path uses native llama.cpp prefills and
full-vocabulary projection before selecting legal candidates.
No text generation or JSON parsing is needed for decisions.

This project extends [system-one-open](https://github.com/mithalouni/system-one-open).
The original E2B experiments and demos remain available; their historical results
are not measurements of this Unified model. See [upstream notes](docs/upstream.md).

## NVFP4 release — October 7, 2026

The trained checkpoint now has a native packed **NVFP4** export for Blackwell,
with its candidate head and image/audio modules retained in BF16. The same B200
and PyTorch 2.10 runtime measured both formats on the frozen release datasets:

| Format | Weight files | Peak CUDA allocated | Fresh BoolQ (256 cases) | Median request latency |
| --- | ---: | ---: | ---: | ---: |
| Trained BF16 | 23.92 GB | 24.19 GB | 89.84% | 38.68 ms |
| Trained NVFP4 | **8.93 GB** | **9.14 GB** | 88.67% | 45.44 ms |

NVFP4 reduces memory in this runtime; it is slower on the measured workload.
Historical native media scores are 31/52 for NVFP4 and 32/52 for the same B200
BF16 reference. These measurements retain their own hardware/runtime scope;
the older A100 results below remain unchanged. All 436 release decisions were
replayed using the exact bundled runtime, with identical candidate probabilities.

The [NVFP4 package is published on Hugging Face](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-NVFP4/tree/5c4f3c2cb5de2be66b9e51160613e868b33c311f),
including the verified runtime wheel and Jev-Omni benchmark evidence.
Use the [NVFP4 runtime and reproduction guide](docs/nvfp4-release.md) and
[complete measured evidence](docs/validation/2026-10-07/nvfp4/README.md).
The `s1-transformers-nvfp4-v1` package requires its bundled S1 loader;
stock Transformers, vLLM and TensorRT-LLM compatibility is not claimed.

## BoolQ specialist training — October 2, 2026

A complete 2,048-update LoRA pass has been merged into standalone BF16 12B
weights, retaining the native image/audio modules. This is LoRA fine-tuning
followed by a full-model merge, not optimization of every base parameter.

On a fresh, balanced 256-case BoolQ validation subset, accuracy increased from
**80.08% to 89.84%** and NLL decreased from **0.441 to 0.273**. Both models use
separate temperatures fitted on 256 disjoint calibration cases. Training took
22.49 minutes on A100-SXM4-80GB. The merged BF16 model's historical media
accuracy stayed at 34/52, while NLL, Brier and ECE worsened; general multimodal
quality is not established.

The full multimodal **MLX 8-bit export is 12.75 GB** and also scores **89.84%**
on all 256 fresh cases after independent calibration. All observed fresh-test
predictions agree with BF16, but individual candidate probabilities differ by
up to 27.48 percentage points. Eight synthetic MPS workloads also execute
successfully; their timings are separate from accuracy evaluation.

The native **GGUF Q8_0 language model plus F16 image/audio projector is 12.79 GB**.
Its independently calibrated fresh BoolQ accuracy is **90.23% (231/256)**;
255/256 predictions agree with trained BF16, with up to 32.89 percentage points
of candidate-probability difference. Historical media accuracy is 33/52,
including 28/32 images and 5/20 audio recordings. The
[GGUF runtime and reproduction guide](docs/gguf-release.md) records the complete
564-case evaluation and the separately verified calibrated decision interface.

The [release protocol and raw results](docs/release.md) and
[reproduction commands](docs/release-reproduce.md) include frozen datasets,
exclusions, weight hashes and full validation. The
[BF16](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One/tree/a66f836b56605039fe040f330180e336d19b3362),
[MLX 8-bit](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-MLX-8bit/tree/a5f89b400f7ef63e162f22866c206ed67cf8f282)
and [GGUF Q8_0 + F16 projector](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-GGUF/tree/c418d37a17689fae554f7fc7d78db05ff7d52cfb)
packages were published on October 4, 2026. The
[publication evidence](docs/validation/2026-10-04/README.md) records immutable
revisions, matching remote file identities and public access checks.

## Matched model comparison — October 7, 2026

Matched Modal runs now cover **19 model/format/seed configurations** on
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
| Cloudflare Clef-27B | **89.84%** | 55.66 GB | 185 ms |
| Jev-Omni-12B BF16 | 87.50% | 24.38 GB | 81 ms |
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
  text models are marked unsupported for native images/audio. Clef scores 27/32
  on MNIST; its 20 audio cases are unsupported, with no fabricated audio accuracy.
- **Clef has a higher BoolQ point estimate than Gemma base**, 115/128 versus 111/128,
  with 55.66 GB peak allocation. Their paired difference is +3.125 percentage points
  (95% CI −3.906 to +10.156), which does not establish a statistical advantage.
  Clef uses the full pinned 27B release and shared reference runtime, not Clef-Flash
  or an optimized hosted service.
- **Jev-Omni scores 112/128**, with a paired difference versus Gemma base of
  +0.781 percentage points (95% CI −6.250 to +7.812). Native media scores 30/52:
  28/32 images and 2/20 audio recordings. It uses PyTorch 2.10, its native
  decision head, and float32 softmax; older rows use PyTorch 2.8. The interval
  does not establish an accuracy advantage.
- **Smaller models offer useful tradeoffs.** Decider scores 88.28% with 4.25 GB
  peak allocation; Laya has the lowest measured mean latency at 33 ms and 82.03%
  accuracy. Decider/Kev use eager reference kernels; AgentJev uses its current
  coding checkpoint. Their optimized or older published runs are different tests.

See the [latest saved result summary](docs/validation/2026-10-07/comparison-summary.json)
for 31 retained attempts and 15 paired comparisons, the
[Jev-Omni report and raw evidence](docs/validation/2026-10-07/jev-comparison.md), the
[Clef evidence and replay commands](docs/validation/2026-10-02/README.md) for published
predictions and manifests, the
[quantization guide](docs/quantization.md) for inference commands, and
[NF4 recovery instructions](docs/modal.md#reproduce-the-fixed-nf4-lora-recovery)
for the fixed training recipe and all three saved adapters.
The [October 1 summary](docs/validation/2026-10-01/comparison-summary.json) remains unchanged.

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
