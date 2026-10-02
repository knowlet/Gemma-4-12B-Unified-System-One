# October 2 specialist release

This release trains the S1 answer-slot interface on English BoolQ. It uses LoRA
and merges the adapter into the complete 12B multimodal checkpoint. It is not
full-parameter optimization, and the training data does not establish general
multimodal, multilingual, Choice, Score or multi-question quality.

## Fixed protocol

Extract the [frozen dataset archive](validation/2026-10-02/release-datasets.tar.gz)
from a clean checkout. It contains the exact training/evaluation JSONL files,
source records, explicit excluded passage groups and dataset attribution. The
BoolQ files are CC-BY-SA-3.0; the media provenance records the separate MNIST and
FSDD licenses. Archive SHA-256:
`9fada5df498cc351bdac29c1d58b44a5efd8e7c05aad6ccbbec01c18e9e8657f`.

```bash
mkdir -p artifacts/release/datasets
tar -xzf docs/validation/2026-10-02/release-datasets.tar.gz \
  -C artifacts/release/datasets
```

The dataset manifest SHA-256 is
`e0ab28e3435a910c19abf3a9a7e58df8b5cff7a806b6c7f4f18021db1d3804dd`.
`scripts/prepare_release_dataset.py` records the deterministic source-selection
algorithm; extracting the archive preserves the exact historical media bytes
without depending on the original workspace or Modal volume.

The source is `google/gemma-4-12B-it` at
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. The dataset preparation script selects
2,048 balanced training cases, 256 balanced calibration cases and 256 balanced
test cases deterministically from the pinned BoolQ source. A passage is a source
group. Fresh calibration and test passages exclude every official training
passage and every previously sampled case. Training excludes earlier held-out
passages. IDs, groups, states and canonical requests are checked for overlap.

The previous 128-case BoolQ test and 52 native media cases remain regression
sets. They have already informed earlier experiments, so their results are not
fresh held-out evidence. The media set comprises 32 MNIST images and 20 FSDD
audio recordings; no media labels are fitted in this release.

The recipe fixes one pass through the 2,048 training examples, rank-8 LoRA with
alpha 16 on q/k/v/o projections, learning rate `2e-5`, and cross-entropy loss.
Training uses BF16 base weights, gradient checkpointing, finite-gradient checks,
and durable adapter/optimizer/RNG checkpoints. A short same-recipe pilot measures
time and peak GPU memory before the complete pass. There is no test-driven
checkpoint selection or hyperparameter search.

```bash
uv sync --locked --extra inference --extra train --extra api --extra modal
# Choose a new run name. Completed or interrupted runs preserve their identity.
uv run --no-sync modal run --detach apps/modal/release.py --stage pilot --run your-release-run
uv run --no-sync modal run --detach apps/modal/release.py --stage train --run your-release-run
uv run --no-sync modal run --detach apps/modal/release.py --stage evaluate --run your-release-run
uv run --no-sync modal run apps/modal/release.py --stage export-results --run your-release-run
```

Stages run sequentially; training requires a matching successful pilot, and
evaluation requires completed training and calibrated merged weights. Use
`--resume` only to continue an interrupted pilot/train with unchanged code,
lockfile, data and recipe. The runtime binds all `src/s1` sources and the trainer
to the run identity. Checkpoints preserve optimizer, RNG, sample order and the
completed-update ledger; abandoned log tails are retained separately. The soft
training budget is five hours and the Modal invocation cap is six hours.

The recorded run is `20261002-boolq-release-01`, seed 42, with training code at
`b2a46b1`. Its A100-SXM4-80GB pilot completed all eight updates in 10.41 training
seconds; the longest of all 2,048 prepared training inputs was 656 tokens, and
the observed peak CUDA allocation was 24.617 GB. Loading, preprocessing preflight
and persistence brought the complete pilot stage to 74.93 seconds. These are
pilot observations, not a minimum deployment memory requirement.

The adapter is retained before merging. The merged checkpoint is reloaded and
its scalar temperature is fitted using only the calibration split. Full test
coverage, accuracy, NLL, multiclass Brier and ECE are reported for the base and
merged checkpoint. The full multimodal MLX 8-bit export receives a separate
temperature fitted on the same calibration cases in its own runtime, followed
by the untouched test and media evaluation. Weight hashes and conversion/runtime
versions bind each result to its artifact.

## Scope of the downloadable models

The Transformers checkpoint implements S1 through this repository's
`UnifiedDecisionModel`. It preserves the source processor and image/audio
modules. The MLX artifact uses MLX-VLM 0.7.4, affine 8-bit quantization with group
size 64, and the experimental selected-head evaluation adapter. Generic chat or
generation runners do not expose the S1 decision contract or its calibrated
candidate probabilities.

The [MPS report](mps.md) and [format validation](model-export.md) describe the
separate base-model performance experiments. Those eight synthetic workloads
are not the accuracy evaluation for this trained release. GGUF conversion has
not been executed; the documented recipe remains a feasibility path.

Training, final measurements and publication receipts will be added here after
their respective runs complete. No successful publication is claimed by this
protocol alone.
