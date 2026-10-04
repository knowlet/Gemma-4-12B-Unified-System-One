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

The fresh calibration and test subsets come from official BoolQ validation,
balanced to equal yes/no counts. These are not official leaderboard scores.
"Fresh" means unseen in the recorded local experiments; foundation-model
pretraining overlap with these public datasets is unknown.

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
evaluation requires completed training and calibrated merged weights. The
current workflow returns a matching completed phase's saved receipt without
rewriting it or loading a model. An interrupted stage requires explicit
`--resume` with unchanged code, lockfile, data and recipe before it can change
the phase state.
The runtime binds all `src/s1` sources and the trainer
to the run identity. Checkpoints preserve optimizer, RNG, sample order and the
completed-update ledger; abandoned log tails are retained separately. The soft
training budget is five hours and the Modal invocation cap is six hours.

Historical train/evaluate/resume stages for this release must use runtime
revision `d4ea1ef79af2e22f666374e529fa56c1de9d3f53`, whose recorded trainer,
lockfile and 55 S1 source hashes match the run. Later integration of the Clef
benchmark changes the complete source identity even though it does not change
the release execution paths. Use a fresh run name with newer code; do not alter
the historical manifests to bypass the identity check.
The pinned historical trainer predates the duplicate-invocation guard; do not
invoke its already completed stages again.

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
by the fresh BoolQ test and prior media regression evaluation. Weight hashes and conversion/runtime
versions bind each result to its artifact.

The [reproduction guide](release-reproduce.md) gives the exact source-download,
input-capture, MLX conversion, hash-pinning, evaluation and summary commands.
It records the separate Transformers and MLX environments.

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

## Completed BF16 training and evaluation

The [training receipt](validation/2026-10-02/release/train-receipt.json) records
all 2,048 updates completed in 1,349.35 seconds (22.49 minutes). The whole stage,
including loading, merge, reload, calibration and persistence, took 1,541.93
seconds (25.70 minutes). Peak CUDA allocation was 24.698 GB on A100-SXM4-80GB.
Initial and final logged losses are different individual examples and are not
a before/after dataset loss comparison.

The adapter and full merged checkpoint have separate file hashes in the
[artifact receipt](validation/2026-10-02/release/artifacts.json). Before/after
merge decisions agreed on the eight calibration cases measured; maximum raw
logit change was 0.375 and maximum temperature-1 probability change was
0.000107. This small check does not establish global or bitwise parity.

The base and reloaded merged model fit separate temperatures using the same
256 calibration cases: 2.5185519609 and 2.7830344470 respectively. Calibration
NLL improvement is an in-sample fitting result. Held-out results below come from
the [evaluation receipt](validation/2026-10-02/release/evaluate-receipt.json),
with full coverage and no warmup errors in all six CUDA reports.

| Dataset | Model | Correct / scored | Accuracy | NLL | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fresh BoolQ | Base BF16 | 205 / 256 | 80.08% | 0.4409 | 0.2874 | 0.0671 |
| Fresh BoolQ | Trained BF16 | 230 / 256 | 89.84% | 0.2727 | 0.1580 | 0.0521 |
| Prior BoolQ | Base BF16 | 111 / 128 | 86.72% | 0.3316 | 0.1983 | 0.0799 |
| Prior BoolQ | Trained BF16 | 118 / 128 | 92.19% | 0.2498 | 0.1373 | 0.0586 |
| Prior media | Base BF16 | 34 / 52 | 65.38% | 1.0500 | 0.4168 | 0.1574 |
| Prior media | Trained BF16 | 34 / 52 | 65.38% | 1.0726 | 0.4236 | 0.2184 |

Fresh BoolQ accuracy improved by 9.77 percentage points in this single-seed
run. A paired source-group bootstrap (10,000 resamples, seed 42) gives a 95%
percentile interval of +5.86 to +14.06 points: 29 cases improved and four
regressed. Media aggregate accuracy was unchanged, but NLL, Brier and ECE worsened;
these results do not establish preservation of general multimodal quality or
media calibration. NLL uses a probability floor of `1e-12`, Brier sums squared
errors over legal candidates, and ECE uses 15 equal-width bins.

Probability decimal length does not imply full-precision logits. The CUDA path
projects and softcaps in BF16 before FP32 temperature scaling and softmax.
Different inputs can share rounded logits or logit differences, producing exact
probability repeats; equal candidate logits produce ties. The saved
[base calibration logits](validation/2026-10-02/release/evaluate/base-calibration.json)
include both repeated `[22.5, 27.625]` pairs and equal-logit pairs. Each case still
runs inference separately. Held-out CUDA raw logits were not retained, so these
calibration examples do not reconstruct every repeated held-out output.

## Apple Silicon execution

The exact merged checkpoint passed all eight synthetic MPS workloads on the
64 GiB M1 Max: three measured complete requests and three separate phase
profiles per workload, following one warmup each. All 24 measured requests
and 87 answer decisions completed with valid probabilities. This is execution
and performance coverage, not Choice/Score or multimodal accuracy evidence.

| Workload | Complete-request p50 |
| --- | ---: |
| Billing / two questions | 696 ms |
| Technical / two questions | 698 ms |
| Score | 502 ms |
| Image | 2,086 ms |
| Audio | 709 ms |
| Image + audio | 2,323 ms |
| Long state | 4,576 ms |
| Sixteen questions | 3,500 ms |

Loading took 75.25 seconds and the first cold request took 29.04 seconds;
those times are excluded from the warm request table. The
[raw profile](validation/2026-10-02/release/trained-mps.json) and
[checkpoint binding](validation/2026-10-02/release/trained-mps-provenance.json)
record BF16, SDPA, package versions, full timing samples and memory snapshots.
Snapshots are not peak-memory measurements.

## Trained MLX 8-bit validation

Conversion completed on October 2 and full validation completed on October 3,
2026, using MLX-VLM 0.7.4, MLX/MLX-Metal 0.32.3 and the same 64 GiB M1 Max.
The three weight shards total **12,754,909,484 bytes**, versus **23,919,549,408
bytes** for the merged BF16 source. All 677 source tensors are represented:
330 weights use affine 8-bit/group-64 quantization and 347 tensors remain BF16.
Native vision/audio modules and tied candidate embeddings are retained.

All **256 calibration + 256 fresh test + 52 media cases** completed successfully.
The MLX temperature was independently fitted to `2.7144176165949068` using only
calibration labels. The
[MLX report](validation/2026-10-02/release/mlx-release-evaluation.json),
[conversion manifest](validation/2026-10-02/release/mlx-conversion-manifest.json)
and [recomputed summary](validation/2026-10-02/release/summary/README.md) bind
these measurements to the exact source and quantized weight files.

| Dataset | Model | Correct / scored | Accuracy | NLL | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fresh BoolQ | Trained BF16 | 230 / 256 | 89.84% | 0.2727 | 0.1580 | 0.0521 |
| Fresh BoolQ | Trained MLX 8-bit | 230 / 256 | 89.84% | 0.2705 | 0.1564 | 0.0455 |
| Prior media | Trained BF16 | 34 / 52 | 65.38% | 1.0726 | 0.4236 | 0.2184 |
| Prior media | Trained MLX 8-bit | 34 / 52 | 65.38% | 1.0820 | 0.4258 | 0.1310 |

Both trained formats score 28/32 on MNIST and 6/20 on FSDD. The older 128-case
BoolQ set was not run in MLX and remains explicitly `not_run` in the summary.

Equal accuracy does not establish confidence equivalence. All 256 fresh BoolQ
predictions agree, but the maximum candidate-probability difference is **27.48
percentage points**. Media predictions agree on **49/52** cases and the maximum
probability difference is **31.29 points**. The three changed FSDD predictions
include one correct-to-wrong, one wrong-to-correct and one wrong-to-wrong change.
See the [paired output comparison](validation/2026-10-02/release/cross-runtime-parity.json).
These are independently calibrated outputs across different runtimes, so the
comparison does not isolate quantization alone. The zero-width fresh-test
bootstrap interval reflects zero observed per-case accuracy differences; it is
not a global parity or noninferiority guarantee.

MLX fresh-test prepared-forward p50/p95 was **1,506.87 / 2,854.65 ms**; media was
**3,282.42 / 3,471.44 ms**. These measurements exclude preprocessing, loading,
file I/O and calibration. They use one warmup on the first case of each split
and one measured forward per case; other input shapes may incur compilation.
CUDA and MPS tables have different hardware or workloads and timing scopes,
so their ratios are not end-to-end speedups.

## Publication and code checks

Both complete model packages were published on October 4, 2026:

| Format | Published package | Immutable weights revision | Published model card |
| --- | --- | --- | --- |
| BF16 | [Public snapshot](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One/tree/a66f836b56605039fe040f330180e336d19b3362) | [`66626de5cbcf8c5fecb8e9b58710805a02a42f67`](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One/tree/66626de5cbcf8c5fecb8e9b58710805a02a42f67) | [BF16 card](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One/blob/a66f836b56605039fe040f330180e336d19b3362/README.md) |
| MLX 8-bit | [Public snapshot](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-MLX-8bit/tree/a5f89b400f7ef63e162f22866c206ed67cf8f282) | [`e304ce0b87148933ce43d5ef9bed779f048accc8`](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-MLX-8bit/tree/e304ce0b87148933ce43d5ef9bed779f048accc8) | [MLX card](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-MLX-8bit/blob/a5f89b400f7ef63e162f22866c206ed67cf8f282/README.md) |

The [October 4 publication evidence](validation/2026-10-04/README.md) and
[publication receipt](validation/2026-10-04/publication-status.json) record
matching remote file sizes, LFS SHA-256 and Git blob identities, anonymous weight
metadata access, and downloaded public cards, manifests and calibration sidecars.
This verification did not redownload complete weight files or run new inference.
Each published package's `artifact-manifest.json` hashes its packaged files
except itself. Final cards and manifests were published after the weight commits;
the table keeps those revisions distinct.

The reproduction commands use these local paths:

- BF16: `artifacts/release/checkpoint` (adapter retained separately).
- MLX: `artifacts/exports/s1-boolq-mlx-8bit`.

The [October 3 preparation receipt](validation/2026-10-02/release/publication-status.json)
and archived [BF16 card](validation/2026-10-02/release/model-cards/bf16/README.md)
and [MLX card](validation/2026-10-02/release/model-cards/mlx-8bit/README.md) retain
their original pending-publication status. These historical records are unchanged;
the October 4 evidence records the completed publication separately. Published
weights retain the training/conversion hashes and the measured results above.

The [code validation receipt](validation/2026-10-02/release/code-validation.json)
records **808 passed, 3 optional-dependency skips**, 44 additional checks in the
contracts-only environment, Ruff, package build and all four source CI jobs.
The runtime/evaluation code revision is
`d4ea1ef79af2e22f666374e529fa56c1de9d3f53`.

After integrating main revision `9bcf9db823b7e96bc62dc0bc9924fafb34835199` and
adding completed-phase receipt protection, the combined local suite passed
**894 tests with 3 optional-dependency skips**, including 39 focused training
workflow tests. Ruff and the package build also passed.
The release summary regenerated from the recorded CUDA/MLX reports is byte-for-byte
identical. Historical training and model-package receipts retain their original
source and artifact hashes.
