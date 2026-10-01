# Modal test, training and deployment

Install `uv sync --extra modal`, then use `uv run --no-sync modal ...`. Modal must
already be authenticated (`modal token new` is an interactive user step).

The new `apps/modal/unified.py` uses `Image.uv_sync(frozen=True)`. Both CPU checks
and the GPU runtime use the repository lockfile. It stores model downloads and
training outputs in the separate `gemma-unified-system-one` volume, preserving
the upstream `jev-replica` volume. GPU jobs default to one A100 40GB container.

```bash
# Short CPU contract tests; no checkpoint download.
uv run --no-sync modal run apps/modal/unified.py --mode smoke \
  --output artifacts/modal-cpu-smoke.json

# Downloads the 12B checkpoint; real text/image/audio forwards, no training.
uv run --no-sync modal run apps/modal/unified.py --mode model-smoke \
  --output artifacts/modal-gemma-smoke.json

# Same benchmark JSONL on Laya or Gemma.
uv run --no-sync modal run apps/modal/unified.py --mode benchmark --backend laya \
  --dataset examples/benchmarks/smoke.jsonl --output artifacts/modal-laya.json
uv run --no-sync modal run apps/modal/unified.py --mode benchmark --backend gemma \
  --dataset examples/benchmarks/smoke.jsonl --output artifacts/modal-gemma.json

# Supervised LoRA + disjoint calibration; saves /vol/runs/unified-lora.
uv run --no-sync modal run apps/modal/unified.py --mode train \
  --dataset data/train.jsonl --calibration data/calibration.jsonl --steps 100

# Persistent API deployment, scales to zero after idle.
uv run --no-sync modal deploy apps/modal/unified.py
```

`S1_GPU` selects hardware, `S1_MODEL` selects a Hub model or saved volume checkpoint,
and `S1_REVISION` pins its revision. Set `S1_HF_SECRET` to the name of an existing
Modal secret containing `HF_TOKEN` if checkpoint access requires it. Empty values
are treated as unset. No local credential files are uploaded.

The deployed web function requires Modal proxy authentication. Call the returned
URL with `Modal-Key` and `Modal-Secret` headers from a Modal proxy token. The
`/v1/systemone` and `/decide` routes use the same application as local serving.
There is a one-container limit and serialized inference. This initial service has
no tenant billing, persistent request queue or autoscaling benchmark.

GitHub's `Modal` workflow runs only via `workflow_dispatch`; ordinary pull requests
do not receive cloud credentials or start paid GPUs. Configure a `modal` GitHub
Environment with `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, and optional
`S1_HF_SECRET` / `S1_REVISION` variables. Configure required reviewers there if
deployment approval is needed. The workflow supports CPU smoke, real model smoke
and deployment. It does not automatically train or publish model weights.

Historical apps still run with paths such as
`uv run --extra modal modal run apps/modal/train.py ...`. The old game demos use
their original, specialized emulator images; they are outside the new deployment.

## Reproduce the October 1 matched comparison

The comparison uses the original 128 BoolQ questions and 52 media cases. Fresh
media preparation can change compressed PNG bytes and float32 audio rounding,
even with the same official samples. Restore the archived native media after
preparing the public data and before launching the comparison:

```bash
uv run --extra inference --extra train --extra modal python \
  scripts/prepare_live_validation_data.py --output artifacts/live-validation/datasets
uv run --no-sync python scripts/restore_comparison_data.py
uv run --no-sync modal run apps/modal/compare.py --run <unique-name>
```

On later runs, skip preparation and start with the idempotent restore command;
the preparation script deliberately refuses to overwrite an existing different
media serialization.

The restore command reads the original M1 file from the existing
`gemma-unified-system-one` Modal volume; it starts no cloud compute. It requires
access to that volume, or `--source /absolute/path/to/original-M1.jsonl` for an
already downloaded copy. It verifies the historical dataset fingerprint, every
text-file hash, the media-file hash, the complete media-bundle hash and the full
provenance hash before replacing media files. Existing generated files are backed
up, and an already restored dataset is left intact.

The restored media fingerprint is
`7f3dff9c8fcfb164df5f8ca05b0fdaadbfb3a84812bfbe4c9b1906b6954d9f35`.
[The recovery receipt](validation/2026-10-01/media-rebuild.json) records the
regeneration checks: decoded pixels matched for all 32 images; all 20 audio
sample counts matched and the largest float32 sample difference was
`8.940696716308594e-08`. The actual comparison uses the archived exact bytes.
`apps/modal/prepare_comparison_data.py` is an optional CPU reconstruction
diagnostic; it refuses to replace files when historical hashes differ.

## Reproduce the fixed NF4 LoRA recovery

The original CE adapters lose accuracy when directly loaded in NF4. The separate
recovery experiment continues **all three** historical CE adapters against the
same pinned NF4 base. Its recipe is fixed: 100 batch-one CE optimizer updates at
learning rate `2e-5`, with the original 256 training cases and 16 calibration
cases. Exactly 100 distinct training cases enter optimizer updates in each seed;
this is less than one epoch. The 128 test questions and 52 media cases are used
only for the subsequent evaluation. The experiment is exploratory, motivated by
the initial quantization results; no test-based seed or checkpoint selection is
performed.

After preparing/restoring the exact datasets above, run:

```bash
uv run --no-sync modal run apps/modal/recover_nf4.py --run <unique-recovery-name>
```

The command needs access to the existing historical CE adapters on the
`gemma-unified-system-one` volume. It uses at most three A100 80GB containers at
once. Each training container verifies a real forward/backward pass with zero
optimizer updates before the fixed recipe, then records actual updates, sampled
frozen-weight identity, changed LoRA weights, and all train/calibration identities.
Only LoRA tensors train; the embedding and output head remain frozen BF16.

Fresh inference containers verify saved-adapter calibration parity and execute
the same BoolQ, native media, and six HTTP load shapes as the main comparison.
The parity model is unloaded before the harness loads its model and resets CUDA
peak counters. Both loads share one fresh inference process, so framework caches
may already be warm; do not attribute memory differences to training alone.
Training memory is recorded separately. Receipts and predictions download to
`artifacts/comparison/<unique-recovery-name>/`; all three adapters remain under
`/vol/comparisons/<unique-recovery-name>/adapters/seed-N`.

To download **all three measured recovered adapters** from the completed
`20261001-nf4-recovery-02` campaign:

```bash
uv run --no-sync modal volume get gemma-unified-system-one \
  /comparisons/20261001-nf4-recovery-02/adapters \
  artifacts/adapters/nf4-recovered
```

This creates `seed-0`, `seed-1`, and `seed-2`, each including its calibrated
`s1_config.json` and `training-provenance.json`. Choose a seed explicitly for an
application; the example below uses seed 0 by convention, which is **not** the
highest test-scoring seed. Evaluate a deployment choice on new heldout data.

```bash
uv run --no-sync s1 decide examples/request.json \
  --backend gemma --device cuda --precision bfloat16 --quantization nf4 \
  --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
  --adapter-path artifacts/adapters/nf4-recovered/seed-0
```

For the existing Modal inference app, set
`S1_ADAPTER_PATH=/vol/comparisons/20261001-nf4-recovery-02/adapters/seed-0`
and `S1_QUANTIZATION=nf4`, with the same pinned revision. The original CE adapters
in the general quantization examples remain separate checkpoints; their NF4
accuracy must not be replaced with the recovered adapters' result. See the
[comparison](comparison.md) for all seeds, native media tradeoffs, and measured
memory/latency. These scripts save adapters and run temporary jobs; deployment
uses the explicit inference commands in the [quantization guide](quantization.md).
