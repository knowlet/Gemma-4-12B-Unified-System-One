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
