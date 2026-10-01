# Quantized Gemma inference

The public CLI and Modal inference app support `none`, `int8`, and `nf4`, with
an optional existing CE decision adapter. They use the same multimodal processor
and candidate-only decision readout as BF16 inference. See the
[measured comparison](comparison.md) for accuracy, memory, and latency results.

INT8/NF4 require CUDA and BF16 floating layers. NF4 uses BF16 matrix compute and
nested quantization. INT8 uses FP16 internally when quantizing BF16 activations;
the surrounding model and dense answer head remain BF16. The tied input
embedding/head and image/audio projections stay dense, and CPU offload is disabled.

## Local CLI

Install inference, quantization, PEFT, and API dependencies:

```bash
uv sync --extra inference --extra quantization --extra train --extra api --extra modal
```

Use an existing adapter directory containing `adapter_config.json`, adapter
weights, and its `s1_config.json` calibration file. For users with access to the
campaign's existing Modal volume, this downloads the predeclared seed-0 adapter:

```bash
uv run --no-sync modal volume get gemma-unified-system-one \
  /runs/20261001-v2-live-04/training-full/checks/training-curve/shots-128-seed-0-ce \
  artifacts/adapters/ce128-seed0
```

These examples load the **original CE adapters**, whose NF4 mean accuracy is
**87.24%** across all three seeds on the shared BoolQ set. The separate fixed
LoRA recovery produces new adapters with mean **90.36%** (seed 0/1/2:
88.28% / 90.63% / 92.19%). To use those checkpoints, follow
[the recovery download and inference commands](modal.md#reproduce-the-fixed-nf4-lora-recovery)
and explicitly choose the recovered adapter path. The recovery result includes
additional training and does not establish >90% accuracy for every seed.

Run a decision, a small contract benchmark, or a local HTTP server:

```bash
uv run --no-sync s1 decide examples/request.json \
  --backend gemma --device cuda --precision bfloat16 --quantization nf4 \
  --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
  --adapter-path artifacts/adapters/ce128-seed0

uv run --no-sync s1 benchmark examples/benchmarks/smoke.jsonl \
  --backend gemma --device cuda --quantization nf4 \
  --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
  --adapter-path artifacts/adapters/ce128-seed0 \
  --warmup 2 --output artifacts/nf4-smoke.json

uv run --no-sync s1 serve --backend gemma --device cuda --quantization nf4 \
  --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
  --adapter-path artifacts/adapters/ce128-seed0 --host 127.0.0.1 --port 8000
```

Replace `nf4` with `int8` or `none` to change the weight format. Omit
`--adapter-path` to evaluate the base model. `--precision` defaults to BF16 on
CUDA and FP32 on CPU; FP32 with INT8/NF4 is rejected. These inference flags are
rejected for Laya, HTTP, and uniform backends, and are not training options.
Set `S1_API_KEY` before `s1 serve` to require its bearer token.

The small smoke JSONL checks the request contract; its score is not the matched
128-case BoolQ accuracy. The full campaign uses the v2 runner and dataset
provenance described in [the Modal guide](modal.md#reproduce-the-october-1-matched-comparison).

## Existing Modal inference app

The app captures these settings when constructing its runtime image:

| Environment variable | Meaning |
| --- | --- |
| `S1_QUANTIZATION` | `none` (default), `int8`, or `nf4` |
| `S1_ADAPTER_PATH` | Existing adapter directory inside the remote container |
| `S1_PRECISION` | Optional `bfloat16` or `float32`; CUDA defaults to BF16 |
| `S1_MODEL`, `S1_REVISION` | Base checkpoint and pinned revision |
| `S1_GPU` | Modal GPU type; the existing app defaults to A100 40GB |

The adapter path below points into the mounted `gemma-unified-system-one`
volume. A local path is not uploaded by setting this variable.

```bash
export S1_QUANTIZATION=nf4
export S1_ADAPTER_PATH=/vol/runs/20261001-v2-live-04/training-full/checks/training-curve/shots-128-seed-0-ce
export S1_REVISION=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
export S1_GPU=A100-80GB

uv run --no-sync modal run apps/modal/unified.py --mode model-smoke \
  --output artifacts/modal-nf4-smoke.json

uv run --no-sync modal run apps/modal/unified.py --mode benchmark --backend gemma \
  --dataset examples/benchmarks/smoke.jsonl --output artifacts/modal-nf4-contract.json

# Temporary authenticated HTTP app; ends when the development session stops.
uv run --no-sync modal serve apps/modal/unified.py
```

The same settings apply to the existing `Server` when deploying with
`modal deploy apps/modal/unified.py`. Its HTTP authentication and lifecycle are
described in [the Modal guide](modal.md). `model-smoke` includes measured CUDA
memory; simple benchmark reports include the actual quantization configuration
and adapter path. The matched campaign records loading, quality, and HTTP load
memory separately.

Modal training rejects nondefault inference settings. Clear them before using
the existing BF16 training entrypoint:

```bash
unset S1_QUANTIZATION S1_ADAPTER_PATH S1_PRECISION
```

Training is not automatically changed to QLoRA. Keep the original base checkpoint
and saved adapter; the quantized inference wrapper rejects a merged full-model
save. Loading a model does not publish its weights or deploy an endpoint.

## Memory, accuracy, and logging

The [parameter-count receipt](validation/2026-10-01/quantization-sizing.json)
gives raw parameter floors of **13.02 GB for INT8** and **7.57 GB for NF4**, before
quantizer state, adapters, activations, CUDA workspace, and allocator reservation.
These are estimates from checkpoint shapes, not measured GPU requirements.
The strict decimal 12 GB INT8 cap cannot be met by this linear-only conversion;
the completed NF4 inference runs measure **7.61 GB model footprint** and
**8.34 GB peak allocated CUDA memory** across quality, media, and HTTP load,
for both original and recovered adapters. This misses a strict decimal 8 GB
peak cap. Allocator reservation and device-wide use are higher; an A100 result
does not certify an 8 GB GPU. See the [complete live comparison](comparison.md)
for all seeds, memory scopes, and latency.

bitsandbytes 0.50.2 reports its expected BF16-to-FP16 activation cast on every
INT8 linear forward. The runtime retains the first exact warning per process
and filters only repeats from that module's logger. Other dtypes, warnings,
and errors remain visible. This changes logging overhead, not tensor operations.
Earlier benchmark snapshots can include the repeated logging in their timings;
their measured latency must not be relabeled as a run with this fix.

Implementation reference: [Transformers 5.17.0 bitsandbytes integration](https://huggingface.co/docs/transformers/v5.17.0/en/quantization/bitsandbytes).
