# Jev-Omni matched benchmark — October 7, 2026

Jev-Omni completed the shared **128-case BoolQ benchmark, 52 native media cases,
and 768 HTTP load requests**. Its BoolQ accuracy was **87.50% (112/128)** and
native media accuracy was **57.69% (30/52)**.

These results use the original fixed 128-case comparison set. They must remain
separate from the trained Unified release's fresh 256-case evaluation.

| Model / format | BoolQ accuracy | Peak CUDA allocated | Mean BoolQ latency |
| --- | ---: | ---: | ---: |
| Gemma base BF16 | 86.72% (111/128) | 24.45 GB | 87.67 ms |
| Gemma CE-128 BF16, three fixed seeds | 91.15% mean | 24.50 GB maximum | 127.49–155.92 ms |
| Jev-Omni BF16 | **87.50% (112/128)** | **24.38 GB** | **81.29 ms** |
| Cloudflare Clef-27B BF16 | 89.84% (115/128) | 55.66 GB | 185.02 ms |
| Decider-2B BF16 | 88.28% (113/128) | 4.25 GB | 75.65 ms |

All rows use the same question identities and labels. Historical measurements
retain their original hardware and runtime: Modal supplied A100 80GB PCIe or
SXM variants, with Jev-Omni measured on A100-SXM4-80GB. GB is decimal; peak
allocation does not establish a minimum GPU capacity. Model setup is excluded
from serial latency, and runtime differences prevent treating timing changes
as model-only effects.

Jev-Omni minus Gemma base is **+0.78125 percentage points**, with a paired
source-group bootstrap 95% interval of **[−6.25, +7.8125] pp** (128 groups,
2,000 resamples, seed 20260930). The interval includes zero; no accuracy
advantage is established. These are descriptive screening comparisons without
multiple-comparison adjustment.

## Native media and HTTP load

| Jev-Omni measurement | Result |
| --- | ---: |
| MNIST images | 28/32 = 87.50% |
| FSDD audio | 2/20 = 10.00% |
| All native media | 30/52 = 57.69%; all 52 supported and valid |
| Media mean / p99 | 100.70 / 114.71 ms |
| Closed-loop C1 / C16 HTTP p99 | 133.79 / 1,504.21 ms |
| Successful HTTP requests | 768/768 |
| Requests exceeding the 1,000 ms SLO | 237/768 |

Media uses the original images and recordings, without transcription, captions
or an oracle. These digit-recognition subsets do not establish broad multimodal
quality. HTTP tests use one serialized model process over loopback, without WAN
latency, batching or result caching; high concurrency includes queueing.

## Immutable runtime and evidence

Checkpoint: `akhilaaa3/Jev-Omni` at
`5addda86ddee081a68fb067477ea100c221b8917`.

The native merged backbone uses BF16 and its stored decision head uses FP32,
with BF16 autocast and one forward per question. The recorded stack is PyTorch
2.10.0, torchvision 0.25.0 and Transformers 5.17.0; earlier comparison runs retain
their own package versions. The adapter computes float32 softmax over native
logits and passes the already-normalized 16 kHz float32 audio directly, avoiding
the publisher helper's temporary WAV/ffmpeg roundtrip. It uses no test fitting,
CUDA graphs, compilation or persistent prefix cache.

The adapter limits support to 20 options and one native image or up to 30 seconds
of audio, and rejects timestamped frames or contexts over 16,384 expanded tokens.
All archived quality cases were eligible. These local reference timings include
preprocessing and differ from the publisher's optimized H200 protocol.

- [Combined comparison summary](comparison-summary.json): 19 selected model
  configurations, 31 retained attempts and 15 paired comparisons.
- [Source/hash index](jev-omni-evidence.json): all 18 original Jev-Omni artifact
  files, totaling 1,240,010 bytes, copied unchanged after SHA-256 verification.
- [Campaign](jev-omni-campaign.json) and [model receipt](jev-omni-local/receipt.json).
- [BoolQ predictions](jev-omni-local/boolq/predictions.jsonl) and
  [manifest](jev-omni-local/boolq/run_manifest.json).
- [Media predictions](jev-omni-local/media/predictions.jsonl) and
  [manifest](jev-omni-local/media/run_manifest.json).
- [All load summaries](jev-omni-local/load-summary.json), with raw
  [C1](jev-omni-local/closed_loop-c1-n128.jsonl),
  [C4](jev-omni-local/closed_loop-c4-n128.jsonl),
  [C16](jev-omni-local/closed_loop-c16-n128.jsonl),
  [C64](jev-omni-local/closed_loop-c64-n128.jsonl),
  [fixed](jev-omni-local/fixed-c16-n128.jsonl) and
  [Poisson](jev-omni-local/poisson-c16-n128.jsonl) request records.

All 12 earlier source campaign hashes, 18 selected historical rows and 30 earlier
attempts were verified unchanged. The paired interval was independently replayed
from archived predictions and manifests and matched the combined summary exactly.
