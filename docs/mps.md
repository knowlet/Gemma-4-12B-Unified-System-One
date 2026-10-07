# Apple Silicon MPS inference

Unified inference resolves automatic devices in this order: CUDA, MPS, CPU.
The former implementation selected CPU on Macs and used FP32 even with explicit
`device="mps"`. Automatic MPS precision now uses BF16 on macOS 14 or newer,
otherwise FP32. CPU remains FP32. The constructor and CLI accept explicit
`dtype` and `attn_implementation` settings; backend reports record the resolved
settings. The historical `DecisionModel` engine is unchanged.

```bash
uv sync --extra inference --extra api
uv run --no-sync s1 decide --device mps --dtype bfloat16 \
  --attn-implementation sdpa examples/request.json
uv run --no-sync python scripts/benchmark_mps.py \
  --model google/gemma-4-12B-it \
  --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
  --device mps --dtype bfloat16 --attn-implementation sdpa \
  --dataset examples/benchmarks/mps.jsonl --warmup 1 --repeat 3 \
  --output artifacts/mps/bf16-sdpa.json
```

Run configurations serially in separate processes so they do not compete for GPU
memory or compute. Compare `--attn-implementation eager` or explicit FP32 only
when memory allows. **FP16 is experimental: the actual 12B checkpoint failed the
image workload with an invalid probability distribution**, although the tiny
model and text cases passed. It is therefore never selected automatically.
Reduced precision changes rounding and
can change candidate probabilities; inspect distributions and decisions, not just
speed. BF16 preserves the base checkpoint's native weight format. No fast-math,
CPU fallback, or memory-limit override is enabled by the runtime.

## Measurement contract

The profiler resolves Hub revisions before loading. For each case it warms up
that shape, measures repeated calls to the public `predict`, then performs extra
forwards to measure preparation, backbone, and projection/softmax/CPU transfer.
All device timings synchronize before and after the measured operation using
[PyTorch's device API](https://docs.pytorch.org/docs/2.8/mps.html).
Phase synchronization adds overhead, so those times are diagnostics and are not
summed to claim end-to-end latency. Model loading is recorded separately.

Reports include raw latency samples and answer distributions, processor-expanded
token lengths, question counts, serial requests/decisions per second, environment
settings and memory snapshots. MPS current/driver allocation snapshots are **not
peak memory**. Errors retain their stage and partial measurements and cause a
nonzero exit. CPU fallback is visible in the environment metadata if a caller
enables it externally.

`examples/benchmarks/mps.jsonl` extends the three original smoke cases with a
synthetic red image, a 0.1-second 440 Hz tone, mixed media, a state repeated 64
times, and 16 repeated questions. It is a reproducible performance workload,
not a representative accuracy dataset. Media is synthetic; the retained golds
refer to the original text questions. Repeated questions do not establish
independent decision quality. Generation tokens/second is not measured because
S1 does not generate tokens.

Small randomly initialized Gemma 4 Unified models can check implementation
correctness, but their latency must never be reported as 12B performance.
Use held-out domain data for calibration and quantization acceptance.

## Measured on 2026-10-02

Apple M1 Max, 64 GiB unified memory; Torch 2.8.0, Transformers 5.17.0, the pinned
official 12B checkpoint above, temperature 1.0. All workloads run serially with
one warmup and three measured requests each. Times below are per-case median
milliseconds, excluding model loading and warmup.

| Workload | Expanded tokens | BF16 SDPA | BF16 eager | BF16 SDPA, prefer Metal |
| --- | ---: | ---: | ---: | ---: |
| Billing text, 2 questions | 86 | 683 | 701 | 705 |
| Technical text, 2 questions | 90 | 684 | 705 | 684 |
| Score text, 1 question | 60 | 488 | 488 | 491 |
| Image, 2 questions | 344 | 2053 | 2205 | 2241 |
| Audio, 2 questions | 91 | 690 | 692 | 700 |
| Image + audio, 2 questions | 349 | 2280 | 2139 | 2222 |
| Repeated longer state, 2 questions | 716 | 4505 | 4486 | 4711 |
| 16 questions | 551 | 3699 | 3553 | 4402 |

All three BF16 configurations completed all eight cases and 24 measured requests
(87 question decisions). First-sample probabilities were identical across these
configurations on all 29 distinct case/question positions. Eager's total measured
time differed by less than 1%; this small run does not establish a meaningful
advantage. `PYTORCH_MPS_PREFER_METAL=1` was about 6.6% slower overall in this run,
so it is not enabled. Keep the default SDPA and native BF16.

The backbone dominates measured latency. For short text, preparation took about
2.6–3.1 ms and selected-head projection/softmax/transfer about 1.1–1.7 ms. Cache or
tokenizer micro-optimizations would address little of the observed runtime.
BF16 weight allocation after load was **23.919 GB** (decimal); the final driver
allocation snapshot was **24.810 GB**, not a peak. FP32 weights alone require
approximately twice the storage by dtype size; a full FP32 12B performance run
was not performed, and no CPU-to-MPS speedup factor is claimed.

FP16 completed three text cases (5 decision positions, maximum probability drift
0.03296 versus BF16) then failed the image warmup with `ValueError: invalid
probability distribution`; subsequent cases were unattempted. The first failing
operator has not been isolated. Do not interpret its faster text timings as a
valid multimodal optimization. Tiny-model tests alone did not reveal this failure.

Cold latency matters: an initial BF16 smoke request took 26.5 seconds, and the
later eager profile's first warmup took 19.1 seconds. Subsequent per-shape warmups
were much shorter. Loading and first-use costs depend on file/cache state and are
not included in the table. Three samples per case and synthetic inputs are a
local performance check, not a production capacity or accuracy claim.

Raw reports and comparison metadata are in
[`validation/2026-10-02/summary.json`](validation/2026-10-02/summary.json).
