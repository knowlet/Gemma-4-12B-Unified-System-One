# Runtime optimization — October 7, 2026

The project now includes the remote release retry/provenance corrections from
`3a183fa339ac299b1a9e522a06c05d6704173597`, preserving the local optimization
commit `1bc818be929c2545ebf65aa76ed6775c8fdc858c`.
The remote release branch and `origin/main` at `ca35708` had identical trees at
the update. Historical model weights, calibration sidecars and release receipts
were not rewritten.

## GGUF output reservation

The S1 native contract permits at most 64 ordered answer slots. Previously,
leaving `llama_context_params.n_outputs_max` at zero reserved outputs for the
entire 8,192-token batch. The optimization sets the reservation to the existing
64-slot limit. It retains the full-vocabulary projection, exact input tokens,
all answer slots, context 16,384, batch/microbatch 8,192, Flash Attention and
request cache clearing.

With the pinned llama.cpp revision
`5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd` on this M1 Max/64 GiB Mac:

| Logged allocation | Original | Output limit 64 |
| --- | ---: | ---: |
| Metal compute buffer | 8,432.00 MiB | 2,192.28 MiB |
| CPU compute buffer | 752.34 MiB | 752.31 MiB |

The Metal compute-buffer reduction is 6,239.72 MiB (6.09 GiB, 74.00%). These
are native buffer allocations, not peak process memory, deployment minimums,
or reductions in model weight storage. Model/KV buffers retain their settings.

The experiment runs original → optimized → original again, serially, with the
same CPU-prepared gold-free payloads. Eight synthetic text/image/audio/mixed
requests are warmed separately and measured three times. Four historical
boundary requests and an additional 64-question request are each warmed and
measured once. The long boundary contains 8,481 tokens and preserves an early
answer slot across two native text decodes. First-request startup is retained
separately. The timing scope is native persistent scoring, including media
projection, excluding CPU HF preparation.

Final equivalence results and reproducible evidence are recorded in the
[experiment index](validation/2026-10-07/gguf-memory/README.md).
This is a bounded runtime equivalence experiment, not a new full-population
quality/calibration release. Timings vary substantially across the two original
runs; no accepted throughput or latency improvement follows from these samples.

## Corrections needed for optimization experiments

- Batch API responses must be lists with exactly one result per input request.
  Missing, excess and malformed responses now produce HTTP 502 instead of a
  partial successful response or an unhandled error.
- Gemma batching retains each request's complete ordered multi-question prompt
  (`independent=False`). Splitting questions into independent prompts changed
  the context relative to the singleton API. A tiny real tensor-model regression
  compares full input IDs, answer slots, option counts and probabilities.
- An image-token research override updates both HF resizing/placeholder
  preparation and native projection. Different budgets require evaluation mode
  and matching fresh captures; the published calibration does not qualify a
  different image budget. Supported HF budgets are 70, 140, 280, 560 and 1,120.
- GGUF conversion records verify every supported quantization label against
  actual `general.file_type`, including the F16 projector. K-quants reject
  direct-conversion claims; model/projector filenames must be distinct local
  GGUF basenames. Structural receipts do not establish quality or prove that
  declared conversion commands were executed.
- The provenance-test virtual environment symlinks the real Python executable
  on macOS, retaining its dynamic-library location and separate environment
  identity. Tests still detect changed dependency bytes, unrecorded modules,
  lockfile changes and missing distribution files.

## Existing K-quant screening

The local October 6 reports at source `1bc818b` already contain 256 calibration,
256 test and 52 media cases for each K-quant. The
[arithmetic audit](validation/2026-10-07/gguf-memory/k-quant-screening.json)
rechecks their recorded IDs, slots, gold labels, probabilities and accuracy.
These are historical executions, distinct from the new output-reservation
experiment and the stricter current runtime qualification.

| Format | GGUF weights including projector, decimal GB | Test correct/256 | Images/32 | Audio/20 |
| --- | ---: | ---: | ---: | ---: |
| Published Q8_0 baseline | 12.79 | 231 | 28 | 5 |
| Q6_K | 9.91 | 229 | 28 | 5 |
| Q5_K_M | 8.67 | 231 | 27 | 6 |
| Q4_K_M | 7.50 | 225 | 27 | 3 |

The earlier local note mixed GB and GiB for the smaller formats. The exact
audited sizes are 9,908,054,048 / 8,669,300,768 / 7,503,415,328 bytes, respectively;
these are storage sizes, not runtime memory requirements. Q5 preserves the
test/media totals on this small population, but its reported timing was slower
and image/audio errors changed. It remains a candidate for a matched comparison;
these reports do not establish general superiority or qualify a new publication.

## Next MLX experiments

The selected MLX path already projects only 52 tied-head rows and uses native
full/sliding attention masks. Its next useful experiments remain caching
immutable selected rows, replacing component-level modality scatter with
token-position scatter, then compiling a pure tensor core after deriving
modality metadata on CPU. These are hypotheses, not measured speedups in this
update. The complete multimodal verifier limitations documented in
[the review status](pr5-review-status.md) remain separate.

The [current shape/storage receipt](validation/2026-10-07/gguf-memory/mlx-sizing.json)
checks the existing test capture's 150 distinct sequence lengths and the Q8
safetensors headers without model inference. MLX's
[compilation documentation](https://ml-explore.github.io/mlx/build/html/usage/compile.html#shapeless-compilation)
explains that input shape changes normally recompile functions; shapeless mode
also requires removing shape-dependent Python assumptions. Preserve the native
40 sliding and 8 full-attention layer masks and pass slot/mask values explicitly.

MLP tensors comprise 70.75% of the recorded Q8 MLX payload. An MLP6/attention8/
embedding8 experiment retaining media precision could save approximately 2.12 GB
of packed payload, an arithmetic estimate. No new MLX quantized checkpoint was
converted or evaluated in this update. Any such candidate needs independent
calibration and complete population accounting before publication.
