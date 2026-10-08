# NVFP4 release validation — October 7, 2026

The trained Unified model was exported to packed NVFP4, restored from the
exported files, calibrated on 256 disjoint calibration cases, and evaluated on
all **256 fresh BoolQ, 128 historical BoolQ and 52 native media cases**. A BF16
reference ran in the same process on the same NVIDIA B200 and package stack.

**Published and publicly verified on October 7, 2026:**
[Hugging Face revision `5c4f3c2cb5de2be66b9e51160613e868b33c311f`](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-NVFP4/tree/5c4f3c2cb5de2be66b9e51160613e868b33c311f).
The [publication receipt](publication.json) verifies all 48 files against the
[preupload inventory](preupload-verification.json): 8,970,264,904 total bytes,
including 8,932,247,864 bytes of weight shards. The package includes the exact
runtime wheel, model card, calibration/evaluation evidence, frozen datasets,
and Jev-Omni's matched benchmark. The [published model card](model-card/README.md)
records its installation instructions and runtime limitations.

A fresh B200 subsequently loaded the immutable public Hub revision through
the normal S1 loader and executed native text, image and audio requests.
All three agreed with the validated release within 1e-6; the largest observed
candidate-probability difference was **3.80205e-8**. The separate
[public inference receipt](hub-inference.json) records the revision and cases.

## Same-B200 quality comparison

| Trained model | Fresh BoolQ, 256 | Historical BoolQ, 128 | MNIST, 32 | FSDD, 20 | All media, 52 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 230/256 = 89.84375% | 118/128 = 92.1875% | 28/32 = 87.50% | 4/20 = 20.00% | 32/52 = 61.5385% |
| NVFP4 | 227/256 = 88.671875% | 114/128 = 89.0625% | 28/32 = 87.50% | 3/20 = 15.00% | 31/52 = 59.6154% |

Every expected case completed successfully, with no warmup errors. NVFP4's
fresh-test NLL is **0.300083**, versus **0.273379** for BF16. The BF16 source keeps
its published temperature **2.783034447**; NVFP4 uses **2.518551961**, fitted only
on the 256 calibration cases. Neither test population was used for fitting.
These native image/audio measurements use original MNIST images and FSDD
recordings, without oracle text, captions or transcription.

| NVFP4 minus BF16 | Accuracy difference | Paired 95% source-group interval |
| --- | ---: | ---: |
| Fresh BoolQ | −1.171875 pp | [−3.90625, +1.5625] pp |
| Historical BoolQ | −3.125 pp | [−6.25, −0.78125] pp |
| Native media | −1.92308 pp | [−15.62653, +9.80392] pp |

These are descriptive, unadjusted bootstrap intervals using seed 20260930 and
2,000 resamples. Whole source groups are resampled: 256 fresh groups,
128 historical groups and 38 media groups, including six FSDD speaker groups.
The historical-text interval is entirely negative in this screening comparison;
the fresh-test interval does not establish equivalent quality. Digit-recognition
subsets do not establish broad multimodal quality.

## Same-B200 latency and memory

Latency below is computed directly from archived per-request records. It includes
preprocessing, inference and returned probabilities, excludes model loading and
one warmup request per population, and is not an HTTP service-load measurement.
Each request contains one question. Percentile estimates from these sample sizes
are screening measurements.

| Population | BF16 mean / p99 | NVFP4 mean / p99 |
| --- | ---: | ---: |
| Fresh BoolQ, 256 | 37.768 / 57.946 ms | 45.873 / 51.579 ms |
| Historical BoolQ, 128 | 32.946 / 43.797 ms | 46.040 / 49.654 ms |
| Native media, 52 | 42.578 / 50.295 ms | 52.624 / 57.451 ms |

| Quantity | BF16 | NVFP4 |
| --- | ---: | ---: |
| Registered model tensor footprint | 23.919463522 GB | 8.932036994 GB |
| Peak CUDA allocated | 24.187527168 GB | 9.142883840 GB |
| Peak CUDA reserved | 24.622661632 GB | 9.770631168 GB |
| Last observed device-wide usage | 25.450708992 GB | 10.602872832 GB |

GB is decimal. NVFP4 reduces measured GPU allocation but has higher mean request
latency in each tested population on this B200. It does not meet an 8 GB
allocation limit. Peak allocation alone does not establish minimum GPU capacity.

The GPU peak is reset once before each model's ordered fresh-test, media and
historical-text evaluation; later split peaks cover all earlier splits and their
warmups. The table uses the final cumulative peak for each model. Host RSS is a
process-lifetime maximum across source loading, both models and export work;
it is not an independent per-model host-memory measurement. Device usage is an
instantaneous device-wide observation, not a per-process peak.

## Fair comparison with Jev-Omni

The [request identity audit](request-identity-audit.json) verifies **every case ID,
request SHA-256 and gold label** against the actual archived Jev-Omni manifests
and predictions. All 128 historical text cases and all 52 media cases match
exactly; there are no exclusions or unmatched rows. Their canonical dataset
hashes are respectively
`7bdb25dc3d8a0371bb73909fe55597de86f66e808bc42d1a5a1cb17d08f44d41` and
`7f3dff9c8fcfb164df5f8ca05b0fdaadbfb3a84812bfbe4c9b1906b6954d9f35`.

| Model and measured GPU | Historical BoolQ, same 128 | Native media, same 52 | Fresh release BoolQ, 256 |
| --- | ---: | ---: | ---: |
| Trained Unified BF16 — B200 | 118/128 = 92.19% | 32/52 = 61.54% | 230/256 = 89.84% |
| Trained Unified NVFP4 — B200 | 114/128 = 89.06% | 31/52 = 59.62% | 227/256 = 88.67% |
| Jev-Omni BF16 — A100-SXM4-80GB | 112/128 = 87.50% | 30/52 = 57.69% | Not evaluated |

NVFP4 minus Jev-Omni on the historical 128 cases is **+1.5625 pp**, with a paired
95% source-group interval **[−3.90625, +7.03125] pp**. This interval includes zero
and does not establish an accuracy advantage. The fresh 256-case NVFP4 score
cannot be ranked against Jev-Omni's different 128-case score. Hardware differs,
so the separate Jev-Omni latency and memory results must not be presented as a
controlled speed or memory comparison with these B200 results.

## Runtime, conversion and archived evidence

Source weights are the trained BF16 model
`knowlet/Gemma-4-12B-Unified-System-One` revision
`a66f836b56605039fe040f330180e336d19b3362`. The worker verified its recorded
source file hashes before loading. Both models used PyTorch 2.10.0,
Transformers 5.17.0 and SDPA on NVIDIA B200 (CUDA capability 10.0).

NVFP4 packs **328 language linear modules** using E2M1 weights, E4M3 block scales
with block size 16 and FP32 global scales. The packed module tensors occupy
6,812,468,512 bytes; retained higher-precision tensors bring the complete model
footprint to 8,932,036,994 bytes. Embeddings, multimodal components and the selected
decision head retain their recorded higher precision. The native kernel is
`kernels-community/nvfp4-gemm` revision
`a66348abcb8cc22da69f2b77a3f5d4e01748f495`. Prefill uses dynamic NVFP4 activations;
one- or two-row computation uses the kernel's W4A16 GEMV path. There is no CPU
offload. Three representative text/image/audio cases had bit-identical candidate
logits before and after packed save/reload.

The [evidence index](evidence-index.json) lists **17 original files / 1,352,280 bytes**
with sizes and SHA-256 hashes. Package sidecars were verified against the conversion manifest;
conversion and evaluation files also matched separate read-only downloads from
the completed Modal volume. Every JSON file was parsed completely and its size
and modification time checked across the read before copying. All six individual
evaluation reports equal their embedded versions in `evaluation.json`.

- [Evaluation](evaluation.json), [conversion manifest](conversion-manifest.json),
  [packed tensor manifest](nvfp4_manifest.json), [S1 calibration config](s1_config.json)
  and [completion receipt](receipt.json).
- Original BF16 reports: [fresh text](evidence/bf16-test.json),
  [historical text](evidence/bf16-regression_text.json),
  [media](evidence/bf16-media.json).
- Original NVFP4 reports: [fresh text](evidence/nvfp4-test.json),
  [historical text](evidence/nvfp4-regression_text.json),
  [media](evidence/nvfp4-media.json).
- [Calibration evidence](evidence/calibration.json) and
  [save/reload parity](evidence/reload-parity.json).
- [Packaged runtime verification](runtime-verification.json): the exact distribution
  wheel reproduced all 436 cases, with maximum absolute candidate-probability
  difference 0.0; its conversion/evaluation bindings and local wheel SHA-256
  were independently verified.
- [Independent calculations](independent-comparison.json), including full metric
  tables, procedure, seed, resample count, intervals and source hashes.
- [Exact request matching](request-identity-audit.json) with original case,
  group, question and request identities plus labels.

Weights and tokenizer files are intentionally excluded from this repository
metadata archive. Their recorded identities remain in the conversion manifest;
this evidence archive alone is not a loadable checkpoint. The source files are
from `gemma-unified-system-one:/nvfp4/20261007-nvfp4-02`.
