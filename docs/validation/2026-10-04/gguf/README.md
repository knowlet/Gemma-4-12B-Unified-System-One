# GGUF conversion, evaluation and publication — October 4, 2026

The trained merged BoolQ specialist is publicly available as a Q8_0 language
model and an F16 vision/audio projector. Both GGUF files total 12,791,679,008
bytes. The [runtime and reproduction guide](../../../gguf-release.md) gives
the pinned download, build, decision and evaluation commands.

| Identity | Immutable value |
| --- | --- |
| Public package | [GGUF snapshot](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-GGUF/tree/c418d37a17689fae554f7fc7d78db05ff7d52cfb) |
| Core weights/files revision | `5940bdbe292b33ac86450093a528d39b13c3a8f4` |
| Final card/manifest revision | `c418d37a17689fae554f7fc7d78db05ff7d52cfb` |
| S1 runtime/evaluator revision | `bcf829acc111fd8054987b1519b8181835f1dfed` |
| llama.cpp revision | `5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd` |
| Source BF16 weights revision | `66626de5cbcf8c5fecb8e9b58710805a02a42f67` |
| Conversion receipt SHA-256 | `464874bc51e8ee21ab56f34a4bf7d22b1fb83299a277cbcf966b42c5fc7802b7` |
| Evaluation report SHA-256 | `c812265d0f4b62cbf69a774c728c0c59d1912f259993ca78c38a04f51bfa5fbb` |
| Final artifact manifest SHA-256 | `1646a81f062fa3fe5727c835de569b669f288bd646749b2c3b96426bff4bf8f0` |

## Conversion and population evidence

- [Conversion manifest](conversion-manifest.json) records the direct Q8_0/F16
  commands, source checkpoint hashes, tools and converted file inventory.
- [Tensor inventory](tensor-inventory.json) records source and native tensors,
  vision/audio coverage and proportional RoPE structural checks.
- [Evaluation report](evaluation.json) records all 256 calibration, 256 fresh
  BoolQ and 52 historical media cases, with 100% case/decision coverage.
- [Independent audit and cross-runtime comparison](cross-runtime-comparison.json)
  refits temperature, recomputes metrics, checks exact IDs/labels/token slots,
  validates runtime bytes, and compares the complete BF16/MLX populations.
- [GGUF calibration](s1_config.json) binds its own fitted temperature to the
  conversion, population and runtime identities. The unchanged
  [BF16 calibration](base_s1_config.json) is retained separately.

| Population | Correct / scored | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh BoolQ | 231 / 256 | 90.23% | 0.2722 | 0.1569 | 0.0466 |
| Historical media | 33 / 52 | 63.46% | 1.1118 | 0.4363 | 0.1923 |
| MNIST images | 28 / 32 | 87.50% | 0.5212 | 0.2086 | 0.2075 |
| FSDD audio | 5 / 20 | 25.00% | 2.0568 | 0.8008 | 0.1993 |

The calibration-only fit selects temperature 2.7830344470383452 from native
GGUF logits. NLL changes from 0.5385953545 to 0.2904123939 in sample. That grid
value happens to equal BF16's; it was independently fitted. Fresh BF16/GGUF
argmax agreement is 255/256, with maximum candidate-probability difference
32.89 percentage points. Media agreement is 47/52, with maximum difference
43.27 points. Numerical parity and general model superiority are not established.
Media is a prior regression population; BoolQ-only fitting does not establish
image/audio calibration. Image/audio rows partition the 52 media cases.

## Native execution and code checks

- [Native smoke outputs](native-smoke-output.jsonl) and [log](native-smoke.log)
  cover the eight synthetic text, image, audio, mixed and multi-question shapes.
- [Boundary receipt](boundary-smoke-receipt.json), [inputs](boundary-smoke-inputs.jsonl),
  [outputs](boundary-smoke-output.jsonl) and [log](boundary-smoke.log) cover four
  additional cases. The first-slot logits in an 8,481-token, two-decode-call
  request match its identical 49-token prefix in that check. The receipt predates
  final conversion-metadata regeneration; source/binary hashes match the frozen
  implementation. Full evaluation binds the final package identity.
- [Calibrated interface receipt](calibrated-smoke-receipt.json),
  [requests](calibrated-smoke-inputs.jsonl), [answers](calibrated-smoke-output.jsonl)
  and [log](calibrated-smoke.log) verify all eight public requests and 29 typed
  answers after fitting the GGUF sidecar. These are execution checks, not task
  accuracy measurements.
- [Code validation](code-validation.json), [test log](full-pytest.log) and
  [package build log](package-build.log) record 957 passing tests, three optional
  skips, Ruff and successful wheel/source builds.
- [Isolated build reproduction](build-reproduction.json) and
  [log](build-reproduction.log) verify the public model card's minimal native
  library/helper recipe in a pristine checkout. The helper's `--help` succeeds;
  this build adds no GPU inference or quality measurements.
- [Hardware](hardware.json) and [full evaluation log](evaluation.log) record
  Apple M1 Max, 64 GiB installed memory and 49/49 model layers offloaded to Metal.
  Buffer logs are not peak RSS or minimum deployment memory measurements.

The native runtime projects the full vocabulary before selecting legal
candidates, clears request cache between calls, and never generates answers.
One decision sequence can require several native prefills. Its recorded timing
includes transport, HF preprocessing and native decoding; CUDA/MLX timing scopes
differ and no cross-runtime speedup is inferred.

## Publication verification

The [combined publication status](all-publication-status.json) records verified
BF16, MLX and GGUF packages while preserving their individual receipts.

[Preupload verification](preupload-verification.json) rehashes all 15 core files
after full evaluation. The [core upload receipt](hub-gguf-weights.json) binds
their immutable Hub revision. The [publication receipt](hub-gguf-publication.json)
verifies all 17 final files against remote LFS SHA-256/size or Git blob identities,
anonymous resolve access for both GGUF files, and fresh public downloads of the
card, manifest and calibrated sidecar. [Publication log](publication.log) records
the completed final verification.

The exact [published card](model-card/README.md), [artifact manifest](model-card/artifact-manifest.json),
[LICENSE](model-card/LICENSE) and [NOTICE](model-card/NOTICE) are retained byte
for byte. The final manifest hashes every packaged file except itself and the
Hub-managed `.gitattributes`. Verification did not redownload the complete
weights or run remote inference. Historical BF16/MLX receipts remain unchanged.
