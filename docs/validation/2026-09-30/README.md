# Implementation validation, 2026-09-30 (Asia/Taipei)

These are implementation smoke results, not representative quality benchmarks.
All live jobs ran in the user's Modal workspace and terminated after completion.
No persistent production deployment or model publication was performed.

## Local checks

- 47 tests passed with inference, training and API extras installed.
- Real randomly initialized Gemma4Unified forwards: text, image, audio and mixed
  processor fixtures, multiple answer slots, masking and full-logit parity.
- Attention LoRA parameter updates, CE+Brier training, disjoint calibration,
  processor/checkpoint saving and model reloading.
- API validation/authentication, Laya/HTTP contract conversion, error coverage,
  shared-subset metrics, CLI packaging and legacy high-cardinality regressions.
- Ruff lint/format, `uv lock --check`, source compilation and wheel/sdist build.
- Real official AutoProcessor/tokenizer metadata: 52 stable candidate tokens,
  and successful preprocessing of text, image and audio (no local 12B weights).

## Cloud evidence

| Run | Evidence |
| --- | --- |
| [CPU smoke](https://modal.com/apps/knowlet/main/ap-n3710yEW5QxvedQ7iHj3Om) | [34 passed, one inference module skipped](modal-cpu-smoke.json); captured before the final added tests |
| [Gemma 12B](https://modal.com/apps/knowlet/main/ap-0PmNs9nlse5FNsRqt3oUb9) | [Real text/image/audio forwards](modal-gemma-smoke.json), one pass per request |
| [Gemma fixture](https://modal.com/apps/knowlet/main/ap-7UH2PlSzHLVWxbJGoAvD9L) | [Five decisions](modal-gemma.json), 100% coverage |
| [Laya fixture](https://modal.com/apps/knowlet/main/ap-t4OfAv9PwyvCQY9EUzeAyZ) | [Same five decisions](modal-laya.json), resolved checkpoint and effective temperatures recorded |

Hardware: one A100 40GB per model job. Gemma checkpoint
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`; Laya checkpoint
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, runtime 0.3.21.
Both answered five of five fixture decisions correctly. The
[comparison](comparison.json) uses the exact successful decision intersection.
Three requests cannot establish reliable latency, calibration or accuracy rankings.
The multimodal smoke verifies execution and normalized outputs, not modality
ablation or broad visual/audio understanding.

Laya emitted a checkpoint-temperature warning: its `choice:11+` value was clamped
to 0.5 by the upstream SDK. This fixture has at most three candidates, so that
bucket was not exercised. The report includes effective calibration settings;
users evaluating larger choices should inspect that warning and recalibrate.

The first cloud import exposed a Modal relocation issue (`/root/unified.py` has
fewer parents than the local script). The entrypoint now resolves local paths only
on the client; the successful CPU and model runs above verified the correction.

Not exercised: live Jev credentials/endpoint, representative held-out multimodal
quality, full 12B fine-tuning, sustained concurrent serving, or production deployment.
