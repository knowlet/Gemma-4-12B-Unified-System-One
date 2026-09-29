# Architecture and implementation scope

The first release implements the decision path described in the linked Gemma 4
discussion. It retains the upstream text-only engine for existing E2B checkpoints
and adds an independent Unified engine with the complete multimodal model.

```text
DecisionRequest (state, typed questions, optional media)
  -> validate labels, levels, question ids and media
  -> AutoProcessor: native user turn and image/audio feature expansion
  -> append typed questions, recording one answer position per question
  -> Gemma4UnifiedModel: ONE forward, use_cache=False
  -> gather final hidden states at answer positions
  -> selected output-embedding rows A-Z/a-z (including bias and softcap)
  -> temperature, mask illegal options, softmax
  -> Choice / P(true) / expected Score + complete probability distribution
```

## Modules

- `contracts.py` owns the public, validated schema. Both a question map and an
  ordered list are accepted; the API always returns an answer map keyed by id.
- `unified.py` owns processor integration, model loading, candidate checks and
  projection. `model.model.language_model` alone would discard the multimodal
  embedders, so the complete `model.model` receives every processor tensor.
- `backends.py` provides Gemma, Laya, HTTP and uniform adapters. No adapter silently
  falls back to another model or changes a media request into a text request.
- `api.py` is shared by local Uvicorn and Modal. A lock serializes model access.
- `benchmark.py` stores per-decision distributions and recomputes comparisons on
  the exact common successful subset.
- `training.py` implements supervised CE + Brier, Choice order permutation and
  separate calibration. `s1 train` applies attention LoRA, then saves a merged
  checkpoint, processor and temperature. There is no new pretrained checkpoint.

## Decision semantics

`choice` maps runtime labels and descriptions onto checked, existing one-token
letters. No vocabulary resizing or random new embeddings are required.
`noul` uses false/true in that order. `score` uses ordered levels; numeric dictionary
keys retain their numeric values, while lists use zero-based indices. Score returns
both the expectation and the most likely level. `confidence` is max probability;
`margin` is the top-two difference. The legacy `/decide` demo used margin under the
name confidence; new API callers should use the explicit fields.

The probabilities are conditional on the supplied candidates. They are not the
full vocabulary's probability mass, and temperature 1.0 is an **uncalibrated**
baseline. Fit temperature on held-out calibration data before trusting confidence.

Multi-slot causal attention allows later questions to see earlier questions and
their unfilled answer prompts. Earlier positions cannot see later questions.
There are no generated prior answers, but the slots are not statistically
independent. Evaluate question-order sensitivity before relying on a multi-slot
accuracy or latency advantage.

## Multimodal input

`media` is a list of base64 images or normalized 16 kHz mono sample arrays:

```json
{
  "media": [
    {"type": "image", "data": "<base64 image bytes>", "timestamp_seconds": 1.5},
    {"type": "audio", "samples": [0.0, 0.1, -0.1], "sampling_rate": 16000}
  ]
}
```

Attach it to the state/questions contract in `examples/request.json`. Timestamped
images support sampled video frames, ordered as provided. Video-file decoding,
resampling and URL fetching are caller responsibilities. The service reads no
caller-specified local paths or URLs. Limits are eight media items, 16 megapixels
per decoded image and 30 seconds per audio item.

The processor expands native modality tokens before question tokens are appended.
Attention masks are extended with ones and modality masks with zeros. A request
that exceeds the configured total context is rejected, never silently truncated.
The initial engine supports one request per forward and up to 64 question slots,
with 2–52 options each. Cross-request batching, shared-prefix cache and hierarchical
large-choice selection are future optimizations, not claimed features.

## Training

Use the same JSONL schema as evaluation, with explicit `split: train` and
`split: calibration`. Gold is keyed by semantic label, so shuffling options cannot
change the answer. Training and calibration ids and identical request bodies must
be disjoint. The test split is never used to fit temperature.

```bash
uv sync --extra inference --extra train
uv run --no-sync s1 train --train-data data/train.jsonl \
  --calibration-data data/calibration.jsonl --output artifacts/checkpoint --steps 100
```

This sequential trainer is intended as a reproducible starting point. It does not
implement distributed training, automatic checkpoint resume, model selection,
or guarantee preservation of base multimodal task accuracy after fine-tuning.
Measure capability retention on independent image/audio data before release.

## Upstream migration

The Python import name remains `s1`; install it through uv before running scripts.
Root Modal scripts now live under `apps/modal/`. `make_report.py` lives under
`scripts/`; the shell demo client lives under `examples/`. Historical results,
media, licenses and demo code remain available. `s1/modal_common.py` now resolves
the common model runtime through `uv.lock`. Specialized historical game/data images
retain their own dependencies; they are not part of the new Unified runtime.

Legacy token truncation now obeys its exact budget. Oversized legacy Choice
evaluation uses the existing hierarchical `decide` path, without inspecting gold
labels. Direct prompt building refuses gold-dependent evaluation subsampling.
The hierarchy prunes candidates before reranking, so its probabilities are
approximate and the previous sampled scores are not directly comparable.

## Primary references

- [Google model card](https://huggingface.co/google/gemma-4-12B-it)
- [Transformers Unified model documentation](https://huggingface.co/docs/transformers/model_doc/gemma4_unified)
- [Transformers Unified source](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4_unified)
- [Laya model card](https://huggingface.co/convaiinnovations/laya)
- [Modal uv integration](https://modal.com/docs/reference/modal.Image#uv_sync)
