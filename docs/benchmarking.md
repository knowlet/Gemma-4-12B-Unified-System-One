# Reproducible comparisons

Every backend receives the same `DecisionRequest`, keyed question ids and candidate
order. A JSONL row contains `id`, `split: test`, `request` and a `gold` map covering
every question. Choice golds are option keys, Score golds are level keys and Noul
golds may be booleans. The loader rejects duplicate ids and invalid labels.

`examples/benchmarks/smoke.jsonl` has three cases and five decisions. It checks that
the pipeline runs; its accuracy is not a model quality claim. Bring a representative
held-out dataset for a meaningful comparison, and keep training/calibration/test
splits separate. Benchmark-specific fine-tuned Laya checkpoints must be identified
as such, rather than compared as zero-shot baselines.

```bash
uv sync --extra inference --extra laya
uv run --no-sync s1 benchmark --backend uniform examples/benchmarks/smoke.jsonl \
  --output artifacts/uniform.json
uv run --no-sync s1 benchmark --backend gemma --model google/gemma-4-12B-it \
  --revision <model-commit> data/test.jsonl --warmup 1 --output artifacts/gemma.json
uv run --no-sync s1 benchmark --backend laya --model convaiinnovations/laya \
  --revision <model-commit> data/test.jsonl --warmup 1 --output artifacts/laya.json
uv run --no-sync s1 benchmark --backend laya --subfolder multilingual --max-len 8192 \
  data/test.jsonl --output artifacts/laya-multilingual.json
uv run --no-sync s1 compare artifacts/gemma.json artifacts/laya.json
```

An HTTP adapter can benchmark Jev or another compatible server. Supply its exact
endpoint, not a guessed base URL, and use an environment variable for the token:

```bash
S1_API_KEY=... uv run --no-sync s1 benchmark --backend http --name jev \
  --endpoint https://YOUR-SERVICE/v1/systemone data/test.jsonl \
  --output artifacts/jev.json
```

HTTP bodies use named `questions` and `criteria`. Responses may have a named answer
map or an answer list with ids. Choice and Score must provide a complete
`probabilities` map, and Noul must provide `noul`. Different providers may need a
small adapter for their specific wire format. Missing distributions, wrong labels,
transport failures and invalid values are errors; no probabilities are invented.
The adapter performs no retries or model fallback. `--supports-media` is only for
endpoints implementing this project's base64/sample-array contract. Laya rejects
media requests explicitly. Jev live access is not required for tests.

Laya's Score outputs use zero-based indices. Its adapter converts numeric rubric
keys to descriptions, then maps probabilities back to the original keys and
recomputes the expected numeric score. Four-decimal probability rounding is
normalized only within its mathematical rounding bound. Larger missing mass is an
error. Laya receives explicit `max_len`; its runtime can truncate to that limit,
so use comparable input lengths and report the selected context budget.

## Report contract

- Dataset SHA-256, exact case ids, backend/model/revision settings, Python/platform,
  timestamps and warmup configuration.
- Per-decision probabilities, golds, status, request latency and error type. Full
  request content and credentials are not copied into reports.
- Accuracy, multiclass Brier (sum over classes), NLL, 15-bin ECE, AURC, selective
  accuracy at 80%/50% coverage and Score MAE; also broken down by primitive.
- Overall successful coverage, unsupported/error counts and p50/p95 successful
  request latency. Latency includes preprocessing and inference (and network time
  for HTTP), excludes model loading and optional warmup. Failed calls are not in
  latency percentiles; consult coverage before comparing.
- `s1 compare` verifies dataset fingerprints and recomputes metrics on the exact
  successful `(case_id, question_id)` intersection. Each individual report remains
  available so missing coverage cannot disappear from the comparison.

Question failure currently marks the complete request as failed. A benchmark with
errors still writes its report, then exits nonzero. Unsupported modalities are
counted separately. Do not rank models using only aggregate accuracy when coverage,
fine-tuning data, context policy or calibration policy differs.
