# Jev Ultrafast browser evaluation

The browser campaign runs the original [Jev Ultrafast policy](https://github.com/browser-use/jev-ultrafast/tree/1231850a0bf1a0c0341fe408ef1668dbbfdfac46)
at `1231850a0bf1a0c0341fe408ef1668dbbfdfac46`. It replaces the decision transport
with S1 and Browser Harness's connection with an isolated Playwright CDP session.
The upstream snapshot reader, operation/target questions, freshness checks and
action executor retain their original source. No personal Chrome profile is used.

This is a public local DOM integration suite. It exercises actual Chromium actions
and page transitions, but does not establish reliability on external websites,
official JevBench scores, sealed performance, or superiority over market models.
The existing finite-state workflow replay remains a separate evaluation.

The bridge serializes structured descriptions without rewriting them. A head with
one observed target returns that sole target with probability 1 and a trace flag;
it does not run inference or invent an alternative. Heads over S1's 52-candidate
limit fail explicitly without truncation. Only the upstream-selected operation's
matching target can execute. Text generation uses a separate optional helper and
the original upstream `TEXT_VALUE` instruction; the six initial tasks do not
require data entry, so their success cannot establish text-helper coverage.

## Reproduce

The local published checkpoint must contain a valid `artifact-manifest.json`,
including weight, processor, tokenizer and calibration hashes. The runner hashes
every listed file before loading it and never overwrites a run directory.

```bash
git clone https://github.com/browser-use/jev-ultrafast artifacts/benchmark-tools/jev-ultrafast
git -C artifacts/benchmark-tools/jev-ultrafast checkout 1231850a0bf1a0c0341fe408ef1668dbbfdfac46
uv venv --python 3.12 artifacts/browser-env
VIRTUAL_ENV="$PWD/artifacts/browser-env" uv sync --active --inexact --frozen --extra inference --extra api
uv pip install --python artifacts/browser-env/bin/python \
  'playwright==1.64.1' 'browser-harness==0.1.13' 'httpx[http2]==0.28.1'
artifacts/browser-env/bin/python -m playwright install chromium
artifacts/browser-env/bin/python scripts/run_ultrafast.py \
  --upstream artifacts/benchmark-tools/jev-ultrafast \
  --model artifacts/release/checkpoint --device mps \
  --output artifacts/ultrafast/your-unique-run \
  --profiles causal_multislot,independent --repeats 3 \
  --max-decisions 12 --task-timeout 90 --native-text-helper
```

Use `--profiles uniform --repeats 1 --tasks dev-reading-choices --max-decisions 3`
for an inexpensive transport/error-path diagnostic; its expected failure to
complete the task is not model performance evidence. Exit 0 requires every
attempt to meet the independent goal predicates within the time budget; exit 1
preserves incomplete/failed campaign evidence.

The pinned task manifest freezes three development and three reserved transfer
goals before inference. Transfer tasks must not guide subsequent tuning. Once
results influence a change, use new unseen tasks for its generalization check.
The verifier receives the task's expected DOM fields; the policy receives only
the natural-language goal, current upstream page observation and action history.
No expected DOM predicate or fixed action sequence enters the model request.

Profiles share the same loaded weights and text helper. First-profile order
alternates across tasks/repeats. Model load time is recorded separately, with no
discarded first-inference samples. This reduces order bias but does not eliminate
kernel warmup or thermal effects. Three repeats of six tasks are descriptive
screening, not evidence of statistical superiority.

## Evidence and success

Each attempt stores the complete observed pages, original decision payloads,
mapped S1 requests, raw answers, normalized probabilities, singleton heads,
backend errors, executed actions, fresh final DOM, final screenshot, CDP log and
Playwright trace. The run archives executed local sources, dependency versions,
checkpoint hashes, Chromium version, hardware and resource limits. Credentials
are excluded. A screenshot failure is recorded separately from DOM verification.

`verified_success` requires `goal_observed` and `within_budget`. Agent `DONE`
alone has no effect on the verifier. False DONE, blocked decisions, exceptions and
budget exhaustion remain in the full attempted denominator. Time limits are
checked between commands: in-process inference is not forcibly interrupted and
any overrun is a failed budget gate. The decision-cycle cap includes stale-page
retries. Browser initialization failures preserve partial traces and close owned
contexts before the next task.

`policy_wall_ms` includes browser initialization and the agent loop.
`wall_ms` and its p50/p95 additionally include final verification, screenshots and
cleanup. Actual backend invocations, text-generation calls and backbone forward
counts are separate; one independent-mode request can require multiple forwards.
Cost remains null because local compute billing is not measured.

## Two JevBench regressions

Optional `--heaven-root` and `--coherence-root` run both pinned suites after the
browser attempts using each still-loaded profile. The first root must be
`fstandhartinger/jevbench@bb05a335bc809e61b20c0f745d25499a82b326fc`; source and
dataset files are verified by `configs/benchmarks/jevbench-public.json`. The
second must be a clean `JevBench/jevbench@e18733694623aa93058e279c3534f8b8e2edefa3`.
Keep their distinct public accuracy and strict metamorphic coherence reports;
neither is an official submission. No calibration fitting occurs in this runner.

The 2026-10-28 attempt to rerun the existing A100 campaign was rejected before
job creation because the Modal workspace exceeded its spend limit. Local MPS
measurements must retain their hardware/runtime identity and cannot substitute
for a controlled A100 latency comparison.
