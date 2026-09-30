# Decision Benchmark v2

E0 is implemented: validated registries, offline capability preflight, deterministic
experiment identities and aggregate budget planning. The frozen v1 baseline is
`f8390676ae942699a71a85ae9a99cfcd7b9b3806`. Existing `s1 benchmark` and `s1 compare`
continue to use their v1 contracts. No model-quality results are implied by E0.

## Run the offline planner

From the repository root:

```bash
uv sync --locked --extra api
uv run --no-sync s1 eval preflight --profile smoke --enable-model uniform
uv run --no-sync s1 eval plan --profile smoke --enable-model uniform \
  --output artifacts/evaluation/smoke-plan.json
uv run --no-sync s1 eval plan --profile inventory \
  --output artifacts/evaluation/inventory-plan.json
```

Both commands perform the same offline validation and budget checks; the artifact's
`kind` distinguishes a preflight from a saved plan. They never import model runtimes,
download weights, resolve Hub revisions, contact endpoints, run inference or spend
the declared budget. `--enable-model` overrides a selected model's disabled flag for
this plan only; it does not edit the registry or authorize a future executor.

All shipped models are disabled. The inventory command therefore writes its report
and exits **1**. Exit **0** means at least one enabled model is ready and every
enabled model passed the checks. Exit **2** means malformed configuration, invalid
data, missing input files or an unknown reference. Disabled models remain visible
without blocking enabled models.

The smoke plan has 3 requests / 5 decisions per repetition, 3 repetitions, and 1
warmup request per repetition: **9 measured requests + 3 warmups = 12 requests**,
with 15 measured decisions and zero estimated compute/API cost for uniform. The
three-case dataset is a contract fixture, not a representative quality suite.

## Configuration contract

TOML uses Python 3.11's standard library and adds no dependencies. Omitted fields
represent unknown values where allowed; TOML has no `null` literal. Unknown keys,
duplicate IDs, missing model/suite references, negative budgets, nonfinite prices,
and incompatible calibration declarations are errors.

| File | Contents |
| --- | --- |
| `configs/benchmarks/models.toml` | Model identity, adapter, precision, execution/calibration policy, capabilities, cost ceiling, environment variable names |
| `configs/benchmarks/suites.toml` | Dataset path, track, information view and optional dataset fingerprint pin |
| `configs/benchmarks/profiles.toml` | Selected models/suite, repetitions, warmups, B/C, required capabilities, seed and aggregate budget |

Override paths with `--registry`, `--suites`, `--profiles`, and `--lockfile`. Dataset
paths are relative to the **suite configuration file**, even when invoked from a
different directory. Other CLI paths are relative to the working directory.

E0 loads the existing v1 JSONL `Case` contract and requires a nonempty test-only
dataset with unique IDs and valid gold labels. Case IDs and SHA-256 appear in plans;
request bodies, gold labels, media payloads and credential values do not. The
fingerprint preserves case, question and candidate order because order can affect
predictions. It is a hash of validated cases, not of the raw JSONL bytes.

The dataset contract still limits each request to 52 candidates per question,
64 questions and 8 media items. A declared larger model limit does not enable the
planned high-cardinality track. Group IDs, richer splits and soft targets belong
to the v2 dataset work in E1/E2.

## Eligibility and readiness

Each request is classified before any model execution:

| Eligibility | Meaning |
| --- | --- |
| `eligible` | Declared modalities, primitives, K and N cover the whole request |
| `unsupported` | A declared capability explicitly excludes part of the request |
| `unknown` | Required capability information is missing and there is no explicit exclusion |

Requests are atomic in this milestone. If one question is unsupported, the entire
request is unsupported. Counts retain all requests and all decisions in separate
denominators. Unknown support blocks an enabled model; an explicitly unsupported
subset remains visible and can be excluded before execution. A model with no eligible
requests cannot be ready. No eligibility state produces a zero quality score.

Capabilities describe the configured **adapter**, including its limits. Batch
support is `native`, `loop_emulated`, `none`, or `unknown`; a loop cannot satisfy
`require_native_batch`. B (independent inputs per batch) and C (concurrent requests)
are separate profile fields. N and K are inspected from each actual request.
`require_independent_questions` rejects causal multi-slot or unknown semantics.
`require_probabilities` requires a declared complete distribution. Uniform accepts
media for a sanity baseline; this does not claim media understanding.

Each selected model then has a readiness status:

- `not_run`: disabled, with all unmet conditions still listed.
- `blocked`: enabled but missing configuration, capability information or budget.
- `ready`: the offline declaration and budget checks passed.

`ready` and `can_execute` are planning results. E0 has **no executor**. They do not
validate installed inference extras, available GPU memory, decoded media, tokenizer
context length, credentials, endpoint reachability or the truth of a vendor claim.
Those require later runtime checks. Runtime failures must not retroactively change
the frozen eligible denominator to unsupported.

Gemma and Laya require a model ID and an immutable 40/64-character lowercase hex
revision. A missing processor revision inherits that model revision. HTTP requires
an operator-recorded provider version and an endpoint supplied through the named
environment variable. If `token_env` is declared, it must contain a nonempty value.
The planner verifies presence only. Endpoint URLs must be HTTP(S), without userinfo,
query credentials or fragments; only their SHA-256 is stored. Capability assertions
for Jev, Kev, Decider, AgentJev and SetFit intentionally remain unverified in the
starter registry. Their adapters/checkpoints are not implemented by this milestone.

## Budget and identity

`max_requests` and `max_estimated_cost_usd` apply to the sum of all enabled, otherwise
ready models. Warmups count against the budget on every repetition. Estimates count
logical requests, including every batch member, not Python calls or GPU forwards.
They assume no retries; a future retry policy must reserve extra budget.

`cost_per_request_usd` is an operator-supplied conservative ceiling covering the
planned input sizes and compute/API costs. Missing cost is **unknown**, including
for local models. Only uniform is intrinsically free. An explicit zero for another
adapter is an operator assertion. Decimal arithmetic avoids rejecting a budget
because of binary floating-point addition. Exceeding either aggregate limit blocks
all candidate models; the planner does not silently select a cheaper subset.
Disabled/otherwise blocked models have individual hypothetical estimates but do
not reserve aggregate budget.

Every cell's `experiment_id` hashes its effective model configuration, suite,
dataset fingerprint, profile, frozen baseline, source-code hash, dependency-lock
hash, installed relevant package versions, Python/platform, and endpoint hash when
configured. `plan_id` hashes the ordered experiment IDs. Timestamps and command
purpose are excluded so repeated equivalent planning is stable. Changes to candidate
order, seed, precision, revision, calibration, cost/budget, source or lock change the
identity. This is an E0 planning identity: no resume is implemented. Before actual
execution, the run manifest must additionally capture resolved runtime/model/data
artifacts, hardware/device, service region, tokenization and preprocessing policies.

## Remaining milestones

| Stage | Deliverable | Acceptance focus |
| --- | --- | --- |
| E0 — implemented | Registry, preflight, budgeted dry-run plans | Offline; unknown ≠ unsupported; deterministic identities; honest budgets |
| E1 | V2 adapters, text datasets, execution records | Gold excluded from adapter input; missing access recorded as not run |
| E2 | Actual choice vs standardized argmax, hard/soft/ordinal metrics, calibration, cluster bootstrap | Explicit denominators; no test fitting; hand-checkable reference metrics |
| E3 | Gemma G4 independent reference, N/K/B sweeps, order sensitivity | Separate causal and independent semantics; reference parity |
| E4 | Concurrency/open-loop load generation, timing and telemetry | Timeouts retained; quality-under-load and correct-within-SLO goodput |
| E5 | Native media, OCR/ASR alternatives, counterfactuals and retention | Content sensitivity and full preprocessing cost |
| E6 | Few-shot/SetFit and Gemma CE/Brier comparison | Matched support IDs, multiple seeds and locked tests |
| E7 | Closed-loop workflows and report | Task completion, false completion and cost per success |

The first usable comparative milestone remains E0–E3. Further work should preserve
separate generalist, supervised specialist, native multimodal, pipeline, systems
and workflow tracks. Local forward time and remote API latency must remain separate;
latency/N is amortized cost, not the time a user waits for an answer.
