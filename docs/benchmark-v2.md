# Decision Benchmark v2

E0 and the E1 execution core are implemented: validated registries, offline capability
preflight, deterministic identities, aggregate budgets, reference adapter bridges,
grouped text fixtures and incremental run records. The frozen v1 baseline is
`f8390676ae942699a71a85ae9a99cfcd7b9b3806`. Existing `s1 benchmark` and `s1 compare`
continue to use their v1 contracts. The fixtures do not establish model quality.

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
this invocation only; planning does not edit the registry or invoke the executor.

All shipped models are disabled. The inventory command therefore writes its report
and exits **1**. Exit **0** means at least one enabled model is ready and every
enabled model passed the checks. Exit **2** means malformed configuration, invalid
data, missing input files or an unknown reference. Disabled models remain visible
without blocking enabled models.

The smoke plan has 3 requests / 5 decisions per repetition, 3 repetitions, and 1
warmup request per repetition: **9 measured requests + 3 warmups = 12 requests**,
with 15 measured decisions and zero estimated compute/API cost for uniform. The
three-case dataset is a contract fixture, not a representative quality suite.

## Execute and recompute a run

```bash
uv run --no-sync s1 eval run --profile text-smoke --enable-model uniform \
  --output artifacts/evaluation/text-smoke-run
uv run --no-sync s1 eval summarize artifacts/evaluation/text-smoke-run
```

`run` performs a fresh preflight and then executes the selected models. Its output
directory **must not exist**: previous results are never overwritten and resume is
not implemented. `summarize` only recomputes `summary.json` from local records.
The text fixture has 10 requests, 18 decisions and 5 translation groups; one warmup
makes 11 prediction calls. These are handwritten English/Traditional Chinese
contract cases, not a public benchmark conversion or representative evaluation set.

The E1 executor accepts **B=1, C=1** and complete distributions. It bridges the
existing Uniform, Gemma, Laya and compatible HTTP adapters. Other model families,
choice-only adapters, native batching, queueing and open-loop scheduling remain
future work. Execution never silently emulates a requested batch or concurrency.

For model runs, explicitly enable the relevant entry, pin its revision, declare
verified capabilities and costs, select a matching profile and install the existing
inference/Laya extras. `run` can load weights and call configured endpoints; the
example above invokes only the free local Uniform backend. A globally blocked plan
loads no adapters. Executor-specific unsupported settings are recorded per model
as `not_run`, while other eligible models can still complete.

| Adapter | Applied settings and limits |
| --- | --- |
| Uniform | Sequential, uncalibrated Python float64 probabilities; no model loading |
| Gemma | Causal multi-slot; explicit device; BF16 for CUDA, float32 otherwise; context limit rejects excess input; calibration is checkpoint temperature or explicit T=1 |
| Laya | Revision, optional subfolder, device and context limit; checkpoint calibration; precision is declared `provider` because the current wrapper cannot control it; context may be truncated |
| HTTP | Endpoint/token environment variables, `http_timeout_seconds`, native-media declaration; version/precision/calibration remain provider assertions |

A separate processor revision and unsupported runtime/calibration policies cannot
be silently ignored. Loaded local model revisions and reported precision must match
the plan before prediction. Laya limits/independence still need verification for the
chosen checkpoint. Endpoint compatibility is an operator responsibility, not a claim
that every Jev service implements this repository's wire contract.

Each adapter receives only a deep copy of `DecisionRequest`. Gold labels, group,
task, language and schema metadata remain in the evaluator. Mutations cannot affect
normalization or later repetitions. Runtime errors, including a late
`UnsupportedRequest`, retain their original eligibility. The runner does not retry
or cache results; provider-side caching is unknown. HTTP timeouts come from the existing client; local computation has
no forced deadline in E1. The seed is recorded only: this executor does not perform
stochastic support selection or claim control over provider randomness.

## Run artifacts and descriptive metrics

| Artifact | Contents |
| --- | --- |
| `run_manifest.json` | Effective plan, source/data/lock identities, model lifecycle, sanitized runtime telemetry, start/finish times |
| `requests.jsonl` | Warmup and measured requests, frozen eligibility, dispatch/readiness/termination times and status |
| `predictions.jsonl` | One row per measured decision, including failures/not-run cases; gold label, actual choice, standardized argmax, complete probabilities and Score expectation where available |
| `errors.jsonl` | Setup, prediction, timeout and cleanup error types, without exception messages or raw backend responses |
| `summary.json` | Recomputable counts, denominators, coverage, descriptive accuracy and successful-request p50/p95 |

Rows are flushed after each completed call. A handled interruption marks the
manifest `interrupted` and retains completed/in-flight records. Hard process or
machine termination can leave a `running` manifest or incomplete final record;
there is no recovery/resume protocol yet. Interrupted runs may be inspected with
`summarize`; `records_complete` and recorded/planned decision counts identify partial
data. The eligible denominator comes from the frozen plan, never from successful
rows alone.

`actual_choice_accuracy` uses the service's valid Choice, preserving tied maxima;
`standardized_argmax_accuracy` uses the first maximum in candidate order for all
primitives. `actual_label_accuracy` uses service Choice and standardized Noul/Score
labels. Score expectation and numeric gold are retained for E2's numeric metrics.
`operational_correctness` divides correct labels by all planned eligible decisions.
For a model with no attempted measured requests, quality and execution coverage are
`null`; disabled or unavailable models do not receive fabricated zero scores.

Warmups count toward the budget and have error records but are excluded from quality
and latency percentiles. Latency includes request copying, backend preprocessing,
inference/network and response validation; it excludes model setup. Failure elapsed
time is available from dispatch/termination fields, never substituted for successful
completion latency. These are sequential wall-clock observations, not GPU kernel
benchmarks or stable tail-latency estimates. Repetitions are not independent source
groups, and this milestone provides no confidence intervals or significance claims.

Exit **0** from `run` means all enabled models completed without setup, warmup,
prediction or cleanup errors. **1** covers errors, no execution or partial execution;
artifacts are still saved. **2** covers invalid input or an existing output directory.

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

The v2 loader accepts v1 JSONL and optional `group_id`, `task_id`, `language` and
`schema_id` fields, and recognizes development splits. Planning/execution still
require a nonempty test-only dataset with unique IDs and valid gold labels. Group
IDs default to case IDs when omitted; other metadata stays null. Case IDs and SHA-256 appear in plans;
request bodies, gold labels, media payloads and credential values do not. The
fingerprint preserves case, question and candidate order because order can affect
predictions. It is a hash of validated cases, not of the raw JSONL bytes. V1
fingerprints remain unchanged when the optional metadata is absent; supplied
metadata participates in the fingerprint.

The dataset contract still limits each request to 52 candidates per question,
64 questions and 8 media items. A declared larger model limit does not enable the
planned high-cardinality track. Soft targets and cross-split group audits remain
E2 work.

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

`ready` and `can_execute` are planning results. They do not
validate installed inference extras, available GPU memory, decoded media, tokenizer
context length, credentials, endpoint reachability or the truth of a vendor claim.
The E1 runner performs additional configuration/setup checks and records runtime
failures. Runtime failures must not retroactively change
the frozen eligible denominator to unsupported.

Gemma and Laya require a model ID and an immutable 40/64-character lowercase hex
revision. A missing processor revision inherits that model revision. HTTP requires
an operator-recorded provider version and an endpoint supplied through the named
environment variable. If `token_env` is declared, it must contain a nonempty value.
The planner verifies presence only. Endpoint URLs must be HTTP(S), without userinfo,
query credentials or fragments; only their SHA-256 is stored. Capability assertions
for Jev, Kev, Decider, AgentJev and SetFit intentionally remain unverified in the
starter registry. A compatible generic HTTP bridge is available; dedicated adapters
for Kev, Decider, AgentJev and SetFit are not implemented by this milestone.

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
identity. No resume is implemented. E1 also records resolved local revision/device,
precision and context policy when exposed by the existing adapter. Hardware SKU,
service region and detailed tokenizer/preprocessing telemetry remain necessary
before formal systems comparisons in E3/E4.

## Remaining milestones

E2 tools are now available:

```bash
uv run --no-sync s1 eval audit-splits data/train.jsonl data/calibration.jsonl data/test.jsonl
uv run --no-sync s1 eval compare artifacts/run-a artifacts/run-b --seed 0 --resamples 2000 --output artifacts/comparison.json
uv run --no-sync s1 eval calibrate artifacts/calibration-run --model gemma-g3 --output artifacts/temperature.json
```

Use a profile with `split = "calibration"` to collect calibration predictions.
Fitting refuses test runs, incomplete records and already domain-calibrated input.
To apply the artifact, set `calibration = "domain"`, `base_calibration`, the
`calibration_file` path (relative to the working directory), and its printed
`calibration_sha256`. The base model settings must match the fitting run. Evaluation
case IDs, group IDs and canonical request hashes must be disjoint from fitting data.
The probability power transform uses a disclosed 1e-12 floor and positive fitted T.

Optional `soft_gold` maps question IDs to complete target distributions, and
`critical_questions` identifies critical fields; neither is sent to adapters.
Summaries now contain sum-over-classes Brier, floored NLL, 15-bin reliability/ECE,
tie-averaged risk–coverage/AURC, NouL ROC-AUC/average precision/FPR/FNR, numeric Score
MAE, ordinal error and RPS normalized by K−1. Soft-target cross-entropy/Brier/KL and
teacher agreement are reported separately from hard-label quality. Undefined metrics
remain null. Missing distributions never become invented probabilities.

Task-macro accuracy requires task IDs; macro-F1 is calculated within task/schema/type
groups. NouL uses positive label `true` and strict P(true)>0.5, retaining false on a
tie. NLL discloses its floor and number of zero-gold-probability observations.
Bootstrap intervals resample source groups, keeping repeated decisions, questions
and translations together. Fewer than two groups are inconclusive. Pair comparisons
report both common-valid conditional and common-eligible operational differences,
plus full counts; intervals are descriptive and not multiplicity-adjusted.

| Stage | Deliverable | Acceptance focus |
| --- | --- | --- |
| E0 — implemented | Registry, preflight, budgeted dry-run plans | Offline; unknown ≠ unsupported; deterministic identities; honest budgets |
| E1 — core implemented | Reference bridges, grouped bilingual fixtures, execution records | Gold isolated; honest failure/coverage records; additional model families and representative datasets still pending |
| E2 — implemented | Hard/soft/ordinal metrics, calibration, split audit, paired cluster bootstrap | Explicit denominators; no test fitting; hand-checkable reference metrics |
| E3 | Gemma G4 independent reference, N/K/B sweeps, order sensitivity | Separate causal and independent semantics; reference parity |
| E4 | Concurrency/open-loop load generation, timing and telemetry | Timeouts retained; quality-under-load and correct-within-SLO goodput |
| E5 | Native media, OCR/ASR alternatives, counterfactuals and retention | Content sensitivity and full preprocessing cost |
| E6 | Few-shot/SetFit and Gemma CE/Brier comparison | Matched support IDs, multiple seeds and locked tests |
| E7 | Closed-loop workflows and report | Task completion, false completion and cost per success |

The first usable comparative milestone remains E0–E3. Further work should preserve
separate generalist, supervised specialist, native multimodal, pipeline, systems
and workflow tracks. Local forward time and remote API latency must remain separate;
latency/N is amortized cost, not the time a user waits for an answer.

## E3/E4 execution additions

Gemma entries `gemma-g0/g1/g2/g4` now select one-token constrained generation,
full head at decision positions, independent sequential candidate readout, and
independent native batches. G3 remains causal. G0 emits hard labels only: probability
metrics stay null. G4 groups compatible processor tensor shapes and right-pads text;
media are preserved, with separate forwards for incompatible shapes. Runtime
records disclose forward count, batch sizes and sequence token counts (whole batch
scope, repeated on its members; do not sum duplicate batch telemetry).

Tiny real Gemma forwards test text/image/audio/mixed input, unrelated-question
changes, reordering, mixed-request batches and full-head parity. FP32 uses 1e-5
absolute/relative tolerance; BF16 cross-batch checks use 2e-3 because GEMM shapes can
change rounding. These checks do not establish real 12B quality or bitwise parity.
Shared-prefix and packed branch caches are not enabled; their future release must
pass these same behavior comparisons against G4.

```bash
uv run --no-sync s1 eval prepare-sweep --output artifacts/shape-sweep
uv run --no-sync s1 eval perturb examples/benchmarks/text-contract.jsonl --seed 3 --output artifacts/reordered.jsonl
uv run --no-sync s1 eval behavior artifacts/reference artifacts/reordered --left-model gemma-g4 --right-model gemma-g4
```

Shape workloads are explicitly synthetic, lengths are characters, and reused states
retain source groups. Register chosen cells as suites, then vary profile B/C. No
full Cartesian grid is executed automatically. Models still require explicit enable,
revision, capabilities and cost ceilings.

Profiles now accept `load_mode = "closed_loop" | "fixed" | "poisson"`, `arrival_rate`
(requests/sec, required for fixed/Poisson), and `slo_ms`. C bounds worker concurrency;
B calls the adapter's declared native or explicitly loop-emulated batch API. Open-loop
uses B=1; server batching is a separate deployment policy. Arrival timestamps are
precomputed independently of service completion. All calls drain before model close.
SLO is an observed deadline, not forced cancellation of local GPU computation. Backend
timeouts remain censored observations and never become successful latency samples.

Summaries include queue dispatch lag, end-to-end p99, successful request/decision
throughput, correct-within-SLO goodput and all-critical-fields-correct goodput. Fewer
than 10,000 completions marks p99 as screening only. Model setup time and process
lifetime peak RSS are separate; CUDA memory telemetry is available for local Gemma.
RSS/GPU allocator peaks are process-scoped, not isolated per-request allocations.
Set model `hardware_label` and `service_region` for the comparison cell. Local Gemma
uses one batch worker; concurrency measurements belong to a configured service.
