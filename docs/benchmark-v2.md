# Decision Benchmark v2

E0–E7 evaluation tooling is implemented. The frozen v1 baseline remains
`f8390676ae942699a71a85ae9a99cfcd7b9b3806`; `s1 benchmark` and `s1 compare` retain their
v1 interfaces. V2 runs are separate, reproducible artifacts. The supplied datasets
are contract fixtures, **not evidence of pretrained model quality**.

## Start with the offline acceptance run

```bash
uv sync --locked --extra api
uv run --no-sync python scripts/benchmark_v2_acceptance.py --output artifacts/evaluation/acceptance
```

This needs no model weights, credentials, GPU or network. It exercises preflight,
three load modes, metrics, paired statistics, support selection, prior fitting,
media-view preparation, resettable workflows and standalone reports. Open
`artifacts/evaluation/acceptance/report/report.html` or `report.md`.

All output run/training/report directories must be new. Existing results are never
overwritten. There is no resume protocol; an interrupted run remains inspectable,
but cannot be treated as a finished comparison.

## Plan and run

```bash
uv run --no-sync s1 eval preflight --profile smoke --enable-model uniform
uv run --no-sync s1 eval plan --profile smoke --enable-model uniform --output artifacts/plan.json
uv run --no-sync s1 eval run --profile text-smoke --enable-model uniform --output artifacts/text
uv run --no-sync s1 eval summarize artifacts/text
```

`preflight` and `plan` never load models or contact endpoints. All registry models
are disabled by default. Explicit enabling applies only to this invocation.
`ready` means configuration is complete; runtime loading, credential validity and
hardware compatibility are checked during execution. `run` may download selected
weights and call configured services.

| Configuration | Contract |
|---|---|
| `configs/benchmarks/models.toml` | Adapter, immutable revision, precision, readout, calibration, declared capabilities, cost ceiling, device/region, credential environment-variable names |
| `configs/benchmarks/suites.toml` | Dataset, optional fingerprint, track and information view |
| `configs/benchmarks/profiles.toml` | Models, suite, split, repetitions, warmup, B/C, required capabilities, arrival policy, SLO, seed, aggregate budget |

Override with `--registry`, `--suites`, `--profiles`, `--lockfile`. Suite dataset paths
are relative to the suite TOML; other paths are relative to the current directory.
Unknown settings are rejected. Unknown capability differs from unsupported. A
runtime failure never changes predeclared eligibility. Unsupported/disabled models
remain visible, without fabricated zero quality.

Budgets reserve **all logical requests**, including warmups and each batch member,
across all ready enabled models. Unknown cost blocks execution; only Uniform is
intrinsically free. `cost_per_request_usd` is a conservative operator ceiling,
including preprocessing where configured. It is not an invoice. There are no retries
or runner result caches. Provider-side caches remain unknown unless observed.

Each experiment ID hashes the effective configuration, source code, dependency lock,
installed runtime versions, dataset and ordered candidates, seed, calibration,
model/processor revisions, hardware/service labels and endpoint hashes. Raw prompts,
media and credential values are not copied into evaluation artifacts. Gold and
evaluation metadata never reach adapters.

Exit 0 means completion, 1 means blocked/partial/errors/inconclusive gate, and 2 means
invalid input. Calibration profiles explicitly use `split = "calibration"`; ordinary
quality profiles require test-only data.

## Model routes and controlled Gemma readouts

| Route | Implemented adapter / output |
|---|---|
| Uniform | Complete uniform distributions, sanity only |
| Gemma G0 | `readout = "generate"`, sequential, one constrained answer token/question, thinking off; hard labels only |
| Gemma G1 | `readout = "full"`, sequential independent questions; full vocabulary only at required answer positions |
| Gemma G2 | Candidate projection, sequential independent questions |
| Gemma G3 | Existing causal multi-slot; later questions can read earlier question text |
| Gemma G4 | Independent question sequences in real native batches; compatible processor tensor shapes share a forward |
| Laya | Existing wrapper, pinned checkpoint/subfolder, provider precision; context truncation disclosed |
| Jev-compatible HTTP | `protocol = "compatible"`, exact configured endpoint |
| Kev / Mapika Decider | `protocol = "typesafe"`; Score criteria sent as ordered descriptions, returned indices mapped to original numeric levels |
| AgentJev | `protocol = "agentjev"`; boolean/choice/score and `results[].answers[]` mapping; text-only, Score ≤10 levels |
| TF-IDF + LR / prior | JSON classifier artifact, fixed schema, training-only fit, complete distributions |
| Embedding / NLI / cross-encoder | Candidate ranking with hard labels; similarity/entailment scores are not converted into fake candidate probabilities |
| SetFit | Local trained checkpoint plus hashed training/class-label artifact; complete classifier distribution |

Install `--extra inference`, `--extra laya`, `--extra train` or `--extra baselines`
only for the selected routes. Heavy libraries are imported lazily. Local Gemma uses
BF16 on CUDA and float32 on CPU; declaring another precision is blocked. Baseline
encoders use float32; TF-IDF/prior math uses float64. Embedding/SetFit context
truncation is disclosed. NLI/cross-encoder reject excess context. NLI checkpoints
must identify their entailment class; cross-encoders must return a scalar score.

Local HF checkpoints need immutable 40/64-character hex revisions. HTTP versions
are operator assertions: record the deployed weights **and runtime** in `revision`,
and set `hardware_label` / `service_region`. These adapters do not prove the remote
service is running the claimed checkpoint. Upstream model families, multilingual
Laya and sizes can use distinct registry entries; never conflate them as one model.

Protocol references verified during implementation:
[Kev contract](https://github.com/jaredpalmer/kev/blob/main/kev/api.py),
[Decider](https://github.com/Mapika/decider),
[AgentJev contract](https://github.com/malevrigns/agent-jev/blob/main/jev_service/contract.py),
[SetFit API](https://huggingface.co/docs/setfit/reference/main),
[Sentence Transformers API](https://sbert.net/docs/package_reference/sentence_transformer/model.html).
Pin the deployed source revision rather than assuming these moving branches stay compatible.

The TypeSafe wire mapping accepts bounded four-decimal probability rounding (at
most K×0.00005 total mass error) and explicitly renormalizes that distribution.
Larger errors or non-four-decimal malformed outputs remain failures. This handles
Kev's documented serialization without loosening the generic probability validator.

## Artifacts and metrics

| Artifact | Contents |
|---|---|
| `run_manifest.json` | Frozen plan, identities, lifecycle, setup time, runtime/allocator telemetry |
| `requests.jsonl` | Warmup/measurement status, scheduled/dispatch/ready/termination times, batch ID, SLO, pipeline stage timing/cost |
| `predictions.jsonl` | One record per measured decision, including failures/not-run; IDs/groups, actual label, standardized argmax, distribution, hard/soft gold, K/N/modality metadata |
| `errors.jsonl` | Phase and exception type, without raw provider error messages |
| `summary.json` | Recomputable quality, coverage, latency, goodput and source-group counts |
| `report.md`, `report.html` | Separate tracks/views/hardware, quality/latency tables, descriptive frontier, reliability and risk–coverage SVGs, workflow results |

Requests are atomic: partial/malformed answers fail the whole request. Output is
flushed as calls complete. Interrupted manifests and incomplete record counts remain
explicit. Warmups consume budget but are excluded from measured quality and latency.

Metrics include actual Choice and standardized ordered argmax accuracy, task-macro
accuracy, per-schema macro-F1, all-fields-correct, critical-field errors, sum-Brier,
NLL (1e-12 floor plus zero-gold-probability count), 15-bin reliability/ECE,
tie-averaged risk–coverage/AURC, Noul AUROC/AP/FPR/FNR, numeric Score MAE, ordinal
error and normalized RPS. Noul positive label is `true`; P(true)=0.5 standardizes to
`false`. Soft-target cross-entropy/Brier/KL and teacher agreement are separate.
Hard-label adapters still receive accuracy/Score MAE; unavailable probability
metrics stay null. Slices cover task, language, type, modality, K and N.

Always examine three denominators: valid-answer conditional quality, valid/eligible
execution coverage, and correct/eligible operational correctness. Failures and
timeouts cannot disappear into a successful-only comparison.

```bash
uv run --no-sync s1 eval compare artifacts/reference artifacts/candidate --seed 0 --resamples 2000 --output artifacts/comparison.json
uv run --no-sync s1 eval report artifacts/reference artifacts/candidate --output artifacts/report
```

Formal quality comparison requires complete finished records, identical dataset
fingerprints and information views. Paired bootstrap resamples source groups, keeping
translations, slots and repetitions together. At least two groups are needed for
an interval; small samples are not strong release evidence. Multiple comparisons
are not adjusted automatically. Comparisons are `right_minus_left`.

`eval gate comparison.json --spec gate.json` applies a preregistered accuracy gate:

```json
{"left_experiment_id":"<64-hex>","right_experiment_id":"<64-hex>","accuracy_margin":0.01,"min_groups":30}
```

The lower operational-accuracy CI must clear `-margin`. Insufficient groups or a CI
crossing the boundary returns `inconclusive`, never pass. This gate does not imply
critical-field safety, retention or serving acceptance; inspect those separately.
Register the gate before observing test results.

## Calibration and data provenance

```bash
uv run --no-sync s1 eval audit-splits data/train.jsonl data/calibration.jsonl data/test.jsonl
uv run --no-sync s1 eval calibrate artifacts/calibration-run --model gemma-g4 --output artifacts/temperature.json
```

Calibration fitting only accepts complete calibration-split probability records.
Apply with `calibration = "domain"`, `base_calibration`, `calibration_file` and
`calibration_sha256`. Base settings must match; evaluation IDs, source groups and
canonical request hashes must not overlap fitting data. Temperature uses a disclosed
1e-12-floored probability power transform; test data never tune it.

Cases may include `group_id`, `task_id`, `schema_id`, `language`, `soft_gold` and
`critical_questions`. Gold stays separate from the request. Group IDs should cover
documents, sessions, recordings and translation families; missing groups fall back
to case IDs. Canonical fingerprints preserve question/candidate ordering.

Public-source conversion uses local JSONL and saves source SHA-256, provenance,
original split, derived evaluation split and the transformed fingerprint:

```bash
uv run --no-sync s1 eval import-data data/boolq-validation.jsonl --format boolq --source-url https://example.org/pinned-source --revision SOURCE_REVISION --license SOURCE_LICENSE --original-split validation --split test --output data/boolq-heldout.jsonl
```

Formats: `boolq` (`passage/question/answer`), `ocnli`
(`sentence1/sentence2/label`) and `choice`
(`state/question/options/answer`, optional language/group_id/id). OCNLI unlabeled
test rows are rejected. A held-out labeled development set stays identified as such;
conversion never claims an official test score. Choice preserves the entire supplied
ontology. K>52 is rejected rather than selecting distractors using gold. Source URL,
revision and license are operator-supplied and must be verified for formal results.

## Parallelism and serving

N = questions per state, K = candidates per question, B = independent requests per
batch, C = client workers. G4 expands questions to independent sequences; media
processor tensors are retained, with separate forwards for incompatible shapes.
`adapter_execution` discloses batch sizes, forward calls and actual sequence lengths;
its scope is the whole adapter batch, repeated on member records. Do not sum duplicates.
G2's per-question forwards are not a native batch.

```bash
uv run --no-sync s1 eval prepare-sweep --output artifacts/shape-sweep
uv run --no-sync s1 eval perturb examples/benchmarks/text-contract.jsonl --seed 3 --output artifacts/reordered.jsonl
uv run --no-sync s1 eval behavior artifacts/reference artifacts/reordered --left-model gemma-g4 --right-model gemma-g4
```

Shape workloads cover one axis at a time: N 1–64, K 2–52, text lengths and state reuse.
Lengths are **characters**, with actual model tokens recorded separately. Register
chosen cells as suites and vary B/C profiles; no full Cartesian product runs implicitly.
Order perturbation preserves semantic labels/source groups. Behavior comparison
reports common validity, flips, max probability delta and grouped accuracy delta;
it deliberately allows changed inputs for order/retention diagnostics.

Tiny real Gemma tests cover FP32/BF16, text/image/audio/mixed input, question reordering,
unrelated-question changes and mixed batches. Full-head parity uses 1e-5 tolerance;
cross-batch BF16 uses 2e-3 because GEMM shapes can change rounding. These are numerical
and isolation tests, not pretrained multimodal understanding evidence.

Profiles support `load_mode = "closed_loop" | "fixed" | "poisson"`, `arrival_rate`
(requests/sec for fixed/Poisson), `slo_ms`, `batch_size`, `concurrency`. Open-loop
uses B=1, with arrivals computed independently of service speed. Offline B>1 invokes
the declared native or explicitly loop-emulated adapter batch. Local Gemma/Laya
concurrency is blocked; use native Gemma batching or a configured service for C sweeps.

Latency is scheduled-arrival to all-answers-ready, including queueing, preprocessing,
inference/network and validation. It is never latency/N. Service latency and dispatch
lag are separate. An SLO miss is an observation, not forced cancellation of local
computation. Backend timeouts remain censored; failed calls do not enter successful
latency percentiles. Work drains before model close.

Summaries contain requests/sec, decisions/sec, correct-within-SLO goodput,
all-critical-fields-correct requests/sec, deadline misses and p99 screening flags.
Fewer than 10,000 completions is marked screening-only. Setup time is separate;
RSS is a process-lifetime peak, CUDA values are allocator peaks, not per-request
memory. Remote engine timing is not inferred from RTT. Repeat formal runs over
multiple time blocks and compare only matched hardware/service regions.

## Native media, counterfactuals and OCR/ASR pipelines

`eval prepare-media bundle.jsonl --output artifacts/media` prepares M0–M3. Each row:

```json
{"case":{"id":"media-a","group_id":"source-pair","request":{"state":"...","questions":["<normal Question objects>"],"media":["<normal image/audio inputs>"]},"gold":{"q":"label"}},"missing_media_gold":{"q":"unknown"},"verified_text":"Human-verified transcript or description"}
```

Use actual contract objects in place of explanatory placeholders. M0 removes media
and **requires separately annotated gold**; M1 retains raw media; M2 adds verified
transcript/description; M3 retains raw media for a measured service pipeline. M2 is
a diagnostic view, not free deployable perception. Different-gold counterfactual
pairs must share text/questions and change media. Both answers must be correct:
flipping alone is insufficient.

M3 model configuration: `preprocessing = "http_ocr" | "http_asr" | "http_media"`,
`preprocessor_endpoint_env`, optional `preprocessor_token_env`, and
`preprocessor_revision`. Use suite `track = "pipeline"`,
`information_view = "pipeline_output"`. The explicitly configured OCR/ASR service
accepts `{"media":[...]}` and returns `{"text":"...","cost_usd":0.01}`; cost is optional,
reported by the service, and not inferred. No specific vendor OCR/ASR is assumed.
The pipeline adapter consumes declared media, adds the observation to state, then
calls the text decision model. Both stages' time is included; preprocessing cost/time
survive downstream failure. Native batching is not claimed for this wrapper.

```bash
uv run --no-sync s1 eval counterfactual artifacts/native-run --model gemma-g4 --views artifacts/media/media-views.json
uv run --no-sync s1 eval behavior artifacts/base-media artifacts/lora-media --left-model gemma-g4 --right-model gemma-lora
```

Run base and decision adapters on identical held-out media for retention. Supply
representative licensed images/audio; bundled synthetic signals only test plumbing.

## Matched specialist and LoRA training

```bash
uv sync --locked --extra inference --extra train --extra baselines
uv run --no-sync s1 eval prepare-support data/train.jsonl --heldout data/calibration.jsonl data/test.jsonl --shots 8 32 128 --output artifacts/support
uv run --no-sync s1 eval fit-baseline artifacts/support/shots-8-seed-0.jsonl --method tfidf --seed 0 --output artifacts/tfidf
uv run --no-sync s1 eval fit-baseline artifacts/support/shots-8-seed-0.jsonl --method setfit --model PINNED_EMBEDDER --revision COMMIT_SHA --seed 0 --steps 20 --output artifacts/setfit
uv run --no-sync s1 eval train-curve artifacts/support/support.json --calibration data/calibration.jsonl --revision GEMMA_COMMIT_SHA --steps 100 --max-updates 1800 --output artifacts/gemma-curve
```

Support preparation requires one fixed text schema, one question/case, training-only
labels, three seeds and disjoint held-out groups/IDs/requests. It takes at most one
representative per source group, stratifies by class, and nests lower-shot supports
inside larger ones for each seed. Insufficient classes/groups are errors, never
silently reduced quotas. Every method receives the exact saved support IDs. This
is matched annotation budget, not a claim that SetFit and LoRA consume equal FLOPs.

TF-IDF/LR and prior save JSON parameters; no pickle is used for these artifacts.
SetFit saves a local checkpoint plus a hashed manifest with class ordering and a
checkpoint content digest. Configure `artifact_file`/`artifact_sha256`; use the saved
SetFit model directory as `model_id` and pin its manifest digest as the local revision.
Evaluation rejects unseen schemas and training overlap. Embedding/NLI can instead
accept new runtime candidates, with their hard-label probability contract.

`train-curve` runs CE and CE+Brier using identical support, initialization seed,
steps, learning rate and LoRA rank/targets for seeds 0/1/2. Calibration data are locked
by the support manifest; test data are never fitted. It saves **separate adapters**,
calibration/provenance and content hashes. Select them with `decision_adapter_path`
and `decision_adapter_sha256`; use the pinned base model revision. Preflight rejects
evaluation overlap with adapter training/calibration data. `calibration = "checkpoint"`
uses the saved adapter temperature; `none` explicitly uses T=1. Compare base/LoRA
quality and held-out media retention using the normal runner. Failed/interrupted
training preserves completed cells, without claiming remaining cells completed.

## Resettable workflows and costs

```bash
uv run --no-sync s1 eval workflow examples/benchmarks/workflows.jsonl --small uniform --strong uniform --enable-model uniform --output artifacts/workflow
uv run --no-sync s1 eval report artifacts/text artifacts/workflow --output artifacts/report
```

The shipped CI and UI/document scenarios are **finite-state replay fixtures**. Each
episode resets to its initial state with a matched seed; action transitions determine
the next observation and externally specified goal state. They execute no arbitrary
shell/browser actions. Replace fixtures with licensed representative environments
before claiming coding-agent or UI task success.

Policies: always-small, always-strong, rules-first, and small→the same strong worker
when confidence is below a fixed threshold or the action is `escalate`. Hard-label
S1 outputs always fall back in the cascade. Rules only inspect public state fields;
models receive observations/candidates, not transition tables or goal truth. Saying
`complete` before the goal is satisfied records a false completion.

Worst-case calls and model/tool cost ceilings are reserved before loading models.
`--max-calls`, `--max-cost-usd`, `--threshold` are explicit; freeze the threshold before
test. Records preserve every routed model call, failure, transition, retry, fallback,
model switch and reported cache hit. Unobserved cache loss remains null. Task latency
is measured replay wall time, not simulated external tool duration.

Preflight checks every episode state, including states that a policy might never
visit, for training/calibration overlap by episode ID, source group and canonical
request hash. An overlap blocks adapter construction. After loading, workflows apply
the same resolved precision/revision checks as normal evaluation before any inference.
The workflow manifest preserves each adapter's resolved telemetry and setup failure.

`workflow_summary.json` reports completion, false completion, fallback, retries,
p95 and cost per success. The numerator includes **all episodes**, including failures.
Reported model bills are separate from conservative ceilings and fixture tool costs;
calls retain preprocessing and decision bills separately, and sum each stage once.
An unknown bill from any executed stage keeps the reported total unknown, including
downstream failures after successful preprocessing. `workflow_comparison.json` pairs
policy deltas by source group. Provider cache carryover is unknown and disclosed.

## Boundaries of this delivery

The reference E0–E7 tooling and offline acceptance are available. Live comparative
results still require pinned weights/API credentials, appropriate hardware, explicit
budgets and representative held-out data. No such results are implied by passing CI.
Shared-prefix KV and branch-isolated packed attention, quantized/MLX/optimized runtime
tracks, K>52, specialized vendor OCR/ASR deployment, production browser/coding agents,
and crash-resume are separate extensions. No loop is mislabeled as a native batch,
and no unimplemented cache is advertised as a speedup.
