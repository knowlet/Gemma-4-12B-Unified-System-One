# Live validation, 2026-10-01 (Asia/Taipei)

Runtime coverage: **completed_for_available_access**; evidence outcome: **completed_with_findings**. This report is generated from actual receipts and saved predictions.
Real runtime completion is separate from model quality, latency SLOs, and production acceptance.
External deployment evaluations remain blocked by missing endpoint, identity, authentication, and cost prerequisites.

## Execution evidence

| Stage | Runtime result | Hardware | Wall seconds | Job |
| --- | --- | --- | ---: | --- |
| baselines | completed / passed | CPU | 91.19 | [Modal](https://modal.com/apps/knowlet/main/ap-iwnAEC27b0CvZdi9AO0OCu) |
| checkpoint | completed / completed_with_failures | NVIDIA A100-SXM4-80GB | 102.76 | [Modal](https://modal.com/apps/knowlet/main/ap-d7sHBbjvAJ7qey70u2sdo4) |
| training | completed / passed | NVIDIA A100-SXM4-80GB | 308.55 | [Modal](https://modal.com/apps/knowlet/main/ap-W6NzNTxhnfQK3VovSJk7uY) |
| diagnostics | completed / completed | NVIDIA A100-SXM4-80GB | 54.90 | [Modal](https://modal.com/apps/knowlet/main/ap-2ZKm7x2JOtmjblqYdoT3Hv) |
| services | completed / see nested results | NVIDIA A100-SXM4-80GB | 1343.39 | [Modal](https://modal.com/apps/knowlet/main/ap-G8bfbdhLl0uAv8DHPYcTme) |
| baselines-full | completed / passed | CPU | 1239.43 | [Modal](https://modal.com/apps/knowlet/main/ap-KYotBV6Nq9DRBsz2TtY2dV) |
| checkpoint-fp32 | completed / passed | NVIDIA A100-SXM4-80GB | 715.72 | [Modal](https://modal.com/apps/knowlet/main/ap-i54t80cLOMjRNUM9WcpUmt) |
| stress | completed / see nested results | NVIDIA A100-SXM4-80GB | 183.66 | [Modal](https://modal.com/apps/knowlet/main/ap-qfkoeHABpdp1hLC64HBdVe) |
| training-full | completed / passed | NVIDIA A100-SXM4-80GB | 1696.33 | [Modal](https://modal.com/apps/knowlet/main/ap-WtiHAD5uDycSRHHaaO5dmz) |
| integrity | completed / passed | NVIDIA A100 80GB PCIe | 41.52 | [Modal](https://modal.com/apps/knowlet/main/ap-Ky8N1kngPEnQXxOCShUMLr) |

Local pytest: 515 passed, 0 skipped, 3 warnings. Ruff/lock/build outcomes remain explicit in live-summary.json.

## Public data and screening quality

BoolQ keeps all passage/question text in a single fixed schema. The 256-case full train is a strict superset of the 128-case pilot; 16 calibration and 128 official labeled validation cases are disjoint by ID, passage group, and request hash.
Media uses 32 official MNIST test images and 20 official FSDD test recordings; speaker grouping leaves six independent audio groups.

| Observation | Correct / decisions | Operational accuracy |
| --- | ---: | ---: |
| checkpoint-fp32 / public-text-quality:gemma-g4 | 111 / 128 | 86.72% |
| checkpoint-fp32 / public-media-quality:gemma-g4 | 33 / 52 | 63.46% |
| training / base:text | 111 / 128 | 86.72% |
| training / base:media | 34 / 52 | 65.38% |
| training-full / base:text | 111 / 128 | 86.72% |
| training-full / base:media | 34 / 52 | 65.38% |
| baselines / prior | 64 / 128 | 50.00% |
| baselines / tfidf | 62 / 128 | 48.44% |
| baselines / embedding | 64 / 128 | 50.00% |
| baselines / nli | 65 / 128 | 50.78% |
| baselines / cross_encoder | 63 / 128 | 49.22% |
| baselines / setfit | 71 / 128 | 55.47% |
| pipeline / M0 | 0 / 52 | 0.00% |
| pipeline / M2 | 52 / 52 | 100.00% |
| pipeline / M3 | 18 / 52 | 34.62% |
| laya / laya-general | 105 / 128 | 82.03% |

M0 removes digit evidence and is separately labeled unknown. M2 uses source-label oracle descriptions; it is a diagnostic upper-information view, not independently transcribed input or a free product capability. M3 uses actual Tesseract and pinned CPU Whisper through real loopback HTTP; recognized text and errors are saved.
training-full / base:media by task: fsdd: 6/20, mnist: 28/32.
pipeline / M3 by task: fsdd: 10/20, mnist: 8/32.
nli rejected 1 oversized context(s), retained as operational errors in the 128-case denominator.
cross_encoder rejected 1 oversized context(s), retained as operational errors in the 128-case denominator.
These digit-only subsets support task-specific screening, not broad visual/audio quality claims.

## Matched learning curves

| Curve | Fits / cells | Support shots per class | Seeds | Actual optimizer updates | Runtime status |
| --- | ---: | --- | --- | ---: | --- |
| training, CE and CE+Brier | 12 | [2, 8] | [0, 1, 2] | 24 | passed |
| training-full, CE and CE+Brier | 18 | [8, 32, 128] | [0, 1, 2] | 1800 | passed |
| Prior / TFIDF / SetFit | 27 | [8, 32, 128] | [0, 1, 2] | 180 SetFit; prior/TFIDF closed form | passed |

| Full curve / method | Shots/class | Seed count | Mean accuracy [min, max] | Mean NLL | Mean Brier sum | Mean ECE-15 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| baselines-full / prior | 8 | 3 | 50.00% [50.00%, 50.00%] | 0.6931 | 0.5000 | 0.0000 |
| baselines-full / prior | 32 | 3 | 50.00% [50.00%, 50.00%] | 0.6931 | 0.5000 | 0.0000 |
| baselines-full / prior | 128 | 3 | 50.00% [50.00%, 50.00%] | 0.6931 | 0.5000 | 0.0000 |
| baselines-full / setfit | 8 | 3 | 51.30% [46.09%, 53.91%] | 0.6941 | 0.5009 | 0.0359 |
| baselines-full / setfit | 32 | 3 | 48.70% [46.09%, 52.34%] | 0.6991 | 0.5060 | 0.0506 |
| baselines-full / setfit | 128 | 3 | 49.48% [48.44%, 50.00%] | 0.6956 | 0.5025 | 0.0537 |
| baselines-full / tfidf | 8 | 3 | 49.22% [47.66%, 50.78%] | 0.6933 | 0.5002 | 0.0288 |
| baselines-full / tfidf | 32 | 3 | 47.14% [44.53%, 51.56%] | 0.6958 | 0.5027 | 0.0562 |
| baselines-full / tfidf | 128 | 3 | 52.34% [52.34%, 52.34%] | 0.6978 | 0.5046 | 0.0868 |
| training-full / ce | 8 | 3 | 88.02% [82.03%, 92.19%] | 0.3047 | 0.1862 | 0.0589 |
| training-full / ce | 32 | 3 | 88.80% [87.50%, 90.62%] | 0.2692 | 0.1625 | 0.0632 |
| training-full / ce | 128 | 3 | 91.15% [90.62%, 91.41%] | 0.2451 | 0.1427 | 0.0524 |
| training-full / ce_brier | 8 | 3 | 87.50% [82.03%, 92.19%] | 0.3132 | 0.1919 | 0.0546 |
| training-full / ce_brier | 32 | 3 | 88.80% [87.50%, 90.62%] | 0.2683 | 0.1631 | 0.0600 |
| training-full / ce_brier | 128 | 3 | 90.89% [90.62%, 91.41%] | 0.2438 | 0.1417 | 0.0560 |

Computed 141 paired source-group bootstrap comparisons using 2,000 resamples and seed 20260930. Intervals are unadjusted descriptive 95% CIs; no post-hoc noninferiority acceptance is performed. Every comparison checks saved request hashes and gold/group/task/schema/type metadata. Repetition 0 is added only in temporary comparison copies.
Optimizer updates and reload equality are recorded per cell. Reload deserializes each adapter independently on the same frozen base; full merged model reload is not claimed.

Selected descriptive comparisons below use all 128 fixed public text cases. The direction is candidate minus the frozen BF16 base; seed 0 rows illustrate the full curve without selecting the best seed.

| Candidate | Accuracy delta, percentage points | 95% source-group CI |
| --- | ---: | --- |
| training-full:shots-128-seed-0-ce:text | +3.906 | [-1.562, +9.375] |
| training-full:shots-128-seed-0-ce_brier:text | +3.906 | [-1.562, +9.375] |
| baselines-full:shots-128-seed-0-prior | -36.719 | [-48.438, -25.781] |
| baselines-full:shots-128-seed-0-tfidf | -34.375 | [-44.531, -24.219] |
| baselines-full:shots-128-seed-0-setfit | -36.719 | [-47.656, -25.781] |
| laya:laya-general | -4.688 | [-13.301, +3.926] |

## Batching, pipelines, workflows, and load

- checkpoint (torch.bfloat16): 30 runner sweeps; g1-g2-runner-parity: True, g1-g2-numerical-parity: True, g2-g4-native-parity: False, question-independence: False.
- checkpoint-fp32 (torch.float32): 30 runner sweeps; g1-g2-runner-parity: True, g1-g2-numerical-parity: True, g2-g4-native-parity: True, question-independence: True.

BF16 shape-dependent backbone/logit differences persisted in controlled mask and readout experiments. The explicit FP32 path and its rerun are a separate numerical control; default BF16 failures remain in the evidence. Batches, candidate counts, question counts, context lengths, and repeated-state sweeps verify execution, not representative task quality.

Workflow gemma: 24 policy-episode records; manifest completed; adapter execution {'gemma-g2': 'completed', 'gemma-g4': 'completed'}. Finite-state replay checks environment goal outcomes. It does not perform actual CI repairs or UI/document operations.

| Policy | Successful / episodes | False completions | Fallback episode rate |
| --- | ---: | ---: | ---: |
| always_small | 6 / 6 | 0 | 0.00% |
| always_strong | 6 / 6 | 0 | 0.00% |
| cascade | 6 / 6 | 0 | 0.00% |
| rules_first | 6 / 6 | 0 | 0.00% |

Workflow laya: 24 policy-episode records; manifest completed; adapter execution {'laya-general': 'completed'}. Finite-state replay checks environment goal outcomes. It does not perform actual CI repairs or UI/document operations.

| Policy | Successful / episodes | False completions | Fallback episode rate |
| --- | ---: | ---: | ---: |
| always_small | 2 / 6 | 1 | 0.00% |
| always_strong | 2 / 6 | 1 | 0.00% |
| cascade | 2 / 6 | 1 | 100.00% |
| rules_first | 6 / 6 | 0 | 0.00% |

Load services: 10,512 requests, 10,512 successful, 10,512 recorded real model forwards.

| Workload | Successful / requests | Offered requests/s | Peak inflight | p99 ms | SLO misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed-c1-n64 | 64 / 64 | not recorded | not recorded | 104.193 | 0 |
| fixed-c4-n64 | 64 / 64 | not recorded | not recorded | 111.980 | 0 |
| fixed-c16-n64 | 64 / 64 | not recorded | not recorded | 101.036 | 0 |
| fixed-c64-n64 | 64 / 64 | not recorded | not recorded | 116.311 | 0 |
| poisson-c1-n64 | 64 / 64 | not recorded | not recorded | 463.222 | 0 |
| poisson-c4-n64 | 64 / 64 | not recorded | not recorded | 555.386 | 0 |
| poisson-c16-n64 | 64 / 64 | not recorded | not recorded | 331.405 | 0 |
| poisson-c64-n64 | 64 / 64 | not recorded | not recorded | 627.252 | 0 |
| fixed-c16-n10000 | 10000 / 10000 | not recorded | not recorded | 105.027 | 0 |

10,000-request p99 block-bootstrap 95% CI: [102.41268452141952, 106.95048028300613] ms. Twenty contiguous time blocks from one machine/workload; no deployment-population claim.


Load stress: 1,536 requests, 1,536 successful, 1,536 recorded real model forwards.

| Workload | Successful / requests | Offered requests/s | Peak inflight | p99 ms | SLO misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| closed_loop-c1-n128 | 128 / 128 | closed loop | 1 | 97.757 | 0 |
| closed_loop-c4-n128 | 128 / 128 | closed loop | 4 | 416.697 | 0 |
| closed_loop-c16-n128 | 128 / 128 | closed loop | 16 | 1564.742 | 118 |
| closed_loop-c64-n128 | 128 / 128 | closed loop | 64 | 6692.645 | 120 |
| fixed-c1-n128 | 128 / 128 | 13.0767 | 1 | 2400.033 | 82 |
| fixed-c4-n128 | 128 / 128 | 13.0767 | 4 | 2300.284 | 80 |
| fixed-c16-n128 | 128 / 128 | 13.0767 | 16 | 2368.989 | 79 |
| fixed-c64-n128 | 128 / 128 | 13.0767 | 28 | 2452.714 | 85 |
| poisson-c1-n128 | 128 / 128 | 13.0767 | 1 | 3185.145 | 87 |
| poisson-c4-n128 | 128 / 128 | 13.0767 | 4 | 2476.947 | 94 |
| poisson-c16-n128 | 128 / 128 | 13.0767 | 16 | 2446.876 | 99 |
| poisson-c64-n128 | 128 / 128 | 13.0767 | 41 | 3678.667 | 96 |

The serving checks verify a protected /decide route and the intentionally public health route. High-concurrency stress can complete all forwards while failing the 1,000 ms SLO. Loopback tests do not measure WAN/deployment latency.
Historical stress 04 recorded the offered rate before workload multipliers; latest stress 05 supersedes it with per-cell actual rates. Historical receipts retain their measured latencies and hardware identities.

## Actual fitted-artifact and telemetry boundaries

Integrity status: passed; independently audited 13 boundary checks, 108 train/calibration ID/group/request rejection dimensions across 18 actual saved LoRA artifacts, and 18 disjoint heldout acceptance checks.
Actual FP32 workflow telemetry was present before four real backbone forwards. Revision and precision mismatches each stopped with zero forwards; provenance overlap fixtures stopped before adapter loading. A genuinely fitted CPU prior rejected request-only overlap in an unreachable state while episode IDs/groups and the initial request stayed disjoint.
Expected setup_error/not_run child outcomes are passing boundary observations, not successful model executions. Blocked adapter checks inspect actual artifact/provenance hashes without reloading their weights; original adapter reload checks remain separate.

## Provenance, costs, and remaining access

- boolq: [CC-BY-SA-3.0](https://raw.githubusercontent.com/google-research-datasets/boolean-questions/90af34107399cc7a446b373dc4ee35b8001da7c2/README.md), [source](https://huggingface.co/datasets/google/boolq), immutable revision `35b264d03638db9f4ce671b711558bf7ff0f80d5`.
- mnist: [MIT](https://huggingface.co/datasets/ylecun/mnist/raw/77f3279092a1c1579b2250db8eafed0ad422088c/README.md), [source](https://huggingface.co/datasets/ylecun/mnist), immutable revision `77f3279092a1c1579b2250db8eafed0ad422088c`.
- fsdd: [CC-BY-SA-4.0](https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/26eb9aaf76e81b692f806f9140c2d2777410d7a1/README.md), [source](https://github.com/Jakobovski/free-spoken-digit-dataset), immutable revision `26eb9aaf76e81b692f806f9140c2d2777410d7a1`.

Lock SHA-256: `88ad0692f469b09258c11396aee6f4e8679c170ec58609f02a59482c7bd4d5c4`. Dataset/source/adapter hashes, model revisions, actual counts and job links are in [live-summary.json](live-summary.json). Raw predictions, receipts and comparison JSON remain under artifacts/live-validation.
Reported per-request preprocessing/provider costs remain unknown. Operator ceilings are reservations, not provider invoices. The completed billing buckets sum to $3.985944; pending/unreported buckets prevent a final bill claim.

External provider runs are not completed:

- jev-http: `JEV_BENCHMARK_ENDPOINT`; token configuration: `JEV_BENCHMARK_TOKEN`; an accessible deployment with immutable resolved model revision and provider cost metadata or an explicit unknown-cost policy.
- kev: `KEV_BENCHMARK_ENDPOINT`; token configuration: `configure ModelSpec.token_env if this deployment requires a bearer token`; an accessible deployment with immutable resolved model revision and provider cost metadata or an explicit unknown-cost policy.
- decider: `DECIDER_BENCHMARK_ENDPOINT`; token configuration: `configure ModelSpec.token_env if this deployment requires a bearer token`; an accessible deployment with immutable resolved model revision and provider cost metadata or an explicit unknown-cost policy.
- agentjev: `AGENTJEV_BENCHMARK_ENDPOINT`; token configuration: `configure ModelSpec.token_env if this deployment requires a bearer token`; an accessible deployment with immutable resolved model revision and provider cost metadata or an explicit unknown-cost policy.

Using recorded official A100-80GB/CPU/memory rates, the unreported execution windows estimate $0.1564 GPU-only to $0.1922 with requested CPU/host memory. Recorded billing plus those window allocations is approximately $4.1782. This is not a final invoice: initialization outside receipts, storage/egress and pending bucket extensions are excluded. The $20 cap applied to this chosen campaign; function timeout ceilings are not automatically summed as a budget.
The final app snapshot records 2 jobs, all stopped with zero tasks: True.
Historical CPU baseline outer Gemma fields described campaign context. Actual inner MiniLM identities and resolved runtime telemetry are authoritative.

## Reproduction

Use an authenticated Modal profile with the `gemma-unified-system-one` volume. These commands start ephemeral jobs; they do not deploy a service. Use a fresh run name and preserve receipts/datasets before the next stage.

```sh
uv sync --all-extras --group dev
uv run python scripts/prepare_live_validation_data.py --output artifacts/live-validation/datasets
uv run modal run apps/modal/validate_v2.py --stage checkpoint --run your-run
uv run modal run apps/modal/validate_v2.py --stage checkpoint-fp32 --run your-run
uv run modal run apps/modal/validate_v2.py --stage training-full --run your-run
uv run modal run apps/modal/validate_v2.py --stage baselines-full --run your-run
uv run modal run apps/modal/validate_v2.py --stage services --run your-run
uv run modal run apps/modal/validate_v2.py --stage stress --run your-run
uv run modal run apps/modal/validate_v2.py --stage integrity --run your-run --training-run your-run
uv run python scripts/summarize_live_validation.py
```

Pilot training, six CPU baseline methods and diagnostics use `--stage training`, `--stage baselines`, and `--stage diagnostics`. Integrity must use the run containing completed training-full artifacts; choose per-stage run names when rerunning an existing stage.

Public small-sample screening does not establish broad visual/audio understanding, pretraining cleanliness, production reliability, a statistically accepted replacement decision, or published model/deployment readiness.
