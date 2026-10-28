# Jev Ultrafast browser evaluation

The browser campaign runs the original [Jev Ultrafast policy](https://github.com/browser-use/jev-ultrafast/tree/1231850a0bf1a0c0341fe408ef1668dbbfdfac46)
at `1231850a0bf1a0c0341fe408ef1668dbbfdfac46`. It replaces the decision transport
with S1 and Browser Harness's connection with an isolated Playwright CDP session.
The upstream snapshot reader, operation/target questions, freshness checks and
action executor retain their original source. No personal Chrome profile is used.

This public DOM integration suite includes local fixtures and separately frozen
external tasks. It exercises actual Chromium actions and page transitions, but
does not establish general reliability on external websites,
official JevBench scores, sealed performance, or superiority over market models.
The existing finite-state workflow replay remains a separate evaluation.

The bridge serializes structured descriptions without rewriting them. A head with
one observed target returns that sole target with probability 1 and a trace flag;
it does not run inference or invent an alternative. Heads over S1's 52-candidate
limit fail explicitly without truncation by default. Only the upstream-selected operation's
matching target can execute. Text generation uses a separate optional helper and
the original upstream `TEXT_VALUE` instruction; the six initial tasks do not
require data entry, so their success cannot establish text-helper coverage.

`--oversized-head-policy hierarchical` explicitly enables a preserve-all
approximation for larger Choice heads. It creates `ceil(N/52)` balanced
contiguous groups in original candidate order, embeds all original IDs and
descriptions in the parent criteria, and scores the parent plus every
conditional group in one joint core request alongside the ordinary heads.
For 57 candidates the groups have 29/28 members. The final distribution is
`P(group) × P(candidate | group)` over every original label. These probabilities
are uncalibrated hierarchical approximations; flat-score or separate-call parity
is not claimed. No gold filtering, “none” option, shortlist or discarded group
is used. Core heads still require at most 52 candidates, the expanded request
must fit the existing head/context limits, and errors fail the whole decision.
`reject` remains the default; requests with at most 52 candidates keep their
original path. The identity and raw trace declare the selected policy, group
map, all internal scores and actual backend-call count.

The optional native field helper accepts strict direct JSON or one complete
optional-`json` fenced block without outside prose. It preserves the decoded
value and original raw output, records transport normalization, and rejects
duplicate keys, nonfinite constants, extra keys and missing/invalid text. The
upstream instruction and greedy 128-token budget are unchanged. The diagnostic
generation parser remains strict so its baseline comparison is preserved.

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

`--manifest configs/benchmarks/ultrafast-external.json` selects the separately
frozen Wikipedia search and cross-page tasks. They use fresh private contexts
and exact destination URL/title/heading predicates. Initial-page observation
confirmed candidate sizes but did not execute the goals. An external task's
model success remains unverified until its own run has completed; fixture
success does not fill those results. The optional native text helper is required
when the agent chooses to type in the search field.

To run the external suite after existing MPS jobs, use
`scripts/run_ultrafast_after.py` with a separate fresh supervisor directory and
child run directory. Pass each currently live predecessor with repeated
`--predecessor-pid` arguments. It captures their original identities, waits for
both to end, checks frozen source hashes, and launches the comparison with the
same model and settings for both profiles. The external tasks use a separately
declared 180-second budget and the same 12-cycle cap; their timing is not pooled
with the 90-second local fixture suite. Receipts preserve the exact command,
process identity, logs and exit code. Observation/source errors prevent launch.
The launcher passes `--mps-call-maintenance` and
`--oversized-head-policy hierarchical` only when explicitly requested; default
child arguments remain unchanged. It retains the selected interpreter.

The later [local-oracle diagnostic](validation/2026-10-28/mps-browser/oracle-diagnostic.json)
found that the frozen local count string can reject a task-compatible empty
search submission. Its original scores remain unchanged, but they cannot prove
an actual task-success advantage. This limitation does not change the separately
frozen external destination checks.

The [first external MPS baseline](validation/2026-10-28/mps-external-browser/README.md)
completed all twelve attempts: causal passed 3/6 and independent 0/6. Search
hit a separate native-helper JSON failure; independent calls failed with invalid
probabilities before generation. Exact masks and an isolated MPS kernel control
identified unsafe fully masked sliding-padding rows. A
[native-window batching guard](validation/2026-10-28/mps-sliding-padding/README.md)
now avoids creating those rows on MPS. Its repaired full-checkpoint numerical
control passes. The subsequent
[consumed-task replay](validation/2026-10-28/mps-postfix-browser/README.md)
passes for both profiles; fresh transfer tasks and both benchmark regressions
remain separate acceptance work.
The [isolated Torch 2.14.1 candidate](validation/2026-10-28/mps-torch214-browser/README.md)
then completes all twelve browser attempts with 6/12 verified: both profiles
pass all follow-link repeats, while search types successfully and stops at the
explicit 57-candidate rejection. Its causal public231 run is raw-audited at
197/231 correct with 231 valid answers; coherence is only a checkpointed
94-entry partial inventory, without an accepted score. The
[hierarchy replay](validation/2026-10-28/mps-hierarchy-browser/README.md)
reaches the search destination for both profiles but exceeds the unchanged
180-second budget, so strict verified success is 0/2. These consumed-task
diagnostics do not establish generalization, speed superiority or complete
benchmark acceptance. The runtime is still a candidate; the separate 0.01
probability-parity control fails.
The [earlier observer run](validation/2026-10-28/mps-observer-safety/README.md)
was invalidated for unsafe copy/conversion and supplies no acceptance evidence.

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
For hierarchy calls, failed core predictions or counter-observation errors leave
actual forward counts unknown (`null`). Completed-return deltas are retained
separately as lower bounds and propagated through task/campaign summaries;
they must not be presented as zero dispatched work. Flat/default reporting is
unchanged. One joint request may contain multiple internal question heads.
Cost remains null because local compute billing is not measured.

`--mps-call-maintenance` is a separate opt-in MPS policy. After a whole decision
or native-helper call, including errors, it synchronizes, records memory,
clears unoccupied cache, synchronizes and records memory again. Cleanup is
included in callback/helper latency and never runs per generated token. A
maintenance fault preserves the current raw result/error, blocks later model
dispatch, fails a separate operational resource gate and makes the campaign
exit nonzero. It does not alter goal verification or memory limits. The trace
records profile/kind, errors and memory snapshots; snapshots are not peaks.
This policy is not evidence that Torch 2.8 graph-cache growth is fixed.

## Two JevBench regressions

Optional `--heaven-root` and `--coherence-root` run both pinned suites after the
browser attempts using each still-loaded profile. The first root must be
`fstandhartinger/jevbench@bb05a335bc809e61b20c0f745d25499a82b326fc`; source and
dataset files are verified by `configs/benchmarks/jevbench-public.json`. The
second must be a clean `JevBench/jevbench@e18733694623aa93058e279c3534f8b8e2edefa3`.
Keep their distinct public accuracy and strict metamorphic coherence reports;
neither is an official submission. No calibration fitting occurs in this runner.

The original long MPS process exhausted memory after the causal public run.
Its incomplete coherence and zero-valid independent public results are preserved
as [runtime failures](validation/2026-10-28/mps-memory-failure/README.md), not
accepted quality scores. To resume missing stages after the captured external
supervisor and its actual child terminate:

```bash
artifacts/browser-env/bin/python scripts/resume_mps_benchmarks.py \
  --parent artifacts/ultrafast/20261028-native-01 \
  --output artifacts/ultrafast/your-unique-resume \
  --coherence-root artifacts/benchmark-tools/jevbench \
  --heaven-root artifacts/benchmark-tools/heaven \
  --wait-for-pid CAPTURED_SUPERVISOR_PID \
  --predecessor-receipt artifacts/ultrafast/your-supervisor/receipt.json \
  --original-pid ORIGINAL_BASELINE_PID
```

The launcher archives and validates original S1 sources, complete pinned
benchmark repositories and checkpoint identities. Successful old cache entries
retain their original epoch; new calls have full raw traces. A measured
unoccupied-cache trim runs after callbacks and stage boundaries, including its
latency and before/after memory snapshots. Its OOM effectiveness remains
unverified until the resumed stages complete. Memory limits stay unchanged.
Completed reports require full coverage and raw-answer score replay.
Use repeated `--additional-predecessor-pid` arguments for any additional live
diagnostics that must finish before the resume loads its model. Their original
identities are captured together before waiting; the selected external
supervisor's terminal child receipt remains independently required.

## Bounded benchmark epochs

The Torch 2.14.1 candidate's original process was deliberately stopped after
its completed causal public231 audit, with terminal exit 143 and verified
absence. Its original identity still says `running`; use the checkpoint and
termination receipts rather than treating that stale field as a live process.
The original coherence stop inventory has 94 entries and no accepted score.
The cache is shared with the resume and may now grow; never archive its changing
contents as if they were the original 94-entry snapshot.

`scripts/supervise_mps_benchmark_epochs.py` freezes the parent execution source,
benchmark/checkpoint identity and its own child protocol before dispatch. It
runs one identified child at a time. A latched checkpoint ends that child when
the predeclared new-call budget or driver/recommended-memory fraction is
reached, preserving completed raw answers. The next epoch starts only after
the owned child has terminated and immutable cache/source checks pass. An
unexpected error is not a clean resource checkpoint. Official completion still
requires full raw-answer coverage and score replay; epoch setup or cached
reconstruction is not fresh model-serving latency.

Use the separately recreated [candidate environment](validation/2026-10-28/mps-torch214/README.md),
not a changed project default. The following is a template, not an additional
run executed for this documentation; substitute the captured original PIDs and
include every other live MPS job as an additional predecessor:

```bash
artifacts/mps-next-env/bin/python scripts/supervise_mps_benchmark_epochs.py \
  --parent artifacts/ultrafast/20261028-torch214-candidate-01 \
  --output artifacts/ultrafast/your-unique-bounded-epochs \
  --coherence-root artifacts/benchmark-tools/torch214-candidate-01/coherence \
  --heaven-root artifacts/benchmark-tools/torch214-candidate-01/heaven \
  --predecessor-receipt artifacts/ultrafast/20261028-hierarchy-supervisor-01/receipt.json \
  --original-pid ORIGINAL_CANDIDATE_PID \
  --wait-for-pid ORIGINAL_HIERARCHY_SUPERVISOR_PID \
  --additional-predecessor-pid OTHER_CAPTURED_MPS_PID \
  --max-new-calls 2000 --max-driver-fraction 0.8 --max-epochs 256
```

The recorded supervisor output
`artifacts/ultrafast/20261028-torch214-benchmark-epochs-01` was running at the
evidence capture with the explicit 2,000-call / 0.8 policy. Its changing outputs
are not an accepted final coherence or independent public231 result. The
supervisor receipt, epoch receipts, immutable cache origin and final offline
audits must support any later completion claim.

The 2026-10-28 attempt to rerun the existing A100 campaign was rejected before
job creation because the Modal workspace exceeded its spend limit. Local MPS
measurements must retain their hardware/runtime identity and cannot substitute
for a controlled A100 latency comparison.
