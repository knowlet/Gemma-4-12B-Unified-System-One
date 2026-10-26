# CPU5 NumPy zero-delta source candidate

This directory is an unselected, nontraining source port. `SOURCE_ONLY = True`
refuses direct `execute` before any dependency or plan is loaded. The CLI refuses
before argument parsing, including without arguments and with `--execute`.
After a future reviewed source release, its default path only describes the
profile without executing model work. Neither CLI, imports, AST,
fakes, formatting, source hashing, builder nor model work was executed during
this authoring pass. `plan.pending.json` is explicitly nonexecutable; new source
pins, checks, freeze, actual results and NumPy catalog remain pending. No runnable
plan, root spec or builder is supplied here. No old source was changed.

## Exact port

The source parent is `decision3-readout-numpy-qualification-20261024-01`.

- `observer.py` and `fixture.json` are copied without edits. The four original
  offered heads retain order and legal counts `[8, 1, 2, 3]`, with no semantic
  expected answer. This invented fixture is not DSL train/dev/sealed data, a
  benchmark item, a browser diagnostic or the exposed Copper task.
- `numpy_probe.py` changes only its owner policy diagnostic from owner07 to
  owner09. The complete native answer validator, including K=1 confidence1,
  exact independent read-only FP32 copies, catalogs and pending payload logic
  remain unchanged. Native05's synthetic smoke-answer validator is not called.
- `actor.py` selects native05 actor `9a7e5af9…`, helper `4df54e06…` and plan
  `095db88b…`, with owner09 controller `81846efe…` and source freeze `52ac647a…`.
  Full digests are in the source and pending metadata. The distinct resource
  binding is CPU15–19, quota500%, five intra-op threads, one inter-op thread,
  memory4GiB, swap0, reserve26,119,172,916B and startup30,414,140,212B. The owner
  limit remains10800s, whole actor120s and complete prediction20s.
- Other actor deltas are the source-only hold, CPU5 profile/metadata and explicit
  executable-plan requirement, five-thread plan checks, clearer
  `PENDING_NONTRAINING` receipt labels, owner09 scope text and testable CLI argv.
  `load_source`, source projection/authentication, hook cleanup and transaction
  logic are retained. `perform_call` differs only in pending status strings.
- Tests retain the original stdlib fake timing/primary-error, singleton,
  distribution/order, import guard and malformed-input cases. New authored
  tests cover the hold before argument parsing, a fake-released dry CLI, CPU5
  metadata, fixed source/owner pins, pending state and
  exact observer/fixture/probe delta plus unchanged math/transaction AST. Their
  temporary plans use invalid source digests and mocked authentication; they
  cannot be selected as real plans. These assertions are not passing results.

Native05's loader, original generic `NativeBatch4Backend`, model math and full
authentication are reused; its synthetic two-call smoke `execute` is not called.
One constructor and one genuine request with four rows are proposed. The CPU
BF16 backbone, FP32255×1024 readout, T=1, original preparation/masks/context16384
and rejection without truncation remain unchanged. No helper predictions,
fallbacks, repaired choices, dummy heads or extra encoder passes are introduced.

## Same-call proof and timing

The original first-layer/final-norm observers return `None`. Detached last-position
features are `[4,1024]` FP32 from that same genuine forward. Torch and NumPy
replay every legal probability with fixed
`abs(replay-native) <= 1e-6 + 1e-5*abs(native)`, exact ordered argmax and exact zero
illegal mass. The singleton head remains legal with confidence1. Independent
NumPy copies are byte/equality checked, made read-only and hashed before/after;
the live baseline tensor identities/versions and W0 must remain unchanged.

The frozen `head_core.py` pin remains
`6761dd654481424f4d6c908a7c9fd2d43e4762eeb5420af9f22612ffe77657ed`.
It imports NumPy, so it is loaded only inside `perform_call`, after the original
native05 `load_runtime` fresh-process guard, constructor and initial full
authentication. No preimport of NumPy/core is added. The proposed numeric work
calls only `core.probabilities` with Delta=0; no loss, gradient, `fit_arrays`,
optimizer or adapter is produced.

The20s scope still begins before NumPy import and catalog discovery. It covers
pre/post actual-library hashes, preparation, one native call, copies and both
replays, durable raw/array/catalog/state evidence, full native authentication,
hook and watchdog cleanup, followed by the literal monotonic check. The120s
scope still covers entry/preflight, cold construction, that call, final full
authentication/counters/state and watchdog cleanup plus its final monotonic
check. Primary `BaseException` and failure demotion are preserved. A retained
candidate state alone cannot establish acceptance.

## Unresolved prerequisites

The parent reports a closed native05 **root** review with digest prefix `ae81`.
This source-only pass has not read that actual receipt. Its full digest, path,
real schema and invocation/owner/source/model/current-closure joins are pending
in `plan.pending.json`. It must be bound honestly in future root preflight or
an explicit prerequisite validator before releasing this source hold. The old
NumPy actor contains no independent-smoke-review validator to inherit; this port
does not invent one or relabel root review as independent. Native05 smoke or
browser evidence cannot qualify this new NumPy path.

Root must first authorize and record source checks, final hashes/freeze and
independent source review; then prepare small metadata and a separate exact root
selection within the source-pool limits. Only a fresh owner09 run may discover
the actual ML3.13 NumPy version/origins, native extensions, BLAS configuration
and mapped mathematical-library candidates. Old CPU4 catalog identities are not
inherited. Actual numeric parity, complete-call latency and memory fit are
unknown. This discovery is not preselected trust or full transitive DSO closure.
Acceptance also requires original root exit0, exact owner09 completion, full
pre/post authentication, authenticated current closure, no timeout artifact and
independent actual review. None is claimed by these new sources.

All h/W0/catalog/raw payloads remain diagnostic `PENDING_NONTRAINING`, labelsNone
and `training_eligible=false`, including after any future numeric success.
No COMMITTED ledger, role-isolated collector/loader or optimizer is added.
Future training must first select an accepted closed catalog, consume only the
existing train split, use valid supervised heads and skip unused target heads.
Dev/sealed remain excluded from optimization; exposed toy tasks and failed
diagnostics remain excluded from training. Separate frozen-candidate evaluation
is still required. No browser quality, speedup, score or fit improvement follows
from this source port.
