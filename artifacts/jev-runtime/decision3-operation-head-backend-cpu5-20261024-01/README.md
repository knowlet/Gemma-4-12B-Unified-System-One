# Held operation-only CPU5 backend

Only source authoring. No import, test, AST, formatter, hash, actual/capability/
labels/data/array, NN or Git operation was run. The original shared backend,
runtime, observers and used/frozen files are unchanged. `SOURCE_ONLY=True` and
`CAPABILITY_SHA256=None` refuse before model/dependency/capability access. No
release, numeric qualification, selected artifact or runnable command is made.

New profile: `decision3-dsl01-operation-readout-cpu5-4g-candidate-v1`.
New alias: `d3-edge-dsl01-operation-delta-cpu5-candidate`.
The `DeltaHeadBackend` factory name is retained for caller composition, but its
new profile, acceptance ABI, SHA argument and projection signature are distinct.
No old shared32 capability or qualification authorizes this implementation.

The source is copied from the reviewed shared `backend_released_v1.py`, with
only the new role/acceptance checks and operation-row projection. Original
native `prepare`, actual `runtime.logits`, same-forward first-layer/final-norm
hooks, original `respond`, context16384, real batch4/one-to-three-row tail,
criterion order, T1, masks, entropy, raw evidence and required-retention failure
semantics remain. No offered operation/target is filtered, reordered, added or
repaired. An absent operation head is allowed and causes no delta matmul.

`project(torch, base_logits, features, delta, counts, ordered_head_ids)` must
receive that actual chunk's complete head IDs in row order. It never infers an
operation from row0, option names, K8, goal or state. Only rows whose exact ID is
`operation` receive `h @ delta.T`; their original per-row count mask is retained.
Every other row copies its original base logits, including illegal -inf and
signed zero, and a real Torch uint8-view equality check verifies unchanged bits.
The delta is never merged into W0 and is not even projected for target rows.
Evidence includes original head order, adapted/preserved IDs, true hook shapes,
legal base/delta/result logits and the target bit-equality result. This does not
claim that target softmax outputs across different batch shapes are universally
bit-identical: real runtime parity still requires its own future qualification.

Root agreed a **new accepted parameter artifact** ABI:

- `schema=root-accepted-dsl01-operation-train-delta-v1`,
  `status=ROOT_ACCEPTED_OPERATION_TRAIN_ONLY_READOUT`, `split=train`,
  `training_eligible=true`, `nontraining=false`, `only_head=operation`.
- Exact new `profile`, `alias`, original CPU5 `owner_profile`, full base `model`,
  `baseline_fp32_bytes_sha256`, `delta_ref`, `root_acceptance_ref`, and
  `fit_capability` path/SHA256/size refs.
- Honest selection provenance: `dev_used_for_model_selection=true`,
  `optimizer_input_dev=false`, `optimizer_input_sealed=false`,
  `candidate_selection_before_dev_or_sealed=false`.

The operation fitter's `root-committed-train-operation-fit-v1` /
`ROOT_COMMITTED_TRAIN_OPERATION_FIT` is only an input-authority ref. It is not
an accepted output delta and cannot pass this backend's accepted-capability
schema/status. ROOT must separately close/review the fit and issue the new
parameter acceptance. No recipe step count or source qualification is borrowed
from the old shared candidate. The accepted artifact cap must not contain this
backend's own source hash, avoiding a pin cycle; original owner selection pins
the cap and backend bytes independently.

Caller ABI adds mandatory `root_capability_sha256`. The owned caller must first
authenticate exact cap/delta bytes, root acceptance and original current closure,
then pass the parsed cap and its SHA equal to the future non-None module pin.
This no-IO backend validates the explicit metadata, exact CPU FP32[255,1024]
finite frozen independent copy and original storage/version identities; these
checks do not replace actual file/value-byte or baseline bit proof by the
caller. Neither a filename nor a fit status grants acceptance.

The original owned-call cooperative20 checks still include all required
retention, hooks, projection, original response and final identity work, with
equality20 rejected and exact BaseException retained through secondary failures.
The injected deadline must retain the caller's full20 authorization scope. The
original outer owner/actor whole120 still covers admission, load, artifact
authentication, calls and final retention; this backend is not a new guardian
or network-delivery timer and does not reset that outer budget. CPU15–19,
threads5/interop1,quota500%,4GiB/swap0 and original reserve remain externally
verified facts, not a new self-issued resource qualification.

Unexecuted source fakes reuse only the old stdlib fake-runtime source and cover
operation reordered away from row0 with K5, no operation, mixed legal counts,
real rows1–4 and a true4+1 tail, native target logits preserved bytewise, old-
cap/false-provenance rejection, pending holds, required-sink failure preserving
the exact original object, hook cleanup, and exact20 after terminal retention.
They cannot qualify Torch/BF16 numerics or any model/latency/memory result. Root
will review/check/release and qualify the new artifact/source before any actual
browser, dev/sealed or official evaluation. No such evaluation is run here.
