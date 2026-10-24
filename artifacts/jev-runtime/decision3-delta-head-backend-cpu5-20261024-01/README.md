# Held CPU5 train-only delta readout backend

Root source checks have passed 12 bounded stdlib fake tests. `SOURCE_ONLY=True`
still rejects production construction/preparation/projection/prediction before
any dependency or delta access; tests explicitly supply invented dependencies.
No genuine model inference, provider registration or browser run has qualified
this backend. The caller/plan must be selected separately; no guardian, artifact
loader or HTTP authority is supplied. All original sources remain unchanged.

The distinct numerical profile is
`decision3-dsl01-delta-readout-cpu5-4g-candidate-v1`, and the returned alias is
`d3-edge-dsl01-train-delta-cpu5-candidate`. Adapted probabilities are never
described as unchanged publisher-native numerics.

## Small composition boundary

The future owned caller injects the already authenticated native05 actor,
NP02 observer, genuine original runtime, Torch, full model metadata and a
separately accepted delta tensor. `NativeBatch4Backend` still authenticates the
original BF16 CPU runtime. Its unchanged `prepare` receives every original
state/question argument and preserves errors, ordering and the rejection-only
16384-token context limit. Unsupported extra kwargs reach that original method
and fail; they are never silently removed. This candidate supports its original
text-only non-permuted route, without inventing media support.

Each original runnable question is included once, with its prepared criteria
order intact, in chunks of four and a genuine one-to-three-row tail. Present
unused heads are not pruned; absent heads are not fabricated. A first-layer
pre-hook and final-norm hook reuse NP02 `FinalNormObserver`, returning `None` and
copying only the actual last-position FP32 features from the same original
forward. The encoder runs once per chunk through unmodified `runtime.logits`.

The exact composition is:

```
base_logits = original runtime.logits(original sequences, original counts)
delta_logits = same_forward_h_FP32 @ independent_delta_FP32.T
adapted_logits = (base_logits + delta_logits).masked_fill(original_count_mask, -inf)
probabilities = original Torch softmax(adapted_logits / 1.0)
raw = original runtime.respond(prepared, ordered probabilities, original token sum)
```

W0 is not merged, mutated or replaced. The native original base-logit operation
retains its original FP32 accumulation/order; adding h@delta.T is a separately
named operation. Original `respond` supplies typed outputs, criterion order,
entropy confidence (including a singleton), and usage. Its exact response
before candidate aliasing is preserved under
`respond_before_candidate_alias`; that response already uses adapted
probabilities and is explicitly **not a baseline control result**. Only the
returned copy's `model` is changed to the candidate alias.

Per-chunk evidence records actual rows/padded width, complete head/criteria
order, counts, logical input tokens, physical padded tokens, original hook
events, host forward elapsed time, and every legal base/delta/adapted logit.
It retains the complete ordered raw candidate response and partial completed
chunks on failure. This is inference evidence, not labels or success scoring.
The metadata/storage/version guard reuses `observer.baseline_identity`; it is
not a hash of live W0 bytes. Independent original-bit proof is still required
from the caller's selected source/model/artifact closure.

## Required caller authority, not produced here

`root_capability` is an explicit **caller-authenticated** dict with:

- `schema=root-accepted-dsl01-train-delta-v1`,
  `status=ROOT_ACCEPTED_TRAIN_ONLY_READOUT`, `split=train`,
  `training_eligible=true`, `nontraining=false`.
- Exact candidate `profile`, `alias`, owner09 `owner_profile`, and full original
  `model` dict matching the independently selected native model metadata.
- `delta_ref` and `root_acceptance_ref`, each with nonempty `path`, SHA256 and
  positive integer `size_bytes`, plus `baseline_fp32_bytes_sha256`.

Root has frozen the train-only candidate capability after the separate accepted32-step fit. This backend does not create or authenticate that receipt; its genuine owned caller and inference selection remain pending. The owned caller must authenticate those refs, the trained tensor's
exact file bytes and FP32 values, the accepted train-only fit and W0 identity,
producer root terminal/owner/current closure, source/numpy/model provenance,
and exclusion of diagnostic PENDING_NONTRAINING/dev/sealed/official/toy data.
A copied status or a filename is not acceptance. `root_acceptance_ref` points
to a separately selected acceptance record, not recursively to this dict.
This module never reads paths or selects a data filename and cannot establish
that an externally supplied tensor corresponds to an artifact on its own.

Delta must be finite, contiguous, frozen CPU FP32[255,1024]. The constructor
takes an independent detached copy; caller mutation cannot alter that owned
copy. The owned delta and original backbone/W0/encoder identities and tensor
versions are rechecked, without assigning any original model attribute.

Resources remain original owner09/native05: CPU15–19, five intra-op threads,
one inter-op thread, quota500%, 4GiB, swap0, original reserve and original CPU
BF16 encoder/reference kernels. Those host/owner facts are not self-qualified
by this module; no new guardian or CUDA disguise is introduced.

## Timing and error boundary

The original full-call `deadline()` callback is mandatory and remains the
outer authority. Cooperative checks surround preparation, each original
forward, projection, hook cleanup, observer callbacks, original response and
validation. A local monotonic20 check also rejects equality20. Final baseline
identity work and final evidence callbacks are charged before the backend's
last candidate authorization. Blocking operations are not hard-cancelled.

The future gateway must still charge its final evidence publication and its
actual HTTP200 authorization to the same full20 budget. This backend does not
claim network delivery by20. The original caller HTTP25/retries, browser ticks,
goal/judge and raw choices are outside this module and remain unchanged.

Original BaseException objects are preserved if hook removal, final identity
verification or an evidence sink also fails. Secondary cleanup failures become
notes. A supplied evidence sink is required retention: prepared/chunk/raw/completed failures reject the result, including BaseException; secondary failure-retention errors are added to the original error. Owned callers must preserve the audit
and latch native/cleanup failures before attempting another request. Hooks
installed by this request are removed in reverse order, including partial
registration. No unrelated hook is removed.

## Explicit unresolved integration boundary

The existing `JevDecision3NativeAdapter` checks raw.model against the original
provider's model_id suffix and rejects CPU in its original constructor. This
backend deliberately fabricates neither a CUDA spec nor original publisher
identity to pass those checks. A reviewed, selected CPU **candidate** caller
must bind this alias/profile explicitly while retaining the original native
DTO/criteria normalization and gateway evidence/20s rules. This is not yet
browser wiring and supplies no `.predict(Decision3NativeRequest)` provider.

Before actual root selection, verify genuine same-forward zero-delta
vs original and nonzero-delta comparisons on the selected runtime. Reconfirm
all original W0 bits, all candidate distributions, true rows/tail events,
owner/current closure and final deadline behavior. No genuine runtime check is claimed here. The12 `test_backend.py` fakes check matrix composition, tail/order,
mask/singleton behavior, independent storage, context/kwargs errors, primary
failure cleanup, required-retention failures and exact20 after observer/final identity work; they cannot
qualify BF16/Torch precision, model quality, speed or resource fit.

Source dependencies to pin later, without changing the originals:

- `decision3-publisher-batch4-cpu-20261024-05/actor.py` and its selected
  native plan, resource helper and original model/runtime source closure.
- `decision3-readout-numpy-cpu5-qualification-20261024-02/observer.py`.
- Original owner09/controller and an independently accepted train-only
  delta/capability. The existing head-core training arithmetic is reference
  only; this backend imports neither head-core nor NumPy.
