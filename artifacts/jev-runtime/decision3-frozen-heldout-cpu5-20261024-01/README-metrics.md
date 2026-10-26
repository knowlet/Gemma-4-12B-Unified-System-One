# Held dev24 cached-head metrics

Only new `eval_metrics_held.py` and its small stdlib fake-test source are added.
No existing collector, used source, profile, data or artifact is changed. These
files have not been imported, parsed with AST, formatted, checked, hashed or
tested. No actual, split, capability, raw dev or model payload was read while
authoring. `SOURCE_ONLY=True` and `CAPABILITY_SHA=None` refuse before arguments,
owner paths, dependency imports or array reads. Root owns later source checks,
pins, review, release, command spec and execution.

This is a distinct evaluation role: **permanently NONTRAINING**, split `dev`.
There is no train-loader import/call, fit invocation, gradient, optimizer, L2,
DSL import, labels export, raw-dev parser, model or browser. Reusing the pinned
fit module means calling only its `source`, original owner09 `owned` admission
and `numpy_runtime` authentication helpers. A small independent read-only facade
provides only preselected trusted runtime metadata to the latter; it does not
inherit the train loader or authorize a train capability.

Root capability ABI, not generated here:

- `schema=root-selected-frozen-dev24-metrics-v1`,
  `status=ROOT_SELECTED_FROZEN_DEV24_METRICS_ONLY`, `split=dev`,
  `nontraining=true`, `training_eligible=false`, `frozen_candidate_steps=32`.
- `bundle_root`: one dedicated ordinary private child directory of this package.
  `arrays` contains exactly `features`, `counts`, `soft_targets`, `W0` refs. Each
  pins path/SHA256/size, `format=npy_no_pickle`, dtype and shape. All four files
  are inside that bundle: FP32[N,1024], int64[N], FP32[N,255], FP32[255,1024].
  W0 additionally pins its original `FP32_bytes_sha256`. No train feature or raw
  collector file is accepted as an array input.
- `cases` preserves all24 actual case IDs/order, each with original four
  `offered_heads` order and authored `supervised_heads` subset/order. Target-head
  insertion order may vary by case and is never canonicalized. Operation
  is supervised in each case; other unused heads are skipped, never fabricated.
- `rows` contains exactly the flattened case/supervised-head order with
  `row_index`, `case_id`, `head_id`, `ordered_criteria_ids`. No duplicate,
  missing, reordered or foreign slot/head is accepted. Array counts must equal
  the ordered criteria count in every row. Root retains this private mapping
  with the capability; raw labels/source DSL are not sent to this actor.
- `candidate_capability` pins the already frozen32-step candidate root metadata
  (SHA239165d4…), `delta_ref` matches that exact metadata (SHA81225613…), and
  `model` matches its complete actual base identity. Its original W0 byte hash
  must match this bundle's W0. No candidate change/tuning is selected by dev.
- `runtime` has only `root_np5_prerequisite_ref` and `numpy_runtime_catalog`,
  each selecting trusted already accepted root metadata. Their exact catalog
  join is verified before the original NumPy runtime checks. All refs are
  separately byte-authenticated and reauthenticated before terminal retention.

The numerical core is the exact pinned head_core `_inputs`/`_probabilities`
forward-only arithmetic: FP32 h@W0.T + h@delta.T, original count mask and T1.
Zero delta and the fixed candidate evaluate **the same complete supervised dev
rows**. Full mean soft CE uses stable log probabilities and the frozen FP32
reduction. Per-head mean CE uses the same FP32 per-row CE values. A first ordered
argmax counts as correct when the authored soft target at that slot is positive;
multiple legal valid targets remain valid. Targets are checked by the frozen
core for finite normalized positive mass and exact zero illegal-slot mass.

Outputs separately report full supervised-row soft CE/hit, each original head
(including operation), and the fraction of all24 cases whose **every supervised
head** is correct. The case denominator never shrinks after failures or skipped
unused heads. A missing case/head refuses evaluation instead of producing a
partial score. Empty unused head groups report null accuracy/loss, not a fake
zero or perfect score. No arrays, probabilities, per-case labels or raw inputs
are written; the exact private mapping remains available by capability ref.

Original owner09 remains the sole external authority: CPU15–19, five threads,
one inter-op profile, quota500%,4GiB,swap0 and original reserve. New ignored
`owner_output/dev-metrics/metrics-nontraining-pending.json` is exclusive and
permanently `NONTRAINING_PENDING_ROOT_ACCEPTANCE`. Whole120 strictly rejects
equality and includes authentication, readonly array-byte retention, selected
NumPy/source rechecks, final fsync and cleanup. No numerical partial result is
returned before those checks. Original root terminal0, owner completion and
current scoped closure are still necessary; this actor does not grant them.

Root must first closed-audit all three dev chunks and construct the entire
private metric bundle/capability in the agreed ABI. Then pin and review the
new actor and capability in a fresh original-owner selection. The included
unexecuted fakes cover early hold,24-case denominator, unused heads, per-head/
all-heads accounting, missing/duplicate/reordered mappings, permanent dev role,
candidate step freeze and denial of unselected reads. They do not establish
NumPy precision, real data validity, model quality or a browser/official score.
