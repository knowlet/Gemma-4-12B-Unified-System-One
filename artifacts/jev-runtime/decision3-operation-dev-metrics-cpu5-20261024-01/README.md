# Held operation-only dev24 metrics

This directory contains source-only candidates. The author has not imported,
executed, parsed, formatted, hashed or tested these sources, and has not read
actual results, capabilities, arrays, labels or other data for this task.
`SOURCE_ONLY=True`; `CAPABILITY_SHA`, `CANDIDATE_SHA` and `DELTA_SHA` are `None`.
The source hold and all three pins are checked before arguments, paths or IO.
Root must qualify and release a new pinned copy before execution.

The source is a minimal successor of the committed
`decision3-frozen-heldout-cpu5-20261024-01/eval_metrics_released_v1.py`.
It retains the original `fit_head_released_v2.py` identity and uses only its
source-loading, original-owner admission and NumPy runtime authentication
helpers. It does not call the fitter, import a train loader, load a backbone,
run an optimizer, or open DSL exports/raw browser responses.

## Input selection and accepted candidate

The fixed future metric capability path is this directory's
`root-selected-operation-dev24-metrics-capability-01.json`. Root must provide:

- `schema='root-selected-operation-dev24-metrics-v1'` and
  `status='ROOT_SELECTED_OPERATION_DEV24_METRICS_ONLY'`.
- `split='dev'`, `nontraining=true`, `training_eligible=false`,
  `frozen_candidate_steps=512`, `only_head='operation'`.
- `model_selection_scope='dev_model_selection_only_no_unseen_claim'`,
  `dev_used_for_model_selection=true`, `optimizer_input_dev=false`,
  `optimizer_input_sealed=false`, `candidate_selection_before_dev_or_sealed=false`.
- Complete ordered `cases` metadata for exactly24 different cases and `rows`
  metadata for exactly33 supervised rows. All four actual offered heads per case
  remain present; only the existing valid supervised subset contributes to loss.
  Operation need not be first. Each row retains its true `row_index`, `case_id`,
  `head_id` and original `ordered_criteria_ids`.
- `arrays` with exactly `features`, `counts`, `soft_targets` and `W0` refs, plus
  `bundle_root`; `runtime` with exactly `root_np5_prerequisite_ref` and
  `numpy_runtime_catalog`; accepted `candidate_capability`, `delta_ref`, `model`.

Root reselects the original private dev bundle refs; no data is copied. The bundle
parent guard remains the original `decision3-frozen-heldout-cpu5-20261024-01`
directory through `BUNDLE_SOURCE_DIR`. Array refs must point directly into that
selected bundle. The allowlisted reader retains original SHA/size/stat checks,
finite exact no-pickle array declarations and readonly value fingerprints.

The candidate metadata path is fixed to
`decision3-operation-readout-cpu5-20261024-01/root-frozen-operation-candidate-capability-01.json`.
Its required schema/status are `root-accepted-dsl01-operation-train-delta-v1` and
`ROOT_ACCEPTED_OPERATION_TRAIN_ONLY_READOUT`. The source checks the operation-only
profile/alias, accepted train-only eligibility, exact model, delta ref and raw
FP32 W0 digest. Hyperparameters must be steps512, learning_rate0.003, l2=0.0001,
seed20261023 and batch_size64. The same explicit dev-selection flags above must
hold and `W0_merge_allowed=false`. No old shared32 before-dev qualification or
artifact pin is reused as authority for this new candidate.

Optional root metadata such as `source_dev_metric_capability`,
`source_dev_metric_acceptance` and `feature_collection_candidate_capability`
may retain the original shared32 collection provenance. Those original collection
refs are not rewritten to identify the new operation candidate. The actor does
not follow those additional refs or claim the data was recollected.

## Complete mixed projection

The actor validates both complete33-row inputs through unchanged core `_inputs`
before forming any subset, including all supervised target distributions.
It computes the original full zero-delta control once. Then it selects the24
operation indices from true `cap.rows[*].head_id`, computes only those rows with
the unchanged core `h @ W0.T + h @ delta.T` projection, and writes them into a
copy at the original indices. The remaining9 supervised target probability and
log-probability rows reuse the full zero-control bytes exactly; byte equality is
required. No offered head, option or target supervision is removed or repaired.

Both results retain the full33 supervised-row denominator, all24 cases and per-head
counts. There are63 unused offered heads, which remain outside supervision. CE
uses the original stable FP32 log-probability reduction once over the complete
original row order, not an average of the operation and target subgroup means.
Correctness remains positive authored target mass at the first offered argmax;
multiple valid targets stay valid. There is no L2, gradient or optimizer update.

The original CPU15-19/500%/five-thread/full4GiB/no-swap owner contract, fresh
NumPy admission, post-read authentication and enclosing120-second deadline remain.
Final output writing, fsync and the last authorization gate are inside that
deadline. A failure returns no partial score.

## Pending output and limits

CLI: `--owner-output <fresh-original-owner-output>`.
Output directory is `<owner-output>/operation-dev-metrics`, with
`metrics-nontraining-pending.json`. Summary schema is
`operation-dev24-cached-head-metrics-nontraining-v1`, status
`NONTRAINING_MODEL_SELECTION_PENDING_ROOT_ACCEPTANCE`, and profile
`dsl01-operation-dev24-cached-head-metrics-cpu5-4g-v1`.

`results.zero_delta` is the full original control; `results.frozen_candidate`
means the already accepted512-step artifact applied only to operation rows.
The name does not claim selection before dev exposure. The summary explicitly
records NONTRAINING, model-selection-only scope, no new unseen evaluation claim,
24 adapted operation rows,9 target rows retained from control, zero optimizer
steps and the full denominators. It contains aggregate metrics and input hashes,
not raw arrays, per-case answers or probabilities. A separate root audit must
accept the genuine completed job and original scoped closure before reporting it.

These are development model-selection metrics. Dev has already informed this
policy; it cannot serve as a new unseen evaluation. Sealed data remains excluded.
DSL train/dev4-head and8/4-choice distributions also differ from real browser
requests, so these metrics cannot establish browser success, an official JevBench
score, a ranking win or a deployment qualification.

`test_operation_dev_metrics_held.py` preserves prior fake coverage and adds fixed
24/33/63 denominator, non-first operation routing, exact target-control reuse,
complete validation before subset projection, and partial-score rejection.
Its stdlib protocol fakes test control flow, not NumPy/model numerical parity.
These are authored test sources with no claimed execution result.
