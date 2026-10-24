# Held operation-only CPU5 readout

These are source-only candidates. They have not been imported, executed,
parsed, formatted, hashed or tested by their author. Both sources remain
`SOURCE_ONLY=True`; the fitter's new capability and loader hashes are `None`.
Root must review/check new source copies, select exact source/capability pins,
and create a fresh original-owner command selection before a genuine run.
There is no runnable capability or execution receipt supplied by these files.

`train_loader_held.py` is a minimal successor of the original
`decision3-train-readout-cpu5-20261024-01/train_loader_released_v1.py`.
`fit_head_held.py` retains the ownership, CPU/runtime authentication and direct
child scope from that directory's `fit_head_released_v2.py`. Existing sources,
collector files, COMMITTED acceptances, datasets and `head_core.py` are unchanged.

## Training view

The loader retains the entire original train-input/label/summary validation,
12 COMMITTED chunk checks, current original closure and actual child joins,
source/model/W0 pins, every same-forward native replay, four actual head rows
per case, all supervised soft targets, and every original payload reference.
`TRAIN_SOURCE_DIR` explicitly preserves the original collector parent guard;
the successor's own directory is not used as an alleged collector origin.

Only after all 96 states, 132 supervised rows and 252 unused rows have been
verified does it select by the retained real `head_id == 'operation'`.
Selection requires exactly 96 different cases, one operation row per case and
K=8 for each operation. The 36 supervised target rows are verified and excluded
from optimization. Unused target heads continue to be skipped without invented
labels. Selection never guesses a head from row position or option count.

`load_training(selection, np, core, *, check_deadline)` retains its interface.
Returned `features[96,1024]`, `targets[96,255]` and `counts[96]` are readonly FP32,
FP32 and integer arrays. Original readonly `baseline[255,1024]`, all
`pinned_refs`, and `aggregate` remain available. Additional outputs are:

- `row_map`: the ordered 96 selected operation metadata rows.
- `excluded_supervised_row_map`: the 36 verified supervised target metadata rows.
- `row_mapping_sha256`: the selected operation mapping digest.

Each mapping row combines the original six identity fields with the validated
producer row: `case_id`, `dsl_version`, `family_id`, `ancestor_id`,
`request_sha256`, `native_payload_sha256`, `head_id`, `ordered_criteria_ids`,
`legal_count`, `h_index`, `same_call_id`, `native_call_index`, and
`supervised_row_index`. The last index is its zero-based position in the full
132 supervised rows, before selection. List order remains the original selected
chunk/case/head order. The digest is SHA256 over
`json.dumps(rows, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
allow_nan=False).encode('utf-8')`. Request/projection digests retain their prior
insertion-order encoding. No case IDs or targets belong in a public report.

## Required root capability

The fitter requires a newly selected SHA and the loader requires:

- `schema='root-committed-train-operation-fit-v1'`,
  `status='ROOT_COMMITTED_TRAIN_OPERATION_FIT'`, `split='train'`.
- `profile='decision3-dsl01-operation-readout-cpu5-4g-candidate-v1'`,
  `alias='d3-edge-dsl01-operation-delta-cpu5-candidate'`,
  `training_strategy='operation_only'`.
- `hyperparameters`: steps512, learning_rate0.003, l2=0.0001,
  seed20261023, batch_size64; `fit_budget_seconds=120`.
- Original `projection`, exact original `model`, `train_inputs` with only
  requests/labels/summary, original eight `source_records`, original
  `collector_source`, `root_np5_prerequisite_ref`, `baseline` including its raw
  FP32 digest, and all 12 original `chunks` with COMMITTED acceptance refs.
- `dev_used_for_model_selection=true`, `optimizer_input_dev=false`,
  `optimizer_input_sealed=false`.

COMMITTED acceptance headers still expect the original train data, source,
collector, model and projection fields; they are not rewritten for this policy.
Changing optimizer policy does not replace their original acceptance authority.

## Fitter and pending output

The root-owned future released loader path is
`train_loader_released_v1.py` in this new directory. The held fitter's `LOADER`
constant points there; no file at that future path is supplied here. Root must
create/review a released copy and pin `LOADER_SHA256` plus `SELECTION_SHA256`.
The fixed original owner09 and head-core identities remain unchanged.

The CLI remains `--selection <new-root-capability> --owner-output <fresh-owner-output>`.
Before arguments/IO, `main` checks the source hold and new pins. `execute` repeats
the hold, uses the original ownership and NumPy admission, validates the new
policy before NumPy import, loads the complete authenticated train view, and
passes only its 96 selected rows to unchanged `core.fit_arrays`.

The original core starts delta at zero, performs deterministic FP32 masked SGD
with the five fixed parameters, and never merges or updates W0. No old candidate
delta is loaded as a warm start. The original CPU15-19/500%/five-thread/full4GiB/
no-swap contract and enclosing120-second deadline remain. Full post-fit payload,
runtime and selected-source authentication, durable output writes and final
deadline checks are retained. Whether512 steps fit within120 seconds is unknown;
no reduced-step fallback or budget increase is authorized by this source.

Fresh output is `<owner-output>/fit-operation-head/delta-pending.npy` and
`fit-summary-pending.json`. Summary schema is
`dsl01-owned-operation-readout-pending-v1`, status
`PENDING_OPERATION_READOUT_ROOT_ACCEPTANCE`, `training_eligible=false`.
The summary includes `row_mapping_sha256` and both top-level and train aggregate
counts: `supervised_heads=96`, `total_verified_supervised_heads=132`,
`supervised_target_heads_excluded=36`, `unused_heads_skipped=252`.
It stores only the mapping hash, not the actual mapping rows. Root can rebuild
the hash from original authenticated requests, labels and chunk row metadata.
Only a separate new closed root audit can accept the trained artifact.

## Scope and remaining qualifications

Dev observations informed this new policy. The candidate cannot claim it was
frozen before any dev exposure. Dev/sealed features and labels are never training
inputs; the already exposed browser toy is not training material.

DSL01 train and dev states contain four heads, eight operation choices and four
choices per target. Real browser requests often contain two heads, four operation
choices and one or two target choices. No offered options are removed, reordered
or relabeled here. A later operation-only backend must route the frozen delta
using true head IDs and preserve all other heads' original base logits, original
prepare/respond behavior, complete20-second authorization and required evidence
retention. This directory's training sources do not qualify that inference path.
Future variable-K augmentation needs an independently authored train-only source
from authentic observed affordances, without dev or exposed toy reuse.

`test_operation_readout_held.py` preserves the prior stdlib inventory, acceptance,
producer identity, GPU stat-only and child-resource fake cases. Added authored
fakes cover132-to96 head-ID selection, equal-count target exclusion, reordered
head positions, missing/duplicate/wrong-K rows, excluded-label rejection,
canonical mapping hashes, the new policy and pending-pin holds. They are test
source only, with no claimed result. No genuine fit, numeric/backend/browser
qualification, evaluation score or ranking improvement is claimed.
