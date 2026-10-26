# Held full-choice CPU5 bridge source

This is source-only, unexecuted, unselected and unfrozen. `SOURCE_ONLY=True`
refuses before the owned verifier callback, dependency imports, gateway/model
construction or service use. There is no CLI, loader, listener or guardian here.
The fake tests are authored source and have not been imported or run.

Parent source: `../decision3-cpu5-browser-worker-candidate-20261024-01/bridge_successor.py`.
The parent and all already-used/frozen sources are unchanged.

## Fixed bridge ABI

- `bind_owned_selection(check)` returns the same process-local `_OwnedRelease`;
  every use rechecks the original owner09/native05 proof without trusting HTTP.
- `create_application(plan_path, *, owned_release, actor, runtime, torch, model,
  model_identity, output, cpu_identity, backend_wrapper)` keeps the original
  baseline composition, now with explicit alias `d3-edge-cpu5-full-choice`.
- `_Gateway`, `_BackendShim`, `_RowWitness`, `_cpu_adapter_type`, `_new_output`
  and `_draft_dependencies` keep their original argument/return contracts.
  An explicitly named adapted provider must supply its own adapter, verify its
  candidate alias and construct `_Gateway`; it must not impersonate the base.
- `_full_choice_scope(payload)` replaces `_toy_scope`. It validates the whole
  request in place without writing to, copying back, selecting, filtering or
  reordering any part of the original payload.

The source references are `src/s1/jev_decision3_native.py` (`parse_native_payload`
and the native DTO), `src/s1/evaluation/decision3.py` (`native_payload` and native
normalization), `scripts/jev_decision3_browser_wire.py` (named wire conversion),
and pinned upstream `jev_ultrafast/model.py::choose` plus `questions.py`.

## Narrow derivative

The only protocol expansion is from operation/click-target with one or two
click candidates to the actual standard choice head IDs: `operation`,
`click_target`, `type_text_target`, `select_target`. Any available combination of
the target heads is allowed; operation remains required but may occur anywhere
in the original ordered mapping. Each offered head has 1..255 labels. All eight
standard operations remain legal when offered, including DONE and BLOCKED.
Those two terminal choices must stay present as in the pinned upstream caller.
No operation is removed because its corresponding target head is absent, and
no missing target head is synthesized. Freeform, media, unknown heads, unknown
operations, empty/oversize heads and missing terminal choices reject the whole
request. The native parser continues strict DTO/outer-schema validation and the
original 255-label and rejection-only context16384 boundaries without truncation.

Other changes are an honest full-choice profile/alias, the early source-only
hold, and removal of the word “toy” from the unchanged 1MiB body-limit error.
The original native `predict`, ordered DTO conversion, actual 1..4 row-tail
witness, raw-before-validation events, normalization, named-wire conversion,
original owner/nonce/selection binding and all Gateway retention code remain.

Gateway serialization and its cooperative20s final200 authorization include
candidate retention, response/exchange file writes, fsync and header preparation.
The equality boundary expires. An expired candidate remains immutable; only the
actual final HTTP response and exchange records are atomically replaced with
504. The original25s external caller budget/retries are not changed. This does
not guarantee network delivery or hard cancellation. First native/abort errors
are retained across later evidence/send failures and lock cleanup is unchanged.

## Helper and execution scope

There is only POST `/v1/systemone`. TYPE_TEXT and SELECT choice heads are policy
decisions, not freeform generation. This gateway never loads or calls a text
helper, so its existing `helper_calls=0` describes this CPU service only. The
matched campaign's unchanged helper is served by its separate NF4 process;
no campaign-wide helper-zero claim is made. No answer/goal repair or fallback
is added. The base CPU constructor remains honest BF16/CPU/context16384 and
preserves all original non-device configuration blockers; legacy v1 is a wire
ABI, not GPU/publisher-native qualification for an adapted candidate.

## Pending root selection

New bridge source SHA, test SHA, upstream source pin, native adapter/wire source
pins, service/plan/spec/source closure, owned admission and final release are
all **None / pending**. They must be supplied and authenticated by the outer
root-selected service/owner plan. No plan SHA is embedded here, avoiding a
source/plan hash cycle. `SOURCE_ONLY` must remain true in this held original;
root can review/check and select a distinct released copy. Existing binding
values do not by themselves qualify this new bridge or any full-ten campaign.
No model/browser/HTTP run, actual artifact read, resource observation, test,
import, AST/format/hash check or qualification was performed while authoring.
