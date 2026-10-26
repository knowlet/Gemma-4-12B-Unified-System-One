# Held original-V7 matched browser caller

`caller_held.py` and `test_matched_caller_held.py` are source-only candidates.
The author has not imported, executed, parsed, formatted, hashed or tested them,
and has not read actual receipts, capabilities, weights, data, DOM or gold.
`SOURCE_ONLY=True`, `CAPABILITY_SHA256=None` and `V7_LAUNCHER_SHA256=None` remain.
No CLI can release these holds; standalone launch is refused even after source
release. Root must qualify a new copy and select exact pins before an owned
worker can call it. The separately owned `worker_held.py` is outside this
author's edits; no NN/service/resource-cap implementation is supplied here.

## Original ten-case protocol

The main comparison is the original known4 plus public6:

- `public-arxiv-title-filter-results-v1`
- `public-arxiv-original-paper-open-v1`
- `public-python-counter-most-common-v1`
- `public-nasa-mars-facts-moons-v1`
- `heldout-public-000` through `heldout-public-005`

Known4 uses `case_module='original'`, public6 uses `case_module='heldout'`.
Each call uses the original V7 `prepare_bundle`/`verify_bundle`, imports that
verified bundle runner, and invokes its original `run_benchmark` with exactly one
`public_case_ids` entry. The original cases/judges, seed20261014, repeats1,
max_steps16, development split and standard action profile remain. There is no
alternative task, travel addition, operation pruning or answer repair.

Observation profile is `composed-visible-text-settled-v7`, with exact helper SHA
`d9f4236d1d46c9472c3d6c1551342639ecb870e6c0dd419246927645410ea868`.
The unchanged launcher authenticates the original benchmark/common helpers,
tracker/profile parents and heldout source, verifies its own permitted inverse
source changes, and verifies bundle files before and after execution. The new
caller authenticates the original launcher through the pending root-selected
`V7_LAUNCHER_SHA256`; it does not create another runner patch.

## Owned-worker API

`call_case(case_id, output, *, policy_endpoint, policy_model, text_endpoint,
text_model='generated-actions-nf4', upstream, chromium, capability_sha256,
verify_owned, policy_api_key=None, text_api_key=None)`

`output` is a fresh per-case directory. Policy endpoint/model and helper
endpoint/model are passed independently to the original runner. The text model
must be the shared `generated-actions-nf4` helper, not the D3 policy alias. API
keys are independent values. The caller neither appends endpoints nor supplies
another helper callback, changes request limits, or creates its own retry loop.
As in original `native_v2.run_groups`, the actual `run_benchmark` call is scoped
with `TEXT_MODEL_REASONING=none`, recorded as `text_model_reasoning='none'`.
The previous environment value, including absence, is restored in `finally`
after normal completion or BaseException. Upstream's default low reasoning is
not allowed to silently change the original NF4 helper request contract.

`verify_owned` is a trusted, worker-supplied callback, called before execution
and during finalization. The worker must authenticate current ownership, actual
capability bytes matching `capability_sha256`, all source selection, and the
separately owned original NF4 helper service. This is a process-local integration
contract, not proof against an arbitrary Python caller that supplies a fake
callback. The matched helper must retain its original base/revision/NF4 runtime,
disabled action adapter, grammar/EOS,128 generated-token and20-second contract.
The caller does not infer any of that from an endpoint/model string, load the
helper, or claim its runtime qualified; returned records explicitly set
`helper_runtime_qualification_claimed=false`. Root's outer evidence must supply
the genuine helper identity and closure before declaring a matched comparison.

The outer worker controls whole-job resources, time, fatal guard, service and
browser lifecycle. This source imposes no substitute120-second decomposition,
does not lower16 decisions, and does not split an individual task history.

## Pass-through HTTP evidence

The caller temporarily wraps the original runner's `load_upstream` context only
to install the same `model_module.CLIENT.post` pass-through observer pattern
used by `tests/test_jev_native_http_browser.py`. Original `model_adapter` still
wraps that client for its normal policy/text records and retries. Every observed
attempt calls the real original `.post(url, **kwargs)` exactly once, with the
same values, and returns the same response object. It never edits `post_json`,
choices, normalized answers, helper text, timeout or retry behavior. Both client
method and original loader are restored on normal and exceptional paths.

Per-case layout is:

- `runtime-bundle/`: original verified V7 derivative and manifest.
- `results/`: original runner outputs, unmodified.
- `http/attempt-0001/{request.bin,response.bin,metadata.json}` and later attempts.
- `http/http-attempts.json`: every real transport attempt in observed order.
- `caller-receipt.json`: infrastructure status and original task result.

Actual `response.request.content` and `response.content` bytes are retained
without decode/re-encode. Original raw header byte pairs, including duplicates,
are stored as base64 pairs; request/response SHA256, response status,
`X-Request-ID` and latency of the real HTTP call are recorded. This is the same
HTTPX object-content scope as the original observer, not a network packet or
compressed-wire-byte claim. No JSON serialization of kwargs is substituted for
an unavailable actual request. A transport exception's actual `.request` is
used when available; no response is invented when transport returned none.

Ordinary observation/retention errors are latched until the case boundary,
allowing the original transport result and retry policy to remain unchanged.
They prevent infrastructure completion even if a later HTTP retry succeeded.
An original transport BaseException is rethrown as the same object; secondary
retention errors cannot replace it. A fatal observation BaseException escapes
immediately when there is no original transport error already primary. There
is no assertion of immediate cancellation for a latched ordinary failure.

Raw bodies and headers are private; headers can contain API credentials.
Fresh directories/files use0700/0600 and the per-case `.gitignore` excludes all
outputs. Do not publish or commit raw traces, task outputs or credentials.
Only reviewed aggregates and immutable metadata belong in a public report.

## Results and complete denominator

A completed case returns `case_id` as the original ID, `group='known4'/'public6'`,
`infrastructure_complete`, and `task_success` copied from the original judge.
The original outcome must identify `case_id+'-000'` and `task_id=case_id`.
The record also includes fixed protocol fields, independent model/endpoints,
`summary_ref`, `http_trace_ref`, `bundle_manifest_ref` and `receipt_ref`.
An ordinary task failure, legal BLOCKED, false DONE or exhausted16-step budget
returns `infrastructure_complete=true/task_success=false` when the original
runner and required evidence completed. Such a result is not raised as a fatal
wiring error. Infrastructure/fatal/retention failures retain a partial record
as `exception.jev_v7_case_record` and rethrow the original primary object.

`aggregate_cases(records)` requires exactly the ten original distinct IDs and
matching group/profile/helper SHA/seed/repeats/16-step settings, shared NF4 helper
identity and one policy model. Different service endpoints per owned case are
allowed. Every completed false task remains in denominator10. Missing or
duplicate cases cannot produce a score. Any incomplete infrastructure keeps
denominator10 but returns `success_rate=null`; it cannot manufacture a9-case
comparison or convert an interrupted task to a complete failure/success.

Root must retain matching fixed fields in any outer infrastructure placeholder.
These are main development-comparison sources, not unseen/sealed evaluation,
official JevBench results, a ranking win, or an executed qualification.

The authored stdlib fakes exercise original HTTP bytes/duplicate headers,
one-call pass-through, each real retry, primary BaseException preservation,
deferred retention rejection without added retries, context restoration,
fixed10 denominators, exact runner arguments and completed false-task return.
They use invented data and mocked runner/launcher modules only, and have not
been executed by their author.
