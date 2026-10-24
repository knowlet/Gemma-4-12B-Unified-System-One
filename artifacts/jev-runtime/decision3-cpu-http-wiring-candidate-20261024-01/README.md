# CPU HTTP wiring source candidate — not executable or qualified

This independent directory is **source only, unexecuted, unfrozen and unselected**.
No import, AST, test, formatter, build, dry run, host observation, HTTP request,
browser, model, or service was run while writing it. Root's later source hashes
and text-only review are not executed validation, a freeze or a selection.
The active NN campaign and every frozen source stay as-is.

`SOURCE_ONLY=True` makes the public factory refuse before plan access, project
imports, dependency construction or output creation. Constructors of the draft
CPU adapter, backend shim and ASGI app are also gated. Removing that flag alone
still refuses: selection, guardian, pins, full resource caps and a reviewed live
handoff are missing. There is no model loader or server/command entry here.
Private source internals are draft code, not an alternative authorized entry.

The proposed wiring is:

1. A future owned entry authenticates a complete CPU source/resource identity,
   admits the original resource profile, then hands in **one already loaded**
   selected `NativeBatch4Backend` runtime. It must not call the CUDA loader.
2. An honest CPU adapter subclass inherits the original `predict` unchanged.
   Its constructor explicitly requires `device==cpu`, BF16 and context16384.
   All original configuration blockers except the combined CUDA/BF16 blocker
   still reject; BF16 is checked independently before filtering that blocker.
3. The shim calls original `native_payload` on the DTO, retains complete ordered
   questions, invokes the original actor's `NativeBatch4Backend`, and returns
   its original raw result. Original inherited `predict` normalizes once.
4. Original `named_browser_response` changes only ordered-list answers to the
   named answer container expected by pinned Jev. Its legacy-v1 marker is only
   an ABI binding, **not GPU qualification**. Separate CPU metadata records the
   honest resource identity, generated request ID and exact body hashes.

The first decoder-layer hook records actual CPU BF16 tensor shapes and question
order. It expects native chunks of at most four real rows, including 1–3-row
tails; no dummy questions make a two-head browser call look like batch4.
Hook events prove entry into that layer, not completed inference. A separate
native raw-return/validation event establishes completion. The hook returns
`None` and does not modify tensor values, candidate lists or probabilities.
Question IDs in the hook are bound from the expected prepared order, not read
from tensor contents. Actual order still depends on the selected original
actor/runtime sources and must be authenticated in the real run.

The only intended first task is the existing invented two-lamp fixture in
`tests/test_jev_native_http_browser.py`. Both controls must initially be visible
and enabled. Strict success requires a fresh visible copper confirmation,
upstream DONE without error, and exactly one trusted copper click. Wrong clicks,
premature DONE, BLOCKED, errors and timeouts remain unchanged evidence.

Pinned upstream `model.py:81–116` builds operation choices from currently
available target kinds and controls, then adds DONE/BLOCKED. The two-button
fixture has no fill/select targets. This proposal preserves **every** original
offered operation/target and their order. It never drops DONE/BLOCKED or repairs
a choice that conflicts with the goal. It even retains TYPE_TEXT/SELECT offered
without corresponding heads; choosing an unsupported action follows the caller
failure path unchanged. Actual text/select/freeform heads are outside this toy
diagnostic and cause whole-request rejection, with the original bytes retained.
There is no helper endpoint, helper load, field generation or fallback: helper
calls must be zero. This is not a general browsing or public-benchmark route.

The proposed ASGI timer starts at request entry and counts body reads, parsing,
native preparation/forwards, native/wire validation, observer work, serialization,
candidate/terminal writes and fsync. After terminal evidence and response headers
are prepared, the final successful authorization check runs immediately before
`http.response.start`. Boundary equality (`elapsed >= 20`) rejects 200. A late
candidate is retained unchanged, but HTTP body/status/length and terminal
exchange become 504. The inherited adapter retains its own original cooperative
20-second checks; no budget is increased. Pinned upstream HTTP25 is unchanged.
Neither the cooperative timer nor ASGI authorization guarantees cancellation of
a blocking CPU forward or successful network delivery after response.start.

The per-runtime lock spans request handling, evidence publication and ASGI sends.
Busy requests perform no inference. This does not claim enforcement against a
different process: single owned runtime/process closure is still pending.
All outputs must go into a newly created ordinary ignored directory. Request
IDs and filenames come from the bridge, never request values. Files use 0600,
request directories 0700; only final HTTP response/exchange can be atomically
replaced. Candidate/events remain immutable. Raw body and canonical JSON hashes
are distinct, and incomplete body captures are marked incomplete. Authorization
headers, API keys and environment variables are not logged. Observer failure
blocks 200; publication/cleanup failure cannot replace an earlier primary error.
New output and request-directory entries are also parent-fsynced. Request-parent
fsync occurs after the request timer starts. A constructed candidate is marked
retained only after its file and directory write/fsync return successfully.
Observer interrupts retain the original exception object, which is rethrown
at the next gateway check even if the original adapter isolates that observer
failure. This does not ensure an immediate abort inside a blocking native call;
the independent guardian remains unimplemented and required.

## Source references and unresolved binding

- `src/s1/jev_decision3_native.py`: `parse_native_payload`, native DTOs and exact
  inherited adapter `predict` (no mutation of the module's profile globals).
- `src/s1/evaluation/decision3.py`: `decision3_spec(device="cpu")`, original
  `configuration_blockers`, `native_payload`, `normalize_native_response`.
- `scripts/jev_decision3_browser_wire.py`: unchanged `named_browser_response`.
- `artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261024-03/actor.py`:
  original `NativeBatch4Backend`, `validate_runtime`, native raw validation.
  Its existing synthetic `ShapeWitness` requires exactly four rows and is not
  reused for the two-head browser case. The separate source observer here
  records actual 1–4 row chunks. No model/scorer/readout math changes.
- Publisher `d3_runtime.py:1673–1698,1996–2023`: original left padding, native
  request-order chunks and true final partial batches; no custom padding added.
- New native04/owner08 CPU16/4GiB sources exist but are not selected here. They
  need their own completed qualification/source pins. Native03/owner07 resource
  identity must not silently be relabeled CPU16/4GiB.

`plan.pending.json` is descriptive and is not read by the blocked factory.
`test_cpu_http_bridge.py` contains unexecuted stdlib-fake test source. It patches
the source gate only inside fake tests; no test loads Torch, a model, a live
endpoint or browser. It cannot establish source correctness, actual native tail
behavior, peak resources, task success, latency or readiness until separately
reviewed and executed under a released owned runtime.
