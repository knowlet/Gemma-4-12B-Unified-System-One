# Held external CPU-controller pump

`pump_held.py` is a source candidate. `SOURCE_ONLY=True` and
`TEMPLATE_SHA256=None` independently prevent construction, polling, cleanup and
closed-child verification. `main()` checks these holds before using arguments
and does not provide a standalone launcher after release. No import, test,
formatter, AST check, source hash, host probe, artifact review or model execution
was performed by this source author. The stdlib fakes are authored verification
source, not passed results.

The integrated root owner must hold `Pump(plan, directory, guard)` outside the
NF4 unit. It calls `poll()` during its normal supervision and `close()` in its
cleanup before claiming outer closure. A CPU controller launched inside the GPU
unit is not supported: stopping the GPU unit must not remove the process which
is responsible for closing the separately owned CPU unit.

The pump uses the frozen original owner09 `controller.py` entry and its original
CPU15–19, 500%, five-thread, 4 GiB, swap-zero scope. Its command specification
selects original hard7200 seconds; the HTTP-only service declares cooperative1200
seconds separately. The outer NF4 40 GiB limit does not cap the CPU sibling.
No native arithmetic, batching, offered choices, HTTP deadline, retry, browser
judge or helper behavior is changed here. The NF4 helper remains in the GPU
campaign; no helper endpoint, helper token or helper model is sent to the CPU
service.

## Selected template and runtime expansion

The outer selected plan adds `matched.cpu_template`, an ordinary small source
reference `{path, sha256, size_bytes}` with repository-relative path. The new
pump source has one pending final template SHA. The root-selected template has:

```text
schema: jev-v7-matched-cpu-service-template-v1
status: ROOT_SELECTED_TEMPLATE_ONLY
controller_source: original owner09 source reference
candidate_capability_ref: exact accepted operation artifact reference
numeric_acceptance_ref: exact accepted operation numeric reference
service_plan: static selected HTTP-only service settings and source references
command_spec: original llama-cpp-command-spec-v1 settings
cases: ten ordered case_index/case_id/group/owner_output/nonce records
```

The static command has one worker and whole-token `{SERVICE_PLAN}` and
`{SERVICE_PLAN_SHA256}` substitutions. Original `{OUTPUT}` substitution is left
for the original CPU controller. At most56 static selected sources leave room
for the runtime service plan and shared serialization metadata under the
original58 selected-source and64 source-union bounds. Original five base refs
are separately reconstructed and compared with the original sealed config.

The template and outer plan select the generator/static dependencies. The pump
reads and authenticates the private `directory/selected-plan.json` bytes and
puts that SHA in the generated child service plan. No child plan, command spec,
template or outer plan needs to embed its own hash or future child hash. The
runtime service plan also contains the admitted GPU actor identity, case index,
case nonce, original CPU output path and serialization descriptor/ref.

The fixed order is the original V7 known4 followed by public6:

1. `public-arxiv-title-filter-results-v1`
2. `public-arxiv-original-paper-open-v1`
3. `public-python-counter-most-common-v1`
4. `public-nasa-mars-facts-moons-v1`
5. `heldout-public-000`
6. `heldout-public-001`
7. `heldout-public-002`
8. `heldout-public-003`
9. `heldout-public-004`
10. `heldout-public-005`

The template must select ten distinct fresh original owner09 `actual-NN`
directories and ten distinct case nonces. The pump accepts one case at a time;
later request files cannot skip a case or create concurrent CPU controllers.

## Private handoff

The pump creates mode0700 `handoff/case-NN` directories. Metadata publications
use a mode0600 temporary file, complete write/fsync, exclusive final hard link
and parent-directory fsync. Reads reject symlinks, duplicate keys, nonfinite
JSON, nonregular or oversized metadata and changing file identity. Callback
and service handoffs additionally require same-user mode0600 files.

The original GPU entry's private actor-intent plus fresh GPU unit invocation,
actor PID/session/start ticks and cgroup authenticate each request before any
CPU controller starts. The root-owned controller `Popen` is kept before further
metadata reads and is started with a separate session, without a shell.

The root/callback ABI is:

| File | Producer | Required identity |
| --- | --- | --- |
| `request.json` | GPU callback | schema `jev-v7-matched-case-request-v1`, root_plan_sha256, case_index, case_id, group, parent_identity |
| `start.json` | Root pump | original controller process, runtime spec ref and case binding |
| `ready.json` | Root pump | schema `jev-v7-matched-case-ready-v1`, root/case/service-plan/case-nonce binding, endpoint, model, owner_output, service_ready_ref |
| `stop.json` | GPU callback | schema `jev-v7-matched-case-stop-v1`, root/case/service-plan/case-nonce binding, caller_record_ref or null |
| `closed.json` | Root pump | schema `jev-v7-matched-case-closed-v1`, binding and original terminal/receipt/closure refs |

The service uses distinct `service-ready.json`, `service-stop.json` and
`service-closed.json` files. The pump only publishes callback readiness after
checking the service's original admission/config/`verify_owned` process,
CPU/BF16/context16384/resource binding and fresh loopback endpoint. The service
ready `nonce` is the case nonce; `verify_owned.nonce` is the original CPU
config's `jev-d3-boundary-…` nonce. They are not substituted for one another.

`service-stop.json` contains exactly schema
`jev-v7-full-choice-service-stop-v1`, case_index, case_id, root_plan_sha256,
plan_sha256 and nonce. A caller record may be null when the original callback
failed; this still requests CPU drain and does not authorize a completed case.

`serialization.json` is created once by the pump before model use and contains
the absolute lock/journal paths, their device/inode identity and a fresh32hex
nonce. Its exact ref is selected in the CPU command sources. The pump does not
take or replace model forward leases; the selected serial-forward source wraps
the original full CausalLM and original D3 backend.

## Original closure remains mandatory

Local `service-closed.json` only proves local drain/artifact/source observations
and explicitly leaves external original owner closure pending. Its worker
receipt is `owner_output/v7-full-choice-service/worker-receipt.json`, whose exact
schema is `jev-v7-full-choice-service-receipt-v1` and success keys are
`service_success` and `final_native_artifact_authentication`. No old toy receipt
keys or fallback schema are accepted.

The pump waits its actual original controller `Popen` exit, retains a private
terminal, rereads original sealed config, receipt, admission and original
`external-boundary-closure.json`, and authenticates their refs. Acceptance joins
original selection/profile/unit/boot/nonce/invocation, exact source union,
command/worker stable process identity, original authenticated stop, unit/cohort
exit, post-stop verification, absent cgroup and two empty process scans. Dynamic
process state such as S/R is excluded through original `stable_process` and
`same_identity`; executable, argv, cwd and stable cohort identity remain exact.
Neither a claimed Boolean nor GPU cgroup absence substitutes for these refs.

Task success is independent of infrastructure: a false task result may still
finish an original owner with exit0 and complete closure. A failed or incomplete
service/controller cannot authorize the next case or a denominator of ten.

On abnormal outer cleanup, `close()` requests service stop and sends SIGINT to
the exact retained original CPU controller. A stop-file retention error does
not skip this cleanup. The pump waits up to45 seconds for that controller's
original cleanup; it never kills the controller or claims its sibling is gone
because the GPU unit stopped. Missing evidence, a live controller or failed
original closure raises and remains pending. If stop publication failed but the
original CPU closure is subsequently verified, the same original exception is
raised with the incomplete pump result attached as `jev_v7_pump_closure`.

`verify_children(result, arm, plan, output)` is the new outer-owner delegate. It
rereads all ten original proofs, rederives runtime plans/specs from the pinned
template, compares root/actor/case identities and checks the actual controller
is absent. It accepts only ten completed exit0 original scopes. It explicitly
does not claim universal process-tree confinement, GPU coverage of CPU sibling
memory, independent actual review, browser quality, model victory or an official
JevBench result.

## Remaining before a genuine campaign

Root must qualify/release new copies of the pump, service, full-choice bridge,
serialization wrapper and integrated owner/campaign sources; select the final
template and exact source/capability/numeric refs; validate all source counts
and command/binary identities; and connect the owner delegate/cleanup seams.
The static template, selected integrated plan and genuine case outputs do not
exist by assertion in this source candidate. Source fakes and then the genuine
matched ten-plus-ten run still require root execution and original positive
closures. This source author has not performed them.
