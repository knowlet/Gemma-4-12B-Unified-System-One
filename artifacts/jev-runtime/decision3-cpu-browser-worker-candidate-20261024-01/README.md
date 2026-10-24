# Owned CPU/browser worker — held source candidate

All files here are **source only, unexecuted, unfrozen and unselected**. No import,
AST, test, formatting, builder, dry run, host observation, HTTP, Chromium, model,
service or Git operation was executed while writing this candidate. There is no
claim of syntax correctness, passing tests, resource fit, readiness or performance.
`worker.py` has `SOURCE_ONLY=True`, and rejects before argument parsing, plan reads,
dependency construction or output/model work. The pending plan/spec have null pins
and deliberately cannot satisfy the original owner selection or worker contract.

The concrete proposed route reuses the **original** owner07 controller, boundary,
observer, admission, nonce, source checks, command intent/started records, resource
properties, bounded logs and external unit/cohort close. No guardian is copied or
reimplemented. The new worker is one selected ComfyPython command inside that
unit. It joins its own PID/start/executable/argv to command-01-started, and its
PGID/SID/UID/cgroup to the admitted wrapper. It does not mistake the wrapper PID
for the NN worker. The started-record publication race has a bounded wait.

After binding the actual admission, it calls original native03 `load_runtime`
**once**, preserving native03/helper03/model pins, CPU4/BF16 backbone, original
FP32 head, temperature1, batch4, context16384 rejection, no permutation and no
GPU optional path. Torch/s1/browser/pytest imports occur after the loader's fresh
process check. The worker never loads a Gemma/helper. The native constructor,
resource floors and checkpoint/scorer math are unchanged.

The owned-release callback uses an explicit environment phase: native load count
must be the integer0 or integer1. Before load it checks original `START_ENV`;
after the one original load it checks `START_ENV` plus `CPU_ENV`, matching the
loader's CPU-thread/environment transition. Other counts, including booleans,
refuse. The maps are copied, never mutated. This source-only review correction
also has unexecuted stdlib-fake tests for the transition and invalid phases.

`bridge_successor.py` is a direct source copy of the b4cf00a bridge with this
explicit small semantic delta (no diff/check was executed here):

- Replace its always-refusing global release placeholders with a private
  `_OwnedRelease` holding an invariant proof and callback. The callback reuses
  the original controller/source/admission/current process/native owner checks.
  No request, environment flag, monkeypatch or fake CUDA spec supplies release.
- Implement `bind_owned_selection` and `create_application`: authenticate that
  release, construct an honest `decision3_spec(device="cpu")`, wrap the original
  native shim in the worker's fatal guard, and construct the existing gateway.
- Require the same explicit lease in CPU adapter/shim/gateway/dependency helper
  constructors. The CPU adapter still inherits original `predict` verbatim and
  retains all non-device blockers plus explicit BF16/CPU/context16384 checks.
- Preserve original copied raw recording, shape witnesses, ordered DTO/native
  payload, single normalization, named-wire conversion, complete candidates,
  original errors, 20s publication gate and HTTP-only terminal replacement.
- Recheck the outer deadline after CPU adapter construction, so owned-selection
  revalidation cannot exhaust the HTTP budget and then start another forward.

The lease trusts the exact selected worker callback. It is a process-local
capability, not proof that an arbitrary Python callback read the original owner
records. Root must pin and review that worker; same-process Python callers are
not an isolation boundary.

The old b4cf00a file and its always-refusing factory/constructors remain unchanged.
This successor does not import the old module and does not alter its flags. Its
only launch path is the held worker and a future exact original owner selection.

The worker passes its own pre-bound `127.0.0.1:0` socket to one Uvicorn ASGI
thread (`workers=1`, `lifespan="off"`). That thread checks actual CPU affinity,
Torch thread counts and no CUDA before accepting work. The main thread calls
the unchanged function in `tests/test_jev_native_http_browser.py` directly, after
the model load; it does not invoke pytest CLI/plugins or start a second worker.
That exact caller runs real HTTP, pinned choose/Agent.tick, PlaywrightCDP and the
guarded executor on the existing invented two-button page. It keeps native
HTTP25/retries, max16 ticks, all original options and wrong choices/errors/DONE.
No oracle, prepared actions, helper, fallback or result repair is added.

A fatal backend/ASGI latch retains the original BaseException object across
threads, rejects subsequent NN calls and lets the main worker rethrow the real
failure even if Uvicorn reports HTTP500. The gateway keeps raw attempts before
that NN guard. This does not claim hard interruption of a blocking forward.
The ASGI adapter's original cooperative20 and outer final-publication20 gate
remain; the old `helper.Deadline` (`os._exit`) is not applied to HTTP requests.
The original owner remains the independent hard runtime/cleanup authority.

The unchanged caller finishes its Agent/Chromium finally before service shutdown.
The worker then requests ASGI exit, joins for10 seconds, and only after confirmed
drain performs native/source authentication and joins actual client/server IDs,
body/response hashes and terminal status. No hooks or model state are released
while the NN thread is live. A non-daemon stuck service thread leaves pending
failure evidence and prevents a normal successful exit; the original external
owner must close that unit. There is no direct external-PID kill or unit escape.
The service binding never attaches to an arbitrary pre-existing endpoint/CDP
browser. Browser/driver children inherit the unit; the original boundary tracks
cgroup membership and PID-start, including children with different sessions.

Worker/server/browser output is confined to a fresh ignored subdirectory beneath
the original owner output. The original caller writes raw HTTP/DOM evidence;
the bridge writes generated request IDs/raw native/actual shapes/terminal files.
The worker joins hashes only after drain, outside model input. Original owner
current closure is still mandatory. A local receipt never claims no-escape,
whole-tree proof or general model qualification.

## Necessary work before any run

1. Root reviews all source, then releases permitted CPU checks; none ran here.
2. Finalize source-only fake tests and genuine CPU tail-row/native contract tests.
3. Pin every new/imported source, original harness/test/upstream static inputs,
   bridge successor, owner/native dependencies and actual binary/tool identities.
   The original owner permits at most58 spec sources plus its base union and
   at most64 total source records; no larger pool is silently accepted.
4. Independently select a released worker source (SOURCE_ONLY changed only in a
   newly reviewed/pinned source), create a final plan, then its SHA, then a command
   spec. The plan does **not** embed the command-spec SHA, avoiding a hash cycle.
   Original sealed config/admission/command argv bind those hashes at runtime.
5. Prove browser/Playwright packages and tools exist within the selected runtime,
   authenticate Chromium/driver identities and complete import origins. No such
   readiness or peak-capacity fact was observed here.
6. Retain R plus the entire original2GiB admission floor, swap0, CPU4 and all
   original owner limits. Model+Chrome+driver share that cap; browser fit is unknown.
   A future CPU16/4GiB campaign requires qualified native04/helper08/owner08 in a
   separate profile; this worker deliberately refuses that relabeling.
7. Only after the active NN campaign is terminal may root select and launch a
   fresh original owner invocation. `command-spec.pending.json` is not that spec.

`test_worker_source.py` is unexecuted stdlib-fake test source for the hold, fatal
object preservation, no second native call, explicit release invariants and
verbatim inherited prediction. It supplies no actual browser/model qualification.
