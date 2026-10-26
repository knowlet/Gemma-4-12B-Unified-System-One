# CPU5/4GiB owned browser worker — source only

Distinct **unexecuted, unfrozen, unselected** port of the CPU4 worker/successor in
`decision3-cpu-browser-worker-candidate-20261024-01`. The destination was checked
absent and created exclusively. No old source was changed. No import, AST, test,
formatter, builder, dry run, host/actual/model read, hash, model, service, Chromium
or Git operation was performed. New owner/native/plan/source pins remain `None`.
The worker still has `SOURCE_ONLY=True`, refusing before args/plan/dependency work.
The pending plan/spec are deliberately invalid and grant no launch authority.

Exact intended port delta:

| Binding | CPU4 parent | CPU5 candidate |
| --- | --- | --- |
| Owner | owner07, CPU16–19/400%/2GiB | owner09, CPU15–19/500%/4GiB |
| Native | native03 | native05 |
| Worker profile | cpu4-2g | cpu5-fast-same-cluster-4g |
| Process-local lease ABI | owned-decision3-cpu-browser-binding-v1 | owned-decision3-cpu5-browser-binding-v1 |
| ASGI actual affinity/threads | 16–19, intraop4/interop1 | 15–19, intraop5/interop1 |
| Lease identity | owner07/native03 and2GiB | owner09/native05 and4GiB, swap0, exact15–19, quota500%, expected5/1 threads |
| New owner/native/plan pins | parent recorded pins | pending `None`; parent pins never reused |

Proof thread fields describe the selected expected policy. Genuine thread
readiness still separately checks actual `sched_getaffinity`, Torch thread counts,
CPU runtime identity and uninitialized CUDA inside the serving thread. The
existing phase-aware validation is retained: model_loads0→START_ENV,
model_loads1→START_ENV+CPU_ENV; other counts refuse. CPU_ENV now comes from the
selected native05 helper (five threads). No original gate is monkeypatched.

All other worker behavior is copied: original external owner controller/boundary
is the only admission/resource/cleanup authority; one selected command worker
joins its PID/start/argv/exe to command-started and its cohort to the admitted
wrapper. One original native `load_runtime` runs before s1/browser/pytest imports.
The same pre-bound127.0.0.1:0 listener is handed to one ASGI thread with workers1
and lifespan off; synchronous Playwright/caller remains on the main thread.

The adjacent bridge successor still inherits the original native `predict`
verbatim, keeps all non-device blockers and honest CPU/BF16/context16384 checks,
all original question/candidate order and native raw probabilities, single
normalization and original named-wire conversion. Native batch remains4,
including actual1–4 row tails; five CPU threads do not become five NN rows.
The pinned Edge model/revision/head bytes and numerical policy are unchanged.

The original `tests/test_jev_native_http_browser.py` function is called directly
after model load, without pytest CLI/plugins or an external service. Its invented
two-button fixture, max16 ticks, strict fresh visible/trusted one-copper-click
judge, helper0, standard offered operations, HTTP25 and original retries remain.
No choices, DONE/BLOCKED, failures or wrong targets are corrected. The unchanged
bridge20s gate includes parse/forward/observers/serialization/fsync and last200
authorization at ASGI response.start. No network-delivery or hard-cancel claim.

Fatal BaseException objects still cross the backend/ASGI latch without losing
their original identity, blocking a subsequent native call. Browser finally
finishes before server shutdown, followed by10s drain. Undrained service leaves
pending failure and no model/hook release; original owner closure remains
mandatory. Final native/source authentication and actual client/server generated
IDs/body/response hashes are joined only after drain, outside model input.
All artifacts are inside a fresh ignored owner-output subdirectory. No new
guardian, general API framework, full benchmark campaign or text helper is added.

`test_worker_source.py` is unexecuted stdlib-fake source retaining source-only
refusal, phase0/1, fatal identity/no-second-NN, ASGI propagation and verbatim
predict/non-device-blocker checks. New CPU5 proof negatives reject CPU16/CPU4
owner identities,2GiB/adjacent caps, swap, another five-core cluster, wide affinity,
old quotas and wrong intra/inter-op threads. None of these tests has run.

Root must first genuinely qualify owner09/native05, then review/check/fill pins,
select actual tools/import/upstream closure, release/freeze exact worker source
and generate a final plan followed by a new original command spec. Keep
R26119172916+the entire4GiB startup floor30414140212, swap0 and original authority
limits. Model/Chromium/driver share that cap; browser fit is unknown. “Fast” is
root's topology-based selection name, not a CPU5 speed/result claim. No prior
CPU4/CPU16 timing, memory, browser, quality or competitor qualification transfers.
