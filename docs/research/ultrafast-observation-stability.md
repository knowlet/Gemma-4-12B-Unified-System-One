# Optional browser observation stability control (2026-10-28)

The consumed MPS hierarchy pilot preserves two strict timeout failures. Raw CDP
observations locate real Wikipedia hydration between observation and later
freshness checks. The causal destination is first observed at 139.957 s, but
its first DONE is discarded after text, input and control semantics change.
The later DONE uses a stable marker. The local synchronous CDP shim neither
rewrites those timestamps nor caches marker results. This is not a demonstrated
freshness-guard defect and does not justify accepting the earlier DONE.

## Relevant reference and decision

The existing local upstream checkout is
[`browser-use/jev-ultrafast@1231850a0bf1a0c0341fe408ef1668dbbfdfac46`](https://github.com/browser-use/jev-ultrafast/tree/1231850a0bf1a0c0341fe408ef1668dbbfdfac46),
MIT copyright Browser Use (license read from that pin). The investigation reads
its browser marker/snapshot and `Agent.command` observe/predict/act/DONE caller
chain together with local `scripts/ultrafast_browser.py::run_task`.
The existing startup waits for complete ready state; observation waits for two
animation frames and a short fixed delay. There is no semantic-stability option.
The original full semantic marker and freshness checks remain authoritative.
No upstream code needs to be copied or modified.

The original marker expression executes snapshot bookkeeping: it can initialize
`window.__jevFast`, allocate IDs for newly observed DOM nodes, and refresh the
page key/guard functions. Extra reads therefore can advance the original
observer state. The experiment records this explicitly; unchanged candidates
means no filtering, truncation, injection or description rewriting, not an
assertion that observation has no bookkeeping effects. A missing body returns
an unavailable marker and must never become a stable value.

Use the existing runner rather than changing the pinned agent: an opt-in hook
immediately before each outer `tick` may poll the original marker for a fixed
250 ms quiet period, up to a 2 s polling window or the remaining task budget.
It receives browser access and timing only, never expected task outcomes. Any
semantic marker change restarts the quiet period; invalid observations cannot
count as stable. If the quiet period is not achieved within the window, retain
that observation and return to the original policy. If the task deadline is
already reached, preserve task_timeout and dispatch no new tick. Never replace
or bypass the original predict/reobserve/act/DONE freshness checks.

All waiting is included in the existing policy clock and 180 s external budget.
The synchronous CDP transport has no strict per-call timeout: 2 s bounds the
polling schedule, not a promise that a blocked transport returns in 2 s. Record
actual elapsed time, soft-cap overrun, errors and stability status. Recheck the
task deadline after waiting. A page can change after quiet time or during a
long inference, so effectiveness remains unknown; the added wait can hurt time
budgets. This is a versioned, explicitly enabled scheduling experiment, with
baseline behavior and launcher argv unchanged by default.

## Verification

Use CPU fake clocks/CDP observations to cover stable, changing, unavailable and
error states; remaining-budget exhaustion; timing inclusion; unchanged candidate
content and original guard behavior. A quiet period cannot make a later stale
DONE pass. Preserve the consumed CDP marker witness separately. Actual MPS
browser replay, repetition, untouched-task generalization and both JevBench
regressions remain required; no historical score is recomputed as a success.


The [retained marker fixture](../../tests/fixtures/ultrafast_done_markers.json)
binds CDP observations 37/38/39/41 to the original consumed pilot trace and its
hash. CPU replay preserves the first stale rejection and later equality; it is
not an additional browser execution or evidence that the quiet window succeeds.
