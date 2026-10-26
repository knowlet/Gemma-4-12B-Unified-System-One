# GX10 reserved-window full-evaluation attempts

Two fresh attempts used the unchanged Edge BF16 four-core CPU profile, fixed
2GiB cap, original Comfy idle gate and original full host reserve. Both were
interrupted when the original observer reported one running Comfy job.

| Owner06 attempt | Root wall seconds | Callbacks completed/started | Public231 score | CoherenceMINI score |
| --- | ---: | ---: | --- | --- |
| actual03 | 23.5591 | 32/33 | incomplete | incomplete |
| actual04 | 299.0210 | 228/229 | incomplete | incomplete |

The callback counts are execution progress, not correct answers. Neither
attempt completed the first 231 callbacks or the full 1,479 callback campaign.
No partial score is promoted or combined across attempts.

Both reached the 2GiB memory boundary. The retained memory-event samples show
no OOM/kill events; `memory.events.max` counts (660 and 528 respectively) are
separate from OOM. Actual04 reported 291 runtime observations but retained 64;
its audit covers the saved snapshots, not a continuous absence of OOM.
Actual04's completed-call median was 0.332 seconds, p95 was 5.159 seconds and
maximum 8.355 seconds. These interrupted CPU callback times do not qualify the
browser's complete twenty-second request path.

The original authenticated stop receipts remain intact. Their first process
scans found a zombie and therefore kept whole-unit/cohort exit flags false.
Subsequent original-method scans independently found no processes and absent
cgroups. This later absence does not rewrite the original closure flags.

The independent reviews and root projections are under
`artifacts/jev-runtime/decision3-publisher-batch4-cpu-both-20261023-04/`;
original owner receipts are under `llama-cpp-owner-20261023-06/actual-03/` and
`actual-04/`. Raw tasks, answers and per-response evidence remain local.
