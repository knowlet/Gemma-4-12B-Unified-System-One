# MPS baseline memory failure

The original serial MPS campaign exited naturally with code 1 after exhausting
MPS memory. No supervisor signal was sent and no memory limit was disabled.

| Stage | Preserved outcome | Accepted model score |
| --- | --- | --- |
| Causal public231 | 231 valid answers, 197 correct | 197/231 |
| Causal coherence | 236 raw outcomes, 25 errors, zero scored checks | unavailable |
| Independent public231 | 231 attempts, zero valid answers | unavailable |
| Independent coherence | first request failed | unavailable |

The independent summary's `complete: true` means the attempts finished; its
zero accuracy cannot be interpreted as decision quality when every response is
invalid. Coherence stopped after 25 consecutive backend errors and has no valid
overall score. Neither result is filled from historical CUDA data.

The MPS error reports 23.91 GiB allocated to tensors and 64.24 GiB in other
allocations, against an 88.13 GiB limit. Whether the latter is reclaimable
allocator cache or graph/driver retention remains unresolved. The next measured
MPS epoch synchronizes, records current/driver/recommended memory, calls
`empty_cache`, synchronizes again and records the result after every dispatched
callback and at stage boundaries. Cleanup latency and errors stay in the trace.
Both question profiles use the same policy, original model, precision,
temperature and scoring rules.

PyTorch documents [`empty_cache`](https://docs.pytorch.org/docs/2.8/generated/torch.mps.empty_cache.html)
as releasing unoccupied cached allocator memory. This is a hypothesis under
test, not a proven OOM fix. The documented
[watermark controls](https://docs.pytorch.org/docs/2.8/mps_environment_variables.html)
remain unchanged.

The [failure archive](failure-evidence.zip) preserves the terminal traceback,
identity, termination receipt, partial coherence report and successful cache
entries, plus the independent public attempt records. [report.json](report.json)
lists every member digest and separates attempted from valid populations. The
original causal public evidence is preserved in the
[public231 archive](../mps-public231/public231-evidence.zip).

`scripts/resume_mps_benchmarks.py` freezes the original archived S1 implementation
and pinned benchmark sources before queueing. It waits for the captured external
supervisor and its actual recorded child to terminate, then reruns missing stages
into a fresh output directory. Existing successful coherence cache entries keep
their original execution epoch; newly dispatched calls have separate raw traces
and allocator telemetry. Full reports must pass coverage, identity and official
raw-answer replay audits before being accepted.
