# Original V7 matched comparison: preparation only

The operation-only candidate passed the unchanged two-control browser diagnostic
in actual30, published in `dd0cb6e`. This preparation adds no model inference,
browser attempt, JevBench score, complete ten-case result, or competitor win.

The primary comparison is the original V7 **known4 + public6**, the same ten
public cases used for the historical NF4 Gemma 4/10. The nine seeded templates
plus travel are a different suite. Both fresh arms must use the original V7
bundle builder/runner, seed 20261014, one repeat, sixteen policy steps, original
judges/options/retries, 20-second policy/helper calls and 25-second HTTP timeout.
Each case will use a fresh browser in both arms; the historical grouped-browser
4/10 is background evidence, not a fresh matched control.

The text helper is the same resident Gemma revision
`a66f836b56605039fe040f330180e336d19b3362`, NF4, with its verified action adapter
disabled during text generation. Keep its original grammar, EOS, 128-token
generation limit, 8192-token context and reasoning disabled. Merely naming that
helper in a request does not establish its runtime identity or execution.

The held thin caller passes through the original HTTP client once per actual
attempt, retaining private serialized request/response bytes, headers, request
IDs and latency. It preserves original transport exceptions and retries. Task
failure remains a completed result in the denominator of ten. Missing cases,
changed order/profile/helper or incomplete infrastructure cannot produce a
complete score. All HTTP bytes and real page evidence remain private.

The held serialization component uses one shared lock and journal, both bound
to their original inode. Decision3 holds the lease across actual CPU prediction.
NF4 hooks cover the actual complete CausalLM returned by PEFT's
`get_base_model()`, including the output head and softcap. A charged CUDA
completion barrier precedes unlock; the original independent backbone resource
gate remains. Failed synchronization or required evidence retains the lease
until process exit. This is cooperative serialization, not a process-escape
prevention mechanism or a GPU execution qualification.

The held outer owner retains the original NF4 40 GiB/swap-zero boundary and full
host-memory admission formula, with a separately named 10800-second integrated
campaign profile. That cap covers its own unit. The existing CPU owner09 keeps
its own 4 GiB/CPU5 limits and original scoped closure; a 1200-second per-case
cooperative budget is not a new owner09 hard limit. The external coordinator
must keep the CPU controller outside the GPU unit so GPU cancellation cannot
leave an unmanaged CPU sibling running. Both owners must close independently.

All four entry components remain held. Campaign/pump/capability source pins and
the child-closure contract remain pending. The campaign coordinator, per-case
HTTP-only Decision3 service, handoff pump and complete closed-result auditor are
not implemented or selected in this milestone. No new NN job may be launched
from this preparation. The three stdlib test groups pass 31 tests: caller 11,
serialization 9 and outer ownership 11; Ruff passes. These use invented fixtures,
including a real two-process file-lock exclusion check, with no framework,
weights, HTTP service, browser, actual traces or official gold loaded.

Both JevBench suites remain separately reported. Historical unadapted Edge public
accuracy is 158/231 (68.3983%); its coherence run is incomplete at 1234/1248 and
has no complete score. The operation-only candidate has no new official score.
See `decision3-edge-cpu4-jevbench-results.json` and
`decision3-cpu5-operation-browser-results.json` for their distinct evidence.
