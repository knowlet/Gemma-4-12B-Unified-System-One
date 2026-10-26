# Native batch4 CPU, both original JevBench suites

This is one source-only campaign entry with separate Edge and Lite candidate
plans. It reuses the frozen batch4 actor's actual `load_runtime` constructor
and `NativeBatch4Backend`. It imports no framework, dataset or model in dry
mode or tests. A candidate plan is not a full-suite speed or memory admission.
In particular, a short Lite probe's headroom cannot establish long-input fit.

The exact official implementation is already local and pinned. This is a
small harness bridge, not a replacement scorer, model implementation or new
dependency. The publisher's Apache-2.0 source was researched in the frozen
batch4 source audit: both model snapshots have the same `d3_runtime.py` SHA
`adf7f7aa085674b5682e492d8b2e3cc4f11c67b0d153515c742e907e9b550917`.
The source supports `D3.from_pretrained(device="cpu", batch_size=4,
verify="full", local_files_only=True, permutation_average=False)` directly.
Its default batch size remains unchanged. It groups the original question
order into chunks of at most four, left-pads without dropping tokens, and uses
the publisher's original prompts, BF16 backbone, FP32 readout, temperature1
and Torch reference numerical kernels. No quantization, custom kernel,
answer repair, truncation, cross-request merging or score fitting is added.
Batch/padding can change BF16 numerics; batch1 quality and latency are not
inherited. AST fingerprints across Python interpreter versions are not
identity gates: exact source SHA and verified AST invariants are the gates.

The existing source population is referenced, not copied: the frozen
`decision3-publisher-lite-cpu-20261023-01/benchmark-source-plan.json` pins
78 original source files and two existing dataset metadata manifests. Public
uses `scripts/run_jevbench_public.py:run_public` against Heaven revision
`bb05a335bc809e61b20c0f745d25499a82b326fc`. Coherence uses official
`jb.evaluate(jb.from_callable(predictor.answers, name=selected_profile),
suite="jevbench-mini", concurrency=1, cache=fresh_path, progress=True)`
against revision `e18733694623aa93058e279c3534f8b8e2edefa3`. The one necessary
orchestration change from the older batch1 bridge is passing the selected
Edge/Lite batch4 profile to the original callable API. No benchmark task,
input, gold, output, cache or scorer math is embedded in this folder.

One constructor serves public231 first and then coherence MINI in original
input order. Completion requires 231 public attempts, 1248 coherence
requests, 240 cases, 50 relations, 1200 planned relation tests and exactly
1479 fresh valid callback returns with 1479 raw native envelopes. A callback
may contain several independent questions and consequently several native
chunks; 1479 is the callback count, not the first-layer forward count. The
original scorer remains responsible for answer wire formats, scoring and
confidence intervals. All input heads and up to255 original options pass
unchanged. The separately declared16384 context ceiling rejects a complete
request that exceeds it; it never truncates, prunes or selects fewer cases.

Before any framework/model import or payload authentication, the future
exact root command must provide all of these:

- the selected campaign plan path and SHA;
- the successful same-model batch4 smoke state path and SHA;
- its independently reviewed closed-smoke audit path and SHA.

The independent audit schema is
`independent-native-batch4-synthetic-result-audit-v1`, kind
`closed_real_smoke_result`, PASS or PASS_WITH_LIMITS. All seven qualification
flags must be true: successful smoke, authenticated external closure, one
actual constructor, two actual four-row events, both durable calls<=20s,
whole root command<=120s, and final full model/source authentication. The
actual model revision/identity/manifest, batch4, owner05 and3GiB cap must
match. Exact audit references to state, native plan, owner receipt, root
execution, selected spec and original closure observation (plus any other
named small evidence) are authenticated again. Relative audit paths resolve
against the repository. No success is fabricated in either candidate plan;
the root chooses exact evidence after independent review. This audit is a
synthetic probe prerequisite, not a benchmark score or browser qualification.

The sole owner is the frozen distinct owner05:
`jev-d3-llama-cpp-cpu4-3g-long-20261023-05.service`, profile
`llama-cpp-cpu4-3g-long-command-owned-v1`, CPU16-19/quota400%,
memory.max3221225472/swap0, startup MemAvailable floor29340398388 and runtime
reserve26119172916. The unchanged observer remains read-only. Owner runtime
is10800s; this campaign is10000s including preflight, load, both suites,
final authentication and durable result/state writes. An owner maximum is
not evidence that the model fits this deadline. Whole and per-call timers
retain a hard watchdog and explicit elapsed checks after durable writes.
Neither owner budgets nor any old frozen/active profile is changed here.

Each callback has the official offline180s default, including preparation,
the original forward, raw retention, backend validation, own-unit counters
and durable progress. Raw native output is saved before backend validation;
nonfinite/unsupported JSON is preserved as a bounded exact-repr diagnostic
without changing the object delivered to validation. Primary errors survive
secondary evidence failures. Late durable completion escapes scorer
`Exception` handlers and cannot be counted as a valid callback. Final
aggregate latencies measure callback work through the durable progress
write, exclude model startup, and provide median, nearest-rank p95 and
maximum separately for each original suite. Failed callbacks retain their
own elapsed/error evidence. These180s timers never establish browser20s
compliance. Fresh output/cache directories are mandatory and symlinks/`..`
are resolved before containment is checked.

Initial and final authentication happens only inside the future owned job:
all20 model files (including tensors), manifest, selected runtime and owner
sources, all78 original scorer sources and both existing dataset metadata
manifests. Final authentication also rechecks this campaign/plan, the frozen
native actor/plan/freeze/helper/source audit and all selected smoke audit
references. Source preparation only reads small local source/metadata and
stat records. Tensor/readout/tokenizer payloads are not read or rehashed
here. Root must authenticate the current campaign's terminal/empty cgroup
closure and peak separately before making a final qualification claim.
Results are profile-specific offline scores with no official rank or
browser score. A saved result file remains pending final durable campaign
validation; only a complete state plus separately reviewed closure permits
a final claim. Partial/failing campaigns retain evidence with no complete
score claim.

Future invocation is the Comfy Python interpreter with `-B -s`, this
`campaign.py --execute --plan ... --plan-sha256 ... --smoke-receipt ...
--smoke-receipt-sha256 ... --smoke-audit ... --smoke-audit-sha256 ...
--output <fresh-contained-path>`, wrapped in a separately frozen owner05
command spec and fresh admission. No actual invocation is performed during
preparation. `local_head_at_creation=1a6a2c6...` is provenance only.
