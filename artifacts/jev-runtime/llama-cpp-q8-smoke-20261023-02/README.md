# Q8 CPU smoke sources

Two actors run sequentially under the existing generic owner:

- `prepare_smoke.py` imports the actual CPU processor and original S1 preparation code, then saves short/long requests, exact native token payloads, slot geometry, and a preparation receipt. It does not load a language model or read model weights.
- `run_smoke.py` imports stdlib plus `smoke_common.py` only. It starts fresh selected native processes, records request/response raw bytes and hashes, validates identities/slots/counts/finite logits and actual context/mode, and compares full/compact cache arithmetic.

Both default to dry output. Explicit execution requires `--run`, `--expected-entry-sha256`, `--selection`, `--selection-sha256`, and a fresh `--output`. Scoring also requires `--payload-root` pointing to the completed preparation from that same selected source set. Root owns the concrete selection and outer unit invocation; none is fabricated during preparation.

The selection schema is exactly:

```text
profile: llama-cpp-q8-smoke-selection-v1
research_scoring_deadline_seconds: 600
sources: distinct [{path: repository-relative, sha256, size_bytes}], at most40
processor: {path: exact local a66 snapshot, artifact_manifest: selected small path}
native: {path: ordinary absolute, sha256, size_bytes, stat, libraries: [{path,stat}]}
package: {receipt: selected small ref, model: {path,sha256,stat}, mmproj: {path,sha256,stat}}
stat: {st_dev,st_ino,st_size,st_mtime_ns}
```

The required source pool includes both actors, the shared module, the original adapter, and original S1 `__init__`, `unified`, `contracts`, `schema`, `checkpoint`, `runtime`, and `errors` files. The artifact manifest must have the known public SHA1646a81f… . The package receipt must be a completed full public download, and both files must be beside that receipt with the exact published sizes/hashes and selected current stats. Weight stats and the prior stream receipt do not prove subsequent arbitrary storage integrity; no second13GB read is performed.

Each arm uses native v3 `--cpu-mmap-only1`, `--gpu-layers0`, `--threads1`, batch512, and the requested full/compact SWA flag. The first short response has60 seconds measured from immediately before Popen, including launch/initial identity observation; long scoring has600 research seconds. Each valid call separately records whether it completed within20 seconds. `quality_N3_Q95_or_20s_qualified` remains false. The8192 arm must report an actual8192 context while using the same at-most2048-token input.

Native stdout is a bounded single JSON line per request, with a1MiB maximum buffer. Stderr is drained/hashed while retaining only its first4MiB. On failure, diagnostic draining is bounded and does not wait for the child; unread later bytes are not included in the observed hash/count. Partial response files and prior completed exchanges remain. A failed arm stops the experiment so another native process is not overlapped with a still-running child. Only the outer authenticated unit closes a failed child's process cohort. Successful arms exit normally on stdinEOF before the next process starts.

The18 pure stdlib tests cover fragmented writes/reads, exact raw-wire hashes, response shape/identity/context refusal, fast invalid replies, malformed JSON, timeout partials, primary errors during receipt failure, failed-start resource ownership, stderr retention, and the preselected numerical predicate. They invoke no processor, Torch, native binary, unit, network, or model payload. Actual source/runtime qualification is separate.


This successor preserves frozen smoke01. Only native launch timestamp capture and warm timing/evidence change: warm deadlines and20-second observations use launch-to-response, while long scoring keeps submission-to-response. Fake-clock regressions cover25-second startup/proc delay, a warm budget already expired before submission, and a long call after startup. Fast invalid responses still cannot count as20-second success. No actual processor or model execution occurred during preparation.
