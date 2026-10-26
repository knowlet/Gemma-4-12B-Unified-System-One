# Bounded natural-exit resolution validation

2026-10-22. The previous `5220799a...` observer and tests remain byte for byte unchanged. This successor changes only read-only observation: generic/provisional uncertainty may be cleared by exact offender disappearance, never by another classification while alive. No process is signaled, stopped, restarted, unloaded or modified.

Known Comfy signatures and argv-intrinsic uncertainty remain immediate blockers. A later intrinsic/Comfy observation dominates any earlier generic uncertainty, including a deleted Comfy cwd. Diagnostics retain cwd failure/status type, proc-owner UID and hashes without raw cwd or argv. The one resolution window shares a maximum 0.5-second budget across all offenders. Read-only pidfd exit readiness and a fresh missing `/proc/PID/stat` are both required; kernel ESRCH from pidfd-open also needs fresh proc absence. Reused/live/unknown identities, unsupported proof and all unresolved uncertainty refuse. Any Comfy observed during that window immediately refuses. All pidfds close in a finally block.

After all eligible offenders disappear, discard the earlier observation and restart **both** full listener/process scans once. No second resolution window is allowed. The whole observation shares a two-second admission deadline across HTTP, scans and the resolution window. Existing byte/process bounds and all external memory caps/floors remain unchanged. Source admission still refuses every unresolved unknown.

**106 pure tests passed in 0.37 seconds**, with Ruff lint/format passing. Tests cover the existing cases plus both-scan restart, one-restart limit, shared offender budget, live external reclassification refusal, identity reuse, proc absence without pidfd readiness, live Comfy observed during the wait, immediate intrinsic blockers, typed/redacted diagnostics, stronger later latches and the whole-observation deadline. Test bounds were address space 256 MiB, CPU 30 seconds and wall 45 seconds. Exact JUnit output is preserved as `pure-tests.xml`.

Actual controlled CPU fixtures used three explicitly owned harmless Python children setting `PR_SET_DUMPABLE=0`:

- The original observer rejected its denied cwd. The successor observed the exact same offender's natural exit in **88.48 ms**, proved pidfd exit and fresh proc absence, then accepted only after **two newly restarted complete clean scans**. The child starts its own 80 ms exit timer on a fixture-owned stdin byte when the resolution window begins.
- A separate undumpable child remained alive throughout the shared budget and was refused in **0.5975 seconds**. An alive external classification did not authorize admission.
- A separate actual child with fake argv0 `comfyui` remained alive and was refused in **0.0960 seconds**, without waiting through its Comfy signature.

All three children exited naturally and were reaped with exit code zero. No signals were sent. An actual read-only 150-call normal exec-churn soak then returned **150/150 confirmed absences, zero refusals in 28.893 seconds**, with **653 owned harmless shell-to-Python children** all reaped/exit zero, no timeouts, and one bounded identity retry. The maximum successful observation lasted **0.199812 seconds**. Natural-exit resolution was exercised by the controlled fixture, not needed in the normal churn soak. These bounded checks do not establish a statistical availability guarantee or a browser score.

No GPU, model, browser, service, systemd or actual Comfy operation was invoked. The preceding raw browser run remains a partial known4 result, not a complete ten-scene evaluation.

Two independent read-only reviewers returned CLEAR on the exact frozen source hash below. They checked the full diff and context/latch/exit-proof/restart/deadline paths and found no remaining must-fix issue at this observation scope. Neither reran tests nor invoked actors.

| Artifact | SHA-256 | Bytes |
| --- | --- | ---: |
| `comfy_observer.py` | `103804b05e3047011aa5daf099c04ee71a60b3e1435c0690dc7b0377bd486df7` | 26395 |
| `test_comfy_observer.py` | `d47b145aa3542f0850b6fd57bd324a0e5b91b90f8640080edde81f3506ffb181` | 33749 |
| `pure-tests.xml` | `c56ce10be3b2be7770526acacc9960aa1f1c9c73e312ae58ab57a8794631460d` | 21839 |
| `controlled-resolution-01.json` | `4ff4cdc91bc4b1cae04d949ca34a496fb698b80f1caebf51a8f9cdbc0e19c6f6` | 10069 |
| `new-observer-owned-exec-soak-01.json` | `c4114f5c97507332820aa6f0827b5234123b154a634336701530015e9aa6c862` | 493550 |

The prior observer remains `5220799afc459a3fb702aff5e0d30a7362d14030293b79872a83def6dd1efae2` (17835 bytes); its tests remain `7f7e391f6cf70780cda508c1e3522750f9a5d4da5b482a1b18ecca1f942f4bbc` (26718 bytes).
