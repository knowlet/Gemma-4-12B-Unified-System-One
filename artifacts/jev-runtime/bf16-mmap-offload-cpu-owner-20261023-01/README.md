# Tiny CPU storage-hook owner

Source-only preparation. No actual unit, observer HTTP, framework import, mmap, model or Comfy operation has been executed by this preparation agent. Independent source review and root execution remain separate evidence.

`controller.py` defaults to dry output. Explicit execution requires the independently reviewed controller SHA, actor SHA, selection SHA and a fresh `actual-NN` directory directly under this owner directory. Use the final `source-freeze.json` refs; actor source revisions require a new selection SHA. The fixed unit must be absent; an existing unit is refused without reuse or stopping it.

The unchanged `7a50c47d` owner and `fd977040` observer/`a2446af3` authority enforce the selected unit and read-only idle gate. The unit has 30 seconds total, 2 GiB memory, no swap, CPU quota 100%, CPU 0, core 0, FD 512, file size 8 MiB, stop timeout 3 seconds and RemainAfterExit=yes. Host admission needs `R+2 GiB`; running/final checks need `R=max(24 GiB,ceil(MemTotal/5))`.

A stdlib bootstrap records PID/start/session/cgroup and waits. Parent verifies the original boundary, kernel and extra resource properties before permitting exec of the fixed ComfyUI Python with `-B -s` and the controlled runtime overlay. A second pre-Torch admission authenticates the same PID/start with its actual interpreter, command line and cwd. The actor then runs in that interpreter; a bounded five-second hold lets the parent close the live exact nonce/Invocation through the unchanged owner. Failure to close during that hold becomes exit 125, never a successful active/exited result.

Receipts keep actor exit, stdout/stderr, selected sources, actual admissions, raw exit/MemoryPeak observations, observed cohort closure and final source/host checks. A later failure cannot replace an earlier primary failure. Runtime host checks all gate the actor; at most the first 63 and latest observations are retained when there are more than 64. Closure concerns the authenticated unit and observed cohort; it does not prove a universal process tree or prevent deliberate cgroup migration.

The actor selection covers critical small source hashes and selected binary stat identities, with no binary payload scan or complete installed-runtime claim. Any generated `tiny-weights.safetensors` is **local-only and excluded from Git**. Stage only an explicit source/JSON evidence allowlist.

`test_controller.py` contains eleven pure stdlib checks, including R/S state tolerance with changed start rejection, wrong source SHA before authority load, existing unit refusal and receipt persistence with primary-error preservation. `static-validation.json` records final results. `static-tool-attempt-01.json` preserves a Ruff address-space-limit failure; the 1 GiB address-space limit applies only to the stdlib tests, and Ruff used inherited limits for final validation.

This tiny true CPU mechanism cannot qualify 12B weights, GPU transfers, UMA behavior, 8192 context, 20-second latency, N3/Q95 or model quality. It does not modify a Comfy service.
