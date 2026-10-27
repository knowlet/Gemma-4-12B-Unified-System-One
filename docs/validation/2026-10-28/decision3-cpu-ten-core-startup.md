# Decision3 ten-core startup diagnostic

The guarded ten-core synthetic experiment on GX10 exited before the worker
published its identity or loaded the model. It produced no timing or model
quality result. The watcher recorded `cleanup_unconfirmed` because it could
not bind the worker identity. Subsequent independent checks found that the
unit had `MainPID=0`, its recorded process and the corresponding process group
were absent, and its dedicated cgroup no longer existed. These later checks
do not turn the original watcher closure into a verified closure.

A separate short-lived diagnostic unit used the same CPU and memory settings
without loading a model. Its main-thread affinity was CPUs 5–9 and 15–19, and
its leaf `cpu.max` was `1000000 100000`. The leaf had no
`cpuset.cpus.effective` file: its delegated controllers were `cpu`, `memory`,
and `pids`; the higher `user.slice` exposed a cpuset of 0–19. This makes the
missing file a plausible cause of the startup failure, but the original unit
did not retain a traceback, so the cause is not proven. Main-thread affinity
alone does not establish cgroup cpuset isolation or all-thread affinity.

The v3 worker records a bounded, fixed-category pre-handshake failure after
validating its private output directory. The watcher labels that record as an
unbound worker report, not as proof of successful startup or cleanup. Failures
before the output directory is validated cannot produce this record. The
existing CPU, memory, swap, identity and cpuset checks remain mandatory; the
v3 release names are separate from the retained v1 and v2 releases.

The proposed change to continue when the cpuset file is absent was rejected
by automatic review as weakening resource verification. It was not applied.
No ten-core model run should be claimed until the existing checks can pass and
the independently audited worker completes.
