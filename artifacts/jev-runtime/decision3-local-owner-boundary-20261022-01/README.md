# Decision3 external unit boundary

This stdlib component verifies and closes one explicitly selected `jev-d3-*.service`
user unit. `Selection` binds its name, nonce, boot, memory cap, independent runtime
and stop timeout. `UnitBoundary.verify()` checks actual systemd properties, kernel
memory/swap limits, main PID/start/PGID/SID and recursive kernel cgroup membership.
`UnitBoundary.close()` authenticates nonce and InvocationID again, stops only that
unit, and requires cgroup absence, two independent process scans and unchanged
source pins before reporting unit and observed-cohort closure. Errors retain
unknown closure and the caller's primary error. Evidence files are exclusive.

The callable owner does not import or modify the separate local transport. It
does not accept a caller-supplied closure boolean. The transport's existing
`whole_tree_exit_confirmed=false` and `continuation_authorized=false` remain
unchanged. A future controller must independently compose actual unit evidence,
the reviewed worker's behavior and its own success requirements; this fixture
does not qualify a Decision3 model or baseline.

The current same-UID delegated cgroup hierarchy permits writes to ancestor
`cgroup.procs` and access to the user manager. Explicit migration or requests to
that manager are not prevented. Accordingly `no_escape_enforced`,
`whole_tree_exit_confirmed`, `continuation_authorized` and `model_qualified` are
always false. Closure facts cover the authenticated unit and the observed
cgroup cohort. They do not establish adversarial containment of unknown
descendants that deliberately migrate before observation.

`run_probe.py` defaults to dry mode. Its only actual execution profile is the
authorized fresh `jev-d3-boundary-probe-20261022-01.service`, 64 MiB memory,
zero swap, 30 seconds maximum runtime, and a 2 second stop timeout with final
SIGKILL. The fixture also sets its own 25 second alarm. It double-forks, starts
a separate session and execs a child with an empty initial environment. The
controller checks that this detached child remains in the selected kernel
cgroup, then verifies authenticated external closure. It does not attempt
cgroup migration or use a GPU, model, browser, Comfy endpoint, or another unit.

`research-decision.json` records the primary sources, reuse and limitations.
Pure tests run with AS 256 MiB, CPU 30 seconds, wall 45 seconds and one BLAS
thread. Actual fixture evidence, when generated, is stored in a fresh `probe-NN`
directory; a sealed plan must be regenerated after any execution source changes.
