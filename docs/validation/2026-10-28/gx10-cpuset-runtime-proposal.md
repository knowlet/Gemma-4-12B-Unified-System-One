# Proposed temporary cpuset delegation for the GX10 user service

The reviewed CUDA smoke plan requires the service's actual
`cpuset.cpus.effective` to match CPUs 15–19. At the time of inspection,
`user@1001.service` delegated `cpu`, `memory` and `pids`, but not `cpuset`;
the proposed model run would therefore stop before loading the model.

Subject to explicit approval, the proposed experiment is to install the
following temporary systemd drop-in as
`/run/systemd/system/user@1001.service.d/90-jev-cpuset.conf`, and run
`systemctl daemon-reload`:

```ini
[Service]
Delegate=
Delegate=cpu memory pids cpuset
```

Immediately before writing, recheck the active unit's invocation ID, current
delegated controllers, existing drop-ins and destination. Proceed only if the
baseline still contains exactly the three controllers above and the destination
is absent: the empty `Delegate=` assignment resets earlier assignments. Create
the new file without overwriting anything. `/run` is temporary across reboots.

The drop-in retains the three observed controllers and adds `cpuset`; it does
not set a system-wide CPU list. No user manager, ComfyUI or other service would
be restarted. `daemon-reload` is a system-wide manager reload: it reruns
generators and rebuilds the dependency tree. Afterwards inspect the same unit
invocation, delegated controllers and cgroup hierarchy; a fresh temporary
test unit must independently confirm its effective cpuset, affinity and quota.
Reload alone does not establish that the running service applied the change.
If it did not, stop and report rather than restart or relax the check.

On failure or rollback, remove only the newly created file after checking its
identity, remove a newly created directory only if still empty, then reload
and inspect the unit and cgroup hierarchy again. These actions cannot promise
to undo controller changes already made by systemd. If the cleanup or state
cannot be verified, record it as unconfirmed.

Systemd may enable controllers in ancestor and sibling cgroups, and whether
an already running user service applies the new delegation without a restart
is unverified. The action changes host systemd configuration and can affect
other workloads, so it has not been executed. It does not provide GPU
exclusivity, and no model run is accepted without the existing resource,
identity, host-queue and cleanup checks.
