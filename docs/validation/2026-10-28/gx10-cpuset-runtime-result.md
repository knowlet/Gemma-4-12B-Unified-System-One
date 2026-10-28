# GX10 temporary cpuset delegation result

The user approved a temporary systemd drop-in and daemon reload without
restarting services. The existing `user@1001.service` remained active with
the same invocation and PID. After the drop-in was loaded, systemd reported
`cpu`, `cpuset`, `memory` and `pids` as delegated controllers. The upper
`user.slice`, `user-1001.slice` and `user@1001.service` cgroups exposed
`cpuset.cpus.effective`; however, the running user manager's `app.slice` did
not propagate cpuset to its transient child.

A fresh, short-lived diagnostic unit requested CPUs 15–19, CPU quota 500%,
8 GiB memory and zero swap. Its process affinity was 15–19 and `cpu.max` was
`500000 100000`, but its own `cpuset.cpus.effective` did not exist. The probe
exited without loading the model. The existing strict short-test gate was not
relaxed, and no user service was restarted.

The user then ran the receipt-bound rollback. The temporary drop-in was
removed and systemd was reloaded. Afterwards the same user service invocation
and PID were active, its delegated controllers and cgroup controllers were
`cpu memory pids`, its subtree control was `cpu memory pids`, and the
temporary drop-in was absent. This verifies those observed paths and unit
properties, not every cgroup or possible host-wide side effect.

The CUDA synthetic plan remains prepared but has not run. Making the control
group requirement available may require a future user-manager restart or a
different independently validated environment; neither was authorized or
performed in this experiment.
