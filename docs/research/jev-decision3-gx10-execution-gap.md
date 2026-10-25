# GX10 Decision3 execution and capacity gaps

2026-10-22 read-only source audit at `80d8694`. This records newly confirmed remaining work; it does not claim an executable comparison, model warmup or quality improvement. Reuses the pinned research in `artifacts/jev-runtime/decision3-GX10-warm-successor-20261020-01/research-decision-01.json`.

The completed local checkpoint transfers include all six models, but `gx10_warm_bootstrap.py` is deliberately a component library: its `Components` owns no listener, scheduler, delivery or process supervisor, its CLI refuses execution, and `validate_config` still accepts only lite/edge. `gx10_warm_probe.py` is a standalone diagnostic, not a browser service. Therefore an idle GPU alone will not make the all-six comparison executable.

Useful existing implementations are the unchanged `scripts/jev_decision3_browser_worker.py:process_request` for native inference/raw evidence/deadlines, `apps/modal/decision3_browser_service.py:Gateway` for ingress/queue/RPC/final-publication/delivery handling, and its `Session` for owned-process and arm-closure semantics. Cloud mounts, H200 runtime validation and Modal RPC cannot be relabelled as a local GX10 service. A successor must inventory its actual source union separately and retain the old projections; it must not monkeypatch the old cache-selection constants.

The missing fresh Q95 checker also requires implementation. Bootstrap `live_helper_check` checks a callback's returned shape and booleans; it does not itself prove fresh process, listener, credentials or simultaneous memory admission. The existing Q95 semantic runner supplies qualification/file/PID checks, while the older field-JSON observer supplies process-group/listener/private-credential mechanisms. Reuse those mechanisms under the correct Q95 identity. `/healthz` bypasses bearer authentication and cannot alone prove credential binding.

Capacity is a separate barrier. With physical RAM 130,595,860,480 bytes, reserve 26,119,172,096 bytes, D3 resident files 52,176,025,120 bytes and Gemma base 23,919,549,408 bytes:

| Existing policy | Required available bytes |
| --- | ---: |
| Q95 load and every forward: reserve + 2 × Gemma base | 73,958,270,912 |
| D3 CORE: reserve + 2 × resident files + 2 GiB | 132,618,705,984 |
| Native CLI before helper: reserve + D3 + max(D3, Gemma) + Gemma | 154,390,771,744 |

The D3 CORE floor alone exceeds physical RAM by 2,022,845,504 bytes. These are existing conservative admission formulas, not measured peaks or proof that the model can never run on this hardware. The component has not declared a complete simultaneous helper/model policy. A future capacity profile would need explicit, enforced bounds and genuine admission measurements, preserving the old policy and receipts; subtracting resident memory or silently reducing the floor is insufficient.

The complete execution stage must launch and freshly bind the helper, authenticate each selected cache and runtime, start a fresh model × workload worker, perform the original publisher text48 warmup, serve the original HTTP payload through native `process_request`, run both original browser groups, and confirm final sources/cache/runtime/UUID and owned-process closure. Keep all six models in the required population, preserve per-group 64/96 request envelopes, 16 steps and original judges, and report any capacity refusal separately from model mistakes. New Gemma controls must actually share the selected helper regime; the existing NF4 in-process helper and historical Q95 results cannot be reused as matched new controls.

CPU integration verification can cover real loopback HTTP and owned fake-child READY/request/CLOSED framing, nonce/model/PID/UUID binding, queue/publication deadline crossings, disconnects, stale helper identity, partial startup and incomplete cleanup. It cannot establish actual model quality, CUDA capacity, speed or superiority. No Decision3 source or guard was changed during this audit.

## 2026-10-25: observed GPU service startup failure

The first fixed-case GPU browser diagnostic loaded the NF4 helper, then failed
before Decision3 service readiness or model loading. The original launch exited
1 after 162.03758619102882 seconds. No candidate or baseline case result was
produced; the paired score and full ten-case score remain null. The independent
audit reauthenticated all 856 selected sources and confirmed both original
observed process scopes had closed. It does not establish OS-level containment.
Audit SHA256: `2e7912a39e452715d46e34e93ee79bf09e5739f3f892cf8449da9b3216e8bb59`.

The deliberately minimal worker environment omitted `XDG_RUNTIME_DIR` and
`DBUS_SESSION_BUS_ADDRESS`. Its first `systemctl --user show` failed with
`No medium found`. Repeating a read-only query with that exact environment
reproduced the failure without loading models; adding only the selected current
user's local bus settings made it succeed. This matches the documented
[systemd user-bus environment and ENOMEDIUM behavior](https://freedesktop.org/software/systemd/man/latest/sd_bus_default_user.html).
The native offline benchmarks did not exercise this service-readback path.

The new [readback library](../../scripts/jev_user_unit_readback.py) constructs an
isolated environment for that query only. It requires explicit current-user bus
selection, verifies the private runtime directory and owned Unix socket before
and after the query, and requests the original five properties. The caller must
still verify their identity values. It preserves primary errors, checks the
caller's deadline and uses a one-second subprocess timeout. It neither changes
the model process environment nor mutates units.

All [10 invented regression controls](../../tests/test_jev_user_unit_readback.py)
passed under Python 3.12, as did Ruff lint and formatting checks. A real read-only
query through the helper returned all five properties without changing the
calling environment or importing models. GPU service integration and browser
quality after this fix remain unverified; no benchmark or competitor result is
inferred from these controls.
