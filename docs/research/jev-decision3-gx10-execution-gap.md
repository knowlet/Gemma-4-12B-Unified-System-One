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

## 2026-10-25: empty Python package sources

The next real launch exited 1 after 162.03695940895705 seconds, before service
readiness or any case result. Its new failure was `complete positive file
identity required`. Replaying the original verifier without models accepted
182 of the 188 source records and rejected six correctly pinned, zero-byte
Uvicorn `__init__.py` files. The failure is in source authentication; it is not
a scored browser mistake.
The independent audit reauthenticated all 859 selected sources and confirmed
both original observed process scopes had closed. Audit SHA256:
`8924e941929a0fa898739f221b886547e7b00674165be55de87e245f6df9b806`.

The upstream Uvicorn 0.34.3 tag resolves to
`5e33d430f13622c8363fe74d97963ab37f3df3c2`. Its
[lifespan package initializer](https://github.com/Kludex/uvicorn/blob/5e33d430f13622c8363fe74d97963ab37f3df3c2/uvicorn/lifespan/__init__.py)
is empty (Git blob `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391`). Uvicorn's
[BSD license](https://github.com/Kludex/uvicorn/blob/5e33d430f13622c8363fe74d97963ab37f3df3c2/LICENSE.md)
was checked; no upstream implementation is copied for this fix. Python also
[permits empty package initializers](https://docs.python.org/3/tutorial/modules.html#packages).

| Existing component | Decision |
| --- | --- |
| `SourcePins.pin` in the public runtime-profile builder at `32c6789` | Reuse for zero-byte Python sources: it checks the exact hash and size, ordinary paths, regular files and stable file-descriptor/path identity. |
| Original GPU actor `verify_record` | Retain for every other source, model and payload; its positive-size policy remains unchanged. |
| Browser worker `pin` | Do not reuse here: its read has no byte limit. |

The service successor will select and authenticate the existing builder source,
then use `SourcePins.pin` only for exact three-field inventory references whose
size is an integer zero and whose path is an ordinary absolute `.py` file.
Dependencies, metadata, model files, delta and explicit source loaders keep
their existing nonempty checks. No upstream file is padded, omitted or edited.
The builder itself is unchanged; its file-descriptor and bounded-hash mechanism
is reused rather than introducing a second implementation.

The new [seven regression controls](../../tests/test_jev_empty_python_sources.py)
passed under Python 3.12 in 0.026 seconds, with Ruff lint and formatting checks
passing. They cover a real temporary empty package source, incorrect pins,
nonempty content, symlinks, path replacement during hashing, invalid size types
and tensor filenames. Actual successor service readiness and browser quality
remain to be tested; paired and complete ten-case scores remain null.

## 2026-10-25: first-case pair completed after the startup fixes

The later [accepted first-case GPU pair](../validation/2026-10-25/decision3-firstcase-gpu-paired-results.md) reached candidate readiness and completed both infrastructure paths. The original launch exited 0 after 243.1005604180391 seconds; candidate and matched NF4 baseline each failed the task, 0/1. The audit verified 860 selected files, the 189-entry source inventory including six empty Python files, original worker identity, both observed owned process scopes and the worker's final allocator zero. The earlier startup failures above remain separate historical attempts; no OS-level containment or whole-machine GPU-zero claim follows.

All four genuine Decision3 requests returned HTTP 200, with matching original client/server bytes. Independent FP64 replay checked 144 legal probability positions at the original tolerance, with maximum absolute error `1.4973786233163366e-7`. The observed failure was task progress: one radio click changed unchecked to checked, then three repeated clicks left the page unchanged and triggered the original stagnation stop. W0 BLOCKED-to-composed CLICK on the same features is a local readout comparison, not a separate unmodified Edge trajectory.

The full ten-case score and both fresh candidate JevBench scores remain null; there is no browser qualification or competitor win. Further generic checked-control and state-progress examples must remain separate from this exposed evaluation case and its goal, with the API action choices and original judge unchanged.
