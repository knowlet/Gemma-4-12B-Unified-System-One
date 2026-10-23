# Tiny CPU storage-hook owner, successor 02

Source-only successor to preserved owner01. Root's actual01 stopped before admission because the user cgroup has no `cpuset.cpus.effective`; no Torch import occurred and original-owner closure passed. Owner01 controller/test were restored to exact `66f8f045`/`76c09a19` bytes, and its actual evidence was left untouched.

The only ownership/resource changes are the new exact `...-02.service` unit and `CPUAffinity=0`. Parent reads the observed main thread's affinity twice and its `/proc` allowed CPU list on a stable PID/start; bootstrap also reports its inherited affinity. The manager affinity string and optional cpuset file are raw evidence. Missing cpuset does not imply prevention of later affinity widening or permanent CPU 0 restriction for every runtime thread. The one-core `cpu.max` quota remains a separate kernel check.

Successful closure requires a live original MainPID, original `authenticated_stop`, the live MainPID in that closure's `before_stop`, and closure finishing before the bootstrap's same-boot monotonic five-second hold deadline. Expiry exit 125 cannot qualify as successful owner completion.

All host, source and other unit controls remain: read-only original idle gate, reserve `R=max(24 GiB,ceil(MemTotal/5))`, initial `R+2 GiB`, runtime/final `R`, memory 2 GiB, swap 0, total runtime 30 seconds, stop 3 seconds, core 0, FD 512 and file size 8 MiB. Original boundary/observer/authority sources stay unchanged.

Default execution is dry. A real run needs the independently reviewed controller SHA, actor SHA and selection SHA, plus fresh output directly under this directory; use `source-freeze.json` refs. Only root may perform the separately authorized actual run. Preparation performed 15 pure stdlib tests and Ruff checks, with no unit, observer HTTP, framework import, mmap, model or Comfy action. See `static-validation.json`; `research-decision.json` distinguishes root-observed facts from this source-only preparation.

The two `*-livepid-proposal.py` files archive unexecuted intermediate proposal bytes before exact owner01 restoration. They are historical source evidence, not launch entries.

Any generated `tiny-weights.safetensors` stays **local-only, excluded from Git**. This tiny CPU mechanism cannot qualify 12B, GPU/UMA transfer, context length, 20-second latency, N3/Q95, complete installed-runtime integrity or model quality. Closure concerns the exact authenticated unit and observed cohort, without a universal process-tree claim.
