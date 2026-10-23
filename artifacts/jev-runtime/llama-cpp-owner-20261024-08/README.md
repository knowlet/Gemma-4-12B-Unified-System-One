Owner08 is an unvalidated SOURCE-ONLY successor of the frozen owner07 controller.
No import, AST comparison, builder, formatter, test, dry run, host observation,
unit, model, service or Git action was executed while authoring these files.
No freeze, validation result, execution command spec or source qualification is
provided. All new source hashes and downstream selections remain PENDING in
`plan.pending.json`; that document is not a runnable command specification.

The parent controller SHA recorded by owner07 is
`685a9f809f429f4e9658083dc8530b563740625f9ba7816908fdc58002de256d`.
It and every old frozen profile remain unchanged. Only the copied controller's
following resource/name values change:

| Setting | Owner07 | Owner08 |
| --- | --- | --- |
| Unit | jev-d3-llama-cpp-cpu4-2g-coexist-20261024-07.service | jev-d3-llama-cpp-cpu16-4g-coexist-20261024-08.service |
| Profile | llama-cpp-cpu4-2g-coexist-command-owned-v1 | llama-cpp-cpu16-4g-coexist-command-owned-v1 |
| CPU cores / affinity | 16-19 (four) | 4-19 (sixteen) |
| CPUQuota / manager readback | 400% / 4s | 1600% / 16s |
| Kernel cpu.max ratio | 4 times positive period | 16 times positive period |
| MemoryMax | 2,147,483,648 bytes | 4,294,967,296 bytes |
| Initial full-cap extra | 2,147,483,648 bytes | 4,294,967,296 bytes |
| Initial required available, declared host | 28,266,656,564 bytes | 30,414,140,212 bytes |
| Runtime/final R, declared host | 26,119,172,916 bytes | 26,119,172,916 bytes |

Root selected CPU4-19 after its own read-only online/allowed observation. This
author did not query host topology or utilization. Future actual admission still
requires original fresh unit/cgroup absence, boot and identity authentication,
full initial R+4GiB, and the selected scheduler/proc/kernel resource readbacks.
R remains `max(24GiB, ceil(MemTotal/5))`; the declared host total is
130,595,864,576 bytes. Busy HTTP evidence provides no memory credit.

The exact original boundary, observer and system-services authority pins are
retained. The known HTTP busy/idle and confirmed-absence rules, strict malformed
or unknown refusal, source/stat budgets, bootstrap environment, swap0, 256MiB
file limit, first4MiB log retention, 30/7200/10800 runtime choices, stop3s,
identity/nonce/argv/cwd checks, primary-error retention, authenticated cleanup
and no-escape/whole-tree non-claims are preserved. The bootstrap's conservative
thread environment remains1. A future separate native profile must explicitly
set its own selected16-thread environment before importing its framework;
no native actor, head math, model, scorer or campaign is changed here.

`test_controller.py` retains the parent stdlib fake regressions with new resource
expectations, old-cap/floor refusal, old four-core/quota refusal, mixed owner07
refusal, exact parent source pin and a complete16-substitution controller byte
comparison. These are test SOURCE, not executed results. All assertions about
unchanged bytes/functions remain pending actual review and authorized checks.

Root must review all files and the exact resource delta, then authorize bounded
pure tests/format/source comparisons, compute new source hashes and create its
own freeze. Native/campaign consumers and their source/plan/freeze/command pins
must be new, explicitly reviewed and resealed. Old smoke, timing, memory, model,
benchmark, browser or closure results cannot qualify owner08. No launch is
authorized or prepared by this source-only package.
