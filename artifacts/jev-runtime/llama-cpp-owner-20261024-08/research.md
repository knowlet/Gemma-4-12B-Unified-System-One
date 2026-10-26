This change uses local source research only: frozen owner07 controller,
test_controller.py, README.md, research.md, plan.json and freeze.json, plus the
already pinned original owner_boundary.Selection schema. There was no new
network research, host probe, import or programmatic check.

Root authorized a separate CPU16/4GiB owner and selected affinity4-19. The narrow
implementation changes six top-level assignments (UNIT, PROFILE, CPU_CORES,
CPU_ALLOWED_LIST, MEMORY_MAX_BYTES, INITIAL_HOST_EXTRA_BYTES), plus the resource
literals/diagnostics in fixed_memory, host, properties, main_affinity, run and
main. It introduces no function, model branch, scorer, admission credit,
observer fallback or cleanup mechanism. The proposed test source enumerates
all16 exact text replacements and compares every other function/class AST with
the pinned parent when root later authorizes execution.

MemoryMax and INITIAL_HOST_EXTRA_BYTES must both equal4GiB, not merely agree.
An old2GiB/2GiB pair, either mixed pair, and adjacent-byte differences must fail
before observer/unit selection. The host reserve formula remains unchanged;
for declared MemTotal130,595,864,576, R=26,119,172,916 and initial required
available=30,414,140,212. Runtime/final observations require the same R. Native
leaf startup admission must later use the entire4GiB as well, without credits.

CPU quota1600% has manager readback16s and a kernel ratio16 times its positive
period. Main-thread affinity must match4-19 in both scheduler reads and proc,
with the same process identity throughout. This preserves the parent scope:
main-thread affinity observation is not delegated cpuset enforcement, an
all-thread lifetime guarantee or no-escape proof. Original no-escape and
whole-tree false qualification fields are not promoted.

The generic boundary already permits a selected cap up to40GiB, so4GiB needs no
authority edit. Its original MemoryHigh=infinity, swap0, OOMPolicy=kill,
KillMode=control-group, stop policy and selected runtime choices remain intact.
The original observer still owns HTTP200/list/count/deadline/absence evidence;
its source SHA and the coexist observation policy do not change.

Pending root work is explicit: independent source review, bounded pure checks
only after release, fresh actual source SHA/size capture, freeze, new native and
campaign source/pin chains, exact root command selection and later admission.
Native batch4, all1479 full requests, per-call180s, campaign10000s, owner10800s,
context16384/reject-no-truncation, original model/readout/masks/temperature and
official scorers must remain unchanged in those future consumers. A genuinely
new same-profile native smoke and closed audit are required before a new full
campaign; none is produced or inherited here.
