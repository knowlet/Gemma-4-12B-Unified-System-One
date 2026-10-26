The local primary sources used before this change are the exact owner06
controller SHAaa2af9911fde9fd06526c3fd5fe45f6521e80103bbccf9c29b98c7f3bba5952d,
its test_controller.py and README, and unchanged original observer
SHAfd977040d659770e26a8fd6cc2b94e6c5440a6a3d3f05099e07d7597d8309e45.
No network, host observation or payload read was used for this source preparation.

Owner06 `host` accepts only `http_idle` or `absent_confirmed` with0/0counts; this
is a selected-profile invariant. It is not a general user prohibition on bounded
CPU coexistence. Root explicitly authorized a separate profile within the user's
GX10 multi-core CPU/smaller Decision3 scope, retaining the old profile unchanged.

Original observer `_queue` lines88-106 requires HTTP200, a bounded1MiB response,
and actual list-valued queue_running/queue_pending, then reports their lengths,
status and SHA256. Its successful HTTP branch lines885-897 includes top-level
counts and nested evidence with identical counts and sets busy iff either is
positive. Original absence lines870-883 reports0/0 with connection_refused
provenance only after its original fresh sequential absence checks. `observe`
lines900-908 shares the original deadline and does not convert an error into a
busy result. The new owner consumes these exact source-authenticated outputs;
it adds no endpoint or alternative observer and performs no Comfy action.

Both top-level and nested HTTP counts must be exact nonnegative integers and
match. Busy is a positive-count state; idle is0/0. Explicit HTTP200/SHA evidence
prevents an unknown/failure-shaped fake observation from being accepted merely
because it says busy. The pinned observer remains responsible for bounded
transport, same-boot memory and original absence verification. SHA evidence
identifies response bytes; it does not identify or authenticate the application
behind the endpoint. No new application-identity or job-origin claim is made.

The resource source remains byte-identical outside the declared policy/name/dry
delta: initial R+full2GiB, runtime/final R, CPU16-19/quota400%,2GiB/swap0,
256MiB files, bounded source/output counts, exact tools and runtime30/7200/10800.
Original child identity, source/stat authentication, timing/primary cleanup,
live hold and authenticated boundary closure functions are not patched.
Meaningful new stdlib fakes exercise accepted busy and strict refusals plus
startup/runtime floor-1 cases, directly contrast unchanged owner06 busy refusal,
and compare every other top-level controller function/class AST. Existing
resource and finalization regressions are retained. Root subsequently released
the bounded source-test window after prior closure verification. Saved validation
records report those checks separately from actual admission or model execution.

Future actors and root selections must explicitly select owner07/policy rather
than carrying owner06 identifiers or actual-fit claims. There is no model,
Torch, native binary or quality result in this source package. The generic owner
contains no model-specific behavior, and changing admission policy does not
change model inputs, math, judges, contexts, budgets or feature qualification.
