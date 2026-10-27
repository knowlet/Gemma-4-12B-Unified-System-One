# CPU Decision3 browser components

The source-only CPU browser adapter and loopback HTTP core are preparatory
components. Their synthetic tests do not load a model or establish browser
task success. Neither component supplies a model owner, a hard process
deadline, resource admission or verified cleanup. They are disabled by default.

The adapter retains the native request and all choices, validates the
publisher response and usage, and latches failures. Its 20-second checks are
cooperative: a blocked model forward continues until an external owner stops
the process. The loopback core accepts one IPv4 localhost connection at a
time, requires a private bearer token, bounds the request and response, and
latches errors and overlapping connections. Socket timeout and disconnection
do not cancel a blocked forward.

An HTTP response can reach a client before delivery bookkeeping or later
cleanup fails; bytes already written cannot be withdrawn. The recorded socket
write does not establish client receipt. A future supervised caller must wait
for the delivery event and handler completion before opening the next
connection, and independently verify evidence and closure before treating a
test as successful. Blocking handlers may make ordinary server closure wait;
the external owner must be able to terminate and verify the whole controlled
process group.

Real-model browser evaluation still requires a separately reviewed model
service, exact process and listener binding, resource checks, an independent
watchdog, and final-state verification. Existing CPU and cpuset requirements
must pass before model loading; missing evidence remains a failure.
