# Explicit FIND_TEXT runtime integration

The prior experimental FIND_TEXT capability passed seven local Chromium
canaries with fake decisions, but the actual client, API and Decision3 worker
still selected the original eight-operation classes. This change connects the
reviewed implementation through an explicit `jev-find-text-experimental-v1`
action profile. The default remains `standard`.

The client uses a separately verified bundle derived from the unchanged V7
bundle. Its four additional edits are reversible. It selects the experimental
Agent with a private sixteen-step namespace, records the profile, and requires
an explicit external HTTP text helper. The canonical benchmark, task order,
judges, original protocol sources and original wire remain byte-identical.
The API and Decision3 worker select the matching private nine-operation adapter
and wire. Private modules coexist with original apps without replacing them.

The current NF4 driver now accepts `--action-profile`, `--text-endpoint`,
`--text-model` and `--text-api-key-env`. The experimental profile requires an
explicit loopback helper; the standard profile can use the same helper for a
matched control. Helper credentials remain in the selected environment variable.
Names overwritten by the driver are refused, and the external path disables
local helper generation. Its source inventory verifies the seven dependencies,
selector and new bundle builder before and after execution.

The fixed real suite remains ten tasks, one repeat and sixteen decisions per
task; the original 20-second request/helper budgets and 25-second HTTP client
remain. The experiment is a new profile and inherits no earlier success rate.

Validation: 162 portable tests plus 15 subtests passed in 6.12 seconds, using
CPU0 with a 1GiB address-space bound and 40 CPU-second limit. A further 34 targeted tests passed after clarifying external-helper token
metadata: the client submits 1024, while the external native cap needs its own
authenticated runtime receipt. These include
original driver behavior, concurrent standard/experimental API instances,
private step limits, reversible bundle edits, helper credential preservation,
local-helper refusal and copied-source tamper rejection. They start no real
model, Chromium or network server. The earlier test collection attempt used
an incompatible Python3.13 overlay with Python3.12 and failed before collecting
tests; the successful run used the repository's Python3.12 environment.

Actual NN-selected FIND_TEXT through the real HTTP helper and browser, followed
by all ten matched tasks, remains required. No new real-browser score or
competitor win is claimed.
