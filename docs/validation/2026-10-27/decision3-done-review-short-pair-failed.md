# Decision3 DONE review short-pair diagnostic

The opt-in DONE review diagnostic stopped during the candidate arm. The original
baseline arm did not run, so this attempt produced no complete paired result or
benchmark score. It was a single development diagnostic and does not establish
browser performance or a leaderboard result.

The candidate recorded a `review_failed` event. The candidate service ended
with `failed_no_task_result`, and the existing reconciliation could not produce
an accepted result from the failed service terminal. The runner recorded two
actions before the failure; this count alone does not establish which input was
delivered or whether the task succeeded.

The public Decision3 native parser delegates validation to the generated-action
parser, which requires an `operation` question and accepts only the action and
named target heads. The DONE review request supplies a standalone
`completion_evidence` question. Independent CPU-only synthetic reproduction
confirmed that both parser entry points reject that shape with
`RequestValidationError` before inference. The previous synthetic transport
mock answered the review head directly and therefore did not exercise this
real parser contract. The reproduced protocol incompatibility is consistent
with the service's recorded validation error; the available aggregate evidence
does not independently prove the contents of that failed request. This attempt
does not establish that the model reviewed the task incorrectly.

The recorded client cleanup reported no remaining tracked processes. The
supervisor reported both owned service control groups absent after termination;
the candidate terminal reported zero final CUDA allocation. These observations
do not establish global GPU exclusivity or transport completeness.
