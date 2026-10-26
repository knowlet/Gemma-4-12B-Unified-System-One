# Held numeric v2: exact comparison after original response formatting

`actor_held_v2.py` and `test_operation_numeric_held_v2.py` are new source copies.
The original `actor_held.py` and its qualified test source remain unchanged.
The v2 author has not imported, executed, parsed, formatted, hashed or tested
these files, and has not read cached publisher source, actual receipts,
capabilities, arrays, labels or other diagnostic payloads for this task.
`SOURCE_ONLY=True` and `PINS['backend']=None` remain in place. Root must review,
check and explicitly pin/release a new copy before a genuine run.

Root established the pinned original publisher contract: Choice response
formatting converts values to Python floats, sums them and returns each divided
by that total. The call chain from `runtime.respond` through the original answer
formatter performs this normalization after Torch replay probabilities exist.
Therefore exact comparison between pre-format replay floats and a retained
post-format response can reject an unchanged target row. V2 preserves exact
comparison and accounts for the original formatter at its proper point.

Immediately after each call's `observer.replay_probabilities`, v2 invokes
`native_response_control(runtime, prepared, keys, lengths, original_replay)`.
It calls the same original
`runtime.respond(prepared, dict(zip(keys, original_replay)), sum(lengths.values()))`
without encoding or invoking logits. The helper checks the original answer and
criteria order, returns that full response plus its ordered probability values,
and records the response privately as `row['native_response_control']`.
Original and candidate branches both pass these publisher-formatted control
values to the unchanged exact target-preservation comparison.

The retained control is permanently NONTRAINING under the same private call
evidence, failure handling, finalization and durability path. It supplies no
desired answer, semantic metric or browser success claim. Formatting, retention
and cleanup remain charged to the original complete20-second call and
120-second whole transaction. No prediction, encoder or backbone forward is
added. The four original/candidate4/2 calls, original same-forward replay,
zero primitive, FP32 logit reconstruction, parameter/W0 guards, mixed tolerance,
selected plan/source authentication and primary-error handling are unchanged.

Target preservation still requires exact FP32 legal-logit bits, zero target
delta logits and exact returned Python-float probability bits. There is no
tolerance increase, probability correction or answer repair. Candidate operation
probabilities may differ; offered options, target order and raw responses are
not changed by the auditor.

The v2 fake source loads `actor_held_v2.py`, retains the original12 tests and adds
one formatter-contract regression. Its invented replay includes0.2+0.3, which
requires normalization to0.4/0.6. The fake verifies the same prepared object,
ordered mapping and token count go through `respond`, the replay input is
unchanged, an extra logits call is forbidden, raw replay floats fail the exact
comparison, normalized control succeeds, and a target change of1e-15 is still
rejected. These are authored test sources, with no execution result claimed.

The new private `native_response_control` field must be included when root
adapts the closed numeric audit. No prior source checks or actual qualification
are silently transferred to v2; new selected pins, a genuine job and closed root
acceptance remain required.
