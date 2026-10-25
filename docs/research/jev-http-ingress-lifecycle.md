# HTTP ingress lifecycle acceptance (2026-10-25)

## Scope and starting evidence

Base: `codex/jev-public07-modal-training` at
`e085f8f513c6a36681d2dea03b0a44b2873ebf54`, including merged PR #13.
This checkout has no repo `AGENTS.md` or `.agents/skills`; `/workspace/.agents`
is empty. No historical source pins, missing historical artifacts, evaluation
gates, model weights, GPU jobs or external devices are part of this work.

Existing paths inspected:

- `src/s1/api.py:create_app`: synchronous inference workers share a thread lock.
- `src/s1/jev_generated.py:JevGeneratedActionAdapter.predict`: native decode
  orchestration, token observation, cooperative deadline and hook cleanup.
- `scripts/jev_native_v2_browser.py:make_application`: the current browser
  service serializes policy/helper requests, preserves HTTP evidence and checks
  the original whole-request 20-second deadline before response authorization.
- `tests/test_browser_server_evidence_api.py` already covers ASGI caller
  cancellation and duplicate request identities; `tests/test_jev_generated.py`
  supplies a deterministic NumPy decoder fixture and hook observation.

A real Uvicorn/loopback experiment sent one body byte with `Content-Length: 100`
to the native browser service. Its serial lock stayed held in `request.body()`;
even with the service's injected clock advanced beyond 20 seconds, another
`/healthz` request received nothing until the first client disconnected. No
decoder ran in that experiment. This is a missing asynchronous ingress deadline,
separate from the documented inability to hard-cancel synchronous model calls.

## References inspected before implementation

| Reference | Match, version and license evidence | Decision |
| --- | --- | --- |
| Python `asyncio.timeout` | Python 3.12.14 installed source `asyncio/timeouts.py`, `Timeout.__aexit__`; [official 3.12 documentation](https://docs.python.org/3.12/library/asyncio-task.html#timeouts), available since 3.11; PSF license | Use the existing standard library to cancel only the pending body await and convert its own cancellation into `TimeoutError`. |
| Starlette `run_in_threadpool` | Locked 1.7.0 installed `starlette/concurrency.py`, BSD-3-Clause license in the installed distribution; wheel SHA-256 `67f8e99895493dd2911a03f11314af6ceebeae4e704bb9f43dfc6a9db151c93e` in `uv.lock` | Retain existing synchronous inference semantics; its AnyIO worker cannot be stopped by cancelling an HTTP caller. |
| Uvicorn `H11Protocol.connection_lost` / `RequestResponseCycle.receive` | Locked 0.54.0 installed `uvicorn/protocols/http/h11_impl.py`, BSD-3-Clause distribution license; wheel SHA-256 `505bdb0f318731d45f1f712071fc781a8981f6847a31c902c9f5e652d4f67faf` in `uv.lock`; [official server behavior](https://uvicorn.dev/server-behavior/) | Use the existing server and real sockets to verify disconnect behavior; server computation evidence alone does not prove client delivery. |

The targeted upstream search covered Uvicorn disconnect behavior and Starlette
thread-pool dispatch. Published older upstream source was consulted for context;
the exact installed, locked sources above determine this change. No upstream
code is copied or rewritten and no new dependency is introduced.

## Decision and acceptance

Bound only native browser HTTP body receipt by the time remaining in its
original 20-second budget, including time already spent waiting for the serial
lock. Keep existing timeout refusal, TRACE restoration, evidence publication and
lock release. Do not wrap synchronous inference in a cancelling timeout or claim
that an in-process model can be killed safely. Body size streaming enforcement
and general API queue admission are separate follow-ups.

Regression evidence must fail before the fix and pass afterwards using actual
Uvicorn/socket ingress, an existing synthetic model descriptor and the existing
injected service clock. The partial client must stay connected while receiving
504 and another request must subsequently succeed. Also check body disconnect,
ordinary complete bodies, refusal evidence and server-thread/socket cleanup.
Separate acceptance uses the existing synthetic deterministic decoder with the
real generated-action adapter and API to cover caller cancellation, read timeout,
RST, duplicate queueing, failed decode cleanup and recovery. None of these
results constitutes Gemma, GPU, browser-task or production acceptance.
