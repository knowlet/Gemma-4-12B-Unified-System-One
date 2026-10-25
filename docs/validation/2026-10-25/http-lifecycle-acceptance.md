# HTTP lifecycle acceptance

Date: 2026-10-25. Base: `codex/jev-public07-modal-training`,
`e085f8f513c6a36681d2dea03b0a44b2873ebf54` (contains PR #13).

## Reproduced defect and bounded fix

The current native browser service's `make_application` acquired its serial
policy/helper lock and awaited `request.body()` without a deadline. A client
sending one byte of a declared 100-byte body could keep all subsequent requests
waiting indefinitely. A real Uvicorn/TCP probe reproduced a blocked `/healthz`
even after the injected service clock exceeded 20 seconds; disconnecting the
partial client immediately restored service.

Body receipt now uses `asyncio.timeout` with the **remaining original** 20-second
HTTP budget. Its timeout follows the existing 504 refusal, evidence publication,
TRACE restoration and lock-release paths. Inference is outside this timeout.
The focused regression initially failed because the partial client received no
response; the other three cases passed. All four now pass: body timeout while
the client remains connected, FIN, RST, and complete-body success. The focused
native/browser profile suite returned 103 passed and three existing local
historical-evidence skips.

An independent run used the **default monotonic clock**, actual native wrapper,
Uvicorn h11 and TCP, without changing the deadline or invoking any decoder:

| Observation | Result |
| --- | --- |
| Body sent / declared | 1 / 100 bytes; socket kept open until response |
| Partial client's response | HTTP 504 after 20.013432 seconds |
| Subsequent fresh `/healthz` | HTTP 200 after 0.011820 seconds |
| Published refusal | Exact response bytes retained; no candidate or inference request |
| Cleanup | Server thread stopped; zero ASGI tasks and connections |

## Synthetic decoder and real socket acceptance

The existing NumPy decoder/Torch shim from `tests/test_jev_generated.py` was
used with the **actual** generated-action adapter, `create_app`, Uvicorn h11 and
loopback TCP. The eight scenarios passed both before and after the fix:

- Async HTTPX caller cancellation, HTTPX read timeout and TCP reset after decode
  entry; the synchronous worker continued, retained its terminal computation
  evidence, removed its forward hook and accepted a succeeding request.
- Identical queued requests retained distinct server IDs and serialized decode;
  no deduplication, caching or automatic retry is claimed.
- Partial-token transport, timeout and internal generation failures retained the
  observed tokens, returned the appropriate 502/504, removed hooks and recovered.
- A late decode crossing the injected cooperative 20-second budget returned 504,
  retained the raw output and recovered on the next request.

All eight server instances released their threads, ASGI tasks, connections and
forward hooks. `server_returned` means server computation completed; it does not
prove the disconnected client received a response.

Additional core API acceptance checked 30 failure/recovery pairs across both
single-request aliases and both batch aliases, using native and fallback batch
execution. Validation, timeout, transport, malformed-output and internal errors
all released the inference lock before a successful next request. An actual
HTTPBackend/slow loopback upstream produced 504 on read timeout, recovered on
the next request, made exactly two upstream requests without retry, and closed
its client connection. No additional defect was found.

## Validation and reproducibility

Environment: Linux x86_64, Python 3.12.14; locked FastAPI 0.141.1,
Starlette 1.7.0, Uvicorn 0.54.0, AnyIO 4.15.1, HTTPX 0.28.1, NumPy 2.5.3.
Only the API/browser extras were installed. All commands used
`UV_CACHE_DIR=/tmp/gemma-uv-cache` and test runs used `HF_HUB_OFFLINE=1`.
The existing upstream checkout stayed at
`1231850a0bf1a0c0341fe408ef1668dbbfdfac46`.

| Check | Result |
| --- | --- |
| Baseline full applicable pytest | 4671 passed, 693 skipped, 112 deselected; exit 0 |
| Fixed full applicable pytest + coverage | 4675 passed, 693 skipped, 112 deselected; exit 0 |
| Final focused regression | 4 passed; exit 0 |
| `ruff check .`, `ruff format --check .` | Exit 0 |
| `uv build --no-sources` | Exit 0 |
| Uniform smoke benchmark, eval smoke plan, text-smoke run | Exit 0 |
| `scripts/benchmark_v2_acceptance.py` | Exit 0 |

The full pytest command was:

```sh
JEV_UPSTREAM_PATH=artifacts/jev-runtime/upstream HF_HUB_OFFLINE=1 \
UV_CACHE_DIR=/tmp/gemma-uv-cache uv run --no-sync pytest \
  -m 'not frozen_decoder' --ignore=tests/test_jev_qualify_public_actions_bf16.py \
  --cov=s1 --cov-report=xml:artifacts/lifecycle-acceptance/final/coverage.xml
```

PR #13's test file was explicitly excluded as requested. Existing skips include
uninstalled inference/tokenizer dependencies, opt-in Chromium and absent local
historical artifacts. The unchanged frozen Rust oracle requires its dedicated
Linux ARM64 environment and was not run here. No gate or skip policy changed.
Coverage produced its XML and existing warnings about synthetic
`::action-profile-find` filenames; the logs preserve them.

The regression's simulated remaining budget was widened from 0.25 to 1 second
during the full run for scheduler tolerance. The final four regressions, lint,
format and build were rerun with stable final source hashes; the product script
did not change during verification. The archive records both test hashes and
the actual source hashes instead of claiming a committed-head test run.

[Evidence archive](http-lifecycle-acceptance.tar.gz) contains commands, exit
codes, logs, JUnit, source/environment hashes, synthetic JSONL events, cleanup
records and the standalone socket/core/default-clock reproduction programs.
Unpack it from the repository root. The broader harness can be rerun with:

```sh
HF_HUB_OFFLINE=1 PYTHONPATH=scripts:tests UV_CACHE_DIR=/tmp/gemma-uv-cache \
  uv run --no-sync pytest -v \
  artifacts/lifecycle-acceptance/socket-after/test_socket_acceptance.py
```

## Remaining acceptance

These are synthetic decoder/service tests, **not Gemma/model, GPU, browser-task,
device or production acceptance**. The existing tiny genuine CPU model fixtures
were unavailable because Torch/Transformers were not installed. No paid model,
large model download, GPU or user computer was used.

Running synchronous inference still cannot be hard-cancelled safely. Core API
queue admission and native wrapper lock waiting remain cooperative; this patch
bounds only body receipt, and does not impose a hard deadline on model execution.
`MAX_BODY` is still checked after buffering; streaming size enforcement remains
separate. PR #9/#11 integration and historical data/device/GPU work remain outside
this change. Remote push, Draft PR and exact-head CI results must be reported
separately; local passing checks do not establish remote CI status.

See [implementation research](../../research/jev-http-ingress-lifecycle.md).
