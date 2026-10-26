# Local Decision3 owned transport

`local_transport.py` is a callable stdlib transport core under the new profile `decision3-gx10-local-owned-transport-v1`. Importing it starts no child, socket, framework or GPU activity. This directory does not qualify any model or authorize a real model launch.

The public parent API is:

```python
selection = Selection(nonce, model, expected_boot_id, expected_memtotal, expected_gpu_uuid)
session = OwnedSession(selection, fresh_empty_owned_output_directory)
ready = session.start(explicit_argv, env=selected_environment, cwd=selected_directory, timeout=1800)
gateway = LocalGateway(session, seconds=20)
http = LoopbackHTTP(gateway, token=private_bearer_token, port=0)
endpoint = http.start()  # Only 127.0.0.1; POST /predict.
# The external original browser runner uses endpoint and the private bearer.
http.close()
closure = session.close()
assert closure["direct_child_exit_confirmed"]
assert closure["observed_process_exit_confirmed"]
assert not closure["owned_exit_confirmed"]
assert not closure["continuation_authorized"]  # External kernel owner proof is still required.
```

The future child entry calls `serve_worker(selection, initialize=..., process_request=..., finalize=...)`. Initialization must perform actual selected source/runtime/cache/Q95/UMA checks, construct the backend and complete publisher text48 warmup before returning its evidence object. The request callback receives the **unchanged original JSONL envelope bytes** and can directly call frozen source17 `process_request(line, adapter=..., wire=..., torch=..., gates=..., output=..., model=..., nonce=..., gpu_uuid=..., startup=...)`. The callback's original `response` object is checked and augmented with the new transport's actual process identity. Finalization must authenticate final model evidence, fence CUDA, close the backend and return a receipt containing a typed `success` boolean.

The original four request fields remain exactly `request_id`, `payload_json`, `http_sha256`, `remaining_seconds`. Control events remain lowercase `ready`, `response`, `closed`. Every new control frame additionally binds the profile, nonce, model, GPU UUID, actual PID and start ticks, selected boot and exact MemTotal. The parent verifies these against its own process observations and freshly checks host identity. GPU UUID is a selected identity at this layer; the real initializer/finalizer must authenticate raw Driver/Torch observations. Transport READY alone proves no CUDA, kernel, warmup or model-quality property.

The parent handles bounded pipe/frame/log/evidence sizes, a maximum of 192 envelopes and HTTP queue/RPC/encoding/durable-publication deadlines. `LoopbackHTTP` passes its publication callback into `LocalGateway.request`, keeping the request serialization lock through publication. The callback prepares complete header bytes first, then binds the durable receipt/response and checks the deadline and blocked state immediately before writing those bytes. The final state check and header write share a lock with gateway blocking. A late or concurrently blocked 200 candidate has its buffered headers discarded, its receipt rebound and its headers rebuilt as 504 before publication. Late raw worker frames are retained during closure. Delivery errors bind the exact durable request and response hash and block the arm. Timing starts at handler entry after request headers; socket accept/header queue, response body delivery and client round-trip are outside the policy deadline.

Only this newly spawned process and authenticated observed descendants are eligible for cleanup. Fresh `/proc` observations use PID/start lineage and an inherited random owner marker; logs contain neither complete command lines nor environments. A detached descendant remains observable through that marker only while it retains the marker. A double-forked descendant that clears its environment can leave both observation mechanisms. Two clean scans therefore establish absence of observed identities, without establishing absence of the entire owned tree. Signals still use pidfds with a start-time check, including libc pidfd functions when the installed Python omits their wrappers. There is no numeric-PID or global process-group signal fallback.

`close()` reports direct-child exit, observed-process exit and a successful bound CLOSED/transport outcome separately. `success` has the explicit scope `transport_and_observed_processes_only`. **`owned_exit_confirmed`, `whole_tree_exit_confirmed` and `continuation_authorized` always remain false**, with reason `external_kernel_ownership_boundary_required`. This core has no mechanism to accept caller booleans or claim an independently owned no-escape boundary. The future full owner must enforce and prove its own kernel ownership boundary before authorizing another arm. Unknown ownership, missing CLOSED, changed identity, forced cleanup or request failure still prevents even observed transport success. A failed instance cannot start another worker; `close()` returns its frozen receipt on repeated calls.

`fake_worker.py` is a CPU fixture only. Tests use actual local sockets and fresh owned fixture subprocesses; they do not import Torch, load checkpoints, run Q95, control ComfyUI, start units, or use cloud resources. The suite includes an actual double-fork/setsid/empty-environment escape: the unobserved fake grandchild remains alive after otherwise clean transport closure, while continuation remains false. The test parent temporarily acts as a subreaper solely to identify, terminate through a bound pidfd and reap that fixture safely. Other tests cover original-body preservation, startup failure/truncation, wrong nonce/PID/start/UUID/total/response binding, worker/queue/publication/delivery deadlines, real client disconnect, marked detached descendants, missing/failed/forged closure, forced cleanup and unknown ownership.

Still required for the first real edge baseline: a separately sealed executable owner and child entry with an enforced kernel ownership/no-escape boundary and independent closure proof, complete actual source union, selected local runtime/environment, fresh Q95 process/listener/credential and simultaneous-memory checker, raw host/Driver/Torch/UUID validation, full cache authentication, actual bootstrap/text48 integration, original browser groups and judges, model-specific final authentication and result publication. Historical H15/source17/18 files and qualification identities remain unchanged; this transport cannot substitute NF4 controls or convert fixture passes into model scores.

Final read-only source verification found a current integration blocker: `src/s1/jev_generated.py` is now 23,569 bytes / SHA `c356e0eb3c046cb0787d7be0c5960803006231e8bdb414de644ae2167e72d849`, while source17 expects 18,825 bytes / SHA `582b520615cf123367436622f4e5ca6f5d94abb227449d866ec64cbb8adc9b3c`. The other sixteen historical source pins match. The old warm bootstrap's original authentication will therefore refuse the current project tree. A real successor must explicitly authenticate its complete current source composition and this delta while preserving the historical projection; rewriting the old manifest or monkeypatching authentication is not an integration. The standalone transport and its CPU fixture do not import any of these project modules.
