# V7 links: synthetic Chromium execution record

This is an execution record for one synthetic, loopback-only browser check, not a benchmark score or a claim of task performance. It records the tool output observed in this session; the original process log was not saved to a separate file.

## Command and session

The successful attempt was started from the repository root with:

```bash
systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 -p CPUQuota=500% -p AllowedCPUs=0-4 taskset -c 0-4 uv run --no-sync python -B /tmp/jev_links_chromium_check.py
```

The command returned exec session `11676`; polling that session returned the stdout below and no continuing session. The polling call displayed only `output`, so the process exit code was not captured as independent evidence. The final JSON was printed after the script's assertions and shutdown checks. The temporary script was present at `/tmp/jev_links_chromium_check.py` when this record was written, with SHA-256 `85bcb41637be6e6a5838242d4c4b31def2f26ce115e5c45ecf2c93b70194a707`; `/tmp` is not durable storage.

An initial attempt exited before starting Chromium because the resource probe tried to read an unavailable `cpuset.cpus.effective` file. Only the temporary probe was corrected; the successful attempt checked five-core process affinity directly and the other limits via cgroup files. No browser was rerun for this record.

## Recorded stdout

```json
{"action_count": 4, "child_processes_remaining": 0, "click_choice_ids_equal": true, "closed": {"browser_disconnected": true, "context_closed": true, "playwright_stopped": true, "port_closed": true, "server_stopped": true}, "external_requests": 0, "fake_model_calls": 0, "fake_post_calls": 2, "fingerprint_equal": true, "memory_peak_bytes": 424849408, "resource_limits": {"cpu.max": "500000 100000", "memory.max": "8589934592", "memory.swap.max": "0"}, "sanitized_links": 2, "selected_action": "e1", "snapshot_original_fields_equal": true, "source_sha256": {"jev_observation_profile_v7.py": "d9f4236d1d46c9472c3d6c1551342639ecb870e6c0dd419246927645410ea868", "jev_observation_profile_v7_links.py": "bba6b9ea3ba53b18bba3015565ea5f211efa443b7e66572f548b87b348d045d8"}, "status": "passed"}
```

The `external_requests` field is the number of non-loopback page requests observed and blocked by the Playwright route, not a system-wide network audit. The two `fake_post_calls` invoked an in-process fake response function; no model inference or model HTTP request was performed. The original snapshot also contains its existing fields, so the query and credential sanitization assertion applies to the newly added link destination field only.
