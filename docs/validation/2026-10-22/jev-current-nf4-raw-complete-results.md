# Current NF4 browser baseline: 2/10

2026-10-22. The full V7 run completed on GX10: **2/10 tasks succeeded** (known four: 1/4; public six: 1/6). Both client groups completed and closed. The external controller verified successful worker exit, cgroup absence, two empty process scans, unchanged execution sources and unchanged model cache aliases. An independent audit replayed all ten DOM judges and obtained exactly the same results.

| Outcome | Tasks |
| --- | ---: |
| Success | 2 |
| Premature DONE | 3 |
| Decision budget exhausted | 4 |
| BLOCKED | 1 |

All four budget failures reached the original 16-decision limit; one also recorded a public navigation scope or request-observation failure. The run made 106 policy calls and 5 text-helper calls. All 111 recorded HTTP responses were 200, and all server journals were sealed. A successful HTTP response does not establish task success or, by itself, client delivery.

This control uses the released Gemma base with NF4, the existing raw96 action adapter, raw JSON prompts, and the same adapter-disabled NF4 base as text helper. It retains context 8192, policy/helper limits of 20 seconds, helper output 128 tokens, HTTP 25 seconds, 16 steps and seed 20261014. No candidate, task or judge was removed. Earlier partial attempts remain separate.

The separate BF16 decision path scored 84.42% on public231 and 41.26/100 on coherence mini. The browser result exposes a practical gap, but these different execution paths do not isolate the effects of precision, adapter training or question mode. Historical Q95/BF16 browser 5/10 is also a separate profile. No cross-model win is established.

Independent verification covered the exact 808-file output population, all output hashes, 111 sealed journals/HTTP joins, 620 small source payload hashes and current cache alias identities. The root controller separately authenticated all 621 payloads before and after, including the model weight. The independent audit did not complete a second weight-payload hash: a bounded attempt exited 137 with an unproven cause and receives no success credit.

The next comparison uses a fresh native V2 adapter with the same current NF4 helper, tasks and judges. Training progress and offline loss are not browser improvement evidence. See the [aggregate JSON](jev-current-nf4-raw-complete-results.json) for exact hashes and per-task outcome accounting; raw DOM, requests, answers and weights remain local.
