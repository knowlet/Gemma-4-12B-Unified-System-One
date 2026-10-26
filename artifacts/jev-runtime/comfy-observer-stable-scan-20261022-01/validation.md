# Stable process scan validation

2026-10-22. The old observer was preserved byte for byte. The successor changes only bounded process observation: one initial attempt plus at most one fresh attempt for the same PID after demonstrable comm/start-time or argv disagreement. Both command contexts and all three surrounding live-process comm observations latch known or uncertain Comfy signatures. A later disappearance or read failure cannot erase those signatures. Permission/cwd/read uncertainty is not retryable. Persistent disagreement remains uncertain. Admission still requires two complete fresh listener/process scans after actual connection refusal.

The old read-only soak reproduced one `uncertain_process_identity_changed` refusal in 150 observations (149 confirmed absences, 16.106 seconds). The stopped training owner's exception did not preserve its evidence, so this reproduction does not establish that owner's exact failure cause.

The successor passed **63 pure tests in 0.21 seconds** under a 256 MiB address-space limit, CPU limit 30 seconds and wall limit 45 seconds. Ruff lint and formatting passed. Regressions include transient/persistent identity and argv changes; a final-stat-only identity change; Comfy in either command context or any live comm; disappearance immediately after an observed Comfy comm; known Comfy followed by permission failure; unreadable cwd; and a failed second attempt. Existing HTTP, listener, two-scan, host identity and structured command cases remain.

A real read-only successor soak completed **150/150 confirmed absences, zero refusals in 25.091 seconds**. During it, **549 explicitly owned harmless shell-to-Python exec children** started and were reaped; all exited zero and none timed out. One actual process identity transition was resolved by the single fresh same-PID attempt. This is a bounded regression check, not a statistical availability guarantee. The soak used a 256 MiB address-space limit, CPU limit 45 seconds and wall limit 60 seconds. It invoked no GPU/model/browser/service/systemd operations and made no ComfyUI changes. Source hashes matched before and after the soak.

Independent read-only review confirmed the three-stat sandwich, both context latches, strict blockers, two-attempt bound and unchanged two complete scans, with no remaining must-fix findings.

| Artifact | SHA-256 | Bytes |
| --- | --- | ---: |
| `comfy_observer.py` | `4084e7651a2f4e1559e2b20b061a73e2bb25426097e039ffaab0dc553c88ce7c` | 17501 |
| `test_comfy_observer.py` | `659625c83e606e3d45e26c13e121e79c852f107e880513c89300780f14f4dc02` | 20657 |
| `run_owned_exec_soak.py` | `951062a8b723924303555cdb60e3c7cb0b50a99b55866eeaef5ce827f8c8dc3b` | 3311 |
| `old-observer-soak-01.json` | `77942a1bbce23f8b77059a82236885e9e2a5b8b95e189ee4f8be6f67efd9e224` | 17622 |
| `new-observer-owned-exec-soak-01.json` | `f7b611d9d8400d8226414c7da7868fb18520905d6321c109bfe3293db9de3bc4` | 486427 |
| `/tmp/comfy-observer-stable-scan-20261022.xml` | `239e4daf3f5b0802d57481fc131759ba477b6668d058922f8e1bcb3bcc220306` | 12859 |

The original observer remains `afbc0ad9248543c7b4ec48a773a799dde779138971636f49ff89acd0b2c47ca9` (13982 bytes), and its original tests remain `d63f62be214ad88b2e3e5d5cc5bb42332eb040b73c2219cfde9280f73b238c3d` (11443 bytes).
