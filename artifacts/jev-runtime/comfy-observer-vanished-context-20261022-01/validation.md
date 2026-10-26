# Proven vanished process context validation

2026-10-22. The original `4084e765...` observer and its tests remain byte for byte unchanged. The actual scorer's retained failure identified `uncertain_python_cwd` after 471 completed rows. An owned fake-proc reproduction removes its PID directory during cwd read: the original records uncertainty even though a fresh stat could prove disappearance; this successor records `vanished` after that proof.

The source change is limited to process context handling. Classify known or argv-intrinsic uncertain signatures before reading cwd. This includes explicit Comfy modules/executables, absolute Comfy scripts, unknown Python flags and malformed invocations. Only provisional cwd-dependent uncertainty can propagate cwd ENOENT/ESRCH to the existing fresh-stat check. The PID must be absent on that fresh check; a still-live process remains unknown. Permission errors, deleted/nonabsolute cwd and all previously observed context/comm uncertainty remain blocking. No uncertainty retry, deadline, maximum-attempt, HTTP/listener or two-complete-scan rule changed.

**88 pure tests passed in 0.26 seconds** under address-space 256 MiB, CPU 30 seconds and wall 45 seconds. Ruff lint and formatting passed. Tests cover both context positions for proven disappearance and still-live missing cwd, definite/intrinsic uncertain Comfy argv, known Comfy comm, permission/deleted/nonabsolute uncertainty, plus all 63 prior cases. The exact JUnit receipt is preserved as `pure-tests.xml`.

Independent read-only review returned CLEAR after identifying and closing one intermediate gap: uncertain Comfy script paths must latch before cwd just as definite Comfy signatures do. No intermediate source is used in the final soak.

The frozen successor completed **150/150 confirmed absences with zero refusals in 29.978 seconds** during **647 explicitly owned harmless shell-to-Python exec children**. Every child exited zero and was reaped; none timed out. One real process identity transition used the existing single fresh attempt successfully. Address-space/CPU/wall limits were 256 MiB/45 seconds/60 seconds. Source hashes matched before and after the soak. This bounded check does not establish an availability guarantee. No GPU, model, browser, service, systemd or ComfyUI mutation occurred.

| Artifact | SHA-256 | Bytes |
| --- | --- | ---: |
| `comfy_observer.py` | `5220799afc459a3fb702aff5e0d30a7362d14030293b79872a83def6dd1efae2` | 17835 |
| `test_comfy_observer.py` | `7f7e391f6cf70780cda508c1e3522750f9a5d4da5b482a1b18ecca1f942f4bbc` | 26718 |
| `pure-tests.xml` | `69f880461aa3e6bfed7d92cefc65dfe3a7fde3b9caa490065fc4e9a239a76069` | 18420 |
| `old-controlled-reproduction-01.json` | `149998f170152973154d486103cc248520085ef39f0f14c0ca0d4ccb33a5ffc2` | 513 |
| `new-controlled-reproduction-01.json` | `82d87c433f45b03ef2172d37bd6a7b709aa0d99456d22d8f918262bffc381a88` | 463 |
| `new-observer-owned-exec-soak-01.json` | `f8f9886b7a816d89bc6ce3396488a73cc8d17be8daea013efbc1722ba4a5e5f2` | 486424 |
| `run_owned_exec_soak.py` | `537b40bca2f582e2ead26604b3c96679bddcdf9a3997b31c4bdcb520c7ab47bc` | 3311 |

The original observer remains `4084e7651a2f4e1559e2b20b061a73e2bb25426097e039ffaab0dc553c88ce7c` (17501 bytes); its tests remain `659625c83e606e3d45e26c13e121e79c852f107e880513c89300780f14f4dc02` (20657 bytes).
