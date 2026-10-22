# Native browser HTTP final publication deadline

2026-10-22, before implementation. Reuse the previously researched pinned Browser Use source and the existing project gateway; no new dependency or downloaded code is needed. This is a narrow correction to the current NF4 browser HTTP path, not a diagnosis proven by the historical 5/10 trial.

| Candidate | Source actually inspected | Decision |
| --- | --- | --- |
| Browser Use Jev `1231850a0bf1a0c0341fe408ef1668dbbfdfac46`, MIT | `jev_ultrafast/model.py` request/response behavior, already recorded in `jev-native-v2-browser.md` | Preserve the original 25-second client, wire and read-only browser tasks. Do not extend deadlines or repair actions. |
| Existing native browser runner, project MIT, HEAD `80d8694` | `scripts/jev_native_v2_browser.py:make_application`, existing timeout regressions in `tests/test_jev_native_v2_browser.py` | Correct its final publication boundary: the last 20-second check currently precedes response write, hashing and exchange fsync; a late candidate can therefore still be returned. Preserve candidate bytes and truthful final refusal evidence. |
| Existing Decision3 gateway, project MIT, same HEAD | `apps/modal/decision3_browser_service.py:Gateway.request`, final-evidence and response-binding deadline checks | Reference the fail-closed ordering already used here. Do not import Modal, change frozen Decision3 source, or replace the browser runner with a different service. |

Verification must exercise the real HTTP middleware using a fake model and deterministic slow evidence publication. Cover response bytes, final exchange metadata, candidate preservation, duration accounting, deadline equality, and publication failures; use no model, CUDA, browser or ComfyUI operations. Existing transport and lifecycle tests must continue passing. The already sealed JevBench plan pins this runner and must receive a new source-hash plan before any later actual launch; the original plan and refusal evidence remain immutable.
