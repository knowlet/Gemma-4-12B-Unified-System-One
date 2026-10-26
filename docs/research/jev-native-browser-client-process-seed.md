# Browser client ownership seed repair

2026-10-22, before implementation. Raw02 completed the four known tasks but could not start public6: the client closure had empty tracked PID identities/groups and reported unconfirmed. Inspection of `matrix_common.owned_rows/extend_owned` shows it only expands an existing known PID seed. `run_client` supplied an empty dictionary throughout.

Reuse the repository MIT reference `artifacts/jev-runtime/public-affordance-matrix-harness-20261019-01/matrix_controller.py` immediately after its `Popen(start_new_session=True)`: capture the actual child stat, require its own PID/PGID/SID, seed its start ticks and process group before expansion. Add direct-parent and positive-start checks. This is the missing caller contract; leave the historical common primitive unchanged. The prior upstream/runner comparison and license research is in `jev-native-v2-browser.md`; no new dependency or browser behavior is introduced.

Verify with actual tiny stdlib children and the real common tracker/cleanup, including a child that exits immediately, plus mismatched identity rejection and retained primary exceptions. No model/GPU/browser/ComfyUI is needed. Then reseal a fresh full raw10 attempt. Preserve raw02 and its executed source bytes; no task, helper, policy, judge, candidate or timing limit changes.
