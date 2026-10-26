# V7 bundle with typed stale-action feedback

This is an explicit experimental agent protocol, `typed-stale-action-feedback-v1`, on the unchanged `composed-visible-text-settled-v7` observation/runtime/task/judge surface. No browser/model score has been measured for this successor.

`feedback_bundle.py` calls the exact V7 preparer (`31450f9002ad216c32b77751997eb27748c491d7bc757c4410294d3704fdfb9e`). After it prepares and verifies the original/heldout bundle, the only runner addition is:

```python
import action_feedback as feedback
model_adapter = feedback.wrap_model_adapter(model_adapter)
```

It is placed after the adapter definition and immediately before the existing main guard. The exact reviewed utility (`79ccf2ccc2d88f9613c1320ee883ce9742b0a88c0535dbfae4de7c97ee428246`) is copied to the fresh bundle. The manifest labels the separate agent protocol, embeds the unchanged original V7 manifest and hashes every file. Verification regenerates the original V7 bundle, checks complete inverse runner bytes/AST, exact copied utility, unchanged other file bytes and exact ordinary population. No original V7 preparer, upstream module, observation helper, task, judge, browser safety guard or candidate inventory is edited.

The wrapper adds at most four typed stale-attempt records to `state.action_feedback` through the original benchmark model adapter, before routing and recording. The first policy request stays identical. Actual executed history remains separate and physical input outcome remains unknown. A real generated-runner fake-I/O regression verifies identical feedback in the sent body, saved model request and saved transport request. A separate pure check exercises the real complete shared-elements-v2 formatter and proves that feedback survives into the exact model input and lossless payload roundtrip.

`feedback_native_driver.py` is a saved-byte successor of the original current-NF4 browser driver (`c513dc5bcb091de4fd0d4bd7a0776078e82be16106328b6de9d8424e32f2ee11`). It changes only its ROOT location, explicit profile/agent-protocol metadata, the selected bundle-preparer import and composition-source labels. `launch_browser.py` separately binds the new driver/profile and `v2-feedback` arm; the copied entry is byte-identical. `assemble_sources.py` reproduces all four copies/forks from fixed parent SHAs. `source-deltas.json` and the patches retain full inverse bytes/AST proofs. Model construction, N3 adapter admission, NF4 policy/helper behavior, serialization, request/HTTP budgets, resources, cleanup and source/cache closure bodies remain unchanged.

The concrete unexecuted `launch-plan-feedback-01.json` selects the same N3 adapter, shared-elements-v2 serializer and current NF4 base helper as the completed V2 control plan `73bff770e242b93d5b7b53d24ef933926bd63897a4707d6d2e2a0f06f4438b6d` (4/10). It retains all785 original selected references and adds exactly five runtime sources. Preparation independently streamed789 small file hashes, then checked the one23.9GB base payload's size plus all11 cache alias targets/links/inode/mtime/ctime against the completed control's recorded closure. It does not claim a new full-weight hash pass. The unchanged real owner performs its full checks on actual launch.

The fresh unit is `jev-current-nf4-v2-feedback-20261022-01.service`,40GiB/swap0 with the original memory floors. Its fresh output is this directory's `v2-feedback-01/browser`. It keeps the original known4/public6 groups, one repeat,16steps,seed20261014 and fixed denominator10. Other model/helper campaigns should be externally closed before collecting this matched successor. Actual execution remains an explicit external-owner operation:

```text
/home/albert_hsieh/ComfyUI/.venv/bin/python -B -s \
  artifacts/jev-runtime/browser-stale-feedback-bundle-20261022-01/launch_browser.py \
  --run \
  --plan /home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/browser-stale-feedback-bundle-20261022-01/launch-plan-feedback-01.json \
  --plan-sha256 17627efbf0f1f2e61f7a233c503fd6d9b7ebae998e4bf11521590bab0bc90ea9
```

This command was not run. Default bundle/owner/entry commands were verified dry. Five bounded CPU tests passed in5.106s under256MiB/CPU30s/wall45s, and Ruff passed for all seven Python sources. Tests cover both groups, actual generated-adapter transport recording, complete V2 model input, full source inverse proofs and tamper/missing/extra/symlink refusal. The reviewed utility's separate six-test receipt remains its unchanged prerequisite evidence. No GPU, model, actual browser, unit, helper endpoint or Comfy operation was invoked.

A future result must still replay all ten actual DOM judges and verify client/server/source/cache/external closure. It can measure this matched protocol change only after completion; no improvement, broad usability, causal claim or cross-model win is asserted now. Existing D3 V7 plans stay unchanged.
