# First local Decision3 Edge browser run

This preparation selects the concrete frozen execution component `901549c3108d572b04e8402c46d7953c982860d41a22d3f2bc8087a8baa2bf5c`, helper checker `b20cdf98a7db4283f75fc8453ce2eafe839b33bfcec35ca2582249fa69b61994`, and Q95 frozen `serve-plan-02.json` (`a14708dbc42ff8526ada639f989ba9effe633c7f64d7dd5c4abd66ba3ff03247`). No actual Decision3 model or browser has been run by this preparation.

`preparation-selection.json` is an authenticated static template. It includes the exact37 base source references, actual selected Chromium binary, current boot/MemTotal/GPU UUID, exact Edge revision and two selected fresh units/nonces/output paths. `prepare_edge_plan.py` embeds this template's SHA; drift refuses rather than accepting current source bytes. Default execution checks the small source references and reports preparation scope. It launches nothing.

The original Q95 controller must first be started by the separate owner under `jev-d3-q95-helper-20261022-01.service`, 40GiB/swap0/runtime10800. Its public `owner-01/helper-selection-seed.json` contains exactly the live checker's13 fields except `co_resident`; actual helper PID/start/session/listening inode are unknowable before service startup. The separate `helper-identity.json` is evidence rather than the descriptor seed. Neither contains credentials. This preparation does not read the private config or contact the helper.

After actual helper readiness, independently select the saved seed's SHA, then prepare the concrete plan:

```text
/home/albert_hsieh/ComfyUI/.venv/bin/python -B -s \
  artifacts/jev-runtime/decision3-edge-browser-plan-20261022-01/prepare_edge_plan.py \
  --prepare \
  --helper-seed /home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/decision3-q95-owner-20261022-01/owner-01/helper-selection-seed.json \
  --helper-seed-sha256 <independently-selected-actual-seed-sha256> \
  --expires-after-seconds 9000
```

The builder copies only the public seed, adds each preselected D3 cgroup/model/resident field, and statically verifies the original95 source and348 installed runtime files through the real frozen checker constructor. It never calls the checker, its text connection, a live endpoint, an owner or a controller. It uses the real frozen `execution.prepare_plan`, compares its exact39 source references to the selected37 plus two helper descriptors, and runs the real `selected_plan` validator, including the selected Chromium hash. It saves `helper-known4-01.json`, `helper-public6-01.json`, `plan-01.json` and `prepared-launch-01.json` exclusively. Existing preparation or execution outputs are never overwritten. Static source checks are not fresh live admission.

Review the exact plan bytes and SHA emitted in `prepared-launch-01.json`. Actual launch is a separate explicit operation, using its `launch_argv_for_external_owner`:

```text
/home/albert_hsieh/ComfyUI/.venv/bin/python -B -s \
  artifacts/jev-runtime/decision3-local-execution-20261022-01/launch_execution.py \
  --execute --role controller \
  --plan /home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/decision3-edge-browser-plan-20261022-01/plan-01.json \
  --plan-sha256 <independently-selected-plan-sha256>
```

The controller handles known4 then public6 with distinct fresh workers, units and external closures. Units are `jev-d3-edge-known4-20261022-01.service` and `jev-d3-edge-public6-20261022-01.service`, each80GiB/swap0/runtime7200. Evidence output is the original execution directory's `edge-01`; model outputs are separate `artifacts/jev-d3-native/20261022-edge-known4-01` and `20261022-edge-public6-01`. No actual output directories are created by preparation.

The selected Edge checkpoint is `vllm-sr/d3-edge@cab71e6ad3461b8086d51ddb572c656babd63b0b`, with resident file bytes1,189,722,528. At MemTotal130,595,864,576 the unchanged CORE floor is30,646,101,620 bytes: reserve `max(24GiB,ceil(T/5))`, plus2D and2GiB, against both fresh host available and real cgroup headroom, with no swap/resident credit. Q95 independently retains its73,958,271,732-byte host floor. The actual launch must pass these fresh checks; this template grants no admission.

Original native255-element capacity, publisher48 text warmup, batch8, BF16 backbone/FP32 readout, context8192, temperature1, no fast path/permutation averaging, and raw request conversion remain in the frozen implementation. Known4 retains64 envelopes and public6 retains96, with one repeat/16steps/seed20261014 and unchanged V7 observations/judges. All six checkpoint identities remain in the selected bootstrap; only Edge is planned here. Largest D3 remains capacity-refused under the unchanged floor.

Six bounded CPU tests passed, including a complete real `prepare_plan`/`selected_plan` roundtrip with saved95/runtime348 and selected Chromium. Only that temporary test uses an explicitly synthetic public worker/listener; its files are deleted, it receives no live admission, and no callbacks, actors, units, models or browser run. Negative tests cover source substitution, extra/private seed fields, boot/MemTotal/helper-cgroup/source-root mismatch and per-arm binding. No tensor payload is read by preparation/tests. Installed-runtime version/source/METADATA/origin observations remain the accepted provenance scope; no whole-runtime execution attestation is added.

The actual first Edge score is still pending. A completed result requires all ten original IDs, exact DOM verification, DONE plus no error, final cache/source/runtime/geometry receipts and successful scoped external closures. Different helper/precision/adapter conditions prevent a current cross-model win claim from this preparation or from an unmatched future result. Stale-feedback experiments do not alter this unchanged D3 V7 plan.
