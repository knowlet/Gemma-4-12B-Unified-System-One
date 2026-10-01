# PR #2 review fixes

Verified against review head `b639f00c06b0f57be5375b116246ba8d0ca92c86`
on 2026-09-30 (Asia/Taipei). All four findings reproduced before modification.

| Finding | Root cause and reproduction | Corrected behavior |
| --- | --- | --- |
| P1: noninferiority population | Comparison retained the joint eligible intersection; the gate checked only its groups/CI. Real planner/runner output for 300 reference-supported versus 30 candidate-supported decisions had 30 groups, CI `[0,0]`, and passed. | Default full-population gate is inconclusive. An explicitly declared common subset requires a minimum support fraction and reports its identities, denominator and scope. Unsupported rows retain their status. |
| P1: workflow fitting overlap | Workflow checked configuration/capabilities but omitted fitting provenance and source groups. A real prior baseline's training ID, group or request blocked the normal planner but workflow completed. | Shared validation audits every episode state, including unvisited states, for training/calibration ID, group and canonical request overlap before adapter construction. |
| P2: pipeline reported cost | Workflow read only decision `cost_usd`. Real PipelineAdapter preprocessing `$0.50` plus decision `$0.20` reported `$0.20`. | Stage bills are retained and summed once; the example reports `$0.70`. Missing bills from dispatched stages keep totals null. Known preprocessing bills survive downstream failures. |
| P2: workflow runtime identity | Workflow omitted the runner's telemetry validation. Float32 against declared float64 and Gemma/Laya revision mismatches still dispatched predictions. | Shared telemetry validation checks all loaded adapters before the first prediction. Resolved identity is persisted; mismatch produces setup_error with zero inference calls and closes loaded adapters. |

The SetFit CI failure was separately confirmed from the
[failed inference job](https://github.com/knowlet/Gemma-4-12B-Unified-System-One/actions/runs/36722520076/job/109910990720).
The model loader and Transformers trainer select their devices independently;
inference-only `device="cpu"` did not constrain fitting. The tiny encoder fixture
now explicitly selects CPU, disables accelerator selection for that test, sets
`ACCELERATE_USE_CPU`, and asserts both the fitted model and trainer are on CPU before
running the actual training call. This passed on the local host with MPS available.
The original MPS out-of-memory error was observed in the CI log, not reproduced locally.

## Verification

All inference, train, API, baseline, Modal and Laya extras installed from `uv.lock`.
No pretrained weight downloads, live providers or full 12B inference were used.

- Final full suite on the macOS host, Python 3.12.11: **499 passed, no skips**;
  three upstream deprecation/pinned-memory warnings.
- **44 added regression cases** across gate scope, workflow provenance/runtime
  integrity and pipeline cost; the original SetFit integration test now checks CPU.
- Ruff lint and format checks: passed.
- `uv lock --check --offline`: passed.
- Offline E0–E7 acceptance script: passed, including persisted workflow telemetry
  and completed adapter status.
- `uv build --no-sources --offline`: wheel and source distribution built.
- Independent cross-review of workflow accounting and pre-inference setup: no
  material correctness gaps found.

Full-suite command:

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q
```

This local verification was completed before committing and pushing the repair.
Remote CI results for the pushed head are reported separately by the PR checks.
Legacy comparisons without population evidence must be regenerated before a
noninferiority gate can pass; a passing common-subset gate does not establish
acceptance for the entire workload.
