# Explicit BF16 diagnostic qualification

The audited child adapter completed the original task **1/1** and the known
exposed development cases **13/13**, versus the unadapted BF16 base **11/13**.
Those are 13 matched cases plus an extra original gate, rather than 27 distinct
tasks. They establish results for these saved fixtures and goals. The subsequent
[audited public BF16 comparison](validation/2026-10-15/jev-public-bf16-comparison.json)
scored **1/2 for both the child and unadapted base**, with exactly equal raw
initial payload pairs. The known-fixture gain did not produce a public-task
score improvement.

The frozen tracked CLI reproduced the original task **1/1**, with
[independently audited native execution](validation/2026-10-15/jev-bf16-qualified-cli-original.json).
The remaining known13 cases were never launched: the next process's memory
admission check refused insufficient available RAM. Full CLI parity remains
pending. Historical qualification authorizes explicit trials; it does not
promote the adapter or establish public or unseen-task performance.

Qualification limits the claims supported by the historical evidence, rather
than the inputs that can be evaluated. The explicit CLI accepts new public or
live tasks, other fixture tasks, seeds, repeat counts and suite splits under the
benchmark's existing scope rules. Its receipt records the requested evaluation
scope separately from historical qualification and retains the completed
benchmark configuration. New inputs produce new diagnostic evidence with no
quality guarantee from the known-fixture results.

The existing `jev_action_lora_v1` manifest, training receipt, NF4 verifier,
attachment function, and default CLI behavior remain unchanged. BF16 requires a
separate qualification record that authenticates the complete fit, recursive
lineage, finalized campaign, independent audit, paired requests and fixtures,
current source bytes and package versions. The issuer derives outcome counts
from saved summaries. It never runs code found in qualification evidence.

Create a new qualification bundle only after the campaign and independent audit
have completed. Run from the repository's qualified Python environment:

```sh
PYTHONPATH=src python scripts/jev_qualify_bf16_action_adapter.py \
  --adapter artifacts/jev-browser/20261015-action-lora-counterfactual-fit-01/adapter \
  --campaign artifacts/jev-browser/20261015-bf16-counterfactual-selection-01 \
  --audit artifacts/jev-browser/20261015-bf16-browser-final-independent-audit.json \
  --audit-sha256 2cbfb9b35384d64524e617bfc1ff09e7b54e471d19f6042043ba57e07d92a877 \
  --audit-source artifacts/jev-browser/20261015-bf16-browser-independent-audit-r3.py \
  --output artifacts/jev-browser/bf16-qualification
```

The issuer uses bounded local file reads and copies. It does not import Torch,
load a model, read the base weights, start a browser, or contact a service. The
bundle includes the complete original evidence, so its size exceeds the adapter
weight file. Source or package changes require a newly reviewed qualification.

After source review and qualification issuance, this command runs the original
diagnostic through the tracked CLI. It is an actual model/browser invocation:

```sh
PYTHONPATH=src python scripts/jev_local_model.py \
  --upstream-path artifacts/jev-runtime/upstream \
  --output-dir artifacts/jev-browser/bf16-qualified-cli-original \
  --quantization none --generated-actions --generated-action-grammar \
  --generated-action-adapter artifacts/jev-browser/bf16-qualification/fit/adapter \
  --generated-action-qualification artifacts/jev-browser/bf16-qualification/qualification.json \
  --server-evidence --tasks upstream_travel
```

The loader constructs the same pinned BF16 model and q/v LoRA configuration as
the diagnostic, initializes exact captured FP32 adapter values, then freezes all
parameters and sets evaluation mode. It preserves the original full vocabulary
head and temperature. The helper disables the adapter under the shared lock and
restores frozen evaluation state on success and exception. PEFT restoration may
briefly re-enable gradients; the lock is retained until the model is frozen
again. Integrity failures remain fatal even when a helper's validation fails.

Telemetry separates the original artifact's NF4 deployment declaration from
actual BF16 computation and the `generated-actions-none` alias. A qualified run
checks the exact local base checkpoint before loading, preserves the complete
qualification before model construction, and verifies head/adapter/evidence
integrity after shutdown. Head hashes and tensor equality checks stay outside
timed policy calls. Policy generation retains its cooperative 20-second budget,
32-token grammar and 8192-token context; helpers retain their 128-token limit
without a claimed 20-second wall-clock deadline. There is no offload or precision
fallback.

Qualified receipt updates use atomic replacement with file and directory
`fsync`. Failed persistence preserves the previous progress receipt when rollback
succeeds and reports any rollback failure. A final write failure is also printed
and makes the invocation fail; earlier model, helper or cleanup errors remain
visible. Decode failures retain their original exception when an exit invariant
also fails, and retained integrity errors still reject finalization.
