# Operation-only readout after the complete DSL01 development comparison

Decision recorded on 2026-10-24 after commit `85a24f9020aa4e2226a9c4789e05d143ba79580e`.

The shared readout improved the complete training objective but reduced valid-target hits on the complete development set from 18/33 to 17/33. Operation hits stayed at 10/24; one select-target hit was lost. The genuine browser diagnostic also failed. The shared candidate remains diagnostic-only. These observations motivate an operation-only experiment; they do not establish that this experiment will improve browser completion.

This is a narrow extension of the existing researched implementation. The reference inventory and licensing are retained in `docs/research/jev-decision3-local-successor-inventory.md`, the DSL01 README and its `UPSTREAM-LICENSE`, and the cached-head/core and native-runtime source records. The pinned Decision3 Edge revision is `cab71e6ad3461b8086d51ddb572c656babd63b0b`. The original Browser Use request construction remains unchanged (model.py SHA `a85ada8458d23642c5c8a4095acf0565e6268f81dcda26f6308942b653e9c9f3`, questions.py SHA `66a7054b85040b949e1757311257ea0c5a843c635f9a40c99f6d15ad660fb422`). No new dependency or upstream API is introduced, and no new external search is claimed for this extension.

| Existing component | Relevant mechanism | Decision |
| --- | --- | --- |
| `decision3-train-readout-cpu5-20261024-01/train_loader_released_v1.py` | Validates all original COMMITTED chunks, train identities, actual head ordering and native probability replay | Copy to a new source; retain all validation, then select only the 96 genuine operation rows |
| `decision3-cached-head-core-20261023-01/head_core.py` | Zero-initialized masked FP32 delta SGD; separate `h @ W0.T + h @ delta.T` | Use the exact frozen core; fixed 512 steps, learning rate .003, L2 .0001, seed 20261023, batch 64 |
| `decision3-delta-head-backend-cpu5-20261024-01/backend_released_v1.py` | One original forward per real chunk; final-norm features, original logits/respond, required evidence and complete-call budget | Copy to a new held source; add delta only to rows whose actual head ID is `operation`; retain original target logits |

The optimizer continues to validate all 132 supervised training rows, then excludes the 36 supervised target rows and the 252 unused rows from fitting. It must not assume operation is the first offered head. No new backbone collection is needed. New candidate/fit capabilities and source hashes must be distinct from the rejected shared candidate. The old source, capabilities and failed results remain immutable.

The new strategy was chosen after development exposure: `dev_used_for_model_selection=true`. Development, sealed, official-gold and exposed browser-diagnostic inputs are forbidden in the optimizer. This is a file-flow restriction, not an OS sealing claim. The sealed set remains unrevealed, and a future protocol/candidate must be fixed before opening it.

Source fakes must distinguish actual reordered/absent operation heads, singleton and partial batches, unchanged target logits, mutation and required-evidence failure. Genuine fitting must independently complete within the unchanged original owner09 CPU5/4GiB/120-second transaction. Numeric inference, browser completion and the two separate JevBench suites require new evidence; none is inherited from the old shared candidate. The fixed four-head/eight-operation DSL training distribution remains a known limitation relative to real browser inputs with fewer offered choices.
