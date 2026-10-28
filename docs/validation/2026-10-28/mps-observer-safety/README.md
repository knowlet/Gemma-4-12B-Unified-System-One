# MPS observer copy safety

The first instrumented full-model comparison is **invalidated**. Its observer
combined a GPU-to-CPU copy with dtype conversion. On this PyTorch 2.8.0 MPS build,
that operation misreads nonzero-offset BF16 views and can mutate an aliased
int64 attention mask. Consequently neither its hidden-state counts nor its
unchanged-model comparison is accepted. Its original source/results are retained.
The earlier uninstrumented external browser baseline and standalone attention
kernel witness do not use this observer and remain separate evidence.

Two small controls used known CPU reference tensors, with no checkpoint:

| Control | Combined transfer/conversion | Raw CPU copy, then CPU cast |
| --- | --- | --- |
| BF16 view, offset 68 rows | 260,612 wrong values of 261,120 | zero wrong values |
| Int64 mask to Boolean | 833 source values mutated | zero source mutations |

The first BF16 chunk at offset zero copies correctly, which is insufficient to
validate later chunked reads. The mask's returned Boolean copy is correct even
when the source is mutated, so checking the returned copy alone is also
insufficient. The [exact installed-build copy implementation](https://github.com/pytorch/pytorch/blob/a1cb3cc05d46d198467bebbb6e8fba50a325d4e7/aten/src/ATen/native/mps/operations/Copy.mm#L75-L144)
provides a primary source reference for its offset and conversion paths.

The observer now preserves dtype during transfer, then casts the CPU tensor.
Offset-view regressions forbid combined device/dtype requests and check original
BF16 values and mask storage. A repaired full-checkpoint control uses this policy
and a fresh output directory; it does not rehabilitate the invalidated run.
Observed timings include synchronization and are not performance measurements.

[report.json](report.json) includes both controls and explicit invalidation.
[observer-safety-evidence.zip](observer-safety-evidence.zip) preserves exact
executed probe/source bytes, hashes, raw control results and the invalidated
full-model run with its sidecar. Archive CRC and all member digests were checked.
No model weights are included, and no CUDA execution occurred.
