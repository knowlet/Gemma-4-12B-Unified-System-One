# MPS post-fix browser replay

Both profiles completed the original external follow-link task in fresh contexts
after the native sliding-window batch guard. The exact goal and destination
URL/title/heading/body predicates match the frozen original task.

| Profile | Verified | Time | Decision calls / backbone forwards |
| --- | --- | ---: | ---: |
| causal_multislot | 1/1 | 91.334 s | 3 / 3 |
| independent | 1/1 | 132.968 s | 3 / 6 |

Both finish within the original 180-second budget and twelve-cycle cap, without
text generation or exceptions. Independent traces record the native 1024-token
batch guard. Every final-state predicate and actual model/forward count was
replayed from raw evidence. Agent DONE alone is insufficient.

The [original uninstrumented baseline](../mps-external-browser/README.md) failed
all three independent attempts of this task with invalid probabilities; causal
passed three of three. This one-attempt diagnostic replay supports the specific
numerical defect repair. The populations are not matched repeated comparisons,
and this consumed task cannot establish generalization or statistical
superiority. Independent's successful completion takes longer than its earlier
failed attempt; no speed improvement is claimed.

Search/text-helper failures remain unresolved, and both JevBench regression
results remain separate. The queued resume first recovers the original baseline
using its archived S1 implementation and measured cache maintenance; those
results must not be labeled post-fix benchmark scores.

[report.json](report.json) contains complete identities, raw-summary audit and
per-member hashes. [postfix-browser-evidence.zip](postfix-browser-evidence.zip)
preserves both full browser traces, snapshots, final screenshots, transport logs,
executed source bytes, verified upstream code/license and terminal supervisor
receipt. CRC and source hashes passed. Cost remains unknown. No CUDA execution
was performed.
