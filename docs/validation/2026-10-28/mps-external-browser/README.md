# MPS external browser baseline

Both separately frozen Wikipedia tasks completed three attempts per profile with
the real 12B checkpoint, original Jev Ultrafast policy and unchanged destination
predicates. Every attempt used a fresh Chromium context, 12-cycle cap and
180-second budget.

| Profile | Destination verified | Search task | Follow-link task | Latency p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| causal_multislot | 3/6 | 0/3 | 3/3 | 93.192 / 96.115 s |
| independent | 0/6 | 0/3 | 0/3 | 55.919 / 58.166 s |

The causal follow-link attempts reached the exact expected URL, title and heading,
with the required body terms. These were independently checked after termination;
agent DONE alone did not determine success. Search failed before typing because
the native text helper produced a code fence and repeated underscores rather
than the required JSON object. The selected operation and search field were
correct, which does not establish task completion.

Every independent attempt failed with an invalid probability distribution before
text generation. This is an inference failure requiring numerical diagnosis;
the zero task-success rate is retained in the attempted denominator. It cannot
measure the quality of unavailable decisions. Correcting the helper alone would
not address these failures.

The profiles made 13/6 decision backend calls and 3/0 text-helper calls,
respectively. Actual complete-backbone forward counts were 397/6; causal's
generation calls each reached the 128-token limit. The helper uses the same
specialist checkpoint as the decisions, while upstream normally uses a separate
JSON-capable text model. Its general text quality and protocol suitability were
not established by the earlier local fixture tasks, which never typed.

The two tasks are a narrow repeated population, not six distinct goals per
profile, statistical superiority or an official ranking. They are now consumed
for diagnosis; later generalization claims need fresh tasks. No frozen scorer
was relaxed. Cost remains unknown. CUDA execution remains assigned to gx10.

[report.json](report.json) retains every attempt, environment, checkpoint hashes,
source identity and audit. [external-browser-evidence.zip](external-browser-evidence.zip)
contains all twelve full traces, observed/final DOM, screenshots, CDP logs, raw
decision/helper records, before-execution local sources, verified upstream
source/license, and supervisor/terminal child logs. Archive CRC, member hashes,
all twelve final-state predicates and actual call counts were checked.

Reproduce using [the external manifest instructions](../../../ultrafast-evaluation.md)
with a fresh output directory. Preserve the 180-second external budget separately
from the 90-second local fixture results. This archive records the original
failure before any numerical batching fix.
