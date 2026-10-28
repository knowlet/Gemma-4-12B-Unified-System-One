# MPS public231 baseline

The causal-multislot MPS baseline completed all 231 pinned public tasks. The
official source-pinned scorer was replayed from hashed raw answers; all task
IDs, request hashes, artifact hashes and summary calculations passed the audit.
No calibration fitting or model selection used these labels.

| Metric | MPS / PyTorch 2.8 | Historical A100 / PyTorch 2.10 |
| --- | ---: | ---: |
| Correct | 197/231 | 197/231 |
| Coverage / strict validity | 100% / 100% | 100% / 100% |
| ECE | 0.0515550 | 0.0484954 |
| Brier mean | 0.2199824 | 0.2295357 |

The recorded weight/config/calibration identities match and all 231 request
hashes match. Three labels differ: one is correct only on MPS, one only on A100,
and one is wrong on both. Maximum absolute probability difference is 0.506384
on `easy-intent-00 / track_order`, although both runs select the correct label.
Equal total accuracy therefore does not establish probability or per-case parity.
This diagnostic neither fits the model nor changes any scoring rule.

MPS uses Apple M1 Max BF16/SDPA. Its callable p50/p95 is 1.456/22.622 seconds,
excluding loading. Host testing and diagnostics overlapped this stage, so these
timings are exploratory. No cross-hardware speed advantage is claimed, and no
new CUDA job was started. CUDA testing remains assigned to gx10.

Source attestation is partial: the original MPS pre-execution archive omitted
`run_jevbench_public.py` and the benchmark configuration. Their recovered bytes
are explicitly marked as collected after execution. The source-pinned official
dataset/scorer and raw outcomes remain auditable; this does not turn recovered
files into before-execution proof. Historical full processor/source attestation
is also partial. Consequently this is not a controlled optimizer-parity test.

[report.json](report.json) includes the paired cases, identities, limitations and
archive member hashes. [public231-evidence.zip](public231-evidence.zip) contains
all MPS raw answers, both record sets, source-pinned official task/scoring files,
license, summaries and the explicitly recovered runner/config. Its CRC passed.
Re-audit after extraction with the current repository:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts").resolve()))
from run_jevbench_public import load_public
from summarize_jevbench_campaign import audit_public

tasks, official, config = load_public("extracted/official")
result = audit_public("extracted/mps", tasks, official, config)
assert result["summary"]["n_correct"] == 197
assert result["summary"]["n_valid"] == 231
```

These are legacy public231 research results, not P300/S1200, sealed results,
official rankings or external browser-task success. Cost remains unknown.
Independent-mode public231 and both coherence results are still pending in the
same serial MPS campaign. Prefix reuse is a separate queued experiment.
