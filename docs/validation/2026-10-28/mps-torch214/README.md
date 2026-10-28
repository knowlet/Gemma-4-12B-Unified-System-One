# Isolated MPS Torch 2.14.1 control

The isolated environment changes only Torch 2.8.0 → 2.14.1 and TorchVision
0.23.0 → 0.29.1 from the recorded 88-package browser environment. The project
dependency pins and default environment remain unchanged. The actual Torch
build reports source commit `5c4886908584029761b579af026dcfb627c84070`.

| Control | Torch 2.8 | Torch 2.14.1 |
| --- | --- | --- |
| Fully masked sliding-padding SDPA | NaNs contaminate real queries | all finite |
| BF16 nonzero-offset view readback | wrong values | exact values |
| Int64 mask conversion source | mutated | unchanged |
| Driver growth after trimming, same first 20 callbacks | 18.748 GB | 0.171 GB |

The three small probes use identical source/input bytes across runtimes. The
20-request replay uses the same verified checkpoint, original archived S1
implementation, BF16/SDPA/temperature and chronological successful callback
requests. All twenty execute actual forwards, with the first sample retained.
Driver allocation after trim stays within the predeclared 1 GiB growth bound.
The replay is bounded; it does not establish unlimited process lifetime. Forward
instrumentation differs from the old run, so these are not controlled speed
measurements.

**Numerical parity fails:** maximum probability difference is 0.060997, exceeding
the unchanged 0.01 gate, and 7 of 239 question heads change argmax. No labels are
used here, so this neither proves accuracy improvement nor degradation. The
memory result cannot override the numerical failure or substitute for fresh
public231/coherence/calibration and actual browser acceptance. The script exits
1, and this runtime has not been made the default.

The same saved native-helper input was also tested on the existing base and
specialist models, sequentially. Their weight hashes differ. Under Torch 2.8
both produce a code fence plus repeated underscores. Under 2.14.1 both produce
`{"text": "Graph theory"}` inside one complete JSON code fence. The unchanged
strict diagnostic parser still rejects those fences. This rules out attributing
the old observed failure solely to S1 fine-tuning; one input does not establish
general helper quality. No second model has been adopted. A separate strict
transport adapter can remove a whole fence without inventing field content or
changing task verification.

The inspected upstream fixes motivate these controls:

- [All-masked MPS softmax handling](https://github.com/pytorch/pytorch/commit/194539e9c33dc793fe67fbb68c7cee12f399e276), present from 2.9.0.
- [Graph-cache clearing in `empty_cache`](https://github.com/pytorch/pytorch/commit/350cda97a10d843e1df5e556ffe67df5def10fbe), present from 2.13.0.
- [Copy-cast storage-offset handling](https://github.com/pytorch/pytorch/commit/1117a170a53e591311a133660b8b8c0547489734), included in 2.14.0/2.14.1.

The official [Torch release](https://github.com/pytorch/pytorch/releases/tag/v2.14.1)
and [TorchVision compatibility statement](https://github.com/pytorch/vision/releases/tag/v0.29.1)
identify the tested pair. Declared dependency compatibility does not establish
Gemma runtime acceptance; the raw controls above supply the bounded evidence.

[report.json](report.json) separates both gates, all argmax differences and asset
omissions. [runtime-control-evidence.zip](runtime-control-evidence.zip) preserves
the environment requirements/version comparison, probes, raw twenty-request
replay, old/new helper outputs, source snapshots and verified model identities.
Weights and large tokenizer assets are excluded; their verified hashes remain.
Archive CRC and member digests passed. Cost remains unknown. No CUDA job ran.

After extraction, recreate the isolated environment with the archived
`setup/requirements.txt`, then install this repository without inference extras:

```bash
uv venv --python 3.12 artifacts/mps-next-env
uv pip install --python artifacts/mps-next-env/bin/python -r setup/requirements.txt
uv pip install --python artifacts/mps-next-env/bin/python --no-deps -e .
```

The original pinned environment remains the reference. The candidate needs its
own source/runtime/cache lineage and full regression run.
