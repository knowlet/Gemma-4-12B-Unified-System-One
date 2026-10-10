# GGUF output reservation and optimization validation

Source integration preserved local `1bc818b` and incorporated remote `3a183fa`
in merge `eaeecdebf5ddd5926edfaaa460c69cec511bc811`.
The final runtime source is
`6df1e36b814fc925746711b1af0d58d990fe6746`; native compilation uses pristine
llama.cpp `5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd` and its existing shared
libraries. The model is the unchanged published Q8_0/F16 package at
`c418d37a17689fae554f7fc7d78db05ff7d52cfb`.

## Measured result

The prototype changes only `cp.n_outputs_max = 64` relative to the original
source. Final source additionally shares the slot-bound constant with request
validation and reports `max_output_slots`. It retains context 16,384,
batch/microbatch 8,192, GPU layers 99, threads 8, image budget 280, native full
vocabulary projection and physical cache clearing.

| Quantity | Original | Prototype | Original rerun |
| --- | ---: | ---: | ---: |
| Metal compute buffer, MiB | 8,432.00 | 2,192.28 | 8,432.00 |
| Stored successful calls compared | 44 | 44 | 44 |
| Decisions compared | 258 | 258 | 258 |
| Argmax agreement with original | 258/258 | 258/258 | 258/258 |
| Maximum absolute legal-logit difference | 0 | 0 | 0 |
| Maximum absolute calibrated-probability difference | 0 | 0 | 0 |

The compute allocation falls by 6,239.72 MiB (6.09 GiB, 74.00%). This is not peak
RSS, total unified-memory consumption or a deployment memory requirement.
The report includes one end-of-run process RSS sample; those samples vary with
residency and do not establish a reduction in peak memory. Model/KV allocation
settings are unchanged.

There are 13 distinct inputs: eight synthetic runtime cases (29 decisions),
four historical boundary cases (five decisions), and a 64-question case.
Each input receives one warmup. The eight synthetic cases receive three measured
repetitions; boundary/64-question cases receive one. This yields 156 measured
decision comparisons. The additional first, warmup and recovery responses raise
the independently audited total to 258. The three native processes run serially:
original → prototype → original rerun.

| Warm measured native median, ms | Original | Prototype | Original rerun |
| --- | ---: | ---: | ---: |
| Short billing text | 356.975 | 365.846 | 972.426 |
| Synthetic image | 3,084.311 | 1,991.259 | 2,476.629 |
| Synthetic audio | 1,125.835 | 899.670 | 990.355 |
| Synthetic mixed media | 2,733.561 | 2,191.285 | 2,510.757 |
| 8,481-token cross-batch boundary | 62,100.837 | 87,894.936 | 73,068.188 |

Native timing includes projection/decoding and physical cache clearing, excludes
HF preparation, and comes from the persistent process. First-request startup is
separate and is affected by page/kernel caches. Timing variability and the slower
long prototype case prevent an accepted latency or throughput claim. These
requests establish bounded output equivalence, not task accuracy, new
calibration or complete 564-case qualification of the current runtime.

## Evidence

- [Raw A/B/A report](reserve-comparison.json) retains every cold/warm/measured
  response, commands, binary hashes, timing scope and original source revision.
- [Independent comparison audit](equivalence-audit.json) checks all 44 stored
  successful calls and 258 decisions per variant, including first/warmup/recovery.
- Native stderr: [original](baseline.stderr.log), [prototype](output64.stderr.log),
  [original rerun](baseline-repeat.stderr.log). [Progress log](experiment.log)
  records sequential completion.
- Frozen source: [original C++](baseline-s1_gguf.cpp),
  [one-line prototype](output64-s1_gguf.cpp), [final C++](final-s1_gguf.cpp),
  [CMake](CMakeLists.txt), and [executed benchmark text](benchmark-v1.py.txt).
  Text suffixes preserve executed Python source bytes without treating historical
  evidence as a maintained module. The report's original absolute paths identify
  execution locations; these copies preserve bytes under portable names.
- Inputs: [runtime](runtime-inputs.jsonl), [boundary](boundary-inputs.jsonl).
  They contain requests and captured slots/tokens, with no gold labels.
- [Final native smoke](final-native-smoke.json) verifies the final executable,
  rehashes both unchanged weights against the calibrated conversion manifest,
  compares 64-slot and recovery logits exactly with the prototype, and checks a
  valid-in-range 65-slot request receives the metadata-count error. See
  [executed smoke source](final-smoke.py.txt), [native log](final-native-smoke.log)
  and [progress](final-smoke-run.log).
- Real public adapter evaluations at image budgets 140 and 70 succeed: HF patch
  shapes are `[1,140,6912]` and `[1,70,6912]`, with 121 and 64 real image tokens,
  respectively. Their input totals are 209 and 152 tokens; native counts and slots
  match. These are pipeline checks, not image-quality measurements. See
  [140 log](image-budget-140.log) and [70 log](image-budget-70.log).
- [Code validation](code-validation.json), [pytest log](pytest.log),
  [Ruff log](ruff.log), [native build](native-build.log), and
  [package build](package-build.log) record 1,034 passing tests, three optional
  skips, successful lint/format and wheel/source builds.
- [File identities](evidence-files.json) hash these archived files, excluding the
  identity file itself.
- [MLX shape/storage receipt](mlx-sizing.json) checks existing captures and
  packed tensor headers; it records 150 distinct test lengths and the arithmetic
  MLP6 sizing estimate without model inference or conversion.
- [Existing K-quant report audit](k-quant-screening.json) recomputes recorded
  probabilities and accuracy on 256+256+52 cases at source `1bc818b`. Archived
  originals: [Q6_K](Q6_K-historical-evaluation.json),
  [Q5_K_M](Q5_K_M-historical-evaluation.json),
  [Q4_K_M](Q4_K_M-historical-evaluation.json). These are historical receipts,
  not executions of the newly optimized runtime; K-quant weight payloads and
  dependency bytes were not rehashed in this arithmetic audit.

The original benchmark's invalid 65 request also appended an out-of-range slot.
All recorded responses identify the metadata-count error, but rejection alone
would not prove the slot limit. The final smoke instead inserts the 65th slot
between two existing positions: all 65 positions are unique, ordered and inside
the original token count. The future reproducer uses this corrected input.

## Reproduce the bounded comparison

Use this checkout containing the optimization evidence and source commit
`6df1e36b814fc925746711b1af0d58d990fe6746`. Install its frozen inference
environment. Reuse only the package-download and pinned shared-library build
steps from [the model guide](../../../gguf-release.md), placing the package in
`artifacts/exports/s1-boolq-gguf`. Keep this checkout; the historical guide's
`bcf829` runtime checkout is not the optimized source. The commands below build
both archived C++ sources in isolated directories. Neither experiment needs the
source BF16 weights.

```sh
mkdir -p artifacts/research/gguf-memory/baseline-src
cp docs/validation/2026-10-07/gguf-memory/CMakeLists.txt artifacts/research/gguf-memory/baseline-src/
cp docs/validation/2026-10-07/gguf-memory/baseline-s1_gguf.cpp artifacts/research/gguf-memory/baseline-src/s1_gguf.cpp
cmake -S artifacts/research/gguf-memory/baseline-src -B artifacts/research/gguf-memory/baseline-build \
  -DLLAMA_CPP_DIR="$PWD/artifacts/export-tools/llama.cpp" -DCMAKE_BUILD_TYPE=Release
cmake --build artifacts/research/gguf-memory/baseline-build -j4
mkdir -p artifacts/research/gguf-memory/current-src
cp docs/validation/2026-10-07/gguf-memory/CMakeLists.txt artifacts/research/gguf-memory/current-src/
cp docs/validation/2026-10-07/gguf-memory/final-s1_gguf.cpp artifacts/research/gguf-memory/current-src/s1_gguf.cpp
cmake -S artifacts/research/gguf-memory/current-src -B artifacts/research/gguf-memory/current-build \
  -DLLAMA_CPP_DIR="$PWD/artifacts/export-tools/llama.cpp" -DCMAKE_BUILD_TYPE=Release
cmake --build artifacts/research/gguf-memory/current-build -j4
.venv/bin/python docs/validation/2026-10-07/gguf-memory/benchmark-reproduce.py \
  --variant baseline=artifacts/research/gguf-memory/baseline-build/bin/s1-gguf \
  --variant current=artifacts/research/gguf-memory/current-build/bin/s1-gguf \
  --variant baseline-repeat=artifacts/research/gguf-memory/baseline-build/bin/s1-gguf \
  --output artifacts/research/gguf-memory/reproduced.json
```

[The corrected reproducer](benchmark-reproduce.py) reads archived inputs, checks
the 65-slot error specifically, and saves a checkpoint after each completed
variant. It has been syntax/lint/CLI checked; the recorded experiment used the
separately archived v1 program. The final smoke executed the corrected boundary.
Keep the machine otherwise idle and repeat measurements before making a latency
decision. Raw result paths and hashes are receipts, not commands to execute.
