# GGUF Q8_0 release and native S1 runtime

The trained BF16 checkpoint has been converted to a Q8_0 language-model GGUF
and an F16 vision/audio projector. Twelve real native Metal smoke cases passed,
followed by successful evaluation of all **256 calibration + 256 fresh BoolQ +
52 media cases**. GGUF scored **231/256 (90.234375%)** on fresh BoolQ and
**33/52 (63.461538%)** on the historical media set after independent calibration.
The complete [GGUF package](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-GGUF/tree/c418d37a17689fae554f7fc7d78db05ff7d52cfb)
was published on October 4, with matching remote file identities and verified
anonymous access.

The previously published [BF16 package](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One/tree/a66f836b56605039fe040f330180e336d19b3362)
and [MLX 8-bit package](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-MLX-8bit/tree/a5f89b400f7ef63e162f22866c206ed67cf8f282)
retain their own [release measurements](release.md) and
[publication evidence](validation/2026-10-04/README.md).

## Recorded conversion

| Item | Recorded value |
| --- | --- |
| Trained BF16 source weights | `knowlet/Gemma-4-12B-Unified-System-One` at `66626de5cbcf8c5fecb8e9b58710805a02a42f67` |
| llama.cpp converter/runtime | `5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd` |
| Language-model file | `s1-boolq-Q8_0.gguf`, 12,669,647,264 bytes |
| Language-model SHA-256 | `9d3bc61851e5b7ed8e595b84327b85271830778d0e62fa9c1349c61843f369ed` |
| Vision/audio projector | `mmproj-s1-boolq-f16.gguf`, 122,031,744 bytes |
| Projector SHA-256 | `cffbff05efd290fef28d084ebcb653de7594f5e26bae0eabbedf30d6c782c1d1` |
| Combined GGUF file size | 12,791,679,008 bytes; this is not measured runtime memory |
| GGUF adapter/evaluator source | `bcf829acc111fd8054987b1519b8181835f1dfed` |
| Recorded conversion receipt SHA-256 | `464874bc51e8ee21ab56f34a4bf7d22b1fb83299a277cbcf966b42c5fc7802b7` |

The [conversion receipt](validation/2026-10-04/gguf/conversion-manifest.json), also
saved at `artifacts/exports/s1-boolq-gguf/conversion-manifest.json`, was recorded
at the frozen source revision above with status `converted_and_structurally_verified`.
Structural checks cover the language architecture, proportional RoPE factors and
both native vision/audio tensor groups. Processor, tokenizer and chat-template
files are retained. `base_s1_config.json` preserves the trained BF16 sidecar; it
is not the GGUF calibration result.

The historical `d4ea1ef79af2e22f666374e529fa56c1de9d3f53` source revision belongs
to BF16/MLX inference and evaluation. It does not contain this new native GGUF
adapter. Use the separate frozen GGUF source revision above for these commands.

## Download the published package

Use a clean checkout at `bcf829acc111fd8054987b1519b8181835f1dfed` and install
the locked inference dependencies. Download the complete published package at
the final immutable revision; this path needs no BF16 model weights:

```bash
uv sync --locked --extra inference
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

package = Path(snapshot_download(
    repo_id="knowlet/Gemma-4-12B-Unified-System-One-GGUF",
    revision="c418d37a17689fae554f7fc7d78db05ff7d52cfb",
    local_dir="artifacts/exports/s1-boolq-gguf",
))
manifest_path = package / "artifact-manifest.json"
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
    "1646a81f062fa3fe5727c835de569b669f288bd646749b2c3b96426bff4bf8f0"
)
for entry in json.loads(manifest_path.read_text())["files"]:
    path = package / entry["path"]
    assert path.stat().st_size == entry["size_bytes"], path
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"], path
print(package)
PY
```

Build the pinned native libraries and helper as shown below, then run the
[calibrated decision command](#live-typed-predictions-after-calibration). The
[published model card](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-GGUF/blob/c418d37a17689fae554f7fc7d78db05ff7d52cfb/README.md)
includes a library-only build recipe independently reproduced in a pristine
checkout; its [build receipt](validation/2026-10-04/gguf/build-reproduction.json)
records successful library/helper compilation and `--help`, with no extra
model-quality measurements.

## Native decision behavior

The adapter retains all unfilled answer slots in one ordered S1 input sequence.
It never samples, appends or feeds generated answers back into later questions.
`decision_passes: 1` means one S1 decision evaluation; text batches and media
chunks can require several native prefill/decode calls, reported separately in
`preprocessing.decode_calls`. The KV cache is cleared between requests.

llama.cpp projects the **full vocabulary** at each requested answer position.
The helper selects the legal candidate logits, then the Python adapter applies
the GGUF model's own temperature and candidate softmax. Native logit softcapping
is applied once. This path does not provide the selected-LM-head-row compute
savings of the PyTorch/MLX implementation.

The saved HF processor constructs the authoritative prefix, text/control token
IDs, media ordering and slot positions on CPU, without loading HF model weights.
Images use the exact processed RGB grid reconstructed from HF patches; the
uint8 roundtrip is checked before native `mmproj` inference. Audio uses normalized
16 kHz float32 samples and native 640-sample framing. Media embeddings and
execution come from llama.cpp/libmtmd, not replayed HF image/audio embedding
tensors. Matching prefix and token counts establishes structure, not numerical
media parity. See the [native adapter contract](../tools/gguf/README.md).

## Reproduce the conversion

Use Apple Silicon with CMake, a C++ compiler and sufficient space for the BF16
source plus converted files. Start from the repository root at the final GGUF
runtime commit. Use fresh output/tool directories; keep the pinned runtime
sources unchanged while evaluating.

```bash
set -euo pipefail
gguf_runtime_revision=bcf829acc111fd8054987b1519b8181835f1dfed
test "$(git rev-parse HEAD)" = "$gguf_runtime_revision"
uv sync --locked --extra inference --extra train
```

The commands below use that revision's locked conversion environment and the
complete trained source at its immutable Hub weight revision:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="knowlet/Gemma-4-12B-Unified-System-One",
    revision="66626de5cbcf8c5fecb8e9b58710805a02a42f67",
    local_dir="artifacts/release/checkpoint",
    allow_patterns=[
        "chat_template.jinja", "config.json", "generation_config.json",
        "model.safetensors", "processor_config.json", "s1_config.json",
        "tokenizer.json", "tokenizer_config.json",
    ],
    token=False,
)
PY
```

Build the pinned shared native libraries with Metal enabled and CUDA disabled,
then build this repository's helper against them:

```bash
mkdir -p artifacts/export-tools
git clone https://github.com/ggml-org/llama.cpp artifacts/export-tools/llama.cpp
git -C artifacts/export-tools/llama.cpp checkout 5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd
cmake -S artifacts/export-tools/llama.cpp -B artifacts/export-tools/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGGML_METAL=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF
cmake --build artifacts/export-tools/llama.cpp/build \
  --target llama mtmd ggml ggml-base -j 4
cmake -S tools/gguf -B artifacts/gguf-runtime \
  -DLLAMA_CPP_DIR="$PWD/artifacts/export-tools/llama.cpp"
cmake --build artifacts/gguf-runtime -j 4
```

The helper's CMake configuration checks the exact llama.cpp revision and rejects
tracked source modifications. The recorded conversion directly emits Q8_0 and
F16 files; it does not require an intermediate BF16 GGUF or a Q4_K_M quantizer run.

```bash
test ! -e artifacts/exports/s1-boolq-gguf
mkdir -p artifacts/exports/s1-boolq-gguf
.venv/bin/python artifacts/export-tools/llama.cpp/convert_hf_to_gguf.py \
  artifacts/release/checkpoint --outtype q8_0 \
  --outfile artifacts/exports/s1-boolq-gguf/s1-boolq-Q8_0.gguf \
  --model-name Gemma-4-12B-Unified-System-One
.venv/bin/python artifacts/export-tools/llama.cpp/convert_hf_to_gguf.py \
  artifacts/release/checkpoint --mmproj --outtype f16 \
  --outfile artifacts/exports/s1-boolq-gguf/mmproj-s1-boolq-f16.gguf
.venv/bin/python scripts/record_gguf_conversion.py \
  --source artifacts/release/checkpoint \
  --model artifacts/exports/s1-boolq-gguf \
  --llama-cpp artifacts/export-tools/llama.cpp \
  --source-capture docs/validation/2026-10-02/release/input-manifests/calibration.json \
  --source-capture-sha256 90e9fc465a6e372c81e1596e387fc7c62107730b41c4f3c1d5c7a8087a9bb51e
```

The recorder checks the original source checkpoint and BF16 sidecar against the
pinned historical capture manifest. It copies the processor/tokenizer/template
files and saves the unchanged source sidecar as `base_s1_config.json`, then writes
the tensor inventory and conversion receipt. It does not create a fitted GGUF
`s1_config.json`. A reproduction writes its own receipt with a new recording
timestamp; the evaluator below pins that receipt's actual hash rather than
requiring the historical receipt hash in the table.

## Frozen population and independent calibration

Use the original [frozen dataset archive](validation/2026-10-02/release-datasets.tar.gz).
The three input captures retain case IDs, gold labels, candidate order, source
identity and exact processor inputs. These commands prepare inputs on CPU:

```bash
mkdir -p artifacts/release/datasets
tar -xzf docs/validation/2026-10-02/release-datasets.tar.gz \
  -C artifacts/release/datasets
.venv/bin/python scripts/prepare_mlx_validation.py \
  --model artifacts/release/checkpoint --split calibration \
  --dataset artifacts/release/datasets/calibration.jsonl \
  --output artifacts/release/mlx-inputs/calibration
.venv/bin/python scripts/prepare_mlx_validation.py \
  --model artifacts/release/checkpoint --split test \
  --dataset artifacts/release/datasets/test.jsonl \
  --output artifacts/release/mlx-inputs/test
.venv/bin/python scripts/prepare_mlx_validation.py \
  --model artifacts/release/checkpoint --split test \
  --dataset artifacts/release/datasets/media-test.jsonl \
  --output artifacts/release/mlx-inputs/media
```

The capture helper's filename predates GGUF. The native evaluator uses its case,
source and slot contract while preparing native media through the adapter; the
captured HF media tensors are not substituted for native GGUF projection.

The finalized evaluator command below hashes this reproduction's manifests,
records its complete arguments before execution and binds the native build.
Regenerated capture archives need not share the historical NPZ byte hashes.
Only the 256 calibration cases fit temperature; test and media labels cannot
select the temperature. The completed results are reported below, with the
exact [evaluation report](validation/2026-10-04/gguf/evaluation.json).

```bash
.venv/bin/python - <<'PY'
import hashlib
import json
import subprocess
from pathlib import Path

runtime_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if runtime_revision != "bcf829acc111fd8054987b1519b8181835f1dfed":
    raise RuntimeError("use the frozen GGUF runtime revision")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

root = Path("artifacts/release")
package = Path("artifacts/exports/s1-boolq-gguf")
argv = [".venv/bin/python", "scripts/evaluate_gguf_release.py", "--model", str(package)]
for role in ("calibration", "test", "media"):
    folder = root / "mlx-inputs" / role
    argv += [f"--{role}-inputs", str(folder),
             f"--{role}-manifest-sha256", sha(folder / "manifest.json")]
for kind, path in (("dataset", root / "datasets/manifest.json"),
                   ("conversion", package / "conversion-manifest.json")):
    argv += [f"--{kind}-manifest", str(path), f"--{kind}-manifest-sha256", sha(path)]
argv += ["--runtime-source-revision", runtime_revision]
runtime_files = {
    Path("tools/gguf/s1_gguf.cpp").resolve(),
    Path("tools/gguf/CMakeLists.txt").resolve(),
    Path("artifacts/gguf-runtime/CMakeCache.txt").resolve(),
    Path("artifacts/gguf-runtime/bin/s1-gguf").resolve(),
}
runtime_files.update(
    path.resolve() for path in Path("artifacts/export-tools/llama.cpp/build/bin").glob("*.dylib")
)
for path in sorted(runtime_files):
    argv += ["--runtime-file", str(path)]
argv += ["--warmup", "1", "--repeat", "1", "--context-size", "16384",
         "--adapter-timeout", "300", "--output", str(root / "gguf-release-evaluation.json")]
argv += ["--adapter-command", ".venv/bin/python", "scripts/gguf_adapter.py",
         "--model", str(package), "--native-runner", "artifacts/gguf-runtime/bin/s1-gguf",
         "--evaluation"]
(root / "gguf-evaluation-command.json").write_text(json.dumps(argv, indent=2) + "\n")
subprocess.run(argv, check=True)
PY
```

`--adapter-command` must be last. The evaluator also discovers the bundled
adapter's native files and checks its Python sources against the committed
runtime. Keep HEAD and implementation files unchanged during execution. A
successful full-population run writes the GGUF-specific `s1_config.json`,
including its fitted temperature and conversion-manifest identity, while
preserving `base_s1_config.json`. Failed evaluation records an error report and
does not write a new calibrated sidecar.

## Live typed predictions after calibration

The published package already contains its fitted GGUF calibration sidecar.
For a new conversion, first run full evaluation to write a matching sidecar.
Default prediction mode rejects a copied BF16 or MLX temperature. Supply one
public `DecisionRequest` per JSON line. For the repository's example request:

```bash
mkdir -p artifacts/release
.venv/bin/python - <<'PY' > artifacts/release/gguf-request.jsonl
import json
from pathlib import Path
print(json.dumps(json.loads(Path("examples/request.json").read_text())))
PY
.venv/bin/python scripts/gguf_adapter.py \
  --model artifacts/exports/s1-boolq-gguf \
  --native-runner artifacts/gguf-runtime/bin/s1-gguf \
  < artifacts/release/gguf-request.jsonl
```

The adapter returns typed Choice/Noul/Score answers. Its default native settings
request Metal/GPU offload, a 16,384-token context and an 8,192-token batch and
microbatch. Actual offloading appears in native stderr. `--gpu-layers 0` selects
CPU; the recorded smoke and formal evaluations used Metal. Context overflow, unsupported
layouts, missing modalities and token-count mismatches produce explicit errors.

## Recorded smoke checks

All twelve real Metal smoke cases completed on Apple M1 Max. Eight cases cover
text, Score, image, audio, mixed media, long state and sixteen questions. Four
additional boundary cases include an 8,481-token two-slot request that crosses
the 8,192-token batch boundary. Its first-slot logits exactly matched the
identical 49-token prefix-only request in that measured check; this is not a
general numerical-parity claim. The native log records all 49/49 model layers
offloaded to Metal.

The [GGUF evidence directory](validation/2026-10-04/gguf/) retains the conversion,
smoke and formal evaluation records. The local smoke outputs were
`artifacts/release/gguf-runtime-smoke-output.jsonl`,
`artifacts/release/gguf-runtime-smoke.log` and
`artifacts/release/gguf-boundary-smoke-receipt.json`, with the corresponding
boundary inputs, outputs and log. These checks did not fit temperature or
measure full-population accuracy.
The boundary receipt predates regeneration of the final conversion metadata.
Its recorded source and native-binary hashes match the frozen implementation;
use the final conversion receipt and full evaluation for current package
identity and quality evidence.

After calibration, the public `DecisionRequest` interface also passed all
**8 synthetic requests and 29 typed answers**: 22 Choice, 6 Noul and 1 Score.
The receipt, inputs, outputs and native log are retained in the same evidence
directory, with the receipt also at
`artifacts/release/gguf-calibrated-smoke-receipt.json`. This check used the fitted
GGUF sidecar and verifies interface execution; it is not task-accuracy evidence.

## Completed calibration and full-population evaluation

The [evaluation report](validation/2026-10-04/gguf/evaluation.json) has status
`ok`, `population_complete: true` and `release_validation_complete: true`.
All **564 cases and 564 decisions** succeeded, with 100% coverage in each split.
The report SHA-256 is
`c812265d0f4b62cbf69a774c728c0c59d1912f259993ca78c38a04f51bfa5fbb`.

The GGUF temperature was independently fitted to its own post-softcap raw legal
logits from all 256 calibration cases. The fitter searches 241 log-spaced values
in `[0.05, 20]`, plus `1`. It selected **2.7830344470383452**, reducing calibration
NLL from **0.5385953545 to 0.2904123939**. This is an in-sample fitting result.
The fitted value happens to equal the source BF16 temperature numerically; the
GGUF fit used its own observations and did not copy the BF16 sidecar's value.
The resulting GGUF sidecar SHA-256 is
`24882a49f5440412a63aecb7e446274d4ba6a4f8bb28930ddc4eaf1922d2863c`.

| Evaluation population | Correct / scored | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh BoolQ test | 231 / 256 | 90.234375% | 0.272152 | 0.156864 | 0.046608 |
| Historical native media | 33 / 52 | 63.461538% | 1.111841 | 0.436341 | 0.192256 |
| MNIST images, media subset | 28 / 32 | 87.500000% | 0.521211 | 0.208580 | 0.207535 |
| FSDD audio, media subset | 5 / 20 | 25.000000% | 2.056850 | 0.800759 | 0.199324 |

NLL uses a `1e-12` probability floor, Brier sums squared error across legal
candidates, and ECE uses 15 equal-width bins. The image and audio rows partition
the 52 media cases; they are not additional cases. These are descriptive single-seed
results on fixed public-data populations. The historical media cases already
informed earlier experiments, and BoolQ-only calibration does not establish
image/audio calibration or broad multimodal quality.

## Cross-runtime comparison

The [independent audit](validation/2026-10-04/gguf/cross-runtime-comparison.json)
recomputed calibration and metrics, checked complete case/slot/candidate
alignment, and compared independently calibrated probabilities on the same
frozen populations. Probability differences below are the largest absolute
difference for any legal candidate, expressed in percentage points.

| Reference → GGUF | Population | Argmax agreement | Max probability difference | Wrong → correct | Correct → wrong |
| --- | --- | ---: | ---: | ---: | ---: |
| Trained BF16 → GGUF | Fresh BoolQ | 255 / 256 | 32.89 pp | 1 | 0 |
| Trained BF16 → GGUF | Historical media | 47 / 52 | 43.27 pp | 1 | 2 |
| Trained MLX 8-bit → GGUF | Fresh BoolQ | 255 / 256 | 9.13 pp | 1 | 0 |
| Trained MLX 8-bit → GGUF | Historical media | 46 / 52 | 35.10 pp | 2 | 3 |

Both fresh-test comparisons change accuracy from 230/256 to 231/256:
**GGUF minus reference = +0.390625 percentage points**, with a paired 95%
percentile-bootstrap interval of **[0, +1.171875] percentage points**. The audit
uses 256 source groups, 10,000 resamples and seed 42. The interval includes zero,
so these measurements do not establish an accuracy improvement.

Media accuracy changes from 34/52 for each reference to 33/52 for GGUF. Besides
the correctness changes in the table, two BF16 comparisons and one MLX
comparison change from one wrong label to another. Native media encoding and
runtime execution differ, and each runtime uses its own calibration. These
comparisons do not isolate quantization effects or establish global numerical
parity. The 5/20 audio result is a material limitation of this release.

## Hardware and timing

The full evaluation ran on **Apple M1 Max, 64 GiB unified memory, macOS 27.0.1**,
with all **49/49 model layers offloaded to Metal**. The 64 GiB figure is installed
memory capacity; the report does not establish a peak-memory requirement.

| Population | Measured calls | Backend-call p50 | Backend-call p95 |
| --- | ---: | ---: | ---: |
| Calibration | 256 | 1,002.79 ms | 1,752.95 ms |
| Fresh BoolQ test | 256 | 911.63 ms | 1,601.15 ms |
| Historical media | 52 | 2,345.01 ms | 2,395.05 ms |

`backend_call_latency_ms` covers persistent adapter request serialization,
transport, HF preprocessing and native decoding through the raw-logit response.
It excludes captured-tensor file I/O, NumPy softmax and temperature fitting.
The command uses one first-case warmup per split and one measured call per case;
first-call loading is excluded only when warmup is enabled. This scope differs
from CUDA complete-backend timing and MLX prepared-tensor timing, so ratios do
not establish end-to-end speedups. Smoke timings are not the formal latency
distribution.

## Publication

The public [GGUF repository](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-GGUF)
contains the Q8_0 model, F16 projector, processor files, calibrated sidecar,
evaluation, cross-runtime comparison and provenance. The core weights/files
revision is `5940bdbe292b33ac86450093a528d39b13c3a8f4`; the final model card and
manifest revision is `c418d37a17689fae554f7fc7d78db05ff7d52cfb`.
The final manifest SHA-256 is
`1646a81f062fa3fe5727c835de569b669f288bd646749b2c3b96426bff4bf8f0`.

The [publication receipt](validation/2026-10-04/gguf/hub-gguf-publication.json)
records all 17 packaged files verified against immutable remote LFS SHA-256/size
or Git blob identities. Anonymous resolve requests verified both GGUF weight
files, and the public model card, manifest and GGUF sidecar were downloaded and
hash-checked. This verification did not redownload the complete 12.79 GB weights
or run remote inference. The [evidence index](validation/2026-10-04/gguf/README.md)
retains exact reports and the published card/manifest. Public availability does
not establish global BF16 parity or production suitability.
