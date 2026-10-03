# Reproduce the trained MLX release

Start from the existing repository root on Apple Silicon. The recorded machine
was a 64 GiB M1 Max. Keep space for the approximately 24 GB BF16 source and 13 GB
MLX export together. The archived evidence is described in [release.md](release.md).

## Pinned runtime and archived evidence

Create a separate detached checkout at the exact recorded runtime revision.
Choose a new sibling directory below; the check rejects an existing destination.
Both pinned commits must already exist in the current repository.

```bash
set -euo pipefail
release_runtime_revision=d4ea1ef79af2e22f666374e529fa56c1de9d3f53
release_evidence_revision=5b3be13c3eacbd17913e5702bf33537886d56d4f
release_worktree=../Gemma-4-12B-Unified-System-One-release-reproduce
test ! -e "$release_worktree"
git worktree add --detach "$release_worktree" "$release_runtime_revision"
git archive --format=tar "$release_evidence_revision" \
  docs/validation/2026-10-02/release-datasets.tar.gz \
  docs/validation/2026-10-02/release/artifacts.json \
  docs/validation/2026-10-02/release/manifest.json \
  docs/validation/2026-10-02/release/evaluate \
  docs/validation/2026-10-02/release/train-receipt.json \
  docs/validation/2026-10-02/release/postmerge-calibration.json \
  | tar -xf - -C "$release_worktree"
cd "$release_worktree"
test "$(git rev-parse HEAD)" = "$release_runtime_revision"
```

The evidence commit supplies only the required archive and original CUDA
receipts, preserving their exact bytes. Runtime sources and the lockfile remain
at `d4ea1ef79af2e22f666374e529fa56c1de9d3f53`. Run every command below from this
new checkout; keep its source files unchanged throughout the reproduction.

## Source and environments

The source must be the **complete trained, merged BF16 checkpoint**, including
processor files and `s1_config.json`, at `artifacts/release/checkpoint`. Hugging
Face publication is pending. The recorded source remains on the existing Modal
volume; with access to that volume, download it as follows:

```bash
uv sync --locked --extra inference --extra train --extra modal
mkdir -p artifacts/release/checkpoint
for name in chat_template.jinja config.json generation_config.json \
  model.safetensors processor_config.json s1_config.json \
  tokenizer.json tokenizer_config.json; do
  uv run --no-sync modal volume get gemma-unified-system-one \
    "/releases/20261002-boolq-release-01/merged/$name" \
    "artifacts/release/checkpoint/$name" || exit 1
done

mkdir -p artifacts/release/datasets
tar -xzf docs/validation/2026-10-02/release-datasets.tar.gz \
  -C artifacts/release/datasets

uv venv --python 3.12 artifacts/mlx-env
uv pip install --python artifacts/mlx-env/bin/python -e . \
  mlx-vlm==0.7.4 mlx==0.32.3 mlx-metal==0.32.3 \
  transformers==5.18.0 huggingface-hub==1.33.0 numpy==2.5.3
```

The eight filenames match the archived artifact receipt. Use these explicit
per-file destinations: a directory download into an existing `checkpoint`
directory creates `checkpoint/merged/`, while a nonexistent destination could
be treated as one output file by concurrent downloads. The dataset archive and
manifest hashes are listed in [release.md](release.md); retain the complete
extracted directory.

## Capture inputs and convert

Input preparation uses the repository's locked Transformers environment. It
captures the exact S1 processor tensors, candidate ordering, gold labels and
source hashes without loading the 12B weights into Torch.

```bash
uv run --no-sync python scripts/prepare_mlx_validation.py \
  --model artifacts/release/checkpoint --split calibration \
  --dataset artifacts/release/datasets/calibration.jsonl \
  --output artifacts/release/mlx-inputs/calibration
uv run --no-sync python scripts/prepare_mlx_validation.py \
  --model artifacts/release/checkpoint --split test \
  --dataset artifacts/release/datasets/test.jsonl \
  --output artifacts/release/mlx-inputs/test
uv run --no-sync python scripts/prepare_mlx_validation.py \
  --model artifacts/release/checkpoint --split test \
  --dataset artifacts/release/datasets/media-test.jsonl \
  --output artifacts/release/mlx-inputs/media

artifacts/mlx-env/bin/python -m mlx_vlm convert \
  --hf-path artifacts/release/checkpoint \
  --mlx-path artifacts/exports/s1-boolq-mlx-8bit \
  -q --q-bits 8 --q-group-size 64
cp artifacts/release/checkpoint/s1_config.json \
  artifacts/exports/s1-boolq-mlx-8bit/s1_config.json
```

Use a fresh export directory. This is the full multimodal MLX-VLM conversion,
with affine 8-bit quantization and group size 64. The three captured populations
contain 256 calibration, 256 fresh test and 52 prior media cases.

Create a conversion receipt from the actual files. `checkpoint_identity` streams
all canonical config/index/weight files; mutable calibration sidecars are hashed
separately. The source is also checked against the archived trained-artifact
receipt, so substituting the Google base checkpoint fails.

```bash
artifacts/mlx-env/bin/python - <<'PY'
import importlib.metadata
import json
import runpy
import subprocess
from pathlib import Path

runtime_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if runtime_revision != "d4ea1ef79af2e22f666374e529fa56c1de9d3f53":
    raise RuntimeError("conversion receipt requires the pinned historical runtime")
source_changes = subprocess.check_output(
    ["git", "status", "--porcelain", "--untracked-files=all", "--",
     "src/s1", "scripts", "uv.lock", "pyproject.toml"], text=True,
)
if source_changes.strip():
    raise RuntimeError("runtime sources or dependency definitions differ from the pinned revision")

helper = runpy.run_path("scripts/prepare_mlx_validation.py")
source = Path("artifacts/release/checkpoint")
export = Path("artifacts/exports/s1-boolq-mlx-8bit")
receipt = json.loads(Path("docs/validation/2026-10-02/release/artifacts.json").read_text())
for name, expected in receipt["merged"].items():
    assert helper["file_sha256"](source / name) == expected, name
before = helper["checkpoint_identity"](source)
after = helper["checkpoint_identity"](export)
manifest = {
    "source_checkpoint_sha256": before["sha256"],
    "source_checkpoint_files": before["files"],
    "source_s1_config_sha256": helper["file_sha256"](source / "s1_config.json"),
    "export_checkpoint_sha256": after["sha256"],
    "files": after["files"],
    "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
    "versions": {name: importlib.metadata.version(name) for name in
                 ("mlx-vlm", "mlx", "transformers", "huggingface-hub", "numpy")},
    "runtime_source_revision": runtime_revision,
}
Path("artifacts/release/mlx-conversion-manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n")
PY
```

## Pin and evaluate

The following records the complete evaluator argv with SHA-256 pins **before**
execution, then invokes it. Pins come from this reproduction's files, rather
than assuming regenerated tensor archives have the historical hashes. The
recorded run used the same paths and evaluator defaults: one warmup on the
first case of each split and one measured forward per case.

```bash
artifacts/mlx-env/bin/python - <<'PY'
import hashlib
import json
import subprocess
from pathlib import Path

root = Path("artifacts/release")
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

argv = ["artifacts/mlx-env/bin/python", "scripts/evaluate_mlx_release.py",
        "--model", "artifacts/exports/s1-boolq-mlx-8bit"]
for role in ("calibration", "test", "media"):
    folder = root / "mlx-inputs" / role
    argv += [f"--{role}-inputs", str(folder),
             f"--{role}-manifest-sha256", sha(folder / "manifest.json")]
for kind, path in (("dataset", root / "datasets/manifest.json"),
                   ("conversion", root / "mlx-conversion-manifest.json")):
    argv += [f"--{kind}-manifest", str(path), f"--{kind}-manifest-sha256", sha(path)]
argv += ["--output", str(root / "mlx-release-evaluation.json")]
(root / "mlx-evaluation-command.json").write_text(json.dumps(argv, indent=2) + "\n")
subprocess.run(argv, check=True)
PY
```

The evaluator verifies source/export identities and complete split coverage,
fits temperature using only the 256 calibration cases, then scores the test and
media sets. It preserves the source sidecar as `base_s1_config.json` and writes
the MLX temperature to `s1_config.json` only after successful evaluation.

## Summarize against the recorded CUDA results

Reconstruct the summary input layout from the archived CUDA evidence:

```bash
mkdir -p artifacts/release/runs/20261002-boolq-release-01/train \
  artifacts/release/runs/20261002-boolq-release-01/merged
cp docs/validation/2026-10-02/release/manifest.json \
  docs/validation/2026-10-02/release/artifacts.json \
  artifacts/release/runs/20261002-boolq-release-01/
cp -R docs/validation/2026-10-02/release/evaluate \
  artifacts/release/runs/20261002-boolq-release-01/
cp docs/validation/2026-10-02/release/train-receipt.json \
  artifacts/release/runs/20261002-boolq-release-01/train/receipt.json
cp docs/validation/2026-10-02/release/postmerge-calibration.json \
  artifacts/release/runs/20261002-boolq-release-01/train/
cp artifacts/release/checkpoint/s1_config.json \
  artifacts/release/runs/20261002-boolq-release-01/merged/
uv run --no-sync python scripts/summarize_release.py \
  --run-dir artifacts/release/runs/20261002-boolq-release-01 \
  --mlx-report artifacts/release/mlx-release-evaluation.json \
  --dataset-dir artifacts/release/datasets --output artifacts/release/summary
```

`summary.json` and `README.md` recompute complete-population metrics and paired
accuracy intervals. Prior-text MLX results remain `not_run`. CUDA latency covers
complete backend requests; MLX latency covers synchronized prepared tensors to
candidate logits, excluding preprocessing, loading and file I/O. Their ratio is
not an end-to-end speedup. These fixed public-data results describe this release;
they do not establish general multimodal quality or production acceptance.
