"""Bounded release workflow. Each invocation runs only its explicitly named stage.

uv run --extra modal modal run apps/modal/release.py --stage pilot --run <name>
Then: --stage train, --stage evaluate, --stage export-results. Use --resume only
for a same-recipe interrupted pilot/train/evaluate, including evaluation ending
with completed_with_errors. Weights stay on the existing Volume.
"""

from __future__ import annotations

import io
import json
import os
import re
import zipfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
DATA = ROOT / "artifacts/release/datasets"
app = modal.App("gemma-s1-release")
cache = modal.Volume.from_name("gemma-unified-system-one", create_if_missing=False)
runtime = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(
        uv_project_dir=ROOT, extras=["inference", "train"], frozen=True, extra_options="--no-dev"
    )
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false", "USE_TF": "0"})
    .add_local_python_source("s1")
    .add_local_file(ROOT / "scripts/release_training.py", "/workspace/scripts/release_training.py")
    .add_local_file(ROOT / "uv.lock", "/workspace/uv.lock")
    .add_local_dir(DATA, "/workspace/artifacts/release/datasets")
)
secrets = (
    [modal.Secret.from_name(os.environ["S1_HF_SECRET"])] if os.environ.get("S1_HF_SECRET") else []
)


def _output(run):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", run):
        raise ValueError("run must be a lowercase name with at most 80 characters")
    return Path("/vol/releases") / run


def _phase(stage, run, resume):
    import sys

    sys.path.insert(0, "/workspace/scripts")
    from release_training import run_phase

    cache.reload()
    return run_phase(stage, DATA, _output(run), commit=cache.commit, resume=resume)


@app.function(
    image=runtime,
    gpu="A100-80GB",
    cpu=4,
    memory=98304,
    volumes={"/vol": cache},
    secrets=secrets,
    timeout=3600,
    max_containers=1,
)
def pilot(run: str, resume: bool = False):
    return _phase("pilot", run, resume)


@app.function(
    image=runtime,
    gpu="A100-80GB",
    cpu=4,
    memory=98304,
    volumes={"/vol": cache},
    secrets=secrets,
    timeout=21600,
    max_containers=1,
)
def train(run: str, resume: bool = False):
    return _phase("train", run, resume)


@app.function(
    image=runtime,
    gpu="A100-80GB",
    cpu=4,
    memory=98304,
    volumes={"/vol": cache},
    secrets=secrets,
    timeout=7200,
    max_containers=1,
)
def evaluate(run: str, resume: bool = False):
    return _phase("evaluate", run, resume)


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={"/vol": cache},
    timeout=600,
    max_containers=1,
)
def export_results(run: str):
    cache.reload()
    root = _output(run)
    if not (root / "manifest.json").is_file():
        raise ValueError("release run does not exist")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".txt", ".md"):
                archive.write(path, path.relative_to(root))
    return buffer.getvalue()


@app.local_entrypoint()
def main(
    stage: str = "pilot",
    run: str = "",
    resume: bool = False,
    output: str = "artifacts/release/runs",
):
    _output(run)
    if stage not in ("pilot", "train", "evaluate", "export-results"):
        raise ValueError("stage must be pilot, train, evaluate or export-results")
    destination = Path(output) / run
    destination.mkdir(parents=True, exist_ok=True)
    if stage == "export-results":
        (destination / "results.zip").write_bytes(export_results.remote(run))
        print(f"Saved {destination / 'results.zip'}", flush=True)
        return
    result = (
        pilot.remote(run, resume)
        if stage == "pilot"
        else train.remote(run, resume)
        if stage == "train"
        else evaluate.remote(run, resume)
    )
    (destination / f"{stage}-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "completed":
        raise RuntimeError(f"{stage} ended with {result['status']}")
