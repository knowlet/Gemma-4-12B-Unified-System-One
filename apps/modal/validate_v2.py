"""Bounded, ephemeral real-checkpoint validation; never deploys a service.

Prepare data first, then invoke each stage separately so receipts survive failures:
modal run apps/modal/validate_v2.py --stage checkpoint --run <unique-name>
"""

import io
import json
import time
import traceback
import zipfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
DATA = ROOT / "artifacts/live-validation/datasets"
MODEL = "google/gemma-4-12B-it"
REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
app = modal.App("gemma-v2-live-validation")
cache = modal.Volume.from_name("gemma-unified-system-one", create_if_missing=False)
runtime = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("tesseract-ocr")
    .uv_sync(
        uv_project_dir=ROOT,
        extras=["inference", "train", "api", "laya", "baselines"],
        frozen=True,
    )
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false", "USE_TF": "0"})
    .add_local_python_source("s1")
    .add_local_dir(ROOT / "scripts", "/workspace/scripts")
    .add_local_dir(ROOT / "configs", "/workspace/configs")
    .add_local_dir(ROOT / "examples", "/workspace/examples")
    .add_local_dir(DATA, "/workspace/artifacts/live-validation/datasets")
    .add_local_file(ROOT / "uv.lock", "/workspace/uv.lock")
    .add_local_dir(ROOT / "tests", "/workspace/tests")
)


def collect(stage, run, operation):
    import importlib.metadata
    import platform

    import torch

    if not run or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run):
        raise ValueError("run must be a unique lowercase name")
    output = Path("/vol/runs") / run / stage
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    receipt = {
        "stage": stage,
        "started_at_unix": started,
        "campaign_checkpoint": {"model_id": MODEL, "revision": REVISION},
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "laya", "setfit")
        },
        "modal_sdk": modal.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory
        if torch.cuda.is_available()
        else None,
        "scope": "public-data screening and real runtime validation; no production deployment",
    }
    try:
        receipt["result"] = operation(output / "checks")
        result = receipt["result"]
        receipt["status"] = (
            "completed_with_failures"
            if result.get("failures")
            or result.get("status")
            in (
                "failed",
                "completed_with_failures",
                "completed_with_errors",
                "partial",
                "not_run",
                "setup_error",
                "interrupted",
            )
            else "completed"
        )
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        (output / "failure.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    receipt["elapsed_seconds"] = time.time() - started
    if torch.cuda.is_available():
        receipt["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        receipt["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False))
    cache.commit()
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".txt", ".csv", ".md"):
                archive.write(path, path.relative_to(output))
    return {"receipt": receipt, "archive": bundle.getvalue()}


@app.function(
    image=runtime,
    gpu="A100-80GB",
    cpu=4,
    memory=49152,
    volumes={"/vol": cache},
    timeout=3600,
    max_containers=1,
)
def checkpoint(run):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from live_checkpoint_checks import run as check

    from s1.unified import UnifiedDecisionModel

    def operation(output):
        model = UnifiedDecisionModel(MODEL, revision=REVISION, device="cuda")
        return check(model, ROOT, output, DATA)

    return collect("checkpoint", run, operation)


@app.function(
    image=runtime,
    gpu="A100-80GB",
    cpu=4,
    memory=49152,
    volumes={"/vol": cache},
    timeout=5400,
    max_containers=1,
)
def training(run):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from live_training_checks import run as check

    return collect("training", run, lambda output: check(ROOT, output, DATA, MODEL, REVISION))


@app.function(
    image=runtime,
    gpu="A100-80GB",
    cpu=4,
    memory=49152,
    volumes={"/vol": cache},
    timeout=7200,
    max_containers=1,
)
def services(run):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from live_services_checks import run as check

    return collect("services", run, lambda output: check(ROOT, output, DATA, MODEL, REVISION))


@app.function(
    image=runtime,
    cpu=4,
    memory=16384,
    volumes={"/vol": cache},
    timeout=1800,
    max_containers=1,
)
def baselines(run):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from live_baseline_checks import run as check

    return collect("baselines", run, lambda output: check(ROOT, output, DATA))


@app.function(
    image=runtime,
    gpu="A100-80GB",
    cpu=4,
    memory=49152,
    volumes={"/vol": cache},
    timeout=1200,
    max_containers=1,
)
def diagnostics(run):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from live_batch_diagnostics import run as check

    from s1.unified import UnifiedDecisionModel

    def operation(output):
        model = UnifiedDecisionModel(MODEL, revision=REVISION, device="cuda")
        return check(model, ROOT, output, DATA)

    return collect("diagnostics", run, operation)


@app.function(
    image=runtime,
    gpu="A100-80GB",
    cpu=4,
    memory=49152,
    volumes={"/vol": cache},
    timeout=1800,
    max_containers=1,
)
def checkpoint_fp32(run):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from live_checkpoint_checks import run as check

    from s1.unified import UnifiedDecisionModel

    def operation(output):
        model = UnifiedDecisionModel(MODEL, revision=REVISION, device="cuda", precision="float32")
        return check(model, ROOT, output, DATA)

    return collect("checkpoint-fp32", run, operation)


@app.function(
    image=runtime,
    gpu="A100-80GB",
    cpu=4,
    memory=49152,
    volumes={"/vol": cache},
    timeout=5400,
    max_containers=1,
)
def training_full(run):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from live_training_checks import run as check

    return collect(
        "training-full",
        run,
        lambda output: check(
            ROOT,
            output,
            DATA,
            MODEL,
            REVISION,
            shots=(8, 32, 128),
            steps=100,
            train_filename="text-train-full.jsonl",
        ),
    )


@app.function(
    image=runtime,
    gpu="A100-80GB",
    cpu=4,
    memory=49152,
    volumes={"/vol": cache},
    timeout=600,
    max_containers=1,
)
def stress(run):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from live_services_checks import _load

    from s1.unified import UnifiedDecisionModel

    def operation(output):
        output.mkdir(parents=True, exist_ok=False)
        model = UnifiedDecisionModel(MODEL, revision=REVISION, device="cuda")
        return _load(model, output, stress=True)

    return collect("stress", run, operation)


@app.function(
    image=runtime,
    cpu=4,
    memory=16384,
    volumes={"/vol": cache},
    timeout=3600,
    max_containers=1,
)
def baselines_full(run):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from live_baseline_checks import run_curve

    return collect("baselines-full", run, lambda output: run_curve(ROOT, output, DATA))


@app.function(
    image=runtime,
    gpu="A100-80GB",
    cpu=4,
    memory=49152,
    volumes={"/vol": cache},
    timeout=900,
    max_containers=1,
)
def integrity(run, training_run):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from live_integrity_checks import run as check

    from s1.unified import UnifiedDecisionModel

    if not training_run or any(
        c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in training_run
    ):
        raise ValueError("training_run must identify a completed training-full run")

    def operation(output):
        model = UnifiedDecisionModel(MODEL, revision=REVISION, device="cuda", precision="float32")
        training_dir = Path("/vol/runs") / training_run / "training-full/checks"
        return check(model, ROOT, output, DATA, training_dir)

    return collect("integrity", run, operation)


@app.local_entrypoint()
def main(
    stage: str = "checkpoint",
    run: str = "",
    output: str = "artifacts/live-validation",
    training_run: str = "",
):
    functions = {
        "checkpoint": checkpoint,
        "checkpoint-fp32": checkpoint_fp32,
        "training": training,
        "services": services,
        "baselines": baselines,
        "diagnostics": diagnostics,
        "training-full": training_full,
        "stress": stress,
        "baselines-full": baselines_full,
        "integrity": integrity,
    }
    if stage not in functions:
        raise ValueError(f"stage must be one of {sorted(functions)}")
    result = (
        integrity.remote(run, training_run)
        if stage == "integrity"
        else functions[stage].remote(run)
    )
    destination = Path(output) / run / stage
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(result["archive"])) as archive:
        archive.extractall(destination)
    print(
        json.dumps(
            {key: value for key, value in result["receipt"].items() if key != "result"},
            indent=2,
            allow_nan=False,
        )
    )
    if result["receipt"]["status"] != "completed":
        raise RuntimeError(f"{stage} failed; receipt saved in {destination}")
