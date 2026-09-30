"""Locked Modal smoke tests, model benchmarks, training and authenticated serving.

Run from the repository root with `uv run --extra modal modal run ...`.
"""

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
model_name = os.environ.get("S1_MODEL") or "google/gemma-4-12B-it"
revision = os.environ.get("S1_REVISION") or None
app = modal.App("gemma-unified-system-one")
cache = modal.Volume.from_name("gemma-unified-system-one", create_if_missing=True)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=ROOT, extras=["api"], frozen=True)
    .add_local_python_source("s1")
    .add_local_dir(ROOT / "tests", "/workspace/tests")
    .add_local_dir(ROOT / "examples", "/workspace/examples")
)
runtime = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(
        uv_project_dir=ROOT,
        extras=["inference", "train", "api", "laya"],
        frozen=True,
        extra_options="--no-dev",
    )
    .env(
        {
            "HF_HOME": "/vol/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "USE_TF": "0",
            "S1_MODEL": model_name,
            "S1_REVISION": revision or "",
        }
    )
    .add_local_python_source("s1")
)
gpu = os.environ.get("S1_GPU", "A100-40GB")
# Optional user-configured Hugging Face secret; no local credentials are uploaded.
hf_secrets = (
    [modal.Secret.from_name(os.environ["S1_HF_SECRET"])] if os.environ.get("S1_HF_SECRET") else []
)


@app.function(image=cpu_image, timeout=180, cpu=2)
def smoke():
    import subprocess

    result = subprocess.run(
        ["python", "-m", "pytest", "-q", "tests"], cwd="/workspace", capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return {"exit_code": result.returncode, "output": result.stdout}


@app.function(
    image=runtime,
    gpu=gpu,
    volumes={"/vol": cache},
    secrets=hf_secrets,
    timeout=1200,
    max_containers=1,
)
def model_smoke():
    import base64
    import io

    from PIL import Image

    from s1.contracts import DecisionRequest
    from s1.unified import UnifiedDecisionModel

    model = UnifiedDecisionModel(model_name, revision=revision)
    image = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(image, format="PNG")
    questions = {
        "color": {
            "type": "choice",
            "instructions": "Which color is mentioned or shown?",
            "criteria": {"red": "red", "blue": "blue"},
        },
        "sound": {"type": "noul", "instructions": "Is speech present?"},
    }
    results = {}
    for name, media in (
        ("text", []),
        ("image", [{"type": "image", "data": base64.b64encode(image.getvalue()).decode()}]),
        ("audio", [{"type": "audio", "samples": [0.0] * 1600}]),
    ):
        request = DecisionRequest(state="The object is red.", questions=questions, media=media)
        results[name] = model.predict(request)
    cache.commit()
    return {"model": model.name, "revision": model.revision, "results": results}


@app.function(
    image=runtime,
    gpu=gpu,
    volumes={"/vol": cache},
    secrets=hf_secrets,
    timeout=1200,
    max_containers=1,
)
def benchmark(rows: list[dict], backend: str = "laya"):
    from s1.backends import GemmaBackend, LayaBackend
    from s1.benchmark import Case, evaluate

    if backend not in ("gemma", "laya"):
        raise ValueError("backend must be gemma or laya")
    runner = (
        GemmaBackend(model_name, revision=revision)
        if backend == "gemma"
        else LayaBackend(device="cuda")
    )
    result = evaluate(runner, [Case.model_validate(row) for row in rows], warmup=1)
    cache.commit()
    return result


@app.function(
    image=runtime,
    gpu=gpu,
    volumes={"/vol": cache},
    secrets=hf_secrets,
    timeout=3600,
    max_containers=1,
)
def train(rows: list[dict], calibration: list[dict], steps: int = 100, run: str = "unified-lora"):
    import torch

    from s1.benchmark import Case, write_report
    from s1.training import train_model
    from s1.unified import UnifiedDecisionModel

    if not run or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run):
        raise ValueError("run must be a lowercase name, not a path")
    torch.manual_seed(0)
    model = UnifiedDecisionModel(
        model_name,
        revision=revision,
        lora={
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
        },
    )
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    result = train_model(
        model,
        [Case.model_validate(r) for r in rows],
        [Case.model_validate(r) for r in calibration],
        steps=steps,
    )
    output = f"/vol/runs/{run}"
    model.save(output)
    write_report(result, f"{output}/training.json")
    cache.commit()
    return {"output": output, **result}


@app.cls(
    image=runtime,
    gpu=gpu,
    volumes={"/vol": cache},
    secrets=hf_secrets,
    timeout=900,
    scaledown_window=60,
    max_containers=1,
)
@modal.concurrent(max_inputs=4)
class Server:
    @modal.enter()
    def load(self):
        from s1.backends import GemmaBackend

        self.backend = GemmaBackend(model_name, revision=revision)

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        from s1.api import create_app

        return create_app(self.backend)


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    dataset: str = "examples/benchmarks/smoke.jsonl",
    calibration: str = "",
    backend: str = "laya",
    output: str = "artifacts/modal-result.json",
    steps: int = 100,
    run: str = "unified-lora",
):
    from s1.benchmark import load_cases, write_report

    if mode == "smoke":
        result = smoke.remote()
    elif mode == "model-smoke":
        result = model_smoke.remote()
    elif mode == "benchmark":
        result = benchmark.remote([c.model_dump() for c in load_cases(dataset)], backend)
    elif mode == "train":
        if not calibration:
            raise ValueError("--calibration must name a separate calibration JSONL file")
        result = train.remote(
            [c.model_dump() for c in load_cases(dataset)],
            [c.model_dump() for c in load_cases(calibration)],
            steps,
            run,
        )
    else:
        raise ValueError("mode must be smoke, model-smoke, benchmark, or train")
    write_report(result, output)
    print(json.dumps(result, indent=2))
    if result.get("counts", {}).get("error"):
        raise RuntimeError("benchmark recorded errors; inspect the saved report")
